"""
WebSocket live odds streaming service.

Provides real-time odds streaming via WebSocket connections. The service:
1. Polls TheOddsAPI on a configurable interval (default 30s)
2. Pushes odds updates to all connected WebSocket clients
3. Detects significant line movements and emits alerts
4. Stores odds snapshots for CLV tracking
5. Gracefully degrades when the API is unavailable

Architecture:
    OddsPoller (async, runs on timer)
        -> fetches from TheOddsAPI / football-data / etc.
        -> compares with last snapshot
        -> on significant change: emits to WebSocket clients
        -> stores in SQLite time-series (odds_snapshots table)

    WebSocket endpoint (/ws/odds)
        -> clients connect and subscribe to games/leagues
        -> receives push updates in real-time

Usage:
    # Server: mounted on FastAPI
    from betting_intel.data.websocket_odds import OddsWebSocketManager
    manager = OddsWebSocketManager()
    app.add_websocket_route("/ws/odds", manager.websocket_endpoint)

    # Client: connect via WebSocket
    ws = await websockets.connect("ws://localhost:8000/ws/odds")
    await ws.send('{"subscribe": ["NBA", "BNXT"]}')
    async for msg in ws:
        odds_update = json.loads(msg)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd

try:
    from fastapi import WebSocket, WebSocketDisconnect
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Data Models ────────────────────────────────────────────────────────────


@dataclass
class OddsSnapshot:
    """A single point-in-time snapshot of odds for a game."""

    game_id: str
    league: str
    home_team: str
    away_team: str
    game_date: str
    home_ml: Optional[float] = None
    away_ml: Optional[float] = None
    spread: Optional[float] = None
    spread_home: Optional[float] = None
    total: Optional[float] = None
    total_over: Optional[float] = None
    total_under: Optional[float] = None
    sportsbook: str = "theoddsapi"
    captured_at: float = field(default_factory=time.time)
    is_live: bool = False


@dataclass
class OddsMovement:
    """Significant odds movement event."""

    game_id: str
    league: str
    matchup: str
    movement_type: str  # "total", "spread", "moneyline"
    old_value: float
    new_value: float
    change_pct: float
    direction: str  # "up", "down"
    is_sharp: bool = False
    triggered_at: float = field(default_factory=time.time)


# ── WebSocket Connection Manager ────────────────────────────────────────────


class ConnectionManager:
    """Manages WebSocket connections with subscription support."""

    def __init__(self):
        self._connections: dict[WebSocket, set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, leagues: Optional[list[str]] = None):
        await websocket.accept()
        async with self._lock:
            self._connections[websocket] = set(leagues or [])

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self._connections.pop(websocket, None)

    async def subscribe(self, websocket: WebSocket, leagues: list[str]):
        async with self._lock:
            if websocket in self._connections:
                self._connections[websocket].update(leagues)

    async def unsubscribe(self, websocket: WebSocket, leagues: list[str]):
        async with self._lock:
            if websocket in self._connections:
                self._connections[websocket].difference_update(leagues)

    async def broadcast(self, message: dict, league: Optional[str] = None):
        """Broadcast a message to all connected clients, optionally filtered by league."""
        payload = json.dumps(message, default=str)
        disconnected = []

        async with self._lock:
            for ws, subscriptions in self._connections.items():
                if league and subscriptions and league not in subscriptions:
                    continue
                try:
                    await ws.send_text(payload)
                except Exception:
                    disconnected.append(ws)

        # Clean up disconnected clients
        for ws in disconnected:
            await self.disconnect(ws)
            logger.debug("Removed disconnected WebSocket client")

    @property
    def active_connections(self) -> int:
        return len(self._connections)


# ── Odds Poller ─────────────────────────────────────────────────────────────


class OddsPoller:
    """Polls odds APIs on a timer and emits updates to WebSocket manager.

    Supports multiple data sources:
    1. TheOddsAPI (primary) — h2h, spreads, totals
    2. Football-data.org (soccer)
    3. Web scraping fallback
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        poll_interval: int = 30,
        odds_api_key: Optional[str] = None,
        db_path: Optional[Path] = None,
        movement_threshold_pct: float = 0.02,  # 2% change = significant
    ):
        self.manager = connection_manager
        self.poll_interval = poll_interval
        self.odds_api_key = odds_api_key
        self.db_path = db_path
        self.movement_threshold = movement_threshold_pct
        self._last_snapshots: dict[str, OddsSnapshot] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._ttl_hours: int = 48  # Evict snapshots older than 48h

        # Initialize database for time-series storage
        self._init_db()

    def _init_db(self):
        """Initialize SQLite table for odds snapshots."""
        if not self.db_path:
            return
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS odds_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT,
                    league TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    home_ml REAL,
                    away_ml REAL,
                    spread REAL,
                    total REAL,
                    sportsbook TEXT,
                    captured_at REAL,
                    is_live INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_odds_game
                ON odds_snapshots(game_id, captured_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_odds_league
                ON odds_snapshots(league, captured_at)
            """)
            conn.commit()
            conn.close()
            logger.info(f"Odds snapshots DB initialized at {self.db_path}")
        except Exception as exc:
            logger.warning(f"Could not initialize odds DB: {exc}")

    async def start(self):
        """Start the polling loop and periodic cleanup."""
        if self._running:
            return
        self._running = True
        self._client = httpx.AsyncClient(timeout=15.0)
        self._task = asyncio.create_task(self._poll_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(f"Odds poller started (interval={self.poll_interval}s)")

    async def stop(self):
        """Stop the polling loop and cleanup."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("Odds poller stopped")

    async def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Odds poll error: {exc}")
            await asyncio.sleep(self.poll_interval)

    async def _poll_once(self):
        """Fetch odds from all available sources and emit updates."""
        all_odds: list[OddsSnapshot] = []

        # 1. TheOddsAPI for US sports
        if self.odds_api_key and self.odds_api_key != "your-api-key-here":
            try:
                odds = await self._fetch_theoddsapi()
                all_odds.extend(odds)
            except Exception as exc:
                logger.warning(f"TheOddsAPI poll failed: {exc}")

        # 2. Emit updates for any significant changes
        for snapshot in all_odds:
            await self._process_snapshot(snapshot)

        # Broadcast heartbeat
        now_ts = time.time()
        await self.manager.broadcast({
            "type": "heartbeat",
            "timestamp": datetime.now().isoformat(),
            "games_tracked": len(self._last_snapshots),
            "connections": self.manager.active_connections,
        })

    async def _cleanup_loop(self):
        """Periodically evict stale snapshots from memory.
        Runs every 30 minutes.
        """
        while self._running:
            try:
                await self._evict_stale_snapshots()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug(f"Snapshot cleanup error: {exc}")
            await asyncio.sleep(1800)  # 30 min

    async def _evict_stale_snapshots(self):
        """Remove snapshots older than TTL from the in-memory dict and SQLite DB."""
        cutoff = time.time() - (self._ttl_hours * 3600)

        # In-memory eviction
        stale = [
            key for key, snap in self._last_snapshots.items()
            if snap.captured_at < cutoff
        ]
        for key in stale:
            del self._last_snapshots[key]
        if stale:
            logger.info(f"Evicted {len(stale)} stale snapshots from memory")

        # SQLite eviction — keep the DB lean too
        await self._evict_stale_db_snapshots(cutoff)

    async def _evict_stale_db_snapshots(self, cutoff: float):
        """Delete odds snapshots older than the cutoff from the SQLite DB."""
        if not self.db_path:
            return

        def _sync_evict():
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.execute(
                    "DELETE FROM odds_snapshots WHERE captured_at < ?",
                    (cutoff,),
                )
                deleted = cursor.rowcount
                conn.commit()
                conn.close()
                if deleted:
                    logger.info(f"Evicted {deleted} stale odds snapshots from SQLite DB")
            except Exception as exc:
                logger.debug(f"Failed to evict stale snapshots from DB: {exc}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _sync_evict)

    async def _fetch_theoddsapi(self) -> list[OddsSnapshot]:
        """Fetch odds from TheOddsAPI v4 — all sports in parallel.

        Uses ``asyncio.wait`` with an overall timeout so that a single slow
        API response does not delay the entire poll cycle. Each sport is
        fetched as a separate task; any task that exceeds the overall timeout
        is cancelled and its partial results are discarded.
        """
        if not self._client:
            return []

        base_url = "https://api.the-odds-api.com/v4"
        sports = [
            "basketball_nba",
            "basketball_wnba",
            "basketball_euroleague",
            "soccer_belgium_first_div",
        ]

        league_map = {
            "basketball_nba": "NBA",
            "basketball_wnba": "WNBA",
            "basketball_euroleague": "EuroLeague",
            "soccer_belgium_first_div": "soccer_belgian_pro_league",
        }

        async def _fetch_sport(sport: str) -> tuple[str, list[OddsSnapshot]]:
            """Fetch and parse odds for a single sport."""
            url = f"{base_url}/sports/{sport}/odds/"
            params = {
                "apiKey": self.odds_api_key,
                "regions": "us,eu",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            }
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            league = league_map.get(sport, sport)
            snapshots: list[OddsSnapshot] = []

            for game in data:
                home_team = game.get("home_team", "")
                away_team = game.get("away_team", "")
                game_date = (game.get("commence_time") or "")[:10]

                # Extract best odds across books
                best_home_ml = None
                best_away_ml = None
                best_spread = None
                best_total = None

                for bookmaker in game.get("bookmakers", []):
                    for market in bookmaker.get("markets", []):
                        key = market.get("key", "")
                        outcomes = market.get("outcomes", [])

                        if key == "h2h":
                            for outcome in outcomes:
                                if outcome.get("name", "").lower() == home_team.lower():
                                    if best_home_ml is None or outcome.get("price", 0) > best_home_ml:
                                        best_home_ml = outcome.get("price")
                                elif outcome.get("name", "").lower() == away_team.lower():
                                    if best_away_ml is None or outcome.get("price", 0) > best_away_ml:
                                        best_away_ml = outcome.get("price")
                        elif key == "spreads":
                            for outcome in outcomes:
                                if outcome.get("name", "").lower() == home_team.lower():
                                    best_spread = outcome.get("point")
                        elif key == "totals":
                            for outcome in outcomes:
                                if outcome.get("name", "").lower() == "over":
                                    best_total = outcome.get("point")

                game_id = f"{league}_{home_team}-{away_team}".replace(" ", "_")

                snapshots.append(OddsSnapshot(
                    game_id=game_id,
                    league=league,
                    home_team=home_team,
                    away_team=away_team,
                    game_date=game_date,
                    home_ml=best_home_ml,
                    away_ml=best_away_ml,
                    spread=best_spread,
                    total=best_total,
                    captured_at=time.time(),
                ))

            return sport, snapshots

        # Create one task per sport so they all run concurrently
        tasks = [asyncio.create_task(_fetch_sport(s)) for s in sports]

        # Overall timeout — if the whole batch takes longer than this,
        # pending tasks are cancelled so one slow API doesn't block the cycle
        OVERALL_TIMEOUT = 25.0
        done, pending = await asyncio.wait(
            tasks,
            timeout=OVERALL_TIMEOUT,
            return_when=asyncio.ALL_COMPLETED,
        )

        # Cancel any tasks that didn't finish in time
        for task in pending:
            task.cancel()
            logger.debug(f"Cancelled timed-out TheOddsAPI task")

        # Collect results from completed tasks
        all_snapshots: list[OddsSnapshot] = []
        for task in done:
            try:
                sport, snapshots = task.result()
                logger.debug(f"Fetched {len(snapshots)} games from TheOddsAPI/{sport}")
                all_snapshots.extend(snapshots)
            except asyncio.CancelledError:
                pass  # Handled above, but safe to ignore
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 401:
                    logger.warning("TheOddsAPI: invalid API key — stopping")
                    # Don't process remaining results; key is bad for all
                    return all_snapshots
                logger.debug(f"TheOddsAPI HTTP error: {exc.response.status_code}")
            except Exception as exc:
                logger.debug(f"TheOddsAPI fetch failed: {exc}")

        return all_snapshots

    async def _process_snapshot(self, snapshot: OddsSnapshot):
        """Process a single odds snapshot: detect movements, store, broadcast."""
        game_key = snapshot.game_id
        previous = self._last_snapshots.get(game_key)

        # Store snapshot (async to avoid blocking the event loop)
        self._last_snapshots[game_key] = snapshot
        await self._store_snapshot_async(snapshot)

        # Detect significant movements
        movements: list[OddsMovement] = []
        if previous:
            for field, label in [
                ("total", "total"),
                ("spread", "spread"),
                ("home_ml", "moneyline"),
            ]:
                old_val = getattr(previous, field)
                new_val = getattr(snapshot, field)
                if old_val is not None and new_val is not None and old_val != 0:
                    change = abs(new_val - old_val) / abs(old_val)
                    if change >= self.movement_threshold:
                        movements.append(OddsMovement(
                            game_id=snapshot.game_id,
                            league=snapshot.league,
                            matchup=f"{snapshot.away_team} @ {snapshot.home_team}",
                            movement_type=label,
                            old_value=old_val,
                            new_value=new_val,
                            change_pct=round(change * 100, 2),
                            direction="up" if new_val > old_val else "down",
                        ))

        # Broadcast if significant or if first snapshot
        if movements:
            for movement in movements:
                await self.manager.broadcast({
                    "type": "odds_movement",
                    "movement": asdict(movement),
                }, league=snapshot.league)

        # Broadcast current odds for this game
        await self.manager.broadcast({
            "type": "odds_update",
            "game": {
                "game_id": snapshot.game_id,
                "league": snapshot.league,
                "matchup": f"{snapshot.away_team} @ {snapshot.home_team}",
                "home_team": snapshot.home_team,
                "away_team": snapshot.away_team,
                "home_ml": snapshot.home_ml,
                "away_ml": snapshot.away_ml,
                "spread": snapshot.spread,
                "total": snapshot.total,
                "captured_at": snapshot.captured_at,
            },
        }, league=snapshot.league)

    async def _store_snapshot_async(self, snapshot: OddsSnapshot):
        """Store odds snapshot in SQLite for CLV tracking.

        Runs sync SQLite in an executor to avoid blocking the event loop.
        """
        if not self.db_path:
            return

        def _sync_store():
            try:
                conn = sqlite3.connect(str(self.db_path))
                conn.execute(
                    """INSERT INTO odds_snapshots
                       (game_id, league, home_team, away_team, home_ml, away_ml,
                        spread, total, sportsbook, captured_at, is_live)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        snapshot.game_id,
                        snapshot.league,
                        snapshot.home_team,
                        snapshot.away_team,
                        snapshot.home_ml,
                        snapshot.away_ml,
                        snapshot.spread,
                        snapshot.total,
                        snapshot.sportsbook,
                        snapshot.captured_at,
                        int(snapshot.is_live),
                    ),
                )
                conn.commit()
                conn.close()
            except Exception as exc:
                logger.debug(f"Failed to store odds snapshot: {exc}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _sync_store)

    def get_current_odds(self, league: Optional[str] = None) -> list[dict]:
        """Get current odds for all tracked games, optionally filtered by league."""
        result = []
        for key, snap in self._last_snapshots.items():
            if league and snap.league != league:
                continue
            result.append(asdict(snap))
        return result

    def get_odds_history(
        self,
        game_id: str,
        minutes: int = 60,
    ) -> pd.DataFrame:
        """Get historical odds snapshots for a game from SQLite."""
        if not self.db_path:
            return pd.DataFrame()

        cutoff = time.time() - (minutes * 60)
        try:
            conn = sqlite3.connect(str(self.db_path))
            df = pd.read_sql_query(
                """SELECT * FROM odds_snapshots
                   WHERE game_id = ? AND captured_at >= ?
                   ORDER BY captured_at ASC""",
                conn,
                params=(game_id, cutoff),
            )
            conn.close()
            return df
        except Exception as exc:
            logger.warning(f"Failed to fetch odds history: {exc}")
            return pd.DataFrame()

    def get_live_movements(
        self, league: Optional[str] = None, since_minutes: int = 30
    ) -> list[dict]:
        """Get significant line movements in the last N minutes."""
        movements = []
        for key, snap in self._last_snapshots.items():
            previous = None
            # Try to get previous snapshot from DB
            if self.db_path:
                try:
                    cutoff = time.time() - (since_minutes * 60)
                    conn = sqlite3.connect(str(self.db_path))
                    cursor = conn.execute(
                        """SELECT home_ml, away_ml, spread, total
                           FROM odds_snapshots
                           WHERE game_id = ? AND captured_at >= ?
                           ORDER BY captured_at ASC LIMIT 1""",
                        (key, cutoff),
                    )
                    row = cursor.fetchone()
                    conn.close()
                    if row:
                        previous = OddsSnapshot(
                            game_id=key, league="", home_team="", away_team="",
                            game_date="", home_ml=row[0], away_ml=row[1],
                            spread=row[2], total=row[3],
                        )
                except Exception:
                    pass

            if previous:
                for field, label in [("total", "total"), ("spread", "spread")]:
                    old = getattr(previous, field)
                    new = getattr(snap, field)
                    if old is not None and new is not None and old != 0:
                        change = abs(new - old) / abs(old)
                        if change >= self.movement_threshold:
                            movements.append({
                                "game_id": snap.game_id,
                                "league": snap.league,
                                "matchup": f"{snap.away_team} @ {snap.home_team}",
                                "type": label,
                                "old_value": old,
                                "new_value": new,
                                "change_pct": round(change * 100, 2),
                            })

        if league:
            movements = [m for m in movements if m.get("league") == league]

        return sorted(movements, key=lambda x: x["change_pct"], reverse=True)


# ── WebSocket Endpoint ──────────────────────────────────────────────────────


class OddsWebSocketManager:
    """Combined WebSocket manager + odds poller for FastAPI integration."""

    def __init__(
        self,
        poll_interval: int = 30,
        odds_api_key: Optional[str] = None,
        db_path: Optional[Path] = None,
    ):
        self.connection_manager = ConnectionManager()
        self.poller = OddsPoller(
            connection_manager=self.connection_manager,
            poll_interval=poll_interval,
            odds_api_key=odds_api_key,
            db_path=db_path,
        )

    async def start(self):
        """Start the odds polling loop."""
        await self.poller.start()

    async def stop(self):
        """Stop the odds polling loop."""
        await self.poller.stop()

    async def websocket_endpoint(self, websocket: WebSocket):
        """FastAPI WebSocket endpoint handler."""
        await self.connection_manager.connect(websocket)

        try:
            # Send initial state
            current_odds = self.poller.get_current_odds()
            await websocket.send_json({
                "type": "initial_state",
                "games": current_odds,
                "timestamp": datetime.now().isoformat(),
            })

            # Handle incoming messages (subscription management)
            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    msg_type = msg.get("type", msg.get("action", ""))

                    if msg_type == "subscribe":
                        leagues = msg.get("leagues", msg.get("params", []))
                        await self.connection_manager.subscribe(websocket, leagues)
                        await websocket.send_json({
                            "type": "subscribed",
                            "leagues": leagues,
                        })

                    elif msg_type == "unsubscribe":
                        leagues = msg.get("leagues", msg.get("params", []))
                        await self.connection_manager.unsubscribe(websocket, leagues)
                        await websocket.send_json({
                            "type": "unsubscribed",
                            "leagues": leagues,
                        })

                    elif msg_type == "get_odds":
                        league = msg.get("league")
                        odds = self.poller.get_current_odds(league=league)
                        await websocket.send_json({
                            "type": "odds_snapshot",
                            "games": odds,
                        })

                    elif msg_type == "get_movements":
                        league = msg.get("league")
                        movements = self.poller.get_live_movements(league=league)
                        await websocket.send_json({
                            "type": "live_movements",
                            "movements": movements,
                        })

                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON",
                    })

        except WebSocketDisconnect:
            await self.connection_manager.disconnect(websocket)
        except Exception as exc:
            logger.error(f"WebSocket error: {exc}")
            await self.connection_manager.disconnect(websocket)

    def get_current_odds_snapshot(self) -> dict:
        """Get current odds snapshot for REST API."""
        return {
            "games": self.poller.get_current_odds(),
            "movements": self.poller.get_live_movements(),
            "active_connections": self.connection_manager.active_connections,
            "timestamp": datetime.now().isoformat(),
        }
