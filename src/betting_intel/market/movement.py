"""
Market Movement Engine — Phase 3.10 of the Professional Betting Intelligence Platform.

Tracks:
    - opening_line: first odds ever recorded for a market
    - current_line: most recent odds
    - closing_line: last odds before game start

Calculates:
    - line_movement: raw difference between any two points
    - movement_velocity: how fast the line is moving (points per hour)
    - direction: which way the market is moving

Stores full history for every game, sportsbook, and market.
"""

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

class MoveDirection(Enum):
    UP = "up"           # Line increased (e.g. total from 218 to 220)
    DOWN = "down"       # Line decreased (e.g. total from 220 to 218)
    STEADY = "steady"   # No significant movement


@dataclass
class LineMovement:
    """Movement between two points in time for a specific market."""
    game_id: str
    market_type: str        # 'moneyline', 'spread', 'total'
    sportsbook: str

    # From/to
    from_odds: Optional[float] = None
    to_odds: Optional[float] = None
    from_timestamp: str = ""
    to_timestamp: str = ""

    # Computed
    raw_difference: float = 0.0
    percentage_change: float = 0.0
    direction: str = "steady"
    hours_elapsed: float = 0.0
    velocity: float = 0.0             # Points per hour
    is_significant: bool = False

    def __post_init__(self):
        if self.raw_difference > 0:
            self.direction = "up"
        elif self.raw_difference < 0:
            self.direction = "down"


@dataclass
class MarketMovementRecord:
    """Complete market movement record for a single game/market/sportsbook."""
    game_id: str
    home_team: str
    away_team: str
    game_date: str
    market_type: str

    # Key lines
    opening_line: Optional[float] = None
    opening_odds: Optional[float] = None
    opening_timestamp: str = ""

    current_line: Optional[float] = None
    current_odds: Optional[float] = None
    current_timestamp: str = ""

    closing_line: Optional[float] = None
    closing_odds: Optional[float] = None
    closing_timestamp: str = ""

    # Movements
    opening_to_current: Optional[LineMovement] = None
    opening_to_closing: Optional[LineMovement] = None
    current_to_closing: Optional[LineMovement] = None

    # Velocity
    max_velocity_24h: float = 0.0
    total_movement_abs: float = 0.0
    num_snapshots: int = 0

    def is_steam(self, threshold: float = 0.5) -> bool:
        """Check if this looks like a steam move (>0.5 pts per hour)."""
        return self.max_velocity_24h > threshold


@dataclass
class MarketTrend:
    """Aggregate market trend across sportsbooks for a single market."""
    game_id: str
    market_type: str

    consensus_opening: Optional[float] = None
    consensus_current: Optional[float] = None
    consensus_closing: Optional[float] = None

    move_direction: str = "steady"
    consensus_velocity: float = 0.0
    num_books: int = 0
    agreement_pct: float = 1.0       # % of books moving the same direction

    is_steam: bool = False


# ═══════════════════════════════════════════════════════════════════════════
#  MARKET MOVEMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class MarketMovementEngine:
    """
    Tracks and analyzes market line movements over time.

    Uses the odds table (from OddsIngestionEngine) to compute movements.

    Usage:
        engine = MarketMovementEngine(DB_PATH)
        movements = engine.get_movements("game_123", "total")
        trends = engine.get_trends("game_123")
        steam = engine.detect_steam_moves(hours_back=24)

        # Quick stats
        summary = engine.get_summary("game_123")
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ═══════════════════════════════════════════════════════════════════
    #  MOVEMENT COMPUTATION
    # ═══════════════════════════════════════════════════════════════════

    def get_movements(self, game_id: str,
                      market: str = "totals",
                      sportsbook: Optional[str] = None) -> List[LineMovement]:
        """
        Get line movements for a specific game/market over time.

        Returns sequential movements: each consecutive pair of snapshots
        produces one LineMovement.

        Args:
            game_id: Game identifier
            market: Market type ('totals', 'h2h', 'spreads')
            sportsbook: Optional filter by sportsbook

        Returns:
            List of LineMovement sorted chronologically
        """
        snapshots = self._get_time_series(game_id, market, sportsbook)
        if len(snapshots) < 2:
            return []

        movements = []
        for i in range(len(snapshots) - 1):
            curr = snapshots[i]
            nxt = snapshots[i + 1]

            raw_diff = nxt["line"] - curr["line"]
            pct_change = raw_diff / curr["line"] if curr["line"] != 0 else 0

            # Hours elapsed
            try:
                t1 = datetime.fromisoformat(curr["ts"].replace("Z", "+00:00"))
                t2 = datetime.fromisoformat(nxt["ts"].replace("Z", "+00:00"))
                hours = (t2 - t1).total_seconds() / 3600
            except Exception:
                hours = 1.0

            velocity = abs(raw_diff) / hours if hours > 0 else 0

            mov = LineMovement(
                game_id=game_id,
                market_type=market,
                sportsbook=curr["book"],
                from_odds=curr["line"],
                to_odds=nxt["line"],
                from_timestamp=curr["ts"],
                to_timestamp=nxt["ts"],
                raw_difference=round(raw_diff, 2),
                percentage_change=round(pct_change, 4),
                hours_elapsed=round(hours, 2),
                velocity=round(velocity, 4),
                is_significant=abs(raw_diff) >= 0.5 or velocity >= 0.5,
            )
            movements.append(mov)

        return movements

    def _get_time_series(self, game_id: str, market: str,
                         sportsbook: Optional[str] = None) -> List[Dict]:
        """Get ordered list of (line, timestamp, book) snapshots for a game/market."""
        query = """
            SELECT o.sportsbook, o.timestamp, o.odds_value
            FROM odds o
            WHERE o.game_id = ? AND o.market = ?
        """
        params = [game_id, market]

        if sportsbook:
            query += " AND o.sportsbook = ?"
            params.append(sportsbook)

        query += " ORDER BY o.sportsbook, o.timestamp ASC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        snapshots = []
        for row in rows:
            try:
                outcomes = json.loads(row["odds_value"])
                line = self._extract_line(outcomes, market)
                if line is not None:
                    snapshots.append({
                        "book": row["sportsbook"],
                        "ts": row["timestamp"],
                        "line": line,
                    })
            except (json.JSONDecodeError, TypeError):
                continue

        return snapshots

    def _extract_line(self, outcomes: List[Dict], market: str) -> Optional[float]:
        """Extract the line value from odds outcomes."""
        if market == "totals":
            # Get the Over's point value as the line
            for o in outcomes:
                if o.get("name", "").lower() in ("over", "o"):
                    return float(o.get("point", o.get("price", 0)))
        elif market == "spreads":
            for o in outcomes:
                if o.get("name", "").lower() in ("home", "h"):
                    return float(o.get("point", 0))
        elif market == "h2h":
            # For moneyline, use the home price
            for o in outcomes:
                if o.get("name", "").lower() in ("home", "h"):
                    return float(o.get("price", 0))
        return None

    # ═══════════════════════════════════════════════════════════════════
    #  FULL MARKET RECORD
    # ═══════════════════════════════════════════════════════════════════

    def get_market_record(self, game_id: str, market: str = "totals",
                          sportsbook: Optional[str] = None) -> Optional[MarketMovementRecord]:
        """Build a full MarketMovementRecord for a game/market."""
        snapshots = self._get_time_series(game_id, market, sportsbook)
        if not snapshots:
            return None

        # Extract opening, current, closing
        books = {}
        for s in snapshots:
            bk = s["book"]
            if bk not in books:
                books[bk] = []
            books[bk].append(s)

        # Use the sportsbook with the most snapshots
        if sportsbook and sportsbook in books:
            primary_book = sportsbook
        else:
            primary_book = max(books, key=lambda b: len(books[b])) if books else None

        if not primary_book:
            return None

        series = books[primary_book]
        opening = series[0]
        current = series[-1]

        # Get metadata
        with self._connect() as conn:
            meta = conn.execute(
                "SELECT home_team, away_team, commence_time FROM odds_meta WHERE game_id = ?",
                (game_id,)
            ).fetchone()

        home = meta["home_team"] if meta else ""
        away = meta["away_team"] if meta else ""
        game_date = meta["commence_time"][:10] if meta and meta["commence_time"] else ""

        # Build movements
        all_movements = self.get_movements(game_id, market, primary_book)
        opening_to_current = all_movements[0] if all_movements else None
        opening_to_closing = LineMovement(
            game_id=game_id, market_type=market, sportsbook=primary_book,
            from_odds=opening["line"], to_odds=current["line"],
            from_timestamp=opening["ts"], to_timestamp=current["ts"],
            raw_difference=round(current["line"] - opening["line"], 2),
        ) if series else None

        # Max velocity
        max_vel = max((m.velocity for m in all_movements), default=0.0)
        total_move = sum(abs(m.raw_difference) for m in all_movements)

        return MarketMovementRecord(
            game_id=game_id,
            home_team=home,
            away_team=away,
            game_date=game_date,
            market_type=market,
            opening_line=opening["line"],
            opening_timestamp=opening["ts"],
            current_line=current["line"],
            current_timestamp=current["ts"],
            opening_to_current=opening_to_current,
            opening_to_closing=opening_to_closing,
            max_velocity_24h=round(max_vel, 4),
            total_movement_abs=round(total_move, 2),
            num_snapshots=len(series),
        )

    # ═══════════════════════════════════════════════════════════════════
    #  MARKET TRENDS (across sportsbooks)
    # ═══════════════════════════════════════════════════════════════════

    def get_trends(self, game_id: str) -> List[MarketTrend]:
        """Get aggregate market trends across all sportsbooks for each market type."""
        trends = []

        for market in ["totals", "spreads", "h2h"]:
            snapshots = self._get_time_series(game_id, market)
            if not snapshots:
                continue

            # Group by book
            books: Dict[str, List] = {}
            for s in snapshots:
                bk = s["book"]
                if bk not in books:
                    books[bk] = []
                books[bk].append(s)

            # Get opening and current per book
            book_moves = []
            for bk, series in books.items():
                if len(series) >= 2:
                    book_moves.append({
                        "book": bk,
                        "opening": series[0]["line"],
                        "current": series[-1]["line"],
                        "direction": "up" if series[-1]["line"] > series[0]["line"]
                                     else "down" if series[-1]["line"] < series[0]["line"]
                                     else "steady",
                    })

            if not book_moves:
                continue

            # Consensus
            consensus_open = sum(b["opening"] for b in book_moves) / len(book_moves)
            consensus_cur = sum(b["current"] for b in book_moves) / len(book_moves)

            # Agreement: % of books moving the same direction
            directions = [b["direction"] for b in book_moves if b["direction"] != "steady"]
            if directions:
                most_common = max(set(directions), key=directions.count)
                agreement = directions.count(most_common) / len(directions)
            else:
                most_common = "steady"
                agreement = 1.0

            trend = MarketTrend(
                game_id=game_id,
                market_type=market,
                consensus_opening=round(consensus_open, 2),
                consensus_current=round(consensus_cur, 2),
                move_direction=most_common,
                consensus_velocity=abs(consensus_cur - consensus_open) / max(len(book_moves), 1),
                num_books=len(book_moves),
                agreement_pct=round(agreement, 2),
                is_steam=agreement >= 0.7 and abs(consensus_cur - consensus_open) >= 1.0,
            )
            trends.append(trend)

        return trends

    # ═══════════════════════════════════════════════════════════════════
    #  BULK / SUMMARY
    # ═══════════════════════════════════════════════════════════════════

    def get_all_active_movements(self) -> Dict[str, List[MarketMovementRecord]]:
        """Get market records for all currently tracked games."""
        with self._connect() as conn:
            active = conn.execute(
                "SELECT game_id FROM odds_meta WHERE is_finished = 0"
            ).fetchall()

        results = {}
        for row in active:
            gid = row["game_id"]
            records = []
            for market in ["totals", "spreads", "h2h"]:
                rec = self.get_market_record(gid, market)
                if rec:
                    records.append(rec)
            if records:
                results[gid] = records
        return results

    def get_summary(self, game_id: str) -> str:
        """Human-readable market movement summary for a game."""
        trends = self.get_trends(game_id)
        if not trends:
            return f"No market data for game {game_id}"

        lines = [f"📊 MARKET MOVEMENT — {game_id}", "─" * 50]
        for t in trends:
            steam_flag = " 🔥 STEAM" if t.is_steam else ""
            lines.append(
                f"\n{t.market_type.upper()}{steam_flag}: "
                f"{t.consensus_opening} → {t.consensus_current} "
                f"({t.move_direction.upper()})"
            )
            lines.append(f"  Books: {t.num_books} | Agreement: {t.agreement_pct:.0%}")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, int]:
        """Get engine statistics."""
        with self._connect() as conn:
            return {
                "games_tracked": conn.execute("SELECT COUNT(DISTINCT game_id) FROM odds_meta").fetchone()[0],
                "active_games": conn.execute("SELECT COUNT(*) FROM odds_meta WHERE is_finished = 0").fetchone()[0],
                "odds_records": conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0],
            }
