"""
LivePredictionEngine — real-time prediction system for live + upcoming games.

CRITICAL DESIGN RULE: TheOddsAPI is ONLY called when force_refresh=True
is explicitly passed. Page loads, WebSocket connections, and all
read-only operations return cached data only. Zero automatic API calls.

The user controls when to refresh by hitting the /api/live/refresh endpoint
or clicking the "Refresh" button on the UI.

Architecture:
  1. ONLY real data — if TheOddsAPI returns nothing, we show nothing
  2. User-initiated refresh only — no background polling
  3. Live detection: games with commence_time in the past = in-play
  4. 2-day rolling window: today's games + tomorrow's games combined
  5. ML predictions computed fresh each refresh cycle using real market lines
  6. No synthetic/random/fake games EVER — they destroy trust
"""

from __future__ import annotations

import json
import logging
import time
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────
ODDS_CACHE_TTL_SECONDS = 300       # Refresh odds every 5 minutes
PREDICTION_REFRESH_INTERVAL = 60   # Re-generate predictions every 60s
LIVE_GAME_LEEWAY_MINUTES = 60      # Game is "live" if started within 60 min of now


# ── Data Models ───────────────────────────────────────────────────────────

@dataclass
class LiveGame:
    """A single game — live or upcoming — with real market data."""
    game_id: str
    sport_key: str
    home_team: str
    away_team: str
    home_team_short: str
    away_team_short: str
    commence_time: str  # ISO 8601
    game_date: str      # YYYY-MM-DD

    # Market lines (from consensus across sportsbooks)
    home_ml: Optional[float] = None
    away_ml: Optional[float] = None
    spread: Optional[float] = None
    market_total: Optional[float] = None
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None

    # Consensus metadata
    n_books_ml: int = 0
    n_books_total: int = 0
    ml_std: Optional[float] = None

    # Status
    is_live: bool = False
    is_today: bool = False
    is_tomorrow: bool = False

    # ML predictions (filled by the model)
    predicted_total: Optional[float] = None
    edge_pct: Optional[float] = None
    direction: Optional[str] = None  # "over" or "under"
    confidence: Optional[str] = None

    # Quarter & Half projections
    q1_home: Optional[float] = None
    q1_away: Optional[float] = None
    q1_total: Optional[float] = None
    q2_home: Optional[float] = None
    q2_away: Optional[float] = None
    q2_total: Optional[float] = None
    q3_home: Optional[float] = None
    q3_away: Optional[float] = None
    q3_total: Optional[float] = None
    q4_home: Optional[float] = None
    q4_away: Optional[float] = None
    q4_total: Optional[float] = None
    h1_home: Optional[float] = None
    h1_away: Optional[float] = None
    h1_total: Optional[float] = None
    h2_home: Optional[float] = None
    h2_away: Optional[float] = None
    h2_total: Optional[float] = None

    # Bet recommendations
    recommended_quarter: Optional[str] = None   # e.g. "Q1", "Q2", "1H"
    recommended_direction: Optional[str] = None # "over" or "under"

    # Timestamps
    odds_fetched_at: Optional[str] = None
    predicted_at: Optional[str] = None

    @property
    def matchup(self) -> str:
        return f"{self.away_team_short} @ {self.home_team_short}"

    @property
    def commence_datetime(self) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(self.commence_time.replace("Z", "+00:00"))
        except Exception:
            return None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LivePredictionSnapshot:
    """Complete snapshot of all live + upcoming predictions."""
    live_games: list[LiveGame] = field(default_factory=list)
    today_games: list[LiveGame] = field(default_factory=list)
    tomorrow_games: list[LiveGame] = field(default_factory=list)
    next_two_days: list[LiveGame] = field(default_factory=list)

    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    n_live: int = 0
    n_today: int = 0
    n_tomorrow: int = 0
    n_total: int = 0
    fresh_odds: bool = False

    # Pre-computed chart data (built once when snapshot is constructed)
    # /api/live/chart-data returns this directly — sub-100ms
    # NOT included in to_dict() — only accessed directly by chart endpoint
    chart_data: Optional[dict] = None

    # Fields to exclude from serialization (chart_data, internal state)
    _exclude_from_dict: set = field(default_factory=lambda: {"chart_data", "_exclude_from_dict"})

    def __post_init__(self):
        self.n_live = len(self.live_games)
        self.n_today = len(self.today_games)
        self.n_tomorrow = len(self.tomorrow_games)
        self.n_total = len(self.next_two_days)
        # Pre-compute chart data immediately when snapshot is built
        self.chart_data = self._build_chart_data()

    def _build_chart_data(self) -> dict:
        """Pre-compute all chart data from game objects.
        
        Called once when the snapshot is constructed, so /api/live/chart-data
        endpoints return instantly instead of re-processing every game.
        """
        edges = []
        confidence_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "neutral": 0}
        over_count = 0
        under_count = 0
        neutral_count = 0

        for g in self.next_two_days:
            d = g.to_dict()
            edge_pct = d.get("edge_pct")
            if edge_pct is not None and edge_pct != 0:
                edges.append({
                    "matchup": d["matchup"],
                    "edge_pct": round(edge_pct * 100, 1),
                    "predicted_total": d.get("predicted_total"),
                    "market_total": d.get("market_total"),
                    "is_live": d.get("is_live", False),
                    "confidence": d.get("confidence", "low"),
                    "home_team": d.get("home_team_short", ""),
                    "away_team": d.get("away_team_short", ""),
                    "spread": d.get("spread"),
                    "n_books_ml": d.get("n_books_ml", 0),
                    "direction": d.get("direction", "neutral"),
                })

            c = d.get("confidence", "low") or "low"
            if c in confidence_counts:
                confidence_counts[c] += 1
            else:
                confidence_counts[c] = 1

            direction = d.get("direction", "neutral")
            if direction == "over":
                over_count += 1
            elif direction == "under":
                under_count += 1
            else:
                neutral_count += 1

        return {
            "n_live": self.n_live,
            "n_today": self.n_today,
            "n_tomorrow": self.n_tomorrow,
            "n_total": self.n_total,
            "edges": edges,
            "confidence_breakdown": confidence_counts,
            "direction_breakdown": {
                "over": over_count,
                "under": under_count,
                "neutral": neutral_count,
            },
            "generated_at": self.generated_at,
            "fresh_odds": self.fresh_odds,
        }

    def to_dict(self) -> dict:
        # Exclude internal fields (chart_data, etc.) — only accessed
        # directly by the chart-data endpoint, not serialized over wire
        d = {
            "live_games": [g.to_dict() for g in self.live_games],
            "today_games": [g.to_dict() for g in self.today_games],
            "tomorrow_games": [g.to_dict() for g in self.tomorrow_games],
            "next_two_days": [g.to_dict() for g in self.next_two_days],
            "generated_at": self.generated_at,
            "n_live": self.n_live,
            "n_today": self.n_today,
            "n_tomorrow": self.n_tomorrow,
            "n_total": self.n_total,
            "fresh_odds": self.fresh_odds,
        }
        # Remove internal-only fields from the serialized dict
        exclude = getattr(self, '_exclude_from_dict', set())
        for field_name in exclude:
            d.pop(field_name, None)
        return d


# ── Live Prediction Engine ───────────────────────────────────────────────

class LivePredictionEngine:
    """
    Continuously produces predictions for live + upcoming games using ONLY real data.

    Zero synthetic games. Zero random matchups. If TheOddsAPI returns nothing,
    every method returns an empty list.

    Thread-safe: uses a lock for the internal snapshot cache so the web app
    can read predictions while the background worker refreshes them.
    """

    def __init__(
        self,
        odds_api_key: Optional[str] = None,
        refresh_interval: int = PREDICTION_REFRESH_INTERVAL,
        model_dir: Optional[Path] = None,
    ):
        self._odds_api_key = odds_api_key or os.getenv("ODDS_API_KEY", "")
        self._refresh_interval = refresh_interval
        self._model_dir = model_dir

        # Thread-safe cache
        self._lock = Lock()
        self._snapshot: Optional[LivePredictionSnapshot] = None
        self._last_refresh: float = 0.0
        self._last_odds_fetch: float = 0.0
        self._cached_odds_raw: Optional[list[dict]] = None

        # Lazy-loaded ML model reference
        self._model = None
        self._feature_cols: list[str] = []

        logger.info(
            "LivePredictionEngine initialized — zero synthetic games. "
            f"Refresh interval: {refresh_interval}s"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def get_snapshot(self, force_refresh: bool = False) -> LivePredictionSnapshot:
        """
        Get the latest prediction snapshot. Thread-safe.

        CRITICAL: When force_refresh=False (the default), this NEVER calls
        TheOddsAPI. It returns whatever is in the cache, or an empty
        snapshot if nothing is cached yet. Only force_refresh=True triggers
        an actual API call.
        """
        if not force_refresh:
            with self._lock:
                if self._snapshot is not None:
                    return self._snapshot
                # No cached data yet — return empty snapshot, never auto-fetch
                logger.debug("No cached snapshot — returning empty (user must refresh)")
                return LivePredictionSnapshot()

        # force_refresh=True: explicitly fetch fresh data from TheOddsAPI
        now = time.time()
        try:
            snapshot = self._build_snapshot()
            with self._lock:
                self._snapshot = snapshot
                self._last_refresh = now
            return snapshot
        except Exception as e:
            logger.error(f"Failed to refresh predictions: {e}")
            with self._lock:
                return self._snapshot or LivePredictionSnapshot()

    def get_live_games(self, force_refresh: bool = False) -> list[LiveGame]:
        """Get ONLY games that are currently live/in-progress (cached only unless force_refresh)."""
        snapshot = self.get_snapshot(force_refresh=force_refresh)
        return snapshot.live_games

    def get_today_games(self, force_refresh: bool = False) -> list[LiveGame]:
        """Get today's games — cached only unless force_refresh."""
        snapshot = self.get_snapshot(force_refresh=force_refresh)
        return snapshot.today_games

    def get_tomorrow_games(self, force_refresh: bool = False) -> list[LiveGame]:
        """Get tomorrow's games — cached only unless force_refresh."""
        snapshot = self.get_snapshot(force_refresh=force_refresh)
        return snapshot.tomorrow_games

    def get_next_two_days(self, force_refresh: bool = False) -> list[LiveGame]:
        """Get ALL games in the next 2 days — cached only unless force_refresh."""
        snapshot = self.get_snapshot(force_refresh=force_refresh)
        return snapshot.next_two_days

    def refresh_now(self) -> LivePredictionSnapshot:
        """Force an immediate refresh and return the new snapshot."""
        return self.get_snapshot(force_refresh=True)

    def clear_cache(self):
        """
        Reset ALL internal caches and state, including external scraper caches.

        Use this to isolate test fixtures between scenarios:
          1. engine.clear_cache()
          2. Change mock / external conditions
          3. Call refresh_now() — will fetch fresh data from scratch

        Resets:
          - Snapshot cache (_snapshot = None)
          - Raw odds cache (_cached_odds_raw = None)
          - Refresh timestamps (_last_refresh, _last_odds_fetch)
          - Lazy-loaded ML model (_model, _feature_cols)
          - StealthBrowser cache (ESPN scraper)
          - DraftKingsScraper cache

        Thread-safe: acquires the internal lock.
        """
        with self._lock:
            self._snapshot = None
            self._last_refresh = 0.0
            self._last_odds_fetch = 0.0
            self._cached_odds_raw = None
            self._model = None
            self._feature_cols = []

        # Clear external scraper caches
        try:
            from betting_intel.data.stealth_scraper import StealthBrowser
            StealthBrowser.clear_cache()
        except Exception:
            pass
        try:
            from betting_intel.data.draftkings_scraper import DraftKingsScraper
            DraftKingsScraper.clear_cache()
        except Exception:
            pass

        logger.debug("LivePredictionEngine cache cleared")

    @property
    def has_cached_data(self) -> bool:
        """Check if the engine has a cached snapshot (without triggering a refresh)."""
        with self._lock:
            return self._snapshot is not None and self._snapshot.n_total > 0

    # ── Odds Fetching ─────────────────────────────────────────────────────

    def _fetch_realtime_odds(self) -> list[dict]:
        """
        Fetch live odds — tries the ESPN stealth scraper (public, free API)
        FIRST, then falls back to TheOddsAPI (requires API key, quota-limited).

        Priority: ESPN public API (free, unlimited) > TheOddsAPI (key required).
        Returns ONLY real game data. Empty list if ALL sources fail.
        Uses in-memory cache with ODDS_CACHE_TTL_SECONDS TTL.
        """
        now = time.time()

        # In-memory cache
        if self._cached_odds_raw is not None and (now - self._last_odds_fetch) < ODDS_CACHE_TTL_SECONDS:
            return self._cached_odds_raw

        # Try ESPN stealth scraper FIRST (free, unlimited, no key needed)
        try:
            stealth_data = self._fetch_stealth_scraper()
        except Exception as e:
            logger.warning(f"ESPN stealth scraper raised unexpected exception: {e}")
            stealth_data = []

        # Then try DraftKings as a second free odds source
        # (ESPN + DraftKings = 2 books for better consensus)
        try:
            dk_data = self._fetch_draftkings_odds()
        except Exception as e:
            logger.warning(f"DraftKings scraper raised unexpected exception: {e}")
            dk_data = []

        # Merge ESPN and DraftKings data
        merged = self._merge_odds_sources(stealth_data, dk_data)
        if merged:
            self._cached_odds_raw = merged
            self._last_odds_fetch = now
            return merged

        # Fallback: TheOddsAPI (requires API key, quota-limited)
        logger.info("Free scrapers returned no data — trying TheOddsAPI")
        theodds_data = self._fetch_via_theoddsapi()
        if theodds_data:
            self._cached_odds_raw = theodds_data
            self._last_odds_fetch = now
            return theodds_data

        logger.warning("All odds sources failed — no data available")
        self._cached_odds_raw = []
        self._last_odds_fetch = now
        return []

    def _fetch_via_theoddsapi(self) -> list[dict]:
        """
        Try fetching odds from TheOddsAPI.

        Returns:
            List of game dicts if successful, or empty list on failure.
        """
        # Validate API key
        if not self._odds_api_key or self._odds_api_key in (
            "your-api-key-here", "", "REPLACE_ME_WITH_YOUR_ODDS_API_KEY"
        ):
            logger.debug("No valid ODDS_API_KEY — skipping TheOddsAPI")
            return []

        try:
            import urllib.request
            import urllib.error

            url = (
                f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
                f"?apiKey={self._odds_api_key}"
                f"&regions=us,us2,eu,uk,au"
                f"&markets=h2h,spreads,totals"
                f"&oddsFormat=american"
                f"&dateFormat=iso"
            )

            req = urllib.request.Request(url, headers={"User-Agent": "betting-intel-live/2.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = resp.status
                raw = resp.read().decode("utf-8")
                if status != 200:
                    logger.warning(f"TheOddsAPI returned HTTP {status}")
                    return []
                data = json.loads(raw)

            if not isinstance(data, list) or len(data) == 0:
                logger.info("TheOddsAPI returned no games")
                return []

            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.info(f"Fetched {len(data)} games from TheOddsAPI (quota: {remaining} remaining)")
            return data

        except urllib.error.HTTPError as e:
            status = e.code
            body = e.read().decode("utf-8", errors="replace")[:200]
            if status == 429:
                logger.warning(f"TheOddsAPI quota exceeded (429)")
            else:
                logger.warning(f"TheOddsAPI HTTP {status}: {body}")
            return []

        except urllib.error.URLError as e:
            logger.warning(f"TheOddsAPI connection failed: {e}")
            return []

        except Exception as e:
            logger.error(f"Unexpected error from TheOddsAPI: {e}")
            return []

    def _fetch_stealth_scraper(self) -> list[dict]:
        """
        Fetch odds via the ESPN HTTP scraper (free, no key required).

        Makes a direct HTTP call to ESPN's public JSON API.
        No browser needed. Returns data in seconds.
        Output format: TheOddsAPI-compatible dicts (bookmakers/markets).

        Returns:
            List of parsed game dicts in TheOddsAPI format,
            or empty list if the scraper fails.
        """
        logger.info("Attempting ESPN stealth scraper...")

        try:
            from betting_intel.data.stealth_scraper import StealthBrowser

            scraped = StealthBrowser.sync_scrape_live_odds(
                odds_api_key=self._odds_api_key,
                timeout=25,
            )

            if scraped:
                logger.info(f"ESPN stealth scraper: {len(scraped)} games")
                return scraped

            logger.info("ESPN stealth scraper: returned no data")
            return []

        except ImportError as e:
            logger.debug(f"ESPN stealth scraper: not available ({e})")
            return []
        except Exception as e:
            logger.warning(f"ESPN stealth scraper failed: {e}")
            return []

    def _fetch_draftkings_odds(self) -> list[dict]:
        """
        Fetch odds via DraftKings scraper (free, no key required).

        Uses Playwright to bypass Cloudflare and access DraftKings'
        internal sportsbook API. Falls back gracefully if DraftKings
        is blocked or unavailable.

        Runs in a separate thread with a hard timeout so a slow
        Playwright launch never blocks the main odds fetch.

        Returns:
            List of parsed game dicts in TheOddsAPI format,
            or empty list if the scraper fails or times out.
        """
        logger.info("Attempting DraftKings scraper...")

        import threading

        result = []
        exception = []

        def _run():
            try:
                from betting_intel.data.draftkings_scraper import DraftKingsScraper
                scraped = DraftKingsScraper.scrape(timeout=20)
                if scraped:
                    result.extend(scraped)
            except Exception as e:
                exception.append(e)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=30)

        if thread.is_alive():
            logger.warning("DraftKings scraper timed out (>30s) — using ESPN data only")
            return []

        if exception:
            logger.warning(f"DraftKings scraper failed: {exception[0]}")
            return []

        if result:
            logger.info(f"DraftKings scraper: {len(result)} games")
            return result

        logger.info("DraftKings scraper: returned no data")
        return []

    def _merge_odds_sources(
        self,
        espn_data: list[dict],
        dk_data: list[dict],
    ) -> list[dict]:
        """
        Merge ESPN and DraftKings odds data into a single list.

        Games with the same matchup (home_team + away_team) get their
        bookmakers merged into one game object. This gives the engine
        multi-book consensus from 2 free sources.

        Args:
            espn_data: Games from ESPN scraper (TheOddsAPI format)
            dk_data: Games from DraftKings scraper (TheOddsAPI format)

        Returns:
            Merged list of game dicts with bookmakers from both sources.
        """
        if not espn_data and not dk_data:
            return []

        if not dk_data:
            return espn_data

        if not espn_data:
            return dk_data

        # Build a lookup by matchup for DraftKings games
        dk_by_matchup: dict[str, list[dict]] = {}
        for dk_game in dk_data:
            home = dk_game.get("home_team", "")
            away = dk_game.get("away_team", "")
            if home and away:
                key = f"{home}|{away}"
                if key not in dk_by_matchup:
                    dk_by_matchup[key] = []
                dk_by_matchup[key].append(dk_game)

        # Merge: for each ESPN game, add matching DraftKings bookmakers
        merged: list[dict] = []
        seen_matchups: set[str] = set()

        for espn_game in espn_data:
            home = espn_game.get("home_team", "")
            away = espn_game.get("away_team", "")
            key = f"{home}|{away}"
            seen_matchups.add(key)

            # Make a shallow copy so we don't mutate the original input
            merged_game = dict(espn_game)
            merged_books = list(merged_game.get("bookmakers", []))

            # Add matching DraftKings bookmakers to this ESPN game
            dk_games = dk_by_matchup.get(key, [])
            for dk_game in dk_games:
                dk_books = dk_game.get("bookmakers", [])
                merged_books.extend(dk_books)

            merged_game["bookmakers"] = merged_books
            merged.append(merged_game)

        # Add any DraftKings games that didn't match an ESPN game
        for dk_game in dk_data:
            home = dk_game.get("home_team", "")
            away = dk_game.get("away_team", "")
            key = f"{home}|{away}"
            if key not in seen_matchups:
                merged.append(dk_game)
                seen_matchups.add(key)

        logger.info(
            f"Merged odds: {len(merged)} total games "
            f"({len(espn_data)} from ESPN, {len(dk_data)} from DraftKings)"
        )

        return merged

    # ── Snapshot Builder ──────────────────────────────────────────────────

    def _build_snapshot(self) -> LivePredictionSnapshot:
        """
        Build a complete prediction snapshot from scratch.

        Flow:
          1. Fetch real odds from TheOddsAPI
          2. Parse into LiveGame objects
          3. Classify as live / today / tomorrow
          4. Run ML predictions (if model available)
          5. Compute edges and confidence
          6. Return snapshot with all categories
        """
        raw_odds = self._fetch_realtime_odds()
        fresh_odds = bool(raw_odds)

        # Parse into LiveGame objects
        all_games = self._parse_games(raw_odds)

        if not all_games:
            logger.info("No real games available — returning empty snapshot (no synthetic data)")
            return LivePredictionSnapshot(fresh_odds=fresh_odds)

        # Classify games by time
        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")
        tomorrow_str = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")
        day_after_str = (now_utc + timedelta(days=2)).strftime("%Y-%m-%d")

        live_games: list[LiveGame] = []
        today_games: list[LiveGame] = []
        tomorrow_games: list[LiveGame] = []
        day_after_games: list[LiveGame] = []

        for game in all_games:
            game.is_today = game.game_date == today_str
            game.is_tomorrow = game.game_date == tomorrow_str

            # Live detection: game has started but within leeway
            commence_dt = game.commence_datetime
            if commence_dt and commence_dt < now_utc:
                age_minutes = (now_utc - commence_dt).total_seconds() / 60
                game.is_live = age_minutes < LIVE_GAME_LEEWAY_MINUTES

            if game.is_live:
                live_games.append(game)
            if game.is_today:
                today_games.append(game)
            if game.is_tomorrow:
                tomorrow_games.append(game)
            if game.game_date == day_after_str:
                day_after_games.append(game)

        # Run ML predictions on all games
        predicted_games = self._predict_games(today_games + tomorrow_games + live_games)

        # Build combined window (deduplicated)
        seen_ids: set[str] = set()
        next_two_days: list[LiveGame] = []
        for game in today_games + tomorrow_games + day_after_games:
            if game.game_id not in seen_ids:
                next_two_days.append(game)
                seen_ids.add(game.game_id)

        snapshot = LivePredictionSnapshot(
            live_games=live_games,
            today_games=[g for g in today_games if not g.is_live] + live_games,
            tomorrow_games=tomorrow_games,
            next_two_days=next_two_days,
            generated_at=datetime.now().isoformat(),
            fresh_odds=fresh_odds,
        )

        if live_games:
            logger.info(f"LIVE: {len(live_games)} games in progress")
        logger.info(
            f"Snapshot built: {len(today_games)} today, "
            f"{len(tomorrow_games)} tomorrow, "
            f"{len(day_after_games)} day after, "
            f"{len(next_two_days)} total in window"
        )

        return snapshot

    def _parse_games(self, raw_odds: list[dict]) -> list[LiveGame]:
        """
        Parse raw TheOddsAPI response into LiveGame objects.

        Maps full team names to short names for display consistency.
        Extracts consensus market lines across all sportsbooks.
        """
        from betting_intel.data.odds_fetcher import ODDS_TO_SHORT_NAME

        games: list[LiveGame] = []
        now_utc = datetime.now(timezone.utc)

        for event in raw_odds:
            try:
                home_full = event.get("home_team", "")
                away_full = event.get("away_team", "")
                if not home_full or not away_full:
                    continue

                home_short = ODDS_TO_SHORT_NAME.get(home_full, home_full.split()[-1] if " " in home_full else home_full)
                away_short = ODDS_TO_SHORT_NAME.get(away_full, away_full.split()[-1] if " " in away_full else away_full)

                commence_time = event.get("commence_time", "")
                game_date = commence_time[:10] if commence_time else ""

                # Skip games older than 2 hours (they're finished)
                try:
                    commence_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                    age_hours = (now_utc - commence_dt).total_seconds() / 3600
                    if age_hours > 3:
                        continue  # Game finished over 3 hours ago — skip
                except Exception:
                    pass

                # Extract consensus odds across all sportsbooks
                home_ml_values: list[float] = []
                away_ml_values: list[float] = []
                total_values: list[float] = []
                over_odds_values: list[float] = []
                under_odds_values: list[float] = []
                spread_values: list[float] = []

                bookmakers = event.get("bookmakers", [])
                for book in bookmakers:
                    markets = book.get("markets", [])
                    for market in markets:
                        key = market.get("key", "")
                        outcomes = market.get("outcomes", [])
                        if key == "h2h":
                            for o in outcomes:
                                name = o.get("name", "")
                                price = o.get("price")
                                if price is not None:
                                    if name == home_full:
                                        home_ml_values.append(float(price))
                                    elif name == away_full:
                                        away_ml_values.append(float(price))
                        elif key == "spreads":
                            for o in outcomes:
                                point = o.get("point")
                                if point is not None and o.get("name", "") == home_full:
                                    spread_values.append(float(point))
                        elif key == "totals":
                            for o in outcomes:
                                point = o.get("point")
                                price = o.get("price")
                                if point is not None:
                                    total_values.append(float(point))
                                    if price is not None:
                                        if o.get("name", "") == "Over":
                                            over_odds_values.append(float(price))
                                        elif o.get("name", "") == "Under":
                                            under_odds_values.append(float(price))

                # Compute consensus (median) — robust against outlier sportsbooks
                def median_or_none(values: list) -> Optional[float]:
                    if not values:
                        return None
                    sorted_vals = sorted(values)
                    n = len(sorted_vals)
                    return sorted_vals[n // 2]

                consensus_home_ml = median_or_none(home_ml_values)
                consensus_away_ml = median_or_none(away_ml_values)
                consensus_total = median_or_none(total_values)
                consensus_spread = median_or_none(spread_values)
                consensus_over_odds = median_or_none(over_odds_values)
                consensus_under_odds = median_or_none(under_odds_values)

                def std_or_none(values: list) -> Optional[float]:
                    if len(values) < 2:
                        return None
                    import statistics
                    return statistics.stdev(values)

                game = LiveGame(
                    game_id=event.get("id", f"{home_short}_{away_short}_{game_date}"),
                    sport_key=event.get("sport_key", "basketball_nba"),
                    home_team=home_full,
                    away_team=away_full,
                    home_team_short=home_short,
                    away_team_short=away_short,
                    commence_time=commence_time,
                    game_date=game_date,
                    home_ml=consensus_home_ml,
                    away_ml=consensus_away_ml,
                    spread=consensus_spread,
                    market_total=consensus_total,
                    over_odds=consensus_over_odds,
                    under_odds=consensus_under_odds,
                    n_books_ml=len(home_ml_values),
                    n_books_total=len(total_values),
                    ml_std=std_or_none(home_ml_values),
                    odds_fetched_at=datetime.now().isoformat(),
                )
                games.append(game)

            except Exception as e:
                logger.debug(f"Skipping malformed game event: {e}")
                continue

        return games

    def _predict_games(self, games: list[LiveGame]) -> list[LiveGame]:
        """
        Run ML predictions on each game using the trained model.

        For each game, computes:
          - predicted_total: model's estimate of total points
          - edge_pct: (predicted - market) / market
          - direction: over/under based on edge
          - confidence: high/medium/low based on edge magnitude

        If no model is loaded, tries to load one from disk.
        Falls back to simple statistical estimates using team averages
        from the database (never random numbers).
        """
        if not games:
            return games

        # Try to load the trained model
        if self._model is None:
            self._load_model()

        # If model is available, use it
        if self._model is not None and self._feature_cols:
            return self._predict_with_model(games)

        # Fallback: use market-implied estimates with no edge
        # This is better than generating fake predictions
        for game in games:
            if game.market_total and game.market_total > 0:
                game.predicted_total = game.market_total
                game.edge_pct = 0.0
                game.direction = "neutral"
                game.confidence = "low"
                game.predicted_at = datetime.now().isoformat()

        return games

    def _load_model(self):
        """Load a pre-trained model from disk."""
        try:
            from betting_intel.pipeline.export import load_engine_model
            model_dir = self._model_dir
            if model_dir is None:
                model_dir = (
                    Path(__file__).resolve().parent.parent.parent.parent
                    / "models" / "saved"
                )
            model, metadata = load_engine_model(model_dir=model_dir)
            if model is not None:
                self._model = model
                self._feature_cols = metadata.get("feature_cols", []) if metadata else []
                logger.info(
                    f"Loaded pre-trained model with {len(self._feature_cols)} features"
                )
        except Exception as e:
            logger.debug(f"No pre-trained model available: {e}")

    def _predict_with_model(self, games: list[LiveGame]) -> list[LiveGame]:
        """Run model prediction on each game using feature vectors from historical data."""
        import numpy as np
        import pandas as pd

        try:
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.data.features import FeatureEngineer

            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is None or raw_df.empty:
                logger.warning("No historical data for feature engineering")
                return games

            games_df = loader.build_game_dataset(raw_df)
            raw_df = loader.compute_rest_days(raw_df)

            fe = FeatureEngineer()
            features_df = fe.build_all_features(games_df, raw_df)

            if features_df is None or features_df.empty:
                return games

            for game in games:
                try:
                    # Build feature vector from historical data
                    feat = self._build_feature_vector(
                        game.home_team_short,
                        game.away_team_short,
                        features_df,
                    )
                    if feat is None:
                        continue

                    X_pred = feat.values.reshape(1, -1)
                    raw_pred = self._model.predict(X_pred)
                    if isinstance(raw_pred, (list, tuple)):
                        raw_pred = raw_pred[0]
                    if hasattr(raw_pred, '__len__') and not isinstance(raw_pred, (str, bytes)):
                        predicted_total = float(np.asarray(raw_pred).flatten()[0])
                    else:
                        predicted_total = float(raw_pred)

                    # Validate prediction range
                    if predicted_total > 300 or predicted_total < 80:
                        logger.debug(
                            f"Unreasonable prediction {predicted_total:.0f} "
                            f"for {game.matchup} — skipping (unreasonable)"
                        )
                        # Don't assign a made-up number. Leave as None so
                        # downstream knows no reliable prediction exists.
                        game.predicted_total = None
                        game.edge_pct = 0.0
                        game.direction = "neutral"
                        game.confidence = "low"
                        game.predicted_at = datetime.now().isoformat()
                        continue

                    game.predicted_total = round(predicted_total, 1)

                    # Compute edge vs real market
                    if game.market_total and game.market_total > 0:
                        edge = (predicted_total - game.market_total) / game.market_total
                        game.edge_pct = round(edge, 4)
                        game.direction = "over" if edge > 0 else "under"

                        abs_edge = abs(edge)
                        if abs_edge > 0.05:
                            game.confidence = "high"
                        elif abs_edge >= 0.02:
                            game.confidence = "medium"
                        else:
                            game.confidence = "low"
                    else:
                        game.edge_pct = 0.0
                        game.direction = "neutral"
                        game.confidence = "low"

                    game.predicted_at = datetime.now().isoformat()

                except Exception as e:
                    logger.debug(f"Prediction failed for {game.matchup}: {e}")
                    continue

            return games

        except Exception as e:
            logger.warning(f"ML prediction pipeline failed: {e}")
            return games

    def _build_feature_vector(
        self,
        home_team: str,
        away_team: str,
        features_df: pd.DataFrame,
    ) -> Optional["pd.Series"]:
        """
        Build a feature vector for a matchup from historical feature data.

        Uses per-team rolling averages across all feature columns.
        Falls back to zero if no historical data exists for a team.
        """
        import numpy as np
        import pandas as pd

        if features_df is None or features_df.empty:
            return None

        feature_cols = self._feature_cols
        if not feature_cols:
            feature_cols = [
                c for c in features_df.select_dtypes(include=[np.number]).columns
                if c not in {
                    "game_id", "game_date", "home_team", "away_team",
                    "total_points", "spread", "label", "home_win",
                    "home_score", "away_score",
                }
            ]

        # Find team name columns
        home_col = "TEAM_NAME_home" if "TEAM_NAME_home" in features_df.columns else "home_team"
        away_col = "TEAM_NAME_away" if "TEAM_NAME_away" in features_df.columns else "away_team"

        def _team_avg(team: str, base_stat: str, n: int = 10) -> float:
            home_stat = f"home_{base_stat}"
            away_stat = f"away_{base_stat}"
            home_vals = pd.Series(dtype=float)
            away_vals = pd.Series(dtype=float)

            if home_stat in features_df.columns and home_col in features_df.columns:
                try:
                    mask = features_df[home_col].astype(str).str.contains(team, case=False, na=False)
                    home_vals = features_df.loc[mask, home_stat]
                except Exception:
                    pass

            if away_stat in features_df.columns and away_col in features_df.columns:
                try:
                    mask = features_df[away_col].astype(str).str.contains(team, case=False, na=False)
                    away_vals = features_df.loc[mask, away_stat]
                except Exception:
                    pass

            combined = pd.concat([home_vals, away_vals]).tail(n)
            return float(combined.mean()) if len(combined) > 0 else 0.0

        feature_dict: dict[str, float] = {}
        for col in feature_cols:
            if col.startswith("home_"):
                base = col[5:]
                feature_dict[col] = _team_avg(home_team, base)
            elif col.startswith("away_"):
                base = col[5:]
                feature_dict[col] = _team_avg(away_team, base)
            elif col.endswith("_diff"):
                base = col.replace("_diff", "")
                feature_dict[col] = _team_avg(home_team, base) - _team_avg(away_team, base)
            elif col.startswith("TEAM_"):
                feature_dict[col] = 0.0
            else:
                feature_dict[col] = float(features_df[col].mean()) if col in features_df.columns else 0.0

        result = pd.Series(feature_dict)
        if result.isnull().any():
            result = result.fillna(0.0)

        return result


# ── Background Worker ───────────────────────────────────────────────────

class LivePredictionWorker:
    """
    Background worker that continuously refreshes predictions.

    Designed to be run in a separate thread or asyncio task.
    Updates the shared engine's snapshot every refresh_interval seconds.
    """

    def __init__(self, engine: LivePredictionEngine):
        self.engine = engine
        self._running = False

    def start(self):
        """Start the continuous refresh loop (blocking)."""
        self._running = True
        logger.info("Live prediction worker started")

        while self._running:
            try:
                snapshot = self.engine.refresh_now()
                if snapshot.n_total > 0:
                    logger.info(
                        f"Refreshed: {snapshot.n_live} live, "
                        f"{snapshot.n_today - snapshot.n_live} upcoming today, "
                        f"{snapshot.n_tomorrow} tomorrow — "
                        f"{snapshot.n_total} total in 2-day window"
                    )
                else:
                    logger.info("Refresh complete — no real games available")
            except Exception as e:
                logger.error(f"Refresh cycle failed: {e}")

            time.sleep(self._refresh_interval)

    def stop(self):
        """Stop the refresh loop."""
        self._running = False
        logger.info("Live prediction worker stopped")

    @property
    def _refresh_interval(self) -> int:
        return self.engine._refresh_interval
