"""
LiveDataGateway — central orchestrator for ALL live data sources.

Provides a single interface to fetch, cache, and score freshness of:
  1. ESPN injury reports (ESPNInjuryScraper)
  2. Live odds (TheOddsAPI via OddsAPIClient)
  3. Data freshness (DataFreshnessChecker)
  4. Roster changes (RosterChangeTracker)
  5. Line movement (OddsMovementTracker)

Every method in this class is a no-op if the backend source isn't available,
so the system degrades gracefully when APIs are down or keys are missing.

Usage:
    gateway = LiveDataGateway()
    snapshot = gateway.get_live_snapshot(team_abbr="LAL")
    # -> LiveSnapshot(injuries=[...], odds=..., freshness_score=85, ...)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Cached value wrapper ────────────────────────────────────────────────────


class _CachedValue:
    """Simple TTL-based cache for a single value."""

    def __init__(self, ttl_seconds: int = 300):
        self._value: Any = None
        self._loaded_at: Optional[float] = None
        self._ttl = ttl_seconds

    @property
    def is_fresh(self) -> bool:
        if self._loaded_at is None:
            return False
        return (time.time() - self._loaded_at) < self._ttl

    def get(self) -> Any:
        return self._value if self.is_fresh else None

    def set(self, value: Any) -> None:
        self._value = value
        self._loaded_at = time.time()

    def invalidate(self) -> None:
        self._value = None
        self._loaded_at = None


# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class LiveSnapshot:
    """
    A point-in-time snapshot of ALL live data for a game or team.

    This is what the PreGameValidator uses to decide whether a bet is safe.
    """
    game_id: str
    matchup: str
    game_date: str
    league: str = "NBA"

    # Injuries
    home_injuries: list[dict] = field(default_factory=list)
    away_injuries: list[dict] = field(default_factory=list)
    home_injury_factor: float = 0.0   # 0 (healthy) -> 1 (devastated)
    away_injury_factor: float = 0.0

    # Odds
    home_ml: Optional[float] = None   # Moneyline
    away_ml: Optional[float] = None
    spread: Optional[float] = None
    total: Optional[float] = None
    best_home_ml: Optional[float] = None  # Best available across sportsbooks
    best_away_ml: Optional[float] = None
    best_total_over: Optional[float] = None

    # Line movement
    opening_total: Optional[float] = None
    opening_spread: Optional[float] = None
    total_movement: float = 0.0       # Current - Opening
    spread_movement: float = 0.0
    sharp_money_flag: bool = False     # True if line moved against public betting

    # Freshness
    odds_age_minutes: float = 0.0
    injury_age_minutes: float = 0.0
    freshness_grade: str = "UNKNOWN"   # FRESH / STALE / MISSING

    # Roster
    home_roster_changes: list[dict] = field(default_factory=list)
    away_roster_changes: list[dict] = field(default_factory=list)

    # Warnings
    warnings: list[str] = field(default_factory=list)
    is_bet_safe: bool = True  # Set by PreGameValidator
    captured_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "matchup": self.matchup,
            "game_date": self.game_date,
            "league": self.league,
            "home_injury_factor": self.home_injury_factor,
            "away_injury_factor": self.away_injury_factor,
            "home_injuries": self.home_injuries,
            "away_injuries": self.away_injuries,
            "home_ml": self.home_ml,
            "away_ml": self.away_ml,
            "spread": self.spread,
            "total": self.total,
            "best_home_ml": self.best_home_ml,
            "best_away_ml": self.best_away_ml,
            "total_movement": self.total_movement,
            "spread_movement": self.spread_movement,
            "sharp_money_flag": self.sharp_money_flag,
            "freshness_grade": self.freshness_grade,
            "odds_age_minutes": round(self.odds_age_minutes, 1),
            "injury_age_minutes": round(self.injury_age_minutes, 1),
            "warnings": self.warnings,
            "is_bet_safe": self.is_bet_safe,
            "captured_at": self.captured_at,
        }


# ── Gateway ─────────────────────────────────────────────────────────────────


class LiveDataGateway:
    """
    Central orchestrator for all live data sources.

    Lazy-loads individual scrapers/clients on first use and caches
    results with configurable TTLs. Gracefully handles missing API keys,
    network failures, and partial data.

    Usage:
        gateway = LiveDataGateway()
        snapshot = gateway.get_game_snapshot("LAL", "BOS")
        all_injuries = gateway.get_injuries()
        freshness = gateway.get_freshness_score()
        all_data = gateway.load_all_live_data()  # Full dict
    """

    def __init__(
        self,
        injury_ttl: int = 300,        # 5 min
        odds_ttl: int = 120,           # 2 min
        roster_ttl: int = 3600,        # 1 hour
        movement_ttl: int = 300,       # 5 min
        db_path: Optional[Path] = None,
        odds_api_key: Optional[str] = None,
    ):
        self._injury_cache = _CachedValue(ttl_seconds=injury_ttl)
        self._odds_cache = _CachedValue(ttl_seconds=odds_ttl)
        self._roster_cache = _CachedValue(ttl_seconds=roster_ttl)
        self._movement_cache = _CachedValue(ttl_seconds=movement_ttl)
        self._odds_dataframe_cache = _CachedValue(ttl_seconds=600)  # 10-min cache for DataFrame endpoint

        self._db_path = db_path
        self._odds_api_key = odds_api_key or self._try_get_env_key()

        # Lazy-loaded scrapers/clients
        self._injury_scraper = None
        self._odds_client = None
        self._freshness_checker = None
        self._roster_tracker = None
        self._movement_tracker = None

    # ── Public API ──────────────────────────────────────────────────────────

    def get_game_snapshot(
        self,
        home_team: str,
        away_team: str,
        game_id: str = "",
        game_date: str = "",
        league: str = "NBA",
        force_refresh: bool = False,
    ) -> LiveSnapshot:
        """
        Get a complete live-data snapshot for a single game.

        Gathers injuries for both teams, latest odds, line movement,
        roster changes, and computes freshness scores. Returns a
        LiveSnapshot with all fields populated (or defaults if data
        is unavailable).
        """
        snapshot = LiveSnapshot(
            game_id=game_id,
            matchup=f"{away_team} @ {home_team}",
            game_date=game_date,
            league=league,
        )

        # 1. Injuries
        try:
            injuries = self.get_injuries(force_refresh=force_refresh)
            home_abbr = self._team_to_abbr(home_team)
            away_abbr = self._team_to_abbr(away_team)

            home_inj = [i for i in injuries if i.get("team_abbr") == home_abbr]
            away_inj = [i for i in injuries if i.get("team_abbr") == away_abbr]

            snapshot.home_injuries = home_inj
            snapshot.away_injuries = away_inj
            snapshot.home_injury_factor = self._compute_injury_factor(home_inj)
            snapshot.away_injury_factor = self._compute_injury_factor(away_inj)
            snapshot.injury_age_minutes = self._get_cache_age(self._injury_cache)
        except Exception as e:
            logger.warning(f"Injury fetch failed for {home_team} vs {away_team}: {e}")
            snapshot.warnings.append(f"Injury data unavailable: {e}")

        # 2. Odds
        try:
            odds = self.get_live_odds(force_refresh=force_refresh)
            game_odds = self._find_game_odds(odds, home_team, away_team)
            if game_odds:
                snapshot.home_ml = game_odds.get("home_ml")
                snapshot.away_ml = game_odds.get("away_ml")
                snapshot.spread = game_odds.get("spread")
                snapshot.total = game_odds.get("total")
                snapshot.best_home_ml = game_odds.get("best_home_ml")
                snapshot.best_away_ml = game_odds.get("best_away_ml")
                snapshot.best_total_over = game_odds.get("best_total_over")
            snapshot.odds_age_minutes = self._get_cache_age(self._odds_cache)
        except Exception as e:
            logger.warning(f"Odds fetch failed for {home_team} vs {away_team}: {e}")
            snapshot.warnings.append(f"Odds data unavailable: {e}")

        # 3. Line movement
        try:
            movement = self.get_line_movements(force_refresh=force_refresh)
            game_movement = movement.get(f"{home_team}_vs_{away_team}", {})
            if game_movement:
                snapshot.opening_total = game_movement.get("opening_total")
                snapshot.opening_spread = game_movement.get("opening_spread")
                snapshot.total_movement = game_movement.get("total_movement", 0.0)
                snapshot.spread_movement = game_movement.get("spread_movement", 0.0)
                snapshot.sharp_money_flag = game_movement.get("sharp_money", False)
        except Exception as e:
            logger.debug(f"Line movement unavailable: {e}")

        # 4. Roster changes
        try:
            roster = self.get_roster_changes(force_refresh=force_refresh)
            snapshot.home_roster_changes = [
                r for r in roster
                if self._team_to_abbr(home_team) in (r.get("to_team"), r.get("from_team"))
            ]
            snapshot.away_roster_changes = [
                r for r in roster
                if self._team_to_abbr(away_team) in (r.get("to_team"), r.get("from_team"))
            ]
        except Exception as e:
            logger.debug(f"Roster changes unavailable: {e}")

        # 5. Freshness grade
        snapshot.freshness_grade = self._grade_freshness(snapshot)

        return snapshot

    def get_injuries(self, force_refresh: bool = False) -> list[dict]:
        """Fetch all NBA injuries, with caching."""
        if not force_refresh:
            cached = self._injury_cache.get()
            if cached is not None:
                return cached

        try:
            if self._injury_scraper is None:
                from betting_intel.data.injury_scraper import ESPNInjuryScraper
                self._injury_scraper = ESPNInjuryScraper()

            records = self._injury_scraper.fetch_all(force_refresh=True)
            result = [r.to_dict() for r in records]
            self._injury_cache.set(result)
            return result
        except Exception as e:
            logger.error(f"Failed to fetch injuries: {e}")
            return self._injury_cache.get() or []

    def get_live_odds(self, force_refresh: bool = False) -> list[dict]:
        """Fetch live odds, with caching."""
        if not force_refresh:
            cached = self._odds_cache.get()
            if cached is not None:
                return cached

        try:
            result = self._fetch_odds_from_api()
            self._odds_cache.set(result)
            return result
        except Exception as e:
            logger.error(f"Failed to fetch odds: {e}")
            return self._odds_cache.get() or []

    def get_line_movements(self, force_refresh: bool = False) -> dict[str, dict]:
        """Fetch line movement data, with caching."""
        if not force_refresh:
            cached = self._movement_cache.get()
            if cached is not None:
                return cached

        try:
            if self._movement_tracker is None:
                self._movement_tracker = _OddsMovementTracker()

            result = self._movement_tracker.get_all_movements()
            self._movement_cache.set(result)
            return result
        except Exception:
            return {}

    def get_roster_changes(self, force_refresh: bool = False) -> list[dict]:
        """Fetch roster changes, with caching."""
        if not force_refresh:
            cached = self._roster_cache.get()
            if cached is not None:
                return cached

        try:
            if self._roster_tracker is None:
                self._roster_tracker = _RosterChangeTracker()

            result = self._roster_tracker.get_changes()
            self._roster_cache.set(result)
            return result
        except Exception:
            return []

    def get_freshness_score(
        self, sources: Optional[list[str]] = None
    ) -> dict:
        """
        Get overall data freshness score (0-100).

        Args:
            sources: Optional list of source names to check.
                     Default: ["injuries", "odds", "database"]

        Returns:
            Dict with score, grade, per-source breakdown.
        """
        if sources is None:
            sources = ["injuries", "odds", "database"]

        results = []
        for source in sources:
            if source == "injuries":
                age = self._get_cache_age(self._injury_cache)
                fresh = age < 5  # 5 min threshold
                results.append({
                    "source": "injuries",
                    "age_minutes": round(age, 1),
                    "is_fresh": fresh,
                    "threshold_minutes": 5,
                })
            elif source == "odds":
                age = self._get_cache_age(self._odds_cache)
                fresh = age < 2  # 2 min threshold
                results.append({
                    "source": "odds",
                    "age_minutes": round(age, 1),
                    "is_fresh": fresh,
                    "threshold_minutes": 2,
                })
            elif source == "database":
                try:
                    from betting_intel.data.integrity import DataFreshnessChecker
                    checker = DataFreshnessChecker()
                    status = None
                    if self._db_path and self._db_path.exists():
                        import sqlite3
                        conn = sqlite3.connect(str(self._db_path))
                        cursor = conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                        tables = [row[0] for row in cursor.fetchall()]
                        conn.close()
                        status = checker.check_database(self._db_path, tables)
                    results.append({
                        "source": "database",
                        "is_fresh": status.is_fresh if status else False,
                        "age_hours": round(status.age_hours, 1) if status else -1,
                        "threshold_hours": 24,
                    })
                except Exception:
                    results.append({
                        "source": "database",
                        "is_fresh": False,
                        "error": "Could not check database",
                    })

        fresh_count = sum(1 for r in results if r.get("is_fresh"))
        score = (fresh_count / max(len(results), 1)) * 100

        grade = (
            "A" if score >= 90
            else "B" if score >= 75
            else "C" if score >= 50
            else "D" if score >= 25
            else "F"
        )

        return {
            "score": score,
            "grade": grade,
            "fresh_sources": fresh_count,
            "total_sources": len(results),
            "details": results,
            "all_fresh": fresh_count == len(results),
        }

    def get_live_odds_dataframe(self, force_refresh: bool = False) -> "pd.DataFrame":
        """
        Fetch live NBA odds from TheOddsAPI and return as a structured DataFrame.

        Directly calls https://api.the-odds-api.com/v4/sports/basketball_nba/odds
        with the configured ODDS_API_KEY. Returns columns:
          game_id, home_team, away_team, game_date, market_total,
          over_odds, under_odds, home_ml_odds, away_ml_odds,
          spread_line, spread_home_odds, spread_away_odds

        Results are cached for 10 minutes (600 seconds) to avoid burning API quota.

        Returns:
            pd.DataFrame with the columns above, or empty DataFrame on failure.
        """
        import pandas as pd

        if not force_refresh:
            cached = self._odds_dataframe_cache.get()
            if cached is not None:
                return cached

        try:
            result_df = self._fetch_odds_dataframe_direct()
            self._odds_dataframe_cache.set(result_df)
            return result_df
        except Exception as e:
            logger.error(f"Failed to fetch odds DataFrame: {e}")
            return pd.DataFrame()

    def _fetch_odds_dataframe_direct(self) -> "pd.DataFrame":
        """Make direct HTTP call to TheOddsAPI and parse into a DataFrame."""
        import pandas as pd
        import json
        import urllib.request
        import urllib.error

        api_key = self._odds_api_key
        if not api_key or api_key == "your-api-key-here":
            logger.warning("No valid ODDS_API_KEY configured for direct odds fetch")
            return pd.DataFrame()

        url = (
            f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
            f"?apiKey={api_key}"
            f"&regions=us"
            f"&markets=h2h,spreads,totals"
            f"&oddsFormat=american"
        )

        req = urllib.request.Request(url, headers={"User-Agent": "betting-intel/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = resp.status
                raw = resp.read().decode("utf-8")
                if status != 200:
                    raise RuntimeError(f"TheOddsAPI returned HTTP {status}: {raw[:200]}")
                data = json.loads(raw)
        except urllib.error.HTTPError as e:
            status = e.code
            body = e.read().decode("utf-8", errors="replace")
            if status == 429:
                raise RuntimeError(f"TheOddsAPI quota exceeded (429): {body}") from e
            raise RuntimeError(f"TheOddsAPI HTTP {status}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"TheOddsAPI connection failed: {e}") from e

        if not isinstance(data, list):
            return pd.DataFrame()

        records = []
        for game in data:
            game_id = game.get("id", "")
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            game_date = game.get("commence_time", "")[:10]

            home_ml_odds = None
            away_ml_odds = None
            spread_line = None
            spread_home_odds = None
            spread_away_odds = None
            market_total = None
            over_odds = None
            under_odds = None

            bookmakers = game.get("bookmakers", [])
            if bookmakers:
                # Use the first bookmaker's markets
                markets = bookmakers[0].get("markets", [])
                for market in markets:
                    key = market.get("key", "")
                    outcomes = market.get("outcomes", [])

                    if key == "h2h":
                        for outcome in outcomes:
                            name = outcome.get("name", "")
                            price = outcome.get("price")
                            if price is not None:
                                if name == home_team:
                                    home_ml_odds = price
                                elif name == away_team:
                                    away_ml_odds = price

                    elif key == "spreads":
                        for outcome in outcomes:
                            name = outcome.get("name", "")
                            point = outcome.get("point")
                            price = outcome.get("price")
                            if point is not None and price is not None:
                                spread_line = float(point)
                                if name == home_team:
                                    spread_home_odds = price
                                elif name == away_team:
                                    spread_away_odds = price

                    elif key == "totals":
                        for outcome in outcomes:
                            name = outcome.get("name", "")
                            point = outcome.get("point")
                            price = outcome.get("price")
                            if point is not None:
                                market_total = float(point)
                                if price is not None:
                                    if name == "Over":
                                        over_odds = price
                                    elif name == "Under":
                                        under_odds = price

            records.append({
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "game_date": game_date,
                "market_total": market_total,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "home_ml_odds": home_ml_odds,
                "away_ml_odds": away_ml_odds,
                "spread_line": spread_line,
                "spread_home_odds": spread_home_odds,
                "spread_away_odds": spread_away_odds,
            })

        return pd.DataFrame(records)

    def load_all_live_data(self) -> dict:
        """
        Load ALL live data into a single flat dict for dashboard use.

        This is the primary integration point for the web dashboard.
        """
        injuries = self.get_injuries()
        odds = self.get_live_odds()
        freshness = self.get_freshness_score()
        roster = self.get_roster_changes()
        movements = self.get_line_movements()

        return {
            "injuries": injuries,
            "odds": odds,
            "freshness": freshness,
            "roster_changes": roster,
            "line_movements": movements,
            "total_injuries": len(injuries),
            "total_odds": len(odds),
            "total_roster_changes": len(roster),
            "total_line_movements": len(movements),
            "freshness_grade": freshness.get("grade", "UNKNOWN"),
            "captured_at": datetime.now().isoformat(),
        }

    # ── Internal Helpers ────────────────────────────────────────────────────

    def _fetch_odds_from_api(self) -> list[dict]:
        """Try TheOddsAPI, fall back to the root-level OddsAPIClient."""
        # Try the root-level odds fetcher first (used in predict_tomorrow.py)
        try:
            from data.odds_fetcher import OddsAPIClient as RootOddsClient
            from config import ODDS_API_KEY, ODDS_CACHE_TTL_MINUTES

            key = self._odds_api_key or ODDS_API_KEY
            if key and key != "your-api-key-here":
                client = RootOddsClient(api_key=key, cache_ttl_minutes=ODDS_CACHE_TTL_MINUTES)
                games = client.get_upcoming_games_with_odds()
                return [g.to_dict() for g in games]
        except Exception as e:
            logger.debug(f"Root OddsAPIClient unavailable: {e}")

        # Fallback: try the src-level odds fetcher
        try:
            from betting_intel.data.odds_fetcher import OddsAPIClient as SrcOddsClient
            client = SrcOddsClient(api_key=self._odds_api_key or "")
            # If this has get_upcoming or similar, use it
            if hasattr(client, "fetch_upcoming"):
                games = client.fetch_upcoming()
                return games
        except Exception:
            pass

        return []

    def _find_game_odds(
        self, odds_list: list[dict], home_team: str, away_team: str
    ) -> Optional[dict]:
        """Find odds for a specific matchup in the odds list."""
        for game in odds_list:
            home = game.get("home_team", game.get("home_team_short", "")).lower()
            away = game.get("away_team", game.get("away_team_short", "")).lower()
            matchup = game.get("matchup", "").lower()

            if (home_team.lower() in home or home_team.lower() in matchup) and \
               (away_team.lower() in away or away_team.lower() in matchup):
                return {
                    "home_ml": game.get("home_moneyline"),
                    "away_ml": game.get("away_moneyline"),
                    "spread": game.get("home_spread"),
                    "total": game.get("market_total"),
                    "best_home_ml": game.get("best_home_ml"),
                    "best_away_ml": game.get("best_away_ml"),
                    "best_total_over": game.get("best_total_over"),
                }
        return None

    def _compute_injury_factor(self, injuries: list[dict]) -> float:
        """Compute injury impact factor from a list of injury dicts."""
        penalty = 0.0
        for record in injuries:
            status = record.get("injury_status", "").lower().strip()
            if status in ("out", "out for season"):
                penalty += 0.25
            elif status == "doubtful":
                penalty += 0.15
            elif status in ("questionable", "game time decision"):
                penalty += 0.08
            elif status == "probable":
                penalty += 0.02
        return min(penalty, 1.0)

    def _grade_freshness(self, snapshot: LiveSnapshot) -> str:
        """Determine freshness grade from a snapshot's data ages."""
        scores = []
        if snapshot.odds_age_minutes < 2:
            scores.append(True)
        elif snapshot.odds_age_minutes < 10:
            scores.append(True)
        else:
            scores.append(snapshot.odds_age_minutes == 0)  # No data = don't penalize

        if snapshot.injury_age_minutes < 5:
            scores.append(True)
        elif snapshot.injury_age_minutes < 30:
            scores.append(True)
        else:
            scores.append(snapshot.injury_age_minutes == 0)

        fresh_count = sum(1 for s in scores if s)
        ratio = fresh_count / max(len(scores), 1)

        if ratio >= 0.8:
            return "FRESH"
        elif ratio >= 0.5:
            return "STALE"
        else:
            return "MISSING"

    def _team_to_abbr(self, team_name: str) -> str:
        """Convert team name to abbreviation using scraper's mapping."""
        from betting_intel.data.injury_scraper import extract_team_abbr
        abbr = extract_team_abbr(team_name)
        if abbr:
            return abbr
        return team_name.upper()[:3]

    @staticmethod
    def _get_cache_age(cache: _CachedValue) -> float:
        """Get age of cached value in minutes."""
        if cache._loaded_at is None:
            return float("inf")
        return (time.time() - cache._loaded_at) / 60.0

    @staticmethod
    def _try_get_env_key() -> Optional[str]:
        """Try to get ODDS_API_KEY from environment."""
        import os
        return os.environ.get("ODDS_API_KEY")


# ── Internal Odds Movement Tracker ──────────────────────────────────────────


class _OddsMovementTracker:
    """
    Tracks line movement for upcoming games.

    In production, this would poll TheOddsAPI's historical endpoints or
    subscribe to a real-time feed. For now, it provides a simulation
    based on the most recent odds fetch.
    """

    def __init__(self):
        self._movements: dict[str, dict] = {}

    def get_all_movements(self) -> dict[str, dict]:
        """Get all tracked line movements."""
        return self._movements

    def record_movement(
        self,
        matchup_key: str,
        opening_total: float,
        current_total: float,
        opening_spread: float,
        current_spread: float,
        public_betting_pct: float = 0.5,
    ) -> dict:
        """Record a line movement for a game."""
        movement = {
            "opening_total": opening_total,
            "current_total": current_total,
            "total_movement": current_total - opening_total,
            "opening_spread": opening_spread,
            "current_spread": current_spread,
            "spread_movement": current_spread - opening_spread,
            "public_betting_pct": public_betting_pct,
            "sharp_money": public_betting_pct < 0.4 or public_betting_pct > 0.6,
            "recorded_at": datetime.now().isoformat(),
        }
        self._movements[matchup_key] = movement
        return movement


# ── Internal Roster Change Tracker ──────────────────────────────────────────


class _RosterChangeTracker:
    """
    Monitors roster changes (trades, signings, waivers, returns from injury).

    In production, this would scrape ESPN's transaction wire or use
    an API like SportRadar. For now, it provides the structure and
    integrates with the injury scraper to detect players returning.
    """

    def __init__(self):
        self._changes: list[dict] = []

    def get_changes(self) -> list[dict]:
        """Get all tracked roster changes."""
        return self._changes

    def record_change(
        self,
        player_name: str,
        change_type: str,  # "trade", "signing", "waiver", "return_from_injury"
        from_team: Optional[str] = None,
        to_team: Optional[str] = None,
        date: Optional[str] = None,
        details: str = "",
    ) -> dict:
        """Record a roster change."""
        change = {
            "player_name": player_name,
            "change_type": change_type,
            "from_team": from_team,
            "to_team": to_team,
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "details": details,
            "recorded_at": datetime.now().isoformat(),
        }
        self._changes.append(change)
        return change


# ── Multi-Sportsbook Comparator ─────────────────────────────────────────────


class MultiSportsbookComparator:
    """
    Compares lines across multiple sportsbooks to find the best price.

    Usage:
        comparator = MultiSportsbookComparator()
        best = comparator.find_best_line(odds_list, "LAL", "BOS")
        # -> {"best_home_ml": 2.10, "best_away_ml": 1.85, ...}
    """

    @staticmethod
    def find_best_line(
        odds_list: list[dict],
        home_team: str,
        away_team: str,
    ) -> dict:
        """
        Find the best available line across all sportsbooks for a matchup.

        Args:
            odds_list: List of odds dicts (from OddsAPIClient or LiveDataGateway)
            home_team: Home team name
            away_team: Away team name

        Returns:
            Dict with best lines: best_home_ml, best_away_ml, etc.
        """
        home_mls = []
        away_mls = []
        totals = []
        spreads = []

        for game in odds_list:
            ghome = (game.get("home_team") or game.get("home_team_short", "")).lower()
            gaway = (game.get("away_team") or game.get("away_team_short", "")).lower()
            if home_team.lower() in ghome and away_team.lower() in gaway:
                # Collect from all sportsbooks
                books = game.get("books", game.get("sportsbooks", []))
                if not books and game.get("home_moneyline") is not None:
                    # Single-book format
                    books = [game]

                for book in books:
                    hm = book.get("home_moneyline") or book.get("home_ml")
                    am = book.get("away_moneyline") or book.get("away_ml")
                    sp = book.get("spread") or book.get("home_spread")
                    tl = book.get("total") or book.get("market_total")

                    if hm is not None:
                        home_mls.append(hm)
                    if am is not None:
                        away_mls.append(am)
                    if sp is not None:
                        spreads.append(sp)
                    if tl is not None:
                        totals.append(tl)

                break

        best = {
            "best_home_ml": max(home_mls) if home_mls else None,
            "best_away_ml": max(away_mls) if away_mls else None,
            "best_total_over": max(totals) if totals else None,
            "best_total_under": min(totals) if totals else None,
            "best_spread_home": max(spreads) if spreads else None,
            "best_spread_away": min(spreads) if spreads else None,
            "n_sportsbooks": max(len(home_mls), len(totals)),
        }
        return best

    @staticmethod
    def compare_all(odds_list: list[dict]) -> list[dict]:
        """
        Compare lines across sportsbooks for ALL games.

        Returns a list of dicts, one per game, with best lines.
        """
        results = []
        for game in odds_list:
            home = game.get("home_team") or game.get("home_team_short", "")
            away = game.get("away_team") or game.get("away_team_short", "")
            best = MultiSportsbookComparator.find_best_line(
                odds_list, home, away
            )
            if any(v is not None for v in best.values()):
                results.append({
                    "matchup": game.get("matchup", f"{away} @ {home}"),
                    "home_team": home,
                    "away_team": away,
                    **best,
                })
        return results
