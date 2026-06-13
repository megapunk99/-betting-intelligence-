"""
LivePredictionEngine — real-time NBA prediction system.

NBA-only. Single-sport focus means:
  - One data pipeline (TheOddsAPI basketball_nba)
  - One prediction strategy (total points regression + home win classification)
  - One model (RobustPredictionSystem with NBA-specific features)
  - One team name mapping (ODDS_TO_SHORT_NAME)

CRITICAL DESIGN RULE: TheOddsAPI is ONLY called when force_refresh=True
is explicitly passed. Page loads, WebSocket connections, and all
read-only operations return cached data only. Zero automatic API calls.

Architecture:
  1. ONLY real data — if TheOddsAPI returns nothing, we show nothing
  2. User-initiated refresh only — no background polling
  3. Live detection: games with commence_time in the past = in-play
  4. 2-day rolling window: today's games + tomorrow's games combined
  5. ML predictions computed fresh each refresh cycle using real market lines
  6. No synthetic/random/fake games EVER
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

    # League / sport identification
    league: str = "NBA"
    sport_group: str = "Basketball"

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

    # ML predictions — moneyline edge (filled by MarketInefficiencySystem)
    predicted_total: Optional[float] = None  # home_win_prob (0-1) from robust system
    edge_pct: Optional[float] = None         # predicted market error for moneyline
    direction: Optional[str] = None           # "home" or "away"
    confidence: Optional[str] = None          # "high", "medium", "low", "neutral"

    # ML predictions — totals edge (filled by TotalsRegressor)
    total_prediction: Optional[float] = None  # predicted total points (e.g. 225.5)
    total_edge_pct: Optional[float] = None    # edge on the total (positive = over)
    total_direction: Optional[str] = None     # "over" or "under" or "neutral"
    total_confidence: Optional[str] = None    # "high", "medium", "low"

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

    # Kelly stake (computed by KellyStaker in _predict_with_robust_system)
    stake_dollars: float = 0.0

    # Feature importance (how the model arrived at this prediction)
    feature_importance: Optional[dict[str, float]] = None  # {human_readable_name: importance_weight}

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
    chart_data: Optional[dict] = None

    # Fields to exclude from serialization (chart_data, internal state)
    _exclude_from_dict: set = field(default_factory=lambda: {"chart_data", "_exclude_from_dict"})

    def __post_init__(self):
        self.n_live = len(self.live_games)
        self.n_today = len(self.today_games)
        self.n_tomorrow = len(self.tomorrow_games)
        self.n_total = len(self.next_two_days)
        self.chart_data = self._build_chart_data()

    def _build_chart_data(self) -> dict:
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
                    "matchup": g.matchup,
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
        odds_api_key_fallback: Optional[str] = None,
        refresh_interval: int = PREDICTION_REFRESH_INTERVAL,
        model_dir: Optional[Path] = None,
    ):
        self._odds_api_key = odds_api_key or os.getenv("ODDS_API_KEY", "")
        self._odds_api_key_fallback = odds_api_key_fallback or os.getenv("ODDS_API_KEY_FALLBACK", "")
        self._refresh_interval = refresh_interval
        self._model_dir = model_dir

        # Thread-safe cache
        self._lock = Lock()               # Protects snapshot, odds cache, auto-resolve timestamp
        self._snapshot: Optional[LivePredictionSnapshot] = None
        self._last_refresh: float = 0.0
        self._last_odds_fetch: float = 0.0
        self._cached_odds_raw: Optional[list[dict]] = None

        # Auto-resolve timestamp (protected by _lock)
        self._last_auto_resolve: Optional[str] = None

        # Model lock — protects lazy-loaded model + robust system + totals model
        # (all modified outside _lock)
        self._model_lock = Lock()
        self._model = None
        self._feature_cols: list[str] = []
        self._robust_system: Any = None
        self._robust_system_fitted: bool = False

        # Totals regression model (XGBoost/LightGBM trained on total_points)
        self._totals_model: Any = None
        self._totals_fitted: bool = False
        self._totals_mae: float = 12.0  # Default MAE (used for confidence bounds)
        self._totals_std: float = 15.0  # Default std (used for normalization)

        # Scraper timeout — made an instance attribute so tests can override it
        self._scraper_timeout = 15.0  # Combined timeout for both scrapers (internal are 12s each)

        # Kelly Criterion bankroll manager
        from betting_intel.recommendations.staking import KellyStaker
        self._kelly_staker = KellyStaker(
            initial_bankroll=10_000.0,
            kelly_fraction=0.25,
        )

        # Market odds store — logs real odds every refresh, provides historical market data
        from betting_intel.db.market_odds_store import MarketOddsStore
        self._market_odds_store = MarketOddsStore()
        self._market_odds_store.ensure_table()

        logger.info(
            "LivePredictionEngine initialized — zero synthetic games. "
            f"Refresh interval: {refresh_interval}s"
        )

    @property
    def last_auto_resolve(self) -> Optional[str]:
        """ISO-8601 timestamp of the last auto-resolve run, or None."""
        with self._lock:
            return self._last_auto_resolve

    # ── Public API ────────────────────────────────────────────────────────

    def get_snapshot(self, force_refresh: bool = False) -> LivePredictionSnapshot:
        if not force_refresh:
            with self._lock:
                if self._snapshot is not None:
                    return self._snapshot
                logger.debug("No cached snapshot — returning empty (user must refresh)")
                return LivePredictionSnapshot()

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
        snapshot = self.get_snapshot(force_refresh=force_refresh)
        return snapshot.live_games

    def get_today_games(self, force_refresh: bool = False) -> list[LiveGame]:
        snapshot = self.get_snapshot(force_refresh=force_refresh)
        return snapshot.today_games

    def get_tomorrow_games(self, force_refresh: bool = False) -> list[LiveGame]:
        snapshot = self.get_snapshot(force_refresh=force_refresh)
        return snapshot.tomorrow_games

    def get_next_two_days(self, force_refresh: bool = False) -> list[LiveGame]:
        snapshot = self.get_snapshot(force_refresh=force_refresh)
        return snapshot.next_two_days

    def refresh_now(self) -> LivePredictionSnapshot:
        return self.get_snapshot(force_refresh=True)

    def clear_cache(self):
        # Must hold BOTH locks since model state is protected by _model_lock
        # while snapshot/odds state is protected by _lock
        with self._model_lock:
            self._model = None
            self._feature_cols = []
            self._robust_system = None
            self._robust_system_fitted = False
            self._totals_model = None
            self._totals_fitted = False
        with self._lock:
            self._snapshot = None
            self._last_refresh = 0.0
            self._last_odds_fetch = 0.0
            self._cached_odds_raw = None

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
        with self._lock:
            return self._snapshot is not None and self._snapshot.n_total > 0

    # ── Odds Fetching ─────────────────────────────────────────────────────

    def _has_valid_api_key(self) -> bool:
        primary_valid = bool(self._odds_api_key) and self._odds_api_key not in (
            "your-api-key-here", "", "REPLACE_ME_WITH_YOUR_ODDS_API_KEY"
        )
        fallback_valid = bool(self._odds_api_key_fallback) and self._odds_api_key_fallback not in (
            "your-api-key-here", "", "REPLACE_ME_WITH_YOUR_ODDS_API_KEY"
        )
        return primary_valid or fallback_valid

    def _fetch_realtime_odds(self) -> list[dict]:
        now = time.time()

        # Thread-safe check of the odds cache (fast path — no API call)
        with self._lock:
            if self._cached_odds_raw is not None and (now - self._last_odds_fetch) < ODDS_CACHE_TTL_SECONDS:
                return self._cached_odds_raw

        if self._has_valid_api_key():
            logger.info("Valid ODDS_API_KEY found — trying TheOddsAPI first")
            theodds_data = self._fetch_via_theoddsapi()
            if theodds_data:
                with self._lock:
                    self._cached_odds_raw = theodds_data
                    self._last_odds_fetch = now
                return theodds_data
            logger.info("TheOddsAPI returned no data — trying free scrapers")
        else:
            logger.info("No valid ODDS_API_KEY — using free scrapers")

        # ── Parallel scraper execution ─────────────────────────────────
        # Run ESPN and DraftKings scrapers CONCURRENTLY with a single
        # combined timeout. Previously they ran sequentially: ESPN (25s)
        # then DraftKings (30s) = up to 55s blocking the entire engine.
        # Now both run in parallel and complete within 20s total.
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

        scraper_timeout = self._scraper_timeout  # Configurable for tests

        scraper_results: dict[str, list[dict]] = {}
        pool = ThreadPoolExecutor(max_workers=2)
        try:
            espn_future = pool.submit(self._fetch_stealth_scraper)
            dk_future = pool.submit(self._fetch_draftkings_odds)
            processed: set[str] = set()  # Track which futures we've collected results from

            try:
                for future in as_completed([espn_future, dk_future], timeout=scraper_timeout):
                    try:
                        data = future.result()  # as_completed guarantees it's done
                    except Exception as e:
                        # Individual scraper failed (ImportError, RuntimeError, etc.)
                        logger.debug(f"Scraper result error: {e}")
                        continue
                    key = "espn" if future == espn_future else "dk"
                    scraper_results[key] = data if data else []
                    processed.add(key)
                    logger.info(f"{'ESPN' if key == 'espn' else 'DraftKings'} scraper: {len(data) if data else 0} games")
            except FuturesTimeoutError:
                logger.warning(f"Scrapers timed out after {scraper_timeout}s — using partial results")

            # Collect results from any futures that finished after as_completed timeout
            for future, key, name in [
                (espn_future, "espn", "ESPN"),
                (dk_future, "dk", "DraftKings"),
            ]:
                if key in processed:
                    continue  # Already collected above
                if future.done():
                    try:
                        data = future.result()
                        scraper_results[key] = data if data else []
                        logger.info(f"{name} scraper (late): {len(data) if data else 0} games")
                    except Exception as e:
                        logger.debug(f"{name} scraper late result error: {e}")
                        scraper_results.setdefault(key, [])
                else:
                    logger.warning(f"{name} scraper did not finish — results discarded")
                    scraper_results.setdefault(key, [])
        finally:
            # Shutdown without blocking — don't wait for hanging threads.
            # The scrapers have internal 12s timeouts so threads will terminate
            # on their own, but we don't block the refresh cycle for them.
            pool.shutdown(wait=False)

        stealth_data = scraper_results.get("espn", [])
        dk_data = scraper_results.get("dk", [])

        merged = self._merge_odds_sources(stealth_data, dk_data)
        with self._lock:
            if merged:
                self._cached_odds_raw = merged
            else:
                self._cached_odds_raw = []
            self._last_odds_fetch = now
            return self._cached_odds_raw

    def _fetch_via_theoddsapi(self) -> list[dict]:
        if not self._has_valid_api_key():
            return []

        from betting_intel.live.sport_configs import get_active_sports, SPORT_KEY_TO_CONFIG
        active_sports = get_active_sports()

        if not active_sports:
            active_sports = [SPORT_KEY_TO_CONFIG.get("basketball_nba")]
            if not active_sports or not active_sports[0]:
                return []

        # Try primary key first
        result = self._fetch_via_theoddsapi_with_key(self._odds_api_key, active_sports)

        # If primary returned no data AND we have a fallback, retry with fallback
        if not result and self._odds_api_key_fallback:
            logger.warning("Primary ODDS_API_KEY returned no data — trying fallback key")
            key_label = self._odds_api_key_fallback[:8] + "..."
            logger.info(f"Using fallback key: {key_label}")
            result = self._fetch_via_theoddsapi_with_key(self._odds_api_key_fallback, active_sports)

        return result

    def _fetch_via_theoddsapi_with_key(self, api_key: str, active_sports: list) -> list[dict]:
        """
        Fetch odds from TheOddsAPI using a specific API key.

        This helper exists so the fallback key logic can retry the entire
        fetch with a different key when the primary key returns 401.
        """
        import urllib.request
        import urllib.error

        all_games: list[dict] = []
        total_quota = "?"
        key_label = api_key[:8] + "..." if len(api_key) > 8 else api_key

        for sport in active_sports:
            try:
                markets_str = ",".join(sport.markets_to_fetch)
                url = (
                    f"https://api.the-odds-api.com/v4/sports/{sport.sport_key}/odds"
                    f"?apiKey={api_key}"
                    f"&regions=us,us2,eu,uk,au"
                    f"&markets={markets_str}"
                    f"&oddsFormat=american"
                    f"&dateFormat=iso"
                )
                req = urllib.request.Request(url, headers={"User-Agent": "betting-intel-live/3.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)

                if isinstance(data, list) and len(data) > 0:
                    for game in data:
                        game["_sport_config_key"] = sport.sport_key
                    all_games.extend(data)
                    remaining = resp.headers.get("x-requests-remaining", "?")
                    if remaining != "?":
                        total_quota = remaining
                    logger.info(f"{sport.display_name}: {len(data)} games (quota: {remaining}, key: {key_label})")
                else:
                    logger.info(f"{sport.display_name}: no games available")

            except urllib.error.HTTPError as e:
                if e.code == 401:
                    logger.warning(f"ODDS_API_KEY ({key_label}) returned 401 — invalid key")
                    return []  # Let caller handle fallback
                if e.code == 429:
                    logger.warning(f"TheOddsAPI quota exceeded (429) — key: {key_label}")
                    break
                logger.debug(f"{sport.display_name}: HTTP {e.code} (skipping)")
                continue
            except urllib.error.URLError as e:
                logger.debug(f"{sport.display_name}: connection failed ({e})")
                continue
            except Exception as e:
                logger.debug(f"{sport.display_name}: error ({e})")
                continue

        logger.info(f"TheOddsAPI ({key_label}) total: {len(all_games)} games across {len(active_sports)} sports")
        return all_games

    def _fetch_stealth_scraper(self) -> list[dict]:
        logger.info("Attempting ESPN stealth scraper...")
        try:
            from betting_intel.data.stealth_scraper import StealthBrowser
            scraped = StealthBrowser.sync_scrape_live_odds(
                odds_api_key=self._odds_api_key,
                timeout=12,
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
        logger.info("Attempting DraftKings scraper...")
        try:
            from betting_intel.data.draftkings_scraper import DraftKingsScraper
            scraped = DraftKingsScraper.scrape(timeout=12)
            if scraped:
                logger.info(f"DraftKings scraper: {len(scraped)} games")
                return scraped
            logger.info("DraftKings scraper: returned no data")
            return []
        except ImportError as e:
            logger.debug(f"DraftKings scraper: not available ({e})")
            return []
        except Exception as e:
            logger.warning(f"DraftKings scraper failed: {e}")
            return []

    def _merge_odds_sources(self, espn_data: list[dict], dk_data: list[dict]) -> list[dict]:
        if not espn_data and not dk_data:
            return []
        if not dk_data:
            return espn_data
        if not espn_data:
            return dk_data

        dk_by_matchup: dict[str, list[dict]] = {}
        for dk_game in dk_data:
            home = dk_game.get("home_team", "")
            away = dk_game.get("away_team", "")
            if home and away:
                key = f"{home}|{away}"
                if key not in dk_by_matchup:
                    dk_by_matchup[key] = []
                dk_by_matchup[key].append(dk_game)

        merged: list[dict] = []
        seen_matchups: set[str] = set()

        for espn_game in espn_data:
            home = espn_game.get("home_team", "")
            away = espn_game.get("away_team", "")
            key = f"{home}|{away}"
            seen_matchups.add(key)
            merged_game = dict(espn_game)
            merged_books = list(merged_game.get("bookmakers", []))
            dk_games = dk_by_matchup.get(key, [])
            for dk_game in dk_games:
                dk_books = dk_game.get("bookmakers", [])
                merged_books.extend(dk_books)
            merged_game["bookmakers"] = merged_books
            merged.append(merged_game)

        for dk_game in dk_data:
            home = dk_game.get("home_team", "")
            away = dk_game.get("away_team", "")
            key = f"{home}|{away}"
            if key not in seen_matchups:
                merged.append(dk_game)
                seen_matchups.add(key)

        logger.info(f"Merged odds: {len(merged)} total games ({len(espn_data)} from ESPN, {len(dk_data)} from DraftKings)")
        return merged

    # ── Robust System ─────────────────────────────────────────────────────

    @property
    def robust_system(self) -> Any:
        """The fitted RobustPredictionSystem, or None."""
        with self._model_lock:
            return self._robust_system

    @property
    def robust_system_summary(self) -> dict:
        """Get a summary of the robust prediction system state."""
        with self._model_lock:
            if self._robust_system is None:
                return {"fitted": False, "status": "not_initialized"}
            if not self._robust_system_fitted:
                return {"fitted": False, "status": "not_fitted"}
            try:
                return self._robust_system.get_summary()
            except Exception:
                return {"fitted": True, "status": "error_reading_summary"}

    @property
    def kelly_staker(self) -> Any:
        """The KellyCriterion staker instance."""
        return self._kelly_staker

    def _build_robust_system(self) -> bool:
        """
        Build and train the MarketInefficiencySystem on historical NBA data.

        v5.0 — Market Inefficiency Training
        ────────────────────────────────────
        Instead of simply training on home_win (binary outcome), this trains
        the model to predict MARKET ERROR — the difference between actual
        outcomes and market-implied probabilities.

        For each historical game:
          1. Compute market_implied_home_prob from ELO proxy
          2. Compute market_error = home_win - market_implied_home_prob
          3. Train classifier on home_win (existing capability)
          4. Train regressor on market_error (new capability)
          5. At inference: final_prob = market_implied_prob + predicted_error

        The model learns: "given these features, how much is the market wrong?"
        This is fundamentally different from "given these features, who wins?"

        Returns:
            True if the system was built and fitted successfully.
        """
        try:
            from betting_intel.models.robust_ensemble import (
                RobustPredictionSystem,
                MarketInefficiencySystem,
            )
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.data.features import FeatureEngineer
            from betting_intel.features.market_inefficiency import (
                compute_market_inefficiency_targets,
            )

            logger.info(
                "Building MarketInefficiencySystem on historical NBA data "
                "(v5.0 — market inefficiency training)..."
            )

            # ── Load and engineer features ──────────────────────────────
            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is None or raw_df.empty:
                logger.warning("No historical NBA data available for robust system")
                return False

            games_df = loader.build_game_dataset(raw_df)
            raw_df = loader.compute_rest_days(raw_df)
            fe = FeatureEngineer()
            features_df = fe.build_all_features(games_df, raw_df)

            if features_df is None or features_df.empty:
                logger.warning("Feature engineering produced no data")
                return False

            # ── Query MarketOddsStore for real historical market data ─────
            # Tier 1: use real moneyline odds from the store (highest quality).
            # Tier 2: fall back to ELO proxy when no real data exists.
            try:
                store_start = features_df["GAME_DATE"].min()
                store_end = features_df["GAME_DATE"].max()
                if hasattr(store_start, 'strftime'):
                    store_start = store_start.strftime("%Y-%m-%d")
                    store_end = store_end.strftime("%Y-%m-%d")
                else:
                    store_start = str(store_start)[:10]
                    store_end = str(store_end)[:10]

                raw_overrides = self._market_odds_store.get_market_probs_for_date_range(
                    start_date=store_start,
                    end_date=store_end,
                )

                # Remap keys from full names ("Boston Celtics") to short names ("Celtics")
                # because the features DataFrame uses short names in TEAM_NAME columns.
                from betting_intel.data.odds_fetcher import ODDS_TO_SHORT_NAME
                market_prob_overrides: dict[tuple[str, str, str], float] = {}
                for (home_full, away_full, game_date), prob in raw_overrides.items():
                    home_short = ODDS_TO_SHORT_NAME.get(home_full, home_full.split()[-1] if " " in home_full else home_full)
                    away_short = ODDS_TO_SHORT_NAME.get(away_full, away_full.split()[-1] if " " in away_full else away_full)
                    market_prob_overrides[(home_short, away_short, game_date)] = prob

                logger.info(
                    f"Loaded {len(market_prob_overrides)} real market probs from store "
                    f"({store_start} to {store_end})"
                )
            except Exception:
                logger.debug("Failed to query MarketOddsStore — using ELO proxy only", exc_info=True)
                market_prob_overrides = None

            # ── Add home_win target column ───────────────────────────
            # compute_market_inefficiency_targets() requires 'home_win' to
            # compute market_error = home_win - market_implied_home_prob.
            # Derive it from point_diff (which is team_pts_home - team_pts_away
            # from build_game_dataset). Positive = home win, negative = away win.
            if "home_win" not in features_df.columns:
                if "point_diff" in features_df.columns:
                    features_df["home_win"] = (
                        features_df["point_diff"] > 0
                    ).astype(int)
                elif "WL_home" in features_df.columns:
                    features_df["home_win"] = (
                        features_df["WL_home"] == "W"
                    ).astype(int)
                else:
                    logger.warning(
                        "Cannot derive home_win — no point_diff or WL_home column. "
                        "MarketInefficiencySystem can't be trained."
                    )
                    return False

            # ── Compute market inefficiency targets ─────────────────────
            # This adds: market_prob_source, market_implied_home_prob, market_error, etc.
            # Passes real odds from the store when available, falling back to ELO proxy.
            features_df = compute_market_inefficiency_targets(
                features_df,
                market_prob_overrides=market_prob_overrides,
            )

            # ── Log proxy source distribution ──────────────────────────
            if "market_proxy_source" in features_df.columns:
                source_counts = features_df["market_proxy_source"].value_counts().to_dict()
                total_games = sum(source_counts.values())
                real_odds_count = source_counts.get("real_odds", 0)
                elo_proxy_count = source_counts.get("elo_proxy", 0)
                logger.info(
                    f"Market proxy source distribution: {source_counts} "
                    f"(real_odds={real_odds_count}/{total_games} = "
                    f"{real_odds_count/max(total_games,1)*100:.1f}%)"
                )
                if real_odds_count == 0:
                    logger.warning(
                        "⚠  ZERO games have real market odds. "
                        "The model will train using ELO proxy only. "
                        "Run the live engine refresh cycle several times to "
                        "accumulate odds data, or backfill via the CLI."
                    )
                elif real_odds_count < total_games * 0.1:
                    logger.info(
                        f"  Only {real_odds_count}/{total_games} games have real odds "
                        f"({real_odds_count/max(total_games,1)*100:.1f}%). "
                        f"ELO proxy fills the rest. Coverage will improve "
                        f"as the engine accumulates more refresh cycles."
                    )
                else:
                    logger.info(
                        f"  ✅  Strong real odds coverage: "
                        f"{real_odds_count}/{total_games} games "
                        f"({real_odds_count/max(total_games,1)*100:.1f}%)"
                    )

            # Build feature matrix and target
            # CRITICAL: Use FeatureEngineer.select_features() to exclude raw
            # box score columns (team_pts_home, team_fgm_home, etc.) that would
            # leak the target. The select_features() method is the authoritative
            # source for clean feature selection.
            import numpy as np

            clean_feature_cols = fe.select_features(features_df)

            # Also exclude market inefficiency target columns that were added
            # AFTER feature engineering (by compute_market_inefficiency_targets)
            _market_target_cols = {
                "market_implied_home_prob", "market_error",
                "abs_market_error", "market_error_clipped",
                "market_error_binary", "total_market_error",
                "weighted_market_error", "elo_error",
                "market_error_ma_5g", "market_error_ma_10g",
                "market_error_trend_home", "recent_edge_streak",
            }
            feature_cols = [c for c in clean_feature_cols if c not in _market_target_cols]

            if len(feature_cols) < 3:
                logger.warning(f"Only {len(feature_cols)} feature cols — too few for robust system")
                return False

            X = features_df[feature_cols].fillna(0).values
            n_samples = len(X)
            if n_samples < 200:
                logger.warning(f"Only {n_samples} samples — need at least 200 for robust system")
                return False

            # ── Build targets ───────────────────────────────────────────
            if "home_win" not in features_df.columns:
                logger.warning("`home_win` column not found — cannot train robust system")
                return False

            y_binary = features_df["home_win"].values.astype(int)

            # Market-implied probabilities (from store, ELO proxy, or fallback)
            market_probs = features_df["market_implied_home_prob"].values.astype(float)

            # ── Train market inefficiency system ────────────────────────
            logger.info(
                f"Training MarketInefficiencySystem on {n_samples} samples "
                f"with {len(feature_cols)} features..."
            )

            system = MarketInefficiencySystem(
                calibrate=True,
                n_folds=5,
                min_train_samples=50,
                random_state=42,
            )

            system.fit(
                X, y_binary,
                market_probs=market_probs,
                feature_names=feature_cols,
                verbose=True,
            )

            with self._model_lock:
                self._robust_system = system
                self._robust_system_fitted = True

            summary = system.get_summary()
            logger.info(
                f"MarketInefficiencySystem built: "
                f"{summary.get('n_models', '?')} classifier models, "
                f"{summary.get('n_error_models', 0)} error regressors, "
                f"Brier={summary.get('calibrated_brier', 'N/A')}"
            )

            return True

        except Exception as e:
            logger.warning(f"Failed to build MarketInefficiencySystem: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _predict_with_robust_system(self, games: list[LiveGame]) -> list[LiveGame]:
        """
        Predict games using the MarketInefficiencySystem.

        v5.0 — Market Inefficiency Prediction
        ──────────────────────────────────────
        The system is trained to predict MARKET ERROR (how much the market
        is wrong) rather than raw home_win probability.

        Flow for each game:
          1. Compute market-implied home probability from moneyline odds
          2. Build feature vector from historical data
          3. Call system which:
             a. Gets classifier's home_win prob (trained on home_win)
             b. Gets error regressor's predicted market error
             c. Blends: final_prob = market_prob + predicted_error
          4. edge_pct = predicted_error (directly — no second computation)

        This means edge_pct IS the market inefficiency prediction, not a
        post-hoc calculation. The model directly outputs "how wrong is the
        market?"
        """
        with self._model_lock:
            if not self._robust_system or not self._robust_system_fitted:
                return games
            robust_system = self._robust_system  # Local reference for lock-free access

        import numpy as np
        import pandas as pd
        from betting_intel.recommendations.staking import american_to_decimal
        from betting_intel.features.market_inefficiency import (
            american_to_implied_prob,
            remove_vig,
        )

        try:
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.data.features import FeatureEngineer

            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is None or raw_df.empty:
                return games

            games_df = loader.build_game_dataset(raw_df)
            raw_df = loader.compute_rest_days(raw_df)
            fe = FeatureEngineer()
            features_df = fe.build_all_features(games_df, raw_df)

            if features_df is None or features_df.empty:
                return games

            for game in games:
                try:
                    # ── Step 1: Compute market-implied probability ───────
                    market_prob = None
                    if game.home_ml is not None and game.away_ml is not None:
                        home_implied = american_to_implied_prob(game.home_ml)
                        away_implied = american_to_implied_prob(game.away_ml)
                        # Remove vig to get true market probability
                        market_prob, _ = remove_vig(home_implied, away_implied)

                    # ── Step 2: Build feature vector ─────────────────────
                    feat = self._build_feature_vector(
                        game.home_team_short,
                        game.away_team_short,
                        features_df,
                    )
                    if feat is None:
                        continue

                    X_pred = feat.values.reshape(1, -1)

                    # ── Step 3: Predict with market awareness ────────────
                    result = robust_system.predict_with_details(
                        X_pred,
                        market_prob=market_prob,
                    )

                    home_win_prob = result.home_win_prob

                    # ── Step 4: Capture feature importance ─────────────
                    # Store the top features that drove this prediction so
                    # the dashboard can show human-readable reasoning like
                    # "Celtics avg scoring (L5): +3.7%" — what the model
                    # considers most important for THIS game.
                    if result.feature_importance:
                        # Take top 8 to have enough for filtering later
                        top_features = dict(
                            sorted(
                                result.feature_importance.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )[:8]
                        )
                        game.feature_importance = top_features

                    # ── Step 5: Apply predictions to game ────────────────
                    # edge_pct IS the predicted market error (direct output)
                    if market_prob is not None:
                        # The predicted_error IS the edge
                        predicted_error = result.edge_pct if result.edge_pct is not None else 0.0

                        game.edge_pct = predicted_error
                        game.direction = "home" if predicted_error > 0 else "away"
                        game.confidence = (result.confidence_label or "low").lower()

                        # Compute Kelly stake recommendation
                        if predicted_error >= 0:
                            decimal_odds = american_to_decimal(game.home_ml)
                            team_for_kelly = game.home_team_short
                            win_prob_for_kelly = home_win_prob
                        else:
                            decimal_odds = american_to_decimal(game.away_ml)
                            team_for_kelly = game.away_team_short
                            win_prob_for_kelly = 1.0 - home_win_prob

                        stake_result = self._kelly_staker.compute_stake(
                            win_probability=max(win_prob_for_kelly, 0.01),
                            decimal_odds=decimal_odds,
                            confidence_score=result.confidence_score,
                            confidence_label=result.confidence_label,
                            edge_pct=abs(predicted_error),
                            league=game.league,
                            team=team_for_kelly,
                            game_id=game.game_id,
                        )
                        game.stake_dollars = stake_result.stake_dollars
                    else:
                        game.edge_pct = 0.0
                        game.direction = "neutral"
                        game.confidence = "low"

                    game.predicted_total = round(home_win_prob, 3)
                    game.predicted_at = datetime.now().isoformat()

                except Exception as e:
                    logger.debug(f"Robust prediction failed for {game.matchup}: {e}")
                    continue

            return games

        except Exception as e:
            logger.warning(f"Robust prediction pipeline failed: {e}")
            return games

    # ── Totals Regression Model ───────────────────────────────────────────

    def _build_totals_model(self) -> bool:
        """
        Build and train a totals regression model on historical NBA data.

        Predicts total_points (actual combined score) using the same features
        as the MarketInefficiencySystem. At inference, the predicted total
        is compared to market_total to produce over/under edge.

        Returns:
            True if the model was built and fitted successfully.
        """
        try:
            from betting_intel.live.totals_model import TotalsRegressor
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.data.features import FeatureEngineer

            logger.info("Building TotalsRegressor on historical NBA data...")

            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is None or raw_df.empty:
                logger.warning("No historical NBA data for totals model")
                return False

            games_df = loader.build_game_dataset(raw_df)
            raw_df = loader.compute_rest_days(raw_df)
            fe = FeatureEngineer()
            features_df = fe.build_all_features(games_df, raw_df)

            if features_df is None or features_df.empty:
                logger.warning("Feature engineering produced no data for totals model")
                return False

            import numpy as np

            # CRITICAL: Use FeatureEngineer.select_features() to exclude raw
            # box score columns that would leak total_points target.
            clean_feature_cols = fe.select_features(features_df)

            # Also exclude total_points target and any market columns
            _exclude_totals = {
                "total_points", "point_diff",
                "market_implied_home_prob", "market_error",
                "abs_market_error", "market_error_clipped",
                "market_error_binary", "total_market_error",
                "weighted_market_error", "elo_error",
                "market_error_ma_5g", "market_error_ma_10g",
                "market_error_trend_home", "recent_edge_streak",
            }
            feature_cols = [c for c in clean_feature_cols if c not in _exclude_totals]

            if len(feature_cols) < 3:
                logger.warning(f"Only {len(feature_cols)} feature cols for totals model")
                return False

            X = features_df[feature_cols].fillna(0).values
            n_samples = len(X)
            if n_samples < 200:
                logger.warning(f"Only {n_samples} samples for totals model")
                return False

            # Target: total_points
            if "total_points" not in features_df.columns:
                logger.warning("`total_points` column not found for totals model")
                return False

            y_total = features_df["total_points"].values.astype(float)

            logger.info(
                f"Training TotalsRegressor on {n_samples} samples "
                f"with {len(feature_cols)} features..."
            )

            regressor = TotalsRegressor(random_state=42)
            regressor.fit(X, y_total, feature_names=feature_cols, verbose=True)

            with self._model_lock:
                self._totals_model = regressor
                self._totals_fitted = True
                self._totals_mae = regressor.mae or 12.0

            logger.info(
                f"TotalsRegressor built: {len(regressor._models)} models, "
                f"MAE={regressor.mae:.1f}"
            )

            return True

        except Exception as e:
            logger.warning(f"Failed to build totals model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _predict_totals(self, games: list[LiveGame]) -> list[LiveGame]:
        """
        Predict total points for each game using the totals regression model.

        For each NBA game with a market total:
          1. Build the feature vector
          2. Predict total points
          3. Compute edge = (predicted - market_total) / market_total
          4. Set total_prediction, total_edge_pct, total_direction, total_confidence

        This runs AFTER _predict_with_robust_system, using the same feature
        engineering pipeline. Each game gets both a moneyline edge (from the
        robust system) and a totals edge (from this function).
        """
        with self._model_lock:
            if not self._totals_model or not self._totals_fitted:
                return games
            totals_model = self._totals_model

        try:
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.data.features import FeatureEngineer

            loader = NBADataLoader()
            raw_df = loader.load_game_logs()
            if raw_df is None or raw_df.empty:
                return games

            games_df = loader.build_game_dataset(raw_df)
            raw_df = loader.compute_rest_days(raw_df)
            fe = FeatureEngineer()
            features_df = fe.build_all_features(games_df, raw_df)

            if features_df is None or features_df.empty:
                return games

            nba_games = [g for g in games if g.sport_group == "Basketball" and g.market_total and g.market_total > 0]
            if not nba_games:
                return games

            predicted_count = 0
            for game in nba_games:
                try:
                    feat = self._build_feature_vector(
                        game.home_team_short,
                        game.away_team_short,
                        features_df,
                    )
                    if feat is None:
                        continue

                    X_pred = feat.values.reshape(1, -1)
                    result = totals_model.predict_single(
                        X_pred,
                        market_total=game.market_total,
                    )

                    game.total_prediction = result.predicted_total
                    game.total_edge_pct = result.edge_pct
                    game.total_direction = result.direction
                    game.total_confidence = result.confidence
                    predicted_count += 1

                except Exception as e:
                    logger.debug(f"Totals prediction failed for {game.matchup}: {e}")
                    continue

            if predicted_count > 0:
                logger.info(f"Totals model: predicted {predicted_count} NBA games")

            return games

        except Exception as e:
            logger.warning(f"Totals prediction pipeline failed: {e}")
            return games

    # ── Auto-Resolve ─────────────────────────────────────────────────────

    def _auto_resolve_completed_games(self) -> int:
        """
        Automatically resolve completed games on every engine refresh.

        Calls ResultsTracker.resolve_all() which loads logged predictions,
        fetches actual results from the NBA database, matches each prediction
        to its outcome, computes P&L, and updates forward_test_results.json.

        Runs silently — errors are logged at debug level and do NOT block
        the refresh cycle.

        Returns:
            Number of newly resolved predictions, or 0 if none/failed.
        """
        try:
            from betting_intel.analytics.tracker import ResultsTracker
            tracker = ResultsTracker()
            n = tracker.resolve_all()
            with self._lock:
                self._last_auto_resolve = datetime.now().isoformat()
            if n > 0:
                logger.info(f"Auto-resolved {n} completed game(s) — results saved to forward_test_results.json")
            return n
        except ImportError:
            return 0
        except Exception as e:
            logger.debug(f"Auto-resolve skipped (non-critical): {e}")
            return 0

    # ── Snapshot Builder ──────────────────────────────────────────────────

    def _build_snapshot(self) -> LivePredictionSnapshot:
        """
        Build a complete prediction snapshot from scratch.

        Flow:
          0. Auto-resolve completed games (check actual results)
          1. Fetch real odds from TheOddsAPI
          2. Parse into LiveGame objects
          3. Log odds to historical store (builds training data)
          4. Classify as live / today / tomorrow
          5. Run ML predictions (if model available)
          6. Compute edges and confidence
          7. Return snapshot with all categories
        """
        # Step 0: Auto-resolve completed games before fetching new odds
        self._auto_resolve_completed_games()

        raw_odds = self._fetch_realtime_odds()
        fresh_odds = bool(raw_odds)
        all_games = self._parse_games(raw_odds)

        if not all_games:
            logger.info("No real games available — returning empty snapshot (no synthetic data)")
            return LivePredictionSnapshot(fresh_odds=fresh_odds)

        # Step N: Log all parsed games to the market odds store
        # Every refresh cycle builds up the historical market data corpus.
        # Future training runs will query this store for REAL market probs
        # instead of falling back to the ELO proxy.
        try:
            self._market_odds_store.log_batch(all_games, source="engine_refresh")
        except Exception:
            logger.debug("Failed to log odds snapshots (non-critical)", exc_info=True)

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

        self._predict_games(all_games)

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
        """Parse raw TheOddsAPI events into LiveGame objects. NBA-only."""
        from betting_intel.data.odds_fetcher import ODDS_TO_SHORT_NAME
        from betting_intel.live.sport_configs import (
            league_from_sport_key, sport_key_to_group,
        )

        games: list[LiveGame] = []
        now_utc = datetime.now(timezone.utc)

        for event in raw_odds:
            try:
                home_full = event.get("home_team", "")
                away_full = event.get("away_team", "")
                if not home_full or not away_full:
                    continue

                sport_key = event.get("_sport_config_key", event.get("sport_key", "basketball_nba"))
                league_name = league_from_sport_key(sport_key)
                sport_group = sport_key_to_group(sport_key)

                # NBA team name mapping
                home_short = ODDS_TO_SHORT_NAME.get(home_full, home_full.split()[-1] if " " in home_full else home_full)
                away_short = ODDS_TO_SHORT_NAME.get(away_full, away_full.split()[-1] if " " in away_full else away_full)

                commence_time = event.get("commence_time", "")
                game_date = commence_time[:10] if commence_time else ""

                try:
                    commence_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                    age_hours = (now_utc - commence_dt).total_seconds() / 3600
                    if age_hours > 3:
                        continue
                except Exception:
                    pass

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
                    game_id=event.get("id", f"{sport_key}_{home_short}_{away_short}_{game_date}"),
                    sport_key=sport_key,
                    league=league_name,
                    sport_group=sport_group,
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
        if not games:
            return games

        # Lazy-load model and robust system (thread-safe via dedicated locks)
        if self._model is None:
            self._load_model()

        if not self._robust_system_fitted:
            self._build_robust_system()

        # ── TIER 1: Robust Prediction System (NBA only) ──
        nba_games = [g for g in games if g.sport_group == "Basketball"]
        with self._model_lock:
            robust_ready = self._robust_system_fitted
        if nba_games and robust_ready:
            try:
                self._predict_with_robust_system(nba_games)
                logger.info(f"Robust system: predicted {sum(1 for g in nba_games if g.edge_pct != 0)} NBA games")
            except Exception as e:
                logger.warning(f"Robust system prediction failed: {e}")

        # ── TIER 1b: Totals Regression Model (NBA only) ──
        # Predicts total points for each NBA game, providing over/under edge
        # alongside the moneyline edge from the robust system.
        with self._model_lock:
            totals_ready = self._totals_fitted
        if not totals_ready and nba_games:
            logger.info("Building totals regression model...")
            self._build_totals_model()
            with self._model_lock:
                totals_ready = self._totals_fitted

        if nba_games and totals_ready:
            try:
                self._predict_totals(nba_games)
            except Exception as e:
                logger.warning(f"Totals prediction failed: {e}")

        # ── TIER 2: Legacy model for games not yet predicted ──
        nba_unpredicted = [g for g in nba_games if g.edge_pct is None]
        with self._model_lock:
            model_ready = self._model is not None and bool(self._feature_cols)
        if nba_unpredicted and model_ready:
            try:
                self._predict_with_model(nba_unpredicted)
            except Exception as e:
                logger.warning(f"Legacy model prediction failed: {e}")

        # Fallback for any games not yet predicted
        for game in games:
            if game.edge_pct is not None:
                continue
            if game.sport_group == "Basketball":
                if game.market_total and game.market_total > 0:
                    game.predicted_total = game.market_total
                    game.edge_pct = 0.0
                    game.direction = "neutral"
                    game.confidence = "low"
            else:
                game.edge_pct = 0.0
                game.direction = "neutral"
                game.confidence = "low"
            game.predicted_at = datetime.now().isoformat()

        return games

    def _load_model(self):
        with self._model_lock:
            if self._model is not None:
                return  # Already loaded by another thread
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
                    logger.info(f"Loaded pre-trained model with {len(self._feature_cols)} features")
            except Exception as e:
                logger.debug(f"No pre-trained model available: {e}")

    def _predict_with_model(self, games: list[LiveGame]) -> list[LiveGame]:
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
                    feat = self._build_feature_vector(
                        game.home_team_short, game.away_team_short, features_df,
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

                    if predicted_total > 300 or predicted_total < 80:
                        logger.debug(f"Unreasonable prediction {predicted_total:.0f} for {game.matchup} — skipping")
                        game.predicted_total = None
                        game.edge_pct = 0.0
                        game.direction = "neutral"
                        game.confidence = "low"
                        game.predicted_at = datetime.now().isoformat()
                        continue

                    game.predicted_total = round(predicted_total, 1)

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
        self, home_team: str, away_team: str, features_df: pd.DataFrame,
    ) -> Optional["pd.Series"]:
        import numpy as np
        import pandas as pd

        if features_df is None or features_df.empty:
            return None

        feature_cols = self._feature_cols
        if not feature_cols:
            # Use FeatureEngineer.select_features() to ensure no leaky columns
            from betting_intel.data.features import FeatureEngineer
            _fe = FeatureEngineer()
            feature_cols = _fe.select_features(features_df)
            feature_cols = [c for c in feature_cols
                            if c not in {"total_points", "point_diff",
                                          "home_score", "away_score",
                                          "spread", "label", "home_win"}]

        home_col = "TEAM_NAME_home" if "TEAM_NAME_home" in features_df.columns else "home_team"
        away_col = "TEAM_NAME_away" if "TEAM_NAME_away" in features_df.columns else "away_team"

        def _team_weighted_avg(team: str, base_stat: str, n: int = 15) -> float:
            """
            Compute recency-weighted average for a team over the last n games.

            Uses exponential decay weights: the most recent game gets the
            highest weight, games further back get progressively less weight.
            This is better than a simple average because recent performance
            is more predictive of future outcomes than stale data.

            Decay factor: 0.85 per game (weight halves after ~4.3 games)
            """
            home_stat = f"home_{base_stat}"
            away_stat = f"away_{base_stat}"
            all_vals = []

            if home_stat in features_df.columns and home_col in features_df.columns:
                try:
                    mask = features_df[home_col].astype(str).str.contains(team, case=False, na=False)
                    home_vals = features_df.loc[mask, home_stat].dropna().tail(n)
                    all_vals.extend(home_vals.values.tolist())
                except Exception:
                    pass
            if away_stat in features_df.columns and away_col in features_df.columns:
                try:
                    mask = features_df[away_col].astype(str).str.contains(team, case=False, na=False)
                    away_vals = features_df.loc[mask, away_stat].dropna().tail(n)
                    all_vals.extend(away_vals.values.tolist())
                except Exception:
                    pass

            if not all_vals:
                return 0.0

            # Apply exponential decay: most recent game (last in list) gets highest weight
            weights = np.array([0.85 ** (len(all_vals) - 1 - i) for i in range(len(all_vals))])
            all_vals_arr = np.array(all_vals, dtype=float)
            return float(np.average(all_vals_arr, weights=weights))

        feature_dict: dict[str, float] = {}
        for col in feature_cols:
            if col.startswith("home_") and not col.startswith("home_rest_"):
                base = col[5:]
                feature_dict[col] = _team_weighted_avg(home_team, base)
            elif col.startswith("away_") and not col.startswith("away_rest_"):
                base = col[5:]
                feature_dict[col] = _team_weighted_avg(away_team, base)
            elif col.endswith("_diff"):
                base = col.replace("_diff", "")
                feature_dict[col] = _team_weighted_avg(home_team, base) - _team_weighted_avg(away_team, base)
            elif col.startswith("TEAM_"):
                feature_dict[col] = 0.0
            else:
                feature_dict[col] = float(features_df[col].mean()) if col in features_df.columns else 0.0

        # Note: momentum features are NOT added here because the model
        # was trained on a fixed feature set. Adding extra columns at
        # inference time would change the feature vector dimensions and
        # crash sklearn's predict_proba(). The recency-weighted averages
        # above already provide better differentiation than simple averages.

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
