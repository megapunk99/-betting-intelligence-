"""
LivePredictionEngine — real-time NBA prediction system.

NBA-only. Single-sport focus means:
  - One data pipeline (TheOddsAPI basketball_nba)
  - One prediction strategy (total points regression + home win classification)
  - One model (RobustPredictionSystem with NBA-specific features)
  - One team name mapping (ODDS_TO_SHORT_NAME)

Architecture:
  LivePredictionEngine (this file) composes focused sub-modules:
    - models.py:      LiveGame, LivePredictionSnapshot dataclasses + constants
    - odds_fetcher.py: OddsFetcher — TheOddsAPI calls + free scrapers + merge
    - odds_parser.py:  OddsParser — raw odds -> LiveGame objects
    - predictor.py:    GamePredictor — robust system + totals model + feature vectors
    - snapshot_builder.py: SnapshotBuilder — end-to-end snapshot construction
    - worker.py:       LivePredictionWorker — background refresh loop

CRITICAL DESIGN RULE: TheOddsAPI is ONLY called when force_refresh=True
is explicitly passed. Page loads, WebSocket connections, and all
read-only operations return cached data only. Zero automatic API calls.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from betting_intel.live.models import (
    LiveGame,
    LivePredictionSnapshot,
    ODDS_CACHE_TTL_SECONDS,
    PREDICTION_REFRESH_INTERVAL,
)
from betting_intel.live.odds_fetcher import OddsFetcher
from betting_intel.live.odds_parser import OddsParser
from betting_intel.live.predictor import GamePredictor
from betting_intel.live.snapshot_builder import SnapshotBuilder
from betting_intel.live.worker import LivePredictionWorker
from betting_intel.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# Re-export symbols that tests and other modules import from engine.py
__all__ = [
    "LiveGame",
    "LivePredictionSnapshot",
    "LivePredictionEngine",
    "LivePredictionWorker",
    "ODDS_CACHE_TTL_SECONDS",
    "PREDICTION_REFRESH_INTERVAL",
]

logger = logging.getLogger(__name__)


class LivePredictionEngine:
    """
    Continuously produces predictions for live + upcoming games using ONLY real data.

    Zero synthetic games. Zero random matchups. If TheOddsAPI returns nothing,
    every method returns an empty list.

    Thread-safe: uses a lock for the internal snapshot cache so the web app
    can read predictions while the background worker refreshes them.

    This class orchestrates the sub-modules:
      - OddsFetcher:   fetching real-time odds from multiple sources
      - OddsParser:    parsing raw odds into LiveGame domain objects
      - GamePredictor: building/predicting with ML models
      - SnapshotBuilder: constructing prediction snapshots
      - LivePredictionWorker: background refresh loop
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

        # Scraper timeout — made an instance attribute so tests can override it
        self._scraper_timeout = 15.0

        # ── Sub-modules ─────────────────────────────────────────────
        self._odds_fetcher = OddsFetcher(
            odds_api_key=self._odds_api_key,
            odds_api_key_fallback=self._odds_api_key_fallback,
            scraper_timeout=self._scraper_timeout,
        )
        self._odds_parser = OddsParser()

        # Kelly staker
        try:
            from betting_intel.recommendations.staking import KellyStaker
            self._kelly_staker = KellyStaker(
                initial_bankroll=10_000.0,
                kelly_fraction=0.25,
            )
        except Exception as e:
            logger.warning(f"Kelly staker initialization failed: {e}")
            self._kelly_staker = None

        # Market odds store
        try:
            from betting_intel.db.market_odds_store import MarketOddsStore
            self._market_odds_store = MarketOddsStore()
            self._market_odds_store.ensure_table()
        except Exception as e:
            logger.warning(f"Market odds store initialization failed: {e}")
            self._market_odds_store = None

        # Predictor
        self._predictor = GamePredictor(
            kelly_staker=self._kelly_staker,
            market_odds_store=self._market_odds_store,
            model_dir=self._model_dir,
        )

        # Snapshot builder (pure assembler — receives already-parsed games)
        self._snapshot_builder = SnapshotBuilder(
            market_odds_store=self._market_odds_store,
        )

        # ── Thread-safe cache (owned by engine) ─────────────────────
        self._lock = Lock()
        self._snapshot: Optional[LivePredictionSnapshot] = None
        self._last_refresh: float = 0.0
        self._last_odds_fetch: float = 0.0
        self._cached_odds_raw: Optional[list[dict]] = None

        # Model lock (owned by engine for thread safety)
        self._model_lock = Lock()

        # Telegram notifier (lazy — only instantiated if configured)
        self._notifier: Optional[Any] = None

        logger.info(
            "LivePredictionEngine initialized — zero synthetic games. "
            f"Refresh interval: {refresh_interval}s"
        )

    # ── Proxy properties (for test compatibility — tests access these
    #    on the engine, but state lives in sub-modules) ────────────

    @property
    def _robust_system(self) -> Any:
        if self._predictor is None:
            return None
        return self._predictor._robust_system

    @_robust_system.setter
    def _robust_system(self, value: Any):
        if self._predictor is not None:
            self._predictor._robust_system = value

    @property
    def _robust_system_fitted(self) -> bool:
        if self._predictor is None:
            return False
        return self._predictor._robust_system_fitted

    @_robust_system_fitted.setter
    def _robust_system_fitted(self, value: bool):
        if self._predictor is not None:
            self._predictor._robust_system_fitted = value

    @property
    def _totals_fitted(self) -> bool:
        if self._predictor is None:
            return False
        return self._predictor._totals_fitted

    @_totals_fitted.setter
    def _totals_fitted(self, value: bool):
        if self._predictor is not None:
            self._predictor._totals_fitted = value

    @property
    def _last_auto_resolve(self) -> Optional[str]:
        if self._snapshot_builder is None:
            return None
        return self._snapshot_builder._last_auto_resolve

    @_last_auto_resolve.setter
    def _last_auto_resolve(self, value: Optional[str]):
        if self._snapshot_builder is not None:
            self._snapshot_builder._last_auto_resolve = value

    # ── Public properties (delegating to sub-modules) ─────────────────

    @property
    def robust_system(self) -> Any:
        return self._predictor.robust_system

    @property
    def robust_system_summary(self) -> dict:
        return self._predictor.robust_system_summary()

    @property
    def kelly_staker(self) -> Any:
        return self._kelly_staker

    @property
    def last_auto_resolve(self) -> Optional[str]:
        return self._snapshot_builder._last_auto_resolve

    @property
    def has_cached_data(self) -> bool:
        with self._lock:
            return self._snapshot is not None and self._snapshot.n_total > 0

    @property
    def has_valid_api_key(self) -> bool:
        return self._odds_fetcher.has_valid_api_key()

    @property
    def last_theoddsapi_fetch(self) -> float:
        return self._odds_fetcher._last_theoddsapi_fetch

    @property
    def theoddsapi_quota_remaining(self) -> Optional[str]:
        return self._odds_fetcher._last_quota_remaining

    @property
    def theoddsapi_schedule(self) -> dict:
        """Return the daily fetch schedule summary."""
        return self._odds_fetcher._schedule_summary(self._odds_fetcher._last_theoddsapi_fetch)

    @property
    def theoddsapi_quota_summary(self) -> dict:
        return self._odds_fetcher.quota_summary

    @property
    def theoddsapi_schedule_status(self) -> dict:
        """Return schedule status for the dashboard API banner."""
        return self._odds_fetcher._get_schedule_status_dict(self._odds_fetcher._last_theoddsapi_fetch)

    # ── Public API ────────────────────────────────────────────────────────

    def get_snapshot(self, force_refresh: bool = False) -> LivePredictionSnapshot:
        if not force_refresh:
            with self._lock:
                if self._snapshot is not None:
                    # Auto-expire: if the cached snapshot is older than the odds TTL,
                    # treat it as stale and fall through to rebuild.
                    age = time.time() - self._last_refresh
                    if age < ODDS_CACHE_TTL_SECONDS:
                        return self._snapshot
                    logger.debug(f"Cached snapshot is {age:.0f}s old — stale, will refresh")
                else:
                    logger.debug("No cached snapshot — returning empty (user must refresh)")
                    return LivePredictionSnapshot()

        # Fall through: force refresh or cache expired
        # Propagate force_refresh to _build_snapshot so user-initiated
        # refreshes bypass the daily schedule but auto-refreshes respect it.
        now = time.time()
        try:
            snapshot = self._build_snapshot(force_theoddsapi=force_refresh)
            with self._lock:
                self._snapshot = snapshot
                self._last_refresh = now
            return snapshot
        except Exception as e:
            logger.error(f"Failed to refresh predictions: {e}")
            with self._lock:
                return self._snapshot or LivePredictionSnapshot()

    def get_live_games(self, force_refresh: bool = False) -> list[LiveGame]:
        return self.get_snapshot(force_refresh=force_refresh).live_games

    def get_today_games(self, force_refresh: bool = False) -> list[LiveGame]:
        return self.get_snapshot(force_refresh=force_refresh).today_games

    def get_tomorrow_games(self, force_refresh: bool = False) -> list[LiveGame]:
        return self.get_snapshot(force_refresh=force_refresh).tomorrow_games

    def get_next_two_days(self, force_refresh: bool = False) -> list[LiveGame]:
        return self.get_snapshot(force_refresh=force_refresh).next_two_days

    def refresh_now(self) -> LivePredictionSnapshot:
        return self.get_snapshot(force_refresh=True)

    def clear_cache(self):
        """Clear all cached data, models, and scraper caches."""
        with self._model_lock:
            self._predictor.clear()
        with self._lock:
            self._snapshot = None
            self._last_refresh = 0.0
            self._last_odds_fetch = 0.0
            self._cached_odds_raw = None

        # Reset fetch timer so the daily morning schedule triggers next time
        self._odds_fetcher._last_theoddsapi_fetch = 0.0

        # Clear external scraper caches (best-effort, non-critical)
        import importlib
        _scraper_modules = [
            ("betting_intel.data.stealth_scraper", "StealthBrowser"),
            ("betting_intel.data.draftkings_scraper", "DraftKingsScraper"),
        ]
        for module_name, cls_name in _scraper_modules:
            try:
                mod = importlib.import_module(module_name)
                cls = getattr(mod, cls_name)
                cls.clear_cache()
                logger.debug(f"Cleared {cls_name} cache")
            except Exception as e:
                logger.debug(f"Could not clear {cls_name} cache: {e}")

        logger.info("LivePredictionEngine cache fully cleared")

    # ── Odds Fetching (delegates to OddsFetcher + engine cache) ───────────

    def _has_valid_api_key(self) -> bool:
        return self._odds_fetcher.has_valid_api_key()

    def _fetch_realtime_odds(self, force_theoddsapi: bool = False) -> list[dict]:
        """Fetch odds, using engine-level cache for TTL management.

        Args:
            force_theoddsapi: If True, bypass the daily schedule and call TheOddsAPI.
                              Set to True on user-initiated "Refresh" button clicks.
        """
        now = time.time()

        # Fast-path: return cached if still fresh
        with self._lock:
            if self._cached_odds_raw is not None and (now - self._last_odds_fetch) < ODDS_CACHE_TTL_SECONDS:
                return self._cached_odds_raw

        result = self._odds_fetcher.fetch(
            cached_odds_raw=self._cached_odds_raw,
            last_odds_fetch=self._last_odds_fetch,
            cache_lock=self._lock,
            now=now,
            force_theoddsapi=force_theoddsapi,
        )

        # Update engine-level cache
        with self._lock:
            self._cached_odds_raw = result
            self._last_odds_fetch = now

        return result

    def _fetch_via_theoddsapi(self) -> list[dict]:
        return self._odds_fetcher._fetch_via_theoddsapi()

    def _fetch_via_theoddsapi_with_key(self, api_key: str, active_sports: list) -> list[dict]:
        return self._odds_fetcher._fetch_via_theoddsapi_with_key(api_key, active_sports)

    def _fetch_stealth_scraper(self) -> list[dict]:
        return self._odds_fetcher._fetch_stealth_scraper()

    def _fetch_draftkings_odds(self) -> list[dict]:
        return self._odds_fetcher._fetch_draftkings_odds()

    def _merge_odds_sources(self, espn_data: list[dict], dk_data: list[dict]) -> list[dict]:
        return self._odds_fetcher.merge_odds_sources(espn_data, dk_data)

    # ── Game Parsing (delegates to OddsParser) ────────────────────────────

    def _parse_games(self, raw_odds: list[dict]) -> list[LiveGame]:
        return self._odds_parser.parse_games(raw_odds)

    # ── Robust System (delegates to GamePredictor) ────────────────────────

    def _build_robust_system(self) -> bool:
        with self._model_lock:
            return self._predictor._build_robust_system()

    def _predict_with_robust_system(self, games: list[LiveGame]) -> list[LiveGame]:
        with self._model_lock:
            return self._predictor._predict_with_robust_system(games)

    # ── Totals Model (delegates to GamePredictor) ─────────────────────────

    def _build_totals_model(self) -> bool:
        with self._model_lock:
            return self._predictor._build_totals_model()

    def _predict_totals(self, games: list[LiveGame]) -> list[LiveGame]:
        with self._model_lock:
            return self._predictor._predict_totals(games)

    # ── Auto-Resolve (engine-owned — snapshot builder is a pure assembler) ─

    def _auto_resolve_completed_games(self) -> int:
        """Resolve completed games via ResultsTracker. Non-blocking on failure."""
        try:
            from betting_intel.analytics.tracker import ResultsTracker
            tracker = ResultsTracker()
            n = tracker.resolve_all()
            if self._snapshot_builder is not None:
                self._snapshot_builder.set_auto_resolve_timestamp(datetime.now().isoformat())
            if n > 0:
                logger.info(f"Auto-resolved {n} completed game(s)")
            return n
        except ImportError:
            return 0
        except Exception as e:
            logger.debug(f"Auto-resolve skipped (non-critical): {e}")
            return 0

    # ── Snapshot Builder — engine orchestrates, snapshot builder assembles ─

    def _build_snapshot(self, force_theoddsapi: bool = False) -> LivePredictionSnapshot:
        """
        Build a complete prediction snapshot.

        Args:
            force_theoddsapi: If True, bypass the daily schedule and call TheOddsAPI.
                              True = user clicked Refresh (force_refresh=True).
                              False = auto-refresh from stale cache — respects schedule.

        Flow:
          1. Auto-resolve completed games (engine-owned, non-blocking)
          2. Fetch real-time odds (engine-owned, respects cache TTL)
          3. Parse odds into LiveGame objects (via OddsParser)
          4. If no real games found, auto-seed with offline data (offseason fallback)
          5. Run ML predictions (engine-owned, via predictor)
          6. Assemble snapshot (via SnapshotBuilder, pure assembler)

        All steps before assembly are engine-level so tests can patch them.
        Each step is self-healing: failure in one step does not block subsequent steps.
        """
        # Step 1: Auto-resolve completed games (always safe, never blocks)
        self._auto_resolve_completed_games()

        # Step 2 - 4: Fetch, parse, predict — each handles empty gracefully
        # force_theoddsapi is True when user clicked Refresh (force_refresh=True).
        # When False (stale cache auto-refresh), the daily schedule governs calls.
        raw_odds = self._fetch_realtime_odds(force_theoddsapi=force_theoddsapi)
        fresh_odds = bool(raw_odds)
        if not fresh_odds:
            logger.debug("No fresh odds — will use cached/empty games")

        all_games: list[LiveGame] = []
        is_seeded = False

        if raw_odds:
            try:
                all_games = self._odds_parser.parse_games(raw_odds)
                logger.debug(f"Parsed {len(all_games)} games from raw odds")
            except Exception as e:
                logger.warning(f"Odds parsing failed: {e}")
                all_games = []

        # Offseason: when no real games are found, show an honest empty state
        if not all_games:
            logger.info("No real games available — displaying empty state (offseason)")

        try:
            self._predict_games(all_games)
        except Exception as e:
            logger.warning(f"ML prediction step failed: {e}")

        # Step 5: Detect arbitrage opportunities from raw odds
        arb_opportunities: list[dict] = []
        if raw_odds and len(raw_odds) > 0:
            try:
                from betting_intel.arbitrage import detect_arbitrage
                arb_results = detect_arbitrage(raw_odds)
                arb_opportunities = [o.to_dict() for o in arb_results]
            except Exception as e:
                logger.debug(f"Arbitrage detection failed (non-critical): {e}")

        # Step 6: Assemble snapshot (pure assembler, no side effects)
        snap = self._snapshot_builder.build_snapshot(
            all_games=all_games,
            fresh_odds=fresh_odds,
        )
        snap.arbitrage_opportunities = arb_opportunities
        snap.n_arbitrage = len(arb_opportunities)
        if arb_opportunities:
            logger.info(f"Arbitrage opportunities: {len(arb_opportunities)} found")

        # Send Telegram alerts for high-confidence picks (non-blocking, best-effort)
        try:
            self._send_high_confidence_alerts(all_games)
        except Exception as e:
            logger.debug(f"Telegram alert dispatch failed: {e}")

        return snap

    # ── Telegram Alerts — high-confidence picks ─────────────────────────

    def _send_high_confidence_alerts(self, games: list[LiveGame]):
        """Send Telegram alerts for high-confidence picks.

        High confidence = confidence is "high" (or equivalent) AND
        edge_pct >= MIN_EDGE_THRESHOLD (3%).

        Non-blocking, best-effort. Silently skips if Telegram is
        not configured or no games qualify.
        """
        from betting_intel.live.models import MIN_EDGE_THRESHOLD

        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return

        # Filter to high-confidence picks that haven't been notified yet
        high_conf = []
        for g in games:
            if g.game_id is None:
                continue
            edge = abs(g.edge_pct or 0.0)
            conf = (g.confidence or "low").lower()
            if edge >= MIN_EDGE_THRESHOLD and conf in ("high", "very_high"):
                high_conf.append(g)

        if not high_conf:
            return

        # Lazy-init notifier (only on first use)
        if self._notifier is None:
            from betting_intel.notifications.telegram_bot import TelegramNotifier
            self._notifier = TelegramNotifier(
                bot_token=TELEGRAM_BOT_TOKEN,
                chat_id=TELEGRAM_CHAT_ID,
            )

        notifier = self._notifier

        # Send a single digest for all new high-confidence picks
        n_sent = notifier.send_digest_sync(high_conf)
        if n_sent > 0:
            logger.info(f"Telegram: sent digest with {n_sent} high-confidence pick(s)")

    def _predict_games(self, games: list[LiveGame]) -> list[LiveGame]:
        """Run all available prediction tiers on games.

        Supports multiple basketball leagues (NBA, NCAAB) through the
        same basketball prediction pipeline. Non-basketball games get
        sensible defaults.

        Uses engine-level methods so tests can patch _build_robust_system
        and _build_totals_model directly.
        """
        if not games:
            return games

        basketball_games = [g for g in games if g.sport_group == "Basketball"]

        # TIER 1: Robust Prediction System (moneyline) — for NBA only
        # NCAAB uses the same system but needs its own training data
        nba_games = [g for g in basketball_games if g.league == "NBA"]

        if not self._robust_system_fitted:
            self._build_robust_system()

        if nba_games and self._robust_system_fitted:
            try:
                self._predict_with_robust_system(nba_games)
                n_predicted = sum(1 for g in nba_games if g.edge_pct != 0 and g.edge_pct is not None)
                logger.info(f"Robust system: predicted {n_predicted} NBA games")
            except Exception as e:
                logger.warning(f"Robust system prediction failed: {e}")

        # TIER 2: Totals Regression Model — basketball generic
        if not self._totals_fitted and basketball_games:
            logger.info("Building totals regression model...")
            self._build_totals_model()

        if basketball_games and self._totals_fitted:
            try:
                self._predict_totals(basketball_games)
            except Exception as e:
                logger.warning(f"Totals prediction failed: {e}")

        # TIER 3: Soccer Prediction — ELO + Poisson for EPL
        soccer_games = [g for g in games if g.sport_group == "Soccer"]
        if soccer_games:
            try:
                from betting_intel.live.soccer_predictor import EPLSoccerPredictor

                for game in soccer_games:
                    try:
                        # Moneyline prediction (home/draw/away)
                        game = EPLSoccerPredictor.predict_moneyline(game)
                        # Totals prediction (over/under goals)
                        game = EPLSoccerPredictor.predict_totals(game)
                    except Exception as e:
                        logger.debug(f"Soccer prediction failed for {game.matchup}: {e}")
                        continue

                n_soccer = sum(1 for g in soccer_games if g.edge_pct != 0 and g.edge_pct is not None)
                logger.info(f"Soccer ELO predictor: predicted {n_soccer} EPL games")
            except Exception as e:
                logger.warning(f"Soccer prediction pipeline failed: {e}")

        # Fallback for unpredicted games — ensures every game has
        # sensible defaults so the dashboard never sees None fields.
        now_iso = datetime.now().isoformat()
        for game in games:
            if game.edge_pct is not None:
                continue

            # Default: no edge detected
            game.edge_pct = 0.0
            game.direction = "neutral"
            game.confidence = "low"
            game.predicted_at = now_iso

            # Basketball: use market total as predicted total proxy
            if game.sport_group == "Basketball":
                game.predicted_total = float(game.market_total) if game.market_total and game.market_total > 0 else None
            elif game.sport_group == "Soccer":
                game.predicted_total = float(game.market_total) if game.market_total and game.market_total > 0 else None
            else:
                game.predicted_total = None

        return games

    def _build_feature_vector(
        self,
        home_team: str,
        away_team: str,
        features_df: "pd.DataFrame",
        feature_cols: Optional[list[str]] = None,
    ) -> Optional["pd.Series"]:
        """Build a feature vector. Import pandas lazily to avoid import-order issues."""
        return self._predictor._build_feature_vector(home_team, away_team, features_df, feature_cols)
