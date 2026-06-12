"""
ScraperCoordinator — unified orchestrator for ALL data sources.

Priority chain:
  1. ESPN HTTP (fast, free, no key)
  2. DraftKings Playwright (free, more books for consensus)
  3. ESPN Playwright stealth (fallback when HTTP fails)
  4. TheOddsAPI (requires API key, quota-limited, last resort)

Design principles:
  - No single point of failure: every source has a fallback
  - Auto-disable failing scrapers: prevent cascading latency
  - Health metrics on every call: know exactly what is working
  - Graceful degradation: partial data is better than nothing
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import betting_intel.data.scraper_utils as scraper_utils
from betting_intel.data.scraper_utils import SourceFreshnessTracker

logger = logging.getLogger(__name__)


class ScraperCoordinator:
    """
    Unified orchestrator that routes every fetch request through the health
    monitor, applies retry logic, and chains fallbacks in priority order.
    """

    def __init__(self):
        self._freshness = SourceFreshnessTracker()

    # ── Odds: multi-source with fallback chain ────────────────────────────

    def fetch_odds(
        self,
        force_refresh: bool = False,
        timeout: int = 25,
    ) -> list[dict]:
        """
        Fetch NBA odds using 4-tier priority chain.
        Returns empty list if ALL sources fail.
        """
        from betting_intel.data.stealth_scraper import StealthBrowser
        from betting_intel.data.draftkings_scraper import DraftKingsScraper

        t0 = time.time()

        # ── Tier 1: ESPN HTTP (fast path) ────────────────────────────────
        m = self._m()
        if not m.is_disabled("espn_http"):
            try:
                t1 = time.time()
                games = StealthBrowser._scrape_via_http(timeout=timeout)
                latency = (time.time() - t1) * 1000
                if games:
                    m.record_success("espn_http", latency)
                    self._freshness.record_fetch("espn_http")
                    logger.info(
                        "Coordinator: ESPN HTTP -> %d games (%.0fms)",
                        len(games), latency,
                    )
                    return games
                m.record_failure("espn_http", "EMPTY_RESPONSE",
                                 latency_ms=latency)
            except Exception as e:
                t1 = time.time()
                latency = (time.time() - t1) * 1000
                m.record_failure("espn_http", type(e).__name__,
                                 latency_ms=latency)
                logger.debug("Coordinator: ESPN HTTP failed: %s", e)

        # ── Tier 2: DraftKings Playwright ────────────────────────────────
        if not m.is_disabled("draftkings"):
            try:
                t2 = time.time()
                games = DraftKingsScraper.scrape(timeout=timeout)
                latency = (time.time() - t2) * 1000
                if games:
                    m.record_success("draftkings", latency)
                    self._freshness.record_fetch("draftkings")
                    logger.info(
                        "Coordinator: DraftKings -> %d games (%.0fms)",
                        len(games), latency,
                    )
                    return games
                m.record_failure("draftkings", "EMPTY_RESPONSE",
                                 latency_ms=latency)
            except Exception as e:
                t2 = time.time()
                latency = (time.time() - t2) * 1000
                m.record_failure("draftkings", type(e).__name__,
                                 latency_ms=latency)
                logger.debug("Coordinator: DraftKings failed: %s", e)

        # ── Tier 3: ESPN Playwright stealth (fallback) ───────────────────
        if not m.is_disabled("espn_playwright"):
            try:
                t3 = time.time()
                games = StealthBrowser._scrape_via_playwright(timeout=timeout)
                latency = (time.time() - t3) * 1000
                if games:
                    m.record_success("espn_playwright", latency)
                    self._freshness.record_fetch("espn_playwright")
                    logger.info(
                        "Coordinator: ESPN Playwright -> %d games (%.0fms)",
                        len(games), latency,
                    )
                    return games
                m.record_failure("espn_playwright", "EMPTY_RESPONSE",
                                 latency_ms=latency)
            except Exception as e:
                t3 = time.time()
                latency = (time.time() - t3) * 1000
                m.record_failure("espn_playwright", type(e).__name__,
                                 latency_ms=latency)
                logger.debug("Coordinator: ESPN Playwright failed: %s", e)

        # ── Tier 4: TheOddsAPI (last resort, needs key) ──────────────────
        if not m.is_disabled("theoddsapi"):
            try:
                from betting_intel.data.live_gateway import LiveDataGateway
                gateway = LiveDataGateway()
                t4 = time.time()
                odds = gateway.get_live_odds(force_refresh=True)
                latency = (time.time() - t4) * 1000
                if odds:
                    m.record_success("theoddsapi", latency)
                    self._freshness.record_fetch("theoddsapi")
                    logger.info(
                        "Coordinator: TheOddsAPI -> %d games (%.0fms)",
                        len(odds), latency,
                    )
                    return odds
                m.record_failure("theoddsapi", "EMPTY_RESPONSE",
                                 latency_ms=latency)
            except Exception as e:
                t4 = time.time()
                latency = (time.time() - t4) * 1000
                m.record_failure("theoddsapi", type(e).__name__,
                                 latency_ms=latency)
                logger.debug("Coordinator: TheOddsAPI failed: %s", e)

        total_time = (time.time() - t0) * 1000
        logger.warning(
            "Coordinator: ALL 4 odds sources failed after %.0fms", total_time,
        )
        return []

    def _m(self):
        """Shortcut to the global scraper monitor (always fresh reference)."""
        return scraper_utils.GLOBAL_SCRAPER_MONITOR

    # ── Injuries: ESPN with fallback ──────────────────────────────────────

    def fetch_injuries(
        self, force_refresh: bool = False
    ) -> list[dict]:
        """Fetch NBA injuries. Returns empty list if ALL sources fail."""
        from betting_intel.data.injury_scraper import ESPNInjuryScraper

        t0 = time.time()
        scraper = ESPNInjuryScraper()

        try:
            records = scraper.fetch_all(force_refresh=force_refresh)
            latency = (time.time() - t0) * 1000
            if records:
                m.record_success("injury_scraper", latency)
                self._freshness.record_fetch("injury_scraper")
                return [r.to_dict() for r in records]
            m.record_failure("injury_scraper", "EMPTY_RESPONSE",
                             latency_ms=latency)
        except Exception as e:
            latency = (time.time() - t0) * 1000
            m.record_failure("injury_scraper", type(e).__name__,
                             latency_ms=latency)
            logger.warning("Coordinator: Injury scraper failed: %s", e)

        return []

    # ── Historical data (ESPN multi-league) ──────────────────────────────

    def fetch_historical(
        self,
        league_key: str = "nba",
        seasons: list | None = None,
    ) -> "pd.DataFrame":
        """Fetch historical game data from ESPN for model training."""
        import pandas as pd
        from betting_intel.data.espn_hoops import ESPNLeagueSource

        source = ESPNLeagueSource()
        t0 = time.time()

        try:
            df = source.load_historical(league_key, seasons=seasons)
            latency = (time.time() - t0) * 1000
            if df is not None and not df.empty:
                self._freshness.record_fetch(f"espn_history_{league_key}")
                logger.info(
                    "Coordinator: ESPN history %s -> %d games (%.0fms)",
                    league_key, len(df), latency,
                )
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.warning(
                "Coordinator: ESPN history %s failed: %s", league_key, e,
            )
            return pd.DataFrame()

    # ── Health & Monitoring ──────────────────────────────────────────────

    def get_health_summary(self) -> dict:
        """Get health metrics for ALL scrapers."""
        return self._m().get_health_summary()

    def get_freshness_summary(self) -> dict:
        """Get data freshness for all sources."""
        return self._freshness.summary()

    def get_full_status(self) -> dict:
        """Get complete status: health + freshness + active sources."""
        health = self.get_health_summary()
        freshness = self.get_freshness_summary()
        active_sources = [
            name for name, info in health.get("scrapers", {}).items()
            if not info.get("disabled", False) and info.get("success_rate", 0) >= 0.5
        ]
        return {
            "health": health,
            "freshness": freshness,
            "active_sources": active_sources,
            "total_sources": health.get("total_scrapers", 0),
            "all_sources_healthy": health.get("all_healthy", False),
            "generated_at": __import__("datetime").datetime.now().isoformat(),  # noqa
        }

    def reset_monitor(self):
        """Reset health monitor state by creating a fresh instance."""
        from betting_intel.data.scraper_utils import ScraperHealthMonitor, SourceFreshnessTracker
        scraper_utils.GLOBAL_SCRAPER_MONITOR = ScraperHealthMonitor()
        self._freshness = SourceFreshnessTracker()
        logger.info("Coordinator: Health monitor reset complete")
