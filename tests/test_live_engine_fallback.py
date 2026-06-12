"""
Tests for LivePredictionEngine's odds-fetching pipeline.

Verifies:
  1. _fetch_realtime_odds() tries ESPN stealth scraper FIRST (free, unlimited)
  2. Falls back to TheOddsAPI when ESPN returns no data
  3. _fetch_stealth_scraper() imports and calls the real scraper module
  4. In-memory cache with ODDS_CACHE_TTL_SECONDS TTL works
  5. Graceful degradation when ALL sources fail
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

from betting_intel.live.engine import LivePredictionEngine, ODDS_CACHE_TTL_SECONDS


_NOW = datetime.now(timezone.utc)
_T0 = (_NOW + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:00Z")
_T0_UPDATE = (_NOW + timedelta(hours=5, minutes=55)).strftime("%Y-%m-%dT%H:%M:00Z")

SAMPLE_ESPN_GAMES = [
    {
        "id": "espn_test_game_1",
        "sport_key": "basketball_nba",
        "sport_title": "NBA",
        "commence_time": _T0,
        "home_team": "Boston Celtics",
        "away_team": "Los Angeles Lakers",
        "bookmakers": [
            {
                "key": "stealth_espn",
                "title": "ESPN (Stealth)",
                "last_update": _T0_UPDATE,
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Boston Celtics", "price": -200},
                            {"name": "Los Angeles Lakers", "price": 175},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 218.5, "price": -110},
                            {"name": "Under", "point": 218.5, "price": -110},
                        ],
                    },
                ],
            }
        ],
    }
]

_T1 = (_NOW + timedelta(hours=6, minutes=30)).strftime("%Y-%m-%dT%H:%M:00Z")

SAMPLE_THEODDSAPI_GAMES = [
    {
        "id": "oddsapi_test_game_1",
        "sport_key": "basketball_nba",
        "sport_title": "NBA",
        "commence_time": _T1,
        "home_team": "Golden State Warriors",
        "away_team": "Denver Nuggets",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": _T0_UPDATE,
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Golden State Warriors", "price": -110},
                            {"name": "Denver Nuggets", "price": -110},
                        ],
                    },
                ],
            }
        ],
    }
]


class TestFetchRealtimeOdds:
    """Verify _fetch_realtime_odds tries ESPN first, then TheOddsAPI."""

    def test_espn_first_success(self):
        """When ESPN scraper returns data, TheOddsAPI is NOT called."""
        engine = LivePredictionEngine(odds_api_key="test_key")

        with patch.object(engine, "_fetch_stealth_scraper", return_value=SAMPLE_ESPN_GAMES) as mock_espn:
            with patch.object(engine, "_fetch_via_theoddsapi") as mock_oddsapi:
                result = engine._fetch_realtime_odds()

                assert len(result) == 1
                assert result[0]["id"] == "espn_test_game_1"
                mock_espn.assert_called_once()
                mock_oddsapi.assert_not_called()

    def test_espn_empty_falls_to_oddsapi(self):
        """When ESPN returns empty, TheOddsAPI is called."""
        engine = LivePredictionEngine(odds_api_key="test_key")

        with patch.object(engine, "_fetch_stealth_scraper", return_value=[]) as mock_espn:
            with patch.object(engine, "_fetch_via_theoddsapi", return_value=SAMPLE_THEODDSAPI_GAMES) as mock_oddsapi:
                result = engine._fetch_realtime_odds()

                assert len(result) == 1
                assert result[0]["id"] == "oddsapi_test_game_1"
                mock_espn.assert_called_once()
                mock_oddsapi.assert_called_once()

    def test_espn_exception_falls_to_oddsapi(self):
        """When ESPN raises an exception, TheOddsAPI is still attempted."""
        engine = LivePredictionEngine(odds_api_key="test_key")

        with patch.object(engine, "_fetch_stealth_scraper", side_effect=Exception("ESPN down")) as mock_espn:
            with patch.object(engine, "_fetch_via_theoddsapi", return_value=SAMPLE_THEODDSAPI_GAMES) as mock_oddsapi:
                result = engine._fetch_realtime_odds()

                assert len(result) == 1
                mock_espn.assert_called_once()
                mock_oddsapi.assert_called_once()

    def test_both_sources_fail(self):
        """When both sources return empty, result is empty list."""
        engine = LivePredictionEngine(odds_api_key="test_key")

        with patch.object(engine, "_fetch_stealth_scraper", return_value=[]):
            with patch.object(engine, "_fetch_via_theoddsapi", return_value=[]):
                result = engine._fetch_realtime_odds()
                assert result == []

    def test_cache_hit(self):
        """In-memory cache is used when within TTL."""
        engine = LivePredictionEngine(odds_api_key="test_key")
        engine._cached_odds_raw = SAMPLE_ESPN_GAMES
        engine._last_odds_fetch = __import__("time").time()

        with patch.object(engine, "_fetch_stealth_scraper") as mock_espn:
            with patch.object(engine, "_fetch_via_theoddsapi") as mock_oddsapi:
                result = engine._fetch_realtime_odds()

                assert len(result) == 1
                mock_espn.assert_not_called()
                mock_oddsapi.assert_not_called()

    def test_cache_ttl_expired(self):
        """Cache is bypassed when TTL has expired."""
        engine = LivePredictionEngine(odds_api_key="test_key")
        engine._cached_odds_raw = SAMPLE_ESPN_GAMES
        engine._last_odds_fetch = 0.0  # Far in the past

        with patch.object(engine, "_fetch_stealth_scraper", return_value=SAMPLE_ESPN_GAMES) as mock_espn:
            result = engine._fetch_realtime_odds()

            assert len(result) == 1
            mock_espn.assert_called_once()

    def test_cache_ttl_constant(self):
        """ODDS_CACHE_TTL_SECONDS should be 300 seconds (5 minutes)."""
        assert ODDS_CACHE_TTL_SECONDS == 300


class TestFetchStealthScraper:
    """Verify _fetch_stealth_scraper imports and calls the real scraper module."""

    def test_successful_scrape(self):
        """When stealth_scraper returns data, it's returned directly."""
        engine = LivePredictionEngine(odds_api_key="test_key")

        with patch(
            "betting_intel.data.stealth_scraper.StealthBrowser.sync_scrape_live_odds",
            return_value=SAMPLE_ESPN_GAMES,
        ) as mock_scrape:
            result = engine._fetch_stealth_scraper()

            assert len(result) == 1
            assert result[0]["home_team"] == "Boston Celtics"
            mock_scrape.assert_called_once()

    def test_import_error_returns_empty(self):
        """When the scraper module can't be imported, empty list is returned."""
        engine = LivePredictionEngine(odds_api_key="test_key")

        with patch(
            "betting_intel.data.stealth_scraper.StealthBrowser.sync_scrape_live_odds",
            side_effect=ImportError("playwright not installed"),
        ):
            result = engine._fetch_stealth_scraper()
            assert result == []

    def test_scrape_exception_returns_empty(self):
        """When the scraper raises an exception, empty list is returned."""
        engine = LivePredictionEngine(odds_api_key="test_key")

        with patch(
            "betting_intel.data.stealth_scraper.StealthBrowser.sync_scrape_live_odds",
            side_effect=RuntimeError("HTTP 500"),
        ):
            result = engine._fetch_stealth_scraper()
            assert result == []


class TestEndToEndRefresh:
    """End-to-end tests that mock _fetch_realtime_odds to test the full parsing pipeline."""

    @patch.object(LivePredictionEngine, "_fetch_realtime_odds")
    def test_full_refresh_with_espn(self, mock_fetch):
        """Full refresh with ESPN data produces LiveGame objects."""
        mock_fetch.return_value = SAMPLE_ESPN_GAMES

        engine = LivePredictionEngine(odds_api_key="test_key")
        snapshot = engine.refresh_now()

        assert snapshot.n_total > 0
        assert snapshot.fresh_odds is True
        mock_fetch.assert_called_once()
        # First game should be Boston Celtics vs Lakers
        assert snapshot.next_two_days[0].home_team == "Boston Celtics"

    @patch.object(LivePredictionEngine, "_fetch_realtime_odds")
    def test_full_refresh_fallback_to_oddsapi(self, mock_fetch):
        """When using TheOddsAPI data, produce LiveGame objects."""
        mock_fetch.return_value = SAMPLE_THEODDSAPI_GAMES

        engine = LivePredictionEngine(odds_api_key="test_key")
        snapshot = engine.refresh_now()

        assert snapshot.n_total > 0
        assert snapshot.fresh_odds is True
        assert snapshot.next_two_days[0].away_team == "Denver Nuggets"
        mock_fetch.assert_called_once()

    @patch.object(LivePredictionEngine, "_fetch_realtime_odds")
    def test_both_fail_returns_empty(self, mock_fetch):
        """When all data sources fail, snapshot has no games but doesn't crash."""
        mock_fetch.return_value = []

        engine = LivePredictionEngine(odds_api_key="test_key")
        snapshot = engine.refresh_now()

        assert snapshot.n_total == 0
        assert snapshot.n_live == 0
        assert snapshot.n_today == 0
        assert snapshot.n_tomorrow == 0
        mock_fetch.assert_called_once()
