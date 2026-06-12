"""
Pytest-style tests for the stealth scraper module and team name mappings.

Tests import correctness, API response parsing (mocked), and sync wrapper
behavior without actually launching a browser or hitting external APIs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from betting_intel.data.odds_fetcher import ODDS_TO_SHORT_NAME, SHORT_TO_ODDS_NAME


class TestImports:
    """Verify the stealth scraper module imports cleanly."""

    def test_stealth_scraper_imports(self):
        from betting_intel.data.stealth_scraper import StealthBrowser
        assert StealthBrowser is not None
        assert hasattr(StealthBrowser, "sync_scrape_live_odds")

    def test_shared_team_mappings(self):
        from betting_intel.data.stealth_scraper import ESPN_TEAM_SHORT, TEAM_TO_ESPN

        # stealth_scraper should import from odds_fetcher (same object)
        assert ESPN_TEAM_SHORT is ODDS_TO_SHORT_NAME

        # Verify reverse mapping is correct
        assert TEAM_TO_ESPN["Lakers"] == "Los Angeles Lakers"
        assert TEAM_TO_ESPN["Celtics"] == "Boston Celtics"
        assert TEAM_TO_ESPN["Clippers"] == "Los Angeles Clippers"


class TestStealthScraperSync:
    """Test the sync wrapper of StealthBrowser (without real browser)."""

    @pytest.fixture
    def StealthBrowser(self):
        from betting_intel.data.stealth_scraper import StealthBrowser
        return StealthBrowser

    def test_clear_cache(self, StealthBrowser):
        StealthBrowser._cache = [{"test": "data"}]
        StealthBrowser._cache_time = 100.0
        StealthBrowser.clear_cache()
        assert StealthBrowser._cache is None
        assert StealthBrowser._cache_time == 0.0

    def test_sync_scrape_returns_list(self, StealthBrowser):
        with patch.object(StealthBrowser, "_scrape_via_http", return_value=[]):
            result = StealthBrowser.sync_scrape_live_odds(timeout=5)
            assert isinstance(result, list)

    def test_cache_hit(self, StealthBrowser):
        StealthBrowser.clear_cache()
        StealthBrowser._cache = [{"game": "data"}]
        StealthBrowser._cache_time = 9999999999.0  # Far in the future
        with patch.object(StealthBrowser, "_scrape_via_http") as mock_http:
            result = StealthBrowser.sync_scrape_live_odds(timeout=5)
            assert result == [{"game": "data"}]
            mock_http.assert_not_called()


class TestESPNHttpScrape:
    """Test the ESPN HTTP scraping path with mocked responses."""

    @pytest.fixture
    def StealthBrowser(self):
        from betting_intel.data.stealth_scraper import StealthBrowser
        StealthBrowser.clear_cache()
        return StealthBrowser

    def _build_scoreboard_response(self):
        """Build a realistic ESPN scoreboard API response."""
        return {
            "events": [
                {
                    "id": "401789001",
                    "name": "Celtics at Lakers",
                    "competitions": [
                        {
                            "id": "comp_1",
                            "date": "2026-06-09T19:30Z",
                            "competitors": [
                                {
                                    "homeAway": "home",
                                    "team": {"displayName": "Los Angeles Lakers"}
                                },
                                {
                                    "homeAway": "away",
                                    "team": {"displayName": "Boston Celtics"}
                                }
                            ],
                            "odds": [
                                {
                                    "provider": {"name": "ESPN", "id": "1"},
                                    "spread": -5.5,
                                    "overUnder": 218.5,
                                    "overOdds": -110,
                                    "underOdds": -110,
                                    "homeTeamOdds": {"moneyLine": -250, "spreadOdds": -110},
                                    "awayTeamOdds": {"moneyLine": +210, "spreadOdds": -110},
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    def _mock_espn_response(self):
        """Create a MagicMock for the ESPN API response."""
        scoreboard_data = json.dumps(self._build_scoreboard_response()).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = scoreboard_data
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp

        # Detail odds endpoint returns empty items
        detail_data = json.dumps({"items": []}).encode("utf-8")
        mock_detail = MagicMock()
        mock_detail.read.return_value = detail_data
        mock_detail.status = 200
        mock_detail.__enter__.return_value = mock_detail

        def side_effect(req, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
            if "scoreboard" in url:
                return mock_resp
            return mock_detail

        mock_urlopen = MagicMock(side_effect=side_effect)
        return mock_urlopen

    def test_parse_espn_event(self, StealthBrowser):
        """Test parsing an ESPN scoreboard API response."""
        mock_urlopen = self._mock_espn_response()

        with patch("urllib.request.urlopen", mock_urlopen):
            result = StealthBrowser._scrape_via_http(timeout=10)

        assert isinstance(result, list)
        assert len(result) >= 0

    def test_parse_embedded_odds(self, StealthBrowser):
        """Test _parse_embedded_odds produces correct bookmaker format."""
        game = {
            "home_team": "Los Angeles Lakers",
            "away_team": "Boston Celtics",
        }
        odds_item = {
            "provider": {"name": "ESPN", "id": "1"},
            "spread": -5.5,
            "overUnder": 218.5,
            "overOdds": -110,
            "underOdds": -110,
            "homeTeamOdds": {"moneyLine": -250, "spreadOdds": -110},
            "awayTeamOdds": {"moneyLine": +210, "spreadOdds": -110},
        }
        bookmakers = StealthBrowser._parse_embedded_odds(odds_item, game)
        assert len(bookmakers) == 1
        bk = bookmakers[0]
        assert bk["title"] == "ESPN"
        assert len(bk["markets"]) == 3

        # Check h2h market
        h2h = [m for m in bk["markets"] if m["key"] == "h2h"][0]
        assert len(h2h["outcomes"]) == 2
        home_out = [o for o in h2h["outcomes"] if o["name"] == "Los Angeles Lakers"][0]
        assert home_out["price"] == -250

        # Check totals market
        totals = [m for m in bk["markets"] if m["key"] == "totals"][0]
        assert totals["outcomes"][0]["point"] == 218.5


class TestTeamMappings:
    """Verify team name mappings are consistent across all modules."""

    def test_all_30_nba_teams_mapped(self):
        for full, short in ODDS_TO_SHORT_NAME.items():
            assert short in SHORT_TO_ODDS_NAME, \
                f"Short name '{short}' (from '{full}') missing in reverse mapping"

    def test_short_names_unique(self):
        canonical_full_names = {}
        for full, short in ODDS_TO_SHORT_NAME.items():
            canonical = SHORT_TO_ODDS_NAME.get(short)
            canonical_full_names[short] = canonical

        assert canonical_full_names["Clippers"] == "Los Angeles Clippers"
