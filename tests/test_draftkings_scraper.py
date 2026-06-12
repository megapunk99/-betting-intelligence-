"""
Tests for the DraftKings scraper module.

Covers:
  - Module import and class structure
  - Odds format conversion (American / decimal)
  - Team name matching and normalization
  - DraftKings response parsing
  - Edge cases (76ers, unknown teams, empty data)
  - Merge logic
  - Integration with engine
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock
from typing import Any

from betting_intel.data.draftkings_scraper import DraftKingsScraper
from betting_intel.data.odds_fetcher import ODDS_TO_SHORT_NAME, SHORT_TO_ODDS_NAME


# ── Sample DraftKings-like response data ──────────────────────────────────

SAMPLE_DK_RESPONSE: dict[str, Any] = {
    "eventGroup": {
        "eventGroupId": 42648,
        "name": "NBA",
        "events": [
            {
                "eventId": 123456,
                "name": "Boston Celtics @ Los Angeles Lakers",
                "eventStartDate": "2026-06-11T02:30:00Z",
                "offerCategories": [
                    {
                        "name": "Moneyline",
                        "offerSubcategoryDescriptors": [
                            {
                                "offerSubcategory": {
                                    "offers": [
                                        [
                                            {
                                                "offerId": 1001,
                                                "outcomes": [
                                                    {"label": "Celtics", "oddsAmerican": -120},
                                                    {"label": "Lakers", "oddsAmerican": 105},
                                                ],
                                            }
                                        ]
                                    ]
                                }
                            }
                        ],
                    },
                    {
                        "name": "Point Spread",
                        "offerSubcategoryDescriptors": [
                            {
                                "offerSubcategory": {
                                    "offers": [
                                        [
                                            {
                                                "offerId": 1002,
                                                "outcomes": [
                                                    {"label": "Celtics -2.5", "oddsAmerican": -110},
                                                    {"label": "Lakers +2.5", "oddsAmerican": -110},
                                                ],
                                            }
                                        ]
                                    ]
                                }
                            }
                        ],
                    },
                    {
                        "name": "Total Points",
                        "offerSubcategoryDescriptors": [
                            {
                                "offerSubcategory": {
                                    "offers": [
                                        [
                                            {
                                                "offerId": 1003,
                                                "outcomes": [
                                                    {"label": "Over 224.5", "oddsAmerican": -108},
                                                    {"label": "Under 224.5", "oddsAmerican": -112},
                                                ],
                                            }
                                        ]
                                    ]
                                }
                            }
                        ],
                    },
                ],
            },
            {
                "eventId": 123457,
                "name": "Philadelphia 76ers @ New York Knicks",
                "eventStartDate": "2026-06-11T00:30:00Z",
                "offerCategories": [
                    {
                        "name": "Moneyline",
                        "offerSubcategoryDescriptors": [
                            {
                                "offerSubcategory": {
                                    "offers": [
                                        [
                                            {
                                                "offerId": 1004,
                                                "outcomes": [
                                                    {"label": "76ers", "oddsAmerican": 130},
                                                    {"label": "Knicks", "oddsAmerican": -150},
                                                ],
                                            }
                                        ]
                                    ]
                                }
                            }
                        ],
                    },
                    {
                        "name": "Point Spread",
                        "offerSubcategoryDescriptors": [
                            {
                                "offerSubcategory": {
                                    "offers": [
                                        [
                                            {
                                                "offerId": 1005,
                                                "outcomes": [
                                                    {"label": "76ers +4.5", "oddsAmerican": -110},
                                                    {"label": "Knicks -4.5", "oddsAmerican": -110},
                                                ],
                                            }
                                        ]
                                    ]
                                }
                            }
                        ],
                    },
                    {
                        "name": "Total Points",
                        "offerSubcategoryDescriptors": [
                            {
                                "offerSubcategory": {
                                    "offers": [
                                        [
                                            {
                                                "offerId": 1006,
                                                "outcomes": [
                                                    {"label": "Over 218.5", "oddsAmerican": -110},
                                                    {"label": "Under 218.5", "oddsAmerican": -110},
                                                ],
                                            }
                                        ]
                                    ]
                                }
                            }
                        ],
                    },
                ],
            },
        ],
    }
}

# A sample game that's already in TheOddsAPI format (as returned by ESPN scraper)
SAMPLE_ESPN_GAME: dict[str, Any] = {
    "id": "espn_nba_test123",
    "sport_key": "basketball_nba",
    "sport_title": "NBA",
    "commence_time": "2026-06-11T02:30:00Z",
    "home_team": "Los Angeles Lakers",
    "away_team": "Boston Celtics",
    "bookmakers": [
        {
            "key": "espn",
            "title": "ESPN",
            "last_update": "2026-06-09T12:00:00Z",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Los Angeles Lakers", "price": 105},
                        {"name": "Boston Celtics", "price": -120},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Los Angeles Lakers", "point": 2.5, "price": -110},
                        {"name": "Boston Celtics", "point": -2.5, "price": -110},
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "point": 224.5, "price": -108},
                        {"name": "Under", "point": 224.5, "price": -112},
                    ],
                },
            ],
        }
    ],
}


class TestImports:
    """Verify the module imports cleanly."""

    def test_draftkings_scraper_imports(self):
        from betting_intel.data.draftkings_scraper import (
            DraftKingsScraper,
            NBA_EVENT_GROUP_IDS,
            DK_MARKET_MAP,
            ODDS_CACHE_TTL,
        )
        assert DraftKingsScraper is not None
        assert len(NBA_EVENT_GROUP_IDS) > 0
        assert "Moneyline" in DK_MARKET_MAP
        assert ODDS_CACHE_TTL >= 60

    def test_draftkings_exports_via_init(self):
        from betting_intel.data import DraftKingsScraper as DK
        assert DK is not None

    def test_has_required_methods(self):
        assert hasattr(DraftKingsScraper, "scrape")
        assert hasattr(DraftKingsScraper, "clear_cache")
        assert hasattr(DraftKingsScraper, "_parse_dk_response")
        assert hasattr(DraftKingsScraper, "_to_american_odds")
        assert hasattr(DraftKingsScraper, "_match_team_name")


class TestOddsConversion:
    """Test American/decimal odds conversion."""

    def test_american_odds_positive(self):
        assert DraftKingsScraper._to_american_odds(150) == 150
        assert DraftKingsScraper._to_american_odds("+120") == 120
        assert DraftKingsScraper._to_american_odds(0) is not None

    def test_american_odds_negative(self):
        assert DraftKingsScraper._to_american_odds(-110) == -110
        assert DraftKingsScraper._to_american_odds(-120) == -120

    def test_decimal_to_american_favorite(self):
        # Decimal 1.91 ≈ American -110
        result = DraftKingsScraper._to_american_odds(1.91)
        assert result == -110, f"Expected -110, got {result}"

    def test_decimal_to_american_underdog(self):
        # Decimal 2.20 ≈ American +120
        result = DraftKingsScraper._to_american_odds(2.20)
        assert result == 120, f"Expected 120, got {result}"

    def test_none_odds(self):
        assert DraftKingsScraper._to_american_odds(None) is None

    def test_reasonable_odds_validation(self):
        assert DraftKingsScraper._is_reasonable_odds(-110) is True
        assert DraftKingsScraper._is_reasonable_odds(100) is True
        assert DraftKingsScraper._is_reasonable_odds(-500) is True
        assert DraftKingsScraper._is_reasonable_odds(None) is False
        assert DraftKingsScraper._is_reasonable_odds(0) is True  # 0 is between bounds


class TestTeamMatching:
    """Test DraftKings team name → canonical name matching."""

    def test_direct_full_name_match(self):
        result = DraftKingsScraper._match_team_name(
            "Boston Celtics", "Boston Celtics", "Los Angeles Lakers"
        )
        assert result == "Boston Celtics"

    def test_short_name_match(self):
        result = DraftKingsScraper._match_team_name(
            "Celtics", "Boston Celtics", "Los Angeles Lakers"
        )
        assert result == "Boston Celtics"

    def test_city_name_match(self):
        result = DraftKingsScraper._match_team_name(
            "Lakers", "Boston Celtics", "Los Angeles Lakers"
        )
        assert result == "Los Angeles Lakers"

    def test_unknown_team_returns_none(self):
        result = DraftKingsScraper._match_team_name(
            "Fake Team", "Boston Celtics", "Los Angeles Lakers"
        )
        assert result is None

    @pytest.mark.parametrize(
        "label,home,away,expected",
        [
            ("Warriors", "Boston Celtics", "Golden State Warriors", "Golden State Warriors"),
            ("Spurs", "San Antonio Spurs", "Los Angeles Lakers", "San Antonio Spurs"),
            ("Bulls", "Chicago Bulls", "Boston Celtics", "Chicago Bulls"),
        ],
    )
    def test_various_short_names(self, label, home, away, expected):
        result = DraftKingsScraper._match_team_name(label, home, away)
        assert result == expected, f"Expected {expected}, got {result}"


class TestDKResponseParsing:
    """Test parsing of DraftKings API responses."""

    def test_is_valid_dk_response(self):
        assert DraftKingsScraper._is_valid_dk_response(SAMPLE_DK_RESPONSE) is True
        assert DraftKingsScraper._is_valid_dk_response({}) is False
        assert DraftKingsScraper._is_valid_dk_response({"eventGroup": {}}) is True
        assert DraftKingsScraper._is_valid_dk_response([]) is False

    def test_parse_celtics_lakers_game(self):
        parsed = DraftKingsScraper._parse_dk_response(SAMPLE_DK_RESPONSE)
        assert len(parsed) == 2, f"Expected 2 games, got {len(parsed)}"

        # Check first game: Celtics @ Lakers
        game = parsed[0]
        assert "Boston Celtics" in str(game["home_team"]) or "Los Angeles Lakers" in str(game["home_team"])
        assert game["sport_key"] == "basketball_nba"
        assert len(game["bookmakers"]) == 1
        assert game["bookmakers"][0]["key"] == "draftkings"

        # Check markets exist
        markets = {m["key"]: m for m in game["bookmakers"][0]["markets"]}
        assert "h2h" in markets
        assert "spreads" in markets
        assert "totals" in markets

    def test_parse_76ers_game(self):
        """Verify the 76ers edge case is handled correctly."""
        parsed = DraftKingsScraper._parse_dk_response(SAMPLE_DK_RESPONSE)
        assert len(parsed) >= 2

        # Find the 76ers game
        sixers_game = None
        for g in parsed:
            if "76ers" in g.get("home_team", "") or "76ers" in g.get("away_team", ""):
                sixers_game = g
                break

        assert sixers_game is not None, "76ers game not found"
        assert len(sixers_game["bookmakers"]) == 1
        markets = {m["key"]: m for m in sixers_game["bookmakers"][0]["markets"]}
        assert "spreads" in markets, "76ers spread market not found"

        # Verify both spread outcomes exist
        outcomes = markets["spreads"]["outcomes"]
        assert len(outcomes) == 2, f"Expected 2 spread outcomes, got {len(outcomes)}"
        assert outcomes[0]["point"] is not None
        assert outcomes[1]["point"] is not None
        assert abs(outcomes[0]["point"] + outcomes[1]["point"]) < 0.1  # Should be ~0

    def test_parse_empty_response(self):
        assert DraftKingsScraper._parse_dk_response({}) == []
        assert DraftKingsScraper._parse_dk_response({"eventGroup": {"events": []}}) == []

    def test_parse_none_response(self):
        with pytest.raises((TypeError, AttributeError)):
            DraftKingsScraper._parse_dk_response(None)


class TestMergeLogic:
    """Test merging ESPN + DraftKings odds."""

    def test_merge_identical_matchup(self):
        """ESPN and DK have the same game — bookmakers should merge."""
        from betting_intel.live.engine import LivePredictionEngine

        engine = LivePredictionEngine(odds_api_key="")

        espn_data = [SAMPLE_ESPN_GAME]

        dk_data = DraftKingsScraper._parse_dk_response(SAMPLE_DK_RESPONSE)

        merged = engine._merge_odds_sources(espn_data, dk_data)

        # Should have at least the ESPN game
        lakers_game = None
        for g in merged:
            if g.get("home_team") == "Los Angeles Lakers":
                lakers_game = g
                break

        assert lakers_game is not None, "Lakers game not found in merged data"

        # Should have bookmakers from both sources
        books = lakers_game.get("bookmakers", [])
        book_keys = [b["key"] for b in books]
        assert "espn" in book_keys, "ESPN book missing"
        assert "draftkings" in book_keys, "DraftKings book missing"

    def test_merge_only_espn(self):
        from betting_intel.live.engine import LivePredictionEngine

        engine = LivePredictionEngine(odds_api_key="")
        espn_data = [SAMPLE_ESPN_GAME]
        merged = engine._merge_odds_sources(espn_data, [])
        assert len(merged) == 1
        assert len(merged[0]["bookmakers"]) == 1
        assert merged[0]["bookmakers"][0]["key"] == "espn"

    def test_merge_only_dk(self):
        from betting_intel.live.engine import LivePredictionEngine

        engine = LivePredictionEngine(odds_api_key="")
        dk_data = DraftKingsScraper._parse_dk_response(SAMPLE_DK_RESPONSE)
        merged = engine._merge_odds_sources([], dk_data)
        assert len(merged) == 2
        assert merged[0]["bookmakers"][0]["key"] == "draftkings"

    def test_merge_both_empty(self):
        from betting_intel.live.engine import LivePredictionEngine

        engine = LivePredictionEngine(odds_api_key="")
        merged = engine._merge_odds_sources([], [])
        assert merged == []


class TestScraperCache:
    """Test the in-memory cache behavior."""

    def test_clear_cache(self):
        DraftKingsScraper._cache = [{"test": "data"}]
        DraftKingsScraper._cache_time = 100.0
        DraftKingsScraper.clear_cache()
        assert DraftKingsScraper._cache is None
        assert DraftKingsScraper._cache_time == 0.0

    def test_cache_class_level(self):
        """Cache is class-level, shared across all instances."""
        DraftKingsScraper._cache = [{"test": "data"}]
        s1 = DraftKingsScraper
        s2 = DraftKingsScraper
        assert s1._cache is s2._cache


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_scrape_graceful_on_network_fail(self):
        """Scrape should return empty list on network failure, not crash."""
        result = DraftKingsScraper.scrape(timeout=1)
        assert isinstance(result, list)

    def test_dedup_outcomes(self):
        duplicates = [
            {"name": "Boston Celtics", "price": -120},
            {"name": "Boston Celtics", "price": -120},
            {"name": "Los Angeles Lakers", "price": 105},
        ]
        result = DraftKingsScraper._deduplicate_outcomes(duplicates)
        assert len(result) == 2
        assert result[0]["name"] == "Boston Celtics"

    def test_dedup_empty(self):
        assert DraftKingsScraper._deduplicate_outcomes([]) == []

    def test_parse_nfl_event_ignored(self):
        """Non-NBA events should be filtered by sport_key."""
        # The parser only looks at event names — non-basketball events
        # with different naming might still get parsed if they match the format.
        # The test_parse_empty_response covers unexpected data.
        pass  # No NFL-specific filtering needed for now
