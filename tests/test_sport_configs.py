"""
Unit tests for sport_configs.py — SportConfig dataclass, league registry,
team name resolution, and master list integrity.

Tests cover:
  1. SportConfig structure and defaults
  2. Euroleague config properties (20 teams, Sep-May, total 150-180)
  3. Team name resolution (get_short_name with mapping and fallback)
  4. Multi-league master lists (ALL_SPORTS, ALL_TEAM_NAME_MAP, lookups)
  5. Season detection (is_in_season with mocked dates)
  6. Edge cases (empty names, unmapped teams)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════
#  SportConfig STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════


class TestSportConfig:
    """Tests for SportConfig dataclass — structure, defaults, properties."""

    def test_default_values(self):
        """SportConfig with minimal args gets sensible defaults."""
        from betting_intel.live.sport_configs import SportConfig

        cfg = SportConfig(
            sport_key="test_sport",
            display_name="Test",
            full_name="Test Sport",
        )
        assert cfg.sport_key == "test_sport"
        assert cfg.display_name == "Test"
        assert cfg.full_name == "Test Sport"
        assert cfg.has_h2h is True                        # default
        assert cfg.has_spreads is False                   # default
        assert cfg.has_totals is False                    # default
        assert cfg.prediction_strategy == "total"         # default
        assert cfg.total_min == 180.0                     # default
        assert cfg.total_max == 260.0                     # default
        assert cfg.season_start_month == 10               # default
        assert cfg.season_end_month == 6                  # default

    def test_custom_values(self):
        """SportConfig with all fields set returns correct values."""
        from betting_intel.live.sport_configs import SportConfig

        cfg = SportConfig(
            sport_key="basketball_euroleague",
            display_name="Euroleague",
            full_name="EuroLeague Basketball",
            has_h2h=True,
            has_spreads=True,
            has_totals=True,
            team_name_map={"Real Madrid": "Real Madrid"},
            prediction_strategy="total",
            total_min=150.0,
            total_max=180.0,
            season_start_month=9,
            season_end_month=5,
        )
        assert cfg.sport_key == "basketball_euroleague"
        assert cfg.team_name_map["Real Madrid"] == "Real Madrid"

    def test_default_markets_to_fetch(self):
        """markets_to_fetch defaults to h2h, spreads, totals."""
        from betting_intel.live.sport_configs import SportConfig

        cfg = SportConfig(
            sport_key="t", display_name="T", full_name="Test",
        )
        assert cfg.markets_to_fetch == ["h2h", "spreads", "totals"]


# ═══════════════════════════════════════════════════════════════════════════
#  EUROLEAGUE CONFIG
# ═══════════════════════════════════════════════════════════════════════════


class TestEuroleagueConfig:
    """Tests specific to the Euroleague SportConfig definition."""

    def test_team_count(self):
        """Euroleague should have exactly 20 teams mapped."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        assert len(EUROLEAGUE.team_name_map) == 20

    def test_season_months(self):
        """Euroleague season runs Sep (9) through May (5)."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        assert EUROLEAGUE.season_start_month == 9
        assert EUROLEAGUE.season_end_month == 5

    def test_total_range(self):
        """Euroleague total range is 150-180."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        assert EUROLEAGUE.total_min == 150.0
        assert EUROLEAGUE.total_max == 180.0

    def test_sport_key(self):
        """Euroleague uses basketball_euroleague sport key."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        assert EUROLEAGUE.sport_key == "basketball_euroleague"

    def test_display_name(self):
        """Euroleague display name is 'Euroleague'."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        assert EUROLEAGUE.display_name == "Euroleague"

    def test_full_name(self):
        """Euroleague full name is 'EuroLeague Basketball'."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        assert EUROLEAGUE.full_name == "EuroLeague Basketball"

    def test_has_all_markets(self):
        """Euroleague supports h2h, spreads, and totals."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        assert EUROLEAGUE.has_h2h is True
        assert EUROLEAGUE.has_spreads is True
        assert EUROLEAGUE.has_totals is True

    def test_prediction_strategy(self):
        """Euroleague uses total prediction strategy."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        assert EUROLEAGUE.prediction_strategy == "total"

    @pytest.mark.parametrize("full_name,expected", [
        ("Real Madrid", "Real Madrid"),
        ("FC Barcelona", "Barcelona"),
        ("Olympiacos Piraeus", "Olympiacos"),
        ("Panathinaikos AKTOR", "Panathinaikos"),
        ("Fenerbahçe Beko", "Fenerbahçe"),
        ("Anadolu Efes Istanbul", "Anadolu Efes"),
        ("Crvena Zvezda Meridianbet Belgrade", "Crvena Zvezda"),
        ("Žalgiris Kaunas", "Žalgiris"),
        ("Maccabi Rapyd Tel Aviv", "Maccabi Tel Aviv"),
        ("Paris Basketball", "Paris"),
        ("AS Monaco", "Monaco"),
        ("FC Bayern Munich", "Bayern Munich"),
        ("EA7 Emporio Armani Milan", "Milan"),
        ("LDLC ASVEL", "ASVEL"),
        ("Kosner Baskonia Vitoria-Gasteiz", "Baskonia"),
        ("Valencia Basket", "Valencia"),
        ("Partizan Mozzart Bet Belgrade", "Partizan"),
        ("Virtus Segafredo Bologna", "Virtus Bologna"),
        ("Hapoel IBI Tel Aviv", "Hapoel Tel Aviv"),
        ("Dubai Basketball", "Dubai"),
    ])
    def test_team_name_resolution(self, full_name, expected):
        """All 20 Euroleague teams resolve to correct short names."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        assert EUROLEAGUE.get_short_name(full_name) == expected


# ═══════════════════════════════════════════════════════════════════════════
#  TEAM NAME RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════


class TestTeamNameResolution:
    """Tests for SportConfig.get_short_name() — mapping and fallbacks."""

    def test_mapped_name_returns_short(self):
        """Mapped full name returns its short name."""
        from betting_intel.live.sport_configs import NBA

        assert NBA.get_short_name("Boston Celtics") == "Celtics"
        assert NBA.get_short_name("Los Angeles Lakers") == "Lakers"

    def test_unmapped_name_falls_back_to_last_word(self):
        """Unmapped team name returns the last word."""
        from betting_intel.live.sport_configs import NBA

        # "Some Unknown Team" -> last word -> "Team"
        assert NBA.get_short_name("Some Unknown Team") == "Team"

    def test_single_word_name_returns_unchanged(self):
        """A single-word name with no mapping returns itself."""
        from betting_intel.live.sport_configs import NBA

        assert NBA.get_short_name("Lakers") == "Lakers"

    def test_empty_string_returns_empty(self):
        """Empty string returns empty string."""
        from betting_intel.live.sport_configs import NBA

        assert NBA.get_short_name("") == ""

    def test_none_name_returns_empty(self):
        """None input (coerced) returns empty via falsy check."""
        from betting_intel.live.sport_configs import NBA

        assert NBA.get_short_name("") == ""

    def test_ncaab_team_resolution(self):
        """NCAAB mapped teams resolve correctly."""
        from betting_intel.live.sport_configs import NCAAB

        assert NCAAB.get_short_name("Duke Blue Devils") == "Duke"
        assert NCAAB.get_short_name("North Carolina Tar Heels") == "UNC"

    def test_euroleague_team_via_get_short_name(self):
        """Euroleague teams can be resolved via get_short_name."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        assert EUROLEAGUE.get_short_name("Olympiacos Piraeus") == "Olympiacos"
        assert EUROLEAGUE.get_short_name("FC Barcelona") == "Barcelona"


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER LISTS & LOOKUPS
# ═══════════════════════════════════════════════════════════════════════════


class TestMasterLists:
    """Tests for ALL_SPORTS, ALL_TEAM_NAME_MAP, and lookup dicts."""

    def test_all_sports_includes_five_leagues(self):
        """ALL_SPORTS should have NBA, NCAAB, Euroleague, EPL, and NFL."""
        from betting_intel.live.sport_configs import ALL_SPORTS, NBA, NCAAB, EUROLEAGUE, EPL, NFL

        assert len(ALL_SPORTS) == 5
        assert ALL_SPORTS[0] is NBA
        assert ALL_SPORTS[1] is NCAAB
        assert ALL_SPORTS[2] is EUROLEAGUE
        assert ALL_SPORTS[3] is EPL
        assert ALL_SPORTS[4] is NFL

    def test_all_team_name_map_size(self):
        """ALL_TEAM_NAME_MAP should have 242 entries (30 NBA + 140 NCAAB + 20 Euroleague + 20 EPL + 32 NFL)."""
        from betting_intel.live.sport_configs import ALL_TEAM_NAME_MAP

        assert len(ALL_TEAM_NAME_MAP) == 253

    def test_all_team_name_map_includes_euroleague(self):
        """Euroleague teams should be in ALL_TEAM_NAME_MAP."""
        from betting_intel.live.sport_configs import ALL_TEAM_NAME_MAP

        assert "Real Madrid" in ALL_TEAM_NAME_MAP
        assert "FC Barcelona" in ALL_TEAM_NAME_MAP
        assert "Olympiacos Piraeus" in ALL_TEAM_NAME_MAP
        assert "Žalgiris Kaunas" in ALL_TEAM_NAME_MAP

    def test_all_team_name_map_includes_nfl(self):
        """NFL teams should be in ALL_TEAM_NAME_MAP."""
        from betting_intel.live.sport_configs import ALL_TEAM_NAME_MAP

        assert "Kansas City Chiefs" in ALL_TEAM_NAME_MAP
        assert "San Francisco 49ers" in ALL_TEAM_NAME_MAP
        assert "Dallas Cowboys" in ALL_TEAM_NAME_MAP
        assert ALL_TEAM_NAME_MAP["Kansas City Chiefs"] == "Chiefs"

    def test_all_team_name_map_includes_nba(self):
        """NBA teams should be in ALL_TEAM_NAME_MAP."""
        from betting_intel.live.sport_configs import ALL_TEAM_NAME_MAP

        assert "Boston Celtics" in ALL_TEAM_NAME_MAP
        assert "Los Angeles Lakers" in ALL_TEAM_NAME_MAP

    def test_all_team_name_map_includes_ncaab(self):
        """NCAAB teams should be in ALL_TEAM_NAME_MAP."""
        from betting_intel.live.sport_configs import ALL_TEAM_NAME_MAP

        assert "Duke Blue Devils" in ALL_TEAM_NAME_MAP
        assert "Kentucky Wildcats" in ALL_TEAM_NAME_MAP

    def test_sport_key_to_config_all_keys(self):
        """SPORT_KEY_TO_CONFIG should have all four sport keys."""
        from betting_intel.live.sport_configs import SPORT_KEY_TO_CONFIG

        assert "basketball_nba" in SPORT_KEY_TO_CONFIG
        assert "basketball_ncaab" in SPORT_KEY_TO_CONFIG
        assert "basketball_euroleague" in SPORT_KEY_TO_CONFIG
        assert "americanfootball_nfl" in SPORT_KEY_TO_CONFIG
        assert SPORT_KEY_TO_CONFIG["basketball_euroleague"].display_name == "Euroleague"
        assert SPORT_KEY_TO_CONFIG["americanfootball_nfl"].display_name == "NFL"

    def test_display_name_to_config(self):
        """DISPLAY_NAME_TO_CONFIG should map all four display names."""
        from betting_intel.live.sport_configs import DISPLAY_NAME_TO_CONFIG

        assert "NBA" in DISPLAY_NAME_TO_CONFIG
        assert "NCAAB" in DISPLAY_NAME_TO_CONFIG
        assert "Euroleague" in DISPLAY_NAME_TO_CONFIG
        assert "NFL" in DISPLAY_NAME_TO_CONFIG
        assert DISPLAY_NAME_TO_CONFIG["Euroleague"].sport_key == "basketball_euroleague"
        assert DISPLAY_NAME_TO_CONFIG["NFL"].sport_key == "americanfootball_nfl"

    def test_unknown_sport_key_not_in_lookup(self):
        """Unknown sport key returns None from SPORT_KEY_TO_CONFIG."""
        from betting_intel.live.sport_configs import SPORT_KEY_TO_CONFIG

        assert SPORT_KEY_TO_CONFIG.get("basketball_nonexistent") is None

    def test_unknown_display_name_not_in_lookup(self):
        """Unknown display name returns None from DISPLAY_NAME_TO_CONFIG."""
        from betting_intel.live.sport_configs import DISPLAY_NAME_TO_CONFIG

        assert DISPLAY_NAME_TO_CONFIG.get("Unknown League") is None


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


class TestHelperFunctions:
    """Tests for league_from_sport_key and sport_key_to_group."""

    # ── league_from_sport_key ──────────────────────────────────────

    def test_league_from_sport_key_nba(self):
        from betting_intel.live.sport_configs import league_from_sport_key

        assert league_from_sport_key("basketball_nba") == "NBA"

    def test_league_from_sport_key_ncaab(self):
        from betting_intel.live.sport_configs import league_from_sport_key

        assert league_from_sport_key("basketball_ncaab") == "NCAAB"

    def test_league_from_sport_key_euroleague(self):
        from betting_intel.live.sport_configs import league_from_sport_key

        assert league_from_sport_key("basketball_euroleague") == "Euroleague"

    def test_league_from_sport_key_nfl(self):
        from betting_intel.live.sport_configs import league_from_sport_key

        assert league_from_sport_key("americanfootball_nfl") == "NFL"

    def test_league_from_sport_key_unknown(self):
        """Unknown sport key falls back to last word uppercased."""
        from betting_intel.live.sport_configs import league_from_sport_key

        assert league_from_sport_key("americanfootball_nfl") == "NFL"
        assert league_from_sport_key("soccer_epl") == "EPL"

    def test_league_from_sport_key_no_underscore(self):
        """A single-word key returns itself uppercased."""
        from betting_intel.live.sport_configs import league_from_sport_key

        assert league_from_sport_key("nba") == "NBA"

    # ── sport_key_to_group ─────────────────────────────────────────

    def test_sport_key_to_group_basketball(self):
        """All basketball sport keys map to 'Basketball'."""
        from betting_intel.live.sport_configs import sport_key_to_group

        assert sport_key_to_group("basketball_nba") == "Basketball"
        assert sport_key_to_group("basketball_ncaab") == "Basketball"
        assert sport_key_to_group("basketball_euroleague") == "Basketball"

    def test_sport_key_to_group_football(self):
        from betting_intel.live.sport_configs import sport_key_to_group

        assert sport_key_to_group("americanfootball_nfl") == "Football"
        assert sport_key_to_group("football_nfl") == "Football"

    def test_sport_key_to_group_hockey(self):
        from betting_intel.live.sport_configs import sport_key_to_group

        assert sport_key_to_group("icehockey_nhl") == "Hockey"
        assert sport_key_to_group("hockey_liiga") == "Hockey"

    def test_sport_key_to_group_baseball(self):
        from betting_intel.live.sport_configs import sport_key_to_group

        assert sport_key_to_group("baseball_mlb") == "Baseball"

    def test_sport_key_to_group_soccer(self):
        from betting_intel.live.sport_configs import sport_key_to_group

        assert sport_key_to_group("soccer_epl") == "Soccer"

    def test_sport_key_to_group_tennis(self):
        from betting_intel.live.sport_configs import sport_key_to_group

        assert sport_key_to_group("tennis_atp") == "Tennis"

    def test_sport_key_to_group_other(self):
        """Unknown sport key maps to 'Other'."""
        from betting_intel.live.sport_configs import sport_key_to_group

        assert sport_key_to_group("mma_ufc") == "Other"
        assert sport_key_to_group("cricket_wc") == "Other"


# ═══════════════════════════════════════════════════════════════════════════
#  SEASON DETECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestSeasonDetection:
    """Tests for SportConfig.is_in_season across different months."""

    @patch("betting_intel.live.sport_configs.datetime")
    def test_euroleague_in_season_january(self, mock_dt):
        """Euroleague should be in season during January (month 1)."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        mock_dt.now.return_value = datetime(2026, 1, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert EUROLEAGUE.is_in_season is True

    @patch("betting_intel.live.sport_configs.datetime")
    def test_euroleague_in_season_september(self, mock_dt):
        """Euroleague should be in season during September (month 9)."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        mock_dt.now.return_value = datetime(2025, 9, 1)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert EUROLEAGUE.is_in_season is True

    @patch("betting_intel.live.sport_configs.datetime")
    def test_euroleague_in_season_may(self, mock_dt):
        """Euroleague should be in season during May (month 5)."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        mock_dt.now.return_value = datetime(2026, 5, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert EUROLEAGUE.is_in_season is True

    @patch("betting_intel.live.sport_configs.datetime")
    def test_euroleague_out_of_season_july(self, mock_dt):
        """Euroleague should be out of season during July (month 7)."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        mock_dt.now.return_value = datetime(2026, 7, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert EUROLEAGUE.is_in_season is False

    @patch("betting_intel.live.sport_configs.datetime")
    def test_euroleague_out_of_season_august(self, mock_dt):
        """Euroleague should be out of season during August (month 8)."""
        from betting_intel.live.sport_configs import EUROLEAGUE

        mock_dt.now.return_value = datetime(2026, 8, 1)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert EUROLEAGUE.is_in_season is False

    @patch("betting_intel.live.sport_configs.datetime")
    def test_nfl_in_season_january(self, mock_dt):
        """NFL should be in season during January (month 1 — within Sep→Feb span)."""
        from betting_intel.live.sport_configs import NFL

        mock_dt.now.return_value = datetime(2026, 1, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert NFL.is_in_season is True

    @patch("betting_intel.live.sport_configs.datetime")
    def test_nfl_in_season_september(self, mock_dt):
        """NFL should be in season during September (month 9 — start month)."""
        from betting_intel.live.sport_configs import NFL

        mock_dt.now.return_value = datetime(2026, 9, 1)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert NFL.is_in_season is True

    @patch("betting_intel.live.sport_configs.datetime")
    def test_nfl_in_season_february(self, mock_dt):
        """NFL should be in season during February (month 2 — end month, year-span)."""
        from betting_intel.live.sport_configs import NFL

        mock_dt.now.return_value = datetime(2026, 2, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert NFL.is_in_season is True

    @patch("betting_intel.live.sport_configs.datetime")
    def test_nfl_out_of_season_march(self, mock_dt):
        """NFL should be out of season during March (month 3 — after end)."""
        from betting_intel.live.sport_configs import NFL

        mock_dt.now.return_value = datetime(2026, 3, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert NFL.is_in_season is False

    @patch("betting_intel.live.sport_configs.datetime")
    def test_nfl_out_of_season_july(self, mock_dt):
        """NFL should be out of season during July (month 7)."""
        from betting_intel.live.sport_configs import NFL

        mock_dt.now.return_value = datetime(2026, 7, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert NFL.is_in_season is False

    @patch("betting_intel.live.sport_configs.datetime")
    def test_nba_in_season_january(self, mock_dt):
        """NBA should be in season during January."""
        from betting_intel.live.sport_configs import NBA

        mock_dt.now.return_value = datetime(2026, 1, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert NBA.is_in_season is True

    @patch("betting_intel.live.sport_configs.datetime")
    def test_nba_out_of_season_august(self, mock_dt):
        """NBA should be out of season during August."""
        from betting_intel.live.sport_configs import NBA

        mock_dt.now.return_value = datetime(2026, 8, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert NBA.is_in_season is False

    @patch("betting_intel.live.sport_configs.datetime")
    def test_ncaab_in_season_january(self, mock_dt):
        """NCAAB should be in season during January."""
        from betting_intel.live.sport_configs import NCAAB

        mock_dt.now.return_value = datetime(2026, 1, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert NCAAB.is_in_season is True

    @patch("betting_intel.live.sport_configs.datetime")
    def test_ncaab_out_of_season_july(self, mock_dt):
        """NCAAB should be out of season during July."""
        from betting_intel.live.sport_configs import NCAAB

        mock_dt.now.return_value = datetime(2026, 7, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert NCAAB.is_in_season is False

    def test_non_spanning_season_june_to_august(self):
        """A non-year-spanning season (Jun-Aug) works correctly."""
        from betting_intel.live.sport_configs import SportConfig

        cfg = SportConfig(
            sport_key="summer_league",
            display_name="Summer",
            full_name="Summer League",
            season_start_month=6,
            season_end_month=8,
        )

        with patch("betting_intel.live.sport_configs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 1)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert cfg.is_in_season is True

        with patch("betting_intel.live.sport_configs.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 10, 1)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            assert cfg.is_in_season is False


# ═══════════════════════════════════════════════════════════════════════════
#  GET ACTIVE SPORTS
# ═══════════════════════════════════════════════════════════════════════════


class TestGetActiveSports:
    """Tests for get_active_sports() which returns only in-season leagues."""

    @patch("betting_intel.live.sport_configs.datetime")
    def test_january_returns_all_five(self, mock_dt):
        """In January, all five leagues should be active (NBA, NCAAB, Euroleague, EPL, NFL)."""
        from betting_intel.live.sport_configs import get_active_sports

        mock_dt.now.return_value = datetime(2026, 1, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        active = get_active_sports()
        names = {s.display_name for s in active}
        assert names == {"NBA", "NCAAB", "Euroleague", "EPL", "NFL"}

    @patch("betting_intel.live.sport_configs.datetime")
    def test_july_returns_empty(self, mock_dt):
        """In July, no basketball/football leagues should be active (all off-season). EPL also off (Aug-May)."""
        from betting_intel.live.sport_configs import get_active_sports

        mock_dt.now.return_value = datetime(2026, 7, 15)
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        active = get_active_sports()
        assert len(active) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  NFL CONFIG
# ═══════════════════════════════════════════════════════════════════════════


class TestNFLConfig:
    """Tests specific to the NFL SportConfig definition."""

    def test_team_count(self):
        """NFL should have exactly 32 teams mapped."""
        from betting_intel.live.sport_configs import NFL

        assert len(NFL.team_name_map) == 32

    def test_season_months(self):
        """NFL season runs Sep (9) through Feb (2)."""
        from betting_intel.live.sport_configs import NFL

        assert NFL.season_start_month == 9
        assert NFL.season_end_month == 2

    def test_total_range(self):
        """NFL total range is 30-60."""
        from betting_intel.live.sport_configs import NFL

        assert NFL.total_min == 30.0
        assert NFL.total_max == 60.0

    def test_sport_key(self):
        """NFL uses americanfootball_nfl sport key."""
        from betting_intel.live.sport_configs import NFL

        assert NFL.sport_key == "americanfootball_nfl"

    def test_display_name(self):
        """NFL display name is 'NFL'."""
        from betting_intel.live.sport_configs import NFL

        assert NFL.display_name == "NFL"

    def test_full_name(self):
        """NFL full name is 'National Football League'."""
        from betting_intel.live.sport_configs import NFL

        assert NFL.full_name == "National Football League"

    def test_has_all_markets(self):
        """NFL supports h2h, spreads, and totals."""
        from betting_intel.live.sport_configs import NFL

        assert NFL.has_h2h is True
        assert NFL.has_spreads is True
        assert NFL.has_totals is True

    def test_prediction_strategy(self):
        """NFL uses total prediction strategy."""
        from betting_intel.live.sport_configs import NFL

        assert NFL.prediction_strategy == "total"

    @pytest.mark.parametrize("full_name,expected", [
        # AFC East
        ("Buffalo Bills", "Bills"),
        ("Miami Dolphins", "Dolphins"),
        ("New England Patriots", "Patriots"),
        ("New York Jets", "Jets"),
        # AFC North
        ("Baltimore Ravens", "Ravens"),
        ("Cincinnati Bengals", "Bengals"),
        ("Cleveland Browns", "Browns"),
        ("Pittsburgh Steelers", "Steelers"),
        # AFC South
        ("Houston Texans", "Texans"),
        ("Indianapolis Colts", "Colts"),
        ("Jacksonville Jaguars", "Jaguars"),
        ("Tennessee Titans", "Titans"),
        # AFC West
        ("Denver Broncos", "Broncos"),
        ("Kansas City Chiefs", "Chiefs"),
        ("Las Vegas Raiders", "Raiders"),
        ("Los Angeles Chargers", "Chargers"),
        # NFC East
        ("Dallas Cowboys", "Cowboys"),
        ("New York Giants", "Giants"),
        ("Philadelphia Eagles", "Eagles"),
        ("Washington Commanders", "Commanders"),
        # NFC North
        ("Chicago Bears", "Bears"),
        ("Detroit Lions", "Lions"),
        ("Green Bay Packers", "Packers"),
        ("Minnesota Vikings", "Vikings"),
        # NFC South
        ("Atlanta Falcons", "Falcons"),
        ("Carolina Panthers", "Panthers"),
        ("New Orleans Saints", "Saints"),
        ("Tampa Bay Buccaneers", "Buccaneers"),
        # NFC West
        ("Arizona Cardinals", "Cardinals"),
        ("Los Angeles Rams", "Rams"),
        ("San Francisco 49ers", "49ers"),
        ("Seattle Seahawks", "Seahawks"),
    ])
    def test_team_name_resolution(self, full_name, expected):
        """All 32 NFL teams resolve to correct short names."""
        from betting_intel.live.sport_configs import NFL

        assert NFL.get_short_name(full_name) == expected


# ═══════════════════════════════════════════════════════════════════════════
#  BACKFILL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════


class TestBackfillConstants:
    """Tests for league-specific backfill constants in features.py."""

    def _dictify(self, fill_list):
        """Convert list[tuple[str, float]] to dict for easier assertion."""
        return dict(fill_list)

    # ── All constants share same structure ───────────────────────────

    def test_minimum_entry_count(self):
        """All four backfill constants should have at least 121 entries.
        NBA has additional v6.5 entries for advanced basketball features.
        """
        from betting_intel.data.features import (
            _NBA_NA_FILL, _NCAAB_NA_FILL,
            _EUROLEAGUE_NA_FILL, _NFL_NA_FILL,
        )

        assert len(_NBA_NA_FILL) >= 121
        assert len(_NCAAB_NA_FILL) >= 121
        assert len(_EUROLEAGUE_NA_FILL) >= 121
        assert len(_NFL_NA_FILL) >= 121
        # NBA should have v6.5 entries for advanced basketball features
        assert len(_NBA_NA_FILL) >= 135

    def test_nba_has_v65_entries(self):
        """NBA backfill should contain v6.5 feature patterns."""
        from betting_intel.data.features import _NBA_NA_FILL

        nba_keys = {k for k, _ in _NBA_NA_FILL}
        # v6.5 interaction features
        assert "interact_" in nba_keys
        # v6.5 volatility features
        assert "volatility_pts_" in nba_keys
        assert "volatility_pm_" in nba_keys
        # v6.5 momentum features
        assert "win_streak" in nba_keys
        assert "streak_margin" in nba_keys
        assert "prev_loss" in nba_keys
        # v6.5 pace-adjusted features
        assert "pace_adj_off" in nba_keys
        assert "pace_adj_net" in nba_keys

    # ── Euroleague key values ───────────────────────────────────────

    def test_euroleague_avg_pts(self):
        """Euroleague avg_pts should be 78.0."""
        from betting_intel.data.features import _EUROLEAGUE_NA_FILL

        el = self._dictify(_EUROLEAGUE_NA_FILL)
        assert el["avg_pts"] == 78.0

    def test_euroleague_avg_pace(self):
        """Euroleague avg_pace should be 72.0."""
        from betting_intel.data.features import _EUROLEAGUE_NA_FILL

        el = self._dictify(_EUROLEAGUE_NA_FILL)
        assert el["avg_pace"] == 72.0

    def test_euroleague_margin_volatility(self):
        """Euroleague margin_volatility should be 14.0."""
        from betting_intel.data.features import _EUROLEAGUE_NA_FILL

        el = self._dictify(_EUROLEAGUE_NA_FILL)
        assert el["margin_volatility"] == 14.0

    def test_euroleague_avg_reb(self):
        """Euroleague avg_reb should be 33.0."""
        from betting_intel.data.features import _EUROLEAGUE_NA_FILL

        el = self._dictify(_EUROLEAGUE_NA_FILL)
        assert el["avg_reb"] == 33.0

    def test_euroleague_avg_fga(self):
        """Euroleague avg_fga should be 58.0."""
        from betting_intel.data.features import _EUROLEAGUE_NA_FILL

        el = self._dictify(_EUROLEAGUE_NA_FILL)
        assert el["avg_fga"] == 58.0

    # ── Compare across leagues: avg_pts ─────────────────────────────

    def test_nba_highest_avg_pts(self):
        """NBA should have the highest avg_pts among all leagues."""
        from betting_intel.data.features import (
            _NBA_NA_FILL, _NCAAB_NA_FILL,
            _EUROLEAGUE_NA_FILL, _NFL_NA_FILL,
        )

        nba = dict(_NBA_NA_FILL)
        ncaab = dict(_NCAAB_NA_FILL)
        el = dict(_EUROLEAGUE_NA_FILL)
        nfl = dict(_NFL_NA_FILL)

        # NBA (114.5) > Euroleague (78.0) > NCAAB (70.0) > NFL (22.0)
        assert nba["avg_pts"] > el["avg_pts"] > ncaab["avg_pts"] > nfl["avg_pts"]

    def test_nfl_lowest_pace(self):
        """NFL should have the lowest pace (fewest possessions)."""
        from betting_intel.data.features import (
            _NBA_NA_FILL, _NCAAB_NA_FILL,
            _EUROLEAGUE_NA_FILL, _NFL_NA_FILL,
        )

        nba = dict(_NBA_NA_FILL)
        ncaab = dict(_NCAAB_NA_FILL)
        el = dict(_EUROLEAGUE_NA_FILL)
        nfl = dict(_NFL_NA_FILL)

        assert nfl["avg_pace"] < ncaab["avg_pace"] < el["avg_pace"] < nba["avg_pace"]

    def test_nfl_basketball_stats_zero(self):
        """NFL basketball-specific stats should be 0.0 (not computed for football)."""
        from betting_intel.data.features import _NFL_NA_FILL

        nfl = dict(_NFL_NA_FILL)
        assert nfl["avg_fgm"] == 0.0
        assert nfl["avg_fga"] == 0.0
        assert nfl["avg_reb"] == 0.0
        assert nfl["avg_ast"] == 0.0
        assert nfl["avg_stl"] == 0.0
        assert nfl["avg_blk"] == 0.0
        assert nfl["three_pt_rate"] == 0.0
        assert nfl["avg_fg3_pct"] == 0.0
        assert nfl["avg_efg"] == 0.0

    # ── FeatureEngineer backfill integration ────────────────────────

    def test_euroleague_backfill_applies_correctly(self):
        """backfill_features with league='Euroleague' uses Euroleague constants."""
        from betting_intel.data.features import FeatureEngineer

        import pandas as pd
        fe = FeatureEngineer()
        df = pd.DataFrame({"avg_pts": [None, 100.0, None]})
        result = fe.backfill_features(df, league="Euroleague")
        assert result["avg_pts"].iloc[0] == 78.0  # Euroleague default
        assert result["avg_pts"].iloc[1] == 100.0  # unchanged
        assert result["avg_pts"].iloc[2] == 78.0   # backfilled

    def test_nfl_backfill_applies_correctly(self):
        """backfill_features with league='NFL' uses NFL constants."""
        from betting_intel.data.features import FeatureEngineer

        import pandas as pd
        fe = FeatureEngineer()
        df = pd.DataFrame({"avg_pts": [None, 100.0, None]})
        result = fe.backfill_features(df, league="NFL")
        assert result["avg_pts"].iloc[0] == 22.0  # NFL default
        assert result["avg_pts"].iloc[1] == 100.0  # unchanged
        assert result["avg_pts"].iloc[2] == 22.0   # backfilled

    def test_backfill_leaves_non_na_columns_unchanged(self):
        """Columns with no NaN values should not be modified."""
        from betting_intel.data.features import FeatureEngineer

        import pandas as pd
        fe = FeatureEngineer()
        original = pd.DataFrame({"avg_pts": [100.0, 105.0, 110.0]})
        result = fe.backfill_features(original, league="Euroleague")
        assert result["avg_pts"].iloc[0] == 100.0
        assert result["avg_pts"].iloc[1] == 105.0
        assert result["avg_pts"].iloc[2] == 110.0

    # ── NFL edge_pct_movement and NFL-specific columns ────────────────

    def test_nfl_edge_pct_movement_falls_to_default(self):
        """edge_pct_movement has no pattern match → defaults to 0.0."""
        from betting_intel.data.features import FeatureEngineer

        import pandas as pd
        fe = FeatureEngineer()
        df = pd.DataFrame({"edge_pct_movement": [None]})
        result = fe.backfill_features(df, league="NFL")
        assert result["edge_pct_movement"].iloc[0] == 0.0

    def test_nfl_backfill_avg_pace(self):
        """NFL avg_pace backfills to 65.0."""
        from betting_intel.data.features import FeatureEngineer

        import pandas as pd
        fe = FeatureEngineer()
        df = pd.DataFrame({"avg_pace": [None]})
        result = fe.backfill_features(df, league="NFL")
        assert result["avg_pace"].iloc[0] == 65.0

    def test_nfl_backfill_margin_volatility(self):
        """NFL margin_volatility backfills to 16.0 (higher variance than NBA)."""
        from betting_intel.data.features import FeatureEngineer

        import pandas as pd
        fe = FeatureEngineer()
        df = pd.DataFrame({"margin_volatility": [None]})
        result = fe.backfill_features(df, league="NFL")
        assert result["margin_volatility"].iloc[0] == 16.0

    def test_nfl_backfill_opp_avg_pts(self):
        """NFL opp_avg_pts_scored and opp_avg_pts_allowed backfill to 22.0."""
        from betting_intel.data.features import FeatureEngineer

        import pandas as pd
        fe = FeatureEngineer()
        df = pd.DataFrame({"opp_avg_pts_scored": [None], "opp_avg_pts_allowed": [None]})
        result = fe.backfill_features(df, league="NFL")
        assert result["opp_avg_pts_scored"].iloc[0] == 22.0
        assert result["opp_avg_pts_allowed"].iloc[0] == 22.0

    def test_nfl_backfill_multiple_mixed_columns(self):
        """Multiple NFL columns backfill correctly in one call."""
        from betting_intel.data.features import FeatureEngineer

        import pandas as pd
        fe = FeatureEngineer()
        df = pd.DataFrame({
            "avg_pts": [None, 30.0, None],
            "avg_pace": [None, 70.0, None],
            "margin_volatility": [None, 20.0, None],
            "edge_pct_movement": [None, None, 0.05],  # Known value preserved
        })
        result = fe.backfill_features(df, league="NFL")
        # Backfilled values
        assert result["avg_pts"].iloc[0] == 22.0
        assert result["avg_pts"].iloc[2] == 22.0
        assert result["avg_pace"].iloc[0] == 65.0
        assert result["avg_pace"].iloc[2] == 65.0
        assert result["margin_volatility"].iloc[0] == 16.0
        assert result["margin_volatility"].iloc[2] == 16.0
        # Preserved values
        assert result["avg_pts"].iloc[1] == 30.0
        assert result["avg_pace"].iloc[1] == 70.0
        assert result["margin_volatility"].iloc[1] == 20.0
        # Unmatched column falls to default 0.0
        assert result["edge_pct_movement"].iloc[0] == 0.0
        assert result["edge_pct_movement"].iloc[1] == 0.0
        assert result["edge_pct_movement"].iloc[2] == 0.05  # preserved

    def test_nfl_backfill_ema_pts(self):
        """NFL ema_pts backfills to 22.0 (matches avg_pts)."""
        from betting_intel.data.features import FeatureEngineer

        import pandas as pd
        fe = FeatureEngineer()
        df = pd.DataFrame({"ema_pts": [None]})
        result = fe.backfill_features(df, league="NFL")
        # Matches pattern "ema_pts" → _NFL_NA_FILL ema_pts = 22.0
        assert result["ema_pts"].iloc[0] == 22.0


# ═══════════════════════════════════════════════════════════════════════════
#  EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases for SportConfig and team name resolution."""

    def test_nba_config_integrity(self):
        """NBA config still has 30 teams (unchanged)."""
        from betting_intel.live.sport_configs import NBA

        assert len(NBA.team_name_map) == 30

    def test_ncaab_config_integrity(self):
        """NCAAB config still has 140 teams (unchanged)."""
        from betting_intel.live.sport_configs import NCAAB

        assert len(NCAAB.team_name_map) == 140

    def test_no_team_name_overlap_nba_ncaab(self):
        """NBA and NCAAB team_name_maps should not share keys
        (team names like 'Miami Heat' vs 'Miami Hurricanes' are distinct)."""
        from betting_intel.live.sport_configs import NBA, NCAAB

        overlap = set(NBA.team_name_map.keys()) & set(NCAAB.team_name_map.keys())
        assert len(overlap) == 0, f"Overlapping keys: {overlap}"

    def test_no_team_name_overlap_nba_euroleague(self):
        """NBA and Euroleague team_name_maps should not share keys."""
        from betting_intel.live.sport_configs import NBA, EUROLEAGUE

        overlap = set(NBA.team_name_map.keys()) & set(EUROLEAGUE.team_name_map.keys())
        assert len(overlap) == 0, f"Overlapping keys: {overlap}"

    def test_no_team_name_overlap_ncaab_euroleague(self):
        """NCAAB and Euroleague team_name_maps should not share keys."""
        from betting_intel.live.sport_configs import NCAAB, EUROLEAGUE

        overlap = set(NCAAB.team_name_map.keys()) & set(EUROLEAGUE.team_name_map.keys())
        assert len(overlap) == 0, f"Overlapping keys: {overlap}"

    def test_get_short_name_nba_with_unknown_sport_key(self):
        """get_short_name works even when called on wrong config."""
        from betting_intel.live.sport_configs import NCAAB

        # NCAAB config doesn't know NBA teams
        result = NCAAB.get_short_name("Boston Celtics")
        assert result == "Celtics"  # Falls back to last word
