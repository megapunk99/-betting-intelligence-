"""
Unit tests for web.app._livegame_to_bet() — bet dict conversion.

Tests cover:
  1. Stake propagation (LiveGame object → bet dict keeps the value)
  2. Clear pick detection (abs(edge_pct) > 0.03 → is_clear_pick=True)
  3. Both code paths: LiveGame objects and plain dicts
  4. Edge cases: None, zero, boundary values
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# Module path for patching the import inside web.app
# sport_key_to_group is imported at the top of web/app.py
_APP = "web.app"


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_sport_key_to_group():
    """Patch sport_key_to_group to return Basketball (default sport group)."""
    with patch(f"{_APP}.sport_key_to_group", return_value="Basketball") as m:
        yield m


@pytest.fixture
def live_game_class():
    """Import the real LiveGame dataclass."""
    from betting_intel.live.engine import LiveGame

    return LiveGame


@pytest.fixture
def base_game(live_game_class):
    """A minimal LiveGame with default fields."""
    return live_game_class(
        game_id="g1",
        sport_key="basketball_nba",
        home_team="Boston Celtics",
        away_team="Los Angeles Lakers",
        home_team_short="BOS",
        away_team_short="LAL",
        commence_time="2025-06-15T19:00:00Z",
        game_date="2025-06-15",
        home_ml=-150,
        away_ml=130,
        market_total=218.5,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  STAKE PROPAGATION
# ═══════════════════════════════════════════════════════════════════════════


class TestStakePropagation:
    """_livegame_to_bet must pass through the Kelly stake amount."""

    def test_stake_defaults_to_zero(self, mock_sport_key_to_group, base_game):
        """LiveGame without stake_dollars set → output has 0.0."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["stake_dollars"] == 0.0

    def test_stake_positive_value(self, mock_sport_key_to_group, base_game):
        """LiveGame with stake_dollars=250.0 → output has 250.0."""
        base_game.stake_dollars = 250.0
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["stake_dollars"] == 250.0

    def test_stake_large_value(self, mock_sport_key_to_group, base_game):
        """Large Kelly stake (e.g. $1,234.56) is preserved precisely."""
        base_game.stake_dollars = 1234.56
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["stake_dollars"] == 1234.56

    def test_stake_zero_explicit(self, mock_sport_key_to_group, base_game):
        """Explicit stake_dollars=0.0 → output 0.0 (no bet recommended)."""
        base_game.stake_dollars = 0.0
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["stake_dollars"] == 0.0

    def test_stake_none_defaults_to_zero(self, mock_sport_key_to_group, base_game):
        """None stake_dollars → output 0.0 (or 0 guard)."""
        base_game.stake_dollars = None
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["stake_dollars"] == 0.0

    def test_stake_from_dict(self, mock_sport_key_to_group):
        """Plain dict with stake_dollars → value is preserved."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(
            {
                "game_id": "g1",
                "sport_key": "basketball_nba",
                "home_team": "BOS",
                "away_team": "LAL",
                "home_team_short": "BOS",
                "away_team_short": "LAL",
                "game_date": "2025-06-15",
                "commence_time": "2025-06-15T19:00:00Z",
                "stake_dollars": 312.50,
                "league": "NBA",
            }
        )
        assert result["stake_dollars"] == 312.50

    def test_stake_zero_from_dict(self, mock_sport_key_to_group):
        """Plain dict with stake_dollars=0 → output 0."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(
            {
                "game_id": "g1",
                "sport_key": "basketball_nba",
                "home_team": "BOS",
                "away_team": "LAL",
                "home_team_short": "BOS",
                "away_team_short": "LAL",
                "game_date": "2025-06-15",
                "commence_time": "2025-06-15T19:00:00Z",
                "stake_dollars": 0.0,
                "league": "NBA",
            }
        )
        assert result["stake_dollars"] == 0.0

    def test_stake_dict_missing_key(self, mock_sport_key_to_group):
        """Plain dict without stake_dollars key → output 0.0."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(
            {
                "game_id": "g1",
                "sport_key": "basketball_nba",
                "home_team": "BOS",
                "away_team": "LAL",
                "home_team_short": "BOS",
                "away_team_short": "LAL",
                "game_date": "2025-06-15",
                "commence_time": "2025-06-15T19:00:00Z",
                "league": "NBA",
            }
        )
        assert result["stake_dollars"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  CLEAR PICK DETECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestClearPickDetection:
    """is_clear_pick must be True when abs(edge_pct) > 0.03."""

    def test_clear_pick_above_threshold(self, mock_sport_key_to_group, base_game):
        """edge_pct = 0.05 > 0.03 → is_clear_pick = True."""
        base_game.edge_pct = 0.05
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["is_clear_pick"] is True

    def test_not_clear_below_threshold(self, mock_sport_key_to_group, base_game):
        """edge_pct = 0.02 < 0.03 → is_clear_pick = False."""
        base_game.edge_pct = 0.02
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["is_clear_pick"] is False

    def test_clear_pick_negative_edge(self, mock_sport_key_to_group, base_game):
        """edge_pct = -0.05 → abs(-0.05) = 0.05 > 0.03 → is_clear_pick = True."""
        base_game.edge_pct = -0.05
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["is_clear_pick"] is True

    def test_not_clear_negative_small_edge(self, mock_sport_key_to_group, base_game):
        """edge_pct = -0.02 → abs(-0.02) = 0.02 < 0.03 → is_clear_pick = False."""
        base_game.edge_pct = -0.02
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["is_clear_pick"] is False

    def test_clear_pick_exactly_at_threshold(self, mock_sport_key_to_group, base_game):
        """edge_pct = 0.03 → abs(0.03) = 0.03, NOT > 0.03 → is_clear_pick = False."""
        base_game.edge_pct = 0.03
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["is_clear_pick"] is False

    def test_clear_pick_just_above_threshold(self, mock_sport_key_to_group, base_game):
        """edge_pct = 0.0301 → abs(0.0301) > 0.03 → is_clear_pick = True."""
        base_game.edge_pct = 0.0301
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["is_clear_pick"] is True

    def test_clear_pick_none_edge_defaults_false(
        self, mock_sport_key_to_group, base_game
    ):
        """edge_pct = None → edge_pct resolves to 0.0 → is_clear_pick = False."""
        base_game.edge_pct = None
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["is_clear_pick"] is False

    def test_clear_pick_zero_edge(self, mock_sport_key_to_group, base_game):
        """edge_pct = 0.0 → is_clear_pick = False."""
        base_game.edge_pct = 0.0
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["is_clear_pick"] is False

    def test_clear_pick_from_dict(self, mock_sport_key_to_group):
        """Plain dict with edge_pct=0.07 → is_clear_pick = True."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(
            {
                "game_id": "g1",
                "sport_key": "basketball_nba",
                "home_team": "BOS",
                "away_team": "LAL",
                "home_team_short": "BOS",
                "away_team_short": "LAL",
                "game_date": "2025-06-15",
                "commence_time": "2025-06-15T19:00:00Z",
                "edge_pct": 0.07,
                "league": "NBA",
            }
        )
        assert result["is_clear_pick"] is True

    def test_not_clear_pick_from_dict(self, mock_sport_key_to_group):
        """Plain dict with edge_pct=0.01 → is_clear_pick = False."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(
            {
                "game_id": "g1",
                "sport_key": "basketball_nba",
                "home_team": "BOS",
                "away_team": "LAL",
                "home_team_short": "BOS",
                "away_team_short": "LAL",
                "game_date": "2025-06-15",
                "commence_time": "2025-06-15T19:00:00Z",
                "edge_pct": 0.01,
                "league": "NBA",
            }
        )
        assert result["is_clear_pick"] is False


# ═══════════════════════════════════════════════════════════════════════════
#  COMBINED BEHAVIOR
# ═══════════════════════════════════════════════════════════════════════════


class TestCombinedBehavior:
    """Both stake and clear pick working together."""

    def test_clear_pick_with_positive_stake(self, mock_sport_key_to_group, base_game):
        """A high-edge pick with a real stake should have both fields set."""
        base_game.edge_pct = 0.06
        base_game.stake_dollars = 435.0
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["is_clear_pick"] is True
        assert result["stake_dollars"] == 435.0

    def test_no_stake_low_edge(self, mock_sport_key_to_group, base_game):
        """Low edge and no stake → both zero/false."""
        base_game.edge_pct = 0.01
        base_game.stake_dollars = 0.0
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["is_clear_pick"] is False
        assert result["stake_dollars"] == 0.0

    def test_high_edge_zero_stake(self, mock_sport_key_to_group, base_game):
        """High edge but zero stake (model edge vs no Kelly recommendation)."""
        base_game.edge_pct = 0.08
        base_game.stake_dollars = 0.0
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["is_clear_pick"] is True  # edge says clear pick
        assert result["stake_dollars"] == 0.0  # Kelly says no stake


# ═══════════════════════════════════════════════════════════════════════════
#  FIELD PRESERVATION & SPORT GROUP
# ═══════════════════════════════════════════════════════════════════════════


class TestFieldPreservation:
    """Other fields in the bet dict should be preserved correctly."""

    def test_edge_pct_preserved_positive(self, mock_sport_key_to_group, base_game):
        """edge_pct=0.04 → output edge_pct=0.04."""
        base_game.edge_pct = 0.04
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["edge_pct"] == 0.04

    def test_edge_pct_preserved_negative(self, mock_sport_key_to_group, base_game):
        """edge_pct=-0.03 → output edge_pct=-0.03."""
        base_game.edge_pct = -0.03
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["edge_pct"] == -0.03

    def test_confidence_preserved(self, mock_sport_key_to_group, base_game):
        """confidence='high' → output confidence='high' and edge_confidence='high'."""
        base_game.confidence = "high"
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["confidence"] == "high"
        assert result["edge_confidence"] == "high"

    def test_confidence_defaults_low(self, mock_sport_key_to_group, base_game):
        """No confidence set → output confidence='low'."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["confidence"] == "low"

    def test_confidence_none_defaults_low(self, mock_sport_key_to_group, base_game):
        """confidence=None → output confidence='low'."""
        base_game.confidence = None
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["confidence"] == "low"

    def test_league_defaults_to_nba(self, mock_sport_key_to_group, base_game):
        """league default is NBA."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["league"] == "NBA"

    def test_sport_group_basketball(self, mock_sport_key_to_group, base_game):
        """Basketball sport_group → bet_type='moneyline', bet_type_display='ML Edge'."""
        mock_sport_key_to_group.return_value = "Basketball"
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        # bet_type is always "moneyline" regardless of sport group
        assert result["bet_type"] == "moneyline"
        # bet_type_display differs by sport group
        assert result["bet_type_display"] == "ML Edge"
        assert result["sport_group"] == "Basketball"

    def test_sport_group_non_basketball(self, mock_sport_key_to_group, base_game):
        """Non-basketball sport_group → bet_type='moneyline', bet_type_display='Moneyline'."""
        mock_sport_key_to_group.return_value = "Other"
        # LiveGame has sport_group="Basketball" as default field — override it
        base_game.sport_group = "Other"
        base_game.sport_key = "tennis_atp"
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(base_game)
        assert result["bet_type"] == "moneyline"
        assert result["bet_type_display"] == "Moneyline"
        assert result["sport_group"] == "Other"


# ═══════════════════════════════════════════════════════════════════════════
#  DICT CODE PATH
# ═══════════════════════════════════════════════════════════════════════════


class TestDictPath:
    """_livegame_to_bet also handles plain dicts (no to_dict method)."""

    def test_basic_dict(self, mock_sport_key_to_group):
        """Minimal dict produces valid output."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(
            {
                "game_id": "d1",
                "sport_key": "basketball_nba",
                "home_team": "Home",
                "away_team": "Away",
                "home_team_short": "HOM",
                "away_team_short": "AWY",
                "game_date": "2025-06-15",
                "commence_time": "2025-06-15T19:00:00Z",
                "league": "NBA",
            }
        )
        assert result["game_id"] == "d1"
        assert result["matchup"] == "AWY @ HOM"
        assert result["is_clear_pick"] is False
        assert result["stake_dollars"] == 0.0

    def test_dict_team_shorts_via_matchup(self, mock_sport_key_to_group):
        """Dict without team shorts falls back to matchup parsing."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(
            {
                "game_id": "d2",
                "sport_key": "basketball_nba",
                "home_team": "Celtics",
                "away_team": "Lakers",
                "game_date": "2025-06-15",
                "commence_time": "2025-06-15T19:00:00Z",
                "matchup": "Lakers @ Celtics",
                "league": "NBA",
            }
        )
        assert result["matchup"] == "Lakers @ Celtics"

    def test_dict_missing_optional_fields(self, mock_sport_key_to_group):
        """Dict missing optional fields should not crash."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(
            {
                "game_id": "d3",
                "sport_key": "basketball_nba",
                "home_team": "X",
                "away_team": "Y",
                "home_team_short": "X",
                "away_team_short": "Y",
                "game_date": "2025-06-15",
                "commence_time": "2025-06-15T19:00:00Z",
                "league": "NBA",
            }
        )
        assert result["game_id"] == "d3"
        # These should all be None or defaults without crashing
        assert result["market_total"] is None
        assert result["predicted_total"] is None
        assert result["home_ml"] is None
        assert result["away_ml"] is None

    def test_dict_clear_pick(self, mock_sport_key_to_group):
        """Dict with edge_pct > 0.03 → is_clear_pick=True."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(
            {
                "game_id": "d4",
                "sport_key": "basketball_nba",
                "home_team": "X",
                "away_team": "Y",
                "home_team_short": "X",
                "away_team_short": "Y",
                "game_date": "2025-06-15",
                "commence_time": "2025-06-15T19:00:00Z",
                "edge_pct": 0.04,
                "league": "NBA",
            }
        )
        assert result["is_clear_pick"] is True

    def test_dict_with_stake(self, mock_sport_key_to_group):
        """Dict with stake_dollars → value is preserved."""
        from web.app import _livegame_to_bet

        result = _livegame_to_bet(
            {
                "game_id": "d5",
                "sport_key": "basketball_nba",
                "home_team": "X",
                "away_team": "Y",
                "home_team_short": "X",
                "away_team_short": "Y",
                "game_date": "2025-06-15",
                "commence_time": "2025-06-15T19:00:00Z",
                "stake_dollars": 100.0,
                "league": "NBA",
            }
        )
        assert result["stake_dollars"] == 100.0
