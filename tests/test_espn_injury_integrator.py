"""Unit tests for ESPNInjuryIntegrator and associated dataclasses."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from betting_intel.data.espn_injury_integrator import (
    ESPNInjuryIntegrator,
    MergedGameInjuryData,
    PlayerInjuryStatus,
    TeamInjurySummary,
)
from betting_intel.data.player_injury import (
    GameInjuryData,
    InjuryImpact,
    PLAYER_DATABASE,
    TEAM_ABBR_TO_SHORT,
)
from betting_intel.data.injury_scraper import InjuryRecord


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_espn_scraper(monkeypatch):
    """Replace ESPNInjuryScraper.fetch_all() with known records."""

    def fake_fetch_all(force_refresh=False):
        return [
            InjuryRecord(
                player_name="Mitchell Robinson",
                team="New York Knicks",
                team_abbr="NYK",
                position="C",
                injury_status="Day-To-Day",
                injury_description="Day-To-Day",
                date_updated="2026-05-30T14:08Z",
            ),
            InjuryRecord(
                player_name="Julius Randle",
                team="New York Knicks",
                team_abbr="NYK",
                position="F",
                injury_status="OUT",
                injury_description="Out",
                date_updated="2026-05-29T14:08Z",
            ),
            InjuryRecord(
                player_name="Jeremy Sochan",
                team="San Antonio Spurs",
                team_abbr="SAS",
                position="F",
                injury_status="OUT",
                injury_description="Out",
                date_updated="2026-05-30T14:08Z",
            ),
        ]

    monkeypatch.setattr(
        "betting_intel.data.injury_scraper.ESPNInjuryScraper.fetch_all",
        fake_fetch_all,
    )


@pytest.fixture
def sample_home_impact() -> InjuryImpact:
    """InjuryImpact for the home team (Spurs) with missing players."""
    return InjuryImpact(
        team_abbr="SAS",
        team_short="Spurs",
        total_player_props_pts=180.5,
        num_players_with_props=8,
        missing_stars=[
            "Jeremy Sochan (11 PPG, STARTER)",
            "Tre Jones (9 PPG, STARTER)",
            "Harrison Barnes (11 PPG, STARTER)",
        ],
        missing_ppg_weighted=14.7,
        prop_players=[
            {"name": "Victor Wembanyama", "point": 24.5, "market": "player_points", "ppg": 23.0, "role": "STAR"},
            {"name": "Devin Vassell", "point": 16.5, "market": "player_points", "ppg": 18.0, "role": "STARTER"},
        ],
    )


@pytest.fixture
def sample_away_impact() -> InjuryImpact:
    """InjuryImpact for the away team (Knicks) with missing players."""
    return InjuryImpact(
        team_abbr="NYK",
        team_short="Knicks",
        total_player_props_pts=210.0,
        num_players_with_props=7,
        missing_stars=[
            "Julius Randle (22 PPG, STAR)",
            "Isaiah Hartenstein (8 PPG, STARTER)",
        ],
        missing_ppg_weighted=23.3,
        prop_players=[
            {"name": "Jalen Brunson", "point": 27.5, "market": "player_points", "ppg": 27.0, "role": "STAR"},
            {"name": "Mikal Bridges", "point": 12.5, "market": "player_points", "ppg": 19.0, "role": "STARTER"},
        ],
    )


@pytest.fixture
def sample_prop_data(sample_home_impact, sample_away_impact) -> dict[str, GameInjuryData]:
    """Full prop-based injury data for one game."""
    return {
        "game_123": GameInjuryData(
            game_id="game_123",
            home_team="San Antonio Spurs",
            away_team="New York Knicks",
            home_impact=sample_home_impact,
            away_impact=sample_away_impact,
            total_prop_pts=390.5,
        )
    }


@pytest.fixture
def empty_prop_data() -> dict[str, GameInjuryData]:
    """No prop injury data at all."""
    return {}


@pytest.fixture
def clean_game_prop_data() -> dict[str, GameInjuryData]:
    """Prop data for a game with NO injuries (all players have props)."""
    return {
        "game_456": GameInjuryData(
            game_id="game_456",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_impact=InjuryImpact(
                team_abbr="BOS", team_short="Celtics",
                total_player_props_pts=200.0, num_players_with_props=10,
                missing_stars=[], missing_ppg_weighted=0.0,
            ),
            away_impact=InjuryImpact(
                team_abbr="LAL", team_short="Lakers",
                total_player_props_pts=195.0, num_players_with_props=9,
                missing_stars=[], missing_ppg_weighted=0.0,
            ),
        )
    }


# ── Dataclass Tests ────────────────────────────────────────────────────────


class TestPlayerInjuryStatus:
    """Tests for the PlayerInjuryStatus dataclass."""

    def test_prop_only_missing(self):
        status = PlayerInjuryStatus(
            player_name="Julius Randle",
            team_abbr="NYK",
            prop_detected_missing=True,
            prop_ppg=22.0,
            prop_role="STAR",
        )
        assert status.player_name == "Julius Randle"
        assert status.team_abbr == "NYK"
        assert status.prop_detected_missing is True
        assert status.prop_ppg == 22.0
        assert status.prop_role == "STAR"
        assert status.espn_status is None  # No ESPN data
        assert status.display_status == "[Props: no line \u2014 likely injured]"

    def test_espn_only_injured(self):
        status = PlayerInjuryStatus(
            player_name="Mitchell Robinson",
            team_abbr="NYK",
            prop_detected_missing=False,
            prop_ppg=6.0,
            prop_role="ROTATION",
            espn_status="Day-To-Day",
            espn_description="Day-To-Day",
        )
        assert status.espn_status == "Day-To-Day"
        assert status.prop_detected_missing is False
        assert status.display_status == "[ESPN: Day-To-Day] [Props: has line]"

    def test_both_sources(self):
        status = PlayerInjuryStatus(
            player_name="Jeremy Sochan",
            team_abbr="SAS",
            prop_detected_missing=True,
            prop_ppg=11.0,
            prop_role="STARTER",
            espn_status="OUT",
            espn_description="Out",
        )
        assert status.prop_detected_missing is True
        assert status.espn_status == "OUT"
        assert status.display_status == "[ESPN: OUT] [Props: confirmed missing]"

    def test_active_player(self):
        status = PlayerInjuryStatus(
            player_name="Jalen Brunson",
            team_abbr="NYK",
        )
        assert status.prop_detected_missing is False
        assert status.espn_status is None
        assert status.display_status == "Active"


class TestTeamInjurySummary:
    """Tests for the TeamInjurySummary dataclass."""

    def test_create_summary(self):
        players = [
            PlayerInjuryStatus(player_name="Player A", team_abbr="NYK", prop_detected_missing=True),
            PlayerInjuryStatus(player_name="Player B", team_abbr="NYK", espn_status="OUT"),
        ]
        summary = TeamInjurySummary(
            team_abbr="NYK",
            team_short="Knicks",
            players=players,
            total_prop_missing=1,
            total_espn_injured=1,
            weighted_ppg_loss=22.0,
        )
        assert summary.team_abbr == "NYK"
        assert len(summary.players) == 2
        assert summary.total_prop_missing == 1
        assert summary.total_espn_injured == 1
        assert summary.weighted_ppg_loss == 22.0


class TestMergedGameInjuryData:
    """Tests for the MergedGameInjuryData dataclass."""

    def test_has_injuries_true(self):
        home = TeamInjurySummary(
            team_abbr="SAS", team_short="Spurs",
            total_prop_missing=3, total_espn_injured=1, weighted_ppg_loss=14.7,
        )
        merged = MergedGameInjuryData(
            game_id="game_123",
            home_team="San Antonio Spurs",
            away_team="New York Knicks",
            home_summary=home,
        )
        assert merged.has_any_injuries is True

    def test_has_injuries_false(self):
        home = TeamInjurySummary(
            team_abbr="BOS", team_short="Celtics",
            total_prop_missing=0, total_espn_injured=0, weighted_ppg_loss=0.0,
        )
        away = TeamInjurySummary(
            team_abbr="LAL", team_short="Lakers",
            total_prop_missing=0, total_espn_injured=0, weighted_ppg_loss=0.0,
        )
        merged = MergedGameInjuryData(
            game_id="game_456",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            home_summary=home,
            away_summary=away,
        )
        assert merged.has_any_injuries is False

    def test_no_summaries(self):
        merged = MergedGameInjuryData(
            game_id="game_789",
            home_team="Team A",
            away_team="Team B",
        )
        assert merged.has_any_injuries is False


# ── ESPNInjuryIntegrator Tests ─────────────────────────────────────────────


class TestESPNInjuryIntegrator:
    """Tests for the ESPNInjuryIntegrator class."""

    def test_merge_empty_data(self, mock_espn_scraper):
        """Merging with empty prop data returns empty dict."""
        integrator = ESPNInjuryIntegrator()
        result = integrator.merge({})
        assert result == {}

    def test_merge_no_injuries(self, mock_espn_scraper, clean_game_prop_data):
        """A game with zero injuries should produce merged data with has_any_injuries=False."""
        integrator = ESPNInjuryIntegrator()
        result = integrator.merge(clean_game_prop_data)
        assert "game_456" in result
        merged = result["game_456"]
        assert merged.game_id == "game_456"
        assert merged.has_any_injuries is False

    def test_merge_full_pipeline(self, mock_espn_scraper, sample_prop_data):
        """Full merge: prop-detected + ESPN cross-reference."""
        integrator = ESPNInjuryIntegrator()
        result = integrator.merge(sample_prop_data)

        assert "game_123" in result
        merged = result["game_123"]

        # Game metadata preserved
        assert merged.game_id == "game_123"
        assert merged.home_team == "San Antonio Spurs"
        assert merged.away_team == "New York Knicks"

        # Should detect injuries
        assert merged.has_any_injuries is True

        # Home (Spurs) summary
        home = merged.home_summary
        assert home is not None
        assert home.team_abbr == "SAS"
        assert home.team_short == "Spurs"
        assert home.total_prop_missing == 3
        assert home.weighted_ppg_loss == 14.7

        # Away (Knicks) summary
        away = merged.away_summary
        assert away is not None
        assert away.team_abbr == "NYK"
        assert away.total_prop_missing == 2

        # Check for prop-detected players
        home_player_names = {p.player_name for p in home.players}
        assert "Jeremy Sochan" in home_player_names
        assert "Tre Jones" in home_player_names
        assert "Harrison Barnes" in home_player_names

    def test_merge_xref_espn_status(self, mock_espn_scraper, sample_prop_data):
        """ESPN status should be attached to matching prop-detected players."""
        integrator = ESPNInjuryIntegrator()
        result = integrator.merge(sample_prop_data)
        merged = result["game_123"]

        # Jeremy Sochan is prop-detected AND in ESPN records -> should have ESPN status
        home = merged.home_summary
        sochan = next((p for p in home.players if p.player_name == "Jeremy Sochan"), None)
        assert sochan is not None
        assert sochan.prop_detected_missing is True
        assert sochan.espn_status == "OUT"

        # Julius Randle is also prop-detected and in ESPN records
        away = merged.away_summary
        randle = next((p for p in away.players if p.player_name == "Julius Randle"), None)
        assert randle is not None
        assert randle.prop_detected_missing is True
        assert randle.espn_status == "OUT"

    def test_merge_espn_only_players(self, mock_espn_scraper, sample_prop_data):
        """ESPN-only injured players (not in prop data) should be included."""
        integrator = ESPNInjuryIntegrator()
        result = integrator.merge(sample_prop_data)
        merged = result["game_123"]

        # Mitchell Robinson is ESPN-only (not in prop missing list)
        away = merged.away_summary
        robinson = next((p for p in away.players if p.player_name == "Mitchell Robinson"), None)
        assert robinson is not None
        assert robinson.prop_detected_missing is False  # Not in props
        assert robinson.espn_status == "Day-To-Day"
        assert robinson.prop_ppg == 6.0  # From PLAYER_DATABASE

    def test_init_no_api_key(self):
        """Integrator should not require an API key (uses public ESPN API)."""
        integrator = ESPNInjuryIntegrator()
        assert integrator.api_key == ""


# ── Display Lines Tests ────────────────────────────────────────────────────


class TestGetDisplayLines:
    """Tests for the ESPNInjuryIntegrator.get_display_lines method."""

    def test_no_injuries_returns_empty(self, mock_espn_scraper, clean_game_prop_data):
        integrator = ESPNInjuryIntegrator()
        merged = integrator.merge(clean_game_prop_data)["game_456"]
        lines = integrator.get_display_lines(merged)
        assert lines == []

    def test_displays_prop_missing_players(self, mock_espn_scraper, sample_prop_data):
        integrator = ESPNInjuryIntegrator()
        result = integrator.merge(sample_prop_data)

        for merged in result.values():
            lines = integrator.get_display_lines(merged)
            if not merged.has_any_injuries:
                continue

            # Should contain team names and player names
            all_text = "\n".join(lines)
            assert "Sochan" in all_text or "Tre Jones" in all_text
            assert "Randle" in all_text or "Hartenstein" in all_text

    def test_espn_status_shown_in_output(self, mock_espn_scraper, sample_prop_data):
        """ESPN status tags like [OUT] should appear in display lines."""
        integrator = ESPNInjuryIntegrator()
        result = integrator.merge(sample_prop_data)

        merged = result["game_123"]
        lines = integrator.get_display_lines(merged)
        all_text = "\n".join(lines)

        # ESPN status should be visible
        assert "[OUT]" in all_text

    def test_display_format_structure(self, mock_espn_scraper, sample_prop_data):
        """Display lines should have proper section headers."""
        integrator = ESPNInjuryIntegrator()
        result = integrator.merge(sample_prop_data)

        merged = result["game_123"]
        lines = integrator.get_display_lines(merged)

        assert len(lines) > 0
        # First line should be the game matchup
        assert "@" in lines[0] or "─" in lines[0]

        # Should have section headers
        all_text = "\n".join(lines)
        assert "Props:" in all_text or "ESPN official status:" in all_text

    def test_weighted_ppg_displayed(self, mock_espn_scraper, sample_prop_data):
        """Weighted PPG loss should be shown in display."""
        integrator = ESPNInjuryIntegrator()
        result = integrator.merge(sample_prop_data)

        merged = result["game_123"]
        lines = integrator.get_display_lines(merged)
        all_text = "\n".join(lines)

        # PPG values should appear
        assert "PPG" in all_text

    def test_display_with_no_espn_data(self, clean_game_prop_data, monkeypatch):
        """When ESPN has no injury records, only prop data should be shown."""
        def fake_empty(force_refresh=False):
            return []

        monkeypatch.setattr(
            "betting_intel.data.injury_scraper.ESPNInjuryScraper.fetch_all",
            fake_empty,
        )

        integrator = ESPNInjuryIntegrator()
        merged = integrator.merge(clean_game_prop_data)["game_456"]
        lines = integrator.get_display_lines(merged)
        assert lines == []


# ── Edge Cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for the integrator."""

    def test_partial_data_one_team_only(self, mock_espn_scraper):
        """Game data with only one team's impact should still work."""
        prop_data = {
            "game_789": GameInjuryData(
                game_id="game_789",
                home_team="Team A",
                away_team="Team B",
                home_impact=InjuryImpact(
                    team_abbr="SAS", team_short="Spurs",
                    total_player_props_pts=100.0, num_players_with_props=5,
                    missing_stars=["Player A (10 PPG, STARTER)"],
                    missing_ppg_weighted=7.0,
                ),
                away_impact=None,
            )
        }

        integrator = ESPNInjuryIntegrator()
        result = integrator.merge(prop_data)
        assert "game_789" in result
        merged = result["game_789"]
        assert merged.home_summary is not None
        assert merged.away_summary is None

    def test_prop_data_with_no_missing_stars(self, mock_espn_scraper):
        """Prop data with 0 missing stars should still produce valid output."""
        prop_data = {
            "game_000": GameInjuryData(
                game_id="game_000",
                home_team="Team A",
                away_team="Team B",
                home_impact=InjuryImpact(
                    team_abbr="ATL", team_short="Hawks",
                    total_player_props_pts=150.0, num_players_with_props=10,
                    missing_stars=[], missing_ppg_weighted=0.0,
                ),
                away_impact=InjuryImpact(
                    team_abbr="BOS", team_short="Celtics",
                    total_player_props_pts=160.0, num_players_with_props=10,
                    missing_stars=[], missing_ppg_weighted=0.0,
                ),
            )
        }

        integrator = ESPNInjuryIntegrator()
        result = integrator.merge(prop_data)
        merged = result["game_000"]
        # No prop-detected missing, but ESPN may still have injured players
        assert merged.home_summary is not None
        assert merged.away_summary is not None
        # has_any_injuries depends on whether ESPN found injuries
        # This is fine — it won't crash

    def test_multiple_games(self, mock_espn_scraper, sample_prop_data, clean_game_prop_data):
        """Merging multiple games should work."""
        combined = {**sample_prop_data, **clean_game_prop_data}
        integrator = ESPNInjuryIntegrator()
        result = integrator.merge(combined)
        assert len(result) == 2
        assert "game_123" in result
        assert "game_456" in result
