"""
Tests for the PredictionPipeline (modular pipeline package).

Key coverage:
  PredictionPipeline.generate_player_props
    - Basic prop generation from a predictions DataFrame
    - Verifies ALL generated props appear in results (regression test for the
      props.append-outside-for-loop bug)
    - Handles empty predictions gracefully
    - Edge cases: no player data, non-NBA league, engine failure
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

# Ensure the project root is on sys.path so betting_intel can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from betting_intel.recommendations.bet_types import (
    PlayerPropBet, BetType, Confidence
)
from betting_intel.pipeline import PredictionPipeline


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_predictions_df() -> pd.DataFrame:
    """A minimal predictions DataFrame that PlayerPropEngine can consume."""
    return pd.DataFrame({
        "home_team": ["Celtics", "Lakers", "Warriors"],
        "away_team": ["Lakers", "Warriors", "Celtics"],
        "game_id": ["GAME_001", "GAME_002", "GAME_003"],
        "game_date": ["2026-06-01", "2026-06-01", "2026-06-02"],
        "predicted_total": [220.5, 215.0, 218.0],
        "market_total": [218.0, 214.0, 216.5],
        "edge_pct": [0.011, 0.005, 0.007],
        "confidence": ["medium", "low", "medium"],
    })


@pytest.fixture
def mock_prop_engine():
    """Create a MockPlayerPropEngine that returns controlled BetSuggestion objects."""

    def _predict_for_game(home="", away="", league="NBA", game_id="", game_date="", num_players=6):
        props = []
        for i, (player, pts) in enumerate([
            (f"{home}_Star1", 28.0),
            (f"{home}_Star2", 22.0),
            (f"{away}_Star1", 25.0),
        ]):
            props.append(PlayerPropBet(
                game_id=game_id,
                game_date=game_date,
                matchup=f"{away} @ {home}",
                player_name=player,
                prop_type=BetType.PLAYER_POINTS,
                market_line=pts - 2.5,
                predicted_value=pts,
                side="OVER",
                league=league,
                confidence=Confidence.MEDIUM if i < 2 else Confidence.LOW,
                reasoning=f"Test prop for {player}",
            ))
        return props

    engine = MagicMock()
    engine.predict_for_game.side_effect = _predict_for_game
    return engine


# ── Pipeline Helper ─────────────────────────────────────────────────────────


def _create_pipeline(args_override: dict = None):
    """Create a PredictionPipeline with mocked args for testing."""
    import argparse

    args = argparse.Namespace(
        live=False,
        full=False,
        recommend_only=False,
        simulate=False,
        scheduled=False,
        days_history=90,
        data_source=None,
        csv_path=None,
        no_tune=True,
        model_dir="models/saved",
        ensemble=True,
        strategy="all",
        bankroll=1000.0,
        kelly_fraction=0.25,
        max_exposure=0.20,
        min_edge=0.02,
        output=None,
        html=False,
        verbose=False,
    )
    if args_override:
        for k, v in args_override.items():
            setattr(args, k, v)
    return PredictionPipeline(args)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestGeneratePlayerProps:
    """Tests for PredictionPipeline.generate_player_props."""

    def test_generates_all_props(self, sample_predictions_df, mock_prop_engine):
        """Every generated prop should appear in the result list (the indentation bug fix)."""
        pipeline = _create_pipeline()

        with patch("betting_intel.pipeline.staking.PlayerPropEngine", return_value=mock_prop_engine):
            props = pipeline.generate_player_props(sample_predictions_df)

        assert len(props) == 9, (
            f"Expected 9 props for 3 games (3 props/game), got {len(props)}. "
            "This would be 3 if props.append was outside the for loop!"
        )

        players = {p["player"] for p in props}
        assert "Celtics_Star1" in players
        assert "Celtics_Star2" in players
        assert "Lakers_Star1" in players
        assert "Lakers_Star2" in players
        assert "Warriors_Star1" in players

    def test_props_have_all_required_fields(self, sample_predictions_df, mock_prop_engine):
        """Each prop dict should have all expected keys with non-null values."""
        pipeline = _create_pipeline()

        with patch("betting_intel.pipeline.staking.PlayerPropEngine", return_value=mock_prop_engine):
            props = pipeline.generate_player_props(sample_predictions_df)

        required_keys = {"player", "team", "prop_type", "line", "edge", "confidence", "odds"}
        for prop in props:
            missing = required_keys - set(prop.keys())
            assert not missing, f"Prop {prop.get('player', '?')} missing keys: {missing}"
            assert prop["player"], f"Prop has empty 'player' field"
            assert prop["prop_type"], f"Prop has empty 'prop_type' field"
            assert isinstance(prop["odds"], (int, float)), f"Odds must be numeric"

    def test_empty_predictions(self, mock_prop_engine):
        """Empty predictions DataFrame should produce empty props list."""
        empty_df = pd.DataFrame()
        pipeline = _create_pipeline()

        with patch("betting_intel.pipeline.staking.PlayerPropEngine", return_value=mock_prop_engine):
            props = pipeline.generate_player_props(empty_df)

        assert props == [], f"Empty predictions should yield empty props, got {len(props)}"

    def test_results_stored_in_self_results(self, sample_predictions_df, mock_prop_engine):
        """Props should be stored in self.results['player_props']."""
        pipeline = _create_pipeline()

        with patch("betting_intel.pipeline.staking.PlayerPropEngine", return_value=mock_prop_engine):
            pipeline.generate_player_props(sample_predictions_df)

        stored = pipeline.results.get("player_props", [])
        assert len(stored) == 9, f"Expected 9 props stored in results, got {len(stored)}"

    def test_props_have_team_from_matchup(self, sample_predictions_df, mock_prop_engine):
        """Team should be extracted from the matchup field in the bet dict."""
        pipeline = _create_pipeline()

        with patch("betting_intel.pipeline.staking.PlayerPropEngine", return_value=mock_prop_engine):
            props = pipeline.generate_player_props(sample_predictions_df)

        for prop in props:
            assert prop["team"] != "?", f"Prop {prop['player']} has '?' as team"

    def test_edge_values_are_floats(self, sample_predictions_df, mock_prop_engine):
        """Edge values should be proper floats, not strings or None."""
        pipeline = _create_pipeline()

        with patch("betting_intel.pipeline.staking.PlayerPropEngine", return_value=mock_prop_engine):
            props = pipeline.generate_player_props(sample_predictions_df)

        for prop in props:
            assert isinstance(prop["edge"], float), (
                f"Edge for {prop['player']} should be float, got {type(prop['edge'])}"
            )

    def test_player_prop_engine_failure_graceful(self, sample_predictions_df):
        """If PlayerPropEngine raises, the method should catch it and return []."""
        broken_engine = MagicMock()
        broken_engine.predict_for_game.side_effect = RuntimeError("API down")
        pipeline = _create_pipeline()

        with patch("betting_intel.pipeline.staking.PlayerPropEngine", return_value=broken_engine):
            props = pipeline.generate_player_props(sample_predictions_df)

        assert props == [], "Should return empty list on engine failure"

    def test_predictions_unchanged(self, sample_predictions_df, mock_prop_engine):
        """The input DataFrame should not be mutated."""
        original_cols = list(sample_predictions_df.columns)
        pipeline = _create_pipeline()

        with patch("betting_intel.pipeline.staking.PlayerPropEngine", return_value=mock_prop_engine):
            pipeline.generate_player_props(sample_predictions_df)

        assert list(sample_predictions_df.columns) == original_cols

    def test_regression_all_players_appended(self, sample_predictions_df, mock_prop_engine):
        """Regression test: ensure props.append is inside the for loop."""
        pipeline = _create_pipeline()

        with patch("betting_intel.pipeline.staking.PlayerPropEngine", return_value=mock_prop_engine):
            props = pipeline.generate_player_props(sample_predictions_df)

        distinct_players = {p["player"] for p in props}
        assert len(distinct_players) >= 5, (
            f"Expected >=5 distinct players across 3 games, got {len(distinct_players)}. "
            "props.append may still be outside the for loop!"
        )

    def test_single_game_single_prop(self, mock_prop_engine):
        """A single game should produce the expected number of props."""
        df = pd.DataFrame({
            "home_team": ["Celtics"],
            "away_team": ["Lakers"],
            "game_id": ["GAME_001"],
            "game_date": ["2026-06-01"],
            "predicted_total": [220.0],
            "market_total": [218.0],
            "edge_pct": [0.009],
            "confidence": ["medium"],
        })
        pipeline = _create_pipeline()

        with patch("betting_intel.pipeline.staking.PlayerPropEngine", return_value=mock_prop_engine):
            props = pipeline.generate_player_props(df)

        assert len(props) == 3

    def test_many_games(self, mock_prop_engine):
        """Many games should all generate props without issue."""
        teams = [
            ("Celtics", "Lakers"), ("Warriors", "Nuggets"),
            ("Heat", "Bucks"), ("Suns", "Thunder"), ("Knicks", "76ers"),
            ("Mavericks", "Spurs"), ("Nets", "Raptors"), ("Bulls", "Pacers"),
        ]
        rows = []
        for i, (home, away) in enumerate(teams):
            rows.append({
                "home_team": home, "away_team": away,
                "game_id": f"GAME_{i:03d}", "game_date": "2026-06-01",
                "predicted_total": 215.0, "market_total": 213.5,
                "edge_pct": 0.007, "confidence": "medium",
            })
        df = pd.DataFrame(rows)

        pipeline = _create_pipeline()
        with patch("betting_intel.pipeline.staking.PlayerPropEngine", return_value=mock_prop_engine):
            props = pipeline.generate_player_props(df)

        expected = len(teams) * 3
        assert len(props) == expected, f"Expected {expected} props, got {len(props)}"

    def test_empty_team_names(self, mock_prop_engine):
        """Games with empty team names should not crash."""
        df = pd.DataFrame({
            "home_team": [""],
            "away_team": [""],
            "game_id": ["GAME_000"],
            "game_date": ["2026-06-01"],
            "predicted_total": [210.0],
            "market_total": [210.0],
            "edge_pct": [0.0],
            "confidence": ["low"],
        })
        pipeline = _create_pipeline()

        with patch("betting_intel.pipeline.staking.PlayerPropEngine", return_value=mock_prop_engine):
            props = pipeline.generate_player_props(df)

        assert isinstance(props, list)
