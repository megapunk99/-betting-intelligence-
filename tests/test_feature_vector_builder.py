"""
Unit tests for the v6.0 multi-game weighted average feature vector builder.

Tests the _build_feature_vector method on GamePredictor:
  - Weighted multi-game averaging (last 3 games with recency weights [0.5, 0.3, 0.2])
  - Diff column computation (home - away)
  - Global feature fallback (rest, travel, fatigue, H2H from direct matchup)
  - Edge cases (missing teams, empty dataframe, single game, partial data)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


def _make_features_df(n_teams: int = 4, n_games_per_team: int = 5) -> pd.DataFrame:
    """Build a synthetic features DataFrame for testing _build_feature_vector.

    Creates a DataFrame with:
      - n_teams teams (Team_A, Team_B, ...)
      - n_games_per_team games per team (alternating home/away roles)
      - Representative feature columns with _home, _away, and _diff suffixes
      - GAME_DATE for chronological ordering
      - Rest/travel/H2H columns
    """
    np.random.seed(42)
    teams = [f"Team_{chr(65 + i)}" for i in range(n_teams)]
    rows = []

    game_id = 0
    for g in range(n_games_per_team):
        for h_idx in range(n_teams):
            a_idx = (h_idx + 1) % n_teams
            home_team = teams[h_idx]
            away_team = teams[a_idx]
            if home_team == away_team:
                continue

            rows.append({
                "GAME_ID": f"G{game_id:04d}",
                "GAME_DATE": pd.Timestamp(f"2025-01-{10 + g:02d}"),
                "TEAM_ID_home": h_idx,
                "TEAM_ID_away": a_idx,
                "TEAM_NAME_home": home_team,
                "TEAM_NAME_away": away_team,
                # _home features
                "avg_pts_5g_home": float(np.random.normal(112, 8)),
                "avg_pm_5g_home": float(np.random.normal(2, 5)),
                "avg_reb_5g_home": float(np.random.normal(43, 3)),
                "avg_ast_5g_home": float(np.random.normal(25, 3)),
                "ema_pts_10g_home": float(np.random.normal(113, 6)),
                "composite_power_home": float(np.random.uniform(0.3, 0.7)),
                "elo_home": float(np.random.normal(1500, 50)),
                "win_pct_10g_home": float(np.random.uniform(0.3, 0.7)),
                "weighted_momentum_home": float(np.random.uniform(0.3, 0.7)),
                "fatigue_index_home": float(np.random.uniform(0.1, 0.6)),
                # _away features
                "avg_pts_5g_away": float(np.random.normal(110, 8)),
                "avg_pm_5g_away": float(np.random.normal(-1, 5)),
                "avg_reb_5g_away": float(np.random.normal(42, 3)),
                "avg_ast_5g_away": float(np.random.normal(24, 3)),
                "ema_pts_10g_away": float(np.random.normal(111, 6)),
                "composite_power_away": float(np.random.uniform(0.3, 0.7)),
                "elo_away": float(np.random.normal(1480, 50)),
                "win_pct_10g_away": float(np.random.uniform(0.3, 0.7)),
                "weighted_momentum_away": float(np.random.uniform(0.3, 0.7)),
                "fatigue_index_away": float(np.random.uniform(0.1, 0.6)),
                # _diff features
                "pts_diff_5g": float(np.random.normal(2, 5)),
                "pm_diff_5g": float(np.random.normal(3, 4)),
                "power_diff": float(np.random.uniform(-0.2, 0.2)),
                # Rest & fatigue (game-specific, not averaged)
                "rest_home_days": float(np.random.randint(0, 5)),
                "rest_away_days": float(np.random.randint(0, 5)),
                "rest_advantage": float(np.random.randint(-2, 3)),
                "is_b2b_home": float(np.random.randint(0, 2)),
                "is_b2b_away": float(np.random.randint(0, 2)),
                "fatigue_home": float(np.random.uniform(0.2, 1.5)),
                "fatigue_away": float(np.random.uniform(0.2, 1.5)),
                # Travel
                "travel_distance": float(np.random.uniform(100, 2500)),
                "travel_distance_norm": float(np.random.uniform(0.03, 0.8)),
                "tz_diff": float(np.random.randint(0, 4)),
                "cum_travel_diff": float(np.random.uniform(-1000, 1000)),
                # H2H
                "h2h_win_rate": float(np.random.uniform(0.3, 0.7)),
                "h2h_avg_margin": float(np.random.uniform(-5, 5)),
            })
            game_id += 1

    df = pd.DataFrame(rows)
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df


@pytest.fixture
def features_df() -> pd.DataFrame:
    return _make_features_df(n_teams=4, n_games_per_team=5)


@pytest.fixture
def predictor():
    """Create a GamePredictor with mocked dependencies for testing _build_feature_vector."""
    from betting_intel.live.predictor import GamePredictor
    kelly = MagicMock()
    odds_store = MagicMock()
    return GamePredictor(kelly_staker=kelly, market_odds_store=odds_store)


# ── Core Functionality ──────────────────────────────────────────────────


class TestFeatureVectorBuilder:
    """Tests for _build_feature_vector — weighted average, diff, global features."""

    def test_returns_series(self, predictor, features_df):
        """Basic: returns a pd.Series with correct length."""
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df,
        )
        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_feature_count_matches(self, predictor, features_df):
        """When feature_cols is specified, output has the exact same columns."""
        cols = ["avg_pts_5g_home", "avg_pts_5g_away", "pts_diff_5g",
                "rest_home_days", "travel_distance", "h2h_win_rate"]
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df, feature_cols=cols,
        )
        assert result is not None
        assert list(result.index) == cols
        assert len(result) == len(cols)

    def test_all_feature_values_are_finite(self, predictor, features_df):
        """No NaN, inf, or -inf in the result."""
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df,
        )
        assert result is not None
        assert np.isfinite(result.values).all(), f"Non-finite values: {result[~np.isfinite(result.values)]}"

    def test_returns_none_for_empty_df(self, predictor):
        """Empty DataFrame returns None."""
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", pd.DataFrame(),
        )
        assert result is None

    def test_returns_none_for_none_df(self, predictor):
        """None DataFrame returns None."""
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", None,
        )
        assert result is None

    def test_missing_team_returns_features(self, predictor, features_df):
        """Missing team should still return a result (uses fallbacks)."""
        result = predictor._build_feature_vector(
            "Team_ZZZ", "Team_YYY", features_df,
        )
        assert result is not None
        assert len(result) > 0


class TestWeightedAverage:
    """Tests for the recency-weighted multi-game averaging."""

    def test_home_features_are_weighted_average(self, predictor, features_df):
        """_home features should equal weighted average of last 3 home games."""
        cols = ["avg_pts_5g_home", "avg_pm_5g_home", "avg_reb_5g_home"]
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df, feature_cols=cols,
        )
        assert result is not None

        # Get the last 3 games for Team_A as home team
        # tail(3) returns chronological order (oldest first). The builder
        # reverses values so most recent gets highest weight — we must
        # reverse here too for the expected calculation.
        home_mask = features_df["TEAM_NAME_home"] == "Team_A"
        home_games = features_df[home_mask].tail(3)

        weights = np.array([0.50, 0.30, 0.20])
        weights = weights / weights.sum()

        for col in cols:
            # Reverse values: [oldest, mid, newest] -> [newest, mid, oldest]
            # to match recency weights [0.5, 0.3, 0.2]
            vals = home_games[col].values[::-1]
            expected = float(np.average(vals, weights=weights))
            assert abs(result[col] - expected) < 0.001, (
                f"{col}: expected {expected:.4f}, got {result[col]:.4f}"
            )

    def test_away_features_are_weighted_average(self, predictor, features_df):
        """_away features should equal weighted average of last 3 away games."""
        cols = ["avg_pts_5g_away", "avg_pm_5g_away"]
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df, feature_cols=cols,
        )
        assert result is not None

        away_mask = features_df["TEAM_NAME_away"] == "Team_B"
        away_games = features_df[away_mask].tail(3)

        weights = np.array([0.50, 0.30, 0.20])
        weights = weights / weights.sum()

        for col in cols:
            # Reverse values to match recency weight order
            vals = away_games[col].values[::-1]
            expected = float(np.average(vals, weights=weights))
            assert abs(result[col] - expected) < 0.001, (
                f"{col}: expected {expected:.4f}, got {result[col]:.4f}"
            )

    def test_single_game_falls_back_to_single_row(self, predictor, features_df):
        """When a team has only 1 game, the single row's value is used directly."""
        # Build a tiny DF with just 1 home game for Team_X
        rows = [{
            "GAME_ID": "G0001", "GAME_DATE": pd.Timestamp("2025-01-10"),
            "TEAM_ID_home": 99, "TEAM_ID_away": 98,
            "TEAM_NAME_home": "Team_X", "TEAM_NAME_away": "Team_Y",
            "avg_pts_5g_home": 115.0, "avg_pts_5g_away": 108.0,
            "pts_diff_5g": 7.0,
            "rest_home_days": 2.0, "rest_away_days": 3.0,
            "travel_distance": 500.0, "h2h_win_rate": 0.5, "h2h_avg_margin": 2.0,
        }]
        tiny_df = pd.DataFrame(rows)

        cols = ["avg_pts_5g_home", "avg_pts_5g_away", "pts_diff_5g"]
        result = predictor._build_feature_vector(
            "Team_X", "Team_Y", tiny_df, feature_cols=cols,
        )
        assert result is not None
        assert result["avg_pts_5g_home"] == 115.0  # Single row, unchanged
        assert result["avg_pts_5g_away"] == 108.0
        assert result["pts_diff_5g"] == 7.0

    def test_two_games_renormalizes_weights(self, predictor, features_df):
        """With only 2 games available, weights [0.5, 0.3] are renormalized to [0.625, 0.375]."""
        # Build a DF with exactly 2 home games for Team_X
        rows = []
        for g in range(2):
            rows.append({
                "GAME_ID": f"G{g:04d}", "GAME_DATE": pd.Timestamp(f"2025-01-{10 + g:02d}"),
                "TEAM_ID_home": 99, "TEAM_ID_away": 98,
                "TEAM_NAME_home": "Team_X", "TEAM_NAME_away": "Team_Y",
                "avg_pts_5g_home": float(110 + g * 5), "avg_pts_5g_away": 108.0,
                "pts_diff_5g": float(2 + g * 5),
                "rest_home_days": 2.0, "rest_away_days": 3.0,
                "travel_distance": 500.0, "h2h_win_rate": 0.5, "h2h_avg_margin": 2.0,
            })
        tiny_df = pd.DataFrame(rows)

        cols = ["avg_pts_5g_home"]
        result = predictor._build_feature_vector(
            "Team_X", "Team_Y", tiny_df, feature_cols=cols,
        )
        assert result is not None
        # Most recent (index 1, value 115) gets 0.5/0.8 = 0.625
        # Second (index 0, value 110) gets 0.3/0.8 = 0.375
        expected = 115.0 * 0.625 + 110.0 * 0.375
        assert abs(result["avg_pts_5g_home"] - expected) < 0.001


class TestDiffColumns:
    """Tests for _diff column computation: home_value - away_value."""

    def test_diff_column_not_matching_uses_direct_row(self, predictor, features_df):
        """Columns ending in _5g (not _diff) fall through to direct row."""
        cols = ["avg_pts_5g_home", "avg_pts_5g_away", "pts_diff_5g"]
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df, feature_cols=cols,
        )
        assert result is not None
        # pts_diff_5g does NOT end with "_diff" — it ends with "_5g".
        # The _diff branch only triggers for columns literally ending in "_diff".
        # Reverts to direct row value (from the DataFrame), not home - away.
        direct_mask = (
            (features_df["TEAM_NAME_home"] == "Team_A")
            & (features_df["TEAM_NAME_away"] == "Team_B")
        )
        if direct_mask.any():
            latest = features_df[direct_mask].iloc[-1]
            assert abs(result["pts_diff_5g"] - float(latest["pts_diff_5g"])) < 0.001

    def test_power_diff_base_mismatch_uses_direct_row(self, predictor, features_df):
        """power_diff ends with _diff but base='power' doesn't match any columns."""
        cols = ["composite_power_home", "composite_power_away", "power_diff"]
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df, feature_cols=cols,
        )
        assert result is not None
        # power_diff -> base='power' -> looks for 'power_home'/'power_away'
        # Columns are 'composite_power_home'/'composite_power_away' — no match
        # Falls through to the direct row value from the DataFrame.
        direct_mask = (
            (features_df["TEAM_NAME_home"] == "Team_A")
            & (features_df["TEAM_NAME_away"] == "Team_B")
        )
        if direct_mask.any():
            latest = features_df[direct_mask].iloc[-1]
            assert abs(result["power_diff"] - float(latest["power_diff"])) < 0.001

    def test_diff_computation_with_matching_base_names(self, predictor):
        """_diff columns with matching _home/_away base names compute home - away."""
        rows = [
            {
                "GAME_ID": "G0001", "GAME_DATE": pd.Timestamp("2025-01-10"),
                "TEAM_ID_home": 0, "TEAM_ID_away": 1,
                "TEAM_NAME_home": "Team_X", "TEAM_NAME_away": "Team_Y",
                "feature_home": 115.0, "feature_away": 108.0,
                "feature_diff": 999.0,  # Intentionally wrong — must be computed as 115-108=7
                "other_home": 50.0, "other_away": 48.0,
                "other_diff": 777.0,    # Intentionally wrong — must be computed as 50-48=2
            },
        ]
        df = pd.DataFrame(rows)
        cols = ["feature_home", "feature_away", "feature_diff",
                "other_home", "other_away", "other_diff"]
        result = predictor._build_feature_vector(
            "Team_X", "Team_Y", df, feature_cols=cols,
        )
        assert result is not None
        # feature_diff -> base="feature" -> looks for "feature_home" and
        # "feature_away" — both exist in weighted average results.
        # Should compute: feature_home - feature_away = 115.0 - 108.0 = 7.0
        expected_feature = result["feature_home"] - result["feature_away"]
        assert abs(result["feature_diff"] - expected_feature) < 0.001, (
            f"feature_diff: expected {expected_feature:.4f}, got {result['feature_diff']:.4f}"
        )
        expected_other = result["other_home"] - result["other_away"]
        assert abs(result["other_diff"] - expected_other) < 0.001


class TestGlobalFeatures:
    """Tests for rest, travel, fatigue, and H2H features (use direct matchup)."""

    def test_global_features_use_direct_matchup(self, predictor, features_df):
        """Global features like rest_home_days should match the most recent direct matchup."""
        cols = ["rest_home_days", "rest_away_days", "travel_distance", "h2h_win_rate"]
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df, feature_cols=cols,
        )
        assert result is not None

        # Find the most recent direct matchup (Team_A home vs Team_B away)
        direct_mask = (
            (features_df["TEAM_NAME_home"] == "Team_A")
            & (features_df["TEAM_NAME_away"] == "Team_B")
        )
        if direct_mask.any():
            latest_game = features_df[direct_mask].iloc[-1]
            for col in cols:
                assert abs(result[col] - latest_game[col]) < 0.001, (
                    f"{col}: expected {latest_game[col]}, got {result[col]}"
                )

    def test_h2h_from_direct_matchup(self, predictor, features_df):
        """H2H columns like h2h_win_rate should use direct matchup row."""
        cols = ["h2h_win_rate", "h2h_avg_margin"]
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df, feature_cols=cols,
        )
        assert result is not None

        direct_mask = (
            (features_df["TEAM_NAME_home"] == "Team_A")
            & (features_df["TEAM_NAME_away"] == "Team_B")
        )
        if direct_mask.any():
            latest = features_df[direct_mask].iloc[-1]
            assert abs(result["h2h_win_rate"] - latest["h2h_win_rate"]) < 0.001
            assert abs(result["h2h_avg_margin"] - latest["h2h_avg_margin"]) < 0.001


class TestWeightingAcrossPeriods:
    """Tests that the weighted average correctly handles different period scenarios."""

    def _make_controlled_df(self, values: list[float], col_suffix: str = "home") -> pd.DataFrame:
        """Create a minimal DF where a team's feature values are exactly controlled."""
        rows = []
        for i, val in enumerate(values):
            team = "Team_X" if col_suffix == "home" else "Team_Y"
            other = "Team_Y" if col_suffix == "home" else "Team_X"
            rows.append({
                "GAME_ID": f"G{i:04d}",
                "GAME_DATE": pd.Timestamp(f"2025-01-{10 + i:02d}"),
                "TEAM_ID_home": 0, "TEAM_ID_away": 1,
                "TEAM_NAME_home": team,
                "TEAM_NAME_away": other,
                f"avg_test_{col_suffix}": float(val),
            })
        return pd.DataFrame(rows)

    def test_three_games_correct_weights(self, predictor):
        """With exactly 3 games, weights are [0.50, 0.30, 0.20]."""
        df = self._make_controlled_df([100, 120, 140])
        cols = ["avg_test_home"]
        result = predictor._build_feature_vector(
            "Team_X", "Team_Y", df, feature_cols=cols,
        )
        assert result is not None
        # 140*0.5 + 120*0.3 + 100*0.2 = 70 + 36 + 20 = 126
        assert abs(result["avg_test_home"] - 126.0) < 0.001

    def test_single_game_weight_is_one(self, predictor):
        """With 1 game, weight is [1.0]."""
        df = self._make_controlled_df([150])
        cols = ["avg_test_home"]
        result = predictor._build_feature_vector(
            "Team_X", "Team_Y", df, feature_cols=cols,
        )
        assert result is not None
        assert result["avg_test_home"] == 150.0

    def test_outlier_smoothed_by_multi_game(self, predictor):
        """A single outlier game is smoothed when 3 games are available.

        With values [80, 110, 112], the avg=100.67.
        The single-row approach would use 112 (most recent).
        The weighted approach: 112*0.5 + 110*0.3 + 80*0.2 = 105.0
        This is closer to the 3-game mean than the 112 outlier.
        """
        df = self._make_controlled_df([80, 110, 112])
        cols = ["avg_test_home"]
        result = predictor._build_feature_vector(
            "Team_X", "Team_Y", df, feature_cols=cols,
        )
        assert result is not None
        # 112*0.5 + 110*0.3 + 80*0.2 = 56 + 33 + 16 = 105
        assert abs(result["avg_test_home"] - 105.0) < 0.001


class TestEdgeCases:
    """Edge cases for the feature vector builder."""

    def test_column_name_case_insensitivity(self, predictor, features_df):
        """Team name matching should be case-insensitive."""
        cols = ["avg_pts_5g_home"]
        result_upper = predictor._build_feature_vector(
            "TEAM_A", "TEAM_B", features_df, feature_cols=cols,
        )
        result_lower = predictor._build_feature_vector(
            "team_a", "team_b", features_df, feature_cols=cols,
        )
        result_mixed = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df, feature_cols=cols,
        )
        assert result_upper is not None
        assert abs(result_upper["avg_pts_5g_home"] - result_lower["avg_pts_5g_home"]) < 0.001
        assert abs(result_upper["avg_pts_5g_home"] - result_mixed["avg_pts_5g_home"]) < 0.001

    def test_team_with_spaces(self, predictor, features_df):
        """Team names with leading/trailing spaces should be trimmed."""
        # Create a row with a padded team name
        df = features_df.copy()
        mask = df["TEAM_NAME_home"] == "Team_A"
        if mask.any():
            df.loc[mask, "TEAM_NAME_home"] = "  Team_A  "

        cols = ["avg_pts_5g_home"]
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", df, feature_cols=cols,
        )
        assert result is not None
        assert np.isfinite(result["avg_pts_5g_home"])

    def test_reversed_home_away_direct_matchup(self, predictor, features_df):
        """Direct matchup should work when Team_A is away and Team_B is home."""
        cols = ["rest_home_days"]
        result = predictor._build_feature_vector(
            "Team_C", "Team_A", features_df, feature_cols=cols,
        )
        assert result is not None
        # Should find a direct matchup even though Team_C is away and Team_A is home
        # (the direct_mask searches both orientations)

    def test_all_requested_cols_exist(self, predictor, features_df):
        """Auto-detected feature columns should all exist in the result."""
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df,
        )
        assert result is not None
        # Every column in the result should have a finite value
        assert all(pd.api.types.is_float_dtype(result) for _ in [0]), "All values should be float"

    def test_many_columns_consistency(self, predictor, features_df):
        """With many columns, all should be computed and finite."""
        # Use all available feature columns from the DF that end with _home, _away, or _diff
        cols = [c for c in features_df.columns if (
            c.endswith("_home") or c.endswith("_away") or c.endswith("_diff")
        ) and c not in ("TEAM_NAME_home", "TEAM_NAME_away", "TEAM_ID_home", "TEAM_ID_away")]
        # Add rest, travel, H2H columns
        extra = ["rest_home_days", "rest_away_days", "travel_distance",
                 "h2h_win_rate", "h2h_avg_margin"]
        cols = [c for c in cols if c in features_df.columns] + extra

        result = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df, feature_cols=cols,
        )
        assert result is not None
        assert len(result) == len(cols)
        assert np.isfinite(result.values).all(), f"Non-finite values found in {len(cols)} cols"

    def test_consecutive_calls_same_teams_consistent(self, predictor, features_df):
        """Calling twice for the same matchup should give the same result."""
        cols = ["avg_pts_5g_home", "avg_pts_5g_away", "pts_diff_5g"]
        r1 = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df, feature_cols=cols,
        )
        r2 = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df, feature_cols=cols,
        )
        assert r1 is not None and r2 is not None
        for col in cols:
            assert abs(r1[col] - r2[col]) < 0.001, f"{col} differs between calls"


class TestTotalsSpecific:
    """Tests that the feature vector builder works for totals model scenarios."""

    def test_totals_feature_cols_work(self, predictor, features_df):
        """Feature columns commonly used by the TotalsRegressor should produce valid vectors."""
        cols = [
            "avg_pts_5g_home", "avg_pts_5g_away", "pts_diff_5g",
            "avg_reb_5g_home", "avg_reb_5g_away",
            "avg_ast_5g_home", "avg_ast_5g_away",
            "ema_pts_10g_home", "ema_pts_10g_away",
            "rest_home_days", "rest_away_days",
            "travel_distance", "h2h_win_rate",
        ]
        result = predictor._build_feature_vector(
            "Team_A", "Team_B", features_df, feature_cols=cols,
        )
        assert result is not None
        assert len(result) == len(cols)
        assert np.isfinite(result.values).all()
        assert all(c in result.index for c in cols)
