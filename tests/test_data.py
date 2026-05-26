"""Tests for data loading and feature engineering modules."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from betting_intel.data.features import FeatureEngineer
from betting_intel.data.loader import NBADataLoader


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""

    @pytest.fixture
    def engineer(self):
        return FeatureEngineer(rolling_windows=[3, 5])

    @pytest.fixture
    def sample_games(self):
        """Merged game-level dataframe (one row per game)."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "GAME_ID": [f"G_{i:04d}" for i in range(n)],
            "GAME_DATE": pd.date_range("2023-01-01", periods=n, freq="D"),
            "TEAM_ID_home": [i % 10 for i in range(n)],
            "TEAM_ID_away": [(i + 5) % 10 for i in range(n)],
            "TEAM_NAME_home": [f"Team_{i % 10}" for i in range(n)],
            "TEAM_NAME_away": [f"Team_{(i + 5) % 10}" for i in range(n)],
            "TEAM_ABBREVIATION_home": [f"T{i % 10}" for i in range(n)],
            "TEAM_ABBREVIATION_away": [f"T{(i + 5) % 10}" for i in range(n)],
            "MATCHUP_home": [f"Team_{i % 10} vs. Team_{(i + 5) % 10}" for i in range(n)],
            "MATCHUP_away": [f"Team_{(i + 5) % 10} @ Team_{i % 10}" for i in range(n)],
            "team_pts_home": np.random.normal(110, 10, n),
            "team_pts_away": np.random.normal(108, 10, n),
            "team_fga_home": np.random.normal(85, 8, n),
            "team_fga_away": np.random.normal(84, 8, n),
            "team_fgm_home": np.random.normal(40, 5, n),
            "team_fgm_away": np.random.normal(39, 5, n),
            "team_fg3m_home": np.random.normal(12, 3, n),
            "team_fg3m_away": np.random.normal(11, 3, n),
            "team_fg3a_home": np.random.normal(33, 5, n),
            "team_fg3a_away": np.random.normal(32, 5, n),
            "team_oreb_home": np.random.normal(10, 3, n),
            "team_oreb_away": np.random.normal(9, 3, n),
            "team_tov_home": np.random.normal(13, 3, n),
            "team_tov_away": np.random.normal(12, 3, n),
            "team_plus_minus_home": np.random.normal(2, 8, n),
            "team_plus_minus_away": np.random.normal(-2, 8, n),
            "WL_home": np.random.choice(["W", "L"], n),
            "WL_away": np.random.choice(["W", "L"], n),
            "MIN_home": np.random.uniform(200, 260, n),
            "MIN_away": np.random.uniform(200, 260, n),
            "eFG_home": np.random.uniform(0.45, 0.58, n),
            "eFG_away": np.random.uniform(0.44, 0.57, n),
            "total_points": np.random.normal(218, 15, n),
            "point_diff": np.random.normal(2, 10, n),
        })
        return df

    @pytest.fixture
    def sample_raw(self):
        """Raw team-level dataframe (two rows per game, one per team)."""
        np.random.seed(42)
        n = 200  # 100 games x 2 teams
        game_ids = []
        for i in range(100):
            game_ids.append(f"G_{i:04d}")
            game_ids.append(f"G_{i:04d}")

        dates = []
        for i in range(100):
            dates.append(pd.Timestamp("2023-01-01") + pd.Timedelta(days=i))
            dates.append(pd.Timestamp("2023-01-01") + pd.Timedelta(days=i))

        df = pd.DataFrame({
            "GAME_ID": game_ids,
            "GAME_DATE": dates,
            "TEAM_ID": [i % 10 for i in range(200)],
            "TEAM_NAME": [f"Team_{i % 10}" for i in range(200)],
            "TEAM_ABBREVIATION": [f"T{i % 10}" for i in range(200)],
            "MATCHUP": [f"Team_{i % 10} vs. Team_{(i + 5) % 10}" if i % 2 == 0 else f"Team_{(i + 5) % 10} @ Team_{i % 10}" for i in range(200)],
            "IS_HOME": [1 if i % 2 == 0 else 0 for i in range(200)],
            "PTS": np.random.normal(109, 10, 200),
            "MIN": np.random.uniform(200, 260, 200),
            "FGM": np.random.normal(40, 5, 200),
            "FGA": np.random.normal(85, 8, 200),
            "FG_PCT": np.random.uniform(0.42, 0.52, 200),
            "FG3M": np.random.normal(12, 3, 200),
            "FG3A": np.random.normal(33, 5, 200),
            "FG3_PCT": np.random.uniform(0.32, 0.40, 200),
            "FTM": np.random.normal(18, 4, 200),
            "FTA": np.random.normal(23, 5, 200),
            "FT_PCT": np.random.uniform(0.72, 0.85, 200),
            "OREB": np.random.normal(10, 3, 200),
            "DREB": np.random.normal(30, 5, 200),
            "REB": np.random.normal(40, 6, 200),
            "AST": np.random.normal(24, 5, 200),
            "STL": np.random.normal(7, 2, 200),
            "BLK": np.random.normal(5, 2, 200),
            "TOV": np.random.normal(13, 3, 200),
            "PF": np.random.normal(20, 4, 200),
            "PLUS_MINUS": np.random.normal(0, 8, 200),
            "WL": np.random.choice(["W", "L"], 200),
            "rest_days": np.random.randint(0, 5, 200).astype(float),
        })
        return df

    def test_select_features_excludes_non_feature_columns(self, engineer, sample_games):
        """Feature selection should exclude metadata and target columns."""
        cols = engineer.select_features(sample_games)
        exclude = {"GAME_ID", "TEAM_NAME_home", "TEAM_NAME_away", "GAME_DATE",
                   "total_points", "point_diff", "WL_home", "WL_away"}
        selected_set = set(cols)
        assert len(exclude & selected_set) == 0, f"Found excluded columns: {exclude & selected_set}"
        assert len(cols) > 0, "Should select at least some features"

    def test_build_all_features_creates_rolling_averages(self, engineer, sample_games, sample_raw):
        """Feature engineering should create rolling average columns."""
        result = engineer.build_all_features(sample_games, sample_raw)
        assert "avg_pts_3g_home" in result.columns or any("avg_pts" in c for c in result.columns)
        assert len(result) == len(sample_games)

    def test_compute_streak_positive(self, engineer):
        """Streak computation should handle winning streaks."""
        wl = pd.Series([1, 1, 1, 0, 0, 1, 0, 1, 1])
        result = engineer._compute_streak(wl)
        # After shift(1), first value is NaN -> fillna(0)
        assert result.iloc[0] == 0
        assert result.iloc[3] == 3  # Three wins before this game

    def test_compute_streak_negative(self, engineer):
        """Streak computation should handle losing streaks."""
        wl = pd.Series([0, 0, 0, 1, 1, 0, 0, 0, 0])
        result = engineer._compute_streak(wl)
        assert result.iloc[0] == 0
        assert result.iloc[3] == -3  # Three losses before this game


class TestNBADataLoader:
    """Tests for NBADataLoader."""

    def test_init_default_path(self):
        """Loader should initialize with default config path."""
        loader = NBADataLoader()
        assert loader.db_path is not None
