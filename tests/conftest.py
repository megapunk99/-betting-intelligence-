"""Shared test fixtures for the betting intelligence system."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Set a test API key so that web.auth API key middleware allows test requests.
# This MUST be set before any module that imports web.app is loaded.
os.environ.setdefault("API_KEY", "test-api-key-for-integration-tests")
TEST_API_KEY = os.environ["API_KEY"]

# Default headers to include in all TestClient-based integration tests
TEST_HEADERS = {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def sample_game_data() -> pd.DataFrame:
    """Generate synthetic game data for testing."""
    np.random.seed(42)
    n_games = 100

    home_teams = [f"Team_{i}" for i in range(10)]
    away_teams = [f"Team_{(i + 5) % 10}" for i in range(10)]

    data = {
        "GAME_ID": [f"GAME_{i:04d}" for i in range(n_games)],
        "GAME_DATE": pd.date_range("2023-01-01", periods=n_games, freq="D"),
        "TEAM_ID_home": [i % 10 for i in range(n_games)],
        "TEAM_ID_away": [(i + 5) % 10 for i in range(n_games)],
        "TEAM_NAME_home": [home_teams[i % 10] for i in range(n_games)],
        "TEAM_NAME_away": [away_teams[i % 10] for i in range(n_games)],
        "team_pts_home": np.random.normal(110, 10, n_games),
        "team_pts_away": np.random.normal(108, 10, n_games),
        "team_fga_home": np.random.normal(85, 8, n_games),
        "team_fga_away": np.random.normal(84, 8, n_games),
        "team_fgm_home": np.random.normal(40, 5, n_games),
        "team_fgm_away": np.random.normal(39, 5, n_games),
        "team_fg3m_home": np.random.normal(12, 3, n_games),
        "team_fg3m_away": np.random.normal(11, 3, n_games),
        "team_fg3a_home": np.random.normal(33, 5, n_games),
        "team_fg3a_away": np.random.normal(32, 5, n_games),
        "team_oreb_home": np.random.normal(10, 3, n_games),
        "team_oreb_away": np.random.normal(9, 3, n_games),
        "team_tov_home": np.random.normal(13, 3, n_games),
        "team_tov_away": np.random.normal(12, 3, n_games),
        "team_plus_minus_home": np.random.normal(2, 8, n_games),
        "team_plus_minus_away": np.random.normal(-2, 8, n_games),
        "WL_home": np.random.choice(["W", "L"], n_games),
        "WL_away": np.random.choice(["W", "L"], n_games),
        "MATCHUP_home": [
            f"Team_{i % 10} vs. Team_{(i + 5) % 10}" for i in range(n_games)
        ],
        "MATCHUP_away": [
            f"Team_{(i + 5) % 10} @ Team_{i % 10}" for i in range(n_games)
        ],
    }
    df = pd.DataFrame(data)

    # Derived columns
    df["total_points"] = df["team_pts_home"] + df["team_pts_away"]
    df["point_diff"] = df["team_pts_home"] - df["team_pts_away"]
    df["pace"] = (
        df["team_fga_home"]
        + df["team_tov_home"]
        - df["team_oreb_home"]
        + df["team_fga_away"]
        + df["team_tov_away"]
        - df["team_oreb_away"]
    )
    df["eFG_home"] = (df["team_fgm_home"] + 0.5 * df["team_fg3m_home"]) / df[
        "team_fga_home"
    ].clip(lower=1)
    df["eFG_away"] = (df["team_fgm_away"] + 0.5 * df["team_fg3m_away"]) / df[
        "team_fga_away"
    ].clip(lower=1)

    return df


@pytest.fixture
def sample_bets_dataframe() -> pd.DataFrame:
    """Generate synthetic betting records for testing."""
    np.random.seed(42)
    n_bets = 50

    data = {
        "game_date": pd.date_range("2023-01-01", periods=n_bets, freq="D"),
        "game_id": [f"GAME_{i:04d}" for i in range(n_bets)],
        "matchup": ["Team_A vs Team_B" for _ in range(n_bets)],
        "strategy": np.random.choice(
            ["momentum", "pace_total", "spread_model"], n_bets
        ),
        "model": np.random.choice(["Ridge", "XGBoost", "Logistic"], n_bets),
        "bet_type": np.random.choice(["TOTAL_OVER", "TOTAL_UNDER", "SPREAD"], n_bets),
        "predicted_total": np.random.normal(220, 10, n_bets),
        "market_line": np.random.normal(218, 8, n_bets),
        "actual_total": np.random.normal(219, 12, n_bets),
        "edge_pct": np.random.uniform(-0.08, 0.08, n_bets),
        "outcome": np.random.choice(
            ["WIN", "LOSS", "PUSH"], n_bets, p=[0.55, 0.40, 0.05]
        ),
        "profit_units": np.where(np.random.random(n_bets) > 0.55, 1.0, -1.0),
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
