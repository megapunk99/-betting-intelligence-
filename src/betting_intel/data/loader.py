"""
Data loader: loads NBA game logs from SQLite and provides game-level views.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from betting_intel.config import DB_PATH, MAX_REST_DAYS


class NBADataLoader:
    """Loads and preprocesses NBA game log data from the SQLite database."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH

    def load_game_logs(self) -> pd.DataFrame:
        """Load raw game logs from SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        query = """
        SELECT
            SEASON_ID, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME,
            GAME_ID, GAME_DATE, MATCHUP, WL, MIN,
            PTS, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT,
            FTM, FTA, FT_PCT,
            OREB, DREB, REB, AST, STL, BLK, TOV, PF, PLUS_MINUS,
            SEASON
        FROM game_logs
        ORDER BY GAME_DATE, TEAM_ID
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Parse dates
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        # Parse numeric columns
        numeric_cols = [
            "MIN", "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL",
            "BLK", "TOV", "PF", "PLUS_MINUS"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def build_game_dataset(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build a game-level dataset where each row is ONE game (both teams).
        Derives home/away, opponent stats, and game-level aggregates.
        """
        if df is None:
            df = self.load_game_logs()

        # Parse home/away from MATCHUP
        df["IS_HOME"] = df["MATCHUP"].fillna("").str.contains("vs.").astype(int)
        df["OPPONENT"] = df["MATCHUP"].fillna("").str.split(" ").str[-1]

        # Rename columns for clarity
        team_cols = {
            "PTS": "team_pts", "FGM": "team_fgm", "FGA": "team_fga",
            "FG_PCT": "team_fg_pct", "FG3M": "team_fg3m", "FG3A": "team_fg3a",
            "FG3_PCT": "team_fg3_pct", "FTM": "team_ftm", "FTA": "team_fta",
            "FT_PCT": "team_ft_pct", "OREB": "team_oreb", "DREB": "team_dreb",
            "REB": "team_reb", "AST": "team_ast", "STL": "team_stl",
            "BLK": "team_blk", "TOV": "team_tov", "PF": "team_pf",
            "PLUS_MINUS": "team_plus_minus"
        }
        df = df.rename(columns=team_cols)

        # Merge both teams' stats per game
        home = df[df["IS_HOME"] == 1].copy()
        away = df[df["IS_HOME"] == 0].copy()

        # Create game ID from GAME_ID (they come in pairs)
        home_games = home.copy()
        away_games = away.copy()

        # Merge on GAME_ID
        games = pd.merge(
            home_games, away_games,
            on="GAME_ID",
            suffixes=("_home", "_away"),
            how="inner"
        )

        # Basic game-level features
        games["GAME_DATE"] = games["GAME_DATE_home"]
        games["total_points"] = games["team_pts_home"] + games["team_pts_away"]
        games["point_diff"] = games["team_pts_home"] - games["team_pts_away"]
        games["pace"] = (
            games["team_fga_home"] + games["team_tov_home"]
            - games["team_oreb_home"] +
            games["team_fga_away"] + games["team_tov_away"]
            - games["team_oreb_away"]
        )
        games["eFG_home"] = (games["team_fgm_home"] + 0.5 * games["team_fg3m_home"]) / games["team_fga_home"]
        games["eFG_away"] = (games["team_fgm_away"] + 0.5 * games["team_fg3m_away"]) / games["team_fga_away"]

        # Sort by date
        games = games.sort_values("GAME_DATE").reset_index(drop=True)

        return games

    def compute_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rest days for each team since their last game.
        Needs raw dataframe (not merged game dataset).
        """
        df = df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values(["TEAM_ID", "GAME_DATE"])
        df["rest_days"] = df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days
        df["rest_days"] = df["rest_days"].fillna(MAX_REST_DAYS).clip(0, MAX_REST_DAYS)
        df["is_back_to_back"] = (df["rest_days"] == 0).astype(int)
        return df


def get_team_season_rolling(df: pd.DataFrame, team_col: str, stat_col: str, window: int) -> pd.Series:
    """
    Compute rolling average of a stat for a specific team.
    This is used for building features.
    """
    team_data = df.sort_values("GAME_DATE")
    return team_data.groupby(team_col)[stat_col].transform(
        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
    )


def get_team_cumulative(df: pd.DataFrame, team_col: str, stat_col: str) -> pd.Series:
    """Compute cumulative season average for a team."""
    team_data = df.sort_values("GAME_DATE")
    cumsum = team_data.groupby(team_col)[stat_col].cumsum()
    cumcount = team_data.groupby(team_col)[stat_col].cumcount() + 1
    return (cumsum / cumcount).shift(1)
