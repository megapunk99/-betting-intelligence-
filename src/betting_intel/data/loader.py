"""
Data loader: loads game logs from SQLite (NBA) or ESPN API (NCAAB) and
provides game-level views for feature engineering.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import zlib

import pandas as pd
import numpy as np

from betting_intel.config import DB_PATH, MAX_REST_DAYS

logger = logging.getLogger(__name__)


# ── Team name column standardization across leagues ──────────────────────
# NBA SQLite uses TEAM_NAME. ESPN-based sources use home_team/away_team.
# These helpers ensure the feature pipeline sees consistent column names.


def _standardize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure team name columns use 'TEAM_NAME' convention."""
    if "TEAM_NAME" not in df.columns:
        if "home_team" in df.columns:
            # ESPN-style: rename to match NBADataLoader convention
            pass  # build_game_dataset handles this
    return df


class NBADataLoader:
    """Loads and preprocesses NBA game log data from the SQLite database."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH

    def load_game_logs(self) -> pd.DataFrame:
        """Load raw game logs from SQLite."""
        if not self.db_path or not self.db_path.exists():
            logger.warning(f"Database not found at {self.db_path} — returning empty DataFrame")
            return pd.DataFrame()

        try:
            conn = sqlite3.connect(str(self.db_path))
        except (sqlite3.Error, OSError) as e:
            logger.error(f"Failed to connect to database at {self.db_path}: {e}")
            return pd.DataFrame()

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
        try:
            df = pd.read_sql_query(query, conn)
        except (pd.io.sql.DatabaseError, sqlite3.Error) as e:
            logger.error(f"Failed to read game_logs table: {e}")
            conn.close()
            return pd.DataFrame()
        finally:
            conn.close()

        if df.empty:
            logger.warning("game_logs table is empty")
            return df

        # Parse dates
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        # Parse numeric columns
        numeric_cols = [
            "MIN", "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL",
            "BLK", "TOV", "PF", "PLUS_MINUS"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def build_game_dataset(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build a game-level dataset where each row is ONE game (both teams).
        Derives home/away, opponent stats, and game-level aggregates.
        """
        if df is None:
            df = self.load_game_logs()

        if df is None or df.empty:
            return pd.DataFrame()

        try:
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
            # Only rename columns that exist
            rename_map = {k: v for k, v in team_cols.items() if k in df.columns}
            df = df.rename(columns=rename_map)

            # Merge both teams' stats per game
            home = df[df["IS_HOME"] == 1].copy()
            away = df[df["IS_HOME"] == 0].copy()

            if home.empty or away.empty:
                logger.warning(f"Missing home ({len(home)}) or away ({len(away)}) rows in game dataset")
                return pd.DataFrame()

            # Merge on GAME_ID
            games = pd.merge(
                home, away,
                on="GAME_ID",
                suffixes=("_home", "_away"),
                how="inner"
            )

            if games.empty:
                logger.warning("Merged game dataset is empty — likely GAME_ID mismatch")
                return pd.DataFrame()

            # Basic game-level features (with divide-by-zero guard)
            games["GAME_DATE"] = games["GAME_DATE_home"]
            games["total_points"] = games["team_pts_home"] + games["team_pts_away"]
            games["point_diff"] = games["team_pts_home"] - games["team_pts_away"]
            games["pace"] = (
                games["team_fga_home"].fillna(0) + games["team_tov_home"].fillna(0)
                - games["team_oreb_home"].fillna(0) +
                games["team_fga_away"].fillna(0) + games["team_tov_away"].fillna(0)
                - games["team_oreb_away"].fillna(0)
            )
            games["eFG_home"] = np.where(
                games["team_fga_home"].fillna(0) > 0,
                (games["team_fgm_home"].fillna(0) + 0.5 * games["team_fg3m_home"].fillna(0)) / games["team_fga_home"].clip(lower=1),
                0.0
            )
            games["eFG_away"] = np.where(
                games["team_fga_away"].fillna(0) > 0,
                (games["team_fgm_away"].fillna(0) + 0.5 * games["team_fg3m_away"].fillna(0)) / games["team_fga_away"].clip(lower=1),
                0.0
            )

            # Sort by date
            games = games.sort_values("GAME_DATE").reset_index(drop=True)

            return games

        except Exception as e:
            logger.error(f"Failed to build game dataset: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return pd.DataFrame()

    def compute_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rest days for each team since their last game.
        Needs raw dataframe (not merged game dataset).
        """
        if df is None or df.empty:
            return pd.DataFrame()
        try:
            df = df.copy()
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

            # Determine team identifier column (TEAM_ID for NBA, TEAM_NAME for NCAAB)
            team_id_col = "TEAM_ID" if "TEAM_ID" in df.columns else "TEAM_NAME"
            if team_id_col not in df.columns:
                logger.warning(f"No team identifier column found for rest days computation")
                df["rest_days"] = MAX_REST_DAYS
                df["is_back_to_back"] = 0
                return df

            df = df.sort_values([team_id_col, "GAME_DATE"])
            df["rest_days"] = df.groupby(team_id_col)["GAME_DATE"].diff().dt.days
            df["rest_days"] = df["rest_days"].fillna(MAX_REST_DAYS).clip(0, MAX_REST_DAYS)
            df["is_back_to_back"] = (df["rest_days"] <= 1).astype(int)
            return df
        except Exception as e:
            logger.error(f"Failed to compute rest days: {e}")
            df = df.copy() if df is not None else pd.DataFrame()
            df["rest_days"] = MAX_REST_DAYS
            df["is_back_to_back"] = 0
            return df


# ── NCAAB Data Loader (ESPN-based) ───────────────────────────────────────

class NCAABDataLoader:
    """
    Loads and preprocesses NCAAB game data from the ESPN API.

    NCAAB does not have a local SQLite database (unlike NBA). Data is
    fetched on-demand from ESPN's public API and cached in memory.

    The ESPN API returns game results with box-score-level stats for
    college basketball, which are transformed into the same schema as
    NBADataLoader for compatibility with the FeatureEngineer pipeline.
    """

    def __init__(self, seasons: Optional[list] = None):
        self._cache: Optional[pd.DataFrame] = None
        self._seasons = seasons  # e.g. [2025, 2026] for the 2024-25, 2025-26 seasons

    def load_game_logs(self) -> pd.DataFrame:
        """
        Load NCAAB game logs from ESPN API.

        Returns a DataFrame with the same column schema as NBADataLoader
        so the FeatureEngineer pipeline works identically.

        Returns empty DataFrame if ESPN is unreachable or no data available.
        """
        if self._cache is not None and not self._cache.empty:
            return self._cache

        try:
            from betting_intel.data.espn_hoops import ESPNLeagueSource

            source = ESPNLeagueSource()
            df = source.load_historical("ncaab", seasons=self._seasons)

            if df is None or df.empty:
                logger.warning("No NCAAB data available from ESPN")
                return pd.DataFrame()

            # Transform ESPN schema to NBADataLoader-compatible schema
            # ESPN returns: game_id, league, season, date, home_team, away_team,
            #               home_score, away_score, total_points, home_win
            # We need: GAME_ID, TEAM_ID, TEAM_NAME, GAME_DATE, MATCHUP, WL,
            #          PTS, MIN, and other box-score stats
            records = []
            for _, row in df.iterrows():
                game_id = row.get("game_id", "")
                season = row.get("season", "")
                game_date = row.get("date", "")
                home_team = row.get("home_team", "")
                away_team = row.get("away_team", "")
                home_score = row.get("home_score", 0)
                away_score = row.get("away_score", 0)
                home_win = row.get("home_win", 1 if home_score > away_score else 0)

                # Home team row
                records.append({
                    "GAME_ID": f"{game_id}_home",
                    "TEAM_ID": abs(zlib.crc32(f"ncaab_{home_team}".encode())) % (2**31),
                    "TEAM_ABBREVIATION": home_team[:3].upper(),
                    "TEAM_NAME": home_team,
                    "GAME_DATE": game_date,
                    "MATCHUP": f"{home_team} vs. {away_team}",
                    "WL": "W" if home_win else "L",
                    "MIN": 200,  # NCAAB games are ~40 min, approximate
                    "PTS": home_score,
                    "FGM": 0, "FGA": 0, "FG_PCT": 0,
                    "FG3M": 0, "FG3A": 0, "FG3_PCT": 0,
                    "FTM": 0, "FTA": 0, "FT_PCT": 0,
                    "OREB": 0, "DREB": 0, "REB": 0,
                    "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "PF": 0,
                    "PLUS_MINUS": home_score - away_score,
                    "SEASON": season,
                    "IS_HOME": 1,
                    "OPPONENT": away_team,
                    "team_pts": home_score,
                })
                # Away team row
                records.append({
                    "GAME_ID": f"{game_id}_away",
                    "TEAM_ID": abs(zlib.crc32(f"ncaab_{away_team}".encode())) % (2**31),
                    "TEAM_ABBREVIATION": away_team[:3].upper(),
                    "TEAM_NAME": away_team,
                    "GAME_DATE": game_date,
                    "MATCHUP": f"{away_team} @ {home_team}",
                    "WL": "W" if not home_win else "L",
                    "MIN": 200,
                    "PTS": away_score,
                    "FGM": 0, "FGA": 0, "FG_PCT": 0,
                    "FG3M": 0, "FG3A": 0, "FG3_PCT": 0,
                    "FTM": 0, "FTA": 0, "FT_PCT": 0,
                    "OREB": 0, "DREB": 0, "REB": 0,
                    "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "PF": 0,
                    "PLUS_MINUS": away_score - home_score,
                    "SEASON": season,
                    "IS_HOME": 0,
                    "OPPONENT": home_team,
                    "team_pts": away_score,
                })

            result = pd.DataFrame(records)
            result["GAME_DATE"] = pd.to_datetime(result["GAME_DATE"])

            self._cache = result
            logger.info(f"NCAAB: loaded {len(result)} rows ({len(result)//2} games) from ESPN")
            return result

        except Exception as e:
            logger.warning(f"Failed to load NCAAB data from ESPN: {e}")
            return pd.DataFrame()

    def build_game_dataset(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build a game-level dataset from NCAAB data."""
        if df is None:
            df = self.load_game_logs()

        if df is None or df.empty:
            return pd.DataFrame()

        # Same merge logic as NBADataLoader
        home = df[df["IS_HOME"] == 1].copy()
        away = df[df["IS_HOME"] == 0].copy()

        games = pd.merge(
            home, away,
            on="GAME_ID",
            suffixes=("_home", "_away"),
            how="inner",
        )

        # Standardize team name columns for FeatureEngineer
        home_team_col = "TEAM_NAME_home" if "TEAM_NAME_home" in games.columns else "home_team"
        away_team_col = "TEAM_NAME_away" if "TEAM_NAME_away" in games.columns else "away_team"

        # Ensure these exist for ELO and other name-dependent features
        if "TEAM_NAME_home" not in games.columns and "home_team" in games.columns:
            games["TEAM_NAME_home"] = games["home_team"]
        if "TEAM_NAME_away" not in games.columns and "away_team" in games.columns:
            games["TEAM_NAME_away"] = games["away_team"]

        # Basic game-level features
        games["GAME_DATE"] = games.get("GAME_DATE_home", games.get("date", pd.NaT))
        games["total_points"] = games["team_pts_home"] + games["team_pts_away"]
        games["point_diff"] = games["team_pts_home"] - games["team_pts_away"]

        # Sort by date
        games = games.sort_values("GAME_DATE").reset_index(drop=True)

        return games

    def compute_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rest days using TEAM_NAME as team identifier."""
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values(["TEAM_NAME", "GAME_DATE"])
        df["rest_days"] = df.groupby("TEAM_NAME")["GAME_DATE"].diff().dt.days
        df["rest_days"] = df["rest_days"].fillna(MAX_REST_DAYS).clip(0, MAX_REST_DAYS)
        df["is_back_to_back"] = (df["rest_days"] <= 1).astype(int)
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
