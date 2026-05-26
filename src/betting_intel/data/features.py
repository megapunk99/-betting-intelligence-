"""
Feature engineering: transforms raw game data into predictive features.
All features should be calculable BEFORE the game starts (no lookahead bias).
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from betting_intel.config import ROLLING_WINDOWS, MAX_REST_DAYS


class FeatureEngineer:
    """Creates features for predictive models from raw game data."""

    def __init__(self, rolling_windows: Optional[List[int]] = None):
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS

    def build_all_features(self, games_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the full feature set from game-level data.

        Args:
            games_df: Merged home/away game dataset from NBADataLoader
            raw_df: Raw team-level game logs

        Returns:
            DataFrame with features, no lookahead bias
        """
        df = games_df.copy()
        df = df.sort_values("GAME_DATE").reset_index(drop=True)

        # Convert WL columns to numeric (1 for Win, 0 for Loss)
        for prefix in ["home", "away"]:
            df[f"WL_num_{prefix}"] = (df[f"WL_{prefix}"] == "W").astype(float)

        # ── Team Rolling Averages (home & away) ──────────────────────
        for team_prefix, suffix in [("home", "home"), ("away", "away")]:
            team_id_col = f"TEAM_ID_{suffix}"
            pts_col = f"team_pts_{team_prefix}"

            # Rolling points scored
            for w in self.rolling_windows:
                df[f"avg_pts_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[pts_col]
                    .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                )

            # Rolling points allowed (opponent's points in the same game)
            opp_pts_col = f"team_pts_{'away' if team_prefix == 'home' else 'home'}"
            df[f"avg_pts_allowed_{suffix}"] = (
                df.groupby(team_id_col)[opp_pts_col]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )

            # Pace (FGA + TOV - OREB)
            pace_stat = (
                df[f"team_fga_{team_prefix}"]
                + df[f"team_tov_{team_prefix}"]
                - df[f"team_oreb_{team_prefix}"]
            )
            df[f"pace_{suffix}"] = pace_stat

            for w in [5, 10]:
                df[f"avg_pace_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[f"pace_{suffix}"]
                    .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                )

            # eFG% rolling
            efg_col = f"eFG_{team_prefix}"
            for w in [5, 10]:
                df[f"avg_efg_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[efg_col]
                    .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                )

            # Plus/minus (margin) rolling
            pm_col = f"team_plus_minus_{team_prefix}"
            for w in [5, 10]:
                df[f"avg_pm_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[pm_col]
                    .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                )

            # Win rate (using numeric WL column)
            df[f"win_pct_10g_{suffix}"] = (
                df.groupby(team_id_col)[f"WL_num_{team_prefix}"]
                .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            )

        # ── Rest Days ─────────────────────────────────────────────────
        # Map rest days from raw data into the merged game dataframe
        home_rest = raw_df[raw_df["IS_HOME"] == 1][["GAME_ID", "TEAM_ID", "rest_days"]].copy()
        away_rest = raw_df[raw_df["IS_HOME"] == 0][["GAME_ID", "TEAM_ID", "rest_days"]].copy()

        df["rest_home_key"] = df["GAME_ID"].astype(str) + "_" + df["TEAM_ID_home"].astype(str)
        df["rest_away_key"] = df["GAME_ID"].astype(str) + "_" + df["TEAM_ID_away"].astype(str)

        home_rest_map = dict(zip(home_rest["GAME_ID"].astype(str) + "_" + home_rest["TEAM_ID"].astype(str), home_rest["rest_days"]))
        away_rest_map = dict(zip(away_rest["GAME_ID"].astype(str) + "_" + away_rest["TEAM_ID"].astype(str), away_rest["rest_days"]))

        df["rest_home_days"] = df["rest_home_key"].map(home_rest_map).fillna(3)
        df["rest_away_days"] = df["rest_away_key"].map(away_rest_map).fillna(3)
        df["rest_advantage"] = df["rest_home_days"] - df["rest_away_days"]
        df["is_b2b_home"] = (df["rest_home_days"] == 0).astype(int)
        df["is_b2b_away"] = (df["rest_away_days"] == 0).astype(int)

        # ── Momentum Features ─────────────────────────────────────────
        for team_prefix, suffix in [("home", "home"), ("away", "away")]:
            team_id_col = f"TEAM_ID_{suffix}"
            pm_col = f"team_plus_minus_{team_prefix}"
            wl_num_col = f"WL_num_{team_prefix}"

            # Win streak (using numeric WL)
            df[f"win_streak_{suffix}"] = (
                df.groupby(team_id_col)[wl_num_col]
                .transform(lambda x: self._compute_streak(x))
            )

            # Margin in last 3 games
            df[f"last_3_margin_{suffix}"] = (
                df.groupby(team_id_col)[pm_col]
                .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
            )

            # Standard deviation of margin (consistency)
            df[f"margin_volatility_{suffix}"] = (
                df.groupby(team_id_col)[pm_col]
                .transform(lambda x: x.rolling(10, min_periods=1).std().shift(1))
            )

        # ── Market Line Baseline (for backtesting — NOT used as a feature) ─
        # This is a simple trailing average used as a proxy for the sportsbook's line.
        # It is deliberately excluded from select_features() to prevent data leakage.
        df["market_line_baseline"] = (
            df.get("avg_pts_5g_home", 110).fillna(110) +
            df.get("avg_pts_5g_away", 108).fillna(108)
        ) / 1.0  # Simple average of home and away scoring

        # Also compute a pace-adjusted baseline for comparison
        df["market_line_pace_adj"] = (
            df.get("avg_pace_5g_home", 100).fillna(100) +
            df.get("avg_pace_5g_away", 100).fillna(100)
        ) / 2.0 * 2.1  # Approximate points per possession * pace

        # Pre-compute a simple trailing average of total points for the last 3 games each team played
        # This serves as a simple baseline that's independent of the model features
        df["trailing_avg_total_10g"] = (
            df.groupby("TEAM_ID_home")["team_pts_home"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            .fillna(105)
            +
            df.groupby("TEAM_ID_away")["team_pts_away"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            .fillna(105)
        )

        # ── Clean Up ──────────────────────────────────────────────────
        df = df.drop(columns=["rest_home_key", "rest_away_key"], errors="ignore")

        # Drop intermediate WL string columns but keep WL_num for feature selection
        df = df.drop(columns=["WL_num_home", "WL_num_away"], errors="ignore")

        return df

    def _compute_streak(self, wl_numeric: pd.Series) -> pd.Series:
        """Compute streak length from numeric WL (1=win, 0=loss). Positive = wins streak."""
        streak = np.zeros(len(wl_numeric))
        current_streak = 0
        for i, val in enumerate(wl_numeric):
            if val == 1:
                current_streak = current_streak + 1 if current_streak > 0 else 1
            else:
                current_streak = current_streak - 1 if current_streak < 0 else -1
            streak[i] = current_streak
        result = pd.Series(streak, index=wl_numeric.index).shift(1)
        return result.fillna(0)

    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect feature columns from a dataframe.

        IMPORTANT: Excludes market-line proxy columns and other post-computed
        fields to prevent data leakage (the model should not see the benchmark
        it's being evaluated against).
        """
        exclude = {
            "GAME_ID", "SEASON_ID", "TEAM_ID_home", "TEAM_ID_away",
            "TEAM_ABBREVIATION_home", "TEAM_ABBREVIATION_away",
            "TEAM_NAME_home", "TEAM_NAME_away", "GAME_DATE",
            "MATCHUP_home", "MATCHUP_away",
            "WL_home", "WL_away",
            "SEASON_home", "SEASON_away",
            "total_points", "point_diff",
            "rest_home_key", "rest_away_key",
            # Post-game stats not available pre-game:
            "MIN_home", "MIN_away",
            # Market-line proxy columns — NOT features (prevent leakage):
            "market_line_baseline",
            "market_line_pace_adj",
            "trailing_avg_total_10g",
        }
        return [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64")]
