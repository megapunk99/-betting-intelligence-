"""
Feature engineering: transforms raw game data into predictive features.
All features should be calculable BEFORE the game starts (no lookahead bias).

v2.2 — Enhanced features:
  - EMA (exponential moving average) rolling stats — more weight to recent games
  - Trend slope calculator — detect if teams are improving/declining
  - Travel distance & fatigue — haversine distance, time zone shifts, cumulative fatigue
  - Consecutive road games counter
  - Enhanced fatigue model (fatigue score, 3in4 nights, rest squared)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from betting_intel.config import ROLLING_WINDOWS, MAX_REST_DAYS


# ── Constants for Advanced Features ───────────────────────────────────────

# NBA team arena coordinates (lat, lon) for travel distance calculation
NBA_TEAM_CENTERS: Dict[str, Tuple[float, float]] = {
    "Hawks": (33.755, -84.396), "Celtics": (42.366, -71.062),
    "Nets": (40.683, -73.975), "Hornets": (35.225, -80.839),
    "Bulls": (41.881, -87.674), "Cavaliers": (41.496, -81.688),
    "Mavericks": (32.790, -96.810), "Nuggets": (39.748, -105.007),
    "Pistons": (42.340, -83.056), "Warriors": (37.750, -122.203),
    "Rockets": (29.751, -95.362), "Pacers": (39.764, -86.156),
    "Clippers": (34.043, -118.267), "Lakers": (34.043, -118.267),
    "Grizzlies": (35.138, -90.051), "Heat": (25.781, -80.187),
    "Bucks": (43.043, -87.917), "Timberwolves": (44.979, -93.276),
    "Pelicans": (29.949, -90.082), "Knicks": (40.750, -73.993),
    "Thunder": (35.463, -97.515), "Magic": (28.539, -81.384),
    "76ers": (39.901, -75.172), "Suns": (33.445, -112.071),
    "Trail Blazers": (45.532, -122.667), "Kings": (38.580, -121.500),
    "Spurs": (29.427, -98.437), "Raptors": (43.643, -79.379),
    "Jazz": (40.768, -111.901), "Wizards": (38.898, -77.021),
}

# NBA team time zones (EST = -5, CST = -6, MST = -7, PST = -8)
NBA_TEAM_TZ: Dict[str, int] = {
    "Celtics": -5, "Nets": -5, "Knicks": -5, "76ers": -5, "Wizards": -5,
    "Hawks": -5, "Heat": -5, "Hornets": -5, "Magic": -5, "Raptors": -5,
    "Pistons": -5, "Pacers": -5, "Cavaliers": -5, "Bulls": -6,
    "Bucks": -6, "Timberwolves": -6, "Pelicans": -6, "Thunder": -6,
    "Mavericks": -6, "Rockets": -6, "Grizzlies": -6, "Spurs": -6,
    "Jazz": -7, "Nuggets": -7, "Suns": -7, "Trail Blazers": -8,
    "Kings": -8, "Warriors": -8, "Lakers": -8, "Clippers": -8,
}


class FeatureEngineer:
    """Creates features for predictive models from raw game data.

    v2.2 features include EMA rolling stats, trend slopes, travel distance,
    and enhanced fatigue modeling for more accurate predictions.
    """

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
            pts_col = f"team_pts_{team_prefix}"
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

            # ══════════════════════════════════════════════════════════
            #  v2.2 ENHANCED FEATURES
            # ══════════════════════════════════════════════════════════

            # ── EMA Rolling Features ─────────────────────────────────
            # Exponential Moving Average — more weight to recent games
            for w in self.rolling_windows:
                span = max(w, 2)  # span must be >= 2 for ewm
                df[f"ema_pts_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[pts_col]
                    .transform(lambda x, sp=span: (
                        x.ewm(span=sp, min_periods=1, adjust=False).mean().shift(1)
                    ))
                )
            for w in [5, 10]:
                span = max(w, 2)
                df[f"ema_pm_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[pm_col]
                    .transform(lambda x, sp=span: (
                        x.ewm(span=sp, min_periods=1, adjust=False).mean().shift(1)
                    ))
                )
                df[f"ema_margin_{w}g_{suffix}"] = df[f"ema_pm_{w}g_{suffix}"]

            # ── Trend Slope Features ─────────────────────────────────
            # Linear trend over recent games: positive = improving, negative = declining
            for w in [5, 10]:
                df[f"trend_pts_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[pts_col]
                    .transform(lambda x, win=w: self._compute_trend_slope(x, window=win))
                )
                df[f"trend_pm_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[pm_col]
                    .transform(lambda x, win=w: self._compute_trend_slope(x, window=win))
                )

            # ── Weighted/Decay Momentum (v2.2) ───────────────────────
            df[f"weighted_momentum_{suffix}"] = (
                df.groupby(team_id_col)[wl_num_col]
                .transform(lambda x: self._weighted_momentum(x, window=10))
            )

            # ── Scoring Volatility (pts_zscore) ──────────────────────
            df[f"pts_zscore_{suffix}"] = (
                df.groupby(team_id_col)[pts_col]
                .transform(lambda x: (
                    (x - x.rolling(10, min_periods=1).mean())
                    / x.rolling(10, min_periods=1).std().replace(0, 1)
                ).shift(1))
            )

            # ── Recent form composite: weighted win% + margin ────────
            weighted_wins = (
                df.groupby(team_id_col)[wl_num_col]
                .transform(lambda x: self._weighted_momentum(x, window=5))
            )
            margin_5g = df.get(f"avg_pm_5g_{suffix}", 0).fillna(0)
            df[f"form_score_{suffix}"] = weighted_wins + 0.02 * margin_5g

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

        # ── v2.1 Advanced Features ──────────────────────────────────────
        # Opponent-adjusted stats
        df = self.compute_opponent_adjusted_features(df)

        # Strength of schedule
        df = self.compute_strength_of_schedule(df)

        # Player-specific / team-style features
        df = self.compute_player_specific_features(df)

        # ── Travel & Fatigue Features (v2.2) ──────────────────────────
        df = self._add_travel_features(df)

        # ── Consecutive Road Games (v2.2) ──────────────────────────────
        df = self._add_consecutive_road_games(df)

        # ── Enhanced Fatigue Features (v2.2) ────────────────────────────
        df = self._add_enhanced_fatigue(df)

        # ── Clean Up ──────────────────────────────────────────────────
        df = df.drop(columns=["rest_home_key", "rest_away_key"], errors="ignore")

        # Drop intermediate WL string columns but keep WL_num for feature selection
        df = df.drop(columns=["WL_num_home", "WL_num_away"], errors="ignore")

        return df

    # ── Opponent-Adjusted Features (v2.1) ──────────────────────────────

    def compute_opponent_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute opponent-adjusted stats: how a team performs relative to
        their opponent's season averages.

        Key insight: scoring 110 pts against the best defense (allows 105)
        is more impressive than scoring 115 pts against the worst defense
        (allows 120). These features adjust raw stats by opponent strength.
        """
        df = df.copy()

        for team_prefix, suffix in [("home", "home"), ("away", "away")]:
            team_id_col = f"TEAM_ID_{suffix}"
            opp_pts_col = f"team_pts_{team_prefix}"
            opp_id_col = f"TEAM_ID_{'away' if suffix == 'home' else 'home'}"
            opp_pts_allowed_col = f"team_pts_{'away' if suffix == 'home' else 'home'}"
            opp_pm_col = f"team_plus_minus_{'away' if suffix == 'home' else 'home'}"

            # Opponent's average points scored (their offensive strength)
            df[f"opp_avg_pts_scored_{suffix}"] = (
                df.groupby(opp_id_col)[opp_pts_col]
                .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            )

            # Opponent's average points allowed (their defensive strength)
            df[f"opp_avg_pts_allowed_{suffix}"] = (
                df.groupby(opp_id_col)[opp_pts_allowed_col]
                .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            )

            # Opponent's average plus/minus (overall strength)
            df[f"opp_avg_pm_{suffix}"] = (
                df.groupby(opp_id_col)[opp_pm_col]
                .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            )

            # This team's scoring relative to opponent's defense
            # If team scores X and opponent usually allows Y, then X/Y > 1 means above-expectation
            team_pts_col = f"avg_pts_10g_{suffix}"
            if team_pts_col in df.columns:
                df[f"offense_vs_defense_{suffix}"] = (
                    df[team_pts_col] / df[f"opp_avg_pts_allowed_{suffix}"].clip(lower=1)
                )

            # Opponent's offense vs this team's defense
            opp_off_col = f"opp_avg_pts_scored_{suffix}"
            team_def_col = f"avg_pts_allowed_{suffix}"
            if team_def_col in df.columns:
                df[f"defense_vs_offense_{suffix}"] = (
                    df[opp_off_col] / df[team_def_col].clip(lower=1)
                )

        # Opponent quality differential (how much better is opponent than average)
        for col in ["opp_avg_pm_home", "opp_avg_pm_away"]:
            if col in df.columns:
                df[f"adj_{col}"] = df[col]  # Already shifted, no further adjustment needed

        return df

    def compute_strength_of_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute strength of schedule (SOS) features.

        SOS = weighted average of opponent quality over recent games.
        Higher SOS = tougher schedule = more informative for edge detection.
        """
        df = df.copy()

        for team_prefix, suffix in [("home", "home"), ("away", "away")]:
            team_id_col = f"TEAM_ID_{suffix}"
            opp_id_col = f"TEAM_ID_{'away' if suffix == 'home' else 'home'}"
            opp_pm_col = f"team_plus_minus_{'away' if suffix == 'home' else 'home'}"

            # Get opponent's trailing margin (shifted so we don't leak)
            opp_trailing_margin = (
                df.groupby(opp_id_col)[opp_pm_col]
                .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            )
            df[f"opp_trailing_margin_{suffix}"] = opp_trailing_margin

            # For each team, average the quality of their recent opponents
            # This creates a rolling average of opponent strength
            df[f"sos_{suffix}"] = (
                df.groupby(team_id_col)[f"opp_trailing_margin_{suffix}"]
                .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            )

            # Recent SOS trend (last 5 vs last 10)
            sos_5 = (
                df.groupby(team_id_col)[f"opp_trailing_margin_{suffix}"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )
            sos_10 = df.get(f"sos_{suffix}", 0)
            df[f"sos_trend_{suffix}"] = sos_5 - sos_10.fillna(0)

        return df

    def compute_player_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute player/position-specific features from team-level data.

        Safely handles missing columns — skips features when source columns
        aren't available (e.g. in test fixtures with minimal schema).

        Features:
        - 3-point attempt rate (proxy for spacing / modern offense)
        - Free throw rate (proxy for aggressive play / foul drawing)
        - Assist ratio (proxy for ball movement / system offense)
        - True shooting percentage
        - Rebound rate
        """
        df = df.copy()

        for team_prefix, suffix in [("home", "home"), ("away", "away")]:
            team_id_col = f"TEAM_ID_{suffix}"
            fga_col = f"team_fga_{team_prefix}"
            fg3a_col = f"team_fg3a_{team_prefix}"
            fta_col = f"team_fta_{team_prefix}"
            ast_col = f"team_ast_{team_prefix}"
            pts_col = f"team_pts_{team_prefix}"
            reb_col = f"team_reb_{team_prefix}"
            tov_col = f"team_tov_{team_prefix}"

            # Skip if required base columns are missing
            if fga_col not in df.columns or pts_col not in df.columns:
                continue

            # ── 3-point attempt rate ────────────────────────────────
            if fg3a_col in df.columns and fga_col in df.columns:
                df[f"three_pt_rate_{suffix}"] = (
                    df[fg3a_col] / df[fga_col].clip(lower=1)
                )
                df[f"three_pt_rate_10g_{suffix}"] = (
                    df.groupby(team_id_col)[f"three_pt_rate_{suffix}"]
                    .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
                )

            # ── Free throw rate ─────────────────────────────────────
            if fta_col in df.columns and fga_col in df.columns:
                df[f"ft_rate_{suffix}"] = (
                    df[fta_col] / df[fga_col].clip(lower=1)
                )
                df[f"ft_rate_10g_{suffix}"] = (
                    df.groupby(team_id_col)[f"ft_rate_{suffix}"]
                    .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
                )

            # ── Assist ratio ────────────────────────────────────────
            if all(c in df.columns for c in [ast_col, fga_col, fta_col, tov_col]):
                df[f"ast_ratio_{suffix}"] = (
                    df[ast_col] / (df[fga_col] + df[fta_col] + df[tov_col]).clip(lower=1)
                )
                df[f"ast_ratio_10g_{suffix}"] = (
                    df.groupby(team_id_col)[f"ast_ratio_{suffix}"]
                    .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
                )

            # ── True shooting percentage ────────────────────────────
            if all(c in df.columns for c in [pts_col, fga_col, fta_col]):
                df[f"ts_pct_{suffix}"] = (
                    df[pts_col] / (2 * (df[fga_col] + 0.44 * df[fta_col])).clip(lower=1)
                )
                df[f"ts_pct_10g_{suffix}"] = (
                    df.groupby(team_id_col)[f"ts_pct_{suffix}"]
                    .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
                )

            # ── Rebound rate ────────────────────────────────────────
            opp_reb_col = f"team_reb_{'away' if team_prefix == 'home' else 'home'}"
            if reb_col in df.columns and opp_reb_col in df.columns:
                df[f"reb_pct_{suffix}"] = (
                    df[reb_col] / (df[reb_col] + df[opp_reb_col]).clip(lower=1)
                )
                df[f"reb_pct_10g_{suffix}"] = (
                    df.groupby(team_id_col)[f"reb_pct_{suffix}"]
                    .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
                )

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

    def _compute_trend_slope(self, values: pd.Series, window: int = 5) -> pd.Series:
        """
        Compute the linear trend slope over a rolling window.
        Positive = improving (increasing values), Negative = declining.

        Uses a simplified formula: slope = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
        where x = position in window, y = value
        """
        n = len(values)
        result = np.full(n, np.nan)

        for i in range(n):
            if i < window:
                result[i] = 0.0
                continue

            window_vals = values.iloc[max(0, i - window):i].values
            if len(window_vals) < 2:
                result[i] = 0.0
                continue

            x = np.arange(len(window_vals))
            y = window_vals
            x_mean = x.mean()
            y_mean = y.mean()

            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            slope = numerator / denominator if denominator > 0 else 0.0
            result[i] = slope

        # No shift needed — the loop already uses [i-window:i) (current row excluded)
        return pd.Series(result, index=values.index).fillna(0.0)

    def _weighted_momentum(self, wl_numeric: pd.Series, window: int = 10) -> pd.Series:
        """Compute exponentially weighted recent performance.
        More recent games get higher weight. Returns weighted average of wins (0-1).
        """
        weights = np.exp(np.linspace(-1, 0, window))
        weights /= weights.sum()

        def rolling_weighted_mean(series):
            if len(series) < window:
                return series.mean() if len(series) > 0 else 0.5
            return np.sum(series.tail(window).values * weights)

        result = wl_numeric.rolling(window, min_periods=1).apply(
            rolling_weighted_mean, raw=False
        )
        return result.shift(1).fillna(0.5)

    @staticmethod
    def _haversine(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Haversine distance between two (lat, lon) points in miles."""
        R = 3959  # Earth radius in miles
        lat1, lon1 = np.radians(loc1)
        lat2, lon2 = np.radians(loc2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return R * c

    def _add_travel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add travel distance and time zone features (v2.2).

        Features:
        - travel_distance: miles between home and away arenas
        - travel_distance_norm: normalized 0-1
        - tz_diff: absolute time zone difference
        - cum_travel_home/away: cumulative travel over last 3 games
        """
        df = df.copy()

        # Team name columns
        df["home_team_name"] = df["TEAM_NAME_home"].astype(str).str.strip()
        df["away_team_name"] = df["TEAM_NAME_away"].astype(str).str.strip()

        def get_travel_distance(row):
            home_team = row.get("home_team_name", "")
            away_team = row.get("away_team_name", "")
            home_loc = NBA_TEAM_CENTERS.get(home_team)
            away_loc = NBA_TEAM_CENTERS.get(away_team)
            if home_loc and away_loc:
                return self._haversine(home_loc, away_loc)
            return 500  # default fallback

        df["travel_distance"] = df.apply(get_travel_distance, axis=1)
        df["travel_distance_norm"] = df["travel_distance"] / 3000.0

        # Time zone difference
        df["home_tz"] = df["home_team_name"].map(NBA_TEAM_TZ).fillna(-5)
        df["away_tz"] = df["away_team_name"].map(NBA_TEAM_TZ).fillna(-5)
        df["tz_diff"] = abs(df["home_tz"] - df["away_tz"])

        # Cumulative travel fatigue over last 3 games for each team
        df["cum_travel_home"] = (
            df.groupby("TEAM_ID_home")["travel_distance"]
            .transform(lambda x: x.rolling(3, min_periods=1).sum().shift(1))
        )
        df["cum_travel_away"] = (
            df.groupby("TEAM_ID_away")["travel_distance"]
            .transform(lambda x: x.rolling(3, min_periods=1).sum().shift(1))
        )
        df["cum_travel_diff"] = df["cum_travel_home"].fillna(0) - df["cum_travel_away"].fillna(0)

        return df

    def _add_consecutive_road_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Count consecutive road games for each team (v2.2).

        Teams on extended road trips tend to underperform, especially
        towards the end of long trips.
        """
        df = df.copy()

        for suffix in ["away"]:
            team_id_col = f"TEAM_ID_{suffix}"

            # Consecutive road games counter for away team
            df[f"consec_road_{suffix}"] = (
                df.groupby(team_id_col)[f"WL_num_{suffix}"]
                .transform(lambda x: self._compute_consecutive_road(x.shift(1)))
            )

        df["road_trip_length"] = df["consec_road_away"].fillna(0).astype(int)
        df["long_road_trip"] = (df["road_trip_length"] >= 4).astype(int)

        return df

    def _compute_consecutive_road(self, wl_numeric: pd.Series) -> pd.Series:
        """Count consecutive games played (proxy for road games when used for away team)."""
        result = np.zeros(len(wl_numeric))
        count = 0
        for i in range(len(wl_numeric)):
            val = wl_numeric.iloc[i] if hasattr(wl_numeric, 'iloc') else wl_numeric[i]
            if not pd.isna(val):
                count += 1
            else:
                count = 0
            result[i] = count
        return pd.Series(result, index=wl_numeric.index).fillna(0)

    def _add_enhanced_fatigue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced fatigue features (v2.2).

        Features:
        - fatigue_home/away: non-linear fatigue score (1/(rest+0.5), capped at 2)
        - fatigue_diff: home minus away fatigue
        - rest_home_sq/away_sq: squared rest days (non-linear effect)
        - rest_3in4_home/away: 3 games in 4 nights flag
        - both_b2b: both teams on back-to-back
        - rest_interaction: rest_advantage * home_court
        - travel_rest_interaction: travel_distance * rest_advantage
        """
        df = df.copy()

        # Fatigue score: lower rest = exponentially more fatigue
        df["fatigue_home"] = np.clip(1.0 / (df["rest_home_days"] + 0.5), 0, 2)
        df["fatigue_away"] = np.clip(1.0 / (df["rest_away_days"] + 0.5), 0, 2)
        df["fatigue_diff"] = df["fatigue_home"] - df["fatigue_away"]

        # Non-linear rest effects
        df["rest_home_sq"] = df["rest_home_days"] ** 2
        df["rest_away_sq"] = df["rest_away_days"] ** 2

        # 3 games in 4 nights
        df["rest_3in4_home"] = (df["rest_home_days"] <= 1).astype(int)
        df["rest_3in4_away"] = (df["rest_away_days"] <= 1).astype(int)

        # Both teams on b2b
        df["both_b2b"] = (
            (df["rest_home_days"] == 0) & (df["rest_away_days"] == 0)
        ).astype(int)

        # Interaction features
        df["rest_adv_sq"] = df["rest_advantage"] ** 2
        df["fatigue_rest_interact"] = df["fatigue_diff"] * df["rest_advantage"]

        return df

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
            "home_team_name", "away_team_name",
            # Post-game stats not available pre-game:
            "MIN_home", "MIN_away",
            # ═══════════════════════════════════════════════════════════
            # RAW PER-GAME TEAM STATS — leak the target!
            # These are the actual points/stats the team scored that game.
            # Including them lets the model perfectly reconstruct
            # total_points = team_pts_home + team_pts_away, giving R² ≈ 1.0.
            # Only their LAGGED rolling averages (avg_pts_*, avg_pm_*, etc.)
            # should be available to the model (already computed above).
            # ═══════════════════════════════════════════════════════════
            "team_pts_home", "team_pts_away",
            "team_fgm_home", "team_fgm_away",
            "team_fga_home", "team_fga_away",
            "team_fg_pct_home", "team_fg_pct_away",
            "team_fg3m_home", "team_fg3m_away",
            "team_fg3a_home", "team_fg3a_away",
            "team_fg3_pct_home", "team_fg3_pct_away",
            "team_ftm_home", "team_ftm_away",
            "team_fta_home", "team_fta_away",
            "team_ft_pct_home", "team_ft_pct_away",
            "team_oreb_home", "team_oreb_away",
            "team_dreb_home", "team_dreb_away",
            "team_reb_home", "team_reb_away",
            "team_ast_home", "team_ast_away",
            "team_stl_home", "team_stl_away",
            "team_blk_home", "team_blk_away",
            "team_tov_home", "team_tov_away",
            "team_pf_home", "team_pf_away",
            "team_plus_minus_home", "team_plus_minus_away",
            # Home/away indicators from game dataset — not predictive
            "IS_HOME_home", "IS_HOME_away",
            "OPPONENT_home", "OPPONENT_away",
            # Market-line proxy columns — NOT features (prevent leakage):
            "market_line_baseline",
            "market_line_pace_adj",
            "trailing_avg_total_10g",
            # Intermediate calculation columns (not features themselves):
            "three_pt_rate_home", "three_pt_rate_away",
            "ft_rate_home", "ft_rate_away",
            "ast_ratio_home", "ast_ratio_away",
            "ts_pct_home", "ts_pct_away",
            "reb_pct_home", "reb_pct_away",
            "home_tz", "away_tz",  # Intermediate: use tz_diff instead
        }
        return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
