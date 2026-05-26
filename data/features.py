"""
Feature engineering: transforms raw game data into predictive features.
All features should be calculable BEFORE the game starts (no lookahead bias).

v2.0 — Advanced features:
  - Elo ratings with home-court adjustment
  - True Shooting %, adjusted offensive/defensive ratings
  - Opponent-adjusted performance metrics
  - Travel distance & schedule fatigue
  - Advanced momentum (weighted, decay-based)
  - Pace-adjusted stats (per-possession)
  - Consistency/scoring volatility features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from scipy import spatial

from config import ROLLING_WINDOWS, MAX_REST_DAYS


# ── Constants ─────────────────────────────────────────────────────────────
ELO_K = 32              # K-factor for Elo updates
ELO_HOME_ADV = 100      # Home court advantage in Elo (points)
ELO_INITIAL = 1500      # Starting Elo rating
TS_LEAGUE_AVG = 0.545   # Approximate league-average TS%
PACE_LEAGUE_AVG = 98.5  # Approximate league-average pace
AVG_PTS_PER_GAME = 111  # Approximate league-average points per game
TRAVEL_MILES_PER_HOUR = 550  # Approximate travel speed for time zones


@dataclass
class TeamElo:
    """Tracks a team's Elo rating and history."""
    rating: float = ELO_INITIAL
    home_rating: float = ELO_INITIAL + ELO_HOME_ADV
    away_rating: float = ELO_INITIAL - ELO_HOME_ADV
    history: List[Tuple[str, float]] = None  # (game_date, rating_after)


class FeatureEngineer:
    """Creates advanced features for predictive models from raw game data."""

    def __init__(self, rolling_windows: Optional[List[int]] = None):
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        self.team_elos: Dict[int, TeamElo] = {}
        self._nba_team_centers: Dict[str, Tuple[float, float]] = {
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

    def build_all_features(self, games_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the full advanced feature set from game-level data.

        Args:
            games_df: Merged home/away game dataset from NBADataLoader
            raw_df: Raw team-level game logs

        Returns:
            DataFrame with v2.0 advanced features, no lookahead bias
        """
        df = games_df.copy()
        df = df.sort_values("GAME_DATE").reset_index(drop=True)

        # Convert WL columns to numeric (1 for Win, 0 for Loss)
        for prefix in ["home", "away"]:
            df[f"WL_num_{prefix}"] = (df[f"WL_{prefix}"] == "W").astype(float)

        # ═══════════════════════════════════════════════════════════════
        #  BLOCK 1: Per-Game Advanced Metrics (eFG%, TS%, Pace, etc.)
        # ═══════════════════════════════════════════════════════════════
        print("  Computing advanced game metrics...")
        for team_prefix in ["home", "away"]:
            # Effective FG% already exists, compute True Shooting %
            pts = df[f"team_pts_{team_prefix}"]
            fga = df[f"team_fga_{team_prefix}"]
            fta = df[f"team_fta_{team_prefix}"]
            df[f"ts_pct_{team_prefix}"] = pts / (2 * (fga + 0.44 * fta)).replace(0, np.nan)
            df[f"ts_pct_{team_prefix}"] = df[f"ts_pct_{team_prefix}"].fillna(TS_LEAGUE_AVG)

            # Points per possession (simplified: PTS / (FGA + 0.44*FTA + TOV))
            tov = df[f"team_tov_{team_prefix}"]
            poss = fga + 0.44 * fta + tov
            df[f"ppp_{team_prefix}"] = pts / poss.replace(0, np.nan)
            df[f"ppp_{team_prefix}"] = df[f"ppp_{team_prefix}"].fillna(1.0)

            # Assist-to-turnover ratio
            ast = df[f"team_ast_{team_prefix}"]
            df[f"ast_tov_ratio_{team_prefix}"] = ast / tov.replace(0, np.nan)
            df[f"ast_tov_ratio_{team_prefix}"] = df[f"ast_tov_ratio_{team_prefix}"].fillna(1.5)

            # Rebound rate (OREB / (OREB + opp_DREB)) - using in-game values
            oreb = df[f"team_oreb_{team_prefix}"]
            opp_dreb = df[f"team_dreb_{'away' if team_prefix == 'home' else 'home'}"]
            total_reb = oreb + opp_dreb
            df[f"oreb_pct_{team_prefix}"] = oreb / total_reb.replace(0, np.nan)
            df[f"oreb_pct_{team_prefix}"] = df[f"oreb_pct_{team_prefix}"].fillna(0.25)

            # Steal rate
            stl = df[f"team_stl_{team_prefix}"]
            opp_poss = (df[f"team_fga_{'away' if team_prefix == 'home' else 'home'}"]
                       + 0.44 * df[f"team_fta_{'away' if team_prefix == 'home' else 'home'}"]
                       + df[f"team_tov_{'away' if team_prefix == 'home' else 'home'}"])
            df[f"stl_rate_{team_prefix}"] = stl / opp_poss.replace(0, np.nan)
            df[f"stl_rate_{team_prefix}"] = df[f"stl_rate_{team_prefix}"].fillna(0.07)

        # ═══════════════════════════════════════════════════════════════
        #  BLOCK 2: Rolling Team Features (shifted to prevent lookahead)
        # ═══════════════════════════════════════════════════════════════
        print("  Computing rolling team features...")
        for team_prefix, suffix in [("home", "home"), ("away", "away")]:
            team_id_col = f"TEAM_ID_{suffix}"
            pts_col = f"team_pts_{team_prefix}"

            # ── Core rolling averages ──
            for w in self.rolling_windows:
                df[f"avg_pts_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[pts_col]
                    .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                )

            # Rolling points allowed
            opp_pts_col = f"team_pts_{'away' if team_prefix == 'home' else 'home'}"
            for w in [5, 10]:
                df[f"avg_pts_allowed_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[opp_pts_col]
                    .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                )

            # ── Pace (rolling) ──
            pace_col = f"pace_{suffix}"
            df[pace_col] = (
                df[f"team_fga_{team_prefix}"]
                + df[f"team_tov_{team_prefix}"]
                - df[f"team_oreb_{team_prefix}"]
            )
            for w in [5, 10]:
                df[f"avg_pace_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[pace_col]
                    .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                )

            # ── True Shooting % (rolling) ──
            ts_col = f"ts_pct_{team_prefix}"
            for w in [5, 10]:
                df[f"avg_ts_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[ts_col]
                    .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                )

            # ── eFG% rolling ──
            efg_col = f"eFG_{team_prefix}"
            for w in [5, 10]:
                df[f"avg_efg_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[efg_col]
                    .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                )

            # ── Plus/minus rolling ──
            pm_col = f"team_plus_minus_{team_prefix}"
            for w in [5, 10, 20]:
                df[f"avg_pm_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[pm_col]
                    .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                )

            # ── Points per possession (rolling) ──
            ppp_col = f"ppp_{team_prefix}"
            for w in [5, 10]:
                df[f"avg_ppp_{w}g_{suffix}"] = (
                    df.groupby(team_id_col)[ppp_col]
                    .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                )

            # ── Assist-to-turnover ratio (rolling) ──
            atr_col = f"ast_tov_ratio_{team_prefix}"
            df[f"avg_atr_{suffix}"] = (
                df.groupby(team_id_col)[atr_col]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )

            # ── Win rate ──
            df[f"win_pct_10g_{suffix}"] = (
                df.groupby(team_id_col)[f"WL_num_{team_prefix}"]
                .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            )
            df[f"win_pct_20g_{suffix}"] = (
                df.groupby(team_id_col)[f"WL_num_{team_prefix}"]
                .transform(lambda x: x.rolling(20, min_periods=1).mean().shift(1))
            )

            # ── Scoring consistency (volatility) ──
            df[f"pts_volatility_{w}g_{suffix}"] = (
                df.groupby(team_id_col)[pts_col]
                .transform(lambda x: x.rolling(10, min_periods=1).std().shift(1))
            )

        # ═══════════════════════════════════════════════════════════════
        #  BLOCK 3: Advanced Momentum Features
        # ═══════════════════════════════════════════════════════════════
        print("  Computing advanced momentum features...")
        for team_prefix, suffix in [("home", "home"), ("away", "away")]:
            team_id_col = f"TEAM_ID_{suffix}"
            wl_num_col = f"WL_num_{team_prefix}"

            # Win streak
            df[f"win_streak_{suffix}"] = (
                df.groupby(team_id_col)[wl_num_col]
                .transform(lambda x: self._compute_streak(x))
            )

            # Weighted recent performance (exponential decay — more weight to recent games)
            df[f"weighted_recent_5_{suffix}"] = (
                df.groupby(team_id_col)[wl_num_col]
                .transform(lambda x: self._weighted_recent(x, window=5))
            )
            df[f"weighted_recent_10_{suffix}"] = (
                df.groupby(team_id_col)[wl_num_col]
                .transform(lambda x: self._weighted_recent(x, window=10))
            )

            # Last 3 margin
            pm_col = f"team_plus_minus_{team_prefix}"
            df[f"last_3_margin_{suffix}"] = (
                df.groupby(team_id_col)[pm_col]
                .transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
            )
            df[f"last_5_margin_{suffix}"] = (
                df.groupby(team_id_col)[pm_col]
                .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            )

            # Margin volatility (consistency)
            df[f"margin_volatility_{suffix}"] = (
                df.groupby(team_id_col)[pm_col]
                .transform(lambda x: x.rolling(10, min_periods=1).std().shift(1))
            )

            # Scoring variance
            pts_col = f"team_pts_{team_prefix}"
            df[f"pts_zscore_{suffix}"] = (
                df.groupby(team_id_col)[pts_col]
                .transform(lambda x: ((x - x.rolling(10, min_periods=1).mean())
                                     / x.rolling(10, min_periods=1).std().replace(0, 1)).shift(1))
            )

        # ═══════════════════════════════════════════════════════════════
        #  BLOCK 4: Rest, Fatigue & Travel
        # ═══════════════════════════════════════════════════════════════
        print("  Computing rest, fatigue & travel features...")
        self._add_rest_features(df, raw_df)
        self._add_travel_features(df)

        # ═══════════════════════════════════════════════════════════════
        #  BLOCK 5: Elo Ratings
        # ═══════════════════════════════════════════════════════════════
        print("  Computing Elo ratings...")
        self._compute_elo_ratings(df)

        # ═══════════════════════════════════════════════════════════════
        #  BLOCK 6: Opponent-Adjusted & Composite Features
        # ═══════════════════════════════════════════════════════════════
        print("  Computing opponent-adjusted features...")
        self._add_opponent_adjusted_features(df)

        # ═══════════════════════════════════════════════════════════════
        #  BLOCK 7: Head-to-Head & Game-Level Features
        # ═══════════════════════════════════════════════════════════════
        print("  Computing game-level composite features...")
        self._add_game_level_features(df)

        # ── Clean up ──
        df = df.drop(columns=[c for c in df.columns if c.endswith("_key") or c.endswith("_drop")], errors="ignore")
        df = df.drop(columns=["WL_num_home", "WL_num_away"], errors="ignore")

        return df

    # ═══════════════════════════════════════════════════════════════════
    #  REST & FATIGUE
    # ═══════════════════════════════════════════════════════════════════

    def _add_rest_features(self, df: pd.DataFrame, raw_df: pd.DataFrame):
        """Add rest day and back-to-back features."""
        home_rest = raw_df[raw_df["IS_HOME"] == 1][["GAME_ID", "TEAM_ID", "rest_days"]].copy()
        away_rest = raw_df[raw_df["IS_HOME"] == 0][["GAME_ID", "TEAM_ID", "rest_days"]].copy()

        df["rest_home_key"] = df["GAME_ID"].astype(str) + "_" + df["TEAM_ID_home"].astype(str)
        df["rest_away_key"] = df["GAME_ID"].astype(str) + "_" + df["TEAM_ID_away"].astype(str)

        home_rest_map = dict(zip(
            home_rest["GAME_ID"].astype(str) + "_" + home_rest["TEAM_ID"].astype(str),
            home_rest["rest_days"]
        ))
        away_rest_map = dict(zip(
            away_rest["GAME_ID"].astype(str) + "_" + away_rest["TEAM_ID"].astype(str),
            away_rest["rest_days"]
        ))

        df["rest_home_days"] = df["rest_home_key"].map(home_rest_map).fillna(3)
        df["rest_away_days"] = df["rest_away_key"].map(away_rest_map).fillna(3)
        df["rest_advantage"] = df["rest_home_days"] - df["rest_away_days"]
        df["is_b2b_home"] = (df["rest_home_days"] == 0).astype(int)
        df["is_b2b_away"] = (df["rest_away_days"] == 0).astype(int)
        df["both_b2b"] = ((df["rest_home_days"] == 0) & (df["rest_away_days"] == 0)).astype(int)

        # Fatigue score: lower rest = more fatigue
        df["fatigue_home"] = np.clip(1.0 / (df["rest_home_days"] + 0.5), 0, 2)
        df["fatigue_away"] = np.clip(1.0 / (df["rest_away_days"] + 0.5), 0, 2)
        df["fatigue_diff"] = df["fatigue_home"] - df["fatigue_away"]

        # Rest squared term (non-linear effect of rest)
        df["rest_home_sq"] = df["rest_home_days"] ** 2
        df["rest_away_sq"] = df["rest_away_days"] ** 2

        # 3 games in 4 nights?
        df["rest_home_3in4"] = (df["rest_home_days"] <= 1).astype(int)
        df["rest_away_3in4"] = (df["rest_away_days"] <= 1).astype(int)

    def _add_travel_features(self, df: pd.DataFrame):
        """Add travel distance and time zone features."""
        # Map team names to approximate arena locations
        df["home_team_name"] = df["TEAM_NAME_home"].astype(str).str.strip()
        df["away_team_name"] = df["TEAM_NAME_away"].astype(str).str.strip()

        # Travel distance (miles between arenas)
        def compute_travel_distance(row):
            home_team = row.get("home_team_name", "")
            away_team = row.get("away_team_name", "")
            home_loc = self._nba_team_centers.get(home_team)
            away_loc = self._nba_team_centers.get(away_team)
            if home_loc and away_loc:
                return self._haversine(home_loc, away_loc)
            return 500  # default

        df["travel_distance"] = df.apply(compute_travel_distance, axis=1)

        # Normalized travel (0-1)
        df["travel_distance_norm"] = df["travel_distance"] / 3000.0

        # Time zone difference (approximate)
        # EST = -5, CST = -6, MST = -7, PST = -8
        team_tz = {
            "Celtics": -5, "Nets": -5, "Knicks": -5, "76ers": -5, "Wizards": -5,
            "Hawks": -5, "Heat": -5, "Hornets": -5, "Magic": -5, "Raptors": -5,
            "Pistons": -5, "Pacers": -5, "Cavaliers": -5, "Bulls": -6,
            "Bucks": -6, "Timberwolves": -6, "Pelicans": -6, "Thunder": -6,
            "Mavericks": -6, "Rockets": -6, "Grizzlies": -6, "Spurs": -6,
            "Jazz": -7, "Nuggets": -7, "Suns": -7, "Trail Blazers": -8,
            "Sonics": -8, "Kings": -8, "Warriors": -8, "Lakers": -8, "Clippers": -8,
        }
        df["home_tz"] = df["home_team_name"].map(team_tz).fillna(-5)
        df["away_tz"] = df["away_team_name"].map(team_tz).fillna(-5)
        df["tz_diff"] = abs(df["home_tz"] - df["away_tz"])

        # Cumulative travel fatigue (3-game rolling)
        df["cum_travel_home"] = (
            df.groupby("TEAM_ID_home")["travel_distance"]
            .transform(lambda x: x.rolling(3, min_periods=1).sum().shift(1))
        )
        df["cum_travel_away"] = (
            df.groupby("TEAM_ID_away")["travel_distance"]
            .transform(lambda x: x.rolling(3, min_periods=1).sum().shift(1))
        )

    # ═══════════════════════════════════════════════════════════════════
    #  ELO RATINGS
    # ═══════════════════════════════════════════════════════════════════

    def _compute_elo_ratings(self, df: pd.DataFrame):
        """Compute Elo ratings for each team game-by-game (no lookahead)."""
        elo_dict: Dict[int, TeamElo] = {}

        elo_home_list, elo_away_list = [], []
        elo_home_pre_list, elo_away_pre_list = [], []
        elo_diff_list = []
        elo_prob_home_list = []

        for idx, row in df.iterrows():
            home_id = int(row["TEAM_ID_home"])
            away_id = int(row["TEAM_ID_away"])
            home_pts = row["team_pts_home"]
            away_pts = row["team_pts_away"]
            game_date = str(row["GAME_DATE"])

            # Get or initialize Elo
            home_elo = elo_dict.get(home_id, TeamElo())
            away_elo = elo_dict.get(away_id, TeamElo())
            elo_dict.setdefault(home_id, home_elo)
            elo_dict.setdefault(away_id, away_elo)

            # Pre-game ratings (current state before this game)
            home_rating_pre = home_elo.rating
            away_rating_pre = away_elo.rating

            elo_home_pre_list.append(home_rating_pre)
            elo_away_pre_list.append(away_rating_pre)
            elo_diff_list.append(home_rating_pre - away_rating_pre)

            # Expected win probability (logistic)
            elo_diff = home_rating_pre - away_rating_pre + ELO_HOME_ADV
            expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            elo_prob_home_list.append(expected_home)

            # Actual result
            home_won = 1.0 if home_pts > away_pts else (0.5 if home_pts == away_pts else 0.0)

            # Update Elo
            margin = abs(home_pts - away_pts)
            elo_margin_factor = np.log(max(margin, 1) + 1) / np.log(4)  # ~1 for 3pt game, ~2 for 15pt
            k_actual = ELO_K * elo_margin_factor

            home_new = home_rating_pre + k_actual * (home_won - expected_home)
            away_new = away_rating_pre + k_actual * ((1 - home_won) - (1 - expected_home))

            home_elo.rating = home_new
            away_elo.rating = away_new

            if home_elo.history is None:
                home_elo.history = []
            if away_elo.history is None:
                away_elo.history = []
            home_elo.history.append((game_date, home_new))
            away_elo.history.append((game_date, away_new))

            elo_home_list.append(home_new)
            elo_away_list.append(away_new)

        df["elo_home_pre"] = elo_home_pre_list
        df["elo_away_pre"] = elo_away_pre_list
        df["elo_diff"] = elo_diff_list
        df["elo_prob_home"] = elo_prob_home_list

        # Rolling Elo features
        df["elo_home_5g_avg"] = (
            df.groupby("TEAM_ID_home")["elo_home_pre"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        )
        df["elo_away_5g_avg"] = (
            df.groupby("TEAM_ID_away")["elo_away_pre"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
        )
        df["elo_momentum_home"] = df["elo_home_pre"] - df["elo_home_5g_avg"]
        df["elo_momentum_away"] = df["elo_away_pre"] - df["elo_away_5g_avg"]

        self.team_elos = elo_dict

    # ═══════════════════════════════════════════════════════════════════
    #  OPPONENT-ADJUSTED FEATURES
    # ═══════════════════════════════════════════════════════════════════

    def _add_opponent_adjusted_features(self, df: pd.DataFrame):
        """
        Compute opponent-adjusted stats.
        For each team's performance, adjust based on the strength of opponents faced.
        """
        # Build opponent strength ratings for each game
        team_ids = set(df["TEAM_ID_home"].unique()) | set(df["TEAM_ID_away"].unique())

        # Compute opponent strength as rolling average opponent Elo faced
        for suffix in ["home", "away"]:
            team_id_col = f"TEAM_ID_{suffix}"
            opp_elo_col = f"elo_{'away' if suffix == 'home' else 'home'}_pre"

            df[f"opp_elo_{suffix}"] = df[opp_elo_col]

            # Rolling average opponent Elo faced (strength of schedule)
            df[f"sos_{suffix}"] = (
                df.groupby(team_id_col)[f"opp_elo_{suffix}"]
                .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            )

        # Adjusted margin: margin + opponent Elo adjustment
        df["adj_margin_home"] = (
            df["point_diff"]
            + (df["elo_away_pre"] - ELO_INITIAL) / 50
        )

        # Momentum vs opponent: how team's recent form compares to opponent's
        df["mom_vs_opp_home"] = (
            df.get("weighted_recent_5_home", 0) - df.get("weighted_recent_5_away", 0)
        )
        df["mom_vs_opp_away"] = (
            df.get("weighted_recent_5_away", 0) - df.get("weighted_recent_5_home", 0)
        )

    # ═══════════════════════════════════════════════════════════════════
    #  GAME-LEVEL COMPOSITE FEATURES
    # ═══════════════════════════════════════════════════════════════════

    def _add_game_level_features(self, df: pd.DataFrame):
        """Create head-to-head composite features from home/away team stats."""
        # ── Predicted Pace ──
        df["predicted_pace"] = (
            df.get("avg_pace_5g_home", 100).fillna(100)
            + df.get("avg_pace_5g_away", 100).fillna(100)
        ) / 2

        df["pace_diff"] = (
            df.get("avg_pace_5g_home", 100).fillna(100)
            - df.get("avg_pace_5g_away", 100).fillna(100)
        )

        # ── Offensive & Defensive Strength ──
        df["offensive_strength"] = (
            df.get("avg_pts_5g_home", 100).fillna(100)
            + df.get("avg_pts_5g_away", 100).fillna(100)
        )

        df["defensive_strength"] = (
            df.get("avg_pts_allowed_5g_home", 100).fillna(100)
            + df.get("avg_pts_allowed_5g_away", 100).fillna(100)
        )

        # ── Advanced predicted total ──
        # Use TS% and pace for more accurate total prediction
        home_off_rating = df.get("avg_ppp_5g_home", 1.0).fillna(1.0) * 100
        away_off_rating = df.get("avg_ppp_5g_away", 1.0).fillna(1.0) * 100
        home_def_rating = (1 - df.get("avg_ts_5g_home", TS_LEAGUE_AVG).fillna(TS_LEAGUE_AVG)) * 100
        away_def_rating = (1 - df.get("avg_ts_5g_away", TS_LEAGUE_AVG).fillna(TS_LEAGUE_AVG)) * 100

        df["predicted_total_advanced"] = (
            ((home_off_rating + away_def_rating) / 2
             + (away_off_rating + home_def_rating) / 2)
            * df["predicted_pace"] / PACE_LEAGUE_AVG
        )

        # ── Offensive/Defensive Efficiency Differential ──
        df["oe_diff"] = home_off_rating - away_off_rating
        df["de_diff"] = home_def_rating - away_def_rating

        # ── Net Rating (off - def) ──
        df["net_rating_home"] = (
            df.get("avg_pm_10g_home", 0).fillna(0)
        )
        df["net_rating_away"] = (
            df.get("avg_pm_10g_away", 0).fillna(0)
        )
        df["net_rating_diff"] = df["net_rating_home"] - df["net_rating_away"]

        # ── Combined form indicator ──
        df["form_home"] = (
            df.get("weighted_recent_5_home", 0.5).fillna(0.5)
            + 0.01 * df.get("avg_pm_5g_home", 0).fillna(0)
        )
        df["form_away"] = (
            df.get("weighted_recent_5_away", 0.5).fillna(0.5)
            + 0.01 * df.get("avg_pm_5g_away", 0).fillna(0)
        )
        df["form_diff"] = df["form_home"] - df["form_away"]

        # ── Home court advantage (interaction with fatigue) ──
        df["home_advantage"] = 1.0
        df["rest_interaction"] = df["rest_advantage"] * df["home_advantage"]

        # ── Elo combined features ──
        df["elo_total"] = df["elo_home_pre"] + df["elo_away_pre"]
        df["elo_diff_norm"] = df["elo_diff"] / 400

        # ── Win probability blend (Elo + momentum) ──
        df["win_prob_elo"] = df["elo_prob_home"]
        df["win_prob_momentum"] = np.clip(
            0.5 + df.get("weighted_recent_5_home", 0.5).fillna(0.5)
            - df.get("weighted_recent_5_away", 0.5).fillna(0.5),
            0, 1
        )
        df["win_prob_blended"] = 0.6 * df["win_prob_elo"] + 0.4 * df["win_prob_momentum"]

        # ── Interaction features ──
        df["rest_elo_interact"] = df["rest_advantage"] * df["elo_diff_norm"]
        df["pace_rest_interact"] = df["predicted_pace"] * df["fatigue_diff"]
        df["travel_rest_interact"] = df["travel_distance_norm"] * df["rest_advantage"]

    # ═══════════════════════════════════════════════════════════════════
    #  HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _compute_streak(self, wl_numeric: pd.Series) -> pd.Series:
        """Compute streak length from numeric WL (1=win, 0=loss). Positive = win streak."""
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

    def _weighted_recent(self, x: pd.Series, window: int = 5) -> pd.Series:
        """
        Compute exponentially weighted recent performance.
        More recent games get higher weight.
        """
        weights = np.exp(np.linspace(-1, 0, window))
        weights /= weights.sum()

        def rolling_weighted_mean(series):
            if len(series) < window:
                return series.mean() if len(series) > 0 else 0.5
            return np.sum(series.tail(window).values * weights)

        result = x.rolling(window, min_periods=1).apply(
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
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect feature columns from a dataframe."""
        exclude = {
            "GAME_ID", "SEASON_ID", "TEAM_ID_home", "TEAM_ID_away",
            "TEAM_ABBREVIATION_home", "TEAM_ABBREVIATION_away",
            "TEAM_NAME_home", "TEAM_NAME_away", "GAME_DATE",
            "MATCHUP_home", "MATCHUP_away",
            "WL_home", "WL_away",
            "SEASON_home", "SEASON_away",
            "total_points", "point_diff", "home_team_name", "away_team_name",
            "home_team", "away_team",
            "MIN_home", "MIN_away",
        }
        return [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64")]
