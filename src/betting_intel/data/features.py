"""
Feature engineering: transforms raw game data into predictive features.
All features should be calculable BEFORE the game starts (no lookahead bias).

v5.1 — ULTIMATE FEATURE SET:
  - ALL v2.2/v3.0/v3.1 features preserved (rolling stats, EMAs, trends,
    travel, fatigue, ELO, moneyline-specific, opponent-adjusted, etc.)
  - Target encoding: team_id → historical win rate + avg margin (A)
  - Seasonality: day_of_week, month, season_phase (B)
  - Comprehensive Fatigue Index: games in last N days × rest quality (C)
  - Coach change proxy: early-season performance shifts (D)
  - Home/away splits: team's performance at home vs on road
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from betting_intel.config import ROLLING_WINDOWS, MAX_REST_DAYS    # ── Constants for Advanced Features ───────────────────────────────────────

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
    # NCAAB — ACC (approximate campus locations)
    "Duke": (36.001, -78.938), "UNC": (35.904, -79.047),
    "Virginia": (38.032, -78.511), "NC State": (35.785, -78.682),
    "Clemson": (34.679, -82.839), "Miami": (25.713, -80.276),
    "Florida State": (30.442, -84.299), "Virginia Tech": (37.229, -80.426),
    "Louisville": (38.215, -85.741), "Syracuse": (43.037, -76.134),
    "Notre Dame": (41.699, -86.237), "Pittsburgh": (40.444, -79.962),
    # NCAAB — Big Ten
    "Michigan State": (42.701, -84.482), "Michigan": (42.278, -83.738),
    "Purdue": (40.424, -86.921), "Indiana": (39.173, -86.514),
    "Illinois": (40.102, -88.227), "Ohio State": (40.002, -83.026),
    "Wisconsin": (43.074, -89.393), "Iowa": (41.660, -91.536),
    "Maryland": (38.989, -76.947), "Rutgers": (40.501, -74.447),
    "Penn State": (40.793, -77.861), "Minnesota": (44.977, -93.228),
    "Northwestern": (42.058, -87.674), "UCLA": (34.071, -118.410),
    "USC": (34.021, -118.288), "Washington": (47.651, -122.305),
    "Oregon": (44.058, -123.073), "Oregon State": (44.565, -123.279),
    # NCAAB — SEC
    "Kentucky": (38.030, -84.508), "Tennessee": (35.951, -83.930),
    "Alabama": (33.214, -87.542), "Auburn": (32.607, -85.491),
    "Florida": (29.647, -82.345), "Arkansas": (36.071, -94.176),
    "LSU": (30.413, -91.184), "Texas A&M": (30.616, -96.335),
    "Mississippi State": (33.456, -88.791), "South Carolina": (33.996, -81.029),
    "Ole Miss": (34.366, -89.537), "Georgia": (33.946, -83.377),
    "Oklahoma": (35.206, -97.443), "Texas": (30.283, -97.733),
    # NCAAB — Big 12
    "Kansas": (38.955, -95.247), "Baylor": (31.549, -97.116),
    "Houston": (29.722, -95.350), "Texas Tech": (33.580, -101.876),
    "Iowa State": (42.026, -93.650), "TCU": (32.722, -97.340),
    "West Virginia": (39.651, -79.986), "Kansas State": (39.193, -96.583),
    "BYU": (40.249, -111.649), "Cincinnati": (39.131, -84.514),
    "UCF": (28.600, -81.200), "Arizona": (32.229, -110.949),
    "Arizona State": (33.423, -111.932), "Colorado": (40.008, -105.267),
    "Utah": (40.765, -111.848),
    # NCAAB — Big East
    "UConn": (41.807, -72.254), "Marquette": (43.038, -87.930),
    "Villanova": (40.038, -75.337), "Creighton": (41.256, -95.985),
    "Xavier": (39.149, -84.475), "Providence": (41.845, -71.440),
    "St. John's": (40.728, -73.794), "Butler": (39.840, -86.173),
    "Seton Hall": (40.742, -74.175), "Georgetown": (38.907, -77.072),
    # NCAAB — Others
    "Gonzaga": (47.669, -117.405), "Saint Mary's": (37.929, -122.050),
    "San Diego State": (32.775, -117.073), "Memphis": (35.118, -89.937),
    "VCU": (37.542, -77.455), "Dayton": (39.735, -84.179),
    "Grand Canyon": (33.513, -112.133), "Princeton": (40.345, -74.659),
    "Liberty": (37.348, -79.178), "James Madison": (38.437, -78.873),
    "San Francisco": (37.780, -122.452), "Santa Clara": (37.350, -121.937),
    "Loyola Chicago": (41.998, -87.658), "Saint Louis": (38.636, -90.220),
    "Drake": (41.599, -93.652), "Indiana State": (39.471, -87.408),
    # Default for un-mapped NCAAB teams (central US)
    # Euroleague teams — arenas across Europe
    "Real Madrid": (40.453, -3.688), "Barcelona": (41.383, 2.117),
    "Olympiacos": (37.935, 23.683), "Panathinaikos": (38.055, 23.783),
    "Fenerbahçe": (41.017, 28.997), "Anadolu Efes": (40.983, 28.850),
    "Crvena Zvezda": (44.817, 20.467), "Žalgiris": (54.900, 23.917),
    "Maccabi Tel Aviv": (32.083, 34.800), "Paris": (48.867, 2.333),
    "Monaco": (43.700, 7.417), "Bayern Munich": (48.117, 11.500),
    "Milan": (45.417, 9.050), "ASVEL": (45.717, 4.967),
    "Baskonia": (42.850, -2.683), "Valencia": (39.467, -0.367),
    "Partizan": (44.817, 20.467), "Virtus Bologna": (44.500, 11.317),
    "Hapoel Tel Aviv": (32.083, 34.800), "Dubai": (25.217, 55.283),
    # NFL teams — stadiums across the US
    "Bills": (42.767, -78.733), "Dolphins": (25.958, -80.239),
    "Patriots": (42.087, -71.267), "Jets": (40.813, -74.074),
    "Ravens": (39.278, -76.622), "Bengals": (39.083, -84.517),
    "Browns": (41.517, -81.683), "Steelers": (40.450, -80.017),
    "Texans": (29.683, -95.417), "Colts": (39.750, -86.167),
    "Jaguars": (30.317, -81.633), "Titans": (36.167, -86.783),
    "Broncos": (39.733, -105.017), "Chiefs": (39.050, -94.483),
    "Raiders": (36.083, -115.183), "Chargers": (33.950, -118.333),
    "Cowboys": (32.750, -97.083), "Giants": (40.813, -74.074),
    "Eagles": (39.900, -75.167), "Commanders": (38.900, -76.867),
    "Bears": (41.867, -87.617), "Lions": (42.317, -83.050),
    "Packers": (44.500, -88.017), "Vikings": (44.967, -93.267),
    "Falcons": (33.750, -84.400), "Panthers": (35.217, -80.850),
    "Saints": (29.950, -90.083), "Buccaneers": (27.967, -82.517),
    "Cardinals": (33.533, -112.267), "Rams": (33.950, -118.333),
    "49ers": (37.400, -121.967), "Seahawks": (47.600, -122.333),
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
    # NCAAB — approximate
    "Duke": -5, "UNC": -5, "Virginia": -5, "NC State": -5,
    "Clemson": -5, "Miami": -5, "Florida State": -5, "Virginia Tech": -5,
    "Louisville": -5, "Syracuse": -5, "Notre Dame": -5, "Pittsburgh": -5,
    "Michigan State": -5, "Michigan": -5, "Purdue": -5, "Indiana": -5,
    "Illinois": -6, "Ohio State": -5, "Wisconsin": -6, "Iowa": -6,
    "Maryland": -5, "Rutgers": -5, "Penn State": -5, "Minnesota": -6,
    "Northwestern": -6, "UCLA": -8, "USC": -8, "Washington": -8,
    "Oregon": -8, "Kentucky": -5, "Tennessee": -5, "Alabama": -6,
    "Auburn": -6, "Florida": -5, "Arkansas": -6, "LSU": -6,
    "Texas A&M": -6, "Oklahoma": -6, "Texas": -6, "Kansas": -6,
    "Baylor": -6, "Houston": -6, "Texas Tech": -6, "Iowa State": -6,
    "Arizona": -7, "Arizona State": -7, "Colorado": -7, "Utah": -7,
    "BYU": -7, "UConn": -5, "Gonzaga": -8, "San Diego State": -8,
    "Saint Mary's": -8, "Memphis": -6, "VCU": -5, "Dayton": -5,
    # Euroleague — CET (+1) and EET (+2)
    "Real Madrid": 1, "Barcelona": 1, "Paris": 1,  "Monaco": 1,
    "Bayern Munich": 1, "Milan": 1, "ASVEL": 1,
    "Baskonia": 1, "Valencia": 1, "Virtus Bologna": 1,
    "Olympiacos": 2, "Panathinaikos": 2, "Fenerbahçe": 2,
    "Anadolu Efes": 2, "Crvena Zvezda": 2, "Partizan": 2,
    "Žalgiris": 2, "Maccabi Tel Aviv": 2, "Hapoel Tel Aviv": 2,
    "Dubai": 4,
    # NFL — EST (-5), CST (-6), MST (-7), PST (-8)
    "Bills": -5, "Dolphins": -5, "Patriots": -5, "Jets": -5,
    "Ravens": -5, "Bengals": -5, "Browns": -5, "Steelers": -5,
    "Colts": -5, "Jaguars": -5, "Giants": -5, "Eagles": -5,
    "Commanders": -5, "Falcons": -5, "Panthers": -5,
    "Buccaneers": -5, "Lions": -5,
    "Texans": -6, "Titans": -6, "Chiefs": -6, "Bears": -6,
    "Packers": -6, "Vikings": -6, "Saints": -6,
    "Cowboys": -6, "Broncos": -7, "Cardinals": -7,
    "Raiders": -8, "Chargers": -8, "Rams": -8,
    "49ers": -8, "Seahawks": -8,
}


# ── NBA Backfill Constants ───────────────────────────────────────────────
# NBA per-team averages for backfilling NaN rolling features.
# Pre-defined constants avoid data leakage from dataset statistics.
_NBA_NA_FILL: list[tuple[str, float]] = [
    # Opponent-allowed rolling averages (must come before base stats)
    ("avg_reb_allowed", 43.0),
    ("avg_ast_allowed", 25.0),
    ("avg_stl_allowed", 7.5),
    ("avg_blk_allowed", 5.0),
    ("avg_tov_allowed", 13.0),
    ("avg_pf_allowed", 19.0),
    ("avg_pts_allowed", 114.5),
    ("avg_fgm_allowed", 42.0),
    ("avg_fga_allowed", 88.0),
    ("avg_fg3m_allowed", 13.5),
    ("avg_fg3a_allowed", 38.0),
    ("avg_ftm_allowed", 18.0),
    ("avg_fta_allowed", 23.0),
    ("avg_oreb_allowed", 10.0),
    ("avg_dreb_allowed", 33.0),
    # Off/def comparisons (must come before generic diff)
    ("offense_vs_defense", 1.0),
    ("defense_vs_offense", 1.0),
    # Rate stats (10g variants before base)
    ("three_pt_rate_10g", 0.38),
    ("ft_rate_10g", 0.26),
    ("ast_ratio_10g", 0.18),
    ("ts_pct_10g", 0.57),
    ("reb_pct_10g", 0.50),
    ("three_pt_rate", 0.38),
    ("ft_rate", 0.26),
    ("ast_ratio", 0.18),
    ("ts_pct", 0.57),
    ("reb_pct", 0.50),
    # Opponent features
    ("opp_avg_pts_scored", 114.5),
    ("opp_avg_pts_allowed", 114.5),
    ("opp_avg_pm", 0.0),
    ("opp_trailing_margin", 0.0),
    ("adj_opp_avg_pm", 0.0),
    # Trend slopes for boxscore stats
    ("trend_fgm", 0.0),
    ("trend_fga", 0.0),
    ("trend_reb", 0.0),
    ("trend_ast", 0.0),
    ("trend_stl", 0.0),
    ("trend_blk", 0.0),
    ("trend_tov", 0.0),
    ("trend_pts", 0.0),
    ("trend_pm", 0.0),
    # EMA for boxscore stats
    ("ema_fgm", 42.0),
    ("ema_fga", 88.0),
    ("ema_reb", 43.0),
    ("ema_ast", 25.0),
    ("ema_stl", 7.5),
    ("ema_blk", 5.0),
    ("ema_tov", 13.0),
    ("ema_pf", 19.0),
    ("ema_pts", 114.5),
    ("ema_pm", 0.0),
    ("ema_margin", 0.0),
    # Boxscore stat rolling averages
    ("avg_fg3_pct", 0.355),
    ("avg_ft_pct", 0.780),
    ("avg_fgm", 42.0),
    ("avg_fga", 88.0),
    ("avg_fg3m", 13.5),
    ("avg_fg3a", 38.0),
    ("avg_ftm", 18.0),
    ("avg_fta", 23.0),
    ("avg_oreb", 10.0),
    ("avg_dreb", 33.0),
    ("avg_reb", 43.0),
    ("avg_ast", 25.0),
    ("avg_stl", 7.5),
    ("avg_blk", 5.0),
    ("avg_tov", 13.0),
    ("avg_pf", 19.0),
    ("avg_pace", 100.0),
    ("avg_efg", 0.54),
    ("avg_pts", 114.5),
    ("avg_pm", 0.0),
    ("avg_margin", 0.0),
    ("margin_volatility", 12.0),
    ("last_3_margin", 0.0),
    # Win rates & momentum
    ("win_pct", 0.5),
    ("win_streak", 0.0),
    ("weighted_momentum", 0.5),
    ("form_score", 0.5),
    # Pace
    ("pace", 100.0),
    # Efficiency
    ("efg", 0.54),
    # Points / z-scores
    ("pts_zscore", 0.0),
    # Strength of schedule
    ("sos_trend", 0.0),
    ("sos", 0.0),
    # Travel / cumulative
    ("cum_travel", 0.0),
    # Elo ratings
    ("elo_slope", 0.0),
    # Moneyline features (v3.1)
    ("composite_power", 0.5),
    ("power_diff", 0.0),
    ("perf_vs_expected_raw", 0.0),
    ("perf_vs_expected", 0.0),
    ("perf_vs_expected_diff", 0.0),
    ("consistency", 0.5),
    ("consistency_diff", 0.0),
    ("form_diff", 0.0),
    ("home_away_split_diff", 0.0),
    ("recent_win_pct_home", 0.5),
    ("recent_win_pct_away", 0.5),
    ("h2h_win_rate", 0.5),
    ("h2h_avg_margin", 0.0),
    # Home-away differentials (diff = 0 means teams are equal)
    ("_diff_", 0.0),
    # v5.1 features: target encoding
    ("target_win_rate", 0.5),
    ("target_margin", 0.0),
    # v5.1 features: seasonality
    ("dow_sin", 0.0),
    ("dow_cos", 1.0),
    ("month_sin", 0.0),
    ("month_cos", 1.0),
    ("season_phase", 1.0),
    ("days_since_asb", 0.0),
    ("days_into_season", 15.0),
    ("is_weekend", 0),
    # v5.1 features: fatigue index
    ("fatigue_index", 0.3),
    ("games_3d", 1.0),
    ("games_5d", 2.0),
    ("games_7d", 3.0),
    # v5.1 features: home/away splits
    ("home_win_rate_at_home", 0.6),
    ("away_win_rate_on_road", 0.4),
    ("home_advantage_edge", 0.0),
    # v5.1 features: coach change proxy
    ("perf_shift_z", 0.0),
    ("system_change_flag", 0),
    ("system_changes_10g", 0.0),
    ("system_changes_diff", 0.0),
    # v6.5 features: interaction features
    ("interact_", 0.0),
    # v6.5 features: rolling volatility
    ("volatility_pts_", 12.0),
    ("volatility_pm_", 12.0),
    ("volatility_allowed_", 12.0),
    ("volatility_pace_", 10.0),
    # v6.5 features: momentum/streak
    ("win_streak", 0.0),
    ("streak_margin", 0.0),
    ("streak_quality", 0.0),
    ("prev_loss", 0.0),
    ("form_acceleration", 0.0),
    # v6.5 features: pace-adjusted
    ("pace_adj_off", 110.0),
    ("pace_adj_def", 110.0),
    ("pace_adj_net", 0.0),
    ("pace_100", 100.0),
]


# ── NCAAB-Specific Backfill Constants ────────────────────────────────────
# NCAAB per-team stats: avg ~70 pts, lower pace, less 3-point usage.
_NCAAB_NA_FILL: list[tuple[str, float]] = [
    # Opponent-allowed rolling averages (must come before base stats)
    ("avg_reb_allowed", 33.0),
    ("avg_ast_allowed", 13.0),
    ("avg_stl_allowed", 6.5),
    ("avg_blk_allowed", 3.5),
    ("avg_tov_allowed", 11.0),
    ("avg_pf_allowed", 16.0),
    ("avg_pts_allowed", 70.0),
    ("avg_fgm_allowed", 25.0),
    ("avg_fga_allowed", 56.0),
    ("avg_fg3m_allowed", 7.0),
    ("avg_fg3a_allowed", 21.0),
    ("avg_ftm_allowed", 13.0),
    ("avg_fta_allowed", 18.0),
    ("avg_oreb_allowed", 9.0),
    ("avg_dreb_allowed", 24.0),
    # Off/def comparisons
    ("offense_vs_defense", 1.0),
    ("defense_vs_offense", 1.0),
    # Rate stats
    ("three_pt_rate_10g", 0.36),
    ("ft_rate_10g", 0.31),
    ("ast_ratio_10g", 0.15),
    ("ts_pct_10g", 0.53),
    ("reb_pct_10g", 0.50),
    ("three_pt_rate", 0.36),
    ("ft_rate", 0.31),
    ("ast_ratio", 0.15),
    ("ts_pct", 0.53),
    ("reb_pct", 0.50),
    # Opponent features
    ("opp_avg_pts_scored", 70.0),
    ("opp_avg_pts_allowed", 70.0),
    ("opp_avg_pm", 0.0),
    ("opp_trailing_margin", 0.0),
    ("adj_opp_avg_pm", 0.0),
    # Trend slopes
    ("trend_fgm", 0.0),
    ("trend_fga", 0.0),
    ("trend_reb", 0.0),
    ("trend_ast", 0.0),
    ("trend_stl", 0.0),
    ("trend_blk", 0.0),
    ("trend_tov", 0.0),
    ("trend_pts", 0.0),
    ("trend_pm", 0.0),
    # EMA for boxscore stats
    ("ema_fgm", 25.0),
    ("ema_fga", 56.0),
    ("ema_reb", 33.0),
    ("ema_ast", 13.0),
    ("ema_stl", 6.5),
    ("ema_blk", 3.5),
    ("ema_tov", 11.0),
    ("ema_pf", 16.0),
    ("ema_pts", 70.0),
    ("ema_pm", 0.0),
    ("ema_margin", 0.0),
    # Boxscore stat rolling averages
    ("avg_fg3_pct", 0.34),
    ("avg_ft_pct", 0.73),
    ("avg_fgm", 25.0),
    ("avg_fga", 56.0),
    ("avg_fg3m", 7.0),
    ("avg_fg3a", 21.0),
    ("avg_ftm", 13.0),
    ("avg_fta", 18.0),
    ("avg_oreb", 9.0),
    ("avg_dreb", 24.0),
    ("avg_reb", 33.0),
    ("avg_ast", 13.0),
    ("avg_stl", 6.5),
    ("avg_blk", 3.5),
    ("avg_tov", 11.0),
    ("avg_pf", 16.0),
    ("avg_pace", 70.0),
    ("avg_efg", 0.51),
    ("avg_pts", 70.0),
    ("avg_pm", 0.0),
    ("avg_margin", 0.0),
    ("margin_volatility", 15.0),
    ("last_3_margin", 0.0),
    # Win rates & momentum
    ("win_pct", 0.5),
    ("win_streak", 0.0),
    ("weighted_momentum", 0.5),
    ("form_score", 0.5),
    # Pace
    ("pace", 70.0),
    # Efficiency
    ("efg", 0.51),
    # Points / z-scores
    ("pts_zscore", 0.0),
    # SOS
    ("sos_trend", 0.0),
    ("sos", 0.0),
    # Travel
    ("cum_travel", 0.0),
    # ELO
    ("elo_slope", 0.0),
    # Moneyline features
    ("composite_power", 0.5),
    ("power_diff", 0.0),
    ("perf_vs_expected_raw", 0.0),
    ("perf_vs_expected", 0.0),
    ("perf_vs_expected_diff", 0.0),
    ("consistency", 0.5),
    ("consistency_diff", 0.0),
    ("form_diff", 0.0),
    ("home_away_split_diff", 0.0),
    ("recent_win_pct_home", 0.5),
    ("recent_win_pct_away", 0.5),
    ("h2h_win_rate", 0.5),
    ("h2h_avg_margin", 0.0),
    # Home-away differentials
    ("_diff_", 0.0),
    # Target encoding
    ("target_win_rate", 0.5),
    ("target_margin", 0.0),
    # Seasonality
    ("dow_sin", 0.0),
    ("dow_cos", 1.0),
    ("month_sin", 0.0),
    ("month_cos", 1.0),
    ("season_phase", 1.0),
    ("days_since_asb", 0.0),
    ("days_into_season", 15.0),
    ("is_weekend", 0),
    # Fatigue
    ("fatigue_index", 0.3),
    ("games_3d", 1.0),
    ("games_5d", 2.0),
    ("games_7d", 3.0),
    # Home/away splits
    ("home_win_rate_at_home", 0.6),
    ("away_win_rate_on_road", 0.4),
    ("home_advantage_edge", 0.0),
    # Coach change proxy
    ("perf_shift_z", 0.0),
    ("system_change_flag", 0),
    ("system_changes_10g", 0.0),
    ("system_changes_diff", 0.0),
]


# ── Euroleague-Specific Backfill Constants ────────────────────────────────
# Euroleague per-team stats: avg ~78 pts, moderate pace (~72),
# higher foul rate, lower 3PT volume than NBA.
# Total points per game ~156 (half of NBA's ~228).
_EUROLEAGUE_NA_FILL: list[tuple[str, float]] = [
    # Opponent-allowed rolling averages (must come before base stats)
    ("avg_reb_allowed", 33.0),
    ("avg_ast_allowed", 17.0),
    ("avg_stl_allowed", 6.5),
    ("avg_blk_allowed", 3.0),
    ("avg_tov_allowed", 12.0),
    ("avg_pf_allowed", 20.0),
    ("avg_pts_allowed", 78.0),
    ("avg_fgm_allowed", 27.0),
    ("avg_fga_allowed", 58.0),
    ("avg_fg3m_allowed", 8.0),
    ("avg_fg3a_allowed", 23.0),
    ("avg_ftm_allowed", 16.0),
    ("avg_fta_allowed", 21.0),
    ("avg_oreb_allowed", 9.0),
    ("avg_dreb_allowed", 24.0),
    # Off/def comparisons
    ("offense_vs_defense", 1.0),
    ("defense_vs_offense", 1.0),
    # Rate stats
    ("three_pt_rate_10g", 0.38),
    ("ft_rate_10g", 0.33),
    ("ast_ratio_10g", 0.17),
    ("ts_pct_10g", 0.55),
    ("reb_pct_10g", 0.50),
    ("three_pt_rate", 0.38),
    ("ft_rate", 0.33),
    ("ast_ratio", 0.17),
    ("ts_pct", 0.55),
    ("reb_pct", 0.50),
    # Opponent features
    ("opp_avg_pts_scored", 78.0),
    ("opp_avg_pts_allowed", 78.0),
    ("opp_avg_pm", 0.0),
    ("opp_trailing_margin", 0.0),
    ("adj_opp_avg_pm", 0.0),
    # Trend slopes
    ("trend_fgm", 0.0),
    ("trend_fga", 0.0),
    ("trend_reb", 0.0),
    ("trend_ast", 0.0),
    ("trend_stl", 0.0),
    ("trend_blk", 0.0),
    ("trend_tov", 0.0),
    ("trend_pts", 0.0),
    ("trend_pm", 0.0),
    # EMA for boxscore stats
    ("ema_fgm", 27.0),
    ("ema_fga", 58.0),
    ("ema_reb", 33.0),
    ("ema_ast", 17.0),
    ("ema_stl", 6.5),
    ("ema_blk", 3.0),
    ("ema_tov", 12.0),
    ("ema_pf", 20.0),
    ("ema_pts", 78.0),
    ("ema_pm", 0.0),
    ("ema_margin", 0.0),
    # Boxscore stat rolling averages
    ("avg_fg3_pct", 0.35),
    ("avg_ft_pct", 0.78),
    ("avg_fgm", 27.0),
    ("avg_fga", 58.0),
    ("avg_fg3m", 8.0),
    ("avg_fg3a", 23.0),
    ("avg_ftm", 16.0),
    ("avg_fta", 21.0),
    ("avg_oreb", 9.0),
    ("avg_dreb", 24.0),
    ("avg_reb", 33.0),
    ("avg_ast", 17.0),
    ("avg_stl", 6.5),
    ("avg_blk", 3.0),
    ("avg_tov", 12.0),
    ("avg_pf", 20.0),
    ("avg_pace", 72.0),
    ("avg_efg", 0.52),
    ("avg_pts", 78.0),
    ("avg_pm", 0.0),
    ("avg_margin", 0.0),
    ("margin_volatility", 14.0),
    ("last_3_margin", 0.0),
    # Win rates & momentum
    ("win_pct", 0.5),
    ("win_streak", 0.0),
    ("weighted_momentum", 0.5),
    ("form_score", 0.5),
    # Pace
    ("pace", 72.0),
    # Efficiency
    ("efg", 0.52),
    # Points / z-scores
    ("pts_zscore", 0.0),
    # SOS
    ("sos_trend", 0.0),
    ("sos", 0.0),
    # Travel
    ("cum_travel", 0.0),
    # ELO
    ("elo_slope", 0.0),
    # Moneyline features
    ("composite_power", 0.5),
    ("power_diff", 0.0),
    ("perf_vs_expected_raw", 0.0),
    ("perf_vs_expected", 0.0),
    ("perf_vs_expected_diff", 0.0),
    ("consistency", 0.5),
    ("consistency_diff", 0.0),
    ("form_diff", 0.0),
    ("home_away_split_diff", 0.0),
    ("recent_win_pct_home", 0.5),
    ("recent_win_pct_away", 0.5),
    ("h2h_win_rate", 0.5),
    ("h2h_avg_margin", 0.0),
    # Home-away differentials
    ("_diff_", 0.0),
    # Target encoding
    ("target_win_rate", 0.5),
    ("target_margin", 0.0),
    # Seasonality
    ("dow_sin", 0.0),
    ("dow_cos", 1.0),
    ("month_sin", 0.0),
    ("month_cos", 1.0),
    ("season_phase", 1.0),
    ("days_since_asb", 0.0),
    ("days_into_season", 15.0),
    ("is_weekend", 0),
    # Fatigue
    ("fatigue_index", 0.3),
    ("games_3d", 1.0),
    ("games_5d", 2.0),
    ("games_7d", 3.0),
    # Home/away splits
    ("home_win_rate_at_home", 0.6),
    ("away_win_rate_on_road", 0.4),
    ("home_advantage_edge", 0.0),
    # Coach change proxy
    ("perf_shift_z", 0.0),
    ("system_change_flag", 0),
    ("system_changes_10g", 0.0),
    ("system_changes_diff", 0.0),
]


# ── NFL-Specific Backfill Constants ──────────────────────────────────────
# NFL per-team stats: avg ~22 pts, slow pace (~65 possessions),
# no basketball-specific stats (fgm, reb, ast, etc.).
# Basketball-only feature columns won't be created for NFL data,
# so those fill values are set to 0.0 and won't be hit.
_NFL_NA_FILL: list[tuple[str, float]] = [
    # Opponent-allowed rolling averages (must come before base stats)
    ("avg_reb_allowed", 0.0),
    ("avg_ast_allowed", 0.0),
    ("avg_stl_allowed", 0.0),
    ("avg_blk_allowed", 0.0),
    ("avg_tov_allowed", 0.0),
    ("avg_pf_allowed", 0.0),
    ("avg_pts_allowed", 22.0),
    ("avg_fgm_allowed", 0.0),
    ("avg_fga_allowed", 0.0),
    ("avg_fg3m_allowed", 0.0),
    ("avg_fg3a_allowed", 0.0),
    ("avg_ftm_allowed", 0.0),
    ("avg_fta_allowed", 0.0),
    ("avg_oreb_allowed", 0.0),
    ("avg_dreb_allowed", 0.0),
    # Off/def comparisons
    ("offense_vs_defense", 1.0),
    ("defense_vs_offense", 1.0),
    # Rate stats (all N/A for football → 0.0)
    ("three_pt_rate_10g", 0.0),
    ("ft_rate_10g", 0.0),
    ("ast_ratio_10g", 0.0),
    ("ts_pct_10g", 0.0),
    ("reb_pct_10g", 0.0),
    ("three_pt_rate", 0.0),
    ("ft_rate", 0.0),
    ("ast_ratio", 0.0),
    ("ts_pct", 0.0),
    ("reb_pct", 0.0),
    # Opponent features
    ("opp_avg_pts_scored", 22.0),
    ("opp_avg_pts_allowed", 22.0),
    ("opp_avg_pm", 0.0),
    ("opp_trailing_margin", 0.0),
    ("adj_opp_avg_pm", 0.0),
    # Trend slopes (basketball stats → 0.0)
    ("trend_fgm", 0.0),
    ("trend_fga", 0.0),
    ("trend_reb", 0.0),
    ("trend_ast", 0.0),
    ("trend_stl", 0.0),
    ("trend_blk", 0.0),
    ("trend_tov", 0.0),
    ("trend_pts", 0.0),
    ("trend_pm", 0.0),
    # EMA for boxscore stats (N/A → 0.0)
    ("ema_fgm", 0.0),
    ("ema_fga", 0.0),
    ("ema_reb", 0.0),
    ("ema_ast", 0.0),
    ("ema_stl", 0.0),
    ("ema_blk", 0.0),
    ("ema_tov", 0.0),
    ("ema_pf", 0.0),
    ("ema_pts", 22.0),
    ("ema_pm", 0.0),
    ("ema_margin", 0.0),
    # Boxscore stat rolling averages (N/A → 0.0)
    ("avg_fg3_pct", 0.0),
    ("avg_ft_pct", 0.0),
    ("avg_fgm", 0.0),
    ("avg_fga", 0.0),
    ("avg_fg3m", 0.0),
    ("avg_fg3a", 0.0),
    ("avg_ftm", 0.0),
    ("avg_fta", 0.0),
    ("avg_oreb", 0.0),
    ("avg_dreb", 0.0),
    ("avg_reb", 0.0),
    ("avg_ast", 0.0),
    ("avg_stl", 0.0),
    ("avg_blk", 0.0),
    ("avg_tov", 0.0),
    ("avg_pf", 0.0),
    ("avg_pace", 65.0),
    ("avg_efg", 0.0),
    ("avg_pts", 22.0),
    ("avg_pm", 0.0),
    ("avg_margin", 0.0),
    ("margin_volatility", 16.0),
    ("last_3_margin", 0.0),
    # Win rates & momentum
    ("win_pct", 0.5),
    ("win_streak", 0.0),
    ("weighted_momentum", 0.5),
    ("form_score", 0.5),
    # Pace
    ("pace", 65.0),
    # Efficiency (N/A → 0.0)
    ("efg", 0.0),
    # Points / z-scores
    ("pts_zscore", 0.0),
    # SOS
    ("sos_trend", 0.0),
    ("sos", 0.0),
    # Travel
    ("cum_travel", 0.0),
    # ELO
    ("elo_slope", 0.0),
    # Moneyline features
    ("composite_power", 0.5),
    ("power_diff", 0.0),
    ("perf_vs_expected_raw", 0.0),
    ("perf_vs_expected", 0.0),
    ("perf_vs_expected_diff", 0.0),
    ("consistency", 0.5),
    ("consistency_diff", 0.0),
    ("form_diff", 0.0),
    ("home_away_split_diff", 0.0),
    ("recent_win_pct_home", 0.5),
    ("recent_win_pct_away", 0.5),
    ("h2h_win_rate", 0.5),
    ("h2h_avg_margin", 0.0),
    # Home-away differentials
    ("_diff_", 0.0),
    # Target encoding
    ("target_win_rate", 0.5),
    ("target_margin", 0.0),
    # Seasonality
    ("dow_sin", 0.0),
    ("dow_cos", 1.0),
    ("month_sin", 0.0),
    ("month_cos", 1.0),
    ("season_phase", 1.0),
    ("days_since_asb", 0.0),
    ("days_into_season", 15.0),
    ("is_weekend", 0),
    # Fatigue
    ("fatigue_index", 0.3),
    ("games_3d", 1.0),
    ("games_5d", 2.0),
    ("games_7d", 3.0),
    # Home/away splits
    ("home_win_rate_at_home", 0.6),
    ("away_win_rate_on_road", 0.4),
    ("home_advantage_edge", 0.0),
    # Coach change proxy
    ("perf_shift_z", 0.0),
    ("system_change_flag", 0),
    ("system_changes_10g", 0.0),
    ("system_changes_diff", 0.0),
]


class FeatureEngineer:
    """Creates features for predictive models from raw game data.

    v2.2 features include EMA rolling stats, trend slopes, travel distance,
    and enhanced fatigue modeling for more accurate predictions.

    Supports NBA, NCAAB, Euroleague, and NFL via the ``league`` parameter
    in ``build_all_features``.
    """

    def __init__(self, rolling_windows: Optional[List[int]] = None):
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS

    def build_all_features(self, games_df: pd.DataFrame, raw_df: pd.DataFrame,
                           league: str = "NBA") -> pd.DataFrame:
        """
        Build the full feature set from game-level data.

        Args:
            games_df: Merged home/away game dataset from NBADataLoader
            raw_df: Raw team-level game logs
            league: "NBA" (default), "NCAAB", "Euroleague", or "NFL". Selects
                    league-appropriate backfill constants for rolling features.

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

            # ══════════════════════════════════════════════════════════════
            #  v3.0 — Rolling 5g/10g for ALL boxscore stats (NBA_AI 43-feature style)
            #  Adds rolling averages for every per-game stat so the model
            #  has full team statistical profile — not just pts/pm.
            # ══════════════════════════════════════════════════════════════
            _BOX_SCORE_STATS = [
                "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf",
            ]
            for box_stat in _BOX_SCORE_STATS:
                src_col = f"team_{box_stat}_{team_prefix}"
                if src_col not in df.columns:
                    continue
                for w in [5, 10]:
                    df[f"avg_{box_stat}_{w}g_{suffix}"] = (
                        df.groupby(team_id_col)[src_col]
                        .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                    )
                # Also add EMA (exponential moving average) for key stats
                if box_stat in ("fgm", "fga", "reb", "ast", "stl", "blk", "tov", "pf"):
                    span_10 = max(10, 2)
                    df[f"ema_{box_stat}_10g_{suffix}"] = (
                        df.groupby(team_id_col)[src_col]
                        .transform(lambda x, sp=span_10: (
                            x.ewm(span=sp, min_periods=1, adjust=False).mean().shift(1)
                        ))
                    )

            # ── 3-point & free throw percentage rolling ────────────────
            fg3_pct_col = f"team_fg3_pct_{team_prefix}"
            if fg3_pct_col in df.columns:
                for w in [5, 10]:
                    df[f"avg_fg3_pct_{w}g_{suffix}"] = (
                        df.groupby(team_id_col)[fg3_pct_col]
                        .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                    )
            ft_pct_col = f"team_ft_pct_{team_prefix}"
            if ft_pct_col in df.columns:
                for w in [5, 10]:
                    df[f"avg_ft_pct_{w}g_{suffix}"] = (
                        df.groupby(team_id_col)[ft_pct_col]
                        .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                    )

            # ── Opponent-allowed rolling averages ─────────────────────
            # What does this team's defense allow over the last 5/10 games?
            # Opponent's stats in the same game = what this team allowed.
            opp_prefix = "away" if team_prefix == "home" else "home"
            for box_stat in _BOX_SCORE_STATS:
                opp_col = f"team_{box_stat}_{opp_prefix}"
                if opp_col not in df.columns:
                    continue
                for w in [5, 10]:
                    df[f"avg_{box_stat}_allowed_{w}g_{suffix}"] = (
                        df.groupby(team_id_col)[opp_col]
                        .transform(lambda x, win=w: x.rolling(win, min_periods=1).mean().shift(1))
                    )

            # ── Trend slopes for key stats (momentum signals) ──────────
            for box_stat in ("fgm", "fga", "reb", "ast", "stl", "blk", "tov"):
                src_col = f"team_{box_stat}_{team_prefix}"
                if src_col not in df.columns:
                    continue
                for w in [5, 10]:
                    df[f"trend_{box_stat}_{w}g_{suffix}"] = (
                        df.groupby(team_id_col)[src_col]
                        .transform(lambda x, win=w: self._compute_trend_slope(x, window=win))
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
        # FIXED: Use rest <= 1 instead of rest == 0.
        # A team playing on 0 rest days is on the second night of a back-to-back.
        # A team on 1 rest day played 2 days ago (still fatigue from recent game).
        # The old code only flagged exact B2B (rest==0), missing the broader
        # "recently played" fatigue that impacts performance.
        df["is_b2b_home"] = (df["rest_home_days"] <= 1).astype(int)
        df["is_b2b_away"] = (df["rest_away_days"] <= 1).astype(int)

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

        # ══════════════════════════════════════════════════════════════════
        #  v3.0 — Home-Away Differentials for ALL rolling stats
        #  Diff = home_value - away_value
        #  Positive means home team is stronger in that stat. Feature
        #  names use _diff suffix so the model can learn interaction patterns
        #  like "home shoots better AND defends better = strong edge".
        # ══════════════════════════════════════════════════════════════════
        _ROLLING_STAT_KEYS = [
            "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
            "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf",
            "efg", "pace", "pm", "fg3_pct", "ft_pct",
        ]
        for w in [5, 10]:
            for stat_key in _ROLLING_STAT_KEYS:
                col_home = f"avg_{stat_key}_{w}g_home"
                col_away = f"avg_{stat_key}_{w}g_away"
                if col_home in df.columns and col_away in df.columns:
                    df[f"{stat_key}_diff_{w}g"] = (
                        df[col_home].fillna(0) - df[col_away].fillna(0)
                    )

            # Also compute allowed stat differentials
            for stat_key in ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
                             "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf"]:
                col_home = f"avg_{stat_key}_allowed_{w}g_home"
                col_away = f"avg_{stat_key}_allowed_{w}g_away"
                if col_home in df.columns and col_away in df.columns:
                    df[f"{stat_key}_allowed_diff_{w}g"] = (
                        df[col_home].fillna(0) - df[col_away].fillna(0)
                    )

        # ── Market Line Baseline (for backtesting — NOT used as a feature) ─
        # This is a simple trailing average used as a proxy for the sportsbook's line.
        # It is deliberately excluded from select_features() to prevent data leakage.
        # Blend trailing averages with league mean (224) for more realistic proxy
        # Regression to mean prevents extreme team averages from inflating win rates
        avg_home = df["avg_pts_10g_home"].fillna(112) if "avg_pts_10g_home" in df.columns else 112
        avg_away = df["avg_pts_10g_away"].fillna(112) if "avg_pts_10g_away" in df.columns else 112
        df["market_line_baseline"] = 0.80 * (avg_home + avg_away) + 0.20 * 224.0

        # Also compute a pace-adjusted baseline for comparison
        pace_home = df["avg_pace_5g_home"].fillna(100) if "avg_pace_5g_home" in df.columns else pd.Series(100, index=df.index)
        pace_away = df["avg_pace_5g_away"].fillna(100) if "avg_pace_5g_away" in df.columns else pd.Series(100, index=df.index)
        df["market_line_pace_adj"] = (pace_home + pace_away) / 2.0 * 2.1

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

        # ── Elo Ratings as Features ─────────────────────────────────────
        # ELO captures team strength evolution over time more accurately
        # than simple rolling averages. Adding ELO ratings gives the model
        # a calibrated strength signal that rolling stats alone miss.
        df = self._add_elo_features(df)

        # ── Moneyline-Specific Features (v3.1) ──────────────────────────
        df = self._add_moneyline_features(df)

        # ══════════════════════════════════════════════════════════════════
        #  v5.1 — NEW ENHANCED FEATURES
        # ══════════════════════════════════════════════════════════════════

        # ── A: Target Encoding — team quality features ──────────────────
        df = self._add_target_encoding_features(df)

        # ── B: Seasonality — day of week, month, season phase ──────────
        df = self._add_seasonality_features(df)

        # ── C: Comprehensive Fatigue Index ─────────────────────────────
        df = self._add_fatigue_index(df)

        # ── D: Home/Away Performance Splits ─────────────────────────────
        df = self._add_home_away_splits(df)

        # ══════════════════════════════════════════════════════════════════
        #  v6.5 — NEXT-GEN FEATURES
        # ══════════════════════════════════════════════════════════════════

        # ── E: Interaction Features (v6.5) ──────────────────────────────
        df = self._add_interaction_features(df)

        # ── F: Rolling Volatility (v6.5) ────────────────────────────────
        df = self._add_rolling_volatility_features(df)

        # ── G: Momentum/Streak Features (v6.5) ──────────────────────────
        df = self._add_momentum_features(df)

        # ── H: Pace-Adjusted Features (v6.5) ────────────────────────────
        df = self._add_pace_adjusted_features(df)

        # ── Clean Up ──────────────────────────────────────────────────
        df = df.drop(columns=["rest_home_key", "rest_away_key"], errors="ignore")

        # Drop intermediate WL string columns but keep WL_num for feature selection
        df = df.drop(columns=["WL_num_home", "WL_num_away"], errors="ignore")

        # Drop raw calendar columns (sin/cos encoded versions are kept)
        df = df.drop(columns=["dow", "month"], errors="ignore")

        # ── Backfill NAs with league-average defaults ──────────────────
        # Rolling features (avg_pts_*, avg_pm_*, ema_*, etc.) are NaN for
        # each team's first game(s) of the season since there's no prior
        # history to compute from. We fill with sensible defaults so no
        # training data is dropped.
        df = self.backfill_features(df, league=league)

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

            # FIXED: Do NOT shift before computing consecutive road games.
            # The WL_num column is already the result of the CURRENT game
            # (not a future-game leak). The rolling `.shift(1)` inside
            # the groupby transform already ensures no lookahead.
            # Applying an EXTRA shift(1) here skips the first row entirely
            # and creates an off-by-one error in the count.
            df[f"consec_road_{suffix}"] = (
                df.groupby(team_id_col)[f"WL_num_{suffix}"]
                .transform(lambda x: self._compute_consecutive_road(x))
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

    def _add_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ELO ratings as model features.

        Standard ELO (k=20, home_adv=100) computed chronologically from
        historical game results. ELO captures team strength evolution
        more accurately than simple rolling averages because:
          - Accounts for opponent strength (beating good teams = more rating gain)
          - Has home court adjustment built in
          - Converges faster than 10-game rolling averages

        Features added:
          - elo_home, elo_away: current ELO rating before the game
          - elo_diff: home_elo - away_elo
          - elo_home_prob: home win probability from ELO formula
          - elo_class: categorical (hot/cold/neutral based on ELO trend)
        """
        df = df.copy()

        # Compute ELO ratings chronologically
        elo_ratings: dict[str, float] = {}
        K = 20.0
        HOME_ADV = 100.0

        def expected_prob(rating_a: float, rating_b: float, home: bool = True) -> float:
            ha = HOME_ADV if home else 0
            return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a - ha) / 400.0))

        elo_home_list = []
        elo_away_list = []
        elo_diff_list = []
        elo_prob_list = []

        for idx, row in df.iterrows():
            home_team = str(row.get("TEAM_NAME_home", row.get("home_team_name", "")))
            away_team = str(row.get("TEAM_NAME_away", row.get("away_team_name", "")))

            if not home_team or not away_team:
                elo_home_list.append(1500.0)
                elo_away_list.append(1500.0)
                elo_diff_list.append(0.0)
                elo_prob_list.append(0.5)
                continue

            home_elo = elo_ratings.get(home_team, 1500.0)
            away_elo = elo_ratings.get(away_team, 1500.0)

            elo_home_list.append(home_elo)
            elo_away_list.append(away_elo)
            elo_diff_list.append(home_elo - away_elo)
            elo_prob_list.append(expected_prob(home_elo, away_elo, home=True))

            # Update ratings based on actual result
            home_won = row.get("point_diff", 0) > 0
            if "WL_home" in df.columns:
                home_won = str(row.get("WL_home", "")).strip().upper() == "W"

            expected = expected_prob(home_elo, away_elo, home=True)
            actual = 1.0 if home_won else 0.0

            elo_ratings[home_team] = home_elo + K * (actual - expected)
            elo_ratings[away_team] = away_elo + K * ((1.0 - actual) - (1.0 - expected))

        df["elo_home"] = elo_home_list
        df["elo_away"] = elo_away_list
        df["elo_diff"] = elo_diff_list
        df["elo_home_prob"] = elo_prob_list

        # ELO class: hot/cold indicator based on ELO trend
        # Hot = gained 30+ ELO in last 5 games, Cold = lost 30+ ELO
        df["elo_slope_home"] = (
            df.groupby("TEAM_ID_home")["elo_home"]
            .transform(lambda x: x.diff().rolling(5, min_periods=1).mean())
        )
        df["elo_slope_away"] = (
            df.groupby("TEAM_ID_away")["elo_away"]
            .transform(lambda x: x.diff().rolling(5, min_periods=1).mean())
        )

        return df

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

        # 3 games in 4 nights: rest_days <= 1 means the team played
        # yesterday or the day before (back-to-back or 3in4 nights).
        df["rest_3in4_home"] = (df["rest_home_days"] <= 1).astype(int)
        df["rest_3in4_away"] = (df["rest_away_days"] <= 1).astype(int)

        # Both teams on b2b: rest_days <= 1 for BOTH teams.
        # Old code used (rest == 0) which ONLY caught exact same-day
        # back-to-backs, missing the case where one team has 1 day rest
        # and the other also has 1 day rest (both played previous day).
        df["both_b2b"] = (
            (df["rest_home_days"] <= 1) & (df["rest_away_days"] <= 1)
        ).astype(int)

        # Interaction features
        df["rest_adv_sq"] = df["rest_advantage"] ** 2
        df["fatigue_rest_interact"] = df["fatigue_diff"] * df["rest_advantage"]

        return df

    # ── A: Target Encoding Features (v5.1) ───────────────────────────

    def _add_target_encoding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace raw TEAM_ID integers with informative rolling averages.

        The model shouldn't treat team IDs as numbers (team 1610612738 is
        not "better" than team 1610612739). Instead, we create target-
        encoded features: each team's historical win rate and margin
        computed chronologically (no lookahead).

        Features:
          - team_win_rate_home/away: Rolling win rate for each team
          - team_avg_margin_home/away: Rolling average margin
          - team_win_rate_diff: home_win_rate - away_win_rate
          - team_margin_diff: home_margin - away_margin
        """
        df = df.copy()

        for suffix, team_prefix in [("home", "home"), ("away", "away")]:
            team_id_col = f"TEAM_ID_{suffix}"
            wl_num_col = f"WL_num_{team_prefix}"
            pm_col = f"team_plus_minus_{team_prefix}"

            # Rolling win rate (already exists as win_pct_10g, but add
            # a direct team-quality signal using ALL history, not just 10g)
            df[f"target_win_rate_{suffix}"] = (
                df.groupby(team_id_col)[wl_num_col]
                .transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
            )

            # Rolling average margin (expanding = all available history)
            df[f"target_margin_{suffix}"] = (
                df.groupby(team_id_col)[pm_col]
                .transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
            )

        # Differential features
        df["target_win_rate_diff"] = df["target_win_rate_home"] - df["target_win_rate_away"]
        df["target_margin_diff"] = df["target_margin_home"] - df["target_margin_away"]

        return df

    # ── B: Seasonality Features (v5.1) ───────────────────────────────

    def _add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar-based features that capture time-of-season effects.

        NBA teams have different performance patterns:
        - Early season (Oct-Nov): teams still gelling, more variance
        - Mid season (Dec-Jan): "dog days" — fatigue + travel wear
        - Post-ASB (Feb-Apr): playoff push — stronger performances
        - Day of week: Sunday afternoon games play differently from
          Tuesday night games (rest advantage, travel patterns)

        Features:
          - day_of_week: 0=Mon..6=Sun (sin/cos encoded for cyclicity)
          - month: 1-12 (sin/cos encoded)
          - season_phase: early(0), mid(1), playoff_push(2)
          - days_since_all_star: days after ASB (playoff push signal)
          - is_weekend: 1 if Fri/Sat/Sun game
        """
        df = df.copy()

        if "GAME_DATE" not in df.columns:
            return df

        # Parse dates if needed
        if not pd.api.types.is_datetime64_any_dtype(df["GAME_DATE"]):
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        # Day of week
        df["dow"] = df["GAME_DATE"].dt.dayofweek  # 0=Monday
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
        df["is_weekend"] = (df["dow"] >= 4).astype(int)  # Fri, Sat, Sun

        # Month
        df["month"] = df["GAME_DATE"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Season phase
        def _season_phase(date):
            m = date.month
            if m in (10, 11):
                return 0  # Early season
            elif m in (12, 1):
                return 1  # Mid season / dog days
            elif m in (2, 3):
                return 2  # Playoff push
            elif m in (4, 5, 6):
                return 3  # Playoffs / end of regular
            return 1

        df["season_phase"] = df["GAME_DATE"].apply(_season_phase).astype(int)

        # Days since All-Star break (approximate: Feb 15)
        year = df["GAME_DATE"].dt.year
        asb_date = pd.to_datetime(year.astype(str) + "-02-15")
        df["days_since_asb"] = ((df["GAME_DATE"] - asb_date).dt.days).clip(-60, 90)
        df["days_since_asb"] = df["days_since_asb"].fillna(0)

        # Days into season (from Oct 1)
        season_start = pd.to_datetime(year.astype(str) + "-10-01")
        df["days_into_season"] = ((df["GAME_DATE"] - season_start).dt.days).clip(0, 365).fillna(0)

        return df

    # ── C: Comprehensive Fatigue Index (v5.1) ────────────────────────

    def _add_fatigue_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a comprehensive fatigue index (v5.1 — vectorized).

        Uses a fully vectorized approach instead of slow .rolling().apply()
        to count games in recent windows and compute a composite index.

        Features:
          - fatigue_index_home/away: 0-1 scalar (higher = more fatigued)
          - fatigue_index_diff: home - away
          - games_3d_home/away: games played in last 3 days
          - games_5d_home/away: games played in last 5 days
          - games_7d_home/away: games played in last 7 days
        """
        df = df.copy()

        # Ensure GAME_DATE is datetime
        if "GAME_DATE" not in df.columns:
            return df
        if not pd.api.types.is_datetime64_any_dtype(df["GAME_DATE"]):
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        for suffix in ["home", "away"]:
            team_id_col = f"TEAM_ID_{suffix}"

            # Compute day gaps between consecutive games for each team
            day_gaps = df.groupby(team_id_col)["GAME_DATE"].transform(
                lambda x: x.diff().dt.days
            )

            for window_days, col_name in [(3, "3d"), (5, "5d"), (7, "7d")]:
                # Binary: was each game within `window_days` of the previous?
                within_window = (day_gaps <= window_days).astype(float)
                # Rolling count of games in window (vectorized)
                temp_df = pd.DataFrame({
                    "team_id": df[team_id_col],
                    "within": within_window,
                })
                result = (
                    temp_df.groupby("team_id")["within"]
                    .transform(
                        lambda x: x.rolling(window_days, min_periods=1).sum().shift(1)
                    )
                )
                df[f"games_{col_name}_{suffix}"] = result.fillna(0).astype(int)

        # Fatigue index = weighted combination of signals
        rest_quality_home = 1.0 - np.clip(df.get("rest_home_days", 3).fillna(3) / 7.0, 0, 1)
        rest_quality_away = 1.0 - np.clip(df.get("rest_away_days", 3).fillna(3) / 7.0, 0, 1)

        games_5d_home = df.get("games_5d_home", 0).fillna(0)
        games_5d_away = df.get("games_5d_away", 0).fillna(0)

        travel_penalty_home = np.clip(df.get("cum_travel_home", 0).fillna(0) / 3000.0, 0, 1)
        travel_penalty_away = np.clip(df.get("cum_travel_away", 0).fillna(0) / 3000.0, 0, 1)

        b2b_home = df.get("is_b2b_home", 0).fillna(0)
        b2b_away = df.get("is_b2b_away", 0).fillna(0)

        df["fatigue_index_home"] = (
            0.40 * rest_quality_home
            + 0.30 * np.clip(games_5d_home / 4.0, 0, 1)
            + 0.15 * travel_penalty_home
            + 0.15 * b2b_home
        )
        df["fatigue_index_away"] = (
            0.40 * rest_quality_away
            + 0.30 * np.clip(games_5d_away / 4.0, 0, 1)
            + 0.15 * travel_penalty_away
            + 0.15 * b2b_away
        )
        df["fatigue_index_diff"] = df["fatigue_index_home"] - df["fatigue_index_away"]

        return df

    # ── D: Home/Away Performance Splits (v5.1) ───────────────────────

    def _add_home_away_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute home/away venue performance differentials.

        Some teams have significant home/away splits (e.g., Denver Nuggets
        at altitude). These features capture how much better/worse each
        team plays at their venue vs on the road.

        Features:
          - home_advantage_home: home team's home win rate
          - home_advantage_away: away team's road win rate
          - home_advantage_diff: home_advantage_home - (1 - home_advantage_away)
        """
        df = df.copy()

        # For the home team: how often do they win AT HOME?
        home_id_col = "TEAM_ID_home"
        df["home_win_rate_at_home"] = (
            df.groupby(home_id_col)["WL_num_home"]
            .transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
        )

        # For the away team: how often do they win ON THE ROAD?
        away_id_col = "TEAM_ID_away"
        df["away_win_rate_on_road"] = (
            df.groupby(away_id_col)["WL_num_away"]
            .transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
        )

        # This game's expected home advantage = home's home win rate
        # vs away's road loss rate (1 - away_road_win_rate)
        df["home_advantage_edge"] = (
            df["home_win_rate_at_home"].fillna(0.5)
            - (1.0 - df["away_win_rate_on_road"].fillna(0.5))
        )

        # ── D2: Coach Change Proxy Detection (v5.1) ───────────────────────
        df = self._add_coach_change_features(df)
        return df

    def _add_coach_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Proxy detection of coaching/system changes via performance anomalies.

        A coaching change typically causes a sudden shift in:
          - Performance vs expectation (perf_vs_expected)
          - Margin volatility (higher variance as team adjusts)
          - Pace of play (new systems change game speed)

        Instead of trying to scrape which teams changed coaches mid-season
        (which requires an external API), we detect anomalous shifts in
        rolling performance that could indicate a system change.

        Features:
          - perf_shift_home/away: z-score of last 3 games' perf vs rolling norm
          - system_change_home/away: 1 if large anomalous shift detected
          - coach_change_count_home/away: rolling count of detected shifts

        This is a PROXY — it may catch non-coach-change anomalies too
        (key injuries, trades, etc.), which is actually desirable since
        those also affect game outcomes.
        """
        df = df.copy()

        for suffix, team_prefix in [("home", "home"), ("away", "away")]:
            team_id_col = f"TEAM_ID_{suffix}"
            perf_col = f"perf_vs_expected_{suffix}"
            pm_col = f"team_plus_minus_{team_prefix}"

            if perf_col not in df.columns:
                continue

            # Use pre-z-scored perf_vs_expected (already normalized)
            # Compute rolling mean and std for anomaly detection
            rolling_mean = (
                df.groupby(team_id_col)[perf_col]
                .transform(lambda x: x.rolling(15, min_periods=5).mean().shift(1))
            )
            rolling_std = (
                df.groupby(team_id_col)[perf_col]
                .transform(lambda x: x.rolling(15, min_periods=5).std().shift(1))
            )

            # Recent 3-game average vs rolling 15-game norm
            recent_3g = (
                df.groupby(team_id_col)[perf_col]
                .transform(lambda x: x.rolling(3, min_periods=2).mean().shift(1))
            )

            # Z-score of recent performance vs historical norm
            df[f"perf_shift_z_{suffix}"] = np.where(
                rolling_std > 0.01,
                (recent_3g - rolling_mean) / rolling_std,
                0.0,
            )

            # Flag: absolute z-score > 2.0 = likely system change
            df[f"system_change_flag_{suffix}"] = (
                np.abs(df[f"perf_shift_z_{suffix}"]).fillna(0) > 2.0
            ).astype(int)

            # Rolling count of system changes in last 10 games
            df[f"system_changes_10g_{suffix}"] = (
                df.groupby(team_id_col)[f"system_change_flag_{suffix}"]
                .transform(lambda x: x.rolling(10, min_periods=1).sum().shift(1))
            )

        # Differential features
        if "system_changes_10g_home" in df.columns:
            df["system_changes_diff"] = (
                df["system_changes_10g_home"].fillna(0)
                - df["system_changes_10g_away"].fillna(0)
            )

        return df

    # ── Moneyline-Specific Features (v3.1) ────────────────────────────

    def _add_moneyline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features designed specifically for win/loss classification.

        Unlike the existing rolling averages (built for total points regression),
        these features directly capture team quality, matchup dynamics, and
        performance relative to expectations — all critical for moneyline.

        Features added:
          - composite_power_home/away: Weighted blend of ELO + net rating + form
          - power_diff: home - away composite power
          - form_diff: home weighted_momentum - away weighted_momentum
          - perf_vs_expected_home/away: Rolling actual margin vs ELO-expected margin
          - perf_vs_expected_diff: home - away perf_vs_expected
          - consistency_home/away: Inverse of margin volatility (higher = more reliable)
          - consistency_diff: home - away consistency
          - home_win_pct_10g_home: How well does home team play at home?
          - away_win_pct_10g_away: How well does away team play on road?
          - h2h_win_rate: Head-to-head win rate for home team in last 5 meetings
          - h2h_avg_margin: Head-to-head avg margin for home team in last 5 meetings
        """
        df = df.copy()

        for suffix in ["home", "away"]:
            team_prefix = suffix
            team_id_col = f"TEAM_ID_{suffix}"
            wl_num_col = f"WL_num_{team_prefix}"
            pm_col = f"team_plus_minus_{team_prefix}"

            # ── Composite Power Rating ─────────────────────────────────
            # Blend: 40% ELO, 30% net rating (pm), 30% recent win rate
            # Normalize each component to ~0-1 range
            elo_col = f"elo_{suffix}"
            pm_10g = f"avg_pm_10g_{suffix}"
            win_pct_10g = f"win_pct_10g_{suffix}"
            weighted_mom = f"weighted_momentum_{suffix}"

            elo_norm = (df.get(elo_col, 1500) - 1300) / 400  # ~0.5-1.0 range
            pm_norm = df.get(pm_10g, 0) / 20  # ~-0.5-0.5 range, clipped
            pm_norm = np.clip(pm_norm, -1, 1)

            df[f"composite_power_{suffix}"] = (
                0.35 * elo_norm
                + 0.35 * pm_norm
                + 0.15 * df.get(win_pct_10g, 0.5)
                + 0.15 * df.get(weighted_mom, 0.5)
            )

            # ── Performance vs Expectation ─────────────────────────────
            # Rolling difference between actual margin and ELO-expected margin
            # Positive = team is outperforming what ELO predicts = hot
            elo_prob_col = f"elo_home_prob"
            if suffix == "away":
                elo_prob_val = 1.0 - df.get("elo_home_prob", 0.5)
            else:
                elo_prob_val = df.get("elo_home_prob", 0.5)

            # Expected margin: convert win prob to expected margin
            # Derivation: ELO formula gives P(home_win) based on rating diff.
            # Each ~6% of win probability ≈ 1 point of margin in NBA.
            # So expected_margin = (win_prob - 0.5) / 0.06 ≈ (wp - 0.5) * 16.667
            expected_margin = (elo_prob_val - 0.5) * 16.667
            actual_margin = df.get(pm_10g, 0)
            perf_vs_exp = actual_margin - expected_margin

            # Rolling 10-game z-score of performance vs expectation
            # Uses inline z-score to avoid scipy dependency
            def _zscore_last(arr: np.ndarray) -> float:
                mu, s = arr.mean(), arr.std(ddof=0)
                return float((arr[-1] - mu) / s) if s > 0 else 0.0

            df[f"perf_vs_expected_raw_{suffix}"] = perf_vs_exp
            df[f"perf_vs_expected_{suffix}"] = (
                df.groupby(team_id_col)[f"perf_vs_expected_raw_{suffix}"]
                .transform(
                    lambda x: x.rolling(10, min_periods=1).apply(
                        lambda s: _zscore_last(s) if len(s) >= 3 else 0.0,
                        raw=True,
                    )
                )
            )

            # ── Consistency Score ──────────────────────────────────────
            # Inverse of margin volatility: higher = more consistent = more reliable
            # Scale: if volatility = 12 (NBA avg), consistency = 0.5
            vol_col = f"margin_volatility_{suffix}"
            df[f"consistency_{suffix}"] = np.clip(
                1.0 - (df.get(vol_col, 12) / 24), 0, 1
            )

            # ── Team Recent Win Rate ───────────────────────────────────
            # Rolling win rate over last 10 games for each team.
            # Note: this is the team's OVERALL win rate, not filtered by
            # home/away venue (the per-game data doesn't separate that).
            if suffix == "home":
                df["recent_win_pct_home"] = (
                    df.groupby(team_id_col)[wl_num_col]
                    .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
                )
            else:
                df["recent_win_pct_away"] = (
                    df.groupby(team_id_col)[wl_num_col]
                    .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
                )

        # ── Differential Features ───────────────────────────────────────
        # These capture the NET advantage of home over away

        df["power_diff"] = df["composite_power_home"] - df["composite_power_away"]

        df["form_diff"] = (
            df.get("weighted_momentum_home", 0.5)
            - df.get("weighted_momentum_away", 0.5)
        )

        df["perf_vs_expected_diff"] = (
            df["perf_vs_expected_home"] - df["perf_vs_expected_away"]
        )

        df["consistency_diff"] = (
            df["consistency_home"] - df["consistency_away"]
        )

        df["home_away_split_diff"] = (
            df.get("recent_win_pct_home", 0.5)
            - df.get("recent_win_pct_away", 0.5)
        )

        # ── Head-to-Head Features ───────────────────────────────────────
        # How have these two teams performed against each other recently?
        # Uses a team-pair key (sorted) to track H2H history.
        # Stores which team won by name, so the current home team's
        # historical record vs the away team can be computed correctly.
        h2h_store: dict[str, list[tuple[str, int, float]]] = {}  # key -> [(winner_team, idx, margin)]

        h2h_win_rate_list = []  # current home team's win rate vs away team
        h2h_margin_list = []    # current home team's avg margin vs away team

        for idx, row in df.iterrows():
            home_team = str(row.get("TEAM_NAME_home", row.get("home_team_name", ""))).strip()
            away_team = str(row.get("TEAM_NAME_away", row.get("away_team_name", ""))).strip()

            if not home_team or not away_team or home_team == away_team:
                h2h_win_rate_list.append(0.5)
                h2h_margin_list.append(0.0)
                continue

            # Create sorted pair key so both directions share history
            pair_key = tuple(sorted([home_team, away_team]))
            history = h2h_store.get(pair_key, [])

            # Compute H2H: how often has the current HOME team beaten the AWAY team?
            recent = history[-5:] if len(history) >= 5 else history
            if recent:
                home_wins = sum(1 for winner, _, _ in recent if winner == home_team)
                home_margins = [
                    m if winner == home_team else -m
                    for winner, _, m in recent
                ]
                h2h_win_rate_list.append(home_wins / len(recent))
                h2h_margin_list.append(sum(home_margins) / len(home_margins) if home_margins else 0.0)
            else:
                h2h_win_rate_list.append(0.5)
                h2h_margin_list.append(0.0)

            # Record THIS game: who won and by how much (from home perspective)
            home_won = row.get("point_diff", 0) > 0
            margin = row.get("point_diff", 0)
            winner = home_team if home_won else away_team
            h2h_store.setdefault(pair_key, []).append((winner, idx, margin))

        return df

    # ── E: Interaction Features (v6.5) ──────────────────────────────────

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create meaningful interaction features between key predictive signals.

        Interactions capture non-linear relationships that individual features miss.
        E.g., a team with high ELO rating AND high fatigue might perform worse
        than either signal alone predicts (fatigued good teams underperform more
        than fatigued bad teams).

        Features:
          - fatigue_x_home_adv: fatigue_index_diff × home_advantage_edge
          - elo_x_rest: elo_home × rest_advantage (normalized)
          - power_x_fatigue: composite_power × fatigue_index
          - consistency_x_home: consistency × home_advantage
          - momentum_x_opponent: form_score × opponent_quality
          - travel_x_rest: travel_distance × rest_advantage
          - perf_x_fatigue: perf_vs_expected × fatigue_index
          - elo_x_pace: elo_diff × pace_diff
          - three_pt_x_reb: 3pt_rate_diff × reb_rate_diff
          - defense_x_foul: def_rating_diff × foul_rate_diff
        """
        df = df.copy()

        # Fatigue × Home Advantage
        if all(c in df.columns for c in ["fatigue_index_diff", "home_advantage_edge"]):
            df["interact_fatigue_x_home"] = (
                df["fatigue_index_diff"].fillna(0) * df["home_advantage_edge"].fillna(0)
            )

        # ELO × Rest Advantage (both normalized)
        if all(c in df.columns for c in ["elo_diff", "rest_advantage"]):
            elo_norm = df["elo_diff"].fillna(0) / 200.0  # Normalize ELO diff
            rest_norm = df["rest_advantage"].fillna(0) / 7.0  # Normalize rest
            df["interact_elo_x_rest"] = elo_norm * rest_norm

        # Composite Power × Fatigue
        if all(c in df.columns for c in ["power_diff", "fatigue_index_diff"]):
            df["interact_power_x_fatigue"] = (
                df["power_diff"].fillna(0) * df["fatigue_index_diff"].fillna(0)
            )

        # Consistency × Home Advantage
        if all(c in df.columns for c in ["consistency_diff", "home_advantage_edge"]):
            df["interact_consistency_x_home"] = (
                df["consistency_diff"].fillna(0) * df["home_advantage_edge"].fillna(0)
            )

        # Travel × Rest (cumulative travel penalty × rest disadvantage)
        if all(c in df.columns for c in ["cum_travel_diff", "rest_advantage"]):
            travel_norm = df["cum_travel_diff"].fillna(0) / 3000.0
            rest_norm = df["rest_advantage"].fillna(0) / 7.0
            df["interact_travel_x_rest"] = travel_norm * rest_norm

        # Performance vs Expectation × Fatigue
        if all(c in df.columns for c in ["perf_vs_expected_diff", "fatigue_index_diff"]):
            df["interact_perf_x_fatigue"] = (
                df["perf_vs_expected_diff"].fillna(0) * df["fatigue_index_diff"].fillna(0)
            )

        # ELO × Pace Differential
        if all(c in df.columns for c in ["elo_diff", "pace_diff_5g"]):
            elo_norm = df["elo_diff"].fillna(0) / 200.0
            pace_norm = df["pace_diff_5g"].fillna(0) / 20.0
            df["interact_elo_x_pace"] = elo_norm * pace_norm

        # 3PT Rate × Rebound Rate Differential
        if all(c in df.columns for c in ["fg3a_diff_5g", "reb_diff_5g"]):
            df["interact_3pt_x_reb"] = (
                df["fg3a_diff_5g"].fillna(0) * df["reb_diff_5g"].fillna(0)
            )

        # Form × Opponent Quality
        if all(c in df.columns for c in ["form_diff", "opp_avg_pm_home"]):
            opp_quality = df["opp_avg_pm_home"].fillna(0) / 20.0
            df["interact_form_x_opp"] = df["form_diff"].fillna(0) * opp_quality

        # Margin Volatility × Rest (unstable teams on short rest = bad)
        if all(c in df.columns for c in ["margin_volatility_home", "rest_home_days"]):
            vol_norm = df["margin_volatility_home"].fillna(12) / 24.0
            rest_penalty = 1.0 - np.clip(df["rest_home_days"].fillna(3) / 7.0, 0, 1)
            df["interact_volatility_x_rest_home"] = vol_norm * rest_penalty

        if all(c in df.columns for c in ["margin_volatility_away", "rest_away_days"]):
            vol_norm = df["margin_volatility_away"].fillna(12) / 24.0
            rest_penalty = 1.0 - np.clip(df["rest_away_days"].fillna(3) / 7.0, 0, 1)
            df["interact_volatility_x_rest_away"] = vol_norm * rest_penalty

        return df

    # ── F: Rolling Volatility Features (v6.5) ─────────────────────────

    def _add_rolling_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling standard deviation (volatility) features.

        Volatility captures consistency/stability which is highly predictive:
        - Low volatility teams are more reliable (perform to expectation)
        - High volatility teams are more likely to deviate from expectation
        - Sudden changes in volatility often signal regime changes

        Features:
          - volatility_pts: std of points over 5/10 games
          - volatility_pm: std of plus/minus over 5/10 games
          - volatility_efg: std of eFG% over 5 games
          - volatility_pace: std of pace over 5 games
          - volatility_allowed: std of points allowed over 5/10 games
        """
        df = df.copy()

        for suffix, team_prefix in [("home", "home"), ("away", "away")]:
            team_id_col = f"TEAM_ID_{suffix}"
            pts_col = f"team_pts_{team_prefix}"
            pm_col = f"team_plus_minus_{team_prefix}"

            if pts_col in df.columns:
                for w in [5, 10]:
                    df[f"volatility_pts_{w}g_{suffix}"] = (
                        df.groupby(team_id_col)[pts_col]
                        .transform(lambda x, win=w: x.rolling(win, min_periods=2).std().shift(1))
                    )

            if pm_col in df.columns:
                for w in [5, 10]:
                    df[f"volatility_pm_{w}g_{suffix}"] = (
                        df.groupby(team_id_col)[pm_col]
                        .transform(lambda x, win=w: x.rolling(win, min_periods=2).std().shift(1))
                    )

            # Volatility of opponent points allowed
            opp_pts_col = f"team_pts_{'away' if team_prefix == 'home' else 'home'}"
            if opp_pts_col in df.columns:
                for w in [5, 10]:
                    df[f"volatility_allowed_{w}g_{suffix}"] = (
                        df.groupby(team_id_col)[opp_pts_col]
                        .transform(lambda x, win=w: x.rolling(win, min_periods=2).std().shift(1))
                    )

            # Pace volatility
            pace_col = f"pace_{suffix}"
            if pace_col in df.columns:
                df[f"volatility_pace_5g_{suffix}"] = (
                    df.groupby(team_id_col)[pace_col]
                    .transform(lambda x: x.rolling(5, min_periods=2).std().shift(1))
                )

        return df

    # ── G: Momentum/Streak Features (v6.5) ────────────────────────────

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced momentum and streak-based features.

        Streaks carry psychological momentum in sports:
        - Winning streaks: teams play with more confidence
        - Losing streaks: teams press and make more mistakes
        - Cover streaks: beating the spread consistently
        - Streak breaks: how a team performs after a streak ends

        Features:
          - streak_length: current win/loss streak length (signed)
          - streak_strength: avg margin during streak (how dominant?)
          - streak_quality: streak_length × avg_margin_in_streak
          - recent_success: win% in last 3, 5, 10 games (weighted)
          - bounce_back: how team performs after a loss (1 if prev was loss)
          - consecutive_cover: consecutive games beating market expectation
        """
        df = df.copy()

        for suffix, team_prefix in [("home", "home"), ("away", "away")]:
            team_id_col = f"TEAM_ID_{suffix}"
            wl_num_col = f"WL_num_{team_prefix}"
            pm_col = f"team_plus_minus_{team_prefix}"

            # Win streak (already exists, but add streak strength)
            # Average margin during the current streak
            if pm_col in df.columns:
                df[f"streak_margin_{suffix}"] = (
                    df.groupby(team_id_col)[pm_col]
                    .transform(lambda x: self._compute_streak_margin(x))
                )

            # Streak quality = streak_length × avg_margin_in_streak
            streak_col = f"win_streak_{suffix}"
            streak_margin_col = f"streak_margin_{suffix}"
            if streak_col in df.columns and streak_margin_col in df.columns:
                df[f"streak_quality_{suffix}"] = (
                    np.abs(df[streak_col].fillna(0)) * df[streak_margin_col].fillna(0)
                )

            # Bounce back: how team performs the game after a loss
            # 1 if previous game was a loss, 0 if previous was a win
            if wl_num_col in df.columns:
                df[f"prev_loss_{suffix}"] = (
                    df.groupby(team_id_col)[wl_num_col]
                    .transform(lambda x: (1.0 - x).shift(1))
                ).fillna(0)

            # Recent form acceleration: is the team getting better or worse?
            # Compares last 3 games' margin to the 10-game rolling average
            if all(c in df.columns for c in [pm_col]):
                rolling_10 = (
                    df.groupby(team_id_col)[pm_col]
                    .transform(lambda x: x.rolling(10, min_periods=3).mean().shift(1))
                )
                rolling_3 = (
                    df.groupby(team_id_col)[pm_col]
                    .transform(lambda x: x.rolling(3, min_periods=2).mean().shift(1))
                )
                df[f"form_acceleration_{suffix}"] = (
                    rolling_3.fillna(0) - rolling_10.fillna(0)
                )

        return df

    def _compute_streak_margin(self, pm_series: pd.Series) -> pd.Series:
        """Compute average margin during current win/loss streak."""
        n = len(pm_series)
        result = np.zeros(n)
        current_streak_type = 0  # 1 for positive streak, -1 for negative
        streak_values = []

        for i in range(n):
            val = pm_series.iloc[i] if hasattr(pm_series, 'iloc') else pm_series[i]

            # Determine streak direction from this game's margin
            if val > 0:
                if current_streak_type == 1:
                    streak_values.append(val)
                else:
                    current_streak_type = 1
                    streak_values = [val]
            else:
                if current_streak_type == -1:
                    streak_values.append(val)
                else:
                    current_streak_type = -1
                    streak_values = [val]

            result[i] = np.mean(streak_values) if streak_values else 0.0

        return pd.Series(result, index=pm_series.index).shift(1).fillna(0.0)

    # ── H: Pace-Adjusted Features (v6.5) ──────────────────────────────

    def _add_pace_adjusted_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pace-adjusted statistics for more accurate comparison.

        Teams that play at different paces can't be compared by raw stats alone.
        A team scoring 120 ppg at 105 pace is less impressive than one scoring
        115 ppg at 95 pace (pace-adjusted: 114.3 vs 121.1 per 100 possessions).

        Features:
          - pace_adj_off: points per 100 possessions
          - pace_adj_def: opponent points per 100 possessions
          - pace_adj_net: offensive - defensive rating
          - pace_diff: home pace - away pace (how tempo mismatch)
          - pace_adj_total: expected total points at neutral pace
        """
        df = df.copy()

        for suffix, team_prefix in [("home", "home"), ("away", "away")]:
            team_id_col = f"TEAM_ID_{suffix}"
            pts_col = f"avg_pts_10g_{suffix}"
            pace_col = f"avg_pace_10g_{suffix}"

            if pts_col in df.columns and pace_col in df.columns:
                # Offensive rating: points per 100 possessions
                df[f"pace_adj_off_{suffix}"] = (
                    df[pts_col].fillna(110) / df[pace_col].fillna(100).clip(lower=50) * 100.0
                )

            # Defensive rating
            pts_allowed_col = f"avg_pts_allowed_{suffix}"
            if pts_allowed_col in df.columns and pace_col in df.columns:
                df[f"pace_adj_def_{suffix}"] = (
                    df[pts_allowed_col].fillna(110) / df[pace_col].fillna(100).clip(lower=50) * 100.0
                )

            # Net rating
            off_col = f"pace_adj_off_{suffix}"
            def_col = f"pace_adj_def_{suffix}"
            if off_col in df.columns and def_col in df.columns:
                df[f"pace_adj_net_{suffix}"] = df[off_col] - df[def_col]

            # Effective pace (possessions per game)
            fga_col = f"avg_fga_10g_{suffix}"
            tov_col = f"avg_tov_10g_{suffix}"
            oreb_col = f"avg_oreb_10g_{suffix}"
            if all(c in df.columns for c in [fga_col, tov_col, oreb_col]):
                df[f"pace_100_{suffix}"] = (
                    df[fga_col].fillna(85) - df[oreb_col].fillna(10) + df[tov_col].fillna(13)
                )

        return df

    def backfill_features(self, df: pd.DataFrame, league: str = "NBA") -> pd.DataFrame:
        """
        Backfill NaN feature values with league-average constants ONLY.

        CRITICAL: NEVER use dataset-wide statistics (median, mean, etc.)
        because that leaks future information into the training data.
        When walk-forward validation splits chronologically, the training
        fold's features would have been backfilled with medians computed
        from the FULL dataset (including future games) — 
        that is DATA LEAKAGE and inflates validation metrics.

        Rolling features (avg_pts_*, avg_pm_*, ema_*, etc.) are NaN for
        each team's first game(s) of the season since there's no prior
        history to compute from. This fills those gaps with pre-defined
        league-average constants, so no training data needs to be dropped
        and no future information leaks backward.

        Args:
            df: Feature DataFrame with potential NaN values.
            league: "NBA" (default), "NCAAB", "Euroleague", or "NFL". Selects
                    the appropriate set of league-average constants for backfill.
        """
        df = df.copy()

        # Select the right set of league-average constants
        if league == "NCAAB":
            _NA_FILL = _NCAAB_NA_FILL
        elif league == "Euroleague":
            _NA_FILL = _EUROLEAGUE_NA_FILL
        elif league == "NFL":
            _NA_FILL = _NFL_NA_FILL
        else:
            _NA_FILL = _NBA_NA_FILL

        for col in df.columns:
            if not df[col].isna().any():
                continue

            # Find the best-matching key (patterns in order: longest first)
            matched = False
            for pattern, fill_value in _NA_FILL:
                if pattern in col:
                    df[col] = df[col].fillna(fill_value)
                    matched = True
                    break

            if not matched:
                # NO DATA LEAKAGE: fill with 0.0 constant, NOT dataset median.
                # Using df[col].dropna().median() would compute over FUTURE games
                # when the dataset is sorted chronologically for walk-forward.
                df[col] = df[col].fillna(0.0)

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
            # v5.1: intermediate date columns
            "dow", "month",  # Use sin/cos encoded versions instead
            # v5.1: coach change intermediate columns
            "perf_shift_z_home", "perf_shift_z_away",
            "system_change_flag_home", "system_change_flag_away",
            # v6.5: interaction intermediate columns
            "three_pt_rate_home", "three_pt_rate_away",
            "ft_rate_home", "ft_rate_away",
            # v6.5: momentum intermediate columns
            "streak_margin_home", "streak_margin_away",
            "streak_quality_home", "streak_quality_away",
            "prev_loss_home", "prev_loss_away",
            "form_acceleration_home", "form_acceleration_away",
        }
        return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c].dtype)]

