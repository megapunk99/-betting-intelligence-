#!/usr/bin/env python3
"""
Forward Test — Real Market Odds Validation.

Trains models on all 695 historical NBA games, then predicts UPCOMING
games using real market lines from TheOddsAPI. Shows you what your model
thinks vs what the sportsbooks are actually offering.

Usage:
    export ODDS_API_KEY="your_key_here"
    python tools/forward_test.py                          # Full run
    python tools/forward_test.py --model totals           # Totals only
    python tools/forward_test.py --model moneyline        # Moneyline only

Key difference from backtest: This uses REAL MARKET LINES from sportsbooks
instead of a trailing-average baseline. The win rate you see here is what
actually matters for profitability."""

import sys
import os
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure src/ and project root are on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))  # For data/odds_fetcher

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer
from betting_intel.models.predictors import (
    TotalPointsPredictor, MomentumModel,
)
from betting_intel.config import MIN_EDGE_THRESHOLD
# ---- Team ID mapping (duck typing — self-contained, no dependency on data/odds_fetcher) ----
# This maps short team names ("Celtics", "Lakers") to NBA API team IDs.
# Used by build_feature_row_for_game to look up historical stats.
SHORT_NAME_TO_TEAM_ID: dict[str, int] = {
    "Hawks": 1610612737, "Celtics": 1610612738, "Nets": 1610612751,
    "Hornets": 1610612766, "Bulls": 1610612741, "Cavaliers": 1610612739,
    "Mavericks": 1610612742, "Nuggets": 1610612743, "Pistons": 1610612765,
    "Warriors": 1610612744, "Rockets": 1610612745, "Pacers": 1610612754,
    "Clippers": 1610612746, "Lakers": 1610612747, "Grizzlies": 1610612763,
    "Heat": 1610612748, "Bucks": 1610612749, "Timberwolves": 1610612750,
    "Pelicans": 1610612740, "Knicks": 1610612752, "Thunder": 1610612760,
    "Magic": 1610612753, "76ers": 1610612755, "Suns": 1610612756,
    "Trail Blazers": 1610612757, "Kings": 1610612758, "Spurs": 1610612759,
    "Raptors": 1610612761, "Jazz": 1610612762, "Wizards": 1610612764,
}

# ---- Odds API import (try both src and root-level paths) ----
try:
    from data.odds_fetcher import OddsAPIClient, OddsGame
    ODDS_AVAILABLE = True
except ImportError:
    ODDS_AVAILABLE = False

# Player injury impact module
try:
    from betting_intel.data.player_injury import PlayerInjuryFetcher, GameInjuryData
    INJURY_AVAILABLE = True
except ImportError:
    INJURY_AVAILABLE = False

# ESPN injury integrator — merges official status with prop-based detection
try:
    from betting_intel.data.espn_injury_integrator import ESPNInjuryIntegrator, MergedGameInjuryData
    ESPN_INTEGRATOR_AVAILABLE = True
except ImportError:
    ESPN_INTEGRATOR_AVAILABLE = False

# Injury adjuster — adjusts features for missing players
try:
    from betting_intel.data.injury_adjuster import InjuryAdjuster
    INJURY_ADJUSTER_AVAILABLE = True
except ImportError:
    INJURY_ADJUSTER_AVAILABLE = False

# ---- ANSI Colors ----
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ---- Data Models for Forward-Test Results ----
@dataclass
class ForwardPrediction:
    """A single prediction for an upcoming game vs the real market line."""
    game_date: str
    matchup: str
    home_team: str
    away_team: str

    # Totals
    model_total: float | None = None
    market_total: float | None = None
    total_edge_pct: float | None = None
    total_verdict: str | None = None          # "OVER", "UNDER", or "no edge"

    # Moneyline
    home_win_prob: float | None = None        # Model's home win probability
    market_home_implied: float | None = None  # Market's no-vig home win probability
    home_ml_edge: float | None = None
    away_win_prob: float | None = None
    market_away_implied: float | None = None
    away_ml_edge: float | None = None
    ml_verdict: str | None = None             # Which side to bet, if any

    # Staking (quarter-Kelly on $10k bankroll)
    recommended_stake: float = 0.0
    kelly_fraction: float = 0.0

    # Raw odds data
    home_ml_raw: float | None = None
    away_ml_raw: float | None = None
    over_odds_raw: float | None = None
    under_odds_raw: float | None = None

    # Explanations
    explanation: list[str] = field(default_factory=list)


# ============================================================================
#  Core Logic
# ============================================================================

def load_and_prepare_data() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Load historical games, engineer features, and return clean data + full feature matrix."""
    print("  Loading historical NBA data...")
    loader = NBADataLoader()
    fe = FeatureEngineer()

    raw_df = loader.load_game_logs()
    games_df = loader.build_game_dataset(raw_df)
    raw_df = loader.compute_rest_days(raw_df)

    print(f"    Games loaded: {len(games_df):,}")
    print(f"    Date range:   {games_df['GAME_DATE'].min().date()} to {games_df['GAME_DATE'].max().date()}")

    print("  Engineering features...")
    feature_df = fe.build_all_features(games_df, raw_df)
    feature_cols = fe.select_features(feature_df)

    print(f"    Features: {len(feature_cols)}")
    print(f"    Rows:     {len(feature_df):,}")

    # Create home_win target
    feature_df["home_win"] = (feature_df["point_diff"] > 0).astype(int)

    # After backfill_features in the FeatureEngineer, all NaN features should
    # be filled with league-average defaults — no rows should be dropped.
    clean_df = feature_df.copy()
    remaining_nas = clean_df[feature_cols].isna().sum().sum()
    if remaining_nas:
        print(f"    {YELLOW}Warning: {remaining_nas} NaN values remain after backfill{RESET}")
        clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2).copy()
        print(f"    Clean:    {len(clean_df):,} rows (dropped {len(feature_df) - len(clean_df)})")
    else:
        print(f"    Clean:    {len(clean_df):,} rows (all features backfilled)")

    return clean_df, feature_cols, feature_df


def train_models(df: pd.DataFrame, feature_cols: list[str], calibrated: bool = False) -> dict:
    """Train all models on the FULL historical dataset."""
    print("\n  Training models on full dataset...")

    models = {}
    X = df[feature_cols].dropna()
    y_total = df.loc[X.index, "total_points"]
    y_win = df.loc[X.index, "home_win"]

    print(f"    Training samples: {len(X):,}")

    # Ridge for totals
    ridge = TotalPointsPredictor("ridge")
    ridge.fit(X.values, y_total.values)
    train_preds = ridge.predict(X.values)
    train_mae = float(np.mean(np.abs(train_preds - y_total.values)))
    train_bias = float(np.mean(train_preds - y_total.values))
    print(f"    [Totals] Ridge:   MAE={train_mae:.1f}, Bias={train_bias:+.2f}")
    models["totals_ridge"] = ridge

    # XGBoost for totals (if available)
    try:
        xgb = TotalPointsPredictor("xgboost")
        xgb.fit(X.values, y_total.values)
        xgb_preds = xgb.predict(X.values)
        xgb_mae = float(np.mean(np.abs(xgb_preds - y_total.values)))
        xgb_bias = float(np.mean(xgb_preds - y_total.values))
        print(f"    [Totals] XGBoost: MAE={xgb_mae:.1f}, Bias={xgb_bias:+.2f}")
        models["totals_xgboost"] = xgb
    except Exception as e:
        print(f"    [Totals] XGBoost: SKIPPED ({e})")

    # Momentum (LogisticRegression) for moneyline
    label = "Platt Calibrated" if calibrated else "Uncalibrated"
    momentum = MomentumModel("logistic", calibrate=calibrated)
    momentum.fit(X.values, y_win.values)
    win_acc = float(np.mean(momentum.predict(X.values) == y_win.values))
    print(f"    [ML]     Momentum ({label}): Train Acc={win_acc:.1%}")
    models["ml_momentum"] = momentum

    return models


def fetch_upcoming_games(api_key: str = "") -> list:
    """Fetch upcoming NBA games with real odds from TheOddsAPI."""
    if not api_key:
        print(f"  {YELLOW}[!] No ODDS_API_KEY found. Set it in your environment{RESET}")
        return []

    if not ODDS_AVAILABLE:
        print(f"  {YELLOW}[!] OddsAPIClient not available (data/odds_fetcher.py not found){RESET}")
        return []

    try:
        client = OddsAPIClient(api_key=api_key)
        if not client._configured:
            print(f"  {YELLOW}[!] Invalid API key{RESET}")
            return []

        games = client.get_upcoming_games_with_odds(
            sport="basketball_nba",
            markets="h2h,spreads,totals",
            use_cache=False,
        )
        print(f"    Fetched {len(games)} upcoming games with real odds")
        return games

    except Exception as e:
        print(f"  {RED}[!] OddsAPI error: {e}{RESET}")
        return []


def _haversine(loc1, loc2) -> float:
    """Haversine distance between two (lat, lon) points in miles."""
    R = 3959.0
    lat1, lon1 = np.radians(loc1)
    lat2, lon2 = np.radians(loc2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


NBA_TEAM_CENTERS: dict[str, tuple[float, float]] = {
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

NBA_TEAM_TZ: dict[str, int] = {
    "Celtics": -5, "Nets": -5, "Knicks": -5, "76ers": -5, "Wizards": -5,
    "Hawks": -5, "Heat": -5, "Hornets": -5, "Magic": -5, "Raptors": -5,
    "Pistons": -5, "Pacers": -5, "Cavaliers": -5, "Bulls": -6,
    "Bucks": -6, "Timberwolves": -6, "Pelicans": -6, "Thunder": -6,
    "Mavericks": -6, "Rockets": -6, "Grizzlies": -6, "Spurs": -6,
    "Jazz": -7, "Nuggets": -7, "Suns": -7, "Trail Blazers": -8,
    "Kings": -8, "Warriors": -8, "Lakers": -8, "Clippers": -8,
}


def _team_feature_value(
    last_row: pd.Series,
    col_name: str,
    team_was_home: bool,
) -> float:
    """Extract a team-specific feature value from their last game row.

    If the feature is e.g. `avg_pts_5g_home` and the team was home in their
    last game, the value is `last_row[avg_pts_5g_home]`. If they were away,
    the value is in `last_row[avg_pts_5g_away]` — so we swap the suffix.
    """
    if col_name.endswith("_home"):
        base = col_name[:-5]  # Remove '_home'
        suffix = "_home" if team_was_home else "_away"
        return last_row.get(base + suffix, 0.0)
    elif col_name.endswith("_away"):
        base = col_name[:-5]  # Remove '_away'
        suffix = "_home" if team_was_home else "_away"
        return last_row.get(base + suffix, 0.0)
    else:
        return last_row.get(col_name, 0.0)


def build_feature_row_for_game(
    game,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict | None:
    """
    Build a feature vector for an upcoming game using the already-computed
    FeatureEngineer matrix.

    For each team, finds their MOST RECENT game in the full feature matrix
    and extracts their team-specific features (rolling avgs, EMA, momentum,
    etc.) — correctly handling whether they were home or away in that game.

    Only game-level features (rest days, travel, fatigue) are computed fresh
    for the upcoming matchup.
    """
    if feature_df is None or feature_df.empty:
        return None

    home_short = game.home_team_short
    away_short = game.away_team_short
    home_id = SHORT_NAME_TO_TEAM_ID.get(home_short)
    away_id = SHORT_NAME_TO_TEAM_ID.get(away_short)

    if not home_id or not away_id:
        return None

    sorted_df = feature_df.sort_values("GAME_DATE")

    # Find last row where each team played
    home_rows = sorted_df[
        (sorted_df["TEAM_ID_home"] == home_id) |
        (sorted_df["TEAM_ID_away"] == home_id)
    ]
    away_rows = sorted_df[
        (sorted_df["TEAM_ID_home"] == away_id) |
        (sorted_df["TEAM_ID_away"] == away_id)
    ]

    if home_rows.empty or away_rows.empty:
        return None

    home_last = home_rows.iloc[-1]
    away_last = away_rows.iloc[-1]

    home_was_home = home_last["TEAM_ID_home"] == home_id
    away_was_home = away_last["TEAM_ID_home"] == away_id

    # Extract team names
    home_team_name = str(home_last.get("TEAM_NAME_home" if home_was_home else "TEAM_NAME_away", ""))
    away_team_name = str(away_last.get("TEAM_NAME_home" if away_was_home else "TEAM_NAME_away", ""))

    # ── Gather all features ──
    feature_row: dict[str, float] = {}

    for col in feature_cols:
        if col.endswith("_home"):
            # Home team's feature
            feature_row[col] = _team_feature_value(home_last, col, home_was_home)
        elif col.endswith("_away"):
            # Away team's feature
            feature_row[col] = _team_feature_value(away_last, col, away_was_home)
        else:
            # Game-level feature — compute fresh or use from home team's last game
            feature_row[col] = home_last.get(col, 0.0)

    # ── Override game-level features with fresh upcoming-game values ──

    # Most recent game dates
    home_last_date = pd.Timestamp(home_last["GAME_DATE"])
    away_last_date = pd.Timestamp(away_last["GAME_DATE"])

    # Upcoming game date
    try:
        game_date = pd.Timestamp(game.commence_time[:10])
    except Exception:
        game_date = max(home_last_date, away_last_date) + pd.Timedelta(days=1)

    # Rest days for upcoming game
    home_rest = max(0, min((game_date - home_last_date).days, 14))
    away_rest = max(0, min((game_date - away_last_date).days, 14))

    override_features = {
        "rest_home_days": float(home_rest),
        "rest_away_days": float(away_rest),
        "rest_advantage": float(home_rest - away_rest),
        "is_b2b_home": 1.0 if home_rest == 0 else 0.0,
        "is_b2b_away": 1.0 if away_rest == 0 else 0.0,
        "both_b2b": 1.0 if home_rest == 0 and away_rest == 0 else 0.0,
        "fatigue_home": np.clip(1.0 / (home_rest + 0.5), 0, 2),
        "fatigue_away": np.clip(1.0 / (away_rest + 0.5), 0, 2),
        "fatigue_diff": np.clip(1.0 / (home_rest + 0.5), 0, 2) - np.clip(1.0 / (away_rest + 0.5), 0, 2),
        "rest_home_sq": float(home_rest ** 2),
        "rest_away_sq": float(away_rest ** 2),
        "rest_3in4_home": 1.0 if home_rest <= 1 else 0.0,
        "rest_3in4_away": 1.0 if away_rest <= 1 else 0.0,
    }

    # Travel features
    travel_distance = _haversine(
        NBA_TEAM_CENTERS.get(home_team_name, (40.0, -95.0)),
        NBA_TEAM_CENTERS.get(away_team_name, (40.0, -95.0)),
    )
    tz_diff = abs(NBA_TEAM_TZ.get(home_team_name, -5) - NBA_TEAM_TZ.get(away_team_name, -5))

    override_features["travel_distance"] = travel_distance
    override_features["travel_distance_norm"] = travel_distance / 3000.0
    override_features["tz_diff"] = float(tz_diff)

    # Rest advantage interaction
    override_features["rest_adv_sq"] = float((home_rest - away_rest) ** 2)
    override_features["fatigue_rest_interact"] = override_features["fatigue_diff"] * (home_rest - away_rest)

    # Apply overrides
    for k, v in override_features.items():
        if k in feature_cols:
            feature_row[k] = v

    # Override game-level `pace` as average of team-level pace features
    # (avoids inflated values from post-game game-level pace column)
    if "pace" in feature_cols:
        pace_home = feature_row.get("pace_home", 100)
        pace_away = feature_row.get("pace_away", 100)
        feature_row["pace"] = (pace_home + pace_away) / 2

    # Set cumulative travel and road-trip features to 0 for upcoming games
    # (extracting from home team's last game row gives stale values)
    for stale_col in ["cum_travel_home", "cum_travel_away", "cum_travel_diff",
                      "consec_road_away", "road_trip_length", "long_road_trip"]:
        if stale_col in feature_cols:
            feature_row[stale_col] = 0.0

    return feature_row


def build_prediction_features(
    odds_games: list,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, dict]:
    """
    Build feature vectors for each upcoming game using the already-computed
    FeatureEngineer matrix. Extracts each team's most recent feature vector
    and overrides game-level features (rest, travel) for the upcoming matchup.
    """
    print("  Building feature vectors for upcoming games...")
    feature_map = {}
    loaded = 0
    skipped = 0

    for game in odds_games:
        feature_row = build_feature_row_for_game(game, feature_df, feature_cols)
        if feature_row is not None:
            key = game.id or f"{game.away_team_short}@{game.home_team_short}"
            feature_map[key] = {
                "features": feature_row,
                "game": game,
            }
            loaded += 1
        else:
            skipped += 1

    print(f"    Feature vectors built: {loaded}  Skipped: {skipped}")
    return feature_map


# ── Feature-to-English explanation helpers ──

_FEATURE_DESC: dict[str, str] = {
    # Rolling scoring averages
    "avg_pts_5g": "pts/100poss (last 5g)",
    "avg_pts_10g": "pts/100poss (last 10g)",
    "avg_pts_20g": "pts/100poss (last 20g)",
    "avg_pts_allowed_5g": "pts allowed/100poss (last 5g)",
    "avg_pts_allowed_10g": "pts allowed/100poss (last 10g)",
    "avg_pts_allowed_20g": "pts allowed/100poss (last 20g)",
    "avg_margin_5g": "avg margin (last 5g)",
    "avg_margin_10g": "avg margin (last 10g)",
    "avg_margin_20g": "avg margin (last 20g)",
    # Pace & efficiency
    "pace": "game pace",
    "pace_home": "home pace",
    "pace_away": "away pace",
    "pace_10g": "pace (last 10g)",
    "avg_pace_5g": "avg pace (5g)",
    "avg_pace_10g": "avg pace (10g)",
    "eFG": "effective FG%",
    "eFG_10g": "eFG% (last 10g)",
    # Rest & scheduling
    "rest_home_days": "home rest",
    "rest_away_days": "away rest",
    "rest_advantage": "rest advantage (home-away)",
    "rest_adv_sq": "rest advantage^2",
    "is_b2b_home": "home on B2B",
    "is_b2b_away": "away on B2B",
    "both_b2b": "both teams B2B",
    "rest_home_sq": "home rest^2",
    "rest_away_sq": "away rest^2",
    "rest_3in4_home": "home: 3 games in 4 days",
    "rest_3in4_away": "away: 3 games in 4 days",
    "fatigue_home": "home fatigue",
    "fatigue_away": "away fatigue",
    "fatigue_diff": "fatigue difference",
    "fatigue_rest_interact": "fatigue x rest interact",
    # Travel
    "travel_distance": "travel distance (miles)",
    "travel_distance_norm": "travel distance (norm)",
    "tz_diff": "timezone diff (hours)",
    "cum_travel_home": "cumulative home travel",
    "cum_travel_away": "cumulative away travel",
    "cum_travel_diff": "cumulative travel diff",
    "consec_road_away": "consecutive road games",
    "road_trip_length": "road trip length",
    "long_road_trip": "long road trip flag",
    # Momentum & form
    "momentum_10g": "momentum (net rating, 10g)",
    "margin_volatility_10g": "margin volatility (10g)",
    "trend_10g": "trend (10g)",
    "wtd_momentum_10g": "weighted momentum (10g)",
    "zscore_pts_10g": "pts z-score (10g)",
    "zscore_margin_10g": "margin z-score (10g)",
    "form_score": "composite form score",
    # Win percentages
    "home_win_pct_5g": "home win% (5g)",
    "home_win_pct_10g": "home win% (10g)",
    "road_win_pct_10g": "road win% (10g)",
    "win_pct_10g": "win% (10g)",
    # Advanced rate stats
    "three_pt_rate_10g": "3PT attempt rate (10g)",
    "fta_rate_10g": "FTA rate (10g)",
    "assist_ratio_10g": "assist ratio (10g)",
    "ts_pct_10g": "true shooting% (10g)",
    "reb_pct_10g": "rebound% (10g)",
    # Ratings
    "off_rtg_10g": "offensive rating (10g)",
    "def_rtg_10g": "defensive rating (10g)",
    "net_rtg_10g": "net rating (10g)",
    # Opponent features
    "opp_avg_pts_scored": "opponent pts scored avg",
    "opp_avg_pts_allowed": "opponent pts allowed avg",
    "opp_avg_margin": "opponent avg margin",
    "opp_three_pt_rate_10g": "opponent 3PT rate (10g)",
    "opp_fta_rate_10g": "opponent FTA rate (10g)",
    "opp_assist_ratio_10g": "opponent assist ratio (10g)",
    "opp_ts_pct_10g": "opponent TS% (10g)",
    "opp_reb_pct_10g": "opponent reb% (10g)",
    # Strength of schedule
    "sos": "strength of schedule",
    "sos_last_10": "SOS (last 10g)",
    "sos_trend_10g": "SOS trend (10g)",
    # EMAs
    "ema_pts_20g": "EMA pts (20g)",
    "ema_margin_20g": "EMA margin (20g)",
    "ema_pts_10g": "EMA pts (10g)",
    "ema_margin_10g": "EMA margin (10g)",
    "ema_pts_3g": "EMA pts (3g)",
    "ema_margin_3g": "EMA margin (3g)",
    "avg_pts_3g": "pts/100poss (last 3g)",
    "avg_pts_allowed_3g": "pts allowed/100poss (last 3g)",
    "avg_margin_3g": "avg margin (last 3g)",
    # Streaks
    "streak": "current streak",
    "streak_net_pts": "streak net pts",
    "streak_5g": "streak (last 5g)",
    "streak_10g": "streak (last 10g)",
    # Trailing / in-game
    "trailing_margin": "trailing margin",
    "trailing_margin_sos": "trailing margin (SOS-adj)",
    "trailing_margin_avg": "trailing margin avg",
    # Cumulative
    "avg_pts_scored_10g_simple": "simple avg pts scored (10g)",
    "avg_pts_allowed_10g_simple": "simple avg pts allowed (10g)",
    "avg_margin_10g_simple": "simple avg margin (10g)",
}


def _feature_to_english(col: str, team_home: str, team_away: str) -> str:
    """Convert a feature column name to a plain-English description."""
    # Determine which team this feature refers to
    if col.endswith("_home"):
        team = team_home
        base = col[:-5]
    elif col.endswith("_away"):
        team = team_away
        base = col[:-5]
    else:
        team = ""
        base = col

    # Check the FULL column name first (catches _home/_away-specific entries
    # like "pace_home" vs "pace", "eFG_home" vs "eFG")
    if col in _FEATURE_DESC:
        desc = _FEATURE_DESC[col]
        if team:
            return f"{team} {desc}"
        return desc

    # Fall back to stripped base name
    if base in _FEATURE_DESC:
        desc = _FEATURE_DESC[base]
        if team:
            return f"{team} {desc}"
        return desc

    # Handle opp_ features with team prefix
    if base.startswith("opp_") and team:
        stripped = base[4:]  # Remove 'opp_'
        if stripped in _FEATURE_DESC:
            return f"{team} opponent: {_FEATURE_DESC[stripped]}"

    # Handle trailing_margin features
    if "trailing_margin" in base and team:
        extra = base.replace("trailing_margin", "").lstrip("_")
        if extra:
            return f"{team} trailing margin ({extra})"
        return f"{team} trailing margin"

    # Handle avg_pts_scored / avg_pts_allowed
    if base.startswith("avg_pts_scored") and team:
        period = base.replace("avg_pts_scored", "").lstrip("_")
        return f"{team} avg pts scored{': ' + period if period else ''}"
    if base.startswith("avg_pts_allowed") and team:
        period = base.replace("avg_pts_allowed", "").lstrip("_")
        return f"{team} avg pts allowed{': ' + period if period else ''}"

    # Fallback: clean up the name
    readable = base.replace("_", " ").replace("avg ", "avg ").strip()
    if team:
        return f"{team} {readable}"
    return readable


def _explain_prediction(
    pred: ForwardPrediction,
    models: dict,
    feat_dict: dict,
    feature_cols: list[str],
    X: np.ndarray,
    verbose: bool = False,
) -> list[str]:
    """Generate plain-English explanations for a prediction by examining
    feature-value × coefficient contributions.

    Shows the top features pushing the prediction up and down,
    so you understand WHY the model thinks what it does.
    """
    explanations = []

    # --- Ridge totals explanation ---
    if "totals_ridge" in models and pred.model_total is not None:
        ridge = models["totals_ridge"]
        coefs = ridge.model.coef_  # shape (119,)
        intercept = float(ridge.model.intercept_)

        contributions = []
        for i, col in enumerate(feature_cols):
            val = feat_dict.get(col, 0.0)
            contrib = val * coefs[i]
            contributions.append((abs(contrib), contrib, col, val, coefs[i]))

        contributions.sort(key=lambda x: -x[0])

        # Determine how many features to show
        top_n = 999 if verbose else 5
        min_contrib = 0.0 if verbose else 0.3

        i = 0
        top_pushers = []
        while len(top_pushers) < top_n and i < len(contributions):
            _, contrib, col, val, coef = contributions[i]
            i += 1
            if abs(contrib) < min_contrib:
                if not verbose:
                    continue
            if "SEASON_ID" in col:
                continue
            top_pushers.append((contrib, col, val, coef))

        # Build readable lines
        if top_pushers:
            explanations.append(f"  Totals breakdown (intercept={intercept:.0f}):")
            for contrib, col, val, coef in top_pushers:
                desc = _feature_to_english(col, pred.home_team, pred.away_team)
                arrow = "[+] " if contrib > 0 else "[-] "
                explanations.append(
                    f"    {arrow}{contrib:+.1f}pts  {desc}"
                )

    # --- Logistic regression explanation (toward home win) ---
    if "ml_momentum" in models and pred.home_win_prob is not None:
        ml_model = models["ml_momentum"]

        # Access coefficients (handles both raw and calibrated models)
        if hasattr(ml_model.model, "coef_"):
            lr_coefs = ml_model.model.coef_.flatten()
        elif hasattr(ml_model.model, "calibrated_classifiers_"):
            # CalibratedClassifierCV — drill into the base estimator
            cal = ml_model.model.calibrated_classifiers_[0]
            # Try multiple attribute names for the base estimator
            for attr in ["estimator", "base_estimator_", "base_estimator"]:
                base_est = getattr(cal, attr, None)
                if base_est is not None and hasattr(base_est, "coef_"):
                    lr_coefs = base_est.coef_.flatten()
                    break
            else:
                lr_coefs = None
        else:
            lr_coefs = None

        if lr_coefs is not None:

            contributions = []
            for i, col in enumerate(feature_cols):
                val = feat_dict.get(col, 0.0)
                contrib = val * lr_coefs[i]
                contributions.append((abs(contrib), contrib, col, val, lr_coefs[i]))

            contributions.sort(key=lambda x: -x[0])

            top_n = 999 if verbose else 5
            min_contrib = 0.0 if verbose else 0.001

            i = 0
            top_ml = []
            while len(top_ml) < top_n and i < len(contributions):
                _, contrib, col, val, coef = contributions[i]
                i += 1
                if abs(contrib) < min_contrib:
                    if not verbose:
                        continue
                if "SEASON_ID" in col:
                    continue
                top_ml.append((contrib, col, val, coef))

            if top_ml:
                home_pct = pred.home_win_prob * 100
                market_pct = pred.market_home_implied * 100 if pred.market_home_implied is not None else 0.0
                explanations.append(
                    f"  ML reasoning (home win prob={home_pct:.0f}%, "
                    f"market={market_pct:.0f}%):"
                )
                for contrib, col, val, coef in top_ml:
                    desc = _feature_to_english(col, pred.home_team, pred.away_team)
                    direction = "-> home" if contrib > 0 else "-> away"
                    explanations.append(
                        f"    {desc}  ({direction})"
                    )

    return explanations


def compute_no_vig_probs(home_ml: float, away_ml: float) -> tuple[float, float]:
    """Strip vig from moneyline odds to get fair probabilities."""
    def ml_to_prob(odds):
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)

    home_imp = ml_to_prob(home_ml)
    away_imp = ml_to_prob(away_ml)
    total = home_imp + away_imp

    if total > 0:
        return (home_imp / total, away_imp / total)
    return (0.5, 0.5)


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal."""
    if odds > 0:
        return 1 + odds / 100.0
    else:
        return 1 + 100.0 / abs(odds)


def compute_kelly(win_prob: float, decimal_odds: float, fraction: float = 0.25) -> tuple[float, float]:
    """Compute fractional Kelly stake on $10k bankroll."""
    if win_prob <= 0 or win_prob >= 1 or decimal_odds <= 1:
        return (0.0, 0.0)
    b = decimal_odds - 1.0
    if b <= 0:
        return (0.0, 0.0)
    full_kelly = (b * win_prob - (1 - win_prob)) / b
    if full_kelly <= 0:
        return (0.0, 0.0)
    frac = full_kelly * fraction
    frac = max(0.0, min(frac, 0.10))  # Cap at 10%
    return (frac, frac * 10000.0)


# ── Adjustment Column Construction ─────────────────────────────────────
# For each base_col from InjuryAdjuster, which window suffix to use when
# constructing the exact feature column name to adjust.
#
# MUST stay in sync with FEATURE_ADJUSTMENT in injury_adjuster.py.
# We intentionally keep this minimal: only avg_pts (primary scoring) and
# avg_pts_allowed (defensive leakage). All other correlated features
# (ema_pts, trend_pts, margin, pace, form_score, etc.) are excluded
# to prevent over-compounding the same signal across 10+ columns.
#
# Key: base_col from adjuster. Value: window suffix ("10g") or "" for
# single-instance features.
_ADJUSTMENT_WINDOWS: dict[str, str] = {
    "avg_pts": "10g",           # avg_pts_10g_home / _away
    "avg_pts_allowed": "10g",    # avg_pts_allowed_10g_home / _away
}


def adjust_features_for_injuries(
    feature_map: dict[str, dict],
    injury_data: dict[str, GameInjuryData],
    odds_games: list,
) -> dict[str, dict]:
    """
    Apply injury adjustments to feature vectors.

    Creates a copy of the feature map with adjusted features for games
    where injuries were detected. Uses InjuryAdjuster to compute the
    per-team scoring loss and translates it to feature-level overrides.

    Unlike the old substring-matching approach (which compounded the same
    adjustment across 13+ correlated columns), this constructs exact
    column names targeting a single mid-range window per base pattern.
    E.g., avg_pts → avg_pts_10g_home (not avg_pts_3g_home, _5g_home,
    _20g_home, or incorrectly avg_pts_allowed_home / opp_avg_pts_scored_home).

    Args:
        feature_map: Original feature map from build_prediction_features()
        injury_data: Dict of game_id -> GameInjuryData from PlayerInjuryFetcher
        odds_games: List of OddsGame objects

    Returns:
        Adjusted copy of feature_map with injury-modified features.
        Returns None if no injuries or no games were actually adjusted.
    """
    if not injury_data or not INJURY_ADJUSTER_AVAILABLE:
        return None

    print("  Applying injury adjustments to features...")
    adjuster = InjuryAdjuster()

    adjusted_map: dict[str, dict] = {}
    games_adjusted = 0
    total_adjustment = 0.0

    for key, entry in feature_map.items():
        game = entry["game"]
        features = dict(entry["features"])  # Copy

        # Match this game to injury data using team names
        game_id = getattr(game, "id", "")
        if not game_id:
            # Try matching by team names
            game_id = None
            for gid, gd in injury_data.items():
                home_short = gd.home_team.split()[-1] if gd.home_team else ""
                away_short = gd.away_team.split()[-1] if gd.away_team else ""
                if (home_short == game.home_team_short and
                        away_short == game.away_team_short):
                    game_id = gid
                    break
                # Try reverse (home/away might be swapped)
                if (home_short == game.away_team_short and
                        away_short == game.home_team_short):
                    game_id = gid
                    break

        if game_id and game_id in injury_data:
            gd = injury_data[game_id]
            if gd.has_injuries:
                adjustment = adjuster.compute_game_adjustment_from_injury_data(gd)

                # Apply adjustments using exact column name construction.
                # Each base_col targets ONE specific window to avoid
                # over-compounding the same signal across correlated features.
                for side, adj_dict in adjustment.items():
                    suffix = f"_{side}"
                    for base_col, adj_value in adj_dict.items():
                        window = _ADJUSTMENT_WINDOWS.get(base_col, "")
                        if window:
                            # Multi-window feature: avg_pts + _10g + _home
                            target = f"{base_col}_{window}{suffix}"
                        else:
                            # Single-instance feature: pace + _home
                            target = f"{base_col}{suffix}"
                        if target in features:
                            features[target] += adj_value

                games_adjusted += 1
                total_adjustment += abs(gd.total_missing_ppg)

        adjusted_map[key] = {
            "features": features,
            "game": game,
        }

    if games_adjusted > 0:
        print(f"    Adjusted {games_adjusted} game(s) for injuries "
              f"({total_adjustment:.0f} total weighted PPG impact)")
        return adjusted_map
    else:
        print(f"    No injury adjustments applied (0 games matched)")
        return None


def predict_and_compare(
    models: dict,
    feature_map: dict[str, dict],
    feature_cols: list[str],
    min_edge: float = MIN_EDGE_THRESHOLD,
    verbose: bool = False,
    label: str = "",
) -> list[ForwardPrediction]:
    """
    For each upcoming game, run model predictions and compare vs real market lines.

    Args:
        label: Optional label for logging (e.g., "(Adjusted)")
    """
    label_str = f" {label}" if label else ""
    print(f"  Running predictions{label_str} vs market lines...")
    predictions = []

    for key, entry in feature_map.items():
        game = entry["game"]
        feat_dict = entry["features"]

        # Build feature vector in the correct column order
        X = np.array([feat_dict.get(c, 0.0) for c in feature_cols]).reshape(1, -1)

        pred = ForwardPrediction(
            game_date=game.commence_time[:10] if game.commence_time else "TBD",
            matchup=f"{game.away_team_short} @ {game.home_team_short}",
            home_team=game.home_team_short,
            away_team=game.away_team_short,
            home_ml_raw=game.home_moneyline,
            away_ml_raw=game.away_moneyline,
        )

        # ---- Totals Prediction ----
        if "totals_ridge" in models:
            ridge_pred = float(models["totals_ridge"].predict(X)[0])
            pred.model_total = round(ridge_pred, 1)

        if "totals_xgboost" in models and pred.model_total is None:
            xgb_pred = float(models["totals_xgboost"].predict(X)[0])
            pred.model_total = round(xgb_pred, 1)

        # Market total line
        if game.market_total is not None:
            pred.market_total = game.market_total
            if pred.model_total is not None and game.market_total > 0:
                edge = (pred.model_total - game.market_total) / game.market_total
                pred.total_edge_pct = edge
                if abs(edge) >= min_edge:
                    pred.total_verdict = "OVER" if edge > 0 else "UNDER"

                    # Kelly stake calculation for totals
                    model_win_prob = 0.55 + abs(edge)
                    model_win_prob = min(model_win_prob, 0.85)
                    kelly, stake = compute_kelly(model_win_prob, 1.91)
                    pred.kelly_fraction = kelly
                    pred.recommended_stake = stake

        # ---- Moneyline Prediction ----
        if "ml_momentum" in models:
            win_probs = models["ml_momentum"].predict_proba(X)[0]
            if len(win_probs) == 2:
                home_prob, away_prob = float(win_probs[1]), float(win_probs[0])
            else:
                home_prob = float(win_probs[0])
                away_prob = 1.0 - home_prob

            pred.home_win_prob = round(home_prob, 3)
            pred.away_win_prob = round(away_prob, 3)

            if game.home_moneyline and game.away_moneyline:
                market_home, market_away = compute_no_vig_probs(
                    game.home_moneyline, game.away_moneyline
                )
                pred.market_home_implied = round(market_home, 3)
                pred.market_away_implied = round(market_away, 3)

                home_edge = home_prob - market_home
                away_edge = away_prob - market_away
                pred.home_ml_edge = round(home_edge, 3)
                pred.away_ml_edge = round(away_edge, 3)

                if home_edge >= min_edge or away_edge >= min_edge:
                    if home_edge > away_edge:
                        pred.ml_verdict = game.home_team_short
                        dec_odds = american_to_decimal(game.home_moneyline)
                        kelly, stake = compute_kelly(home_prob, dec_odds)
                        pred.kelly_fraction = kelly
                        pred.recommended_stake = stake
                    else:
                        pred.ml_verdict = game.away_team_short
                        dec_odds = american_to_decimal(game.away_moneyline)
                        kelly, stake = compute_kelly(away_prob, dec_odds)
                        pred.kelly_fraction = kelly
                        pred.recommended_stake = stake

        # Generate plain-English explanations
        pred.explanation = _explain_prediction(pred, models, feat_dict, feature_cols, X,
                                                verbose=verbose)

        predictions.append(pred)

    return predictions


# ============================================================================
#  Display
# ============================================================================

def print_header():
    print()
    print(f"{CYAN}{BOLD}{'=' * 75}{RESET}")
    print(f"{CYAN}{BOLD}  FORWARD TEST — REAL MARKET ODDS VALIDATION{RESET}")
    print(f"{CYAN}{BOLD}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 75}{RESET}")


def print_comparison_table(predictions: list[ForwardPrediction]):
    """Print the main comparison table showing model vs market for each game."""
    if not predictions:
        print(f"\n  {YELLOW}No predictions to display.{RESET}")
        return

    # Filter to games with enough data
    valid = [p for p in predictions if p.market_total is not None or p.market_home_implied is not None]
    if not valid:
        print(f"\n  {YELLOW}Could not build features for any upcoming game.{RESET}")
        return

    # ---- Totals Comparison ----
    has_totals = any(p.market_total is not None for p in valid)
    if has_totals:
        print(f"\n  {BOLD}TOTALS — Model vs Market Line{RESET}")
        print(f"  {'-' * 88}")
        print(f"  {'Date':<12s} {'Matchup':<28s} {'Model':>7s} {'Market':>7s} {'Edge':>7s} {'Verdict':>10s} {'Stake':>8s}")
        print(f"  {'-' * 88}")

        total_stake = 0.0
        for p in valid:
            if p.market_total is None:
                continue

            date_str = p.game_date[:10]
            model_str = f"{p.model_total:.1f}" if p.model_total else "N/A"
            market_str = f"{p.market_total:.1f}"
            edge_str = f"{p.total_edge_pct:+.1%}" if p.total_edge_pct is not None else "N/A"
            verdict_str = p.total_verdict if p.total_verdict else "—"
            stake_str = f"${p.recommended_stake:.0f}" if p.recommended_stake > 0 else "—"

            if p.total_verdict:
                verdict_display = f"{GREEN}{verdict_str:>10s}{RESET}"
                stake_display = f"{GREEN}{stake_str:>8s}{RESET}"
                total_stake += p.recommended_stake
            else:
                verdict_display = f"{verdict_str:>10s}"
                stake_display = f"{stake_str:>8s}"

            print(f"  {date_str:<12s} {p.matchup:<28s} {model_str:>7s} {market_str:>7s} {edge_str:>7s} "
                  f"{verdict_display} {stake_display}")

        if total_stake > 0:
            print(f"  {'-' * 88}")
            print(f"  {'':<12s} {'':<28s} {'':>7s} {'':>7s} {'':>7s} {'TOTAL':>10s} "
                  f"{GREEN}${total_stake:.0f}{RESET}")
        print(f"  {'-' * 88}")

    # ---- Moneyline Comparison ----
    has_ml = any(p.market_home_implied is not None for p in valid)
    if has_ml:
        print(f"\n  {BOLD}MONEYLINE — Model vs Market (No-Vig){RESET}")
        print(f"  {'-' * 98}")
        print(f"  {'Date':<12s} {'Matchup':<28s} {'Model H':>7s} {'Market H':>7s} {'Edge H':>7s} "
              f"{'Model A':>7s} {'Market A':>7s} {'Edge A':>7s}")
        print(f"  {'-' * 98}")

        for p in valid:
            if p.market_home_implied is None:
                continue

            date_str = p.game_date[:10]
            ml_h = f"{p.home_win_prob:.1%}" if p.home_win_prob else "N/A"
            mk_h = f"{p.market_home_implied:.1%}"
            ed_h = f"{p.home_ml_edge:+.1%}" if p.home_ml_edge is not None else "N/A"
            ml_a = f"{p.away_win_prob:.1%}" if p.away_win_prob else "N/A"
            mk_a = f"{p.market_away_implied:.1%}"
            ed_a = f"{p.away_ml_edge:+.1%}" if p.away_ml_edge is not None else "N/A"

            print(f"  {date_str:<12s} {p.matchup:<28s} {ml_h:>7s} {mk_h:>7s} {ed_h:>7s} "
                  f"{ml_a:>7s} {mk_a:>7s} {ed_a:>7s}")

        # Print ML verdicts
        print(f"\n  {BOLD}ML Bets to Consider:{RESET}")
        print(f"  {'-' * 68}")
        print(f"  {'Date':<12s} {'Matchup':<28s} {'Side':>8s} {'Edge':>7s} {'Stake':>8s}")
        print(f"  {'-' * 68}")

        ml_stake = 0.0
        for p in valid:
            if p.ml_verdict and p.recommended_stake > 0:
                edge = p.home_ml_edge if p.ml_verdict == p.home_team else p.away_ml_edge
                print(f"  {p.game_date[:10]:<12s} {p.matchup:<28s} {GREEN}{p.ml_verdict:>8s}{RESET} "
                      f"{edge:+.1%}  ${p.recommended_stake:.0f}")
                ml_stake += p.recommended_stake

        if ml_stake > 0:
            print(f"  {'-' * 68}")
            print(f"  {'':<12s} {'':<28s} {'':>8s} {'TOTAL':>7s} {GREEN}${ml_stake:.0f}{RESET}")
            print(f"  {'-' * 68}")


def print_opportunity_summary(predictions: list[ForwardPrediction]):
    """Print a concise list of actionable +EV opportunities ranked by edge."""
    print(f"\n  {BOLD}+EV OPPORTUNITIES (Ranked by Edge){RESET}")
    print(f"  {'-' * 60}")
    print(f"  {'#':<3s} {'Bet':<55s} {'Edge':>7s} {'Stake':>8s}")
    print(f"  {'-' * 73}")

    opportunities = []
    for p in predictions:
        if p.total_verdict and p.recommended_stake > 0:
            opportunities.append((
                f"{p.game_date[:10]} Total {p.total_verdict} {p.market_total:.0f} — {p.matchup}",
                p.total_edge_pct,
                p.recommended_stake,
            ))
        if p.ml_verdict and p.recommended_stake > 0:
            opportunities.append((
                f"{p.game_date[:10]} ML {p.ml_verdict} — {p.matchup}",
                p.home_ml_edge if p.ml_verdict == p.home_team else p.away_ml_edge,
                p.recommended_stake,
            ))

    # Sort by absolute edge descending
    opportunities.sort(key=lambda x: abs(x[1]), reverse=True)

    if not opportunities:
        print(f"  {YELLOW}  No actionable opportunities found (edge below threshold).{RESET}")
        return

    total_stake = 0.0
    for i, (bet, edge, stake) in enumerate(opportunities, 1):
        edge_color = GREEN if edge > 0 else RED
        print(f"  {i:<3d} {bet:<55s} {edge_color}{edge:+.1%}{RESET}  ${stake:.0f}")
        total_stake += stake

    print(f"  {'-' * 73}")
    print(f"  {'':<3s} {'TOTAL EXPOSURE':<55s} {'':>7s} {GREEN}${total_stake:.0f}{RESET}")
    print(f"  {'-' * 73}")
    print(f"  {'':<3s} {'Bankroll: $10,000 | Kelly: 0.25x | Edge threshold: ' + f'{MIN_EDGE_THRESHOLD:.0%}':<55s}")
    print()


# ============================================================================
#  Per-game Explanations
# ============================================================================

def print_explanations(predictions: list[ForwardPrediction]):
    """Print plain-English explanations for each prediction."""
    if not predictions:
        return

    has_expl = any(p.explanation for p in predictions)
    if not has_expl:
        return

    print(f"\n  {BOLD}WHY THE MODEL PREDICTS WHAT IT DOES - Per-Game Breakdown{RESET}")
    print()

    for p in predictions:
        if not p.explanation:
            continue
        print(f"  {CYAN}{BOLD}-- {p.game_date[:10]}  {p.matchup}{RESET}")
        for line in p.explanation:
            print(line)
        print()


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Forward Test — Validate model against real market odds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  export ODDS_API_KEY="your_key_here"
  python tools/forward_test.py                   # Full run
  python tools/forward_test.py --model totals    # Totals only
  python tools/forward_test.py --model moneyline # Moneyline only
        """,
    )
    parser.add_argument("--model", type=str, default=None,
                        choices=["totals", "moneyline"],
                        help="Run only one model type")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE_THRESHOLD,
                        help=f"Minimum edge threshold (default: {MIN_EDGE_THRESHOLD:.0%})")
    parser.add_argument("--calibrated", action="store_true",
                        help="Apply Platt scaling to probability estimates")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show ALL feature contributions instead of top 5")

    args = parser.parse_args()
    min_edge = args.min_edge

    print_header()

    # Phase 1: Load data and train models
    print(f"\n  {CYAN}{BOLD}[Phase 1/4] Training on Historical Data{RESET}")
    df, feature_cols, feature_df = load_and_prepare_data()
    models = train_models(df, feature_cols, calibrated=args.calibrated)

    # Phase 2: Fetch upcoming games with real odds
    print(f"\n  {CYAN}{BOLD}[Phase 2/4] Fetching Upcoming Games{RESET}")
    odds_games = fetch_upcoming_games()

    if not odds_games:
        print(f"\n  {RED}[!] No upcoming games. Nothing to predict.{RESET}")
        print(f"   Set ODDS_API_KEY in your environment and try again.\n")
        return 1

    game_list_str = "\n  ".join(
        f"  {getattr(g, 'commence_time', '')[:10]:<12s} {g.away_team_short:>20s} @ {g.home_team_short:<20s}  "
        f"(ML: {g.home_moneyline}/{g.away_moneyline}, Total: {g.market_total})"
        for g in odds_games[:10]
    )
    print(f"  Upcoming games (scheduled date):\n{game_list_str}")
    if len(odds_games) > 10:
        print(f"  ... and {len(odds_games) - 10} more")

    # Phase 3: Fetch injury data (player props + ESPN status) for upcoming games
    print(f"\n  {CYAN}{BOLD}[Phase 3/4] Injury Impact Assessment & Feature Adjustment{RESET}")
    injury_data: dict[str, GameInjuryData] = {}
    merged_injury_data: dict[str, MergedGameInjuryData] = {}

    api_key = os.environ.get("ODDS_API_KEY", "")
    if api_key and INJURY_AVAILABLE:
        try:
            # Source 1: Prop-based detection (TheOddsAPI)
            fetcher = PlayerInjuryFetcher(api_key=api_key)
            results = fetcher.fetch_injury_impact_for_upcoming_games()
            for gd in results:
                injury_data[gd.game_id] = gd

            # Source 2: ESPN official injury status (roster API)
            if ESPN_INTEGRATOR_AVAILABLE:
                print()
                print(f"  ESPN Injury Status:")
                try:
                    integrator = ESPNInjuryIntegrator()
                    merged_injury_data = integrator.merge(injury_data)
                    for game_id, merged in sorted(merged_injury_data.items()):
                        lines = integrator.get_display_lines(merged)
                        for line in lines:
                            # Safely encode for Windows console
                            try:
                                print(line)
                            except UnicodeEncodeError:
                                safe = line.encode('ascii', 'replace').decode('ascii')
                                print(safe)
                    if not merged_injury_data:
                        print(f"    No injury data from either source")
                    else:
                        any_injuries = any(m.has_any_injuries for m in merged_injury_data.values())
                        if not any_injuries:
                            print(f"    No significant injuries detected")
                except Exception as e:
                    print(f"    [!] ESPN integration failed: {e}")
                    print(f"    (Prop-based injury detection still works)")
            else:
                print(f"  ESPN integrator not available")

        except Exception as e:
            print(f"  {YELLOW}[!] Injury fetch failed: {e}{RESET}")
    else:
        print(f"  {YELLOW}[!] Skipping (no API key or module unavailable){RESET}")

    # Phase 4: Predict and compare
    print(f"\n  {CYAN}{BOLD}[Phase 4/4] Model vs Market Comparison{RESET}")

    feature_map = build_prediction_features(odds_games, feature_df, feature_cols)

    if not feature_map:
        print(f"  {YELLOW}[!] Could not build features for any upcoming game.{RESET}")
        print(f"     This usually means the teams in upcoming games don't have")
        print(f"     enough history in the database to compute features.")
        return 1

    # Run predictions WITHOUT injury adjustments
    predictions = predict_and_compare(models, feature_map, feature_cols, min_edge,
                                         verbose=args.verbose)

    # Run predictions WITH injury adjustments (if injury data available)
    adjusted_predictions = None
    if injury_data and INJURY_ADJUSTER_AVAILABLE:
        adjusted_map = adjust_features_for_injuries(feature_map, injury_data, odds_games)
        if adjusted_map:
            adjusted_predictions = predict_and_compare(
                models, adjusted_map, feature_cols, min_edge,
                verbose=args.verbose, label="(Adjusted)"
            )

    # Display results
    print(f"\n  {CYAN}[Unadjusted Predictions]{RESET}")
    print_comparison_table(predictions)
    print_explanations(predictions)

    if adjusted_predictions:
        print(f"\n  {CYAN}[Injury-Adjusted Predictions]{RESET}")
        print_comparison_table(adjusted_predictions)
        print_explanations(adjusted_predictions)

        # Show comparison summary
        print(f"\n  {BOLD}INJURY ADJUSTMENT IMPACT{RESET}")
        print(f"  {'-' * 60}")
        for orig, adj in zip(predictions, adjusted_predictions):
            if orig.matchup == adj.matchup:
                shifts = []
                if orig.model_total is not None and adj.model_total is not None:
                    diff = adj.model_total - orig.model_total
                    shifts.append(f"Total shift: {diff:+.1f} pts")
                if orig.home_win_prob is not None and adj.home_win_prob is not None:
                    diff = (adj.home_win_prob - orig.home_win_prob) * 100
                    shifts.append(f"Home win prob shift: {diff:+.1f}%")
                if shifts:
                    print(f"  {orig.matchup}:")
                    for s in shifts:
                        print(f"    {s}")

    # Injury impact already shown in Phase 3 with both ESPN + prop data
    print_opportunity_summary(adjusted_predictions or predictions)

    print(f"\n  {GREEN}{BOLD}Done.{RESET} Set up cron to run daily: python tools/forward_test.py")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
