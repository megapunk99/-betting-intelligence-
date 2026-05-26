"""
Configuration: All paths, parameters, and strategy settings.
v2.0 — Updated for advanced models, features, and Monte Carlo simulation.
"""

from pathlib import Path
import os

# ── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
DB_PATH = PROJECT_ROOT / "data" / "nba_data.db"

# ── Data Settings ──────────────────────────────────────────────────────────
MIN_GAMES_PER_TEAM = 5       # Minimum games before a team gets features
TRAIN_START_DATE = "2022-10-01"
TEST_START_DATE = "2023-11-01"
TEST_END_DATE = "2024-04-14"

# ── Feature Engineering (v2.0) ─────────────────────────────────────────────
ROLLING_WINDOWS = [3, 5, 10]     # Game windows for rolling averages (20 removed for speed)
MAX_REST_DAYS = 7                 # Cap for rest days feature
BACK_TO_BACK_PENALTY = True       # Include B2B flag
ENABLE_ELO_RATINGS = True         # Compute Elo ratings (v2.0)
ENABLE_TRAVEL_FEATURES = True     # Compute travel distance & fatigue (v2.0)
ENABLE_OPPONENT_ADJUSTED = True   # Compute opponent-adjusted stats (v2.0)
ENABLE_ADVANCED_MOMENTUM = True   # Weighted/decay-based momentum (v2.0)

# ── Modeling (v2.0) ────────────────────────────────────────────────────────
TEST_SIZE = 0.20                  # Fraction of data held out for final test
WALK_FORWARD_WINDOW = 200        # Games in each training window
WALK_FORWARD_STEP = 20           # Step size for walk-forward
MIN_TRAIN_SAMPLES = 50           # Minimum training samples required

# Model selection (v2.0)
PREFERRED_MODEL = "lightgbm"     # Options: lightgbm, catboost, xgboost, ridge, ensemble

# These are disabled by default for performance. Enable with --tune or config changes.
ENABLE_HYPERPARAMETER_TUNING = False  # Use Optuna for tuning (off by default for speed)
ENABLE_STACKING_ENSEMBLE = False      # Use stacking ensemble meta-model (off by default)
ENABLE_PROBABILITY_CALIBRATION = True # Calibrate classification probabilities
ENABLE_BAYESIAN_UNCERTAINTY = True   # Use Bayesian models for uncertainty

# Fast mode: when True, runs only LightGBM + Momentum (1-2 models instead of 7)
FAST_MODE = True                     # Runs only essential models for quick results

# ── Monte Carlo Simulation (v2.0) ──────────────────────────────────────────
ENABLE_MONTE_CARLO = True            # Run Monte Carlo simulation
MONTE_CARLO_SIMULATIONS = 1000       # Number of simulated seasons (reduced from 5000 for speed)
MONTE_CARLO_BETS_PER_SEASON = 500    # Bets per simulated season

# ── Betting Simulation ─────────────────────────────────────────────────────
INITIAL_BANKROLL = 10_000.0       # Starting bankroll ($)
UNIT_SIZE = 0.02                  # 2% Kelly fraction (conservative)
MAX_KELLY_FRACTION = 0.25         # Cap on Kelly stake (% of bankroll)
MIN_EDGE_THRESHOLD = 0.02         # Minimum edge to bet (2%)
VIG = 0.045                       # Assumed bookmaker vig (4.5%)
CONFIDENCE_THRESHOLD = 0.55       # Minimum model confidence to bet

# ── Model Selection ─────────────────────────────────────────────────────────
ENABLE_LINEAR_MODEL = True
ENABLE_XGBOOST_MODEL = True
ENABLE_ENSEMBLE = True

# ── Output ─────────────────────────────────────────────────────────────────
VERBOSE = True
SAVE_PLOTS = True
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# ── Strategy Names ─────────────────────────────────────────────────────────
STRATEGIES = {
    "pace_total": "Pace-Adjusted Total Prediction (v2.0)",
    "rest_edge": "Rest & Fatigue Edge (v2.0)",
    "momentum": "Momentum Reversion (v2.0)",
    "spread_model": "Spread Prediction Model (v2.0)",
    "quarter_scoring": "Quarter Scoring Pattern",
    "elo_based": "Elo Rating Based Prediction (v2.0)",
    "advanced_ensemble": "Stacking Ensemble (v2.0)",
}

# ── XGBoost availability (checked at import) ───────────────────────────────
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── TheOddsAPI Integration ──────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "your-api-key-here")
"""API key for TheOddsAPI. Get one free at https://the-odds-api.com/"""

ODDS_API_BASE_URL = "https://api.the-odds-api.com"
"""Base URL for TheOddsAPI v4"""

ODDS_CACHE_TTL_MINUTES = 15
"""How long to cache odds data before fetching fresh (minutes)"""

ODDS_DEFAULT_MARKETS = "h2h,spreads,totals"
"""Markets to fetch from TheOddsAPI"""

ODDS_DEFAULT_REGIONS = "us"
"""Bookmaker regions to include"""

CACHE_DIR = PROJECT_ROOT / "cache"
"""Directory for caching API responses"""
