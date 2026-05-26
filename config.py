"""
Configuration: All paths, parameters, and strategy settings.
Change these to adapt the system to different leagues or strategies.
"""

from pathlib import Path

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

# ── Feature Engineering ────────────────────────────────────────────────────
ROLLING_WINDOWS = [3, 5, 10, 20]  # Game windows for rolling averages
MAX_REST_DAYS = 7                 # Cap for rest days feature
BACK_TO_BACK_PENALTY = True       # Include B2B flag

# ── Modeling ────────────────────────────────────────────────────────────────
TEST_SIZE = 0.20                  # Fraction of data held out for final test
WALK_FORWARD_WINDOW = 200        # Games in each training window
WALK_FORWARD_STEP = 20           # Step size for walk-forward
MIN_TRAIN_SAMPLES = 50           # Minimum training samples required

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
    "pace_total": "Pace-Adjusted Total Prediction",
    "rest_edge": "Rest & Fatigue Edge",
    "momentum": "Momentum Reversion",
    "spread_model": "Spread Prediction Model",
    "quarter_scoring": "Quarter Scoring Pattern",
}
