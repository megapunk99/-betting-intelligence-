"""Configuration package — exposes settings and module-level constants."""
from pathlib import Path
from betting_intel.config.settings import Settings, get_settings

_settings = get_settings()
settings = _settings  # Export the instance

# ── Core paths ──────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR: Path = _settings.data_dir
OUTPUT_DIR: Path = _settings.output_dir
DB_PATH: Path = _settings.nba_db_path

# ── Data Settings ───────────────────────────────────────────────────
MIN_GAMES_PER_TEAM = 5
TRAIN_START_DATE = "2022-10-01"
TEST_START_DATE = "2023-11-01"
TEST_END_DATE = "2024-04-14"

# ── Feature Engineering ─────────────────────────────────────────────
ROLLING_WINDOWS = [3, 5, 10, 20]
MAX_REST_DAYS = 14
BACK_TO_BACK_PENALTY = True

# ── Modeling ────────────────────────────────────────────────────────
TEST_SIZE = 0.2
WALK_FORWARD_WINDOW = 50
WALK_FORWARD_STEP = 10
MIN_TRAIN_SAMPLES = 100

# ── Betting Simulation ──────────────────────────────────────────────
INITIAL_BANKROLL = _settings.initial_bankroll
MIN_EDGE_THRESHOLD = _settings.min_edge_threshold
UNIT_SIZE = INITIAL_BANKROLL / 100
MAX_KELLY_FRACTION = 0.25
VIG = 0.045
CONFIDENCE_THRESHOLD = 0.55

# ── Model Selection ─────────────────────────────────────────────────
ENABLE_LINEAR_MODEL = True
ENABLE_XGBOOST_MODEL = True
ENABLE_ENSEMBLE = True

# ── Output ──────────────────────────────────────────────────────────
VERBOSE = True
SAVE_PLOTS = True
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# ── Strategy Names ──────────────────────────────────────────────────
STRATEGIES = {
    "pace_total": "Pace-Adjusted Total Prediction",
    "rest_edge": "Rest & Fatigue Edge",
    "momentum": "Momentum Reversion",
    "spread_model": "Spread Prediction Model",
    "quarter_scoring": "Quarter Scoring Pattern",
}

# ── Pipeline Control ────────────────────────────────────────────────
FAST_MODE = True
ENABLE_HYPERPARAMETER_TUNING = False
ENABLE_STACKING_ENSEMBLE = True
ENABLE_MONTE_CARLO = True
MONTE_CARLO_SIMULATIONS = 10000
PREFERRED_MODEL = "lightgbm"

# ── API / External Services ─────────────────────────────────────────
ODDS_API_KEY = _settings.odds_api_key
ODDS_API_BASE_URL = "https://api.the-odds-api.com"
ODDS_CACHE_TTL_MINUTES = 5
CACHE_DIR = PROJECT_ROOT / "cache"

__all__ = [
    "Settings", "get_settings", "settings",
    "PROJECT_ROOT", "DATA_DIR", "OUTPUT_DIR", "DB_PATH",
    "ROLLING_WINDOWS", "MAX_REST_DAYS",
    "WALK_FORWARD_WINDOW", "WALK_FORWARD_STEP", "MIN_TRAIN_SAMPLES",
    "INITIAL_BANKROLL", "UNIT_SIZE", "MAX_KELLY_FRACTION", "MIN_EDGE_THRESHOLD",
    "ENABLE_LINEAR_MODEL", "ENABLE_XGBOOST_MODEL", "ENABLE_ENSEMBLE",
    "VERBOSE", "STRATEGIES",
    "FAST_MODE", "ENABLE_HYPERPARAMETER_TUNING",
    "ENABLE_STACKING_ENSEMBLE", "ENABLE_MONTE_CARLO", "MONTE_CARLO_SIMULATIONS",
    "PREFERRED_MODEL",
    "ODDS_API_KEY", "ODDS_API_BASE_URL", "ODDS_CACHE_TTL_MINUTES", "CACHE_DIR",
]
