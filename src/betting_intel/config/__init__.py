"""
Configuration package — exposes both the pydantic-settings Settings class
and module-level constants matching the original config.py API for backward
compatibility with the core engine modules.
"""
from pathlib import Path
from betting_intel.config.settings import Settings, get_settings

_settings = get_settings()

# ── Core constants (matching original config.py API) ──────────────────
PROJECT_ROOT = _settings.project_root
DATA_DIR = _settings.resolved_data_dir
OUTPUT_DIR = _settings.resolved_output_dir
DB_PATH = _settings.resolved_nba_db_path

# ── Data Settings ─────────────────────────────────────────────────────
MIN_GAMES_PER_TEAM = 5
TRAIN_START_DATE = "2022-10-01"
TEST_START_DATE = "2023-11-01"
TEST_END_DATE = "2024-04-14"

# ── Feature Engineering ───────────────────────────────────────────────
ROLLING_WINDOWS = _settings.rolling_windows
MAX_REST_DAYS = _settings.max_rest_days
BACK_TO_BACK_PENALTY = True

# ── Modeling ──────────────────────────────────────────────────────────
TEST_SIZE = _settings.test_size
WALK_FORWARD_WINDOW = _settings.walk_forward_window
WALK_FORWARD_STEP = _settings.walk_forward_step
MIN_TRAIN_SAMPLES = _settings.min_train_samples

# ── Betting Simulation ────────────────────────────────────────────────
INITIAL_BANKROLL = _settings.initial_bankroll
UNIT_SIZE = _settings.unit_size
MAX_KELLY_FRACTION = _settings.max_kelly_fraction
MIN_EDGE_THRESHOLD = _settings.min_edge_threshold
VIG = 0.045
CONFIDENCE_THRESHOLD = 0.55

# ── Model Selection ───────────────────────────────────────────────────
ENABLE_LINEAR_MODEL = _settings.enable_linear_model
ENABLE_XGBOOST_MODEL = _settings.enable_xgboost_model
ENABLE_ENSEMBLE = _settings.enable_ensemble

# ── Output ────────────────────────────────────────────────────────────
VERBOSE = True
SAVE_PLOTS = True
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# ── Strategy Names ────────────────────────────────────────────────────
STRATEGIES = {
    "pace_total": "Pace-Adjusted Total Prediction",
    "rest_edge": "Rest & Fatigue Edge",
    "momentum": "Momentum Reversion",
    "spread_model": "Spread Prediction Model",
    "quarter_scoring": "Quarter Scoring Pattern",
}

__all__ = [
    "Settings", "get_settings", "settings",
    # Constants
    "PROJECT_ROOT", "DATA_DIR", "OUTPUT_DIR", "DB_PATH",
    "ROLLING_WINDOWS", "MAX_REST_DAYS",
    "WALK_FORWARD_WINDOW", "WALK_FORWARD_STEP", "MIN_TRAIN_SAMPLES",
    "INITIAL_BANKROLL", "UNIT_SIZE", "MAX_KELLY_FRACTION", "MIN_EDGE_THRESHOLD",
    "ENABLE_LINEAR_MODEL", "ENABLE_XGBOOST_MODEL", "ENABLE_ENSEMBLE",
    "VERBOSE", "STRATEGIES",
]
