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

# ── Feature Engineering ─────────────────────────────────────────────
ROLLING_WINDOWS = [3, 5, 10, 20]
MAX_REST_DAYS = 14

# ── API / External Services ─────────────────────────────────────────
ODDS_API_KEY = _settings.odds_api_key
ODDS_API_BASE_URL = "https://api.the-odds-api.com"
ODDS_CACHE_TTL_SECONDS = 300       # Refresh odds every 5 minutes
CACHE_DIR = PROJECT_ROOT / "cache"

__all__ = [
    "Settings", "get_settings", "settings",
    "PROJECT_ROOT", "DATA_DIR", "OUTPUT_DIR", "DB_PATH",
    "ROLLING_WINDOWS", "MAX_REST_DAYS",
    "ODDS_API_KEY", "ODDS_API_BASE_URL", "ODDS_CACHE_TTL_SECONDS", "CACHE_DIR",
]
