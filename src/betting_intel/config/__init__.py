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
ODDS_CACHE_TTL_SECONDS = 300  # Refresh odds every 5 minutes
THEODDSAPI_REFRESH_INTERVAL = 0  # Throttle removed — scheduler handles daily cadence
CACHE_DIR = PROJECT_ROOT / "cache"

# ── Telegram Notifications ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = _settings.telegram_bot_token
TELEGRAM_CHAT_ID = _settings.telegram_chat_id

# ── Daily Fetch Schedule (minimize API usage) ─────────────────────────
DAILY_FETCH_ENABLED = _settings.daily_fetch_enabled
DAILY_FETCH_HOUR = _settings.daily_fetch_hour
DAILY_FETCH_TIMEZONE = _settings.daily_fetch_timezone
# Cost estimate: ~3 credits/sport × 5 sports = ~15 credits/full refresh
# At once per day, that's ~450 credits/month (within 500 free tier)

__all__ = [
    "Settings",
    "get_settings",
    "settings",
    "PROJECT_ROOT",
    "DATA_DIR",
    "OUTPUT_DIR",
    "DB_PATH",
    "ROLLING_WINDOWS",
    "MAX_REST_DAYS",
    "ODDS_API_KEY",
    "ODDS_API_BASE_URL",
    "ODDS_CACHE_TTL_SECONDS",
    "CACHE_DIR",
    "DAILY_FETCH_ENABLED",
    "DAILY_FETCH_HOUR",
    "DAILY_FETCH_TIMEZONE",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
]
