"""Application settings — pydantic-settings with environment variable support."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — loaded from .env / environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Core Paths ──────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent.parent.parent
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")

    # ── Database ────────────────────────────────────────────────────
    nba_db_path: Path = Path("./data/nba_data.db")
    database_url: str = ""  # PostgreSQL/other URL; empty = SQLite fallback

    # ── API Server ──────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "change-me-to-a-random-secret"
    api_workers: int = 1
    api_log_level: str = "info"
    cors_origins: str = "*"

    # ── Live Odds ───────────────────────────────────────────────────
    enable_live_odds: bool = False
    odds_poll_interval: int = 60
    odds_snapshots_db: str = "./data/odds_snapshots.db"

    # ── Logging ─────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── TheOddsAPI ──────────────────────────────────────────────────
    odds_api_key: str = "your-api-key-here"

    # ── API Schedule (daily morning fetch to minimize usage) ──────────
    # Controls when TheOddsAPI is automatically called.
    # Default: once per day at 6:00 AM local time.
    daily_fetch_enabled: bool = True  # Enable daily morning fetch
    daily_fetch_hour: int = 6  # Hour (0-23) for the daily fetch
    daily_fetch_timezone: str = "local"  # "local" or "utc"

    # ── Telegram Notifications ────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Kelly Staking ───────────────────────────────────────────────
    initial_bankroll: float = 10_000.0
    min_edge_threshold: float = 0.02


@lru_cache()
def get_settings() -> Settings:
    return Settings()
