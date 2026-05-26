"""Production-grade configuration using pydantic-settings.

Loads from:
1. .env file (local development)
2. Environment variables (Docker/production)
3. Default values (sensible defaults)
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Project Paths ──────────────────────────────────────────────────
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent
    )
    data_dir: Path = Field(default="./data")
    output_dir: Path = Field(default="./output")
    logs_dir: Path = Field(default="./logs")

    # ── Database ───────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///./data/betting_intel.db",
        description="Database connection URL. Supports SQLite and PostgreSQL.",
    )
    nba_db_path: Path = Field(
        default="./data/nba_data.db",
        description="Path to the NBA game logs SQLite database.",
    )

    # ── Logging ────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")
    log_format: Literal["json", "pretty"] = Field(
        default="pretty", description="Log output format"
    )
    log_file: Optional[Path] = Field(default=None, description="Path to log file")

    # ── API Server ─────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=4, ge=1, le=32)
    api_log_level: str = Field(default="info")
    api_key: str = Field(default="change-me-to-a-random-secret")
    cors_origins: str = Field(default="*")

    # ── Monitoring ─────────────────────────────────────────────────────
    enable_prometheus: bool = Field(default=False)
    prometheus_port: int = Field(default=9090, ge=1, le=65535)

    # ── Kelly Staking ──────────────────────────────────────────────────
    initial_bankroll: float = Field(default=10_000.0, ge=0)
    unit_size: float = Field(default=0.02, ge=0, le=1)
    max_kelly_fraction: float = Field(default=0.25, ge=0, le=1)
    min_edge_threshold: float = Field(default=0.02, ge=0, le=1)

    # ── Model Configuration ────────────────────────────────────────────
    enable_linear_model: bool = Field(default=True)
    enable_xgboost_model: bool = Field(default=True)
    enable_ensemble: bool = Field(default=True)
    walk_forward_window: int = Field(default=200, ge=10)
    walk_forward_step: int = Field(default=20, ge=1)
    min_train_samples: int = Field(default=50, ge=10)
    test_size: float = Field(default=0.20, ge=0.05, le=0.5)

    # ── Rolling Windows for Feature Engineering ────────────────────────
    rolling_windows: list[int] = Field(default=[3, 5, 10, 20])
    max_rest_days: int = Field(default=7, ge=1)

    # ── Small League Configuration ──────────────────────────────────────
    enable_small_leagues: bool = Field(
        default=True,
        description="Enable small-league data ingestion (LNB Pro B, CEBL, BNXT)",
    )
    small_league_cache_dir: Optional[str] = Field(
        default=None,
        description="Cache directory for small-league data.",
    )
    # PrivateAttrs store raw env-var values; properties parse them.
    # We use PrivateAttr because pydantic-settings struggles with list/dict
    # types from .env files (trying to JSON-parse comma-separated values).
    _active_small_leagues_raw: str = PrivateAttr(default="lnb_pro_b,cebl,bnxt")
    _small_league_seasons_raw: str = PrivateAttr(default="")

    def __init__(self, **kwargs):
        # Intercept env-var style keys that match our PrivateAttr aliases
        for raw_key in ("active_small_leagues", "small_league_historical_seasons"):
            if raw_key in kwargs:
                kwargs["_" + raw_key + "_raw"] = kwargs.pop(raw_key)
        super().__init__(**kwargs)

    @property
    def active_small_leagues(self) -> list[str]:
        """Parse comma-separated active_small_leagues into a list."""
        raw = self._active_small_leagues_raw
        import json
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return [x.strip() for x in raw.split(",") if x.strip()]

    @property
    def small_league_historical_seasons(self) -> dict[str, list[str]]:
        """Parse JSON seasons config, or return defaults."""
        raw = self._small_league_seasons_raw
        if raw:
            import json
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return {
            "lnb_pro_b": ["2025-2026", "2024-2025"],
            "cebl": ["2025", "2024"],
            "bnxt": ["2025-2026", "2024-2025"],
        }

    # ── Derived Paths ──────────────────────────────────────────────────
    @property
    def resolved_data_dir(self) -> Path:
        path = Path(self.data_dir)
        return path if path.is_absolute() else self.project_root / path

    @property
    def resolved_output_dir(self) -> Path:
        path = Path(self.output_dir)
        return path if path.is_absolute() else self.project_root / path

    @property
    def resolved_logs_dir(self) -> Path:
        path = Path(self.logs_dir or "logs")
        return path if path.is_absolute() else self.project_root / path

    @property
    def resolved_nba_db_path(self) -> Path:
        path = Path(self.nba_db_path)
        return path if path.is_absolute() else self.project_root / path

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"Invalid log level: {v}. Must be one of {allowed}")
        return upper


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
