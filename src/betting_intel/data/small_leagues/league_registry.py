"""
League Registry — central registry for all small-league data sources.

Provides:
1. A unified registry mapping league keys to source classes
2. Health checks for each league (freshness, data age, API status)
3. Auto-discovery of available sources
4. League metadata lookup

Usage:
    from betting_intel.data.small_leagues.league_registry import LeagueRegistry

    registry = LeagueRegistry()
    registry.list_leagues()
    # -> {"lnb_pro_b": {"name": "French LNB Pro B", ...}, ...}

    registry.check_health("cebl")
    # -> {"league": "cebl", "status": "healthy", "games_last_24h": 0, ...}

    registry.discover_new_sources()
    # -> ["wnba", "ligat_haal"]  (not yet registered but discovered as modules)
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LeagueHealthStatus:
    """Health status for a single league data source."""

    league_key: str
    league_name: str
    data_source: str = "unknown"
    status: str = "unknown"  # "healthy", "degraded", "unavailable", "unknown"
    games_last_24h: int = 0
    games_last_7d: int = 0
    total_games: int = 0
    last_data_fetch: Optional[datetime] = None
    last_error: Optional[str] = None
    fetch_success_rate: float = 1.0  # 0.0 to 1.0
    is_available: bool = True
    warnings: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.now)

    @property
    def freshness_grade(self) -> str:
        if self.last_data_fetch is None:
            return "NO_DATA"
        hours_since = (datetime.now() - self.last_data_fetch).total_seconds() / 3600
        if hours_since < 1:
            return "A"
        elif hours_since < 6:
            return "B"
        elif hours_since < 24:
            return "C"
        elif hours_since < 72:
            return "D"
        return "F"


class LeagueRegistry:
    """
    Central registry for all small-league data sources.

    The registry is the single point of entry for discovering, loading,
    and monitoring small-league data sources. New sources register
    themselves when the module is imported.
    """

    # Built-in sources
    _sources: dict[str, type] = {}
    _metadata: dict[str, dict[str, Any]] = {}

    def __init__(self):
        self._loaded: dict[str, Any] = {}
        self._auto_discover()

    def _auto_discover(self):
        """Auto-discover league source modules in the small_leagues package."""
        try:
            import betting_intel.data.small_leagues as pkg

            for importer, modname, ispkg in pkgutil.iter_modules(
                pkg.__path__, prefix="betting_intel.data.small_leagues."
            ):
                if ispkg:
                    continue
                try:
                    module = importlib.import_module(modname)
                    if hasattr(module, "REGISTER_LEAGUE"):
                        key, cls, meta = module.REGISTER_LEAGUE
                        self.register(key, cls, meta)
                        logger.info(f"Auto-discovered league source: {key} ({meta.get('name', '?')})")
                except Exception as exc:
                    logger.debug(f"Failed to load {modname}: {exc}")
        except Exception as exc:
            logger.warning(f"Auto-discovery failed: {exc}")

    def register(
        self,
        league_key: str,
        source_class: type,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Register a league data source."""
        self._sources[league_key] = source_class
        if metadata:
            self._metadata[league_key] = metadata
        logger.info(f"Registered league: {league_key}")

    def get_source(self, league_key: str, **kwargs) -> Any:
        """Get or create a source instance for a league."""
        if league_key not in self._sources:
            raise ValueError(
                f"Unknown league '{league_key}'. Available: {', '.join(sorted(self._sources))}"
            )
        if league_key not in self._loaded:
            cls = self._sources[league_key]
            self._loaded[league_key] = cls(**kwargs)
        return self._loaded[league_key]

    def list_leagues(self) -> dict[str, dict[str, Any]]:
        """List all registered leagues with metadata."""
        result = {}
        for key, cls in self._sources.items():
            meta = dict(self._metadata.get(key, {}))
            meta["source_class"] = cls.__name__
            result[key] = meta
        return result

    def check_health(self, league_key: str) -> LeagueHealthStatus:
        """Check health of a specific league's data source."""
        if league_key not in self._sources:
            return LeagueHealthStatus(
                league_key=league_key,
                league_name=league_key,
                status="unavailable",
                data_source="unknown",
                is_available=False,
                warnings=["League not registered"],
            )

        try:
            source = self.get_source(league_key)
            meta = self._metadata.get(league_key, {})

            status = LeagueHealthStatus(
                league_key=league_key,
                league_name=meta.get("name", league_key),
                data_source=meta.get("data_source", "unknown"),
            )

            # Try to load historical data to check availability
            try:
                df = source.load_historical(seasons=None)
                if df.empty:
                    status.status = "degraded"
                    status.warnings.append("No historical data returned")
                else:
                    status.total_games = len(df)
                    status.is_available = True

                    # Check recent data
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                        now = datetime.now()
                        last_24h = now - timedelta(hours=24)
                        last_7d = now - timedelta(days=7)
                        status.games_last_24h = int(
                            (df["date"] >= pd.Timestamp(last_24h)).sum()
                        )
                        status.games_last_7d = int(
                            (df["date"] >= pd.Timestamp(last_7d)).sum()
                        )
                        if not df["date"].empty:
                            status.last_data_fetch = df["date"].max().to_pydatetime()

                    if status.total_games > 0:
                        status.status = "healthy"
                    else:
                        status.status = "degraded"
            except Exception as exc:
                status.status = "unavailable"
                status.last_error = str(exc)
                status.is_available = False
                status.warnings.append(f"Data fetch failed: {exc}")

            status.checked_at = datetime.now()
            return status

        except Exception as exc:
            return LeagueHealthStatus(
                league_key=league_key,
                league_name=league_key,
                status="unavailable",
                data_source="unknown",
                last_error=str(exc),
                is_available=False,
                warnings=[f"Health check failed: {exc}"],
            )

    def check_all_health(self) -> dict[str, LeagueHealthStatus]:
        """Check health of all registered leagues."""
        return {key: self.check_health(key) for key in self._sources}

    def get_available_leagues(self) -> list[str]:
        """Get list of leagues that are currently available (healthy or degraded)."""
        healthy = []
        for key in self._sources:
            status = self.check_health(key)
            if status.is_available:
                healthy.append(key)
        return healthy

    def load_all_historical(
        self, seasons: Optional[dict[str, list]] = None
    ) -> dict[str, pd.DataFrame]:
        """Load historical data from all available leagues."""
        result = {}
        for key in self._sources:
            try:
                league_seasons = (seasons or {}).get(key)
                source = self.get_source(key)
                df = source.load_historical(seasons=league_seasons)
                if not df.empty:
                    result[key] = df
                    logger.info(f"Loaded {len(df)} games from {key}")
            except Exception as exc:
                logger.warning(f"Failed to load {key}: {exc}")
        return result


# Global singleton
league_registry = LeagueRegistry()


__all__ = [
    "LeagueRegistry",
    "LeagueHealthStatus",
    "league_registry",
]
