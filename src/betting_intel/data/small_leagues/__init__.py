"""Small-league data sources: LNB Pro B, CEBL, BNXT League.

Each source implements a standard interface (load_historical, load_upcoming, get_teams)
and outputs pandas DataFrames in a unified canonical schema.

Usage:
    from betting_intel.data.small_leagues import SmallLeagueIngestion
    ing = SmallLeagueIngestion()
    lnb_df = ing.load_league("lnb_pro_b", seasons=["2025-2026"])
    cebl_df = ing.load_league("cebl", seasons=[2025])
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from betting_intel.data.small_leagues.base import (
    CANONICAL_SCHEMA,
    LEAGUE_METADATA,
    SmallLeagueSource,
)
from betting_intel.data.small_leagues.thesportsdb_source import TheSportsDBSource
from betting_intel.data.small_leagues.cebl_source import CEBLSource
from betting_intel.data.small_leagues.bnxt_source import BNXTSource


class SmallLeagueIngestion:
    """Central ingestion coordinator for all small-league data sources."""

    SOURCES: dict[str, type[SmallLeagueSource]] = {
        "lnb_pro_b": TheSportsDBSource,
        "cebl": CEBLSource,
        "bnxt": BNXTSource,
    }

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._sources: dict[str, SmallLeagueSource] = {}

    def _get_source(self, league_key: str) -> SmallLeagueSource:
        if league_key not in self.SOURCES:
            available = ", ".join(sorted(self.SOURCES))
            raise ValueError(
                f"Unknown league '{league_key}'. Available: {available}"
            )
        if league_key not in self._sources:
            cls = self.SOURCES[league_key]
            self._sources[league_key] = cls(cache_dir=self.cache_dir)
        return self._sources[league_key]

    def load_historical(
        self, league_key: str, seasons: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Load historical game data for a given small league.

        Args:
            league_key: One of 'lnb_pro_b', 'cebl', 'bnxt'.
            seasons: List of season identifiers. Defaults to most recent season.

        Returns:
            DataFrame in CANONICAL_SCHEMA.
        """
        source = self._get_source(league_key)
        return source.load_historical(seasons=seasons)

    def load_upcoming(
        self, league_key: str, limit: int = 20
    ) -> pd.DataFrame:
        """Load upcoming scheduled games for a given small league.

        Args:
            league_key: One of 'lnb_pro_b', 'cebl', 'bnxt'.
            limit: Maximum number of upcoming games to fetch.

        Returns:
            DataFrame in CANONICAL_SCHEMA (scores will be None/NaN).
        """
        source = self._get_source(league_key)
        return source.load_upcoming(limit=limit)

    def get_teams(self, league_key: str) -> pd.DataFrame:
        """Get team metadata for a given small league."""
        source = self._get_source(league_key)
        return source.get_teams()

    @staticmethod
    def list_available_leagues() -> dict[str, dict]:
        """Return metadata about all available small leagues."""
        return dict(LEAGUE_METADATA)


__all__ = [
    "SmallLeagueIngestion",
    "CANONICAL_SCHEMA",
    "LEAGUE_METADATA",
]
