"""Small-league data sources: LNB Pro B, CEBL, BNXT League, WNBA, EuroLeague Women, soccer.

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
from betting_intel.data.small_leagues.wnba_source import WNBASource
from betting_intel.data.small_leagues.euroleague_women_source import EuroLeagueWomenSource
from betting_intel.data.small_leagues.soccer_source import SoccerLeagueSource, SoccerLeagueFactory
from betting_intel.data.small_leagues.league_registry import LeagueRegistry, LeagueHealthStatus, league_registry


class SmallLeagueIngestion:
    """Central ingestion coordinator for all small-league data sources."""

    SOURCES: dict[str, type[SmallLeagueSource]] = {
        "lnb_pro_b": TheSportsDBSource,
        "cebl": CEBLSource,
        "bnxt": BNXTSource,
        "wnba": WNBASource,
        "euroleague_women": EuroLeagueWomenSource,
    }

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._sources: dict[str, SmallLeagueSource] = {}

    def _get_source(self, league_key: str) -> SmallLeagueSource:
        if league_key.startswith("soccer_"):
            return self._get_soccer_source(league_key)
        if league_key not in self.SOURCES:
            available = ", ".join(sorted(self.SOURCES))
            raise ValueError(
                f"Unknown league '{league_key}'. Available: {available}"
            )
        if league_key not in self._sources:
            cls = self.SOURCES[league_key]
            self._sources[league_key] = cls(cache_dir=self.cache_dir)
        return self._sources[league_key]

    def _get_soccer_source(self, league_key: str) -> SoccerLeagueSource:
        """Get or create a soccer league source."""
        if league_key not in self._sources:
            soccer_key = league_key.replace("soccer_", "")
            self._sources[league_key] = SoccerLeagueSource(
                league_key=soccer_key,
                cache_dir=self.cache_dir,
            )
        return self._sources[league_key]

    def load_historical(
        self, league_key: str, seasons: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Load historical game data for a given small league.

        Args:
            league_key: One of 'lnb_pro_b', 'cebl', 'bnxt', 'wnba', 'euroleague_women',
                        'soccer_belgian_pro_league', 'soccer_scottish_championship', etc.
            seasons: List of season identifiers. Defaults to most recent season.

        Returns:
            DataFrame in CANONICAL_SCHEMA.
        """
        source = self._get_source(league_key)
        return source.load_historical(seasons=seasons)

    def load_upcoming(
        self, league_key: str, limit: int = 20
    ) -> pd.DataFrame:
        """Load upcoming scheduled games for a given small league."""
        source = self._get_source(league_key)
        return source.load_upcoming(limit=limit)

    def get_teams(self, league_key: str) -> pd.DataFrame:
        """Get team metadata for a given small league."""
        source = self._get_source(league_key)
        return source.get_teams()

    @staticmethod
    def list_available_leagues() -> dict[str, dict]:
        """Return metadata about all available small leagues."""
        from betting_intel.data.small_leagues.soccer_source import SOCCER_LEAGUES
        result = dict(LEAGUE_METADATA)
        for key, meta in SOCCER_LEAGUES.items():
            result[f"soccer_{key}"] = {
                **meta,
                "data_source": "football-data.org / Flashscore",
            }
        return result


__all__ = [
    "SmallLeagueIngestion",
    "CANONICAL_SCHEMA",
    "LEAGUE_METADATA",
    "LeagueRegistry",
    "LeagueHealthStatus",
    "league_registry",
    "WNBASource",
    "EuroLeagueWomenSource",
    "SoccerLeagueSource",
    "SoccerLeagueFactory",
]
