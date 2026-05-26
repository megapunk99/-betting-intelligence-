"""Base schema, metadata, and abstract interface for small-league data sources.

All small-league sources output DataFrames conforming to CANONICAL_SCHEMA.

The canonical schema mirrors the NBA game-log structure used by the rest of the
pipeline, so ``unified_bridge.py`` can map directly into the 84-feature engine.

Why canonical? The whole point of targeting small leagues is that betting markets
are softer (less liquidity, fewer sharps), which means bookmaker errors are
larger and more persistent. But the ML pipeline doesn't care *which* league
the data comes from -- it only cares that columns are consistent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical column schema (one row = one team's performance in one game)
# ---------------------------------------------------------------------------

CANONICAL_SCHEMA: dict[str, str] = {
    # ── Identifiers ──────────────────────────────────────────────────────
    "game_id": "Unique game identifier (e.g. 'LNB_2476456')",
    "league": "League key: 'lnb_pro_b', 'cebl', 'bnxt'",
    "season": "Season string (e.g. '2025-2026', '2025')",
    "date": "Game date as ISO string (YYYY-MM-DD)",
    # ── Teams ────────────────────────────────────────────────────────────
    "team_id": "Team identifier for this row's team",
    "team_name": "Full team name",
    "opponent_id": "Opponent team identifier",
    "opponent_name": "Full opponent team name",
    "is_home": "1 if this team is home, 0 if away",
    # ── Scores ───────────────────────────────────────────────────────────
    "team_score": "Points scored by this team",
    "opponent_score": "Points scored by opponent",
    "total_points": "Sum of both scores (team_score + opponent_score)",
    "result": "1 if win (team_score > opponent_score), 0 if loss, NaN if upcoming",
    # ── Game metadata ────────────────────────────────────────────────────
    "venue": "Venue name (if available)",
    "status": "Game status: 'scheduled', 'completed', or 'live'",
}

# ---------------------------------------------------------------------------
# League metadata
# ---------------------------------------------------------------------------

LEAGUE_METADATA: dict[str, dict[str, Any]] = {
    "lnb_pro_b": {
        "name": "French LNB Pro B",
        "country": "France",
        "tier": "Second division",
        "num_teams": 18,
        "season_format": "Regular season + playoffs",
        "typical_season_months": "September → May",
        "data_source": "TheSportsDB (free API)",
        "market_notes": (
            "French second-division basketball. Low betting volume = "
            "softer lines. Bookmakers rely on algorithms trained on "
            "top-tier leagues, creating exploitable gaps."
        ),
    },
    "cebl": {
        "name": "Canadian Elite Basketball League",
        "country": "Canada",
        "tier": "First division (Canada)",
        "num_teams": 10,
        "season_format": "Regular season (20-24 games) + playoffs",
        "typical_season_months": "May → August",
        "data_source": "ceblpy Python package",
        "market_notes": (
            "Summer league with short season. Low data volume means "
            "bookmakers have limited historical baselines. Momentum and "
            "rest-day edges are amplified."
        ),
    },
    "bnxt": {
        "name": "BNXT League",
        "country": "Belgium / Netherlands",
        "tier": "First division (combined league)",
        "num_teams": 19,
        "season_format": "Regular season (National + BNXT phases) + playoffs",
        "typical_season_months": "September → May",
        "data_source": "Proballers.com (web scraping)",
        "market_notes": (
            "Cross-border league with complex scheduling (national + "
            "international phases). Low international betting attention. "
            "Home-court and travel-distance edges likely underpriced."
        ),
    },
}


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class SmallLeagueSource(ABC):
    """Abstract interface that every small-league data source implements.

    Subclasses must implement:
        - load_historical(seasons) -> pd.DataFrame
        - load_upcoming(limit)      -> pd.DataFrame
        - get_teams()               -> pd.DataFrame

    The returned DataFrames must conform to CANONICAL_SCHEMA (extra columns
    are allowed and preserved).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir

    @abstractmethod
    def load_historical(
        self, seasons: Optional[list[Any]] = None
    ) -> pd.DataFrame:
        """Load historical completed games for the given seasons.

        Args:
            seasons: List of season identifiers (type varies by source).
                     If None, loads the most recent complete season.

        Returns:
            DataFrame conforming to CANONICAL_SCHEMA.
        """
        ...

    @abstractmethod
    def load_upcoming(self, limit: int = 20) -> pd.DataFrame:
        """Load upcoming scheduled games.

        Returns:
            DataFrame conforming to CANONICAL_SCHEMA with result/score
            columns set to NaN for unscheduled games.
        """
        ...

    @abstractmethod
    def get_teams(self) -> pd.DataFrame:
        """Load team metadata (id, name, city, logo URL if available).

        Returns:
            DataFrame with at minimum 'team_id' and 'team_name' columns.
        """
        ...

    def validate_schema(self, df: pd.DataFrame, name: str = "") -> None:
        """Validate that a DataFrame conforms to CANONICAL_SCHEMA."""
        required = set(CANONICAL_SCHEMA.keys())
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame '{name}' missing required columns: {missing}"
            )

    def _sort_and_dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by date and drop duplicate game rows."""
        if "date" in df.columns and not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        return df.drop_duplicates(subset=["game_id", "team_id"], keep="last")
