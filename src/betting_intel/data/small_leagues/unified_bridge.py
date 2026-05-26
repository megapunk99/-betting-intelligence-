"""Unified data bridge: transform small-league game data into the NBA-compatible
format expected by the feature engineering and ML pipeline.

The key insight: our 84-feature engine (rolling averages, pace, rest, eFG%)
operates on per-team game logs with columns like ``team_score``, ``opponent_score``,
``is_home``, ``date``, ``team_id``, ``opponent_id``. Small-league data in
CANONICAL_SCHEMA already has these exact fields, so the bridge is primarily
about:

1. **League tagging** — Add ``league`` column so the pipeline can learn
   league-specific patterns
2. **Team ID normalization** — Prefix team IDs with league code to avoid
   collisions (e.g., both LNB Pro B and BNXT might have "Paris" or "Brussels")
3. **Season formatting** — Normalize to the format expected by the pipeline
4. **Feature compatibility** — Ensure the output matches what the NBA data
   loader produces (same dtypes, same column names, same index structure)

Usage::
    from betting_intel.data.small_leagues import SmallLeagueIngestion
    from betting_intel.data.small_leagues.unified_bridge import SmallLeagueBridge

    ing = SmallLeagueIngestion()
    bridge = SmallLeagueBridge()

    # Load and bridge all three leagues
    lnb_df = ing.load_historical("lnb_pro_b")
    bridged = bridge.bridge_dataframe(lnb_df)
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from betting_intel.data.small_leagues.base import CANONICAL_SCHEMA

logger = logging.getLogger(__name__)


class SmallLeagueBridge:
    """Transforms small-league CANONICAL_SCHEMA DataFrames into the form
    expected by the NBA feature engineering pipeline.

    The bridge handles:
        - Team ID namespacing (``lnb_pro_b:135248`` instead of ``135248``)
        - Column renaming / dtype normalization
        - Season string formatting
        - Merging multiple league DataFrames into a single training set
    """

    # Mapping from canonical column names to the names expected by the NBA pipeline
    CANONICAL_TO_NBA_PIPELINE = {
        "game_id": "game_id",
        "league": "league",
        "season": "season",
        "date": "date",
        "team_id": "team_id",
        "team_name": "team_name",
        "opponent_id": "opponent_id",
        "opponent_name": "opponent_name",
        "is_home": "IS_HOME",
        "team_score": "team_score",
        "opponent_score": "opponent_score",
        "total_points": "total_points",
        "result": "result",
    }

    def __init__(self, league_prefix: bool = True):
        """
        Args:
            league_prefix: If True, prefix team IDs with league code
                          (e.g. ``lnb_pro_b:135248``) to avoid collisions.
        """
        self.league_prefix = league_prefix

    def bridge_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert a CANONICAL_SCHEMA DataFrame to NBA-pipeline format.

        The output DataFrame maintains the same index structure as the
        NBA game logs used by ``build_all_features()``, with columns
        like ``PTS``, ``opponent_PTS``, ``IS_HOME``, ``date``, ``team_id``,
        ``opponent_id``, and ``result``.

        Args:
            df: DataFrame conforming to CANONICAL_SCHEMA (e.g., from
                ``SmallLeagueIngestion.load_historical()``).

        Returns:
            DataFrame in NBA-pipeline compatible format.
        """
        if df.empty:
            return df

        result = df.copy()

        # 1. League-prefix team IDs to avoid cross-league collisions
        if self.league_prefix and "league" in result.columns:
            league = result["league"].iloc[0] if result["league"].nunique() == 1 else "multi"
            prefix = result["league"].iloc[0] if result["league"].nunique() == 1 else ""

            for col in ["team_id", "opponent_id"]:
                if col in result.columns:
                    result[col] = result[col].astype(str)
                    if "" in result[col].values:
                        pass  # keep empty strings as-is
                    if prefix:
                        result[col] = result[col].apply(
                            lambda x: f"{prefix}:{x}" if x and ":" not in str(x) else x
                        )

        # 2. Ensure date is datetime
        if "date" in result.columns:
            result["date"] = pd.to_datetime(result["date"], errors="coerce")

        # 3. Ensure numerical columns are proper types
        for col in ["team_score", "opponent_score", "total_points", "is_home"]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        # 4. Add NBA-pipeline aliases for compatibility
        if "team_score" in result.columns and "PTS" not in result.columns:
            result["PTS"] = result["team_score"]
            result["opponent_PTS"] = result["opponent_score"]

        # 5. Fix is_home naming convention
        if "is_home" in result.columns:
            result["IS_HOME"] = result["is_home"]

        # 6. Remove games with null scores (upcoming/unplayed)
        result = result.dropna(subset=["team_score", "opponent_score"]).copy()

        # 7. Sort by date for rolling average computation
        if "date" in result.columns and not result.empty:
            result = result.sort_values("date").reset_index(drop=True)

        return result

    def merge_leagues(
        self,
        league_dfs: dict[str, pd.DataFrame],
        add_league_indicator: bool = True,
    ) -> pd.DataFrame:
        """Merge multiple league DataFrames into one training set.

        Args:
            league_dfs: Dict mapping league key to its bridged DataFrame.
            add_league_indicator: If True, keep the ``league`` column as
                                  a categorical feature for the model.

        Returns:
            Combined DataFrame with all league data.
        """
        combined: list[pd.DataFrame] = []
        for _league_key, df in league_dfs.items():
            if not df.empty:
                combined.append(df)

        if not combined:
            return pd.DataFrame()

        result = pd.concat(combined, ignore_index=True)

        # Ensure consistent dtypes across leagues
        for col in ["team_score", "opponent_score", "PTS", "opponent_PTS"]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        if "date" in result.columns:
            result["date"] = pd.to_datetime(result["date"], errors="coerce")

        return result.sort_values("date").reset_index(drop=True)

    @staticmethod
    def generate_fake_nba_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add dummy statistical columns required by the feature engine.

        The NBA feature engine expects columns like ``MIN_*``, ``eFG_*``,
        ``Pace_*``, etc. Small-league sources don't provide these, so we
        seed sensible league-aware defaults.

        .. important::
            This is a **stopgap**. These are NOT real advanced stats — they
            are static league-average estimates. The model will treat all
            teams as identical on these dimensions, which reduces signal
            but still allows the core features (rolling averages, rest days)
            to work. The real solution is to compute advanced stats from
            small-league play-by-play data.

        **Default values by league:**
            - NBA (48 min games, ~100 pace, ~0.52 eFG%)
            - FIBA / European (40 min games, ~72 pace, ~0.50 eFG%)
            - CEBL (48 min games, ~95 pace, ~0.51 eFG%)
        """
        if df.empty:
            return df

        # Determine league type for appropriate defaults
        if "league" in df.columns and not df.empty:
            league = str(df["league"].iloc[0])
        else:
            league = ""

        if league == "cebl":
            mins = 240.0   # 48 min CEBL games
            pace = 95.0
            efg = 0.51
        elif league in ("lnb_pro_b", "bnxt"):
            mins = 200.0   # 40 min FIBA games
            pace = 72.0
            efg = 0.50
        else:
            mins = 240.0   # Default to NBA standard
            pace = 100.0
            efg = 0.52

        defaults = {
            "MIN_home": mins,
            "MIN_away": mins,
            "eFG_home": efg,
            "eFG_away": efg,
            "Pace_home": pace,
            "Pace_away": pace,
            "rest_days_home": 2.0,
            "rest_days_away": 2.0,
        }

        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        return df
