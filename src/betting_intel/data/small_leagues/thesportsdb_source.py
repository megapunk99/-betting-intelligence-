"""LNB Pro B data source via TheSportsDB free API.

TheSportsDB provides a free tier (key=``123``) with 30 requests/minute.
League ID for French LNB Pro B: **4577**.

API endpoints used:
    - ``eventspastleague.php?id=4577``  → historical games
    - ``eventsnextleague.php?id=4577``  → upcoming games
    - ``search_all_teams.php?l=French_LNB_Pro_B`` → team roster

Each game includes quarter-by-quarter breakdowns in ``strResult``, which we
parse into play-by-play style features.

**Limitations** (free tier):
    - ~100 most recent past events only (no deep history)
    - No player-level box scores
    - No advanced stats (eFG%, pace, etc.) -- we compute those ourselves
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import requests

from betting_intel.data.small_leagues.base import (
    CANONICAL_SCHEMA,
    SmallLeagueSource,
)

logger = logging.getLogger(__name__)

# TheSportsDB free API key (default, rate-limited to 30/min)
API_BASE = "https://www.thesportsdb.com/api/v1/json/123/"
LEAGUE_ID = "4577"  # French LNB Pro B
TEAM_SEARCH_TERM = "French_LNB_Pro_B"


class TheSportsDBSource(SmallLeagueSource):
    """Data source for French LNB Pro B via TheSportsDB API."""

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self._teams_cache: Optional[pd.DataFrame] = None
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "BettingIntel/0.1 (research project)"}
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_historical(
        self, seasons: Optional[list[Any]] = None
    ) -> pd.DataFrame:
        """Load recent completed games from TheSportsDB.

        .. note::
            The free tier returns only the ~100 most recent events. For
            deep historical data, a paid API key or alternative source
            is required.
        """
        logger.info("Fetching LNB Pro B past events from TheSportsDB...")
        events = self._fetch(f"eventspastleague.php?id={LEAGUE_ID}")

        if not events:
            logger.warning("No past events returned. TheSportsDB may be rate-limited.")
            return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

        logger.info(f"Got {len(events)} past events")
        records = [self._event_to_record(e) for e in events]
        records = [r for r in records if r is not None]

        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

        # Filter by season if specified
        if seasons:
            df = df[df["season"].isin(seasons)]

        # Also create away-team rows
        df = self._expand_home_away(df)
        df["league"] = "lnb_pro_b"
        self.validate_schema(df, "lnb_pro_b_historical")
        return self._sort_and_dedup(df)

    def load_upcoming(self, limit: int = 20) -> pd.DataFrame:
        """Load upcoming LNB Pro B games."""
        logger.info("Fetching LNB Pro B upcoming events...")
        events = self._fetch(f"eventsnextleague.php?id={LEAGUE_ID}")

        if not events:
            logger.warning("No upcoming events.")
            return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

        logger.info(f"Got {len(events)} upcoming events")
        records = []
        for e in events[:limit]:
            rec = self._event_to_record(e, upcoming=True)
            if rec:
                records.append(rec)

        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

        df = self._expand_home_away(df)
        df["league"] = "lnb_pro_b"
        self.validate_schema(df, "lnb_pro_b_upcoming")
        return self._sort_and_dedup(df)

    def get_teams(self) -> pd.DataFrame:
        """Load LNB Pro B team metadata."""
        if self._teams_cache is not None:
            return self._teams_cache

        logger.info("Fetching LNB Pro B teams...")
        data = self._fetch_raw(f"search_all_teams.php?l={TEAM_SEARCH_TERM}")
        teams = (data or {}).get("teams", [])

        records = []
        for t in teams:
            records.append(
                {
                    "team_id": t.get("idTeam"),
                    "team_name": t.get("strTeam"),
                    "team_short": t.get("strTeamShort", ""),
                    "team_badge": t.get("strBadge", ""),
                    "team_stadium": t.get("strStadium", ""),
                    "city": t.get("strLocation", ""),
                    "country": "France",
                    "league": "lnb_pro_b",
                }
            )

        self._teams_cache = pd.DataFrame(records)
        return self._teams_cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(self, endpoint: str) -> list[dict]:
        """Fetch JSON list from TheSportsDB endpoint."""
        data = self._fetch_raw(endpoint)
        # TheSportsDB returns { "events": [...] } or { "event": [...] } or { "teams": [...] }
        if isinstance(data, dict):
            for key in ("events", "event", "teams"):
                val = data.get(key)
                if isinstance(val, list):
                    return val
        return []

    def _fetch_raw(self, endpoint: str) -> Optional[dict]:
        """Raw API call."""
        url = API_BASE + endpoint
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error(f"TheSportsDB API error: {exc}")
            return None

    def _event_to_record(
        self, e: dict, upcoming: bool = False
    ) -> Optional[dict]:
        """Convert a single TheSportsDB event to a canonical game record (home team row)."""
        try:
            home_score = e.get("intHomeScore")
            away_score = e.get("intAwayScore")

            if upcoming:
                home_score = None
                away_score = None

            home_score = int(home_score) if home_score is not None else None
            away_score = int(away_score) if away_score is not None else None
            total = (home_score + away_score) if (home_score is not None) else None

            timestamp = e.get("strTimestamp") or e.get("dateEvent", "")
            date_str = timestamp[:10] if timestamp else ""

            return {
                "game_id": f"LNB_{e.get('idEvent')}",
                "league": "lnb_pro_b",
                "season": e.get("strSeason", ""),
                "date": date_str,
                "team_id": e.get("idHomeTeam"),
                "team_name": e.get("strHomeTeam", ""),
                "opponent_id": e.get("idAwayTeam"),
                "opponent_name": e.get("strAwayTeam", ""),
                "is_home": 1,
                "team_score": home_score,
                "opponent_score": away_score,
                "total_points": total,
                "result": (
                    1
                    if (home_score is not None and away_score is not None and home_score > away_score)
                    else (0 if (home_score is not None and away_score is not None and home_score < away_score) else None)
                ),
                "venue": e.get("strVenue", ""),
                "status": "completed" if not upcoming else "scheduled",
            }
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(f"Skipping malformed event {e.get('idEvent')}: {exc}")
            return None

    @staticmethod
    def _expand_home_away(df: pd.DataFrame) -> pd.DataFrame:
        """Create away-team rows by swapping team/opponent columns."""
        if df.empty:
            return df

        home_rows = df.copy()

        away_rows = home_rows.copy()
        away_rows["is_home"] = 0
        # Swap team and opponent
        away_rows["team_id"], away_rows["opponent_id"] = (
            away_rows["opponent_id"],
            away_rows["team_id"],
        )
        away_rows["team_name"], away_rows["opponent_name"] = (
            away_rows["opponent_name"],
            away_rows["team_name"],
        )
        # Scores stay the same relative to home/away perspective
        # team_score stays as what the *original home team* scored
        # But we need to flip: the away team's score is opponent_score
        away_rows["team_score"], away_rows["opponent_score"] = (
            away_rows["opponent_score"],
            away_rows["team_score"],
        )
        # Result flips
        away_rows["result"] = away_rows["result"].apply(
            lambda x: 1 - x if pd.notna(x) else None
        )

        return pd.concat([home_rows, away_rows], ignore_index=True)
