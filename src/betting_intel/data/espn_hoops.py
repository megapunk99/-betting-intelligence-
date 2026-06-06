"""
ESPN Basketball Data Source — fetches historical game data for multiple basketball leagues.

Supports:
  - NBA (basketball/nba)
  - WNBA (basketball/wnba)
  - NCAAB (basketball/ncaab/men)

Uses ESPN's public API (no key required). Returns data in a format compatible
with the existing feature engineering pipeline.

Usage:
    from betting_intel.data.espn_hoops import ESPNLeagueSource

    source = ESPNLeagueSource()
    wnba_df = source.load_historical("wnba", seasons=[2024, 2025])
    ncaab_df = source.load_historical("ncaab", seasons=["2024-2025"])
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ESPN API base URLs
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/{sport}/scoreboard"
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/{sport}/summary"

# Mapping from our league keys to ESPN sport paths
LEAGUE_TO_ESPN_PATH: dict[str, str] = {
    "nba": "basketball/nba",
    "wnba": "basketball/wnba",
    "ncaab": "basketball/ncaab/men",
    "wncaab": "basketball/ncaab/women",
}

# Mapping from ESPN team names to short names (for matching with DB)
ESPN_TEAM_TO_SHORT: dict[str, str] = {
    # NBA
    "Atlanta Hawks": "Hawks", "Boston Celtics": "Celtics", "Brooklyn Nets": "Nets",
    "Charlotte Hornets": "Hornets", "Chicago Bulls": "Bulls", "Cleveland Cavaliers": "Cavaliers",
    "Dallas Mavericks": "Mavericks", "Denver Nuggets": "Nuggets", "Detroit Pistons": "Pistons",
    "Golden State Warriors": "Warriors", "Houston Rockets": "Rockets", "Indiana Pacers": "Pacers",
    "LA Clippers": "Clippers", "Los Angeles Lakers": "Lakers", "Memphis Grizzlies": "Grizzlies",
    "Miami Heat": "Heat", "Milwaukee Bucks": "Bucks", "Minnesota Timberwolves": "Timberwolves",
    "New Orleans Pelicans": "Pelicans", "New York Knicks": "Knicks",
    "Oklahoma City Thunder": "Thunder", "Orlando Magic": "Magic",
    "Philadelphia 76ers": "76ers", "Phoenix Suns": "Suns",
    "Portland Trail Blazers": "Trail Blazers", "Sacramento Kings": "Kings",
    "San Antonio Spurs": "Spurs", "Toronto Raptors": "Raptors",
    "Utah Jazz": "Jazz", "Washington Wizards": "Wizards",
    # WNBA
    "Atlanta Dream": "Dream", "Chicago Sky": "Sky", "Connecticut Sun": "Sun",
    "Dallas Wings": "Wings", "Indiana Fever": "Fever", "Las Vegas Aces": "Aces",
    "Los Angeles Sparks": "Sparks", "Minnesota Lynx": "Lynx", "New York Liberty": "Liberty",
    "Phoenix Mercury": "Mercury", "Seattle Storm": "Storm", "Washington Mystics": "Mystics",
}

# Default season dates per league (used to find the right year to query)
LEAGUE_DEFAULT_MONTHS: dict[str, tuple[int, int]] = {
    "nba": (10, 6),      # Oct-Jun
    "wnba": (5, 9),      # May-Sep
    "ncaab": (11, 4),    # Nov-Apr
    "wncaab": (11, 4),   # Nov-Apr
}


def _team_short(name: str) -> str:
    """Convert an ESPN team name to a short name."""
    return ESPN_TEAM_TO_SHORT.get(name, name.split()[-1] if name else name)


class ESPNLeagueSource:
    """
    Fetch historical game data from ESPN's public API for supported leagues.

    The ESPN API provides scoreboard data with game results, team names,
    and scores. No API key is required.

    Usage:
        source = ESPNLeagueSource()
        df = source.load_historical("nba", seasons=[2024])
        upcoming = source.load_upcoming("nba", limit=10)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        })
        self._last_request = 0.0

    def _rate_limit(self):
        """Rate limit API requests (no key = be respectful)."""
        elapsed = time.time() - self._last_request
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request = time.time()

    def load_historical(
        self, league_key: str, seasons: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Load historical game results for a league from ESPN.

        Args:
            league_key: 'nba', 'wnba', 'ncaab', or 'wncaab'
            seasons: List of season identifiers (years for pro, strings for college)

        Returns:
            DataFrame with columns: game_id, date, home_team, away_team,
            home_score, away_score, total_points, league, season
        """
        if league_key not in LEAGUE_TO_ESPN_PATH:
            logger.warning(f"League '{league_key}' not available on ESPN API")
            return pd.DataFrame()

        espn_path = LEAGUE_TO_ESPN_PATH[league_key]

        if seasons is None:
            # Default: current year and previous year
            current_year = datetime.now().year
            seasons = [current_year, current_year - 1]

        all_records = []
        for season in seasons:
            try:
                records = self._fetch_season(espn_path, league_key, season)
                all_records.extend(records)
                logger.info(f"ESPN {league_key} {season}: {len(records)} games")
            except Exception as e:
                logger.warning(f"ESPN {league_key} {season} failed: {e}")

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
        return df

    def load_upcoming(self, league_key: str, limit: int = 20) -> pd.DataFrame:
        """
        Load upcoming scheduled games from ESPN.

        Returns DataFrame with home_team, away_team, date, league columns.
        """
        if league_key not in LEAGUE_TO_ESPN_PATH:
            return pd.DataFrame()

        espn_path = LEAGUE_TO_ESPN_PATH[league_key]
        self._rate_limit()

        try:
            url = ESPN_SCOREBOARD_URL.format(sport=espn_path)
            params = {"dates": datetime.now().strftime("%Y%m%d"), "limit": 300}
            resp = self._session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            records = []
            events = data.get("events", []) + data.get("upcoming", [])
            for event in events[:limit]:
                comp = self._get_competition(event)
                if not comp:
                    continue

                home, away = self._get_teams(comp)
                if not home or not away:
                    continue

                date_str = event.get("date", "")[:10]
                status = comp.get("status", {}).get("type", {}).get("name", "")
                is_completed = status in {"STATUS_FINAL", "STATUS_FINAL_ALT"}

                records.append({
                    "game_id": event.get("id", f"espn_{league_key}_{date_str}"),
                    "league": league_key,
                    "date": date_str,
                    "home_team": _team_short(home.get("displayName", "")),
                    "away_team": _team_short(away.get("displayName", "")),
                    "home_score": home.get("score", None) if is_completed else None,
                    "away_score": away.get("score", None) if is_completed else None,
                    "status": "completed" if is_completed else "scheduled",
                })

            return pd.DataFrame(records)

        except Exception as e:
            logger.warning(f"ESPN upcoming {league_key} failed: {e}")
            return pd.DataFrame()

    def _fetch_season(self, espn_path: str, league_key: str, season: Any) -> list[dict]:
        """
        Fetch all games for a season by iterating through dates.
        ESPN's API returns scoreboard data date-by-date.
        """
        # Determine the date range for this season
        start_month, end_month = LEAGUE_DEFAULT_MONTHS.get(league_key, (1, 12))
        season_year = int(season)

        # For seasons spanning two years (like NBA), the season label is the
        # starting year. So "2024" means Oct 2024 - Jun 2025.
        if start_month > end_month:
            # Season spans two years
            from datetime import date as dt_date, timedelta
            start_date = dt_date(season_year, start_month, 1)
            end_date = dt_date(season_year + 1, end_month + 1, 1) + timedelta(days=-1)
        else:
            from datetime import date as dt_date, timedelta
            start_date = dt_date(season_year, start_month, 1)
            end_date = dt_date(season_year, end_month + 1, 1) + timedelta(days=-1)

        all_games = {}
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y%m%d")
            records = self._fetch_date(espn_path, league_key, date_str)
            for rec in records:
                gid = rec["game_id"]
                if gid not in all_games:
                    all_games[gid] = rec
            current += timedelta(days=1)  # Step by day to capture ALL games

        return list(all_games.values())

    def _fetch_date(
        self, espn_path: str, league_key: str, date_str: str
    ) -> list[dict]:
        """Fetch all games for a specific date."""
        self._rate_limit()

        try:
            url = ESPN_SCOREBOARD_URL.format(sport=espn_path)
            resp = self._session.get(
                url, params={"dates": date_str, "limit": 300}, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        records = []
        events = data.get("events", []) + data.get("upcoming", [])
        for event in events:
            comp = self._get_competition(event)
            if not comp:
                continue

            home, away = self._get_teams(comp)
            if not home or not away:
                continue

            status = comp.get("status", {}).get("type", {}).get("name", "")
            is_completed = status in {"STATUS_FINAL", "STATUS_FINAL_ALT", "STATUS_FINAL_ALT_2"}
            if not is_completed:
                continue

            home_score = home.get("score")
            away_score = away.get("score")
            if home_score is None or away_score is None:
                continue

            home_short = _team_short(home.get("displayName", ""))
            away_short = _team_short(away.get("displayName", ""))
            event_date = event.get("date", "")[:10]

            # Two rows (one per team) to match CANONICAL_SCHEMA
            gid = event.get("id", f"espn_{league_key}_{event_date}_{home_short}_{away_short}")

            records.append({
                "game_id": gid,
                "league": league_key,
                "season": str(season_year_from_date(event_date)),
                "date": event_date,
                "home_team": home_short,
                "away_team": away_short,
                "home_score": int(home_score),
                "away_score": int(away_score),
                "total_points": int(home_score) + int(away_score),
                "home_win": 1 if int(home_score) > int(away_score) else 0,
            })

        return records

    @staticmethod
    def _get_competition(event: dict) -> Optional[dict]:
        """Extract the competition object from an ESPN event."""
        competitions = event.get("competitions", [])
        return competitions[0] if competitions else None

    @staticmethod
    def _get_teams(competition: dict) -> tuple[Optional[dict], Optional[dict]]:
        """Extract home and away teams from a competition."""
        competitors = competition.get("competitors", [])
        home = None
        away = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home = c.get("team", {})
            else:
                away = c.get("team", {})
        return home, away


def season_year_from_date(date_str: str) -> int:
    """Extract a season year from a date string."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        # For NBA/college seasons that start in one year and end in the next,
        # the season label is the starting year
        if dt.month >= 10:
            return dt.year
        return dt.year
    except Exception:
        return datetime.now().year
