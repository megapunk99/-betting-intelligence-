"""
Minor/niche soccer leagues data source.

Why niche soccer leagues:
- Extremely low betting volume = very soft lines
- Limited data used by algorithmic bookmakers
- Multi-league coverage creates cross-league arbitrage
- Complicated promotion/relegation dynamics create inefficiencies
- Lower-tier leagues have minimal sharp-bettor attention

Target leagues:
1. Scottish Championship (second division Scotland)
2. Belgian Pro League (smaller European league)
3. Portuguese Segunda Liga (second division Portugal)
4. Nordic leagues (Swedish Allsvenskan, Norwegian Eliteserien) — summer seasons
5. Women's Super League (England)

Data sources:
1. Football-data.org (free tier, historical results)
2. API-Football (RapidAPI, comprehensive coverage)
3. ESPN FC (web scraping fallback)
4. Flashscore (web scraping fallback)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from betting_intel.data.small_leagues.base import (
    CANONICAL_SCHEMA,
    SmallLeagueSource,
)

logger = logging.getLogger(__name__)

# Target soccer leagues and their identifiers
SOCCER_LEAGUES = {
    "scottish_championship": {
        "name": "Scottish Championship",
        "country": "Scotland",
        "tier": "Second division",
        "api_football_id": 2046,
        "num_teams": 10,
    },
    "belgian_pro_league": {
        "name": "Belgian Pro League",
        "country": "Belgium",
        "tier": "First division",
        "api_football_id": 144,
        "num_teams": 16,
    },
    "portuguese_segunda": {
        "name": "Portuguese Segunda Liga",
        "country": "Portugal",
        "tier": "Second division",
        "api_football_id": 2129,
        "num_teams": 18,
    },
    "swedish_allsvenskan": {
        "name": "Swedish Allsvenskan",
        "country": "Sweden",
        "tier": "First division",
        "api_football_id": 113,
        "num_teams": 16,
    },
    "womens_super_league": {
        "name": "Women's Super League (England)",
        "country": "England",
        "tier": "First division (women)",
        "api_football_id": 1988,
        "num_teams": 12,
    },
}

FOOTBALL_DATA_BASE = "https://www.football-data.org"
FLASHSCORE_BASE = "https://www.flashscore.com"


class SoccerLeagueSource(SmallLeagueSource):
    """Data source for niche soccer leagues (NEW soccer format).

    Soccer data is mapped to the basketball CANONICAL_SCHEMA:
    - team_score -> goals scored
    - opponent_score -> goals conceded
    - total_points -> total goals in match
    """

    LEAGUE_SUPPORTED = list(SOCCER_LEAGUES.keys())

    def __init__(
        self,
        league_key: str = "belgian_pro_league",
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(cache_dir)
        if league_key not in self.LEAGUE_SUPPORTED:
            raise ValueError(
                f"Unsupported soccer league '{league_key}'. "
                f"Supported: {', '.join(self.LEAGUE_SUPPORTED)}"
            )
        self.league_key = league_key
        self.league_meta = SOCCER_LEAGUES[league_key]
        self._teams_cache: Optional[pd.DataFrame] = None
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        })
        if self._api_key:
            self._session.headers.update({"X-Auth-Token": self._api_key})
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_historical(self, seasons: Optional[list[Any]] = None) -> pd.DataFrame:
        """Load completed games for this soccer league."""
        if seasons is None:
            seasons = ["2024", "2025"]

        all_records: list[dict] = []
        for season in seasons:
            try:
                records = self._scrape_season(str(season))
                if records:
                    logger.info(
                        f"Soccer {self.league_key} {season}: {len(records)} games"
                    )
                    all_records.extend(records)
            except Exception as exc:
                logger.error(f"Failed to scrape {self.league_key} {season}: {exc}")

        if not all_records:
            return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

        df = pd.DataFrame(all_records)
        df["league"] = f"soccer_{self.league_key}"
        self.validate_schema(df, f"soccer_{self.league_key}_historical")
        return self._sort_and_dedup(df)

    def load_upcoming(self, limit: int = 20) -> pd.DataFrame:
        """Load upcoming games."""
        try:
            records = self._scrape_upcoming_football_data(limit)
            if records:
                df = pd.DataFrame(records)
                df["league"] = f"soccer_{self.league_key}"
                self.validate_schema(df, f"soccer_{self.league_key}_upcoming")
                return self._sort_and_dedup(df)
        except Exception as exc:
            logger.warning(f"Upcoming fetch failed for {self.league_key}: {exc}")

        return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

    def get_teams(self) -> pd.DataFrame:
        """Get team metadata from historical data."""
        if self._teams_cache is not None:
            return self._teams_cache

        teams: dict[str, dict] = {}
        for season in ["2024", "2025"]:
            try:
                records = self._scrape_season(season)
                for rec in records:
                    for col in ("team_name", "opponent_name"):
                        name = rec.get(col, "")
                        if name and name not in teams:
                            teams[name] = {
                                "team_id": name.lower().replace(" ", "-"),
                                "team_name": name,
                                "team_short": name[:3].upper(),
                                "country": self.league_meta.get("country", ""),
                                "league": f"soccer_{self.league_key}",
                            }
            except Exception:
                continue

        self._teams_cache = pd.DataFrame(list(teams.values()))
        return self._teams_cache

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def _scrape_season(self, season: str) -> list[dict]:
        """Scrape a season from football-data.org or Flashscore."""
        if self._api_key:
            return self._scrape_via_football_data(season)
        return self._scrape_via_flashscore(season)

    def _scrape_via_football_data(self, season: str) -> list[dict]:
        """Use football-data.org API."""
        league_code = self._get_league_code()
        url = f"{FOOTBALL_DATA_BASE}/v4/competitions/{league_code}/matches"

        self._rate_limit()
        logger.info(f"Fetching {self.league_key} from football-data.org...")

        try:
            resp = self._session.get(url, timeout=15, params={"season": season})
            resp.raise_for_status()
            data = resp.json()
            return self._parse_football_data_response(data)
        except requests.RequestException as exc:
            logger.warning(f"football-data.org error: {exc}")
            return []

    def _scrape_via_flashscore(self, season: str) -> list[dict]:
        """Fallback: scrape from Flashscore."""
        league_slug = self._get_flashscore_slug()
        url = f"{FLASHSCORE_BASE}/football/{league_slug}/"

        self._rate_limit()
        logger.info(f"Fetching {self.league_key} from Flashscore...")

        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            records: list[dict] = []
            matches = (
                soup.select("div.event__match")
                or soup.select("div[class*='match']")
            )

            for match in matches:
                text = match.get_text(" ", strip=True)
                m = re.search(r"(.+?)\s+(\d+)\s*[-:]\s*(\d+)\s+(.+)", text)
                if m:
                    home = m.group(1).strip()
                    away = m.group(4).strip()
                    hs, aws = int(m.group(2)), int(m.group(3))

                    if not home or not away:
                        continue

                    game_id = f"{self.league_key}_fs_{home}-{away}"
                    game_id = re.sub(r"[^a-zA-Z0-9_-]", "_", game_id)

                    records.append({
                        "game_id": game_id,
                        "league": f"soccer_{self.league_key}",
                        "season": season,
                        "date": "",
                        "team_id": home.lower().replace(" ", "-"),
                        "team_name": home,
                        "opponent_id": away.lower().replace(" ", "-"),
                        "opponent_name": away,
                        "is_home": 1,
                        "team_score": hs,
                        "opponent_score": aws,
                        "total_points": hs + aws,
                        "result": 1 if hs > aws else 0,
                        "venue": "",
                        "status": "completed",
                    })
            return records
        except Exception as exc:
            logger.warning(f"Flashscore failed: {exc}")
            return []

    def _scrape_upcoming_football_data(self, limit: int) -> list[dict]:
        """Fetch upcoming matches from football-data.org."""
        if not self._api_key:
            return []

        league_code = self._get_league_code()
        url = f"{FOOTBALL_DATA_BASE}/v4/competitions/{league_code}/matches"
        params = {"status": "SCHEDULED", "limit": min(limit, 100)}

        try:
            resp = self._session.get(url, timeout=15, params=params)
            resp.raise_for_status()
            data = resp.json()
            records: list[dict] = []
            for match in (data.get("matches") or [])[:limit]:
                home = match.get("homeTeam", {}).get("name", "")
                away = match.get("awayTeam", {}).get("name", "")
                date = (match.get("utcDate") or "")[:10]
                match_id = match.get("id", "unknown")

                records.append({
                    "game_id": f"{self.league_key}_fd_{match_id}_{home}-{away}",
                    "league": f"soccer_{self.league_key}",
                    "season": (match.get("season") or {}).get("startDate", "2024")[:4],
                    "date": date,
                    "team_id": home.lower().replace(" ", "-"),
                    "team_name": home,
                    "opponent_id": away.lower().replace(" ", "-"),
                    "opponent_name": away,
                    "is_home": 1,
                    "team_score": None,
                    "opponent_score": None,
                    "total_points": None,
                    "result": None,
                    "venue": match.get("venue", ""),
                    "status": "scheduled",
                })
            return records
        except Exception as exc:
            logger.warning(f"football-data.org upcoming failed: {exc}")
            return []

    def _parse_football_data_response(self, data: dict) -> list[dict]:
        """Parse football-data.org API response."""
        records: list[dict] = []
        for match in data.get("matches") or []:
            if match.get("status") == "SCHEDULED":
                continue

            home = match.get("homeTeam", {}).get("name", "")
            away = match.get("awayTeam", {}).get("name", "")
            score = match.get("score", {})
            full_time = score.get("fullTime", {})
            hs = full_time.get("home")
            aws = full_time.get("away")

            if hs is None or aws is None or not home or not away:
                continue

            date = (match.get("utcDate") or "")[:10]
            match_id = match.get("id", "unknown")
            game_id = f"{self.league_key}_fd_{match_id}_{home}-{away}"
            game_id = re.sub(r"[^a-zA-Z0-9_-]", "_", game_id)

            records.append({
                "game_id": game_id,
                "league": f"soccer_{self.league_key}",
                "season": (match.get("season") or {}).get("startDate", "2024")[:4],
                "date": date,
                "team_id": home.lower().replace(" ", "-"),
                "team_name": home,
                "opponent_id": away.lower().replace(" ", "-"),
                "opponent_name": away,
                "is_home": 1,
                "team_score": hs,
                "opponent_score": aws,
                "total_points": hs + aws,
                "result": 1 if hs > aws else 0,
                "venue": "",
                "status": "completed",
            })
        return records

    def _get_league_code(self) -> str:
        """Map league_key to football-data.org competition code."""
        mapping = {
            "belgian_pro_league": "BPL",
            "scottish_championship": "SC",
            "portuguese_segunda": "PSL",
            "swedish_allsvenskan": "AL",
            "womens_super_league": "WSL",
        }
        return mapping.get(self.league_key, self.league_key.upper())

    def _get_flashscore_slug(self) -> str:
        """Map league_key to Flashscore URL slug."""
        mapping = {
            "belgian_pro_league": "belgium/belgian-pro-league",
            "scottish_championship": "scotland/scottish-championship",
            "portuguese_segunda": "portugal/segunda-liga",
            "swedish_allsvenskan": "sweden/allsvenskan",
            "womens_super_league": "england/womens-super-league",
        }
        return mapping.get(self.league_key, self.league_key)


# Create a factory for soccer leagues
class SoccerLeagueFactory:
    """Factory to create soccer league sources by key."""

    @staticmethod
    def create(
        league_key: str,
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> SoccerLeagueSource:
        return SoccerLeagueSource(
            league_key=league_key,
            cache_dir=cache_dir,
            api_key=api_key,
        )

    @staticmethod
    def list_leagues() -> dict[str, dict]:
        return dict(SOCCER_LEAGUES)


# Register all soccer leagues
try:
    from betting_intel.data.small_leagues.league_registry import league_registry

    for league_key, meta in SOCCER_LEAGUES.items():
        REGISTER_META = dict(meta)
        REGISTER_META["data_source"] = "football-data.org / Flashscore"
        REGISTER_META["market_notes"] = (
            f"Niche soccer league with low betting volume. "
            f"{meta.get('tier', 'League')} with {meta.get('num_teams', '?')} teams. "
            "Lower liquidity means softer lines and persistent market inefficiencies."
        )

        # We register a factory function that creates SoccerLeagueSource instances
        # The closure captures league_key; extra kwargs (cache_dir, api_key) are forwarded
        def _make_source(league_key=league_key, **kwargs):
            return SoccerLeagueSource(league_key=league_key, **kwargs)

        league_registry.register(f"soccer_{league_key}", _make_source, REGISTER_META)
        logger.info(f"Registered soccer league: {league_key}")
except ImportError as exc:
    logger.debug(f"Could not register soccer leagues: {exc}")
