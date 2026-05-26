"""CEBL (Canadian Elite Basketball League) data source.

Primary (Python 3.12+): ``ceblpy`` package.
Fallback (Python 3.10+): Web scraping CEBL official site + Basketball-Reference.

The CEBL season runs **May → August** (summer league). Short season means
every game carries more statistical weight and market inefficiencies are
amplified. The CEBL replaced the now-defunct NBL Canada.
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

# Try ceblpy (Python 3.12+)
try:
    import ceblpy.ceblpy as cebl  # type: ignore
    CEBLPY_AVAILABLE = True
except ImportError:
    CEBLPY_AVAILABLE = False
    cebl = None
    logger.info("ceblpy not available (needs Python 3.12+). Using web fallback.")

# Fallback URLs for web scraping
CEBL_SCHEDULE_URL = "https://www.cebl.ca/schedule"
CEBL_STANDINGS_URL = "https://www.cebl.ca/standings"
# Basketball-Reference CEBL page (if available)
BBREF_CEBL = "https://www.basketball-reference.com/international/cebl/"


class CEBLSource(SmallLeagueSource):
    """CEBL data via ceblpy (primary) or web scraping (fallback)."""

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self._teams_cache: Optional[pd.DataFrame] = None
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        })
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
        """Load completed CEBL games."""
        if seasons is None:
            seasons = [2024, 2025]

        if CEBLPY_AVAILABLE:
            return self._load_via_ceblpy(seasons)
        else:
            logger.info("ceblpy unavailable, using web fallback for CEBL")
            return self._load_via_web(seasons)

    def load_upcoming(self, limit: int = 20) -> pd.DataFrame:
        """Load upcoming CEBL games."""
        if CEBLPY_AVAILABLE:
            return self._upcoming_via_ceblpy(limit)
        return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

    def get_teams(self) -> pd.DataFrame:
        """Load CEBL teams."""
        if self._teams_cache is not None:
            return self._teams_cache

        if CEBLPY_AVAILABLE:
            return self._teams_via_ceblpy()

        # Fallback: known CEBL teams
        known_teams = [
            "Brampton Honey Badgers", "Calgary Surge", "Edmonton Stingers",
            "Montreal Alliance", "Niagara River Lions", "Ottawa BlackJacks",
            "Saskatchewan Rattlers", "Scarborough Shooting Stars",
            "Vancouver Bandits", "Winnipeg Sea Bears",
        ]
        records = [
            {"team_id": t.lower().replace(" ", "-"), "team_name": t,
             "team_short": self._abbrev(t), "country": "Canada", "league": "cebl"}
            for t in known_teams
        ]
        self._teams_cache = pd.DataFrame(records)
        return self._teams_cache

    @staticmethod
    def _abbrev(name: str) -> str:
        parts = name.split()
        if len(parts) >= 2:
            return (parts[0][0] + parts[-1][0]).upper()
        return name[:3].upper()

    # ------------------------------------------------------------------
    # ceblpy path
    # ------------------------------------------------------------------

    def _load_via_ceblpy(self, seasons: list[Any]) -> pd.DataFrame:
        all_records = []
        for year in seasons:
            try:
                sched = cebl.load_cebl_schedule(int(year))  # type: ignore
                if sched is None or sched.empty:
                    continue
                records = self._schedule_to_records(sched, year)
                all_records.extend(records)
            except Exception as exc:
                logger.error(f"ceblpy {year} failed: {exc}")
        if not all_records:
            return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))
        df = pd.DataFrame(all_records)
        df["league"] = "cebl"
        self.validate_schema(df, "cebl_historical")
        return self._sort_and_dedup(df)

    def _upcoming_via_ceblpy(self, limit: int) -> pd.DataFrame:
        for year in [2025, 2024]:
            try:
                sched = cebl.load_cebl_schedule(int(year))  # type: ignore
                if sched is None or sched.empty:
                    continue
                upcoming = sched[sched["home_team_score"].isna()].head(limit)
                if upcoming.empty:
                    continue
                records = self._schedule_to_records(upcoming, year)
                df = pd.DataFrame(records)
                df["league"] = "cebl"
                self.validate_schema(df, "cebl_upcoming")
                return self._sort_and_dedup(df)
            except Exception:
                continue
        return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

    def _teams_via_ceblpy(self) -> pd.DataFrame:
        teams: dict[str, dict] = {}
        for year in [2025, 2024]:
            try:
                sched = cebl.load_cebl_schedule(int(year))
                if sched is None or sched.empty:
                    continue
                for col_prefix in ["home", "away"]:
                    name_col = f"{col_prefix}_team"
                    if name_col in sched.columns:
                        for _, row in sched.iterrows():
                            name = row.get(name_col)
                            if name and name not in teams:
                                teams[name] = {
                                    "team_id": str(row.get(f"{col_prefix}_team_id", "")),
                                    "team_name": name,
                                    "team_short": self._abbrev(str(name)),
                                    "country": "Canada",
                                    "league": "cebl",
                                }
            except Exception:
                continue
        self._teams_cache = pd.DataFrame(list(teams.values()))
        return self._teams_cache

    # ------------------------------------------------------------------
    # Web scraping fallback
    # ------------------------------------------------------------------

    def _load_via_web(self, seasons: list[Any]) -> pd.DataFrame:
        """Fallback: attempt to scrape CEBL data from official site."""
        try:
            self._rate_limit()
            resp = self._session.get(CEBL_SCHEDULE_URL, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            records = self._parse_cebl_schedule(soup)
            if records:
                df = pd.DataFrame(records)
                df["league"] = "cebl"
                self.validate_schema(df, "cebl_web")
                return self._sort_and_dedup(df)
        except Exception as exc:
            logger.warning(f"CEBL web scrape failed: {exc}")

        return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

    def _parse_cebl_schedule(self, soup: BeautifulSoup) -> list[dict]:
        """Parse CEBL schedule page HTML. Handles multiple layouts."""
        records: list[dict] = []
        game_els = (
            soup.select("div.schedule-game")
            or soup.select("div.game-card")
            or soup.select("div.match-item")
            or soup.select("tr")
        )

        for el in game_els:
            text = el.get_text(separator=" ", strip=True)
            match = re.search(
                r"([A-Za-z\s.]+?)\s+(\d+)\s+([A-Za-z\s.]+?)\s+(\d+)", text
            )
            if match:
                home = match.group(1).strip()
                away = match.group(3).strip()
                hs, aws = int(match.group(2)), int(match.group(4))
                records.append({
                    "game_id": f"CEBL_web_{home}-{away}",
                    "league": "cebl", "season": "2024", "date": "",
                    "team_id": home.lower().replace(" ", "-"), "team_name": home,
                    "opponent_id": away.lower().replace(" ", "-"), "opponent_name": away,
                    "is_home": 1, "team_score": hs, "opponent_score": aws,
                    "total_points": hs + aws,
                    "result": 1 if hs > aws else 0, "venue": "", "status": "completed",
                })
                # Create away-team row
                records.append({
                    "game_id": f"CEBL_web_{home}-{away}",
                    "league": "cebl", "season": "2024", "date": "",
                    "team_id": away.lower().replace(" ", "-"), "team_name": away,
                    "opponent_id": home.lower().replace(" ", "-"), "opponent_name": home,
                    "is_home": 0, "team_score": aws, "opponent_score": hs,
                    "total_points": hs + aws,
                    "result": 1 if aws > hs else 0, "venue": "", "status": "completed",
                })
        return records

    # ------------------------------------------------------------------
    # Shared helpers (ceblpy path)
    # ------------------------------------------------------------------

    def _schedule_to_records(self, sched: pd.DataFrame, year: int) -> list[dict[str, Any]]:
        """Convert ceblpy schedule DataFrame to canonical records."""
        records = []
        col_map = {
            "game_id": ["game_id", "gameid", "id"],
            "date": ["date", "game_date", "day", "start_date"],
            "home_team": ["home_team", "home", "homeTeam"],
            "away_team": ["away_team", "away", "awayTeam", "visitor"],
            "home_score": ["home_team_score", "home_score", "homeScore", "home_pts"],
            "away_score": ["away_team_score", "away_score", "awayScore", "visitor_pts"],
            "home_id": ["home_team_id", "home_id", "homeTeamId"],
            "away_id": ["away_team_id", "away_id", "awayTeamId"],
            "venue": ["venue", "location", "arena"],
        }

        def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        gid_col = _find_col(sched, col_map["game_id"])
        date_col = _find_col(sched, col_map["date"])
        home_col = _find_col(sched, col_map["home_team"])
        away_col = _find_col(sched, col_map["away_team"])
        hs_col = _find_col(sched, col_map["home_score"])
        as_col = _find_col(sched, col_map["away_score"])
        hid_col = _find_col(sched, col_map["home_id"])
        aid_col = _find_col(sched, col_map["away_id"])
        venue_col = _find_col(sched, col_map["venue"])

        if not all([home_col, away_col]):
            return records

        for idx, row in sched.iterrows():
            try:
                home_score = row.get(hs_col) if hs_col else None
                away_score = row.get(as_col) if as_col else None
                home_score = int(home_score) if pd.notna(home_score) else None
                away_score = int(away_score) if pd.notna(away_score) else None
                total = (home_score + away_score) if (home_score is not None) else None
                gid = row.get(gid_col) if gid_col else f"CEBL_{year}_{idx}"
                raw_date = row.get(date_col) if date_col else ""
                date_str = raw_date.strftime("%Y-%m-%d") if hasattr(raw_date, "strftime") else str(raw_date)[:10]

                records.append({
                    "game_id": f"CEBL_{gid}", "league": "cebl", "season": str(year),
                    "date": date_str,
                    "team_id": str(row.get(hid_col) if hid_col else ""),
                    "team_name": str(row.get(home_col, "")),
                    "opponent_id": str(row.get(aid_col) if aid_col else ""),
                    "opponent_name": str(row.get(away_col, "")),
                    "is_home": 1, "team_score": home_score, "opponent_score": away_score,
                    "total_points": total,
                    "result": (1 if home_score and away_score and home_score > away_score
                               else (0 if home_score and away_score and home_score < away_score else None)),
                    "venue": str(row.get(venue_col) if venue_col else ""),
                    "status": "completed" if home_score is not None else "scheduled",
                })
                records.append({
                    "game_id": f"CEBL_{gid}", "league": "cebl", "season": str(year),
                    "date": date_str,
                    "team_id": str(row.get(aid_col) if aid_col else ""),
                    "team_name": str(row.get(away_col, "")),
                    "opponent_id": str(row.get(hid_col) if hid_col else ""),
                    "opponent_name": str(row.get(home_col, "")),
                    "is_home": 0, "team_score": away_score, "opponent_score": home_score,
                    "total_points": total,
                    "result": (1 if away_score and home_score and away_score > home_score
                               else (0 if away_score and home_score and away_score < home_score else None)),
                    "venue": str(row.get(venue_col) if venue_col else ""),
                    "status": "completed" if home_score is not None else "scheduled",
                })
            except (TypeError, ValueError) as exc:
                logger.debug(f"Skipping CEBL row {idx}: {exc}")
        return records
