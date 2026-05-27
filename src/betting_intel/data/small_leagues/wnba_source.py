"""
WNBA data source — fetches game results and schedules via web scraping.

Why WNBA is a target:
- Lower betting volume than NBA = softer lines
- Limited historical data used by bookmaker algorithms
- Key player absences (rest, injury) are often mispriced
- Short season (40 games) means each game has higher variance that
  models can exploit

Data sources (in priority order):
1. Basketball-Reference.com — comprehensive WNBA stats
2. ESPN WNBA schedule page — fallback

Note: TheSportsDB has WNBA league ID 4387 (free tier available).
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

# WNBA sources
BBREF_WNBA_BASE = "https://www.basketball-reference.com/wnba"
ESPN_WNBA_URL = "https://www.espn.com/wnba/scoreboard"

# Register this source with the league registry
try:
    from betting_intel.data.small_leagues.league_registry import league_registry

    REGISTER_LEAGUE = (
        "wnba",
        None,  # Will be set after class definition
        {
            "name": "WNBA (Women's National Basketball Association)",
            "country": "USA",
            "tier": "First division (professional)",
            "num_teams": 12,
            "season_format": "Regular season (40 games) + playoffs",
            "typical_season_months": "May -> September",
            "data_source": "Basketball-Reference.com (web scraping)",
            "market_notes": (
                "Women's professional basketball. Lower betting volume = softer lines. "
                "Short season amplifies momentum and fatigue edges. Key player absences "
                "are often mispriced by algorithmic bookmakers."
            ),
        },
    )
except ImportError:
    REGISTER_LEAGUE = None


class WNBASource(SmallLeagueSource):
    """Data source for WNBA via Basketball-Reference.com scraping."""

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
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.5:  # Slower rate limit for BBRef (they're strict)
            time.sleep(1.5 - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_historical(self, seasons: Optional[list[Any]] = None) -> pd.DataFrame:
        """Load completed WNBA games from Basketball-Reference."""
        if seasons is None:
            seasons = [2024, 2025]

        all_records: list[dict] = []
        for year in seasons:
            try:
                records = self._scrape_bbref_season(int(year))
                if records:
                    logger.info(f"WNBA {year}: {len(records)} games found")
                    all_records.extend(records)
                else:
                    logger.warning(f"WNBA {year}: no games found")
            except Exception as exc:
                logger.error(f"Failed to scrape WNBA {year}: {exc}")

        if not all_records:
            return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

        df = pd.DataFrame(all_records)
        df["league"] = "wnba"
        self.validate_schema(df, "wnba_historical")
        return self._sort_and_dedup(df)

    def load_upcoming(self, limit: int = 20) -> pd.DataFrame:
        """Load upcoming WNBA games."""
        self._rate_limit()
        try:
            resp = self._session.get(ESPN_WNBA_URL, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            records = self._parse_espn_scoreboard(soup, upcoming_only=True)
            if records:
                df = pd.DataFrame(records[:limit])
                df["league"] = "wnba"
                self.validate_schema(df, "wnba_upcoming")
                return self._sort_and_dedup(df)
        except Exception as exc:
            logger.error(f"Failed to fetch WNBA upcoming: {exc}")

        return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

    def get_teams(self) -> pd.DataFrame:
        """Load WNBA team metadata."""
        if self._teams_cache is not None:
            return self._teams_cache

        known_teams = [
            "Atlanta Dream", "Chicago Sky", "Connecticut Sun",
            "Dallas Wings", "Indiana Fever", "Las Vegas Aces",
            "Los Angeles Sparks", "Minnesota Lynx", "New York Liberty",
            "Phoenix Mercury", "Seattle Storm", "Washington Mystics",
        ]
        records = [
            {
                "team_id": t.lower().replace(" ", "-"),
                "team_name": t,
                "team_short": self._abbrev(t),
                "country": "USA",
                "league": "wnba",
            }
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
    # Scraping
    # ------------------------------------------------------------------

    def _scrape_bbref_season(self, year: int) -> list[dict]:
        """Scrape a single WNBA season from Basketball-Reference."""
        url = f"{BBREF_WNBA_BASE}/schedule_{year}.html"
        self._rate_limit()
        logger.info(f"Scraping WNBA {year} from {url}")

        try:
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"BBRef WNBA error for {url}: {exc}")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        table = soup.select_one("table#schedule")
        if not table:
            table = soup.select_one("table.stats_table")
        if not table:
            logger.warning(f"No schedule table found for WNBA {year}")
            return []

        return self._parse_bbref_table(table, year)

    def _parse_bbref_table(self, table, year: int) -> list[dict]:
        """Parse Basketball-Reference schedule table."""
        records: list[dict] = []
        rows = table.select("tr")

        for row in rows:
            if "thead" in (row.get("class") or []):
                continue

            cells = row.find_all("td")
            if len(cells) < 6:
                continue

            try:
                # BBRef columns: Date, Visitor/Neutral, Home/Neutral, Visitor Pts, Home Pts, OT, Att.
                date_cell = cells[0] if len(cells) > 0 else None
                away_cell = cells[1] if len(cells) > 1 else None
                home_cell = cells[3] if len(cells) > 3 else None
                away_pts_cell = cells[2] if len(cells) > 2 else None
                home_pts_cell = cells[4] if len(cells) > 4 else None

                if not all([away_cell, home_cell]):
                    continue

                away_team = away_cell.get_text(strip=True)
                home_team = home_cell.get_text(strip=True)
                away_score = self._extract_score(away_pts_cell)
                home_score = self._extract_score(home_pts_cell)
                date_str = self._extract_date_bbref(date_cell, year)

                if not away_team or not home_team:
                    continue

                game_id = f"WNBA_{away_team}-{home_team}-{date_str or year}"
                game_id = re.sub(r"[^a-zA-Z0-9_-]", "_", game_id)

                if home_score is not None and away_score is not None:
                    # Home team row
                    records.append({
                        "game_id": game_id,
                        "league": "wnba",
                        "season": str(year),
                        "date": date_str or "",
                        "team_id": home_team.lower().replace(" ", "-"),
                        "team_name": home_team,
                        "opponent_id": away_team.lower().replace(" ", "-"),
                        "opponent_name": away_team,
                        "is_home": 1,
                        "team_score": home_score,
                        "opponent_score": away_score,
                        "total_points": home_score + away_score,
                        "result": 1 if home_score > away_score else 0,
                        "venue": "",
                        "status": "completed",
                    })
                    # Away team row
                    records.append({
                        "game_id": game_id,
                        "league": "wnba",
                        "season": str(year),
                        "date": date_str or "",
                        "team_id": away_team.lower().replace(" ", "-"),
                        "team_name": away_team,
                        "opponent_id": home_team.lower().replace(" ", "-"),
                        "opponent_name": home_team,
                        "is_home": 0,
                        "team_score": away_score,
                        "opponent_score": home_score,
                        "total_points": home_score + away_score,
                        "result": 1 if away_score > home_score else 0,
                        "venue": "",
                        "status": "completed",
                    })
            except (ValueError, AttributeError, TypeError) as exc:
                logger.debug(f"Skipping unparseable row: {exc}")
                continue

        return records

    def _parse_espn_scoreboard(self, soup, upcoming_only: bool = False) -> list[dict]:
        """Parse ESPN WNBA scoreboard page."""
        records: list[dict] = []
        games = soup.select("section.game-card") or soup.select("div.game-module")

        for game in games:
            text = game.get_text(" ", strip=True)
            match = re.search(
                r"([A-Za-z\s.]+?)\s+(\d+)?\s+([A-Za-z\s.]+?)\s+(\d+)?", text
            )
            if match:
                away = match.group(1).strip()
                home = match.group(3).strip()
                away_pts = int(match.group(2)) if match.group(2) else None
                home_pts = int(match.group(4)) if match.group(4) else None

                if upcoming_only and away_pts is not None:
                    continue

                if away_pts is not None and home_pts is not None:
                    records.append({
                        "game_id": f"WNBA_espn_{away}-{home}",
                        "league": "wnba",
                        "season": "2025",
                        "date": "",
                        "team_id": home.lower().replace(" ", "-"),
                        "team_name": home,
                        "opponent_id": away.lower().replace(" ", "-"),
                        "opponent_name": away,
                        "is_home": 1,
                        "team_score": home_pts,
                        "opponent_score": away_pts,
                        "total_points": home_pts + away_pts,
                        "result": 1 if home_pts > away_pts else 0,
                        "venue": "",
                        "status": "completed",
                    })
                    records.append({
                        "game_id": f"WNBA_espn_{away}-{home}",
                        "league": "wnba",
                        "season": "2025",
                        "date": "",
                        "team_id": away.lower().replace(" ", "-"),
                        "team_name": away,
                        "opponent_id": home.lower().replace(" ", "-"),
                        "opponent_name": home,
                        "is_home": 0,
                        "team_score": away_pts,
                        "opponent_score": home_pts,
                        "total_points": home_pts + away_pts,
                        "result": 1 if away_pts > home_pts else 0,
                        "venue": "",
                        "status": "completed",
                    })
        return records

    @staticmethod
    def _extract_score(cell) -> Optional[int]:
        """Extract score from a table cell."""
        if cell is None:
            return None
        text = cell.get_text(strip=True)
        if not text or text == "":
            return None
        try:
            return int(text)
        except ValueError:
            return None

    @staticmethod
    def _extract_date_bbref(cell, fallback_year: int) -> Optional[str]:
        """Extract date from a BBRef table cell."""
        if cell is None:
            return None
        text = cell.get_text(strip=True)
        if not text:
            return None

        # BBRef format: "Tue, May 15 2024" or "Tue, May 15"
        try:
            from datetime import datetime
            for fmt in ["%a, %b %d %Y", "%a, %b %d", "%b %d %Y", "%b %d"]:
                try:
                    dt = datetime.strptime(text, fmt)
                    if dt.year == 1900:
                        dt = dt.replace(year=fallback_year)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        except Exception:
            pass

        # Direct YYYY-MM-DD match
        m = re.match(r"(\d{4}-\d{2}-\d{2})", text)
        if m:
            return m.group(1)

        return None


# Register with league registry
if REGISTER_LEAGUE:
    _key, _cls_placeholder, _meta = REGISTER_LEAGUE
    league_registry.register("wnba", WNBASource, _meta)
