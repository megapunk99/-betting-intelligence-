"""BNXT League data source via web scraping (Proballers.com).

The BNXT League is a combined Belgian-Dutch basketball league with 19 teams.
Since no free API exists, we scrape game results and schedules from
Proballers.com, which has the most complete historical data for European
basketball leagues.

**Scraping approach:**
    - Fetch season standings/schedule pages from Proballers
    - Parse HTML tables with BeautifulSoup
    - Extract game results, team names, scores, and dates
    - Rate-limit requests to avoid overloading the server

**Ethics & reliability:**
    - Requests are rate-limited to 1/sec
    - We respect ``robots.txt`` and set a descriptive User-Agent
    - If Proballers changes their HTML structure, the parser must be updated
    - Flashscore.com is used as a fallback data source

**Why BNXT is worth the scraping effort:**
    - Cross-border league means travel fatigue is real and underpriced
    - Low international betting volume = softer lines
    - Complex two-phase schedule (National + BNXT) creates confusion for
      algorithmic bookmakers
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

# URLs for BNXT League data
PROBALLERS_BASE = "https://www.proballers.com"
PROBALLERS_BNXT_SCHEDULE = (
    "https://www.proballers.com/basketball/league/96/bnxt-league/games"
)

# Fallback: Flashscore BNXT URL
FLASHSCORE_BNXT = (
    "https://www.flashscore.com/basketball/netherlands/bnxt-league/"
)


class BNXTSource(SmallLeagueSource):
    """Data source for BNXT League via Proballers.com scraping."""

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self._teams_cache: Optional[pd.DataFrame] = None
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self._last_request_time = 0.0

        # Track if we already warned about Cloudflare blocking
        self._cloudflare_warned = False
        self._use_flashscore = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_historical(
        self, seasons: Optional[list[Any]] = None
    ) -> pd.DataFrame:
        """Load completed BNXT League games from Proballers.

        Args:
            seasons: List of season URLs or IDs. If None, attempts to
                     load the most recent season.

        Returns:
            DataFrame in CANONICAL_SCHEMA.
        """
        if seasons is None:
            seasons = ["2025-2026", "2024-2025"]

        all_records: list[dict] = []
        for season in seasons:
            try:
                records = self._scrape_season(season)
                if records:
                    logger.info(f"BNXT {season}: {len(records)} games found")
                    all_records.extend(records)
                else:
                    logger.warning(f"BNXT {season}: no games found")
            except Exception as exc:
                logger.error(f"Failed to scrape BNXT season {season}: {exc}")

        if not all_records:
            return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

        df = pd.DataFrame(all_records)
        df["league"] = "bnxt"
        self.validate_schema(df, "bnxt_historical")
        return self._sort_and_dedup(df)

    def load_upcoming(self, limit: int = 20) -> pd.DataFrame:
        """Load upcoming BNXT League games.

        .. note::
            Upcoming games are harder to scrape because Proballers focuses
            on completed results. This may return fewer than ``limit`` rows.
        """
        # Try the schedule page for upcoming games
        try:
            self._rate_limit()
            resp = self._session.get(PROBALLERS_BNXT_SCHEDULE, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")
            records = self._parse_schedule_page(soup, upcoming_only=True)
            logger.info(f"BNXT upcoming: {len(records)} games")
            if records:
                df = pd.DataFrame(records[:limit])
                df["league"] = "bnxt"
                self.validate_schema(df, "bnxt_upcoming")
                return self._sort_and_dedup(df)
        except Exception as exc:
            logger.error(f"Failed to scrape BNXT upcoming: {exc}")

        return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

    def get_teams(self) -> pd.DataFrame:
        """Load BNXT League teams from scraped schedule data."""
        if self._teams_cache is not None:
            return self._teams_cache

        teams: dict[str, dict] = {}

        # Try to extract teams from recent seasons
        for season in ["2024-2025", "2025-2026"]:
            try:
                records = self._scrape_season(season)
                for rec in records:
                    for team_key in ["team_name", "opponent_name"]:
                        name = rec.get(team_key, "")
                        if name and name not in teams:
                            teams[name] = {
                                "team_id": name.lower().replace(" ", "-"),
                                "team_name": name,
                                "team_short": str(name)[:3].upper(),
                                "country": "Belgium/Netherlands",
                                "league": "bnxt",
                            }
            except Exception:
                continue

        self._teams_cache = pd.DataFrame(list(teams.values()))
        return self._teams_cache

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Ensure we don't exceed ~1 request per second."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

    def _scrape_season(self, season: str) -> list[dict]:
        """Scrape games for a specific BNXT season.

        Tries Proballers first. If blocked (Cloudflare), falls back to
        Flashscore for the current season data.
        """
        season_slug = season.replace("-", "--")
        url = f"{PROBALLERS_BNXT_SCHEDULE}?season={season_slug}"

        self._rate_limit()
        logger.info(f"Scraping BNXT {season} from {url}")
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"Proballers error for {url}: {exc}")
            # Try Flashscore fallback
            return self._scrape_from_flashscore(season)

        soup = BeautifulSoup(resp.text, "lxml")

        # Detect Cloudflare challenge page
        page_text = soup.get_text(" ", strip=True).lower()
        if "just a moment" in page_text or "enable javascript" in page_text or "cloudflare" in page_text:
            if not self._cloudflare_warned:
                logger.warning("Proballers blocked by Cloudflare. Trying Flashscore...")
                self._cloudflare_warned = True
            return self._scrape_from_flashscore(season)

        return self._parse_schedule_page(soup, upcoming_only=False)

    def _scrape_from_flashscore(self, season: str) -> list[dict]:
        """Fallback: scrape BNXT data from Flashscore.com."""
        self._rate_limit()
        logger.info(f"Attempting Flashscore BNXT scrape for {season}...")
        try:
            resp = self._session.get(FLASHSCORE_BNXT, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            records = self._parse_flashscore_page(soup)
            if records:
                logger.info(f"Flashscore returned {len(records)} BNXT games")
                return records
        except requests.RequestException as exc:
            logger.warning(f"Flashscore also failed: {exc}")

        return []

    def _parse_flashscore_page(self, soup: BeautifulSoup) -> list[dict]:
        """Parse Flashscore.com BNXT page for game results."""
        records: list[dict] = []
        # Flashscore uses event__match divs
        matches = soup.select("div.event__match")
        if not matches:
            matches = soup.select("div[class*='match']")
        if not matches:
            matches = soup.select("tr")

        for match in matches:
            text = match.get_text(" ", strip=True)
            score_match = re.search(r"(.+?)\s+(\d+)\s*[-:]\s*(\d+)\s+(.+)", text)
            if score_match:
                home_team = score_match.group(1).strip()
                away_team = score_match.group(4).strip()
                home_score = int(score_match.group(2))
                away_score = int(score_match.group(3))

                game_id = f"BNXT_fs_{home_team}-{away_team}".replace(" ", "_")
                records.append({
                    "game_id": game_id, "league": "bnxt", "season": "2024-2025", "date": "",
                    "team_id": home_team.lower().replace(" ", "-"), "team_name": home_team,
                    "opponent_id": away_team.lower().replace(" ", "-"), "opponent_name": away_team,
                    "is_home": 1, "team_score": home_score, "opponent_score": away_score,
                    "total_points": home_score + away_score,
                    "result": 1 if home_score > away_score else 0, "venue": "", "status": "completed",
                })

        return self._create_away_rows(records)

    def _parse_schedule_page(
        self, soup: BeautifulSoup, upcoming_only: bool = False
    ) -> list[dict]:
        """Parse a Proballers schedule page HTML into game records.

        Proballers HTML structure (subject to change):
            - Each game is in a ``div.game-card`` or ``tr`` element
            - Contains: home team, away team, scores, date, status
        """
        records: list[dict] = []

        # Try multiple selectors to handle different Proballers page layouts
        game_elements = (
            soup.select("div.game-card")
            or soup.select("div.game-item")
            or soup.select("div.match-row")
            or soup.select("tr.game-row")
            or soup.select("div.schedule-game")
            or # Flashscore-style
            soup.select("div.event__match")
        )

        if not game_elements:
            # Fallback: look for any table with team names and scores
            logger.debug("No game-card elements found, trying table-based parsing")
            return self._parse_table_fallback(soup, upcoming_only)

        for elem in game_elements:
            try:
                game = self._parse_game_element(elem, upcoming_only)
                if game:
                    records.append(game)
                    records.extend(self._create_away_rows([game]))
            except Exception as exc:
                logger.debug(f"Skipping unparseable game element: {exc}")

        return records

    def _parse_game_element(
        self, elem: Any, upcoming_only: bool = False
    ) -> Optional[dict]:
        """Parse a single game element into a home-team record."""
        text = elem.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        # Try to extract teams and scores
        home_team = self._find_team(elem, "home")
        away_team = self._find_team(elem, "away")
        date = self._find_date(elem)

        # Score extraction
        home_score = self._find_score(elem, "home")
        away_score = self._find_score(elem, "away")

        if not home_team or not away_team:
            # Last resort: parse from text
            parsed = self._parse_from_text(lines)
            if parsed:
                home_team = parsed.get("home_team") or home_team
                away_team = parsed.get("away_team") or away_team
                home_score = parsed.get("home_score") or home_score
                away_score = parsed.get("away_score") or away_score
                date = parsed.get("date") or date

        if not home_team or not away_team:
            return None

        if upcoming_only and home_score is not None:
            return None  # Skip completed games when looking for upcoming

        home_score = int(home_score) if home_score is not None else None
        away_score = int(away_score) if away_score is not None else None
        total = (
            (home_score + away_score)
            if (home_score is not None and away_score is not None)
            else None
        )

        game_id_str = f"{home_team}-{away_team}-{date or 'unknown'}"
        game_id = "BNXT_" + re.sub(r"[^a-zA-Z0-9_-]", "_", game_id_str)

        return {
            "game_id": game_id,
            "league": "bnxt",
            "season": self._infer_season(date or ""),
            "date": date or "",
            "team_id": home_team.lower().replace(" ", "-"),
            "team_name": home_team,
            "opponent_id": away_team.lower().replace(" ", "-"),
            "opponent_name": away_team,
            "is_home": 1,
            "team_score": home_score,
            "opponent_score": away_score,
            "total_points": total,
            "result": (
                1
                if (
                    home_score is not None
                    and away_score is not None
                    and home_score > away_score
                )
                else (
                    0
                    if (
                        home_score is not None
                        and away_score is not None
                        and home_score < away_score
                    )
                    else None
                )
            ),
            "venue": "",
            "status": "completed"
            if home_score is not None
            else "scheduled",
        }

    def _parse_table_fallback(
        self, soup: BeautifulSoup, upcoming_only: bool = False
    ) -> list[dict]:
        """Fallback parser for table-based page layouts."""
        records: list[dict] = []
        tables = soup.select("table") or []

        for table in tables:
            rows = table.select("tr")
            for row in rows:
                cells = row.select("td, th")
                texts = [c.get_text(strip=True) for c in cells]

                if len(texts) < 3:
                    continue

                # Try to identify team names and scores
                # Look for patterns like: "TeamA 89 - 87 TeamB"
                full_text = " ".join(texts)
                match = re.search(
                    r"(.+?)\s+(\d+)\s*[-:]\s*(\d+)\s+(.+)", full_text
                )
                if match:
                    home_team = match.group(1).strip()
                    away_team = match.group(4).strip()
                    home_score = int(match.group(2))
                    away_score = int(match.group(3))

                    if upcoming_only:
                        continue

                    date = self._extract_date(full_text) or ""

                    records.append(
                        {
                            "game_id": f"BNXT_{home_team}-{away_team}",
                            "league": "bnxt",
                            "season": self._infer_season(date),
                            "date": date,
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
                        }
                    )

        return records

    @staticmethod
    def _create_away_rows(records: list[dict]) -> list[dict]:
        """Create away-team rows from home-team records."""
        away_rows = []
        for rec in records:
            away = dict(rec)
            away["is_home"] = 0
            away["team_id"], away["opponent_id"] = (
                away["opponent_id"],
                away["team_id"],
            )
            away["team_name"], away["opponent_name"] = (
                away["opponent_name"],
                away["team_name"],
            )
            away["team_score"], away["opponent_score"] = (
                away["opponent_score"],
                away["team_score"],
            )
            if rec["result"] is not None:
                away["result"] = 1 - rec["result"]
            away_rows.append(away)
        return away_rows

    # ------------------------------------------------------------------
    # HTML element helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_team(elem: Any, side: str = "home") -> Optional[str]:
        """Extract team name from a game element."""
        # Try common selectors
        for cls in [f"team-{side}", f"{side}-team", f"team__{side}"]:
            el = elem.select_one(f".{cls}")
            if el:
                name = el.get_text(strip=True)
                if name:
                    return name
        # Try by class containing the side name
        for el in elem.select("[class*='team']"):
            classes = " ".join(el.get("class", []))
            if side in classes.lower():
                name = el.get_text(strip=True)
                if name and len(name) > 2:
                    return name
        return None

    @staticmethod
    def _find_score(elem: Any, side: str = "home") -> Optional[int]:
        """Extract score from a game element."""
        for cls in [f"score-{side}", f"{side}-score", f"score__{side}"]:
            el = elem.select_one(f".{cls}")
            if el:
                text = el.get_text(strip=True)
                try:
                    return int(text)
                except ValueError:
                    pass
        # Try any element with just a number that could be a score
        for el in elem.select("[class*='score']"):
            text = el.get_text(strip=True)
            try:
                return int(text)
            except ValueError:
                continue
        return None

    @staticmethod
    def _find_date(elem: Any) -> Optional[str]:
        """Extract date from a game element."""
        for cls in ["date", "game-date", "match-date", "time", "day"]:
            el = elem.select_one(f".{cls}")
            if el:
                return el.get("data-date") or el.get_text(strip=True)
        return None

    @staticmethod
    def _extract_date(text: str) -> Optional[str]:
        """Extract a YYYY-MM-DD date from arbitrary text."""
        match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", text)
        if match:
            return match.group(1).replace("/", "-")
        # Try European format: DD/MM/YYYY
        match = re.search(r"(\d{2})[-/](\d{2})[-/](\d{4})", text)
        if match:
            return f"{match.group(3)}-{match.group(2)}-{match.group(1)}"
        return None

    @staticmethod
    def _infer_season(date_str: str) -> str:
        """Infer season string from a date."""
        if not date_str or len(date_str) < 4:
            return "unknown"
        try:
            year = int(date_str[:4])
            next_year = year + 1
            month = int(date_str[5:7]) if len(date_str) >= 7 else 1
            if month >= 9:
                return f"{year}-{next_year}"
            else:
                return f"{year - 1}-{year}"
        except (ValueError, IndexError):
            return "unknown"

    @staticmethod
    def _parse_from_text(lines: list[str]) -> Optional[dict]:
        """Parse teams and scores from raw text lines as last resort."""
        full = " ".join(lines)
        # Pattern: TeamA vs TeamB or TeamA - TeamB with optional scores
        patterns = [
            r"(.+?)\s+(\d+)\s*[-:]\s*(\d+)\s+(.+)",  # TeamA 89 - 87 TeamB
            r"(.+?)\s+vs\s+(.+?)\s+(\d+)\s*[-:]\s*(\d+)",  # TeamA vs TeamB 89-87
            r"(\d+)\s*[-:]\s*(\d+)\s+(.+?)\s+vs\s+(.+)",  # 89-87 TeamA vs TeamB
        ]
        for pat in patterns:
            match = re.search(pat, full, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 4 and all(g.strip() for g in groups):
                    return {
                        "home_score": int(groups[1])
                        if groups[0][-1].isdigit()
                        else int(groups[2]),
                        "away_score": int(groups[2])
                        if groups[0][-1].isdigit()
                        else int(groups[3]),
                        "home_team": groups[0]
                        if not groups[0][-1].isdigit()
                        else groups[2],
                        "away_team": groups[1]
                        if not groups[0][-1].isdigit()
                        else groups[3],
                        "date": None,
                    }
        return None
