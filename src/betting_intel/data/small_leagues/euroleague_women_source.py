"""
Women's EuroLeague basketball data source.

Why Women's EuroLeague:
- Niche women's basketball with very low betting volume
- Multi-country league (teams from Spain, France, Russia, Turkey, etc.)
- Complex travel schedules create fatigue edges that are underpriced
- Limited historical data means bookmaker models are less accurate
- EuroLeague Women has growing viewership but still inefficient markets

Data sources:
1. FIBA website (euroleaguewomen.basketball) — official schedule/results
2. Flashscore.com — fallback for game results
3. Proballers.com — European basketball stats
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

# EuroLeague Women data URLs
EUROLEAGUE_WOMEN_BASE = "https://www.euroleaguebasketball.net/en/wrl/"
EUROLEAGUE_WOMEN_RESULTS = "https://www.euroleaguebasketball.net/en/wrl/game-center/"
FLASHSCORE_WOMEN = "https://www.flashscore.com/basketball/europe/euroleague-women/"

# Register this source
try:
    from betting_intel.data.small_leagues.league_registry import league_registry

    REGISTER_LEAGUE = (
        "euroleague_women",
        None,
        {
            "name": "EuroLeague Women",
            "country": "Europe (multi-country)",
            "tier": "First division (continental)",
            "num_teams": 16,
            "season_format": "Regular season (14 games) + playoffs + Final Four",
            "typical_season_months": "October -> April",
            "data_source": "EuroLeague website (web scraping)",
            "market_notes": (
                "Multi-country women's basketball. Very low betting volume = "
                "extremely soft lines. Complex travel across Europe creates fatigue "
                "edges. Cross-league play (domestic + European) means load management "
                "is frequently mispriced."
            ),
        },
    )
except ImportError:
    REGISTER_LEAGUE = None


class EuroLeagueWomenSource(SmallLeagueSource):
    """Data source for Women's EuroLeague via web scraping."""

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
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.5:
            time.sleep(1.5 - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_historical(self, seasons: Optional[list[Any]] = None) -> pd.DataFrame:
        """Load completed Women's EuroLeague games."""
        if seasons is None:
            seasons = ["2024-2025", "2023-2024"]

        all_records: list[dict] = []
        for season in seasons:
            try:
                records = self._scrape_season(season)
                if records:
                    logger.info(f"EuroLeague Women {season}: {len(records)} games")
                    all_records.extend(records)
            except Exception as exc:
                logger.error(f"Failed to scrape EuroLeague Women {season}: {exc}")

        if not all_records:
            return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

        df = pd.DataFrame(all_records)
        df["league"] = "euroleague_women"
        self.validate_schema(df, "euroleague_women_historical")
        return self._sort_and_dedup(df)

    def load_upcoming(self, limit: int = 20) -> pd.DataFrame:
        """Load upcoming Women's EuroLeague games."""
        self._rate_limit()
        try:
            resp = self._session.get(EUROLEAGUE_WOMEN_RESULTS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            records = self._parse_game_center(soup, upcoming_only=True)
            if records:
                df = pd.DataFrame(records[:limit])
                df["league"] = "euroleague_women"
                self.validate_schema(df, "euroleague_women_upcoming")
                return self._sort_and_dedup(df)
        except Exception as exc:
            logger.error(f"Failed to fetch EuroLeague Women upcoming: {exc}")

        # Fallback: try Flashscore
        try:
            self._rate_limit()
            resp = self._session.get(FLASHSCORE_WOMEN, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            records = self._parse_flashscore(soup)
            if records:
                df = pd.DataFrame(records[:limit])
                df["league"] = "euroleague_women"
                self.validate_schema(df, "euroleague_women_upcoming_fs")
                return self._sort_and_dedup(df)
        except Exception as exc:
            logger.error(f"Flashscore fallback also failed: {exc}")

        return pd.DataFrame(columns=list(CANONICAL_SCHEMA.keys()))

    def get_teams(self) -> pd.DataFrame:
        """Load Women's EuroLeague team metadata."""
        if self._teams_cache is not None:
            return self._teams_cache

        # Try to extract from historical data
        teams: dict[str, dict] = {}
        for season in ["2024-2025", "2023-2024"]:
            try:
                records = self._scrape_season(season)
                for rec in records:
                    for team_key in ["team_name", "opponent_name"]:
                        name = rec.get(team_key, "")
                        if name and name not in teams:
                            teams[name] = {
                                "team_id": name.lower().replace(" ", "-"),
                                "team_name": name,
                                "team_short": name[:3].upper(),
                                "country": "Europe",
                                "league": "euroleague_women",
                            }
            except Exception:
                continue

        self._teams_cache = pd.DataFrame(list(teams.values()))
        return self._teams_cache

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def _scrape_season(self, season: str) -> list[dict]:
        """Scrape a single season."""
        season_slug = season.replace("-", "")
        url = f"{EUROLEAGUE_WOMEN_RESULTS}?season={season_slug}"

        self._rate_limit()
        logger.info(f"Scraping EuroLeague Women {season} from {url}")

        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            return self._parse_game_center(soup, upcoming_only=False)
        except requests.RequestException as exc:
            logger.error(f"EuroLeague website error: {exc}")
            return self._scrape_flashscore_fallback(season)

    def _parse_game_center(self, soup: BeautifulSoup, upcoming_only: bool = False) -> list[dict]:
        """Parse the EuroLeague game center page."""
        records: list[dict] = []
        game_cards = (
            soup.select("div.game-card")
            or soup.select("article.match")
            or soup.select("div.event-item")
            or soup.select("div.match-row")
            or soup.select("div[class*='game']")
        )

        for card in game_cards:
            try:
                game = self._parse_game_card(card, upcoming_only)
                if game:
                    records.append(game)
            except Exception as exc:
                logger.debug(f"Skipping unparseable card: {exc}")

        if not records:
            records = self._parse_table_fallback(soup, upcoming_only)

        return records

    def _parse_game_card(self, card, upcoming_only: bool = False) -> Optional[dict]:
        """Parse a single game card element."""
        text = card.get_text(separator=" ", strip=True)

        # Extract teams and scores
        # Try multiple patterns
        patterns = [
            r"([A-Za-z\s.]+?)\s+(\d+)\s*[-:]\s*(\d+)\s+([A-Za-z\s.]+)",  # TeamA 89-87 TeamB
            r"([A-Za-z\s.]+?)\s+vs\s+([A-Za-z\s.]+?)\s+(\d+)\s*[-:]\s*(\d+)",  # TeamA vs TeamB 89-87
        ]

        home_team = None
        away_team = None
        home_score = None
        away_score = None
        date_str = None

        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 4:
                    # Determine which is home/away based on pattern
                    if "vs" in pat:
                        away_team = groups[0].strip()
                        home_team = groups[1].strip()
                        away_score = int(groups[2])
                        home_score = int(groups[3])
                    else:
                        home_team = groups[0].strip()
                        away_team = groups[3].strip()
                        home_score = int(groups[1])
                        away_score = int(groups[2])
                    break

        if not home_team or not away_team:
            return None

        if upcoming_only and home_score is not None:
            return None  # Skip completed games

        # Try to extract date
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if date_match:
            date_str = date_match.group(1)
        else:
            date_text = card.select_one("time, span.date, div.date")
            if date_text:
                date_str = date_text.get("datetime") or date_text.get_text(strip=True)

        game_id = f"ELW_{home_team}-{away_team}-{date_str or 'unknown'}"
        game_id = re.sub(r"[^a-zA-Z0-9_-]", "_", game_id)
        season = self._infer_season(date_str or "")

        if home_score is not None and away_score is not None:
            return {
                "game_id": game_id,
                "league": "euroleague_women",
                "season": season,
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
            }
        return None

    def _parse_table_fallback(self, soup: BeautifulSoup, upcoming_only: bool = False) -> list[dict]:
        """Fallback parser for table-based layouts."""
        records: list[dict] = []
        for table in soup.select("table"):
            rows = table.select("tr")
            for row in rows[1:]:
                cells = row.select("td")
                if len(cells) < 4:
                    continue

                texts = [c.get_text(strip=True) for c in cells]
                full = " ".join(texts)
                match = re.search(
                    r"([A-Za-z\s.]+?)\s+(\d+)\s*[-:]\s*(\d+)\s+([A-Za-z\s.]+)", full
                )
                if match:
                    home = match.group(1).strip()
                    away = match.group(4).strip()
                    hs, aws = int(match.group(2)), int(match.group(3))
                    date_str = self._extract_date(full) or ""

                    if upcoming_only:
                        continue

                    records.append({
                        "game_id": f"ELW_tab_{home}-{away}",
                        "league": "euroleague_women",
                        "season": self._infer_season(date_str),
                        "date": date_str,
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

    def _scrape_flashscore_fallback(self, season: str) -> list[dict]:
        """Fallback: scrape from Flashscore."""
        self._rate_limit()
        logger.info(f"Attempting Flashscore for EuroLeague Women {season}")
        try:
            resp = self._session.get(FLASHSCORE_WOMEN, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            return self._parse_flashscore(soup)
        except Exception as exc:
            logger.warning(f"Flashscore failed: {exc}")
            return []

    def _parse_flashscore(self, soup: BeautifulSoup) -> list[dict]:
        """Parse Flashscore page."""
        records: list[dict] = []
        matches = (
            soup.select("div.event__match")
            or soup.select("div[class*='match']")
            or soup.select("tr")
        )

        for match in matches:
            text = match.get_text(" ", strip=True)
            m = re.search(r"(.+?)\s+(\d+)\s*[-:]\s*(\d+)\s+(.+)", text)
            if m:
                away = m.group(1).strip()
                home = m.group(4).strip()
                aws, hs = int(m.group(2)), int(m.group(3))

                records.append({
                    "game_id": f"ELW_fs_{home}-{away}",
                    "league": "euroleague_women",
                    "season": "2024-2025",
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

    @staticmethod
    def _extract_date(text: str) -> Optional[str]:
        m = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", text)
        if m:
            return m.group(1).replace("/", "-")
        m = re.search(r"(\d{2})[-/](\d{2})[-/](\d{4})", text)
        if m:
            return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        return None

    @staticmethod
    def _infer_season(date_str: str) -> str:
        if not date_str or len(date_str) < 4:
            return "unknown"
        try:
            year = int(date_str[:4])
            month = int(date_str[5:7]) if len(date_str) >= 7 else 1
            if month >= 9:
                return f"{year}-{year + 1}"
            return f"{year - 1}-{year}"
        except (ValueError, IndexError):
            return "unknown"


# Register with league registry
if REGISTER_LEAGUE:
    _key, _cls_placeholder, _meta = REGISTER_LEAGUE
    league_registry.register("euroleague_women", EuroLeagueWomenSource, _meta)
