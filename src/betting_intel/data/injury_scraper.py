"""
ESPN NBA injury scraper — fetches real-time injury reports.

Scrapes ESPN's NBA injuries page (https://www.espn.com/nba/injuries)
to get structured injury data for all 30 teams. This data powers:

1. **Injury-adjusted predictions** — downgrade teams missing key players
2. **Minutes projections** — estimate playing time for replacement players
3. **Prop model inputs** — injury status for player prop probability
4. **Alert generation** — flag when key players return or get injured

ESPN's injuries page is updated regularly by ESPN's injury analysts.
It's a free, public page (no auth required), updated daily during the
NBA season.

Data source: https://www.espn.com/nba/injuries
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

ESPN_INJURIES_URL = "https://www.espn.com/nba/injuries"

# Abbreviation → full team name mapping
TEAM_ABBREVIATIONS: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

# Injury status severity mapping — used for minutes projection and model weighting
INJURY_SEVERITY: dict[str, str] = {
    "out": "critical",
    "out for season": "critical",
    "doubtful": "high",
    "questionable": "moderate",
    "probable": "low",
    "game time decision": "moderate",
    "day-to-day": "low",
}

# ── Data Structures ───────────────────────────────────────────────────────


@dataclass
class InjuryRecord:
    """A single player injury report from ESPN."""

    player_name: str
    team: str
    team_abbr: str
    position: str
    injury_status: str  # e.g., "OUT", "Questionable", "Probable"
    injury_description: str  # e.g., "Right Ankle Sprain"
    date_updated: Optional[str] = None
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_active(self) -> bool:
        """Whether this player is likely to miss the next game."""
        return self.injury_status.lower() in ("out", "doubtful", "out for season")

    @property
    def severity(self) -> str:
        """Severity classification for model weighting."""
        status_lower = self.injury_status.lower().strip()
        return INJURY_SEVERITY.get(status_lower, "unknown")

    @property
    def estimated_minutes_penalty(self) -> float:
        """
        Estimated playing time penalty as a fraction of normal minutes.

        Returns a multiplier to apply to a player's projected minutes:
        - OUT / Out for Season → 0.0 (will not play)
        - Doubtful → 0.2 (unlikely to play)
        - Questionable → 0.5 (coin flip)
        - Probable → 0.85 (likely to play reduced minutes or start)
        - Day-to-day → 0.9 (minor, likely plays)
        - Game Time Decision → 0.6 (uncertain)
        """
        status_lower = self.injury_status.lower().strip()
        penalties = {
            "out": 0.0,
            "out for season": 0.0,
            "doubtful": 0.2,
            "questionable": 0.5,
            "game time decision": 0.6,
            "probable": 0.85,
            "day-to-day": 0.9,
        }
        return penalties.get(status_lower, 0.5)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "player_name": self.player_name,
            "team": self.team,
            "team_abbr": self.team_abbr,
            "position": self.position,
            "injury_status": self.injury_status,
            "injury_description": self.injury_description,
            "date_updated": self.date_updated,
            "severity": self.severity,
            "minutes_penalty": self.estimated_minutes_penalty,
            "is_active": self.is_active,
            "scraped_at": self.scraped_at,
        }


# ── Scraper ────────────────────────────────────────────────────────────────


class ESPNInjuryScraper:
    """
    Scrapes ESPN's NBA injuries page for structured injury data.

    Usage::

        scraper = ESPNInjuryScraper()
        all_injuries = scraper.fetch_all()
        lakers_injuries = scraper.fetch_by_team("LAL")
        summary = scraper.get_league_summary()

    The scraper parses ESPN's HTML table layout:
    - Each team has a section with a roster table
    - Each row = player name, position, injury status, return date
    """

    def __init__(self, user_agent: Optional[str] = None, timeout: int = 15):
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                user_agent
                or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        self._last_fetch: Optional[list[InjuryRecord]] = None
        self._last_fetch_time: Optional[datetime] = None

    def fetch_all(self, force_refresh: bool = False) -> list[InjuryRecord]:
        """
        Fetch all NBA injuries from ESPN.

        Args:
            force_refresh: If False, may return cached results from the
                           last fetch within the same session.

        Returns:
            List of InjuryRecord for all injured players across all teams.
        """
        if not force_refresh and self._last_fetch is not None:
            # Cache is valid for 5 minutes within a session
            if self._last_fetch_time and (datetime.now() - self._last_fetch_time).seconds < 300:
                return self._last_fetch

        logger.info(f"Fetching NBA injuries from {ESPN_INJURIES_URL}...")
        try:
            resp = self._session.get(ESPN_INJURIES_URL, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch ESPN injuries page: {e}")
            return self._last_fetch or []

        records = self._parse_injuries_page(resp.text)
        logger.info(f"Parsed {len(records)} injury records from ESPN")

        self._last_fetch = records
        self._last_fetch_time = datetime.now()
        return records

    def fetch_by_team(self, team_abbr: str) -> list[InjuryRecord]:
        """
        Fetch injuries for a specific team.

        Args:
            team_abbr: Team abbreviation (e.g., "LAL", "BOS", "GSW")

        Returns:
            List of InjuryRecord for the specified team.
        """
        team_abbr = team_abbr.upper()
        all_injuries = self.fetch_all()
        return [r for r in all_injuries if r.team_abbr == team_abbr]

    def get_active_out_count(self, team_abbr: Optional[str] = None) -> int:
        """
        Count players who are actively OUT (will miss next game).

        Args:
            team_abbr: Optional team filter. If None, counts league-wide.

        Returns:
            Number of active outs.
        """
        if team_abbr:
            records = self.fetch_by_team(team_abbr)
        else:
            records = self.fetch_all()
        return sum(1 for r in records if r.is_active)

    def get_league_summary(self) -> dict:
        """
        Get a per-team summary of injury impact.

        Returns:
            Dict with per-team injury counts and league-wide totals.
        """
        all_injuries = self.fetch_all()
        if not all_injuries:
            return {"teams": {}, "total_injured": 0, "total_active_outs": 0}

        by_team: dict[str, dict] = {}
        for record in all_injuries:
            if record.team_abbr not in by_team:
                by_team[record.team_abbr] = {
                    "team": record.team,
                    "total_injured": 0,
                    "active_outs": 0,
                    "questionable": 0,
                    "players": [],
                }
            by_team[record.team_abbr]["total_injured"] += 1
            by_team[record.team_abbr]["players"].append(record.player_name)
            if record.is_active:
                by_team[record.team_abbr]["active_outs"] += 1
            if record.severity == "moderate":
                by_team[record.team_abbr]["questionable"] += 1

        return {
            "teams": by_team,
            "total_injured": len(all_injuries),
            "total_active_outs": sum(1 for r in all_injuries if r.is_active),
            "scraped_at": datetime.now().isoformat(),
        }

    def to_dataframe(self, records: Optional[list[InjuryRecord]] = None) -> pd.DataFrame:
        """
        Convert injury records to a pandas DataFrame.

        Args:
            records: List of InjuryRecord. If None, uses last fetch.

        Returns:
            DataFrame with columns: player_name, team, team_abbr, position,
            injury_status, injury_description, severity, minutes_penalty,
            is_active, scraped_at.
        """
        if records is None:
            records = self._last_fetch or []
        if not records:
            return pd.DataFrame(columns=[
                "player_name", "team", "team_abbr", "position",
                "injury_status", "injury_description", "severity",
                "minutes_penalty", "is_active", "scraped_at",
            ])
        return pd.DataFrame([r.to_dict() for r in records])

    def get_key_injuries(
        self, min_severity: str = "moderate", min_players: int = 2
    ) -> list[dict]:
        """
        Get teams significantly impacted by injuries.

        Args:
            min_severity: Minimum severity to flag ("moderate" or "high")
            min_players: Minimum number of injured players to flag

        Returns:
            List of dicts with team info and injury details for flagged teams.
        """
        summary = self.get_league_summary()
        flagged = []
        for abbr, info in summary.get("teams", {}).items():
            if info["total_injured"] >= min_players:
                flagged.append({
                    "team_abbr": abbr,
                    "team": info["team"],
                    "total_injured": info["total_injured"],
                    "active_outs": info["active_outs"],
                    "players": info["players"],
                    "impact": (
                        "high" if info["active_outs"] >= 3
                        else "moderate" if info["active_outs"] >= 1
                        else "low"
                    ),
                })
        return sorted(flagged, key=lambda x: x["active_outs"], reverse=True)

    def compute_team_injury_factor(self, team_abbr: str) -> float:
        """
        Compute a single injury impact factor for a team (0.0 to 1.0).

        0.0 = fully healthy (no injuries)
        1.0 = devastated (multiple key players out)

        This factor can be used to adjust team strength predictions.

        Args:
            team_abbr: Team abbreviation.

        Returns:
            Injury impact factor.
        """
        injuries = self.fetch_by_team(team_abbr)
        if not injuries:
            return 0.0

        # Penalty scale: active out = 0.25, doubtful = 0.15,
        # questionable = 0.08, probable = 0.02
        penalty = 0.0
        for record in injuries:
            status_lower = record.injury_status.lower().strip()
            if status_lower in ("out", "out for season"):
                penalty += 0.25
            elif status_lower == "doubtful":
                penalty += 0.15
            elif status_lower in ("questionable", "game time decision"):
                penalty += 0.08
            elif status_lower == "probable":
                penalty += 0.02

        return min(penalty, 1.0)

    # ── HTML Parsing ────────────────────────────────────────────────────

    def _parse_injuries_page(self, html: str) -> list[InjuryRecord]:
        """
        Parse ESPN's injuries HTML page.

        ESPN's layout:
            - Each team section has a <h2> with the team name
            - Followed by a <table> with columns:
                Player Name | Position | Injury Status | Return Date (optional)
            - Team abbreviation is extracted from the section ID or URL

        Args:
            html: Raw HTML of the ESPN injuries page.

        Returns:
            List of parsed InjuryRecord objects.
        """
        soup = BeautifulSoup(html, "html.parser")
        records: list[InjuryRecord] = []

        # Find all team sections — ESPN uses <h2> with team names
        # Try the modern ESPN layout structure first
        team_sections = []

        # Modern ESPN layout: each team injury table is in a div with
        # data attribute or class containing the team abbreviation
        for section in soup.find_all(["section", "div"], class_=lambda c: c and "team" in c.lower() if c else False):
            team_sections.append(section)

        # Fallback: look for tables with specific column structure
        tables = soup.find_all("table")
        if not tables:
            logger.warning("No tables found on ESPN injuries page — layout may have changed")
            return []

        current_team = ""
        current_team_abbr = ""

        for table in tables:
            # Determine which team this table belongs to
            # Strategy: look at preceding headers/text
            prev = table.find_previous(["h1", "h2", "h3", "h4", "span", "a"],
                                       class_=lambda c: c and "team" in c.lower() if c else False)

            if prev:
                team_name_text = prev.get_text(strip=True)
                # Try to match full team name
                team_abbr = extract_team_abbr(team_name_text)
                if team_abbr:
                    current_team = TEAM_ABBREVIATIONS.get(team_abbr, team_name_text)
                    current_team_abbr = team_abbr

            # Parse the table rows
            rows = table.find_all("tr")
            for row in rows[1:]:  # Skip header
                cols = row.find_all("td")
                if len(cols) < 3:
                    continue

                player_name = cols[0].get_text(strip=True) if len(cols) > 0 else ""
                position = cols[1].get_text(strip=True) if len(cols) > 1 else "N/A"
                injury_status = cols[2].get_text(strip=True) if len(cols) > 2 else "N/A"

                # Injury description (sometimes in col 3, or embedded in the status)
                injury_description = ""
                date_updated = None
                if len(cols) > 3:
                    injury_text = cols[3].get_text(strip=True)
                    # Check if this looks like a date or a description
                    if injury_text and not self._looks_like_date(injury_text):
                        injury_description = injury_text
                    else:
                        date_updated = injury_text

                if not player_name:
                    continue

                # If injury description is still empty, try to get it from
                # the injury_status cell's title attribute or nested elements
                if not injury_description:
                    status_cell = cols[2]
                    title_attr = status_cell.get("title", "")
                    if title_attr:
                        injury_description = title_attr
                    else:
                        # Fallback: use the status text as-is
                        injury_description = injury_status

                record = InjuryRecord(
                    player_name=player_name,
                    team=current_team or "Unknown Team",
                    team_abbr=current_team_abbr or "???" if current_team_abbr else self._guess_team_abbr(player_name),
                    position=position,
                    injury_status=injury_status,
                    injury_description=injury_description,
                    date_updated=date_updated,
                )
                records.append(record)

        # Fallback: if no records found via table parsing, try the alternate
        # ESPN layout with div-based injury cards
        if not records:
            records = self._parse_alt_layout(soup)

        return records

    def _parse_alt_layout(self, soup: BeautifulSoup) -> list[InjuryRecord]:
        """
        Fallback parser for ESPN's alternative layout (div-based cards).

        Some ESPN pages use a card-based layout instead of tables.
        """
        records: list[InjuryRecord] = []
        current_team = ""
        current_team_abbr = ""

        # Look for team headers followed by player cards
        for elem in soup.find_all(["h2", "h3", "h4"]):
            text = elem.get_text(strip=True)
            abbr = extract_team_abbr(text)
            if abbr:
                current_team = TEAM_ABBREVIATIONS.get(abbr, text)
                current_team_abbr = abbr

            # Check if next sibling contains player items
            parent = elem.parent
            items = parent.find_all(["li", "div"], class_=lambda c: c and "player" in c.lower() if c else False)
            if not items:
                items = parent.find_all("tr")

            for item in items:
                tds = item.find_all("td") if item.name == "tr" else []
                if tds:
                    # Same table-row parsing as above
                    player_name = tds[0].get_text(strip=True) if len(tds) > 0 else ""
                    position = tds[1].get_text(strip=True) if len(tds) > 1 else ""
                    status = tds[2].get_text(strip=True) if len(tds) > 2 else ""
                    desc = tds[3].get_text(strip=True) if len(tds) > 3 else ""

                    if player_name and current_team_abbr:
                        records.append(InjuryRecord(
                            player_name=player_name,
                            team=current_team,
                            team_abbr=current_team_abbr,
                            position=position or "N/A",
                            injury_status=status or "N/A",
                            injury_description=desc,
                        ))

        return records

def extract_team_abbr(text: str) -> Optional[str]:
    """Try to extract a team abbreviation from text."""
    if not text:
        return None
    text = text.strip()
    # Direct match
    if text.upper() in TEAM_ABBREVIATIONS:
        return text.upper()

    # Full name → abbreviation
    for abbr, full_name in TEAM_ABBREVIATIONS.items():
        if full_name.lower() in text.lower() or text.lower() in full_name.lower():
            return abbr

    # Try city name matching
    city_to_abbr = {
        "atlanta": "ATL",
        "boston": "BOS",
        "brooklyn": "BKN",
        "charlotte": "CHA",
        "chicago": "CHI",
        "cleveland": "CLE",
        "dallas": "DAL",
        "denver": "DEN",
        "detroit": "DET",
        "golden state": "GSW",
        "houston": "HOU",
        "indiana": "IND",
        "la clippers": "LAC",
        "clippers": "LAC",
        "la lakers": "LAL",
        "lakers": "LAL",
        "memphis": "MEM",
        "miami": "MIA",
        "milwaukee": "MIL",
        "minnesota": "MIN",
        "new orleans": "NOP",
        "new york": "NYK",
        "oklahoma city": "OKC",
        "orlando": "ORL",
        "philadelphia": "PHI",
        "phoenix": "PHX",
        "portland": "POR",
        "sacramento": "SAC",
        "san antonio": "SAS",
        "toronto": "TOR",
        "utah": "UTA",
        "washington": "WAS",
    }
    for city, abbr in city_to_abbr.items():
        if city in text.lower():
            return abbr

    return None

    @staticmethod
    def _guess_team_abbr(player_name: str) -> str:
        """Fallback when team can't be determined."""
        return "???"

    @staticmethod
    def _looks_like_date(text: str) -> bool:
        """Rough check if text looks like a date string."""
        import re
        # Match patterns like "Jan 15", "2025-01-15", "01/15", etc.
        date_patterns = [
            r"\w{3,9}\s+\d{1,2}",  # "Jan 15"
            r"\d{4}-\d{2}-\d{2}",  # "2025-01-15"
            r"\d{1,2}/\d{1,2}",    # "01/15"
        ]
        return any(re.search(p, text) for p in date_patterns)
