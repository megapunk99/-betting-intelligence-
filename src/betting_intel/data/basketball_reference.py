"""
Basketball-Reference.com Scraper — fetches NBA game data, team stats, and schedules.

Data Sources:
  - https://www.basketball-reference.com/boxscores/   → Daily box scores
  - https://www.basketball-reference.com/leagues/NBA_{year}.html  → Season standings & team stats
  - https://www.basketball-reference.com/friv/dailyleaders.cgi  → Daily leaders

Output format matches the existing FeatureEngineer so data flows directly into the pipeline.

Key design decisions:
  1. Uses requests + BeautifulSoup with polite rate limiting (1 req/3 sec)
  2. Retry with exponential backoff on HTTP errors
  3. Caches HTML responses to disk to avoid hammering the server
  4. Team name normalization matches ODDS_TO_SHORT_NAME from odds_fetcher
  5. Handles both regular season and playoff games

Usage:
    from betting_intel.data.basketball_reference import BasketballReferenceScraper

    scraper = BasketballReferenceScraper()
    games = scraper.fetch_today_games()          # Live/upcoming games
    box = scraper.fetch_boxscore("202506120BOS") # Single box score by game ID
    dfs = scraper.fetch_recent_days(days=7)      # Last 7 days of completed games
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Cloudscraper for bypassing Cloudflare protection (optional, graceful fallback)
_has_cloudscraper = False
try:
    import cloudscraper
    _has_cloudscraper = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── Rate Limiting ────────────────────────────────────────────────────────
MIN_REQUEST_INTERVAL = 3.0  # Seconds between requests (be polite!)
REQUEST_TIMEOUT = 15        # HTTP request timeout
MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 10]  # Exponential backoff seconds

# ── URLs ────────────────────────────────────────────────────────────────
BR_BASE_URL = "https://www.basketball-reference.com"
BOXSCORE_URL = f"{BR_BASE_URL}/boxscores"
DAILY_LEADERS_URL = f"{BR_BASE_URL}/friv/dailyleaders.cgi"
SEASON_URL = f"{BR_BASE_URL}/leagues/NBA_{{year}}.html"

# ── Cache ────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "cache" / "basketball_ref"
CACHE_TTL_SECONDS = 3600  # 1 hour

# ── Team Name Mapping (short name → basketball-reference abbreviation) ──
# basketball-reference uses 3-letter abbreviations like BOS, LAL, GSW
TEAM_TO_BR_ABBR: dict[str, str] = {
    "Hawks": "ATL", "Celtics": "BOS", "Nets": "BRK",
    "Hornets": "CHO", "Bulls": "CHI", "Cavaliers": "CLE",
    "Mavericks": "DAL", "Nuggets": "DEN", "Pistons": "DET",
    "Warriors": "GSW", "Rockets": "HOU", "Pacers": "IND",
    "Clippers": "LAC", "Lakers": "LAL", "Grizzlies": "MEM",
    "Heat": "MIA", "Bucks": "MIL", "Timberwolves": "MIN",
    "Pelicans": "NOP", "Knicks": "NYK", "Thunder": "OKC",
    "Magic": "ORL", "76ers": "PHI", "Suns": "PHO",
    "Trail Blazers": "POR", "Kings": "SAC", "Spurs": "SAS",
    "Raptors": "TOR", "Jazz": "UTA", "Wizards": "WAS",
}

# Reverse mapping: BR abbreviation → short name
BR_ABBR_TO_TEAM: dict[str, str] = {v: k for k, v in TEAM_TO_BR_ABBR.items()}

# BR abbreviation → full name (for matching with TheOddsAPI names)
BR_ABBR_TO_FULL: dict[str, str] = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BRK": "Brooklyn Nets",
    "CHO": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


class BasketballReferenceScraper:
    """
    Scrapes NBA game data from basketball-reference.com.

    Features:
      - Fetch today's scheduled games
      - Fetch completed box scores (with full team stats)
      - Fetch recent days of games
      - Fetch season-level team stats
      - Cache HTML to avoid repeated requests
      - Polite rate limiting (1 request per 3 seconds minimum)
    """

    def __init__(self, cache_dir: Optional[Path] = None, respect_robots: bool = True):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if _has_cloudscraper:
            try:
                # Use cloudscraper to bypass Cloudflare protection
                self._session = cloudscraper.create_scraper()
            except Exception:
                logger.debug("cloudscraper init failed, falling back to requests")
                self._session = requests.Session()
        else:
            self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        })
        self._last_request_time = 0.0
        self._respect_robots = respect_robots

    # ═══════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════════════

    def fetch_today_games(self) -> list[dict[str, Any]]:
        """
        Fetch today's NBA schedule from basketball-reference.

        Returns list of game dicts with: home_team, away_team, game_id,
        game_date, home_score (None if not started), away_score (None),
        status ("scheduled", "live", "final").
        """
        today_str = date.today().strftime("%Y%m%d")
        url = f"{BOXSCORE_URL}/?month={date.today().month}&day={date.today().day}&year={date.today().year}"
        html = self._fetch(url, cache_key=f"scoreboard_{today_str}")
        if not html:
            return []

        return self._parse_scoreboard(html, today_str)

    def fetch_date_games(self, target_date: str) -> list[dict[str, Any]]:
        """
        Fetch games for a specific date (YYYY-MM-DD format).

        Returns list of game dicts with full box score data for completed games,
        or partial data for live/scheduled games.
        """
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {target_date} (use YYYY-MM-DD)")
            return []

        date_str = dt.strftime("%Y%m%d")
        url = f"{BOXSCORE_URL}/?month={dt.month}&day={dt.day}&year={dt.year}"
        html = self._fetch(url, cache_key=f"scoreboard_{date_str}")
        if not html:
            return []

        return self._parse_scoreboard(html, date_str)

    def fetch_recent_days(self, days: int = 7) -> list[dict[str, Any]]:
        """
        Fetch completed games for the last N days.

        Returns a flat list of game dicts with full box score data.
        Includes: home_team, away_team, home_score, away_score, total_points,
        and per-team stats for feature engineering.
        """
        all_games: list[dict[str, Any]] = []
        today = date.today()

        for offset in range(days):
            check_date = today - timedelta(days=offset)
            date_str = check_date.strftime("%Y-%m-%d")
            games = self.fetch_date_games(date_str)
            # Only include completed games
            for g in games:
                if g.get("status") == "final" and g.get("home_score") is not None:
                    all_games.append(g)
            if games:
                logger.info(f"  {date_str}: {sum(1 for g in games if g.get('status') == 'final')} completed games")

        return all_games

    def fetch_boxscore(self, game_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch a detailed box score for a specific game by game ID.

        Game ID format: YYYYMMDD{away_abbr}{home_abbr}
        Example: "202506120BOS" = June 12, 2025, BOS @ ??? (last 3 chars = home)

        Returns dict with full team stats per quarter, player stats, etc.
        Returns None if the box score page doesn't exist (game not played yet).
        """
        url = f"{BOXSCORE_URL}/{game_id}.html"
        html = self._fetch(url, cache_key=f"boxscore_{game_id}")
        if not html:
            return None

        return self._parse_boxscore(html, game_id)

    def fetch_season_standings(self, year: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Fetch team standings and per-game stats for a season.

        Args:
            year: The ending year of the season (e.g., 2025 for 2024-25 season).
                  Defaults to current season.

        Returns list of dicts with team name, wins, losses, win_pct,
        points_per_game, opp_points_per_game, pace, etc.
        """
        if year is None:
            now = datetime.now()
            year = now.year if now.month >= 10 else now.year - 1

        url = SEASON_URL.format(year=year)
        html = self._fetch(url, cache_key=f"season_{year}")
        if not html:
            return []

        return self._parse_season_standings(html, year)

    def fetch_team_schedule(self, team_abbr: str, year: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Fetch a team's complete game schedule for a season.

        Args:
            team_abbr: Team abbreviation (e.g., "BOS", "LAL")
            year: Season ending year. Defaults to current season.

        Returns list of game dicts with opponent, location, result, score, etc.
        """
        if year is None:
            now = datetime.now()
            year = now.year if now.month >= 10 else now.year - 1

        team_abbr = team_abbr.upper()
        url = f"{BR_BASE_URL}/teams/{team_abbr}/{year}_games.html"
        html = self._fetch(url, cache_key=f"schedule_{team_abbr}_{year}")
        if not html:
            return []

        return self._parse_team_schedule(html, team_abbr, year)

    def fetch_team_stats(self, team_abbr: str, year: Optional[int] = None) -> Optional[dict[str, float]]:
        """
        Fetch per-game team stats for a specific team and season.

        Args:
            team_abbr: Team abbreviation (e.g., "BOS")
            year: Season ending year. Defaults to current season.

        Returns dict with team averages: pts, fgm, fga, fg3m, fg3a, ftm, fta,
        oreb, dreb, reb, ast, stl, blk, tov, pf, pace, etc.
        """
        if year is None:
            now = datetime.now()
            year = now.year if now.month >= 10 else now.year - 1

        url = f"{BR_BASE_URL}/teams/{team_abbr}/{year}.html"
        html = self._fetch(url, cache_key=f"team_stats_{team_abbr}_{year}")
        if not html:
            return None

        return self._parse_team_stats(html, team_abbr, year)

    # ═══════════════════════════════════════════════════════════════════
    #  PARSERS
    # ═══════════════════════════════════════════════════════════════════

    def _parse_scoreboard(self, html: str, date_str: str) -> list[dict[str, Any]]:
        """Parse the basketball-reference scoreboard page.

        Scoreboard HTML structure:
          div.game_summary
            table.teams
              tr.winner   (first row)
                td  -> <a href="/teams/ABBR/year.html">Team Name</a>
                td  -> score (number)
                td  -> <a href="/boxscores/YYYYMMDD0HHH.html">Final</a>
              tr.loser    (second row)
                td  -> <a href="/teams/ABBR/year.html">Team Name</a>
                td  -> score (number)

        Home/away is determined from the boxscore URL:
          /boxscores/YYYYMMDD0HHH.html where HHH = home team abbreviation
        """
        soup = BeautifulSoup(html, "html.parser")
        games: list[dict[str, Any]] = []

        game_summaries = soup.find_all("div", class_="game_summary")
        if not game_summaries:
            return games

        for summary in game_summaries:
            try:
                table = summary.find("table", class_="teams")
                if not table:
                    continue

                rows = table.find_all("tr")
                if len(rows) < 2:
                    continue

                # Extract team abbreviation from <a href="/teams/ABBR/year.html">
                def _extract_abbr(row):
                    cells = row.find_all("td")
                    if len(cells) < 2:
                        return None, None, None
                    link = cells[0].find("a", href=re.compile(r"/teams/([A-Z0-9]{2,4})/"))
                    abbr = link["href"].split("/")[2] if link and link.get("href") else None
                    score = self._parse_score_cell(cells[1])
                    game_link = None
                    if len(cells) > 2:
                        gl = cells[2].find("a", href=re.compile(r"/boxscores/"))
                        if gl and gl.get("href"):
                            m = re.search(r'/boxscores/(\d+[A-Z]{3})\.html', gl["href"])
                            if m:
                                game_link = m.group(1)
                    return abbr, score, game_link

                row0_abbr, row0_score, game_id = _extract_abbr(rows[0])
                row1_abbr, row1_score, _ = _extract_abbr(rows[1])

                if not row0_abbr or not row1_abbr:
                    continue

                row0_abbr = row0_abbr.upper()
                row1_abbr = row1_abbr.upper()

                # Determine home/away from the game URL
                # Format: /boxscores/YYYYMMDD0HHH.html where last 3 chars = home team
                home_abbr = ""
                away_abbr = ""
                if game_id and len(game_id) >= 11:
                    # Last 3 chars of game_id before .html = home team
                    home_abbr = game_id[-3:].upper()
                    away_abbr = row0_abbr if row1_abbr == home_abbr else row1_abbr
                else:
                    # Fallback: treat first row as away (traditional scoreboard convention)
                    away_abbr = row0_abbr
                    home_abbr = row1_abbr
                    game_id = game_id or f"BR_{date_str}_{away_abbr}{home_abbr}"

                home_short = BR_ABBR_TO_TEAM.get(home_abbr, home_abbr)
                away_short = BR_ABBR_TO_TEAM.get(away_abbr, away_abbr)

                # Scores: row0 is winner, row1 is loser
                # Map to correct home/away
                if row0_abbr == home_abbr:
                    home_score = row0_score
                    away_score = row1_score
                else:
                    home_score = row1_score
                    away_score = row0_score

                # Game status
                status = "final" if home_score is not None and away_score is not None else "scheduled"
                if status == "scheduled" and self._has_live_indicator(summary):
                    status = "live"

                game = {
                    "game_id": game_id,
                    "game_date": self._date_str_to_iso(date_str),
                    "home_team": home_short,
                    "away_team": away_short,
                    "home_team_full": BR_ABBR_TO_FULL.get(home_abbr, home_short),
                    "away_team_full": BR_ABBR_TO_FULL.get(away_abbr, away_short),
                    "home_team_abbr": home_abbr,
                    "away_team_abbr": away_abbr,
                    "home_score": home_score,
                    "away_score": away_score,
                    "total_points": (home_score + away_score) if home_score is not None and away_score is not None else None,
                    "status": status,
                    "source": "basketball_reference",
                }

                # For completed games, try to get full box score
                if status == "final" and game_id:
                    boxscore = self.fetch_boxscore(game_id)
                    if boxscore:
                        game.update(boxscore)

                games.append(game)

            except Exception as e:
                logger.debug(f"Error parsing game summary: {e}")
                continue

        return games

    def _parse_boxscore(self, html: str, game_id: str) -> Optional[dict[str, Any]]:
        """Parse a detailed box score page."""
        soup = BeautifulSoup(html, "html.parser")
        result: dict[str, Any] = {}

        try:
            # Extract team basic stats (four factors table)
            # basketball-reference has two tables: one for each team
            tables = soup.find_all("table", class_="stats_table")
            if not tables:
                # Try to find any table with stats
                tables = soup.find_all("table")

            for table in tables:
                table_id = table.get("id", "")
                if "team_stats" in table_id:
                    team_stats = self._parse_team_stats_table(table)
                    for key, val in team_stats.items():
                        result[f"team_stat_{key}"] = val
                elif "four_factors" in table_id:
                    ff_stats = self._parse_four_factors_table(table)
                    for key, val in ff_stats.items():
                        result[f"four_factor_{key}"] = val
                elif "q1" in table_id.lower() or "quarter" in table_id.lower():
                    q_stats = self._parse_quarter_stats(table)
                    if q_stats:
                        for key, val in q_stats.items():
                            result[key] = val

            # Extract by-quarter scoring from the game summary
            score_table = soup.find("table", id=re.compile(r"game_summary|scoring", re.I))
            if not score_table:
                score_table = soup.find("table", class_="stats_table", summary=re.compile("scoring|quarter", re.I))

            if score_table:
                quarter_data = self._parse_scoring_by_quarter(score_table)
                if quarter_data:
                    result["quarters"] = quarter_data

        except Exception as e:
            logger.debug(f"Error parsing boxscore {game_id}: {e}")
            return None

        return result if result else None

    def _parse_season_standings(self, html: str, year: int) -> list[dict[str, Any]]:
        """Parse the season standings and team stats page.

        Basketball-reference season page table structure:
          table#per_game-team  (team per-game averages)
            thead -> header row
            tbody
              tr (one per team)
                td[data-stat=team] -> <a href="/teams/ABBR/year.html">Team Name</a>
                td[data-stat=g]  -> games played
                td[data-stat=pts] -> points per game
                ... etc
        """
        soup = BeautifulSoup(html, "html.parser")
        teams: list[dict[str, Any]] = []

        table = soup.find("table", id="per_game-team")
        if not table:
            logger.warning(f"No per_game-team table found for season {year}")
            return teams

        body = table.find("tbody")
        if not body:
            return teams

        # Team full name -> short name mapping
        FULL_TO_SHORT = {
            "Atlanta Hawks": "Hawks", "Boston Celtics": "Celtics",
            "Brooklyn Nets": "Nets", "Charlotte Hornets": "Hornets",
            "Chicago Bulls": "Bulls", "Cleveland Cavaliers": "Cavaliers",
            "Dallas Mavericks": "Mavericks", "Denver Nuggets": "Nuggets",
            "Detroit Pistons": "Pistons", "Golden State Warriors": "Warriors",
            "Houston Rockets": "Rockets", "Indiana Pacers": "Pacers",
            "LA Clippers": "Clippers", "Los Angeles Lakers": "Lakers",
            "Memphis Grizzlies": "Grizzlies", "Miami Heat": "Heat",
            "Milwaukee Bucks": "Bucks", "Minnesota Timberwolves": "Timberwolves",
            "New Orleans Pelicans": "Pelicans", "New York Knicks": "Knicks",
            "Oklahoma City Thunder": "Thunder", "Orlando Magic": "Magic",
            "Philadelphia 76ers": "76ers", "Phoenix Suns": "Suns",
            "Portland Trail Blazers": "Trail Blazers", "Sacramento Kings": "Kings",
            "San Antonio Spurs": "Spurs", "Toronto Raptors": "Raptors",
            "Utah Jazz": "Jazz", "Washington Wizards": "Wizards",
        }

        # Stat name mapping: data-stat attribute -> friendly key
        # Uses short names (actual data-stat values on the page)
        STAT_MAP = {
            "g": "games_played", "mp": "mp_per_game",
            "fg": "fg_per_game", "fga": "fga_per_game", "fg_pct": "fg_pct",
            "fg3": "fg3_per_game", "fg3a": "fg3a_per_game", "fg3_pct": "fg3_pct",
            "ft": "ft_per_game", "fta": "fta_per_game", "ft_pct": "ft_pct",
            "orb": "orb_per_game", "drb": "drb_per_game", "trb": "trb_per_game",
            "ast": "ast_per_game", "stl": "stl_per_game", "blk": "blk_per_game",
            "tov": "tov_per_game", "pf": "pf_per_game", "pts": "pts_per_game",
            "pace": "pace",
        }

        for row in body.find_all("tr"):
            if row.get("class") and "thead" in " ".join(row.get("class", [])):
                continue
            try:
                # Extract team name + abbreviation from the <a> link
                team_cell = row.find("td", {"data-stat": "team"})
                if not team_cell:
                    continue

                team_link = team_cell.find("a")
                team_text = team_cell.get_text(strip=True).rstrip("*")  # Strip playoff indicator
                team_abbr = ""
                if team_link and team_link.get("href"):
                    m = re.search(r'/teams/([A-Z0-9]{2,4})/', team_link["href"])
                    if m:
                        team_abbr = m.group(1)

                # Map full name to short name
                team_short = None
                for full_name, short in FULL_TO_SHORT.items():
                    if full_name in team_text:
                        team_short = short
                        break
                if not team_short:
                    team_short = team_text  # fallback

                def get_stat(stat_name: str) -> Optional[float]:
                    cell = row.find("td", {"data-stat": stat_name})
                    if cell:
                        text = cell.get_text(strip=True)
                        try:
                            return float(text)
                        except (ValueError, TypeError):
                            pass
                    return None

                team_stats: dict[str, Any] = {
                    "team": team_short,
                    "team_full": team_text,
                    "team_abbr": team_abbr,
                }

                # Collect all available stats using short stat names
                for short_name, friendly_name in SHORT_STAT_MAP.items():
                    val = get_stat(short_name)
                    if val is not None:
                        team_stats[friendly_name] = val

                teams.append(team_stats)

            except Exception as e:
                logger.debug(f"Error parsing team row: {e}")
                continue

        return teams

    def _parse_team_schedule(self, html: str, team_abbr: str, year: int) -> list[dict[str, Any]]:
        """Parse a team's game schedule page."""
        soup = BeautifulSoup(html, "html.parser")
        games: list[dict[str, Any]] = []

        table = soup.find("table", id="games")
        if not table:
            logger.warning(f"No games table found for {team_abbr} {year}")
            return games

        body = table.find("tbody")
        if not body:
            return games

        for row in body.find_all("tr"):
            if row.get("class") and "thead" in row.get("class", []):
                continue
            try:
                def get_val(stat: str) -> str:
                    cell = row.find("td", {"data-stat": stat})
                    return cell.get_text(strip=True) if cell else ""

                date_str = get_val("date_game")
                opponent_abbr = get_val("opp_id")
                location = get_val("game_location")
                result = get_val("game_result")
                team_score = get_val("pts")
                opp_score = get_val("opp_pts")
                overtime = get_val("overtimes")

                if not date_str or not opponent_abbr:
                    continue

                is_home = location != "@"
                home_abbr = team_abbr if is_home else opponent_abbr
                away_abbr = opponent_abbr if is_home else team_abbr

                home_short = BR_ABBR_TO_TEAM.get(home_abbr, home_abbr)
                away_short = BR_ABBR_TO_TEAM.get(away_abbr, away_abbr)

                is_final = result in ("W", "L")
                home_score = int(team_score) if is_final and is_home else (int(opp_score) if is_final else None)
                away_score = int(opp_score) if is_final and is_home else (int(team_score) if is_final else None)

                games.append({
                    "game_id": f"BR_{date_str}_{away_abbr}{home_abbr}",
                    "game_date": self._nba_date_to_iso(date_str, year),
                    "home_team": home_short,
                    "away_team": away_short,
                    "home_team_full": BR_ABBR_TO_FULL.get(home_abbr, home_short),
                    "away_team_full": BR_ABBR_TO_FULL.get(away_abbr, away_short),
                    "home_team_abbr": home_abbr,
                    "away_team_abbr": away_abbr,
                    "home_score": home_score,
                    "away_score": away_score,
                    "total_points": (home_score + away_score) if home_score is not None and away_score is not None else None,
                    "status": "final" if is_final else "scheduled",
                    "overtime": overtime if overtime else "",
                    "source": "basketball_reference",
                })

            except Exception as e:
                logger.debug(f"Error parsing schedule row: {e}")
                continue

        return games

    def _parse_team_stats(self, html: str, team_abbr: str, year: int) -> Optional[dict[str, float]]:
        """Parse per-game team stats from a team's season page.

        The team page (/teams/ABBR/YEAR.html) contains per_game_stats for PLAYERS,
        not the team summary. We use the team's season page by delegating to
        _parse_season_standings and filtering for the requested team.

        This is a thin wrapper that fetches the season page and finds the team row.
        """
        soup = BeautifulSoup(html, "html.parser")
        stats: dict[str, float] = {}

        # Try team summary section at top of page
        team_info = soup.find("div", id="team")
        if team_info:
            # Parse basic stats from the team info section
            p_tags = team_info.find_all("p")
            for p in p_tags:
                text = p.get_text(strip=True)
                # Record: 64-18 (.781), Home: 36-6, Away: 28-12
                rec_match = re.search(r'Record:\s+(\d+)-(\d+)', text)
                if rec_match:
                    stats["wins"] = float(rec_match.group(1))
                    stats["losses"] = float(rec_match.group(2))
                # Expected W-L (pythagorean)
                exp_match = re.search(r'Expected W-L:\s+(\d+)-(\d+)', text)
                if exp_match:
                    stats["expected_wins"] = float(exp_match.group(1))
                    stats["expected_losses"] = float(exp_match.group(2))

        # Get team-level per-game averages from the season standings page
        try:
            season_url = SEASON_URL.format(year=year)
            season_html = self._fetch(season_url, cache_key=f"season_{year}")
            if season_html:
                season_teams = self._parse_season_standings(season_html, year)
                for t in season_teams:
                    if t.get("team_abbr", "").upper() == team_abbr.upper():
                        # Merge in season-level stats (more comprehensive)
                        for k, v in t.items():
                            if k not in ("team", "team_full", "team_abbr") and isinstance(v, (int, float)):
                                stats[k] = v
                        break
        except Exception as e:
            logger.debug(f"Could not fetch season standings for team stats: {e}")

        return stats if stats else None

    def _parse_team_stats_table(self, table) -> dict[str, float]:
        """Parse a team stats table from a box score."""
        stats: dict[str, float] = {}
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["th", "td"])
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True).lower().replace(" ", "_")
                try:
                    val = float(cells[1].get_text(strip=True))
                    stats[label] = val
                except (ValueError, TypeError):
                    pass
        return stats

    def _parse_four_factors_table(self, table) -> dict[str, float]:
        """Parse the four factors table."""
        stats: dict[str, float] = {}
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["th", "td"])
            if len(cells) >= 3:
                label = cells[0].get_text(strip=True)
                home_val = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                away_val = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                try:
                    stats[f"home_{label.lower().replace(' ', '_')}"] = float(home_val) if home_val else 0.0
                    stats[f"away_{label.lower().replace(' ', '_')}"] = float(away_val) if away_val else 0.0
                except (ValueError, TypeError):
                    pass
        return stats

    def _parse_scoring_by_quarter(self, table) -> list[dict[str, Any]]:
        """Parse the by-quarter scoring breakdown."""
        quarters: list[dict[str, Any]] = []
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["th", "td"])
            labels = [c.get_text(strip=True) for c in cells]
            if "Quarter" in labels or "Q1" in labels or "1st" in labels:
                # Header row — skip
                continue
            if len(labels) >= 3:
                team_name = labels[0]
                # Skip empty rows
                if not team_name:
                    continue
                team_short = None
                for full, short in BR_ABBR_TO_TEAM.items():
                    if full in team_name or short in team_name:
                        team_short = short
                        break
                if not team_short:
                    team_short = team_name

                q_data = {"team": team_short}
                for i, label in enumerate(labels[1:], 1):
                    try:
                        q_data[f"q{i}"] = int(label)
                    except (ValueError, IndexError, TypeError):
                        pass
                if len(q_data) > 1:
                    quarters.append(q_data)

        return quarters

    def _parse_quarter_stats(self, table) -> Optional[dict[str, float]]:
        """Parse quarter-by-quarter stats from a box score table."""
        data: dict[str, float] = {}
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            label = cells[0].get_text(strip=True)
            for i, cell in enumerate(cells[1:], 1):
                try:
                    val = float(cell.get_text(strip=True))
                    data[f"{label.lower()}_col{i}"] = val
                except (ValueError, TypeError):
                    pass
        return data

    # ═══════════════════════════════════════════════════════════════════
    #  HTTP & CACHE
    # ═══════════════════════════════════════════════════════════════════

    def _fetch(self, url: str, cache_key: str = "", force_refresh: bool = False) -> Optional[str]:
        """
        Fetch a URL with caching and rate limiting.

        Args:
            url: The full URL to fetch
            cache_key: Key for the disk cache (if empty, caching is skipped)
            force_refresh: If True, bypass cache

        Returns:
            HTML string, or None on failure
        """
        # Check cache first
        if cache_key and not force_refresh:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

        # Fetch with retries
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"Fetching: {url} (attempt {attempt + 1}/{MAX_RETRIES})")
                resp = self._session.get(url, timeout=REQUEST_TIMEOUT)
                self._last_request_time = time.time()

                if resp.status_code == 200:
                    html = resp.text
                    if cache_key:
                        self._save_cache(cache_key, html)
                    return html
                elif resp.status_code == 404:
                    logger.debug(f"404: {url}")
                    return None
                elif resp.status_code == 429:
                    wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 30
                    logger.warning(f"Rate limited (429). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(f"HTTP {resp.status_code}: {url}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt])
                        continue
                    return None

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout: {url}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                return None
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                return None

        return None

    def _load_cache(self, key: str) -> Optional[str]:
        """Load cached HTML from disk if fresh."""
        cache_path = self.cache_dir / f"{self._sanitize_key(key)}.html"
        if not cache_path.exists():
            return None
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age > CACHE_TTL_SECONDS:
                return None
            with open(cache_path, "r", encoding="utf-8") as f:
                logger.debug(f"Cache HIT: {key} ({age:.0f}s old)")
                return f.read()
        except Exception:
            return None

    def _save_cache(self, key: str, html: str):
        """Save HTML to disk cache."""
        try:
            cache_path = self.cache_dir / f"{self._sanitize_key(key)}.html"
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    @staticmethod
    def _sanitize_key(key: str) -> str:
        """Sanitize a cache key to a safe filename."""
        return re.sub(r'[^a-zA-Z0-9_-]', '_', key)[:100]

    # ═══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_score_cell(cell) -> Optional[int]:
        """Parse a score from a table cell."""
        if cell is None:
            return None
        # Check for links (box score links mean game is done)
        link = cell.find("a")
        if link:
            text = link.get_text(strip=True)
        else:
            text = cell.get_text(strip=True)
        try:
            return int(text)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _has_live_indicator(summary) -> bool:
        """Check if a game summary shows a game in progress."""
        text = summary.get_text().lower()
        indicators = ["q1", "q2", "q3", "q4", "ot", "final", "in progress", "halftime"]
        return any(ind in text for ind in indicators)

    @staticmethod
    def _date_str_to_iso(date_str: str) -> str:
        """Convert YYYYMMDD to YYYY-MM-DD."""
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return date_str

    @staticmethod
    def _nba_date_to_iso(date_str: str, season_year: int) -> str:
        """Convert an NBA date string (e.g., 'Mon, Dec 25, 2024') to YYYY-MM-DD."""
        try:
            from dateutil import parser as dateparser
            dt = dateparser.parse(date_str)
            return dt.strftime("%Y-%m-%d")
        except ImportError:
            pass
        # Manual parsing attempt
        try:
            months = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
            }
            parts = date_str.replace(",", "").split()
            if len(parts) >= 4:
                month_str = parts[1][:3].lower()
                day = int(parts[2])
                year = int(parts[3]) if parts[3].isdigit() else season_year
                month = months.get(month_str, 1)
                return f"{year:04d}-{month:02d}-{day:02d}"
        except Exception:
            pass
        # Fallback: use the season year
        if season_year:
            return f"{season_year}-01-01"
        return date_str

    def close(self):
        """Close the HTTP session."""
        self._session.close()


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def fetch_recent_completed_games(days: int = 3) -> list[dict[str, Any]]:
    """
    Convenience: fetch completed NBA games from basketball-reference for the last N days.

    Returns game dicts with team stats ready for feature engineering.
    Falls back to empty list on failure (never raises).
    """
    try:
        scraper = BasketballReferenceScraper()
        games = scraper.fetch_recent_days(days=days)
        scraper.close()
        logger.info(f"Basketball-Reference: {len(games)} games from last {days} days")
        return games
    except Exception as e:
        logger.warning(f"Basketball-Reference fetch failed: {e}")
        return []


def fetch_upcoming_games() -> list[dict[str, Any]]:
    """
    Convenience: fetch today's scheduled games from basketball-reference.

    Returns list of game dicts. Each has home_team, away_team, status="scheduled".
    Falls back to empty list on failure.
    """
    try:
        scraper = BasketballReferenceScraper()
        games = scraper.fetch_today_games()
        scraper.close()
        return [g for g in games if g.get("status") == "scheduled"]
    except Exception as e:
        logger.warning(f"Basketball-Reference upcoming fetch failed: {e}")
        return []


__all__ = [
    "BasketballReferenceScraper",
    "fetch_recent_completed_games",
    "fetch_upcoming_games",
]
