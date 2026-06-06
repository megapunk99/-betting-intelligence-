"""
NBA Player Stats Updater — fetches player-level game logs from boxscore APIs.

Supports two data sources:
- NBA CDN (cdn.nba.com): NBA-style game IDs (0022..., 0042..., etc.)
- ESPN summary API (site.api.espn.com): ESPN-style game IDs (4017...)

Pulls player stats (PTS, MIN, FGM, FGA, FG3M, FG3A, FTM, FTA, REB, AST, STL,
BLK, TOV, PF, PLUS_MINUS) and stores them in a `player_game_logs` table.

Usage:
    from betting_intel.data.player_stats import PlayerStatsManager

    manager = PlayerStatsManager()
    manager.update_all()                          # Fetch all unprocessed games
    pg = manager.get_player_ppg("Jalen Brunson")  # 27.0
    team_pts = manager.get_team_missing_ppg("NYK", ["Julius Randle"])
    # Returns 22.0 (Randle's season PPG)
"""

from __future__ import annotations

import re
import sys
import time
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests

from betting_intel.config import DB_PATH

logger = logging.getLogger(__name__)

# ── API Configuration ─────────────────────────────────────────────────────
BOXSCORE_TEMPLATE = (
    "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
)

ESPN_SUMMARY_TEMPLATE = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
    "?event={game_id}"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
}

ESPN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.espn.com/nba/scoreboard",
}

# ── Player Table Creation ─────────────────────────────────────────────────

_PLAYER_GAME_LOGS_DDL = """
CREATE TABLE IF NOT EXISTS player_game_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    GAME_ID TEXT NOT NULL,
    TEAM_ID INTEGER NOT NULL,
    PLAYER_ID INTEGER NOT NULL,
    PLAYER_NAME TEXT NOT NULL,
    POSITION TEXT DEFAULT '',
    MINUTES INTEGER DEFAULT 0,
    PTS INTEGER DEFAULT 0,
    FGM INTEGER DEFAULT 0,
    FGA INTEGER DEFAULT 0,
    FG_PCT REAL DEFAULT 0.0,
    FG3M INTEGER DEFAULT 0,
    FG3A INTEGER DEFAULT 0,
    FG3_PCT REAL DEFAULT 0.0,
    FTM INTEGER DEFAULT 0,
    FTA INTEGER DEFAULT 0,
    FT_PCT REAL DEFAULT 0.0,
    OREB INTEGER DEFAULT 0,
    DREB INTEGER DEFAULT 0,
    REB INTEGER DEFAULT 0,
    AST INTEGER DEFAULT 0,
    STL INTEGER DEFAULT 0,
    BLK INTEGER DEFAULT 0,
    TOV INTEGER DEFAULT 0,
    PF INTEGER DEFAULT 0,
    PLUS_MINUS INTEGER DEFAULT 0,
    SEASON_ID INTEGER DEFAULT 0,
    GAME_DATE TEXT DEFAULT '',
    scraped_at TEXT DEFAULT '',
    UNIQUE(GAME_ID, PLAYER_ID)
)
"""

_PLAYER_TRACKING_DDL = """
CREATE TABLE IF NOT EXISTS player_tracking (
    player_name TEXT PRIMARY KEY,
    team_abbr TEXT NOT NULL,
    games_played INTEGER DEFAULT 0,
    season_ppg REAL DEFAULT 0.0,
    season_min REAL DEFAULT 0.0,
    season_reb REAL DEFAULT 0.0,
    season_ast REAL DEFAULT 0.0,
    season_fgm REAL DEFAULT 0.0,
    season_fga REAL DEFAULT 0.0,
    season_fg3m REAL DEFAULT 0.0,
    season_fg_pct REAL DEFAULT 0.0,
    last_updated TEXT DEFAULT ''
)
"""


# ── Parser Helpers ────────────────────────────────────────────────────────


def _parse_minutes(min_str: str | None) -> int:
    """Parse 'PT26M01.00S' -> 26 (integer minutes played)."""
    if not min_str:
        return 0
    m = re.search(r"(\d+)M", str(min_str))
    return int(m.group(1)) if m else 0


def _fmt_pct(val) -> float:
    """Convert boxscore percentage to float (may be 0.44 or None)."""
    if val is None:
        return 0.0
    return round(float(val), 3)


def _parse_stat_pair(val: str) -> tuple[int, int]:
    """Parse '10-15' -> (10, 15) for FG/3PT/FT compound stats."""
    if not val or "-" not in val:
        return (0, 0)
    parts = val.split("-")
    try:
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (0, 0)


# ── Manager ───────────────────────────────────────────────────────────────


class PlayerStatsManager:
    """
    Manages player-level game logs from NBA CDN and ESPN boxscore APIs.

    Automatically dispatches to the correct API based on game ID prefix:
    - `0022...`, `0042...`, etc. → NBA CDN (cdn.nba.com)
    - `4017...` (ESPN IDs) → ESPN summary API (site.api.espn.com)

    Usage:
        manager = PlayerStatsManager()
        manager.update_all()  # Fetch all unprocessed games
        ppg = manager.get_player_ppg("Jalen Brunson")
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self):
        """Ensure the player stats tables exist."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(_PLAYER_GAME_LOGS_DDL)
        conn.execute(_PLAYER_TRACKING_DDL)
        conn.commit()
        conn.close()

    # ── Public API ──────────────────────────────────────────────────────

    def get_player_ppg(self, player_name: str) -> float:
        """
        Get a player's current season PPG from the tracking table.

        Tries exact match first. If not found, extracts the last name
        and does a LIKE search (handles CDN abbreviated names like
        "J. Randle" when queried with "Julius Randle", and vice versa
        after the _rebuild_tracking merge step).

        Args:
            player_name: Full player name (e.g., "Jalen Brunson")

        Returns:
            Season PPG average, or 0.0 if not found.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Strategy 1: Exact match (handles merged full-name entries)
        cursor.execute(
            "SELECT season_ppg FROM player_tracking WHERE player_name = ?",
            (player_name,),
        )
        row = cursor.fetchone()
        if row:
            conn.close()
            return row[0]

        # Strategy 2: Extract last name and search for any variant
        # Handles both directions:
        #   - Abbreviated query ("D. Lillard") → find full name ("Damian Lillard")
        #   - Full-name query ("Damian Lillard") → find abbreviated entry
        # Uses the entry with the highest PPG (real stats over zero-stats)
        parts = player_name.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            cursor.execute(
                "SELECT season_ppg FROM player_tracking "
                "WHERE player_name LIKE ? "
                "ORDER BY season_ppg DESC LIMIT 1",
                (f"%{last_name}%",),
            )
            row = cursor.fetchone()
            if row:
                conn.close()
                return row[0]

        conn.close()
        return 0.0

    def get_player_stats(self, player_name: str) -> dict:
        """
        Get full season stats for a player.

        Tries exact match first, then falls back to last-name search
        (handles both abbreviated and full name formats).

        Returns:
            Dict with keys: ppg, min, reb, ast, fgm, fga, fg_pct, games_played
            Empty dict if player not found.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Strategy 1: Exact match
        cursor.execute(
            "SELECT games_played, season_ppg, season_min, season_reb, "
            "season_ast, season_fgm, season_fga, season_fg_pct, team_abbr "
            "FROM player_tracking WHERE player_name = ?",
            (player_name,),
        )
        row = cursor.fetchone()

        if not row:
            # Strategy 2: Last-name search (handles format mismatch)
            parts = player_name.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                cursor.execute(
                    "SELECT games_played, season_ppg, season_min, season_reb, "
                    "season_ast, season_fgm, season_fga, season_fg_pct, team_abbr "
                    "FROM player_tracking "
                    "WHERE player_name LIKE ? "
                    "ORDER BY season_ppg DESC LIMIT 1",
                    (f"%{last_name}%",),
                )
                row = cursor.fetchone()

        conn.close()
        if not row:
            return {}
        return {
            "games_played": row[0],
            "ppg": row[1],
            "min": row[2],
            "reb": row[3],
            "ast": row[4],
            "fgm": row[5],
            "fga": row[6],
            "fg_pct": row[7],
            "team": row[8],
        }

    def get_team_players(self, team_abbr: str) -> list[dict]:
        """
        Get all tracked players for a team with their season stats.

        Args:
            team_abbr: Team abbreviation (e.g., "NYK", "SAS")

        Returns:
            List of dicts with player stats, sorted by PPG descending.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT player_name, games_played, season_ppg, season_min, "
            "season_fgm, season_fga "
            "FROM player_tracking WHERE team_abbr = ? "
            "ORDER BY season_ppg DESC",
            (team_abbr.upper(),),
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "name": r[0],
                "games_played": r[1],
                "ppg": r[2],
                "min": r[3],
                "fgm": r[4],
                "fga": r[5],
            }
            for r in rows
        ]

    def search_player(self, query: str) -> list[dict]:
        """
        Search for players by name (fuzzy, case-insensitive).

        Args:
            query: Partial player name (e.g., "Brunson", "James")

        Returns:
            List of matching player dicts.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT player_name, team_abbr, games_played, season_ppg "
            "FROM player_tracking WHERE player_name LIKE ? "
            "ORDER BY season_ppg DESC LIMIT 20",
            (f"%{query}%",),
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {"name": r[0], "team": r[1], "games_played": r[2], "ppg": r[3]}
            for r in rows
        ]

    def count_unprocessed_games(self) -> int:
        """
        Count how many games in game_logs have NOT been processed for player data.

        Returns:
            Number of unprocessed game IDs.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(DISTINCT g.GAME_ID)
            FROM game_logs g
            LEFT JOIN player_game_logs p ON g.GAME_ID = p.GAME_ID
            WHERE p.GAME_ID IS NULL
        """)
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_unprocessed_game_ids(self, limit: int = 50) -> list[str]:
        """
        Get game IDs that haven't been processed for player data yet.

        Args:
            limit: Maximum number of IDs to return.

        Returns:
            List of game ID strings.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT g.GAME_ID
            FROM game_logs g
            LEFT JOIN player_game_logs p ON g.GAME_ID = p.GAME_ID
            WHERE p.GAME_ID IS NULL
            ORDER BY g.GAME_DATE DESC
            LIMIT ?
        """, (limit,))
        ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return ids

    def update_all(self, limit: int = 50, delay: float = 0.75) -> dict:
        """
        Fetch player data for unprocessed games and update the database.

        Args:
            limit: Max games to process in one call.
            delay: Seconds between API calls.

        Returns:
            Summary dict with keys: processed, players_added, errors, games_in_db
        """
        game_ids = self.get_unprocessed_game_ids(limit=limit)

        if not game_ids:
            return {"processed": 0, "players_added": 0, "errors": 0, "games_in_db": 0}

        total_players = 0
        errors = 0

        for i, gid in enumerate(game_ids):
            try:
                players = self._fetch_and_store_game(gid)
                if players is None:
                    errors += 1
                else:
                    total_players += len(players)
            except Exception as e:
                logger.warning(f"Error processing game {gid}: {e}")
                errors += 1

            if delay > 0:
                time.sleep(delay)

            if (i + 1) % 10 == 0:
                logger.info(f"  Player stats: {i+1}/{len(game_ids)} games processed")

        # Refresh the tracking table
        self._rebuild_tracking()

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT GAME_ID) FROM player_game_logs")
        games_in_db = cursor.fetchone()[0]
        conn.close()

        return {
            "processed": len(game_ids),
            "players_added": total_players,
            "errors": errors,
            "games_in_db": games_in_db,
        }

    def update_recent(self, num_games: int = 10) -> dict:
        """
        Fetch player data for the N most recent unprocessed games.

        Args:
            num_games: Number of most recent games to process.

        Returns:
            Summary dict.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT g.GAME_ID
            FROM game_logs g
            LEFT JOIN player_game_logs p ON g.GAME_ID = p.GAME_ID
            WHERE p.GAME_ID IS NULL
            ORDER BY g.GAME_DATE DESC
            LIMIT ?
        """, (num_games,))
        ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not ids:
            return {"processed": 0, "players_added": 0, "errors": 0, "games_in_db": 0}

        return self.update_all(limit=num_games, delay=0.75)

    # ── Internal Methods ────────────────────────────────────────────────

    def _fetch_and_store_game(self, game_id: str) -> list[dict] | None:
        """
        Fetch boxscore for a game and store player stats.

        Dispatches to NBA CDN or ESPN API based on game ID prefix:
        - `0022...`, `0042...`, etc. → NBA CDN (cdn.nba.com)
        - `4017...` (ESPN IDs) → ESPN summary API

        Args:
            game_id: NBA game ID (e.g., "0042500316" or "401716954")

        Returns:
            List of player dicts stored, or None on failure.
        """
        # Dispatch based on game ID prefix
        if game_id.startswith("4017"):
            return self._fetch_and_store_game_espn(game_id)
        else:
            return self._fetch_and_store_game_nba(game_id)

    def _fetch_and_store_game_nba(self, game_id: str) -> list[dict] | None:
        """
        Fetch boxscore from NBA CDN and store player stats.

        Args:
            game_id: NBA game ID (e.g., "0042500316")

        Returns:
            List of player dicts stored, or None on failure.
        """
        url = BOXSCORE_TEMPLATE.format(game_id=game_id)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:
            return None

        game = data.get("game", {})
        game_status = game.get("gameStatus", 0)
        if game_status < 2:  # Not final/complete
            return None

        home_team = game.get("homeTeam", {})
        away_team = game.get("awayTeam", {})

        # Extract game metadata
        home_tricode = home_team.get("teamTricode", "")
        away_tricode = away_team.get("teamTricode", "")
        home_id = home_team.get("teamId", 0)
        away_id = away_team.get("teamId", 0)
        game_date = game.get("gameEt", "")[:10]
        season_id = self._derive_season_id(game_date)

        # Extract player data from both teams
        all_players = []
        for team_box, team_tricode, team_id in [
            (home_team, home_tricode, home_id),
            (away_team, away_tricode, away_id),
        ]:
            for player in team_box.get("players", []):
                stats = player.get("statistics", {})
                player_entry = {
                    "GAME_ID": game_id,
                    "TEAM_ID": team_id,
                    "PLAYER_ID": player.get("personId", 0),
                    "PLAYER_NAME": player.get("nameI", ""),
                    "POSITION": player.get("position", ""),
                    "MINUTES": _parse_minutes(stats.get("minutes")),
                    "PTS": int(stats.get("points", 0)),
                    "FGM": int(stats.get("fieldGoalsMade", 0)),
                    "FGA": int(stats.get("fieldGoalsAttempted", 0)),
                    "FG_PCT": _fmt_pct(stats.get("fieldGoalsPercentage")),
                    "FG3M": int(stats.get("threePointersMade", 0)),
                    "FG3A": int(stats.get("threePointersAttempted", 0)),
                    "FG3_PCT": _fmt_pct(stats.get("threePointersPercentage")),
                    "FTM": int(stats.get("freeThrowsMade", 0)),
                    "FTA": int(stats.get("freeThrowsAttempted", 0)),
                    "FT_PCT": _fmt_pct(stats.get("freeThrowsPercentage")),
                    "OREB": int(stats.get("reboundsOffensive", 0)),
                    "DREB": int(stats.get("reboundsDefensive", 0)),
                    "REB": int(stats.get("reboundsOffensive", 0)) + int(stats.get("reboundsDefensive", 0)),
                    "AST": int(stats.get("assists", 0)),
                    "STL": int(stats.get("steals", 0)),
                    "BLK": int(stats.get("blocks", 0)),
                    "TOV": int(stats.get("turnovers", 0)),
                    "PF": int(stats.get("foulsPersonal", 0)),
                    "PLUS_MINUS": int(stats.get("plusMinusPoints", 0)),
                    "SEASON_ID": season_id,
                    "GAME_DATE": game_date,
                    "scraped_at": datetime.now().isoformat(),
                }
                all_players.append(player_entry)

        if not all_players:
            return None

        # Store in DB
        self._store_players_bulk(all_players)
        return all_players

    def _fetch_and_store_game_espn(self, game_id: str) -> list[dict] | None:
        """
        Fetch boxscore from ESPN summary API and store player stats.

        The ESPN API returns player stats as flat lists matched to labels:
            labels = ['MIN', 'PTS', 'FG', '3PT', 'FT', 'REB', 'AST',
                      'TO', 'STL', 'BLK', 'OREB', 'DREB', 'PF', '+/-']
            stats  = ['36', '28', '10-15', '2-4', '6-7', '12', '8', ...]

        Compound fields (FG, 3PT, FT) are "made-attempted" format.

        Args:
            game_id: ESPN game ID (e.g., "401716954")

        Returns:
            List of player dicts stored, or None on failure.
        """
        url = ESPN_SUMMARY_TEMPLATE.format(game_id=game_id)
        try:
            resp = requests.get(url, headers=ESPN_HEADERS, timeout=30)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:
            return None

        # Verify game is finished
        header = data.get("header", {})
        competitions = header.get("competitions", [])
        if not competitions:
            return None

        comp = competitions[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            return None

        # Extract game date (format: "2024-10-07T...Z")
        game_date = (comp.get("date", "") or "")[:10]
        season_id = self._derive_season_id(game_date)

        # Build team_id lookup: competitors -> {homeAway: teamId}
        team_map = {}  # homeAway -> {id, abbr}
        for competitor in comp.get("competitors", []):
            team = competitor.get("team", {})
            ha = competitor.get("homeAway", "")
            tid = int(team.get("id", 0))
            abbr = team.get("abbreviation", "")
            team_map[ha] = {"id": tid, "abbr": abbr}

        # Parse player data from boxscore.players
        players_data = data.get("boxscore", {}).get("players", [])

        all_players = []
        for team_entry in players_data:
            team_info = team_entry.get("team", {})
            home_away = "home" if team_entry.get("homeAway") == "home" else "away"
            if home_away in team_map:
                team_id = team_map[home_away]["id"]
            else:
                team_id = int(team_info.get("id", 0))

            for stat_group in team_entry.get("statistics", []):
                labels = stat_group.get("labels", [])
                if "PTS" not in labels or "MIN" not in labels:
                    continue  # Skip non-standard stat groups

                for athlete_entry in stat_group.get("athletes", []):
                    if athlete_entry.get("didNotPlay", False):
                        continue  # Skip DNP players

                    athlete = athlete_entry.get("athlete", {})
                    stats_list = athlete_entry.get("stats", [])

                    if not stats_list:
                        continue

                    # Build stat dict from labels + flat stats list
                    stat_dict = {}
                    for i, label in enumerate(labels):
                        if i < len(stats_list):
                            stat_dict[label] = stats_list[i]

                    # Parse compound fields
                    fgm, fga = _parse_stat_pair(stat_dict.get("FG", ""))
                    fg3m, fg3a = _parse_stat_pair(stat_dict.get("3PT", ""))
                    ftm, fta = _parse_stat_pair(stat_dict.get("FT", ""))

                    # Parse +/- (may be "+2", "-5", "--", or "E")
                    pm_str = stat_dict.get("PLUS_MINUS", "0")
                    try:
                        plus_minus = int(pm_str)
                    except ValueError:
                        plus_minus = 0

                    # Parse minutes (ESPN uses integer "36", not "PT36M...S")
                    min_str = stat_dict.get("MIN", "0")
                    try:
                        minutes = int(min_str) if min_str else 0
                    except ValueError:
                        minutes = 0

                    player_entry = {
                        "GAME_ID": game_id,
                        "TEAM_ID": team_id,
                        "PLAYER_ID": int(athlete.get("id", 0)),
                        "PLAYER_NAME": athlete.get("displayName", ""),
                        "POSITION": athlete.get("position", {}).get("abbreviation", ""),
                        "MINUTES": minutes,
                        "PTS": int(stat_dict.get("PTS", 0)),
                        "FGM": fgm,
                        "FGA": fga,
                        "FG_PCT": _fmt_pct(fgm / fga) if fga > 0 else 0.0,
                        "FG3M": fg3m,
                        "FG3A": fg3a,
                        "FG3_PCT": _fmt_pct(fg3m / fg3a) if fg3a > 0 else 0.0,
                        "FTM": ftm,
                        "FTA": fta,
                        "FT_PCT": _fmt_pct(ftm / fta) if fta > 0 else 0.0,
                        "OREB": int(stat_dict.get("OREB", 0)),
                        "DREB": int(stat_dict.get("DREB", 0)),
                        "REB": int(stat_dict.get("REB", 0)),
                        "AST": int(stat_dict.get("AST", 0)),
                        "STL": int(stat_dict.get("STL", 0)),
                        "BLK": int(stat_dict.get("BLK", 0)),
                        "TOV": int(stat_dict.get("TO", 0)),
                        "PF": int(stat_dict.get("PF", 0)),
                        "PLUS_MINUS": plus_minus,
                        "SEASON_ID": season_id,
                        "GAME_DATE": game_date,
                        "scraped_at": datetime.now().isoformat(),
                    }
                    all_players.append(player_entry)

        if not all_players:
            return None

        self._store_players_bulk(all_players)
        return all_players

    def _store_players_bulk(self, players: list[dict]):
        """Bulk-insert player game logs into the database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        columns = [
            "GAME_ID", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "POSITION",
            "MINUTES", "PTS", "FGM", "FGA", "FG_PCT",
            "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
            "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
            "PLUS_MINUS", "SEASON_ID", "GAME_DATE", "scraped_at",
        ]

        placeholders = ", ".join(["?" for _ in columns])
        col_names = ", ".join(columns)

        insert_sql = (
            f"INSERT OR IGNORE INTO player_game_logs "
            f"({col_names}) VALUES ({placeholders})"
        )

        for player in players:
            values = [player.get(c) for c in columns]
            try:
                cursor.execute(insert_sql, values)
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        conn.close()

    @staticmethod
    def _normalize_abbreviated_name(name: str) -> str | None:
        """
        Convert an abbreviated NBA CDN name ('D. Lillard') to its last name ('Lillard').

        Returns the last name if the name matches the abbreviated pattern
        (single initial + '. ' + last name), or None otherwise.
        """
        m = re.match(r"^[A-Z]\.\s+(.+)$", name.strip())
        if m:
            return m.group(1).strip()
        return None

    def _rebuild_tracking(self):
        """
        Rebuild the player_tracking table from player_game_logs.

        Computes season averages PPG, MIN, etc., grouped by player name
        and current team (last team they played for).

        After initial aggregation, merges entries where the same player
        appears under both abbreviated (NBA CDN "D. Lillard") and full
        (ESPN "Damian Lillard") names. The full name is kept as canonical
        with combined stats from both sources.
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Clear and rebuild
        cursor.execute("DELETE FROM player_tracking")

        # Step 1: Build tracking from player_game_logs joined with game_logs
        cursor.execute("""
            INSERT INTO player_tracking (player_name, team_abbr, games_played,
                season_ppg, season_min, season_reb, season_ast,
                season_fgm, season_fga, season_fg3m, season_fg_pct,
                last_updated)
            SELECT
                p.PLAYER_NAME,
                t.TEAM_ABBREVIATION AS team_abbr,
                COUNT(*) AS games_played,
                ROUND(AVG(p.PTS), 1) AS season_ppg,
                ROUND(AVG(p.MINUTES), 1) AS season_min,
                ROUND(AVG(p.REB), 1) AS season_reb,
                ROUND(AVG(p.AST), 1) AS season_ast,
                ROUND(AVG(p.FGM), 1) AS season_fgm,
                ROUND(AVG(p.FGA), 1) AS season_fga,
                ROUND(AVG(p.FG3M), 1) AS season_fg3m,
                ROUND(AVG(CAST(p.FGM AS REAL) / NULLIF(p.FGA, 0)), 3) AS season_fg_pct,
                datetime('now') AS last_updated
            FROM player_game_logs p
            JOIN (
                SELECT GAME_ID, TEAM_ID, TEAM_ABBREVIATION
                FROM game_logs
                WHERE GAME_ID IN (SELECT DISTINCT GAME_ID FROM player_game_logs)
            ) t ON p.GAME_ID = t.GAME_ID AND p.TEAM_ID = t.TEAM_ID
            GROUP BY p.PLAYER_NAME
            ORDER BY season_ppg DESC
        """)

        conn.commit()

        # Step 2: Detect and merge abbreviated names into full names.
        # Query PLAYER_GAME_LOGS directly to catch entries that may not have
        # joined with game_logs (e.g., test records, partial data).
        cursor.execute("""
            SELECT PLAYER_NAME,
                   COUNT(*) as games_played,
                   ROUND(AVG(PTS), 1) as season_ppg,
                   ROUND(AVG(MINUTES), 1) as season_min,
                   ROUND(AVG(REB), 1) as season_reb,
                   ROUND(AVG(AST), 1) as season_ast,
                   ROUND(AVG(FGM), 1) as season_fgm,
                   ROUND(AVG(FGA), 1) as season_fga,
                   ROUND(AVG(CAST(FGM AS REAL) / NULLIF(FGA, 0)), 3) as season_fg_pct
            FROM player_game_logs
            GROUP BY PLAYER_NAME
        """)
        all_raw = cursor.fetchall()

        # Build lookup: last_name -> list of (player_name, row) for non-abbreviated names
        full_by_last: dict[str, list[tuple[str, tuple]]] = {}
        abbr_entries: list[tuple[str, str, tuple]] = []  # (abbrev_name, last_name, row)

        for row in all_raw:
            pname = row[0]
            last = self._normalize_abbreviated_name(pname)
            if last:
                abbr_entries.append((pname, last.lower(), row))
            else:
                parts = pname.strip().split()
                if parts:
                    key = parts[-1].lower()
                    full_by_last.setdefault(key, []).append((pname, row))

        # For each abbreviated entry with zero stats, find matching full-name
        # entry with actual stats and merge them into the tracking table.
        updates: list[tuple] = []
        abbr_to_delete: list[str] = []

        for abbr_name, last_lower, abbr_row in abbr_entries:
            # Abbreviation already has stats — no need to merge
            if abbr_row[2] > 0 or abbr_row[3] > 0:
                continue

            matches = full_by_last.get(last_lower, [])
            if not matches:
                continue

            # Among matches, pick the one with the most games (likely the real player)
            matches.sort(key=lambda m: m[1][1], reverse=True)
            full_name, full_row = matches[0]

            # Only merge if full-name entry actually has stats
            if full_row[2] == 0 and full_row[3] == 0:
                continue

            # Use the FULL-NAME entry's stats directly (not weighted average)
            # because the abbreviated entry's zeros are corrupt data (NBA CDN
            # returned 0 PTS/MIN for known star players as roster listing,
            # not actual game stats). Weighted averaging would dilute real
            # stats with these zeros.
            updates.append((
                full_name,
                full_row[1],           # games_played from real stats
                full_row[2],           # ppg
                full_row[3],           # min
                full_row[4],           # reb
                full_row[5],           # ast
                full_row[6],           # fgm
                full_row[7],           # fga
                full_row[8],           # fg_pct
            ))
            abbr_to_delete.append(abbr_name)

        # Apply: upsert full-name entries, delete merged abbreviated ones
        for (full_name, total_gp, ppg, min_, reb, ast, fgm, fga, fg_pct) in updates:
            # Check if entry already exists in tracking (has team_abbr from SQL JOIN)
            cursor.execute(
                "SELECT team_abbr FROM player_tracking WHERE player_name = ?",
                (full_name,),
            )
            existing = cursor.fetchone()
            if existing:
                # Entry exists — preserve team_abbr, update stats
                cursor.execute(
                    "UPDATE player_tracking SET games_played = ?, season_ppg = ?, "
                    "season_min = ?, season_reb = ?, season_ast = ?, "
                    "season_fgm = ?, season_fga = ?, season_fg_pct = ?, "
                    "last_updated = datetime('now') WHERE player_name = ?",
                    (total_gp, ppg, min_, reb, ast, fgm, fga, fg_pct, full_name)
                )
            else:
                # Entry not in tracking — insert with team abbreviation
                # derived from the abbreviated entry's tracking row
                cursor.execute(
                    "SELECT team_abbr FROM player_tracking WHERE player_name LIKE ?",
                    (f"%{full_name.split()[-1]}%",),
                )
                team_row = cursor.fetchone()
                team_abbr = team_row[0] if team_row else ''
                cursor.execute(
                    "INSERT INTO player_tracking "
                    "(player_name, team_abbr, games_played, season_ppg, season_min, "
                    "season_reb, season_ast, season_fgm, season_fga, season_fg_pct, last_updated) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                    (full_name, team_abbr, total_gp, ppg, min_, reb, ast, fgm, fga, fg_pct)
                )

        for abbr_name in abbr_to_delete:
            cursor.execute("DELETE FROM player_tracking WHERE player_name = ?", (abbr_name,))

        if updates:
            logger.info(f"  Merged {len(updates)} abbreviated entries into full names")
            logger.info(f"  Deleted {len(abbr_to_delete)} stale zero-stat entries")

        conn.commit()
        conn.close()

    @staticmethod
    def _derive_season_id(game_date: str) -> int:
        """Derive SEASON_ID (e.g., 202526) from a game date string YYYY-MM-DD."""
        if not game_date or len(game_date) < 4:
            return 202526
        try:
            year = int(game_date[:4])
            month = int(game_date[5:7]) if len(game_date) >= 7 else 10
            display_year = year if month >= 10 else year - 1
            return int(f"{display_year}{display_year + 1}")
        except (ValueError, IndexError):
            return 202526

    def get_team_missing_ppg(self, team_abbr: str, missing_player_names: list[str]) -> float:
        """
        Compute total PPG missing for injured/out players on a team.

        Looks up each player's actual season PPG from the tracking table.

        Args:
            team_abbr: Team abbreviation (e.g., "NYK")
            missing_player_names: List of player names suspected injured

        Returns:
            Sum of actual season PPG for missing players.
        """
        total = 0.0
        for name in missing_player_names:
            # Extract just the player name (may contain " (22 PPG, STAR)" suffix)
            clean_name = name.split(" (")[0].strip()
            ppg = self.get_player_ppg(clean_name)
            total += ppg
        return total


# ── Standalone CLI ────────────────────────────────────────────────────────


def main():
    """Run the player stats updater as a CLI."""
    # ── Windows Console Encoding Fix ──────────────────────────────
    if sys.stdout.encoding and sys.stdout.encoding.upper() not in ('UTF-8', 'UTF8'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except AttributeError:
            pass

    import argparse

    parser = argparse.ArgumentParser(
        description="Update player stats from NBA CDN boxscore API",
    )
    parser.add_argument(
        "--recent", type=int, default=10,
        help="Number of most recent games to process (default: 10)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process ALL unprocessed games"
    )
    parser.add_argument(
        "--delay", type=float, default=0.75,
        help="Delay between API calls in seconds (default: 0.75)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip delay (risks rate limiting - not recommended)"
    )
    args = parser.parse_args()

    manager = PlayerStatsManager()
    unprocessed = manager.count_unprocessed_games()
    print(f"Unprocessed games: {unprocessed}")

    if unprocessed == 0:
        print("All games processed. Nothing to do.")
        return 0

    delay = 0.0 if args.fast else args.delay
    if args.all:
        result = manager.update_all(limit=unprocessed, delay=delay)
    else:
        result = manager.update_recent(num_games=args.recent)

    print(f"Processed: {result['processed']} games")
    print(f"Players added: {result['players_added']}")
    print(f"Errors: {result['errors']}")
    print(f"Games in player DB: {result['games_in_db']}")

    # Show top scorers found
    conn = sqlite3.connect(str(manager.db_path))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT player_name, team_abbr, season_ppg FROM player_tracking "
        "WHERE games_played >= 5 ORDER BY season_ppg DESC LIMIT 10"
    )
    top = cursor.fetchall()
    conn.close()

    if top:
        print("\nTop scorers found:")
        for name, team, ppg in top:
            print(f"  {name:25s} {team:3s}  {ppg:.1f} PPG")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
