#!/usr/bin/env python3
"""
Update NBA Data — Scrapes 2025-26 season game logs from official NBA CDN API.

Fetches the full schedule, then for each completed game fetches the boxscore
with detailed team stats and inserts them into the SQLite database.

Usage:
    python tools/update_data.py                  # Full update (all completed games)
    python tools/update_data.py --dry-run        # Show what would be inserted
    python tools/update_data.py --recent 10      # Only fetch the N most recent games
"""

import sys
import os
import time
import re
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime

import requests

# Ensure src/ is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from betting_intel.config import DB_PATH

# ── API Endpoints ──────────────────────────────────────────────────────────
SCHEDULE_URL = (
    "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
)
BOXSCORE_TEMPLATE = (
    "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
}

# ── NBA Team tricodes (30 NBA teams) ─────────────────────────────────────
NBA_TRICODES: set[str] = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
}

# ── NBA Team ID Map (teamTricode → teamId) ────────────────────────────────
TEAM_ID_MAP: dict[str, int] = {
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751,
    "CHA": 1610612766, "CHI": 1610612741, "CLE": 1610612739,
    "DAL": 1610612742, "DEN": 1610612743, "DET": 1610612765,
    "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
    "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750,
    "NOP": 1610612740, "NYK": 1610612752, "OKC": 1610612760,
    "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759,
    "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764,
}


def _parse_minutes(min_str: str | None) -> int:
    """Parse 'PT290M00.00S' → 240 (total minutes played)."""
    if not min_str:
        return 240
    m = re.search(r"(\d+)M", str(min_str))
    return int(m.group(1)) if m else 240


def _fmt_pct(val) -> float:
    """Convert boxscore percentage (may be 0.44 or None) to DB format."""
    if val is None:
        return 0.0
    return round(float(val), 3)


def fetch_schedule() -> dict | None:
    """Fetch the full NBA season schedule from the CDN."""
    print("  Fetching schedule from NBA CDN...")
    resp = requests.get(SCHEDULE_URL, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        print(f"  {RED}[!] Schedule fetch failed: HTTP {resp.status_code}{RESET}")
        return None
    data = resp.json()
    schedule = data.get("leagueSchedule", {})
    season_year = schedule.get("seasonYear")
    print(f"    Season: {season_year}")
    game_dates = schedule.get("gameDates", [])
    print(f"    Game dates: {len(game_dates)}")
    return schedule


def fetch_boxscore(game_id: str) -> dict | None:
    """Fetch boxscore for a single game."""
    url = BOXSCORE_TEMPLATE.format(game_id=game_id)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def is_game_completed(game: dict) -> bool:
    """Check if a game has been played (final score exists)."""
    home_score = game.get("homeTeam", {}).get("score")
    away_score = game.get("awayTeam", {}).get("score")
    if home_score is None or away_score is None:
        return False
    try:
        return int(home_score) > 0 and int(away_score) > 0
    except (ValueError, TypeError):
        return False


def game_to_rows(game: dict, boxscore_data: dict) -> list[dict]:
    """
    Convert a game + boxscore into two database rows (home + away).

    Returns list of dicts, one per team, with keys matching the DB schema.
    """
    game_id = game.get("gameId", "")
    gdate = game.get("gameDateUTC", "")[:10]  # YYYY-MM-DD

    home_info = game.get("homeTeam", {})
    away_info = game.get("awayTeam", {})

    home_tricode = home_info.get("teamTricode", "")
    away_tricode = away_info.get("teamTricode", "")

    home_score = int(home_info.get("score", 0))
    away_score = int(away_info.get("score", 0))

    # Determine winner
    home_won = home_score > away_score

    # Team names from boxscore data
    game_data = boxscore_data.get("game", {})
    home_box = game_data.get("homeTeam", {})
    away_box = game_data.get("awayTeam", {})

    home_name = home_box.get("teamName", home_tricode)
    away_name = away_box.get("teamName", away_tricode)

    def _extract_stats(team_box: dict, is_home: bool, won: bool) -> dict:
        stats = team_box.get("statistics", {})
        plus_minus = stats.get("plusMinusPoints")
        if plus_minus is None:
            # Compute plus/minus from score difference
            if is_home:
                plus_minus = home_score - away_score
            else:
                plus_minus = away_score - home_score

        # Derive SEASON_ID from the game date
        gdate_parts = gdate.split("-")
        if len(gdate_parts) == 3:
            year = int(gdate_parts[0])
            month = int(gdate_parts[1])
            season_display_year = year if month >= 10 else year - 1
            season_str = f"{season_display_year}-{str(season_display_year + 1)[-2:]}"
            season_id = int(f"{season_display_year}{season_display_year + 1}")
        else:
            season_id = 202526
            season_str = "2025-26"

        row = {
            "SEASON_ID": season_id,
            "TEAM_ID": team_box.get("teamId") or 0,
            "TEAM_ABBREVIATION": team_box.get("teamTricode", ""),
            "TEAM_NAME": team_box.get("teamName", ""),
            "GAME_ID": game_id,
            "GAME_DATE": gdate,
            "MATCHUP": (
                f"{team_box.get('teamName', '')} vs. {away_name}"
                if is_home else
                f"{team_box.get('teamName', '')} @ {home_name}"
            ),
            "WL": "W" if won else "L",
            "MIN": _parse_minutes(stats.get("minutes")),
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
            "PLUS_MINUS": int(plus_minus) if plus_minus is not None else 0,
            "SEASON": season_str,
        }
        return row

    home_row = _extract_stats(home_box, is_home=True, won=home_won)
    away_row = _extract_stats(away_box, is_home=False, won=not home_won)

    return [home_row, away_row]


def get_existing_game_ids(db_path: Path) -> set[str]:
    """Return set of GAME_IDs already in the database."""
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT GAME_ID FROM game_logs")
    ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    return ids


def insert_rows(db_path: Path, rows: list[dict], dry_run: bool = False) -> int:
    """Insert game log rows into the database. Returns count of rows inserted."""
    if not rows:
        return 0

    if dry_run:
        return len(rows)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    columns = [
        "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
        "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT",
        "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
        "PLUS_MINUS", "SEASON",
    ]

    placeholders = ", ".join(["?" for _ in columns])
    col_names = ", ".join(columns)

    insert_sql = f"INSERT OR IGNORE INTO game_logs ({col_names}) VALUES ({placeholders})"

    inserted = 0
    for row in rows:
        values = [row.get(c) for c in columns]
        try:
            cursor.execute(insert_sql, values)
            inserted += cursor.rowcount
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    return inserted

# ── ANSI Colors ──────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(
        description="Update NBA data — scrape 2025-26 season from NBA CDN API",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be inserted without inserting")
    parser.add_argument("--recent", type=int, default=0,
                        help="Only fetch the N most recent completed games")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay in seconds between API calls (default: 0.1)")
    args = parser.parse_args()

    db_path = DB_PATH
    print(f"{CYAN}{BOLD}NBA Data Updater{RESET}")
    print(f"  Database: {db_path}")
    print(f"  Dry run:  {args.dry_run}")
    print()

    # Step 1: Fetch schedule
    schedule = fetch_schedule()
    if not schedule:
        print(f"{RED}[!] Failed to fetch schedule{RESET}")
        return 1

    game_dates = schedule.get("gameDates", [])

    # Step 2: Collect all completed games, in chronological order
    all_completed: list[tuple[str, dict]] = []  # (date, game)
    total_in_schedule = 0

    for gd in game_dates:
        date_str = gd.get("gameDate", "")
        for game in gd.get("games", []):
            total_in_schedule += 1
            if is_game_completed(game):
                all_completed.append((date_str, game))

    print(f"  Total games in schedule: {total_in_schedule}")
    print(f"  Completed games:         {len(all_completed)}")

    # Step 3: Check what's already in the database
    existing_ids = get_existing_game_ids(db_path)
    print(f"  Already in DB:           {len(existing_ids) // 2} games "
          f"({len(existing_ids)} rows)")
    print()

    # Step 4: Filter to new games only
    new_games = [
        (date, game) for date, game in all_completed
        if game.get("gameId", "") not in existing_ids
    ]

    # Apply --recent limit (from the end = most recent)
    if args.recent > 0 and len(new_games) > args.recent:
        new_games = new_games[-args.recent:]

    print(f"  New games to fetch:      {len(new_games)}")
    if not new_games:
        print(f"\n{GREEN}Database is up to date!{RESET}")
        return 0

    # Step 5: Fetch boxscores and build rows
    print(f"\n  Fetching boxscores ({len(new_games)} games)...")

    total_rows = 0
    total_inserted = 0
    errors = 0
    skipped = 0

    for i, (date_str, game) in enumerate(new_games):
        game_id = game.get("gameId", "?")
        home_tri = game.get("homeTeam", {}).get("teamTricode", "?")
        away_tri = game.get("awayTeam", {}).get("teamTricode", "?")

        # Progress indicator
        if (i + 1) % 25 == 0 or i == 0 or i == len(new_games) - 1:
            pct = (i + 1) / len(new_games) * 100
            print(f"    [{i+1}/{len(new_games)}] {pct:.0f}% — "
                  f"{away_tri} @ {home_tri} ({date_str})...", end="")

        # Fetch boxscore with retry
        boxscore = None
        for attempt in range(3):
            boxscore = fetch_boxscore(game_id)
            if boxscore:
                break
            if attempt < 2:
                time.sleep(0.5)
        if not boxscore:
            if (i + 1) % 25 == 0 or i == 0 or i == len(new_games) - 1:
                print(f" {YELLOW}no boxscore{RESET}")
            errors += 1
            continue

        # Skip non-NBA teams (preseason vs international/G League)
        if home_tri not in NBA_TRICODES or away_tri not in NBA_TRICODES:
            skipped += 1
            continue

        # Check if stats are available (game may be scheduled but not started)
        game_data = boxscore.get("game", {})
        home_stats = game_data.get("homeTeam", {}).get("statistics", {})
        if not home_stats or not home_stats.get("fieldGoalsMade"):
            if (i + 1) % 25 == 0 or i == 0 or i == len(new_games) - 1:
                print(f" {YELLOW}no stats yet{RESET}")
            skipped += 1
            continue

        rows = game_to_rows(game, boxscore)
        inserted = insert_rows(
            db_path, rows, dry_run=args.dry_run
        )
        total_inserted += inserted
        total_rows += len(rows)

        if (i + 1) % 25 == 0 or i == 0 or i == len(new_games) - 1:
            print(f" {GREEN}done{RESET}")

        # Rate limit delay
        if args.delay > 0:
            time.sleep(args.delay)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"{BOLD}Summary{RESET}")
    print(f"  Games processed:   {len(new_games)}")
    print(f"  Rows inserted:     {total_inserted} ({total_inserted // 2} games)")
    print(f"  Errors (no bxsc):  {errors}")
    print(f"  Skipped (no stat): {skipped}")

    if args.dry_run:
        print(f"\n{YELLOW}Dry run — no data was written to the database.{RESET}")
    else:
        total_in_db = len(existing_ids) + total_rows
        print(f"\n  Total rows in DB:   {total_in_db}")
        print(f"  Total games in DB:  {total_in_db // 2}")
        print(f"{GREEN}Update complete!{RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
