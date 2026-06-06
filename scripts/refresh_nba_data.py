#!/usr/bin/env python3
"""
Refresh NBA Data — Fetches fresh game logs from NBA.com via nba_api
and upserts them into nba_data.db.

Called daily by the retrain pipeline (daily_retrain.py, weekly_retrain.py)
to ensure the database has the latest game results.

Usage:
    python scripts/refresh_nba_data.py                  # Full refresh
    python scripts/refresh_nba_data.py --season 2025-26 # Specific season
    python scripts/refresh_nba_data.py --dry-run         # Show what would happen
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── nba_api imports (try gracefully) ──────────────────────────────────
try:
    from nba_api.stats.endpoints import LeagueGameLog
    from nba_api.stats.library.parameters import SeasonType
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

# ── ANSI colors ───────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ── Expected game_logs columns (must match DB schema) ─────────────────
GAME_LOG_COLUMNS = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
    "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS",
    "SEASON",
]


def fetch_season_game_logs(season: str = "2025-26") -> list[dict]:
    """Fetch game logs for a season from the NBA API."""
    if not NBA_API_AVAILABLE:
        print(f"  {YELLOW}[!] nba_api not installed. Install via: pip install nba_api{RESET}")
        return []

    try:
        print(f"  Fetching game logs for season {season}...")
        start = time.time()

        gamelog = LeagueGameLog(
            season=season,
            season_type_all_star=SeasonType.regular
        )

        df = gamelog.get_data_frames()[0]
        records = df.to_dict("records")
        elapsed = time.time() - start
        print(f"    Fetched {len(records)} rows in {elapsed:.1f}s")

        # Add SEASON column if not present
        for rec in records:
            if "SEASON" not in rec:
                rec["SEASON"] = season

        return records
    except Exception as e:
        print(f"  {RED}[!] NBA API error: {e}{RESET}")
        return []


def fetch_multiple_seasons(seasons: list[str] | None = None) -> list[dict]:
    """Fetch game logs for multiple seasons and combine them.

    Defaults to current season if no seasons specified.
    Also tries to get the previous season for context.
    """
    if seasons is None:
        seasons = ["2024-25", "2025-26"]

    all_records = []
    for season in seasons:
        try:
            records = fetch_season_game_logs(season)
            all_records.extend(records)
        except Exception as e:
            print(f"  {YELLOW}[!] Season {season} failed: {e}{RESET}")

    return all_records


def get_db_columns(db_path: Path) -> list[str]:
    """Get the actual column names from the game_logs table."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(game_logs)")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()
    return columns


def upsert_game_logs(db_path: Path, records: list[dict], dry_run: bool = False) -> dict:
    """Insert or update game logs in the database.

    Dynamically detects the DB schema and only inserts columns that exist
    in both the incoming records and the database table.

    Returns:
        dict with counts of inserted, updated, and total
    """
    if not records:
        return {"inserted": 0, "updated": 0, "total": 0, "skipped": 0}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Dynamically detect what columns exist in the DB
    db_columns = get_db_columns(db_path)

    # Get existing (GAME_ID, TEAM_ID) pairs
    cursor.execute("SELECT GAME_ID, TEAM_ID FROM game_logs")
    existing_pairs = {(row[0], row[1]) for row in cursor.fetchall()}

    new_game_ids = set()
    inserted = 0
    updated = 0
    skipped = 0

    # Only use columns that exist in BOTH the DB and our expected columns
    columns = [c for c in GAME_LOG_COLUMNS if c in db_columns]

    if not columns:
        print(f"  {RED}[!] No matching columns found between schema and expected columns{RESET}")
        conn.close()
        return {"inserted": 0, "updated": 0, "total": 0, "skipped": len(records)}

    placeholders = ", ".join(["?" for _ in columns])
    col_names = ", ".join(columns)

    for rec in records:
        # Extract values in column order
        values = []
        skip = False
        for col in columns:
            if col in rec and rec[col] is not None:
                values.append(rec[col])
            elif col == "SEASON":
                # Add default season if record doesn't have it
                values.append("2025-26")
            else:
                # Try with 0 or empty string default for missing values
                default = 0 if col not in ("SEASON_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "SEASON") else ""
                values.append(default)

        if skip or len(values) != len(columns):
            skipped += 1
            continue

        game_id = rec.get("GAME_ID", "")
        team_id = rec.get("TEAM_ID", "")
        pair = (game_id, team_id)

        if pair in existing_pairs:
            # Update existing row
            if not dry_run:
                set_clause = ", ".join([f"{c}=?" for c in columns])
                update_sql = f"UPDATE game_logs SET {set_clause} WHERE GAME_ID=? AND TEAM_ID=?"
                try:
                    cursor.execute(update_sql, values + [game_id, team_id])
                except Exception as e:
                    skipped += 1
                    continue
            updated += 1
        else:
            # Insert new row
            if not dry_run:
                insert_sql = f"INSERT INTO game_logs ({col_names}) VALUES ({placeholders})"
                try:
                    cursor.execute(insert_sql, values)
                except sqlite3.IntegrityError:
                    # Conflict — update instead
                    set_clause = ", ".join([f"{c}=?" for c in columns])
                    update_sql = f"UPDATE game_logs SET {set_clause} WHERE GAME_ID=? AND TEAM_ID=?"
                    try:
                        cursor.execute(update_sql, values + [game_id, team_id])
                    except Exception:
                        skipped += 1
                        continue
                except Exception:
                    skipped += 1
                    continue
            inserted += 1
            existing_pairs.add(pair)

        if game_id:
            new_game_ids.add(game_id)

    if not dry_run:
        conn.commit()

    conn.close()

    return {
        "inserted": inserted,
        "updated": updated,
        "total": len(records),
        "skipped": skipped,
        "new_games": len(new_game_ids - existing_game_ids),
        "total_games": len(new_game_ids | existing_game_ids),
    }


def print_summary(stats: dict, db_path: Path, elapsed: float):
    """Print a human-readable summary."""
    print(f"\n  {BOLD}SUMMARY{RESET}")
    print(f"  {'-' * 50}")
    print(f"  Database:        {db_path}")
    print(f"  Records fetched: {stats['total']}")
    print(f"  New rows:        {GREEN}{stats['inserted']}{RESET}")
    print(f"  Updated rows:    {YELLOW}{stats['updated']}{RESET}")
    print(f"  Skipped:         {RED}{stats['skipped']}{RESET}")
    print(f"  New games:       {GREEN}{stats['new_games']}{RESET}")
    print(f"  Total games:     {stats['total_games']}")
    print(f"  Duration:        {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch fresh NBA game logs from nba_api and update the database.",
    )
    parser.add_argument("--season", type=str, default=None,
                        help="Specific season (e.g., '2025-26'). Default: current + previous")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without modifying the database")
    args = parser.parse_args()

    db_path = PROJECT_ROOT / "data" / "nba_data.db"

    if not db_path.exists():
        print(f"  {RED}[!] Database not found: {db_path}{RESET}")
        print(f"  Run the full pipeline first to create it.")
        return 1

    print(f"\n{CYAN}{BOLD}{'=' * 60}{RESET}")
    print(f"{CYAN}{BOLD}  NBA DATA REFRESH{RESET}")
    print(f"{CYAN}{BOLD}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 60}{RESET}")

    if args.dry_run:
        print(f"\n  {YELLOW}[DRY RUN] No changes will be made{RESET}")

    seasons = [args.season] if args.season else ["2024-25", "2025-26"]
    print(f"\n  Fetching seasons: {', '.join(seasons)}")

    start = time.time()

    records = fetch_multiple_seasons(seasons)
    if not records:
        print(f"  {RED}[!] No records fetched. Check your nba_api installation.{RESET}")
        return 1

    stats = upsert_game_logs(db_path, records, dry_run=args.dry_run)

    elapsed = time.time() - start
    print_summary(stats, db_path, elapsed)

    # Check if anything actually changed
    new_data_added = stats["inserted"] > 0 or stats["new_games"] > 0
    if new_data_added:
        print(f"\n  {GREEN}New data available. Run daily_retrain.py to update model caches.{RESET}")
    else:
        print(f"\n  {DIM}No new games found. Data is already up to date.{RESET}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
