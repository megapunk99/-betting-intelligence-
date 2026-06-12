#!/usr/bin/env python3
"""
Check NBA database health — extracted from CI workflows.

Usage:
    python tools/check_db.py
    # Exit code 0 if DB has sufficient data, 1 otherwise

Used by weekly_retraining.yml, monthly_full_retraining.yml, and ci-slow-tests.yml.
"""

import sys
import argparse
from pathlib import Path

# Add src/ to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from betting_intel.config import DB_PATH
except ImportError:
    DB_PATH = PROJECT_ROOT / "data" / "betting_intel.db"


def check_database(min_rows: int = 50) -> bool:
    """Check that the database exists and has sufficient game data.

    Args:
        min_rows: Minimum number of game_logs rows required.

    Returns:
        True if database is healthy with sufficient data, False otherwise.
    """
    db_path = Path(DB_PATH) if isinstance(DB_PATH, str) else DB_PATH

    if not db_path.exists():
        print(f"NBA database not found at: {db_path}")
        return False

    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        # Try multiple possible table names
        for possible_table in ["game_logs", "game_data", "games", "nba_games"]:
            if possible_table in table_names:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {possible_table}"
                ).fetchone()[0]
                conn.close()
                print(f"NBA database has {count} rows in '{possible_table}' table")
                if count >= min_rows:
                    return True
                print(f"Not enough data: {count} < {min_rows}")
                return False

        # Fallback: count total rows across all tables
        total_rows = 0
        for tname in table_names:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()
                total_rows += row[0]
            except Exception:
                pass
        conn.close()

        print(f"NBA database has ~{total_rows} total rows across {len(table_names)} tables")
        if total_rows >= min_rows:
            return True
        print(f"Not enough data: {total_rows} < {min_rows}")
        return False

    except Exception as e:
        print(f"Error checking database: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check NBA database health")
    parser.add_argument("--min-rows", type=int, default=50,
                        help="Minimum number of game rows required (default: 50)")
    args = parser.parse_args()
    healthy = check_database(min_rows=args.min_rows)
    sys.exit(0 if healthy else 1)
