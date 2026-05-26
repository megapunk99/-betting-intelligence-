"""
Populate the NBA SQLite database with game logs from nba_api.
Fetches data for the last 3 full seasons + current season.
Optimized: fetches all teams at once per season (only 3 API calls).

Usage:
    python scripts/populate_nba_data.py

Requires: pip install nba-api pandas
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

from config import DB_PATH

# ── Team ID to name mapping (NBA API) ────────────────────────────────────
TEAMS = {
    1610612737: "Hawks", 1610612738: "Celtics", 1610612751: "Nets",
    1610612766: "Hornets", 1610612741: "Bulls", 1610612739: "Cavaliers",
    1610612742: "Mavericks", 1610612743: "Nuggets", 1610612765: "Pistons",
    1610612744: "Warriors", 1610612745: "Rockets", 1610612754: "Pacers",
    1610612746: "Clippers", 1610612747: "Lakers", 1610612763: "Grizzlies",
    1610612748: "Heat", 1610612749: "Bucks", 1610612750: "Timberwolves",
    1610612740: "Pelicans", 1610612752: "Knicks", 1610612760: "Thunder",
    1610612753: "Magic", 1610612755: "76ers", 1610612756: "Suns",
    1610612757: "Trail Blazers", 1610612758: "Kings", 1610612759: "Spurs",
    1610612761: "Raptors", 1610612762: "Jazz", 1610612764: "Wizards",
}

TEAM_ABBREV = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN",
    1610612766: "CHA", 1610612741: "CHI", 1610612739: "CLE",
    1610612742: "DAL", 1610612743: "DEN", 1610612765: "DET",
    1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
    1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM",
    1610612748: "MIA", 1610612749: "MIL", 1610612750: "MIN",
    1610612740: "NOP", 1610612752: "NYK", 1610612760: "OKC",
    1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS",
    1610612761: "TOR", 1610612762: "UTA", 1610612764: "WAS",
}


def fetch_season_game_logs(season: str = "2024-25") -> pd.DataFrame:
    """Fetch ALL game logs for a season in one API call."""
    from nba_api.stats.endpoints import LeagueGameLog

    try:
        print(f"  Fetching season {season}...")
        game_logs = LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="T",
        )
        df = game_logs.get_data_frames()[0]

        # Map team names
        df["TEAM_NAME"] = df["TEAM_ID"].map(TEAMS)
        df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(TEAM_ABBREV)

        # Season ID
        season_id = int(season.replace("-", "")) if "-" in season else int(season)
        df["SEASON_ID"] = season_id

        print(f"    {len(df)} game log entries, {df['TEAM_ID'].nunique()} teams")
        return df

    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()


def fetch_all_seasons(seasons: list = None) -> pd.DataFrame:
    """Fetch all seasons."""
    if seasons is None:
        now = datetime.now()
        # Determine current season end year
        current_season_end = now.year if now.month >= 10 else now.year - 1
        seasons = [
            f"{current_season_end - 2}-{str(current_season_end - 1)[-2:]}",
            f"{current_season_end - 1}-{str(current_season_end)[-2:]}",
            f"{current_season_end}-{str(current_season_end + 1)[-2:]}",
        ]

    all_dfs = []

    for season in seasons:
        df = fetch_season_game_logs(season)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1.0)  # Be polite to the API

    if not all_dfs:
        print("\n[!] No data fetched. Check internet connection or nba_api installation.")
        return pd.DataFrame()

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["GAME_ID", "TEAM_ID", "TEAM_NAME"])

    # Parse dates
    if "GAME_DATE" in full_df.columns:
        full_df["GAME_DATE"] = pd.to_datetime(full_df["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  Total: {len(full_df)} game log entries")
    print(f"  Seasons: {', '.join(seasons)}")
    print(f"  Teams: {full_df['TEAM_NAME'].nunique()}")
    print(f"  Games: {full_df['GAME_ID'].nunique()}")
    if "GAME_DATE" in full_df.columns:
        print(f"  Date range: {full_df['GAME_DATE'].min()} to {full_df['GAME_DATE'].max()}")
    print(f"{'='*60}")

    return full_df


def populate_database(df: pd.DataFrame):
    """Write to SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))

    df.to_sql("game_logs", conn, if_exists="replace", index=False)

    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_game_id ON game_logs(GAME_ID)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_team_id ON game_logs(TEAM_ID)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_game_date ON game_logs(GAME_DATE)")
    conn.commit()

    count = cursor.execute("SELECT COUNT(*) FROM game_logs").fetchone()[0]
    conn.close()

    print(f"\n  [OK] Database populated: {count} rows in 'game_logs' table")
    print(f"  [OK] Database: {DB_PATH}")


def show_column_info(df: pd.DataFrame):
    """Show column mapping info for debugging."""
    print(f"\n  Columns in fetched data ({len(df.columns)}):")
    for col in sorted(df.columns):
        dtype = str(df[col].dtype)
        print(f"    {col:30s} ({dtype})")


if __name__ == "__main__":
    print("=" * 60)
    print("  NBA DATA POPULATOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n[+] Fetching NBA game logs from nba_api (one call per season)...")
    df = fetch_all_seasons()

    if df.empty:
        print("\n[!] No data fetched. Trying most recent season...")
        df = fetch_all_seasons(seasons=["2024-25"])

    if not df.empty:
        show_column_info(df)
        populate_database(df)

        print(f"\n{'='*60}")
        print("  [OK] DATA POPULATION COMPLETE")
        print("  You can now run the prediction engine!")
        print(f"{'='*60}")
    else:
        print("\n[!] Could not fetch NBA data.")
        print("   Check your internet connection.")
        print("   Try: pip install --upgrade nba-api")
