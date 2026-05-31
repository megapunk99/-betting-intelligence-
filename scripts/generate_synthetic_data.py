"""
Generate synthetic NBA game log data for the betting intelligence system.
Creates realistic-looking data when nba_api is unreachable.

Usage:
    python scripts/generate_synthetic_data.py
"""

import sys
import os
from pathlib import Path

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings("ignore")

# Add src/ to path so we can import from betting_intel.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from betting_intel.config import DB_PATH

np.random.seed(42)
random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
#  NBA Teams
# ═══════════════════════════════════════════════════════════════════════════

TEAMS = [
    {"id": 1610612737, "name": "Hawks", "abbr": "ATL", "conf": "East", "off_rating": 112.3, "def_rating": 114.1},
    {"id": 1610612738, "name": "Celtics", "abbr": "BOS", "conf": "East", "off_rating": 118.5, "def_rating": 110.2},
    {"id": 1610612751, "name": "Nets", "abbr": "BKN", "conf": "East", "off_rating": 113.1, "def_rating": 115.8},
    {"id": 1610612766, "name": "Hornets", "abbr": "CHA", "conf": "East", "off_rating": 108.2, "def_rating": 117.6},
    {"id": 1610612741, "name": "Bulls", "abbr": "CHI", "conf": "East", "off_rating": 112.8, "def_rating": 114.9},
    {"id": 1610612739, "name": "Cavaliers", "abbr": "CLE", "conf": "East", "off_rating": 115.2, "def_rating": 112.5},
    {"id": 1610612742, "name": "Mavericks", "abbr": "DAL", "conf": "West", "off_rating": 117.1, "def_rating": 115.3},
    {"id": 1610612743, "name": "Nuggets", "abbr": "DEN", "conf": "West", "off_rating": 118.0, "def_rating": 113.7},
    {"id": 1610612765, "name": "Pistons", "abbr": "DET", "conf": "East", "off_rating": 109.5, "def_rating": 117.2},
    {"id": 1610612744, "name": "Warriors", "abbr": "GSW", "conf": "West", "off_rating": 117.8, "def_rating": 115.1},
    {"id": 1610612745, "name": "Rockets", "abbr": "HOU", "conf": "West", "off_rating": 113.5, "def_rating": 114.8},
    {"id": 1610612754, "name": "Pacers", "abbr": "IND", "conf": "East", "off_rating": 119.2, "def_rating": 116.5},
    {"id": 1610612746, "name": "Clippers", "abbr": "LAC", "conf": "West", "off_rating": 116.3, "def_rating": 112.9},
    {"id": 1610612747, "name": "Lakers", "abbr": "LAL", "conf": "West", "off_rating": 115.6, "def_rating": 114.7},
    {"id": 1610612763, "name": "Grizzlies", "abbr": "MEM", "conf": "West", "off_rating": 114.8, "def_rating": 113.2},
    {"id": 1610612748, "name": "Heat", "abbr": "MIA", "conf": "East", "off_rating": 113.2, "def_rating": 112.4},
    {"id": 1610612749, "name": "Bucks", "abbr": "MIL", "conf": "East", "off_rating": 117.5, "def_rating": 113.8},
    {"id": 1610612750, "name": "Timberwolves", "abbr": "MIN", "conf": "West", "off_rating": 114.5, "def_rating": 108.9},
    {"id": 1610612740, "name": "Pelicans", "abbr": "NOP", "conf": "West", "off_rating": 114.2, "def_rating": 113.5},
    {"id": 1610612752, "name": "Knicks", "abbr": "NYK", "conf": "East", "off_rating": 116.1, "def_rating": 113.0},
    {"id": 1610612760, "name": "Thunder", "abbr": "OKC", "conf": "West", "off_rating": 119.5, "def_rating": 111.0},
    {"id": 1610612753, "name": "Magic", "abbr": "ORL", "conf": "East", "off_rating": 111.8, "def_rating": 111.6},
    {"id": 1610612755, "name": "76ers", "abbr": "PHI", "conf": "East", "off_rating": 115.9, "def_rating": 113.3},
    {"id": 1610612756, "name": "Suns", "abbr": "PHX", "conf": "West", "off_rating": 116.8, "def_rating": 114.4},
    {"id": 1610612757, "name": "Trail Blazers", "abbr": "POR", "conf": "West", "off_rating": 110.8, "def_rating": 117.0},
    {"id": 1610612758, "name": "Kings", "abbr": "SAC", "conf": "West", "off_rating": 116.0, "def_rating": 115.5},
    {"id": 1610612759, "name": "Spurs", "abbr": "SAS", "conf": "West", "off_rating": 112.7, "def_rating": 116.8},
    {"id": 1610612761, "name": "Raptors", "abbr": "TOR", "conf": "East", "off_rating": 112.5, "def_rating": 115.0},
    {"id": 1610612762, "name": "Jazz", "abbr": "UTA", "conf": "West", "off_rating": 113.0, "def_rating": 117.5},
    {"id": 1610612764, "name": "Wizards", "abbr": "WAS", "conf": "East", "off_rating": 109.8, "def_rating": 118.0},
]

TEAM_LOOKUP = {t["id"]: t for t in TEAMS}


def generate_season_schedule(season_label: str, start_date: datetime, n_games: int = 1230) -> pd.DataFrame:
    """
    Generate a realistic NBA schedule with home/away pairings.
    Each team plays ~82 games, ~41 at home.
    """
    team_ids = [t["id"] for t in TEAMS]
    rows = []
    game_id_counter = 1
    date = start_date

    # Track games per team to roughly hit 82
    games_per_team = {tid: 0 for tid in team_ids}
    target_games = 82
    days_active = 0

    while all(games_per_team[tid] < target_games for tid in team_ids) and days_active < 200:
        if date.weekday() < 6:  # Skip Sundays (rest days)
            # Generate matchups for this day
            available = [tid for tid in team_ids if games_per_team[tid] < target_games]
            random.shuffle(available)

            # Pair up teams (make even number)
            available = available[:len(available) - (len(available) % 2)]

            for i in range(0, len(available) - 1, 2):
                home_id = available[i]
                away_id = available[i + 1]

                if games_per_team[home_id] >= target_games or games_per_team[away_id] >= target_games:
                    continue

                game_id = f"{season_label[:4]}0{game_id_counter:05d}"
                game_id_counter += 1
                date_str = date.strftime("%Y-%m-%d")

                home = TEAM_LOOKUP[home_id]
                away = TEAM_LOOKUP[away_id]

                # Generate realistic box score
                home_pts = int(np.random.normal(home["off_rating"] - away["def_rating"] + 105, 10))
                away_pts = int(np.random.normal(away["off_rating"] - home["def_rating"] + 105, 10))

                # Ensure realistic scores
                home_pts = max(70, min(140, home_pts))
                away_pts = max(70, min(140, away_pts))

                home_wl = "W" if home_pts > away_pts else "L"
                away_wl = "W" if away_pts > home_pts else "L"

                # Generate per-game stats (shared helper)
                def gen_team_stats(pts, opponent_pts):
                    fgm = int(np.random.normal(pts * 0.36, 3))
                    fga = int(fgm + np.random.normal(30, 5))
                    fg_pct = fgm / max(fga, 1)
                    fg3m = int(np.random.normal(pts * 0.11, 2))
                    fg3a = int(fg3m + np.random.normal(15, 4))
                    fg3_pct = fg3m / max(fg3a, 1)
                    ftm = pts - (fgm * 2 + fg3m)
                    fta = int(ftm + np.random.normal(5, 2))
                    ft_pct = ftm / max(fta, 1)
                    oreb = int(np.random.normal(10, 3))
                    dreb = int(np.random.normal(30, 4))
                    reb = oreb + dreb
                    ast = int(np.random.normal(24, 4))
                    stl = int(np.random.normal(7, 2))
                    blk = int(np.random.normal(5, 2))
                    tov = int(np.random.normal(13, 3))
                    pf = int(np.random.normal(20, 3))
                    plus_minus = pts - opponent_pts
                    return fgm, fga, fg_pct, fg3m, fg3a, fg3_pct, ftm, fta, ft_pct, oreb, dreb, reb, ast, stl, blk, tov, pf, plus_minus

                # Home team stats
                h_stats = gen_team_stats(home_pts, away_pts)
                # Away team stats
                a_stats = gen_team_stats(away_pts, home_pts)

                # Home team row
                rows.append({
                    "SEASON_ID": int(season_label.replace("-", "")),
                    "TEAM_ID": home_id,
                    "TEAM_ABBREVIATION": home["abbr"],
                    "TEAM_NAME": home["name"],
                    "GAME_ID": game_id,
                    "GAME_DATE": date_str,
                    "MATCHUP": f"{home['name']} vs. {away['name']}",
                    "WL": home_wl,
                    "MIN": 240,
                    "PTS": home_pts,
                    "FGM": h_stats[0], "FGA": h_stats[1], "FG_PCT": round(h_stats[2], 3),
                    "FG3M": h_stats[3], "FG3A": h_stats[4], "FG3_PCT": round(h_stats[5], 3),
                    "FTM": h_stats[6], "FTA": h_stats[7], "FT_PCT": round(h_stats[8], 3),
                    "OREB": h_stats[9], "DREB": h_stats[10], "REB": h_stats[11],
                    "AST": h_stats[12], "STL": h_stats[13], "BLK": h_stats[14],
                    "TOV": h_stats[15], "PF": h_stats[16], "PLUS_MINUS": home_pts - away_pts,
                    "SEASON": f"{season_label}",
                })

                # Away team row
                rows.append({
                    "SEASON_ID": int(season_label.replace("-", "")),
                    "TEAM_ID": away_id,
                    "TEAM_ABBREVIATION": away["abbr"],
                    "TEAM_NAME": away["name"],
                    "GAME_ID": game_id,
                    "GAME_DATE": date_str,
                    "MATCHUP": f"{away['name']} @ {home['name']}",
                    "WL": away_wl,
                    "MIN": 240,
                    "PTS": away_pts,
                    "FGM": a_stats[0], "FGA": a_stats[1], "FG_PCT": round(a_stats[2], 3),
                    "FG3M": a_stats[3], "FG3A": a_stats[4], "FG3_PCT": round(a_stats[5], 3),
                    "FTM": a_stats[6], "FTA": a_stats[7], "FT_PCT": round(a_stats[8], 3),
                    "OREB": a_stats[9], "DREB": a_stats[10], "REB": a_stats[11],
                    "AST": a_stats[12], "STL": a_stats[13], "BLK": a_stats[14],
                    "TOV": a_stats[15], "PF": a_stats[16], "PLUS_MINUS": away_pts - home_pts,
                    "SEASON": f"{season_label}",
                })

                games_per_team[home_id] += 1
                games_per_team[away_id] += 1

        date += timedelta(days=1)
        days_active += 1

    df = pd.DataFrame(rows)
    return df


def generate_all_seasons():
    """Generate 3 seasons of NBA data."""
    seasons = [
        ("2022-23", datetime(2022, 10, 18)),
        ("2023-24", datetime(2023, 10, 24)),
        ("2024-25", datetime(2024, 10, 22)),
    ]

    all_dfs = []
    total_games = 0

    for season_label, start_date in seasons:
        print(f"  Generating {season_label} schedule...")
        df = generate_season_schedule(season_label, start_date)
        all_dfs.append(df)
        n_unique = df["GAME_ID"].nunique()
        total_games += n_unique
        print(f"    {len(df)} rows ({n_unique} games)")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df = full_df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"  Generated: {len(full_df)} game log entries")
    print(f"  Seasons: {', '.join(s[0] for s in seasons)}")
    print(f"  Teams: {full_df['TEAM_NAME'].nunique()}")
    print(f"  Games: {full_df['GAME_ID'].nunique()}")
    print(f"  Date range: {full_df['GAME_DATE'].min()} to {full_df['GAME_DATE'].max()}")
    print(f"  Total points range: {full_df.groupby('GAME_ID')['PTS'].sum().min():.0f}-{full_df.groupby('GAME_ID')['PTS'].sum().max():.0f}")
    print(f"{'='*60}")

    return full_df


def populate_database(df: pd.DataFrame):
    """Write to SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove old database
    if DB_PATH.exists():
        DB_PATH.unlink()

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
    print(f"  [OK] Database: {DB_PATH} ({DB_PATH.stat().st_size / 1024:.0f} KB)")


def show_sample_games(df: pd.DataFrame):
    """Show some sample games."""
    sample = df[df["IS_HOME"] == 1].head(5) if "IS_HOME" in df.columns else df.drop_duplicates("GAME_ID").head(5)
    print(f"\n  Sample games:")
    for _, row in sample.iterrows():
        home_pts = row.get("PTS", "?")
        home_name = row.get("TEAM_NAME", "?")
        matchup = row.get("MATCHUP", "?")
        date = row.get("GAME_DATE", "?")
        print(f"    {date} | {matchup:35s} | {home_pts} pts")


if __name__ == "__main__":
    print("=" * 60)
    print("  SYNTHETIC NBA DATA GENERATOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n[+] Generating realistic synthetic NBA data...")
    df = generate_all_seasons()

    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    populate_database(df)
    show_sample_games(df)

    print(f"\n{'='*60}")
    print("  [OK] SYNTHETIC DATA GENERATED")
    print("  The prediction engine can now run with realistic NBA data!")
    print("  Note: This is synthetic data. Replace with real nba_api data")
    print("        when stats.nba.com is reachable from this network.")
    print(f"{'='*60}")
