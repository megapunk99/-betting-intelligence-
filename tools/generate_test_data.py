#!/usr/bin/env python3
"""
Generate synthetic NBA game log data in SQLite for backtesting.

Creates ~800 NBA games with realistic box scores across 3 seasons,
so the full feature engineering pipeline and backtest can run.

Usage:
    python tools/generate_test_data.py
"""

import logging
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_test_data")

# ── NBA Teams ──────────────────────────────────────────────────────────
NBA_TEAMS = [
    ("ATL", "Atlanta Hawks"), ("BOS", "Boston Celtics"), ("BKN", "Brooklyn Nets"),
    ("CHA", "Charlotte Hornets"), ("CHI", "Chicago Bulls"), ("CLE", "Cleveland Cavaliers"),
    ("DAL", "Dallas Mavericks"), ("DEN", "Denver Nuggets"), ("DET", "Detroit Pistons"),
    ("GSW", "Golden State Warriors"), ("HOU", "Houston Rockets"), ("IND", "Indiana Pacers"),
    ("LAC", "LA Clippers"), ("LAL", "Los Angeles Lakers"), ("MEM", "Memphis Grizzlies"),
    ("MIA", "Miami Heat"), ("MIL", "Milwaukee Bucks"), ("MIN", "Minnesota Timberwolves"),
    ("NOP", "New Orleans Pelicans"), ("NYK", "New York Knicks"), ("OKC", "Oklahoma City Thunder"),
    ("ORL", "Orlando Magic"), ("PHI", "Philadelphia 76ers"), ("PHX", "Phoenix Suns"),
    ("POR", "Portland Trail Blazers"), ("SAC", "Sacramento Kings"), ("SAS", "San Antonio Spurs"),
    ("TOR", "Toronto Raptors"), ("UTA", "Utah Jazz"), ("WAS", "Washington Wizards"),
]

# Team quality tiers for realistic differentials
TIER1 = {"Celtics", "Nuggets", "Bucks", "Thunder", "Timberwolves"}  # Elite
TIER2 = {"Knicks", "76ers", "Mavericks", "Lakers", "Cavaliers", "Suns", "Clippers", "Heat"}
TIER3 = {"Pelicans", "Magic", "Pacers", "Kings", "Warriors", "Rockets", "Bulls", "Hawks"}
TIER4 = {"Raptors", "Jazz", "Grizzlies", "Spurs", "Nets", "Hornets", "Trail Blazers", "Wizards", "Pistons"}

# Team IDs based on abbreviation hash
TEAM_IDS = {abbr: abs(hash(f"NBA_{abbr}")) % (2**31) for abbr, _ in NBA_TEAMS}


def _team_tier(name: str) -> int:
    if name in TIER1: return 0
    if name in TIER2: return 1
    if name in TIER3: return 2
    return 3


def _gen_team_stats(rng: np.random.Generator, base_pts: float, base_pace: float,
                     opponent_tier: int, is_home: bool) -> dict:
    """Generate realistic box score stats for one team."""
    home_boost = 2.5 if is_home else 0.0
    tier_boost = (3 - opponent_tier) * 1.5  # better teams score more
    pts = base_pts + home_boost + tier_boost + rng.normal(0, 8)

    fga = base_pace * 0.85 + rng.normal(0, 4)
    fg_pct = np.clip(0.45 + rng.normal(0, 0.04) + tier_boost * 0.005, 0.35, 0.60)
    fgm = fga * fg_pct
    fg3a = fga * np.clip(0.38 + rng.normal(0, 0.04), 0.25, 0.50)
    fg3_pct = np.clip(0.35 + rng.normal(0, 0.035), 0.20, 0.50)
    fg3m = fg3a * fg3_pct

    # Two-pointers (FGA - 3PA) * 2P%
    fg2a = fga - fg3a
    fg2_pct = np.clip(fg_pct + 0.07 + rng.normal(0, 0.03), 0.40, 0.65)
    fg2m = fg2a * fg2_pct

    # Free throws: pts should ≈ 2*fg2m + 3*fg3m + ftm
    expected_ft_points = pts - (2*fg2m + 3*fg3m)
    ft_attempts = max(0, expected_ft_points / 0.75 + rng.normal(0, 3))
    ft_pct = np.clip(0.77 + rng.normal(0, 0.03), 0.65, 0.90)
    ftm = max(0, ft_attempts * ft_pct)

    # Rebounding
    team_tot_reb = 38 + rng.normal(0, 5)
    oreb_pct = np.clip(0.25 + rng.normal(0, 0.03), 0.15, 0.35)
    oreb = team_tot_reb * oreb_pct
    dreb = team_tot_reb - oreb

    # Other stats
    ast = (fgm * np.clip(0.60 + rng.normal(0, 0.04), 0.40, 0.80))
    stl = 7 + rng.normal(0, 2)
    blk = 5 + rng.normal(0, 1.5)
    tov = 13 + rng.normal(0, 2.5)
    pf = 19 + rng.normal(0, 3)
    min_played = 240 + rng.normal(0, 2)  # 48 min * 5 players (NBA OT)

    return {
        "PTS": max(60, round(pts)),
        "FGM": max(20, round(fgm)),
        "FGA": max(50, round(fga)),
        "FG_PCT": round(fgm / max(fga, 1), 3),
        "FG3M": max(5, round(fg3m)),
        "FG3A": max(15, round(fg3a)),
        "FG3_PCT": round(fg3m / max(fg3a, 1), 3),
        "FTM": max(5, round(ftm)),
        "FTA": max(8, round(ft_attempts)),
        "FT_PCT": round(ftm / max(ft_attempts, 1), 3),
        "OREB": max(5, round(oreb)),
        "DREB": max(15, round(dreb)),
        "REB": max(20, round(team_tot_reb)),
        "AST": max(10, round(ast)),
        "STL": max(2, round(stl)),
        "BLK": max(1, round(blk)),
        "TOV": max(5, round(tov)),
        "PF": max(10, round(pf)),
        "MIN": max(240, round(min_played)),
    }


def generate_game_logs(n_games: int = 800, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic NBA game logs."""
    rng = np.random.default_rng(seed)
    rows = []

    # Season structure: ~82 games per team, ~1230 total games per season
    # We'll spread across 3 seasons
    seasons = [2024, 2025, 2026]
    team_pool = [name for _, name in NBA_TEAMS]
    team_abbrs = [abbr for abbr, _ in NBA_TEAMS]

    # Base date per season (Oct 15 of season-1 year)
    base_dates = [
        datetime(2024, 10, 15),
        datetime(2025, 10, 15),
        datetime(2024, 10, 15),  # Reuse dates for 2026
    ]

    # Track last game date per team for realistic scheduling
    last_game = {name: None for _, name in NBA_TEAMS}

    for game_id in range(1, n_games + 1):
        # Pick season
        season_idx = min(game_id // 270, 2)
        season = seasons[season_idx]
        base_date = base_dates[season_idx]

        # Pick two random teams
        home_idx = rng.integers(0, len(team_pool))
        away_idx = rng.integers(0, len(team_pool))
        while away_idx == home_idx:
            away_idx = rng.integers(0, len(team_pool))

        home_name = team_pool[home_idx]
        away_name = team_pool[away_idx]
        home_abbr = team_abbrs[home_idx]
        away_abbr = team_abbrs[away_idx]

        # Generate game date (in-season: Oct-Apr)
        day_offset = rng.integers(0, 170)  # ~5.6 months of season
        game_date = base_date + timedelta(days=int(day_offset))

        # Ensure at least 2 days gap between consecutive games for each team
        if last_game[home_name] is not None and (game_date - last_game[home_name]).days < 2:
            game_date = last_game[home_name] + timedelta(days=2)
        if last_game[away_name] is not None and (game_date - last_game[away_name]).days < 2:
            game_date = max(game_date, last_game[away_name] + timedelta(days=2))

        last_game[home_name] = game_date
        last_game[away_name] = game_date

        # Base pace and scoring (league avg ~100 possessions, ~228 total points)
        home_pace = 100 + rng.normal(0, 3)
        away_pace = 100 + rng.normal(0, 3)
        base_pts_home = 114 + rng.normal(0, 3)
        base_pts_away = 114 + rng.normal(0, 3)

        # Generate stats
        home_tier = _team_tier(home_name)
        away_tier = _team_tier(away_name)
        home_stats = _gen_team_stats(rng, base_pts_home, home_pace, away_tier, True)
        away_stats = _gen_team_stats(rng, base_pts_away, away_pace, home_tier, False)

        home_pts = home_stats["PTS"]
        away_pts = away_stats["PTS"]

        home_wl = "W" if home_pts > away_pts else "L"
        home_plus_minus = home_pts - away_pts
        away_plus_minus = away_pts - home_pts

        game_date_str = game_date.strftime("%Y-%m-%d")

        # Home team row
        gid = f"SYNTH_{game_id:05d}"
        rows.append({
            "SEASON_ID": season,
            "TEAM_ID": TEAM_IDS[home_abbr],
            "TEAM_ABBREVIATION": home_abbr,
            "TEAM_NAME": home_name,
            "GAME_ID": gid,
            "GAME_DATE": game_date_str,
            "MATCHUP": f"{home_abbr} vs. {away_abbr}",
            "WL": home_wl,
            **home_stats,
            "PLUS_MINUS": home_plus_minus,
            "SEASON": season,
        })

        # Away team row
        rows.append({
            "SEASON_ID": season,
            "TEAM_ID": TEAM_IDS[away_abbr],
            "TEAM_ABBREVIATION": away_abbr,
            "TEAM_NAME": away_name,
            "GAME_ID": gid,
            "GAME_DATE": game_date_str,
            "MATCHUP": f"{away_abbr} @ {home_abbr}",
            "WL": "W" if away_pts > home_pts else "L",
            **away_stats,
            "PLUS_MINUS": away_plus_minus,
            "SEASON": season,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    logger.info(f"Generated {len(df)} rows ({len(df)//2} games)")
    return df


def populate_sqlite(df: pd.DataFrame, db_path: str = "data/nba_data.db"):
    """Write game_logs table to SQLite."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create game_logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_logs (
            SEASON_ID INTEGER,
            TEAM_ID INTEGER,
            TEAM_ABBREVIATION TEXT,
            TEAM_NAME TEXT,
            GAME_ID TEXT,
            GAME_DATE TEXT,
            MATCHUP TEXT,
            WL TEXT,
            MIN INTEGER,
            PTS INTEGER,
            FGM INTEGER,
            FGA INTEGER,
            FG_PCT REAL,
            FG3M INTEGER,
            FG3A INTEGER,
            FG3_PCT REAL,
            FTM INTEGER,
            FTA INTEGER,
            FT_PCT REAL,
            OREB INTEGER,
            DREB INTEGER,
            REB INTEGER,
            AST INTEGER,
            STL INTEGER,
            BLK INTEGER,
            TOV INTEGER,
            PF INTEGER,
            PLUS_MINUS INTEGER,
            SEASON INTEGER
        )
    """)

    # Clear existing data
    cursor.execute("DELETE FROM game_logs")

    # Insert data
    df.to_sql("game_logs", conn, if_exists="append", index=False)
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM game_logs")
    count = cursor.fetchone()[0]
    conn.close()
    logger.info(f"Wrote {count} rows to {db_path}")


if __name__ == "__main__":
    logger.info("Generating synthetic NBA game logs...")
    df = generate_game_logs(n_games=800, seed=42)
    populate_sqlite(df)
    logger.info("Done!")
