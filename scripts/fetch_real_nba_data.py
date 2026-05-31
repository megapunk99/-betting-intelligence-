"""
Real NBA data fetcher using ESPN API.
Fetches ALL completed NBA games from 2022-23 through 2025-26 seasons.
Two-phase approach:
  Phase 1: Fetch scoreboard data (fast, ~21 API calls) -> writes DB with basic stats
  Phase 2: Backfill detailed boxscore stats (parallel, many calls)

Usage:
    python scripts/fetch_real_nba_data.py              # fetch all 4 seasons
    python scripts/fetch_real_nba_data.py --season 2024  # single season
    python scripts/fetch_real_nba_data.py --fast          # scoreboard only (no boxscore backfill)
    python scripts/fetch_real_nba_data.py --backfill      # only backfill boxscore data (from existing cache/games)
"""

import sys
import os
from pathlib import Path

import sqlite3
import requests
import json
import time
import hashlib
import random
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Add src/ to path so we can import from betting_intel.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from betting_intel.config import DB_PATH

# ─── CONFIG ──────────────────────────────────────────────────────────────
SEASONS = {
    2022: "2022-23",
    2023: "2023-24",
    2024: "2024-25",
    2025: "2025-26",
}
MONTHS = [10, 11, 12, 1, 2, 3, 4]  # NBA regular season months
CACHE_DIR = Path("cache/espn_api")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.espn.com/nba/scoreboard",
}

# ─── TEAM NAME MAPPING ──────────────────────────────────────────────────
ESPN_TEAM_MAP = {
    "Atlanta Hawks": {"name": "Hawks", "abbr": "ATL", "id": 1610612737},
    "Boston Celtics": {"name": "Celtics", "abbr": "BOS", "id": 1610612738},
    "Brooklyn Nets": {"name": "Nets", "abbr": "BKN", "id": 1610612751},
    "Charlotte Hornets": {"name": "Hornets", "abbr": "CHA", "id": 1610612766},
    "Chicago Bulls": {"name": "Bulls", "abbr": "CHI", "id": 1610612741},
    "Cleveland Cavaliers": {"name": "Cavaliers", "abbr": "CLE", "id": 1610612739},
    "Dallas Mavericks": {"name": "Mavericks", "abbr": "DAL", "id": 1610612742},
    "Denver Nuggets": {"name": "Nuggets", "abbr": "DEN", "id": 1610612743},
    "Detroit Pistons": {"name": "Pistons", "abbr": "DET", "id": 1610612765},
    "Golden State Warriors": {"name": "Warriors", "abbr": "GSW", "id": 1610612744},
    "Houston Rockets": {"name": "Rockets", "abbr": "HOU", "id": 1610612745},
    "Indiana Pacers": {"name": "Pacers", "abbr": "IND", "id": 1610612754},
    "LA Clippers": {"name": "Clippers", "abbr": "LAC", "id": 1610612746},
    "Los Angeles Clippers": {"name": "Clippers", "abbr": "LAC", "id": 1610612746},
    "Los Angeles Lakers": {"name": "Lakers", "abbr": "LAL", "id": 1610612747},
    "Memphis Grizzlies": {"name": "Grizzlies", "abbr": "MEM", "id": 1610612763},
    "Miami Heat": {"name": "Heat", "abbr": "MIA", "id": 1610612748},
    "Milwaukee Bucks": {"name": "Bucks", "abbr": "MIL", "id": 1610612749},
    "Minnesota Timberwolves": {"name": "Timberwolves", "abbr": "MIN", "id": 1610612750},
    "New Orleans Pelicans": {"name": "Pelicans", "abbr": "NOP", "id": 1610612740},
    "New York Knicks": {"name": "Knicks", "abbr": "NYK", "id": 1610612752},
    "Oklahoma City Thunder": {"name": "Thunder", "abbr": "OKC", "id": 1610612760},
    "Orlando Magic": {"name": "Magic", "abbr": "ORL", "id": 1610612753},
    "Philadelphia 76ers": {"name": "76ers", "abbr": "PHI", "id": 1610612755},
    "Phoenix Suns": {"name": "Suns", "abbr": "PHX", "id": 1610612756},
    "Portland Trail Blazers": {"name": "Trail Blazers", "abbr": "POR", "id": 1610612757},
    "Sacramento Kings": {"name": "Kings", "abbr": "SAC", "id": 1610612758},
    "San Antonio Spurs": {"name": "Spurs", "abbr": "SAS", "id": 1610612759},
    "Toronto Raptors": {"name": "Raptors", "abbr": "TOR", "id": 1610612761},
    "Utah Jazz": {"name": "Jazz", "abbr": "UTA", "id": 1610612762},
    "Washington Wizards": {"name": "Wizards", "abbr": "WAS", "id": 1610612764},
}

ABBR_TO_ESPN = {v["abbr"]: k for k, v in ESPN_TEAM_MAP.items()}
SHORT_TO_ESPN = {v["name"]: k for k, v in ESPN_TEAM_MAP.items()}


def cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def get_cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def fetch_json(url: str, use_cache: bool = True, max_retries: int = 2) -> Optional[dict]:
    """Fetch JSON from URL with caching and retries."""
    key = cache_key(url)
    cache_path = get_cache_path(key)

    if use_cache and cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)

    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if use_cache:
                    with open(cache_path, "w") as f:
                        json.dump(data, f)
                return data
            elif r.status_code == 429 and attempt < max_retries:
                wait_time = 2 ** attempt + random.random()
                time.sleep(wait_time)
                continue
            else:
                return None
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Fetch scoreboard data (fast - gets basic stats for ALL games)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_month_scoreboard(season_year: int, month: int) -> Tuple[int, List[dict]]:
    """Fetch scoreboard data for a month (with pagination), return (completed_count, games)."""
    if month >= 10:
        date_str = f"{season_year}{month:02d}"
    else:
        date_str = f"{season_year + 1}{month:02d}"

    all_completed = []
    page = 1

    while True:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}&page={page}&limit=100"
        data = fetch_json(url)

        if not data:
            break

        events = data.get("events", [])
        if not events:
            break

        for event in events:
            comps = event.get("competitions", [])
            if comps:
                status = comps[0].get("status", {}).get("type", {})
                if status.get("completed", False):
                    all_completed.append({
                        "event": event,
                        "competition": comps[0],
                    })

        # Check if there are more pages
        page_count = data.get("pageCount", 1)
        if page >= page_count:
            break
        page += 1

    return len(all_completed), all_completed


def parse_scoreboard_stats(competition: dict) -> Tuple[Optional[dict], Optional[dict]]:
    """Parse team stats from scoreboard data (basic stats only)."""
    competitors = competition.get("competitors", [])
    home_data, away_data = None, None

    for team in competitors:
        is_home = team.get("homeAway") == "home"
        team_info = team.get("team", {})
        espn_name = team_info.get("displayName", "")
        mapped = ESPN_TEAM_MAP.get(espn_name, {})

        stats = {
            "team_id": mapped.get("id", 0),
            "team_name": mapped.get("name", espn_name),
            "team_abbreviation": mapped.get("abbr", ""),
            "points": int(team.get("score", 0)),
            "home_away": "home" if is_home else "away",
            "fgm": 0, "fga": 0, "fg_pct": 0.0,
            "fg3m": 0, "fg3a": 0, "fg3_pct": 0.0,
            "ftm": 0, "fta": 0, "ft_pct": 0.0,
            "reb": 0, "ast": 0, "oreb": 0, "dreb": 0,
            "stl": 0, "blk": 0, "tov": 0, "pf": 0,
        }

        for s in team.get("statistics", []):
            name = s.get("name", "")
            val = s.get("displayValue", "0")
            try:
                if name == "fieldGoalsMade":
                    stats["fgm"] = int(val)
                elif name == "fieldGoalsAttempted":
                    stats["fga"] = int(val)
                elif name == "threePointFieldGoalsMade":
                    stats["fg3m"] = int(val)
                elif name == "threePointFieldGoalsAttempted":
                    stats["fg3a"] = int(val)
                elif name == "freeThrowsMade":
                    stats["ftm"] = int(val)
                elif name == "freeThrowsAttempted":
                    stats["fta"] = int(val)
                elif name == "rebounds":
                    stats["reb"] = int(val)
                elif name == "assists":
                    stats["ast"] = int(val)
            except (ValueError, TypeError):
                pass

        # Compute percentages
        if stats["fga"] > 0:
            stats["fg_pct"] = round(stats["fgm"] / stats["fga"], 3)
        if stats["fg3a"] > 0:
            stats["fg3_pct"] = round(stats["fg3m"] / stats["fg3a"], 3)
        if stats["fta"] > 0:
            stats["ft_pct"] = round(stats["ftm"] / stats["fta"], 3)

        if is_home:
            home_data = stats
        else:
            away_data = stats

    return home_data, away_data


def scoreboard_row_to_db_rows(event_id: str, competition: dict, season_label: str) -> List[dict]:
    """Convert scoreboard competition data to DB rows (no boxscore needed)."""
    home, away = parse_scoreboard_stats(competition)
    if not home or not away:
        return []

    date_str = competition.get("date", competition.get("startDate", ""))
    try:
        game_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except:
        game_date = datetime.now()

    home_pts = home["points"]
    away_pts = away["points"]
    home_wl = "W" if home_pts > away_pts else "L"
    away_wl = "W" if away_pts > home_pts else "L"

    home_name = home["team_name"]
    away_name = away["team_name"]

    season_id = int(season_label.replace("-", "")) if "-" in season_label else int(season_label) * 10000 + (int(season_label) + 1)

    def make_row(td, is_home):
        pts = td["points"]
        opp_pts = away_pts if is_home else home_pts
        return {
            "SEASON_ID": season_id,
            "TEAM_ID": td["team_id"],
            "TEAM_ABBREVIATION": td["team_abbreviation"],
            "TEAM_NAME": td["team_name"],
            "GAME_ID": f"ESPN{event_id}",
            "GAME_DATE": game_date.strftime("%Y-%m-%d"),
            "MATCHUP": f"{home_name} vs. {away_name}" if is_home else f"{away_name} @ {home_name}",
            "WL": home_wl if is_home else away_wl,
            "MIN": 240,
            "PTS": pts,
            "FGM": td["fgm"],
            "FGA": td["fga"],
            "FG_PCT": td["fg_pct"],
            "FG3M": td["fg3m"],
            "FG3A": td["fg3a"],
            "FG3_PCT": td["fg3_pct"],
            "FTM": td["ftm"],
            "FTA": td["fta"],
            "FT_PCT": td["ft_pct"],
            "OREB": td.get("oreb", 0),
            "DREB": td.get("dreb", 0),
            "REB": td["reb"],
            "AST": td["ast"],
            "STL": td.get("stl", 0),
            "BLK": td.get("blk", 0),
            "TOV": td.get("tov", 0),
            "PF": td.get("pf", 0),
            "PLUS_MINUS": pts - opp_pts,
            "SEASON": season_label,
        }

    return [
        make_row(home, is_home=True),
        make_row(away, is_home=False),
    ]


def phase1_fetch_scoreboards(seasons_to_fetch: List[int]) -> int:
    """Phase 1: Fetch all scoreboard data and write to DB. Returns game count."""
    print("\n" + "="*60)
    print("  PHASE 1: Fetching scoreboard data (basic stats)")
    print("  Only ~28 API calls needed for all 4 seasons!")
    print("="*60)

    all_rows = []
    total_completed = 0

    for season_year in seasons_to_fetch:
        season_label = SEASONS[season_year]
        print(f"\n  Season: {season_label}")
        for month in MONTHS:
            count, games = fetch_month_scoreboard(season_year, month)
            if count == 0:
                continue

            total_completed += count
            print(f"    Month {month:2d}: {count} completed games", end="")

            for game_data in games:
                event = game_data["event"]
                competition = game_data["competition"]
                event_id = event.get("id", "")
                rows = scoreboard_row_to_db_rows(event_id, competition, season_label)
                all_rows.extend(rows)

            print(f"  ({len(all_rows)} rows so far)")

    # Deduplicate
    seen = set()
    unique = []
    for row in all_rows:
        key = f"{row['GAME_ID']}_{row['TEAM_ID']}"
        if key not in seen:
            seen.add(key)
            unique.append(row)

    print(f"\n  Total unique rows from scoreboard: {len(unique)} ({len(unique)//2} games)")

    # Write to DB
    if unique:
        import pandas as pd
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        if DB_PATH.exists():
            DB_PATH.unlink()

        df = pd.DataFrame(unique)
        df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

        conn = sqlite3.connect(str(DB_PATH))
        df.to_sql("game_logs", conn, if_exists="replace", index=False)
        _create_indexes(conn)
        count = conn.execute("SELECT COUNT(*) FROM game_logs").fetchone()[0]
        conn.close()
        print(f"\n  [OK] Phase 1 complete: {count} rows in database ({count//2} games)")
        print(f"  [OK] Basic stats (PTS, FGM, FGA, FG3M, FG3A, FTM, FTA, REB, AST) from scoreboard")
        print(f"  [OK] Advanced stats (OREB, DREB, STL, BLK, TOV, PF) need boxscore backfill")
        print(f"  [OK] DB size: {DB_PATH.stat().st_size / 1024:.0f} KB")

    return len(unique) // 2


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Backfill boxscore details (parallel, many API calls)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_boxscore(event_id: str) -> Optional[dict]:
    """Fetch boxscore data for a single event via the summary endpoint."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
    return fetch_json(url)


def parse_boxscore_stats(boxscore_data: dict) -> Optional[Dict[str, dict]]:
    """Parse boxscore data into per-team stat dicts. Returns {home_team, away_team -> stats}."""
    boxscore = boxscore_data.get("boxscore")
    if not boxscore:
        return None

    teams = boxscore.get("teams", [])
    if len(teams) < 2:
        return None

    result = {}
    for team_data in teams:
        team_info = team_data.get("team", {})
        espn_name = team_info.get("displayName", "")
        mapped = ESPN_TEAM_MAP.get(espn_name, {})
        home_away = team_data.get("homeAway", "")

        stats = {}
        for stat_entry in team_data.get("statistics", []):
            name = stat_entry.get("name", "")
            display = stat_entry.get("displayValue", "0")

            try:
                if "fieldGoalsMade-fieldGoalsAttempted" == name and "-" in display:
                    parts = display.split("-")
                    stats["fgm"] = int(parts[0])
                    stats["fga"] = int(parts[1])
                elif "threePointFieldGoalsMade-threePointFieldGoalsAttempted" == name and "-" in display:
                    parts = display.split("-")
                    stats["fg3m"] = int(parts[0])
                    stats["fg3a"] = int(parts[1])
                elif "freeThrowsMade-freeThrowsAttempted" == name and "-" in display:
                    parts = display.split("-")
                    stats["ftm"] = int(parts[0])
                    stats["fta"] = int(parts[1])
                elif name == "offensiveRebounds":
                    stats["oreb"] = int(display)
                elif name == "defensiveRebounds":
                    stats["dreb"] = int(display)
                elif name == "steals":
                    stats["stl"] = int(display)
                elif name == "blocks":
                    stats["blk"] = int(display)
                elif name == "turnovers":
                    stats["tov"] = int(display)
                elif name == "fouls":
                    stats["pf"] = int(display)
                elif name == "totalRebounds":
                    stats["reb"] = int(display)
                elif name == "assists":
                    stats["ast"] = int(display)
                elif name == "fastBreakPoints":
                    stats["fast_break_pts"] = int(display)
                elif name == "pointsInPaint":
                    stats["paint_pts"] = int(display)
            except (ValueError, TypeError):
                pass

        result[home_away] = stats

    return result


def phase2_backfill_boxscores(max_workers: int = 20, batch_size: int = 100):
    """Phase 2: Backfill detailed boxscore stats for games missing them."""
    print("\n" + "="*60)
    print("  PHASE 2: Backfilling boxscore stats (parallel)")
    print(f"  Workers: {max_workers}")
    print("="*60)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Check if OREB column has data (all zeros = not backfilled)
    sample = cursor.execute("SELECT OREB FROM game_logs LIMIT 1").fetchone()
    if sample and sample[0] != 0:
        count = cursor.execute("SELECT COUNT(*) FROM game_logs WHERE OREB > 0").fetchone()[0]
        total = cursor.execute("SELECT COUNT(*) FROM game_logs").fetchone()[0]
        if count == total:
            print("\n  [OK] Boxscore data already present. Skipping backfill.")
            conn.close()
            return

    # Get all unique GAME_IDs
    game_ids = [r[0] for r in cursor.execute("SELECT DISTINCT GAME_ID FROM game_logs ORDER BY GAME_DATE").fetchall()]
    conn.close()

    total_games = len(game_ids)
    print(f"\n  Games to backfill: {total_games}")

    done = 0
    errors = 0
    updated = 0

    # Use a single DB connection for all updates (much faster)
    backfill_conn = sqlite3.connect(str(DB_PATH))

    # Process in batches with ThreadPoolExecutor
    for batch_start in range(0, total_games, batch_size):
        batch = game_ids[batch_start:batch_start + batch_size]
        batch_end = min(batch_start + batch_size, total_games)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_boxscore, gid.replace("ESPN", "")): gid for gid in batch}

            for future in as_completed(futures):
                gid = futures[future]
                done += 1
                try:
                    box_data = future.result()
                    if box_data:
                        stats = parse_boxscore_stats(box_data)
                        if stats:
                            home_s = stats.get("home", {})
                            away_s = stats.get("away", {})

                            rows1 = _update_boxscore_stats_batch(backfill_conn, gid, home_s, is_home=True)
                            rows2 = _update_boxscore_stats_batch(backfill_conn, gid, away_s, is_home=False)
                            updated += (1 if rows1 > 0 else 0) + (1 if rows2 > 0 else 0)
                        else:
                            errors += 1
                    else:
                        errors += 1
                except Exception as e:
                    errors += 1

                if done % 200 == 0 or done == total_games:
                    pct = done / total_games * 100
                    print(f"    Progress: {done}/{total_games} ({pct:.0f}%) - {updated} rows updated, {errors} errors")

        # Commit batch and brief pause
        backfill_conn.commit()
        if batch_end < total_games:
            time.sleep(1)

    backfill_conn.close()

    print(f"\n  Phase 2 complete: {updated} rows updated across {done} games ({errors} errors)")

    # Verify
    conn = sqlite3.connect(str(DB_PATH))
    has_oreb = conn.execute("SELECT COUNT(*) FROM game_logs WHERE OREB > 0").fetchone()[0]
    has_stl = conn.execute("SELECT COUNT(*) FROM game_logs WHERE STL > 0").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM game_logs").fetchone()[0]
    conn.close()
    print(f"\n  Verification:")
    print(f"    Games with OREB: {has_oreb}/{total} ({has_oreb/total*100:.0f}%)")
    print(f"    Games with STL:  {has_stl}/{total} ({has_stl/total*100:.0f}%)")


def _update_boxscore_stats_batch(conn, game_id: str, stats: dict, is_home: bool):
    """Update a single row with boxscore stats using an existing connection."""
    if not stats:
        return 0

    set_parts = []
    params = []
    for col in ["oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf"]:
        if col in stats and stats[col] > 0:
            set_parts.append(f"{col.upper()} = ?")
            params.append(stats[col])

    for col in ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]:
        if col in stats and stats[col] > 0:
            set_parts.append(f"{col.upper()} = ?")
            params.append(stats[col])

    if not set_parts:
        return 0

    matchup_pattern = "%vs.%" if is_home else "%@%"
    params.extend([game_id, matchup_pattern])

    query = f"UPDATE game_logs SET {', '.join(set_parts)} WHERE GAME_ID = ? AND MATCHUP LIKE ?"
    try:
        cur = conn.execute(query, params)
        return cur.rowcount
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _create_indexes(conn):
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_game_id ON game_logs(GAME_ID)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_team_id ON game_logs(TEAM_ID)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_logs_game_date ON game_logs(GAME_DATE)")
    conn.commit()


def verify_database():
    """Print stats about the database."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    total = c.execute("SELECT COUNT(*) FROM game_logs").fetchone()[0]
    games = total // 2
    date_range = c.execute("SELECT MIN(GAME_DATE), MAX(GAME_DATE) FROM game_logs").fetchone()
    seasons = [r[0] for r in c.execute("SELECT DISTINCT SEASON FROM game_logs ORDER BY SEASON").fetchall()]
    teams = c.execute("SELECT COUNT(DISTINCT TEAM_NAME) FROM game_logs").fetchone()[0]

    # Per-season
    print(f"\n  {'='*55}")
    print(f"  {'Season':12s} {'Games':8s} {'Rows':8s}")
    print(f"  {'='*55}")
    for season in seasons:
        cnt = c.execute("SELECT COUNT(*) FROM game_logs WHERE SEASON = ?", (season,)).fetchone()[0]
        gms = cnt // 2
        print(f"  {season:12s} {gms:<8} {cnt:<8}")
    print(f"  {'='*55}")
    print(f"  {'TOTAL':12s} {games:<8} {total:<8}")
    print(f"  {teams} teams, {date_range[0]} to {date_range[1]}")

    # Stats completeness
    print(f"\n  Statistics availability:")
    for col in ["PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "REB", "AST"]:
        has = c.execute(f"SELECT COUNT(*) FROM game_logs WHERE {col} > 0").fetchone()[0]
        print(f"    {col:7s}: {has}/{total} ({has/total*100:.0f}%)")
    for col in ["OREB", "DREB", "STL", "BLK", "TOV", "PF"]:
        has = c.execute(f"SELECT COUNT(*) FROM game_logs WHERE {col} > 0").fetchone()[0]
        print(f"    {col:7s}: {has}/{total} ({has/total*100:.0f}%)")

    # Sample
    print(f"\n  Sample games:")
    for r in c.execute("SELECT GAME_DATE, TEAM_NAME, MATCHUP, PTS, FGM, FGA, REB, AST FROM game_logs LIMIT 4").fetchall():
        print(f"    {r[0]} | {r[1]:20s} | {r[2]:35s} | PTS={r[3]} FG={r[4]}/{r[5]} REB={r[6]} AST={r[7]}")

    conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch real NBA data from ESPN API")
    parser.add_argument("--season", type=int, choices=[2022, 2023, 2024, 2025],
                        help="Single season to fetch (default: all 4)")
    parser.add_argument("--fast", action="store_true",
                        help="Scoreboard only - skip boxscore backfill")
    parser.add_argument("--backfill", action="store_true",
                        help="Only run boxscore backfill (skip scoreboard)")
    parser.add_argument("--workers", type=int, default=25,
                        help="Parallel workers for boxscore (default: 25)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify database contents only")

    args = parser.parse_args()

    print("=" * 60)
    print("  REAL NBA DATA FETCHER (ESPN API)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  No synthetic data - every stat from real NBA games!")
    print("=" * 60)

    if args.verify:
        if DB_PATH.exists():
            verify_database()
        else:
            print("\n  [!] No database found. Run without --verify first.")
        return

    seasons_to_fetch = [args.season] if args.season else list(SEASONS.keys())

    if args.backfill:
        # Phase 2 only
        phase2_backfill_boxscores(max_workers=args.workers)
        verify_database()
    else:
        # Phase 1 (+ optionally Phase 2)
        game_count = phase1_fetch_scoreboards(seasons_to_fetch)
        if game_count > 0 and not args.fast:
            phase2_backfill_boxscores(max_workers=args.workers)

        if game_count > 0:
            verify_database()
            print(f"\n{'='*60}")
            print("  [OK] REAL NBA DATA SUCCESSFULLY FETCHED!")
            print(f"  Ready for prediction: python predict_tomorrow.py")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
