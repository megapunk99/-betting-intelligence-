"""
NBA & Basketball League Schedule Checker

Queries multiple sources (ESPN API, nba_api, season calendars) to find
which basketball leagues currently have active games scheduled for today
and tomorrow.

Usage:
    python scripts/check_schedule.py                     # Today's games
    python scripts/check_schedule.py --days 7             # Next 7 days
    python scripts/check_schedule.py --league wnba       # Just WNBA
    python scripts/check_schedule.py --json              # JSON output
"""

import sys
import os
import json
import warnings
from datetime import date, timedelta, datetime
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_src = _project_root / "src"
for p in (_project_root, _src):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ["LOG_LEVEL"] = "CRITICAL"
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.CRITICAL)

# Fix Windows encoding for emoji-safe output
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from betting_intel.data.basketball_leagues import (
    ALL_BASKETBALL_LEAGUES, PRIMARY_BASKETBALL_LEAGUES, LEAGUE_BY_KEY,
)
from betting_intel.data.espn_hoops import LEAGUE_TO_ESPN_PATH


def _months_overlap(months_str: str) -> bool:
    """Check if current date falls within a season's month range."""
    if not months_str:
        return False
    try:
        parts = months_str.lower().replace("\u2192", "|").replace("\u2013", "|").replace("-", "|").split("|")
        if len(parts) < 2:
            return False
        import calendar
        month_names = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
        month_names.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
        start_name = parts[0].strip().split()[-1]
        end_name = parts[1].strip().split()[0]
        start_m = month_names.get(start_name, 0)
        end_m = month_names.get(end_name, 0)
        if start_m == 0 or end_m == 0:
            return False
        curr = datetime.now().month
        if start_m <= end_m:
            return start_m <= curr <= end_m
        else:
            return curr >= start_m or curr <= end_m
    except Exception:
        return False


def check_espn_games(league_key: str, days: int = 1) -> list[dict]:
    """Query ESPN scoreboard API for upcoming/completed games."""
    if league_key not in LEAGUE_TO_ESPN_PATH:
        return []

    import requests
    path = LEAGUE_TO_ESPN_PATH[league_key]
    results = []

    for offset in range(days):
        check_date = (date.today() + timedelta(days=offset)).strftime("%Y%m%d")
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"
            resp = requests.get(url, params={"dates": check_date, "limit": 300},
                                timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            data = resp.json()
            events = data.get("events", [])
            for ev in events:
                comp = ev.get("competitions", [{}])[0]
                competitors = comp.get("competitors", [])
                home = away = ""
                home_score = away_score = None
                home_record = away_record = None
                for c in competitors:
                    team = c.get("team", {})
                    name = team.get("displayName", team.get("shortDisplayName", ""))
                    score = c.get("score")
                    record = c.get("records", [{}])[0].get("summary", "") if c.get("records") else ""
                    if c.get("homeAway") == "home":
                        home = name
                        home_score = int(score) if score else None
                        home_record = record
                    else:
                        away = name
                        away_score = int(score) if score else None
                        away_record = record
                status = comp.get("status", {}).get("type", {}).get("name", "")
                is_completed = "FINAL" in status.upper() if status else False
                results.append({
                    "league": league_key,
                    "date": ev.get("date", "")[:10] or (date.today() + timedelta(days=offset)).isoformat(),
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_record": home_record,
                    "away_record": away_record,
                    "status": "completed" if is_completed else ("scheduled" if "PRE" in status.upper() or "SCHEDULED" in status.upper() else "live" if status else "unknown"),
                    "source": "espn",
                })
        except Exception:
            pass

    return results


def check_nba_api_games(days: int = 1) -> list[dict]:
    """Query nba_api for NBA game schedule."""
    results = []
    try:
        from nba_api.stats.endpoints import scoreboardv2
        for offset in range(days):
            check_date = (date.today() + timedelta(days=offset)).strftime("%Y-%m-%d")
            try:
                sb = scoreboardv2.ScoreboardV2(game_date=check_date)
                games = sb.get_data_frames()[0]
                if len(games) == 0:
                    continue
                for _, g in games.iterrows():
                    home = g.get("HOME_TEAM_NAME", g.get("HOME_TEAM_ABBREVIATION", ""))
                    away = g.get("VISITOR_TEAM_NAME", g.get("VISITOR_TEAM_ABBREVIATION", ""))
                    home_score = g.get("HOME_TEAM_SCORE", g.get("HOME_TEAM_PTS", None))
                    away_score = g.get("VISITOR_TEAM_SCORE", g.get("VISITOR_TEAM_PTS", None))
                    try:
                        home_score = int(home_score) if home_score and home_score != 0 else None
                        away_score = int(away_score) if away_score and away_score != 0 else None
                    except (ValueError, TypeError):
                        home_score = away_score = None
                    results.append({
                        "league": "nba",
                        "date": check_date,
                        "home_team": home or "?",
                        "away_team": away or "?",
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_record": "",
                        "away_record": "",
                        "status": "completed" if home_score else "scheduled",
                        "source": "nba_api",
                    })
            except Exception:
                pass
    except ImportError:
        pass
    return results


def check_theoddsapi_sports() -> list[dict]:
    """Check which basketball sports are available on TheOddsAPI."""
    from betting_intel.pipeline.bootstrap import ODDS_API_KEY
    if not ODDS_API_KEY or ODDS_API_KEY in ("your-api-key-here", "", "REPLACE_ME_WITH_YOUR_ODDS_API_KEY"):
        return []
    results = []
    try:
        import requests
        url = "https://api.the-odds-api.com/v4/sports"
        resp = requests.get(url, params={"apiKey": ODDS_API_KEY}, timeout=15)
        if resp.status_code == 200:
            sports = resp.json()
            basketball_sports = [s for s in sports if "basketball" in s.get("key", "")]
            for s in basketball_sports:
                results.append({
                    "key": s.get("key", ""),
                    "title": s.get("title", ""),
                    "active": s.get("active", False),
                })
    except Exception:
        pass
    return results


def print_schedule_summary(all_games: dict[str, list[dict]], odds_sports: list[dict]):
    """Print a formatted summary of all scheduled games."""
    today_str = date.today().isoformat()
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()

    print()
    print("=" * 72)
    print("  [BB]  BASKETBALL SCHEDULE CHECKER")
    print("        %s" % datetime.now().strftime('%B %d, %Y at %H:%M'))
    print("=" * 72)

    # League status header
    print()
    print("  %-25s %-15s %-20s %6s" % ("League", "Status", "Season", "Games"))
    print("  " + "-" * 68)

    for league in PRIMARY_BASKETBALL_LEAGUES:
        games = all_games.get(league.key, [])
        in_season = _months_overlap(league.season_months)
        status = "ACTIVE" if in_season else "offseason"
        season_label = league.season_months or "?"
        n_games = len(games)
        print("  %-25s %-15s %-20s %4d" % (league.name[:24], status, season_label, n_games))

    # Also show minor leagues that are in season
    for league in ALL_BASKETBALL_LEAGUES:
        if league.key in [l.key for l in PRIMARY_BASKETBALL_LEAGUES]:
            continue
        if not league.season_months:
            continue
        in_season = _months_overlap(league.season_months)
        if not in_season:
            continue
        games = all_games.get(league.key, [])
        status = "ACTIVE" if in_season else ""
        print("  %-25s %-15s %-20s %4d" % (league.name[:24], status, league.season_months, len(games)))

    print()

    # Detailed game list per league
    for league in PRIMARY_BASKETBALL_LEAGUES:
        games = all_games.get(league.key, [])
        if not games:
            continue
        sep = "-- %s (%d games) " % (league.name, len(games))
        print("  %s" % (sep + "-" * max(0, 68 - len(sep))))
        for g in games:
            gdate = g.get("date", "")
            rel = "TODAY" if gdate == today_str else ("TOM" if gdate == tomorrow_str else gdate[-5:])
            icons = {"completed": "[x]", "scheduled": "[ ]", "live": "[!]", "unknown": "[?]"}
            status_icon = icons.get(g.get("status", ""), "[?]")
            score_str = ""
            hs = g.get("home_score")
            as_ = g.get("away_score")
            if hs is not None and as_ is not None:
                score_str = "  %d - %d" % (as_, hs)
            home_rec = g.get("home_record", "")
            away_rec = g.get("away_record", "")
            rec_str = ""
            if home_rec and away_rec:
                rec_str = "  (%s) @ (%s)" % (away_rec, home_rec)
            source = g.get("source", "")
            print("    %-5s %s  %-20s @ %-20s%s%s" % (
                rel, status_icon, g['away_team'][:20], g['home_team'][:20], score_str, rec_str))
        print()

    # TheOddsAPI availability
    if odds_sports:
        sep = "-- TheOddsAPI Sports (%d basketball) " % len(odds_sports)
        print("  %s" % (sep + "-" * max(0, 68 - len(sep))))
        for s in odds_sports:
            status = "[x]" if s.get("active") else "[ ]"
            print("    %s %-35s %s" % (status, s['key'], s['title']))
        print()

    # Season calendar summary
    sep = "-- Season Calendar "
    print("  %s" % (sep + "-" * max(0, 68 - len(sep))))
    for league in ALL_BASKETBALL_LEAGUES:
        if not league.season_months:
            continue
        in_season = _months_overlap(league.season_months)
        icon = "[x]" if in_season else "[ ]"
        has_odds = "[x]" if league.odds_sport_key else "[ ]"
        has_espn = "[x]" if league.has_espn_api else "[ ]"
        print("    %s %-20s %s odds %s espn  %-22s %s" % (
            icon, league.key, has_odds, has_espn, league.season_months, league.name[:35]))
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check basketball league schedules")
    parser.add_argument("--days", type=int, default=1, help="Days ahead to check (default: 1)")
    parser.add_argument("--league", type=str, default=None, help="Check specific league only")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Determine which leagues to check
    if args.league:
        leagues_to_check = [args.league]
    else:
        leagues_to_check = [lg.key for lg in ALL_BASKETBALL_LEAGUES if lg.has_espn_api]

    # Query ESPN API for each league
    all_games = {}
    for lg_key in leagues_to_check:
        espn_games = check_espn_games(lg_key, args.days)
        all_games.setdefault(lg_key, []).extend(espn_games)

    # Always check nba_api for NBA
    nba_games = check_nba_api_games(args.days)
    all_games.setdefault("nba", []).extend(nba_games)

    # Deduplicate by (date, home_team, away_team)
    for lg_key in list(all_games.keys()):
        seen = set()
        unique = []
        for g in all_games[lg_key]:
            key = (g["date"], g["home_team"], g["away_team"])
            if key not in seen:
                seen.add(key)
                unique.append(g)
        all_games[lg_key] = unique
        all_games[lg_key].sort(key=lambda x: (x["date"], x.get("status", "")))

    # Check TheOddsAPI once
    odds_sports = check_theoddsapi_sports()

    # Output
    if args.json:
        output = {
            "checked_at": datetime.now().isoformat(),
            "days_checked": args.days,
            "leagues": {},
        }
        for lg_key, games in all_games.items():
            output["leagues"][lg_key] = {
                "total_games": len(games),
                "games": games,
            }
        output["season_status"] = {}
        for lg_key, lg in LEAGUE_BY_KEY.items():
            in_season = _months_overlap(lg.season_months) if lg.season_months else False
            output["season_status"][lg_key] = {
                "in_season": in_season,
                "season_months": lg.season_months,
                "games_today": len(all_games.get(lg_key, [])),
            }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_schedule_summary(all_games, odds_sports)

    total = sum(len(v) for v in all_games.values())
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
