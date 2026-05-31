#!/usr/bin/env python3
"""
Data Quality Report - Daily database health monitoring for NBA data.

Checks:
  1. Record counts per table (with day-over-day change tracking)
  2. NULL value analysis per column
  3. Anomaly detection (future dates, negative stats, impossible values, duplicates)
  4. Data freshness (most recent game, days since update)
  5. Team coverage (all 30 NBA teams present?)
  6. Season distribution
  7. Overall health score

Usage:
    python tools/data_quality_report.py                       # Full report
    python tools/data_quality_report.py --history-file DATA    # Custom history path
    python tools/data_quality_report.py --skip-anomalies       # Skip anomaly checks
"""

import sys
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, date

# Ensure src/ is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from betting_intel.config import DB_PATH

# -- ANSI Colors --
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# -- Constants --
NBA_TEAMS_COUNT = 30
HISTORY_DIR = PROJECT_ROOT / "logs"
HISTORY_FILE = HISTORY_DIR / "data_quality_history.json"
DATE_FORMAT = "%Y-%m-%d"

# Columns to check for NULLs in game_logs (all meaningful stat columns)
GAME_LOG_NULL_COLS = [
    "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL",
    "BLK", "TOV", "PF", "PLUS_MINUS", "MIN", "WL", "TEAM_ID",
    "MATCHUP", "GAME_DATE",
]

PLAYER_LOG_NULL_COLS = [
    "PLAYER_NAME", "PTS", "FGM", "FGA", "MINUTES", "TEAM_ID",
]


# ============================================================================
#  Report Class
# ============================================================================

class DataQualityReport:
    """Generates a structured data quality report from the NBA database."""

    def __init__(self, db_path: Path, history_file: Path | None = None):
        self.db_path = db_path
        self.history_file = history_file or HISTORY_FILE

    # -- Connection --

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    # -- Record Counts --

    def _get_table_counts(self) -> dict[str, int]:
        """Get row counts for all tables."""
        conn = self._connect()
        c = conn.cursor()
        tables = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        counts: dict[str, int] = {}
        for (name,) in tables:
            if name == "sqlite_sequence":
                continue
            count = c.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            counts[name] = count

        conn.close()
        return counts

    def _get_unique_counts(self) -> dict[str, int]:
        """Get unique game and team counts."""
        conn = self._connect()
        c = conn.cursor()

        unique_games = c.execute(
            "SELECT COUNT(DISTINCT GAME_ID) FROM game_logs"
        ).fetchone()[0]

        unique_teams = c.execute(
            "SELECT COUNT(DISTINCT TEAM_ID) FROM game_logs"
        ).fetchone()[0]

        unique_players = 0
        try:
            unique_players = c.execute(
                "SELECT COUNT(DISTINCT PLAYER_ID) FROM player_game_logs"
            ).fetchone()[0]
        except Exception:
            pass

        conn.close()

        return {
            "unique_games": unique_games,
            "unique_teams": unique_teams,
            "unique_players": unique_players,
        }

    # -- NULL Analysis --

    def _check_nulls(self, table: str, columns: list[str]) -> list[dict]:
        """Check NULL counts for specified columns in a table."""
        conn = self._connect()
        c = conn.cursor()
        total = c.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

        null_results = []
        for col in columns:
            try:
                null_count = c.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE "{col}" IS NULL'
                ).fetchone()[0]
                if null_count > 0:
                    null_results.append({
                        "table": table,
                        "column": col,
                        "null_count": null_count,
                        "null_pct": round(null_count / total * 100, 2),
                    })
            except Exception:
                pass  # Column might not exist

        conn.close()
        return null_results

    # -- Anomaly Detection --

    def _check_anomalies(self) -> list[dict]:
        """Check for data anomalies in game_logs."""
        conn = self._connect()
        c = conn.cursor()
        issues = []

        # 1. Future dates
        today_str = date.today().strftime(DATE_FORMAT)
        future_dates = c.execute(
            'SELECT COUNT(*) FROM game_logs WHERE GAME_DATE > ?',
            (today_str,)
        ).fetchone()[0]
        if future_dates > 0:
            # Show them
            future_rows = c.execute(
                'SELECT DISTINCT GAME_ID, GAME_DATE FROM game_logs WHERE GAME_DATE > ? ORDER BY GAME_DATE LIMIT 10',
                (today_str,)
            ).fetchall()
            issues.append({
                "severity": "critical" if future_dates > 50 else "warning",
                "check": "future_dates",
                "count": future_dates,
                "detail": f"{future_dates} rows with game_date in the future",
                "samples": [f"{gid} ({gd})" for gid, gd in future_rows],
                "recommendation": "Check the data source - future games should not have stat rows",
            })

        # 2. Negative or zero stat values
        negative_checks = [
            ("PTS", "PTS < 0", "Negative points"),
            ("FGA", "FGA < 0", "Negative FGA"),
            ("FGM", "FGM < 0", "Negative FGM"),
            ("REB", "REB < 0", "Negative rebounds"),
            ("AST", "AST < 0", "Negative assists"),
            ("MIN", "MIN < 0", "Negative minutes"),
            ("PLUS_MINUS", "PLUS_MINUS IS NULL AND PTS > 0", "NULL plus_minus with points"),
        ]
        for col, condition, label in negative_checks:
            try:
                count = c.execute(
                    f'SELECT COUNT(*) FROM game_logs WHERE {condition}'
                ).fetchone()[0]
                if count > 0:
                    issues.append({
                        "severity": "critical" if count > 10 else "warning",
                        "check": f"negative_{col.lower()}",
                        "count": count,
                        "detail": f"{count} rows: {label}",
                        "recommendation": f"Investigate source of invalid values in {col}",
                    })
            except Exception:
                pass

        # 3. FG_PCT out of range [0, 1]
        invalid_fg_pct = c.execute(
            'SELECT COUNT(*) FROM game_logs WHERE FG_PCT < 0 OR FG_PCT > 1'
        ).fetchone()[0]
        if invalid_fg_pct > 0:
            issues.append({
                "severity": "critical",
                "check": "invalid_fg_pct",
                "count": invalid_fg_pct,
                "detail": f"{invalid_fg_pct} rows with FG_PCT outside [0, 1]",
                "recommendation": "FG_PCT must be between 0 and 1",
            })

        # 4. FG3_PCT out of range
        for pct_col in ["FG3_PCT", "FT_PCT"]:
            invalid = c.execute(
                f'SELECT COUNT(*) FROM game_logs WHERE "{pct_col}" < 0 OR "{pct_col}" > 1'
            ).fetchone()[0]
            if invalid > 0:
                issues.append({
                    "severity": "critical" if invalid > 5 else "warning",
                    "check": f"invalid_{pct_col.lower()}",
                    "count": invalid,
                    "detail": f"{invalid} rows with {pct_col} outside [0, 1]",
                    "recommendation": f"{pct_col} must be between 0 and 1",
                })

        # 5. Duplicate (GAME_ID, TEAM_ID) pairs
        duplicates = c.execute("""
            SELECT GAME_ID, TEAM_ID, COUNT(*) as cnt
            FROM game_logs
            GROUP BY GAME_ID, TEAM_ID
            HAVING cnt > 1
        """).fetchall()
        if duplicates:
            total_dup = sum(cnt for _, _, cnt in duplicates)
            issues.append({
                "severity": "critical" if len(duplicates) > 5 else "warning",
                "check": "duplicate_game_team",
                "count": len(duplicates),
                "detail": f"{len(duplicates)} (GAME_ID, TEAM_ID) pairs have {total_dup} total extra rows",
                "recommendation": "Each team should appear exactly once per game",
            })

        # 6. WL values not in ('W', 'L')
        invalid_wl = c.execute(
            "SELECT COUNT(*) FROM game_logs WHERE WL NOT IN ('W', 'L')"
        ).fetchone()[0]
        if invalid_wl > 0:
            issues.append({
                "severity": "warning",
                "check": "invalid_wl",
                "count": invalid_wl,
                "detail": f"{invalid_wl} rows with WL not in ['W', 'L']",
                "recommendation": "WL should only be 'W' or 'L'",
            })

        # 7. Team coverage
        distinct_teams = c.execute(
            "SELECT COUNT(DISTINCT TEAM_ID) FROM game_logs"
        ).fetchone()[0]
        if distinct_teams < NBA_TEAMS_COUNT:
            issues.append({
                "severity": "warning",
                "check": "missing_teams",
                "count": NBA_TEAMS_COUNT - distinct_teams,
                "detail": f"Only {distinct_teams}/{NBA_TEAMS_COUNT} NBA teams in database",
                "recommendation": "Check that all 30 teams are being scraped",
            })

        # 8. Points but no minutes
        pts_no_min = c.execute(
            "SELECT COUNT(*) FROM game_logs WHERE PTS > 0 AND (MIN IS NULL OR MIN = 0)"
        ).fetchone()[0]
        if pts_no_min > 0:
            issues.append({
                "severity": "warning",
                "check": "pts_without_minutes",
                "count": pts_no_min,
                "detail": f"{pts_no_min} rows with PTS > 0 but no MIN",
                "recommendation": "Players with points should have minutes recorded",
            })

        # 9. Check that PTS >= FGM * 2 (minimum possible points)
        impossible_pts = c.execute(
            "SELECT COUNT(*) FROM game_logs WHERE FGM > 0 AND PTS < FGM * 2"
        ).fetchone()[0]
        if impossible_pts > 0:
            issues.append({
                "severity": "warning",
                "check": "impossible_pts",
                "count": impossible_pts,
                "detail": f"{impossible_pts} rows where PTS < FGM * 2 (impossible)",
                "recommendation": "Points must be at least 2x field goals made",
            })

        conn.close()
        return issues

    # -- Player Game Logs Checks --

    def _check_player_anomalies(self) -> list[dict]:
        """Check player_game_logs for anomalies."""
        conn = self._connect()
        c = conn.cursor()
        issues = []

        try:
            # Check for games with 0 players tracked
            games_with_no_players = c.execute("""
                SELECT COUNT(DISTINCT gl.GAME_ID)
                FROM game_logs gl
                LEFT JOIN player_game_logs pgl ON gl.GAME_ID = pgl.GAME_ID
                WHERE pgl.GAME_ID IS NULL
            """).fetchone()[0]
            if games_with_no_players > 0:
                issues.append({
                    "severity": "info",
                    "check": "games_without_player_stats",
                    "count": games_with_no_players,
                    "detail": f"{games_with_no_players} games have no player stats",
                    "recommendation": "Expected for games where boxscore parsing hasn't run yet",
                })

            # Games with very few players (< 5 per team is suspicious)
            sparse_games = c.execute("""
                SELECT GAME_ID, TEAM_ID, COUNT(*) as cnt
                FROM player_game_logs
                GROUP BY GAME_ID, TEAM_ID
                HAVING cnt < 5
                LIMIT 10
            """).fetchall()
            if sparse_games:
                issues.append({
                    "severity": "info",
                    "check": "sparse_player_games",
                    "count": len(sparse_games),
                    "detail": f"{len(sparse_games)} game-team entries have < 5 players",
                    "recommendation": "May indicate incomplete boxscore parsing",
                })

        except Exception:
            pass

        conn.close()
        return issues

    # -- Data Freshness --

    def _get_freshness(self) -> dict:
        """Get data freshness metrics."""
        conn = self._connect()
        c = conn.cursor()

        # Most recent game date
        last_game = c.execute(
            "SELECT MAX(GAME_DATE) FROM game_logs"
        ).fetchone()[0]

        # Earliest game date
        first_game = c.execute(
            "SELECT MIN(GAME_DATE) FROM game_logs"
        ).fetchone()[0]

        conn.close()

        today = date.today()
        last_date = datetime.strptime(last_game, DATE_FORMAT).date() if last_game else today
        first_date = datetime.strptime(first_game, DATE_FORMAT).date() if first_game else today

        days_since_update = (today - last_date).days
        total_days = (last_date - first_date).days

        freshness_score = max(0, 100 - days_since_update * 5)  # 100 if today, 0 if 20+ days stale
        freshness_grade = (
            "A" if days_since_update <= 1 else
            "B" if days_since_update <= 3 else
            "C" if days_since_update <= 7 else
            "D" if days_since_update <= 14 else
            "F"
        )

        return {
            "first_game_date": str(first_date),
            "last_game_date": str(last_date),
            "days_since_update": days_since_update,
            "total_days_span": total_days,
            "freshness_score": freshness_score,
            "freshness_grade": freshness_grade,
        }

    # -- Season Distribution --

    def _get_season_distribution(self) -> list[dict]:
        """Get game counts per season."""
        conn = self._connect()
        c = conn.cursor()

        seasons = c.execute("""
            SELECT SEASON, COUNT(DISTINCT GAME_ID) as game_count, COUNT(*) as row_count
            FROM game_logs
            GROUP BY SEASON
            ORDER BY SEASON
        """).fetchall()

        conn.close()

        return [
            {
                "season": s,
                "games": gc,
                "rows": rc,
            }
            for s, gc, rc in seasons
        ]

    # -- Change Tracking --

    def _load_history(self) -> dict | None:
        """Load previous report from history file."""
        if self.history_file and self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _save_history(self, report: dict):
        """Save current report to history file."""
        if not self.history_file:
            return
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        # Keep only last 30 reports
        summary = {
            "timestamp": report["timestamp"],
            "table_counts": report["table_counts"],
            "unique_counts": report["unique_counts"],
            "freshness": report["freshness"],
            "issue_count": len(report["issues"]),
            "alert_count": len(report["alerts"]),
            "health_score": report["health_score"],
        }

        history = self._load_history() or {"reports": []}
        history.setdefault("reports", []).append(summary)

        # Keep last 30 entries
        history["reports"] = history["reports"][-30:]

        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)

    def _compute_changes(self, report: dict) -> dict:
        """Compare current report with previous to detect changes."""
        history = self._load_history()
        if not history or not history.get("reports"):
            return {}

        prev = history["reports"][-1]
        current_counts = report["table_counts"]
        prev_counts = prev["table_counts"]

        changes = {}
        for table in set(list(current_counts.keys()) + list(prev_counts.keys())):
            cur = current_counts.get(table, 0)
            prev_c = prev_counts.get(table, 0)
            diff = cur - prev_c
            if diff != 0:
                changes[table] = {
                    "previous": prev_c,
                    "current": cur,
                    "delta": diff,
                    "delta_pct": round(diff / max(prev_c, 1) * 100, 1),
                }

        return changes

    # -- Health Score --

    def _compute_health_score(self, freshness: dict, issues: list[dict], unique_counts: dict) -> dict:
        """Compute overall database health score 0-100."""
        score = 100.0
        deductions = []

        # Freshness penalty
        days_stale = freshness["days_since_update"]
        if days_stale > 1:
            pen = min(days_stale * 3, 30)
            score -= pen
            deductions.append(f"Data {days_stale}d stale (-{pen:.0f})")

        # Issue penalties
        severity_weights = {"critical": 15, "warning": 5, "info": 1}
        for issue in issues:
            sev = issue.get("severity", "info")
            pen = severity_weights.get(sev, 1) * min(issue.get("count", 1), 10)
            score -= pen
            if sev != "info":
                deductions.append(f"{sev}: {issue['check']} ({issue.get('count', 0)}) (-{min(pen, 15):.0f})")

        # Team coverage
        if unique_counts["unique_teams"] < NBA_TEAMS_COUNT:
            missing = NBA_TEAMS_COUNT - unique_counts["unique_teams"]
            score -= missing * 2
            deductions.append(f"{missing} missing teams (-{missing * 2})")

        score = max(0, min(100, round(score, 1)))

        grade = (
            "A" if score >= 90 else
            "B" if score >= 75 else
            "C" if score >= 60 else
            "D" if score >= 40 else
            "F"
        )

        return {
            "score": score,
            "grade": grade,
            "deductions": deductions,
            "status": "HEALTHY" if score >= 75 else "DEGRADED" if score >= 50 else "CRITICAL",
        }

    # -- Main Report Generation --

    def generate(self, skip_anomalies: bool = False) -> dict:
        """Generate a complete data quality report."""
        report: dict = {
            "timestamp": datetime.now().isoformat(),
            "database": str(self.db_path),
        }

        # 1. Record counts
        report["table_counts"] = self._get_table_counts()
        report["unique_counts"] = self._get_unique_counts()

        # 2. NULL analysis
        nulls = self._check_nulls("game_logs", GAME_LOG_NULL_COLS)
        try:
            nulls.extend(self._check_nulls("player_game_logs", PLAYER_LOG_NULL_COLS))
        except Exception:
            pass
        report["null_issues"] = nulls

        # 3. Anomalies
        report["issues"] = []
        if not skip_anomalies:
            report["issues"] = self._check_anomalies()
            report["issues"].extend(self._check_player_anomalies())

        # 4. Freshness
        report["freshness"] = self._get_freshness()

        # 5. Season distribution
        report["season_distribution"] = self._get_season_distribution()

        # 6. Change tracking
        report["changes"] = self._compute_changes(report)

        # 7. Health score
        report["health_score"] = self._compute_health_score(
            report["freshness"], report["issues"], report["unique_counts"]
        )

        # 8. Alerts (health score alerts)
        report["alerts"] = []
        if report["health_score"]["status"] == "CRITICAL":
            report["alerts"].append({
                "severity": "critical",
                "message": f"Database health score is {report['health_score']['score']}/100 ({report['health_score']['grade']}). Immediate attention needed.",
            })
        elif report["health_score"]["status"] == "DEGRADED":
            report["alerts"].append({
                "severity": "warning",
                "message": f"Database health score is {report['health_score']['score']}/100 ({report['health_score']['grade']}). Review recommended.",
            })
        if report["freshness"]["days_since_update"] > 3:
            report["alerts"].append({
                "severity": "warning",
                "message": f"No new data in {report['freshness']['days_since_update']} days. Last game: {report['freshness']['last_game_date']}.",
            })

        # Save history
        self._save_history(report)

        return report


# ============================================================================
#  Formatting
# ============================================================================

def format_report(report: dict) -> str:
    """Format the data quality report as a readable string."""
    lines = []

    def section(title: str):
        lines.append("")
        lines.append(f"  {BOLD}{title}{RESET}")
        lines.append(f"  {'-' * 60}")

    def kv(key: str, val: str, color: str = ""):
        if color:
            lines.append(f"  {key:<30s} {color}{val}{RESET}")
        else:
            lines.append(f"  {key:<30s} {val}")

    # -- Header --
    lines.append(f"\n{CYAN}{BOLD}{'=' * 70}{RESET}")
    lines.append(f"{CYAN}{BOLD}  DATA QUALITY REPORT{RESET}")
    lines.append(f"{CYAN}{BOLD}  {report['timestamp'][:19].replace('T', ' ')}{RESET}")
    lines.append(f"{CYAN}{BOLD}{'=' * 70}{RESET}")

    # -- Health Score --
    hs = report["health_score"]
    score_color = GREEN if hs["score"] >= 75 else YELLOW if hs["score"] >= 50 else RED
    lines.append("")
    lines.append(f"  {'OVERALL HEALTH':30s} {score_color}{hs['grade']} ({hs['score']}/100) - {hs['status']}{RESET}")
    for d in hs["deductions"]:
        lines.append(f"  {'':30s} {YELLOW}! {d}{RESET}")

    # -- Record Counts --
    section("Record Counts")
    for table, count in sorted(report["table_counts"].items()):
        kv(f"  {table}", f"{count:,} rows")
    uc = report["unique_counts"]
    kv(f"  Unique games", f"{uc['unique_games']:,}")
    kv(f"  Unique teams", f"{uc['unique_teams']}/{NBA_TEAMS_COUNT}")
    if uc["unique_players"]:
        kv(f"  Unique players", f"{uc['unique_players']:,}")

    # -- Change Tracking --
    if report.get("changes"):
        section("Changes vs Previous Report")
        for table, ch in report["changes"].items():
            delta_color = GREEN if ch["delta"] > 0 else RED
            arrow = "+" if ch["delta"] > 0 else "-"
            kv(f"  {table}", f"{ch['previous']:,} -> {ch['current']:,}  ({delta_color}{arrow} {ch['delta']:+d}{RESET}, {ch['delta_pct']:+.1f}%)")

    # -- Data Freshness --
    section("Data Freshness")
    f = report["freshness"]
    fresh_color = GREEN if f["freshness_grade"] in ("A", "B") else YELLOW if f["freshness_grade"] == "C" else RED
    kv(f"  Date range", f"{f['first_game_date']} -> {f['last_game_date']} ({f['total_days_span']} days)")
    days_text = f"{f['days_since_update']} day{'s' if f['days_since_update'] != 1 else ''} ago"
    kv(f"  Last update", f"{days_text} [{f['freshness_grade']}]", color=fresh_color)

    # -- Season Distribution --
    section("Season Distribution")
    for s in report["season_distribution"]:
        kv(f"  {s['season']}", f"{s['games']} games ({s['rows']} rows)")

    # -- NULL Analysis --
    nulls = report.get("null_issues", [])
    if nulls:
        section("NULL Values Detected")
        for n in nulls:
            n_color = YELLOW if n["null_pct"] > 5 else RESET
            kv(f"  {n['table']}.{n['column']}", f"{n['null_count']:,} ({n['null_pct']:.1f}%)", color=n_color)
    else:
        section("NULL Values")
        kv(f"  (none)", f"{GREEN}All checked columns are fully populated{RESET}")

    # -- Issues / Anomalies --
    issues = report.get("issues", [])
    if issues:
        section("Anomalies Detected")
        for issue in issues:
            sev_color = RED if issue["severity"] == "critical" else YELLOW if issue["severity"] == "warning" else CYAN
            sev_label = issue["severity"].upper()
            kv(f"  [{sev_label}] {issue['check']}", f"{sev_color}{issue['detail']}{RESET}", color=sev_color)
            kv(f"  {'':30s}", f">> {issue['recommendation']}", color=CYAN)
    else:
        section("Anomalies")
        kv(f"  (none)", f"{GREEN}No anomalies detected{RESET}")

    # -- Alerts --
    alerts = report.get("alerts", [])
    if alerts:
        section("Alerts")
        for a in alerts:
            a_color = RED if a["severity"] == "critical" else YELLOW
            kv(f"  [{a['severity'].upper()}]", f"{a_color}{a['message']}{RESET}")

    # -- Footer --
    lines.append("")
    lines.append(f"{CYAN}{BOLD}{'=' * 70}{RESET}")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
#  Main Entry Point
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Data Quality Report - daily database health monitoring",
    )
    parser.add_argument("--history-file", type=str, default=str(HISTORY_FILE),
                        help="Path to history JSON file for change tracking")
    parser.add_argument("--skip-anomalies", action="store_true",
                        help="Skip anomaly detection checks (faster)")
    args = parser.parse_args()

    history_path = Path(args.history_file) if args.history_file else None

    reporter = DataQualityReport(
        db_path=DB_PATH,
        history_file=history_path,
    )

    report = reporter.generate(skip_anomalies=args.skip_anomalies)
    output = format_report(report)
    print(output)

    # Return exit code based on health
    if report["health_score"]["status"] == "CRITICAL":
        return 2
    elif report["health_score"]["status"] == "DEGRADED":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
