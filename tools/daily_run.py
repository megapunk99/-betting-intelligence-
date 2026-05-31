#!/usr/bin/env python3
"""
Daily NBA Pipeline — Updates data and runs the forward test.

Runs on a daily schedule (Windows Task Scheduler) to:
  1. Scrape new completed games from the NBA CDN API
  2. Retrain models on all historical data
  3. Compare model predictions vs real sportsbook odds
  4. Log everything to a timestamped file in logs/

Usage:
    python tools/daily_run.py                          # Full pipeline
    python tools/daily_run.py --skip-update            # Skip data scrape
    python tools/daily_run.py --skip-odds              # Skip forward test
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Load .env so subprocesses inherit ODDS_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable or "python"
LOGS_DIR = PROJECT_ROOT / "logs"
UPDATE_SCRIPT = PROJECT_ROOT / "tools" / "update_data.py"
FORWARD_SCRIPT = PROJECT_ROOT / "tools" / "forward_test.py"
ENV_FILE = PROJECT_ROOT / ".env"

# ── ANSI Colors ────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def log(msg: str, color: str = ""):
    """Print to both stdout and the log file."""
    line = f"{color}{msg}{RESET}" if color else msg
    print(line)


def run_step(description: str, cmd: list[str], timeout: int = 600) -> tuple[bool, str]:
    """Run a subprocess step and return (success, output)."""
    log(f"\n  [{description}] Running: {' '.join(cmd[-3:])}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},
            cwd=str(PROJECT_ROOT),
        )
        # Write stderr to stdout for logging purposes
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        success = result.returncode == 0
        if success:
            log(f"  [{description}] {GREEN}OK{RESET}")
        else:
            log(f"  [{description}] {RED}FAILED (exit code {result.returncode}){RESET}")
        return success, output
    except subprocess.TimeoutExpired:
        log(f"  [{description}] {RED}TIMEOUT after {timeout}s{RESET}")
        return False, f"[TIMEOUT after {timeout}s]"
    except Exception as e:
        log(f"  [{description}] {RED}ERROR: {e}{RESET}")
        return False, str(e)


def write_log(log_path: Path, sections: list[tuple[str, str]]):
    """Write the combined log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"{'=' * 70}",
        f"  NBA Daily Pipeline — {timestamp}",
        f"{'=' * 70}",
        "",
    ]
    for header, content in sections:
        lines.append(f"── {header} ─{'─' * (60 - len(header))}")
        lines.append("")
        lines.append(content.strip())
        lines.append("")

    lines.append(f"{'=' * 70}")
    lines.append("  End of daily run")
    lines.append(f"{'=' * 70}")
    lines.append("")

    log_path.write_text("\n".join(lines), encoding="utf-8")

    # Also print log location
    log(f"\n  {CYAN}Log written: {log_path}{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Daily NBA pipeline: scrape + predict + log",
    )
    parser.add_argument("--skip-update", action="store_true",
                        help="Skip the data update step")
    parser.add_argument("--skip-odds", action="store_true",
                        help="Skip the odds/prediction step")
    args = parser.parse_args()

    # ── Banner ──────────────────────────────────────────────────────────
    print()
    print(f"{CYAN}{BOLD}{'=' * 70}{RESET}")
    print(f"{CYAN}{BOLD}  NBA DAILY PIPELINE{RESET}")
    print(f"{CYAN}{BOLD}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 70}{RESET}")

    # ── Ensure logs dir exists ──────────────────────────────────────────
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Update data ─────────────────────────────────────────────
    sections: list[tuple[str, str]] = []

    if not args.skip_update:
        log(f"\n  {BOLD}[Step 1/2] Scraping new NBA games...{RESET}")
        cmd = [str(PYTHON), "-B", str(UPDATE_SCRIPT)]
        success, output = run_step("Data update", cmd, timeout=600)
        sections.append(("DATA UPDATE", output))
        if not success:
            log(f"\n  {RED}[!] Data update failed. Continuing anyway...{RESET}")
    else:
        log(f"\n  {YELLOW}[Step 1/2] Skipped (--skip-update){RESET}")
        sections.append(("DATA UPDATE", "[SKIPPED]"))

    # ── Step 2: Forward test ────────────────────────────────────────────
    if not args.skip_odds:
        log(f"\n  {BOLD}[Step 2/2] Running forward test with real odds...{RESET}")
        cmd = [str(PYTHON), "-B", str(FORWARD_SCRIPT), "--calibrated"]
        success, output = run_step("Forward test", cmd, timeout=600)
        sections.append(("FORWARD TEST", output))
        if not success:
            log(f"\n  {RED}[!] Forward test had errors.{RESET}")
    else:
        log(f"\n  {YELLOW}[Step 2/2] Skipped (--skip-odds){RESET}")
        sections.append(("FORWARD TEST", "[SKIPPED]"))

    # ── Write log file ──────────────────────────────────────────────────
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_filename = f"daily_{date_str}.log"
    log_path = LOGS_DIR / log_filename
    write_log(log_path, sections)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{CYAN}{BOLD}{'=' * 70}{RESET}")
    print(f"  {GREEN}Daily pipeline complete.{RESET}")
    print(f"  Log: {log_path}")
    if not args.skip_update:
        print(f"  Tip: Check the log for new games scraped today.")
    if not args.skip_odds:
        print(f"  Tip: The forward test output shows +EV opportunities (if any).")
    print(f"{CYAN}{BOLD}{'=' * 70}{RESET}\n")

    # Track failures: return non-zero if any step failed
    # (sections with non-skipped content that contain error keywords)
    any_failure = any(
        "FAILED" in content or "TIMEOUT" in content
        for _, content in sections
        if content not in ("[SKIPPED]", "")
    )
    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
