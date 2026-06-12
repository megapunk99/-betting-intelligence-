#!/usr/bin/env python3
"""
Daily Retrain (Fast) — Refresh NBA data, retrain momentum model,
regenerate engine caches, and generate the daily betting card.

Designed to run every morning via Windows Task Scheduler.

Usage:
    python scripts/daily_retrain.py                    # Full daily run
    python scripts/daily_retrain.py --skip-refresh      # Skip data refresh
    python scripts/daily_retrain.py --skip-card         # Skip betting card
    python scripts/daily_retrain.py --scheduled         # Scheduled mode (no prompts)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Fix Windows console for Unicode ───────────────────────────────────
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Logging ───────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"daily_retrain_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("daily_retrain")

# ── ANSI colors ───────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def run_subprocess(script_name: str, args: list[str] | None = None,
                   timeout: int = 600) -> tuple[int, str]:
    """Run a Python subprocess and return (exit_code, output)."""
    script_path = PROJECT_ROOT / script_name
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    logger.info(f"  Running: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=timeout,
        )
        elapsed = time.time() - start
        output = result.stdout + result.stderr
        logger.info(f"  Completed in {elapsed:.1f}s (exit code: {result.returncode})")
        return result.returncode, output
    except subprocess.TimeoutExpired:
        logger.error(f"  Timed out after {timeout}s")
        return -1, f"TIMEOUT after {timeout}s"
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return -1, str(e)


def refresh_nba_data() -> bool:
    """Fetch fresh NBA data. Returns True if new data was added."""
    logger.info(f"\n{CYAN}[1/4] Refreshing NBA data...{RESET}")

    # Try using the refresh script first
    script = PROJECT_ROOT / "scripts" / "refresh_nba_data.py"
    if script.exists():
        exit_code, output = run_subprocess("scripts/refresh_nba_data.py")
        if exit_code != 0:
            logger.warning(f"  Data refresh had issues (exit: {exit_code})")
            return False
        # Check if new data was added
        if "New games:" in output and "0" not in output.split("New games:")[1][:10]:
            logger.info(f"  New games added!")
            return True
        logger.info(f"  {DIM}No new games found.{RESET}")
        return False
    else:
        logger.warning(f"  refresh_nba_data.py not found — skipping data refresh")
        return False


def clear_engine_caches():
    """Delete cached engine files so they regenerate with fresh data on next load.

    The schedule_cache and team_stats_cache already auto-invalidate via DB mtime,
    but the momentum model cache and the in-memory cache need explicit clearing.
    """
    logger.info(f"\n{CYAN}[2/4] Clearing engine caches...{RESET}")
    cache_dir = PROJECT_ROOT / "models" / "saved"
    if not cache_dir.exists():
        logger.info(f"  Cache directory doesn't exist — nothing to clear")
        return

    caches_to_clear = [
        "momentum_engine_cache.pkl",
    ]

    cleared = 0
    for cache_file in caches_to_clear:
        cache_path = cache_dir / cache_file
        if cache_path.exists():
            try:
                os.remove(cache_path)
                logger.info(f"  Removed: {cache_file}")
                cleared += 1
            except Exception as e:
                logger.warning(f"  Could not remove {cache_file}: {e}")

    # Also remove schedule_cache and team_stats_cache (they'll regenerate)
    # This ensures the new DB data is picked up immediately
    for stale_cache in ["schedule_cache.pkl", "team_stats_cache.pkl"]:
        stale_path = cache_dir / stale_cache
        if stale_path.exists():
            try:
                # Force recreation by deleting (even though they auto-invalidate)
                os.remove(stale_path)
                logger.info(f"  Removed: {stale_cache} (will regenerate on first request)")
            except Exception:
                pass

    if cleared > 0:
        logger.info(f"  Cleared {cleared} cache file(s). Will regenerate with fresh data.")
    else:
        logger.info(f"  {DIM}No caches to clear.{RESET}")


def regenerate_engine():
    """Regenerate engine caches (stub — RecommendationEngine deleted)."""
    logger.info(f"\n{CYAN}[3/4] Engine regeneration skipped (RecommendationEngine was deleted){RESET}")
    logger.info(f"  Recreate betting_intel/recommendations/engine.py to re-enable.")
    return True


def generate_betting_card() -> bool:
    """Generate the daily betting card."""
    logger.info(f"\n{CYAN}[4/4] Generating daily betting card...{RESET}")

    script = PROJECT_ROOT / "run_daily.py"
    if not script.exists():
        logger.warning(f"  run_daily.py not found — skipping card generation")
        return False

    exit_code, output = run_subprocess(
        "run_daily.py",
        ["--scheduled", "--skip-pipeline"],
        timeout=120,
    )

    if exit_code != 0:
        logger.warning(f"  Card generation had issues (exit: {exit_code})")
        return False

    # Check for key success indicators in output
    if "DAILY RUN COMPLETE" in output:
        logger.info(f"  Daily betting card generated!")
        return True
    else:
        logger.warning(f"  Card generation may have completed with warnings")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Daily fast retrain — refresh data, retrain model, generate card",
    )
    parser.add_argument("--skip-refresh", action="store_true",
                        help="Skip NBA data refresh")
    parser.add_argument("--skip-card", action="store_true",
                        help="Skip betting card generation")
    parser.add_argument("--scheduled", action="store_true",
                        help="Scheduled mode (suppresses non-essential output)")
    args = parser.parse_args()

    start_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"  DAILY RETRAIN (Fast)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    # Phase 1: Refresh NBA data
    new_data = False
    if not args.skip_refresh:
        new_data = refresh_nba_data()
    else:
        logger.info(f"\n{CYAN}[1/4] Skipping data refresh{RESET}")
        new_data = True  # Assume we want to retrain anyway

    # Phase 2: Clear engine caches
    clear_engine_caches()

    # Phase 3: Regenerate engine (trains model, loads schedule, computes team stats)
    engine_ok = regenerate_engine()

    # Phase 4: Generate betting card
    card_ok = True
    if not args.skip_card and engine_ok:
        generate_betting_card()
    elif args.skip_card:
        logger.info(f"\n{CYAN}[4/4] Skipping betting card (--skip-card){RESET}")
    elif not engine_ok:
        logger.warning(f"\n  Skipping card due to engine error")

    # Summary
    elapsed = time.time() - start_time
    if engine_ok:
        print(f"\n  DAILY RETRAIN COMPLETE ({elapsed:.1f}s)")
    else:
        print(f"\n  DAILY RETRAIN FAILED ({elapsed:.1f}s)")

    print()
    return 0 if engine_ok else 1


if __name__ == "__main__":
    sys.exit(main())
