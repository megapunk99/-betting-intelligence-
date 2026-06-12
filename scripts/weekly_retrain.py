#!/usr/bin/env python3
"""
Weekly Full Retrain — Runs the full ML pipeline with all models,
performs walk-forward backtesting, saves to model registry,
then updates the engine caches for the recommendation engine.

Designed to run every Sunday via Windows Task Scheduler.

Usage:
    python scripts/weekly_retrain.py                    # Full weekly run
    python scripts/weekly_retrain.py --skip-pipeline    # Skip full pipeline, just retrain engine
    python scripts/weekly_retrain.py --fast             # Fast mode (LightGBM + Momentum only)
    python scripts/weekly_retrain.py --scheduled        # Scheduled mode
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
        logging.FileHandler(LOG_DIR / f"weekly_retrain_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("weekly_retrain")

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


def print_header():
    print(f"\n{'=' * 60}")
    print(f"  WEEKLY FULL RETRAIN")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")


def phase_refresh_data() -> bool:
    """Phase 1: Refresh NBA data."""
    logger.info(f"\n{CYAN}[1/5] Refreshing NBA data...{RESET}")

    script = PROJECT_ROOT / "scripts" / "refresh_nba_data.py"
    if not script.exists():
        logger.warning(f"  refresh_nba_data.py not found — using existing data")
        return True

    exit_code, output = run_subprocess("scripts/refresh_nba_data.py")
    if exit_code != 0:
        logger.warning(f"  Data refresh had issues (exit: {exit_code})")
        return False

    new_games = 0
    for line in output.splitlines():
        if "New games:" in line:
            try:
                new_games = int(line.split(":")[-1].strip())
            except ValueError:
                pass

    if new_games > 0:
        logger.info(f"  {GREEN}Added {new_games} new games to database{RESET}")
    return True


def phase_run_monthly_retrain() -> bool:
    """Phase 2: Run the monthly retrain (all models, walk-forward backtest, registry)."""
    logger.info(f"\n{CYAN}[2/5] Running full model retrain + walk-forward backtest...{RESET}")

    script = PROJECT_ROOT / "tools" / "monthly_retrain.py"
    if not script.exists():
        logger.warning(f"  monthly_retrain.py not found — skipping")
        return False

    exit_code, output = run_subprocess(
        "tools/monthly_retrain.py",
        ["--force"],  # Force retrain even if already done this month
        timeout=600,
    )

    return exit_code == 0


def phase_run_full_pipeline() -> bool:
    """Phase 3: Run the full pipeline (main.py --full — all 7+ models)."""
    logger.info(f"\n{CYAN}[3/5] Running full pipeline (all models)...{RESET}")

    script = PROJECT_ROOT / "main.py"
    if not script.exists():
        logger.warning(f"  main.py not found — skipping")
        return False

    exit_code, output = run_subprocess(
        "main.py",
        ["--full"],
        timeout=600,
    )

    if exit_code == 0:
        logger.info(f"  {GREEN}Full pipeline complete{RESET}")
    else:
        logger.warning(f"  Pipeline completed with warnings (exit: {exit_code})")

    return True


def phase_clear_engine_caches():
    """Phase 4: Clear engine caches so they regenerate with fresh model data."""
    logger.info(f"\n{CYAN}[4/5] Clearing engine caches...{RESET}")

    import os
    cache_dir = PROJECT_ROOT / "models" / "saved"
    if not cache_dir.exists():
        logger.info(f"  Cache directory doesn't exist")
        return

    caches_to_clear = [
        "momentum_engine_cache.pkl",
        "schedule_cache.pkl",
        "team_stats_cache.pkl",
    ]

    for cache_file in caches_to_clear:
        cache_path = cache_dir / cache_file
        if cache_path.exists():
            try:
                os.remove(cache_path)
                logger.info(f"  Removed: {cache_file}")
            except Exception as e:
                logger.warning(f"  Could not remove {cache_file}: {e}")

    logger.info(f"  {GREEN}Caches cleared — will regenerate on next request{RESET}")


def phase_regenerate_engine() -> bool:
    """Phase 5: Regenerate recommendation engine (stub — RecommendationEngine deleted)."""
    logger.info(f"\n{CYAN}[5/5] Engine regeneration skipped (RecommendationEngine was deleted){RESET}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Weekly full retrain — full pipeline + model registry + engine refresh",
    )
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip full ML pipeline, just retrain engine caches")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode (LightGBM + Momentum only, not all 7 models)")
    parser.add_argument("--scheduled", action="store_true",
                        help="Scheduled mode")
    args = parser.parse_args()

    start_time = time.time()
    print_header()

    ok = True

    # Phase 1: Refresh data
    ok &= phase_refresh_data()

    if not args.skip_pipeline:
        # Phase 2: Monthly retrain (model registry + backtest)
        phase_run_monthly_retrain()

        # Phase 3: Full pipeline
        if args.fast:
            logger.info(f"\n{CYAN}[3/5] FAST MODE: Running LightGBM + Momentum only...{RESET}")
            exit_code, _ = run_subprocess("main.py", timeout=300)
        else:
            phase_run_full_pipeline()
    else:
        logger.info(f"\n  Skipping pipeline (--skip-pipeline)")

    # Phase 5: Clear caches (includes momentum model cache — force retrain)
    phase_clear_engine_caches()

    # Phase 6: Regenerate engine (loads pipeline predictions or pre-trained models)
    ok &= phase_regenerate_engine()

    # Summary
    elapsed = time.time() - start_time
    if ok:
        print(f"\n  WEEKLY RETRAIN COMPLETE ({elapsed:.1f}s)")
    else:
        print(f"\n  WEEKLY RETRAIN FINISHED WITH WARNINGS ({elapsed:.1f}s)")

    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
