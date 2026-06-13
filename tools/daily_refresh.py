#!/usr/bin/env python3
"""
Daily refresh — Automated data collection, model retraining, and tomorrow predictions.

This script is designed to be run daily (via cron / Windows Task Scheduler) to:
  1. Fetch recent completed games from basketball-reference.com
  2. Fetch upcoming game odds from TheOddsAPI
  3. Update local database with new game results
  4. Retrain prediction models on fresh data
  5. Generate tomorrow's betting predictions
  6. Save predictions to output/
  7. Log results for monitoring

Usage:
    python tools/daily_refresh.py                        # Full daily refresh
    python tools/daily_refresh.py --mode fetch-only      # Just fetch new data
    python tools/daily_refresh.py --mode train-only      # Just retrain models
    python tools/daily_refresh.py --mode predict-only    # Just generate predictions
    python tools/daily_refresh.py --days-back 7          # Fetch last 7 days of games
    python tools/daily_refresh.py --verbose              # Detailed logging

Environment Variables:
    ODDS_API_KEY (required): Your TheOddsAPI key for live odds
    LOG_DIR: Directory for log files (default: logs/)

Requirements:
    pip install requests beautifulsoup4 lxml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Logging ───────────────────────────────────────────────────────────
LOG_DIR = Path(os.environ.get("LOG_DIR", PROJECT_ROOT / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"daily_refresh_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("daily_refresh")

# Suppress noisy libraries
for lib in ["urllib3", "requests", "selenium", "websocket"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1: FETCH DATA
# ══════════════════════════════════════════════════════════════════════


def stage_fetch_recent_games(days_back: int = 3) -> int:
    """
    Fetch recent completed games from basketball-reference.com.

    Returns:
        Number of games fetched.
    """
    logger.info("═" * 60)
    logger.info("  STAGE 1: Fetch recent games from Basketball-Reference")
    logger.info("═" * 60)

    try:
        from betting_intel.data.basketball_reference import BasketballReferenceScraper

        scraper = BasketballReferenceScraper()
        games = scraper.fetch_recent_days(days=days_back)
        scraper.close()

        logger.info(f"  Fetched {len(games)} completed games from last {days_back} days")

        if not games:
            logger.warning("  No games fetched. Check basketball-reference.com accessibility.")
            return 0

        # Save to JSON for the trainer to consume
        output_path = PROJECT_ROOT / "data" / "recent_games.json"
        with open(output_path, "w") as f:
            json.dump(games, f, indent=2, default=str)
        logger.info(f"  Saved recent games to {output_path}")

        return len(games)

    except ImportError as e:
        logger.error(f"  Basketball-Reference scraper not available: {e}")
        logger.error("  Install: pip install requests beautifulsoup4 lxml")
        return 0
    except Exception as e:
        logger.error(f"  Failed to fetch games: {e}")
        return 0


def stage_fetch_live_odds() -> int:
    """
    Fetch upcoming game odds from TheOddsAPI.

    Returns:
        Number of upcoming games found.
    """
    logger.info("─" * 50)
    logger.info("  Fetching live odds from TheOddsAPI...")

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        logger.warning("  No ODDS_API_KEY set in .env. Skipping odds fetch.")
        logger.warning("  Get a free key at: https://the-odds-api.com/")
        return 0

    try:
        from betting_intel.data.odds_fetcher import OddsAPIClient

        client = OddsAPIClient(api_key=api_key)
        games = client.get_upcoming_games_with_odds(
            sport="basketball_nba",
            markets="h2h,spreads,totals",
            use_cache=False,
        )
        logger.info(f"  Found {len(games)} upcoming games with odds")

        # Save to JSON for the prediction stage
        output_path = PROJECT_ROOT / "data" / "live_odds.json"
        with open(output_path, "w") as f:
            json.dump([g.to_dict() for g in games], f, indent=2, default=str)
        logger.info(f"  Saved live odds to {output_path}")

        return len(games)

    except Exception as e:
        logger.error(f"  Failed to fetch live odds: {e}")
        return 0


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2: TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════


def stage_train_models() -> bool:
    """
    Train/retrain prediction models on the latest data.

    Returns:
        True if models were trained successfully.
    """
    logger.info("═" * 60)
    logger.info("  STAGE 2: Train/retrain prediction models")
    logger.info("═" * 60)

    try:
        # Run the pipeline in historical mode to retrain on all data
        from betting_intel.pipeline.pipeline import PredictionPipeline
        from betting_intel.pipeline.cli import parse_args

        # Use only known CLI flags, then setattr for remaining attrs
        args = parse_args(["--live", "--no-tune"])
        pipeline = PredictionPipeline(args)

        # Ensure critical attributes exist (setattr fallback for any parse_args doesn't define)
        for key, val in [("live", True), ("no_tune", True), ("model_dir", "models"),
                          ("min_edge", 0.02), ("bankroll", 10000.0)]:
            if not hasattr(pipeline.args, key):
                setattr(pipeline.args, key, val)

        results = pipeline.run()

        # Check if model was saved
        model_path = Path("models/pipeline_ensemble_full.pkl")
        if model_path.exists():
            logger.info(f"  Model saved: {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")
        else:
            logger.warning("  Model file not found after training — checking alternatives...")
            for p in Path("models").glob("*.pkl"):
                logger.info(f"    Found: {p} ({p.stat().st_size / 1024:.0f} KB)")

        # Log training metrics
        metadata = results.get("metadata", {})
        if metadata:
            n_features = metadata.get("n_features_selected", metadata.get("n_features_raw", "?"))
            n_folds = metadata.get("n_folds", "?")
            logger.info(f"  Training complete: {n_features} features, {n_folds} folds")

        return True

    except Exception as e:
        logger.error(f"  Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3: GENERATE TOMORROW PREDICTIONS
# ══════════════════════════════════════════════════════════════════════


def stage_predict_tomorrow() -> list:
    """
    Generate tomorrow's betting predictions using the trained model and live odds.

    Returns:
        List of prediction dicts.
    """
    logger.info("═" * 60)
    logger.info("  STAGE 3: Generate tomorrow's predictions")
    logger.info("═" * 60)

    try:
        from betting_intel.pipeline.predict_tomorrow import PredictTomorrowPipeline

        pipeline = PredictTomorrowPipeline(
            bankroll=10000.0,
            kelly_fraction=0.25,
            min_edge=0.02,
            save_results=True,
        )
        results = pipeline.run()

        logger.info(f"  Generated {len(results)} total opportunities")
        actionable = [r for r in results if r["stake"] > 0]
        logger.info(f"  Actionable bets: {len(actionable)}")

        if actionable:
            logger.info("  Top picks:")
            for r in actionable[:5]:
                logger.info(f"    {r['game']} | {r['prediction']} "
                           f"| Edge: {r['edge']:.2%} | Stake: ${r['stake']:.0f}")

        return results

    except Exception as e:
        logger.error(f"  Tomorrow predictions failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# ══════════════════════════════════════════════════════════════════════
#  STAGE 4: EXPORT TO WEB
# ══════════════════════════════════════════════════════════════════════


def stage_export_for_web():
    """
    Export prediction results to the web app's data directory.
    """
    logger.info("─" * 50)
    logger.info("  Exporting predictions for web app...")

    try:
        from betting_intel.pipeline.export import export_predictions_to_web
        result = export_predictions_to_web()
        logger.info(f"  Web export: {result}")
    except Exception as e:
        logger.warning(f"  Web export skipped: {e}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════


def run_full_refresh(days_back: int = 3, verbose: bool = False) -> dict:
    """
    Run the complete daily refresh pipeline.

    Args:
        days_back: Number of past days to fetch game data for
        verbose: Enable DEBUG-level logging

    Returns:
        Dict with status of each stage
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start_time = time.time()
    logger.info("")
    logger.info("█" * 60)
    logger.info("  🏀  DAILY REFRESH — Betting Intelligence System")
    logger.info(f"     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("█" * 60)
    logger.info("")

    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stages": {},
        "success": True,
    }

    # Stage 1: Fetch data
    try:
        n_games = stage_fetch_recent_games(days_back=days_back)
        status["stages"]["fetch_games"] = {"games_fetched": n_games, "success": True}
        if n_games == 0:
            logger.warning("  No new games fetched. Continuing with existing data.")
    except Exception as e:
        logger.error(f"  Fetch stage failed: {e}")
        status["stages"]["fetch_games"] = {"error": str(e), "success": False}
        status["success"] = False

    # Stage 1b: Fetch live odds
    try:
        n_odds = stage_fetch_live_odds()
        status["stages"]["fetch_odds"] = {"games_found": n_odds, "success": True}
    except Exception as e:
        logger.error(f"  Odds fetch failed: {e}")
        status["stages"]["fetch_odds"] = {"error": str(e), "success": False}

    # Stage 2: Train models
    try:
        model_trained = stage_train_models()
        status["stages"]["train_models"] = {"trained": model_trained, "success": model_trained}
        if not model_trained:
            logger.warning("  Model training did not complete. Predictions may use stale model.")
    except Exception as e:
        logger.error(f"  Training stage failed: {e}")
        status["stages"]["train_models"] = {"error": str(e), "success": False}

    # Stage 3: Generate tomorrow predictions
    try:
        predictions = stage_predict_tomorrow()
        n_actionable = sum(1 for r in predictions if r.get("stake", 0) > 0) if predictions else 0
        status["stages"]["predict_tomorrow"] = {
            "total": len(predictions),
            "actionable": n_actionable,
            "success": True,
        }
    except Exception as e:
        logger.error(f"  Prediction stage failed: {e}")
        status["stages"]["predict_tomorrow"] = {"error": str(e), "success": False}

    # Stage 4: Export for web
    try:
        stage_export_for_web()
        status["stages"]["export_web"] = {"success": True}
    except Exception as e:
        logger.warning(f"  Web export skipped: {e}")
        status["stages"]["export_web"] = {"success": False, "error": str(e)}

    # Summary
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("═" * 60)
    logger.info(f"  Daily refresh complete in {elapsed:.1f}s")
    success_count = sum(1 for s in status["stages"].values() if s.get("success"))
    total_count = len(status["stages"])
    logger.info(f"  Stages: {success_count}/{total_count} successful")
    logger.info("═" * 60)
    logger.info("")

    # Save status report
    report_path = PROJECT_ROOT / "data" / "daily_refresh_status.json"
    with open(report_path, "w") as f:
        json.dump(status, f, indent=2, default=str)
    logger.info(f"Status report saved to {report_path}")

    return status


def main():
    parser = argparse.ArgumentParser(
        description="Daily refresh — automated data collection, model retraining, and predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/daily_refresh.py                        # Full daily refresh
  python tools/daily_refresh.py --mode fetch-only      # Just fetch new data
  python tools/daily_refresh.py --mode train-only      # Just retrain models
  python tools/daily_refresh.py --mode predict-only    # Just generate predictions
  python tools/daily_refresh.py --days-back 7          # Fetch last 7 days
  python tools/daily_refresh.py --verbose              # Detailed logging
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "fetch-only", "train-only", "predict-only"],
        help="Which stages to run (default: full)",
    )
    parser.add_argument(
        "--days-back", type=int, default=3,
        help="Number of past days to fetch game data for (default: 3)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )

    args = parser.parse_args()

    if args.mode == "fetch-only":
        n = stage_fetch_recent_games(days_back=args.days_back)
        stage_fetch_live_odds()
        print(f"\nFetch complete: {n} games")
    elif args.mode == "train-only":
        ok = stage_train_models()
        print(f"\nTraining {'succeeded' if ok else 'failed'}")
    elif args.mode == "predict-only":
        preds = stage_predict_tomorrow()
        print(f"\nPredictions: {len(preds)} opportunities")
    else:
        status = run_full_refresh(days_back=args.days_back, verbose=args.verbose)
        if not status["success"]:
            print("\n⚠  Some stages failed. Check logs for details.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
