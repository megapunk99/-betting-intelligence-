"""
CLI module — argument parser and main() entry point.

Extracted from the monolithic predict_tomorrow.py.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from betting_intel.pipeline.bootstrap import PROJECT_ROOT, ODDS_API_KEY


# ── CLI Argument Parser ─────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="🏀 Betting Intelligence — Full Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict_tomorrow.py                           # Historical mode\n"
            "  python predict_tomorrow.py --live                    # Live predictions\n"
            "  python predict_tomorrow.py --live --no-tune          # Skip tuning\n"
            "  python predict_tomorrow.py --full                    # Full pipeline\n"
        ),
    )

    # Mode
    mode = parser.add_argument_group("Mode")
    mode.add_argument("--live", action="store_true", help="Fetch live upcoming games from TheOddsAPI")
    mode.add_argument("--full", action="store_true", help="Run full pipeline: predictions → recommendations → risk → simulation")
    mode.add_argument("--recommend-only", action="store_true", help="Generate recommendations from existing predictions only")
    mode.add_argument("--simulate", action="store_true", help="Run Monte Carlo simulation on results")
    mode.add_argument("--scheduled", action="store_true", help="Run in scheduled mode (auto-save, JSON summary to stdout)")

    # Data
    data_grp = parser.add_argument_group("Data Options")
    data_grp.add_argument("--days-history", type=int, default=90, help="Days of historical data to load")
    data_grp.add_argument("--data-source", choices=["csv", "sqlite", "api"], default=None, help="Force data source")
    data_grp.add_argument("--csv-path", type=str, help="Path to CSV data file")

    # Model
    model_grp = parser.add_argument_group("Model Options")
    model_grp.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning")
    model_grp.add_argument("--model-dir", type=str, default="models/saved", help="Directory for saved models")
    model_grp.add_argument("--ensemble", action="store_true", default=True, help="Use ensemble of all strategies")
    model_grp.add_argument("--strategy", type=str, choices=["lightgbm", "catboost", "random_forest", "bayesian", "ridge", "all"], default="all",
                           help="Which prediction strategy to use")

    # Risk
    risk_grp = parser.add_argument_group("Risk Options")
    risk_grp.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll for Kelly sizing")
    risk_grp.add_argument("--kelly-fraction", type=float, default=0.25, help="Kelly fraction (0.0-1.0)")
    risk_grp.add_argument("--max-exposure", type=float, default=0.20, help="Max exposure per game as fraction of bankroll")
    risk_grp.add_argument("--min-edge", type=float, default=0.02, help="Minimum edge threshold (2%% = 0.02)")

    # Output
    out_grp = parser.add_argument_group("Output Options")
    out_grp.add_argument("--output", type=str, default=None, help="Save predictions to JSON file")
    out_grp.add_argument("--html", action="store_true", help="Generate HTML report")
    out_grp.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser.parse_args(argv)


# ── Entry Point ─────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    """Run the prediction pipeline from CLI arguments.

    Returns 0 if predictions were generated, 1 otherwise.
    """
    from betting_intel.pipeline.pipeline import PredictionPipeline

    args = parse_args(argv)

    if args.live and not ODDS_API_KEY:
        print("  ⚠  --live mode requires ODDS_API_KEY env var or .env file.")
        print("  ℹ  Falling back to historical.")
        args.live = False

    # Scheduled mode: force live+no-tune, auto-save results, JSON to stdout
    if args.scheduled:
        if not ODDS_API_KEY:
            print("  ⚠  --scheduled mode requires ODDS_API_KEY env var or .env file.")
            return 1
        args.live = True
        args.no_tune = True
        args.html = False
        if not args.output:
            from betting_intel.pipeline.bootstrap import PROJECT_ROOT
            args.output = str(PROJECT_ROOT / "reports" / "latest.json")

    pipeline = PredictionPipeline(args)
    results = pipeline.run()

    if results.get("clear_picks") or results.get("recommendations"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
