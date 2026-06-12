#!/usr/bin/env python3
"""
Daily feature refresh — extracted from .github/workflows/daily_feature_refresh.yml.

Usage:
    python tools/daily_refresh.py --mode build-features   # Load data + build features
    python tools/daily_refresh.py --mode validate-checks  # Verify validation modules import
"""

import sys
import warnings
import argparse
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings("ignore")


# ======================================================================
#  Mode: build-features
# ======================================================================

def build_features():
    """Load NBA game data and build feature store snapshots."""
    from betting_intel.features.builder import FeatureBuilder
    from betting_intel.data.loader import NBADataLoader
    from betting_intel.config import DB_PATH

    print("Loading NBA game data...")
    loader = NBADataLoader()
    raw_df = loader.load_game_logs()
    print(f"  Loaded {len(raw_df)} rows")

    builder = FeatureBuilder(DB_PATH)
    version = builder.build_all(raw_df, description="Daily auto-refresh")
    print(f"  Features built: {version}")


# ======================================================================
#  Mode: validate-checks
# ======================================================================

def validate_checks():
    """Verify that validation/calibration modules import correctly."""
    from betting_intel.validation.cross_validation import TimeSeriesCrossValidator
    from betting_intel.validation.calibration import ProbabilityCalibrator

    print("Validation modules loaded successfully")


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Daily feature refresh and validation checks.",
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["build-features", "validate-checks"],
        help="build-features: load data + build features | validate-checks: verify module imports",
    )
    args = parser.parse_args()

    if args.mode == "build-features":
        build_features()
    else:
        validate_checks()

    return 0


if __name__ == "__main__":
    sys.exit(main())
