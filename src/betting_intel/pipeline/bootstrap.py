"""
Bootstrap module — environment setup, imports, and module-level constants.

This module is imported early by the pipeline and the thin root-level
entry point (predict_tomorrow.py).  It handles:

    - Unicode configuration for Windows terminals
    - sys.path manipulation for project-relative imports
    - .env file loading
    - ODDS_API_KEY retrieval
    - All internal betting_intel imports (canonical package paths)
    - Module availability flags (HAS_RECOMMENDATIONS, HAS_RISK, etc.)

Usage:
    from betting_intel.pipeline.bootstrap import (
        PROJECT_ROOT, ODDS_API_KEY, logger,
        HAS_RECOMMENDATIONS, HAS_RISK, HAS_BETTING,
        HAS_VALIDATION, HAS_MONITORING, HAS_BACKTESTING, HAS_ROOT_PREDICTORS,
        FeatureEngineer, TotalPointsPredictor, …,
    )
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Unicode setup (must happen early on Windows) ────────────────────────
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

# ── Project root ────────────────────────────────────────────────────────
# When this file is imported from within src/betting_intel, the project
# root is two levels up.  The thin entry point (root predict_tomorrow.py)
# also sets PROJECT_ROOT before importing this module, so we only fall
# back to this heuristic when it hasn't been set yet.
if not os.environ.get("_PIPELINE_PROJECT_ROOT"):
    _probe = Path(__file__).resolve()
    PROJECT_ROOT = _probe.parent.parent.parent  # src/betting_intel/pipeline -> project root
else:
    PROJECT_ROOT = Path(os.environ["_PIPELINE_PROJECT_ROOT"])

os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
# Ensure both project root and src/ are on sys.path
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("LOG_LEVEL", "INFO")

# ── .env ────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ── Imports from the betting_intel package ──────────────────────────────

from betting_intel.data.features import FeatureEngineer
from betting_intel.models.predictors import (
    TotalPointsPredictor,
    SpreadPredictor,
    MomentumModel,
    StackingEnsemblePredictor,
)
from betting_intel.recommendations.engine import RecommendationEngine
from betting_intel.recommendations.bet_types import BetType
from betting_intel.recommendations.ranker import BetRanker
from betting_intel.recommendations.ev_scanner import PositiveEVScanner
from betting_intel.recommendations.arbitrage import ArbitrageDetector
from betting_intel.recommendations.player_props import PlayerPropEngine
from betting_intel.risk.kelly import KellyCalculator
from betting_intel.risk.exposure import ExposureManager
from betting_intel.risk.correlation import BetCorrelationTracker
from betting_intel.betting.edge import EdgeDetector
from betting_intel.betting.monte_carlo import MonteCarloSimulator
from betting_intel.validation.calibration import ProbabilityCalibrator
from betting_intel.validation.overfitting import OverfittingDetector
from betting_intel.validation.cross_validation import TimeSeriesCrossValidator
from betting_intel.monitoring.drift import PerformanceTracker
from betting_intel.backtesting.metrics import BacktestMetrics
from betting_intel.services.logging import get_logger
from betting_intel.services.journal import BetJournal

# ── Module availability flags ───────────────────────────────────────────

HAS_RECOMMENDATIONS = True
HAS_RISK = True
HAS_BETTING = True
HAS_VALIDATION = True
HAS_MONITORING = True
HAS_BACKTESTING = True
HAS_ROOT_PREDICTORS = True

# Module-level logger
logger = get_logger(__name__)
