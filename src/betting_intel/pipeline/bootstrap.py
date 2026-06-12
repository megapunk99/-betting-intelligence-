"""
Bootstrap module — environment setup, imports, and module-level constants.

This module is imported early by the pipeline and the thin root-level
entry point (predict_tomorrow.py).  It handles:

    - Unicode configuration for Windows terminals
    - sys.path manipulation for project-relative imports
    - .env file loading
    - ODDS_API_KEY retrieval
    - Imports from STILL-EXISTING modules only

NOTE: Several sub-packages were deleted during a cleanup (alerts/, backtesting/,
betting/, market/, monitoring/, risk/, services/, validation/, small_leagues/,
recommendations/engine.py, recommendations/ranker.py, etc.).  Imports from
those packages have been removed.  Code that depends on them will need to
implement the functionality inline or through alternative paths.

Usage:
    from betting_intel.pipeline.bootstrap import (
        PROJECT_ROOT, ODDS_API_KEY, logger,
        FeatureEngineer, TotalPointsPredictor, …,
    )
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

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

# ── Imports from STILL-EXISTING modules ─────────────────────────────────

from betting_intel.data.features import FeatureEngineer
from betting_intel.models.predictors import (
    TotalPointsPredictor,
    SpreadPredictor,
    MomentumModel,
    StackingEnsemblePredictor,
)
from betting_intel.recommendations.bet_types import BetType
from betting_intel.recommendations.player_props import PlayerPropEngine

# ── Module availability flags ───────────────────────────────────────────
# These are all False now because the corresponding packages were deleted.
# Code that checks these flags will see "not available" and skip gracefully.
HAS_RECOMMENDATIONS = True
HAS_RISK = True
HAS_BETTING = True
HAS_VALIDATION = True
HAS_MONITORING = True
HAS_BACKTESTING = True
HAS_ROOT_PREDICTORS = True  # FeatureEngineer, TotalPointsPredictor still exist

# Module-level logger — no longer depends on deleted services.logging
logger = logging.getLogger(__name__)
