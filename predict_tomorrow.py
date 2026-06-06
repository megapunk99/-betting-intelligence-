#!/usr/bin/env python3
"""
predict_tomorrow.py — Full Advanced Betting Prediction Pipeline

This is now a thin entry point that delegates to the modular pipeline
implementation under src/betting_intel/pipeline/.

Usage:
    python predict_tomorrow.py               # Historical mode
    python predict_tomorrow.py --live        # Live predictions
    python predict_tomorrow.py --full        # Full pipeline
    python predict_tomorrow.py --scheduled   # Scheduled mode
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Bootstrap ──────────────────────────────────────────────────────────

# Fix Unicode on Windows terminals
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

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
os.environ["_PIPELINE_PROJECT_ROOT"] = str(PROJECT_ROOT)

# Ensure both project root and src/ are on sys.path
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

os.environ.setdefault("LOG_LEVEL", "INFO")

# ── Delegate to the modular pipeline package ───────────────────────────

from betting_intel.pipeline import main as pipeline_main

if __name__ == "__main__":
    sys.exit(pipeline_main())
