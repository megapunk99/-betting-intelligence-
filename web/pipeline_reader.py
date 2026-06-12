"""
Pipeline output reader — loads saved predictions from forward_test_results.json.

This is a compatibility shim for any code that still imports from web.pipeline_reader.
All new code should use web.app.load_predictions() directly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
FORWARD_TEST_JSON = PROJECT_ROOT / "data" / "forward_test_results.json"


def load_saved_dashboard_data() -> dict | None:
    """Load predictions from forward_test_results.json."""
    if not FORWARD_TEST_JSON.exists():
        return None
    try:
        with open(FORWARD_TEST_JSON) as f:
            data = json.load(f)
        return {
            "summary": data.get("summary", {}),
            "all_bets": data.get("all_bets", []),
            "clear_picks": data.get("clear_picks", []),
            "generated_at": data.get("generated_at", ""),
            "source": "forward_test_results.json",
        }
    except Exception as e:
        logger.warning(f"Failed to load predictions: {e}")
        return None


def predictions_file_info() -> list[dict]:
    """Return whether the predictions file exists."""
    return [{"name": "Forward Test Results", "exists": FORWARD_TEST_JSON.exists(), "path": str(FORWARD_TEST_JSON)}]
