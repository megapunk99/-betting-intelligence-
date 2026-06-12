"""
Multi-league model training and prediction stub.

The full multi_league module was part of the deleted packages. This stub
prevents ModuleNotFoundError when the pipeline calls train_multi_league_models()
or predict_multi_league_games() in live mode. It returns empty results.

To restore multi-league functionality, re-create this module with:
  - train_all_basketball_models(leagues, output_dir)
  - predict_league_games(league_key, upcoming_df, model, feature_cols)
  - load_league_model(league_key)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def train_all_basketball_models(
    leagues: Optional[List[str]] = None,
    output_dir: str = "models/saved",
) -> Dict[str, Any]:
    """Stub — original was in the deleted multi-league package.

    Returns empty results dict. No models are trained.
    """
    logger.info(
        "[multi_league stub] train_all_basketball_models called "
        f"(leagues={leagues}, output_dir={output_dir}) — no-op"
    )
    return {}


def predict_league_games(
    league_key: str,
    upcoming_df: Any,
    model: Any = None,
    feature_cols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Stub — original was in the deleted multi-league package.

    Returns empty list. No predictions are generated.
    """
    logger.debug(
        "[multi_league stub] predict_league_games called "
        f"(league_key={league_key}) — no-op"
    )
    return []


def load_league_model(league_key: str) -> tuple:
    """Stub — original was in the deleted multi-league package.

    Returns (None, None) so callers skip gracefully.
    """
    logger.debug(
        f"[multi_league stub] load_league_model({league_key}) — returning None"
    )
    return None, None
