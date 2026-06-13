"""
Pipeline export — saves trained models and predictions to engine-compatible format.

This module bridges the gap between the sophisticated ML pipeline (`main.py`)
and the recommendation engine (`engine.py`). After the pipeline trains models
and generates predictions, this module exports them to a shared location that
the engine can load as its primary signal source.

Architecture:
    Pipeline (main.py) → export_predictions() → models/saved/pipeline_predictions.pkl
                                                     ↓
    Engine (engine.py) ← _load_pipeline_predictions() ← reads .pkl
    
    Pipeline (main.py) → train_full_data_model() → models/saved/ → save EnhancedEnsemble
                                                     ↓
    Engine (engine.py) ← _load_enhanced_ensemble() ← reads joblib artifacts

Usage:
    from betting_intel.pipeline.export import export_predictions, load_latest_predictions
    
    # After pipeline completes:
    export_predictions(pipeline_results, output_dir=Path("models/saved"))
    
    # In the engine:
    preds = load_latest_predictions()
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from betting_intel.utils.safe_serialize import (
    safe_pickle_dump, safe_pickle_load,
    safe_joblib_dump, safe_joblib_load,
    ModelIntegrityError,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Default export location (relative to project root)
DEFAULT_EXPORT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "saved"
PREDICTIONS_FILENAME = "pipeline_predictions.pkl"
METADATA_FILENAME = "pipeline_metadata.json"


def export_predictions(
    results: Dict[str, Any],
    output_dir: Optional[Path] = None,
    model_name: str = "EnhancedEnsemble",
) -> Path:
    """
    Export pipeline predictions to a pickle file the engine can consume.

    Extracts the following from pipeline results:
      - predictions (from results["predictions"])
      - tomorrow predictions (from results["tomorrow_predictions"])
      - metadata about model, features, timestamp

    Args:
        results: The pipeline results dict (from PredictionPipeline.run())
        output_dir: Directory to save to. Defaults to models/saved/
        model_name: Name to tag the predictions with

    Returns:
        Path to the exported predictions file
    """
    output_dir = output_dir or DEFAULT_EXPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = results.get("predictions", [])
    tomorrow_preds = results.get("tomorrow_predictions", [])
    metadata = results.get("metadata", {})

    # Build a unified predictions DataFrame
    records = []

    # Historical predictions (from backtest walk-forward)
    if predictions:
        for pred in predictions:
            if isinstance(pred, dict):
                records.append({
                    "game_id": pred.get("game_id", ""),
                    "home_team": pred.get("home_team", pred.get("home", "")),
                    "away_team": pred.get("away_team", pred.get("away", "")),
                    "game_date": pred.get("game_date", ""),
                    "predicted_total": pred.get("predicted_total", 0),
                    "predicted_spread": pred.get("predicted_spread", 0),
                    "market_total": pred.get("market_total", pred.get("market_line", 0)),
                    "market_line": pred.get("market_line", pred.get("market_total", 0)),
                    "edge_pct": pred.get("edge_pct", 0),
                    "direction": pred.get("direction", "over"),
                    "confidence": pred.get("confidence", "medium"),
                    "model_name": pred.get("model_name", model_name),
                    "prediction_type": "historical",
                })

    # Tomorrow predictions (live mode — real upcoming games with odds)
    if tomorrow_preds:
        for pred in tomorrow_preds:
            if isinstance(pred, dict):
                records.append({
                    "game_id": pred.get("game_id", ""),
                    "home_team": pred.get("home_team", ""),
                    "away_team": pred.get("away_team", ""),
                    "game_date": pred.get("game_date", ""),
                    "predicted_total": pred.get("predicted_total", 0),
                    "predicted_spread": pred.get("predicted_spread", 0),
                    "market_total": pred.get("market_total", 0),
                    "market_line": pred.get("market_total", 0),
                    "edge_pct": pred.get("edge_pct", 0),
                    "direction": pred.get("direction", "over"),
                    "confidence": pred.get("confidence", "medium"),
                    "model_name": pred.get("model_name", model_name),
                    "prediction_type": "tomorrow",
                })

    if not records:
        # Try to find predictions in other result fields
        for key in ["recommendations", "clear_picks", "ev_opportunities"]:
            items = results.get(key, [])
            if items and isinstance(items, list):
                for item in items[:50]:
                    if isinstance(item, dict):
                        records.append({
                            "game_id": item.get("game_id", ""),
                            "home_team": item.get("home_team", item.get("team", "")),
                            "away_team": item.get("away_team", ""),
                            "game_date": item.get("game_date", ""),
                            "predicted_total": item.get("predicted_value", item.get("predicted_total", 0)),
                            "predicted_spread": item.get("predicted_spread", 0),
                            "market_total": item.get("market_line", item.get("market_total", 0)),
                            "market_line": item.get("market_line", 0),
                            "edge_pct": item.get("edge", item.get("edge_pct", 0)),
                            "direction": item.get("direction", "over"),
                            "confidence": item.get("confidence", "medium"),
                            "model_name": item.get("model_name", model_name),
                            "prediction_type": key,
                        })
            if records:
                logger.info(f"Found {len(records)} predictions in '{key}' field")
                break

    if not records:
        logger.warning("No predictions found in pipeline results to export")
        # Return the path anyway — caller can check if file exists
        return output_dir / PREDICTIONS_FILENAME

    # Build DataFrame
    df = pd.DataFrame(records)

    # De-duplicate by game_id + prediction_type
    if "game_id" in df.columns:
        df = df.drop_duplicates(subset=["game_id", "prediction_type"], keep="first")

    # Create export package with metadata
    export_package = {
        "exported_at": datetime.now().isoformat(),
        "pipeline_version": results.get("pipeline_version", "3.0"),
        "model_name": model_name,
        "n_predictions": len(df),
        "predictions_df": df,
        "metadata": metadata,
    }

    # Save to disk with hash verification
    export_path = output_dir / PREDICTIONS_FILENAME
    safe_pickle_dump(export_package, export_path)

    # Also save human-readable metadata
    meta_path = output_dir / METADATA_FILENAME
    meta = {
        "exported_at": export_package["exported_at"],
        "pipeline_version": export_package["pipeline_version"],
        "model_name": model_name,
        "n_predictions": len(df),
        "prediction_types": df["prediction_type"].value_counts().to_dict() if "prediction_type" in df.columns else {},
        "n_games": df["game_id"].nunique() if "game_id" in df.columns else 0,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Exported {len(df)} predictions to {export_path}")
    return export_path


def load_latest_predictions(
    export_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Load the latest pipeline predictions for use by the recommendation engine.

    Args:
        export_dir: Directory to load from. Defaults to models/saved/

    Returns:
        DataFrame of predictions, or None if no predictions found
    """
    export_dir = export_dir or DEFAULT_EXPORT_DIR
    export_path = export_dir / PREDICTIONS_FILENAME

    if not export_path.exists():
        logger.info("No pipeline predictions found (file doesn't exist)")
        return None

    try:
        package = safe_pickle_load(export_path)

        df = package.get("predictions_df")
        if df is None or df.empty:
            logger.info("Pipeline predictions file exists but is empty")
            return None

        n_preds = len(df)
        n_games = df["game_id"].nunique() if "game_id" in df.columns else n_preds
        exported_at = package.get("exported_at", "unknown")
        logger.info(
            f"Loaded {n_preds} predictions ({n_games} games) "
            f"exported at {exported_at} from pipeline"
        )

        # Add model_name if missing
        if "model_name" not in df.columns:
            df["model_name"] = package.get("model_name", "PipelineEnsemble")

        return df

    except (FileNotFoundError, ModelIntegrityError) as e:
        logger.warning(f"Pipeline predictions integrity error: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load pipeline predictions: {e}")
        return None


def export_full_model_artifacts(
    ensemble,
    feature_cols: List[str],
    metrics: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Export a trained EnhancedEnsemble (or other model) for the engine to load.

    Saves:
      - model.joblib: The trained model object
      - model_metadata.json: Feature columns, metrics, timestamp

    Args:
        ensemble: Trained model (EnhancedEnsemble, LGBMRegressor, etc.)
        feature_cols: Feature column names used during training
        metrics: Optional performance metrics
        output_dir: Directory to save to

    Returns:
        Path to the saved model directory
    """
    import joblib

    output_dir = output_dir or DEFAULT_EXPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "engine_ensemble"

    model_path = output_dir / f"{model_name}.joblib"
    safe_joblib_dump(ensemble, model_path)

    metadata = {
        "model_name": model_name,
        "exported_at": datetime.now().isoformat(),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "metrics": metrics or {},
        "artifact_path": str(model_path),
    }

    meta_path = output_dir / f"{model_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Exported model '{model_name}' to {model_path} ({len(feature_cols)} features)")
    return output_dir


def load_engine_model(
    model_dir: Optional[Path] = None,
    model_name: str = "engine_ensemble",
) -> tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Load a pre-trained engine model and its metadata.

    Args:
        model_dir: Directory to load from
        model_name: Name of the model (without extension)

    Returns:
        Tuple of (model, metadata_dict) or (None, None) if not found
    """
    import joblib

    model_dir = model_dir or DEFAULT_EXPORT_DIR
    model_path = model_dir / f"{model_name}.joblib"
    meta_path = model_dir / f"{model_name}_metadata.json"

    if not model_path.exists():
        logger.info(f"No pre-trained model found at {model_path}")
        return None, None

    try:
        model = safe_joblib_load(model_path)
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        logger.info(
            f"Loaded pre-trained model '{model_name}' "
            f"({metadata.get('n_features', '?')} features)"
        )
        return model, metadata
    except (ModelIntegrityError, FileNotFoundError) as e:
        logger.warning(f"Pre-trained model integrity check failed: {e}")
        return None, None
    except Exception as e:
        logger.warning(f"Failed to load pre-trained model: {e}")
        return None, None
