"""Prediction API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from betting_intel.api.schemas import PredictionRequest, PredictionResponse
from betting_intel.models.persistence import model_registry

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/game")
async def predict_game(request: PredictionRequest):
    """
    Predict total points and spread for a game.

    NOTE: This endpoint requires a fully trained model pipeline to return
    real predictions. The model_registry.load() call below will raise
    FileNotFoundError if no model has been trained yet.

    To train the pipeline, run:
        betting-intel run-pipeline

    Or use the live prediction dashboard at / for real-time predictions
    from the LivePredictionEngine.
    """
    try:
        # Load the latest ensemble model if available
        model, metadata = model_registry.load("total_ridge")
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model available. "
                "Run the pipeline first: betting-intel run-pipeline\n\n"
                "Alternatively, use the live prediction dashboard at / "
                "for real-time predictions from the LivePredictionEngine."
            ),
        )

    feature_cols = metadata.get("feature_cols", [])
    model_version = metadata.get("version", "unknown")

    # Actual prediction requires feature engineering from historical data.
    # The model_registry stored model expects a feature vector built by
    # the pipeline. Use the live engine endpoint at /api/predictions for
    # real predictions on upcoming games.
    raise HTTPException(
        status_code=501,
        detail=(
            "Real-time per-game prediction is not yet implemented. "
            f"Model '{metadata.get('name', 'total_ridge')}' v{model_version} is available "
            f"with {len(feature_cols)} features, but feature computation for "
            "arbitrary matchups at request time is not wired up.\n\n"
            "Use the live prediction dashboard at / or the /api/predictions "
            "endpoint for predictions on upcoming games."
        ),
    )


@router.get("/models", response_model=list[dict])
async def list_models():
    """List all available trained models."""
    return model_registry.list_models()
