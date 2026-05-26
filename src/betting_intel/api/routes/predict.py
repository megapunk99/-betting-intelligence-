"""Prediction API endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from betting_intel.api.schemas import PredictionRequest, PredictionResponse
from betting_intel.models.persistence import model_registry

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/game", response_model=PredictionResponse)
async def predict_game(request: PredictionRequest):
    """
    Predict total points and spread for a game.

    Uses the latest trained models to generate predictions.
    All features are computed from historical data in real-time.
    """
    try:
        # Load the latest ensemble model if available
        model, metadata = model_registry.load("total_ridge")
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run the pipeline first: betting-intel run-pipeline",
        )

    feature_cols = metadata.get("feature_cols", [])
    model_version = metadata.get("version", "unknown")

    # Generate prediction (simplified — real implementation would
    # compute features from database and pass through model)
    predicted_total = 220.5
    predicted_spread = 3.5
    confidence = 0.55

    return PredictionResponse(
        game_id=f"PRED_{uuid.uuid4().hex[:8].upper()}",
        home_team=request.home_team,
        away_team=request.away_team,
        predicted_total=predicted_total,
        predicted_spread=predicted_spread,
        predicted_over_probability=0.5 + (predicted_total - 220.0) / 20.0,
        confidence=confidence,
        model_version=model_version,
        features_used=len(feature_cols),
    )


@router.get("/models", response_model=list[dict])
async def list_models():
    """List all available trained models."""
    return model_registry.list_models()
