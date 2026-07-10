"""Prediction API endpoints — wired to LivePredictionEngine and FutureGamePredictor."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from betting_intel.api.schemas import PredictionRequest
from betting_intel.models.persistence import model_registry

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Lazy-initialized singletons (avoid creating a new engine/predictor per request)
_live_engine: Optional["LivePredictionEngine"] = None
_future_predictor: Optional["FutureGamePredictor"] = None


def _get_engine():
    """Get or create the singleton LivePredictionEngine (lazy-init)."""
    global _live_engine
    if _live_engine is None:
        from betting_intel.live.engine import LivePredictionEngine

        _live_engine = LivePredictionEngine()
    return _live_engine


def _get_future_predictor():
    """Get or create the singleton FutureGamePredictor (lazy-init)."""
    global _future_predictor
    if _future_predictor is None:
        from betting_intel.live.future_predictor import FutureGamePredictor

        _future_predictor = FutureGamePredictor()
        _future_predictor.load()
    return _future_predictor


@router.post("/game")
async def predict_game(request: PredictionRequest):
    """
    Predict a specific game matchup.

    Checks the LivePredictionEngine's current snapshot first (upcoming games
    with real market odds). If the matchup isn't there, falls back to the
    FutureGamePredictor which uses stat-based predictions.

    For arbitrary matchups without real odds, see GET /predict/upcoming
    or the live dashboard at / for all current predictions.
    """
    # Try LivePredictionEngine first (has real market odds + ML models)
    try:
        engine = _get_engine()
        snapshot = engine.get_snapshot(force_refresh=False)
        all_games = snapshot.next_two_days

        for game in all_games:
            if (
                game.home_team_short.lower() in request.home_team.lower()
                or request.home_team.lower() in game.home_team_short.lower()
            ) and (
                game.away_team_short.lower() in request.away_team.lower()
                or request.away_team.lower() in game.away_team_short.lower()
            ):
                return {
                    "game_id": game.game_id,
                    "home_team": game.home_team_short,
                    "away_team": game.away_team_short,
                    "matchup": game.matchup,
                    "game_date": game.game_date,
                    "predicted_total": game.predicted_total,
                    "edge_pct": game.edge_pct,
                    "direction": game.direction,
                    "confidence": game.confidence,
                    "market_total": game.market_total,
                    "total_prediction": game.total_prediction,
                    "total_edge_pct": game.total_edge_pct,
                    "total_direction": game.total_direction,
                    "total_confidence": game.total_confidence,
                    "stake_dollars": game.stake_dollars,
                    "source": "live_engine",
                    "predicted_at": game.predicted_at,
                    "feature_importance": game.feature_importance,
                }
    except Exception:
        pass

    # Fall back to FutureGamePredictor (stat-based predictions)
    try:
        predictor = _get_future_predictor()
        upcoming = predictor.predict_upcoming_games(num_games=30)
        for pred in upcoming:
            home = pred.get("home_team_short", "").lower()
            away = pred.get("away_team_short", "").lower()
            if (
                home in request.home_team.lower() or request.home_team.lower() in home
            ) and (
                away in request.away_team.lower() or request.away_team.lower() in away
            ):
                return {
                    **pred,
                    "source": "future_predictor",
                }
    except Exception:
        pass

    raise HTTPException(
        status_code=404,
        detail=(
            f"No prediction found for {request.home_team} vs {request.away_team}. "
            "Check team names (use short names like 'Celtics', 'Lakers') or "
            "use GET /predict/upcoming to see all available predictions. "
            "To refresh odds, call POST /api/live/refresh first."
        ),
    )


@router.get("/models")
async def list_models():
    """List all available trained models from the registry."""
    return model_registry.list_models()


@router.get("/upcoming")
async def list_upcoming_predictions(
    num_games: int = Query(
        20, description="Number of predictions to return", ge=1, le=50
    ),
    source: Optional[str] = Query(
        None, description="Source: 'engine', 'future', or None for both"
    ),
):
    """
    List upcoming game predictions from the LivePredictionEngine + FutureGamePredictor.

    Returns predictions for the next 2 days from the engine (real market odds + ML)
    and predictions for the next 14 days from the FutureGamePredictor (stat-based).
    """
    results: dict[str, list] = {
        "live_engine": [],
        "future_predictor": [],
    }

    # LivePredictionEngine (next 2 days, real odds + ML)
    if source is None or source == "engine":
        try:
            engine = _get_engine()
            snapshot = engine.get_snapshot(force_refresh=False)
            results["live_engine"] = [
                {
                    "game_id": g.game_id,
                    "matchup": g.matchup,
                    "game_date": g.game_date,
                    "league": g.league,
                    "home_team": g.home_team_short,
                    "away_team": g.away_team_short,
                    "predicted_total": g.predicted_total,
                    "market_total": g.market_total,
                    "edge_pct": g.edge_pct,
                    "direction": g.direction,
                    "confidence": g.confidence,
                    "total_prediction": g.total_prediction,
                    "total_edge_pct": g.total_edge_pct,
                    "total_direction": g.total_direction,
                    "total_confidence": g.total_confidence,
                    "stake_dollars": g.stake_dollars,
                    "is_live": g.is_live,
                    "home_ml": g.home_ml,
                    "away_ml": g.away_ml,
                    "spread": g.spread,
                    "n_books_ml": g.n_books_ml,
                    "source": "live_engine",
                }
                for g in snapshot.next_two_days
            ][:num_games]
        except Exception:
            pass

    # FutureGamePredictor (next 14 days, stat-based)
    if source is None or source == "future":
        try:
            predictor = _get_future_predictor()
            results["future_predictor"] = predictor.predict_upcoming_games(
                num_games=num_games
            )
        except Exception:
            pass

    # Combine for a unified view
    combined = list(results["live_engine"])
    seen_ids = set(g.get("game_id", "") for g in combined)
    for g in results["future_predictor"]:
        if g.get("game_id") not in seen_ids:
            g["source"] = "future_predictor"
            combined.append(g)
            seen_ids.add(g.get("game_id", ""))

    return {
        "n_predictions": len(combined),
        "predictions": combined,
        "sources": {k: len(v) for k, v in results.items()},
        "generated_at": datetime.now().isoformat(),
    }
