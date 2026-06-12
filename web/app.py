"""
Betting Intelligence — Web Dashboard.

A FastAPI app that loads predictions from forward_test_results.json
and the LivePredictionEngine for real-time odds.

Run:
    uvicorn web.app:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

import asyncio

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FORWARD_TEST_JSON = PROJECT_ROOT / "data" / "forward_test_results.json"

# ── App ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Betting Intelligence", version="0.3.0")

static_dir = HERE / "static"
static_dir.mkdir(parents=True, exist_ok=True)
(static_dir / "css").mkdir(exist_ok=True)
(static_dir / "js").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))
templates.env.globals["api_key"] = os.getenv("API_KEY", "change-me-to-a-random-secret")


# ── Live Prediction Engine Singleton ────────────────────────────────────

_engine_lock = Lock()
_live_engine: Optional["LivePredictionEngine"] = None


def get_live_engine() -> "LivePredictionEngine":
    """Get or create the singleton LivePredictionEngine instance."""
    global _live_engine
    if _live_engine is None:
        with _engine_lock:
            if _live_engine is None:
                from betting_intel.live.engine import LivePredictionEngine
                _live_engine = LivePredictionEngine()
    return _live_engine


# ── Helper: LiveGame → bet dict ─────────────────────────────────────────

def _livegame_to_bet(game: Any) -> dict:
    """Convert a LiveGame (or dict) to a bet dict for API responses."""
    if hasattr(game, "to_dict"):
        g = game.to_dict()
        # Properties like 'matchup' are not included in asdict() — compute them explicitly
        matchup = getattr(game, "matchup", f"{getattr(game, 'away_team_short', '?')} @ {getattr(game, 'home_team_short', '?')}")
        direction = getattr(game, "direction", g.get("direction", "neutral")) or "neutral"
        confidence = getattr(game, "confidence", g.get("confidence", "low")) or "low"
        edge_pct = getattr(game, "edge_pct", g.get("edge_pct", 0.0)) or 0.0
        predicted_total = getattr(game, "predicted_total", g.get("predicted_total"))
        market_total = getattr(game, "market_total", g.get("market_total"))
    else:
        g = game
        matchup = g.get("matchup", f"{g.get('away_team_short', '?')} @ {g.get('home_team_short', '?')}")
        direction = g.get("direction", "neutral") or "neutral"
        confidence = g.get("confidence", "low") or "low"
        edge_pct = g.get("edge_pct", 0.0) or 0.0
        predicted_total = g.get("predicted_total")
        market_total = g.get("market_total")

    return {
        "game_id": g.get("game_id", ""),
        "game_date": g.get("game_date", ""),
        "matchup": matchup,
        "home_team": g.get("home_team", ""),
        "away_team": g.get("away_team", ""),
        "league": "NBA",
        "bet_type": "total",
        "bet_type_display": "Total",
        "bet_side": f"Total {'OVER' if direction == 'over' else 'UNDER'} {market_total or ''}",
        "edge_pct": edge_pct,
        "stake_dollars": 0.0,
        "confidence": confidence,
        "edge_confidence": confidence,
        "is_clear_pick": False,
        "reasoning": f"Model predicts {predicted_total or '?'} vs market {market_total or '?'}",
        "model_name": "pipeline_ensemble",
        "sport_key": g.get("sport_key", "basketball_nba"),
        "sport_title": "NBA",
        "commence_time": g.get("commence_time", ""),
        "home_team_short": g.get("home_team_short", ""),
        "away_team_short": g.get("away_team_short", ""),
        "home_ml": g.get("home_ml"),
        "away_ml": g.get("away_ml"),
        "market_total": market_total,
        "predicted_total": predicted_total,
        "direction": direction,
        "n_books_ml": g.get("n_books_ml", 0),
        "n_books_total": g.get("n_books_total", 0),
    }


def _livegame_to_clear_pick(game: Any) -> dict:
    """Convert a LiveGame to a clear-pick dict for API responses."""
    bet = _livegame_to_bet(game)
    return {
        "bet": bet,
        "clear_score": abs(bet.get("edge_pct", 0) or 0) * 100,
        "risk_level": "low" if abs(bet.get("edge_pct", 0) or 0) > 0.05 else "medium",
        "reasons": [
            f"Predicted total: {bet.get('predicted_total')}",
            f"Market total: {bet.get('market_total')}",
            f"Edge: {bet.get('edge_pct', 0):.1%}",
        ],
    }


# ── Data Loading (from forward_test_results.json) ──────────────────────

def load_predictions() -> dict:
    """Load predictions from forward_test_results.json. Returns empty data on failure."""
    empty = {
        "bets": [], "clear_picks": [], "summary": {},
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_bets": 0, "n_games": 0, "n_clear": 0,
    }

    if not FORWARD_TEST_JSON.exists():
        return empty

    try:
        with open(FORWARD_TEST_JSON) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load predictions: {e}")
        return empty

    bets = data.get("all_bets", [])
    clear = [b for b in bets if b.get("is_clear_pick", False)]
    summary = data.get("summary", {})
    games = len(set(b.get("matchup", "") for b in bets))

    avg_edge = summary.get("avg_edge", 0)
    avg_edge_str = f"{avg_edge:.1%}" if avg_edge else "—"

    return {
        "bets": bets,
        "clear_picks": clear[:10],
        "summary": summary,
        "avg_edge": avg_edge_str,
        "generated_at": data.get("generated_at", ""),
        "n_bets": len(bets),
        "n_games": games,
        "n_clear": len(clear),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  LIVE ENGINE ROUTES
# ═══════════════════════════════════════════════════════════════════════════


@app.post("/api/live/refresh")
async def live_refresh():
    """Force-refresh live predictions from odds sources."""
    try:
        engine = get_live_engine()
        snapshot = engine.refresh_now()
        return JSONResponse(content={
            "n_total": snapshot.n_total,
            "n_today": snapshot.n_today,
            "n_tomorrow": snapshot.n_tomorrow,
            "n_live": snapshot.n_live,
            "fresh_odds": snapshot.fresh_odds,
            "refreshed": True,
            "generated_at": snapshot.generated_at,
        })
    except Exception as e:
        logger.error(f"Live refresh failed: {e}")
        return JSONResponse(content={
            "n_total": 0, "n_today": 0, "n_tomorrow": 0,
            "n_live": 0, "fresh_odds": False, "refreshed": False,
            "error": str(e),
        })


@app.get("/api/live/snapshot")
async def live_snapshot():
    """Get the current live prediction snapshot."""
    try:
        engine = get_live_engine()
        snapshot = engine.get_snapshot(force_refresh=False)
        return JSONResponse(content=snapshot.to_dict())
    except Exception as e:
        logger.error(f"Snapshot failed: {e}")
        return JSONResponse(content={
            "n_total": 0, "n_today": 0, "n_tomorrow": 0, "n_live": 0,
            "fresh_odds": False, "next_two_days": [],
            "live_games": [], "today_games": [], "tomorrow_games": [],
            "generated_at": datetime.now().isoformat(),
        })


@app.get("/api/live/chart-data")
async def live_chart_data():
    """Get pre-computed chart data for the dashboard."""
    try:
        engine = get_live_engine()
        snapshot = engine.get_snapshot(force_refresh=False)
        if snapshot.chart_data:
            return JSONResponse(content=snapshot.chart_data)
        return JSONResponse(content={
            "n_total": 0, "n_today": 0, "n_tomorrow": 0,
            "edges": [], "confidence_breakdown": {},
            "direction_breakdown": {},
            "generated_at": datetime.now().isoformat(),
            "fresh_odds": False,
        })
    except Exception as e:
        logger.error(f"Chart data failed: {e}")
        return JSONResponse(content={
            "n_total": 0, "edges": [], "confidence_breakdown": {},
            "direction_breakdown": {},
            "generated_at": datetime.now().isoformat(),
            "fresh_odds": False,
        })


@app.get("/api/live/games")
async def live_games():
    """Get the list of live games."""
    try:
        engine = get_live_engine()
        games = engine.get_live_games(force_refresh=False)
        return JSONResponse(content=[g.to_dict() for g in games])
    except Exception as e:
        logger.error(f"Live games failed: {e}")
        return JSONResponse(content=[])


@app.post("/api/live/clear-cache")
async def live_clear_cache():
    """Clear all cached predictions and odds."""
    try:
        engine = get_live_engine()
        engine.clear_cache()
        return JSONResponse(content={"cleared": True})
    except Exception as e:
        logger.error(f"Clear cache failed: {e}")
        return JSONResponse(content={"cleared": False, "error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════
#  HTML PAGE ROUTES
# ═══════════════════════════════════════════════════════════════════════════


def _games_context() -> dict:
    """Build template context from live engine games."""
    try:
        engine = get_live_engine()
        # Use get_next_two_days to get games from the snapshot
        games = engine.get_next_two_days(force_refresh=False)
        logger.debug(f"_games_context: {len(games)} games from engine")
        bets = [_livegame_to_bet(g) for g in games]
        clear = [_livegame_to_bet(g) for g in games
                 if abs((g.edge_pct or 0)) > 0.03]
        return {
            "bets": bets,
            "clear_picks": clear[:10],
            "summary": {
                "n_games": len(games),
                "n_bets": len(bets),
                "n_clear": len(clear),
            },
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_bets": len(bets),
            "n_games": len(games),
            "n_clear": len(clear),
            "today": date.today().isoformat(),
        }
    except Exception as e:
        logger.warning(f"_games_context failed: {e}")
        return {
            "bets": [], "clear_picks": [], "summary": {},
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_bets": 0, "n_games": 0, "n_clear": 0,
            "today": date.today().isoformat(),
        }


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard — shows predictions from forward_test_results.json."""
    data = load_predictions()
    return templates.TemplateResponse(request, "index.html", {
        "bets": data["bets"],
        "clear_picks": data["clear_picks"],
        "summary": data["summary"],
        "generated_at": data["generated_at"],
        "n_bets": data["n_bets"],
        "n_games": data["n_games"],
        "n_clear": data["n_clear"],
        "today": date.today().isoformat(),
    })


# Note: All duplicate page routes (/dashboard, /live, /todays-card, /tomorrow,
# /pre-match-prediction, /all-bets, /clear-picks) have been consolidated
# into the single / route to eliminate confusion.


# ═══════════════════════════════════════════════════════════════════════════
#  FUTURE PREDICTIONS ROUTE
# ═══════════════════════════════════════════════════════════════════════════

_future_predictor = None
_future_predictor_lock = Lock()


def get_future_predictor() -> Optional[Any]:
    """Get or create the FutureGamePredictor singleton."""
    global _future_predictor
    if _future_predictor is None:
        with _future_predictor_lock:
            if _future_predictor is None:
                try:
                    from betting_intel.live.future_predictor import FutureGamePredictor
                    predictor = FutureGamePredictor()
                    predictor.load()
                    _future_predictor = predictor
                except Exception as e:
                    logger.warning(f"Failed to load FutureGamePredictor: {e}")
                    return None
    return _future_predictor


@app.get("/api/future-predictions")
async def api_future_predictions(num_games: int = 6):
    """JSON API — returns AI-predicted future games with quarter/half breakdowns."""
    try:
        predictor = get_future_predictor()
        if predictor is None:
            return JSONResponse(content={
                "predictions": [],
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "unavailable",
                "message": "FutureGamePredictor failed to load — check NBA database and model files",
            })
        preds = predictor.predict_upcoming_games(num_games=max(1, min(num_games, 30)))
        return JSONResponse(content={
            "predictions": preds,
            "n_predictions": len(preds),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "ok",
        })
    except Exception as e:
        logger.error(f"Future predictions API failed: {e}")
        return JSONResponse(content={
            "predictions": [],
            "n_predictions": 0,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "error",
            "error": str(e),
        })


@app.get("/future-predictions", response_class=HTMLResponse)
async def future_predictions_page(request: Request):
    """Future predictions page — AI-generated game predictions with quarter/half breakdowns."""
    try:
        predictor = get_future_predictor()
        if predictor:
            try:
                preds = predictor.predict_upcoming_games(num_games=20)
            except Exception:
                preds = []
        else:
            preds = []
    except Exception:
        preds = []

    return templates.TemplateResponse(request, "future_predictions.html", {
        "predictions": preds,
        "n_predictions": len(preds),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "today": date.today().isoformat(),
    })


# ═══════════════════════════════════════════════════════════════════════════
#  JSON API ROUTES
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/api/predictions")
async def api_predictions(limit: int = 50, min_edge: float = 0.0):
    """JSON API — returns predictions sorted by edge descending."""
    data = load_predictions()
    bets = data["bets"]
    if min_edge > 0:
        bets = [b for b in bets if abs(b.get("edge_pct", 0)) >= min_edge]
    bets.sort(key=lambda b: abs(b.get("edge_pct", 0)), reverse=True)
    return JSONResponse(content={
        "n_bets": len(bets[:limit]),
        "generated_at": data["generated_at"],
        "bets": bets[:limit],
    })


@app.get("/api/bets")
async def api_bets():
    """JSON API — returns all bets from the live engine as a flat list."""
    try:
        engine = get_live_engine()
        games = engine.get_next_two_days(force_refresh=False)
        return JSONResponse(content=[_livegame_to_bet(g) for g in games])
    except Exception:
        return JSONResponse(content=[])


@app.get("/api/clear-picks")
async def api_clear_picks():
    """JSON API — returns clear picks from the live engine."""
    try:
        engine = get_live_engine()
        games = engine.get_next_two_days(force_refresh=False)
        picks = []
        for g in games:
            edge = abs(g.edge_pct or 0)
            if edge > 0.03:  # Only strong edges
                picks.append(_livegame_to_clear_pick(g))
        return JSONResponse(content=picks)
    except Exception:
        return JSONResponse(content=[])


@app.get("/api/refresh")
async def api_refresh():
    """JSON API — refresh predictions and return summary."""
    try:
        engine = get_live_engine()
        snapshot = engine.refresh_now()
        bets = [_livegame_to_bet(g) for g in snapshot.next_two_days]
        total_stake = 0.0  # No stake info from live engine directly
        return JSONResponse(content={
            "total_bets": len(bets),
            "games_available": snapshot.n_total,
            "total_stake": total_stake,
            "generated_at": snapshot.generated_at,
            "n_today": snapshot.n_today,
            "n_tomorrow": snapshot.n_tomorrow,
            "refreshed": True,
        })
    except Exception as e:
        logger.error(f"Refresh API failed: {e}")
        return JSONResponse(content={
            "total_bets": 0, "games_available": 0,
            "total_stake": 0, "generated_at": datetime.now().isoformat(),
            "refreshed": False,
        })


@app.get("/api/resolve")
async def api_resolve():
    """Resolve predictions against actual results and return updated data."""
    try:
        from betting_intel.analytics.tracker import ResultsTracker
        tracker = ResultsTracker()
        n = tracker.resolve_all()
        data = load_predictions()
        return JSONResponse(content={
            "resolved": n,
            "message": f"Resolved {n} predictions",
            "n_bets": data["n_bets"],
            "n_clear": data["n_clear"],
            "generated_at": data["generated_at"],
        })
    except Exception as e:
        logger.error(f"Resolve failed: {e}")
        return JSONResponse(content={"error": str(e), "resolved": 0}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════════
#  WEBSOCKET LIVE REFRESH
# ═══════════════════════════════════════════════════════════════════════════


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """WebSocket endpoint for live dashboard refresh.

    On connection, immediately sends the current prediction data.
    Then pushes updated data every 30 seconds to all connected clients.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Send current predictions
            try:
                data = load_predictions()
                await websocket.send_json({
                    "type": "predictions",
                    "n_bets": data["n_bets"],
                    "n_games": data["n_games"],
                    "n_clear": data["n_clear"],
                    "generated_at": data["generated_at"],
                })
            except Exception as e:
                logger.warning(f"WebSocket data error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

            # Wait 30 seconds before next push
            await asyncio.sleep(30)

            # Try to refresh prediction data from file
            try:
                # Re-read forward_test_results.json — file may have been updated
                # by the background pipeline
                pass
            except Exception:
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  HEALTH ROUTES
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/api/health")
async def api_health():
    """Lightweight health check."""
    db_ok = Path("data/nba_data.db").exists()
    predictions_ok = FORWARD_TEST_JSON.exists()
    return JSONResponse(content={
        "status": "ok" if db_ok else "degraded",
        "database": db_ok,
        "predictions_file": predictions_ok,
        "version": "0.3.0",
    })


@app.get("/api/health/live")
async def api_health_live():
    """Live health check — is the live engine running?"""
    try:
        engine = get_live_engine()
        has_data = engine.has_cached_data
        return JSONResponse(content={
            "status": "ok" if has_data else "degraded",
            "cached_data": has_data,
        })
    except Exception:
        return JSONResponse(content={
            "status": "degraded",
            "cached_data": False,
        })


@app.get("/api/health/ready")
async def api_health_ready():
    """Ready check — is the app ready to serve requests?"""
    return JSONResponse(content={
        "status": "ok",
        "ready": True,
        "version": "0.3.0",
    })


# ── Run ─────────────────────────────────────────────────────────────────

def run(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    import uvicorn
    uvicorn.run("web.app:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    run()
