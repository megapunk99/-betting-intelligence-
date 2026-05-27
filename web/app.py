"""
FastAPI web application for Betting Intelligence.
A proper web GUI replacing the Streamlit dashboard.

Run:
    betting-intel web start
    # or
    uvicorn web.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

app = FastAPI(
    title="Betting Intelligence",
    description="AI-powered basketball betting recommendations",
    version="0.2.0",
)

# Mount static files
static_dir = HERE / "static"
static_dir.mkdir(parents=True, exist_ok=True)
(static_dir / "css").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates = Jinja2Templates(directory=str(HERE / "templates"))


# ── Engine Integration ─────────────────────────────────────────────────────

# Simple TTL cache: regenerates data every 30 seconds
_cache: dict[str, Any] = {"data": None, "timestamp": 0.0}
CACHE_TTL = 30.0  # seconds


def _now():
    import time
    return time.time()


def get_engine():
    """Lazy-load the recommendation engine."""
    from betting_intel.recommendations import RecommendationEngine
    return RecommendationEngine()


def load_dashboard_data(force_refresh: bool = False):
    """Load all data needed for the dashboard with 30s TTL caching."""
    now = _now()
    if not force_refresh and _cache["data"] is not None and (now - _cache["timestamp"]) < CACHE_TTL:
        return _cache["data"]

    engine = get_engine()
    all_bets = engine.generate_all_bets()
    clear_picks = engine.get_clear_picks()
    summary = engine.get_summary()
    todays = engine.get_todays_card()

    # Group today's bets by game
    games = {}
    for bet in todays:
        key = bet.matchup
        if key not in games:
            games[key] = {"league": bet.league, "bets": []}
        games[key]["bets"].append(bet)

    data = {
        "all_bets": all_bets,
        "clear_picks": clear_picks,
        "summary": summary,
        "todays_bets": todays,
        "games": games,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _cache["data"] = data
    _cache["timestamp"] = now
    return data


def bet_to_dict(bet) -> dict:
    """Convert a BetSuggestion to a plain dict for templates."""
    return {
        "game_id": bet.game_id,
        "game_date": bet.game_date,
        "matchup": bet.matchup,
        "league": bet.league,
        "bet_type": bet.bet_type.value,
        "bet_type_display": bet.bet_type.display_name(),
        "bet_type_icon": bet.bet_type.icon(),
        "bet_side": bet.bet_side,
        "action": bet.action,
        "edge_pct": round(bet.edge_pct * 100, 1),  # percentage
        "stake_dollars": round(bet.stake_dollars, 0),
        "kelly_fraction": round(bet.kelly_fraction, 4),
        "win_probability": round(bet.win_probability, 3),
        "confidence": bet.confidence.value,
        "is_clear_pick": bet.is_clear_pick,
        "reasoning": bet.reasoning,
        "model_name": bet.model_name,
        "tags": bet.tags,
        "market_line": bet.market_line,
        "predicted_value": round(bet.predicted_value, 1),
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    data = load_dashboard_data()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "summary": data["summary"],
            "clear_picks": data["clear_picks"],
            "todays_bets": data["todays_bets"],
            "games": data["games"],
            "generated_at": data["generated_at"],
            "today": date.today().isoformat(),
        },
    )


@app.get("/clear-picks", response_class=HTMLResponse)
async def clear_picks_page(
    request: Request,
    league: str = "all",
    bet_type: str = "all",
    min_edge: float = 0.0,
):
    """Clear picks page with HTMX filtering."""
    data = load_dashboard_data()
    picks = data["clear_picks"]

    # Filter
    filtered = []
    for cp in picks:
        bet = cp.bet
        if league != "all" and bet.league.lower() != league.lower():
            continue
        if bet_type != "all" and bet.bet_type.value != bet_type:
            continue
        if bet.edge_pct < min_edge:
            continue
        filtered.append(cp)

    leagues = sorted(set(cp.bet.league for cp in picks))
    types = sorted(set(cp.bet.bet_type.value for cp in picks))

    # Check if HTMX request (partial update)
    is_htmx = request.headers.get("HX-Request") == "true"

    template = "partials/clear_picks_list.html" if is_htmx else "clear_picks.html"
    return templates.TemplateResponse(
        template,
        {
            "request": request,
            "clear_picks": filtered,
            "leagues": leagues,
            "types": types,
            "selected_league": league,
            "selected_type": bet_type,
            "min_edge": min_edge,
            "summary": data["summary"],
            "generated_at": data["generated_at"],
        },
    )


@app.get("/all-bets", response_class=HTMLResponse)
async def all_bets_page(
    request: Request,
    league: str = "all",
    bet_type: str = "all",
    min_edge: float = 0.0,
    sort_by: str = "edge",
):
    """All bets page with sorting and filtering."""
    data = load_dashboard_data()
    bets = data["all_bets"]

    # Filter
    if league != "all":
        bets = [b for b in bets if b.league.lower() == league.lower()]
    if bet_type != "all":
        bets = [b for b in bets if b.bet_type.value == bet_type]
    if min_edge > 0:
        bets = [b for b in bets if b.edge_pct >= min_edge]

    # Sort
    if sort_by == "edge":
        bets.sort(key=lambda b: b.edge_pct, reverse=True)
    elif sort_by == "stake":
        bets.sort(key=lambda b: b.stake_dollars, reverse=True)
    elif sort_by == "confidence":
        bets.sort(key=lambda b: b.confidence.numeric(), reverse=True)

    leagues = sorted(set(b.league for b in data["all_bets"]))
    types = sorted(set(b.bet_type.value for b in data["all_bets"]))

    is_htmx = request.headers.get("HX-Request") == "true"
    template = "partials/bets_table.html" if is_htmx else "all_bets.html"

    return templates.TemplateResponse(
        template,
        {
            "request": request,
            "bets": [bet_to_dict(b) for b in bets],
            "leagues": leagues,
            "types": types,
            "selected_league": league,
            "selected_type": bet_type,
            "min_edge": min_edge,
            "sort_by": sort_by,
            "total": len(bets),
            "summary": data["summary"],
            "generated_at": data["generated_at"],
        },
    )


@app.get("/todays-card", response_class=HTMLResponse)
async def todays_card_page(request: Request):
    """Today's betting card."""
    data = load_dashboard_data()
    return templates.TemplateResponse(
        "todays_card.html",
        {
            "request": request,
            "games": data["games"],
            "todays_bets": data["todays_bets"],
            "summary": data["summary"],
            "generated_at": data["generated_at"],
            "today": date.today().isoformat(),
        },
    )


@app.get("/tomorrow", response_class=HTMLResponse)
async def tomorrow_page(request: Request):
    """Tomorrow's betting card — one-day-ahead predictions."""
    engine = get_engine()
    tomorrow_bets = engine.get_tomorrows_card()
    summary = engine.get_summary()

    # Group by game
    games = {}
    for bet in tomorrow_bets:
        key = bet.matchup
        if key not in games:
            games[key] = {"league": bet.league, "series": "", "bets": []}
        games[key]["bets"].append(bet)

    tomorrow_date = (date.today() + timedelta(days=1)).isoformat()

    return templates.TemplateResponse(
        "tomorrow.html",
        {
            "request": request,
            "games": games,
            "tomorrow_bets": tomorrow_bets,
            "summary": summary,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tomorrow": tomorrow_date,
            "today": date.today().isoformat(),
        },
    )


@app.get("/player-props", response_class=HTMLResponse)
async def player_props_page(
    request: Request,
    home: str = "Spurs",
    away: str = "Thunder",
    league: str = "NBA",
):
    """Player props page."""
    from betting_intel.recommendations.player_props import PlayerPropEngine

    engine = PlayerPropEngine()
    props = engine.predict_for_game(home=home, away=away, league=league)
    props_dicts = [bet_to_dict(p) for p in props]

    # Group by team
    home_props = [p for p in props_dicts if home in p["bet_side"]]
    away_props = [p for p in props_dicts if away in p["bet_side"]]

    return templates.TemplateResponse(
        "player_props.html",
        {
            "request": request,
            "home_team": home,
            "away_team": away,
            "league": league,
            "home_props": home_props,
            "away_props": away_props,
            "total": len(props),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


@app.get("/api/refresh")
async def api_refresh():
    """Refresh cached data and return summary JSON."""
    data = load_dashboard_data(force_refresh=True)
    return JSONResponse(content={
        "total_bets": data["summary"]["total_bets"],
        "clear_picks": data["summary"]["clear_picks"],
        "games_available": data["summary"]["games_available"],
        "avg_edge": round(data["summary"]["avg_edge"] * 100, 1),
        "total_stake": round(data["summary"]["total_stake"], 0),
        "generated_at": data["generated_at"],
    })


@app.get("/api/bets")
async def api_bets(
    league: str = "all",
    bet_type: str = "all",
    min_edge: float = 0.0,
    limit: int = 50,
):
    """JSON API for bets."""
    data = load_dashboard_data()
    bets = data["all_bets"]
    if league != "all":
        bets = [b for b in bets if b.league.lower() == league.lower()]
    if bet_type != "all":
        bets = [b for b in bets if b.bet_type.value == bet_type]
    if min_edge > 0:
        bets = [b for b in bets if b.edge_pct >= min_edge]
    bets.sort(key=lambda b: b.edge_pct, reverse=True)
    return JSONResponse(content=[bet_to_dict(b) for b in bets[:limit]])


@app.get("/api/clear-picks")
async def api_clear_picks():
    """JSON API for clear picks."""
    data = load_dashboard_data()
    return JSONResponse(content=[
        {
            "bet": bet_to_dict(cp.bet),
            "clear_score": round(cp.clear_score, 1),
            "risk_level": cp.risk_level,
            "reasons": cp.reasons,
        }
        for cp in data["clear_picks"]
    ])


# ── Run ────────────────────────────────────────────────────────────────────

def run(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the web server via uvicorn."""
    import uvicorn
    uvicorn.run(
        "web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run()
