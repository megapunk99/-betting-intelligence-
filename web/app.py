"""
Betting Intelligence — Web Dashboard.

A FastAPI app that uses the LivePredictionEngine to fetch real NBA games
and predictions from TheOddsAPI + ESPN. Shows live predictions for the
next 2 days with ML-powered edge analysis.

Run:
    uvicorn web.app:app --reload --port 8000
"""

from __future__ import annotations

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
import csv
import io
import json
import re

from betting_intel.live.sport_configs import sport_key_to_group


# ── Feature name → human-readable description mapping ────────────────────
# Maps model feature column names to what they mean for the dashboard.
# The model returns feature_importance as {column_name: importance_weight}.
# This function converts column names like "avg_pts_5g_home" into
# readable descriptions like "Scoring avg (L5)".

_FEATURE_DESCRIPTIONS: dict[str, str] = {
    # Short stat keys (used by diff features like "pts_diff_5g")
    "pts": "Scoring",
    "reb": "Rebounding",
    "ast": "Assists",
    "stl": "Steals",
    "blk": "Blocks",
    "tov": "Turnovers",
    "pf": "Fouls",
    "fgm": "FGM",
    "fga": "FGA",
    "fg3m": "3PTM",
    "fg3a": "3PTA",
    "ftm": "FTM",
    "fta": "FTA",
    "oreb": "OREB",
    "dreb": "DREB",
    "efg": "eFG %",
    "pm": "Net rating",
    "pace": "Pace",

    # Rolling averages (points, rebounds, assists, etc.)
    "avg_pts": "Scoring avg",
    "avg_reb": "Rebounding avg",
    "avg_ast": "Assists avg",
    "avg_stl": "Steals avg",
    "avg_blk": "Blocks avg",
    "avg_tov": "Turnovers avg",
    "avg_pf": "Fouls avg",
    "avg_oreb": "Off. rebounding",
    "avg_dreb": "Def. rebounding",
    "avg_fgm": "Field goals made",
    "avg_fga": "Field goals attempted",
    "avg_fg3m": "3PT made",
    "avg_fg3a": "3PT attempted",
    "avg_ftm": "FT made",
    "avg_fta": "FT attempted",
    "avg_pts_allowed": "Defense: pts allowed",
    "avg_reb_allowed": "Defense: rebounding",
    "avg_ast_allowed": "Defense: assists",
    "avg_stl_allowed": "Defense: steals",
    "avg_blk_allowed": "Defense: blocks",
    "avg_tov_allowed": "Defense: turnovers forced",
    "avg_pf_allowed": "Defense: fouls",
    "avg_fgm_allowed": "Defense: FGM allowed",
    "avg_fga_allowed": "Defense: FGA allowed",
    "avg_fg3m_allowed": "Defense: 3PT allowed",
    "avg_fg3a_allowed": "Defense: 3PTA allowed",
    "avg_ftm_allowed": "Defense: FTM allowed",
    "avg_fta_allowed": "Defense: FTA allowed",
    "avg_oreb_allowed": "Defense: OREB allowed",
    "avg_dreb_allowed": "Defense: DREB allowed",
    "avg_fg3_pct": "3PT %",
    "avg_ft_pct": "FT %",
    "avg_efg": "eFG %",
    "avg_pace": "Pace",
    "avg_pm": "Net rating",
    "avg_margin": "Avg margin",

    # EMA (exponential moving average)
    "ema_pts": "Weighted scoring",
    "ema_reb": "Weighted rebounding",
    "ema_ast": "Weighted assists",
    "ema_stl": "Weighted steals",
    "ema_blk": "Weighted blocks",
    "ema_tov": "Weighted turnovers",
    "ema_pf": "Weighted fouls",
    "ema_fgm": "Weighted FGM",
    "ema_fga": "Weighted FGA",
    "ema_pm": "Weighted net rating",
    "ema_margin": "Weighted margin",

    # Trends
    "trend_pts": "Scoring trend",
    "trend_pm": "Net rating trend",
    "trend_reb": "Rebounding trend",
    "trend_ast": "Assist trend",
    "trend_stl": "Steal trend",
    "trend_blk": "Block trend",
    "trend_tov": "Turnover trend",
    "trend_fgm": "FGM trend",
    "trend_fga": "FGA trend",

    # Momentum & form
    "win_pct": "Win rate",
    "weighted_momentum": "Weighted momentum",
    "form_score": "Form score",
    "win_streak": "Win streak",
    "last_3_margin": "Recent margin (L3)",
    "margin_volatility": "Consistency (volatility)",
    "pts_zscore": "Scoring form (z-score)",

    # Moneyline-specific features
    "composite_power": "Power rating",
    "perf_vs_expected": "Performance vs expected",
    "consistency": "Reliability",
    "recent_win_pct": "Recent win rate",
    "home_away_split_diff": "Home/away split",

    # Head-to-head
    "h2h_win_rate": "H2H record",
    "h2h_avg_margin": "H2H avg margin",

    # ELO
    "elo_home": "ELO rating",
    "elo_away": "ELO rating",
    "elo_diff": "ELO rating edge",
    "elo_slope": "ELO trend",

    # Rest & fatigue
    "rest_home_days": "Rest days",
    "rest_away_days": "Rest days",
    "rest_advantage": "Rest advantage",
    "is_b2b": "Back-to-back",
    "fatigue": "Fatigue",
    "fatigue_diff": "Fatigue advantage",

    # Travel
    "travel_distance": "Travel distance",
    "tz_diff": "Time zone diff",
    "cum_travel": "Trip fatigue",
    "consec_road": "Road trip length",
    "long_road_trip": "Long road trip",

    # Differential features
    "power_diff": "Power edge",
    "form_diff": "Form edge",
    "perf_vs_expected_diff": "Performance edge",
    "consistency_diff": "Reliability edge",

    # Opponent-adjusted
    "offense_vs_defense": "Off vs opp. defense",
    "defense_vs_offense": "Def vs opp. offense",
    "opp_avg_pts_scored": "Opponent scoring",
    "opp_avg_pts_allowed": "Opponent defense",
    "opp_avg_pm": "Opponent net rating",
    "sos": "Strength of schedule",
    "sos_trend": "SOS trend",

}


def _describe_feature(feature_name: str, home_team: str, away_team: str) -> str:
    """
    Convert a model feature column name into a human-readable description.

    Handles patterns like:
      - "avg_pts_5g_home" → "Celtics: Scoring avg (L5)"
      - "avg_pts_5g_away" → "Spurs: Scoring avg (L5)"
      - "pts_diff_5g"     → "Scoring edge (L5)"
      - "elo_home"        → "Celtics: ELO rating"
      - "travel_distance"  → "Travel distance"
      - "rest_advantage"   → "Rest advantage"
    """
    name = feature_name

    # Determine the suffix pattern: _home, _away, _diff, or neither
    home_team_label = home_team
    away_team_label = away_team

    # ── Strip numeric window suffixes (e.g. _5g, _10g) to find the base key ──
    # Patterns: "avg_pts_5g_home" → key="avg_pts", window="L5"
    #           "pts_diff_5g" → key="pts", window="L5", is_home_away_diff=True
    #           "avg_pts_5g" → key="avg_pts", window="L5"
    base = name
    window_label = ""

    # Detect window: e.g. _5g, _10g, _20g
    window_match = re.search(r"_(\d+)g$|_(\d+)g_", base)
    if window_match:
        w = window_match.group(1) or window_match.group(2)
        window_label = f"L{w}"
        # Remove the window pattern to get the core key
        base = re.sub(r"_(\d+)g$", "", base)
        base = re.sub(r"_(\d+)g_", "_", base)

    is_home = base.endswith("_home")
    is_away = base.endswith("_away")
    is_diff = "_diff" in base or "_diff_" in base or base.endswith("_diff")

    if is_home:
        core = base[:-5]  # remove "_home"
    elif is_away:
        core = base[:-5]  # remove "_away"
    elif is_diff:
        # Remove _diff and _diff_5g etc.
        core = re.sub(r"_diff(_\d+[a-z]*)?$", "", base).strip("_")
    else:
        core = base

    # Look up the core key in descriptions
    desc = _FEATURE_DESCRIPTIONS.get(core)
    if not desc:
        # Fallback: prettify the raw name (e.g., "home_away_split_diff" → "Home Away Split")
        desc = core.replace("_", " ").title()

    # Add window label if available
    if window_label:
        desc = f"{desc} ({window_label})"

    # Build final string
    if is_home:
        return f"{home_team_label}: {desc}"
    elif is_away:
        return f"{away_team_label}: {desc}"
    elif is_diff:
        return f"Edge: {desc}"
    else:
        return desc


from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
        stake_dollars = getattr(game, "stake_dollars", g.get("stake_dollars", 0.0)) or 0.0
        is_clear_pick_val = abs(edge_pct) > 0.03
        home_ml = getattr(game, 'home_ml', g.get('home_ml'))
        away_ml = getattr(game, 'away_ml', g.get('away_ml'))
        predicted_at = getattr(game, 'predicted_at', g.get('predicted_at', ''))
        # Totals fields
        total_prediction = getattr(game, 'total_prediction', g.get('total_prediction'))
        total_edge_pct = getattr(game, 'total_edge_pct', g.get('total_edge_pct'))
        total_direction = getattr(game, 'total_direction', g.get('total_direction'))
        total_confidence = getattr(game, 'total_confidence', g.get('total_confidence'))
    else:
        g = game
        matchup = g.get("matchup", f"{g.get('away_team_short', '?')} @ {g.get('home_team_short', '?')}")
        direction = g.get("direction", "neutral") or "neutral"
        confidence = g.get("confidence", "low") or "low"
        edge_pct = g.get("edge_pct", 0.0) or 0.0
        predicted_total = g.get("predicted_total")
        market_total = g.get("market_total")
        stake_dollars = g.get("stake_dollars", 0.0) or 0.0
        is_clear_pick_val = abs(edge_pct) > 0.03
        home_ml = g.get('home_ml')
        away_ml = g.get('away_ml')
        predicted_at = g.get('predicted_at', '')
        # Totals fields
        total_prediction = g.get('total_prediction')
        total_edge_pct = g.get('total_edge_pct')
        total_direction = g.get('total_direction')
        total_confidence = g.get('total_confidence')

    sport_key = g.get("sport_key", "basketball_nba")
    league_name = g.get("league", "NBA")
    sport_group = g.get("sport_group", sport_key_to_group(sport_key))

    home_short = g.get('home_team_short', '?')
    away_short = g.get('away_team_short', '?')

    # ── Compute rich reasoning ──────────────────────────────────────────
    # The edge_pct represents the model's predicted market error:
    #   Positive = the model believes the market underestimates the home team
    #   Negative = the model believes the market overestimates the home team
    edge_pct_val = edge_pct or 0.0
    abs_edge = abs(edge_pct_val)
    edge_direction = "home" if edge_pct_val > 0 else "away"

    # Build multi-line reasoning
    reasoning_lines = []

    # Line 1: What the model predicts
    favored_team = home_short if edge_pct_val >= 0 else away_short
    underdog_team = away_short if edge_pct_val >= 0 else home_short
    reasoning_lines.append(
        f"ML Edge: Model favors {favored_team} ({edge_pct_val:+.1%} edge vs market)"
    )

    # Line 2: Market context
    if home_ml and away_ml:
        reasoning_lines.append(
            f"Market odds: {home_short} {'+' if home_ml > 0 else ''}{home_ml}, "
            f"{away_short} {'+' if away_ml > 0 else ''}{away_ml}"
        )

    # Line 3: Totals prediction from the totals regression model
    # This is SEPARATE from the moneyline edge — it predicts total points
    # (combined score) vs the market total line.
    if total_prediction is not None and total_prediction > 0 and market_total and market_total > 10:
        total_abs_edge = abs(total_edge_pct or 0)
        total_dir_label = (total_direction or "neutral").upper()
        if total_abs_edge > 0.005:
            reasoning_lines.append(
                f"Total {total_dir_label}: Model predicts {total_prediction:.0f} pts "
                f"(market {market_total:.0f}, edge {total_edge_pct:+.1%})"
            )
            reasoning_lines.append(
                f"Total confidence: {total_confidence or 'low'}"
            )

    # Line 4: Confidence explanation
    abs_edge_pct = abs(edge_pct_val)
    if abs_edge_pct >= 0.08:
        confidence_reason = "Very high edge — strong market disagreement"
    elif abs_edge_pct >= 0.05:
        confidence_reason = "Significant edge — clear market inefficiency"
    elif abs_edge_pct >= 0.03:
        confidence_reason = "Moderate edge — potential value opportunity"
    elif abs_edge_pct >= 0.01:
        confidence_reason = "Small edge — marginal opportunity"
    else:
        confidence_reason = "No significant edge detected"

    reasoning_lines.append(confidence_reason)

    # Line 5: Data quality
    n_books_ml = g.get("n_books_ml", 0)
    if n_books_ml > 1:
        reasoning_lines.append(f"Consensus from {n_books_ml} sportsbooks")
    elif n_books_ml == 0:
        reasoning_lines.append("Limited market data — single source")

    # Line 6: Feature importance (top 4 model drivers)
    # Shows what stats the model considers most influential for this prediction
    feature_imp = None
    if hasattr(game, 'feature_importance') and game.feature_importance:
        feature_imp = game.feature_importance
    elif g.get('feature_importance'):
        feature_imp = g['feature_importance']

    feature_display_lines = []
    if feature_imp:
        # Sort by importance descending, take top 4
        sorted_features = sorted(
            feature_imp.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:4]

        for feat_name, importance in sorted_features:
            desc = _describe_feature(feat_name, home_short, away_short)
            # Show importance as a visual weight indication
            pct = round(importance * 100, 1)
            line = f"Key: {desc}"
            feature_display_lines.append(line)
            reasoning_lines.append(f"Key driver: {desc}")

    # Build a structured feature importance display for the template
    feature_display = []
    if feature_imp:
        sorted_features = sorted(
            feature_imp.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:6]
        for feat_name, importance in sorted_features:
            desc = _describe_feature(feat_name, home_short, away_short)
            feature_display.append({
                "description": desc,
                "weight": round(importance * 100, 1),
                "width_pct": int(round(importance * 100, 0)),
            })

    # Determine bet type display
    if sport_group == "Basketball":
        # Direction is "home" or "away" (moneyline edge), not over/under
        bet_type_display = "ML Edge"
        bet_side_display = (
            f"{favored_team} ML ({edge_pct_val:+.1%} edge)"
        )
    else:
        bet_type_display = "Moneyline"
        home_ml_str = home_ml
        away_ml_str = away_ml
        bet_side_display = (
            f"ML: {away_short} {'+' if away_ml_str and away_ml_str > 0 else ''}{away_ml_str or '?'} / "
            f"{home_short} {'+' if home_ml_str and home_ml_str > 0 else ''}{home_ml_str or '?'}"
        )

    return {
        "game_id": g.get("game_id", ""),
        "game_date": g.get("game_date", ""),
        "matchup": matchup,
        "home_team": g.get("home_team", ""),
        "away_team": g.get("away_team", ""),
        "league": league_name,
        "sport_group": sport_group,
        "sport_key": sport_key,
        "bet_type": "moneyline",
        "bet_type_display": bet_type_display,
        "bet_side": bet_side_display,
        "edge_pct": edge_pct_val,
        "stake_dollars": stake_dollars,
        "confidence": confidence,
        "edge_confidence": confidence,
        "is_clear_pick": is_clear_pick_val,
        "reasoning": " | ".join(reasoning_lines),
        "reasoning_lines": reasoning_lines,
        "model_name": "MarketInefficiencySystem",
        "sport_title": league_name,
        "commence_time": g.get("commence_time", ""),
        "home_team_short": home_short,
        "away_team_short": away_short,
        "home_ml": home_ml,
        "away_ml": away_ml,
        "market_total": market_total,
        "predicted_total": predicted_total,
        "total_prediction": total_prediction,
        "total_edge_pct": total_edge_pct,
        "total_direction": total_direction,
        "total_confidence": total_confidence,
        "direction": direction if direction in ("home", "away") else edge_direction,
        "n_books_ml": g.get("n_books_ml", 0),
        "n_books_total": g.get("n_books_total", 0),
        "predicted_at": predicted_at,
        "is_live": g.get("is_live", False),
        "feature_display": feature_display,
    }


def _livegame_to_clear_pick(game: Any) -> dict:
    """Convert a LiveGame to a clear-pick dict for API responses."""
    bet = _livegame_to_bet(game)
    return {
        "bet": bet,
        "clear_score": abs(bet.get("edge_pct", 0) or 0) * 100,
        "risk_level": "low" if abs(bet.get("edge_pct", 0) or 0) > 0.05 else "medium",
        "reasons": bet.get("reasoning_lines", []),
    }


# ── Resolved Bets (from forward_test_results.json) ─────────────────────

FORWARD_TEST_JSON = PROJECT_ROOT / "data" / "forward_test_results.json"


def load_resolved_bets(max_age_days: int = 30) -> list[dict]:
    """
    Load resolved (completed) bets from forward_test_results.json.

    Filters to only bets with an actual_result field, sorted by date descending.
    Returns empty list if file doesn't exist or has no resolved bets.
    """
    if not FORWARD_TEST_JSON.exists():
        return []

    try:
        with open(FORWARD_TEST_JSON) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load forward_test_results.json: {e}")
        return []

    all_bets = data.get("all_bets", [])
    resolved = [b for b in all_bets if b.get("actual_result") is not None]

    # Filter by age if a cutoff date is available
    if resolved:
        cutoff = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
        resolved = [b for b in resolved if b.get("game_date", "")[:10] >= cutoff]

    # Sort by date descending (most recent first)
    resolved.sort(key=lambda b: b.get("game_date", ""), reverse=True)

    return resolved


# ═══════════════════════════════════════════════════════════════════════════
#  LIVE ENGINE ROUTES
# ═══════════════════════════════════════════════════════════════════════════


@app.post("/api/live/refresh")
@app.get("/api/live/refresh")
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
    except (ValueError, RuntimeError, ConnectionError, OSError) as e:
        logger.error(f"Live refresh failed: {e}")
        return JSONResponse(content={
            "n_total": 0, "n_today": 0, "n_tomorrow": 0,
            "n_live": 0, "fresh_odds": False, "refreshed": False,
            "error": str(e),
        })
    except Exception as e:
        logger.critical(f"Unexpected live refresh error: {e}", exc_info=True)
        return JSONResponse(content={
            "n_total": 0, "n_today": 0, "n_tomorrow": 0,
            "n_live": 0, "fresh_odds": False, "refreshed": False,
            "error": "internal_error",
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


def _games_context(force_refresh: bool = False) -> dict:
    """Build template context from live engine games + resolved bets.

    Merges live predictions (next 2 days) with recently resolved bets
    from forward_test_results.json so the dashboard shows both upcoming
    games and past results.

    If force_refresh=True, fetches fresh data from TheOddsAPI.
    NOTE: Page loads NEVER auto-refresh — only user-initiated refresh.
    """
    try:
        engine = get_live_engine()
        # NEVER auto-refresh on page load — only user-initiated refresh.
        # This prevents the dashboard from hanging when odds sources timeout.
        games = engine.get_next_two_days(force_refresh=force_refresh)
        logger.debug(f"_games_context: {len(games)} games from engine (force_refresh={force_refresh})")

        bets = [_livegame_to_bet(g) for g in games]
        clear = [_livegame_to_bet(g) for g in games
                 if abs((g.edge_pct or 0)) > 0.03]

        # Load resolved bets from history
        resolved_bets = load_resolved_bets(max_age_days=30)
        n_resolved = len(resolved_bets)

        # Compute stats from resolved bets
        resolved_wins = sum(1 for b in resolved_bets if b.get("actual_result") == "WIN")
        resolved_losses = sum(1 for b in resolved_bets if b.get("actual_result") == "LOSS")
        resolved_profits = [b.get("actual_profit", 0) for b in resolved_bets if b.get("actual_profit") is not None]
        total_pnl = sum(resolved_profits) if resolved_profits else 0.0

        # Build per-league breakdown from resolved bets
        league_stats: dict[str, dict] = {}
        for b in resolved_bets:
            league = b.get("league", "NBA") or "NBA"
            if league not in league_stats:
                league_stats[league] = {"wins": 0, "losses": 0, "profit": 0.0, "n": 0}
            st = league_stats[league]
            st["n"] += 1
            result = b.get("actual_result")
            if result == "WIN":
                st["wins"] += 1
            elif result == "LOSS":
                st["losses"] += 1
            profit = b.get("actual_profit", 0)
            if profit is not None:
                st["profit"] += profit
        # Sort leagues by profit descending
        sport_pnl = [
            {
                "league": league,
                "wins": st["wins"],
                "losses": st["losses"],
                "n": st["n"],
                "profit": round(st["profit"], 2),
            }
            for league, st in sorted(league_stats.items(), key=lambda x: x[1]["profit"], reverse=True)
        ]

        # Auto-resolve timestamp from engine
        last_auto_resolve = getattr(engine, 'last_auto_resolve', None)

        # Merge bets + resolved_bets into one unified list sorted by date descending
        all_bets = list(bets) + list(resolved_bets)
        all_bets.sort(key=lambda b: b.get("game_date", ""), reverse=True)

        # Build chart data from resolved profits (oldest first for cumulative chart)
        # resolved_bets are sorted by date descending; reverse for plotting
        chart_profits = list(reversed(resolved_profits)) if len(resolved_profits) > 1 else []

        return {
            "bets": bets,
            "clear_picks": clear[:10],
            "resolved_bets": resolved_bets,
            "all_bets": all_bets,
            "n_all_bets": len(all_bets),
            "n_resolved": n_resolved,
            "resolved_wins": resolved_wins,
            "resolved_losses": resolved_losses,
            "resolved_pnl": round(total_pnl, 2),
            "sport_pnl": sport_pnl, "n_sport_count": len(sport_pnl),
            "chart_profits": json.dumps(chart_profits),
            "last_auto_resolve": last_auto_resolve or "",
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
            "bets": [], "clear_picks": [], "resolved_bets": [], "all_bets": [],
            "n_resolved": 0, "n_all_bets": 0, "resolved_wins": 0, "resolved_losses": 0, "resolved_pnl": 0.0,
            "sport_pnl": [], "n_sport_count": 0,
            "last_auto_resolve": "",
            "summary": {},
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_bets": 0, "n_games": 0, "n_clear": 0,
            "today": date.today().isoformat(),
        }


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard — shows predictions from the LivePredictionEngine.

    Uses the live engine to fetch real NBA games + predictions from TheOddsAPI
    and the ML model. Auto-refreshes if no cached data is available.
    """
    ctx = _games_context()
    return templates.TemplateResponse(request, "index.html", ctx)


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
    """Future predictions page — predictions loaded client-side from /api/future-predictions.

    The server now only returns the HTML shell with skeleton loading.
    JavaScript fetches predictions asynchronously and renders cards client-side.
    """
    # No need to load predictions server-side anymore — JS loads them asynchronously
    return templates.TemplateResponse(request, "future_predictions.html", {
        "predictions": [],
        "n_predictions": 0,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "today": date.today().isoformat(),
    })


# ═══════════════════════════════════════════════════════════════════════════
#  JSON API ROUTES
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/api/predictions")
async def api_predictions(limit: int = 50, min_edge: float = 0.0):
    """JSON API — returns live predictions sorted by edge descending."""
    ctx = _games_context()
    bets = ctx["bets"]
    if min_edge > 0:
        bets = [b for b in bets if abs(b.get("edge_pct", 0)) >= min_edge]
    bets.sort(key=lambda b: abs(b.get("edge_pct", 0)), reverse=True)
    return JSONResponse(content={
        "n_bets": len(bets[:limit]),
        "generated_at": ctx["generated_at"],
        "bets": bets[:limit],
    })


@app.get("/api/bets")
async def api_bets():
    """JSON API — returns all bets from the live engine as a flat list."""
    try:
        engine = get_live_engine()
        needs_refresh = not engine.has_cached_data
        games = engine.get_next_two_days(force_refresh=needs_refresh)
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
    """Resolve predictions against actual results and refresh the live engine.

    After resolving, refreshes the live engine so the dashboard immediately
    shows fresh predictions for upcoming games and resolved results for past games.
    """
    try:
        from betting_intel.analytics.tracker import ResultsTracker
        tracker = ResultsTracker()
        n = tracker.resolve_all()
        # Refresh live engine after resolving so we get fresh predictions
        ctx = _games_context(force_refresh=True)
        return JSONResponse(content={
            "resolved": n,
            "message": f"Resolved {n} predictions and refreshed live engine",
            "n_bets": ctx["n_bets"],
            "n_games": ctx["n_games"],
            "n_clear": ctx["n_clear"],
            "generated_at": ctx["generated_at"],
        })
    except Exception as e:
        logger.error(f"Resolve failed: {e}")
        return JSONResponse(content={"error": str(e), "resolved": 0}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/api/resolved-bets/csv")
async def api_resolved_bets_csv(max_age_days: int = 365):
    """Download resolved bets as CSV."""
    bets = load_resolved_bets(max_age_days=max_age_days)
    if not bets:
        return JSONResponse(content={"error": "No resolved bets found"}, status_code=404)

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "game_date", "league", "matchup", "home_team", "away_team",
        "bet_type", "bet_side", "edge_pct", "stake_dollars",
        "actual_result", "actual_profit",
    ])

    for b in bets:
        writer.writerow([
            b.get("game_date", ""),
            b.get("league", ""),
            b.get("matchup", ""),
            b.get("home_team", ""),
            b.get("away_team", ""),
            b.get("bet_type", ""),
            b.get("bet_side", ""),
            b.get("edge_pct", 0),
            b.get("stake_dollars", 0),
            b.get("actual_result", ""),
            b.get("actual_profit", ""),
        ])

    output.seek(0)
    # Prepend BOM for Excel UTF-8 compatibility
    csv_content = "\ufeff" + output.getvalue()
    today_str = date.today().isoformat()
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="resolved_bets_{today_str}.csv"',
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
#  WEBSOCKET LIVE REFRESH
# ═══════════════════════════════════════════════════════════════════════════


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """WebSocket endpoint for live dashboard refresh.

    On connection, immediately sends the current prediction data.
    Then pushes updated data every 30 seconds — refreshing from the live engine.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Send current predictions from the live engine
            try:
                ctx = _games_context()
                await websocket.send_json({
                    "type": "predictions",
                    "n_bets": ctx["n_bets"],
                    "n_games": ctx["n_games"],
                    "n_clear": ctx["n_clear"],
                    "generated_at": ctx["generated_at"],
                })
            except WebSocketDisconnect:
                raise  # Clean disconnect — outer handler logs it
            except Exception as e:
                logger.warning(f"WebSocket data error: {e}")
                # Try to notify the client — if this also fails, the
                # socket is closed and we bail out of the loop.
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except Exception:
                    raise  # Socket is dead — break out of loop

            # Wait 30 seconds before next push
            await asyncio.sleep(30)

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
    # Check if the live engine has cached data
    try:
        engine_ok = get_live_engine().has_cached_data
    except Exception:
        engine_ok = False
    return JSONResponse(content={
        "status": "ok" if db_ok else "degraded",
        "database": db_ok,
        "live_engine_ready": engine_ok,
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
