#!/usr/bin/env python3
"""
Seed the web dashboard with mock data for all 4 supported leagues.

Generates realistic-looking LiveGame objects for NBA, NCAAB, Euroleague,
and NFL so you can visually verify league badges, sport filters, edge bars,
and prediction cards in the browser without live API keys.

Usage:
    # Seed + start the dashboard with mock data:
    python tools/seed_dashboard.py

    # Seed only (write data files, don't start server):
    python tools/seed_dashboard.py --seed-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Optional

# ── Ensure project root is on sys.path ─────────────────────────────────
_HERE = Path(__file__).resolve().parent          # tools/
_PROJECT_ROOT = _HERE.parent                      # betting-intelligence/
sys.path.insert(0, str(_PROJECT_ROOT))            # for web.app, data/, etc.
sys.path.insert(0, str(_PROJECT_ROOT / "src"))    # for betting_intel.*


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("seed_dashboard")


# ═══════════════════════════════════════════════════════════════════════════
#  MOCK DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

# Realistic edges by league (edge_pct range, total_mean, total_range)
_LEAGUE_PROFILES = {
    "NBA": {
        "sport_key": "basketball_nba",
        "sport_group": "Basketball",
        "edge_base": 0.035, "edge_range": 0.04,       # ~0-7% edges
        "total_mean": 228.0, "total_range": 20.0,
        "total_min": 180.0, "total_max": 260.0,
        "home_teams": ["Celtics", "Lakers", "Warriors", "Knicks", "Nuggets", "Bucks"],
        "away_teams": ["Heat", "Spurs", "Mavericks", "76ers", "Suns", "Thunder"],
        "home_ml": [-350, -180, -120, -110, 150, 120],
        "away_ml": [280, 160, 100, -105, -175, -140],
        "spreads": [-7.5, -3.5, -1.5, -1.0, 3.5, 2.5],
    },
    "NCAAB": {
        "sport_key": "basketball_ncaab",
        "sport_group": "Basketball",
        "edge_base": 0.045, "edge_range": 0.05,
        "total_mean": 148.0, "total_range": 15.0,
        "total_min": 110.0, "total_max": 190.0,
        "home_teams": ["Duke", "Kentucky", "UNC", "Kansas", "UConn", "Gonzaga"],
        "away_teams": ["Arizona", "Michigan State", "Tennessee", "Baylor", "Purdue", "Marquette"],
        "home_ml": [-250, -150, -110, 110, -130, -200],
        "away_ml": [200, 130, -105, -130, 115, 170],
        "spreads": [-6.5, -4.5, -1.5, 2.5, -2.5, -5.5],
    },
    "Euroleague": {
        "sport_key": "basketball_euroleague",
        "sport_group": "Basketball",
        "edge_base": 0.03, "edge_range": 0.04,
        "total_mean": 162.0, "total_range": 12.0,
        "total_min": 150.0, "total_max": 180.0,
        "home_teams": ["Real Madrid", "Barcelona", "Olympiacos", "Fenerbahçe"],
        "away_teams": ["Panathinaikos", "Maccabi Tel Aviv", "Monaco", "Žalgiris"],
        "home_ml": [-220, -150, -120, 105],
        "away_ml": [185, 130, 100, -125],
        "spreads": [-5.5, -4.0, -2.0, 1.5],
    },
    "NFL": {
        "sport_key": "americanfootball_nfl",
        "sport_group": "Football",
        "edge_base": 0.03, "edge_range": 0.04,
        "total_mean": 46.0, "total_range": 8.0,
        "total_min": 30.0, "total_max": 60.0,
        "home_teams": ["Chiefs", "49ers", "Ravens", "Lions", "Eagles", "Cowboys"],
        "away_teams": ["Bills", "Packers", "Bengals", "Vikings", "Rams", "Giants"],
        "home_ml": [-280, -140, -110, -120, -150, -130],
        "away_ml": [230, 120, -105, 100, 130, 115],
        "spreads": [-5.5, -2.5, -1.5, -2.0, -3.5, -3.0],
    },
}

# Confidence distribution per league for visual variety
_CONFIDENCE_DIST = ["high", "high", "medium", "medium", "medium", "low"]


def _pick_random(items: list, idx: int) -> Any:
    """Pick item by index with wrap-around."""
    return items[idx % len(items)]


def _compute_totals(predicted_total: float, market_total: float) -> dict:
    """Compute totals fields for a mock game."""
    edge_pct = round((predicted_total - market_total) / max(market_total, 1), 4)
    direction = "over" if edge_pct > 0 else "under"
    abs_e = abs(edge_pct)
    confidence = "high" if abs_e > 0.05 else ("medium" if abs_e >= 0.02 else "low")

    return {
        "total_prediction": round(predicted_total, 1),
        "market_total": round(market_total, 1),
        "total_edge_pct": edge_pct,
        "total_direction": direction,
        "total_confidence": confidence,
    }


def _league_to_moneyline(profile: dict, idx: int) -> list[float]:
    """Get home/away moneyline values."""
    h = _pick_random(profile["home_ml"], idx)
    a = _pick_random(profile["away_ml"], idx)
    return [h, a]


def create_mock_live_games() -> list[dict]:
    """Create mock LiveGame-compatible dicts for all 4 leagues.

    Returns a list of dicts that can be used to construct LivePredictionSnapshot
    or seeded directly into the engine cache.
    """
    today_str = date.today().isoformat()
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()
    day_after_str = (date.today() + timedelta(days=2)).isoformat()
    games: list[dict] = []
    game_idx = 0

    for league, profile in _LEAGUE_PROFILES.items():
        n_games = len(profile["home_teams"])
        for i in range(n_games):
            home = _pick_random(profile["home_teams"], i)
            away = _pick_random(profile["away_teams"], i)

            # Vary game dates: today, tomorrow, day after
            date_choices = [today_str, tomorrow_str, day_after_str]
            gdate = date_choices[i % 3]

            # Generate realistic edge
            import random
            rng = random.Random(hash(f"{league}{home}{away}{game_idx}") & 0xFFFFFFFF)

            edge_pct = round(profile["edge_base"] + rng.uniform(-profile["edge_range"]/2, profile["edge_range"]/2), 4)
            direction = "home" if edge_pct > 0 else "away"
            abs_e = abs(edge_pct)
            confidence = _pick_random(_CONFIDENCE_DIST, i)

            # Spread related
            spread = _pick_random(profile["spreads"], i)
            home_ml, away_ml = _league_to_moneyline(profile, i)

            # Totals
            market_total = round(profile["total_mean"] + rng.uniform(-profile["total_range"]/2, profile["total_range"]/2), 1)
            market_total = max(profile["total_min"], min(profile["total_max"], market_total))
            predicted_total = round(market_total * (1 + edge_pct), 1)
            predicted_total = max(profile["total_min"], min(profile["total_max"], predicted_total))

            totals = _compute_totals(predicted_total, market_total)

            # Game ID
            game_id = f"{home}_{away}_{gdate}"

            # Commence time
            hour = 19 + (i % 4)  # 19:00 - 22:00
            commence = f"{gdate}T{hour}:00:00Z"

            feature_importance = _mock_feature_importance(home, away, league)

            game = {
                "game_id": game_id,
                "sport_key": profile["sport_key"],
                "home_team": home,
                "away_team": away,
                "home_team_short": home,
                "away_team_short": away,
                "commence_time": commence,
                "game_date": gdate,
                "league": league,
                "sport_group": profile["sport_group"],
                "home_ml": float(home_ml),
                "away_ml": float(away_ml),
                "spread": float(spread),
                # market_total is set via **totals below
                "over_odds": -110.0,
                "under_odds": -110.0,
                "n_books_ml": rng.randint(3, 12),
                "n_books_total": rng.randint(2, 8),
                "ml_std": round(rng.uniform(0.02, 0.08), 3),
                "is_live": i == 0 and league == "NBA",  # First NBA game is "live"
                "is_today": gdate == today_str,
                "is_tomorrow": gdate == tomorrow_str,
                "predicted_total": predicted_total,
                "edge_pct": edge_pct,
                "direction": direction,
                "confidence": confidence,
                **totals,
                "stake_dollars": round(abs(edge_pct) * 500, 2),
                "feature_importance": feature_importance,
                "recommended_quarter": "FULL",
                "recommended_direction": direction,
                "odds_fetched_at": datetime.now().isoformat(),
                "predicted_at": datetime.now().isoformat(),
                "quarter_projections": {
                    "q1_home": round(predicted_total * 0.24, 1),
                    "q1_away": round(predicted_total * 0.23, 1),
                    "q2_home": round(predicted_total * 0.25, 1),
                    "q2_away": round(predicted_total * 0.24, 1),
                    "q3_home": round(predicted_total * 0.26, 1),
                    "q3_away": round(predicted_total * 0.25, 1),
                    "q4_home": round(predicted_total * 0.25, 1),
                    "q4_away": round(predicted_total * 0.26, 1),
                },
            }
            games.append(game)
            game_idx += 1

    # Add one very strong edge game per league for clear-pick visibility
    strong_edge_games = []
    for league, profile in _LEAGUE_PROFILES.items():
        if not games:
            continue
        # Clone the first game of each league with a high edge
        template = deepcopy(games[len(strong_edge_games)])
        template["edge_pct"] = 0.095 if league != "NFL" else 0.085
        template["confidence"] = "high"
        template["direction"] = "home"
        template["stake_dollars"] = 475.0
        template["game_id"] = template["game_id"] + "_strong"
        strong_edge_games.append(template)
    games.extend(strong_edge_games)

    return games


def _mock_feature_importance(home: str, away: str, league: str) -> dict[str, float]:
    """Generate realistic feature importance dict."""
    import random
    rng = random.Random(hash(f"{league}{home}{away}") & 0xFFFFFFFF)

    features = {
        "avg_pts_5g_home": 0.18,
        "avg_pts_5g_away": 0.14,
        "margin_volatility_home": 0.12,
        "elo_home": 0.11,
        "elo_away": 0.09,
        "avg_pace_5g_home": 0.08,
        "weighted_momentum_home": 0.07,
        "weighted_momentum_away": 0.06,
        "travel_distance": 0.05,
        "rest_advantage": 0.04,
        "h2h_win_rate": 0.03,
        "sos_home": 0.03,
    }
    # Jitter values slightly
    for k in features:
        features[k] = round(max(0.01, features[k] + rng.uniform(-0.02, 0.02)), 3)

    return features


def create_mock_future_predictions() -> list[dict]:
    """Create mock prediction dicts matching FutureGamePredictor output format."""
    import random
    today_str = date.today().isoformat()
    games: list[dict] = []

    for league_idx, (league, profile) in enumerate(_LEAGUE_PROFILES.items()):
        n = len(profile["home_teams"])
        for i in range(n):
            rng = random.Random(hash(f"future_{league}{i}") & 0xFFFFFFFF)
            home = _pick_random(profile["home_teams"], i)
            away = _pick_random(profile["away_teams"], i)
            gdate = (date.today() + timedelta(days=(i % 3))).isoformat()

            market = round(profile["total_mean"] + rng.uniform(-profile["total_range"]/3, profile["total_range"]/3), 1)
            market = max(profile["total_min"], min(profile["total_max"], market))
            predicted = round(market * (1 + rng.uniform(-0.03, 0.05)), 1)
            predicted = max(profile["total_min"], min(profile["total_max"], predicted))
            edge = round((predicted - market) / max(market, 1), 4)
            direction = "over" if edge > 0 else "under"
            abs_e = abs(edge)
            conf = "high" if abs_e > 0.05 else ("medium" if abs_e >= 0.02 else "low")

            hpct = 0.51 + (i - n/2) * 0.02
            hs = round(predicted * hpct, 1)
            aws = round(predicted * (1 - hpct), 1)

            game = {
                "game_id": f"{home}_{away}_{gdate}",
                "game_date": gdate,
                "league": league,
                "home_team": home,
                "away_team": away,
                "home_team_short": home,
                "away_team_short": away,
                "matchup": f"{away} @ {home}",
                "predicted_total": predicted,
                "market_total": market,
                "edge_pct": edge,
                "direction": direction,
                "confidence": conf,
                "model_mae": "N/A",
                "home_score": hs,
                "away_score": aws,
                "best_quarter": "Q3",
                "best_quarter_edge": round(abs(edge) * 2, 2),
                "best_quarter_direction": direction,
                "recommended_quarter": "Q3",
                "recommended_direction": direction,
                "q1_home": round(hs * 0.24, 1), "q1_away": round(aws * 0.23, 1),
                "q1_total": round(predicted * 0.242, 1), "q1_market": round(market * 0.242, 1), "q1_edge": 0.005,
                "q2_home": round(hs * 0.25, 1), "q2_away": round(aws * 0.24, 1),
                "q2_total": round(predicted * 0.251, 1), "q2_market": round(market * 0.251, 1), "q2_edge": 0.004,
                "q3_home": round(hs * 0.26, 1), "q3_away": round(aws * 0.25, 1),
                "q3_total": round(predicted * 0.253, 1), "q3_market": round(market * 0.253, 1), "q3_edge": 0.006,
                "q4_home": round(hs * 0.25, 1), "q4_away": round(aws * 0.26, 1),
                "q4_total": round(predicted * 0.254, 1), "q4_market": round(market * 0.254, 1), "q4_edge": 0.004,
                "h1_home": round(hs * 0.49, 1), "h1_away": round(aws * 0.47, 1),
                "h1_total": round(predicted * 0.493, 1), "h1_market": round(market * 0.493, 1), "h1_edge": 0.005,
                "h2_home": round(hs * 0.51, 1), "h2_away": round(aws * 0.53, 1),
                "h2_total": round(predicted * 0.507, 1), "h2_market": round(market * 0.507, 1), "h2_edge": 0.005,
            }
            games.append(game)

    games.sort(key=lambda g: (g["game_date"], g["league"]))
    return games


# ═══════════════════════════════════════════════════════════════════════════
#  FRESH ODDS CACHE (so the engine thinks it has data)
# ═══════════════════════════════════════════════════════════════════════════

def create_mock_odds_data() -> list[dict]:
    """Create mock raw odds dicts matching OddsFetcher output format."""
    odds = []
    for league, profile in _LEAGUE_PROFILES.items():
        for i, (home, away) in enumerate(zip(profile["home_teams"], profile["away_teams"])):
            gdate = (date.today() + timedelta(days=(i % 3))).isoformat()
            h_ml, a_ml = _league_to_moneyline(profile, i)
            import random
            rng = random.Random(hash(f"odds_{league}{i}") & 0xFFFFFFFF)
            market_total = round(profile["total_mean"] + rng.uniform(-5, 5), 1)
            odds.append({
                "id": f"{home}_{away}_{gdate}",
                "sport_key": profile["sport_key"],
                "sport_title": league,
                "commence_time": f"{gdate}T20:00:00Z",
                "home_team": home,
                "away_team": away,
                "bookmakers": [
                    {
                        "key": "mock",
                        "title": "Mock Sportsbook",
                        "last_update": datetime.now().isoformat(),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": home, "price": h_ml},
                                    {"name": away, "price": a_ml},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": home, "price": -110, "point": _pick_random(profile["spreads"], i)},
                                    {"name": away, "price": -110, "point": -_pick_random(profile["spreads"], i)},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "price": -110, "point": market_total},
                                    {"name": "Under", "price": -110, "point": market_total},
                                ],
                            },
                        ],
                    },
                ],
            })
    return odds


# ═══════════════════════════════════════════════════════════════════════════
#  FILE-BASED SEEDING
# ═══════════════════════════════════════════════════════════════════════════

def seed_data_files(project_root: Optional[Path] = None) -> Path:
    """Write seed data files. Returns the project root path."""
    if project_root is None:
        project_root = _PROJECT_ROOT

    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Future predictions cache ──────────────────────────────────
    future_preds = create_mock_future_predictions()
    cache_path = data_dir / "prediction_cache.json"
    with open(cache_path, "w") as f:
        json.dump({
            "predictions": future_preds,
            "_cached_at": datetime.now().timestamp(),
        }, f, indent=2, default=str)
    logger.info(f"Wrote {len(future_preds)} future predictions to {cache_path}")

    # ── 2. Snapshot seed data (for engine patch) ─────────────────────
    games_list = create_mock_live_games()
    snapshot_path = data_dir / "seed_snapshot.json"
    with open(snapshot_path, "w") as f:
        json.dump({
            "live_games": [g for g in games_list if g.get("is_live")],
            "today_games": [g for g in games_list if g.get("is_today")],
            "tomorrow_games": [g for g in games_list if g.get("is_tomorrow")],
            "next_two_days": games_list,
            "generated_at": datetime.now().isoformat(),
            "n_live": sum(1 for g in games_list if g.get("is_live")),
            "n_today": sum(1 for g in games_list if g.get("is_today")),
            "n_tomorrow": sum(1 for g in games_list if g.get("is_tomorrow")),
            "n_total": len(games_list),
            "fresh_odds": True,
        }, f, indent=2, default=str)
    logger.info(f"Wrote {len(games_list)} live games to {snapshot_path}")

    # ── 3. Mock odds raw data ────────────────────────────────────────
    odds_data = create_mock_odds_data()
    odds_path = data_dir / "seed_odds.json"
    with open(odds_path, "w") as f:
        json.dump(odds_data, f, indent=2, default=str)
    logger.info(f"Wrote {len(odds_data)} mock odds to {odds_path}")

    return project_root


# ═══════════════════════════════════════════════════════════════════════════
#  ENGINE PATCHING (for seeded server)
# ═══════════════════════════════════════════════════════════════════════════

def patch_live_engine(project_root: Optional[Path] = None):
    """Patch web.app's get_live_engine to return a pre-seeded engine.

    This function:
    1. Loads seed data from data/seed_snapshot.json
    2. Creates LiveGame dataclass instances
    3. Builds a LivePredictionSnapshot
    4. Patches get_live_engine to return an engine whose get_snapshot returns the seeded snapshot
    """
    if project_root is None:
        project_root = _PROJECT_ROOT

    from betting_intel.live.models import LiveGame, LivePredictionSnapshot

    snapshot_path = project_root / "data" / "seed_snapshot.json"
    if not snapshot_path.exists():
        logger.error(f"Seed data not found at {snapshot_path}. Run with --seed-only first.")
        sys.exit(1)

    with open(snapshot_path) as f:
        snap_data = json.load(f)

    def _dict_to_livegame(d: dict) -> LiveGame:
        """Convert a dict to a LiveGame dataclass, filtering unknown fields."""
        from dataclasses import fields as dc_fields
        valid_keys = {f.name for f in dc_fields(LiveGame)}
        kwargs = {k: v for k, v in d.items() if k in valid_keys}
        return LiveGame(**kwargs)

    live_games = [_dict_to_livegame(g) for g in snap_data.get("live_games", [])]
    today_games = [_dict_to_livegame(g) for g in snap_data.get("today_games", [])]
    tomorrow_games = [_dict_to_livegame(g) for g in snap_data.get("tomorrow_games", [])]
    next_two = [_dict_to_livegame(g) for g in snap_data.get("next_two_days", [])]

    seeded_snapshot = LivePredictionSnapshot(
        live_games=live_games,
        today_games=today_games,
        tomorrow_games=tomorrow_games,
        next_two_days=next_two,
        generated_at=snap_data.get("generated_at", datetime.now().isoformat()),
        n_live=snap_data.get("n_live", len(live_games)),
        n_today=snap_data.get("n_today", len(today_games)),
        n_tomorrow=snap_data.get("n_tomorrow", len(tomorrow_games)),
        n_total=snap_data.get("n_total", len(next_two)),
        fresh_odds=True,
    )

    # Patch the web app's singleton getter
    import web.app as web_app
    original_getter = web_app.get_live_engine

    def seeded_getter():
        """Return a pre-seeded engine."""
        engine = original_getter()
        import time
        engine._snapshot = seeded_snapshot
        engine._last_refresh = time.time()
        # Leave _cached_odds_raw as None — page loads use force_refresh=False
        # which never touches _cached_odds_raw. The refresh button would
        # rebuild the snapshot (and fail gracefully with no API keys), but
        # the seeded snapshot remains intact for the next page load.
        logger.info(f"LivePredictionEngine seeded with {seeded_snapshot.n_total} mock games")
        return engine

    web_app.get_live_engine = seeded_getter
    logger.info("Web app patched — get_live_engine will return seeded data")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Seed the web dashboard with mock data for all 4 leagues",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only write seed data files, don't start the server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to run the dashboard on (default: 8001)",
    )
    args = parser.parse_args()

    # 1. Write seed data files
    project_root = seed_data_files()

    if args.seed_only:
        logger.info("\n" + "=" * 60)
        logger.info("Seed data written to data/ directory")
        logger.info("Start the dashboard manually:")
        logger.info("  python tools/seed_dashboard.py")
        logger.info("  # or: uvicorn web.app:app --port 8001")
        logger.info("=" * 60)
        return

    # 2. Patch the engine
    patch_live_engine(project_root)

    # 3. Start uvicorn with the already-imported app object
    #    (uvicorn.run with a string re-imports the module in a subprocess,
    #     losing the monkey-patch applied above)
    logger.info("Starting dashboard with seeded data on port %d...", args.port)
    from web.app import app as _seeded_app
    import uvicorn
    uvicorn.run(_seeded_app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
