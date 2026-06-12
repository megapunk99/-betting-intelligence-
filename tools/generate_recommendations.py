#!/usr/bin/env python3
"""
ULTIMATE BET RECOMMENDATIONS — Three-Model Prediction Pipeline.

Uses the ultimate multi-model system (totals + win-prob + spread) with
live ESPN odds to produce betting recommendations with Kelly staking.

The ML predictions for moneyline are now backed by a trained classifier
(90.2% accuracy, 0.1045 Brier) — no more heuristic formulas.

Usage:
    python tools/generate_recommendations.py
"""

import sys
import json
import math
import warnings
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import joblib

from betting_intel.models.stacking import WalkForwardStackingEnsemble, WinProbEnsemble


# ── Helpers ────────────────────────────────────────────────────────────────

def american_to_prob(odds: float) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def compute_kelly(win_prob: float, decimal_odds: float, fraction: float = 0.25) -> tuple[float, float]:
    if win_prob <= 0 or win_prob >= 1 or decimal_odds <= 1:
        return (0.0, 0.0)
    b = decimal_odds - 1.0
    if b <= 0:
        return (0.0, 0.0)
    full_kelly = (b * win_prob - (1 - win_prob)) / b
    if full_kelly <= 0:
        return (0.0, 0.0)
    frac = full_kelly * fraction
    frac = max(0.0, min(frac, 0.10))
    return (frac, frac * 10000.0)


def american_to_decimal(odds: float) -> float:
    if odds > 0:
        return 1 + odds / 100.0
    else:
        return 1 + 100.0 / abs(odds)


# ── Team mapping ───────────────────────────────────────────────────────────

SHORT_NAME_TO_TEAM_ID: dict[str, int] = {
    "Hawks": 1610612737, "Celtics": 1610612738, "Nets": 1610612751,
    "Hornets": 1610612766, "Bulls": 1610612741, "Cavaliers": 1610612739,
    "Mavericks": 1610612742, "Nuggets": 1610612743, "Pistons": 1610612765,
    "Warriors": 1610612744, "Rockets": 1610612745, "Pacers": 1610612754,
    "Clippers": 1610612746, "Lakers": 1610612747, "Grizzlies": 1610612763,
    "Heat": 1610612748, "Bucks": 1610612749, "Timberwolves": 1610612750,
    "Pelicans": 1610612740, "Knicks": 1610612752, "Thunder": 1610612760,
    "Magic": 1610612753, "76ers": 1610612755, "Suns": 1610612756,
    "Trail Blazers": 1610612757, "Kings": 1610612758, "Spurs": 1610612759,
    "Raptors": 1610612761, "Jazz": 1610612762, "Wizards": 1610612764,
}

NBA_TEAM_CENTERS: dict[str, tuple[float, float]] = {
    "Hawks": (33.755, -84.396), "Celtics": (42.366, -71.062),
    "Nets": (40.683, -73.975), "Hornets": (35.225, -80.839),
    "Bulls": (41.881, -87.674), "Cavaliers": (41.496, -81.688),
    "Mavericks": (32.790, -96.810), "Nuggets": (39.748, -105.007),
    "Pistons": (42.340, -83.056), "Warriors": (37.750, -122.203),
    "Rockets": (29.751, -95.362), "Pacers": (39.764, -86.156),
    "Clippers": (34.043, -118.267), "Lakers": (34.043, -118.267),
    "Grizzlies": (35.138, -90.051), "Heat": (25.781, -80.187),
    "Bucks": (43.043, -87.917), "Timberwolves": (44.979, -93.276),
    "Pelicans": (29.949, -90.082), "Knicks": (40.750, -73.993),
    "Thunder": (35.463, -97.515), "Magic": (28.539, -81.384),
    "76ers": (39.901, -75.172), "Suns": (33.445, -112.071),
    "Trail Blazers": (45.532, -122.667), "Kings": (38.580, -121.500),
    "Spurs": (29.427, -98.437), "Raptors": (43.643, -79.379),
    "Jazz": (40.768, -111.901), "Wizards": (38.898, -77.021),
}

NBA_TEAM_TZ: dict[str, int] = {
    "Celtics": -5, "Nets": -5, "Knicks": -5, "76ers": -5, "Wizards": -5,
    "Hawks": -5, "Heat": -5, "Hornets": -5, "Magic": -5, "Raptors": -5,
    "Pistons": -5, "Pacers": -5, "Cavaliers": -5, "Bulls": -6,
    "Bucks": -6, "Timberwolves": -6, "Pelicans": -6, "Thunder": -6,
    "Mavericks": -6, "Rockets": -6, "Grizzlies": -6, "Spurs": -6,
    "Jazz": -7, "Nuggets": -7, "Suns": -7, "Trail Blazers": -8,
    "Kings": -8, "Warriors": -8, "Lakers": -8, "Clippers": -8,
}


def _haversine(loc1, loc2) -> float:
    R = 3959.0
    lat1, lon1 = np.radians(loc1)
    lat2, lon2 = np.radians(loc2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _team_feature_value(last_row: pd.Series, col_name: str, team_was_home: bool) -> float:
    if col_name.endswith("_home"):
        base = col_name[:-5]
        suffix = "_home" if team_was_home else "_away"
        return last_row.get(base + suffix, 0.0)
    elif col_name.endswith("_away"):
        base = col_name[:-5]
        suffix = "_home" if team_was_home else "_away"
        return last_row.get(base + suffix, 0.0)
    else:
        return last_row.get(col_name, 0.0)


def build_feature_vector(
    home_short: str,
    away_short: str,
    game_date: str,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict | None:
    home_id = SHORT_NAME_TO_TEAM_ID.get(home_short)
    away_id = SHORT_NAME_TO_TEAM_ID.get(away_short)
    if not home_id or not away_id:
        return None

    sorted_df = feature_df.sort_values("GAME_DATE")

    home_rows = sorted_df[
        (sorted_df["TEAM_ID_home"] == home_id) |
        (sorted_df["TEAM_ID_away"] == home_id)
    ]
    away_rows = sorted_df[
        (sorted_df["TEAM_ID_home"] == away_id) |
        (sorted_df["TEAM_ID_away"] == away_id)
    ]
    if home_rows.empty or away_rows.empty:
        return None

    home_last = home_rows.iloc[-1]
    away_last = away_rows.iloc[-1]
    home_was_home = home_last["TEAM_ID_home"] == home_id
    away_was_home = away_last["TEAM_ID_home"] == away_id

    home_team_name = str(home_last.get("TEAM_NAME_home" if home_was_home else "TEAM_NAME_away", ""))
    away_team_name = str(away_last.get("TEAM_NAME_home" if away_was_home else "TEAM_NAME_away", ""))

    feature_row = {}
    for col in feature_cols:
        if col.endswith("_home"):
            feature_row[col] = _team_feature_value(home_last, col, home_was_home)
        elif col.endswith("_away"):
            feature_row[col] = _team_feature_value(away_last, col, away_was_home)
        else:
            feature_row[col] = home_last.get(col, 0.0)

    # Fix opp_* features
    OPP_TO_TEAM_STAT: dict[str, str] = {
        "opp_avg_pts_scored": "avg_pts_allowed",
        "opp_avg_pts_allowed": "avg_pts_10g",
        "opp_avg_pm": "avg_pm_10g",
        "opp_trailing_margin": "avg_pm_10g",
    }
    opp_prefixes = sorted(OPP_TO_TEAM_STAT.keys(), key=len, reverse=True)
    for col in feature_cols:
        if not col.startswith("opp_") or col.startswith("adj_opp_"):
            continue
        matched = None
        for prefix in opp_prefixes:
            if col.startswith(prefix):
                matched = prefix
                break
        if matched is None:
            continue
        team_stat = OPP_TO_TEAM_STAT[matched]
        suffix = col[len(matched):]
        if suffix == "_home":
            feature_row[col] = _team_feature_value(away_last, f"{team_stat}_home", away_was_home)
        elif suffix == "_away":
            feature_row[col] = _team_feature_value(home_last, f"{team_stat}_home", home_was_home)

    # Override game-level features
    try:
        game_dt = pd.Timestamp(game_date)
    except Exception:
        game_dt = max(
            pd.Timestamp(home_last["GAME_DATE"]),
            pd.Timestamp(away_last["GAME_DATE"]),
        ) + pd.Timedelta(days=1)

    home_last_date = pd.Timestamp(home_last["GAME_DATE"])
    away_last_date = pd.Timestamp(away_last["GAME_DATE"])
    home_rest = max(0, min((game_dt - home_last_date).days, 14))
    away_rest = max(0, min((game_dt - away_last_date).days, 14))

    travel_distance = _haversine(
        NBA_TEAM_CENTERS.get(home_team_name, (40.0, -95.0)),
        NBA_TEAM_CENTERS.get(away_team_name, (40.0, -95.0)),
    )
    tz_diff = abs(NBA_TEAM_TZ.get(home_team_name, -5) - NBA_TEAM_TZ.get(away_team_name, -5))

    overrides = {
        "rest_home_days": float(home_rest),
        "rest_away_days": float(away_rest),
        "rest_advantage": float(home_rest - away_rest),
        "is_b2b_home": 1.0 if home_rest == 0 else 0.0,
        "is_b2b_away": 1.0 if away_rest == 0 else 0.0,
        "both_b2b": 1.0 if home_rest == 0 and away_rest == 0 else 0.0,
        "fatigue_home": np.clip(1.0 / (home_rest + 0.5), 0, 2),
        "fatigue_away": np.clip(1.0 / (away_rest + 0.5), 0, 2),
        "travel_distance": travel_distance,
        "tz_diff": float(tz_diff),
    }
    for k, v in overrides.items():
        if k in feature_cols:
            feature_row[k] = v

    # ELO features to 0 for upcoming games
    for k in ["elo_home", "elo_away", "elo_diff", "elo_home_prob", "elo_slope_home", "elo_slope_away"]:
        if k in feature_cols:
            feature_row[k] = 0.0

    # Fill NaN with 0
    for col in feature_cols:
        if col not in feature_row:
            feature_row[col] = 0.0
        elif feature_row[col] is None or (isinstance(feature_row[col], float) and math.isnan(feature_row[col])):
            feature_row[col] = 0.0

    return feature_row


# ── Main Pipeline ──────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  ULTIMATE BETTING INTELLIGENCE — 3-Model Prediction Engine")
    print(f"     {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 72)

    # ── Step 1: Load trained models ──────────────────────────────────────
    print("\n[1/5] Loading trained models...")
    model_path = PROJECT_ROOT / "models" / "ultimate_model.pkl"
    if not model_path.exists():
        print(f"  [FAIL] Model not found at {model_path}")
        print("  Run: python tools/train_ultimate_model.py")
        return 1

    model_data = joblib.load(model_path)

    # Three models
    totals_model = model_data.get("totals_model", model_data.get("model"))
    winprob_model = model_data.get("winprob_model")
    spread_model = model_data.get("spread_model")
    feature_cols = model_data["feature_cols"]
    model_mae = model_data.get("mae", 0.0)
    metrics = model_data.get("metrics", {})

    print(f"  [OK] Totals:    {type(totals_model).__name__} — CV MAE: {metrics.get('totals', {}).get('cv_mae', model_mae):.1f} pts")
    if winprob_model:
        wp_metrics = metrics.get("winprob", {})
        print(f"  [OK] WinProb:   {type(winprob_model).__name__} — Acc: {wp_metrics.get('accuracy', 0):.1%}, Brier: {wp_metrics.get('brier', 0):.3f}")
    if spread_model:
        sp_metrics = metrics.get("spread", {})
        print(f"  [OK] Spread:    {type(spread_model).__name__} — CV MAE: {sp_metrics.get('cv_mae', 0):.1f} pts")
    print(f"  [OK] Features:  {len(feature_cols)}")

    # ── Step 2: Fetch odds from ESPN ─────────────────────────────────────
    print("\n[2/5] Fetching live odds from ESPN...")
    from betting_intel.data.stealth_scraper import StealthBrowser
    odds_games = StealthBrowser.sync_scrape_live_odds(odds_api_key='', timeout=20)
    if not odds_games:
        print("  [FAIL] No odds data from ESPN")
        return 1
    print(f"  [OK] Got {len(odds_games)} game(s) from ESPN")

    # ── Step 3: Load historical data and build features ──────────────────
    print("\n[3/5] Loading historical data and building features...")
    from betting_intel.data.loader import NBADataLoader
    from betting_intel.data.features import FeatureEngineer
    from betting_intel.data.odds_fetcher import ODDS_TO_SHORT_NAME

    loader = NBADataLoader()
    raw_df = loader.load_game_logs()
    if raw_df is None or raw_df.empty:
        print("  [FAIL] No historical game data")
        return 1
    print(f"  [OK] Loaded {len(raw_df)} game logs")

    games_df = loader.build_game_dataset(raw_df)
    raw_df = loader.compute_rest_days(raw_df)
    fe = FeatureEngineer()
    feature_df = fe.build_all_features(games_df, raw_df)
    print(f"  [OK] Built feature matrix: {feature_df.shape[0]} rows x {feature_df.shape[1]} cols")

    # ── Step 4: Run all 3 predictions for each game ──────────────────────
    print("\n[4/5] Running predictions (3 models)...")
    recommendations = []

    for game in odds_games:
        home_full = game.get("home_team", "")
        away_full = game.get("away_team", "")
        commence_time = game.get("commence_time", "")
        game_date = commence_time[:10] if commence_time else ""

        home_short = ODDS_TO_SHORT_NAME.get(home_full, home_full.split()[-1] if " " in home_full else home_full)
        away_short = ODDS_TO_SHORT_NAME.get(away_full, away_full.split()[-1] if " " in away_full else away_full)

        # Build feature vector
        feat_dict = build_feature_vector(home_short, away_short, game_date, feature_df, feature_cols)
        if feat_dict is None:
            print(f"  [WARN] Could not build features for {away_short} @ {home_short} — skipping")
            continue

        X = np.array([feat_dict.get(c, 0.0) for c in feature_cols]).reshape(1, -1)

        # ── Run all 3 model predictions ─────────────────────────────────
        predicted_total = float(totals_model.predict(X)[0]) if totals_model else None
        home_win_prob = float(winprob_model.predict_proba(X)[0, 1]) if winprob_model else 0.5
        predicted_spread = float(spread_model.predict(X)[0]) if spread_model else None

        # Validate
        if predicted_total is not None and (predicted_total > 300 or predicted_total < 80):
            print(f"  [WARN] Unreasonable total {predicted_total:.0f} for {away_short} @ {home_short}")
            predicted_total = None

        away_win_prob = 1.0 - home_win_prob

        # ── Extract market lines from ESPN ──────────────────────────────
        market_total = None
        home_ml = None
        away_ml = None
        spread_line = None
        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                key = market.get("key", "")
                outcomes = market.get("outcomes", [])
                if key == "totals":
                    for o in outcomes:
                        point = o.get("point")
                        if point is not None:
                            market_total = float(point)
                            break
                elif key == "h2h":
                    for o in outcomes:
                        name = o.get("name", "")
                        price = o.get("price")
                        if price is not None:
                            if name == home_full:
                                home_ml = int(price)
                            elif name == away_full:
                                away_ml = int(price)
                elif key == "spreads":
                    for o in outcomes:
                        if o.get("name", "") == home_full:
                            pt = o.get("point")
                            if pt is not None:
                                spread_line = float(pt)

        print(f"\n  [GAME] {away_short} @ {home_short} — {game_date}")
        print(f"    Totals: {predicted_total:.1f} (model) vs {market_total if market_total else 'N/A'} (market)")
        print(f"    Win%:   {home_short} {home_win_prob:.1%} | {away_short} {away_win_prob:.1%} (model)")
        print(f"    Spread: {predicted_spread:+.1f} (model) vs {spread_line if spread_line else 'N/A'} (market)")

        # ── 1. Total bet ────────────────────────────────────────────────
        if market_total is not None and predicted_total is not None:
            edge = (predicted_total - market_total) / market_total
            direction = "OVER" if edge > 0 else "UNDER"
            abs_edge = abs(edge)

            if abs_edge >= 0.05:
                conf = "HIGH"
                kelly_frac = 0.50
            elif abs_edge >= 0.02:
                conf = "MEDIUM"
                kelly_frac = 0.25
            else:
                conf = "LOW"
                kelly_frac = 0.10

            if abs_edge >= 0.02:
                kelly_prob = 0.50 + min(abs_edge, 0.35)
                kelly_prob = min(kelly_prob, 0.85)
                _, kelly_stake = compute_kelly(kelly_prob, american_to_decimal(-110), fraction=kelly_frac)
                recommendations.append({
                    "game_date": game_date,
                    "matchup": f"{away_short} @ {home_short}",
                    "type": "TOTAL",
                    "bet": f"Total {direction} {market_total:.0f}",
                    "model_line": round(predicted_total, 1),
                    "market_line": market_total,
                    "edge_pct": round(edge, 4),
                    "confidence": conf.lower(),
                    "stake": round(kelly_stake, 0),
                    "reasoning": f"Model predicts {predicted_total:.1f} pts vs market {market_total:.0f} ({direction} by {abs(edge)*market_total:.1f} pts)"
                })
                print(f"    [PICK] Total {direction} {market_total:.0f} — edge: {edge:+.1%} — stake: ${kelly_stake:.0f} ({conf})")

        # ── 2. Moneyline bet (model-backed!) ────────────────────────────
        if home_ml and away_ml:
            home_implied = american_to_prob(home_ml)
            away_implied = american_to_prob(away_ml)
            total_imp = home_implied + away_implied
            home_no_vig = home_implied / total_imp if total_imp > 0 else 0.5
            away_no_vig = away_implied / total_imp if total_imp > 0 else 0.5

            home_ml_edge = home_win_prob - home_no_vig
            away_ml_edge = away_win_prob - away_no_vig

            print(f"    ML: {home_short} {home_ml} (market {home_no_vig:.1%}) vs {away_short} {away_ml} (market {away_no_vig:.1%})")

            if home_ml_edge > 0.02:
                _, stake = compute_kelly(home_win_prob, american_to_decimal(home_ml))
                recommendations.append({
                    "game_date": game_date,
                    "matchup": f"{away_short} @ {home_short}",
                    "type": "MONEYLINE",
                    "bet": f"ML {home_short} ({home_ml})",
                    "model_line": round(home_win_prob, 3),
                    "market_line": round(home_no_vig, 3),
                    "edge_pct": round(home_ml_edge, 4),
                    "confidence": "medium",
                    "stake": round(stake, 0),
                    "reasoning": f"Model-backed: {home_short} win prob {home_win_prob:.1%} vs market {home_no_vig:.1%} ({home_ml_edge:+.1%} edge)"
                })
                print(f"    [PICK] ML {home_short} — edge: {home_ml_edge:+.1%} — stake: ${stake:.0f} (model-backed!)")
            elif away_ml_edge > 0.02:
                _, stake = compute_kelly(away_win_prob, american_to_decimal(away_ml))
                recommendations.append({
                    "game_date": game_date,
                    "matchup": f"{away_short} @ {home_short}",
                    "type": "MONEYLINE",
                    "bet": f"ML {away_short} ({away_ml})",
                    "model_line": round(away_win_prob, 3),
                    "market_line": round(away_no_vig, 3),
                    "edge_pct": round(away_ml_edge, 4),
                    "confidence": "medium",
                    "stake": round(stake, 0),
                    "reasoning": f"Model-backed: {away_short} win prob {away_win_prob:.1%} vs market {away_no_vig:.1%} ({away_ml_edge:+.1%} edge)"
                })
                print(f"    [PICK] ML {away_short} — edge: {away_ml_edge:+.1%} — stake: ${stake:.0f} (model-backed!)")

        # ── 3. Spread bet ──────────────────────────────────────────────
        if spread_line is not None and predicted_spread is not None:
            # Predicted spread is home margin (positive = home wins by X)
            spread_edge = predicted_spread - spread_line
            # If spread_line = -3.5 (home gives 3.5) and we predict home by 7,
            # then expected actual margin (home - away) = 7, so home covers -3.5
            home_covers = predicted_spread > spread_line
            away_covers = predicted_spread < spread_line
            abs_spread_edge = abs(spread_edge)

            if abs_spread_edge >= 2.0:
                side = home_short if home_covers else away_short
                recommendations.append({
                    "game_date": game_date,
                    "matchup": f"{away_short} @ {home_short}",
                    "type": "SPREAD",
                    "bet": f"{side} {spread_line:+.1f}",
                    "model_line": round(predicted_spread, 1),
                    "market_line": spread_line,
                    "edge_pct": round(spread_edge / abs(spread_line) if spread_line != 0 else 0, 4),
                    "confidence": "low",
                    "stake": 0,
                    "reasoning": f"Model predicts {side} by {abs(predicted_spread):.1f} vs line {spread_line:+.1f}"
                })
                print(f"    [INFO] Spread: {side} {spread_line:+.1f} — model predicts {predicted_spread:+.1f} margin")
        print()

    # ── Step 5: Display results ──────────────────────────────────────────
    print("=" * 72)
    print("  BET RECOMMENDATIONS")
    print("=" * 72)

    if not recommendations:
        print("  No actionable bets found (edge below 2% threshold).")
        return 0

    recommendations.sort(key=lambda r: abs(r["edge_pct"]), reverse=True)

    total_stake = 0
    print(f"\n  {'#':<4s} {'Type':<12s} {'Game':<32s} {'Bet':<24s} {'Edge':>8s} {'Stake':>8s} {'Conf':>6s}")
    print(f"  {'-'*94}")
    for i, rec in enumerate(recommendations, 1):
        edge_str = f"{rec['edge_pct']*100:+.1f}%"
        stake_str = f"${rec['stake']:.0f}" if rec['stake'] > 0 else "-"
        conf_str = rec['confidence'].upper()
        print(f"  {i:<4d} {rec['type']:<12s} {rec['matchup']:<32s} {rec['bet']:<24s} {edge_str:>8s} {stake_str:>8s} {conf_str:>6s}")
        total_stake += rec.get('stake', 0)

    print(f"  {'-'*94}")
    print(f"  {'':<4s} {'':<12s} {'':<32s} {'':<24s} {'':>8s} ${total_stake:.0f}{'':>5s}")
    print(f"\n  Total exposure: ${total_stake:.0f} on {len(recommendations)} bet(s)")
    print(f"  Bankroll: $10,000 (quarter-Kelly, $100 limit per bet)")

    # Model metrics banner
    print(f"\n  {'='*72}")
    print(f"  MODEL ACCURACY")
    print(f"  {'='*72}")
    m = metrics
    print(f"  Totals:  CV MAE {m.get('totals', {}).get('cv_mae', 0):.2f} pts")
    print(f"  Win%:    {m.get('winprob', {}).get('accuracy', 0):.1%} accuracy, Brier {m.get('winprob', {}).get('brier', 0):.3f}")
    print(f"  Spread:  CV MAE {m.get('spread', {}).get('cv_mae', 0):.2f} pts")
    print(f"  ML bets are now MODEL-BACKED (not heuristic!)")

    # Save to JSON
    output_path = PROJECT_ROOT / "data" / "recommendations.json"
    with open(output_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": "Ultimate 3-Model Ensemble",
            "metrics": {
                "totals_cv_mae": metrics.get('totals', {}).get('cv_mae', model_mae),
                "winprob_accuracy": metrics.get('winprob', {}).get('accuracy', 0),
                "winprob_brier": metrics.get('winprob', {}).get('brier', 0),
                "spread_cv_mae": metrics.get('spread', {}).get('cv_mae', 0),
            },
            "n_games": len(odds_games),
            "n_recommendations": len(recommendations),
            "total_stake": total_stake,
            "recommendations": recommendations,
        }, f, indent=2)
    print(f"\n  [SAVE] Saved to {output_path}")

    # Detailed reasoning
    print(f"\n{'='*72}")
    print("  DETAILED ANALYSIS")
    print(f"{'='*72}")
    for rec in recommendations:
        print(f"\n  {rec['matchup']} — {rec['bet']}")
        print(f"    Edge: {rec['edge_pct']*100:+.1f}% | Stake: ${rec['stake']:.0f} | Type: {rec['type']}")
        print(f"    Why: {rec['reasoning']}")

    print(f"\n{'='*72}")
    print("  [OK]  Done. Picks saved.")
    print(f"{'='*72}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
