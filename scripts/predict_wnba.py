"""
WNBA Prediction — trains a LightGBM model on ESPN historical data and
predicts upcoming WNBA games with model-based totals.

Usage:
    python scripts/predict_wnba.py
    python scripts/predict_wnba.py --json
"""

import sys
import os
import json
import warnings
from datetime import date, timedelta
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_src = _project_root / "src"
for p in (_project_root, _src):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ["LOG_LEVEL"] = "CRITICAL"
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.CRITICAL)

import pandas as pd
import numpy as np

from betting_intel.data.espn_hoops import ESPNLeagueSource

today = date.today()
tomorrow = today + timedelta(days=1)


def fetch_historical() -> pd.DataFrame:
    """Fetch WNBA historical data from ESPN."""
    source = ESPNLeagueSource()
    df = source.load_historical("wnba", seasons=[2025, 2026])
    if df.empty:
        df = source.load_historical("wnba", seasons=[2026])
    if df.empty:
        print("  FAILED: No WNBA historical data from ESPN")
        sys.exit(1)
    return df


def engineer_features(df: pd.DataFrame) -> tuple:
    """Build feature vectors and targets from game results."""
    df = df.copy()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["total_points"] = df["home_score"] + df["away_score"]
    df = df.dropna(subset=["total_points"])
    df = df.sort_values("date").reset_index(drop=True)

    features, targets = [], []
    for i in range(20, len(df)):
        row = df.iloc[i]
        home, away = row["home_team"], row["away_team"]
        past = df.iloc[:i]

        home_games = past[(past["home_team"] == home) | (past["away_team"] == home)].tail(10)
        away_games = past[(past["home_team"] == away) | (past["away_team"] == away)].tail(10)

        if len(home_games) < 3 or len(away_games) < 3:
            continue

        # Home team avg points
        home_total = home_count = 0
        for _, g in home_games.iterrows():
            home_total += g["home_score"] if g["home_team"] == home else g["away_score"]
            home_count += 1
        home_avg = home_total / home_count

        # Home games only
        home_home = past[past["home_team"] == home].tail(5)
        home_home_avg = home_home["home_score"].mean() if len(home_home) > 0 else home_avg

        # Away team avg points
        away_total = away_count = 0
        for _, g in away_games.iterrows():
            away_total += g["away_score"] if g["away_team"] == away else g["home_score"]
            away_count += 1
        away_avg = away_total / away_count

        # Away games only
        away_away = past[past["away_team"] == away].tail(5)
        away_away_avg = away_away["away_score"].mean() if len(away_away) > 0 else away_avg

        # Recent form (last 3 games)
        home_recent = home_games.tail(3)
        away_recent = away_games.tail(3)
        home_wins = sum(
            1 for _, g in home_recent.iterrows()
            if (g["home_team"] == home and g["home_score"] > g["away_score"])
            or (g["away_team"] == home and g["away_score"] > g["home_score"])
        )
        away_wins = sum(
            1 for _, g in away_recent.iterrows()
            if (g["home_team"] == away and g["home_score"] > g["away_score"])
            or (g["away_team"] == away and g["away_score"] > g["home_score"])
        )

        # Head-to-head trend
        h2h = past[
            ((past["home_team"] == home) & (past["away_team"] == away))
            | ((past["home_team"] == away) & (past["away_team"] == home))
        ].tail(5)
        h2h_avg = h2h["home_score"].mean() + h2h["away_score"].mean() if len(h2h) > 0 else 165.0

        features.append({
            "home_avg_pts": home_avg,
            "away_avg_pts": away_avg,
            "home_home_avg": home_home_avg,
            "away_away_avg": away_away_avg,
            "home_form_3": home_wins / 3,
            "away_form_3": away_wins / 3,
            "h2h_avg_total": h2h_avg,
            "home_games_played": len(home_games),
            "away_games_played": len(away_games),
        })
        targets.append(row["total_points"])

    return pd.DataFrame(features), np.array(targets)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="WNBA Prediction")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    print()
    print("=" * 65)
    print("  WNBA PREDICTION TRAINING & FORECAST")
    print("  %s" % today)
    print("=" * 65)

    # Step 1: Fetch data
    print("\n[1/5] Fetching WNBA historical data from ESPN...")
    df = fetch_historical()
    print("  Games loaded: %d" % len(df))

    # Step 2: Engineer features
    print("\n[2/5] Engineering features from %d games..." % len(df))
    X, y = engineer_features(df)
    teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    print("  Feature vectors: %d, Teams: %d" % (len(X), len(teams)))

    # Step 3: Train model
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    print("\n[3/5] Training LightGBM (%d train, %d test)..." % (len(X_train), len(X_test)))
    from lightgbm import LGBMRegressor
    from sklearn.metrics import r2_score, mean_absolute_error

    model = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=16, random_state=42, verbosity=-1,
    )
    model.fit(X_train, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    print("  Train R2: %.3f, Test R2: %.3f, Test MAE: %.2f" % (train_r2, test_r2, test_mae))

    # Feature importance
    imp = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print("\n  Top features:")
    for _, r in imp.head(5).iterrows():
        print("    %-20s %d" % (r["feature"], r["importance"]))

    # Step 4: Fetch upcoming
    print("\n[4/5] Fetching upcoming WNBA games...")
    source = ESPNLeagueSource()
    upcoming = source.load_upcoming("wnba", limit=20)
    if upcoming is None or upcoming.empty:
        print("  No upcoming WNBA games found on ESPN")
        return

    print("  Games: %d" % len(upcoming))
    for _, r in upcoming.iterrows():
        print("    %s - %s @ %s" % (str(r.get("date", "?"))[:10], r.get("away_team", "?"), r.get("home_team", "?")))

    # Step 5: Predict
    print("\n[5/5] Predicting upcoming WNBA games...")
    wnba_avg = 165.0

    predictions = []
    for _, row in upcoming.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        gdate = str(row.get("date", ""))[:10]

        # Get historical stats for these teams
        home_g = df[(df["home_team"] == home) | (df["away_team"] == home)].tail(10)
        away_g = df[(df["home_team"] == away) | (df["away_team"] == away)].tail(10)
        home_pts = home_g["home_score"].mean() if not home_g.empty else 82
        away_pts = away_g["away_score"].mean() if not away_g.empty else 82
        home_home_avg = df[df["home_team"] == home]["home_score"].tail(5).mean() if len(df[df["home_team"] == home]) > 0 else home_pts
        away_away_avg = df[df["away_team"] == away]["away_score"].tail(5).mean() if len(df[df["away_team"] == away]) > 0 else away_pts

        feat = pd.DataFrame([{
            "home_avg_pts": home_pts,
            "away_avg_pts": away_pts,
            "home_home_avg": home_home_avg,
            "away_away_avg": away_away_avg,
            "home_form_3": 0.5,
            "away_form_3": 0.5,
            "h2h_avg_total": wnba_avg,
            "home_games_played": min(len(home_g), 10),
            "away_games_played": min(len(away_g), 10),
        }])

        pred = model.predict(feat)[0]
        edge = (pred - wnba_avg) / wnba_avg
        direction = "over" if edge > 0 else "under"

        predictions.append({
            "date": gdate,
            "home": home,
            "away": away,
            "predicted_total": round(float(pred), 1),
            "market_est": wnba_avg,
            "edge_pct": round(float(edge) * 100, 1),
            "direction": direction,
        })

    # Output
    if args.json:
        print(json.dumps({
            "model": {"train_r2": round(float(train_r2), 3), "test_r2": round(float(test_r2), 3), "test_mae": round(float(test_mae), 2)},
            "predictions": predictions,
        }, indent=2))
    else:
        print()
        print("=" * 65)
        print("  WNBA PREDICTIONS")
        print("=" * 65)
        for p in predictions:
            icon = "[UP]" if p["direction"] == "over" else "[DN]"
            rel = "TODAY" if p["date"] == str(today) else ("TOM" if p["date"] == str(tomorrow) else p["date"])
            print("  %-5s %s %-20s @ %-20s  pred=%5.1f  mkt=%.0f  edge=%+.1f%%  %s" % (
                rel, icon, p["away"], p["home"],
                p["predicted_total"], p["market_est"], p["edge_pct"], p["direction"],
            ))
        print()
        print("  Model: %d train / %d test games  |  Test R2: %.3f  MAE: %.2f pts" % (
            len(X_train), len(X_test), test_r2, test_mae))
        print("=" * 65)


if __name__ == "__main__":
    main()
