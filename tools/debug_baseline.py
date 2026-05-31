"""
Debug: Trace exactly why baseline model predictions aren't flowing through to edge calculation.

This loads data, runs ONE walk-forward window, and prints every step of the pipeline
to find where baseline_prediction gets dropped.
"""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# Add src/ to path so we can import from betting_intel.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer
from betting_intel.backtesting.engine import WalkForwardEngine, BacktestResult
from betting_intel.models.predictors import TotalPointsPredictor

# ── Load data ───────────────────────────────────────────────────────
print("=" * 65)
print("  BASELINE MODEL DEBUG")
print("=" * 65)

loader = NBADataLoader()
fe = FeatureEngineer()
engine = WalkForwardEngine(train_window=200, step=20, min_train=50)

raw_df = loader.load_game_logs()
games_df = loader.build_game_dataset(raw_df)
raw_df = loader.compute_rest_days(raw_df)

feature_df = fe.build_all_features(games_df, raw_df)
feature_cols = fe.select_features(feature_df)

print(f"\n  Total features: {len(feature_cols)}")
print(f"  Sample: {feature_cols[:5]}...")

clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols)//2).copy()
clean_df = clean_df.reset_index(drop=True)
print(f"  Clean samples: {len(clean_df)}")

# ── Check which basic features exist ────────────────────────────────
basic_prefixes = ["avg_pts_", "avg_pm_", "avg_ts_", "avg_efg_",
                  "pace_", "rest_", "is_b2b_", "tz_",
                  "home_rest", "away_rest", "home_is_b2b", "away_is_b2b",
                  "net_rating_", "opp_avg_pts_allowed_", "opp_avg_pts_scored_"]

basic_cols = []
for col in feature_cols:
    for prefix in basic_prefixes:
        if col.startswith(prefix):
            basic_cols.append(col)
            break

print(f"\n  Basic features selected: {len(basic_cols)}")
for c in basic_cols[:20]:
    print(f"    - {c}")
if len(basic_cols) > 20:
    print(f"    ... and {len(basic_cols) - 20} more")
if len(basic_cols) < 5:
    print(f"  [!] FEWER THAN 5 basic features! Would fall back to feature_cols[:20]")
    print(f"  [!] Fallback cols: {feature_cols[:20]}")
    basic_cols = feature_cols[:20]

# Check that all basic_cols exist in clean_df
missing_cols = [c for c in basic_cols if c not in clean_df.columns]
if missing_cols:
    print(f"\n  [!] MISSING columns in df: {missing_cols[:5]}...")
else:
    print(f"\n  All {len(basic_cols)} basic features exist in df.")

# ── Manually run ONE walk-forward window ────────────────────────────
df = clean_df.sort_values("GAME_DATE").reset_index(drop=True)
n = len(df)

start_idx = 0
train_end = start_idx + 200
test_end = min(train_end + 20, n)

train_df = df.iloc[start_idx:train_end]
test_df = df.iloc[train_end:test_end]

print(f"\n  Train rows: {len(train_df)}, Test rows: {len(test_df)}")
print(f"  Train date range: {train_df['GAME_DATE'].iloc[0].date()} to {train_df['GAME_DATE'].iloc[-1].date()}")
print(f"  Test date range:  {test_df['GAME_DATE'].iloc[0].date()} to {test_df['GAME_DATE'].iloc[-1].date()}")

# Train MAIN model (LightGBM)
X_train = train_df[feature_cols].dropna()
y_train = train_df.loc[X_train.index, "total_points"]
print(f"\n  X_train after dropna: {X_train.shape}")

main_model = TotalPointsPredictor("lightgbm")
main_model.fit(X_train.values, y_train.values)
print(f"  Main model (LightGBM) trained OK")

# Train BASELINE model (Ridge on basic features)
baseline_model = None
X_base_train = train_df[basic_cols].dropna()
y_base_train = train_df.loc[X_base_train.index, "total_points"]

print(f"  X_base_train after dropna: {X_base_train.shape}")

if len(X_base_train) >= 50:
    baseline_model = TotalPointsPredictor("ridge")
    baseline_model.fit(X_base_train.values, y_base_train.values)
    print(f"  Baseline model (Ridge on {len(basic_cols)} basic features) trained OK")
else:
    print(f"  [!] X_base_train too small: {len(X_base_train)}")
    baseline_model = None

# Test
X_test = test_df[feature_cols].dropna()
y_pred = main_model.predict(X_test.values)
y_actual = test_df.loc[X_test.index, "total_points"].values

print(f"\n  Test predictions: {len(y_pred)}")

# Baseline predictions
baseline_pred_map = {}
if baseline_model is not None:
    X_base_test = test_df[basic_cols].dropna()
    print(f"  X_base_test after dropna: {X_base_test.shape}")
    if len(X_base_test) > 0:
        y_baseline_pred = baseline_model.predict(X_base_test.values)
        for b_idx, b_val in zip(X_base_test.index, y_baseline_pred):
            baseline_pred_map[b_idx] = float(b_val)
        print(f"  Baseline predictions: {len(baseline_pred_map)}")
    else:
        print(f"  [!] X_base_test is empty!")
else:
    print(f"  [!] No baseline model!")

# Check alignment
test_indices = list(X_test.index)
baseline_available = sum(1 for idx in test_indices if idx in baseline_pred_map)
print(f"\n  Test indices with matching baseline pred: {baseline_available} / {len(test_indices)}")

# ── Compare predictions ────────────────────────────────────────────
print(f"\n  -- PREDICTION COMPARISON (first 10 test rows) --")
print(f"  {'Idx':>6} {'Main Pred':>10} {'Baseline':>10} {'Actual':>8} {'Edge%':>8} {'Win?':>6}")
print(f"  {'-'*48}")

for i, idx in enumerate(test_indices[:10]):
    main_p = y_pred[i]
    actual = y_actual[i]
    baseline_p = baseline_pred_map.get(idx)
    
    if baseline_p is not None and baseline_p > 0:
        edge = (main_p - baseline_p) / baseline_p
    else:
        # Fallback to column-based
        row = df.loc[idx]
        market_line = row.get("market_line_baseline", row.get("trailing_avg_total_10g", main_p))
        edge = (main_p - market_line) / market_line if market_line > 0 else 0
        baseline_p = market_line  # For display
    
    edge_str = f"{edge*100:+.2f}%" if abs(edge) < 999 else f"{edge:+.2f}"
    won = (edge > 0 and actual > baseline_p) or (edge < 0 and actual < baseline_p)
    
    print(f"  {idx:>6} {main_p:>10.1f} {baseline_p:>10.1f} {actual:>8.1f} {edge_str:>8} {'WIN' if won else 'LOSS':>6}")

# Check if any bets would be placed with edge > 2%
if baseline_model is not None:
    bets_with_edge = 0
    for idx in test_indices:
        baseline_p = baseline_pred_map.get(idx)
        if baseline_p is not None:
            main_p = y_pred[list(test_indices).index(idx)]
            if baseline_p > 0:
                edge = (main_p - baseline_p) / baseline_p
                if abs(edge) >= 0.02:
                    bets_with_edge += 1
    
    total_possible = len([idx for idx in test_indices if idx in baseline_pred_map])
    print(f"\n  Bets with edge >= 2% (baseline model): {bets_with_edge} / {total_possible}")
else:
    # Check fallback behavior
    bets_with_edge = 0
    for i, idx in enumerate(test_indices):
        row = df.loc[idx]
        market_line = row.get("market_line_baseline", row.get("trailing_avg_total_10g", y_pred[i]))
        if market_line > 0:
            edge = (y_pred[i] - market_line) / market_line
            if abs(edge) >= 0.02:
                bets_with_edge += 1
    
    print(f"\n  (Using fallback column-based market line)")
    print(f"  Bets with edge >= 2% (fallback column): {bets_with_edge} / {len(test_indices)}")

# ── Check fallback columns ──────────────────────────────────────────
print(f"\n  -- COLUMN CHECK --")
sample_row = df.iloc[test_indices[0]]
has_market_line = "market_line_baseline" in df.columns
has_trailing_avg = "trailing_avg_total_10g" in df.columns
print(f"  has market_line_baseline: {has_market_line}")
print(f"  has trailing_avg_total_10g: {has_trailing_avg}")

if has_market_line:
    vals = df["market_line_baseline"].dropna().values[:5]
    print(f"  market_line_baseline sample: {vals}")
if has_trailing_avg:
    vals = df["trailing_avg_total_10g"].dropna().values[:5]
    print(f"  trailing_avg_total_10g sample: {vals}")

print(f"\n  Done.")
