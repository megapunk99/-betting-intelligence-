#!/usr/bin/env python3
"""Deep dive: manually run ONE walk-forward fold and inspect every prediction."""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"

# Add src/ to path so we can import from betting_intel.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer
from betting_intel.models.predictors import TotalPointsPredictor
from betting_intel import config as cfg

print("Loading data...")
loader = NBADataLoader()
raw_df = loader.load_game_logs()
games_df = loader.build_game_dataset(raw_df)
raw_df = loader.compute_rest_days(raw_df)

fe = FeatureEngineer()
feature_df = fe.build_all_features(games_df, raw_df)
feature_cols = fe.select_features(feature_df)
print("Features: %d" % len(feature_cols))

clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols)//2).copy()
clean_df = clean_df.sort_values("GAME_DATE").reset_index(drop=True)
print("Clean samples: %d" % len(clean_df))

# Manual Walk-Forward (ONE fold)
train_window = cfg.WALK_FORWARD_WINDOW
step = cfg.WALK_FORWARD_STEP
threshold = cfg.MIN_EDGE_THRESHOLD

train_df = clean_df.iloc[0:train_window]
test_df = clean_df.iloc[train_window:train_window + step]

print()
print("Train: %s to %s (%d rows)" % (train_df['GAME_DATE'].min().date(), train_df['GAME_DATE'].max().date(), len(train_df)))
print("Test:  %s to %s (%d rows)" % (test_df['GAME_DATE'].min().date(), test_df['GAME_DATE'].max().date(), len(test_df)))

# Check temporal split
if test_df['GAME_DATE'].min() <= train_df['GAME_DATE'].max():
    print("FAIL: Test data NOT after train data -- temporal split leak!")
else:
    print("OK: Test is strictly after train")

# Train Main Model
X_train = train_df[feature_cols].dropna()
y_train = train_df.loc[X_train.index, "total_points"]
print("X_train shape: %s" % str(X_train.shape))

main_model = TotalPointsPredictor("ridge")
main_model.fit(X_train.values, y_train.values)

# Train Baseline Models
baseline_features = [c for c in feature_cols
    if any(c.startswith(p) for p in ["avg_pts_5g_", "avg_pts_10g_", "avg_pts_allowed_",
                                       "avg_pace_5g_", "ema_pts_5g_"])]
X_base_train = train_df[baseline_features].dropna()
y_base = train_df.loc[X_base_train.index, "total_points"]
baseline_ridge = Ridge(alpha=1.0)
baseline_ridge.fit(X_base_train.values, y_base.values)

# Baseline B: Ridge on ALL features
X_full_train = train_df[feature_cols].dropna()
y_full = train_df.loc[X_full_train.index, "total_points"]
full_ridge = Ridge(alpha=10.0)
full_ridge.fit(X_full_train.values, y_full.values)

# Test
X_test = test_df[feature_cols].dropna()
X_base_test = test_df[baseline_features].dropna()
common_idx = X_test.index.intersection(X_base_test.index)
y_actual_c = test_df.loc[common_idx, "total_points"].values
y_main_c = main_model.predict(X_test.loc[common_idx].values)
y_baseline = baseline_ridge.predict(X_base_test.loc[common_idx].values)
y_full = full_ridge.predict(X_test.loc[common_idx].values)

# Analyze
rows = []
for i, idx in enumerate(common_idx):
    row = test_df.loc[idx]
    naive_line = row.get("trailing_avg_total_10g", y_baseline[i])

    edge_vs_naive = (y_main_c[i] - naive_line) / max(naive_line, 1)
    edge_vs_baseline = (y_main_c[i] - y_baseline[i]) / max(y_baseline[i], 1)
    edge_vs_full = (y_main_c[i] - y_full[i]) / max(y_full[i], 1)

    win_vs_naive = (y_main_c[i] > naive_line and y_actual_c[i] > naive_line) or \
                   (y_main_c[i] < naive_line and y_actual_c[i] < naive_line)
    win_vs_baseline = (y_main_c[i] > y_baseline[i] and y_actual_c[i] > y_baseline[i]) or \
                      (y_main_c[i] < y_baseline[i] and y_actual_c[i] < y_baseline[i])
    win_vs_full = (y_main_c[i] > y_full[i] and y_actual_c[i] > y_full[i]) or \
                  (y_main_c[i] < y_full[i] and y_actual_c[i] < y_full[i])

    rows.append({
        "idx": idx,
        "date": str(row["GAME_DATE"].date()),
        "home": row.get("TEAM_NAME_home", "?"),
        "away": row.get("TEAM_NAME_away", "?"),
        "actual": float(y_actual_c[i]),
        "main_pred": float(y_main_c[i]),
        "naive_line": float(naive_line) if not pd.isna(naive_line) else float(y_baseline[i]),
        "baseline_pred": float(y_baseline[i]),
        "full_pred": float(y_full[i]),
        "edge_naive": float(edge_vs_naive),
        "edge_baseline": float(edge_vs_baseline),
        "edge_full": float(edge_vs_full),
        "win_naive": bool(win_vs_naive),
        "win_baseline": bool(win_vs_baseline),
        "win_full": bool(win_vs_full),
    })

results_df = pd.DataFrame(rows)

# === Print Summary ===
print()
print("SINGLE-FOLD DIAGNOSTIC RESULTS")
print("=" * 60)

# Checks
print()
print("1. Win rates by market line (edge > %.0f%%):" % (threshold * 100))
for label, edge_col, win_col in [
    ('Naive Trailing Avg', 'edge_naive', 'win_naive'),
    ('Ridge Baseline (5 feat)', 'edge_baseline', 'win_baseline'),
    ('Ridge Full (159 feat, a=10)', 'edge_full', 'win_full'),
]:
    mask = results_df[edge_col].abs() >= threshold
    subset = results_df[mask]
    if len(subset) > 0:
        wr = subset[win_col].mean()
        losses = subset[~subset[win_col]]
        loss_str = "%d losses" % len(losses) if len(losses) > 0 else "NO LOSSES"
        print("  %-35s: bets=%3d  wr=%.1f%%  %s" % (label, len(subset), wr*100, loss_str))
    else:
        print("  %-35s: no bets" % label)

print()
print("2. Systematic bias (main_pred - actual):")
print("  Main model bias:   %.2f points" % (results_df['main_pred'] - results_df['actual']).mean())
print("  Naive line bias:   %.2f points" % (results_df['naive_line'] - results_df['actual']).mean())
print("  Baseline Ridge:    %.2f points" % (results_df['baseline_pred'] - results_df['actual']).mean())
print("  Full Ridge:        %.2f points" % (results_df['full_pred'] - results_df['actual']).mean())

print()
print("3. Model accuracy:")
print("  Main model MAE:    %.2f points" % abs(results_df['main_pred'] - results_df['actual']).mean())
print("  Naive line MAE:    %.2f points" % abs(results_df['naive_line'] - results_df['actual']).mean())
print("  Baseline Ridge:    %.2f points" % abs(results_df['baseline_pred'] - results_df['actual']).mean())
print("  Full Ridge:        %.2f points" % abs(results_df['full_pred'] - results_df['actual']).mean())

print()
print("4. Directional agreement (main_pred vs actual, same side of market line):")
for label, line_col in [('Naive Line', 'naive_line'), ('Baseline Ridge', 'baseline_pred'), ('Full Ridge', 'full_pred')]:
    above_both = (results_df['main_pred'] > results_df[line_col]) & (results_df['actual'] > results_df[line_col])
    below_both = (results_df['main_pred'] < results_df[line_col]) & (results_df['actual'] < results_df[line_col])
    disagree = ((results_df['main_pred'] > results_df[line_col]) & (results_df['actual'] < results_df[line_col])) | \
               ((results_df['main_pred'] < results_df[line_col]) & (results_df['actual'] > results_df[line_col]))
    total = len(results_df)
    n_disagree = disagree.sum()
    n_disagree_large = 0
    if n_disagree > 0:
        edges = abs(results_df['main_pred'] - results_df[line_col]) / results_df[line_col].clip(lower=1)
        n_disagree_large = ((edges > threshold) & disagree).sum()
    print("  %-20s: agree=%d (%.1f%%)  disagree=%d  disagree+edge>%.0f%%=%d" % (
        label, total - n_disagree, (total - n_disagree)/total*100, n_disagree, threshold*100, n_disagree_large))

# Show the disagree cases with large edge
if 'full_pred' in results_df.columns:
    for line_label, line_col in [('Naive Line', 'naive_line'), ('Full Ridge', 'full_pred')]:
        disagree = ((results_df['main_pred'] > results_df[line_col]) & (results_df['actual'] < results_df[line_col])) | \
                   ((results_df['main_pred'] < results_df[line_col]) & (results_df['actual'] > results_df[line_col]))
        if disagree.sum() > 0:
            edges = abs(results_df['main_pred'] - results_df[line_col]) / results_df[line_col].clip(lower=1)
            bad = (edges > threshold) & disagree
            if bad.sum() > 0:
                print()
                print("5. Cases where model disagrees with actual (edge > %.0f%% vs %s):" % (threshold*100, line_label))
                bad_cases = results_df[bad].head(10)
                for _, r in bad_cases.iterrows():
                    print("  %s %12s vs %-12s pred=%.1f %s=%.1f actual=%.1f" % (
                        r['date'], r['home'], r['away'], r['main_pred'], line_label[:8], r[line_col], r['actual']))

# Show first 10 bets
print()
print("6. Sample predictions (first 10):")
mask = results_df['edge_naive'].abs() >= threshold
sample = results_df[mask].head(10)
for _, r in sample.iterrows():
    print("  %s %12s vs %-12s pred=%.1f naive=%.1f baseln=%.1f full=%.1f actual=%.1f %s" % (
        r['date'], r['home'], r['away'],
        r['main_pred'], r['naive_line'], r['baseline_pred'], r['full_pred'],
        r['actual'], 'WIN' if r['win_naive'] else 'LOSS'))
