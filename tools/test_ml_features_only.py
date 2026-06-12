#!/usr/bin/env python3
"""Quick test: verify new moneyline features are computed correctly and have predictive signal."""
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer

print("Loading NBA data...")
loader = NBADataLoader()
fe = FeatureEngineer()
raw_df = loader.load_game_logs()
games_df = loader.build_game_dataset(raw_df)
raw_df = loader.compute_rest_days(raw_df)
feature_df = fe.build_all_features(games_df, raw_df)
feature_cols = fe.select_features(feature_df)

y = (feature_df["point_diff"].values > 0).astype(int)

# List all moneyline-specific features
ml_features = sorted([c for c in feature_cols if any(k in c for k in [
    "composite_power", "power_diff", "form_diff", "perf_vs_expected",
    "consistency", "h2h_win_rate", "h2h_avg_margin", "home_away_split",
    "recent_win_pct",
])])

print(f"\nTotal features: {len(feature_cols)}")
print(f"Moneyline features: {len(ml_features)}")
for f in ml_features:
    vals = feature_df[f].fillna(0.5).values
    try:
        auc = roc_auc_score(y, vals)
        auc_str = f"AUC={auc:.4f}"
    except:
        auc_str = "AUC=ERR"
    marker = " <--" if auc > 0.53 else ""
    print(f"  {f:40s}  {auc_str}{marker}")

# Combined signal tests
print("\n--- Combined Signal ---")
# Check that power_diff alone has signal
vals = feature_df["power_diff"].fillna(0).values
auc = roc_auc_score(y, vals)
print(f"  power_diff AUC: {auc:.4f}")

# Check form_diff
if "form_diff" in feature_df.columns:
    vals = feature_df["form_diff"].fillna(0).values
    auc = roc_auc_score(y, vals)
    print(f"  form_diff AUC:   {auc:.4f}")

# Check perf_vs_expected_diff
if "perf_vs_expected_diff" in feature_df.columns:
    vals = feature_df["perf_vs_expected_diff"].fillna(0).values
    auc = roc_auc_score(y, vals)
    print(f"  perf_vs_exp AUC: {auc:.4f}")

# Check consistency_diff
if "consistency_diff" in feature_df.columns:
    vals = feature_df["consistency_diff"].fillna(0).values
    auc = roc_auc_score(y, vals)
    print(f"  consistency AUC: {auc:.4f}")

# Simple combined model: equal-weighted average of top individual features
print("\n--- Equal-Weight Ensemble (no training) ---")
top_features = [f for f in ml_features if roc_auc_score(y, feature_df[f].fillna(0.5).values) > 0.52]
if top_features:
    combined = np.mean([feature_df[f].fillna(0.5).values for f in top_features], axis=0)
    auc = roc_auc_score(y, combined)
    print(f"  {len(top_features)} features combined: AUC={auc:.4f}")
    for f in top_features:
        print(f"    + {f}")

print("\nDone.")
