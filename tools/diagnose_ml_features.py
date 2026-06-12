#!/usr/bin/env python3
"""Diagnose why MoneylinePredictor AUC=0.5 — check per-feature predictive power."""

import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer

print("=== Loading NBA data...")
loader = NBADataLoader()
fe = FeatureEngineer()
raw_df = loader.load_game_logs()
games_df = loader.build_game_dataset(raw_df)
raw_df = loader.compute_rest_days(raw_df)
feature_df = fe.build_all_features(games_df, raw_df)
feature_cols = fe.select_features(feature_df)

print(f"  Features: {len(feature_cols)}, Games: {len(feature_df)}")

clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols)//2).copy()
clean_df = clean_df.reset_index(drop=True)
y = (clean_df["point_diff"].values > 0).astype(int)

print(f"  Class balance: {y.sum()} wins / {len(y)-y.sum()} losses ({y.mean():.1%} home win rate)")
print()

# ── Check ELO features ──
print("=== ELO Feature Diagnostic ===")
for col in ["elo_diff", "elo_home_prob", "elo_home", "elo_away", "elo_slope_home", "elo_slope_away"]:
    if col in clean_df.columns:
        vals = clean_df[col].fillna(0).values
        try:
            auc = roc_auc_score(y, vals)
            brier = brier_score_loss(y, vals)
            # Accuracy at 0.5 threshold
            acc = ((vals >= 0.5) == y).mean()
            print(f"  {col:25s}  AUC={auc:.4f}  Brier={brier:.4f}  Acc={acc:.1%}")
        except Exception as e:
            print(f"  {col:25s}  ERROR: {e}")

print()

# ── Check top features by mutual information ──
print("=== Top 20 Features by Mutual Information (for classification) ===")
X = clean_df[feature_cols].fillna(0).values
mi = mutual_info_classif(X, y, random_state=42)
top_idx = np.argsort(mi)[-30:][::-1]

for i, idx in enumerate(top_idx):
    col = feature_cols[idx]
    mi_val = mi[idx]
    vals = clean_df[col].fillna(0).values
    try:
        auc = roc_auc_score(y, vals)
    except:
        auc = 0.5
    star = " ★" if auc > 0.53 else ""
    print(f"  {i+1:2d}. MI={mi_val:.4f}  AUC={auc:.4f}  {col}{star}")

print()

# ── Check seasonality ──
print("=== Season Splits ===")
if "SEASON_ID_home" in clean_df.columns:
    seasons = clean_df["SEASON_ID_home"].unique()
    for s in sorted(seasons):
        mask = clean_df["SEASON_ID_home"] == s
        if mask.sum() > 0:
            home_win_rate = y[mask.values].mean()
            n = mask.sum()
            print(f"  Season {s}: {n} games, home win rate = {home_win_rate:.1%}")

# ── Feature pair analysis ──
print()
print("=== Best Feature Pairs (simple average) ===")
# Check if combining top features improves AUC
feature_names = []
for idx in top_idx[:5]:
    feature_names.append(feature_cols[idx])

for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        col_i = clean_df[feature_names[i]].fillna(0).values
        col_j = clean_df[feature_names[j]].fillna(0).values
        # Simple average
        combined = (col_i + col_j) / 2
        try:
            auc = roc_auc_score(y, combined)
            if auc > 0.55:
                print(f"  AUC={auc:.4f}  {feature_names[i]} + {feature_names[j]}")
        except:
            pass

print()
print("=== Done ===")
