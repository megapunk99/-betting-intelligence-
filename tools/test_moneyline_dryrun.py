#!/usr/bin/env python3
"""Quick dry-run of the MoneylinePredictor on both real NBA data and synthetic data."""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
from betting_intel.models.moneyline_predictor import MoneylinePredictor

print("=" * 60)
print("  TEST 1: Synthetic data — full API smoke test")
print("=" * 60)
np.random.seed(42)
n = 200
X_syn = np.random.randn(n, 10)
y_syn = (X_syn[:, 0] + X_syn[:, 1] + np.random.randn(n) * 0.5 > 0).astype(int)
feature_cols = [f"f{i}" for i in range(10)]

p = MoneylinePredictor(calibrate=True, feature_selection=False)
p.fit(X_syn, y_syn, feature_names=feature_cols)
probs = p.predict_proba(X_syn)
preds = p.predict(X_syn)
m = p.evaluate(X_syn, y_syn)

print(f"  Brier:    {m['brier']:.4f}")
print(f"  LogLoss:  {m['log_loss']:.4f}")
print(f"  Accuracy: {m['accuracy']:.1%}")
print(f"  AUC:      {m['auc_roc']:.3f}")
print(f"  CalErr:   {m['calibration_error']:.4f}")
print(f"  Models:   {list(p.models_.keys())}")
print(f"  Calibrated: {list(p.calibrated_models_.keys())}")

# Cross-validate
cv = p.cross_validate(X_syn, y_syn, feature_names=feature_cols, n_splits=3)
print(f"  CV Folds: {cv['n_folds']}, CV Brier: {cv['avg_brier']:.4f}")

# Feature importance
fi = p.get_feature_importance(top_n=3)
for name, top_k in fi.items():
    print(f"  Top-3 [{name}]: {list(top_k.keys())}")

print()
print("=" * 60)
print("  TEST 2: Real NBA data — walk-forward CV")
print("=" * 60)

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer

loader = NBADataLoader()
fe = FeatureEngineer()

raw_df = loader.load_game_logs()
if raw_df is None or len(raw_df) == 0:
    print("  [SKIP] No NBA database found — skipping real-data test")
    sys.exit(0)

games_df = loader.build_game_dataset(raw_df)
raw_df = loader.compute_rest_days(raw_df)
feature_df = fe.build_all_features(games_df, raw_df)
feature_cols = fe.select_features(feature_df)

print(f"  Features: {len(feature_cols)}, Games: {len(feature_df)}")

clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2).copy()
clean_df = clean_df.reset_index(drop=True)
X = clean_df[feature_cols].fillna(0).values
y = (clean_df["point_diff"].values > 0).astype(int)
print(f"  X shape: {X.shape}, y: {y.sum()} wins / {len(y) - y.sum()} losses")

predictor = MoneylinePredictor(calibrate=True, n_folds=3, feature_selection=True, n_select=30)
print()
print("  Running walk-forward CV (3 splits)...")
results = predictor.cross_validate(X, y, feature_names=feature_cols, n_splits=3)

print()
print("=" * 40)
print("  REAL-DATA CV RESULTS")
print("=" * 40)
print(f"  Folds:       {results['n_folds']}")
print(f"  OOS samples: {results['n_oos']}")
print(f"  Avg Brier:   {results['avg_brier']:.4f}")
print(f"  Avg LogLoss: {results['avg_log_loss']:.4f}")
print(f"  Avg Acc:     {results['avg_accuracy']:.1%}")
print(f"  Avg AUC:     {results['avg_auc_roc']:.3f}")
print(f"  OOS Brier:   {results['oos_brier']:.4f}")
print(f"  OOS AUC:     {results['oos_auc_roc']:.3f}")

if predictor.is_fitted:
    fi = predictor.get_feature_importance(top_n=5)
    for model_name, top_k in fi.items():
        print(f"  Top-5 [{model_name}]: {list(top_k.keys())}")

print()
print("=" * 60)
print("  ALL TESTS PASSED")
print("=" * 60)
