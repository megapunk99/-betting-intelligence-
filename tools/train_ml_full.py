#!/usr/bin/env python3
"""Full MoneylinePredictor training on real NBA data + save via ModelRegistry."""

import sys, os, warnings, json
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from datetime import datetime
import numpy as np
import pandas as pd

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer
from betting_intel.models.moneyline_predictor import (
    MoneylinePredictor,
    train_moneyline_model,
    HAS_XGBOOST,
    HAS_LIGHTGBM,
)

results = {}

print("=" * 65)
print(f"  MONEYLINE FULL TRAINING — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

print(f"\n[1/5] Available models:")
print(f"  XGBoost:  {'YES' if HAS_XGBOOST else 'NO'}")
print(f"  LightGBM: {'YES' if HAS_LIGHTGBM else 'NO'}")
print(f"  Logistic: ALWAYS")

print("\n[2/5] Loading NBA data...")
loader = NBADataLoader()
fe = FeatureEngineer()
raw_df = loader.load_game_logs()

if raw_df is None or len(raw_df) == 0:
    print("  [FAIL] No NBA database found")
    sys.exit(1)

print(f"  Raw rows: {len(raw_df)}")
games_df = loader.build_game_dataset(raw_df)
raw_df = loader.compute_rest_days(raw_df)
print(f"  Games:    {len(games_df)}")

print("\n[3/5] Building features (v3.1 with moneyline features)...")
feature_df = fe.build_all_features(games_df, raw_df)
feature_cols = fe.select_features(feature_df)

ml_features = [c for c in feature_cols if any(k in c for k in [
    "composite_power", "power_diff", "form_diff", "perf_vs_expected",
    "consistency", "h2h_win_rate", "h2h_avg_margin", "home_away_split",
    "recent_win_pct",
])]
print(f"  Total features:    {len(feature_cols)}")
print(f"  Moneyline features: {len(ml_features)}")

clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols)//2).copy()
clean_df = clean_df.reset_index(drop=True)
X = clean_df[feature_cols].fillna(0).values
y = (clean_df["point_diff"].values > 0).astype(int)
print(f"  Training samples:  {len(X)}")
print(f"  Home win rate:     {y.mean():.1%} ({y.sum()}W / {len(y)-y.sum()}L)")

print("\n[4/5] Running walk-forward CV (5 splits, feature selection=50)...")
predictor = MoneylinePredictor(
    calibrate=True,
    n_folds=5,
    feature_selection=True,
    n_select=50,
    random_state=42,
)

cv_results = predictor.cross_validate(
    X, y,
    feature_names=feature_cols,
    n_splits=5,
)

results["cv"] = {
    "n_folds": cv_results["n_folds"],
    "n_oos": cv_results.get("n_oos", 0),
    "avg_brier": round(float(cv_results["avg_brier"]), 4),
    "avg_log_loss": round(float(cv_results["avg_log_loss"]), 4),
    "avg_accuracy": round(float(cv_results["avg_accuracy"]), 4),
    "avg_auc_roc": round(float(cv_results["avg_auc_roc"]), 3),
    "oos_brier": round(float(cv_results.get("oos_brier", 0)), 4),
    "oos_auc_roc": round(float(cv_results.get("oos_auc_roc", 0)), 3),
}

print(f"\n{'=' * 65}")
print(f"  WALK-FORWARD CV RESULTS")
print(f"{'=' * 65}")
print(f"  Folds:            {cv_results['n_folds']}")
print(f"  OOS samples:      {cv_results.get('n_oos', 0)}")
print(f"  Avg Brier:        {cv_results['avg_brier']:.4f}")
print(f"  Avg Log Loss:     {cv_results['avg_log_loss']:.4f}")
print(f"  Avg Accuracy:     {cv_results['avg_accuracy']:.1%}")
print(f"  Avg AUC-ROC:      {cv_results['avg_auc_roc']:.3f}")
print(f"  Pooled OOS Brier: {cv_results.get('oos_brier', 0):.4f}")
print(f"  Pooled OOS AUC:   {cv_results.get('oos_auc_roc', 0):.3f}")

if predictor.is_fitted:
    fi = predictor.get_feature_importance(top_n=15)
    print(f"\n  Top-15 Features:")
    all_top = {}
    for model_name, top_k in fi.items():
        for feat, imp in top_k.items():
            all_top[feat] = all_top.get(feat, 0) + imp
    sorted_top = sorted(all_top.items(), key=lambda x: -x[1])[:15]
    for i, (feat, imp) in enumerate(sorted_top):
        ml_marker = " [ML]" if any(k in feat for k in [
            "composite_power", "power_diff", "form_diff", "perf_vs_expected",
            "consistency", "h2h_win_rate", "h2h_avg_margin", "home_away_split",
            "recent_win_pct",
        ]) else ""
        print(f"    {i+1:2d}. {feat:45s} {imp:.4f}{ml_marker}")

print(f"\n[5/5] Saving model via train_moneyline_model...")
try:
    predictor_saved, metrics_saved = train_moneyline_model(
        clean_df, feature_cols, target_col="point_diff",
        calibrate=True, cv=True, save=True,
        model_name="moneyline_ensemble",
    )
    print(f"  Model saved to registry as 'moneyline_ensemble'")
    results["saved"] = True
except Exception as e:
    print(f"  [SKIP] Model save failed: {e}")
    results["saved"] = False

# Comparison with previous run (AUC=0.500)
prev_auc = 0.500
new_auc = cv_results.get("avg_auc_roc", cv_results.get("oos_auc_roc", 0.5))
improvement = new_auc - prev_auc

print(f"\n{'=' * 65}")
print(f"  COMPARISON VS BASELINE")
print(f"{'=' * 65}")
print(f"  Previous AUC (regression features):   {prev_auc:.3f}")
print(f"  New AUC (moneyline features + model): {new_auc:.3f}")
print(f"  Improvement:                         +{improvement:.3f}")

if improvement >= 0.08:
    verdict = "EXCELLENT — strong predictive signal"
elif improvement >= 0.04:
    verdict = "GOOD — meaningful improvement"
elif improvement >= 0.02:
    verdict = "MODEST — detectable signal"
else:
    verdict = "MINIMAL — further feature work needed"

print(f"  Verdict: {verdict}")
print(f"\n  Training complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 65}")

# Save results to JSON
out_path = os.path.join(PROJECT_ROOT, "tools", "ml_training_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
