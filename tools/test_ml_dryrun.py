#!/usr/bin/env python3
"""Full dry-run: load NBA data, build new moneyline features, train MoneylinePredictor, report AUC."""

import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer
from betting_intel.models.moneyline_predictor import MoneylinePredictor

print("=" * 65)
print("  MONEYLINE FEATURE DRY-RUN — v3.1 Moneyline Features")
print("=" * 65)

print("\n[1/4] Loading NBA data...")
loader = NBADataLoader()
fe = FeatureEngineer()

raw_df = loader.load_game_logs()
if raw_df is None or len(raw_df) == 0:
    print("  [FAIL] No NBA database found")
    sys.exit(1)

games_df = loader.build_game_dataset(raw_df)
raw_df = loader.compute_rest_days(raw_df)

print(f"  Raw games loaded: {len(games_df)}")

print("\n[2/4] Building features (v3.1 with moneyline features)...")
feature_df = fe.build_all_features(games_df, raw_df)
feature_cols = fe.select_features(feature_df)

# Print what moneyline features exist
ml_features = [c for c in feature_cols if any(k in c for k in [
    "composite_power", "power_diff", "form_diff", "perf_vs_expected",
    "consistency", "h2h_win_rate", "h2h_avg_margin", "home_away_split",
    "recent_win_pct",
])]
print(f"  Total features: {len(feature_cols)}")
print(f"  Moneyline-specific features: {len(ml_features)}")
for f in sorted(ml_features):
    print(f"    - {f}")

clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols)//2).copy()
clean_df = clean_df.reset_index(drop=True)
X = clean_df[feature_cols].fillna(0).values
y = (clean_df["point_diff"].values > 0).astype(int)

print(f"\n[3/4] Training data: {X.shape[0]} games, {X.shape[1]} features")
print(f"  Home win rate: {y.mean():.1%} ({y.sum()}W / {len(y)-y.sum()}L)")

print("\n[4/4] Running walk-forward CV (3 splits, XGBoost+LightGBM+Logistic)...")
predictor = MoneylinePredictor(
    calibrate=True, n_folds=3, feature_selection=True, n_select=30,
)
results = predictor.cross_validate(X, y, feature_names=feature_cols, n_splits=3)

print()
print("=" * 65)
print("  RESULTS")
print("=" * 65)
print(f"  Walk-forward folds:    {results['n_folds']}")
print(f"  Out-of-sample games:   {results['n_oos']}")
print(f"  Avg Brier score:       {results['avg_brier']:.4f}")
print(f"  Avg Log Loss:          {results['avg_log_loss']:.4f}")
print(f"  Avg Accuracy:          {results['avg_accuracy']:.1%}")
print(f"  Avg AUC-ROC:           {results['avg_auc_roc']:.3f}")
print(f"  OOS Brier (pooled):    {results['oos_brier']:.4f}")
print(f"  OOS AUC-ROC (pooled):  {results['oos_auc_roc']:.3f}")

if predictor.is_fitted:
    fi = predictor.get_feature_importance(top_n=10)
    print("\n  Top-10 Features:")
    for model_name, top_k in fi.items():
        print(f"    [{model_name}]:")
        for i, (feat, imp) in enumerate(list(top_k.items())[:10]):
            print(f"      {i+1:2d}. {feat:40s} {imp:.4f}")

print()
# Check if we have any moneyline features in the top 20
all_top = set()
for model_name, top_k in fi.items():
    all_top.update(top_k.keys())

ml_top = [f for f in all_top if any(k in f for k in [
    "composite_power", "power_diff", "form_diff", "perf_vs_expected",
    "consistency", "h2h_win_rate", "h2h_avg_margin", "home_away_split",
    "recent_win_pct",
])]
print(f"  Moneyline features in top-20: {len(ml_top)} / {len(all_top)}")
for f in sorted(ml_top):
    print(f"    - {f}")

prev_auc = 0.500  # from earlier run
new_auc = results.get("avg_auc_roc", results.get("oos_auc_roc", 0.5))
improvement = new_auc - prev_auc

if improvement > 0.02:
    verdict = f"✅ STRONG IMPROVEMENT (+{improvement:.3f} AUC)"
elif improvement > 0.005:
    verdict = f"👍 MODEST IMPROVEMENT (+{improvement:.3f} AUC)"
elif improvement > -0.005:
    verdict = f"➡️ NO CHANGE ({improvement:+.3f} AUC)"
else:
    verdict = f"❌ REGRESSION ({improvement:+.3f} AUC)"

print(f"\n  PREVIOUS AUC:  {prev_auc:.3f}")
print(f"  NEW AUC:       {new_auc:.3f}")
print(f"  VERDICT:       {verdict}")
print()
print("=" * 65)
