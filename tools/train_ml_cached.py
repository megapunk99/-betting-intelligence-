#!/usr/bin/env python3
"""Step 2: Train MoneylinePredictor from cached features, with timing."""
import sys, os, warnings, json, time, joblib
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from betting_intel.models.moneyline_predictor import MoneylinePredictor, HAS_XGBOOST, HAS_LIGHTGBM

# Load cached data
data = joblib.load(os.path.join(PROJECT_ROOT, "data", "ml_training_data.joblib"))
X, y, feature_cols = data["X"], data["y"], data["feature_cols"]

results = {}
t0 = time.time()

print(f"Models available: XGBoost={'YES' if HAS_XGBOOST else 'NO'}, LightGBM={'YES' if HAS_LIGHTGBM else 'NO'}")
print(f"Training: {X.shape}, y: {y.sum()}/{len(y)} wins, {len(feature_cols)} features")
print()

# Train with 5-fold walk-forward CV
print("Walk-forward CV (5 folds, feature selection 50)...")
t1 = time.time()

predictor = MoneylinePredictor(calibrate=True, feature_selection=True, n_select=50)
cv_results = predictor.cross_validate(X, y, feature_names=feature_cols, n_splits=5)

train_time = time.time() - t1
results["train_time_seconds"] = round(train_time, 1)
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

print(f"\n{'='*55}")
print(f"  WALK-FORWARD CV RESULTS")
print(f"{'='*55}")
print(f"  Training time:  {train_time:.1f}s")
print(f"  Folds:          {cv_results['n_folds']}")
print(f"  OOS samples:    {cv_results.get('n_oos', 0)}")
print(f"  Avg Brier:      {cv_results['avg_brier']:.4f}")
print(f"  Avg Log Loss:   {cv_results['avg_log_loss']:.4f}")
print(f"  Avg Accuracy:   {cv_results['avg_accuracy']:.1%}")
print(f"  Avg AUC:        {cv_results['avg_auc_roc']:.3f}")
print(f"  OOS Brier:      {cv_results.get('oos_brier', 0):.4f}")
print(f"  OOS AUC:        {cv_results.get('oos_auc_roc', 0):.3f}")

# Feature importance
if predictor.is_fitted:
    fi = predictor.get_feature_importance(top_n=15)
    all_top = {}
    for model_name, top_k in fi.items():
        for feat, imp in top_k.items():
            all_top[feat] = all_top.get(feat, 0) + imp
    sorted_top = sorted(all_top.items(), key=lambda x: -x[1])[:15]

    print(f"\n  Top-15 Features (aggregated across models):")
    ml_count = 0
    for i, (feat, imp) in enumerate(sorted_top):
        is_ml = any(k in feat for k in [
            "composite_power", "power_diff", "form_diff", "perf_vs_expected",
            "consistency", "h2h_win_rate", "h2h_avg_margin", "home_away_split",
            "recent_win_pct",
        ])
        if is_ml:
            ml_count += 1
        marker = " ★" if is_ml else ""
        print(f"    {i+1:2d}. {feat:45s} {imp:.4f}{marker}")
    print(f"    --- {ml_count}/15 are moneyline-specific features")

# Comparison
prev_auc = 0.500
new_auc = cv_results.get("avg_auc_roc", 0.5)
improvement = new_auc - prev_auc

print(f"\n{'='*55}")
print(f"  VS BASELINE")
print(f"{'='*55}")
print(f"  Previous AUC (regression features): {prev_auc:.3f}")
print(f"  New AUC (moneyline features):       {new_auc:.3f}")
print(f"  Improvement:                       +{improvement:.3f}")
if improvement >= 0.06:
    print(f"  Verdict: STRONG SIGNAL")
elif improvement >= 0.03:
    print(f"  Verdict: GOOD SIGNAL")
elif improvement >= 0.01:
    print(f"  Verdict: MODEST SIGNAL")
else:
    print(f"  Verdict: MINIMAL")

results["improvement"] = round(improvement, 3)
results["total_time"] = round(time.time() - t0, 1)

out = os.path.join(PROJECT_ROOT, "tools", "ml_training_results.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out}")

# Cleanup cached data
os.remove(os.path.join(PROJECT_ROOT, "data", "ml_training_data.joblib"))
print(f"Cached data cleaned up")
print(f"Total time: {results['total_time']}s")
