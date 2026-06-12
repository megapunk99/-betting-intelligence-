#!/usr/bin/env python3
"""Quick synthetic dry-run of the MoneylinePredictor."""

import sys
import os
import warnings
import json

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
from betting_intel.models.moneyline_predictor import MoneylinePredictor

results = {}

# --- Synthetic test ---
np.random.seed(42)
n = 200
X = np.random.randn(n, 10)
y = (X[:, 0] + X[:, 1] + np.random.randn(n) * 0.5 > 0).astype(int)
feature_cols = [f"f{i}" for i in range(10)]

p = MoneylinePredictor(calibrate=True, feature_selection=False)
p.fit(X, y, feature_names=feature_cols)
probs = p.predict_proba(X)
preds = p.predict(X)
m = p.evaluate(X, y)

results["synthetic"] = {
    "brier": round(m["brier"], 4),
    "log_loss": round(m["log_loss"], 4),
    "accuracy": round(m["accuracy"], 4),
    "auc_roc": round(m["auc_roc"], 3),
    "cal_error": round(m["calibration_error"], 4),
    "models": list(p.models_.keys()),
    "calibrated": list(p.calibrated_models_.keys()),
}

cv = p.cross_validate(X, y, feature_names=feature_cols, n_splits=3)
results["synthetic_cv"] = {
    "n_folds": cv["n_folds"],
    "avg_brier": round(cv["avg_brier"], 4),
    "avg_auc": round(cv["avg_auc_roc"], 3),
}

fi = p.get_feature_importance(top_n=3)
results["feature_importance"] = {name: list(keys.keys()) for name, keys in fi.items()}

# --- Error handling tests ---
try:
    p2 = MoneylinePredictor()
    p2.predict_proba(X)
    results["error_before_fit"] = "FAIL: should have raised"
except ValueError:
    results["error_before_fit"] = "PASS"

try:
    p3 = MoneylinePredictor()
    p3.fit(np.array([]), np.array([]))
    results["error_empty_data"] = "FAIL: should have raised"
except ValueError:
    results["error_empty_data"] = "PASS"

print(json.dumps(results, indent=2))
