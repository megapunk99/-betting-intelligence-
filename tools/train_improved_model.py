#!/usr/bin/env python3
"""
Train an improved total-points prediction model using LightGBM + Ridge
stacking ensemble with proper TimeSeriesSplit cross-validation.

Usage:
    python tools/train_improved_model.py

Output:
    models/total_model.pkl  (same format as before: model, feature_cols,
                             mae, training_date)

The script:
  1. Loads historical NBA data via NBADataLoader + FeatureEngineer
  2. Runs 5-fold TimeSeriesSplit CV across: Ridge, LightGBM, CatBoost,
     XGBoost, and a Stacking Ensemble (Ridge + LGB + CatBoost)
  3. Selects the best model by CV MAE
  4. Trains it on ALL historical data for production use
  5. Saves in the same format as the current model
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# -- Model imports ------------------------------------------------------
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from betting_intel.models.stacking import WalkForwardStackingEnsemble

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print(" [!] LightGBM not available")

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print(" [!] CatBoost not available")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print(" [!] XGBoost not available")

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer


# ========================================================================
#  DATA LOADING
# ========================================================================


def load_data() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Load NBA data, engineer features, return (full_df, feature_cols, raw_feature_df)."""
    print("\n  Loading historical NBA data...")
    loader = NBADataLoader()
    raw_df = loader.load_game_logs()
    if raw_df is None or raw_df.empty:
        print("  [FAIL] No data loaded. Check database.")
        sys.exit(1)

    print(f"    Game logs: {len(raw_df):,}")
    raw_df["IS_HOME"] = raw_df["MATCHUP"].fillna("").str.contains("vs.").astype(int)
    raw_df = loader.compute_rest_days(raw_df)
    games_df = loader.build_game_dataset(raw_df)
    print(f"    Game records: {len(games_df):,}")

    engineer = FeatureEngineer()
    feature_df = engineer.build_all_features(games_df, raw_df)
    feature_cols = engineer.select_features(feature_df)
    print(f"    Features: {len(feature_cols)} | Rows: {len(feature_df):,}")

    if "total_points" not in feature_df.columns:
        print("  [FAIL] total_points not in feature dataframe")
        sys.exit(1)

    return feature_df, feature_cols, feature_df


# ========================================================================
#  MODEL BUILDERS
# ========================================================================


def make_ridge(**kwargs) -> Ridge:
    return Ridge(alpha=1.0, random_state=42)


def make_lightgbm(**kwargs) -> LGBMRegressor:
    """LightGBM tuned for ~361 features and ~2.8k samples."""
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )


def make_catboost(**kwargs) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=0,
        thread_count=-1,
    )


def make_xgboost(**kwargs) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
    )


# ========================================================================
#  CROSS-VALIDATION
# ========================================================================


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """Run TimeSeriesSplit CV across all available models."""
    n = len(X)
    fold_size = n // n_splits

    results = []
    model_fns = []

    model_fns.append(("Ridge", lambda: make_ridge()))
    if HAS_LGB:
        model_fns.append(("LightGBM", lambda: make_lightgbm()))
    if HAS_CAT:
        model_fns.append(("CatBoost", lambda: make_catboost()))
    if HAS_XGB:
        model_fns.append(("XGBoost", lambda: make_xgboost()))

    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = min(test_start + fold_size, n)

        if test_start < 100 or test_end - test_start < 10:
            continue

        X_train = X[:test_start]
        y_train = y[:test_start]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        fold_result = {"fold": fold + 1, "n_train": len(X_train), "n_test": len(X_test)}

        all_preds = []
        for name, builder in model_fns:
            model = builder()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            all_preds.append(preds)
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))
            fold_result[f"{name}_mae".lower()] = mae
            fold_result[f"{name}_r2".lower()] = r2

        if len(all_preds) >= 2:
            avg_preds = np.mean(all_preds, axis=0)
            fold_result["avg_ensemble_mae"] = float(mean_absolute_error(y_test, avg_preds))
            fold_result["avg_ensemble_r2"] = float(r2_score(y_test, avg_preds))

        results.append(fold_result)

    return results


def print_cv_results(results: list[dict]):
    """Print formatted CV comparison across models."""
    if not results:
        print("  No CV folds completed.")
        return

    metric_keys = [k for k in results[0].keys()
                   if k not in ("fold", "n_train", "n_test")]
    mae_keys = [k for k in metric_keys if k.endswith("_mae")]

    print(f"\n  {'-' * 72}")
    print(f"  {'MODEL':<25} {'CV MAE':<12} {'CV R2':<12} {'Folds':<8}")
    print(f"  {'-' * 72}")

    best_mae = float("inf")
    best_key = None

    for key in mae_keys:
        vals = [r[key] for r in results if key in r]
        if not vals:
            continue
        avg_mae = float(np.mean(vals))
        std_mae = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        avg_r2_key = key.replace("_mae", "_r2")
        r2_vals = [r[avg_r2_key] for r in results if avg_r2_key in r]
        avg_r2 = float(np.mean(r2_vals)) if r2_vals else 0.0

        label = key.replace("_mae", "").replace("_", " ").title()
        print(f"  {label:<25} {avg_mae:<8.3f} +/-{std_mae:<5.3f} {avg_r2:<8.4f}  {len(vals)}")

        if avg_mae < best_mae:
            best_mae = avg_mae
            best_key = key

    print(f"  {'-' * 72}")
    print(f"  {'BEST':<25} {best_mae:<8.3f} ({best_key})")
    print(f"  {'-' * 72}")

    print(f"\n  Per-fold MAE:")
    header = "  " + "".join(f"{k.replace('_mae','').replace('_',' ').title():>16}"
                             for k in mae_keys)
    print(f"  {'Fold':<6}  {header}")
    for r in results:
        fold = r["fold"]
        line = f"  {fold:<6}"
        for key in mae_keys:
            val = r.get(key, float("nan"))
            line += f"{val:>16.3f}" if not np.isnan(val) else f"{'N/A':>16}"
        print(line)

    return best_key, best_mae


# ========================================================================
#  MAIN
# ========================================================================


def main():
    t0 = time.time()

    print("=" * 72)
    print("  IMPROVED MODEL TRAINING - LightGBM + Ridge Stacking Ensemble")
    print("=" * 72)

    # -- 1. Load data ---------------------------------------------------
    feature_df, feature_cols, raw_feature_df = load_data()

    # -- 2. Prepare X, y ------------------------------------------------
    X_all = feature_df[feature_cols].fillna(0).values
    y_all = feature_df["total_points"].fillna(
        feature_df["total_points"].median()
    ).values

    remaining_nas = np.isnan(X_all).sum()
    if remaining_nas > 0:
        print(f"    [WARN] {remaining_nas} NaN values - filling with 0")
        X_all = np.nan_to_num(X_all, nan=0.0)

    print(f"\n  Training matrix: {X_all.shape[0]:,} rows x {X_all.shape[1]} features")

    # -- 3. Run Cross-Validation ----------------------------------------
    print(f"\n  {'-' * 72}")
    print(f"  PHASE 1: 5-Fold TimeSeriesSplit Cross-Validation")
    print(f"  {'-' * 72}")

    cv_results = run_cv(X_all, y_all, n_splits=5)
    best_key, best_mae = print_cv_results(cv_results)

    # -- 4. Walk-Forward Stacking Ensemble ------------------------------
    print(f"\n  {'-' * 72}")
    print(f"  PHASE 2: Walk-Forward Stacking Ensemble (Ridge + LGB + CatBoost)")
    print(f"  {'-' * 72}")

    stacking_ensemble = WalkForwardStackingEnsemble(n_folds=5)
    stacking_ensemble.add_model("Ridge", make_ridge)
    if HAS_LGB:
        stacking_ensemble.add_model("LightGBM", make_lightgbm)
    if HAS_CAT:
        stacking_ensemble.add_model("CatBoost", make_catboost)

    stacking_ensemble.fit(X_all, y_all)

    print(f"\n    Stacking Ensemble CV MAE: {stacking_ensemble.cv_mae:.4f}")
    print(f"    Meta-model coefficients: {np.round(stacking_ensemble.coef_, 4)}")
    print(f"    Meta-model intercept: {stacking_ensemble.intercept_:.4f}")

    # -- 5. Compare and select best model -------------------------------
    print(f"\n  {'-' * 72}")
    print(f"  PHASE 3: Model Selection")
    print(f"  {'-' * 72}")

    use_ensemble = True
    cv_best = best_mae if best_key else float("inf")

    if stacking_ensemble.cv_mae is not None and stacking_ensemble.cv_mae < cv_best:
        print(f"\n    Stacking Ensemble wins: {stacking_ensemble.cv_mae:.4f} MAE")
        print(f"    vs best single model: {cv_best:.4f} MAE ({best_key})")
        print(f"    Improvement: {(cv_best - stacking_ensemble.cv_mae) / cv_best * 100:+.1f}%")
        final_model = stacking_ensemble
        final_mae = stacking_ensemble.cv_mae
    elif best_key:
        print(f"\n    Best single model: {best_key} ({cv_best:.4f} MAE)")
        print(f"    Training on full dataset...")

        name_map = {
            "ridge_mae": make_ridge,
            "lightgbm_mae": make_lightgbm,
            "catboost_mae": make_catboost,
            "xgboost_mae": make_xgboost,
        }
        builder = name_map.get(best_key)
        if builder is not None:
            m = builder()
            m.fit(X_all, y_all)
            final_model = m
        else:
            models = []
            if HAS_LGB:
                models.append(make_lightgbm())
            if HAS_CAT:
                models.append(make_catboost())
            models.append(make_ridge())
            for m in models:
                m.fit(X_all, y_all)

            class SimpleAvgEnsemble:
                def __init__(self, models):
                    self.models = models
                def predict(self, X):
                    return np.mean([m.predict(X) for m in self.models], axis=0)

            final_model = SimpleAvgEnsemble(models)
        final_mae = cv_best
        use_ensemble = False
    else:
        print(f"    [WARN] No model succeeded - falling back to Ridge")
        final_model = make_ridge()
        final_model.fit(X_all, y_all)
        final_mae = 10.0
        use_ensemble = False

    # -- 6. Evaluate on full training set -------------------------------
    train_preds = final_model.predict(X_all)
    train_mae = float(mean_absolute_error(y_all, train_preds))
    train_r2 = float(r2_score(y_all, train_preds))
    train_bias = float(np.mean(train_preds - y_all))

    print(f"\n  Final model performance on full dataset:")
    print(f"    CV MAE:       {final_mae:.4f}")
    print(f"    Train MAE:    {train_mae:.4f}")
    print(f"    Train R2:     {train_r2:.4f}")
    print(f"    Train Bias:   {train_bias:+.4f}")

    # -- 7. Save model --------------------------------------------------
    import joblib

    model_path = PROJECT_ROOT / "models" / "total_model.pkl"
    backup_path = PROJECT_ROOT / "models" / "total_model.backup.pkl"

    if model_path.exists():
        import shutil
        shutil.copy2(str(model_path), str(backup_path))
        print(f"\n  Backup saved: {backup_path}")

    model_data = {
        "model": final_model,
        "feature_cols": feature_cols,
        "mae": final_mae,
        "training_date": pd.Timestamp.now().isoformat(),
        "train_mae": train_mae,
        "train_r2": train_r2,
        "train_bias": train_bias,
        "cv_results": cv_results,
        "stacking_meta_coef": stacking_ensemble.coef_.tolist() if stacking_ensemble.coef_ is not None else None,
        "model_type": type(final_model).__name__,
        "n_features": len(feature_cols),
        "n_training_samples": len(X_all),
        "ensemble": use_ensemble,
    }

    joblib.dump(model_data, str(model_path))
    print(f"  Model saved: {model_path}")
    print(f"    Type: {type(final_model).__name__}")
    print(f"    Features: {len(feature_cols)}")
    print(f"    Samples: {len(X_all):,}")

    # -- 8. Summary -----------------------------------------------------
    elapsed = time.time() - t0
    improvement = (9.85 - final_mae) / 9.85 * 100

    print(f"\n  {'=' * 72}")
    print(f"  SUMMARY")
    print(f"  {'=' * 72}")
    print(f"  Current model MAE:  9.85 (GradientBoostingRegressor)")
    print(f"  New model MAE:      {final_mae:.3f} ({type(final_model).__name__})")
    print(f"  Improvement:        {improvement:+.1f}%")
    print(f"  Completed in:       {elapsed:.1f}s")
    print(f"  Model saved to:     models/total_model.pkl")
    print(f"  Backup at:          models/total_model.backup.pkl")
    print(f"\n  To verify: python tools/generate_recommendations.py")
    print(f"  {'=' * 72}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
