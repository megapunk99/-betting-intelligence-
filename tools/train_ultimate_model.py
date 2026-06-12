#!/usr/bin/env python3
"""
ULTIMATE MODEL TRAINING — Optuna-Tuned Multi-Target Stacking Ensembles.

Trains 3 separate models (totals, win-prob, spread) using:
  - Optuna hyperparameter search (30 trials per model type)
  - Mutual information feature selection (top N features)
  - Walk-forward stacking ensemble (Ridge + LGB + CatBoost + XGBoost)
  - TimeSeriesSplit cross-validation

Output:
    models/ultimate_model.pkl  — dict with all 3 models + metadata

Usage:
    python tools/train_ultimate_model.py
    python tools/train_ultimate_model.py --features 150   # limit to top 150 features
    python tools/train_ultimate_model.py --trials 50      # more Optuna trials
"""

from __future__ import annotations

import sys
import time
import warnings
import json
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

# ── ML imports ──────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, brier_score_loss, log_loss
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV

from betting_intel.models.stacking import WalkForwardStackingEnsemble, WinProbEnsemble

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print(" [!] LightGBM not available")

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print(" [!] CatBoost not available")

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print(" [!] XGBoost not available")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print(" [!] Optuna not available — using default params")

from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer


# ════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════


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
    print(f"    Raw features: {len(feature_cols)} | Rows: {len(feature_df):,}")

    # Create target columns
    feature_df["home_win"] = (feature_df["point_diff"] > 0).astype(int)

    for target in ["total_points", "home_win", "point_diff"]:
        if target not in feature_df.columns:
            print(f"  [FAIL] {target} not in feature dataframe")
            sys.exit(1)

    print(f"    Targets: total_points, home_win, point_diff")
    print(f"    Date range: {feature_df['GAME_DATE'].min().date()} to {feature_df['GAME_DATE'].max().date()}")

    return feature_df, feature_cols, feature_df


# ════════════════════════════════════════════════════════════════════════
#  FEATURE SELECTION (Mutual Information)
# ════════════════════════════════════════════════════════════════════════


def select_top_features(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_clf: np.ndarray,
    feature_cols: list[str],
    n_select: int = 200,
) -> list[str]:
    """
    Select top N features using mutual information, averaged across
    regression (total_points) and classification (home_win) targets.
    """
    if n_select >= len(feature_cols):
        print(f"    Using all {len(feature_cols)} features (n_select >= total)")
        return feature_cols

    print(f"\n  Selecting top {n_select} features via mutual information...")

    X_clean = np.nan_to_num(X, nan=0.0)
    y_reg_clean = np.nan_to_num(y_reg, nan=np.nanmedian(y_reg))
    y_clf_clean = np.nan_to_num(y_clf, nan=0).astype(int)

    mi_reg = mutual_info_regression(X_clean, y_reg_clean, random_state=42)
    mi_clf = mutual_info_classif(X_clean, y_clf_clean, random_state=42)

    # Average MI across both targets (normalize first)
    mi_reg_norm = (mi_reg - mi_reg.min()) / (mi_reg.max() - mi_reg.min() + 1e-10)
    mi_clf_norm = (mi_clf - mi_clf.min()) / (mi_clf.max() - mi_clf.min() + 1e-10)
    mi_avg = (mi_reg_norm + mi_clf_norm) / 2.0

    top_indices = np.argsort(mi_avg)[-n_select:][::-1]
    selected = [feature_cols[i] for i in top_indices]

    # Show top features
    print(f"    Top 10 features by MI:")
    for i, idx in enumerate(top_indices[:10]):
        print(f"      {i+1}. {feature_cols[idx]} (MI: {mi_avg[idx]:.4f})")

    return selected


# ════════════════════════════════════════════════════════════════════════
#  OPTUNA HYPERPARAMETER TUNING
# ════════════════════════════════════════════════════════════════════════


def objective_ridge_reg(trial, X_train, y_train, X_val, y_val):
    alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return float(mean_absolute_error(y_val, preds))


def objective_lgb_reg(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
        "random_state": 42,
        "verbosity": -1,
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return float(mean_absolute_error(y_val, preds))


def objective_cat_reg(trial, X_train, y_train, X_val, y_val):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_seed": 42,
        "verbose": 0,
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return float(mean_absolute_error(y_val, preds))


def objective_xgb_reg(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
        "random_state": 42,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return float(mean_absolute_error(y_val, preds))


def objective_lgb_clf(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
        "random_state": 42,
        "verbosity": -1,
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    return float(brier_score_loss(y_val, probs))


def objective_cat_clf(trial, X_train, y_train, X_val, y_val):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_seed": 42,
        "verbose": 0,
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    return float(brier_score_loss(y_val, probs))


def objective_xgb_clf(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
        "random_state": 42,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    return float(brier_score_loss(y_val, probs))


def tune_hyperparams(
    X_train, y_train, X_val, y_val,
    model_type: str,
    target_type: str,  # "reg" or "clf"
    n_trials: int = 30,
):
    """Run Optuna hyperparameter search for a given model type."""
    if not HAS_OPTUNA:
        return None

    objectives = {
        "ridge_reg": objective_ridge_reg,
        "lgb_reg": objective_lgb_reg,
        "cat_reg": objective_cat_reg,
        "xgb_reg": objective_xgb_reg,
        "lgb_clf": objective_lgb_clf,
        "cat_clf": objective_cat_clf,
        "xgb_clf": objective_xgb_clf,
    }

    key = f"{model_type}_{target_type}"
    if key not in objectives:
        return None

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objectives[key](trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    return study.best_params


# ════════════════════════════════════════════════════════════════════════
#  TIMESERIES CROSS-VALIDATION
# ════════════════════════════════════════════════════════════════════════


def run_cv_all_models(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_clf: np.ndarray,
    y_spread: np.ndarray,
    n_splits: int = 5,
    n_trials: int = 30,
) -> tuple[list[dict], dict]:
    """
    Run TimeSeriesSplit CV across all models for all 3 targets.
    Returns per-fold results and best hyperparameters.
    """
    n = len(X)
    fold_size = n // n_splits
    results = []
    best_params = {}

    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = min(test_start + fold_size, n)

        if test_start < 100 or test_end - test_start < 10:
            continue

        X_train, X_test = X[:test_start], X[test_start:test_end]
        y_reg_train, y_reg_test = y_reg[:test_start], y_reg[test_start:test_end]
        y_clf_train, y_clf_test = y_clf[:test_start], y_clf[test_start:test_end]
        y_sprd_train, y_sprd_test = y_spread[:test_start], y_spread[test_start:test_end]

        fold_res = {"fold": fold + 1, "n_train": len(X_train), "n_test": len(X_test)}

        # ── Ridge (baseline) ──────────────────────────────────────────
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_reg_train)
        preds = ridge.predict(X_test)
        fold_res["ridge_mae"] = float(mean_absolute_error(y_reg_test, preds))
        fold_res["ridge_r2"] = float(r2_score(y_reg_test, preds))

        # ── LightGBM ──────────────────────────────────────────────────
        if HAS_LGB:
            if fold == 0:
                # Tune on first fold only
                best = tune_hyperparams(X_train, y_reg_train, X_test, y_reg_test, "lgb", "reg", n_trials)
                if best:
                    best_params["lgb_reg"] = best

            params = best_params.get("lgb_reg", {})
            lgb = LGBMRegressor(
                n_estimators=params.get("n_estimators", 300),
                learning_rate=params.get("learning_rate", 0.03),
                max_depth=params.get("max_depth", 5),
                num_leaves=params.get("num_leaves", 31),
                min_child_samples=params.get("min_child_samples", 20),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.7),
                reg_alpha=params.get("reg_alpha", 0.1),
                reg_lambda=params.get("reg_lambda", 0.5),
                random_state=42, verbosity=-1,
            )
            lgb.fit(X_train, y_reg_train)
            preds = lgb.predict(X_test)
            fold_res["lgb_mae"] = float(mean_absolute_error(y_reg_test, preds))
            fold_res["lgb_r2"] = float(r2_score(y_reg_test, preds))

        # ── CatBoost ──────────────────────────────────────────────────
        if HAS_CAT:
            if fold == 0:
                best = tune_hyperparams(X_train, y_reg_train, X_test, y_reg_test, "cat", "reg", n_trials)
                if best:
                    best_params["cat_reg"] = best

            params = best_params.get("cat_reg", {})
            cat = CatBoostRegressor(
                iterations=params.get("iterations", 500),
                learning_rate=params.get("learning_rate", 0.03),
                depth=params.get("depth", 5),
                l2_leaf_reg=params.get("l2_leaf_reg", 5.0),
                random_seed=42, verbose=0,
            )
            cat.fit(X_train, y_reg_train)
            preds = cat.predict(X_test)
            fold_res["cat_mae"] = float(mean_absolute_error(y_reg_test, preds))
            fold_res["cat_r2"] = float(r2_score(y_reg_test, preds))

        # ── XGBoost ───────────────────────────────────────────────────
        if HAS_XGB:
            if fold == 0:
                best = tune_hyperparams(X_train, y_reg_train, X_test, y_reg_test, "xgb", "reg", n_trials)
                if best:
                    best_params["xgb_reg"] = best

            params = best_params.get("xgb_reg", {})
            xgb = XGBRegressor(
                n_estimators=params.get("n_estimators", 300),
                learning_rate=params.get("learning_rate", 0.03),
                max_depth=params.get("max_depth", 5),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.7),
                reg_alpha=params.get("reg_alpha", 0.1),
                reg_lambda=params.get("reg_lambda", 0.5),
                random_state=42,
            )
            xgb.fit(X_train, y_reg_train)
            preds = xgb.predict(X_test)
            fold_res["xgb_mae"] = float(mean_absolute_error(y_reg_test, preds))
            fold_res["xgb_r2"] = float(r2_score(y_reg_test, preds))

        # ── Simple average ensemble ───────────────────────────────────
        all_preds = [v for k, v in fold_res.items() if k.endswith("_mae") and k != "avg_ensemble_mae"]
        if len([m for m in [HAS_LGB, HAS_CAT, HAS_XGB] if m]) >= 2:
            ensemble_preds = []
            if HAS_LGB:
                ensemble_preds.append(lgb.predict(X_test))
            if HAS_CAT:
                ensemble_preds.append(cat.predict(X_test))
            if HAS_XGB:
                ensemble_preds.append(xgb.predict(X_test))
            avg_preds = np.mean(ensemble_preds, axis=0)
            fold_res["avg_ensemble_mae"] = float(mean_absolute_error(y_reg_test, avg_preds))
            fold_res["avg_ensemble_r2"] = float(r2_score(y_reg_test, avg_preds))

        # ── Win Probability (classification) ──────────────────────────
        if HAS_LGB:
            if fold == 0:
                best = tune_hyperparams(X_train, y_clf_train, X_test, y_clf_test, "lgb", "clf", n_trials)
                if best:
                    best_params["lgb_clf"] = best

            params = best_params.get("lgb_clf", {})
            lgb_clf = LGBMClassifier(
                n_estimators=params.get("n_estimators", 200),
                learning_rate=params.get("learning_rate", 0.03),
                max_depth=params.get("max_depth", 4),
                num_leaves=params.get("num_leaves", 24),
                random_state=42, verbosity=-1,
            )
            lgb_clf.fit(X_train, y_clf_train)
            probs = lgb_clf.predict_proba(X_test)[:, 1]
            fold_res["lgb_clf_brier"] = float(brier_score_loss(y_clf_test, probs))
            fold_res["lgb_clf_acc"] = float(accuracy_score(y_clf_test, (probs > 0.5).astype(int)))

        if HAS_CAT:
            if fold == 0:
                best = tune_hyperparams(X_train, y_clf_train, X_test, y_clf_test, "cat", "clf", n_trials)
                if best:
                    best_params["cat_clf"] = best

            params = best_params.get("cat_clf", {})
            cat_clf = CatBoostClassifier(
                iterations=params.get("iterations", 300),
                learning_rate=params.get("learning_rate", 0.03),
                depth=params.get("depth", 4),
                random_seed=42, verbose=0,
            )
            cat_clf.fit(X_train, y_clf_train)
            probs = cat_clf.predict_proba(X_test)[:, 1]
            fold_res["cat_clf_brier"] = float(brier_score_loss(y_clf_test, probs))
            fold_res["cat_clf_acc"] = float(accuracy_score(y_clf_test, (probs > 0.5).astype(int)))

        # ── Spread (margin) regression ────────────────────────────────
        ridge_spread = Ridge(alpha=1.0, random_state=42)
        ridge_spread.fit(X_train, y_sprd_train)
        spreds = ridge_spread.predict(X_test)
        fold_res["ridge_spread_mae"] = float(mean_absolute_error(y_sprd_test, spreds))

        results.append(fold_res)

    return results, best_params


def print_results(results: list[dict], best_params: dict):
    """Print formatted results."""
    if not results:
        print("  No CV folds completed.")
        return

    # ── Totals comparison ─────────────────────────────────────────────
    mae_keys = [k for k in results[0].keys()
                if k.endswith("_mae") and k.startswith(("ridge_", "lgb_", "cat_", "xgb_", "avg_"))]

    print(f"\n  {'=' * 72}")
    print(f"  TOTALS PREDICTION — Cross-Validation Results")
    print(f"  {'=' * 72}")
    print(f"  {'MODEL':<25} {'CV MAE':<12} {'CV R2':<12} {'Folds':<8}")
    print(f"  {'-' * 60}")

    best_mae = float("inf")
    best_tag = None
    for key in mae_keys:
        vals = [r[key] for r in results if key in r]
        if not vals:
            continue
        avg_mae = float(np.mean(vals))
        std_mae = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        r2_key = key.replace("_mae", "_r2")
        r2_vals = [r[r2_key] for r in results if r2_key in r]
        avg_r2 = float(np.mean(r2_vals)) if r2_vals else 0.0
        label = key.replace("_mae", "").replace("_", " ").title()
        print(f"  {label:<25} {avg_mae:<8.3f} +/-{std_mae:<5.3f} {avg_r2:<8.4f}  {len(vals)}")
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_tag = key

    print(f"  {'-' * 60}")
    print(f"  BEST: {best_tag} ({best_mae:.3f} MAE)")
    print(f"  {'-' * 60}")

    # ── Win Probability comparison ────────────────────────────────────
    clf_keys = [k for k in results[0].keys() if k.endswith("_brier")]

    print(f"\n  {'=' * 72}")
    print(f"  WIN PROBABILITY — Cross-Validation Results")
    print(f"  {'=' * 72}")
    print(f"  {'MODEL':<25} {'Brier':<12} {'Accuracy':<12} {'Folds':<8}")
    print(f"  {'-' * 60}")

    for key in clf_keys:
        vals = [r[key] for r in results if key in r]
        if not vals:
            continue
        avg_brier = float(np.mean(vals))
        acc_key = key.replace("_brier", "_acc")
        acc_vals = [r[acc_key] for r in results if acc_key in r]
        avg_acc = float(np.mean(acc_vals)) if acc_vals else 0.0
        label = key.replace("_brier", "").replace("_", " ").title()
        print(f"  {label:<25} {avg_brier:<8.4f}      {avg_acc:<8.1%}   {len(vals)}")

    # ── Spread comparison ─────────────────────────────────────────────
    spread_keys = [k for k in results[0].keys() if k.endswith("_spread_mae")]

    print(f"\n  {'=' * 72}")
    print(f"  SPREAD PREDICTION — Cross-Validation Results")
    print(f"  {'=' * 72}")
    for key in spread_keys:
        vals = [r[key] for r in results if key in r]
        if not vals:
            continue
        avg_mae = float(np.mean(vals))
        label = key.replace("_spread_mae", "").replace("_", " ").title()
        print(f"  {label:<25} {avg_mae:<8.3f} MAE  ({len(vals)} folds)")

    # ── Hyperparameters ───────────────────────────────────────────────
    if best_params:
        print(f"\n  {'=' * 72}")
        print(f"  OPTUNA BEST HYPERPARAMETERS")
        print(f"  {'=' * 72}")
        for model_key, params in best_params.items():
            print(f"  {model_key}:")
            for k, v in params.items():
                print(f"    {k}: {v}")
        print(f"  {'-' * 60}")

    return best_tag, best_mae


# ════════════════════════════════════════════════════════════════════════
#  BUILD FINAL ENSEMBLES
# ════════════════════════════════════════════════════════════════════════


def build_and_save_models(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_clf: np.ndarray,
    y_spread: np.ndarray,
    feature_cols: list[str],
    best_params: dict,
    cv_results: list[dict],
    cv_best_mae: float,
):
    """Build walk-forward stacking ensembles for all 3 targets and save."""

    print(f"\n  {'=' * 72}")
    print(f"  BUILDING FINAL ENSEMBLES")
    print(f"  {'=' * 72}")

    # ── 1. Totals Ensemble ────────────────────────────────────────────
    print(f"\n  [1/3] Totals Prediction Ensemble...")
    totals_ensemble = WalkForwardStackingEnsemble(n_folds=5)
    totals_ensemble.add_model("Ridge", lambda: Ridge(alpha=1.0, random_state=42))
    if HAS_LGB:
        p = best_params.get("lgb_reg", {})
        totals_ensemble.add_model("LightGBM", lambda p=p: LGBMRegressor(
            n_estimators=p.get("n_estimators", 300),
            learning_rate=p.get("learning_rate", 0.03),
            max_depth=p.get("max_depth", 5),
            num_leaves=p.get("num_leaves", 31),
            min_child_samples=p.get("min_child_samples", 20),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.7),
            reg_alpha=p.get("reg_alpha", 0.1),
            reg_lambda=p.get("reg_lambda", 0.5),
            random_state=42, verbosity=-1,
        ))
    if HAS_CAT:
        p = best_params.get("cat_reg", {})
        totals_ensemble.add_model("CatBoost", lambda p=p: CatBoostRegressor(
            iterations=p.get("iterations", 500),
            learning_rate=p.get("learning_rate", 0.03),
            depth=p.get("depth", 5),
            l2_leaf_reg=p.get("l2_leaf_reg", 5.0),
            random_seed=42, verbose=0,
        ))
    if HAS_XGB:
        p = best_params.get("xgb_reg", {})
        totals_ensemble.add_model("XGBoost", lambda p=p: XGBRegressor(
            n_estimators=p.get("n_estimators", 300),
            learning_rate=p.get("learning_rate", 0.03),
            max_depth=p.get("max_depth", 5),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.7),
            reg_alpha=p.get("reg_alpha", 0.1),
            reg_lambda=p.get("reg_lambda", 0.5),
            random_state=42,
        ))

    totals_ensemble.fit(X, y_reg)
    totals_cv_mae = totals_ensemble.cv_mae
    print(f"    Totals CV MAE: {totals_cv_mae:.3f}")

    # ── 2. Win Probability Ensemble ───────────────────────────────────
    print(f"\n  [2/3] Win Probability Ensemble...")

    # For classification, we need to build the stacking differently
    # Use a simpler approach: calibrated classification ensemble
    winprob_models = {}
    if HAS_LGB:
        p = best_params.get("lgb_clf", {})
        m = LGBMClassifier(
            n_estimators=p.get("n_estimators", 200),
            learning_rate=p.get("learning_rate", 0.03),
            max_depth=p.get("max_depth", 4),
            num_leaves=p.get("num_leaves", 24),
            random_state=42, verbosity=-1,
        )
        m.fit(X, y_clf)
        winprob_models["lgb"] = m

    if HAS_CAT:
        p = best_params.get("cat_clf", {})
        m = CatBoostClassifier(
            iterations=p.get("iterations", 300),
            learning_rate=p.get("learning_rate", 0.03),
            depth=p.get("depth", 4),
            random_seed=42, verbose=0,
        )
        m.fit(X, y_clf)
        winprob_models["cat"] = m

    # Also use calibrated logistic regression as a stable baseline
    lr_base = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    lr_calibrated = CalibratedClassifierCV(estimator=lr_base, method="sigmoid", cv=5)
    lr_calibrated.fit(X, y_clf)
    winprob_models["lr"] = lr_calibrated

    # Evaluate win-prob ensemble
    win_probs = []
    for name, m in winprob_models.items():
        probs = m.predict_proba(X)[:, 1]
        win_probs.append(probs)

    avg_probs = np.mean(win_probs, axis=0)
    win_brier = float(brier_score_loss(y_clf, avg_probs))
    win_acc = float(accuracy_score(y_clf, (avg_probs > 0.5).astype(int)))
    print(f"    Win Prob: Brier={win_brier:.4f}, Accuracy={win_acc:.1%}")

    winprob_ensemble = WinProbEnsemble(winprob_models)

    # ── 3. Spread Ensemble ────────────────────────────────────────────
    print(f"\n  [3/3] Spread (Margin) Ensemble...")
    spread_ensemble = WalkForwardStackingEnsemble(n_folds=5)
    spread_ensemble.add_model("Ridge", lambda: Ridge(alpha=1.0, random_state=42))
    if HAS_LGB:
        spread_ensemble.add_model("LightGBM", lambda: LGBMRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=4,
            num_leaves=24, random_state=42, verbosity=-1,
        ))
    if HAS_CAT:
        spread_ensemble.add_model("CatBoost", lambda: CatBoostRegressor(
            iterations=300, learning_rate=0.03, depth=4,
            random_seed=42, verbose=0,
        ))

    spread_ensemble.fit(X, y_spread)
    spread_cv_mae = spread_ensemble.cv_mae
    print(f"    Spread CV MAE: {spread_cv_mae:.3f}")

    # ── Evaluate all on training set ──────────────────────────────────
    train_preds_total = totals_ensemble.predict(X)
    train_preds_win = winprob_ensemble.predict_proba(X)[:, 1]
    train_preds_spread = spread_ensemble.predict(X)

    train_total_mae = float(mean_absolute_error(y_reg, train_preds_total))
    train_total_r2 = float(r2_score(y_reg, train_preds_total))
    train_win_brier = float(brier_score_loss(y_clf, train_preds_win))
    train_win_acc = float(accuracy_score(y_clf, (train_preds_win > 0.5).astype(int)))
    train_spread_mae = float(mean_absolute_error(y_spread, train_preds_spread))

    print(f"\n  {'=' * 72}")
    print(f"  TRAINING PERFORMANCE (on all {len(X):,} samples)")
    print(f"  {'=' * 72}")
    print(f"  Totals:  MAE={train_total_mae:.3f}, R2={train_total_r2:.4f}")
    print(f"  Win%:    Brier={train_win_brier:.4f}, Accuracy={train_win_acc:.1%}")
    print(f"  Spread:  MAE={train_spread_mae:.3f}")

    # ── Save everything ───────────────────────────────────────────────
    import joblib

    model_path = PROJECT_ROOT / "models" / "ultimate_model.pkl"

    model_data = {
        # Legacy format (generate_recommendations.py compat)
        "model": totals_ensemble,
        "feature_cols": feature_cols,
        "mae": totals_cv_mae or cv_best_mae,
        "training_date": pd.Timestamp.now().isoformat(),

        # New multi-model format
        "totals_model": totals_ensemble,
        "winprob_model": winprob_ensemble,
        "winprob_models": winprob_models,  # individual models too
        "spread_model": spread_ensemble,

        # Metadata
        "n_features": len(feature_cols),
        "n_training_samples": len(X),
        "best_params": best_params,
        "cv_results": cv_results,
        "metrics": {
            "totals": {"mae": train_total_mae, "r2": train_total_r2, "cv_mae": totals_cv_mae},
            "winprob": {"brier": train_win_brier, "accuracy": train_win_acc},
            "spread": {"mae": train_spread_mae, "cv_mae": spread_cv_mae},
        },
        "model_info": {
            "totals": "WalkForwardStackingEnsemble (Ridge + LGB + CatBoost + XGB)",
            "winprob": "Ensemble (LR + LGB + CatBoost classifiers)",
            "spread": "WalkForwardStackingEnsemble (Ridge + LGB + CatBoost)",
        },
    }

    joblib.dump(model_data, str(model_path))
    print(f"\n  [SAVED] {model_path}")
    print(f"    Totals:  {model_data['model_info']['totals']}")
    print(f"    WinProb: {model_data['model_info']['winprob']}")
    print(f"    Spread:  {model_data['model_info']['spread']}")

    return model_data


# ════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ultimate model training")
    parser.add_argument("--features", type=int, default=200, help="Number of features (0 = all)")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per model")
    parser.add_argument("--folds", type=int, default=5, help="CV folds")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 72)
    print("  ULTIMATE MODEL TRAINING — Optuna-Tuned Multi-Target Ensembles")
    print("=" * 72)

    # ── 1. Load data ───────────────────────────────────────────────────
    feature_df, feature_cols, _ = load_data()

    # ── 2. Prepare data ────────────────────────────────────────────────
    X_all = feature_df[feature_cols].fillna(0).values
    y_reg = feature_df["total_points"].fillna(feature_df["total_points"].median()).values
    y_clf = feature_df["home_win"].fillna(0).astype(int).values
    y_spread = feature_df["point_diff"].fillna(0).values

    X_all = np.nan_to_num(X_all, nan=0.0)
    print(f"\n  Data: {X_all.shape[0]:,} rows x {X_all.shape[1]} features")
    print(f"  Targets: totals [mean={y_reg.mean():.1f}], win% [{y_clf.mean():.1%}], spread [{y_spread.mean():.1f}]")

    # ── 3. Feature Selection ──────────────────────────────────────────
    print(f"\n  {'=' * 72}")
    print(f"  PHASE 1: Feature Selection")
    print(f"  {'=' * 72}")

    feature_cols_selected = select_top_features(X_all, y_reg, y_clf, feature_cols, args.features)
    X_selected = feature_df[feature_cols_selected].fillna(0).values
    X_selected = np.nan_to_num(X_selected, nan=0.0)
    print(f"  Using {len(feature_cols_selected)} features")

    # ── 4. Cross-Validation with Hyperparameter Tuning ────────────────
    print(f"\n  {'=' * 72}")
    print(f"  PHASE 2: {args.folds}-Fold TimeSeriesSplit CV + Optuna ({args.trials} trials)")
    print(f"  {'=' * 72}")

    cv_results, best_params = run_cv_all_models(
        X_selected, y_reg, y_clf, y_spread,
        n_splits=args.folds, n_trials=args.trials,
    )

    best_tag, best_mae = print_results(cv_results, best_params)

    # ── 5. Build and save final models ─────────────────────────────────
    print(f"\n  {'=' * 72}")
    print(f"  PHASE 3: Building Final Ensembles on All Data")
    print(f"  {'=' * 72}")

    model_data = build_and_save_models(
        X_selected, y_reg, y_clf, y_spread,
        feature_cols_selected, best_params, cv_results, best_mae,
    )

    # ── 6. Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n  {'=' * 72}")
    print(f"  SUMMARY")
    print(f"  {'=' * 72}")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Features:         {len(feature_cols_selected)} (selected from {len(feature_cols)})")
    print(f"  Samples:          {len(X_selected):,}")
    print(f"  Totals CV MAE:    {model_data['metrics']['totals']['cv_mae']:.3f}")
    print(f"  WinProb Brier:    {model_data['metrics']['winprob']['brier']:.4f}")
    print(f"  WinProb Accuracy: {model_data['metrics']['winprob']['accuracy']:.1%}")
    print(f"  Spread CV MAE:    {model_data['metrics']['spread']['cv_mae']:.3f}")
    print(f"  Saved to:         models/ultimate_model.pkl")
    print(f"  {'=' * 72}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
