#!/usr/bin/env python3
"""
Retrain All Models — extracted from .github/workflows/ weekly & monthly retraining YAML.

Usage:
    python tools/retrain_all.py --mode weekly     # Train total + spread models (Monday cron)
    python tools/retrain_all.py --mode monthly    # Train total + spread + win prob + report (1st cron)

Supports both modes so the CI workflows can call this single script instead of
inlining 100+ lines of Python in YAML.
"""

import sys
import os
import warnings
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Project imports ───────────────────────────────────────────────────
from betting_intel.data.loader import NBADataLoader
from betting_intel.data.features import FeatureEngineer
from betting_intel.pipeline.performance import get_performance_tracker as _get_perf_tracker


# ======================================================================
#  Data Loading (shared by both modes)
# ======================================================================

def load_and_prepare_data() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Load historical NBA data and engineer features.

    Returns:
        clean_df: DataFrame with features and targets (NaN rows dropped)
        feature_cols: List of selected feature column names
        feature_df: Full feature DataFrame (before NaN dropping)
    """
    print("Loading data...")
    loader = NBADataLoader()
    fe = FeatureEngineer()

    raw_df = loader.load_game_logs()
    games_df = loader.build_game_dataset(raw_df)
    raw_df = loader.compute_rest_days(raw_df)
    feature_df = fe.build_all_features(games_df, raw_df)
    feature_cols = fe.select_features(feature_df)

    print(f"  Features: {len(feature_cols)}")
    print(f"  Games:    {len(feature_df)}")

    clean_df = feature_df.dropna(subset=feature_cols, thresh=len(feature_cols) // 2).copy()
    clean_df = clean_df.reset_index(drop=True)
    print(f"  Clean:    {len(clean_df)} rows (dropped {len(feature_df) - len(clean_df)})")

    return clean_df, feature_cols, feature_df


# ======================================================================
#  Model Training — Enhanced with CatBoost + Optuna tuning
# ======================================================================

_HAS_CATBOOST = False
try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    pass


def _build_ensemble_models():
    """Return dict of models used for the ensemble — now includes CatBoost if available."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    models = {
        "gbr": GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05),
        "rfr": RandomForestRegressor(n_estimators=150, max_depth=6),
        "ridge": Ridge(alpha=1.0),
    }
    if _HAS_CATBOOST:
        models["catboost"] = CatBoostRegressor(
            iterations=400, learning_rate=0.05, depth=4,
            l2_leaf_reg=5.0, random_seed=42, verbose=0,
        )
    return models


# ── Optuna Hyperparameter Tuning ────────────────────────────────────────

def _tune_lightgbm_params(X_train, y_train, X_val, y_val, n_trials=30) -> dict:
    """Run quick Optuna hyperparameter search for LightGBM.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of trials (default 30 for fast tuning)

    Returns:
        Dict of best hyperparameters found
    """
    try:
        import optuna
        from lightgbm import LGBMRegressor
        from sklearn.metrics import mean_absolute_error
    except ImportError:
        return {}  # No tuning available

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 8, 48),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "random_state": 42,
            "verbosity": -1,
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return float(mean_absolute_error(y_val, preds))

    try:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=120, show_progress_bar=False)
        print(f"  🎛  Optuna: best MAE={study.best_value:.3f} in {len(study.trials)} trials")
        return study.best_params
    except Exception as e:
        print(f"  ℹ  Optuna tuning failed: {e}")
        return {}


def _tune_ridge_alpha(X_train, y_train, X_val, y_val) -> float:
    """Find optimal Ridge alpha via grid search."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    best_mae = float("inf")
    best_alpha = 1.0
    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    return best_alpha


def train_total_models(
    df: pd.DataFrame, feature_cols: list[str],
    tune: bool = False,
) -> dict:
    """Train total points model ensemble and save to models/total_model.pkl.

    Args:
        df: Clean DataFrame with features and total_points target
        feature_cols: Feature column names
        tune: If True, run Optuna hyperparameter tuning

    Returns dict with training metrics.
    """
    from sklearn.metrics import mean_absolute_error
    import joblib

    X = df[feature_cols].values
    y_total = df["total_points"].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_total[:split_idx], y_total[split_idx:]

    models = _build_ensemble_models()
    predictions = []

    # ── Hyperparameter tuning ───────────────────────────────────────
    if tune and len(X_train) > 200:
        print("  🔬  Tuning hyperparameters...")
        try:
            lgb_best = _tune_lightgbm_params(X_train, y_train, X_val, y_val, n_trials=30)
            if lgb_best:
                from lightgbm import LGBMRegressor
                tuned_lgb = LGBMRegressor(**lgb_best)
                tuned_lgb.fit(X_train, y_train)
                models["tuned_lgb"] = tuned_lgb
                print(f"  ✅  Tuned LightGBM: {lgb_best.get('learning_rate', '?')} lr, "
                      f"{lgb_best.get('max_depth', '?')} depth, "
                      f"{lgb_best.get('num_leaves', '?')} leaves")
        except Exception as e:
            print(f"  ℹ  Tuning skipped: {e}")

        best_ridge_alpha = _tune_ridge_alpha(X_train, y_train, X_val, y_val)
        models["ridge"] = Ridge(alpha=best_ridge_alpha, random_state=42)
        print(f"  ✅  Tuned Ridge: alpha={best_ridge_alpha}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        predictions.append(preds)
        mae = mean_absolute_error(y_val, preds)
        print(f"  Total {name}: MAE={mae:.2f}")

    ensemble = np.mean(predictions, axis=0)
    ensemble_mae = mean_absolute_error(y_val, ensemble)
    print(f"  Total Ensemble ({len(models)} models): MAE={ensemble_mae:.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model_type": "total_points_ensemble",
        "models": {name: models[name] for name in models},
        "feature_cols": feature_cols,
        "val_mae": ensemble_mae,
        "training_samples": len(X_train),
        "training_date": pd.Timestamp.now().isoformat(),
    }, "models/total_model.pkl")
    print("  -> Saved to models/total_model.pkl")

    # ── Record to persistent performance history ────────────────
    try:
        tracker = _get_perf_tracker()
        tracker.record_run(
            model_name="retrain_total",
            test_mae=ensemble_mae,
            n_features=len(feature_cols),
            n_samples=len(X_train),
            n_folds=len(models),
            models_used=list(models.keys()),
            mode="retrain",
            tune=tune,
        )
    except Exception as e:
        print(f"  ℹ  Performance tracking: {e}")

    return {"mae": ensemble_mae, "n_train": len(X_train), "n_val": len(X_val), "n_models": len(models)}


def train_spread_models(
    df: pd.DataFrame, feature_cols: list[str],
    tune: bool = False,
) -> dict:
    """Train spread model ensemble with CatBoost + tuning if requested."""
    from sklearn.metrics import mean_absolute_error
    import joblib

    X = df[feature_cols].values
    y_spread = df["point_diff"].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_spread[:split_idx], y_spread[split_idx:]

    models = _build_ensemble_models()
    predictions = []

    if tune and len(X_train) > 200:
        print("  🔬  Tuning spread hyperparameters...")
        best_alpha = _tune_ridge_alpha(X_train, y_train, X_val, y_val)
        models["ridge"] = Ridge(alpha=best_alpha, random_state=42)
        print(f"  ✅  Tuned Ridge: alpha={best_alpha}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        predictions.append(preds)
        mae = mean_absolute_error(y_val, preds)
        print(f"  Spread {name}: MAE={mae:.2f}")

    ensemble = np.mean(predictions, axis=0)
    ensemble_mae = mean_absolute_error(y_val, ensemble)
    print(f"  Spread Ensemble ({len(models)} models): MAE={ensemble_mae:.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model_type": "spread_ensemble",
        "models": {name: models[name] for name in models},
        "feature_cols": feature_cols,
        "val_mae": ensemble_mae,
        "training_samples": len(X_train),
        "training_date": pd.Timestamp.now().isoformat(),
    }, "models/spread_model.pkl")
    print("  -> Saved to models/spread_model.pkl")

    # ── Record to persistent performance history ────────────────
    try:
        tracker = _get_perf_tracker()
        tracker.record_run(
            model_name="retrain_spread",
            test_mae=ensemble_mae,
            n_features=len(feature_cols),
            n_samples=len(X_train),
            n_folds=len(models),
            models_used=list(models.keys()),
            mode="retrain",
            tune=tune,
        )
    except Exception as e:
        print(f"  ℹ  Performance tracking: {e}")

    return {"mae": ensemble_mae, "n_train": len(X_train), "n_val": len(X_val), "n_models": len(models)}


def train_win_probability_model(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Train win probability model (classification with calibration).

    Used by monthly retraining only. Saves to models/win_model.pkl.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    import joblib

    X = df[feature_cols].values
    y_win = (df["point_diff"].values > 0).astype(int)

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_win[:split_idx], y_win[split_idx:]

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(clf, cv=3, method="sigmoid")
    calibrated.fit(X_train, y_train)

    win_probs = calibrated.predict_proba(X_val)[:, 1]
    brier = brier_score_loss(y_val, win_probs)
    print(f"  Win Prob: Brier={brier:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model_type": "win_probability",
        "model": calibrated,
        "feature_cols": feature_cols,
        "val_brier": brier,
        "training_samples": len(X_train),
        "training_date": pd.Timestamp.now().isoformat(),
    }, "models/win_model.pkl")
    print("  -> Saved to models/win_model.pkl")

    # ── Record to persistent performance history ────────────────
    try:
        tracker = _get_perf_tracker()
        tracker.record_run(
            model_name="retrain_win_prob",
            n_features=len(feature_cols),
            n_samples=len(X_train),
            models_used=["LogisticRegression+CalibratedClassifierCV"],
            mode="retrain",
            val_brier=round(brier, 4),
        )
    except Exception as e:
        print(f"  ℹ  Performance tracking: {e}")

    return {"brier": brier, "n_train": len(X_train), "n_val": len(X_val)}


def run_calibration_check():
    """Verify calibration modules load correctly."""
    try:
        from betting_intel.validation.calibration import ProbabilityCalibrator, PlattCalibrator, IsotonicCalibrator
        print("  Calibration modules verified")
        return True
    except ImportError as e:
        print(f"  [WARN] Calibration modules not available: {e}")
        return False


# ======================================================================
#  Weekly Mode
# ======================================================================

def run_weekly(tune: bool = False):
    """Full weekly retrain: total + spread models."""
    print("=" * 60)
    print(f"  WEEKLY RETRAINING — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    df, feature_cols, _ = load_and_prepare_data()
    print()

    total_metrics = train_total_models(df, feature_cols, tune=tune)
    print()
    spread_metrics = train_spread_models(df, feature_cols, tune=tune)

    print(f"\n{'=' * 60}")
    print(f"  WEEKLY RETRAINING COMPLETE")
    print(f"  Total MAE: {total_metrics['mae']:.2f}  |  Spread MAE: {spread_metrics['mae']:.2f}")
    print(f"  Models: {total_metrics.get('n_models', '?')} total, {spread_metrics.get('n_models', '?')} spread")
    print(f"  Models saved to models/")
    print(f"{'=' * 60}")


# ======================================================================
#  Monthly Mode
# ======================================================================

def run_monthly(tune: bool = False):
    """Full monthly retrain: total + spread + win prob + moneyline + calibration check."""
    print("=" * 60)
    print(f"  FULL MONTHLY RETRAINING — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    df, feature_cols, _ = load_and_prepare_data()

    print("\n--- Total Points Model ---")
    total_metrics = train_total_models(df, feature_cols, tune=tune)

    print("\n--- Spread Model ---")
    spread_metrics = train_spread_models(df, feature_cols, tune=tune)

    print("\n--- Win Probability Model ---")
    win_metrics = train_win_probability_model(df, feature_cols)

    print("\n--- MoneylinePredictor (XGBoost+LightGBM+Logistic, walk-forward CV) ---")
    try:
        from betting_intel.models.moneyline_predictor import train_moneyline_model
        ml_predictor, ml_metrics = train_moneyline_model(
            df, feature_cols, target_col="point_diff",
            calibrate=True, cv=True, save=True,
            model_name="moneyline_ensemble",
        )
        cv = ml_metrics.get("cv", {})
        print(f"  Moneyline CV: Brier={cv.get('avg_brier', '?'):.4f}, "
              f"AUC={cv.get('avg_auc_roc', '?'):.3f}, "
              f"Accuracy={cv.get('avg_accuracy', '?'):.1%}, "
              f"{cv.get('n_folds', 0)} folds, "
              f"{cv.get('n_oos', 0)} OOS samples")
    except Exception as e:
        print(f"  [SKIP] MoneylinePredictor: {e}")

    print("\n--- Calibration Check ---")
    run_calibration_check()

    print(f"\n{'=' * 60}")
    print(f"  MONTHLY RETRAINING COMPLETE")
    print(f"  Total MAE:  {total_metrics['mae']:.2f}")
    print(f"  Spread MAE: {spread_metrics['mae']:.2f}")
    print(f"  Win Brier:  {win_metrics['brier']:.4f}")
    print(f"  Models: {total_metrics.get('n_models', '?')} total, {spread_metrics.get('n_models', '?')} spread")
    print(f"  Models saved to models/")
    print(f"{'=' * 60}")


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Retrain all prediction models. Extracted from CI workflows.",
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["weekly", "monthly"],
        help="weekly: total + spread models | monthly: total + spread + win prob + moneyline + calibration",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter tuning (monthly mode recommended for tuning)",
    )
    args = parser.parse_args()

    if args.mode == "weekly":
        run_weekly(tune=args.tune)
    else:
        run_monthly(tune=args.tune)

    return 0


if __name__ == "__main__":
    sys.exit(main())
