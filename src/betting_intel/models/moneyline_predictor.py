"""
MoneylinePredictor — dedicated binary classifier for home/away win probability.

Unlike the existing MomentumModel (simple logistic regression) or ELO (heuristic),
this predictor uses gradient-boosted trees (XGBoost, LightGBM) with Platt scaling
calibration for well-calibrated win probabilities.

Pipeline:
  1. Feature engineering (via FeatureEngineer) → raw feature matrix
  2. Feature selection via mutual information → top N features
  3. Walk-forward TimeSeriesSplit → out-of-sample predictions
  4. Platt scaling (CalibratedClassifierCV, sigmoid) → calibrated probs
  5. Ensemble averaging across model types → final probability

Usage:
    from betting_intel.models.moneyline_predictor import MoneylinePredictor

    predictor = MoneylinePredictor()
    predictor.fit(X_train, y_train)
    probs = predictor.predict_proba(X_test)       # calibrated probabilities
    preds = predictor.predict(X_test)              # 0/1 predictions
    metrics = predictor.evaluate(X_test, y_test)   # brier, log-loss, accuracy
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Optional model libraries ─────────────────────────────────────────────

HAS_XGBOOST = False
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    pass

HAS_LIGHTGBM = False
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
#  MONEYLINE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════


class MoneylinePredictor:
    """
    Home win probability classifier with calibrated probabilities.

    Uses an ensemble of gradient-boosted trees with Platt scaling to
    produce well-calibrated win probabilities suitable for edge calculation
    against market-implied probabilities.

    Training uses walk-forward (TimeSeriesSplit) validation to avoid
    lookahead bias — the model is evaluated on out-of-sample predictions
    across chronological folds.

    Parameters
    ----------
    model_types : list of str
        Which model types to include. Options: "xgboost", "lightgbm", "logistic".
        Default: ["xgboost", "lightgbm", "logistic"]
    calibrate : bool
        If True, applies Platt scaling (sigmoid calibration) via CalibratedClassifierCV.
        Default: True.
    n_folds : int
        Number of walk-forward folds for cross-validation. Default: 5.
    feature_selection : bool
        If True, selects top N features via mutual information. Default: True.
    n_select : int
        Number of top features to keep when feature_selection=True. Default: 50.
    random_state : int
        Random seed for reproducibility. Default: 42.

    Attributes
    ----------
    models_ : dict of {str: object}
        Trained base models before calibration.
    calibrated_models_ : dict of {str: CalibratedClassifierCV}
        Calibrated versions of each base model.
    is_fitted : bool
        Whether the predictor has been fitted.
    feature_names_ : list of str
        Feature column names used during training.
    selected_feature_names_ : list of str
        Subset of feature_names_ after feature selection.
    metrics_ : dict of {str: float}
        Latest evaluation metrics from evaluate() or cross-validated.
    fold_metrics_ : list of dict
        Per-fold metrics from walk-forward validation.

    Usage
    -----
        predictor = MoneylinePredictor()
        predictor.fit(X_train, y_train, feature_names=feature_cols)

        # Calibrated probabilities
        probs = predictor.predict_proba(X_test)

        # Evaluate
        metrics = predictor.evaluate(X_val, y_val)
        print(metrics["brier"], metrics["log_loss"], metrics["accuracy"])

        # Cross-validated metrics (no lookahead bias)
        cv_results = predictor.cross_validate(X, y, feature_names=feature_cols)
        print(cv_results["avg_brier"])
    """

    def __init__(
        self,
        model_types: Optional[List[str]] = None,
        calibrate: bool = True,
        n_folds: int = 5,
        feature_selection: bool = True,
        n_select: int = 50,
        random_state: int = 42,
    ):
        self.model_types = model_types or ["xgboost", "lightgbm", "logistic"]
        self.calibrate = calibrate
        self.n_folds = n_folds
        self.feature_selection = feature_selection
        self.n_select = n_select
        self.random_state = random_state

        # Internal state
        self.models_: Dict[str, Any] = {}
        self.calibrated_models_: Dict[str, Any] = {}
        self.is_fitted = False
        self.feature_names_: List[str] = []
        self.selected_feature_names_: List[str] = []
        self._feature_sel_indices_: Optional[np.ndarray] = None  # for ndarray column selection after feature reduction
        self.metrics_: Dict[str, float] = {}
        self.fold_metrics_: List[Dict[str, Any]] = []
        self._feature_importances_: Dict[str, np.ndarray] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "MoneylinePredictor":
        """
        Fit the moneyline predictor on full training data.

        Trains each requested model type on (X, y), optionally applies
        Platt scaling calibration, and stores all models for prediction.

        Args:
            X: (n_samples, n_features) training features.
            y: (n_samples,) binary labels (1 = home win, 0 = home loss).
            feature_names: Optional list of feature column names.

        Returns:
            self
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data")
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)}) and y ({len(y)}) length mismatch")

        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # Feature selection
        if self.feature_selection and X.shape[1] > self.n_select:
            try:
                mi = mutual_info_classif(X, y, random_state=self.random_state)
                top_idx = np.argsort(mi)[-self.n_select:][::-1]
                self.selected_feature_names_ = [self.feature_names_[i] for i in top_idx]
                self._feature_sel_indices_ = top_idx  # store for predict_proba ndarray path
                X_sel = X[:, top_idx]
                logger.info(
                    f"Feature selection: {X.shape[1]} → {len(top_idx)} "
                    f"(top: {', '.join(self.selected_feature_names_[:5])}...)"
                )
            except Exception as e:
                logger.debug(f"Feature selection failed: {e}")
                X_sel = X
                self.selected_feature_names_ = self.feature_names_
                self._feature_sel_indices_ = None  # prevent stale indices from previous fit()
        else:
            X_sel = X
            self.selected_feature_names_ = self.feature_names_

        # Train each model type
        for model_type in self.model_types:
            model = self._build_model(model_type)
            if model is None:
                continue

            try:
                model.fit(X_sel, y)
                self.models_[model_type] = model

                # Extract feature importance
                if hasattr(model, "feature_importances_"):
                    self._feature_importances_[model_type] = model.feature_importances_
                elif hasattr(model, "coef_"):
                    self._feature_importances_[model_type] = np.abs(model.coef_[0])

                # Apply Platt scaling calibration
                if self.calibrate:
                    calibrated = CalibratedClassifierCV(
                        estimator=self._build_model(model_type),
                        method="sigmoid",
                        cv=3,
                    )
                    calibrated.fit(X_sel, y)
                    self.calibrated_models_[model_type] = calibrated

                logger.info(f"Trained: {model_type}")
            except Exception as e:
                logger.warning(f"Failed to train {model_type}: {e}")

        if not self.models_:
            raise RuntimeError(
                "No models trained successfully. "
                f"Available model types: {self.model_types}. "
                f"XGBoost: {HAS_XGBOOST}, LightGBM: {HAS_LIGHTGBM}"
            )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary outcomes (0/1).

        Uses ensemble average of calibrated probabilities, thresholded at 0.5.

        Args:
            X: (n_samples, n_features) input features.

        Returns:
            (n_samples,) binary predictions.
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict calibrated home win probabilities.

        Averages predictions across all trained (and calibrated) models.

        Args:
            X: (n_samples, n_features) input features.

        Returns:
            (n_samples,) probability of home win (0-1).
        """
        if not self.is_fitted:
            raise ValueError("Not fitted yet. Call fit() first.")

        # Select features: if feature selection reduced dimensions during fit(),
        # we need to apply the same reduction to prediction data.
        # For ndarray: use stored column indices from fit().
        # For DataFrame: select columns by name.
        if self._feature_sel_indices_ is not None and X.shape[1] > len(self.selected_feature_names_):
            # ndarray path — use stored column indices
            X_sel = X[:, self._feature_sel_indices_]
        elif isinstance(X, pd.DataFrame) and self.selected_feature_names_:
            # DataFrame path — select columns by name
            available = [c for c in self.selected_feature_names_ if c in X.columns]
            if available:
                X_sel = X[available].values
            else:
                X_sel = X.values
                logger.warning(
                    f"Feature name mismatch: selected {len(self.selected_feature_names_)}, "
                    f"got {X.shape[1]}, none matched by name"
                )
        else:
            X_sel = X

        predictions = []
        for name in self.models_:
            try:
                if self.calibrate and name in self.calibrated_models_:
                    # Calibrated model returns (n, 2) — take class 1 prob
                    proba = self.calibrated_models_[name].predict_proba(X_sel)
                    if proba.ndim == 2:
                        predictions.append(proba[:, 1])
                    else:
                        predictions.append(proba)
                elif hasattr(self.models_[name], "predict_proba"):
                    proba = self.models_[name].predict_proba(X_sel)
                    if proba.ndim == 2:
                        predictions.append(proba[:, 1])
                    else:
                        predictions.append(proba)
                else:
                    preds = self.models_[name].predict(X_sel)
                    predictions.append(preds.astype(float))
            except Exception as e:
                logger.debug(f"{name} predict_proba failed: {e}")
                continue

        if not predictions:
            return np.full(X.shape[0], 0.5)

        return np.mean(predictions, axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the predictor on test data.

        Args:
            X: (n_samples, n_features) test features.
            y: (n_samples,) ground truth labels.

        Returns:
            Dict with brier, log_loss, accuracy, auc_roc.
        """
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)

        metrics: Dict[str, float] = {
            "brier": float(brier_score_loss(y, probs)),
            "accuracy": float(accuracy_score(y, preds)),
            "n_samples": len(y),
        }

        # Log loss requires both classes present
        try:
            metrics["log_loss"] = float(log_loss(y, probs))
        except Exception:
            metrics["log_loss"] = 1.0

        # AUC-ROC requires both classes present
        if len(np.unique(y)) == 2:
            try:
                metrics["auc_roc"] = float(roc_auc_score(y, probs))
            except Exception:
                metrics["auc_roc"] = 0.5
        else:
            metrics["auc_roc"] = 0.5

        # Calibration error: mean absolute difference between predicted prob
        # and actual frequency across decile bins
        try:
            bins = np.linspace(0, 1, 11)
            bin_indices = np.digitize(probs, bins) - 1
            bin_indices = np.clip(bin_indices, 0, 9)
            cal_error = 0.0
            n_bins_used = 0
            for b in range(10):
                mask = bin_indices == b
                if mask.sum() > 0:
                    avg_prob = probs[mask].mean()
                    actual = y[mask].mean()
                    cal_error += abs(avg_prob - actual) * mask.sum()
                    n_bins_used += mask.sum()
            metrics["calibration_error"] = float(cal_error / max(n_bins_used, 1))
        except Exception:
            metrics["calibration_error"] = 1.0

        self.metrics_ = metrics
        return metrics

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """
        Walk-forward cross-validation with TimeSeriesSplit.

        Each fold trains on past data, predicts on future data (no lookahead).
        Metrics are computed only on out-of-sample predictions.

        Args:
            X: (n_samples, n_features).
            y: (n_samples,) binary labels.
            feature_names: Optional feature names.
            n_splits: Number of chronological splits.

        Returns:
            Dict with avg metrics, per-fold metrics, and OOS predictions.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oos_probs = np.full(len(y), np.nan)
        self.fold_metrics_ = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if len(train_idx) < 50 or len(test_idx) < 10:
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                fold_predictor = MoneylinePredictor(
                    model_types=self.model_types,
                    calibrate=self.calibrate,
                    feature_selection=self.feature_selection,
                    n_select=self.n_select,
                    random_state=self.random_state,
                )
                fold_predictor.fit(X_train, y_train, feature_names=feature_names)
                fold_probs = fold_predictor.predict_proba(X_test)
                oos_probs[test_idx] = fold_probs

                fold_metrics = fold_predictor.evaluate(X_test, y_test)
                fold_metrics["fold"] = fold
                fold_metrics["n_train"] = len(X_train)
                fold_metrics["n_test"] = len(X_test)
                self.fold_metrics_.append(fold_metrics)
            except Exception as e:
                logger.debug(f"Fold {fold} failed: {e}")
                continue

        if not self.fold_metrics_:
            return {"avg_brier": 1.0, "avg_log_loss": 1.0, "avg_accuracy": 0.5, "n_folds": 0}

        valid_oos = oos_probs[~np.isnan(oos_probs)]
        valid_y = y[~np.isnan(oos_probs)]

        avg_metrics = {
            "avg_brier": float(np.mean([m["brier"] for m in self.fold_metrics_])),
            "avg_log_loss": float(np.mean([m["log_loss"] for m in self.fold_metrics_])),
            "avg_accuracy": float(np.mean([m["accuracy"] for m in self.fold_metrics_])),
            "avg_auc_roc": float(np.mean([m.get("auc_roc", 0.5) for m in self.fold_metrics_])),
            "n_folds": len(self.fold_metrics_),
            "n_oos": len(valid_oos),
            "fold_metrics": self.fold_metrics_,
        }

        if len(valid_oos) > 0:
            avg_metrics["oos_brier"] = float(brier_score_loss(valid_y, valid_oos))
            avg_metrics["oos_log_loss"] = float(log_loss(valid_y, valid_oos))
            avg_metrics["oos_accuracy"] = float(accuracy_score((valid_oos >= 0.5).astype(int), valid_y))
            if len(np.unique(valid_y)) == 2:
                try:
                    avg_metrics["oos_auc_roc"] = float(roc_auc_score(valid_y, valid_oos))
                except Exception:
                    avg_metrics["oos_auc_roc"] = 0.5

        # Log summary
        logger.info(
            f"Walk-forward CV: {avg_metrics['n_folds']} folds, "
            f"{avg_metrics.get('n_oos', 0)} OOS samples, "
            f"Brier={avg_metrics['avg_brier']:.4f}, "
            f"LogLoss={avg_metrics['avg_log_loss']:.4f}, "
            f"AUC={avg_metrics['avg_auc_roc']:.3f}"
        )

        # Refit on full data
        self.fit(X, y, feature_names=feature_names)

        return avg_metrics

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance from each trained model.

        Returns:
            Dict of {model_name: {feature_name: importance_score}}.
        """
        result: Dict[str, Dict[str, float]] = {}
        for name, importance in self._feature_importances_.items():
            if len(importance) != len(self.selected_feature_names_):
                continue
            indices = np.argsort(importance)[-top_n:][::-1]
            result[name] = {
                self.selected_feature_names_[i]: float(importance[i])
                for i in indices
                if i < len(self.selected_feature_names_)
            }
        return result

    def get_params(self) -> Dict[str, Any]:
        """Get predictor configuration parameters."""
        return {
            "name": "MoneylinePredictor",
            "model_types": self.model_types,
            "calibrate": self.calibrate,
            "n_folds": self.n_folds,
            "feature_selection": self.feature_selection,
            "n_select": self.n_select,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "n_models": len(self.models_),
            "model_names": list(self.models_.keys()),
            "n_features": len(self.selected_feature_names_),
        }

    # ── Internal: Model Builders ──────────────────────────────────────────

    def _build_model(self, model_type: str) -> Any:
        """Build a base classifier for the given model type."""
        if model_type == "xgboost" and HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=3.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
                eval_metric="logloss",
            )
        elif model_type == "lightgbm" and HAS_LIGHTGBM:
            return LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=24,
                min_child_samples=25,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=3.0,
                random_state=self.random_state,
                verbosity=-1,
            )
        elif model_type == "logistic":
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
            )
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING WRAPPER (for retrain_all.py integration)
# ═══════════════════════════════════════════════════════════════════════════


def train_moneyline_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "point_diff",
    calibrate: bool = True,
    cv: bool = True,
    save: bool = True,
    model_name: str = "moneyline_ensemble",
) -> Tuple[MoneylinePredictor, Dict[str, Any]]:
    """
    Train and save a MoneylinePredictor for home win probability.

    Designed to be called from retrain_all.py. Works with the existing
    FeatureEngineer output (feature matrix with total_points, point_diff).

    Args:
        df: Feature DataFrame with feature_cols + target_col.
        feature_cols: Feature column names.
        target_col: Column to derive binary target from (point_diff > 0).
        calibrate: Apply Platt scaling calibration.
        cv: Run walk-forward cross-validation.
        save: Save model via ModelRegistry.
        model_name: Name for ModelRegistry.

    Returns:
        Tuple of (trained MoneylinePredictor, metrics dict).
    """
    X = df[feature_cols].fillna(0).values
    y = (df[target_col].values > 0).astype(int)

    predictor = MoneylinePredictor(calibrate=calibrate)

    metrics: Dict[str, Any] = {}

    if cv:
        cv_results = predictor.cross_validate(X, y, feature_names=feature_cols)
        metrics["cv"] = cv_results
        logger.info(
            f"Moneyline CV: Brier={cv_results.get('avg_brier', '?'):.4f}, "
            f"AUC={cv_results.get('avg_auc_roc', '?'):.3f}, "
            f"{cv_results.get('n_folds', 0)} folds"
        )
    else:
        predictor.fit(X, y, feature_names=feature_cols)
        probs = predictor.predict_proba(X)
        train_metrics = predictor.evaluate(X, y)
        metrics["train"] = train_metrics
        logger.info(
            f"Moneyline train: Brier={train_metrics['brier']:.4f}, "
            f"Accuracy={train_metrics['accuracy']:.1%}"
        )

    # Save model
    if save:
        _save_moneyline_model(predictor, model_name, metrics, feature_cols)

    return predictor, metrics


def _save_moneyline_model(
    predictor: MoneylinePredictor,
    model_name: str,
    metrics: Dict[str, Any],
    feature_cols: List[str],
):
    """Save the trained predictor via ModelRegistry."""
    try:
        from betting_intel.models.persistence import model_registry

        model_registry.save(
            model=predictor,
            model_name=model_name,
            feature_cols=predictor.selected_feature_names_ or feature_cols,
            metrics=metrics,
            parameters=predictor.get_params(),
        )
        logger.info(f"Moneyline model saved: {model_name}")
    except Exception as e:
        logger.warning(f"Failed to save moneyline model: {e}")


def load_moneyline_model(
    model_name: str = "moneyline_ensemble",
    version: Optional[str] = None,
) -> Optional[MoneylinePredictor]:
    """Load a trained MoneylinePredictor from the model registry."""
    try:
        from betting_intel.models.persistence import model_registry

        model, metadata = model_registry.load(model_name, version)
        if not isinstance(model, MoneylinePredictor):
            logger.warning(
                f"Loaded model is {type(model).__name__}, "
                f"expected MoneylinePredictor"
            )
            return None
        return model
    except FileNotFoundError:
        logger.info(f"No saved moneyline model found for '{model_name}'")
        return None
    except Exception as e:
        logger.warning(f"Failed to load moneyline model: {e}")
        return None


__all__ = [
    "MoneylinePredictor",
    "train_moneyline_model",
    "load_moneyline_model",
]
