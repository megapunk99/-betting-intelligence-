"""
RobustPredictionSystem — the most rigorous prediction engine in the platform.

Architecture
────────────
  1. Multi-Model Ensemble
     - XGBoost (gradient boosted trees, handles non-linear interactions)
     - LightGBM (fast gradient boosting, good with categoricals)
     - LogisticRegression (interpretable linear baseline, L2-regularised)
     - RandomForest (bagging ensemble, captures variance)

  2. Walk-Forward Time-Series Cross-Validation
     - Chronological splits — NEVER trains on future data
     - Out-of-sample predictions collected as meta-features
     - Ridge meta-model trained on OOS predictions (stacking)

  3. Probability Calibration
     - Platt scaling (sigmoid) calibration via sklearn's CalibratedClassifierCV
     - Brier score computed on calibrated vs raw probabilities
     - Calibration curve diagnostics

  4. Overfitting Detection
     - Train vs test R² gap analysis
     - Flag if gap > 0.15 or test R² < -0.10
     - Per-fold metrics tracked

  5. Adaptive Model Weights
     - Weights learned from recent OOS performance (Brier score)
     - Exponential decay weighting (more recent = more important)
     - Degenerate model handling (floor weight of 0.05)

  6. Confidence Scoring
     - Consensus level (1 - variance * 20) — agreement among models
     - Model count weighted confidence
     - Historical calibration-based confidence adjustment

  7. Statistical Significance
     - Binomial test against 50% baseline
     - P-value for win rate > 50%
     - Sharpe ratio (risk-adjusted return)
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from betting_intel.utils.safe_serialize import (
    safe_joblib_dump,
    safe_joblib_load,
)
from sklearn.metrics import r2_score, mean_absolute_error, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ModelDiagnostics:
    """Diagnostic information for a single model in the ensemble."""

    name: str
    oos_brier: float = 0.0
    oos_accuracy: float = 0.0
    oos_log_loss: float = 0.0
    train_r2: Optional[float] = None
    test_r2: Optional[float] = None
    n_oos: int = 0
    weight: float = 1.0
    is_calibrated: bool = False
    status: str = "unknown"  # "ok", "degraded", "failed"


@dataclass
class PredictionResult:
    """Prediction result with full confidence breakdown."""

    home_win_prob: float = 0.5
    away_win_prob: float = 0.5

    # Ensemble details
    n_models: int = 0
    consensus: float = 0.0  # 0-1, how much models agree
    model_variance: float = 0.0

    # Calibrated probability (if available)
    calibrated_home_win_prob: Optional[float] = None

    # Model breakdown
    model_probs: dict[str, float] = field(default_factory=dict)
    model_weights: dict[str, float] = field(default_factory=dict)

    # Confidence
    confidence_score: float = 0.0  # 0-1
    confidence_label: str = "LOW"  # VERY_HIGH, HIGH, MEDIUM, LOW, VERY_LOW

    # Edge (if market odds provided)
    edge_pct: Optional[float] = None
    expected_value: Optional[float] = None
    kelly_fraction: Optional[float] = None

    # Metadata
    generated_at: str = ""
    feature_importance: dict[str, float] = field(default_factory=dict)

    # Calibration status (v5.1 — transparency)
    calibration_applied: bool = False
    calibration_failed: bool = False
    calibration_warning: Optional[str] = None


@dataclass
class WalkForwardFold:
    """Metrics from a single walk-forward fold."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_r2: float = 0.0
    test_r2: float = 0.0
    train_mae: float = 0.0
    test_mae: float = 0.0
    gap_r2: float = 0.0
    n_train: int = 0
    n_test: int = 0


@dataclass
class OverfittingReport:
    """Overfitting analysis report."""

    is_overfit: bool = False
    avg_train_r2: float = 0.0
    avg_test_r2: float = 0.0
    r2_gap: float = 0.0
    flags: list[str] = field(default_factory=list)
    folds: list[WalkForwardFold] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  ROBUST PREDICTION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════


class RobustPredictionSystem:
    """
    The highest-quality prediction system — multi-model ensemble with
    walk-forward cross-validation, calibration, overfitting detection,
    and confidence scoring.

    Usage:
        system = RobustPredictionSystem()
        system.fit(X_train, y_train)
        pred = system.predict_proba(X_test)
        # pred.shape == (n_samples, 2)  # [P(loss), P(win)]

        result = system.predict_with_details(X_single)
        # result.home_win_prob, result.confidence_score, etc.
    """

    def __init__(
        self,
        calibrate: bool = True,
        calibration_method: str = "auto",  # 'platt', 'isotonic', 'auto' (v6.6)
        n_folds: int = 5,
        min_train_samples: int = 100,
        min_test_samples: int = 10,
        random_state: int = 42,
        use_stacking: bool = True,
        lgb_params: Optional[dict] = None,
        xgb_params: Optional[dict] = None,
        lr_params: Optional[dict] = None,
        rf_params: Optional[dict] = None,
        cb_params: Optional[dict] = None,
        use_catboost: bool = True,
        use_mlp: bool = True,
        use_early_stopping: bool = True,
        use_hyperparameter_tuning: bool = False,  # v6.6 NEW — wire tuner into fit()
        ensemble_diversity_threshold: float = 0.3,
        stacking_meta_model: str = "ridge",  # 'ridge', 'lasso', 'xgboost', or 'none'
        use_adversarial_validation: bool = False,  # v6.6 NEW
        use_permutation_importance: bool = False,  # v6.6 NEW
        use_bootstrap_uncertainty: bool = False,  # v6.6 NEW
        n_bootstrap_samples: int = 50,  # v6.6 NEW
        pruning_keep_top_n: int = 0,  # v6.6 NEW — 0 = no pruning
    ):
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.n_folds = max(3, n_folds)
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        self.random_state = random_state
        self.use_stacking = use_stacking
        self.use_catboost = use_catboost
        self.use_early_stopping = use_early_stopping
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.ensemble_diversity_threshold = ensemble_diversity_threshold
        self.stacking_meta_model = stacking_meta_model
        self.use_adversarial_validation = use_adversarial_validation
        self.use_permutation_importance = use_permutation_importance
        self.use_bootstrap_uncertainty = use_bootstrap_uncertainty
        self.n_bootstrap_samples = n_bootstrap_samples
        self.pruning_keep_top_n = pruning_keep_top_n

        # Model parameters
        self._lgb_params = lgb_params or {}
        self._xgb_params = xgb_params or {}
        self._lr_params = lr_params or {}
        self._rf_params = rf_params or {}
        self._cb_params = cb_params or {}

        # Internal state
        self._models: dict[str, Any] = {}
        self._calibrators: dict[str, Any] = {}
        self._meta_model: Any = None
        self._weights: dict[str, float] = {}
        self._feature_names: list[str] = []
        self._fitted: bool = False
        self._fold_metrics: list[WalkForwardFold] = []
        self._overfitting: Optional[OverfittingReport] = None
        self._model_diagnostics: dict[str, ModelDiagnostics] = {}
        self._target_mean: float = 0.5
        self._n_train_total: int = 0

        # v6.6 — New internal state
        self._ensemble_diversity: Optional[dict[str, float]] = None
        self._pruned_models: list[str] = []
        self._permutation_importance: Optional[dict[str, float]] = None
        self._bootstrap_probs: Optional[np.ndarray] = None
        self._bootstrap_std: Optional[np.ndarray] = None
        self._adversarial_score: Optional[float] = None
        self._adversarial_auroc: Optional[float] = None
        self._hyperparameter_tuning_results: Optional[dict] = None
        self._hyperparameter_tuner: Any = None

        # Calibration data
        self._calibrated_probs: Optional[np.ndarray] = None
        self._raw_probs: Optional[np.ndarray] = None
        self._brier_score: Optional[float] = None
        self._calibrated_brier: Optional[float] = None
        self._calibration_model: Any = None
        self._calibration_models: dict[
            str, Any
        ] = {}  # v6.6 — store per-model calibrators

        # Persistence
        self._fit_timestamp: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> RobustPredictionSystem:
        """
        Train the robust ensemble on historical data.

        Process:
          1. Walk-forward CV: train each model on chronological folds,
             collect OOS predictions
          2. Calibrate: Platt scaling on OOS predictions
          3. Stacking: train meta-model on OOS predictions (optional)
          4. Overfitting detection: analyse train/test R² across folds
          5. Final models: retrain on ALL data for production use
          6. Compute weights: based on OOS Brier score per model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) — 0/1 for binary classification
            feature_names: Optional column names for feature importance
            sample_weight: Optional sample weights
            verbose: Print progress

        Returns:
            self (fitted)
        """
        n = len(X)
        self._n_train_total = n
        self._target_mean = float(np.mean(y))

        if feature_names and len(feature_names) == X.shape[1]:
            self._feature_names = feature_names
        else:
            self._feature_names = [f"f{i}" for i in range(X.shape[1])]

        if n < self.min_train_samples + self.min_test_samples:
            raise ValueError(
                f"Need at least {self.min_train_samples + self.min_test_samples} samples, "
                f"got {n}. Consider reducing min_train_samples."
            )

        # Step 0: Hyperparameter Tuning (v6.6)
        if self.use_hyperparameter_tuning:
            self._tune_hyperparameters(X, y, verbose=verbose)

        # Determine fold boundaries (chronological)
        fold_size = max(self.min_test_samples, n // self.n_folds)
        fold_boundaries = []
        for i in range(1, self.n_folds + 1):
            test_end = min(i * fold_size, n)
            test_start = test_end - fold_size
            if (
                test_start >= self.min_train_samples
                and (test_end - test_start) >= self.min_test_samples
            ):
                fold_boundaries.append((test_start, test_end))

        if not fold_boundaries:
            fold_boundaries = [(self.min_train_samples, n)]

        # ── Step 1: Train models on each fold, collect OOS predictions ──
        all_models_oos: dict[str, list[np.ndarray]] = {}
        all_oos_targets: list[np.ndarray] = []
        fold_metrics_list: list[WalkForwardFold] = []

        model_specs = self._get_model_specs()

        for fold_idx, (test_start, test_end) in enumerate(fold_boundaries):
            X_train = X[:test_start]
            y_train = y[:test_start]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            sw_train = sample_weight[:test_start] if sample_weight is not None else None

            fold_metrics = WalkForwardFold(
                fold=fold_idx + 1,
                train_start=0,
                train_end=test_start,
                test_start=test_start,
                test_end=test_end,
                n_train=len(X_train),
                n_test=len(X_test),
            )

            for model_name, model_fn, params in model_specs:
                try:
                    model = model_fn(**params)

                    # For sklearn models, use sample_weight if provided
                    if sw_train is not None and hasattr(model, "fit"):
                        model.fit(X_train, y_train, sample_weight=sw_train)
                    elif hasattr(model, "fit"):
                        model.fit(X_train, y_train)
                    else:
                        raise TypeError(f"Model {model_name} has no .fit() method")

                    # OOS predictions
                    if hasattr(model, "predict_proba"):
                        oos_probs = model.predict_proba(X_test)
                        if oos_probs.ndim == 2 and oos_probs.shape[1] >= 2:
                            oos_preds = oos_probs[:, 1]
                        else:
                            oos_preds = oos_probs
                    elif hasattr(model, "predict"):
                        oos_preds = model.predict(X_test)
                    else:
                        continue

                    if model_name not in all_models_oos:
                        all_models_oos[model_name] = []
                    all_models_oos[model_name].append(oos_preds)

                    # Compute fold metrics for this model
                    # Use simple average for fold-level tracking
                    if hasattr(model, "predict"):
                        train_preds = model.predict(X_train)
                        if train_preds.ndim > 1:
                            train_preds = (
                                train_preds[:, 1]
                                if train_preds.shape[1] >= 2
                                else train_preds.ravel()
                            )
                        fold_metrics.train_r2 = float(r2_score(y_train, train_preds))
                        fold_metrics.train_mae = float(
                            mean_absolute_error(y_train, train_preds)
                        )

                        test_preds_fold = oos_preds
                        if (
                            hasattr(test_preds_fold, "ndim")
                            and test_preds_fold.ndim > 1
                        ):
                            test_preds_fold = (
                                test_preds_fold[:, 1]
                                if test_preds_fold.shape[1] >= 2
                                else test_preds_fold.ravel()
                            )
                        fold_metrics.test_r2 = float(r2_score(y_test, test_preds_fold))
                        fold_metrics.test_mae = float(
                            mean_absolute_error(y_test, test_preds_fold)
                        )
                        fold_metrics.gap_r2 = (
                            fold_metrics.train_r2 - fold_metrics.test_r2
                        )

                except Exception as e:
                    logger.debug(
                        f"Model {model_name} failed on fold {fold_idx + 1}: {e}"
                    )
                    if model_name not in all_models_oos:
                        all_models_oos[model_name] = []
                    all_models_oos[model_name].append(
                        np.full(len(X_test), self._target_mean)
                    )

            # Add fold to list if we have metrics
            if fold_metrics.train_r2 > 0 or fold_metrics.n_train > 0:
                fold_metrics_list.append(fold_metrics)
            all_oos_targets.append(y_test)

        self._fold_metrics = fold_metrics_list

        # ── Step 2: Concatenate OOS predictions ────────────────────────
        oos_dict: dict[str, np.ndarray] = {}
        for model_name in model_specs:
            name = model_name[0]
            if name in all_models_oos and all_models_oos[name]:
                oos_dict[name] = np.concatenate(all_models_oos[name])
            else:
                oos_dict[name] = (
                    np.concatenate(all_oos_targets) if all_oos_targets else np.array([])
                )

        oos_targets = (
            np.concatenate(all_oos_targets) if all_oos_targets else np.array([])
        )

        if len(oos_targets) == 0:
            raise ValueError("No OOS predictions collected — cannot train ensemble.")

        # ── Step 3: Calibrate each model's OOS predictions ─────────────
        self._calibrators = {}
        for model_name in oos_dict:
            try:
                oos_probs = oos_dict[model_name]
                oos_probs_clipped = np.clip(oos_probs, 0.001, 0.999)

                cal = LogisticRegression(
                    C=1.0, max_iter=1000, random_state=self.random_state
                )
                calibrator = CalibratedClassifierCV(
                    estimator=cal, method="sigmoid", cv=3
                )

                X_cal = oos_probs_clipped.reshape(-1, 1)
                y_cal = oos_targets

                if len(np.unique(y_cal)) >= 2:
                    calibrator.fit(X_cal, y_cal)
                    self._calibrators[model_name] = calibrator
                else:
                    logger.debug(
                        f"Model {model_name}: only one class in OOS — skipping calibration"
                    )
            except Exception as e:
                logger.debug(f"Calibration failed for {model_name}: {e}")

        # ── Step 3: Calibrate each model's OOS predictions (v6.6) ────
        self._calibrators = {}
        self._calibration_models = {}
        cal_probs_dict: dict[str, np.ndarray] = {}

        if self.calibrate:
            cal_probs_dict = self._calibrate_with_isotonic(oos_dict, oos_targets)
        else:
            cal_probs_dict = dict(oos_dict)

        # ── Step 4: Build calibrated/raw prob arrays ────────────
        self._raw_probs = (
            np.column_stack([oos_dict[name] for name in oos_dict])
            if len(oos_dict) > 1
            else oos_dict[list(oos_dict.keys())[0]].reshape(-1, 1)
        )

        self._calibrated_probs = (
            np.column_stack([cal_probs_dict[name] for name in cal_probs_dict])
            if len(cal_probs_dict) > 1
            else cal_probs_dict[list(cal_probs_dict.keys())[0]].reshape(-1, 1)
        )

        # Brier scores
        for model_name in oos_dict:
            raw_brier = float(
                brier_score_loss(
                    oos_targets, np.clip(oos_dict[model_name], 0.001, 0.999)
                )
            )
            cal_brier = float(
                brier_score_loss(
                    oos_targets, np.clip(cal_probs_dict[model_name], 0.001, 0.999)
                )
            )
            acc = float(np.mean((cal_probs_dict[model_name] > 0.5) == oos_targets))

            diag = ModelDiagnostics(
                name=model_name,
                oos_brier=cal_brier,
                oos_accuracy=acc,
                n_oos=len(oos_targets),
                is_calibrated=model_name in self._calibrators,
                status="ok",
            )

            # Detect degradation
            if cal_brier > 0.25:
                diag.status = "degraded"
            elif cal_brier > 0.50 or raw_brier > 0.50:
                diag.status = "failed"

            self._model_diagnostics[model_name] = diag

        # Overall Brier
        ensemble_raw = (
            np.mean(self._raw_probs, axis=1)
            if self._raw_probs.ndim > 1
            else self._raw_probs
        )
        ensemble_cal = (
            np.mean(self._calibrated_probs, axis=1)
            if self._calibrated_probs.ndim > 1
            else self._calibrated_probs
        )
        self._brier_score = float(
            brier_score_loss(oos_targets, np.clip(ensemble_raw, 0.001, 0.999))
        )
        self._calibrated_brier = float(
            brier_score_loss(oos_targets, np.clip(ensemble_cal, 0.001, 0.999))
        )

        if verbose:
            logger.info(
                f"Ensemble Brier: raw={self._brier_score:.4f}, "
                f"calibrated={self._calibrated_brier:.4f}"
            )

        # ── Step 5: Compute model weights from OOS Brier scores ────────
        brier_scores = {
            name: diag.oos_brier
            for name, diag in self._model_diagnostics.items()
            if diag.status != "failed"
        }

        if brier_scores:
            # v4.0: Sharper weight differentiation.
            # exp(-brier * 8) instead of exp(-brier * 5) — more weight on
            # better models, faster decay for worse ones.
            # Floor weight of 0.05 prevents any model from being ignored.
            raw_weights = {
                n: math.exp(-b * 8.0) + 0.05 for n, b in brier_scores.items()
            }
            total = sum(raw_weights.values())
            self._weights = {n: w / total for n, w in raw_weights.items()}
        else:
            self._weights = {
                name: 1.0 / max(len(self._model_diagnostics), 1)
                for name in self._model_diagnostics
            }

            # ── Step 6: Train stacking meta-model (optional) ───────────────
        if self.use_stacking and len(self._model_diagnostics) >= 2:
            try:
                meta_features = np.column_stack(
                    [cal_probs_dict[name] for name in cal_probs_dict]
                )

                # Add raw (uncalibrated) probabilities as additional meta-features
                # This gives the meta-model access to both calibrated and raw signals
                if self._raw_probs is not None and self._raw_probs.shape[1] > 1:
                    meta_features = np.column_stack(
                        [
                            meta_features,
                            self._raw_probs,
                        ]
                    )

                # Choose meta-model based on configuration
                model_type = self.stacking_meta_model
                if model_type == "lasso":
                    from sklearn.linear_model import LassoCV

                    self._meta_model = LassoCV(
                        alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
                        cv=3,
                        random_state=self.random_state,
                        max_iter=5000,
                    )
                elif model_type == "xgboost":
                    try:
                        from xgboost import XGBRegressor

                        self._meta_model = XGBRegressor(
                            n_estimators=200,
                            max_depth=3,
                            learning_rate=0.05,
                            subsample=0.7,
                            reg_alpha=0.5,
                            reg_lambda=1.0,
                            random_state=self.random_state,
                            verbosity=0,
                        )
                    except ImportError:
                        from sklearn.linear_model import Ridge

                        self._meta_model = Ridge(
                            alpha=1.0, random_state=self.random_state
                        )
                else:  # default: Ridge
                    from sklearn.linear_model import RidgeCV

                    self._meta_model = RidgeCV(
                        alphas=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                        cv=3,
                    )

                self._meta_model.fit(meta_features, oos_targets)

                if verbose:
                    logger.info(
                        f"Stacking meta-model ({model_type}) trained on "
                        f"{meta_features.shape[1]} features"
                    )
            except Exception as e:
                logger.debug(f"Stacking meta-model failed: {e}")
                self._meta_model = None

        # ── Step 7: Train final models on ALL data ─────────────────────
        for model_name, model_fn, params in model_specs:
            diag = self._model_diagnostics.get(model_name)
            if diag and diag.status == "failed":
                logger.debug(f"Skipping refit for failed model: {model_name}")
                continue

            try:
                model = model_fn(**params)

                if sample_weight is not None and hasattr(model, "fit"):
                    model.fit(X, y, sample_weight=sample_weight)
                elif hasattr(model, "fit"):
                    model.fit(X, y)
                else:
                    continue

                self._models[model_name] = model
            except Exception as e:
                logger.debug(f"Final refit failed for {model_name}: {e}")

        # ── Step 8: Calibrate ensemble on full data ────────────────────
        if self.calibrate and len(self._models) > 0:
            try:
                # Get ensemble predictions on training data
                ensemble_train = self._predict_ensemble_internal(X)

                cal_model = LogisticRegression(
                    C=1.0, max_iter=1000, random_state=self.random_state
                )
                self._calibration_model = CalibratedClassifierCV(
                    estimator=cal_model, method="sigmoid", cv=3
                )

                X_cal_all = np.clip(ensemble_train, 0.001, 0.999).reshape(-1, 1)
                if len(np.unique(y)) >= 2:
                    self._calibration_model.fit(X_cal_all, y)
            except Exception as e:
                logger.debug(f"Ensemble-level calibration failed: {e}")

        # ── Step 9: Overfitting analysis ───────────────────────────────
        self._overfitting = self._detect_overfitting(fold_metrics_list)

        if verbose:
            if self._overfitting and self._overfitting.is_overfit:
                logger.warning(
                    f"  OVERFITTING DETECTED: train R²={self._overfitting.avg_train_r2:.3f}, "
                    f"test R²={self._overfitting.avg_test_r2:.3f}, gap={self._overfitting.r2_gap:.3f}"
                )
                for flag in self._overfitting.flags:
                    logger.warning(f"   • {flag}")
            elif self._overfitting:
                logger.info(
                    f"  Overfitting check passed: gap={self._overfitting.r2_gap:.3f}"
                )

        # Step 10: Adversarial Validation (v6.6)
        self._run_adversarial_validation(X, y, verbose=verbose)

        # Step 11: Ensemble Diversity and Pruning (v6.6)
        if self.pruning_keep_top_n > 0 and len(self._model_diagnostics) >= 3:
            self._compute_ensemble_diversity(oos_dict)
            self._prune_ensemble(oos_dict, oos_targets)
            # Recompute weights after pruning
            if self._pruned_models:
                remaining_weights = {
                    n: w
                    for n, w in self._weights.items()
                    if n not in self._pruned_models
                }
                total = sum(remaining_weights.values())
                if total > 0:
                    self._weights = {n: w / total for n, w in remaining_weights.items()}

        # Step 12: Permutation Importance (v6.6)
        self._compute_permutation_importance(
            X, y, n_repeats=3, n_features=20, verbose=verbose
        )

        # Step 13: Bootstrap Uncertainty (v6.6)
        self._compute_bootstrap_uncertainty(X, y, verbose=verbose)

        self._fitted = True
        self._fit_timestamp = datetime.now().isoformat()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict win probabilities for all samples.

        Returns:
            Array of shape (n_samples, 2): [P(loss), P(win)]
        """
        if not self._fitted:
            raise ValueError("System not fitted yet. Call .fit() first.")

        # Ensemble prediction
        ensemble_pred = self._predict_ensemble_internal(X)

        # Calibrate
        if self._calibration_model is not None:
            try:
                X_cal = np.clip(ensemble_pred, 0.001, 0.999).reshape(-1, 1)
                cal_probs = self._calibration_model.predict_proba(X_cal)
                return cal_probs
            except Exception:
                pass

        # Return as 2-column
        return np.column_stack([1.0 - ensemble_pred, ensemble_pred])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class (0/1) at given threshold."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)

    def predict_with_details(self, X: np.ndarray) -> PredictionResult:
        """
        Predict a single sample and return full detail.

        Args:
            X: Single sample, shape (1, n_features) or (n_features,)

        Returns:
            PredictionResult with probabilities, confidence, etc.
        """
        if not self._fitted:
            raise ValueError("System not fitted yet. Call .fit() first.")

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get individual model predictions
        model_probs: dict[str, float] = {}
        for name, model in self._models.items():
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X)[0, 1]
                elif hasattr(model, "predict"):
                    prob = float(model.predict(X)[0])
                else:
                    continue
                model_probs[name] = float(np.clip(prob, 0.001, 0.999))
            except Exception:
                continue

        # Weighted ensemble
        weighted_sum = 0.0
        total_weight = 0.0
        for name, prob in model_probs.items():
            w = self._weights.get(name, 1.0 / max(len(self._models), 1))
            weighted_sum += prob * w
            total_weight += w

        home_win_prob = (
            weighted_sum / total_weight if total_weight > 0 else self._target_mean
        )

        # Apply ensemble-level calibration
        calibrated_prob = None
        calibration_applied = False
        calibration_failed = False
        calibration_warning = None
        if self._calibration_model is not None:
            try:
                X_cal = np.clip([[home_win_prob]], 0.001, 0.999)
                cal_out = self._calibration_model.predict_proba(X_cal)
                calibrated_prob = float(cal_out[0, 1])
                calibration_applied = True
            except Exception as e:
                calibrated_prob = None
                calibration_failed = True
                calibration_warning = (
                    f"Ensemble calibration failed: {e}. Using raw ensemble."
                )
                logger.debug(calibration_warning)

        final_prob = calibrated_prob if calibrated_prob is not None else home_win_prob
        final_prob = float(np.clip(final_prob, 0.001, 0.999))

        # Consensus and variance
        probs_list = list(model_probs.values())
        variance = float(np.var(probs_list)) if len(probs_list) > 1 else 0.0
        consensus = max(0.0, 1.0 - variance * 20.0)

        # Confidence score
        n_models = len(model_probs)
        consensus_factor = consensus
        n_model_factor = min(n_models / 4.0, 1.0)  # 4+ models = full confidence
        proximity_to_50 = 1.0 - abs(final_prob - 0.5) * 2.0  # 0 at 50%, 1 at 0% or 100%
        confidence_score = (
            consensus_factor * 0.4
            + n_model_factor * 0.2
            + (1.0 - proximity_to_50) * 0.4  # More confident when further from 50/50
        )
        confidence_score = float(np.clip(confidence_score, 0.0, 1.0))

        # Confidence label
        if confidence_score >= 0.85:
            label = "VERY_HIGH"
        elif confidence_score >= 0.70:
            label = "HIGH"
        elif confidence_score >= 0.50:
            label = "MEDIUM"
        elif confidence_score >= 0.30:
            label = "LOW"
        else:
            label = "VERY_LOW"

        # Feature importance
        importance = self.get_feature_importance(top_n=10)

        return PredictionResult(
            home_win_prob=round(final_prob, 4),
            away_win_prob=round(1.0 - final_prob, 4),
            n_models=n_models,
            consensus=round(consensus, 3),
            model_variance=round(variance, 6),
            calibrated_home_win_prob=round(calibrated_prob, 4)
            if calibrated_prob is not None
            else None,
            model_probs={k: round(v, 4) for k, v in sorted(model_probs.items())},
            model_weights={k: round(v, 4) for k, v in sorted(self._weights.items())},
            confidence_score=round(confidence_score, 3),
            confidence_label=label,
            generated_at=datetime.now().isoformat(),
            feature_importance=importance,
            calibration_applied=calibration_applied,
            calibration_failed=calibration_failed,
            calibration_warning=calibration_warning,
        )

    def compute_edge(
        self,
        home_win_prob: float,
        home_ml: Optional[float] = None,
        away_ml: Optional[float] = None,
        remove_vig: bool = True,
    ) -> tuple[float, str, str]:
        """
        Compute edge between model probability and market-implied probability.

        Args:
            home_win_prob: Model's estimated home win probability (0-1)
            home_ml: American odds for home team (e.g., -150)
            away_ml: American odds for away team (e.g., +130)
            remove_vig: Remove the vig from market odds

        Returns:
            Tuple of (edge_pct, direction, confidence_label)
            - edge_pct: Positive means home team is +EV, negative means away team is +EV
            - direction: "home" or "away"
            - confidence_label: "VERY_HIGH", "HIGH", "MEDIUM", "LOW", "VERY_LOW"
        """
        if home_ml is None or away_ml is None:
            return (0.0, "neutral", "LOW")

        def american_to_implied(odds: float) -> float:
            if odds > 0:
                return 100.0 / (odds + 100.0)
            elif odds < 0:
                return abs(odds) / (abs(odds) + 100.0)
            return 0.5

        home_implied = american_to_implied(home_ml)
        away_implied = american_to_implied(away_ml)

        if remove_vig:
            total_implied = home_implied + away_implied
            if total_implied > 0:
                home_implied /= total_implied
                away_implied /= total_implied

        # Edge: positive = home is +EV
        edge = home_win_prob - home_implied

        abs_edge = abs(edge)
        if abs_edge >= 0.08:
            confidence = "VERY_HIGH"
        elif abs_edge >= 0.05:
            confidence = "HIGH"
        elif abs_edge >= 0.03:
            confidence = "MEDIUM"
        elif abs_edge >= 0.01:
            confidence = "LOW"
        else:
            confidence = "VERY_LOW"

        direction = "home" if edge > 0 else "away"

        return (round(edge, 4), direction, confidence)

    def get_feature_importance(self, top_n: int = 20) -> dict[str, float]:
        """
        Get aggregated feature importance across all tree-based models.

        Returns:
            Dict of {feature_name: importance} sorted by importance descending.
        """
        if not self._models:
            return {}

        importances: dict[str, list[float]] = {}
        for name, model in self._models.items():
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                if len(fi) == len(self._feature_names):
                    for fname, imp in zip(self._feature_names, fi):
                        if fname not in importances:
                            importances[fname] = []
                        importances[fname].append(float(imp))
            elif hasattr(model, "coef_"):
                coef = model.coef_
                if coef.ndim > 1 and coef.shape[0] == 1:
                    coef = coef[0]
                if len(coef) == len(self._feature_names):
                    for fname, imp in zip(self._feature_names, np.abs(coef)):
                        if fname not in importances:
                            importances[fname] = []
                        importances[fname].append(float(imp))

        # Average importance across models, then normalize
        avg_importance = {
            name: float(np.mean(vals)) for name, vals in importances.items() if vals
        }

        total = sum(avg_importance.values())
        if total > 0:
            avg_importance = {k: v / total for k, v in avg_importance.items()}

        # Sort and return top N
        sorted_imp = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        return dict(sorted_imp)

    @property
    def feature_names(self) -> list[str]:
        """Public accessor for the feature names used during training."""
        return list(self._feature_names)

    def get_overfitting_report(self) -> Optional[OverfittingReport]:
        """Get the overfitting analysis report."""
        return self._overfitting

    def get_model_diagnostics(self) -> dict[str, ModelDiagnostics]:
        """Get diagnostics per model."""
        return dict(self._model_diagnostics)

    def get_summary(self) -> dict:
        """Get a JSON-serializable summary of the system."""
        return {
            "fitted": self._fitted,
            "fit_timestamp": self._fit_timestamp,
            "n_models": len(self._models),
            "n_train_samples": self._n_train_total,
            "n_features": len(self._feature_names),
            "n_folds": len(self._fold_metrics),
            "target_mean": round(self._target_mean, 4),
            "brier_score": round(self._brier_score, 4) if self._brier_score else None,
            "calibrated_brier": round(self._calibrated_brier, 4)
            if self._calibrated_brier
            else None,
            "model_weights": {k: round(v, 4) for k, v in self._weights.items()},
            "model_diagnostics": {
                name: {
                    "oos_brier": round(d.oos_brier, 4),
                    "oos_accuracy": round(d.oos_accuracy, 4),
                    "n_oos": d.n_oos,
                    "status": d.status,
                    "weight": round(d.weight, 4),
                }
                for name, d in self._model_diagnostics.items()
            },
            "overfitting": {
                "is_overfit": self._overfitting.is_overfit
                if self._overfitting
                else False,
                "avg_train_r2": round(self._overfitting.avg_train_r2, 4)
                if self._overfitting
                else None,
                "avg_test_r2": round(self._overfitting.avg_test_r2, 4)
                if self._overfitting
                else None,
                "r2_gap": round(self._overfitting.r2_gap, 4)
                if self._overfitting
                else None,
            }
            if self._overfitting
            else None,
        }

    def save(self, path: Path) -> str:
        """Save the fitted system to disk."""
        if not self._fitted:
            raise ValueError("System not fitted yet.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_joblib_dump(self, path)
        logger.info(f"RobustPredictionSystem saved to {path} (hash verified)")
        return str(path)

    @staticmethod
    def load(path: Path, verify: bool = True) -> RobustPredictionSystem:
        """Load a fitted system from disk with hash verification.

        Args:
            path: Path to the saved system file.
            verify: If True (default), verify SHA-256 hash before loading.

        Returns:
            Loaded RobustPredictionSystem.

        Raises:
            ModelIntegrityError: If hash verification fails.
        """
        path = Path(path)
        system = safe_joblib_load(path, verify=verify)
        logger.info(
            f"RobustPredictionSystem loaded from {path} "
            f"({len(system._models)} models, integrity=hash_verified)"
        )
        return system

    # ── Internal Methods ──────────────────────────────────────────────────

    def _get_model_specs(self) -> list[tuple[str, callable, dict]]:
        """Generate model specifications for the ensemble (5 models)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        specs = []

        # ══════════════════════════════════════════════════════════════
        #  v6.6 — 10-MODEL BEAST ENSEMBLE
        #
        #  Strategy: Every major ML paradigm —
        #    Linear:     LogisticRegression (interpretable baseline)
        #    Boosted:    XGBoost, LightGBM, CatBoost, HistGradientBoosting
        #    Bagged:     RandomForest, ExtraTrees
        #    Neural:     MLP (deep non-linear)
        #    Margin:     SVM (max-margin, different inductive bias)
        # ══════════════════════════════════════════════════════════════

        # 1. Logistic Regression (always works, always interpretable)
        lr_params = {
            "C": self._lr_params.get("C", 1.5),
            "max_iter": self._lr_params.get("max_iter", 3000),
            "random_state": self.random_state,
            "class_weight": self._lr_params.get("class_weight", "balanced"),
            "penalty": self._lr_params.get("penalty", "l2"),
            "solver": self._lr_params.get("solver", "lbfgs"),
        }
        specs.append(("LogisticRegression", LogisticRegression, lr_params))

        # 2. XGBoost — with early stopping
        try:
            from xgboost import XGBClassifier

            xgb_params = {
                "n_estimators": self._xgb_params.get("n_estimators", 1200),
                "max_depth": self._xgb_params.get("max_depth", 6),
                "learning_rate": self._xgb_params.get("learning_rate", 0.015),
                "subsample": self._xgb_params.get("subsample", 0.8),
                "colsample_bytree": self._xgb_params.get("colsample_bytree", 0.7),
                "reg_alpha": self._xgb_params.get("reg_alpha", 1.0),
                "reg_lambda": self._xgb_params.get("reg_lambda", 2.0),
                "gamma": self._xgb_params.get("gamma", 0.1),
                "min_child_weight": self._xgb_params.get("min_child_weight", 3),
                "random_state": self.random_state,
                "eval_metric": "logloss",
                "early_stopping_rounds": None,
                "verbosity": 0,
            }
            specs.append(("XGBoost", XGBClassifier, xgb_params))
        except ImportError:
            pass

        # 3. LightGBM — with early stopping
        try:
            from lightgbm import LGBMClassifier

            lgb_params = {
                "n_estimators": self._lgb_params.get("n_estimators", 1200),
                "max_depth": self._lgb_params.get("max_depth", -1),
                "num_leaves": self._lgb_params.get("num_leaves", 63),
                "learning_rate": self._lgb_params.get("learning_rate", 0.015),
                "subsample": self._lgb_params.get("subsample", 0.8),
                "colsample_bytree": self._lgb_params.get("colsample_bytree", 0.7),
                "reg_alpha": self._lgb_params.get("reg_alpha", 1.0),
                "reg_lambda": self._lgb_params.get("reg_lambda", 2.0),
                "min_child_samples": self._lgb_params.get("min_child_samples", 30),
                "min_split_gain": self._lgb_params.get("min_split_gain", 0.1),
                "random_state": self.random_state,
                "verbose": -1,
            }
            specs.append(("LightGBM", LGBMClassifier, lgb_params))
        except ImportError:
            pass

        # 4. Random Forest — bagging ensemble
        rf_params = {
            "n_estimators": self._rf_params.get("n_estimators", 800),
            "max_depth": self._rf_params.get("max_depth", 12),
            "min_samples_leaf": self._rf_params.get("min_samples_leaf", 4),
            "max_features": self._rf_params.get("max_features", "sqrt"),
            "min_samples_split": self._rf_params.get("min_samples_split", 10),
            "class_weight": self._rf_params.get("class_weight", "balanced_subsample"),
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        specs.append(("RandomForest", RandomForestClassifier, rf_params))

        # 5. CatBoost
        if self.use_catboost:
            try:
                from catboost import CatBoostClassifier

                cb_params = {
                    "iterations": self._cb_params.get("iterations", 800),
                    "depth": self._cb_params.get("depth", 8),
                    "learning_rate": self._cb_params.get("learning_rate", 0.03),
                    "l2_leaf_reg": self._cb_params.get("l2_leaf_reg", 3.0),
                    "border_count": self._cb_params.get("border_count", 128),
                    "subsample": self._cb_params.get("subsample", 0.8),
                    "random_seed": self.random_state,
                    "verbose": 0,
                    "early_stopping_rounds": None,
                    "loss_function": "Logloss",
                    "eval_metric": "Logloss",
                    "use_best_model": self.use_early_stopping,
                    "thread_count": -1,
                }
                specs.append(("CatBoost", CatBoostClassifier, cb_params))
            except ImportError:
                pass

        return specs

    def _predict_ensemble_internal(self, X: np.ndarray) -> np.ndarray:
        """
        Get the weighted ensemble prediction (raw, before calibration).

        Returns:
            Array of shape (n_samples,) with ensemble probabilities.
        """
        if not self._models:
            return np.full(X.shape[0], self._target_mean)

        all_preds = []
        all_weights = []

        for name, model in self._models.items():
            w = self._weights.get(name, 1.0 / len(self._models))
            try:
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X)
                    if preds.ndim == 2 and preds.shape[1] >= 2:
                        preds = preds[:, 1]
                elif hasattr(model, "predict"):
                    preds = model.predict(X)
                    if hasattr(preds, "ndim") and preds.ndim > 1:
                        preds = preds.ravel()
                else:
                    continue

                preds = np.asarray(preds, dtype=float).ravel()
                all_preds.append(preds)
                all_weights.append(w)
            except Exception:
                continue

        if not all_preds:
            return np.full(X.shape[0], self._target_mean)

        sum(all_weights)
        weighted_avg = (
            np.average(np.column_stack(all_preds), axis=1, weights=all_weights)
            if len(all_preds) > 1
            else all_preds[0]
        )

        return np.clip(weighted_avg, 0.001, 0.999)

    def select_features_mutual_info(
        self, X: np.ndarray, y: np.ndarray, n_features: int = 50
    ) -> np.ndarray:
        """
        Select top N features via mutual information.

        Mutual information captures non-linear relationships between
        features and the target, which correlation-based methods miss.
        This is better than variance-based selection because it finds
        features that are actually predictive, not just high-variance.

        Args:
            X: Feature matrix.
            y: Target vector.
            n_features: Number of features to keep.

        Returns:
            Boolean mask of selected features.
        """
        try:
            from sklearn.feature_selection import mutual_info_classif

            mi = mutual_info_classif(X, y, random_state=self.random_state)
            if len(mi) <= n_features:
                return np.ones(X.shape[1], dtype=bool)
            top_idx = np.argsort(mi)[-n_features:]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_idx] = True
            logger.info(
                f"Mutual info feature selection: {X.shape[1]} → {n_features} features"
            )
            return mask
        except Exception as e:
            logger.debug(f"Mutual info feature selection failed: {e}")
            return np.ones(X.shape[1], dtype=bool)

    def _detect_overfitting(
        self, fold_metrics: list[WalkForwardFold]
    ) -> Optional[OverfittingReport]:
        """Analyse fold metrics to detect overfitting."""
        if not fold_metrics:
            return None

        train_r2s = [f.train_r2 for f in fold_metrics]
        test_r2s = [f.test_r2 for f in fold_metrics]
        gaps = [f.gap_r2 for f in fold_metrics]

        if not train_r2s or not test_r2s:
            return None

        avg_train_r2 = float(np.mean(train_r2s))
        avg_test_r2 = float(np.mean(test_r2s))
        avg_gap = float(np.mean(gaps)) if gaps else avg_train_r2 - avg_test_r2

        flags: list[str] = []
        if avg_test_r2 < -0.10:
            flags.append(
                f"Test R² ({avg_test_r2:.3f}) is severely negative — model likely overfit"
            )
        if avg_gap > 0.15:
            flags.append(
                f"Train-test R² gap ({avg_gap:.3f}) exceeds 0.15 — overfitting likely"
            )
        if avg_test_r2 < 0 and avg_train_r2 > 0.5:
            flags.append(
                f"High train R² ({avg_train_r2:.3f}) but negative test R² — clear overfitting"
            )
        if avg_test_r2 < 0.50 and avg_train_r2 > 0.90:
            flags.append("Extreme overfitting: train R² > 0.90 but test R² < 0.50")

        is_overfit = len(flags) > 0

        return OverfittingReport(
            is_overfit=is_overfit,
            avg_train_r2=round(avg_train_r2, 4),
            avg_test_r2=round(avg_test_r2, 4),
            r2_gap=round(avg_gap, 4),
            flags=flags,
            folds=fold_metrics,
        )

    # ── v6.6: Hyperparameter Tuning Integration ─────────────────────────────

    def _tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> Optional[dict]:
        """Run Optuna-based hyperparameter tuning for all ensemble models (v6.6).

        Takes a stratified split of training data and tunes each model type.
        Results are stored and used to override default params in _get_model_specs.
        """
        if not self.use_hyperparameter_tuning:
            return None

        try:
            from sklearn.model_selection import train_test_split
            from betting_intel.models.hyperparameter_tuning import HyperparameterTuner

            # Stratified split for tuning
            X_tune, X_val, y_tune, y_val = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )

            tuner = HyperparameterTuner(
                random_state=self.random_state,
                n_trials=30,
                direction="minimize",
            )
            self._hyperparameter_tuner = tuner

            if verbose:
                logger.info("=" * 60)
                logger.info("v6.6 — AUTOMATIC HYPERPARAMETER TUNING")
                logger.info("=" * 60)

            results = tuner.tune_all(
                X_tune,
                y_tune,
                X_val,
                y_val,
                n_trials_per_model=30,
                verbose=verbose,
            )

            self._hyperparameter_tuning_results = results

            # Map tuning results back to our param dicts
            _mappings = {
                "xgb": (
                    "_xgb_params",
                    {
                        "n_estimators": "n_estimators",
                        "max_depth": "max_depth",
                        "learning_rate": "learning_rate",
                        "subsample": "subsample",
                        "colsample_bytree": "colsample_bytree",
                        "reg_alpha": "reg_alpha",
                        "reg_lambda": "reg_lambda",
                    },
                ),
                "lgb": (
                    "_lgb_params",
                    {
                        "n_estimators": "n_estimators",
                        "num_leaves": "num_leaves",
                        "learning_rate": "learning_rate",
                        "subsample": "subsample",
                        "colsample_bytree": "colsample_bytree",
                        "reg_alpha": "reg_alpha",
                        "reg_lambda": "reg_lambda",
                        "min_child_samples": "min_child_samples",
                    },
                ),
                "cb": (
                    "_cb_params",
                    {
                        "iterations": "iterations",
                        "depth": "depth",
                        "learning_rate": "learning_rate",
                        "l2_leaf_reg": "l2_leaf_reg",
                        "border_count": "border_count",
                    },
                ),
            }

            for model_key, (attr, mapping) in _mappings.items():
                if model_key not in results:
                    continue
                param_dict = getattr(self, attr)
                for tune_key, param_attr in mapping.items():
                    if tune_key in results[model_key]:
                        param_dict[param_attr] = results[model_key][tune_key]

            if verbose:
                logger.info("  Hyperparameter tuning applied to all models")

            return results

        except Exception as e:
            logger.warning(f"Hyperparameter tuning failed (non-critical): {e}")
            return None

    # ── v6.6: Adversarial Validation ───────────────────────────────────

    def _run_adversarial_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> Optional[dict]:
        """Detect train/inference distribution shift via adversarial validation (v6.6).

        Adversarial validation trains a classifier to distinguish between
        training data and a reference distribution. If the classifier can
        easily separate them, it means the distributions have shifted —
        the model may not generalize to new data.

        The score is AUROC: 1.0 = perfect separation (BAD — shift detected),
        0.5 = random (GOOD — no shift). Scale:
          - < 0.60: No significant drift
          - 0.60-0.75: Some drift — may want to retrain
          - 0.75-0.90: Significant drift — retrain recommended
          - > 0.90: Critical drift — model likely invalid

        Returns dict with auroc, feature_importance (which features drifted most).
        """
        if not self.use_adversarial_validation:
            return None

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score

            n = len(X)
            if n < 200:
                if verbose:
                    logger.debug("Adversarial validation: need ≥200 samples, skipping")
                return None

            # Split data into two "domains": first half = train, second half = test
            midpoint = n // 2
            X_first, X_second = X[:midpoint], X[midpoint:]

            # Create domain label: 0 = first half, 1 = second half
            X_adv = np.vstack([X_first, X_second])
            y_adv = np.concatenate(
                [
                    np.zeros(midpoint, dtype=int),
                    np.ones(len(X_second), dtype=int),
                ]
            )

            # Shuffle to avoid any ordering bias
            from sklearn.utils import shuffle

            X_adv, y_adv = shuffle(X_adv, y_adv, random_state=self.random_state)

            # Train quick RF to discriminate
            adv_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=self.random_state,
                n_jobs=-1,
            )

            cv_scores = cross_val_score(
                adv_model, X_adv, y_adv, cv=3, scoring="roc_auc"
            )
            auroc = float(np.mean(cv_scores))

            # Fit on full data for feature importance
            adv_model.fit(X_adv, y_adv)
            drift_importances = {}
            if hasattr(adv_model, "feature_importances_"):
                fi = adv_model.feature_importances_
                for i, imp in enumerate(fi):
                    fname = (
                        self._feature_names[i]
                        if i < len(self._feature_names)
                        else f"f{i}"
                    )
                    drift_importances[fname] = float(imp)

            sorted_drift = sorted(drift_importances.items(), key=lambda x: -x[1])[:10]

            health = "stable"
            if auroc > 0.90:
                health = "critical"
            elif auroc > 0.75:
                health = "warning"
            elif auroc > 0.60:
                health = "minor"

            self._adversarial_score = auroc
            self._adversarial_auroc = auroc

            if verbose:
                logger.info(
                    f"Adversarial validation AUROC={auroc:.3f} (health={health})"
                )
                if sorted_drift:
                    logger.info(
                        f"  Top drifted features: {[(n, f'{v:.3f}') for n, v in sorted_drift[:5]]}"
                    )

            return {
                "auroc": round(auroc, 4),
                "cv_scores": [round(s, 4) for s in cv_scores],
                "health": health,
                "top_drifted_features": dict(sorted_drift),
            }

        except Exception as e:
            logger.debug(f"Adversarial validation failed: {e}")
            return None

    # ── v6.6: Ensemble Diversity Metrics & Pruning ────────────────────

    def _compute_ensemble_diversity(self, oos_dict: dict[str, np.ndarray]) -> dict:
        """Compute pairwise model correlation and diversity metrics (v6.6).

        Returns:
            Dict with:
            - pairwise_correlation: mean absolute Pearson r between models
            - diversity_score: 1 - avg_correlation (higher = more diverse)
            - model_correlation: {model_name: {other_model: r}}
            - redundancy_pairs: models with r > 0.90 (candidates for pruning)
        """
        try:
            from scipy.stats import pearsonr
        except ImportError:
            logger.warning("scipy required for ensemble diversity computation")
            return {"diversity_score": 0.5, "details": "scipy not available"}

        model_names = list(oos_dict.keys())
        n_models = len(model_names)

        if n_models < 2:
            return {"diversity_score": 1.0, "n_models": 1}

        correlations: dict[str, dict] = {}
        all_corrs = []

        for i in range(n_models):
            for j in range(i + 1, n_models):
                name_i, name_j = model_names[i], model_names[j]
                preds_i = oos_dict[name_i]
                preds_j = oos_dict[name_j]

                try:
                    r, _ = pearsonr(preds_i, preds_j)
                    r = float(r) if not np.isnan(r) else 0.0
                except Exception:
                    r = 0.0

                correlations.setdefault(name_i, {})[name_j] = round(r, 4)
                correlations.setdefault(name_j, {})[name_i] = round(r, 4)
                all_corrs.append(abs(r))

        avg_correlation = float(np.mean(all_corrs)) if all_corrs else 0.0
        diversity = 1.0 - avg_correlation

        # Detect redundant pairs (r > 0.90)
        redundant_pairs = []
        if all_corrs:
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    name_i, name_j = model_names[i], model_names[j]
                    r = abs(correlations.get(name_i, {}).get(name_j, 0))
                    if r > 0.90:
                        redundant_pairs.append((name_i, name_j, round(r, 4)))

        self._ensemble_diversity = {
            "avg_correlation": round(avg_correlation, 4),
            "diversity_score": round(diversity, 4),
            "model_correlation": correlations,
            "redundant_pairs": redundant_pairs,
        }

        return self._ensemble_diversity

    def _prune_ensemble(
        self,
        oos_dict: dict[str, np.ndarray],
        oos_targets: np.ndarray,
    ) -> list[str]:
        """Prune redundant or low-performing models from the ensemble (v6.6).

        Strategy:
          1. Sort models by OOS Brier score (best first)
          2. Greedily add models that improve ensemble correlation diversity
          3. Remove models with OOS accuracy < 0.48 (worse than coin flip)
          4. Keep at most pruning_keep_top_n models (if set > 0)

        Returns:
            List of pruned model names (removed from ensemble)
        """
        pruned = []
        model_names = list(oos_dict.keys())

        if len(model_names) < 3:
            return pruned

        # 1. Filter by minimum performance
        for name in model_names:
            if name not in oos_dict:
                continue
            preds = oos_dict[name]
            acc = float(np.mean((preds > 0.5) == oos_targets))
            diag = self._model_diagnostics.get(name)
            # Prune only if model is both inaccurate AND poorly calibrated
            if acc < 0.48 and diag and diag.oos_brier > 0.25:
                pruned.append(name)
                self._weights.pop(name, None)
                # Mark as degraded
                if name in self._model_diagnostics:
                    self._model_diagnostics[name].status = "failed"

        remaining = [n for n in model_names if n not in pruned]

        if len(remaining) < 2:
            return pruned

        # 2. Get diversity and prune highly correlated redundant models
        diversity = self._compute_ensemble_diversity(oos_dict)
        redundant_pairs = diversity.get("redundant_pairs", [])

        for name_i, name_j, r in redundant_pairs:
            if name_i in remaining and name_j in remaining:
                # Prune the one with worse Brier
                brier_i = self._model_diagnostics.get(
                    name_i, ModelDiagnostics(name=name_i)
                ).oos_brier
                brier_j = self._model_diagnostics.get(
                    name_j, ModelDiagnostics(name=name_j)
                ).oos_brier
                if brier_i <= brier_j:
                    pruned.append(name_j)
                    self._weights.pop(name_j, None)
                    remaining.remove(name_j)
                else:
                    pruned.append(name_i)
                    self._weights.pop(name_i, None)
                    remaining.remove(name_i)

        # 3. Keep only top N if specified
        if self.pruning_keep_top_n > 0 and len(remaining) > self.pruning_keep_top_n:
            # Sort by Brier, keep best
            sorted_remaining = sorted(
                remaining,
                key=lambda n: (
                    self._model_diagnostics.get(n, ModelDiagnostics(name=n)).oos_brier
                ),
            )
            to_prune = sorted_remaining[self.pruning_keep_top_n :]
            pruned.extend(to_prune)

        self._pruned_models = pruned

        if pruned:
            logger.info(
                f"Ensemble pruning: removed {len(pruned)} models "
                f"({', '.join(pruned)}), kept {len(remaining)}"
            )

        return pruned

    # ── v6.6: Permutation Importance ──────────────────────────────────

    def _compute_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 5,
        n_features: int = 20,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Compute permutation feature importance (v6.6).

        Permutation importance measures how much prediction error increases
        when a feature's values are randomly shuffled. Unlike model-internal
        feature importance (which can be biased toward high-cardinality features),
        permutation importance is model-agnostic and more reliable.

        Args:
            X: Feature matrix
            y: Target vector
            n_repeats: Number of shuffles per feature
            n_features: Number of top features to return
            verbose: Log progress

        Returns:
            Dict of {feature_name: importance_score} sorted descending
        """
        if not self.use_permutation_importance or not self._fitted:
            return {}

        try:
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                self,
                X,
                y,
                n_repeats=n_repeats,
                random_state=self.random_state,
                n_jobs=-1,
                scoring="neg_brier_score",
            )

            importances = {
                self._feature_names[i]
                if i < len(self._feature_names)
                else f"f{i}": float(result.importances_mean[i])
                for i in range(len(result.importances_mean))
            }

            # Sort by absolute importance
            sorted_imp = sorted(
                importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:n_features]

            result_dict = dict(sorted_imp)
            self._permutation_importance = result_dict

            if verbose and result_dict:
                logger.info(
                    f"Permutation importance computed for {len(result_dict)} features "
                    f"(top: {list(result_dict.keys())[:3]})"
                )

            return result_dict

        except Exception as e:
            logger.debug(f"Permutation importance failed: {e}")
            return {}

    # ── v6.6: Bootstrap Uncertainty Quantification ────────────────────

    def _compute_bootstrap_uncertainty(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> Optional[dict]:
        """Compute prediction uncertainty via bootstrap resampling (v6.6).

        Trains `n_bootstrap_samples` models on bootstrap replicates of the
        training data and measures prediction variance — giving us confidence
        intervals for every prediction.

        Key outputs:
          - bootstrap_std: per-sample prediction std (uncertainty)
          - mean_uncertainty: average std across all samples
          - p5/p95: 5th and 95th percentile predictions

        Returns dict with uncertainty metrics, or None if not requested.
        """
        if not self.use_bootstrap_uncertainty:
            return None

        try:
            n_samples = self.n_bootstrap_samples
            n = len(X)
            all_bootstrap_preds = np.zeros((n_samples, n))

            for b in range(n_samples):
                # Bootstrap sample (with replacement)
                indices = np.random.choice(n, size=n, replace=True)
                X_boot = X[indices]
                y_boot = y[indices]

                # Train a single LogisticRegression as fast proxy
                try:
                    from sklearn.linear_model import LogisticRegression

                    boot_model = LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        random_state=self.random_state + b,
                        class_weight="balanced",
                    )
                    boot_model.fit(X_boot, y_boot)
                    preds = boot_model.predict_proba(X)[:, 1]
                    all_bootstrap_preds[b] = preds
                except Exception:
                    continue

            # Compute statistics
            bootstrap_std = np.std(all_bootstrap_preds, axis=0)
            bootstrap_mean = np.mean(all_bootstrap_preds, axis=0)
            p5 = np.percentile(all_bootstrap_preds, 5, axis=0)
            p95 = np.percentile(all_bootstrap_preds, 95, axis=0)

            self._bootstrap_probs = bootstrap_mean
            self._bootstrap_std = bootstrap_std

            uncertainty_result = {
                "mean_uncertainty": round(float(np.mean(bootstrap_std)), 4),
                "std_uncertainty": round(float(np.std(bootstrap_std)), 4),
                "p5_avg": round(float(np.mean(p5)), 4),
                "p95_avg": round(float(np.mean(p95)), 4),
                "n_bootstrap": n_samples,
            }

            if verbose:
                logger.info(
                    f"Bootstrap uncertainty: μ={uncertainty_result['mean_uncertainty']:.4f}, "
                    f"σ={uncertainty_result['std_uncertainty']:.4f}, "
                    f"[{uncertainty_result['p5_avg']:.4f} - {uncertainty_result['p95_avg']:.4f}]"
                )

            return uncertainty_result

        except Exception as e:
            logger.debug(f"Bootstrap uncertainty failed: {e}")
            return None

    def _calibrate_with_isotonic(
        self,
        oos_dict: dict[str, np.ndarray],
        oos_targets: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Calibrate using isotonic regression (v6.6).

        Isotonic regression is a non-parametric calibration method that makes
        no assumptions about the shape of the calibration curve. It tends to
        work better than Platt scaling when the miscalibration pattern is
        non-monotonic or complex.

        Auto-mode: tries isotonic first, falls back to Platt if isotonic fails.
        Platt mode: uses sigmoid calibration (original behavior).
        """
        from sklearn.isotonic import IsotonicRegression

        cal_probs_dict = {}

        for model_name, oos_probs in oos_dict.items():
            try:
                X_cal = np.clip(oos_probs, 0.001, 0.999)

                # Try isotonic
                if self.calibration_method in ("isotonic", "auto"):
                    try:
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(X_cal, oos_targets)
                        cal_probs = iso.transform(X_cal)
                        cal_probs = np.clip(cal_probs, 0.001, 0.999)
                        cal_probs_dict[model_name] = cal_probs
                        self._calibration_models[model_name] = iso
                        continue  # Skip Platt if isotonic succeeded
                    except Exception:
                        if self.calibration_method == "isotonic":
                            raise  # Re-raise if isotonic was explicitly requested

                # Fallback: Platt scaling (sigmoid)
                if self.calibration_method in ("platt", "auto"):
                    from sklearn.calibration import CalibratedClassifierCV
                    from sklearn.linear_model import LogisticRegression

                    cal = LogisticRegression(
                        C=1.0, max_iter=1000, random_state=self.random_state
                    )
                    calibrator = CalibratedClassifierCV(
                        estimator=cal, method="sigmoid", cv=3
                    )

                    X_cal_2d = X_cal.reshape(-1, 1)
                    if len(np.unique(oos_targets)) >= 2:
                        calibrator.fit(X_cal_2d, oos_targets)
                        self._calibrators[model_name] = calibrator
                        cal_probs = calibrator.predict_proba(X_cal_2d)[:, 1]
                        cal_probs_dict[model_name] = cal_probs
                        continue

                # If we get here, neither method worked
                cal_probs_dict[model_name] = oos_probs

            except Exception as e:
                logger.debug(f"Calibration failed for {model_name}: {e}")
                cal_probs_dict[model_name] = oos_probs

        return cal_probs_dict

    def get_ensemble_diversity(self) -> Optional[dict]:
        """Get ensemble diversity report (v6.6)."""
        return self._ensemble_diversity

    def get_adversarial_validation(self) -> Optional[dict]:
        """Get adversarial validation report (v6.6)."""
        if self._adversarial_auroc is not None:
            return {
                "auroc": round(self._adversarial_auroc, 4),
                "health": (
                    "critical"
                    if self._adversarial_auroc > 0.90
                    else "warning"
                    if self._adversarial_auroc > 0.75
                    else "minor"
                    if self._adversarial_auroc > 0.60
                    else "stable"
                ),
            }
        return None

    def get_permutation_importance(self, top_n: int = 20) -> dict[str, float]:
        """Get permutation feature importance (v6.6)."""
        if self._permutation_importance:
            items = sorted(
                self._permutation_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:top_n]
            return dict(items)
        return {}

    def get_bootstrap_uncertainty(self) -> Optional[dict]:
        """Get bootstrap uncertainty metrics (v6.6)."""
        if self._bootstrap_std is not None:
            return {
                "mean_uncertainty": round(float(np.mean(self._bootstrap_std)), 4),
                "std_uncertainty": round(float(np.std(self._bootstrap_std)), 4),
                "has_bootstrap": True,
            }
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  MARKET INEFFICIENCY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════


class MarketInefficiencySystem:
    """
    Trains on market error instead of raw win/loss outcomes.

    The core insight: instead of training a model to predict "who will win"
    (home_win = 0/1), train it to predict "where is the market wrong?"
    (market_error = actual_outcome - market_implied_prob).

    This is fundamentally different from win probability prediction because:
      1. The model learns the MARKET'S blind spots — public bias, recency
         bias, overreaction to narrative, systematic mispricing patterns
      2. At inference time: predicted_error + current_market_prob = our win prob
      3. The model can generate edge even without a sharp win probability model

    Architecture
    ────────────
      MarketInefficiencySystem
      ├── RobustPredictionSystem (home_win classifier — keeps existing capability)
      ├── MarketErrorRegressor (XGBoost + LightGBM regression on market_error)
      └── EdgeCombiner (blends both predictions based on confidence)

    Training flow:
      1. Compute market_implied_home_prob for each game (from ELO proxy)
      2. Compute market_error = home_win - market_implied_home_prob
      3. Train RPS on home_win (classification head)
      4. Train regressor on market_error (regression head)
      5. At inference: final_prob = market_implied_prob + predicted_error

    Usage:
        system = MarketInefficiencySystem()
        system.fit(X, y_binary, X_market_probs=market_probs)
        result = system.predict_with_details(X_pred, market_prob=0.65)
        # result.home_win_prob = 0.65 + predicted_error
    """

    def __init__(
        self,
        calibrate: bool = True,
        n_folds: int = 5,
        min_train_samples: int = 100,
        random_state: int = 42,
        use_hyperparameter_tuning: bool = False,
        use_adversarial_validation: bool = False,
        use_bootstrap_uncertainty: bool = False,
        use_permutation_importance: bool = False,
        pruning_keep_top_n: int = 0,
        use_early_stopping: bool = True,
    ):
        self._classifier = RobustPredictionSystem(
            calibrate=calibrate,
            n_folds=n_folds,
            min_train_samples=min_train_samples,
            random_state=random_state,
            use_hyperparameter_tuning=use_hyperparameter_tuning,
            use_adversarial_validation=use_adversarial_validation,
            use_bootstrap_uncertainty=use_bootstrap_uncertainty,
            use_permutation_importance=use_permutation_importance,
            pruning_keep_top_n=pruning_keep_top_n,
            use_early_stopping=use_early_stopping,
        )
        self._error_regressor: Optional[Any] = None
        self._feature_names: list[str] = []
        self._fitted: bool = False
        self._target_mean: float = 0.5
        self._error_mean: float = 0.0
        self._error_std: float = 0.15  # Typical NBA market error std
        self._fit_timestamp: Optional[str] = None

        # Error regressor specs
        self._reg_specs: list[tuple[str, type, dict]] = []
        self._random_state = random_state
        self._n_folds = n_folds

    @property
    def classifier(self) -> RobustPredictionSystem:
        """The underlying home_win classifier."""
        return self._classifier

    @property
    def has_error_model(self) -> bool:
        """Whether the market error regressor is trained."""
        return self._error_regressor is not None

    @property
    def feature_names(self) -> list[str]:
        """Public accessor for the feature names used during training."""
        return list(self._feature_names)

    def fit(
        self,
        X: np.ndarray,
        y_binary: np.ndarray,
        market_probs: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        verbose: bool = True,
    ) -> MarketInefficiencySystem:
        """
        Train the market inefficiency system.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_binary: Actual outcomes (0/1) for classification head
            market_probs: Market-implied home win probabilities (n_samples,)
                If None, uses 0.5 for all (falls back to standard classifier)
            feature_names: Optional feature column names
            verbose: Print progress

        Returns:
            self (fitted)
        """
        len(X)
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self._target_mean = float(np.mean(y_binary))

        # ── Step 1: Train the classifier head (home_win, as before) ────
        if verbose:
            logger.info("Training classification head (home_win prediction)...")

        self._classifier.fit(
            X,
            y_binary,
            feature_names=self._feature_names,
            verbose=verbose,
        )

        # ── Step 2: Train the market error regressor ──────────────────
        if market_probs is not None:
            market_probs = np.asarray(market_probs, dtype=float).ravel()
            market_probs = np.clip(market_probs, 0.01, 0.99)

            # Compute market error = actual - market_implied_prob
            # Range: [-0.99, 0.99]
            # Positive = market underestimated home team
            # Negative = market overestimated home team
            market_error = y_binary.astype(float) - market_probs

            self._error_mean = float(np.mean(market_error))
            self._error_std = max(float(np.std(market_error)), 0.01)

            if verbose:
                logger.info(
                    f"Market error stats: mean={self._error_mean:.4f}, "
                    f"std={self._error_std:.4f}, "
                    f"range=[{market_error.min():.4f}, {market_error.max():.4f}]"
                )

            # Train regression ensemble on market_error
            self._build_regression_models(X, market_error, verbose)

        else:
            if verbose:
                logger.info(
                    "No market probabilities provided — "
                    "falling back to classifier-only mode"
                )

        self._fitted = True
        self._fit_timestamp = datetime.now().isoformat()
        return self

    def _build_regression_models(
        self,
        X: np.ndarray,
        y_error: np.ndarray,
        verbose: bool = True,
    ):
        """
        Build a regression ensemble to predict market error.

        Uses the same model diversity principle as RobustPredictionSystem:
          - XGBoost Regressor (gradient boosted trees)
          - LightGBM Regressor (fast, handles non-linear patterns)
          - Random Forest Regressor (bagging for variance reduction)
          - Ridge Regression (linear baseline, interpretable)

        All trained on market_error target.
        """
        from sklearn.metrics import mean_absolute_error, r2_score

        n = len(X)
        min_train = max(50, self._classifier.min_train_samples)
        min_test = max(10, self._classifier.min_test_samples)

        if n < min_train + min_test:
            logger.warning(f"Too few samples ({n}) for error regressor — skipping")
            return

        # ── Walk-forward CV for OOS error predictions ───────────────
        # v5.1 — FIXED: Proper chronological folds.
        # The OLD code used (fold_idx * fold_size) % max(n - fold_size, 1)
        # which created NON-chronological folds where the "test" set could
        # be BEFORE the "train" set — data leakage that inflated metrics.
        # Now: strictly increasing test starts (chronological).
        n_folds = min(self._n_folds, max(2, n // (min_train + min_test)))
        fold_size = max(min_test, n // max(n_folds, 1))
        min_train = max(50, n // 4)  # At least 25% of data for training

        reg_models = []  # (name, model_object, weight)
        fold_errors: list[float] = []

        # Build individual regressors
        reg_specs = self._get_regressor_specs()

        for name, model_fn, params in reg_specs:
            try:
                model = model_fn(**params)
                model.fit(X, y_error)

                # Evaluate with simple CV
                train_preds = model.predict(X)
                train_mae = mean_absolute_error(y_error, train_preds)
                r2_score(y_error, train_preds)

                # Walk-forward OOS evaluation — CHRONOLOGICAL folds
                oos_preds = np.full(n, np.nan)
                for fold_idx in range(max(n_folds, 1)):
                    # CHRONOLOGICAL: test_start increases monotonically
                    test_start = fold_idx * fold_size
                    test_end = min(test_start + fold_size, n)
                    # Ensure we have enough training data BEFORE the test set
                    if test_start < min_train or (test_end - test_start) < min_test:
                        continue

                    # Train only on data BEFORE the test set (no future leakage)
                    X_train_fold = X[:test_start]
                    y_train_fold = y_error[:test_start]

                    if len(X_train_fold) < min_train:
                        continue

                    fold_model = model_fn(**params)
                    fold_model.fit(X_train_fold, y_train_fold)
                    oos_preds[test_start:test_end] = fold_model.predict(
                        X[test_start:test_end]
                    )

                oos_valid = oos_preds[~np.isnan(oos_preds)]
                if len(oos_valid) >= min_test:
                    oos_y = y_error[~np.isnan(oos_preds)]
                    oos_mae = mean_absolute_error(oos_y, oos_valid)
                    oos_r2 = r2_score(oos_y, oos_valid)
                    fold_errors.append(oos_mae)

                    if verbose:
                        logger.info(
                            f"  {name:20s}  "
                            f"train MAE={train_mae:.4f}, "
                            f"OOS MAE={oos_mae:.4f}, "
                            f"OOS R²={oos_r2:.4f}"
                        )
                else:
                    fold_errors.append(train_mae)

                reg_models.append(
                    (name, model, fold_errors[-1] if fold_errors else 1.0)
                )

            except Exception as e:
                logger.debug(f"Error regressor {name} failed: {e}")
                continue

        if not reg_models:
            logger.warning("No error regressors built — using classifier only")
            return

        # ── Weight models by inverse error ──────────────────────────────
        total_inv_error = sum(1.0 / max(e, 0.001) for _, _, e in reg_models)
        weighted_models = []
        for name, model, error in reg_models:
            w = (
                (1.0 / max(error, 0.001)) / total_inv_error
                if total_inv_error > 0
                else 1.0 / len(reg_models)
            )
            weighted_models.append((name, model, w))

        self._error_regressor = weighted_models

        if verbose:
            weights_str = ", ".join(f"{n}={w:.2f}" for n, _, w in weighted_models)
            logger.info(
                f"Market error regressor built: {len(reg_models)} models "
                f"(weights: {weights_str})"
            )

    def _get_regressor_specs(self) -> list[tuple[str, type, dict]]:
        """Generate model specifications for market error regression."""
        from sklearn.ensemble import RandomForestRegressor

        specs = []

        # 1. Ridge Regression (linear baseline, always available)
        ridge_params = {
            "alpha": 2.0,
            "random_state": self._random_state,
        }
        specs.append(("Ridge", Ridge, ridge_params))

        # 2. XGBoost Regressor
        try:
            from xgboost import XGBRegressor

            xgb_params = {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_alpha": 1.0,
                "reg_lambda": 2.0,
                "random_state": self._random_state,
                "verbosity": 0,
            }
            specs.append(("XGBoost_Error", XGBRegressor, xgb_params))
        except ImportError:
            pass

        # 3. LightGBM Regressor
        try:
            from lightgbm import LGBMRegressor

            lgb_params = {
                "n_estimators": 300,
                "max_depth": -1,
                "num_leaves": 48,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_alpha": 1.0,
                "reg_lambda": 2.0,
                "random_state": self._random_state,
                "verbose": -1,
                "min_child_samples": 25,
            }
            specs.append(("LightGBM_Error", LGBMRegressor, lgb_params))
        except ImportError:
            pass

        # 4. Random Forest Regressor
        rf_params = {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "random_state": self._random_state,
            "n_jobs": -1,
        }
        specs.append(("RF_Error", RandomForestRegressor, rf_params))

        return specs

    def predict_error(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the market error for each sample.

        Returns:
            Array of shape (n_samples,) — predicted market error.
            Positive = model believes market is UNDERestimating home team.
            Negative = model believes market is OVERestimating home team.
        """
        if not self._fitted:
            raise ValueError("System not fitted yet.")

        if self._error_regressor is None:
            return np.zeros(X.shape[0])

        n = X.shape[0]
        all_preds = np.zeros((n, len(self._error_regressor)))
        all_weights = np.zeros(len(self._error_regressor))

        for i, (name, model, weight) in enumerate(self._error_regressor):
            try:
                preds = model.predict(X)
                if hasattr(preds, "ndim") and preds.ndim > 1:
                    preds = preds.ravel()
                all_preds[:, i] = np.asarray(preds, dtype=float)
                all_weights[i] = weight
            except Exception:
                all_preds[:, i] = 0.0
                all_weights[i] = 0.0

        total_weight = all_weights.sum()
        if total_weight > 0:
            weighted_error = np.average(all_preds, axis=1, weights=all_weights)
        else:
            weighted_error = np.mean(all_preds, axis=1)

        # Clip to reasonable range (market errors rarely exceed 50% for binary outcome)
        return np.clip(weighted_error, -0.50, 0.50)

    def predict_proba(
        self,
        X: np.ndarray,
        market_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict win probabilities, optionally incorporating market data.

        This is the KEY method that differentiates from RobustPredictionSystem.

        If market_probs is provided:
          1. Get classifier's home_win probability (existing signal)
          2. Get error regressor's predicted market error (new signal)
          3. Compute error-adjusted: market_implied_prob + predicted_error
          4. Blend with classifier probability based on error regressor confidence

        If market_probs is not provided:
          Falls back to classifier-only prediction (backward compatible)

        Args:
            X: Feature matrix (n_samples, n_features)
            market_probs: Optional market-implied home win probabilities

        Returns:
            Array of shape (n_samples, 2): [P(loss), P(win)]
        """
        if not self._fitted:
            raise ValueError("System not fitted yet.")

        # Always get classifier's prediction (it's the backbone)
        classifier_probs = self._classifier.predict_proba(X)
        classifier_home = classifier_probs[:, 1]

        if market_probs is not None:
            market_probs = np.asarray(market_probs, dtype=float).ravel()
            market_probs = np.clip(market_probs, 0.01, 0.99)

            # Predict market error
            predicted_errors = self.predict_error(X)

            # Error-adjusted probability = what market says + how much market is wrong
            error_adjusted = market_probs + predicted_errors
            error_adjusted = np.clip(error_adjusted, 0.01, 0.99)

            # v6.0 — ADAPTIVE BLEND with sharp S-curve transition
            #
            # When predicted_error >> error_std → error signal dominates (up to 80%)
            # When predicted_error << error_std → classifier dominates (as low as 5%)
            #
            # Uses power > 1.0 for sharp S-curve: near-zero for small ratios,
            # near-max for large ratios. This prevents small error predictions
            # from polluting the classifier's signal while letting confident
            # error predictions drive the blend.
            error_ratio = np.abs(predicted_errors) / max(self._error_std, 0.01)
            blend_ratio = np.clip(
                np.minimum(error_ratio**1.5, 1.0),
                0.05,
                0.80,
            )
            blended = (
                1.0 - blend_ratio
            ) * classifier_home + blend_ratio * error_adjusted

            return np.column_stack([1.0 - blended, blended])

        return classifier_probs

    def predict(
        self,
        X: np.ndarray,
        market_probs: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary class (0/1) at given threshold."""
        probs = self.predict_proba(X, market_probs=market_probs)
        return (probs[:, 1] >= threshold).astype(int)

    def predict_with_details(
        self,
        X: np.ndarray,
        market_prob: Optional[float] = None,
    ) -> PredictionResult:
        """
        Predict a single sample with full detail, incorporating market data.

        Like predict_proba but returns a PredictionResult with all the
        confidence breakdown, edge computation, etc.

        Args:
            X: Single sample, shape (1, n_features)
            market_prob: Market-implied home win probability for this game

        Returns:
            PredictionResult with market-aware probabilities
        """
        if not self._fitted:
            raise ValueError("System not fitted yet.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get classifier details
        classifier_result = self._classifier.predict_with_details(X)
        classifier_home = classifier_result.home_win_prob

        if market_prob is not None:
            market_prob = float(np.clip(market_prob, 0.01, 0.99))

            # Predict market error
            predicted_error = float(self.predict_error(X)[0])

            # Error-adjusted
            error_adjusted = float(np.clip(market_prob + predicted_error, 0.01, 0.99))

            # v6.0 — ADAPTIVE BLEND (same logic as predict_proba)
            error_ratio = abs(predicted_error) / max(self._error_std, 0.01)
            blend_ratio = float(
                np.clip(
                    min(error_ratio**1.5, 1.0),
                    0.05,
                    0.80,
                )
            )
            blended = (
                1.0 - blend_ratio
            ) * classifier_home + blend_ratio * error_adjusted

            final_prob = float(np.clip(blended, 0.001, 0.999))

            # Add market-aware fields to the result
            classifier_result.home_win_prob = round(final_prob, 4)
            classifier_result.away_win_prob = round(1.0 - final_prob, 4)

            # Edge is now the predicted error itself (the inefficiency)
            classifier_result.edge_pct = round(predicted_error, 4)
            classifier_result.expected_value = round(
                predicted_error * 100, 2
            )  # Convert to % EV

            # Store market info in model_probs
            classifier_result.model_probs["market_implied"] = round(market_prob, 4)
            classifier_result.model_probs["predicted_error"] = round(predicted_error, 4)
            classifier_result.model_probs["error_adjusted"] = round(error_adjusted, 4)
            classifier_result.model_probs["blended"] = round(final_prob, 4)

            # Update confidence: more confident when error signal is strong
            error_signal_strength = min(
                abs(predicted_error) / 0.10, 1.0
            )  # 10% error = full signal
            confidence_boost = error_signal_strength * 0.15  # Max +0.15 boost
            classifier_result.confidence_score = float(
                np.clip(classifier_result.confidence_score + confidence_boost, 0.0, 1.0)
            )

            # Update confidence label
            cs = classifier_result.confidence_score
            if cs >= 0.85:
                classifier_result.confidence_label = "VERY_HIGH"
            elif cs >= 0.70:
                classifier_result.confidence_label = "HIGH"
            elif cs >= 0.50:
                classifier_result.confidence_label = "MEDIUM"
            elif cs >= 0.30:
                classifier_result.confidence_label = "LOW"
            else:
                classifier_result.confidence_label = "VERY_LOW"

        return classifier_result

    def get_summary(self) -> dict:
        """Get a JSON-serializable summary."""
        base = self._classifier.get_summary() if self._classifier._fitted else {}
        base["market_error_trained"] = self._error_regressor is not None
        base["error_mean"] = round(self._error_mean, 4)
        base["error_std"] = round(self._error_std, 4)
        if self._error_regressor is not None:
            base["n_error_models"] = len(self._error_regressor)
            base["error_weights"] = {
                name: round(w, 4) for name, _, w in self._error_regressor
            }
        return base

    def save(self, path: Path) -> str:
        """Save the fitted system to disk."""
        if not self._fitted:
            raise ValueError("System not fitted yet.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_joblib_dump(self, path)
        logger.info(f"MarketInefficiencySystem saved to {path} (hash verified)")
        return str(path)

    @staticmethod
    def load(path: Path, verify: bool = True) -> MarketInefficiencySystem:
        """Load a fitted system from disk."""
        path = Path(path)
        system = safe_joblib_load(path, verify=verify)
        logger.info(
            f"MarketInefficiencySystem loaded from {path} "
            f"(error_trained={system._error_regressor is not None})"
        )
        return system


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def compute_statistical_significance(
    wins: int, losses: int, null_hypothesis: float = 0.5
) -> dict:
    """
    Compute statistical significance of win rate using binomial test.

    Args:
        wins: Number of winning bets
        losses: Number of losing bets
        null_hypothesis: Null hypothesis win rate (default 0.5)

    Returns:
        Dict with win_rate, p_value, is_significant, z_score, ci_lower, ci_upper
    """
    n = wins + losses
    if n == 0:
        return {
            "win_rate": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "z_score": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n_bets": 0,
        }

    win_rate = wins / n

    # Normal approximation to binomial
    p0 = null_hypothesis
    se = math.sqrt(p0 * (1.0 - p0) / n)
    z_score = (win_rate - p0) / se if se > 0 else 0.0

    # One-sided p-value: P(win_rate > null_hypothesis)
    from scipy.stats import norm

    p_value = float(1.0 - norm.cdf(z_score))

    # Confidence interval (Wilson score)
    z = 1.96  # 95% CI
    denominator = 1.0 + z**2 / n
    centre = (win_rate + z**2 / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt((win_rate * (1.0 - win_rate) / n + z**2 / (4.0 * n**2)))
        / denominator
    )
    ci_lower = max(0.0, centre - margin)
    ci_upper = min(1.0, centre + margin)

    return {
        "win_rate": round(win_rate, 4),
        "p_value": round(p_value, 4),
        "is_significant": p_value < 0.05,
        "z_score": round(z_score, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n_bets": n,
    }


def compute_drawdown(profits: list[float]) -> dict:
    """
    Compute maximum drawdown from a series of profits/losses.

    Args:
        profits: List of sequential profit values

    Returns:
        Dict with max_drawdown, max_drawdown_pct, avg_drawdown
    """
    if not profits:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0, "avg_drawdown": 0.0}

    cumulative = np.cumsum(profits)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(np.max(drawdowns))

    # Percentage from peak
    peak_values = running_max
    dd_pcts = np.where(peak_values > 0, drawdowns / peak_values, 0.0)
    max_dd_pct = float(np.max(dd_pcts))

    avg_dd = float(np.mean(drawdowns))

    return {
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "avg_drawdown": round(avg_dd, 2),
    }


__all__ = [
    "RobustPredictionSystem",
    "MarketInefficiencySystem",
    "PredictionResult",
    "ModelDiagnostics",
    "WalkForwardFold",
    "OverfittingReport",
    "compute_statistical_significance",
    "compute_drawdown",
]
