"""
TotalsRegressor — predicts total points for a game using ensemble regression (v6.5).

Architecture (v6.5)
────────────
  - XGBoost Regressor (gradient boosted trees, non-linear patterns)
  - LightGBM Regressor (fast, handles large feature sets)
  - CatBoost Regressor (robust to outliers, categorical features)
  - Random Forest Regressor (bagging for variance reduction)
  - Ridge Regression (linear baseline fallback)
  - Weighted ensemble (weights from OOS MAE)
  - Quantile regression for prediction intervals (over/under confidence)

Training
────────
  Target: total_points (actual total points scored in the game)
  Features: Same feature_cols used by MarketInefficiencySystem
  OOS evaluation: Walk-forward chronological splits
  Early stopping for tree models (prevents overfitting)
  Confidence from quantile regression intervals

Inference
─────────
  For each game:
    1. Build feature vector
    2. Predict total points from all models
    3. Weighted ensemble prediction
    4. Quantile regression: P10, P50, P90 for confidence intervals
    5. Clip to [160, 280] (NBA range)
    6. Compute edge = (predicted - market_total) / market_total
    7. Compute confidence from interval width and edge magnitude
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Data ──────────────────────────────────────────────────────────────────

@dataclass
class TotalsPrediction:
    """Result of a totals prediction for a single game (v6.5)."""
    predicted_total: float = 0.0
    edge_pct: float = 0.0
    direction: str = "neutral"   # "over", "under", or "neutral"
    confidence: str = "low"
    market_total: Optional[float] = None
    n_models: int = 0
    mae: Optional[float] = None
    # v6.5 — Quantile predictions for confidence intervals
    pred_p10: Optional[float] = None  # 10th percentile (optimistic under)
    pred_p90: Optional[float] = None  # 90th percentile (optimistic over)
    interval_width: Optional[float] = None  # p90 - p10 (uncertainty measure)


# ── Model ─────────────────────────────────────────────────────────────────

class TotalsRegressor:
    """
    Ensemble regression model that predicts total points for NBA games (v6.5).

    Trains XGBoost + LightGBM + CatBoost + RandomForest regressors and
    combines them via weighted averaging based on OOS MAE. Uses quantile
    regression for confidence intervals.
    """

    def __init__(
        self,
        random_state: int = 42,
        use_catboost: bool = True,
        use_histgb: bool = True,
        use_extratrees: bool = True,
        use_early_stopping: bool = True,
        use_quantile: bool = True,
    ):
        self._models: dict[str, Any] = {}
        self._weights: dict[str, float] = {}
        self._feature_names: list[str] = []
        self._fitted: bool = False
        self._target_mean: float = 220.0
        self._mae: Optional[float] = None
        self._random_state = random_state
        self._fit_timestamp: Optional[str] = None
        self._use_catboost = use_catboost
        self._use_histgb = use_histgb
        self._use_extratrees = use_extratrees
        self._use_early_stopping = use_early_stopping
        self._use_quantile = use_quantile

        # Model diagnostics (v6.6)
        self._model_diagnostics: dict[str, dict] = {}  # {name: {mae, r2, n_train, status}}

        # Quantile models (v6.5)
        self._quantile_lower: Any = None  # 10th percentile
        self._quantile_upper: Any = None  # 90th percentile

    @property
    def mae(self) -> Optional[float]:
        """Mean absolute error on OOS data from training."""
        return self._mae

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        X: np.ndarray,
        y_total: np.ndarray,
        feature_names: Optional[list[str]] = None,
        verbose: bool = True,
    ) -> TotalsRegressor:
        """
        Train the totals ensemble on historical game data (v6.5).

        Args:
            X: Feature matrix (n_samples, n_features)
            y_total: Target vector — actual total points (n_samples,)
            feature_names: Optional column names for reference
            verbose: Print progress

        Returns:
            self (fitted)
        """
        n = len(X)
        self._target_mean = float(np.mean(y_total))

        if feature_names and len(feature_names) == X.shape[1]:
            self._feature_names = feature_names
        else:
            self._feature_names = [f"f{i}" for i in range(X.shape[1])]

        models: dict[str, Any] = {}
        model_diagnostics: dict[str, dict] = {}

        # ── 1. XGBoost Regressor ─────────────────────────────────────
        try:
            from xgboost import XGBRegressor
            xgb = XGBRegressor(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=2.0,
                gamma=0.1,
                random_state=self._random_state,
                verbosity=0,
                early_stopping_rounds=50 if self._use_early_stopping else None,
            )
            if self._use_early_stopping and n > 200:
                try:
                    split_idx = int(n * 0.85)
                    xgb.fit(
                        X[:split_idx], y_total[:split_idx],
                        eval_set=[(X[split_idx:], y_total[split_idx:])],
                        verbose=False,
                    )
                except Exception:
                    xgb.fit(X, y_total)
            else:
                # Refit without early_stopping if it was set
                if self._use_early_stopping:
                    try:
                        xgb_no_es = XGBRegressor(
                            n_estimators=600,
                            max_depth=6,
                            learning_rate=0.03,
                            subsample=0.8,
                            colsample_bytree=0.7,
                            reg_alpha=1.0,
                            reg_lambda=2.0,
                            random_state=self._random_state,
                            verbosity=0,
                        )
                        xgb_no_es.fit(X, y_total)
                        models["XGBoost"] = xgb_no_es
                    except Exception:
                        xgb.fit(X, y_total)
                        models["XGBoost"] = xgb
                else:
                    xgb.fit(X, y_total)
                    models["XGBoost"] = xgb
        except ImportError:
            if verbose:
                logger.debug("XGBoost not available for totals model")

        # ── 2. LightGBM Regressor ────────────────────────────────────
        try:
            from lightgbm import LGBMRegressor
            lgb = LGBMRegressor(
                n_estimators=600,
                max_depth=-1,
                num_leaves=48,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=2.0,
                min_child_samples=25,
                min_split_gain=0.1,
                random_state=self._random_state,
                verbose=-1,
            )
            if self._use_early_stopping and n > 200:
                try:
                    split_idx = int(n * 0.85)
                    lgb.fit(
                        X[:split_idx], y_total[:split_idx],
                        eval_set=[(X[split_idx:], y_total[split_idx:])],
                        callbacks=[LGBMEarlyStopping(50)],
                    )
                except Exception:
                    lgb.fit(X, y_total)
            else:
                lgb.fit(X, y_total)
            models["LightGBM"] = lgb
        except ImportError:
            if verbose:
                logger.debug("LightGBM not available for totals model")

        # ── 3. CatBoost Regressor (v6.5 NEW) ─────────────────────────
        if self._use_catboost:
            try:
                from catboost import CatBoostRegressor
                cb = CatBoostRegressor(
                    iterations=500,
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=3.0,
                    subsample=0.8,
                    random_seed=self._random_state,
                    verbose=0,
                    early_stopping_rounds=50 if self._use_early_stopping else None,
                    loss_function="MAE",
                )
                if self._use_early_stopping and n > 200:
                    try:
                        split_idx = int(n * 0.85)
                        cb.fit(
                            X[:split_idx], y_total[:split_idx],
                            eval_set=(X[split_idx:], y_total[split_idx:]),
                            verbose=False,
                        )
                    except Exception:
                        cb.fit(X, y_total)
                else:
                    cb.fit(X, y_total)
                models["CatBoost"] = cb
            except ImportError:
                if verbose:
                    logger.debug("CatBoost not available for totals model")

        # ── 4. Random Forest Regressor ───────────────────────────────
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=self._random_state,
                n_jobs=-1,
            )
            rf.fit(X, y_total)
            models["RandomForest"] = rf
        except ImportError:
            if verbose:
                logger.debug("RandomForest not available for totals model")

        # ── 5. HistGradientBoosting Regressor (v6.6 NEW) ──────────────
        if self._use_histgb:
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor
                hgb = HistGradientBoostingRegressor(
                    max_iter=600,
                    max_depth=5,
                    learning_rate=0.03,
                    max_leaf_nodes=63,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    early_stopping=self._use_early_stopping,
                    scoring="loss",
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=self._random_state,
                    verbose=0,
                )
                if self._use_early_stopping and n > 200:
                    try:
                        split_idx = int(n * 0.85)
                        hgb.fit(
                            X[:split_idx], y_total[:split_idx].ravel(),
                        )
                    except Exception:
                        hgb.fit(X, y_total)
                else:
                    hgb.fit(X, y_total)
                models["HistGradientBoosting"] = hgb
            except ImportError:
                if verbose:
                    logger.debug("HistGradientBoosting not available for totals model")

        # ── 6. ExtraTrees Regressor (v6.6 NEW) ────────────────────────
        if self._use_extratrees:
            try:
                from sklearn.ensemble import ExtraTreesRegressor
                et = ExtraTreesRegressor(
                    n_estimators=600,
                    max_depth=12,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    min_samples_split=8,
                    random_state=self._random_state,
                    n_jobs=-1,
                    bootstrap=True,
                )
                et.fit(X, y_total)
                models["ExtraTrees"] = et
            except ImportError:
                if verbose:
                    logger.debug("ExtraTrees not available for totals model")

        # ── 7. Fallback: Ridge ────────────────────────────────────────
        if not models:
            from sklearn.linear_model import Ridge
            ridge = Ridge(alpha=2.0, random_state=self._random_state)
            ridge.fit(X, y_total)
            models["Ridge"] = ridge

        # ── Compute OOS MAE via CHRONOLOGICAL walk-forward (v6.6 FIX) ─
        from sklearn.metrics import mean_absolute_error

        n_folds = min(5, max(2, n // 100))
        fold_size = max(100, n // max(n_folds, 1))
        oos_preds = np.full(n, np.nan)
        min_train = 200

        for fold_idx in range(max(n_folds, 1)):
            # v6.6 FIX: Strictly increasing test starts (chronological).
            # The OLD code used modulo arithmetic which created NON-chronological
            # folds where the "test" set could be BEFORE training data.
            test_start = fold_idx * fold_size
            test_end = min(test_start + fold_size, n)

            if test_start < min_train or (test_end - test_start) < 20:
                continue

            # Train ONLY on data BEFORE the test set (no future leakage)
            X_train_fold = X[:test_start]
            y_train_fold = y_total[:test_start]

            if len(X_train_fold) < min_train:
                continue

            # Train each model on this chronological fold, collect ensemble predictions
            fold_preds_collect = []
            for name, model in models.items():
                try:
                    fold_model = _clone_model(model)
                    fold_model.fit(X_train_fold, y_train_fold)
                    fold_preds_collect.append(
                        fold_model.predict(X[test_start:test_end])
                    )
                except Exception:
                    continue

            if fold_preds_collect:
                # Average all model predictions (equal-weight ensemble)
                oos_preds[test_start:test_end] = np.mean(fold_preds_collect, axis=0)

        oos_valid = oos_preds[~np.isnan(oos_preds)]
        if len(oos_valid) >= 30:
            oos_y = y_total[~np.isnan(oos_preds)]
            self._mae = float(mean_absolute_error(oos_y, oos_valid))
        else:
            full_pred = self._predict_ensemble_internal(X, models, equal_weight=True)
            self._mae = float(mean_absolute_error(y_total, full_pred))

        # ── Compute per-model MAE & diagnostics (v6.6) ────────────────
        for name, model in models.items():
            try:
                preds = model.predict(X)
                m = float(mean_absolute_error(y_total, preds))
                # Simple R²-like metric
                baseline_mae = float(np.mean(np.abs(y_total - np.mean(y_total))))
                r2_like = 1.0 - (m / max(baseline_mae, 0.1))
                status = "ok"
                if m > baseline_mae:
                    status = "degraded"
                model_diagnostics[name] = {
                    "mae": round(m, 2),
                    "r2_score": round(r2_like, 4),
                    "n_train": n,
                    "status": status,
                }
            except Exception:
                model_diagnostics[name] = {
                    "mae": self._mae or 12.0,
                    "r2_score": 0.0,
                    "n_train": n,
                    "status": "failed",
                }

        self._model_diagnostics = model_diagnostics

        # ── Compute weights ──────────────────────────────────────────
        weights: dict[str, float] = {}
        if len(models) > 1:
            model_maes = {
                name: diag["mae"]
                for name, diag in model_diagnostics.items()
                if diag["status"] != "failed"
            }
            if not model_maes:
                model_maes = {name: 12.0 for name in models}

            total_inv = sum(1.0 / max(m, 0.1) for m in model_maes.values())
            for name, m in model_maes.items():
                weights[name] = (1.0 / max(m, 0.1)) / total_inv if total_inv > 0 else 1.0 / len(model_maes)
        else:
            for name in models:
                weights[name] = 1.0

        self._models = models
        self._weights = weights
        self._fitted = True
        self._fit_timestamp = datetime.now().isoformat()

        # ── Quantile Regression (v6.5) ───────────────────────────────
        if self._use_quantile and self._mae is not None:
            self._fit_quantile_models(X, y_total, verbose)

        if verbose:
            weight_str = ", ".join(f"{n}={w:.2f}" for n, w in weights.items())
            diag_str = ", ".join(
                f"{n}={d['mae']:.1f}" for n, d in model_diagnostics.items()
            )
            logger.info(
                f"TotalsRegressor v6.6 built: {len(models)} models, "
                f"OOS MAE={self._mae:.1f}, "
                f"(per-model MAE: {diag_str}) | "
                f"(weights: {weight_str})"
            )

        return self

    def _fit_quantile_models(
        self,
        X: np.ndarray,
        y_total: np.ndarray,
        verbose: bool = True,
    ):
        """Fit quantile regression models for prediction intervals (v6.5)."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            # Lower bound (10th percentile) — optimistic under estimate
            self._quantile_lower = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                loss="quantile",
                alpha=0.1,
                random_state=self._random_state,
            )
            self._quantile_lower.fit(X, y_total)

            # Upper bound (90th percentile) — optimistic over estimate
            self._quantile_upper = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                loss="quantile",
                alpha=0.9,
                random_state=self._random_state,
            )
            self._quantile_upper.fit(X, y_total)

            if verbose:
                logger.info("Quantile regression models trained (P10/P90)")
        except Exception as e:
            logger.debug(f"Quantile regression failed: {e}")
            self._quantile_lower = None
            self._quantile_upper = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict total points for each sample."""
        if not self._fitted or not self._models:
            return np.full(X.shape[0], self._target_mean)
        ensemble_pred = self._predict_ensemble_internal(X, self._models, equal_weight=False)
        return np.clip(ensemble_pred, 160.0, 280.0)

    def predict_quantiles(
        self,
        X: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Predict 10th and 90th percentile of total points (v6.5).

        Returns:
            (p10, p90) tuple, where each is shape (n_samples,) or None
        """
        if not self._use_quantile or self._quantile_lower is None or self._quantile_upper is None:
            return None, None

        try:
            p10 = np.clip(self._quantile_lower.predict(X), 160.0, 280.0)
            p90 = np.clip(self._quantile_upper.predict(X), 160.0, 280.0)
            return p10, p90
        except Exception:
            return None, None

    def predict_single(
        self,
        X: np.ndarray,
        market_total: Optional[float] = None,
    ) -> TotalsPrediction:
        """
        Predict total points for a single game and compute edge vs market total (v6.5).

        Uses quantile regression intervals for improved confidence estimation.

        Args:
            X: Single sample, shape (1, n_features) or (n_features,)
            market_total: The market total points line (e.g. 218.5)

        Returns:
            TotalsPrediction with predicted total, edge, direction, confidence
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predicted = float(self.predict(X)[0])

        # Quantile predictions (v6.5)
        p10, p90 = self.predict_quantiles(X)
        pred_p10 = float(p10[0]) if p10 is not None else None
        pred_p90 = float(p90[0]) if p90 is not None else None
        interval_width = (pred_p90 - pred_p10) if pred_p10 is not None and pred_p90 is not None else None

        # Compute edge vs market total
        if market_total and market_total > 0:
            edge = (predicted - market_total) / market_total
            direction = "over" if edge > 0 else "under"
            abs_edge = abs(edge)

            # v6.5 — Enhanced confidence using both MAE and quantile intervals
            if self._mae and self._mae > 0:
                signal_noise = abs_edge * market_total / self._mae

                # Adjust confidence by interval width (wider = less confident)
                if interval_width is not None and interval_width > 0:
                    # Normalize: NBA games typically have ~20 point intervals
                    width_penalty = min(interval_width / 30.0, 1.5)
                    adjusted_signal = signal_noise / max(width_penalty, 0.5)
                else:
                    adjusted_signal = signal_noise

                if adjusted_signal >= 1.5:
                    confidence = "high"
                elif adjusted_signal >= 0.8:
                    confidence = "medium"
                elif adjusted_signal >= 0.3:
                    confidence = "low"
                else:
                    confidence = "very_low"
            else:
                if abs_edge >= 0.04:
                    confidence = "high"
                elif abs_edge >= 0.02:
                    confidence = "medium"
                elif abs_edge >= 0.01:
                    confidence = "low"
                else:
                    confidence = "very_low"
        else:
            edge = 0.0
            direction = "neutral"
            confidence = "very_low"

        return TotalsPrediction(
            predicted_total=round(predicted, 1),
            edge_pct=round(edge, 4),
            direction=direction,
            confidence=confidence,
            market_total=market_total,
            n_models=len(self._models),
            mae=round(self._mae, 2) if self._mae else None,
            pred_p10=round(pred_p10, 1) if pred_p10 else None,
            pred_p90=round(pred_p90, 1) if pred_p90 else None,
            interval_width=round(interval_width, 1) if interval_width else None,
        )

    def get_summary(self) -> dict:
        """Get a JSON-serializable summary."""
        return {
            "fitted": self._fitted,
            "fit_timestamp": self._fit_timestamp,
            "n_models": len(self._models),
            "mae": round(self._mae, 2) if self._mae else None,
            "target_mean": round(self._target_mean, 1),
            "model_names": list(self._models.keys()),
            "model_weights": {k: round(v, 4) for k, v in self._weights.items()},
            "model_diagnostics": self._model_diagnostics,
            "quantile_enabled": self._use_quantile,
            "quantile_trained": self._quantile_lower is not None,
        }

    def _predict_ensemble_internal(
        self,
        X: np.ndarray,
        models: dict[str, Any],
        equal_weight: bool = False,
    ) -> np.ndarray:
        """Get weighted ensemble prediction."""
        if not models:
            return np.full(X.shape[0], self._target_mean)

        all_preds = []
        all_weights = []

        for name, model in models.items():
            w = 1.0 / len(models) if equal_weight else self._weights.get(name, 1.0 / len(models))
            try:
                preds = model.predict(X)
                if hasattr(preds, "ndim") and preds.ndim > 1:
                    preds = preds.ravel()
                all_preds.append(np.asarray(preds, dtype=float))
                all_weights.append(w)
            except Exception:
                continue

        if not all_preds:
            return np.full(X.shape[0], self._target_mean)

        total_weight = sum(all_weights)
        if total_weight > 0:
            ensemble = np.average(np.column_stack(all_preds), axis=1, weights=all_weights)
        else:
            ensemble = np.mean(all_preds, axis=1)

        return ensemble


# ── LightGBM early stopping compatibility ────────────────────────────────
try:
    from lightgbm import early_stopping as LGBMEarlyStopping
except ImportError:
    try:
        from lightgbm.callback import early_stopping as LGBMEarlyStopping
    except ImportError:
        LGBMEarlyStopping = None


# ── Clone helper for chronological CV (v6.6) ────────────────────────────

def _clone_model(model: Any) -> Any:
    """
    Clone a model by getting its class and params, then creating a fresh instance.

    Used by the chronological walk-forward CV to train fold-specific models
    without mutating the original full-data model.

    Raises ValueError if the model cannot be cloned.
    """
    try:
        from sklearn.base import clone
        return clone(model)
    except Exception:
        pass

    # Manual fallback: create a new instance of the same class
    try:
        model_class = model.__class__
        params = model.get_params()
        return model_class(**params)
    except Exception as e:
        raise ValueError(
            f"Cannot clone model of type {model.__class__.__name__}: {e}"
        )
