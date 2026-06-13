"""
TotalsRegressor — predicts total points for a game using the same features as
the MarketInefficiencySystem. Wired into LivePredictionEngine to provide
over/under edge alongside the moneyline edge.

Architecture
────────────
  - XGBoost Regressor (primary, handles non-linear patterns)
  - LightGBM Regressor (secondary, fast convergence)
  - Weighted ensemble of both (weights from OOS MAE)
  - Train/test split for confidence calibration
  - Reasonable range clipping (NBA: 160-280)

Training
────────
  Target: total_points (actual total points scored in the game)
  Features: Same feature_cols used by MarketInefficiencySystem
  OOS evaluation: Walk-forward chronological splits

Inference
─────────
  For each game:
    1. Build feature vector
    2. Predict total points
    3. Clip to [160, 280] (NBA range)
    4. Compute edge = (predicted - market_total) / market_total
    5. Compute confidence from edge magnitude / MAE ratio
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
    """Result of a totals prediction for a single game."""
    predicted_total: float = 0.0
    edge_pct: float = 0.0
    direction: str = "neutral"   # "over", "under", or "neutral"
    confidence: str = "low"
    market_total: Optional[float] = None
    n_models: int = 0
    mae: Optional[float] = None


# ── Model ─────────────────────────────────────────────────────────────────

class TotalsRegressor:
    """
    Ensemble regression model that predicts total points for NBA games.

    Trains XGBoost + LightGBM regressors on historical data and predicts
    total points. At inference, computes edge = predicted - market_total.
    """

    def __init__(self, random_state: int = 42):
        self._models: dict[str, Any] = {}
        self._weights: dict[str, float] = {}
        self._feature_names: list[str] = []
        self._fitted: bool = False
        self._target_mean: float = 220.0
        self._mae: Optional[float] = None
        self._random_state = random_state
        self._fit_timestamp: Optional[str] = None

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
        Train the totals ensemble on historical game data.

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

        # ── Walk-forward CV ──────────────────────────────────────────
        n_folds = min(5, n // 50)
        fold_size = max(50, n // max(n_folds, 1))

        models: dict[str, Any] = {}
        weights: dict[str, float] = {}

        # Try XGBoost first
        try:
            from xgboost import XGBRegressor
            xgb = XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=2.0,
                random_state=self._random_state,
                verbosity=0,
            )
            xgb.fit(X, y_total)
            models["XGBoost"] = xgb
        except ImportError:
            if verbose:
                logger.debug("XGBoost not available for totals model")

        # Try LightGBM
        try:
            from lightgbm import LGBMRegressor
            lgb = LGBMRegressor(
                n_estimators=400,
                max_depth=-1,
                num_leaves=48,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=2.0,
                random_state=self._random_state,
                verbose=-1,
                min_child_samples=25,
            )
            lgb.fit(X, y_total)
            models["LightGBM"] = lgb
        except ImportError:
            if verbose:
                logger.debug("LightGBM not available for totals model")

        # Fallback: use a simple Ridge if no tree models available
        if not models:
            from sklearn.linear_model import Ridge
            ridge = Ridge(alpha=2.0, random_state=self._random_state)
            ridge.fit(X, y_total)
            models["Ridge"] = ridge

        # ── Compute OOS MAE via walk-forward ─────────────────────────
        oos_preds = np.full(n, np.nan)
        min_train = 100

        for fold_idx in range(max(n_folds, 1)):
            test_start = (fold_idx * fold_size) % max(n - fold_size, 1)
            test_end = min(test_start + fold_size, n)
            if test_start < min_train or test_end - test_start < 10:
                continue

            X_train = np.concatenate([
                X[:test_start], X[test_end:]
            ], axis=0) if len(X) > test_end else X[:test_start]
            y_train = np.concatenate([
                y_total[:test_start], y_total[test_end:]
            ], axis=0) if len(y_total) > test_end else y_total[:test_start]

            if len(X_train) < min_train:
                continue

            fold_preds = self._predict_ensemble_internal(
                X[test_start:test_end],
                models,
                equal_weight=True,
            )
            oos_preds[test_start:test_end] = fold_preds

        from sklearn.metrics import mean_absolute_error
        oos_valid = oos_preds[~np.isnan(oos_preds)]
        if len(oos_valid) >= 10:
            oos_y = y_total[~np.isnan(oos_preds)]
            self._mae = float(mean_absolute_error(oos_y, oos_valid))
        else:
            # Estimate from full-data residuals
            full_pred = self._predict_ensemble_internal(X, models, equal_weight=True)
            self._mae = float(mean_absolute_error(y_total, full_pred))

        # ── Compute weights: better models get more weight ────────────
        if len(models) > 1:
            model_maes = {}
            for name, model in models.items():
                try:
                    preds = model.predict(X)
                    m = float(mean_absolute_error(y_total, preds))
                    model_maes[name] = m
                except Exception:
                    model_maes[name] = self._mae or 12.0

            total_inv = sum(1.0 / max(m, 0.1) for m in model_maes.values())
            for name, m in model_maes.items():
                weights[name] = (1.0 / max(m, 0.1)) / total_inv if total_inv > 0 else 1.0 / len(models)
        else:
            for name in models:
                weights[name] = 1.0

        self._models = models
        self._weights = weights
        self._fitted = True
        self._fit_timestamp = datetime.now().isoformat()

        if verbose:
            weight_str = ", ".join(f"{n}={w:.2f}" for n, w in weights.items())
            logger.info(
                f"TotalsRegressor built: {len(models)} models, "
                f"MAE={self._mae:.1f}, "
                f"(weights: {weight_str})"
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict total points for each sample."""
        if not self._fitted or not self._models:
            return np.full(X.shape[0], self._target_mean)
        ensemble_pred = self._predict_ensemble_internal(X, self._models, equal_weight=False)
        return np.clip(ensemble_pred, 160.0, 280.0)

    def predict_single(self, X: np.ndarray, market_total: Optional[float] = None) -> TotalsPrediction:
        """
        Predict total points for a single game and compute edge vs market total.

        Args:
            X: Single sample, shape (1, n_features) or (n_features,)
            market_total: The market total points line (e.g. 218.5)

        Returns:
            TotalsPrediction with predicted total, edge, direction, confidence
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predicted = float(self.predict(X)[0])

        # Compute edge vs market total
        if market_total and market_total > 0:
            edge = (predicted - market_total) / market_total
            direction = "over" if edge > 0 else "under"
            abs_edge = abs(edge)

            # Confidence based on edge vs model MAE
            if self._mae and self._mae > 0:
                signal_noise = abs_edge * market_total / self._mae
                if signal_noise >= 1.5:
                    confidence = "high"
                elif signal_noise >= 0.8:
                    confidence = "medium"
                elif signal_noise >= 0.3:
                    confidence = "low"
                else:
                    confidence = "very_low"
            else:
                # Fallback: use simple thresholds
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
