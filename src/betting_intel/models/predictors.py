"""
Prediction models for basketball betting strategies.
Implements simple robust models with interpretability as priority.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, log_loss, brier_score_loss
)

from betting_intel.config import ENABLE_XGBOOST_MODEL

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class BasePredictor:
    """Base class for all predictors."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BasePredictor":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> Dict:
        return {"name": self.name, "type": self.__class__.__name__}


class TotalPointsPredictor(BasePredictor):
    """
    Predicts total points in a game.
    Edge hypothesis: Sportsbooks use league-average baselines;
    we can improve by incorporating pace, rest, and recent form.
    """

    def __init__(self, model_type: str = "ridge"):
        super().__init__(f"TotalPred_{model_type}")
        self.model_type = model_type

        if model_type == "ridge":
            self.model = Ridge(alpha=1.0)
        elif model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            self.model = XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1
            )
        elif model_type == "xgboost":
            self.model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
        else:
            self.model = Ridge(alpha=1.0)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "TotalPointsPredictor":
        X_scaled = self.scaler.fit_transform(X)
        if sample_weight is not None:
            self.model.fit(X_scaled, y, sample_weight=sample_weight)
        else:
            self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Extract feature importance if available
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            self.feature_importance = self.model.coef_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        preds = self.predict(X)
        return {
            "mae": mean_absolute_error(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
            "r2": r2_score(y, preds),
            "mean_pred": np.mean(preds),
            "mean_actual": np.mean(y),
            "bias": np.mean(preds - y),
        }


class SpreadPredictor(BasePredictor):
    """
    Predicts point spread (margin of victory).
    Edge hypothesis: Market overreacts to recent results;
    we can identify regression-to-mean opportunities.
    """

    def __init__(self, model_type: str = "ridge"):
        super().__init__(f"SpreadPred_{model_type}")
        if model_type == "ridge":
            self.model = Ridge(alpha=2.0)
        elif model_type == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
                self.model = LGBMRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    random_state=42, verbosity=-1
                )
            except ImportError:
                self.model = Ridge(alpha=2.0)
        else:
            self.model = Ridge(alpha=2.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpreadPredictor":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        if hasattr(self.model, "coef_"):
            self.feature_importance = self.model.coef_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        # ATS accuracy: correct side (not margin)
        ats_correct = np.mean((preds > 0) == (y > 0))
        return {
            "mae": mean_absolute_error(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
            "r2": r2_score(y, preds),
            "ats_accuracy": ats_correct,
            "mean_pred": np.mean(preds),
            "mean_actual": np.mean(y),
        }


class MomentumModel(BasePredictor):
    """
    Models momentum reversion.
    Edge hypothesis: Teams on streaks are overvalued by the market.
    Bets against extreme streaks should have positive EV.

    When calibrate=True, applies Platt scaling (sigmoid calibration) via
    CalibratedClassifierCV to correct overconfident probability estimates.
    """

    def __init__(self, model_type: str = "logistic", calibrate: bool = False):
        name = "Momentum_Logistic_Calibrated" if calibrate else "Momentum_Logistic"
        super().__init__(name)
        self.calibrate = calibrate
        self._base_logistic = LogisticRegression(
            C=0.5, class_weight="balanced", random_state=42
        )
        if calibrate:
            # sklearn >= 1.2 renamed base_estimator to estimator
            try:
                self.model = CalibratedClassifierCV(
                    estimator=self._base_logistic,
                    method="sigmoid",
                    cv=5,
                )
            except TypeError:
                # Fallback for older sklearn versions
                self.model = CalibratedClassifierCV(
                    base_estimator=self._base_logistic,
                    method="sigmoid",
                    cv=5,
                )
        else:
            self.model = self._base_logistic

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "MomentumModel":
        X_scaled = self.scaler.fit_transform(X)
        if sample_weight is not None:
            self.model.fit(X_scaled, y, sample_weight=sample_weight)
        else:
            self.model.fit(X_scaled, y)
        self.is_fitted = True
        self._extract_feature_importance()
        return self

    def _extract_feature_importance(self):
        """Extract coefficients from the underlying logistic regression."""
        if self.calibrate:
            # CalibratedClassifierCV stores one calibrated classifier per fold
            coefs = []
            for cc in self.model.calibrated_classifiers_:
                # Access the fitted base estimator (attribute name varies by sklearn version)
                # sklearn >= 1.2 uses estimator_, older uses base_estimator_
                base = (getattr(cc, "estimator", None) or
                        getattr(cc, "base_estimator_", None) or
                        getattr(cc, "base_estimator", None))
                if base is not None and hasattr(base, "coef_"):
                    coefs.append(base.coef_[0])
        else:
            if hasattr(self.model, "coef_"):
                self.feature_importance = self.model.coef_[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        proba = self.predict_proba(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "log_loss": log_loss(y, proba),
            "brier": brier_score_loss(y, proba[:, 1]),
            "class_balance": float(np.mean(preds)),
        }


class StackingEnsemblePredictor:
    """
    Stacking ensemble that combines multiple base predictors via a meta-model.

    Usage:
        ensemble = StackingEnsemblePredictor(prediction_type="regression")
        ensemble.add_base_model(TotalPointsPredictor("ridge"))
        ensemble.add_base_model(TotalPointsPredictor("lightgbm"))
        ensemble.fit(X_train, y_train)
        preds = ensemble.predict(X_test)
    """

    def __init__(self, prediction_type: str = "regression"):
        self.prediction_type = prediction_type
        self.base_models: list[BasePredictor] = []
        self.meta_model = None
        self.is_fitted = False

    def add_base_model(self, model: BasePredictor) -> "StackingEnsemblePredictor":
        """Add a base model to the ensemble."""
        self.base_models.append(model)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemblePredictor":
        """Fit all base models and the meta-model."""
        if not self.base_models:
            raise ValueError("At least one base model must be added before fitting")

        # Split data for meta-training (simple holdout)
        n = len(X)
        split = max(1, int(n * 0.8))
        X_base, X_meta = X[:split], X[split:]
        y_base, y_meta = y[:split], y[split:]

        # Train base models
        for model in self.base_models:
            model.fit(X_base, y_base)

        # Generate meta-features from base models on held-out data
        meta_features = np.column_stack([
            model.predict(X_meta) for model in self.base_models
        ])

        # Train meta-model
        if self.prediction_type == "regression":
            from sklearn.linear_model import LinearRegression
            self.meta_model = LinearRegression()
        else:
            from sklearn.linear_model import LogisticRegression
            self.meta_model = LogisticRegression(C=1.0, max_iter=1000)

        self.meta_model.fit(meta_features, y_meta)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the stacked ensemble."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        # Generate predictions from all base models
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models
        ])

        # Meta-model final prediction
        return self.meta_model.predict(meta_features)

    def get_params(self) -> Dict[str, Any]:
        return {
            "prediction_type": self.prediction_type,
            "n_base_models": len(self.base_models),
            "meta_model": type(self.meta_model).__name__ if self.meta_model else None,
        }
