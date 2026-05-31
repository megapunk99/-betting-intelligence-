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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TotalPointsPredictor":
        X_scaled = self.scaler.fit_transform(X)
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

    def __init__(self):
        super().__init__("SpreadPred_Ridge")
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
            self.model = CalibratedClassifierCV(
                base_estimator=self._base_logistic,
                method="sigmoid",   # Platt scaling
                cv=5,               # 5-fold CV within training set
            )
        else:
            self.model = self._base_logistic

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MomentumModel":
        X_scaled = self.scaler.fit_transform(X)
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
                base = getattr(cc, "base_estimator_", None) or getattr(cc, "base_estimator", None) or getattr(cc, "estimator", None)
                if base is not None and hasattr(base, "coef_"):
                    coefs.append(base.coef_[0])
            if coefs:
                self.feature_importance = np.mean(coefs, axis=0)
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
