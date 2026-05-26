"""
Prediction models for basketball betting strategies.
v2.0 — Advanced models with hyperparameter tuning, ensembles, and uncertainty quantification.

Models:
  - LightGBM Regressor/Classifier (gradient boosting on steroids)
  - CatBoost Regressor/Classifier (native categorical support)
  - Bayesian Ridge (uncertainty quantification)
  - Stacking Ensemble (meta-learner on base models)
  - Optuna hyperparameter optimization
  - Probability calibration (Platt scaling)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import warnings
import json

from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression, BayesianRidge
from sklearn.ensemble import (
    GradientBoostingRegressor, GradientBoostingClassifier,
    RandomForestRegressor, RandomForestClassifier,
    StackingRegressor, StackingClassifier,
    VotingRegressor, VotingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
)
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, log_loss, brier_score_loss, roc_auc_score
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from config import ENABLE_XGBOOST_MODEL, XGBOOST_AVAILABLE

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════
#  BASE & UTILITY CLASSES
# ═════════════════════════════════════════════════════════════════════════


class BasePredictor:
    """Base class for all predictors."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.is_fitted = False
        self.training_history: Dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BasePredictor":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> Dict:
        return {"name": self.name, "type": self.__class__.__name__}

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Optional[Dict]:
        """Return feature importance mapping."""
        if self.feature_importance is None:
            return None
        if feature_names and len(feature_names) == len(self.feature_importance):
            return dict(zip(feature_names, self.feature_importance))
        return {"importance": self.feature_importance.tolist()}


class TunedPredictor(BasePredictor):
    """Base for predictors that support Optuna hyperparameter tuning."""

    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 30,
        cv: int = 3,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
    ) -> Dict:
        """
        Tune hyperparameters using Optuna with time-series cross-validation.

        Args:
            X: Training features
            y: Training targets
            n_trials: Number of Optuna trials
            cv: Number of cross-validation folds
            timeout: Optional timeout in seconds
            study_name: Name for the study

        Returns:
            Best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            print(f"  [!] Optuna not available. Using default params for {self.name}")
            return {}

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=cv)

        def objective(trial):
            params = self._get_trial_params(trial)
            model_copy = self._build_model(params)
            scores = cross_val_score(
                model_copy, X, y, cv=tscv,
                scoring=self._get_scoring(),
                n_jobs=1
            )
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name or f"{self.name}_tuning",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        try:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        except Exception as e:
            print(f"  [!] Tuning failed for {self.name}: {e}")
            return {}

        self.training_history["optuna_study"] = study
        self.training_history["best_params"] = study.best_params
        self.training_history["best_value"] = study.best_value

        print(f"  Tuned {self.name}: best score={study.best_value:.4f} ({n_trials} trials)")
        return study.best_params

    def _get_trial_params(self, trial: optuna.Trial) -> Dict:
        raise NotImplementedError

    def _build_model(self, params: Dict) -> Any:
        raise NotImplementedError

    def _get_scoring(self) -> str:
        return "neg_mean_absolute_error"


# ═════════════════════════════════════════════════════════════════════════
#  TOTAL POINTS PREDICTORS (Regression)
# ═════════════════════════════════════════════════════════════════════════


class TotalPointsPredictor(BasePredictor):
    """
    Predicts total points in a game.
    Supports multiple model backends with automatic best-model selection.
    """

    def __init__(self, model_type: str = "ridge"):
        super().__init__(f"TotalPred_{model_type}")
        self.model_type = model_type

        if model_type == "ridge":
            self.model = Ridge(alpha=1.0)
        elif model_type == "lightgbm" and LGBM_AVAILABLE:
            self.model = LGBMRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                min_child_samples=10, random_state=42, n_jobs=-1,
                verbose=-1,
            )
        elif model_type == "catboost" and CATBOOST_AVAILABLE:
            self.model = CatBoostRegressor(
                iterations=500, depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bylevel=0.7,
                l2_leaf_reg=3.0, random_seed=42, verbose=0,
                early_stopping_rounds=50,
            )
        elif model_type == "bayesian":
            self.model = BayesianRidgeRegressor()  # see below
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=300, max_depth=10, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
        elif model_type == "xgboost":
            if XGBOOST_AVAILABLE:
                from xgboost import XGBRegressor
                self.model = XGBRegressor(
                    n_estimators=500, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=1.0,
                    random_state=42, n_jobs=-1,
                )
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42,
                )
        else:
            self.model = Ridge(alpha=1.0)

        # Wrap in tuner if applicable
        self._is_tunable = model_type in ("lightgbm", "xgboost", "catboost", "random_forest", "gradient_boosting")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TotalPointsPredictor":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self._extract_importance()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates.
        Returns (predictions, standard_deviations).
        """
        preds = self.predict(X)
        # Simple heuristic uncertainty based on model type
        if hasattr(self.model, "estimators_"):
            # Tree ensemble: use tree variance
            all_preds = np.array([tree.predict(self.scaler.transform(X))
                                  for tree in self.model.estimators_])
            uncertainties = np.std(all_preds, axis=0)
        elif isinstance(self.model, BayesianRidge):
            _, std = self.model.predict(self.scaler.transform(X), return_std=True)
            uncertainties = std
        else:
            # Default: use residual-based estimate
            uncertainties = np.ones_like(preds) * 8.0  # ~8 points std

        return preds, uncertainties

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        return {
            "mae": mean_absolute_error(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
            "r2": r2_score(y, preds),
            "mean_pred": np.mean(preds),
            "mean_actual": np.mean(y),
            "bias": np.mean(preds - y),
            "correlation": np.corrcoef(preds, y)[0, 1] if len(preds) > 1 else 0,
        }

    def _extract_importance(self):
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            self.feature_importance = self.model.coef_
        elif hasattr(self.model, "feature_importances_") is False and hasattr(self.model, "coef_") is False:
            self.feature_importance = np.array([])


# ═════════════════════════════════════════════════════════════════════════
#  SPREAD PREDICTOR
# ═════════════════════════════════════════════════════════════════════════


class SpreadPredictor(BasePredictor):
    """
    Predicts point spread (margin of victory).
    Uses LightGBM for better spread modeling with uncertainty.
    """

    def __init__(self, model_type: str = "lightgbm"):
        super().__init__("SpreadPred")
        self.model_type = model_type

        if model_type == "lightgbm" and LGBM_AVAILABLE:
            self.model = LGBMRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=2.0, reg_lambda=2.0,
                min_child_samples=15, random_state=42, n_jobs=-1,
                verbose=-1,
            )
        elif model_type == "catboost" and CATBOOST_AVAILABLE:
            self.model = CatBoostRegressor(
                iterations=400, depth=4, learning_rate=0.05,
                subsample=0.8, l2_leaf_reg=5.0,
                random_seed=42, verbose=0,
            )
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=2.0, reg_lambda=2.0,
                random_state=42, n_jobs=-1,
            )
        elif model_type == "ridge":
            self.model = Ridge(alpha=2.0)
        else:
            self.model = LGBMRegressor(n_estimators=400, max_depth=4, verbose=-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpreadPredictor":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
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
        preds = self.predict(X)
        ats_correct = np.mean((preds > 0) == (y > 0))
        return {
            "mae": mean_absolute_error(y, preds),
            "rmse": np.sqrt(mean_squared_error(y, preds)),
            "r2": r2_score(y, preds),
            "ats_accuracy": ats_correct,
            "mean_pred": np.mean(preds),
            "mean_actual": np.mean(y),
            "correlation": np.corrcoef(preds, y)[0, 1] if len(preds) > 1 else 0,
        }


# ═════════════════════════════════════════════════════════════════════════
#  MOMENTUM REVERSION MODEL (Classification)
# ═════════════════════════════════════════════════════════════════════════


class MomentumModel(BasePredictor):
    """
    Models momentum reversion using gradient boosting.
    Edge hypothesis: Teams on streaks are overvalued by the market.
    Provides calibrated win probabilities.
    """

    def __init__(self, model_type: str = "lightgbm", calibrate: bool = True):
        super().__init__("Momentum")
        self.model_type = model_type
        self.calibrate = calibrate
        self._base_model = None

        if model_type == "lightgbm" and LGBM_AVAILABLE:
            self.model = LGBMClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                min_child_samples=15, class_weight="balanced",
                random_state=42, n_jobs=-1, verbose=-1,
            )
        elif model_type == "catboost" and CATBOOST_AVAILABLE:
            self.model = CatBoostClassifier(
                iterations=400, depth=4, learning_rate=0.05,
                subsample=0.8, l2_leaf_reg=3.0,
                class_weights=[1, 1.5], random_seed=42,
                verbose=0, early_stopping_rounds=50,
            )
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                scale_pos_weight=1.2, random_state=42, n_jobs=-1,
            )
        else:
            self.model = LogisticRegression(C=0.5, class_weight="balanced", random_state=42)

        self._base_model = self.model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MomentumModel":
        X_scaled = self.scaler.fit_transform(X)

        if self.calibrate and not isinstance(self._base_model, LogisticRegression):
            # Use calibrated classifier for better probability estimates
            self.model = CalibratedClassifierCV(
                self._base_model, method="sigmoid", cv=3
            )

        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self._extract_importance()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        elif hasattr(self.model, "decision_function"):
            # Convert decision function to probability via sigmoid
            scores = self.model.decision_function(X_scaled)
            probs = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack((1 - probs, probs))
        return np.ones((X_scaled.shape[0], 2)) * 0.5

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        proba = self.predict_proba(X)
        metrics = {
            "accuracy": accuracy_score(y, preds),
            "log_loss": log_loss(y, proba),
            "brier": brier_score_loss(y, proba[:, 1]),
            "class_balance": float(np.mean(preds)),
        }
        try:
            metrics["roc_auc"] = roc_auc_score(y, proba[:, 1])
        except Exception:
            metrics["roc_auc"] = 0.5
        return metrics

    def _extract_importance(self):
        """Extract feature importance from the model."""
        if hasattr(self._base_model, "feature_importances_"):
            self.feature_importance = self._base_model.feature_importances_
        elif hasattr(self._base_model, "coef_"):
            self.feature_importance = self._base_model.coef_[0]


# ═════════════════════════════════════════════════════════════════════════
#  BAYESIAN RIDGE REGRESSOR (Uncertainty Quantification)
# ═════════════════════════════════════════════════════════════════════════


class BayesianRidgeRegressor(BasePredictor):
    """
    Bayesian Ridge Regression with uncertainty estimation.
    Provides prediction intervals naturally through the Bayesian framework.
    """

    def __init__(self, alpha_init: float = 1.0, lambda_init: float = 1.0):
        super().__init__("BayesianRidge")
        self.model = BayesianRidge(
            alpha_1=1e-6, alpha_2=1e-6,
            lambda_1=1e-6, lambda_2=1e-6,
            alpha_init=alpha_init, lambda_init=lambda_init,
            compute_score=True,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianRidgeRegressor":
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

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (mean, std, 95% confidence interval width).
        """
        X_scaled = self.scaler.transform(X)
        mean, std = self.model.predict(X_scaled, return_std=True)
        return mean, std, std * 1.96


# ═════════════════════════════════════════════════════════════════════════
#  ENSEMBLE / STACKING
# ═════════════════════════════════════════════════════════════════════════


class StackingEnsemblePredictor(BasePredictor):
    """
    Stacking ensemble that combines multiple base models with a meta-learner.
    Uses diverse model types to reduce variance and improve accuracy.
    """

    def __init__(self, prediction_type: str = "regression"):
        super().__init__("StackingEnsemble")
        self.prediction_type = prediction_type
        self.base_models: List[BasePredictor] = []
        self.meta_model = None
        self.base_predictions_train: Optional[np.ndarray] = None

    def add_base_model(self, model: BasePredictor):
        """Add a base model to the ensemble."""
        self.base_models.append(model)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemblePredictor":
        """Fit all base models and the meta-learner."""
        X_scaled = self.scaler.fit_transform(X)

        # Train base models
        base_preds = []
        for model in self.base_models:
            try:
                model.fit(X_scaled, y)
                train_preds = model.predict(X_scaled)
                base_preds.append(train_preds)
            except Exception as e:
                print(f"  [!] Base model {model.name} failed: {e}")
                continue

        if len(base_preds) == 0:
            raise ValueError("No base models could be fitted")

        self.base_predictions_train = np.column_stack(base_preds)

        # Train meta-learner on out-of-fold predictions would be better,
        # but for simplicity we use a Ridge/LR meta-model
        if self.prediction_type == "regression":
            self.meta_model = Ridge(alpha=0.5)
        else:
            self.meta_model = LogisticRegression(C=1.0, class_weight="balanced")

        self.meta_model.fit(self.base_predictions_train, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_scaled = self.scaler.transform(X)

        base_preds = []
        for model in self.base_models:
            try:
                base_preds.append(model.predict(X_scaled))
            except Exception:
                continue

        if len(base_preds) == 0:
            return np.zeros(X.shape[0])

        meta_features = np.column_stack(base_preds)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """For classification ensembles."""
        X_scaled = self.scaler.transform(X)
        base_preds = []

        for model in self.base_models:
            try:
                if hasattr(model, "predict_proba"):
                    base_preds.append(model.predict_proba(X_scaled))
                else:
                    p = model.predict(X_scaled)
                    base_preds.append(np.column_stack((1 - p, p)))
            except Exception:
                continue

        if len(base_preds) == 0:
            return np.ones((X.shape[0], 2)) * 0.5

        # Average probabilities from base models
        avg_probs = np.mean(np.array(base_preds), axis=0)
        return avg_probs


# ═════════════════════════════════════════════════════════════════════════
#  OPTUNA HYPERPARAMETER TUNING UTILITY
# ═════════════════════════════════════════════════════════════════════════


def create_tuned_lgbm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
    name: str = "LGBM_Tuned",
    verbose: bool = True,
) -> TotalPointsPredictor:
    """
    Create a tuned LightGBM regressor using Optuna.
    """
    if not OPTUNA_AVAILABLE:
        if verbose:
            print("  [!] Optuna not available. Creating default LGBM.")
        return TotalPointsPredictor("lightgbm")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "verbosity": -1,
            "random_state": 42,
            "n_jobs": -1,
        }

        model = LGBMRegressor(**params)
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X_train, y_train, cv=tscv,
                                  scoring="neg_mean_absolute_error", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        study_name=name,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    except Exception as e:
        if verbose:
            print(f"  [!] Tuning failed: {e}")
        return TotalPointsPredictor("lightgbm")

    if verbose:
        print(f"  Tuned LGBM: MAE={study.best_value:.2f} ({n_trials} trials)")

    best = study.best_params
    model_params = {k: v for k, v in best.items() if k not in ("num_leaves",)}
    model_params["num_leaves"] = best.get("num_leaves", 31)
    model_params["verbosity"] = -1
    model_params["random_state"] = 42
    model_params["n_jobs"] = -1

    predictor = TotalPointsPredictor("lightgbm")
    predictor.model = LGBMRegressor(**model_params)
    predictor.training_history["optuna_best"] = study.best_value

    return predictor


# ═════════════════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ═════════════════════════════════════════════════════════════════════════


def create_best_model(prediction_type: str, X_train: Optional[np.ndarray] = None,
                       y_train: Optional[np.ndarray] = None,
                       tune: bool = False) -> BasePredictor:
    """
    Create the best available model for the given prediction type.
    Automatically selects between LightGBM, CatBoost, XGBoost, and Ridge
    based on what's available.

    Args:
        prediction_type: 'regression' or 'classification'
        X_train: Optional training data for tuning
        y_train: Optional training targets for tuning
        tune: Whether to run hyperparameter tuning

    Returns:
        Best available predictor
    """
    if prediction_type == "regression":
        if LGBM_AVAILABLE:
            if tune and X_train is not None and OPTUNA_AVAILABLE:
                return create_tuned_lgbm_regressor(X_train, y_train)
            return TotalPointsPredictor("lightgbm")
        elif CATBOOST_AVAILABLE:
            return TotalPointsPredictor("catboost")
        elif XGBOOST_AVAILABLE:
            return TotalPointsPredictor("xgboost")
        else:
            return TotalPointsPredictor("ridge")
    else:
        if LGBM_AVAILABLE:
            model = MomentumModel("lightgbm", calibrate=True)
            if tune and X_train is not None and OPTUNA_AVAILABLE:
                print("  Tuning Momentum model with Optuna...")
                # Simple tuning for classification
                def obj(trial):
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
                        "max_depth": trial.suggest_int("max_depth", 3, 7),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
                        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
                        "verbosity": -1,
                        "random_state": 42,
                        "n_jobs": -1,
                    }
                    m = LGBMClassifier(**params)
                    tscv = TimeSeriesSplit(n_splits=3)
                    scores = cross_val_score(m, X_train, y_train, cv=tscv,
                                              scoring="accuracy", n_jobs=1)
                    return scores.mean()

                study = optuna.create_study(direction="maximize",
                                            sampler=optuna.samplers.TPESampler(seed=42))
                try:
                    study.optimize(obj, n_trials=15, show_progress_bar=False)
                    best = study.best_params
                    best["verbosity"] = -1
                    best["random_state"] = 42
                    best["n_jobs"] = -1
                    model.model = LGBMClassifier(**best)
                    print(f"  Tuned: accuracy={study.best_value:.4f}")
                except Exception:
                    pass
            return model
        elif CATBOOST_AVAILABLE:
            return MomentumModel("catboost", calibrate=True)
        elif XGBOOST_AVAILABLE:
            return MomentumModel("xgboost", calibrate=True)
        else:
            return MomentumModel(calibrate=True)
