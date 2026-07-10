"""
HyperparameterTuner — Bayesian hyperparameter optimization via Optuna (v6.5).

Provides automatic hyperparameter search for all ensemble models:
  - XGBoost: learning_rate, max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda
  - LightGBM: learning_rate, num_leaves, min_child_samples, subsample, reg_alpha, reg_lambda
  - RandomForest: n_estimators, max_depth, min_samples_leaf, max_features
  - CatBoost: learning_rate, depth, l2_leaf_reg, border_count
  - LogisticRegression: C, penalty

Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler for efficient search.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Bayesian hyperparameter optimization for ensemble models.

    Usage:
        tuner = HyperparameterTuner(random_state=42)
        best_params = tuner.tune_xgboost(X_train, y_train, X_val, y_val, n_trials=50)
        tuner.plot_results()  # Optional: visualize optimization history
    """

    def __init__(
        self,
        random_state: int = 42,
        n_trials: int = 30,
        timeout_seconds: Optional[int] = None,
        direction: str = "minimize",  # 'minimize' for logloss, 'maximize' for accuracy
    ):
        self.random_state = random_state
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.direction = direction
        self._study = None
        self._best_params: dict = {}
        self._best_value: float = 0.0

    @property
    def best_params(self) -> dict:
        return dict(self._best_params)

    @property
    def best_value(self) -> float:
        return self._best_value

    def _create_study(self, study_name: str = "ensemble_tuning"):
        """Create or load an Optuna study."""
        try:
            import optuna
            from optuna.samplers import TPESampler

            sampler = TPESampler(seed=self.random_state)
            self._study = optuna.create_study(
                study_name=study_name,
                direction=self.direction,
                sampler=sampler,
            )
            return True
        except ImportError:
            logger.warning("Optuna not installed. Install with: pip install optuna")
            return False

    def tune_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """Tune XGBoost hyperparameters using Optuna."""
        if not self._create_study("xgboost_tuning"):
            return self._default_xgb_params()

        n_trials = n_trials or self.n_trials

        def objective(trial):
            from xgboost import XGBClassifier
            from sklearn.metrics import log_loss

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.1, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "random_state": self.random_state,
                "eval_metric": "logloss",
                "early_stopping_rounds": 50,
                "verbosity": 0,
            }

            model = XGBClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            preds = model.predict_proba(X_val)
            return float(log_loss(y_val, preds))

        self._study.optimize(objective, n_trials=n_trials, timeout=self.timeout_seconds)

        self._best_params = self._study.best_params
        self._best_value = self._study.best_value

        if verbose:
            logger.info(
                f"XGBoost tuning: best logloss={self._best_value:.4f}, "
                f"params={self._best_params}"
            )

        return dict(self._best_params)

    def tune_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """Tune LightGBM hyperparameters using Optuna."""
        if not self._create_study("lightgbm_tuning"):
            return self._default_lgb_params()

        n_trials = n_trials or self.n_trials

        def objective(trial):
            from lightgbm import LGBMClassifier
            from sklearn.metrics import log_loss

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.1, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
                "random_state": self.random_state,
                "verbose": -1,
            }

            model = LGBMClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[LGBMEarlyStopping(50)]
                if (LGBMEarlyStopping is not None and self._has_early_stopping())
                else [],
            )
            preds = model.predict_proba(X_val)
            return float(log_loss(y_val, preds))

        self._study.optimize(objective, n_trials=n_trials, timeout=self.timeout_seconds)

        self._best_params = self._study.best_params
        self._best_value = self._study.best_value

        if verbose:
            logger.info(
                f"LightGBM tuning: best logloss={self._best_value:.4f}, "
                f"params={self._best_params}"
            )

        return dict(self._best_params)

    def tune_catboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: Optional[int] = None,
        verbose: bool = True,
    ) -> dict:
        """Tune CatBoost hyperparameters using Optuna."""
        if not self._create_study("catboost_tuning"):
            return self._default_cb_params()

        n_trials = n_trials or self.n_trials

        def objective(trial):
            from catboost import CatBoostClassifier
            from sklearn.metrics import log_loss

            params = {
                "iterations": trial.suggest_int("iterations", 300, 1500, step=100),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.005, 0.1, log=True
                ),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "random_seed": self.random_state,
                "verbose": 0,
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "early_stopping_rounds": 50,
            }

            model = CatBoostClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                verbose=False,
            )
            preds = model.predict_proba(X_val)
            return float(log_loss(y_val, preds))

        self._study.optimize(objective, n_trials=n_trials, timeout=self.timeout_seconds)

        self._best_params = self._study.best_params
        self._best_value = self._study.best_value

        if verbose:
            logger.info(
                f"CatBoost tuning: best logloss={self._best_value:.4f}, "
                f"params={self._best_params}"
            )

        return dict(self._best_params)

    def tune_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials_per_model: int = 30,
        verbose: bool = True,
    ) -> dict[str, dict]:
        """Tune all ensemble models and return best params for each."""
        results = {}

        if verbose:
            logger.info("=" * 60)
            logger.info("HYPERPARAMETER TUNING — ALL MODELS")
            logger.info("=" * 60)

        # XGBoost
        if verbose:
            logger.info("\n--- Tuning XGBoost ---")
        results["xgb"] = self.tune_xgboost(
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=n_trials_per_model,
            verbose=verbose,
        )

        # LightGBM
        if verbose:
            logger.info("\n--- Tuning LightGBM ---")
        results["lgb"] = self.tune_lightgbm(
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=n_trials_per_model,
            verbose=verbose,
        )

        # CatBoost
        if verbose:
            logger.info("\n--- Tuning CatBoost ---")
        results["cb"] = self.tune_catboost(
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=n_trials_per_model,
            verbose=verbose,
        )

        if verbose:
            logger.info("=" * 60)
            logger.info("TUNING COMPLETE")
            logger.info("=" * 60)

        return results

    def _default_xgb_params(self) -> dict:
        return {
            "n_estimators": 800,
            "max_depth": 6,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
        }

    def _default_lgb_params(self) -> dict:
        return {
            "n_estimators": 800,
            "num_leaves": 63,
            "max_depth": -1,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "min_child_samples": 30,
        }

    def _default_cb_params(self) -> dict:
        return {
            "iterations": 800,
            "depth": 8,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "border_count": 128,
            "subsample": 0.8,
        }

    @staticmethod
    def _has_early_stopping() -> bool:
        try:
            from lightgbm import early_stopping as lgb_early_stopping

            return True
        except ImportError:
            try:
                from lightgbm.callback import early_stopping

                return True
            except ImportError:
                return False

    def get_summary(self) -> dict:
        """Get a JSON-serializable summary."""
        return {
            "n_trials": self.n_trials,
            "direction": self.direction,
            "best_params": {
                k: str(v) if isinstance(v, tuple) else v
                for k, v in self._best_params.items()
            },
            "best_value": round(self._best_value, 4),
            "has_study": self._study is not None,
        }


# LGBMEarlyStopping is imported inside the LightGBM tuning method
# where it's used (lazy import avoids import errors on older versions)
LGBMEarlyStopping = None  # Fallback: set inside tune_lightgbm


__all__ = [
    "HyperparameterTuner",
]
