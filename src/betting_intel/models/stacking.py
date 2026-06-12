"""
Walk-Forward Stacking Ensemble — picklable, reusable, and compatible with
joblib.dump/load for production model storage.

Usage:
    from betting_intel.models.stacking import WalkForwardStackingEnsemble

    ensemble = WalkForwardStackingEnsemble(n_folds=5)
    ensemble.add_model("Ridge", lambda: Ridge(alpha=1.0))
    ensemble.add_model("LightGBM", lambda: LGBMRegressor(...))
    ensemble.fit(X_train, y_train)
    preds = ensemble.predict(X_test)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


class WalkForwardStackingEnsemble:
    """
    Stacking ensemble using walk-forward (chronological) validation.

    How it works:
      1. Split data into n_folds chronologically.
      2. For each fold, train base models on [:fold_end], predict OOS
         on [fold_end:fold_end+fold_size].
      3. Collect ALL out-of-sample predictions as meta-features.
      4. Train a Ridge meta-model on the OOS predictions.
      5. Refit base models on all data for production use.

    The meta-model never sees future data — every meta-training sample
    was predicted by models that had no access to that game's outcome.

    Pickle-safe: can be saved with joblib.dump() and loaded with joblib.load()
    as long as this module is importable in the loading environment.

    Pickle protocol: Uses __getstate__/__setstate__ to exclude the model
    builder functions (which can't be pickled), keeping only the fitted
    model objects and meta-model.
    """

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.base_model_specs: list[tuple[str, callable, dict]] = []
        self._refitted_models: list[tuple[str, object]] = []
        self.meta_model: Ridge | None = None
        self.is_fitted = False
        self.cv_mae: float | None = None
        self.fold_metrics: list[dict] = []

    def __getstate__(self):
        """Exclude non-picklable model builder functions from serialization."""
        state = self.__dict__.copy()
        state["base_model_specs"] = []  # Don't pickle the builder functions
        return state

    def __setstate__(self, state):
        """Restore state, re-initializing non-pickled fields."""
        self.__dict__.update(state)
        if "base_model_specs" not in state:
            self.base_model_specs = []

    def add_model(self, name: str, model_fn: callable, **kwargs):
        """
        Register a base model.

        Args:
            name: Identifier for the model (e.g. "Ridge", "LightGBM")
            model_fn: Callable that returns a fresh unfitted model instance
            **kwargs: Passed to model_fn on every instantiation
        """
        self.base_model_specs.append((name, model_fn, kwargs))

    def fit(self, X: np.ndarray, y: np.ndarray,
            min_train: int = 500) -> WalkForwardStackingEnsemble:
        """
        Walk-forward stacking fit.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Target vector, shape (n_samples,)
            min_train: Minimum training samples required for a fold.
                       Raise this if features >> samples to prevent
                       base models from overfitting on tiny folds.
        """
        n = len(X)
        fold_size = n // self.n_folds

        all_meta_features: list[np.ndarray] = []
        all_meta_targets: list[np.ndarray] = []
        self.fold_metrics = []

        for fold in range(self.n_folds):
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, n)

            if test_start < min_train or test_end - test_start < 10:
                if fold < self.n_folds - 1:
                    continue
                # Last fold: if not enough training data, skip
                if test_start < min_train:
                    break

            X_train_fold = X[:test_start]
            y_train_fold = y[:test_start]
            X_test_fold = X[test_start:test_end]
            y_test_fold = y[test_start:test_end]

            # Train all base models on this fold's training set
            fold_preds = []
            for name, model_fn, kwargs in self.base_model_specs:
                try:
                    model = model_fn(**kwargs)
                    model.fit(X_train_fold, y_train_fold)
                    preds = model.predict(X_test_fold)
                    fold_preds.append(preds)
                except Exception as e:
                    print(f"  ⚠  {name} FAILED on fold {fold + 1}: {e}")
                    # Fill with average of other models' predictions if available
                    if fold_preds:
                        fold_preds.append(np.mean(fold_preds, axis=0))
                    else:
                        fold_preds.append(np.zeros(X_test_fold.shape[0]))

            # Collect OOS meta-features for this fold
            oos_preds = np.column_stack(fold_preds)
            all_meta_features.append(oos_preds)
            all_meta_targets.append(y_test_fold)

            # Simple-average ensemble for fold-level metrics
            ensemble_pred = np.mean([p for p in fold_preds], axis=0)
            fold_mae = float(mean_absolute_error(y_test_fold, ensemble_pred))
            fold_r2 = float(r2_score(y_test_fold, ensemble_pred))
            self.fold_metrics.append({
                "fold": fold + 1,
                "train_size": len(X_train_fold),
                "test_size": len(X_test_fold),
                "mae": fold_mae,
                "r2": fold_r2,
            })

        if not all_meta_features:
            raise ValueError(
                f"Not enough data for {self.n_folds}-fold walk-forward stacking. "
                f"Need at least {min_train + fold_size} samples, got {n}."
            )

        # Train meta-model on ALL out-of-sample predictions
        meta_X = np.vstack(all_meta_features)
        meta_y = np.concatenate(all_meta_targets)

        self.meta_model = Ridge(alpha=1.0, random_state=42)
        self.meta_model.fit(meta_X, meta_y)

        # CV performance of the meta-model on OOS data
        meta_preds = self.meta_model.predict(meta_X)
        self.cv_mae = float(mean_absolute_error(meta_y, meta_preds))
        cv_r2 = float(r2_score(meta_y, meta_preds))
        print(f"      Meta-model CV: MAE={self.cv_mae:.3f}, R2={cv_r2:.4f}")
        print(f"      Meta coefs: {np.round(self.meta_model.coef_, 4)}")

        # Refit base models on ALL data for production use
        refitted = []
        for name, model_fn, kwargs in self.base_model_specs:
            try:
                model = model_fn(**kwargs)
                model.fit(X, y)
                refitted.append((name, model))
            except Exception as e:
                print(f"  ⚠  {name} failed on full-data refit: {e} — excluding from ensemble")
        if not refitted:
            raise RuntimeError("All base models failed during full-data refit — cannot build ensemble")
        self._refitted_models = refitted

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the stacked ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet. Call .fit() first.")

        meta_features = np.column_stack([
            model.predict(X) for _, model in self._refitted_models
        ])
        return self.meta_model.predict(meta_features)

    @property
    def coef_(self) -> np.ndarray | None:
        """Meta-model coefficients (importance of each base model)."""
        if self.meta_model is not None and hasattr(self.meta_model, "coef_"):
            return self.meta_model.coef_
        return None

    @property
    def intercept_(self) -> float:
        """Meta-model intercept."""
        if self.meta_model is not None:
            return float(self.meta_model.intercept_)
        return 0.0


class WinProbEnsemble:
    """
    Simple average of win-probability classifier models.

    Pickle-safe: all member models are standard sklearn/lightgbm/catboost
    objects that serialize correctly.

    Usage:
        models = {
            "lgb": LGBMClassifier(...),
            "cat": CatBoostClassifier(...),
            "lr": CalibratedClassifierCV(...),
        }
        ensemble = WinProbEnsemble(models)
        probs = ensemble.predict_proba(X)  # shape (n, 2)
        preds = ensemble.predict(X)        # shape (n,)
    """

    def __init__(self, models: dict):
        self.models = models

    def predict_proba(self, X):
        """Return probability estimates averaged across all models."""
        import numpy as np
        probs = np.mean([m.predict_proba(X)[:, 1] for m in self.models.values()], axis=0)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        """Return class predictions (0/1)."""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
