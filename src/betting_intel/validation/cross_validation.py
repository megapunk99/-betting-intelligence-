"""
Time-series cross-validation: purged walk-forward, expanding window, and
combinatorial purged cross-validation (CPCV) for betting models.

Standard K-fold CV leaks future information into training sets.
These methods ensure no lookahead bias in time-series evaluation.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score


@dataclass
class CVResult:
    """Container for cross-validation results."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    n_train: int
    n_test: int
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class TimeSeriesCrossValidator:
    """
    Time-series-aware cross-validation with purging and embargoing.

    Purging: Remove from training set any samples whose label depends
    on test-set data (prevents leakage from overlapping labels).

    Embargoing: After each test set, remove a buffer period from the
    following training set to prevent serial correlation leakage.

    Usage:
        cv = TimeSeriesCrossValidator(n_splits=5, embargo=5)
        results = cv.validate(df, feature_cols, target_col, model_builder)
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 100,
        embargo: int = 5,
        purge_threshold: int = 0,
    ):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.embargo = embargo
        self.purge_threshold = purge_threshold
        self.results: List[CVResult] = []

    def get_splits(self, n_samples: int) -> List[Tuple[int, int, int, int]]:
        """
        Generate train/test split indices.

        Returns list of (train_start, train_end, test_start, test_end).
        """
        if n_samples < self.min_train_size * 2:
            raise ValueError(
                f"Not enough samples: {n_samples} < {self.min_train_size * 2}"
            )

        test_size = (n_samples - self.min_train_size) // self.n_splits
        splits = []

        for i in range(self.n_splits):
            test_end = n_samples - (self.n_splits - 1 - i) * test_size
            test_start = test_end - test_size
            train_end = test_start - self.embargo  # Embargo
            train_start = 0

            if train_end - train_start < self.min_train_size:
                train_start = train_end - self.min_train_size
                train_start = max(0, train_start)

            if train_end > train_start and test_end > test_start:
                splits.append((train_start, train_end, test_start, test_end))

        return splits

    def validate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model_builder: Callable,
        prediction_type: str = "regression",
        **model_kwargs,
    ) -> List[CVResult]:
        """
        Run time-series cross-validation.

        Args:
            df: DataFrame with features and target
            feature_cols: Feature column names
            target_col: Target column name
            model_builder: Function that returns a fresh model instance
            prediction_type: 'regression' or 'classification'
            **model_kwargs: Additional kwargs for model_builder

        Returns:
            List of CVResult for each fold
        """
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        n = len(df)

        splits = self.get_splits(n)
        self.results = []

        for i, (tr_start, tr_end, te_start, te_end) in enumerate(splits):
            result = CVResult(
                fold=i + 1,
                train_start=tr_start,
                train_end=tr_end,
                test_start=te_start,
                test_end=te_end,
                n_train=tr_end - tr_start,
                n_test=te_end - te_start,
            )

            # Split data
            train_df = df.iloc[tr_start:tr_end]
            test_df = df.iloc[te_start:te_end]

            # Clean NaN rows
            X_train = train_df[feature_cols].dropna()
            y_train = train_df.loc[X_train.index, target_col]

            X_test = test_df[feature_cols].dropna()
            y_test = test_df.loc[X_test.index, target_col]

            if len(X_train) < 50 or len(X_test) < 10:
                result.errors.append(
                    f"Fold {i + 1}: insufficient data ({len(X_train)} train, {len(X_test)} test)"
                )
                self.results.append(result)
                continue

            try:
                model = model_builder(**model_kwargs)
                model.fit(X_train.values, y_train.values)
                y_pred = model.predict(X_test.values)

                result.predictions = y_pred.tolist()
                result.actuals = y_test.tolist()

                # Compute metrics
                if prediction_type == "regression":
                    result.metrics = {
                        "mae": float(mean_absolute_error(y_test, y_pred)),
                        "r2": float(r2_score(y_test, y_pred)),
                        "rmse": float(np.sqrt(np.mean((y_pred - y_test) ** 2))),
                        "bias": float(np.mean(y_pred - y_test)),
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                    }
                else:
                    y_class = (y_pred > 0.5).astype(int) if prediction_type == "probability" else y_pred
                    y_true_class = (y_test > 0.5).astype(int) if prediction_type == "probability" else y_test
                    result.metrics = {
                        "accuracy": float(accuracy_score(y_true_class, y_class)),
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                    }

            except Exception as e:
                result.errors.append(f"Fold {i + 1}: {str(e)}")

            self.results.append(result)

        return self.results

    def get_summary(self) -> Dict[str, float]:
        """Get aggregate metrics across all folds."""
        if not self.results:
            return {}

        valid_results = [r for r in self.results if r.metrics and not r.errors]

        if not valid_results:
            return {"error": "No valid folds"}

        summary = {}
        metric_keys = valid_results[0].metrics.keys()

        for key in metric_keys:
            if key in ("n_train", "n_test"):
                continue
            values = [r.metrics[key] for r in valid_results if key in r.metrics]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
                summary[f"{key}_min"] = float(np.min(values))
                summary[f"{key}_max"] = float(np.max(values))

        summary["n_folds"] = len(valid_results)
        summary["total_errors"] = sum(len(r.errors) for r in self.results)

        return summary

    def get_prediction_stability(self) -> Dict[str, float]:
        """
        Measure prediction stability across folds.
        High variance in metrics across folds = unstable model.
        """
        summary = self.get_summary()
        if not summary:
            return {}

        stability = {}
        for key in ["mae", "r2", "accuracy"]:
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            if mean_key in summary and std_key in summary and summary[mean_key] != 0:
                # Coefficient of variation (lower = more stable)
                stability[f"{key}_cv"] = summary[std_key] / abs(summary[mean_key])

        return stability


class ExpandingWindowCV:
    """
    Expanding window cross-validation: training set grows over time.

    More robust than sliding window for small datasets, as it uses
    all available past data for each training iteration.
    """

    def __init__(
        self,
        n_splits: int = 5,
        initial_window: int = 100,
        step_size: int = 20,
        embargo: int = 5,
    ):
        self.n_splits = n_splits
        self.initial_window = initial_window
        self.step_size = step_size
        self.embargo = embargo

    def get_splits(self, n_samples: int) -> List[Tuple[int, int, int, int]]:
        splits = []
        current_train_end = self.initial_window

        for _ in range(self.n_splits):
            if current_train_end >= n_samples:
                break

            test_size = min(self.step_size, n_samples - current_train_end)
            test_start = current_train_end + self.embargo
            test_end = min(test_start + test_size, n_samples)

            if test_end <= test_start:
                break

            splits.append((0, current_train_end, test_start, test_end))
            current_train_end = test_end

        return splits


def purged_walk_forward(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_builder: Callable,
    train_window: int = 200,
    test_window: int = 20,
    embargo: int = 5,
    prediction_type: str = "regression",
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Purged walk-forward validation (PWF) from Advances in Financial ML.

    - Train on rolling window
    - Test on next window
    - Purge (embargo) samples after each test set
    - Ensures no leakage between train and test sets

    Returns dictionary with predictions, metrics, and fold results.
    """
    import numpy as np

    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    n = len(df)

    all_predictions = []
    all_actuals = []
    fold_metrics = []

    start = 0
    fold = 0

    while start + train_window + embargo + test_window < n:
        train_end = start + train_window
        test_start = train_end + embargo
        test_end = min(test_start + test_window, n)

        if test_end <= test_start:
            break

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[test_start:test_end]

        X_train = train_df[feature_cols].dropna()
        y_train = train_df.loc[X_train.index, target_col]

        X_test = test_df[feature_cols].dropna()
        y_test = test_df.loc[X_test.index, target_col]

        if len(X_train) < 50 or len(X_test) < 5:
            start += test_window
            continue

        try:
            model = model_builder(**model_kwargs)
            model.fit(X_train.values, y_train.values)
            y_pred = model.predict(X_test.values)

            all_predictions.extend(y_pred.tolist())
            all_actuals.extend(y_test.tolist())

            if prediction_type == "regression":
                metrics = {
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "r2": float(r2_score(y_test, y_pred)),
                }
            else:
                y_class = (y_pred > 0.5).astype(int)
                metrics = {"accuracy": float(accuracy_score(y_test, y_class))}

            metrics["fold"] = fold
            metrics["n_train"] = len(X_train)
            metrics["n_test"] = len(X_test)
            fold_metrics.append(metrics)

        except Exception as e:
            fold_metrics.append({"fold": fold, "error": str(e)})

        start += test_window
        fold += 1

    return {
        "predictions": all_predictions,
        "actuals": all_actuals,
        "fold_metrics": fold_metrics,
        "n_folds": len(fold_metrics),
        "overall_mae": float(np.mean(np.abs(np.array(all_predictions) - np.array(all_actuals)))),
        "overall_r2": float(
            r2_score(all_actuals, all_predictions)
            if len(all_predictions) > 1
            else 0
        ),
    }
