"""
Unit tests for MoneylinePredictor — fit, predict, evaluate, walk-forward CV,
feature importance, error handling, and the train/load wrapper functions.

All tests use synthetic data so they run fast with zero external dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from betting_intel.models.moneyline_predictor import (
    MoneylinePredictor,
    train_moneyline_model,
    load_moneyline_model,
    HAS_XGBOOST,
    HAS_LIGHTGBM,
)


# ═══════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate synthetic binary classification data with 10 features.

    Features 0 and 1 are predictive (correlated with target).
    Features 2-9 are noise.
    """
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 10)
    # y = f0 + f1 > 0 (with noise)
    y = (X[:, 0] + X[:, 1] + np.random.randn(n) * 0.5 > 0).astype(int)
    feature_names = [f"f{i}" for i in range(10)]
    return X, y, feature_names


@pytest.fixture
def synthetic_dataframe(synthetic_data) -> pd.DataFrame:
    """Same data as a DataFrame — for testing DataFrame path in predict_proba."""
    X, y, feature_names = synthetic_data
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


@pytest.fixture
def trained_predictor(synthetic_data) -> MoneylinePredictor:
    """A MoneylinePredictor fitted on synthetic data (no calibration, no feature selection)."""
    X, y, feature_names = synthetic_data
    p = MoneylinePredictor(calibrate=False, feature_selection=False)
    p.fit(X, y, feature_names=feature_names)
    return p


# ═══════════════════════════════════════════════════════════════════════════
#  INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════


class TestInit:
    """Constructor defaults and parameter validation."""

    def test_default_params(self):
        """Default constructor uses sensible defaults."""
        p = MoneylinePredictor()
        assert p.model_types == ["xgboost", "lightgbm", "logistic"]
        assert p.calibrate is True
        assert p.n_folds == 5
        assert p.feature_selection is True
        assert p.n_select == 50
        assert p.random_state == 42
        assert p.is_fitted is False
        assert p.models_ == {}
        assert p.calibrated_models_ == {}
        assert p._feature_importances_ == {}

    def test_custom_params(self):
        """All constructor params can be overridden."""
        p = MoneylinePredictor(
            model_types=["logistic"],
            calibrate=False,
            n_folds=3,
            feature_selection=False,
            n_select=10,
            random_state=7,
        )
        assert p.model_types == ["logistic"]
        assert p.calibrate is False
        assert p.n_folds == 3
        assert p.feature_selection is False
        assert p.n_select == 10
        assert p.random_state == 7

    def test_get_params_returns_config(self):
        """get_params() returns a dict with all config fields."""
        p = MoneylinePredictor()
        params = p.get_params()
        assert params["name"] == "MoneylinePredictor"
        assert params["model_types"] == ["xgboost", "lightgbm", "logistic"]
        assert "is_fitted" in params
        assert "n_models" in params
        assert "model_names" in params
        assert "n_features" in params


# ═══════════════════════════════════════════════════════════════════════════
#  FIT
# ═══════════════════════════════════════════════════════════════════════════


class TestFit:
    """Training logic."""

    def test_fit_returns_self(self, synthetic_data):
        """fit() returns self for chaining."""
        X, y, feature_names = synthetic_data
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        result = p.fit(X, y, feature_names=feature_names)
        assert result is p

    def test_fit_sets_is_fitted(self, synthetic_data):
        """After fit(), is_fitted is True."""
        X, y, feature_names = synthetic_data
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        p.fit(X, y, feature_names=feature_names)
        assert p.is_fitted is True

    def test_fit_trains_all_available_models(self, synthetic_data):
        """Each requested model type that's available gets trained."""
        X, y, _ = synthetic_data
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        p.fit(X, y)

        expected = ["logistic"]
        if HAS_XGBOOST:
            expected.append("xgboost")
        if HAS_LIGHTGBM:
            expected.append("lightgbm")

        for name in expected:
            assert name in p.models_, f"{name} should be in models_"

    def test_fit_stores_feature_names(self, synthetic_data):
        """fit() stores the provided feature names."""
        X, y, feature_names = synthetic_data
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        p.fit(X, y, feature_names=feature_names)
        assert p.feature_names_ == feature_names

    def test_fit_generates_default_feature_names(self, synthetic_data):
        """Without feature_names, fit() generates f0, f1, ... names."""
        X, y, _ = synthetic_data
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        p.fit(X, y)
        assert p.feature_names_ == [f"f{i}" for i in range(X.shape[1])]

    def test_fit_with_calibration(self, synthetic_data):
        """When calibrate=True, calibrated models are also trained."""
        X, y, _ = synthetic_data
        p = MoneylinePredictor(calibrate=True, feature_selection=False)
        p.fit(X, y)
        # Only logistic might be available if XGBoost/LightGBM are missing
        if "logistic" in p.models_:
            assert "logistic" in p.calibrated_models_

    def test_fit_with_feature_selection(self, synthetic_data):
        """Feature selection reduces feature count when n_select < n_features."""
        X, y, feature_names = synthetic_data
        p = MoneylinePredictor(
            calibrate=False, feature_selection=True,
            n_select=5,  # 10 features → keep 5
        )
        p.fit(X, y, feature_names=feature_names)
        assert len(p.selected_feature_names_) == 5
        assert len(p.selected_feature_names_) < len(p.feature_names_)

    def test_fit_empty_data_raises(self):
        """Empty X raises ValueError."""
        p = MoneylinePredictor()
        with pytest.raises(ValueError, match="Empty training data"):
            p.fit(np.array([]), np.array([]))

    def test_fit_length_mismatch_raises(self):
        """X and y with different lengths raises ValueError."""
        p = MoneylinePredictor()
        with pytest.raises(ValueError, match="length mismatch"):
            p.fit(np.array([[1], [2], [3]]), np.array([1, 2]))


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICT_PROBA
# ═══════════════════════════════════════════════════════════════════════════


class TestPredictProba:
    """Probability prediction."""

    def test_predict_proba_before_fit_raises(self, synthetic_data):
        """Calling predict_proba before fit() raises ValueError."""
        X, y, _ = synthetic_data
        p = MoneylinePredictor()
        with pytest.raises(ValueError, match="Not fitted yet"):
            p.predict_proba(X)

    def test_predict_proba_returns_correct_shape(self, trained_predictor, synthetic_data):
        """predict_proba returns (n_samples,) array with values in [0, 1]."""
        X, y, _ = synthetic_data
        probs = trained_predictor.predict_proba(X)
        assert probs.shape == (len(X),)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_predict_proba_from_dataframe(self, trained_predictor, synthetic_dataframe):
        """predict_proba works with pd.DataFrame input (column-name matching path)."""
        df = synthetic_dataframe
        X_df = df[[c for c in df.columns if c != "target"]]
        probs = trained_predictor.predict_proba(X_df)
        assert probs.shape == (len(X_df),)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_predict_proba_without_feature_names(self, synthetic_data):
        """Works without feature_names provided during fit."""
        X, y, _ = synthetic_data
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        p.fit(X, y)  # No feature_names
        probs = p.predict_proba(X)
        assert probs.shape == (len(X),)

    def test_predict_proba_all_models_fail_returns_05(self, trained_predictor):
        """When all model predictions fail, returns 0.5 for each sample."""
        # Force models_ to be empty by patching
        trained_predictor.models_ = {}
        probs = trained_predictor.predict_proba(np.array([[1.0, 2.0]]))
        assert probs.shape == (1,)
        assert probs[0] == 0.5

    def test_dataframe_wrong_columns_warns(self, synthetic_data, caplog):
        """DataFrame with no matching columns logs a warning."""
        import logging
        caplog.set_level(logging.WARNING)
        X, y, _ = synthetic_data
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        p.fit(X, y, feature_names=[f"f{i}" for i in range(10)])

        wrong_df = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])
        probs = p.predict_proba(wrong_df)
        assert probs.shape == (5,)
        # Should have logged a warning
        assert any("Feature name mismatch" in rec.message for rec in caplog.records)


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICT
# ═══════════════════════════════════════════════════════════════════════════


class TestPredict:
    """Binary prediction."""

    def test_predict_returns_binary(self, trained_predictor, synthetic_data):
        """predict() returns 0/1 integers."""
        X, y, _ = synthetic_data
        preds = trained_predictor.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_before_fit_raises(self, synthetic_data):
        """Calling predict before fit() raises ValueError."""
        X, y, _ = synthetic_data
        p = MoneylinePredictor()
        with pytest.raises(ValueError, match="Not fitted yet"):
            p.predict(X)


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATE
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluate:
    """Evaluation metrics."""

    def test_evaluate_returns_all_metrics(self, trained_predictor, synthetic_data):
        """evaluate() returns dict with brier, log_loss, accuracy, n_samples, auc_roc, calibration_error."""
        X, y, _ = synthetic_data
        metrics = trained_predictor.evaluate(X, y)
        assert "brier" in metrics
        assert "log_loss" in metrics
        assert "accuracy" in metrics
        assert "auc_roc" in metrics
        assert "calibration_error" in metrics
        assert "n_samples" in metrics

    def test_evaluate_metrics_in_reasonable_bounds(self, trained_predictor, synthetic_data):
        """Metrics are within expected bounds for synthetic data."""
        X, y, _ = synthetic_data
        metrics = trained_predictor.evaluate(X, y)
        assert 0.0 <= metrics["brier"] <= 0.5
        assert 0.0 <= metrics["log_loss"] <= 1.0
        assert 0.5 <= metrics["accuracy"] <= 1.0
        assert 0.5 <= metrics["auc_roc"] <= 1.0
        assert 0.0 <= metrics["calibration_error"] <= 1.0
        assert metrics["n_samples"] == len(y)

    def test_evaluate_stores_metrics(self, trained_predictor, synthetic_data):
        """evaluate() stores metrics on self.metrics_."""
        X, y, _ = synthetic_data
        metrics = trained_predictor.evaluate(X, y)
        assert trained_predictor.metrics_ == metrics

    def test_evaluate_single_class_only(self, synthetic_data):
        """When y has only one class, auc_roc defaults to 0.5."""
        X, y, fn = synthetic_data
        y_all_ones = np.ones_like(y)
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        p.fit(X, y_all_ones)
        metrics = p.evaluate(X, y_all_ones)
        assert metrics["auc_roc"] == 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossValidate:
    """Walk-forward cross-validation."""

    def test_cross_validate_returns_expected_keys(self, synthetic_data):
        """cross_validate() returns dict with all expected keys."""
        X, y, feature_names = synthetic_data
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        cv_results = p.cross_validate(X, y, feature_names=feature_names, n_splits=3)
        expected_keys = {
            "avg_brier", "avg_log_loss", "avg_accuracy", "avg_auc_roc",
            "n_folds", "n_oos", "fold_metrics",
        }
        assert expected_keys.issubset(cv_results.keys())

    def test_cross_validate_sets_fold_metrics(self, synthetic_data):
        """cross_validate() populates fold_metrics_ with per-fold results."""
        X, y, feature_names = synthetic_data
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        cv_results = p.cross_validate(X, y, feature_names=feature_names, n_splits=3)
        assert cv_results["n_folds"] >= 1
        assert len(p.fold_metrics_) >= 1
        for fold in p.fold_metrics_:
            assert "fold" in fold
            assert "n_train" in fold
            assert "n_test" in fold

    def test_cross_validate_includes_oos_metrics(self, synthetic_data):
        """cross_validate includes pooled OOS metrics when data is sufficient."""
        X, y, feature_names = synthetic_data
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        cv_results = p.cross_validate(X, y, feature_names=feature_names, n_splits=3)
        if cv_results["n_folds"] > 0:
            assert "oos_brier" in cv_results
            assert "oos_accuracy" in cv_results

    def test_cross_validate_refits_on_full_data(self, synthetic_data):
        """After cross_validate(), the predictor is fitted on full data."""
        X, y, feature_names = synthetic_data
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        p.cross_validate(X, y, feature_names=feature_names, n_splits=3)
        assert p.is_fitted is True
        probs = p.predict_proba(X)
        assert probs.shape == (len(X),)

    def test_cross_validate_small_data(self):
        """Very small dataset returns fallback metrics (no folds)."""
        X_small = np.random.randn(30, 5)
        y_small = (X_small[:, 0] > 0).astype(int)
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        cv_results = p.cross_validate(
            X_small, y_small, n_splits=3,
        )
        assert cv_results["n_folds"] == 0
        assert cv_results["avg_brier"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureImportance:
    """Feature importance extraction."""

    def test_get_feature_importance_returns_dict(self, trained_predictor):
        """get_feature_importance() returns dict of {model_name: {feat: score}}."""
        fi = trained_predictor.get_feature_importance(top_n=5)
        assert isinstance(fi, dict)
        for model_name, feat_dict in fi.items():
            assert isinstance(feat_dict, dict)
            assert len(feat_dict) <= 5
            for feat, score in feat_dict.items():
                assert isinstance(feat, str)
                assert isinstance(score, float)
                assert score >= 0

    def test_get_feature_importance_does_not_raise_before_fit(self):
        """Before fit, get_feature_importance() returns empty dict (no crash)."""
        p = MoneylinePredictor()
        fi = p.get_feature_importance()
        assert fi == {}

    def test_top_features_are_most_predictive(self, trained_predictor, synthetic_data):
        """f0 and f1 should appear in top features (they're the predictive columns)."""
        X, y, _ = synthetic_data
        fi = trained_predictor.get_feature_importance(top_n=10)
        # At least one model should rank f0 or f1 in top features
        all_top_features = set()
        for feat_dict in fi.values():
            all_top_features.update(feat_dict.keys())
        assert "f0" in all_top_features or "f1" in all_top_features


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN & LOAD WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainMoneylineModel:
    """train_moneyline_model() wrapper function."""

    def test_train_without_cv(self, synthetic_dataframe):
        """train_moneyline_model(cv=False) trains and returns metrics."""
        df = synthetic_dataframe
        feature_cols = [c for c in df.columns if c != "target"]

        # Patch model_registry.save to avoid disk I/O
        with patch("betting_intel.models.moneyline_predictor._save_moneyline_model"):
            predictor, metrics = train_moneyline_model(
                df, feature_cols, target_col="target",
                calibrate=False, cv=False, save=False,
            )

        assert predictor.is_fitted
        assert "train" in metrics
        assert metrics["train"]["accuracy"] >= 0.5
        assert "brier" in metrics["train"]

    def test_train_with_cv(self, synthetic_dataframe):
        """train_moneyline_model(cv=True) runs walk-forward CV."""
        df = synthetic_dataframe
        feature_cols = [c for c in df.columns if c != "target"]

        with patch("betting_intel.models.moneyline_predictor._save_moneyline_model"):
            predictor, metrics = train_moneyline_model(
                df, feature_cols, target_col="target",
                calibrate=False, cv=True, save=False,
            )

        assert "cv" in metrics
        cv = metrics["cv"]
        assert "avg_brier" in cv
        assert "n_folds" in cv
        assert cv["n_folds"] >= 1

    def test_train_saves_via_registry(self, synthetic_dataframe):
        """When save=True, _save_moneyline_model is called."""
        df = synthetic_dataframe
        feature_cols = [c for c in df.columns if c != "target"]

        with patch("betting_intel.models.moneyline_predictor._save_moneyline_model") as mock_save:
            predictor, metrics = train_moneyline_model(
                df, feature_cols, target_col="target",
                calibrate=False, cv=False, save=True,
            )

        mock_save.assert_called_once()


class TestLoadMoneylineModel:
    """load_moneyline_model() wrapper function."""

    def test_load_model_not_found_returns_none(self):
        """When no model exists, returns None (no crash)."""
        with patch("betting_intel.models.persistence.model_registry.load", side_effect=FileNotFoundError):
            model = load_moneyline_model("nonexistent_model")
        assert model is None

    def test_load_model_wrong_type_returns_none(self):
        """When registry returns wrong type, returns None and logs warning."""
        with patch("betting_intel.models.persistence.model_registry.load", return_value=("not_a_model", {})):
            model = load_moneyline_model("moneyline_ensemble")
        assert model is None


# ═══════════════════════════════════════════════════════════════════════════
#  EDGE CASES & ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_single_feature_does_not_crash(self):
        """Model trains with just 1 feature."""
        X = np.random.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)
        p = MoneylinePredictor(calibrate=False, feature_selection=False)
        p.fit(X, y)
        probs = p.predict_proba(X)
        assert probs.shape == (100,)

    def test_constant_target_calibrates_gracefully(self):
        """All-zeros constant target trains without crashing (degenerate model)."""
        X = np.random.randn(100, 5)
        y = np.zeros(100, dtype=int)
        p = MoneylinePredictor(calibrate=True, feature_selection=False)
        p.fit(X, y)  # Should not raise — models train fine on constant targets
        assert p.is_fitted

    def test_get_feature_importance_after_feature_selection(self, synthetic_data):
        """Feature importance works after feature selection (different indices)."""
        X, y, feature_names = synthetic_data
        p = MoneylinePredictor(
            calibrate=False, feature_selection=True, n_select=5,
        )
        p.fit(X, y, feature_names=feature_names)
        fi = p.get_feature_importance(top_n=3)
        for feat_dict in fi.values():
            for feat in feat_dict.keys():
                assert feat in p.selected_feature_names_

    def test_random_state_reproducibility(self):
        """Same random_state produces same predictions."""
        np.random.seed(123)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        p1 = MoneylinePredictor(calibrate=False, feature_selection=False, random_state=42)
        p1.fit(X, y)
        probs1 = p1.predict_proba(X)

        p2 = MoneylinePredictor(calibrate=False, feature_selection=False, random_state=42)
        p2.fit(X, y)
        probs2 = p2.predict_proba(X)

        np.testing.assert_array_almost_equal(probs1, probs2)
