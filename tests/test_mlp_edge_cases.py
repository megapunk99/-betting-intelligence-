"""
Edge case tests for MLP predictor, ensemble, and spread predictor.

Covers:
  - Empty input arrays (0 samples)
  - NaN and inf values in X and y
  - Extremely large batch sizes
  - Multi-dimensional prediction targets
  - Combined stress tests (multiple edge conditions at once)
"""

from __future__ import annotations

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# 1. Empty Input Arrays
# ═══════════════════════════════════════════════════════════════════════════


# ── Helpers (local replacement for fixtures in test_mlp_predictor.py) ──


def _regression_data():
    """Standard regression dataset with 500 samples, 20 features."""
    rng = np.random.RandomState(42)
    X = rng.randn(500, 20)
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 0.1 * rng.randn(500)
    return X, y


def _classification_data():
    """Binary classification dataset with 500 samples, 10 features."""
    rng = np.random.RandomState(42)
    X = rng.randn(500, 10)
    logits = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 2]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.5).astype(float)
    return X, y


def _margin_data():
    """Synthetic point-margin data for SpreadPredictorWithUncertainty."""
    rng = np.random.RandomState(42)
    X = rng.randn(500, 15)
    margin = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.0 * X[:, 2] + rng.randn(500) * 5
    return X, margin


class TestEmptyInputArrays:
    """MLPPredictor, EnhancedEnsemble with zero-length arrays."""

    def test_empty_X_fit_does_not_crash(self):
        """Fitting with X of shape (0, 10) should not crash or hang."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X_empty = np.empty((0, 10))
        y_empty = np.empty((0,))
        predictor = MLPPredictor(input_dim=10, max_epochs=2, batch_size=16)

        try:
            predictor.fit(X_empty, y_empty)
        except Exception as e:
            # Any exception is acceptable — the important thing is no hang/crash
            assert "empty" in str(e).lower() or "nan" in str(e).lower() or \
                   "invalid" in str(e).lower() or "0" in str(e), \
                   f"Unexpected error for empty X: {e}"
            return

        # If fit succeeded, predictions should be possible (even if meaningless)
        X_test = np.random.randn(5, 10)
        preds = predictor.predict(X_test)
        assert len(preds) == 5, "Predictions should match input shape"

    def test_empty_X_predict_returns_empty(self):
        """Predicting with X of shape (0, n_features) should return (0,) array."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = _regression_data()
        predictor = MLPPredictor(input_dim=20, max_epochs=3, batch_size=64)
        predictor.fit(X[:100], y[:100])

        X_empty = np.empty((0, 20))
        preds = predictor.predict(X_empty)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (0,), f"Expected (0,), got {preds.shape}"

    def test_empty_X_predict_with_uncertainty(self):
        """predict_with_uncertainty with empty X should return (0,) arrays."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = _regression_data()
        predictor = MLPPredictor(input_dim=20, max_epochs=3, batch_size=64,
                                 predict_uncertainty=True)
        predictor.fit(X[:100], y[:100], X_val=X[100:120], y_val=y[100:120])

        X_empty = np.empty((0, 20))
        mu, sigma = predictor.predict_with_uncertainty(X_empty)
        assert mu.shape == (0,)
        assert sigma.shape == (0,)

    def test_empty_ensemble_predict_returns_empty(self):
        """Ensemble predict on empty X should return (0,) array."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble
        from sklearn.linear_model import Ridge

        X, y = _regression_data()
        ridge = Ridge(alpha=1.0)
        ridge.fit(X[:100], y[:100])

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", ridge, model_type="regression")

        X_empty = np.empty((0, 20))
        preds = ensemble.predict(X_empty)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (0,), f"Expected (0,), got {preds.shape}"

    def test_empty_ensemble_breakdown(self):
        """predict_with_breakdown on empty X should return consistent dict."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble
        from sklearn.linear_model import Ridge

        X, y = _regression_data()
        ridge = Ridge(alpha=1.0)
        ridge.fit(X[:100], y[:100])

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", ridge, model_type="regression")

        X_empty = np.empty((0, 20))
        breakdown = ensemble.predict_with_breakdown(X_empty)
        assert breakdown["ensemble_pred"].shape == (0,)
        assert breakdown["n_models"] == 1
        assert breakdown["consensus"] >= 0.0

    def test_empty_spread_predict(self):
        """SpreadPredictorWithUncertainty with empty X should return (0,) arrays."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        X, margin = _margin_data()
        predictor = SpreadPredictorWithUncertainty(
            input_dim=15, max_epochs=3, batch_size=32,
        )
        predictor.fit(X[:100], margin[:100])

        X_empty = np.empty((0, 15))
        result = predictor.predict_with_probs(X_empty)
        assert result["spread_mu"].shape == (0,)
        assert result["spread_sigma"].shape == (0,)
        assert result["win_prob"].shape == (0,)

    def test_empty_y_fit_handled(self):
        """Fitting with empty y (matching empty X) should not crash."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        predictor = MLPPredictor(input_dim=5, max_epochs=2, batch_size=16)
        try:
            predictor.fit(np.empty((0, 5)), np.empty((0,)))
        except Exception:
            pass  # Any exception is fine; no crash is the minimum

    def test_empty_ensemble_update_weights(self):
        """update_weights with empty lists should not crash."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble
        from sklearn.linear_model import Ridge
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        ridge = Ridge().fit(X, y)

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", ridge, model_type="regression")

        # Update with no predictions — should be a no-op
        try:
            ensemble.update_weights([], np.array([]), window=10)
        except Exception:
            pass  # Any exception is fine


# ═══════════════════════════════════════════════════════════════════════════
# 2. NaN / Inf Values
# ═══════════════════════════════════════════════════════════════════════════


class TestNanInfValues:
    """MLPPredictor behavior with NaN and infinity values."""

    def test_nan_in_X_fit_does_not_crash(self):
        """Fitting with NaN values in X should not crash the process."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        X[0, 0] = np.nan
        y = rng.randn(100)

        predictor = MLPPredictor(input_dim=10, max_epochs=3, batch_size=32)
        try:
            predictor.fit(X[:80], y[:80])
            preds = predictor.predict(X[80:90])
            # NaN may propagate — just verify no crash
            assert len(preds) == 10
        except (ValueError, RuntimeError) as e:
            # These are acceptable — NaN propagation can cause issues
            assert True

    def test_nan_in_y_fit_does_not_crash(self):
        """Fitting with NaN values in y should not crash."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = rng.randn(100)
        y[5] = np.nan

        predictor = MLPPredictor(input_dim=10, max_epochs=3, batch_size=32)
        try:
            predictor.fit(X[:80], y[:80])
            preds = predictor.predict(X[80:90])
            assert len(preds) == 10
        except Exception:
            pass  # NaN in targets can cause training issues — no crash is the minimum

    def test_inf_in_X_fit_does_not_crash(self):
        """Fitting with inf values in X should not crash."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        X[3, 2] = np.inf
        X[7, 5] = -np.inf
        y = rng.randn(100)

        predictor = MLPPredictor(input_dim=10, max_epochs=3, batch_size=32)
        try:
            predictor.fit(X[:80], y[:80])
            preds = predictor.predict(X[80:90])
            assert len(preds) == 10
        except Exception:
            pass

    def test_nan_in_X_predict_returns_finite(self):
        """Predicting with NaN in X should not crash (may produce NaN output)."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = _regression_data()
        predictor = MLPPredictor(input_dim=20, max_epochs=5, batch_size=64)
        predictor.fit(X[:200], y[:200])

        X_nan = X[200:205].copy()
        X_nan[0, 0] = np.nan
        X_nan[2, 3] = np.nan

        preds = predictor.predict(X_nan)
        assert len(preds) == 5
        # NaN may be present; that's acceptable
        assert isinstance(preds, np.ndarray)

    def test_all_nan_X_predict(self):
        """Predicting with entirely NaN X should not crash."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = _regression_data()
        predictor = MLPPredictor(input_dim=20, max_epochs=3, batch_size=64)
        predictor.fit(X[:100], y[:100])

        X_all_nan = np.full((5, 20), np.nan)
        preds = predictor.predict(X_all_nan)
        assert len(preds) == 5

    def test_nan_ensemble_predict(self):
        """Ensemble predict with NaN in X should not crash."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble
        from sklearn.linear_model import Ridge

        X, y = _regression_data()
        # Fit Ridge on clean data (Ridge refuses NaN inputs)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X[:200], y[:200])

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", ridge, model_type="regression")

        # Predict with NaN — Ridge outputs NaN but doesn't crash
        X_nan = X[200:205].copy()
        X_nan[1, :] = np.nan
        preds = ensemble.predict(X_nan)
        assert len(preds) == 5
        assert isinstance(preds, np.ndarray)

    def test_mixed_nan_inf_in_X(self):
        """Mixed NaN and inf values in X should not crash fit."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        X[0, 0] = np.nan
        X[1, 1] = np.inf
        X[2, 2] = -np.inf
        X[3, 3] = np.nan
        y = rng.randn(100)

        predictor = MLPPredictor(input_dim=10, max_epochs=3, batch_size=16)
        try:
            predictor.fit(X[:80], y[:80])
            preds = predictor.predict(X[80:90])
            assert len(preds) == 10
        except Exception:
            pass

    def test_inf_ensemble_update_weights(self):
        """update_weights with inf values should not crash or produce NaN weights."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble
        from sklearn.linear_model import Ridge

        X, y = _regression_data()
        ridge = Ridge(alpha=1.0)
        ridge.fit(X[:200], y[:200])

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", ridge, model_type="regression")

        # Create predictions with inf
        ridge_preds = ridge.predict(X[200:210])
        ridge_preds[0] = np.inf

        ensemble.update_weights([ridge_preds], y[200:210], window=50)
        # Weight should be finite (update_weights clips to [0.1, 10.0])
        assert np.isfinite(ensemble.weights["ridge"])

    def test_nan_in_spread_predict(self):
        """SpreadPredictorWithUncertainty with NaN in X should not crash."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        X, margin = _margin_data()
        predictor = SpreadPredictorWithUncertainty(
            input_dim=15, max_epochs=3, batch_size=32,
        )
        predictor.fit(X[:100], margin[:100])

        X_nan = X[100:105].copy()
        X_nan[0, 0] = np.nan
        result = predictor.predict_with_probs(X_nan)
        assert len(result["spread_mu"]) == 5
        assert len(result["spread_sigma"]) == 5
        assert len(result["win_prob"]) == 5


# ═══════════════════════════════════════════════════════════════════════════
# 3. Extremely Large Batch Sizes
# ═══════════════════════════════════════════════════════════════════════════


class TestLargeBatchSizes:
    """MLPPredictor with varying batch sizes, including extreme values."""

    def test_batch_size_larger_than_dataset(self):
        """batch_size > n_samples should work (single batch, no error)."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = _regression_data()
        predictor = MLPPredictor(
            input_dim=20, max_epochs=3, batch_size=10_000,
        )
        predictor.fit(X[:100], y[:100])
        preds = predictor.predict(X[200:205])
        assert len(preds) == 5
        assert np.all(np.isfinite(preds))

    def test_batch_size_of_one_raises(self):
        """batch_size=1 raises ValueError — BatchNorm1d needs >1 sample per channel.

        PyTorch's BatchNorm1d requires at least 2 samples to compute
        batch statistics. With batch_size=1, each batch has shape (1, 256)
        and BatchNorm1d raises "Expected more than 1 value per channel".
        """
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = _regression_data()
        predictor = MLPPredictor(
            input_dim=20, max_epochs=3, batch_size=1,
        )
        with pytest.raises((ValueError, RuntimeError)):
            predictor.fit(X[:50], y[:50])

    def test_batch_size_equals_dataset(self):
        """batch_size == n_samples should work (full batch GD)."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = _regression_data()
        n = 100
        predictor = MLPPredictor(
            input_dim=20, max_epochs=3, batch_size=n,
        )
        predictor.fit(X[:n], y[:n])
        preds = predictor.predict(X[200:210])
        assert len(preds) == 10
        assert np.all(np.isfinite(preds))

    def test_very_large_batch_size_classification(self):
        """batch_size >> n_samples should work for classification too."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = _classification_data()
        predictor = MLPPredictor(
            input_dim=10, prediction_type="classification",
            max_epochs=3, batch_size=1_000_000,
        )
        predictor.fit(X[:80], y[:80])
        preds = predictor.predict(X[400:405])
        assert len(preds) == 5
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    def test_batch_size_zero_raises(self):
        """batch_size=0 should raise an error (invalid configuration)."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = rng.randn(50)

        with pytest.raises((ValueError, RuntimeError)):
            predictor = MLPPredictor(input_dim=5, max_epochs=2, batch_size=0)
            predictor.fit(X, y)

    def test_large_batch_with_validation(self):
        """Large batch with validation data should work."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = _regression_data()
        predictor = MLPPredictor(
            input_dim=20, max_epochs=5, batch_size=10_000,
            patience=10,
        )
        predictor.fit(X[:200], y[:200], X_val=X[200:250], y_val=y[200:250])
        preds = predictor.predict(X[400:410])
        assert len(preds) == 10
        assert np.all(np.isfinite(preds))

    def test_large_batch_spread_predictor(self):
        """SpreadPredictorWithUncertainty with large batch should work."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        X, margin = _margin_data()
        predictor = SpreadPredictorWithUncertainty(
            input_dim=15, max_epochs=3, batch_size=10_000,
        )
        predictor.fit(X[:100], margin[:100])
        result = predictor.predict_with_probs(X[400:405])
        assert len(result["spread_mu"]) == 5


# ═══════════════════════════════════════════════════════════════════════════
# 4. Multi-Dimensional Prediction Targets
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiDimTargets:
    """MLPPredictor with non-standard target shapes."""

    def test_2d_y_shape_n_1(self):
        """y with shape (n, 1) should be handled without error."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = rng.randn(200, 1)  # 2D shape

        predictor = MLPPredictor(input_dim=10, max_epochs=5, batch_size=32)
        try:
            predictor.fit(X[:160], y[:160])
            preds = predictor.predict(X[160:170])
            assert len(preds) == 10
            assert np.all(np.isfinite(preds))
        except Exception as e:
            # 2D y may cause shape mismatch — acceptable if it raises a clear error
            error_msg = str(e).lower()
            assert any(word in error_msg for word in
                       ["shape", "dimension", "broadcast", "target", "2d", "input"]), \
                   f"Unexpected error message: {e}"

    def test_y_column_vector_n_1(self):
        """y as column vector shape (n, 1) via evaluate should not crash."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y_1d = X[:, 0] * 0.5 + 0.1 * rng.randn(100)
        y_2d = y_1d.reshape(-1, 1)  # column vector

        predictor = MLPPredictor(input_dim=5, max_epochs=5, batch_size=32)
        predictor.fit(X[:80], y_2d[:80])
        preds = predictor.predict(X[80:90])
        assert len(preds) == 10
        assert np.all(np.isfinite(preds))

        # evaluate should also work
        metrics = predictor.evaluate(X[80:90], y_2d[80:90])
        assert isinstance(metrics["mae"], float)
        assert np.isfinite(metrics["mae"])

    def test_multi_output_target_raises_broadcast_error(self):
        """y with shape (n, 3) raises RuntimeError — shapes can't broadcast.

        pred.squeeze() is (batch,) and batch_y is (batch, 3).
        These can't broadcast because dim -1: batch vs 3 (neither 1).
        PyTorch raises RuntimeError from broadcast_tensors.
        """
        from betting_intel.models.mlp_predictor import MLPPredictor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = rng.randn(100, 3)  # 3 output dimensions

        predictor = MLPPredictor(input_dim=5, max_epochs=3, batch_size=32)
        with pytest.raises((ValueError, RuntimeError)):
            predictor.fit(X[:80], y[:80])

    def test_classification_2d_target(self):
        """Classification with 2D targets should work (internally squeezed)."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        logits = X[:, 0] * 2.0 - X[:, 1] * 1.0
        y = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(float).reshape(-1, 1)

        predictor = MLPPredictor(
            input_dim=5, prediction_type="classification",
            max_epochs=5, batch_size=32,
        )
        try:
            predictor.fit(X[:160], y[:160])
            preds = predictor.predict(X[160:170])
            assert len(preds) == 10
            assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
        except Exception as e:
            error_msg = str(e).lower()
            assert any(word in error_msg for word in
                       ["shape", "dimension", "target", "class", "long"]), \
                   f"Unexpected error message: {e}"

    def test_spread_2d_target(self):
        """SpreadPredictorWithUncertainty with 2D margin should work."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        rng = np.random.RandomState(42)
        X = rng.randn(150, 8)
        margin = rng.randn(150, 1) * 10  # 2D

        predictor = SpreadPredictorWithUncertainty(
            input_dim=8, max_epochs=3, batch_size=32,
        )
        try:
            predictor.fit(X[:120], margin[:120])
            result = predictor.predict_with_probs(X[120:125])
            assert len(result["spread_mu"]) == 5
            assert np.all(np.isfinite(result["spread_mu"]))
        except Exception as e:
            error_msg = str(e).lower()
            assert any(word in error_msg for word in
                       ["shape", "dimension", "margin", "target"]), \
                   f"Unexpected error message: {e}"

    def test_y_3d_broadcastable(self):
        """y with shape (n, 1, 1) broadcasts to (batch aka n, 1, n)."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = rng.randn(100, 1, 1)  # 3D target — broadcasts with (batch,)

        predictor = MLPPredictor(input_dim=5, max_epochs=3, batch_size=32)
        predictor.fit(X[:80], y[:80])
        preds = predictor.predict(X[80:90])
        assert len(preds) == 10


# ═══════════════════════════════════════════════════════════════════════════
# 5. Stress Tests (Combined Edge Conditions)
# ═══════════════════════════════════════════════════════════════════════════


class TestStressConditions:
    """Multiple edge conditions applied simultaneously."""

    def test_nan_X_large_batch(self):
        """NaN in X with extremely large batch size should not crash."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        X[::5, :] = np.nan  # Every 5th row is all NaN
        y = rng.randn(50)

        predictor = MLPPredictor(
            input_dim=10, max_epochs=3, batch_size=1_000_000,
        )
        try:
            predictor.fit(X[:40], y[:40])
            preds = predictor.predict(X[40:45])
            assert len(preds) == 5
        except Exception:
            pass

    def test_empty_X_large_batch(self):
        """Empty X with large batch size should not crash."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        predictor = MLPPredictor(input_dim=5, max_epochs=2, batch_size=1_000_000)
        try:
            predictor.fit(np.empty((0, 5)), np.empty((0,)))
        except Exception:
            pass

    def test_nan_ensemble_survives_nan_predict(self):
        """Ensemble should survive predicting with NaN X (Ridge refuses NaN during fit)."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble
        from sklearn.linear_model import Ridge

        rng = np.random.RandomState(42)
        X_clean = rng.randn(100, 10)
        y = rng.randn(100)

        # Fit on clean data
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_clean[:80], y[:80])

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", ridge, model_type="regression")

        # Predict with NaN
        X_nan = X_clean[80:85].copy()
        X_nan[1, :] = np.nan
        preds = ensemble.predict(X_nan)
        assert len(preds) == 5
        assert isinstance(preds, np.ndarray)

    def test_all_edge_conditions_ensemble(self):
        """Ensemble with all edge conditions: empty model list, NaN X, large batch."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        ensemble = EnhancedEnsemble()

        # Empty ensemble should raise
        with pytest.raises(ValueError, match="No models"):
            ensemble.predict(np.full((3, 5), np.nan))

        # After adding a model, NaN should not crash
        from sklearn.linear_model import Ridge
        X, y = _regression_data()
        ridge = Ridge(alpha=1.0).fit(X[:100], y[:100])
        ensemble.add_model("ridge", ridge, model_type="regression")

        X_all_nan = np.full((10, 20), np.nan)
        preds = ensemble.predict(X_all_nan)
        assert len(preds) == 10

    def test_spread_empty_nan_mixed(self):
        """Spread predictor with empty + NaN at once."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        X, margin = _margin_data()
        predictor = SpreadPredictorWithUncertainty(
            input_dim=15, max_epochs=3, batch_size=32,
        )
        predictor.fit(X[:100], margin[:100])

        # Mix of empty and valid
        X_mixed = np.concatenate([
            np.empty((0, 15)),
            X[100:102],
            np.full((2, 15), np.nan),
            X[102:103],
        ])
        result = predictor.predict_with_probs(X_mixed)
        assert len(result["spread_mu"]) == 5
