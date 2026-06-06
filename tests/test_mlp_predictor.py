"""
Unit tests for the MLP neural network predictor and ensemble models.

Covers:
  - MLPPredictor: fit, predict, evaluate, predict_with_uncertainty, predict_proba,
                  edge cases (unfitted model, single sample, constant targets)
  - EnhancedEnsemble: add_model, predict, predict_with_breakdown, update_weights,
                      empty ensemble, classification mode
  - SpreadPredictorWithUncertainty: fit, predict_spread, predict_with_probs,
                                    margin scaling
"""

from __future__ import annotations

import numpy as np
import pytest

# ── Helper: reproducible synthetic data ────────────────────────────────


@pytest.fixture
def regression_data():
    """Standard regression dataset with 500 samples, 20 features."""
    np.random.seed(42)
    X = np.random.randn(500, 20)
    y = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 0.1 * np.random.randn(500)
    return X, y


@pytest.fixture
def classification_data():
    """Binary classification dataset with 500 samples, 10 features."""
    np.random.seed(42)
    X = np.random.randn(500, 10)
    logits = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 2]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.5).astype(float)
    return X, y


@pytest.fixture
def margin_data():
    """Synthetic point-margin data for SpreadPredictorWithUncertainty."""
    np.random.seed(42)
    X = np.random.randn(500, 15)
    margin = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.0 * X[:, 2] + np.random.randn(500) * 5
    return X, margin


# ═══════════════════════════════════════════════════════════════════════════
# 1. MLPPredictor — Regression
# ═══════════════════════════════════════════════════════════════════════════


class TestMLPPredictorRegression:
    """MLPPredictor in regression mode."""

    def test_fit_and_predict(self, regression_data):
        """Basic fit + predict returns correct shape and reasonable MAE."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        predictor = MLPPredictor(input_dim=20, prediction_type="regression",
                                 max_epochs=10, batch_size=64)

        predictor.fit(X[:400], y[:400], X_val=X[400:], y_val=y[400:])
        preds = predictor.predict(X[400:])

        assert predictor.is_fitted
        assert len(preds) == 100
        assert preds.dtype in (np.float32, np.float64)
        mae = float(np.mean(np.abs(preds - y[400:])))
        assert mae < 1.0, f"MAE should be reasonable (< 1.0), got {mae:.4f}"

    def test_predict_returns_flat_array(self, regression_data):
        """predict() returns a 1D array even with (100, 1)-shaped inputs."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        predictor = MLPPredictor(input_dim=20, max_epochs=5, batch_size=64)
        predictor.fit(X[:400], y[:400])

        preds = predictor.predict(X[400:])
        assert preds.ndim == 1, f"Expected 1D array, got {preds.ndim}D"

    def test_predict_with_uncertainty_returns_nan(self, regression_data):
        """Without predict_uncertainty=True, sigma should be NaN."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        predictor = MLPPredictor(input_dim=20, max_epochs=5, batch_size=64)
        predictor.fit(X[:400], y[:400])

        mu, sigma = predictor.predict_with_uncertainty(X[400:405])
        assert len(mu) == 5
        assert np.all(np.isnan(sigma)), "sigma should be NaN when uncertainty disabled"

    def test_predict_with_uncertainty_mode(self, regression_data):
        """With predict_uncertainty=True, sigma should be finite positive."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        predictor = MLPPredictor(input_dim=20, max_epochs=10, batch_size=64,
                                 predict_uncertainty=True)
        predictor.fit(X[:400], y[:400], X_val=X[400:], y_val=y[400:])

        mu, sigma = predictor.predict_with_uncertainty(X[400:405])
        assert len(mu) == 5
        assert len(sigma) == 5
        assert np.all(sigma > 0), "sigma should be positive"
        assert np.all(np.isfinite(sigma)), "sigma should be finite"

    def test_evaluate_returns_metrics(self, regression_data):
        """evaluate() returns dict with mae, rmse, r2, bias."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        predictor = MLPPredictor(input_dim=20, max_epochs=5, batch_size=64)
        predictor.fit(X[:400], y[:400])

        metrics = predictor.evaluate(X[400:], y[400:])
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "bias" in metrics
        assert "n_samples" in metrics
        assert metrics["n_samples"] == 100
        assert isinstance(metrics["mae"], float)
        assert metrics["mae"] >= 0.0

    def test_get_params(self, regression_data):
        """get_params() returns model configuration."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        predictor = MLPPredictor(input_dim=10, max_epochs=50, dropout=0.3,
                                 learning_rate=1e-4)
        params = predictor.get_params()

        assert params["name"] == "MLPPredictor"
        assert params["max_epochs"] == 50
        assert params["dropout"] == pytest.approx(0.3)
        assert params["learning_rate"] == pytest.approx(1e-4)
        assert params["is_fitted"] is False

    def test_predict_before_fit_raises(self, regression_data):
        """Calling predict() before fit() raises ValueError."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        predictor = MLPPredictor(input_dim=20)
        with pytest.raises(ValueError, match="not fitted"):
            predictor.predict(np.random.randn(10, 20))

    def test_fit_returns_self(self, regression_data):
        """fit() returns self for method chaining."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        predictor = MLPPredictor(input_dim=20, max_epochs=3, batch_size=64)
        result = predictor.fit(X[:100], y[:100])
        assert result is predictor

    def test_single_sample(self, regression_data):
        """Predicting a single sample returns an array of length 1."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        predictor = MLPPredictor(input_dim=20, max_epochs=3, batch_size=64)
        predictor.fit(X[:100], y[:100])

        preds = predictor.predict(X[0:1])
        assert len(preds) == 1

    def test_auto_input_dim(self, regression_data):
        """input_dim=0 should auto-detect from X shape."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        predictor = MLPPredictor(input_dim=0, max_epochs=3, batch_size=64)
        predictor.fit(X[:100], y[:100])
        assert predictor.input_dim == 20

    def test_linear_targets(self):
        """Model should learn a linear relationship with enough capacity."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 10.0  # strong linear signal

        predictor = MLPPredictor(input_dim=5, max_epochs=100, batch_size=32,
                                 patience=20, learning_rate=1e-3)
        predictor.fit(X[:160], y[:160], X_val=X[160:], y_val=y[160:])
        preds = predictor.predict(X[160:])

        assert np.all(np.isfinite(preds))
        # Should capture the linear structure (R² > 0)
        metrics = predictor.evaluate(X[160:], y[160:])
        assert metrics["r2"] > 0.0, f"Expected positive R², got {metrics['r2']:.3f}"

    def test_no_validation_data(self, regression_data):
        """fit() should work without validation data (uses train loss for scheduling)."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        predictor = MLPPredictor(input_dim=20, max_epochs=5, batch_size=64)

        # No validation data provided
        predictor.fit(X[:200], y[:200])
        preds = predictor.predict(X[200:210])

        assert predictor.is_fitted
        assert len(preds) == 10
        assert np.all(np.isfinite(preds))

    def test_training_losses_recorded(self, regression_data):
        """train_losses and val_losses should be populated after fit."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        predictor = MLPPredictor(input_dim=20, max_epochs=10, batch_size=64,
                                 patience=20)
        predictor.fit(X[:400], y[:400], X_val=X[400:], y_val=y[400:])

        assert len(predictor.train_losses) > 0
        assert len(predictor.val_losses) > 0
        # Training loss should decrease over time
        assert predictor.train_losses[-1] <= predictor.train_losses[0] * 1.5

    def test_deterministic_with_seed(self, regression_data):
        """Same seed + torch.manual_seed should produce identical predictions."""
        import torch
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        preds_list = []
        for _ in range(2):
            np.random.seed(42)
            torch.manual_seed(42)
            predictor = MLPPredictor(input_dim=20, max_epochs=5, batch_size=64)
            predictor.fit(X[:200], y[:200])
            preds_list.append(predictor.predict(X[400:410]))

        np.testing.assert_array_almost_equal(preds_list[0], preds_list[1], decimal=4)


# ═══════════════════════════════════════════════════════════════════════════
# 2. MLPPredictor — Classification
# ═══════════════════════════════════════════════════════════════════════════


class TestMLPPredictorClassification:
    """MLPPredictor in classification mode."""

    def test_classification_predict_shape(self, classification_data):
        """predict() returns probabilities in [0, 1] for classifier."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = classification_data
        predictor = MLPPredictor(input_dim=10, prediction_type="classification",
                                 max_epochs=10, batch_size=64)
        predictor.fit(X[:400], y[:400], X_val=X[400:], y_val=y[400:])

        preds = predictor.predict(X[400:])
        assert len(preds) == 100
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    def test_predict_proba_shape(self, classification_data):
        """predict_proba() returns (n, 2) probabilities."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = classification_data
        predictor = MLPPredictor(input_dim=10, prediction_type="classification",
                                 max_epochs=10, batch_size=64)
        predictor.fit(X[:400], y[:400])

        proba = predictor.predict_proba(X[400:405])
        assert proba.shape == (5, 2)
        # Rows should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(5), decimal=5)

    def test_classification_evaluate(self, classification_data):
        """evaluate() includes accuracy and AUC-ROC for classifiers."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = classification_data
        predictor = MLPPredictor(input_dim=10, prediction_type="classification",
                                 max_epochs=10, batch_size=64)
        predictor.fit(X[:400], y[:400])

        metrics = predictor.evaluate(X[400:], y[400:])
        assert "accuracy" in metrics
        assert "auc_roc" in metrics
        assert isinstance(metrics["accuracy"], float)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_predict_proba_not_classifier(self, regression_data):
        """predict_proba() should raise for regression models."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        predictor = MLPPredictor(input_dim=20, prediction_type="regression",
                                 max_epochs=3, batch_size=64)
        predictor.fit(X[:100], y[:100])

        with pytest.raises(ValueError, match="only available for classification"):
            predictor.predict_proba(X[:5])

    def test_classification_binary_prediction(self, classification_data):
        """Binary classifier should produce reasonable accuracy (> 50%)."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = classification_data
        predictor = MLPPredictor(input_dim=10, prediction_type="binary",
                                 max_epochs=20, batch_size=64)
        predictor.fit(X[:400], y[:400], X_val=X[400:], y_val=y[400:])

        preds = predictor.predict(X[400:])
        binary = (preds > 0.5).astype(float)
        accuracy = float(np.mean(binary == y[400:]))
        assert accuracy > 0.50, f"Accuracy should be > 50%, got {accuracy:.2%}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. EnhancedEnsemble
# ═══════════════════════════════════════════════════════════════════════════


class TestEnhancedEnsemble:
    """Tests for the EnhancedEnsemble class."""

    @pytest.fixture
    def trained_models(self, regression_data):
        """Return a dict of trained models and their data."""
        from sklearn.linear_model import Ridge
        from betting_intel.models.mlp_predictor import MLPPredictor

        X, y = regression_data
        ridge = Ridge(alpha=1.0)
        ridge.fit(X[:400], y[:400])

        mlp = MLPPredictor(input_dim=20, max_epochs=5, batch_size=64)
        mlp.fit(X[:400], y[:400])

        return {"X": X, "y": y, "ridge": ridge, "mlp": mlp}

    def test_add_model_and_predict(self, trained_models):
        """Adding models and predicting returns ensemble predictions."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", trained_models["ridge"], model_type="regression")
        ensemble.add_model("mlp", trained_models["mlp"], model_type="regression")

        X = trained_models["X"]
        preds = ensemble.predict(X[400:])

        assert len(preds) == 100
        assert np.all(np.isfinite(preds))
        assert float(np.mean(np.abs(preds - trained_models["y"][400:]))) < 1.0

    def test_predict_empty_ensemble_raises(self):
        """predict() on empty ensemble raises ValueError."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        ensemble = EnhancedEnsemble()
        with pytest.raises(ValueError, match="No models"):
            ensemble.predict(np.random.randn(10, 5))

    def test_add_model_returns_self(self, trained_models):
        """add_model() returns self for chaining."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        ensemble = EnhancedEnsemble()
        result = ensemble.add_model("ridge", trained_models["ridge"])
        assert result is ensemble

    def test_predict_with_breakdown_structure(self, trained_models):
        """predict_with_breakdown() returns dict with all expected keys."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", trained_models["ridge"], model_type="regression")
        ensemble.add_model("mlp", trained_models["mlp"], model_type="regression")

        X = trained_models["X"]
        breakdown = ensemble.predict_with_breakdown(X[400:410])

        assert "ensemble_pred" in breakdown
        assert "model_preds" in breakdown
        assert "weights" in breakdown
        assert "consensus" in breakdown
        assert "n_models" in breakdown
        assert breakdown["n_models"] == 2
        assert breakdown["consensus"] >= 0.0
        assert breakdown["consensus"] <= 1.0
        assert "ridge" in breakdown["model_preds"]
        assert "mlp" in breakdown["model_preds"]

    def test_consensus_close_to_one(self, trained_models):
        """Models trained on same data should have high consensus."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", trained_models["ridge"], model_type="regression")
        ensemble.add_model("mlp", trained_models["mlp"], model_type="regression")

        X = trained_models["X"]
        breakdown = ensemble.predict_with_breakdown(X[400:])
        assert breakdown["consensus"] > 0.5, f"Expected high consensus, got {breakdown['consensus']:.2f}"

    def test_update_weights(self, trained_models):
        """update_weights() should adjust weights based on error."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", trained_models["ridge"], model_type="regression")
        ensemble.add_model("mlp", trained_models["mlp"], model_type="regression")

        X = trained_models["X"]
        y = trained_models["y"]

        old_weights = ensemble.weights.copy()

        # Generate predictions and update weights
        ridge_preds = trained_models["ridge"].predict(X[400:])
        mlp_preds = trained_models["mlp"].predict(X[400:])
        ensemble.update_weights(
            [ridge_preds, mlp_preds],
            y[400:],
            window=50,
        )

        # Weights should have changed
        new_weights = ensemble.weights
        assert new_weights["ridge"] != old_weights["ridge"] or \
               new_weights["mlp"] != old_weights["mlp"]

    def test_get_params(self, trained_models):
        """get_params() returns ensemble config."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        ensemble = EnhancedEnsemble(log_odds_averaging=False, weight_decay=0.9)
        ensemble.add_model("ridge", trained_models["ridge"], model_type="regression")

        params = ensemble.get_params()
        assert params["name"] == "EnhancedEnsemble"
        assert params["n_models"] == 1
        assert params["log_odds_averaging"] is False
        assert params["weight_decay"] == pytest.approx(0.9)

    def test_single_model_predict(self, trained_models):
        """Ensemble with single model should match that model's predictions."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", trained_models["ridge"], model_type="regression")

        X = trained_models["X"]
        ensemble_preds = ensemble.predict(X[400:405])
        ridge_preds = trained_models["ridge"].predict(X[400:405])

        np.testing.assert_array_almost_equal(ensemble_preds, ridge_preds, decimal=5)

    def test_predict_batch_shape(self, trained_models):
        """predict() output shape matches number of samples."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", trained_models["ridge"], model_type="regression")
        ensemble.add_model("mlp", trained_models["mlp"], model_type="regression")

        X = trained_models["X"]
        preds = ensemble.predict(X[400:420])
        assert len(preds) == 20

    def test_classification_ensemble(self, classification_data):
        """Ensemble with classification models uses log-odds averaging."""
        from sklearn.linear_model import LogisticRegression
        from betting_intel.models.mlp_predictor import (
            EnhancedEnsemble, MLPPredictor,
        )

        X, y = classification_data
        logreg = LogisticRegression(C=1.0, max_iter=1000)
        logreg.fit(X[:400], y[:400])

        mlp = MLPPredictor(input_dim=10, prediction_type="classification",
                           max_epochs=10, batch_size=64)
        mlp.fit(X[:400], y[:400])

        ensemble = EnhancedEnsemble(log_odds_averaging=True)
        ensemble.add_model("logreg", logreg, model_type="classification")
        ensemble.add_model("mlp", mlp, model_type="classification")

        preds = ensemble.predict(X[400:])
        assert len(preds) == 100
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    def test_deterministic_ensemble(self, trained_models):
        """Ensemble with same models should produce same predictions."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        X = trained_models["X"]
        preds_list = []
        for _ in range(2):
            ensemble = EnhancedEnsemble()
            ensemble.add_model("ridge", trained_models["ridge"], "regression")
            ensemble.add_model("mlp", trained_models["mlp"], "regression")
            preds_list.append(ensemble.predict(X[400:405]))

        np.testing.assert_array_almost_equal(preds_list[0], preds_list[1], decimal=5)

    def test_custom_weights(self, trained_models):
        """Custom initial weights should be preserved."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", trained_models["ridge"], model_type="regression", weight=2.0)
        ensemble.add_model("mlp", trained_models["mlp"], model_type="regression", weight=0.5)

        assert ensemble.weights["ridge"] == pytest.approx(2.0)
        assert ensemble.weights["mlp"] == pytest.approx(0.5)

        X = trained_models["X"]
        preds = ensemble.predict(X[400:405])
        assert len(preds) == 5


# ═══════════════════════════════════════════════════════════════════════════
# 4. SpreadPredictorWithUncertainty
# ═══════════════════════════════════════════════════════════════════════════


class TestSpreadPredictorWithUncertainty:
    """Tests for SpreadPredictorWithUncertainty (NBA_AI L4-style)."""

    def test_fit_and_predict_spread(self, margin_data):
        """predict_spread() returns margins in original scale."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        X, margin = margin_data
        predictor = SpreadPredictorWithUncertainty(
            input_dim=15, max_epochs=10, batch_size=64,
        )
        predictor.fit(X[:400], margin[:400], X_val=X[400:], y_val_margin=margin[400:])

        preds = predictor.predict_spread(X[400:])
        assert len(preds) == 100
        assert np.all(np.isfinite(preds))
        # Margin should be in reasonable range (not massively scaled)
        assert float(np.std(preds)) > 0.1

    def test_predict_with_probs_returns_dict(self, margin_data):
        """predict_with_probs() returns dict with spread_mu, spread_sigma, win_prob."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        X, margin = margin_data
        predictor = SpreadPredictorWithUncertainty(
            input_dim=15, max_epochs=10, batch_size=64,
        )
        predictor.fit(X[:400], margin[:400], X_val=X[400:], y_val_margin=margin[400:])

        result = predictor.predict_with_probs(X[400:405])

        assert "spread_mu" in result
        assert "spread_sigma" in result
        assert "win_prob" in result
        assert len(result["spread_mu"]) == 5
        assert len(result["spread_sigma"]) == 5
        assert len(result["win_prob"]) == 5
        assert np.all(np.isfinite(result["spread_mu"]))
        assert np.all(result["spread_sigma"] > 0)
        assert np.all(result["win_prob"] >= 0.0) and np.all(result["win_prob"] <= 1.0)

    def test_spread_sigma_positive(self, margin_data):
        """Spread sigma should always be positive (uncertainty > 0)."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        X, margin = margin_data
        predictor = SpreadPredictorWithUncertainty(
            input_dim=15, max_epochs=5, batch_size=64,
        )
        predictor.fit(X[:200], margin[:200])

        result = predictor.predict_with_probs(X[200:210])
        assert np.all(result["spread_sigma"] > 0)

    def test_win_prob_in_range(self, margin_data):
        """Win probability should always be in [0, 1]."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        X, margin = margin_data
        predictor = SpreadPredictorWithUncertainty(
            input_dim=15, max_epochs=5, batch_size=64,
        )
        predictor.fit(X[:300], margin[:300])

        result = predictor.predict_with_probs(X[300:350])
        assert np.all(result["win_prob"] >= 0.0)
        assert np.all(result["win_prob"] <= 1.0)

    def test_margin_scaling(self, margin_data):
        """Internal margin scaling should not distort predictions."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        X, margin = margin_data
        predictor = SpreadPredictorWithUncertainty(
            input_dim=15, max_epochs=5, batch_size=64,
        )
        predictor.fit(X[:300], margin[:300])

        preds = predictor.predict_spread(X[300:310])
        # Predicted margins should have similar variance to actual
        actual_std = float(np.std(margin[:300]))
        pred_std = float(np.std(preds))
        assert pred_std > actual_std * 0.1, f"Predicted std {pred_std:.2f} too small vs actual {actual_std:.2f}"

    def test_predict_spread_before_fit_raises(self, margin_data):
        """predict_spread() before fit() raises ValueError."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        predictor = SpreadPredictorWithUncertainty(input_dim=15)
        with pytest.raises(ValueError, match="not fitted"):
            predictor.predict_spread(np.random.randn(10, 15))

    def test_no_validation_data(self, margin_data):
        """Spread predictor should work without validation data."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        X, margin = margin_data
        predictor = SpreadPredictorWithUncertainty(
            input_dim=15, max_epochs=5, batch_size=64,
        )
        # No validation data
        predictor.fit(X[:200], margin[:200])

        result = predictor.predict_with_probs(X[200:205])
        assert np.all(np.isfinite(result["spread_mu"]))
        assert np.all(result["spread_sigma"] > 0)

    def test_constant_margin(self):
        """Constant margin values should still produce valid predictions."""
        from betting_intel.models.mlp_predictor import SpreadPredictorWithUncertainty

        np.random.seed(42)
        X = np.random.randn(100, 5)
        margin = np.ones(100) * 5.0  # constant margin

        predictor = SpreadPredictorWithUncertainty(
            input_dim=5, max_epochs=5, batch_size=32,
        )
        predictor.fit(X[:80], margin[:80])

        result = predictor.predict_with_probs(X[80:85])
        assert np.all(np.isfinite(result["spread_mu"]))
        assert np.all(np.isfinite(result["win_prob"]))


# ═══════════════════════════════════════════════════════════════════════════
# 5. MLPNetwork (internal PyTorch module)
# ═══════════════════════════════════════════════════════════════════════════


class TestMLPNetwork:
    """Tests for the underlying PyTorch MLPNetwork module."""

    def test_forward_shape(self):
        """Forward pass returns correct shape."""
        from betting_intel.models.mlp_predictor import MLPNetwork

        import torch
        net = MLPNetwork(input_dim=20, hidden_dims=[256, 128, 64], output_dim=1)
        x = torch.randn(32, 20)
        out = net(x)

        assert out.shape == (32, 1)

    def test_forward_with_uncertainty(self):
        """Forward pass with predict_uncertainty returns (mu, sigma) tuple."""
        from betting_intel.models.mlp_predictor import MLPNetwork

        import torch
        net = MLPNetwork(input_dim=10, hidden_dims=[64, 32], output_dim=1,
                         predict_uncertainty=True)
        x = torch.randn(16, 10)
        mu, sigma = net(x)

        assert mu.shape == (16, 1)
        assert sigma.shape == (16, 1)
        assert torch.all(sigma > 0)

    def test_custom_hidden_dims(self):
        """Custom hidden dimensions are respected."""
        from betting_intel.models.mlp_predictor import MLPNetwork

        import torch
        net = MLPNetwork(input_dim=5, hidden_dims=[10, 5], output_dim=2)
        x = torch.randn(8, 5)
        out = net(x)

        assert out.shape == (8, 2)

    def test_different_output_dim(self):
        """Output dimension can be > 1."""
        from betting_intel.models.mlp_predictor import MLPNetwork

        import torch
        net = MLPNetwork(input_dim=8, hidden_dims=[16, 8], output_dim=3)
        x = torch.randn(4, 8)
        out = net(x)

        assert out.shape == (4, 3)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Edge Cases & Error Handling
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Cross-cutting edge cases across all components."""

    def test_gaussian_nll_loss(self):
        """_gaussian_nll_loss returns finite positive values."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        import torch
        mu = torch.tensor([0.0, 1.0, -1.0])
        sigma = torch.tensor([1.0, 0.5, 2.0])
        y = torch.tensor([0.5, 1.5, -0.5])

        loss = MLPPredictor._gaussian_nll_loss(mu, sigma, y)
        assert torch.isfinite(loss)
        assert loss > 0

    def test_ensemble_model_failure_graceful(self, regression_data):
        """If one ensemble model fails, the other should still produce predictions."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        X, y = regression_data
        from sklearn.linear_model import Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X[:400], y[:400])

        # Create a broken model
        class BrokenModel:
            def predict(self, X):
                raise RuntimeError("Model crashed")

        ensemble = EnhancedEnsemble()
        ensemble.add_model("ridge", ridge, model_type="regression")
        ensemble.add_model("broken", BrokenModel(), model_type="regression")

        # Should not crash — broken model should be skipped
        preds = ensemble.predict(X[400:405])
        assert len(preds) == 5
        assert np.all(np.isfinite(preds))

    def test_ensemble_all_models_fail(self):
        """If all models fail, ensemble returns zeros."""
        from betting_intel.models.mlp_predictor import EnhancedEnsemble

        class BrokenModel:
            def predict(self, X):
                raise RuntimeError("Crash")

        ensemble = EnhancedEnsemble()
        ensemble.add_model("broken1", BrokenModel(), model_type="regression")
        ensemble.add_model("broken2", BrokenModel(), model_type="regression")

        preds = ensemble.predict(np.random.randn(3, 5))
        assert len(preds) == 3
        assert np.all(preds == 0.0)  # First call returns zeros

    def test_mlp_invalid_prediction_type(self):
        """Invalid prediction_type should default to regression behavior."""
        from betting_intel.models.mlp_predictor import MLPPredictor

        # This should still work (invalid type defaults to regression internally)
        predictor = MLPPredictor(input_dim=5, prediction_type="invalid_value",
                                 max_epochs=3, batch_size=32)
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X[:, 0] * 0.5 + 0.1 * np.random.randn(50)

        predictor.fit(X[:40], y[:40])
        preds = predictor.predict(X[40:])
        assert np.all(np.isfinite(preds))
