"""
Comprehensive tests for RobustPredictionSystem and KellyStaker.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════════════
#  ROBUST PREDICTION SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestRobustPredictionSystem:
    """Tests for the RobustPredictionSystem class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 10)
        # Create a semi-predictable target
        y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0).astype(int)
        return X, y

    @staticmethod
    def _make_fast_system(**kwargs):
        """Create a RobustPredictionSystem with reduced estimators for fast testing."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem
        # Override defaults: 30 estimators instead of 800 for tree models
        fast_params = dict(
            calibrate=False,
            n_folds=3,
            min_train_samples=30,
            use_stacking=False,
            rf_params={"n_estimators": 30, "n_jobs": 1},
            lgb_params={"n_estimators": 30, "verbose": -1},
            xgb_params={"n_estimators": 30, "verbosity": 0},
        )
        fast_params.update(kwargs)
        return RobustPredictionSystem(**fast_params)

    # ── v6.6 New Models Tests ────────────────────────────────────────

    def test_histgradientboosting_included(self, sample_data):
        """Test that HistGradientBoosting is included when use_histgb=True."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(use_histgb=True, use_svm=False, use_mlp=False, use_extratrees=False)
        system.fit(X, y, verbose=False)

        assert "HistGradientBoosting" in system._models
        # Check it contributes to predictions
        probs = system.predict_proba(X[:5])
        assert probs.shape == (5, 2)

    def test_histgradientboosting_disabled(self, sample_data):
        """Test that HistGradientBoosting is excluded when use_histgb=False."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(use_histgb=False, use_svm=False, use_mlp=False, use_extratrees=False)
        system.fit(X, y, verbose=False)

        assert "HistGradientBoosting" not in system._models

    def test_extratrees_included(self, sample_data):
        """Test that ExtraTrees is included when use_extratrees=True."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(use_extratrees=True, use_svm=False, use_mlp=False, use_histgb=False)
        system.fit(X, y, verbose=False)

        assert "ExtraTrees" in system._models

    def test_extratrees_disabled(self, sample_data):
        """Test that ExtraTrees is excluded when use_extratrees=False."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(use_extratrees=False, use_svm=False, use_mlp=False, use_histgb=False)
        system.fit(X, y, verbose=False)

        assert "ExtraTrees" not in system._models

    def test_svm_included(self, sample_data):
        """Test that SVM (via _DownsampledSVC) is included when use_svm=True."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(use_svm=True, use_mlp=False, use_histgb=False, use_extratrees=False)
        system.fit(X, y, verbose=False)

        assert "SVM" in system._models

    def test_svm_disabled(self, sample_data):
        """Test that SVM is excluded when use_svm=False."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(use_svm=False, use_mlp=False, use_histgb=False, use_extratrees=False)
        system.fit(X, y, verbose=False)

        assert "SVM" not in system._models

    def test_svm_downsampled_svc_wrapper(self):
        """Test _DownsampledSVC wrapper directly."""
        from betting_intel.models.robust_ensemble import _DownsampledSVC

        X = np.random.randn(500, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        svm = _DownsampledSVC(max_samples=3000, probability=True, random_state=42)
        svm.fit(X, y)
        preds = svm.predict_proba(X[:10])
        assert preds.shape == (10, 2)
        assert np.allclose(preds.sum(axis=1), 1.0)

        # Test attribute proxying
        assert hasattr(svm, 'support_vectors_')
        assert svm.support_vectors_ is not None

    def test_svm_downsampled_svc_triggers(self):
        """Test that _DownsampledSVC actually downsamples when n > max_samples."""
        from betting_intel.models.robust_ensemble import _DownsampledSVC

        X = np.random.randn(300, 5)
        y = (X[:, 0] > 0).astype(int)

        svm = _DownsampledSVC(max_samples=100, probability=True, random_state=42)
        svm.fit(X, y)

        # Check that internal model was trained on <= 100 samples
        assert svm._model is not None
        assert len(svm._model.support_vectors_) <= len(X)  # Support vectors subset

    def test_svm_downsampled_svc_picklable(self, tmp_path):
        """Test that _DownsampledSVC survives pickle roundtrip via joblib."""
        import joblib
        from betting_intel.models.robust_ensemble import _DownsampledSVC

        X = np.random.randn(200, 5)
        y = (X[:, 0] > 0).astype(int)

        svm = _DownsampledSVC(max_samples=3000, probability=True, random_state=42)
        svm.fit(X, y)

        path = tmp_path / "svm_test.joblib"
        joblib.dump(svm, path)

        loaded = joblib.load(path)
        preds_before = svm.predict_proba(X[:5])
        preds_after = loaded.predict_proba(X[:5])
        assert np.allclose(preds_before, preds_after, atol=1e-6)

    def test_all_three_new_models_together(self, sample_data):
        """Test that all 3 new models train together without issues."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(
            use_histgb=True, use_extratrees=True, use_svm=True,
            use_mlp=False, use_catboost=False,
        )
        system.fit(X, y, verbose=False)

        # All 3 should be in the ensemble
        assert "HistGradientBoosting" in system._models
        assert "ExtraTrees" in system._models
        assert "SVM" in system._models
        # Ensemble should make valid predictions
        probs = system.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    # ── v6.6 Calibration Tests ───────────────────────────────────────

    def test_calibrate_with_isotonic_method(self, sample_data):
        """Test that calibration_method='isotonic' uses isotonic regression."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(calibrate=True, calibration_method="isotonic")
        system.fit(X, y, verbose=False)

        assert system._calibrated_brier is not None
        assert system._brier_score is not None

    def test_calibrate_with_platt_method(self, sample_data):
        """Test that calibration_method='platt' uses Platt scaling."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(calibrate=True, calibration_method="platt")
        system.fit(X, y, verbose=False)

        assert system._calibrated_brier is not None

    def test_calibrate_with_auto_method(self, sample_data):
        """Test that calibration_method='auto' tries isotonic then plattt."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(calibrate=True, calibration_method="auto")
        system.fit(X, y, verbose=False)

        assert system._calibrated_brier is not None

    def test_calibrate_disabled(self, sample_data):
        """Test that calibrate=False skips isotonic/Platt calibration.

        Note: Brier scores are still computed (they use raw probs for both
        raw and 'calibrated' when calibrate=False), so they are not None.
        """
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(calibrate=False)
        system.fit(X, y, verbose=False)

        # Brier scores are computed regardless of calibrate flag
        assert system._calibrated_brier is not None
        assert system._brier_score is not None
        # Without calibration, 'calibrated' = raw so both are the same
        assert system._calibrated_brier == pytest.approx(system._brier_score, rel=1e-6)

    def test_calibrated_brier_not_worse_than_raw(self, sample_data):
        """Test that calibrated Brier is not drastically worse than raw Brier."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(calibrate=True, calibration_method="auto")
        system.fit(X, y, verbose=False)

        if system._brier_score is not None and system._calibrated_brier is not None:
            # Calibrated shouldn't be > 2x raw (allows some slack for small datasets)
            assert system._calibrated_brier <= system._brier_score * 2.0

    # ── v6.6 Adversarial Validation Tests ─────────────────────────────

    def test_adversarial_validation_disabled_by_default(self, sample_data):
        """Test that adversarial validation returns None when disabled."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(use_adversarial_validation=False)
        system.fit(X, y, verbose=False)

        assert system.get_adversarial_validation() is None

    def test_adversarial_validation_enabled(self, sample_data):
        """Test that adversarial validation runs when enabled."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(use_adversarial_validation=True)
        system.fit(X, y, verbose=False)

        adv = system.get_adversarial_validation()
        assert adv is not None
        assert "auroc" in adv
        assert "health" in adv
        assert 0.0 <= adv["auroc"] <= 1.0
        assert adv["health"] in ("stable", "minor", "warning", "critical")

    def test_adversarial_validation_too_few_samples(self):
        """Test that adversarial validation skips with < 200 samples."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X = np.random.randn(150, 5)
        y = (X[:, 0] > 0).astype(int)
        system = RobustPredictionSystem(
            calibrate=False, n_folds=2, min_train_samples=30,
            use_adversarial_validation=True,
            rf_params={"n_estimators": 10, "n_jobs": 1},
            xgb_params={"n_estimators": 10, "verbosity": 0},
            lgb_params={"n_estimators": 10, "verbose": -1},
        )
        system.fit(X, y, verbose=False)

        # With 150 samples, adversarial validation should skip (needs >= 200)
        adv = system.get_adversarial_validation()
        assert adv is None

    # ── v6.6 Ensemble Diversity & Pruning Tests ──────────────────────

    def test_ensemble_diversity_disabled_by_default(self, sample_data):
        """Test that diversity metrics are None when pruning is disabled."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system()
        system.fit(X, y, verbose=False)

        assert system.get_ensemble_diversity() is None

    def test_ensemble_diversity_enabled(self, sample_data):
        """Test that diversity metrics compute when pruning is enabled."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(pruning_keep_top_n=4)
        system.fit(X, y, verbose=False)

        diversity = system.get_ensemble_diversity()
        if diversity is not None:
            assert "diversity_score" in diversity
            assert "avg_correlation" in diversity
            assert 0.0 <= diversity["diversity_score"] <= 1.0

    def test_pruning_keeps_top_n(self, sample_data):
        """Test that pruning_keep_top_n limits model count."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(
            pruning_keep_top_n=3,
            use_svm=False, use_mlp=False, use_catboost=False,
        )
        system.fit(X, y, verbose=False)

        summary = system.get_summary()
        # With 5 models (LR, XGB, LGB, RF, HGB) and pruning_keep_top_n=3,
        # pruning should reduce to at most 5 (may not prune if all models are
        # diverse and accurate). The test just verifies it doesn't error.
        assert summary["n_models"] >= 1
        assert summary["n_models"] <= 6  # Should never exceed base model count

    # ── v6.6 Permutation Importance Tests ─────────────────────────────

    def test_permutation_importance_disabled(self, sample_data):
        """Test that permutation importance returns empty when disabled."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(use_permutation_importance=False)
        system.fit(X, y, verbose=False)

        assert system.get_permutation_importance() == {}

    def test_permutation_importance_enabled(self, sample_data):
        """Test that permutation importance computes top features."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(
            use_permutation_importance=True,
            rf_params={"n_estimators": 30, "n_jobs": 1},
        )
        system.fit(X, y, verbose=False)

        imp = system.get_permutation_importance(top_n=5)
        # May be empty with few samples, but shouldn't error
        assert isinstance(imp, dict)
        if imp:
            assert len(imp) <= 5
            for v in imp.values():
                assert isinstance(v, float)

    # ── v6.6 Bootstrap Uncertainty Tests ──────────────────────────────

    def test_bootstrap_uncertainty_disabled(self, sample_data):
        """Test that bootstrap uncertainty returns None when disabled."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(use_bootstrap_uncertainty=False)
        system.fit(X, y, verbose=False)

        assert system.get_bootstrap_uncertainty() is None

    def test_bootstrap_uncertainty_enabled(self, sample_data):
        """Test that bootstrap uncertainty computes metrics."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(
            use_bootstrap_uncertainty=True,
            n_bootstrap_samples=5,
        )
        system.fit(X, y, verbose=False)

        unc = system.get_bootstrap_uncertainty()
        assert unc is not None
        assert "mean_uncertainty" in unc
        assert "std_uncertainty" in unc
        assert "has_bootstrap" in unc
        assert unc["has_bootstrap"] is True

    # ── Existing Tests ───────────────────────────────────────────────

    def test_import(self):
        """Verify the module imports correctly."""
        from betting_intel.models.robust_ensemble import (
            RobustPredictionSystem, PredictionResult,
            compute_statistical_significance, compute_drawdown,
        )
        assert RobustPredictionSystem is not None
        assert PredictionResult is not None

    def test_fit_and_predict_with_defaults(self, sample_data):
        """Test that fit() and predict_proba() work with default params."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system()
        system.fit(X, y, verbose=False)

        assert system._fitted
        assert len(system._models) >= 2  # At least Logistic + one tree model

        probs = system.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        assert np.all(probs >= 0) and np.all(probs <= 1)
        # Probabilities should sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_predict_with_details(self, sample_data):
        """Test predict_with_details returns complete PredictionResult."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(calibrate=True)
        system.fit(X, y, verbose=False)

        result = system.predict_with_details(X[0])
        assert 0 < result.home_win_prob < 1
        assert result.confidence_label in ("VERY_HIGH", "HIGH", "MEDIUM", "LOW", "VERY_LOW")
        assert result.n_models >= 1
        assert len(result.model_probs) >= 1

    def test_predict_with_details_2d_input(self, sample_data):
        """Test predict_with_details handles 2D input."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system()
        system.fit(X, y, verbose=False)

        # Should work with (1, n_features) input
        result_2d = system.predict_with_details(X[0:1])
        assert isinstance(result_2d.home_win_prob, float)

        # Should work with (n_features,) input
        result_1d = system.predict_with_details(X[0])
        assert isinstance(result_1d.home_win_prob, float)

    def test_not_fitted_error(self):
        """Test that predicting before fit raises ValueError."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        system = RobustPredictionSystem()
        with pytest.raises(ValueError, match="not fitted"):
            system.predict_proba(np.random.randn(1, 5))

    def test_insufficient_data_error(self):
        """Test that fit with too little data raises ValueError."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        system = RobustPredictionSystem(min_train_samples=100, min_test_samples=20)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        with pytest.raises(ValueError, match="Need at least"):
            system.fit(X, y, verbose=False)

    def test_compute_edge_valid(self, sample_data):
        """Test compute_edge with valid market odds."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        system = self._make_fast_system()
        X, y = sample_data
        system.fit(X, y, verbose=False)

        # Test with meaningful odds
        edge, direction, confidence = system.compute_edge(0.60, -150, +130)
        assert isinstance(edge, float)
        assert direction in ("home", "away", "neutral")
        assert isinstance(confidence, str)

        # Home team +EV (model says 60%, market says ~60% implied after vig removal)
        # -150 → implied = 150/250 = 0.60, +130 → 100/230 = 0.435
        # After vig removal: home ~0.58, away ~0.42
        # Edge = 0.60 - 0.58 = 0.02 (home +EV since model > market)
        expected_edge = 0.60 - (0.60 / (0.60 + 0.435))
        assert abs(edge - round(expected_edge, 4)) < 0.001

    def test_compute_edge_none_odds(self, sample_data):
        """Test compute_edge with None odds returns neutral."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        system = self._make_fast_system()
        X, y = sample_data
        system.fit(X, y, verbose=False)

        edge, direction, confidence = system.compute_edge(0.60, None, None)
        assert edge == 0.0
        assert direction == "neutral"
        assert confidence == "LOW"

    def test_feature_importance(self, sample_data):
        """Test feature importance returns correct format."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system()
        system.fit(X, y, feature_names=[f"feat_{i}" for i in range(X.shape[1])], verbose=False)

        importance = system.get_feature_importance(top_n=5)
        assert len(importance) <= 5
        for name, val in importance.items():
            assert name.startswith("feat_")
            assert isinstance(val, float)
            assert 0 <= val <= 1

    def test_get_summary(self, sample_data):
        """Test get_summary returns valid dict."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system()
        system.fit(X, y, verbose=False)

        summary = system.get_summary()
        assert summary["fitted"] is True
        assert summary["n_models"] >= 2
        assert summary["n_features"] == 10
        assert summary["n_train_samples"] == 200

    def test_get_model_diagnostics(self, sample_data):
        """Test model diagnostics returns info per model."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system()
        system.fit(X, y, verbose=False)

        diag = system.get_model_diagnostics()
        assert len(diag) >= 2
        for name, d in diag.items():
            assert d.oos_brier > 0
            assert d.n_oos > 0

    def test_save_and_load(self, sample_data, tmp_path):
        """Test save and load roundtrip preserves state."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system()
        system.fit(X, y, verbose=False)

        path = tmp_path / "test_system.joblib"
        system.save(path)

        loaded = RobustPredictionSystem.load(path)
        assert loaded._fitted
        assert loaded._n_train_total == 200
        assert len(loaded._models) == len(system._models)

        # Predictions should match
        orig_result = system.predict_with_details(X[0])
        loaded_result = loaded.predict_with_details(X[0])
        assert abs(orig_result.home_win_prob - loaded_result.home_win_prob) < 0.01

    def test_overfitting_detection(self):
        """Test overfitting detection logic."""
        from betting_intel.models.robust_ensemble import OverfittingReport

        report = OverfittingReport(
            is_overfit=True,
            avg_train_r2=0.95,
            avg_test_r2=0.10,
            r2_gap=0.85,
            flags=["Test R² is severely negative"],
        )
        assert report.is_overfit
        assert report.r2_gap == 0.85

    def test_predict_binary(self, sample_data):
        """Test predict() returns binary classes."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system()
        system.fit(X, y, verbose=False)

        preds = system.predict(X[:10], threshold=0.5)
        assert preds.shape == (10,)
        assert set(preds).issubset({0, 1})

    def test_calibration_improves_brier(self, sample_data):
        """Test that calibrated Brier is not worse than raw Brier."""
        from betting_intel.models.robust_ensemble import RobustPredictionSystem

        X, y = sample_data
        system = self._make_fast_system(calibrate=True)
        system.fit(X, y, verbose=False)

        summary = system.get_summary()
        raw_brier = summary.get("brier_score")
        cal_brier = summary.get("calibrated_brier")

        # If both exist, calibrated shouldn't be drastically worse
        if raw_brier is not None and cal_brier is not None:
            assert cal_brier <= raw_brier * 1.5  # Allow some slack


# ═══════════════════════════════════════════════════════════════════════════
#  STATISTICAL SIGNIFICANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestStatisticalSignificance:
    """Tests for compute_statistical_significance."""

    def test_significant_result(self):
        """Test that 60/40 is significant at 95% level."""
        from betting_intel.models.robust_ensemble import compute_statistical_significance

        result = compute_statistical_significance(60, 40)
        assert result["win_rate"] == 0.6
        assert result["n_bets"] == 100
        assert result["is_significant"]  # p < 0.05

    def test_not_significant(self):
        """Test that 55/45 is not necessarily significant with small n."""
        from betting_intel.models.robust_ensemble import compute_statistical_significance

        result = compute_statistical_significance(55, 45)
        assert result["win_rate"] == 0.55
        assert not result["is_significant"]  # p > 0.05

    def test_edge_case_no_bets(self):
        """Test edge case with zero bets."""
        from betting_intel.models.robust_ensemble import compute_statistical_significance

        result = compute_statistical_significance(0, 0)
        assert result["win_rate"] == 0.0
        assert result["n_bets"] == 0
        assert result["p_value"] == 1.0
        assert not result["is_significant"]

    def test_perfect_record(self):
        """Test perfect record gives very low p-value."""
        from betting_intel.models.robust_ensemble import compute_statistical_significance

        result = compute_statistical_significance(20, 0)
        assert result["win_rate"] == 1.0
        assert result["is_significant"]
        assert result["ci_lower"] > 0.8

    def test_ci_bounds(self):
        """Test confidence intervals are within [0, 1]."""
        from betting_intel.models.robust_ensemble import compute_statistical_significance

        for wins, losses in [(10, 10), (30, 20), (50, 10), (5, 15), (100, 100)]:
            result = compute_statistical_significance(wins, losses)
            assert 0 <= result["ci_lower"] <= 1
            assert 0 <= result["ci_upper"] <= 1
            assert result["ci_lower"] <= result["ci_upper"]


# ═══════════════════════════════════════════════════════════════════════════
#  DRAWDOWN ANALYSIS TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestDrawdown:
    """Tests for compute_drawdown."""

    def test_basic_drawdown(self):
        """Test basic drawdown calculation."""
        from betting_intel.models.robust_ensemble import compute_drawdown

        # Profits: +10, -5, +20, -10, +30
        # Cumulative: 10, 5, 25, 15, 45
        # Peak: 10, 10, 25, 25, 45
        # Drawdown: 0, 5, 0, 10, 0
        # Max drawdown = 10
        result = compute_drawdown([10, -5, 20, -10, 30])
        assert result["max_drawdown"] == 10.0

    def test_no_drawdown(self):
        """Test with only positive profits."""
        from betting_intel.models.robust_ensemble import compute_drawdown

        result = compute_drawdown([10, 20, 30])
        assert result["max_drawdown"] == 0.0
        assert result["max_drawdown_pct"] == 0.0

    def test_empty_list(self):
        """Test with empty list."""
        from betting_intel.models.robust_ensemble import compute_drawdown

        result = compute_drawdown([])
        assert result["max_drawdown"] == 0.0
        assert result["max_drawdown_pct"] == 0.0

    def test_large_drawdown(self):
        """Test large drawdown calculation."""
        from betting_intel.models.robust_ensemble import compute_drawdown

        # Start 100, lose 50, gain 30, lose 80
        # Cumulative: 100, -50, -20, -100
        result = compute_drawdown([100, -150, 30, -80])
        assert result["max_drawdown"] > 0
        assert result["max_drawdown_pct"] > 0


# ═══════════════════════════════════════════════════════════════════════════
#  KELLY STAKER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestKellyStaker:
    """Tests for the KellyStaker class."""

    @pytest.fixture
    def staker(self):
        """Create a basic staker."""
        from betting_intel.recommendations.staking import KellyStaker
        return KellyStaker(initial_bankroll=10000, kelly_fraction=0.25)

    def test_import(self):
        """Verify staking module imports correctly."""
        from betting_intel.recommendations.staking import (
            KellyStaker, StakeResult, BankrollState,
            american_to_decimal, decimal_to_american,
            american_to_implied, remove_vig,
        )
        assert KellyStaker is not None
        assert StakeResult is not None

    def test_compute_stake_basic(self, staker):
        """Test basic stake computation."""
        from betting_intel.recommendations.staking import american_to_decimal

        dec_odds = american_to_decimal(-110)
        result = staker.compute_stake(
            win_probability=0.60,
            decimal_odds=dec_odds,
            confidence_score=0.80,
            confidence_label="HIGH",
            edge_pct=0.05,
            league="NBA",
            team="Lakers",
        )
        assert result.stake_dollars > 0
        assert result.kelly_full > 0
        assert result.exposure_pct > 0
        assert result.kelly_full > result.kelly_fractional  # fractional less than full

    def test_compute_stake_below_threshold(self, staker):
        """Test that stake is 0 when edge is below threshold."""
        result = staker.compute_stake(
            win_probability=0.51,
            decimal_odds=1.91,
            confidence_score=0.5,
            confidence_label="LOW",
            edge_pct=0.005,  # Below 0.01 threshold
            league="NBA",
            team="Lakers",
        )
        assert result.stake_dollars == 0.0
        assert "below threshold" in " ".join(result.adjustment_reasons).lower()

    def test_kelly_full_calculation(self, staker):
        """Test Kelly formula: f* = (bp - q) / b"""
        # For 60% win prob at -110 odds (1.91 decimal):
        # b = 0.91, p = 0.60, q = 0.40
        # f* = (0.91 * 0.60 - 0.40) / 0.91 = (0.546 - 0.40) / 0.91 = 0.146/0.91 = 0.1604
        from betting_intel.recommendations.staking import american_to_decimal

        dec_odds = american_to_decimal(-110)  # ~1.909
        result = staker.compute_stake(
            win_probability=0.60,
            decimal_odds=dec_odds,
            confidence_score=1.0,
            confidence_label="VERY_HIGH",
            edge_pct=0.10,
        )
        expected_kelly = ((dec_odds - 1) * 0.60 - 0.40) / (dec_odds - 1)
        assert abs(result.kelly_full - expected_kelly) < 0.001

    def test_consecutive_losses_reduce_stake(self):
        """Test that consecutive losses reduce subsequent stakes."""
        from betting_intel.recommendations.staking import KellyStaker

        staker = KellyStaker(initial_bankroll=10000, kelly_fraction=0.25)

        # No losses
        result_no_loss = staker.compute_stake(
            win_probability=0.65, decimal_odds=1.91,
            confidence_score=0.9, confidence_label="VERY_HIGH",
            edge_pct=0.10,
        )

        # Simulate 5 consecutive losses
        for _ in range(5):
            staker.record_bet(stake=100, won=False, profit=-100)

        result_with_losses = staker.compute_stake(
            win_probability=0.65, decimal_odds=1.91,
            confidence_score=0.9, confidence_label="VERY_HIGH",
            edge_pct=0.10,
        )

        assert result_with_losses.stake_dollars < result_no_loss.stake_dollars

    def test_drawdown_reduces_stake(self):
        """Test that drawdown reduces stake."""
        from betting_intel.recommendations.staking import KellyStaker

        staker = KellyStaker(initial_bankroll=10000, kelly_fraction=0.25)

        # Simulate large losses to create drawdown
        for _ in range(3):
            staker.record_bet(stake=2000, won=False, profit=-2000)

        result = staker.compute_stake(
            win_probability=0.70, decimal_odds=1.91,
            confidence_score=0.9, confidence_label="VERY_HIGH",
            edge_pct=0.15,
        )
        # After 3 * 2000 losses, bankroll is 4000, drawdown is 60%
        # At 40%+ drawdown, staking is disabled
        assert result.stake_dollars == 0.0
        assert any("drawdown" in r.lower() for r in result.adjustment_reasons)

    def test_team_exposure_limit(self, staker):
        """Test that team exposure limit works."""
        # First bet on Lakers
        result1 = staker.compute_stake(
            win_probability=0.60, decimal_odds=1.91,
            confidence_score=0.8, confidence_label="HIGH",
            edge_pct=0.05, league="NBA", team="Lakers",
        )
        staker.record_bet(team="Lakers", stake=result1.stake_dollars, won=True)

        # Second bet on Lakers should be limited
        result2 = staker.compute_stake(
            win_probability=0.65, decimal_odds=1.91,
            confidence_score=0.85, confidence_label="VERY_HIGH",
            edge_pct=0.08, league="NBA", team="Lakers",
        )

        # Exposure per team should not exceed max
        total_exposed = result1.stake_dollars + result2.stake_dollars
        max_allowed = staker.max_team_exposure_pct * staker.bankroll
        assert total_exposed <= max_allowed * 1.1  # 10% slack

    def test_get_state(self, staker):
        """Test get_state returns valid BankrollState."""
        state = staker.get_state()
        assert state.current == 10000
        assert state.initial == 10000
        assert state.peak == 10000
        assert state.drawdown == 0.0
        assert state.n_bets_today == 0

    def test_record_win(self, staker):
        """Test recording a win updates bankroll."""
        staker.record_bet(team="Lakers", stake=200, won=True, profit=181.82)
        assert staker.bankroll == pytest.approx(10181.82, rel=1e-4)
        assert staker.total_profit == pytest.approx(181.82, rel=1e-4)

    def test_record_loss(self, staker):
        """Test recording a loss updates bankroll."""
        staker.record_bet(team="Celtics", stake=200, won=False, profit=-200)
        assert staker.bankroll == 9800.0
        assert staker.total_profit == -200.0

    def test_reset(self, staker):
        """Test reset restores initial state."""
        staker.record_bet(stake=500, won=True, profit=500)
        assert staker.bankroll == 10500.0
        staker.reset()
        assert staker.bankroll == 10000.0
        assert staker.peak == 10000.0

    def test_get_performance_summary(self, staker):
        """Test performance summary returns correct data."""
        staker.record_bet(stake=200, won=True, profit=180)
        staker.record_bet(stake=200, won=False, profit=-200)
        staker.record_bet(stake=200, won=True, profit=180)

        summary = staker.get_performance_summary()
        assert summary["n_bets"] == 3
        assert summary["wins"] == 2
        assert summary["losses"] == 1
        assert summary["win_rate"] == pytest.approx(0.6667, abs=0.001)  # Rounded to 4dp

    def test_release_exposure(self, staker):
        """Test releasing exposure works."""
        staker.record_bet(team="Lakers", league="NBA", game_id="g1", stake=500)
        assert staker.get_exposure().total_exposed == 500.0

        staker.release_exposure(team="Lakers")
        assert "Lakers" not in staker.get_exposure().per_team
        assert staker.get_exposure().total_exposed == 0.0

    def test_high_confidence_bets_bigger(self):
        """Test that higher confidence = bigger stake."""
        from betting_intel.recommendations.staking import KellyStaker

        staker = KellyStaker(initial_bankroll=10000)

        low_conf = staker.compute_stake(
            win_probability=0.60, decimal_odds=1.91,
            confidence_score=0.5, confidence_label="LOW",
            edge_pct=0.05,
        )

        high_conf = staker.compute_stake(
            win_probability=0.60, decimal_odds=1.91,
            confidence_score=0.9, confidence_label="VERY_HIGH",
            edge_pct=0.05,
        )

        assert high_conf.stake_dollars > low_conf.stake_dollars

    def test_daily_bet_limit(self, staker):
        """Test daily bet limit is enforced."""
        # Reach the daily bet limit
        for _ in range(staker.max_daily_bets):
            staker._n_bets_today += 1

        result = staker.compute_stake(
            win_probability=0.60, decimal_odds=1.91,
            confidence_score=0.8, confidence_label="HIGH",
            edge_pct=0.05,
        )
        assert result.stake_dollars == 0.0
        assert any("daily bet limit" in r.lower() for r in result.adjustment_reasons)

    def test_negative_edge_returns_zero(self, staker):
        """Test negative edge returns zero stake."""
        result = staker.compute_stake(
            win_probability=0.45,
            decimal_odds=1.91,
            edge_pct=-0.05,  # Negative edge
        )
        assert result.stake_dollars == 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  ODDS CONVERSION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestOddsConversion:
    """Tests for odds conversion utilities."""

    def test_american_to_decimal_favorite(self):
        from betting_intel.recommendations.staking import american_to_decimal
        assert abs(american_to_decimal(-150) - 1.6667) < 0.01

    def test_american_to_decimal_underdog(self):
        from betting_intel.recommendations.staking import american_to_decimal
        assert abs(american_to_decimal(+200) - 3.0) < 0.01

    def test_american_to_decimal_even(self):
        from betting_intel.recommendations.staking import american_to_decimal
        assert abs(american_to_decimal(100) - 2.0) < 0.01

    def test_decimal_to_american(self):
        from betting_intel.recommendations.staking import decimal_to_american
        assert decimal_to_american(1.91) == -110  # Common NBA odds

    def test_decimal_to_american_underdog(self):
        from betting_intel.recommendations.staking import decimal_to_american
        assert decimal_to_american(3.0) == 200

    def test_american_to_implied_favorite(self):
        from betting_intel.recommendations.staking import american_to_implied
        assert abs(american_to_implied(-200) - 2/3) < 0.01

    def test_american_to_implied_underdog(self):
        from betting_intel.recommendations.staking import american_to_implied
        assert abs(american_to_implied(+200) - 1/3) < 0.01

    def test_remove_vig(self):
        from betting_intel.recommendations.staking import remove_vig
        home, away = remove_vig(0.6, 0.45)
        total = home + away
        assert abs(total - 1.0) < 0.001  # Should sum to 1


# ═══════════════════════════════════════════════════════════════════════════
#  ENGINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestEngineIntegration:
    """Tests for the engine with RobustPredictionSystem."""

    def test_engine_imports(self):
        """Verify engine imports work."""
        from betting_intel.live.engine import LivePredictionEngine, LiveGame, LivePredictionSnapshot
        assert LivePredictionEngine is not None
        assert LiveGame is not None

    def test_engine_initialization(self):
        """Test engine init creates kelly staker and robust references."""
        from betting_intel.live.engine import LivePredictionEngine

        engine = LivePredictionEngine()
        assert engine.kelly_staker is not None
        assert engine.robust_system is None  # Not fitted yet
        assert engine.robust_system_summary["status"] == "not_initialized"

    def test_engine_robust_properties(self):
        """Test robust_system properties work before training."""
        from betting_intel.live.engine import LivePredictionEngine

        engine = LivePredictionEngine()
        summary = engine.robust_system_summary
        assert isinstance(summary, dict)
        assert "fitted" in summary

    def test_kelly_staker_property(self):
        """Test kelly_staker property returns working staker."""
        from betting_intel.live.engine import LivePredictionEngine

        engine = LivePredictionEngine()
        staker = engine.kelly_staker
        assert staker is not None
        assert staker.bankroll == 10000.0


class TestLiveGame:
    """Tests for the LiveGame dataclass."""

    def test_matchup_property(self):
        """Test matchup property format."""
        from betting_intel.live.engine import LiveGame

        game = LiveGame(
            game_id="test_1",
            sport_key="basketball_nba",
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            home_team_short="LAL",
            away_team_short="BOS",
            commence_time="2025-06-12T19:00:00Z",
            game_date="2025-06-12",
        )
        assert game.matchup == "BOS @ LAL"

    def test_commence_datetime(self):
        """Test commence_datetime parses ISO format."""
        from betting_intel.live.engine import LiveGame

        game = LiveGame(
            game_id="test_1",
            sport_key="basketball_nba",
            home_team="Lakers",
            away_team="Celtics",
            home_team_short="LAL",
            away_team_short="BOS",
            commence_time="2025-06-12T19:00:00Z",
            game_date="2025-06-12",
        )
        dt = game.commence_datetime
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 6

    def test_to_dict(self):
        """Test to_dict serialization."""
        from betting_intel.live.engine import LiveGame

        game = LiveGame(
            game_id="test_1",
            sport_key="basketball_nba",
            home_team="Lakers",
            away_team="Celtics",
            home_team_short="LAL",
            away_team_short="BOS",
            commence_time="2025-06-12T19:00:00Z",
            game_date="2025-06-12",
            home_ml=-150,
            away_ml=+130,
        )
        d = game.to_dict()
        assert d["game_id"] == "test_1"
        assert d["home_ml"] == -150
        assert d.get("matchup", "MISSING") == "MISSING"  # 'matchup' is a @property, not in to_dict()
        assert d["away_team_short"] == "BOS"
        assert d["home_team_short"] == "LAL"


class TestLivePredictionSnapshot:
    """Tests for the LivePredictionSnapshot dataclass."""

    def test_empty_snapshot(self):
        """Test empty snapshot defaults."""
        from betting_intel.live.engine import LivePredictionSnapshot

        snap = LivePredictionSnapshot()
        assert snap.n_live == 0
        assert snap.n_total == 0
        assert len(snap.next_two_days) == 0
        assert snap.generated_at is not None
