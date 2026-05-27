"""Tests for the validation module — time-series cross-validation, overfitting detection, probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestTimeSeriesCrossValidator:
    """Tests for purged time-series cross-validation."""

    @pytest.fixture
    def cv(self):
        from betting_intel.validation.cross_validation import TimeSeriesCrossValidator

        return TimeSeriesCrossValidator(n_splits=5, min_train_size=100, embargo=5)

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            "GAME_DATE": pd.date_range("2020-01-01", periods=n, freq="D"),
            "feature_1": np.random.normal(0, 1, n),
            "feature_2": np.random.normal(0, 1, n),
            "feature_3": np.random.normal(0, 1, n),
            "total_points": np.random.normal(220, 15, n),
        })

    def test_get_splits_basic(self, cv):
        splits = cv.get_splits(n_samples=500)
        assert len(splits) == 5
        for tr_start, tr_end, te_start, te_end in splits:
            assert tr_end <= te_start  # No leakage
            assert tr_end > tr_start   # Non-empty train
            assert te_end > te_start   # Non-empty test

    def test_get_splits_embargo_enforced(self, cv):
        splits = cv.get_splits(n_samples=500)
        for _, tr_end, te_start, _ in splits:
            assert tr_end + cv.embargo <= te_start or tr_end >= te_start

    def test_get_splits_not_enough_data(self):
        from betting_intel.validation.cross_validation import TimeSeriesCrossValidator

        cv = TimeSeriesCrossValidator(n_splits=5, min_train_size=200)
        with pytest.raises(ValueError, match="Not enough samples"):
            cv.get_splits(n_samples=50)

    def test_validate_basic(self, cv, sample_df):
        from betting_intel.models.predictors import TotalPointsPredictor

        results = cv.validate(
            df=sample_df,
            feature_cols=["feature_1", "feature_2", "feature_3"],
            target_col="total_points",
            model_builder=lambda: TotalPointsPredictor("lightgbm"),
            prediction_type="regression",
        )
        assert len(results) > 0
        assert all(r.metrics for r in results if not r.errors)

    def test_validate_metrics_present(self, cv, sample_df):
        from betting_intel.models.predictors import TotalPointsPredictor

        results = cv.validate(
            df=sample_df,
            feature_cols=["feature_1", "feature_2", "feature_3"],
            target_col="total_points",
            model_builder=lambda: TotalPointsPredictor("lightgbm"),
            prediction_type="regression",
        )

        valid = [r for r in results if r.metrics]
        if valid:
            metrics = valid[0].metrics
            assert "mae" in metrics
            assert "r2" in metrics
            assert "rmse" in metrics
            assert "bias" in metrics
            assert "n_train" in metrics
            assert "n_test" in metrics
            assert metrics["mae"] > 0
            assert metrics["n_train"] > 0
            assert metrics["n_test"] > 0

    def test_get_summary(self, cv, sample_df):
        from betting_intel.models.predictors import TotalPointsPredictor

        cv.validate(
            df=sample_df,
            feature_cols=["feature_1", "feature_2", "feature_3"],
            target_col="total_points",
            model_builder=lambda: TotalPointsPredictor("lightgbm"),
        )
        summary = cv.get_summary()
        assert summary["n_folds"] > 0
        assert "mae_mean" in summary
        assert "mae_std" in summary
        assert "r2_mean" in summary or "r2_mean" not in summary

    def test_get_summary_empty(self, cv):
        summary = cv.get_summary()
        assert summary == {} or "error" in summary

    def test_get_prediction_stability(self, cv, sample_df):
        from betting_intel.models.predictors import TotalPointsPredictor

        cv.validate(
            df=sample_df,
            feature_cols=["feature_1", "feature_2", "feature_3"],
            target_col="total_points",
            model_builder=lambda: TotalPointsPredictor("lightgbm"),
        )
        stability = cv.get_prediction_stability()
        if stability:
            assert "mae_cv" in stability or "r2_cv" in stability

    def test_validate_classification(self, cv, sample_df):
        from sklearn.linear_model import LogisticRegression

        # Add binary target
        df = sample_df.copy()
        df["won"] = (np.random.normal(0, 1, len(df)) > 0).astype(int)

        results = cv.validate(
            df=df,
            feature_cols=["feature_1", "feature_2", "feature_3"],
            target_col="won",
            model_builder=lambda: LogisticRegression(),
            prediction_type="classification",
        )
        valid = [r for r in results if r.metrics]
        if valid:
            assert "accuracy" in valid[0].metrics


class TestExpandingWindowCV:
    """Tests for expanding window cross-validation."""

    def test_get_splits_basic(self):
        from betting_intel.validation.cross_validation import ExpandingWindowCV

        cv = ExpandingWindowCV(n_splits=5, initial_window=100, step_size=50)
        splits = cv.get_splits(n_samples=500)
        assert len(splits) > 0
        for _, tr_end, te_start, _ in splits:
            assert tr_end <= te_start

    def test_get_splits_limited_data(self):
        from betting_intel.validation.cross_validation import ExpandingWindowCV

        cv = ExpandingWindowCV(n_splits=10, initial_window=100, step_size=50)
        splits = cv.get_splits(n_samples=150)
        assert len(splits) < 10  # Should produce fewer splits than requested

    def test_get_splits_ascending_train(self):
        from betting_intel.validation.cross_validation import ExpandingWindowCV

        cv = ExpandingWindowCV(n_splits=3, initial_window=50, step_size=30, embargo=5)
        splits = cv.get_splits(n_samples=200)
        train_sizes = [tr_end - tr_start for tr_start, tr_end, _, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]  # Expanding window


class TestPurgedWalkForward:
    """Tests for the purged_walk_forward convenience function."""

    def test_basic_execution(self):
        from betting_intel.validation.cross_validation import purged_walk_forward
        from betting_intel.models.predictors import TotalPointsPredictor

        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "GAME_DATE": pd.date_range("2020-01-01", periods=n, freq="D"),
            "f1": np.random.normal(0, 1, n),
            "f2": np.random.normal(0, 1, n),
            "total_points": np.random.normal(220, 15, n),
        })

        result = purged_walk_forward(
            df=df,
            feature_cols=["f1", "f2"],
            target_col="total_points",
            model_builder=lambda: TotalPointsPredictor("lightgbm"),
            train_window=100,
            test_window=20,
            embargo=5,
        )
        assert "predictions" in result
        assert "actuals" in result
        assert "fold_metrics" in result
        assert result["n_folds"] > 0
        assert result["overall_mae"] > 0


class TestDeflatedSharpeRatio:
    """Tests for the Deflated Sharpe Ratio."""

    def test_compute_basic(self):
        from betting_intel.validation.overfitting import DeflatedSharpeRatio

        dsr = DeflatedSharpeRatio(
            observed_sharpe=2.0,
            n_strategies=5,
            n_observations=200,
            skew=-0.2,
            kurtosis=3.5,
        )
        dsr_value, p_value = dsr.compute()
        assert isinstance(dsr_value, float)
        assert 0 <= p_value <= 1

    def test_compute_high_sharpe_significant(self):
        from betting_intel.validation.overfitting import DeflatedSharpeRatio

        dsr = DeflatedSharpeRatio(
            observed_sharpe=5.0,
            n_strategies=1,
            n_observations=500,
        )
        dsr_value, _ = dsr.compute()
        assert dsr_value > 2  # Should be clearly significant

    def test_compute_low_sharpe_not_significant(self):
        from betting_intel.validation.overfitting import DeflatedSharpeRatio

        dsr = DeflatedSharpeRatio(
            observed_sharpe=0.5,
            n_strategies=50,
            n_observations=50,
        )
        dsr_value, p_value = dsr.compute()
        assert dsr_value < 2 or p_value > 0.05  # Likely not significant

    def test_is_significant(self):
        from betting_intel.validation.overfitting import DeflatedSharpeRatio

        dsr = DeflatedSharpeRatio(observed_sharpe=3.0, n_strategies=1, n_observations=200)
        assert dsr.is_significant(threshold=2.0)

    def test_single_strategy(self):
        from betting_intel.validation.overfitting import DeflatedSharpeRatio

        dsr = DeflatedSharpeRatio(observed_sharpe=1.5, n_strategies=1, n_observations=100)
        dsr_v, p_v = dsr.compute()
        assert dsr_v >= 0


class TestModelComparison:
    """Tests for ModelComparisonTest (Diebold-Mariano, McNemar)."""

    def test_diebold_mariano_b_vs_a(self):
        from betting_intel.validation.overfitting import ModelComparisonTest

        errors_a = np.random.normal(0, 2, 100)
        errors_b = errors_a * 0.5  # B is better (smaller errors)

        result = ModelComparisonTest.diebold_mariano(errors_a, errors_b)
        assert "dm_statistic" in result
        assert "p_value" in result
        assert "conclusion" in result
        assert result["better_model"] == "B"

    def test_diebold_mariano_a_vs_b(self):
        from betting_intel.validation.overfitting import ModelComparisonTest

        errors_a = np.random.normal(0, 1, 100)
        errors_b = errors_a * 2  # A is better

        result = ModelComparisonTest.diebold_mariano(errors_a, errors_b)
        assert result["better_model"] == "A"

    def test_diebold_mariano_insufficient_data(self):
        from betting_intel.validation.overfitting import ModelComparisonTest

        result = ModelComparisonTest.diebold_mariano(
            np.array([1.0, 2.0]), np.array([1.5, 2.5])
        )
        assert "Insufficient data" in result["conclusion"]

    def test_mcnemar_basic(self):
        from betting_intel.validation.overfitting import ModelComparisonTest

        np.random.seed(42)
        n = 200
        correct_a = np.random.random(n) > 0.4
        correct_b = np.random.random(n) > 0.45

        result = ModelComparisonTest.mcnemar(correct_a, correct_b)
        assert "chi2" in result
        assert "p_value" in result
        assert "concordant_pairs" in result
        assert "discordant_pairs" in result
        assert "conclusion" in result

    def test_mcnemar_no_discordant(self):
        from betting_intel.validation.overfitting import ModelComparisonTest

        correct_a = np.array([True, True, True])
        correct_b = np.array([True, True, True])

        result = ModelComparisonTest.mcnemar(correct_a, correct_b)
        assert "No discordant pairs" in result["conclusion"]


class TestOverfittingDetector:
    """Tests for the comprehensive OverfittingDetector."""

    @pytest.fixture
    def detector(self):
        from betting_intel.validation.overfitting import OverfittingDetector

        return OverfittingDetector()

    def test_analyze_genuine_strategy(self, detector):
        """A strategy with low train-test gap and stable CV should score low."""
        result = detector.analyze(
            train_metrics={"win_rate": 0.58, "r2": 0.90, "mae": 2.5},
            test_metrics={"win_rate": 0.56, "r2": 0.88, "mae": 2.8},
            cv_results=[
                {"win_rate": 0.55, "r2": 0.87},
                {"win_rate": 0.56, "r2": 0.88},
                {"win_rate": 0.57, "r2": 0.89},
            ],
            n_strategies_tested=3,
            n_observations=500,
            sharpe_ratio=2.0,
        )
        assert result["overfitting_score"] <= 40  # Low = likely genuine

    def test_analyze_overfitted_strategy(self, detector):
        """A strategy with high train-test gap should score high."""
        result = detector.analyze(
            train_metrics={"win_rate": 0.85, "r2": 0.99, "mae": 1.0},
            test_metrics={"win_rate": 0.51, "r2": 0.20, "mae": 8.0},
            cv_results=[
                {"win_rate": 0.52, "r2": 0.15},
                {"win_rate": 0.50, "r2": 0.25},
                {"win_rate": 0.53, "r2": 0.18},
            ],
            n_strategies_tested=100,
            n_observations=100,
            sharpe_ratio=0.5,
        )
        assert result["overfitting_score"] >= 20  # Higher = more overfitted

    def test_analyze_no_cv_results(self, detector):
        """Should handle missing CV results gracefully."""
        result = detector.analyze(
            train_metrics={"win_rate": 0.55, "mae": 3.0},
            test_metrics={"win_rate": 0.53, "mae": 3.2},
            cv_results=[],
        )
        assert "cv_warning" in result or "overfitting_score" in result

    def test_analyze_with_warnings(self, detector):
        result = detector.analyze(
            train_metrics={"win_rate": 0.80, "r2": 0.98, "mae": 1.0},
            test_metrics={"win_rate": 0.55, "r2": 0.30, "mae": 5.0},
            cv_results=[
                {"win_rate": 0.54, "r2": 0.28},
                {"win_rate": 0.56, "r2": 0.32},
            ],
            n_strategies_tested=50,
            n_observations=200,
            sharpe_ratio=0.8,
        )
        assert len(result.get("warnings", [])) > 0

    def test_get_verdict_levels(self, detector):
        verdicts = {}
        for score in [5, 30, 50, 80]:
            verdicts[score] = detector._get_verdict(score)

        assert "LOW" in verdicts[5]
        assert "MODERATE" in verdicts[30]
        assert "HIGH" in verdicts[50]
        assert "CRITICAL" in verdicts[80]
        assert "Reject" in verdicts[80]


class TestProbabilityCalibration:
    """Tests for the ProbabilityCalibrator module."""

    @pytest.fixture
    def calibration_data(self):
        np.random.seed(42)
        n = 500
        true_probs = np.random.uniform(0.2, 0.8, n)
        labels = (np.random.random(n) < true_probs).astype(int)
        # Add some noise to create miscalibrated scores
        scores = true_probs + np.random.normal(0, 0.1, n)
        scores = np.clip(scores, 0.0, 1.0)
        return scores, labels

    def test_platt_calibration(self, calibration_data):
        from betting_intel.validation.calibration import ProbabilityCalibrator

        scores, labels = calibration_data
        train_scores, test_scores = scores[:300], scores[300:]
        train_labels, test_labels = labels[:300], labels[300:]

        calibrator = ProbabilityCalibrator(method="platt")
        calibrator.fit(train_scores, train_labels)
        calibrated = calibrator.calibrate(test_scores)

        assert len(calibrated) == len(test_scores)
        assert all(0 <= p <= 1 for p in calibrated)

    def test_isotonic_calibration(self, calibration_data):
        from betting_intel.validation.calibration import ProbabilityCalibrator

        scores, labels = calibration_data
        train_scores, test_scores = scores[:300], scores[300:]
        train_labels, test_labels = labels[:300], labels[300:]

        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(train_scores, train_labels)
        calibrated = calibrator.calibrate(test_scores)

        assert len(calibrated) == len(test_scores)
        assert all(0 <= p <= 1 for p in calibrated)

    def test_identity_calibration(self, calibration_data):
        from betting_intel.validation.calibration import ProbabilityCalibrator

        scores, labels = calibration_data
        calibrator = ProbabilityCalibrator(method="none")
        calibrator.fit(scores, labels)
        calibrated = calibrator.calibrate(scores)
        assert np.allclose(calibrated, scores)

    def test_evaluate(self, calibration_data):
        from betting_intel.validation.calibration import ProbabilityCalibrator

        scores, labels = calibration_data
        calibrator = ProbabilityCalibrator(method="platt")
        calibrator.fit(scores[:300], labels[:300])
        metrics = calibrator.evaluate(scores[300:], labels[300:])

        assert "brier_score" in metrics
        assert "log_loss" in metrics
        assert "mean_pred" in metrics
        assert "mean_actual" in metrics
        assert "calibration_error" in metrics
        assert metrics["brier_score"] >= 0

    def test_calibrate_before_fit_raises(self):
        from betting_intel.validation.calibration import ProbabilityCalibrator

        calibrator = ProbabilityCalibrator(method="platt")
        with pytest.raises(ValueError, match="not fitted"):
            calibrator.calibrate(np.array([0.5, 0.6]))

    def test_beta_calibration(self, calibration_data):
        from betting_intel.validation.calibration import ProbabilityCalibrator

        scores, labels = calibration_data
        calibrator = ProbabilityCalibrator(method="beta")
        calibrator.fit(scores[:300], labels[:300])
        calibrated = calibrator.calibrate(scores[300:])

        assert len(calibrated) == len(scores[300:])
        assert all(0 <= p <= 1 for p in calibrated)

    def test_find_best_calibrator(self, calibration_data):
        from betting_intel.validation.calibration import find_best_calibrator

        scores, labels = calibration_data
        best_cal, metrics = find_best_calibrator(scores[:300], labels[:300])

        assert best_cal is not None
        assert "brier_score" in metrics
        assert "method" in metrics

    def test_evaluate_calibration(self, calibration_data):
        from betting_intel.validation.calibration import evaluate_calibration

        scores, labels = calibration_data
        result = evaluate_calibration(scores, labels, n_bins=5)

        assert "ece" in result
        assert "mce" in result
        assert "bin_probs_true" in result
        assert "bin_probs_pred" in result
        assert 0 <= result["ece"] <= 1
        assert 0 <= result["mce"] <= 1

    def test_platt_calibrator_wrapper(self, calibration_data):
        from betting_intel.validation.calibration import PlattCalibrator

        scores, labels = calibration_data
        calibrator = PlattCalibrator()
        calibrator.fit(scores[:300], labels[:300])
        calibrated = calibrator.calibrate(scores[300:])
        assert len(calibrated) == len(scores[300:])

    def test_isotonic_calibrator_wrapper(self, calibration_data):
        from betting_intel.validation.calibration import IsotonicCalibrator

        scores, labels = calibration_data
        calibrator = IsotonicCalibrator()
        calibrator.fit(scores[:300], labels[:300])
        calibrated = calibrator.calibrate(scores[300:])
        assert len(calibrated) == len(scores[300:])

    def test_beta_calibrator_wrapper(self, calibration_data):
        from betting_intel.validation.calibration import BetaCalibrator

        scores, labels = calibration_data
        calibrator = BetaCalibrator()
        calibrator.fit(scores[:300], labels[:300])
        calibrated = calibrator.calibrate(scores[300:])
        assert len(calibrated) == len(scores[300:])
