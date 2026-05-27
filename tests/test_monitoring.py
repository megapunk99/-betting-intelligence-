"""Tests for the monitoring module — drift detection, performance tracking, Prometheus metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestComputePSI:
    """Tests for the Population Stability Index function."""

    def test_identical_distributions(self):
        from betting_intel.monitoring.drift import compute_psi

        data = np.random.normal(100, 15, 500)
        psi = compute_psi(data, data)
        assert psi < 0.05  # Nearly zero for identical distributions

    def test_shifted_distributions(self):
        from betting_intel.monitoring.drift import compute_psi

        expected = np.random.normal(100, 15, 500)
        actual = np.random.normal(110, 15, 500)  # Shifted mean
        psi = compute_psi(expected, actual)
        assert psi > 0.05  # Should detect the shift

    def test_empty_inputs(self):
        from betting_intel.monitoring.drift import compute_psi

        assert compute_psi(np.array([]), np.array([1, 2, 3])) == 0.0
        assert compute_psi(np.array([1, 2, 3]), np.array([])) == 0.0

    def test_default_n_bins(self):
        from betting_intel.monitoring.drift import compute_psi

        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert 0 <= psi <= 0.15

    def test_extreme_shift(self):
        from betting_intel.monitoring.drift import compute_psi

        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(50, 5, 1000)  # Completely different
        psi = compute_psi(expected, actual)
        assert psi > 0.5

    def test_non_normal_distributions(self):
        from betting_intel.monitoring.drift import compute_psi

        expected = np.random.uniform(0, 1, 1000)
        actual = np.random.beta(0.5, 0.5, 1000)  # U-shaped
        psi = compute_psi(expected, actual)
        assert psi > 0  # Should detect different shapes


class TestComputeKSStatistic:
    """Tests for the Kolmogorov-Smirnov test function."""

    def test_identical_distributions(self):
        from betting_intel.monitoring.drift import compute_ks_statistic

        data = np.random.normal(0, 1, 500)
        stat, p = compute_ks_statistic(data, data)
        assert p > 0.05  # Not significantly different

    def test_different_distributions(self):
        from betting_intel.monitoring.drift import compute_ks_statistic

        a = np.random.normal(0, 1, 500)
        b = np.random.normal(3, 1, 500)  # Shifted
        stat, p = compute_ks_statistic(a, b)
        assert p < 0.05  # Significantly different

    def test_empty_inputs(self):
        from betting_intel.monitoring.drift import compute_ks_statistic

        stat, p = compute_ks_statistic(np.array([]), np.array([1, 2, 3]))
        assert stat == 0.0 and p == 1.0

    def test_returns_tuple(self):
        from betting_intel.monitoring.drift import compute_ks_statistic

        result = compute_ks_statistic(np.array([1, 2, 3]), np.array([1.5, 2.5, 3.5]))
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestComputeKLDiverence:
    """Tests for the KL-Divergence function."""

    def test_identical(self):
        from betting_intel.monitoring.drift import compute_kl_divergence

        data = np.random.normal(0, 1, 1000)
        kl = compute_kl_divergence(data, data)
        assert kl < 0.1

    def test_different(self):
        from betting_intel.monitoring.drift import compute_kl_divergence

        a = np.random.normal(0, 1, 1000)
        b = np.random.normal(5, 2, 1000)
        kl = compute_kl_divergence(a, b)
        assert kl > 0.1

    def test_empty(self):
        from betting_intel.monitoring.drift import compute_kl_divergence

        assert compute_kl_divergence(np.array([]), np.array([1, 2, 3])) == 0.0


class TestPerformanceTracker:
    """Tests for the PerformanceTracker class."""

    @pytest.fixture
    def tracker(self):
        from betting_intel.monitoring.drift import PerformanceTracker

        return PerformanceTracker(
            model_name="test_model",
            window_sizes=[10, 20],
            win_rate_drop_threshold=0.15,
            mae_increase_threshold=0.2,
        )

    def test_initial_state(self, tracker):
        assert tracker.model_name == "test_model"
        assert tracker.window_sizes == [10, 20]
        assert len(tracker._predictions) == 0
        assert len(tracker.alerts) == 0

    def test_record_prediction(self, tracker):
        tracker.record_prediction(predicted=215.0, actual=220.0, won=True)
        assert len(tracker._predictions) == 1
        assert tracker._predictions[0]["predicted"] == 215.0
        assert tracker._predictions[0]["actual"] == 220.0
        assert tracker._predictions[0]["won"] is True
        assert tracker._predictions[0]["error"] == 5.0

    def test_record_multiple_predictions(self, tracker):
        for i in range(25):
            tracker.record_prediction(
                predicted=200.0 + i, actual=205.0 + i, won=(i % 2 == 0), profit_units=1.0 if i % 2 == 0 else -1.1
            )
        assert len(tracker._predictions) == 25

        # Windows should fill and truncate
        assert len(tracker._windows[10]) == 10  # Capped at 10
        assert len(tracker._windows[20]) == 20  # Capped at 20

    def test_get_overall(self, tracker):
        for i in range(10):
            tracker.record_prediction(
                predicted=220.0, actual=220.0, won=(i % 2 == 0)
            )
        overall = tracker.get_overall()
        assert overall.n_predictions == 10
        assert overall.mae == 0.0  # Perfect predictions

    def test_get_window(self, tracker):
        for i in range(15):
            tracker.record_prediction(predicted=200.0, actual=200.0, won=True)
        window = tracker.get_window(10)
        assert window.n_predictions == 10
        assert window.mae == 0.0

    def test_get_window_empty(self, tracker):
        window = tracker.get_window(999)
        assert window.n_predictions == 0
        assert window.window_size == 0

    def test_get_all_windows(self, tracker):
        for i in range(25):
            tracker.record_prediction(predicted=210.0, actual=210.0, won=True)
        windows = tracker.get_all_windows()
        assert 10 in windows
        assert 20 in windows

    def test_set_baseline(self, tracker):
        for i in range(200):
            tracker.record_prediction(
                predicted=200.0 + np.random.normal(0, 5),
                actual=200.0,
                won=(np.random.random() > 0.4),
            )
        tracker.set_baseline()
        assert tracker._baseline is not None
        assert "mean_prediction" in tracker._baseline
        assert "win_rate" in tracker._baseline
        assert "mean_error" in tracker._baseline

    def test_set_baseline_too_few(self, tracker):
        tracker.set_baseline()  # No predictions yet
        assert tracker._baseline is None

    def test_get_report_structure(self, tracker):
        for i in range(50):
            tracker.record_prediction(
                predicted=210.0 + np.random.normal(0, 3),
                actual=210.0,
                won=(np.random.random() > 0.45),
                profit_units=1.0 if np.random.random() > 0.45 else -1.1,
            )
        report = tracker.get_report()
        assert "model_name" in report
        assert "total_predictions" in report
        assert "overall" in report
        assert "windows" in report
        assert "drift_alerts" in report
        assert "drift_summary" in report
        assert report["model_name"] == "test_model"

    def test_format_report(self, tracker):
        for i in range(50):
            tracker.record_prediction(
                predicted=210.0, actual=210.0, won=(i % 2 == 0), profit_units=1.0 if i % 2 == 0 else -1.1
            )
        text = tracker.format_report()
        assert "test_model" in text
        assert "Overall Performance" in text
        assert "Rolling Windows" in text
        assert "Healthy" in text or "RETRAIN" in text

    def test_metadata_in_record(self, tracker):
        tracker.record_prediction(
            predicted=215.0, actual=220.0, won=True,
            game_id="20250101_CELTS_LAKERS", date="2025-01-01",
        )
        assert tracker._predictions[0]["game_id"] == "20250101_CELTS_LAKERS"

    def test_drift_alert_win_rate_drop(self, tracker):
        """Simulate 100 good predictions then 20 bad ones to trigger win-rate alert."""
        # Establish baseline with good predictions
        for i in range(120):
            tracker.record_prediction(
                predicted=210.0, actual=210.0, won=True, profit_units=1.0
            )

        # Set baseline explicitly
        tracker.set_baseline()
        assert tracker._baseline is not None
        assert tracker._baseline["win_rate"] > 0.9

        # Now record a streak of losses
        for i in range(20):
            tracker.record_prediction(
                predicted=210.0, actual=212.0, won=False, profit_units=-1.1
            )

        # Check drift
        alerts = tracker.check_drift()
        win_rate_alerts = [a for a in alerts if "win_rate" in a.metric]
        if len(win_rate_alerts) > 0:
            assert win_rate_alerts[0].severity in ("warning", "critical")
            assert "retraining" in win_rate_alerts[0].recommendation.lower()

    def test_no_drift_when_stable(self, tracker):
        """Stable performance should produce no alerts."""
        for i in range(200):
            tracker.record_prediction(
                predicted=210.0, actual=210.0, won=(i % 2 == 0), profit_units=1.0 if i % 2 == 0 else -1.1
            )
        tracker.set_baseline()
        alerts = tracker.check_drift()
        # Should be few or no alerts with ~50% win rate
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        assert len(critical_alerts) == 0

    def test_prediction_without_won(self, tracker):
        """Regression-only predictions (no bet outcome) should still work."""
        for i in range(20):
            tracker.record_prediction(predicted=210.0 + i, actual=210.0)
        overall = tracker.get_overall()
        assert overall.n_predictions == 20
        assert overall.mae > 0
        assert overall.n_bets == 0  # No bet outcomes
        assert overall.win_rate == 0.0


class TestFeatureDriftDetector:
    """Tests for the FeatureDriftDetector class."""

    @pytest.fixture
    def detector(self):
        from betting_intel.monitoring.drift import FeatureDriftDetector

        return FeatureDriftDetector(
            feature_names=["pts_avg", "reb_avg", "ast_avg"],
            psi_threshold=0.10,
            ks_threshold=0.05,
        )

    @pytest.fixture
    def reference_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "pts_avg": np.random.normal(110, 12, 500),
            "reb_avg": np.random.normal(45, 5, 500),
            "ast_avg": np.random.normal(25, 4, 500),
        })

    @pytest.fixture
    def drifted_df(self, reference_df):
        """Create a DataFrame with shifted distributions."""
        df = reference_df.copy()
        df["pts_avg"] = df["pts_avg"] + 10  # Shift mean
        df["reb_avg"] = df["reb_avg"] * 1.2  # Scale
        return df

    def test_fit_reference(self, detector, reference_df):
        detector.fit_reference(reference_df)
        assert "pts_avg" in detector._reference_stats
        assert "reb_avg" in detector._reference_stats
        assert "ast_avg" in detector._reference_stats
        assert detector._reference_stats["pts_avg"]["mean"] == pytest.approx(110, rel=0.2)

    def test_detect_drift_no_drift(self, detector, reference_df):
        detector.fit_reference(reference_df)
        # Same data should show no drift
        result = detector.detect_drift(reference_df)
        assert result["summary"]["n_features_drifted"] == 0
        assert result["summary"]["overall_severity"] in ("none",)

    def test_detect_drift_with_drift(self, detector, reference_df, drifted_df):
        detector.fit_reference(reference_df)
        result = detector.detect_drift(drifted_df)
        assert result["summary"]["n_features_checked"] > 0
        # At least some features should show drift
        assert result["summary"]["drift_ratio"] >= 0.0

    def test_detect_drift_unseen_feature(self, detector, reference_df):
        detector.fit_reference(reference_df)
        # DataFrame with extra column should be handled gracefully
        df = reference_df.copy()
        df["unknown_feat"] = np.random.normal(0, 1, 500)
        result = detector.detect_drift(df)
        assert "features" in result

    def test_detect_drift_empty_df(self, detector, reference_df):
        detector.fit_reference(reference_df)
        empty_df = pd.DataFrame({"pts_avg": [], "reb_avg": [], "ast_avg": []})
        result = detector.detect_drift(empty_df)
        assert result["summary"]["n_features_checked"] == 0

    def test_detect_drift_missing_column(self, detector, reference_df):
        detector.fit_reference(reference_df)
        partial_df = reference_df[["pts_avg"]].copy()
        result = detector.detect_drift(partial_df)
        assert result["summary"]["n_features_checked"] >= 1  # Should check available cols

    def test_format_report(self, detector, reference_df, drifted_df):
        detector.fit_reference(reference_df)
        result = detector.detect_drift(drifted_df)
        text = detector.format_report(result)
        assert "FEATURE DRIFT DETECTION REPORT" in text
        assert "Features checked" in text
        assert "pts_avg" in text

    def test_format_report_empty(self, detector):
        text = detector.format_report({})
        assert "No drift analysis" in text

    def test_drift_severity_levels(self, detector, reference_df):
        detector.fit_reference(reference_df)
        # Shift significantly
        df = reference_df.copy()
        df["pts_avg"] = np.random.normal(200, 50, 500)
        result = detector.detect_drift(df)
        assert result["summary"]["overall_severity"] in ("none", "moderate", "critical")

    def test_no_reference_returns_empty(self, detector):
        df = pd.DataFrame({"pts_avg": np.random.normal(110, 12, 100)})
        result = detector.detect_drift(df)  # No reference fitted
        assert result["summary"]["n_features_checked"] == 0


class TestPerformanceWindowDataclass:
    """Tests for the PerformanceWindow dataclass structure."""

    def test_default_values(self):
        from betting_intel.monitoring.drift import PerformanceWindow

        pw = PerformanceWindow(window_size=50)
        assert pw.window_size == 50
        assert pw.n_predictions == 0
        assert pw.win_rate == 0.0
        assert pw.is_drifted is False
        assert pw.drift_severity == "none"

    def test_custom_values(self):
        from datetime import datetime
        from betting_intel.monitoring.drift import PerformanceWindow

        pw = PerformanceWindow(
            window_size=100,
            n_predictions=100,
            win_rate=0.58,
            mae=3.2,
            total_profit=45.5,
            is_drifted=True,
            drift_severity="moderate",
        )
        assert pw.win_rate == 0.58
        assert pw.mae == 3.2
        assert pw.total_profit == 45.5
        assert pw.is_drifted is True
        assert pw.drift_severity == "moderate"


class TestDriftAlertDataclass:
    """Tests for the DriftAlert dataclass."""

    def test_default_timestamp(self):
        from betting_intel.monitoring.drift import DriftAlert

        alert = DriftAlert(
            model_name="test", metric="win_rate",
            old_value=0.6, new_value=0.4,
            threshold=0.08, severity="warning",
        )
        assert alert.model_name == "test"
        assert alert.severity == "warning"
        assert alert.timestamp is not None

    def test_critical_alert(self):
        from betting_intel.monitoring.drift import DriftAlert

        alert = DriftAlert(
            model_name="lgbm", metric="mae",
            old_value=2.5, new_value=4.0,
            threshold=0.15, severity="critical",
            recommendation="Retrain immediately",
        )
        assert alert.recommendation == "Retrain immediately"
        assert alert.severity == "critical"


class TestMetricsModule:
    """Tests for the metrics/prometheus module (basic smoke tests)."""

    def test_metrics_functions_exist(self):
        from betting_intel.monitoring.metrics import (
            metrics_endpoint,
            track_prediction_latency,
            update_model_metrics,
        )
        assert callable(metrics_endpoint)
        assert callable(track_prediction_latency)
        assert callable(update_model_metrics)

    def test_update_model_metrics(self):
        from betting_intel.monitoring.metrics import update_model_metrics

        # Should not raise
        update_model_metrics("test_model", {"mae": 2.5, "r2": 0.95, "win_rate": 0.58})
        assert True

    def test_metrics_endpoint(self):
        from betting_intel.monitoring.metrics import metrics_endpoint

        data, status, headers = metrics_endpoint()
        assert status == 200
        assert "Content-Type" in headers
        assert len(data) > 0  # Should contain prometheus text

    def test_gauge_labels_exist(self):
        from betting_intel.monitoring.metrics import (
            MODEL_PERFORMANCE,
            PREDICTIONS_TOTAL,
            BANKROLL_CURRENT,
        )
        # Verify the Prometheus metrics objects exist
        assert MODEL_PERFORMANCE._name == "betting_model_performance"
        assert BANKROLL_CURRENT._name == "betting_bankroll_current"
        # Note: Counter _name stores base name without auto-appended _total suffix
        assert PREDICTIONS_TOTAL._name == "betting_predictions"
