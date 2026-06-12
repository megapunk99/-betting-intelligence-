"""
Unit tests for ModelPerformanceTracker.
"""

import json
import tempfile
from pathlib import Path

import pytest

from betting_intel.pipeline.performance import ModelPerformanceTracker


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_history() -> Path:
    """Return a temporary file path for history JSON."""
    return Path(tempfile.mktemp(suffix=".json"))


@pytest.fixture
def tracker(tmp_history: Path) -> ModelPerformanceTracker:
    """Return a tracker backed by a temporary file."""
    return ModelPerformanceTracker(history_path=tmp_history)


# ── Basic record_run ────────────────────────────────────────────────────


class TestRecordRun:
    def test_records_basic_metrics(self, tracker: ModelPerformanceTracker):
        entry = tracker.record_run(
            model_name="test_model",
            test_r2=0.57,
            test_mae=11.2,
        )
        assert entry["model_name"] == "test_model"
        assert entry["test_r2"] == 0.57
        assert entry["test_mae"] == 11.2
        assert "timestamp" in entry
        assert entry["mode"] == "walk-forward"

    def test_records_with_train_metrics(self, tracker: ModelPerformanceTracker):
        entry = tracker.record_run(
            model_name="test_model",
            test_r2=0.57,
            test_mae=11.2,
            train_r2=0.72,
            train_mae=9.1,
        )
        assert entry["train_r2"] == 0.72
        assert entry["train_mae"] == 9.1
        assert entry["r2_gap"] == pytest.approx(0.15)

    def test_optional_r2_and_mae(self, tracker: ModelPerformanceTracker):
        """Retrain paths may not have R²/MAE (e.g. classification)."""
        entry = tracker.record_run(
            model_name="retrain_win_prob",
            n_samples=800,
            val_brier=0.215,
        )
        assert "test_r2" not in entry
        assert "test_mae" not in entry
        assert entry["val_brier"] == 0.215
        assert entry["n_samples"] == 800

    def test_extra_kwargs_stored(self, tracker: ModelPerformanceTracker):
        entry = tracker.record_run(
            model_name="test",
            test_r2=0.5,
            test_mae=10.0,
            tune=True,
            val_brier=0.2,
        )
        assert entry["tune"] is True
        assert entry["val_brier"] == 0.2

    def test_mode_defaults(self, tracker: ModelPerformanceTracker):
        entry = tracker.record_run(model_name="m", test_r2=0.5, test_mae=10.0)
        assert entry["mode"] == "walk-forward"

    def test_mode_explicit(self, tracker: ModelPerformanceTracker):
        entry = tracker.record_run(
            model_name="m", test_r2=0.5, test_mae=10.0, mode="monthly"
        )
        assert entry["mode"] == "monthly"


# ── Persistence (save/load) ────────────────────────────────────────────


class TestPersistence:
    def test_saves_to_disk(self, tmp_history: Path):
        t1 = ModelPerformanceTracker(history_path=tmp_history)
        t1.record_run(model_name="m", test_r2=0.5, test_mae=10.0)
        assert tmp_history.exists()
        data = json.loads(tmp_history.read_text())
        assert len(data) == 1
        assert data[0]["model_name"] == "m"

    def test_loads_from_disk(self, tmp_history: Path):
        t1 = ModelPerformanceTracker(history_path=tmp_history)
        t1.record_run(model_name="m", test_r2=0.5, test_mae=10.0)
        t2 = ModelPerformanceTracker(history_path=tmp_history)
        assert len(t2.get_history()) == 1

    def test_multiple_runs_accumulate(self, tmp_history: Path):
        t = ModelPerformanceTracker(history_path=tmp_history)
        for i in range(5):
            t.record_run(model_name="m", test_r2=0.5 + i * 0.05, test_mae=10.0)
        assert len(t.get_history()) == 5

    def test_empty_history_no_crash(self, tmp_history: Path):
        t = ModelPerformanceTracker(history_path=tmp_history)
        assert t.get_history() == []
        assert t.get_summary()["n_runs"] == 0

    def test_corrupt_json_handled_gracefully(self, tmp_history: Path):
        tmp_history.write_text("{invalid json!!!")
        t = ModelPerformanceTracker(history_path=tmp_history)
        assert t.get_history() == []  # Loaded as empty, not crashed

    def test_legacy_format_runs_key(self, tmp_history: Path):
        tmp_history.write_text(json.dumps({"runs": [{"model_name": "legacy"}]}))
        t = ModelPerformanceTracker(history_path=tmp_history)
        assert len(t.get_history()) == 1
        assert t.get_history()[0]["model_name"] == "legacy"


# ── Drift Detection ─────────────────────────────────────────────────────


class TestDriftDetection:
    def test_insufficient_history(self, tracker: ModelPerformanceTracker):
        result = tracker.check_drift(model_name="nonexistent", verbose=False)
        assert result["drift_detected"] is False
        assert "insufficient_history" in result["status"]

    def test_r2_degradation_detected(self, tracker: ModelPerformanceTracker):
        """3 good runs then 3 bad runs should trigger R² alert."""
        for i in range(3):
            tracker.record_run(
                model_name="m", test_r2=0.60 - i * 0.01, test_mae=10.0
            )
        for i in range(3):
            tracker.record_run(
                model_name="m", test_r2=0.40 - i * 0.02, test_mae=15.0
            )
        result = tracker.check_drift(model_name="m", verbose=False)
        assert result["drift_detected"] is True
        assert any(a["type"] == "r2_degradation" for a in result["alerts"])

    def test_mae_degradation_detected(self, tracker: ModelPerformanceTracker):
        for i in range(3):
            tracker.record_run(
                model_name="m", test_r2=0.5, test_mae=10.0 + i
            )
        for i in range(3):
            tracker.record_run(
                model_name="m", test_r2=0.5, test_mae=18.0 + i
            )
        result = tracker.check_drift(model_name="m", verbose=False)
        assert result["drift_detected"] is True
        assert any(a["type"] == "mae_degradation" for a in result["alerts"])

    def test_no_drift_when_stable(self, tracker: ModelPerformanceTracker):
        for _ in range(6):
            tracker.record_run(model_name="m", test_r2=0.56, test_mae=11.0)
        result = tracker.check_drift(model_name="m", verbose=False)
        assert result["drift_detected"] is False

    def test_custom_thresholds(self, tracker: ModelPerformanceTracker):
        for _ in range(3):
            tracker.record_run(model_name="m", test_r2=0.5, test_mae=10.0)
        for _ in range(3):
            tracker.record_run(model_name="m", test_r2=0.48, test_mae=10.5)
        # Tight thresholds
        result = tracker.check_drift(
            model_name="m", r2_threshold=-0.01, mae_threshold=0.01, verbose=False
        )
        assert result["drift_detected"] is True

    def test_skips_runs_without_r2_or_mae(self, tracker: ModelPerformanceTracker):
        """Runs without test_r2/test_mae should not crash drift check."""
        for _ in range(6):
            # Record retrain-style entries (no R²)
            tracker.record_run(model_name="retrain_m", val_brier=0.2)
        result = tracker.check_drift(model_name="retrain_m", verbose=False)
        # Should not crash, should detect insufficient metrics
        assert "drift_detected" in result


# ── History Filtering ───────────────────────────────────────────────────


class TestHistoryFiltering:
    def test_get_history_all(self, tracker: ModelPerformanceTracker):
        for name in ["a", "b", "a"]:
            tracker.record_run(model_name=name, test_r2=0.5, test_mae=10.0)
        assert len(tracker.get_history()) == 3

    def test_get_history_by_model(self, tracker: ModelPerformanceTracker):
        for name in ["a", "b", "a"]:
            tracker.record_run(model_name=name, test_r2=0.5, test_mae=10.0)
        assert len(tracker.get_history(model_name="a")) == 2
        assert len(tracker.get_history(model_name="b")) == 1

    def test_get_history_n_last(self, tracker: ModelPerformanceTracker):
        for i in range(5):
            tracker.record_run(model_name="m", test_r2=0.5, test_mae=float(i))
        hist = tracker.get_history(n_last=2)
        assert len(hist) == 2

    def test_get_history_model_and_n_last(self, tracker: ModelPerformanceTracker):
        for name in ["a", "a", "a", "b", "b"]:
            tracker.record_run(model_name=name, test_r2=0.5, test_mae=10.0)
        hist = tracker.get_history(model_name="a", n_last=2)
        assert len(hist) == 2


# ── Summary ─────────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_empty(self, tracker: ModelPerformanceTracker):
        s = tracker.get_summary()
        assert s["n_runs"] == 0
        assert s["models_tracked"] == 0

    def test_summary_with_runs(self, tracker: ModelPerformanceTracker):
        tracker.record_run(model_name="a", test_r2=0.5, test_mae=10.0)
        tracker.record_run(model_name="a", test_r2=0.6, test_mae=9.0)
        tracker.record_run(model_name="b", test_r2=0.4, test_mae=12.0)
        s = tracker.get_summary()
        assert s["n_runs"] == 3
        assert s["models_tracked"] == 2
        assert s["per_model"]["a"]["last_r2"] == 0.6
        assert s["per_model"]["a"]["max_r2"] == 0.6
        assert s["per_model"]["a"]["min_r2"] == 0.5
        assert s["per_model"]["a"]["last_mae"] == 9.0
        assert s["per_model"]["b"]["last_r2"] == 0.4


# ── Clear History ───────────────────────────────────────────────────────


class TestClearHistory:
    def test_clear_all(self, tracker: ModelPerformanceTracker):
        for name in ["a", "b"]:
            tracker.record_run(model_name=name, test_r2=0.5, test_mae=10.0)
        n = tracker.clear_history()  # Clears all
        assert n == 2
        assert len(tracker.get_history()) == 0

    def test_clear_by_model(self, tracker: ModelPerformanceTracker):
        for name in ["a", "a", "b"]:
            tracker.record_run(model_name=name, test_r2=0.5, test_mae=10.0)
        n = tracker.clear_history(model_name="a")
        assert n == 2
        assert len(tracker.get_history()) == 1
        assert len(tracker.get_history(model_name="b")) == 1

    def test_clear_nonexistent_model(self, tracker: ModelPerformanceTracker):
        tracker.record_run(model_name="a", test_r2=0.5, test_mae=10.0)
        n = tracker.clear_history(model_name="nonexistent")
        assert n == 0
        assert len(tracker.get_history()) == 1

    def test_clear_empty(self, tracker: ModelPerformanceTracker):
        n = tracker.clear_history()
        assert n == 0


# ── Integration with modeling.py ────────────────────────────────────────


class TestIntegration:
    def test_walk_forward_metrics_stored(self, tmp_history: Path):
        """Simulate the pattern used in modeling.py _try_stacking_ensemble()."""
        t = ModelPerformanceTracker(history_path=tmp_history)
        t.record_run(
            model_name="pipeline_ensemble",
            test_r2=0.5732,
            test_mae=11.23,
            train_r2=0.7189,
            train_mae=9.07,
            n_features=50,
            n_samples=1234,
            n_folds=5,
            models_used=["ridge", "lightgbm", "catboost"],
            mode="walk-forward",
        )
        entries = t.get_history(model_name="pipeline_ensemble")
        assert len(entries) == 1
        e = entries[0]
        assert e["test_r2"] == pytest.approx(0.5732)
        assert e["test_mae"] == pytest.approx(11.23)
        assert e["train_r2"] == pytest.approx(0.7189)
        assert e["r2_gap"] == pytest.approx(0.1457, abs=0.001)
        assert e["n_folds"] == 5

    def test_full_data_metrics_stored(self, tmp_history: Path):
        """Simulate the pattern used in modeling.py _train_all_data_model()."""
        t = ModelPerformanceTracker(history_path=tmp_history)
        t.record_run(
            model_name="pipeline_ensemble_full",
            test_r2=0.8102,
            test_mae=7.84,
            n_features=320,
            n_samples=1234,
            models_used=["lightgbm", "catboost", "ridge", "mlp_256"],
            mode="full-data",
        )
        entries = t.get_history(model_name="pipeline_ensemble_full")
        assert len(entries) == 1
        assert entries[0]["models_used"] == ["lightgbm", "catboost", "ridge", "mlp_256"]

    def test_retrain_total_metrics_stored(self, tmp_history: Path):
        """Simulate the pattern used in retrain_all.py train_total_models()."""
        t = ModelPerformanceTracker(history_path=tmp_history)
        t.record_run(
            model_name="retrain_total",
            test_mae=12.5,
            n_features=250,
            n_samples=900,
            n_folds=4,
            models_used=["gbr", "rfr", "ridge", "catboost"],
            mode="retrain",
            tune=True,
        )
        entries = t.get_history(model_name="retrain_total")
        assert len(entries) == 1
        assert "test_r2" not in entries[0]  # Optional — not provided
        assert entries[0]["test_mae"] == 12.5
        assert entries[0]["tune"] is True

    def test_retrain_win_prob_metrics_stored(self, tmp_history: Path):
        """Simulate the pattern used in retrain_all.py train_win_probability_model()."""
        t = ModelPerformanceTracker(history_path=tmp_history)
        t.record_run(
            model_name="retrain_win_prob",
            n_features=250,
            n_samples=900,
            models_used=["LogisticRegression+CalibratedClassifierCV"],
            mode="retrain",
            val_brier=0.2154,
        )
        entries = t.get_history(model_name="retrain_win_prob")
        assert len(entries) == 1
        assert "test_r2" not in entries[0]
        assert "test_mae" not in entries[0]
        assert entries[0]["val_brier"] == pytest.approx(0.2154)

    def test_cross_run_drift_detection(self, tmp_history: Path):
        """Simulate a full pipeline: 3 good runs → 3 bad runs → drift."""
        t = ModelPerformanceTracker(history_path=tmp_history)
        # Good runs
        for _ in range(3):
            t.record_run(model_name="pipeline_ensemble", test_r2=0.57, test_mae=11.0)
        # Bad runs
        for _ in range(3):
            t.record_run(model_name="pipeline_ensemble", test_r2=0.38, test_mae=16.0)
        result = t.check_drift(model_name="pipeline_ensemble", verbose=False)
        assert result["drift_detected"] is True
        assert result["baseline_r2"] == pytest.approx(0.57, abs=0.01)
        assert result["recent_r2"] == pytest.approx(0.38, abs=0.01)
