"""
Model Performance Tracker — persistent R²/MAE history for drift detection.

Logs performance metrics every time a model is trained, then compares
new runs against historical baselines to flag drift (degradation).

Data persisted to: logs/model_performance.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from betting_intel.pipeline.bootstrap import PROJECT_ROOT, logger

# Default storage location
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "logs" / "model_performance.json"


class ModelPerformanceTracker:
    """Persistent tracker that records R²/MAE over time and detects drift.

    Each training run appends a metrics entry to a JSON file.
    On subsequent runs, the tracker compares new metrics against
    a rolling baseline to flag degradation.

    Usage:
        tracker = ModelPerformanceTracker()
        tracker.record_run(
            model_name="pipeline_ensemble",
            test_r2=0.57,
            test_mae=11.2,
            train_r2=0.72,
            train_mae=9.1,
            n_features=50,
            n_samples=1200,
            n_folds=5,
            models_used=["ridge", "lightgbm", "catboost"],
        )
        alerts = tracker.check_drift(model_name="pipeline_ensemble")
    """

    def __init__(self, history_path: Optional[Path] = None):
        self.history_path = history_path or DEFAULT_HISTORY_PATH
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, Any]] = self._load_history()

    # ── Public API ─────────────────────────────────────────────────

    def record_run(
        self,
        model_name: str,
        test_r2: Optional[float] = None,
        test_mae: Optional[float] = None,
        train_r2: Optional[float] = None,
        train_mae: Optional[float] = None,
        n_features: int = 0,
        n_samples: int = 0,
        n_folds: int = 0,
        models_used: Optional[List[str]] = None,
        mode: str = "walk-forward",
        **extra: Any,
    ) -> Dict[str, Any]:
        """Record a model training run's performance metrics.

        Both test_r2 and test_mae are optional — classification or retrain
        runs may only log Brier score or no MAE at all.

        Args:
            model_name: Identifier for the trained model
            test_r2: Out-of-sample R² (optional — not computed in retrain paths)
            test_mae: Out-of-sample MAE (optional)
            train_r2: In-sample R² (for overfitting gap tracking)
            train_mae: In-sample MAE
            n_features: Number of features used
            n_samples: Number of training samples
            n_folds: Number of cross-validation folds
            models_used: List of model names in the ensemble
            mode: Training mode (e.g. "walk-forward", "full-data", "weekly", "monthly")
            **extra: Any additional metrics to store

        Returns:
            The recorded entry dict
        """
        entry: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "n_features": n_features,
            "n_samples": n_samples,
            "n_folds": n_folds,
            "models_used": models_used or [],
            "mode": mode,
        }
        if test_r2 is not None:
            entry["test_r2"] = round(test_r2, 4)
        if test_mae is not None:
            entry["test_mae"] = round(test_mae, 2)
        if train_r2 is not None:
            entry["train_r2"] = round(train_r2, 4)
            if test_r2 is not None:
                entry["r2_gap"] = round(train_r2 - test_r2, 4)
        if train_mae is not None:
            entry["train_mae"] = round(train_mae, 2)

        # Store any extra metrics (brier, etc.)
        for k, v in extra.items():
            entry[k] = v

        self._history.append(entry)
        self._save_history()

        # Print summary
        r2_str = f"R²={entry['test_r2']:.3f}  " if "test_r2" in entry else ""
        mae_str = f"MAE={entry['test_mae']:.1f}  " if "test_mae" in entry else ""
        gap_str = f"gap={entry['r2_gap']:.3f}  " if "r2_gap" in entry else ""
        extra_vals = " | ".join(f"{k}={v}" for k, v in extra.items())
        extra_str = f" {extra_vals}" if extra_vals else ""
        print(
            f"  📈  Performance: {model_name}  "
            f"{r2_str}{mae_str}{gap_str}"
            f"({n_samples} samples, {n_features} features, {n_folds} folds)"
            f"{extra_str}"
        )

        return entry

    def check_drift(
        self,
        model_name: str = "pipeline_ensemble",
        n_baseline: int = 3,
        r2_threshold: float = -0.15,
        mae_threshold: float = 0.20,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Check for drift by comparing recent runs against historical baseline.

        Args:
            model_name: Which model's history to check
            n_baseline: Number of earliest runs to use as baseline
            r2_threshold: Max acceptable R² drop (e.g. -0.15 = -0.15 R²)
            mae_threshold: Max acceptable MAE increase fraction (e.g. 0.20 = 20%)
            verbose: Print drift report to console

        Returns:
            Dict with drift status, alerts, and baseline/current metrics
        """
        runs = [r for r in self._history if r.get("model_name") == model_name]

        # Only runs with BOTH test_r2 AND test_mae are drift-checkable
        # (retrain/classification runs only log Brier, not R²/MAE)
        runs_with_metrics = [
            r for r in runs if "test_r2" in r and "test_mae" in r
        ]

        if len(runs_with_metrics) < 3:
            reason = (
                f"insufficient_history ({len(runs_with_metrics)} runs with R²/MAE, need ≥3"
                f" of {len(runs)} total)"
            )
            return {
                "drift_detected": False,
                "status": reason,
                "n_runs": len(runs),
                "n_runs_with_metrics": len(runs_with_metrics),
                "alerts": [],
            }

        baseline = runs_with_metrics[:n_baseline]
        recent = runs_with_metrics[-n_baseline:]

        baseline_r2 = float(sum(r["test_r2"] for r in baseline)) / len(baseline)
        recent_r2 = float(sum(r["test_r2"] for r in recent)) / len(recent)
        baseline_mae = float(sum(r["test_mae"] for r in baseline)) / len(baseline)
        recent_mae = float(sum(r["test_mae"] for r in recent)) / len(recent)

        r2_change = recent_r2 - baseline_r2
        mae_change_frac = (recent_mae - baseline_mae) / max(baseline_mae, 0.01)

        alerts = []
        if r2_change < r2_threshold:
            alerts.append({
                "type": "r2_degradation",
                "metric": "test_r2",
                "baseline": round(baseline_r2, 4),
                "current": round(recent_r2, 4),
                "change": round(r2_change, 4),
                "severity": "high" if r2_change < r2_threshold * 1.5 else "medium",
                "message": (
                    f"R² dropped from {baseline_r2:.3f} → {recent_r2:.3f} "
                    f"(Δ={r2_change:+.3f})"
                ),
            })
        if mae_change_frac > mae_threshold:
            alerts.append({
                "type": "mae_degradation",
                "metric": "test_mae",
                "baseline": round(baseline_mae, 2),
                "current": round(recent_mae, 2),
                "change": round(mae_change_frac, 4),
                "severity": "high" if mae_change_frac > mae_threshold * 1.5 else "medium",
                "message": (
                    f"MAE increased from {baseline_mae:.1f} → {recent_mae:.1f} "
                    f"(+{mae_change_frac:.1%})"
                ),
            })

        drift_detected = len(alerts) > 0

        result: Dict[str, Any] = {
            "drift_detected": drift_detected,
            "status": "drift_detected" if drift_detected else "ok",
            "n_runs": len(runs),
            "n_baseline_runs": n_baseline,
            "baseline_r2": round(baseline_r2, 4),
            "baseline_mae": round(baseline_mae, 2),
            "recent_r2": round(recent_r2, 4),
            "recent_mae": round(recent_mae, 2),
            "r2_change": round(r2_change, 4),
            "mae_change_frac": round(mae_change_frac, 4),
            "alerts": alerts,
        }

        if verbose:
            self._print_drift_report(result, model_name)

        return result

    def get_history(
        self,
        model_name: Optional[str] = None,
        n_last: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get performance history, optionally filtered by model name.

        Args:
            model_name: If set, only return runs for this model
            n_last: If set, only return the last N runs

        Returns:
            List of performance entries
        """
        runs = self._history
        if model_name:
            runs = [r for r in runs if r.get("model_name") == model_name]
        if n_last:
            runs = runs[-n_last:]
        return list(runs)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked performance data.

        Returns:
            Dict with total runs, models tracked, and per-model stats
        """
        if not self._history:
            return {"n_runs": 0, "models_tracked": 0, "per_model": {}}

        model_names = set(r.get("model_name", "unknown") for r in self._history)
        per_model = {}
        for mn in sorted(model_names):
            runs = [r for r in self._history if r.get("model_name") == mn]
            r2s = [r.get("test_r2", 0) for r in runs if "test_r2" in r]
            maes = [r.get("test_mae", 0) for r in runs if "test_mae" in r]
            per_model[mn] = {
                "n_runs": len(runs),
                "last_r2": round(r2s[-1], 4) if r2s else None,
                "max_r2": round(max(r2s), 4) if r2s else None,
                "min_r2": round(min(r2s), 4) if r2s else None,
                "last_mae": round(maes[-1], 2) if maes else None,
                "avg_mae": round(float(sum(maes)) / len(maes), 2) if maes else None,
                "first_run": runs[0].get("timestamp", ""),
                "last_run": runs[-1].get("timestamp", ""),
            }

        return {
            "n_runs": len(self._history),
            "models_tracked": len(model_names),
            "per_model": per_model,
        }

    def clear_history(self, model_name: Optional[str] = None) -> int:
        """Clear performance history for a specific model or all models.

        Args:
            model_name: If set, only clear runs for this model

        Returns:
            Number of entries removed
        """
        if model_name is None:
            n = len(self._history)
            self._history = []
            self._save_history()
            return n

        before = len(self._history)
        self._history = [r for r in self._history if r.get("model_name") != model_name]
        n_removed = before - len(self._history)
        if n_removed > 0:
            self._save_history()
        return n_removed

    # ── Internal ───────────────────────────────────────────────────

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load performance history from disk."""
        if not self.history_path.exists():
            logger.debug(f"No performance history found at {self.history_path}")
            return []
        try:
            with open(self.history_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            # Legacy format: {"runs": [...]}
            if isinstance(data, dict) and "runs" in data:
                return data["runs"]
            logger.warning(f"Unexpected format in {self.history_path}, starting fresh")
            return []
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load performance history: {e}")
            return []

    def _save_history(self) -> None:
        """Save performance history to disk."""
        try:
            with open(self.history_path, "w") as f:
                json.dump(self._history, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save performance history: {e}")

    def _print_drift_report(
        self, result: Dict[str, Any], model_name: str
    ) -> None:
        """Print drift check results to console."""
        alerts = result.get("alerts", [])
        if alerts:
            print(f"\n  🌊  DRIFT DETECTED for {model_name}:")
            for alert in alerts:
                sev = "🔴" if alert.get("severity") == "high" else "🟡"
                print(f"       {sev}  {alert['message']}")
        else:
            print(
                f"  🌊  Drift check: OK  "
                f"(R²: {result['baseline_r2']:.3f} → {result['recent_r2']:.3f}, "
                f"MAE: {result['baseline_mae']:.1f} → {result['recent_mae']:.1f})"
            )


# ── Singleton for convenient import ──────────────────────────────────────
_performance_tracker: Optional[ModelPerformanceTracker] = None


def get_performance_tracker() -> ModelPerformanceTracker:
    """Get or create the global ModelPerformanceTracker singleton."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = ModelPerformanceTracker()
    return _performance_tracker
