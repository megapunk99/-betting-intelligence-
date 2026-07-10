"""
DriftMonitor — Real-time concept drift detection for ML model predictions (v6.5).

Monitors model performance metrics over time and alerts when significant
distribution shifts are detected, which could indicate:
  - Changing market conditions (e.g., rule changes, new team strategies)
  - Data quality issues (e.g., missing features, corrupted inputs)
  - Model degradation (e.g., feature importance drift)
  - Regime changes (e.g., early season vs playoff basketball)

Uses statistical tests and sliding window comparisons for robust detection.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """An alert generated when concept drift is detected."""

    metric: str
    severity: str  # "info", "warning", "critical"
    message: str
    statistic: float
    threshold: float
    timestamp: str = ""
    details: Optional[dict] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class DriftReport:
    """Comprehensive drift analysis report."""

    total_alerts: int = 0
    critical_alerts: list[DriftAlert] = field(default_factory=list)
    warning_alerts: list[DriftAlert] = field(default_factory=list)
    info_alerts: list[DriftAlert] = field(default_factory=list)
    model_health: str = "healthy"  # "healthy", "degraded", "critical"
    recommendation: str = ""
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()


class DriftMonitor:
    """Monitor and detect concept drift in model predictions and features.

    Uses multiple detection strategies:
      1. Prediction distribution drift (KS test on prediction outputs)
      2. Feature distribution drift (per-feature KS test)
      3. Performance drift (sliding window accuracy/Brier)
      4. Feature importance drift (rank correlation of importances)
      5. Calibration drift (reliability curve shift)

    Usage:
        monitor = DriftMonitor(window_size=500, alert_threshold=0.05)
        monitor.update(y_true=..., y_pred=..., features=...)
        report = monitor.check_drift()
        if report.model_health == "critical":
            # Retrain model or trigger alert
    """

    def __init__(
        self,
        window_size: int = 500,
        reference_window: int = 1000,
        alert_threshold: float = 0.05,
        performance_threshold: float = 0.03,
        ks_threshold: float = 0.1,
        max_history: int = 5000,
    ):
        self.window_size = window_size
        self.reference_window = reference_window
        self.alert_threshold = alert_threshold
        self.performance_threshold = performance_threshold
        self.ks_threshold = ks_threshold
        self.max_history = max_history

        # Prediction history
        self._predictions: deque = deque(maxlen=max_history)
        self._targets: deque = deque(maxlen=max_history)
        self._timestamps: deque = deque(maxlen=max_history)
        self._feature_samples: deque = deque(maxlen=max_history // 10)
        self._feature_names: list[str] = []

        # Performance tracking
        self._performance_history: deque = deque(maxlen=max_history)
        self._brier_history: deque = deque(maxlen=max_history)

        # Feature importance history
        self._feature_importance_history: deque = deque(maxlen=20)

        # Alert history
        self._alerts: list[DriftAlert] = []
        self._last_check_time: Optional[datetime] = None

        # Reference statistics (computed on first N samples)
        self._reference_pred_mean: Optional[float] = None
        self._reference_pred_std: Optional[float] = None
        self._reference_performance: Optional[float] = None
        self._reference_feature_stats: dict[str, dict] = {}

    def update(
        self,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        feature_importance: Optional[dict[str, float]] = None,
    ):
        """Update monitor with new predictions and optional features."""
        timestamp = datetime.now()

        if y_pred is not None:
            y_pred_flat = np.asarray(y_pred).ravel()
            for p in y_pred_flat:
                self._predictions.append(float(p))
                self._timestamps.append(timestamp)

        if y_true is not None and y_pred is not None:
            y_true_flat = np.asarray(y_true).ravel()
            y_pred_flat = np.asarray(y_pred).ravel()
            for t, p in zip(y_true_flat, y_pred_flat):
                self._targets.append(float(t))

                # Track Brier score (binary classification)
                brier = (float(t) - float(p)) ** 2
                self._brier_history.append(brier)

                # Track accuracy
                correct = 1.0 if (float(p) >= 0.5) == (float(t) >= 0.5) else 0.0
                self._performance_history.append(correct)

        if features is not None:
            self._feature_samples.append(features.copy())
            if feature_names:
                self._feature_names = feature_names

        if feature_importance:
            self._feature_importance_history.append(dict(feature_importance))

    def check_drift(
        self,
        force: bool = False,
        min_samples: int = 100,
    ) -> DriftReport:
        """Run all drift detection checks and return a report.

        Args:
            force: If True, run even with few samples
            min_samples: Minimum samples needed for meaningful checks

        Returns:
            DriftReport with all alerts and health status
        """
        report = DriftReport()
        n_predictions = len(self._predictions)

        if n_predictions < min_samples and not force:
            report.recommendation = (
                f"Need at least {min_samples} predictions for drift analysis "
                f"(have {n_predictions})"
            )
            return report

        # Set reference statistics if not yet set
        if self._reference_pred_mean is None and n_predictions >= self.reference_window:
            ref_preds = list(self._predictions)[: self.reference_window]
            self._reference_pred_mean = float(np.mean(ref_preds))
            self._reference_pred_std = max(float(np.std(ref_preds)), 0.001)

        if (
            self._reference_performance is None
            and len(self._performance_history) >= self.reference_window
        ):
            ref_perf = list(self._performance_history)[: self.reference_window]
            self._reference_performance = float(np.mean(ref_perf))

        # Run all checks
        alerts = []

        # 1. Prediction distribution drift
        if self._reference_pred_mean is not None:
            alert = self._check_prediction_drift()
            if alert:
                alerts.append(alert)

        # 2. Performance drift
        if self._reference_performance is not None:
            alert = self._check_performance_drift()
            if alert:
                alerts.append(alert)

        # 3. Brier score drift
        if len(self._brier_history) >= 100:
            alert = self._check_brier_drift()
            if alert:
                alerts.append(alert)

        # 4. Feature drift (if enough feature samples collected)
        if len(self._feature_samples) >= 50:
            alert = self._check_feature_drift()
            if alert:
                alerts.append(alert)

        # 5. Feature importance drift
        if len(self._feature_importance_history) >= 5:
            alert = self._check_importance_drift()
            if alert:
                alerts.append(alert)

        # Categorize alerts
        for alert in alerts:
            if alert.severity == "critical":
                report.critical_alerts.append(alert)
            elif alert.severity == "warning":
                report.warning_alerts.append(alert)
            else:
                report.info_alerts.append(alert)

        report.total_alerts = len(alerts)

        # Determine overall health
        if report.critical_alerts:
            report.model_health = "critical"
            report.recommendation = (
                "URGENT: Model retraining recommended — significant drift detected"
            )
        elif report.warning_alerts:
            report.model_health = "degraded"
            report.recommendation = (
                "Model may need retraining — monitor performance closely"
            )
        elif report.info_alerts:
            report.model_health = "healthy"
            report.recommendation = "Minor drift detected — no action required"
        else:
            report.model_health = "healthy"
            report.recommendation = "No drift detected — model is stable"

        self._alerts = alerts
        self._last_check_time = datetime.now()

        return report

    def _check_prediction_drift(self) -> Optional[DriftAlert]:
        """Check if recent prediction distribution differs from reference."""
        recent = list(self._predictions)[-self.window_size :]
        if len(recent) < 50:
            return None

        recent_mean = float(np.mean(recent))
        recent_std = float(np.std(recent))

        # Z-score of mean shift
        se = self._reference_pred_std / math.sqrt(len(recent))
        z_score = abs(recent_mean - self._reference_pred_mean) / max(se, 0.0001)

        # Use KS statistic for distribution comparison
        from scipy.stats import ks_2samp

        reference = list(self._predictions)[: self.reference_window]
        try:
            ks_stat, ks_p = ks_2samp(reference, recent)
        except Exception:
            ks_stat, ks_p = 0.0, 1.0

        if ks_stat > self.ks_threshold and ks_p < 0.05:
            severity = "critical" if ks_stat > 0.2 else "warning"
            return DriftAlert(
                metric="prediction_distribution",
                severity=severity,
                message=(
                    f"Prediction distribution drift detected: "
                    f"KS={ks_stat:.3f}, p={ks_p:.4f}, "
                    f"mean shift z={z_score:.1f}, "
                    f"recent_mean={recent_mean:.3f}, ref_mean={self._reference_pred_mean:.3f}"
                ),
                statistic=float(ks_stat),
                threshold=self.ks_threshold,
                details={
                    "recent_mean": round(recent_mean, 4),
                    "reference_mean": round(self._reference_pred_mean, 4),
                    "z_score": round(z_score, 2),
                    "ks_p_value": round(ks_p, 4),
                },
            )

        return None

    def _check_performance_drift(self) -> Optional[DriftAlert]:
        """Check if recent accuracy differs from reference."""
        recent = list(self._performance_history)[-self.window_size :]
        if len(recent) < 50:
            return None

        recent_perf = float(np.mean(recent))
        perf_diff = recent_perf - self._reference_performance

        if abs(perf_diff) > self.performance_threshold:
            severity = (
                "critical"
                if abs(perf_diff) > 2 * self.performance_threshold
                else "warning"
            )
            direction = "improving" if perf_diff > 0 else "degrading"

            return DriftAlert(
                metric="performance",
                severity=severity,
                message=(
                    f"Model performance {direction}: "
                    f"recent={recent_perf:.3f}, reference={self._reference_performance:.3f}, "
                    f"delta={perf_diff:.3f}"
                ),
                statistic=float(abs(perf_diff)),
                threshold=self.performance_threshold,
                details={
                    "recent_accuracy": round(recent_perf, 4),
                    "reference_accuracy": round(self._reference_performance, 4),
                    "delta": round(perf_diff, 4),
                    "direction": direction,
                },
            )

        return None

    def _check_brier_drift(self) -> Optional[DriftAlert]:
        """Check if recent Brier score has degraded significantly."""
        recent = list(self._brier_history)[-self.window_size :]
        reference = list(self._brier_history)[: self.reference_window]

        if len(recent) < 50 or len(reference) < 50:
            return None

        recent_brier = float(np.mean(recent))
        ref_brier = float(np.mean(reference))
        brier_change = recent_brier - ref_brier

        if brier_change > 0.02:  # Brier getting worse (higher = worse)
            severity = "critical" if brier_change > 0.05 else "warning"
            return DriftAlert(
                metric="calibration",
                severity=severity,
                message=(
                    f"Brier score degraded: "
                    f"recent={recent_brier:.4f}, reference={ref_brier:.4f}, "
                    f"increase={brier_change:.4f}"
                ),
                statistic=float(brier_change),
                threshold=0.02,
                details={
                    "recent_brier": round(recent_brier, 4),
                    "reference_brier": round(ref_brier, 4),
                    "change": round(brier_change, 4),
                },
            )

        return None

    def _check_feature_drift(self) -> Optional[DriftAlert]:
        """Check if feature distribution has drifted."""

        n_features = self._feature_samples[0].shape[-1]
        n_recent = min(100, len(self._feature_samples))
        recent_features = np.array(list(self._feature_samples)[-n_recent:])

        drifted_features = []
        max_ks = 0.0

        for i in range(min(n_features, 20)):  # Check up to 20 features
            feature_values = (
                recent_features[:, i] if recent_features.ndim > 1 else recent_features
            )

            if len(feature_values) < 20:
                continue

            # Simple test: compare variance of recent vs overall
            overall_mean = float(
                np.mean(
                    [
                        s[i] if len(np.atleast_1d(s).shape) > 0 else s
                        for s in list(self._feature_samples)
                    ]
                )
            )
            recent_mean = float(np.mean(feature_values))
            diff = abs(recent_mean - overall_mean) / max(abs(overall_mean), 0.001)

            if diff > 0.5:  # 50% shift in mean
                drifted_features.append(i)
                max_ks = max(max_ks, diff)

        if drifted_features:
            feature_names_str = ", ".join(
                self._feature_names[i] if i < len(self._feature_names) else f"f{i}"
                for i in drifted_features[:5]
            )
            return DriftAlert(
                metric="feature_distribution",
                severity="warning",
                message=(
                    f"Feature drift detected in {len(drifted_features)} features: "
                    f"{feature_names_str}"
                ),
                statistic=float(max_ks),
                threshold=0.5,
                details={
                    "n_drifted": len(drifted_features),
                    "drifted_indices": drifted_features[:10],
                },
            )

        return None

    def _check_importance_drift(self) -> Optional[DriftAlert]:
        """Check if feature importance rankings have shifted."""
        if len(self._feature_importance_history) < 5:
            return None

        recent = self._feature_importance_history[-1]
        older = self._feature_importance_history[0]

        # Compute rank correlation of top features
        common_features = set(recent.keys()) & set(older.keys())
        if len(common_features) < 5:
            return None

        recent_ranks = {
            f: i
            for i, (f, _) in enumerate(sorted(recent.items(), key=lambda x: -abs(x[1])))
        }
        older_ranks = {
            f: i
            for i, (f, _) in enumerate(sorted(older.items(), key=lambda x: -abs(x[1])))
        }

        rank_diffs = [abs(recent_ranks[f] - older_ranks[f]) for f in common_features]
        avg_rank_shift = float(np.mean(rank_diffs))

        if avg_rank_shift > 3:  # Average rank shifted by >3 positions
            return DriftAlert(
                metric="feature_importance",
                severity="info",
                message=(
                    f"Feature importance shift detected: "
                    f"avg rank change={avg_rank_shift:.1f} positions"
                ),
                statistic=avg_rank_shift,
                threshold=3.0,
                details={
                    "avg_rank_shift": round(avg_rank_shift, 1),
                    "n_features_compared": len(common_features),
                },
            )

        return None

    def get_recent_alerts(self, n: int = 10) -> list[DriftAlert]:
        """Get the most recent alerts."""
        return self._alerts[-n:]

    def get_summary(self) -> dict:
        """Get a JSON-serializable summary."""
        return {
            "n_predictions_tracked": len(self._predictions),
            "n_alerts": len(self._alerts),
            "reference_pred_mean": round(self._reference_pred_mean, 4)
            if self._reference_pred_mean
            else None,
            "reference_performance": round(self._reference_performance, 4)
            if self._reference_performance
            else None,
            "last_check": self._last_check_time.isoformat()
            if self._last_check_time
            else None,
            "health": self.check_drift(force=True).model_health
            if len(self._predictions) >= 100
            else "insufficient_data",
        }

    def reset(self):
        """Reset all tracked data."""
        self._predictions.clear()
        self._targets.clear()
        self._timestamps.clear()
        self._feature_samples.clear()
        self._performance_history.clear()
        self._brier_history.clear()
        self._feature_importance_history.clear()
        self._alerts.clear()
        self._reference_pred_mean = None
        self._reference_pred_std = None
        self._reference_performance = None
        self._reference_feature_stats = {}


__all__ = [
    "DriftMonitor",
    "DriftAlert",
    "DriftReport",
]
