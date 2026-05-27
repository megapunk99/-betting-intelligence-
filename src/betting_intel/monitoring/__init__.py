"""Monitoring module: performance tracking, drift detection, Prometheus metrics."""

from betting_intel.monitoring.metrics import (
    metrics_endpoint,
    track_prediction_latency,
    update_model_metrics,
)
from betting_intel.monitoring.drift import (
    compute_psi,
    compute_ks_statistic,
    compute_kl_divergence,
    PerformanceTracker,
    FeatureDriftDetector,
    PerformanceWindow,
    DriftAlert,
)

__all__ = [
    "metrics_endpoint",
    "track_prediction_latency",
    "update_model_metrics",
    "compute_psi",
    "compute_ks_statistic",
    "compute_kl_divergence",
    "PerformanceTracker",
    "FeatureDriftDetector",
    "PerformanceWindow",
    "DriftAlert",
]
