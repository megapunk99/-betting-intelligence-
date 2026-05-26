"""
Prometheus metrics for production monitoring.
Tracks prediction counts, model performance, API latency, and error rates.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from betting_intel.config import settings
from betting_intel.services import logger

# ── Prediction Metrics ─────────────────────────────────────────────────────
PREDICTIONS_TOTAL = Counter(
    "betting_predictions_total",
    "Total number of predictions made",
    ["model", "strategy"],
)

PREDICTIONS_ERRORS = Counter(
    "betting_predictions_errors",
    "Total prediction errors",
    ["model", "error_type"],
)

PREDICTION_LATENCY = Histogram(
    "betting_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ── Model Metrics ──────────────────────────────────────────────────────────
MODEL_PERFORMANCE = Gauge(
    "betting_model_performance",
    "Model performance metrics",
    ["model", "metric"],
)

MODEL_LAST_TRAINED = Gauge(
    "betting_model_last_trained_timestamp",
    "Timestamp of last model training",
    ["model"],
)

# ── Backtest Metrics ───────────────────────────────────────────────────────
BACKTEST_WIN_RATE = Gauge(
    "betting_backtest_win_rate",
    "Backtest win rate by strategy",
    ["strategy"],
)

BACKTEST_SHARPE = Gauge(
    "betting_backtest_sharpe_ratio",
    "Backtest Sharpe ratio by strategy",
    ["strategy"],
)

BACKTEST_PROFIT = Gauge(
    "betting_backtest_total_profit",
    "Backtest total profit in units",
    ["strategy"],
)

# ── Bankroll Metrics ───────────────────────────────────────────────────────
BANKROLL_CURRENT = Gauge(
    "betting_bankroll_current",
    "Current bankroll in dollars",
)

BANKROLL_PEAK = Gauge(
    "betting_bankroll_peak",
    "Peak bankroll in dollars",
)

BANKROLL_DRAWDOWN = Gauge(
    "betting_bankroll_drawdown_pct",
    "Current drawdown percentage",
)

# ── System Metrics ─────────────────────────────────────────────────────────
PIPELINE_RUN_DURATION = Histogram(
    "betting_pipeline_duration_seconds",
    "Pipeline run duration in seconds",
    buckets=(10, 30, 60, 120, 300, 600, 1800),
)

PIPELINE_RUNS_TOTAL = Counter(
    "betting_pipeline_runs_total",
    "Total pipeline runs",
    ["status"],
)

DATABASE_QUERY_LATENCY = Histogram(
    "betting_db_query_latency_seconds",
    "Database query latency",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

API_REQUEST_LATENCY = Histogram(
    "betting_api_request_latency_seconds",
    "API request latency by endpoint",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

API_REQUESTS_TOTAL = Counter(
    "betting_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)


def track_prediction_latency(model_name: str) -> Callable:
    """Decorator to track prediction latency."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                result = func(*args, **kwargs)
                PREDICTIONS_TOTAL.labels(model=model_name, strategy="unknown").inc()
                return result
            except Exception as e:
                PREDICTIONS_ERRORS.labels(model=model_name, error_type=type(e).__name__).inc()
                raise
            finally:
                PREDICTION_LATENCY.labels(model=model_name).observe(time.time() - start)

        return wrapper

    return decorator


def update_model_metrics(model_name: str, metrics: dict[str, float]):
    """Update Prometheus metrics for a trained model."""
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            MODEL_PERFORMANCE.labels(model=model_name, metric=metric_name).set(value)
    MODEL_LAST_TRAINED.labels(model=model_name).set(time.time())


def metrics_endpoint() -> tuple[bytes, int, dict]:
    """Generate prometheus metrics endpoint response."""
    data = generate_latest()
    return data, 200, {"Content-Type": CONTENT_TYPE_LATEST}
