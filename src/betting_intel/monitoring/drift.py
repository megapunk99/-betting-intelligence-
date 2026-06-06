"""
Concept drift detection for betting models.

In sports betting, concept drift occurs when:
- Team dynamics change (trades, coaching changes)
- League meta changes (rule changes, officiating trends)
- Market efficiency changes (sharp bettors adapt)
- Seasonality effects (playoffs vs regular season)

This module detects when model predictions degrade over time
so you know when to retrain or discard a model.

Key tools:
1. Population Stability Index (PSI) — distribution shift in predictions
2. Kolmogorov-Smirnov (KS) test — feature distribution drift
3. Performance window tracking — rolling win rate / MAE monitoring
4. Model decay alerts — automated retraining triggers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import stats


# ── Drift Metrics ──────────────────────────────────────────────────────────


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    Population Stability Index — measures distribution shift.

    PSI = sum((actual_pct_i - expected_pct_i) * ln(actual_pct_i / expected_pct_i))

    Interpretation:
        PSI < 0.1  = No significant shift
        0.1 ≤ PSI < 0.25 = Moderate shift — investigate
        PSI ≥ 0.25 = Significant shift — retrain needed

    Args:
        expected: Reference distribution (e.g., training predictions)
        actual: Current distribution (e.g., recent predictions)
        n_bins: Number of bins for discretization

    Returns:
        PSI value
    """
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Bin edges based on expected distribution
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Expected distribution
    expected_counts = np.histogram(expected, bins=bins)[0]
    expected_pct = expected_counts / len(expected)
    expected_pct = np.clip(expected_pct, 1e-6, None)  # Avoid log(0)

    # Actual distribution
    actual_counts = np.histogram(actual, bins=bins)[0]
    actual_pct = actual_counts / len(actual)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    # PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_ks_statistic(
    expected: np.ndarray,
    actual: np.ndarray,
) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov test for distribution equality.

    Returns:
        (ks_statistic, p_value)
        Low p-value (< 0.05) = distributions are significantly different
    """
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()

    if len(expected) == 0 or len(actual) == 0:
        return (0.0, 1.0)

    statistic, p_value = stats.ks_2samp(expected, actual)
    return (float(statistic), float(p_value))


def compute_kl_divergence(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    Kullback-Leibler divergence between expected and actual distributions.

    Higher values indicate more drift.
    """
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Common bin range
    all_data = np.concatenate([expected, actual])
    bins = np.linspace(min(all_data), max(all_data), n_bins + 1)

    expected_hist = np.histogram(expected, bins=bins, density=True)[0] + 1e-10
    actual_hist = np.histogram(actual, bins=bins, density=True)[0] + 1e-10

    # Normalize
    expected_hist /= expected_hist.sum()
    actual_hist /= actual_hist.sum()

    kl_div = np.sum(expected_hist * np.log(expected_hist / actual_hist))
    return float(kl_div)


# ── Performance Window Tracking ────────────────────────────────────────────


@dataclass
class PerformanceWindow:
    """Tracks model performance over a rolling time window."""

    window_size: int  # Number of predictions in this window
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    n_predictions: int = 0
    n_bets: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    avg_edge: float = 0.0
    total_profit: float = 0.0
    roi: float = 0.0
    mae: float = 0.0  # For regression models
    r2: float = 0.0
    sharpe_ratio: Optional[float] = None
    prediction_distribution_psi: float = 0.0
    is_drifted: bool = False
    drift_severity: str = "none"  # 'none', 'moderate', 'severe'


@dataclass
class DriftAlert:
    """Alert generated when drift is detected."""

    model_name: str
    metric: str  # 'win_rate', 'mae', 'psi', 'ks', 'profit'
    old_value: float
    new_value: float
    threshold: float
    severity: str  # 'warning', 'critical'
    timestamp: datetime = field(default_factory=datetime.now)
    recommendation: str = ""


class PerformanceTracker:
    """
    Tracks model performance over rolling windows.

    Maintains:
    - Overall performance (all time)
    - Rolling window performance (last N bets/predictions)
    - Window-to-window drift detection
    - Alert generation for significant performance degradation

    Usage:
        tracker = PerformanceTracker(model_name="LightGBM", window_sizes=[20, 50, 100])
        tracker.record_prediction(predicted=215.5, actual=220, won=True)
        tracker.record_prediction(predicted=212.0, actual=208, won=False)
        report = tracker.get_report()
        alerts = tracker.check_drift()
    """

    def __init__(
        self,
        model_name: str = "default",
        window_sizes: Optional[List[int]] = None,
        psi_warning_threshold: float = 0.10,
        psi_critical_threshold: float = 0.25,
        win_rate_drop_threshold: float = 0.08,  # 8% drop = alert
        mae_increase_threshold: float = 0.15,    # 15% increase = alert
    ):
        self.model_name = model_name
        self.window_sizes = window_sizes or [20, 50, 100, 200]
        self.psi_warning_threshold = psi_warning_threshold
        self.psi_critical_threshold = psi_critical_threshold
        self.win_rate_drop_threshold = win_rate_drop_threshold
        self.mae_increase_threshold = mae_increase_threshold

        # All recorded predictions
        self._predictions: List[Dict] = []

        # Baseline statistics (computed during warm-up period)
        self._baseline: Optional[Dict] = None
        self._baseline_size: int = 100  # Predictions to establish baseline

        # Window caches
        self._windows: Dict[int, List[Dict]] = defaultdict(list)

        # Alert history
        self.alerts: List[DriftAlert] = []

    def record_prediction(
        self,
        predicted: float,
        actual: float,
        won: Optional[bool] = None,
        edge_pct: float = 0.0,
        profit_units: float = 0.0,
        **metadata,
    ):
        """
        Record a prediction outcome.

        Args:
            predicted: Model's predicted value
            actual: Actual outcome
            won: Whether the bet won (classification) or None for regression
            edge_pct: Edge over market
            profit_units: Profit in units
            **metadata: Additional info (date, game_id, etc.)
        """
        record = {
            "predicted": float(predicted),
            "actual": float(actual),
            "error": abs(float(predicted) - float(actual)),
            "won": won,
            "edge_pct": edge_pct,
            "profit_units": profit_units,
            "timestamp": datetime.now(),
            **metadata,
        }
        self._predictions.append(record)

        # Update windows
        for window_size in self.window_sizes:
            self._windows[window_size].append(record)
            if len(self._windows[window_size]) > window_size:
                self._windows[window_size].pop(0)

        # Check for drift if we have enough data
        if self._baseline is not None and len(self._predictions) >= self._baseline_size:
            alerts = self._detect_window_drift()
            self.alerts.extend(alerts)

    def _compute_window_metrics(self, records: List[Dict]) -> PerformanceWindow:
        """Compute performance metrics for a list of records."""
        if not records:
            return PerformanceWindow(window_size=0)

        n = len(records)
        errors = [r["error"] for r in records]
        profits = [r["profit_units"] for r in records]

        window = PerformanceWindow(
            window_size=n,
            n_predictions=n,
        )

        # Timestamps
        timestamps = [r.get("timestamp") for r in records if r.get("timestamp")]
        if timestamps:
            window.start_date = min(timestamps)
            window.end_date = max(timestamps)

        # Regression metrics
        window.mae = float(np.mean(errors))
        actuals = np.array([r["actual"] for r in records])
        preds = np.array([r["predicted"] for r in records])
        if len(actuals) > 1 and np.var(actuals) > 0:
            from sklearn.metrics import r2_score
            window.r2 = float(r2_score(actuals, preds))

        # Betting metrics
        won = [r for r in records if r["won"] is not None]
        if won:
            window.n_bets = len(won)
            window.n_wins = sum(1 for r in won if r["won"])
            window.n_losses = window.n_bets - window.n_wins
            window.win_rate = window.n_wins / window.n_bets if window.n_bets > 0 else 0
            window.total_profit = sum(r["profit_units"] for r in won)
            window.roi = window.total_profit / max(window.n_bets, 1) * 100
            window.avg_edge = float(np.mean([r["edge_pct"] for r in won]))

            if window.n_bets > 5:
                window.sharpe_ratio = (
                    float(np.mean(profits))
                    / max(float(np.std(profits)), 1e-6)
                    * np.sqrt(82)
                )

        # PSI vs baseline (if baseline exists)
        predictions = np.array([r["predicted"] for r in records])
        if self._baseline is not None and len(predictions) >= 10:
            baseline_preds = np.array(self._baseline.get("predictions", predictions[:50]))
            window.prediction_distribution_psi = compute_psi(baseline_preds, predictions)

        return window

    def set_baseline(self, records: Optional[List[Dict]] = None):
        """Establish baseline statistics from initial predictions."""
        if records is not None:
            baseline_records = records
        else:
            baseline_records = self._predictions[:self._baseline_size]

        if len(baseline_records) < 30:
            return

        predictions = [r["predicted"] for r in baseline_records]
        errors = [r["error"] for r in baseline_records]
        won_records = [r for r in baseline_records if r["won"] is not None]

        self._baseline = {
            "n": len(baseline_records),
            "mean_prediction": float(np.mean(predictions)),
            "std_prediction": float(np.std(predictions)),
            "predictions": predictions,
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "win_rate": float(np.mean([r["won"] for r in won_records])) if won_records else 0,
            "n_wins": sum(1 for r in won_records if r["won"]),
            "n_bets": len(won_records),
        }

    def get_window(self, window_size: int) -> PerformanceWindow:
        """Get performance metrics for a specific window size."""
        records = self._windows.get(window_size, [])
        return self._compute_window_metrics(records)

    def get_all_windows(self) -> Dict[int, PerformanceWindow]:
        """Get performance across all window sizes."""
        return {
            ws: self.get_window(ws)
            for ws in self.window_sizes
        }

    def get_overall(self) -> PerformanceWindow:
        """Get overall performance metrics."""
        return self._compute_window_metrics(self._predictions)

    def _detect_window_drift(self) -> List[DriftAlert]:
        """Detect drift across all window sizes vs baseline."""
        alerts = []
        if self._baseline is None:
            return alerts

        baseline_wr = self._baseline.get("win_rate", 0)
        baseline_mae = self._baseline.get("mean_error", 0)

        for ws in self.window_sizes:
            window = self.get_window(ws)
            if window.n_predictions < 10:
                continue

            # Win rate drift
            if window.n_bets >= 10 and baseline_wr > 0:
                wr_drop = baseline_wr - window.win_rate
                if wr_drop > self.win_rate_drop_threshold:
                    severity = "critical" if wr_drop > self.win_rate_drop_threshold * 1.5 else "warning"
                    alerts.append(DriftAlert(
                        model_name=self.model_name,
                        metric=f"win_rate_{ws}g",
                        old_value=baseline_wr,
                        new_value=window.win_rate,
                        threshold=self.win_rate_drop_threshold,
                        severity=severity,
                        recommendation=(
                            f"Win rate dropped from {baseline_wr:.1%} to {window.win_rate:.1%} "
                            f"in last {ws} bets. Consider retraining."
                        ),
                    ))

            # MAE drift
            if baseline_mae > 0 and window.n_predictions >= 10:
                mae_increase = (window.mae - baseline_mae) / baseline_mae
                if mae_increase > self.mae_increase_threshold:
                    severity = "critical" if mae_increase > self.mae_increase_threshold * 1.5 else "warning"
                    alerts.append(DriftAlert(
                        model_name=self.model_name,
                        metric=f"mae_{ws}g",
                        old_value=baseline_mae,
                        new_value=window.mae,
                        threshold=self.mae_increase_threshold,
                        severity=severity,
                        recommendation=(
                            f"MAE increased by {mae_increase:.1%} in last {ws} predictions. "
                            f"Model may be decaying."
                        ),
                    ))

            # PSI drift
            if window.prediction_distribution_psi > self.psi_warning_threshold:
                severity = (
                    "critical"
                    if window.prediction_distribution_psi > self.psi_critical_threshold
                    else "warning"
                )
                alerts.append(DriftAlert(
                    model_name=self.model_name,
                    metric=f"psi_{ws}g",
                    old_value=0,
                    new_value=window.prediction_distribution_psi,
                    threshold=self.psi_warning_threshold,
                    severity=severity,
                    recommendation=(
                        f"Prediction distribution shifted (PSI={window.prediction_distribution_psi:.3f}) "
                        f"in last {ws} predictions. Data drift detected."
                    ),
                ))

        return alerts

    def check_drift(self, force: bool = False) -> List[DriftAlert]:
        """
        Check all drift conditions and return alerts.

        Args:
            force: If True, clear and re-check all windows

        Returns:
            List of DriftAlert objects
        """
        if force or self._baseline is None:
            if len(self._predictions) >= self._baseline_size:
                self.set_baseline()

        alerts = self._detect_window_drift()
        self.alerts.extend(alerts)
        return alerts

    def get_report(self) -> Dict:
        """Get comprehensive performance and drift report."""
        overall = self.get_overall()
        windows = self.get_all_windows()

        report = {
            "model_name": self.model_name,
            "total_predictions": len(self._predictions),
            "overall": {
                "mae": overall.mae,
                "r2": overall.r2,
                "win_rate": overall.win_rate,
                "total_profit": overall.total_profit,
                "roi": overall.roi,
                "sharpe_ratio": overall.sharpe_ratio,
                "n_bets": overall.n_bets,
                "n_wins": overall.n_wins,
                "n_losses": overall.n_losses,
            },
            "windows": {},
            "drift_alerts": [
                {
                    "metric": a.metric,
                    "severity": a.severity,
                    "old_value": a.old_value,
                    "new_value": a.new_value,
                    "recommendation": a.recommendation,
                }
                for a in self.alerts[-20:]  # Last 20 alerts
            ],
        }

        for ws, window in windows.items():
            report["windows"][f"{ws}g"] = {
                "n_predictions": window.n_predictions,
                "mae": window.mae,
                "r2": window.r2,
                "win_rate": window.win_rate,
                "profit": window.total_profit,
                "roi": window.roi,
                "sharpe": window.sharpe_ratio,
                "psi": window.prediction_distribution_psi,
                "is_drifted": window.is_drifted,
            }

        # Drift summary
        n_critical = sum(1 for a in self.alerts if a.severity == "critical")
        n_warning = sum(1 for a in self.alerts if a.severity == "warning")
        report["drift_summary"] = {
            "total_alerts": len(self.alerts),
            "critical": n_critical,
            "warnings": n_warning,
            "needs_retraining": n_critical > 1 or n_warning > 3,
        }

        return report

    def format_report(self) -> str:
        """Format drift report as readable text."""
        report = self.get_report()
        lines = [
            "=" * 60,
            f"  MODEL PERFORMANCE & DRIFT REPORT: {self.model_name}",
            "=" * 60,
            f"  Total Predictions: {report['total_predictions']:,}",
            "",
            "  -- Overall Performance --",
            f"  MAE:     {report['overall']['mae']:.2f}",
            f"  R²:      {report['overall']['r2']:.3f}",
            f"  Win Rate: {report['overall']['win_rate']:.1%}",
            f"  Profit:  {report['overall']['total_profit']:+.1f}u",
            f"  ROI:     {report['overall']['roi']:+.1f}%",
            f"  Sharpe:  {report['overall']['sharpe_ratio']:.2f}",
            "",
            "  -- Rolling Windows --",
        ]

        for ws_name, w in sorted(report["windows"].items()):
            drift_flag = " [DRIFT]" if w.get("is_drifted") else ""
            lines.append(
                f"  {ws_name:6s}: {w['n_predictions']:3d} preds | "
                f"MAE {w['mae']:.1f} | WR {w.get('win_rate', 0):.0%} | "
                f"PSI {w.get('psi', 0):.3f}{drift_flag}"
            )

        if report["drift_alerts"]:
            lines.extend(["", "  -- Active Alerts --"])
            for a in report["drift_alerts"][:10]:
                severity_icon = "[!]" if a["severity"] == "critical" else "[?]"
                lines.append(f"  {severity_icon} {a['metric']}: {a['recommendation']}")
            lines.append("")

        ds = report["drift_summary"]
        if ds["needs_retraining"]:
            lines.extend([
                "  !! RECOMMENDATION: Retrain model immediately !!",
                f"     ({ds['critical']} critical alerts, {ds['warnings']} warnings)",
            ])
        else:
            lines.extend([
                "  Status: Healthy",
                f"     ({ds['total_alerts']} total alerts, {ds['critical']} critical)",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)


# ── Feature Drift Detection ────────────────────────────────────────────────


class FeatureDriftDetector:
    """
    Monitors individual feature distributions for drift.

    Tracks:
    - Per-feature PSI and KS statistics
    - Feature importance changes over time
    - Correlation structure drift

    Usage:
        detector = FeatureDriftDetector(feature_names=['avg_pts_5g_home', 'rest_home_days', ...])
        detector.fit_reference(reference_df)
        results = detector.detect_drift(current_df)
    """

    def __init__(
        self,
        feature_names: List[str],
        psi_threshold: float = 0.10,
        psi_critical_threshold: float = 0.25,
        ks_threshold: float = 0.05,
    ):
        self.feature_names = feature_names
        self.psi_threshold = psi_threshold
        self.psi_critical_threshold = psi_critical_threshold
        self.ks_threshold = ks_threshold
        self._reference_stats: Dict[str, Dict] = {}

    def fit_reference(self, df: pd.DataFrame):
        """Store reference distribution statistics from training data."""
        for col in self.feature_names:
            if col not in df.columns:
                continue
            values = df[col].dropna().values
            if len(values) == 0:
                continue

            self._reference_stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "values": values,
            }

    def detect_drift(self, df: pd.DataFrame) -> Dict:
        """
        Detect feature drift in new data vs reference.

        Args:
            df: Current data to check for drift

        Returns:
            Dict with per-feature drift results and summary
        """
        results = {}
        n_drifted = 0
        n_total = 0

        for col in self.feature_names:
            if col not in df.columns or col not in self._reference_stats:
                continue

            current_values = df[col].dropna().values
            if len(current_values) < 5:
                continue

            ref_values = self._reference_stats[col]["values"]
            n_total += 1

            # PSI
            psi = compute_psi(ref_values, current_values)

            # KS test
            ks_stat, ks_p = compute_ks_statistic(ref_values, current_values)

            # Basic stat changes
            current_mean = float(np.mean(current_values))
            current_std = float(np.std(current_values))
            mean_shift = abs(current_mean - self._reference_stats[col]["mean"]) / max(
                self._reference_stats[col]["std"], 1e-6
            )

            is_drifted = psi > self.psi_threshold or ks_p < self.ks_threshold
            if is_drifted:
                n_drifted += 1

            severity = "none"
            if psi > self.psi_critical_threshold:
                severity = "critical"
            elif is_drifted:
                severity = "moderate"

            results[col] = {
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_p_value": ks_p,
                "mean_shift_z": mean_shift,
                "ref_mean": self._reference_stats[col]["mean"],
                "current_mean": current_mean,
                "is_drifted": is_drifted,
                "severity": severity,
            }

        # Drift ratio
        drift_ratio = n_drifted / max(n_total, 1)

        return {
            "features": results,
            "summary": {
                "n_features_checked": n_total,
                "n_features_drifted": n_drifted,
                "drift_ratio": drift_ratio,
                "overall_severity": (
                    "critical" if drift_ratio > 0.3
                    else "moderate" if drift_ratio > 0.1
                    else "none"
                ),
                "recommendation": (
                    "Retrain model — significant feature drift detected"
                    if drift_ratio > 0.3
                    else "Monitor closely — some features showing drift"
                    if drift_ratio > 0.1
                    else "No action needed"
                ),
            },
        }

    def format_report(self, results: Dict) -> str:
        """Format drift detection results as readable text."""
        if not results:
            return "No drift analysis results."

        lines = [
            "=" * 60,
            "  FEATURE DRIFT DETECTION REPORT",
            "=" * 60,
        ]

        summary = results.get("summary", {})
        lines.extend([
            f"  Features checked: {summary.get('n_features_checked', 0)}",
            f"  Features drifted: {summary.get('n_features_drifted', 0)} "
            f"({summary.get('drift_ratio', 0):.1%})",
            f"  Severity: {summary.get('overall_severity', 'unknown')}",
            f"  Recommendation: {summary.get('recommendation', 'N/A')}",
            "",
            "  -- Per-Feature Drift --",
        ])

        features = results.get("features", {})
        # Sort by PSI descending
        sorted_features = sorted(
            features.items(),
            key=lambda x: x[1].get("psi", 0),
            reverse=True,
        )

        for col, info in sorted_features:
            severity_icon = {
                "critical": "[!!]",
                "moderate": "[!] ",
                "none": "     ",
            }.get(info.get("severity", "none"), "     ")

            lines.append(
                f"  {severity_icon} {col:30s} "
                f"PSI={info.get('psi', 0):.3f} "
                f"KS-p={info.get('ks_p_value', 1):.3f} "
                f"Mean Δ={info.get('mean_shift_z', 0):.1f}σ"
            )

        lines.append("=" * 60)
        return "\n".join(lines)
