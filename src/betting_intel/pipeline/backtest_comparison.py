"""
Backtest Comparison — Train RobustPredictionSystem + MarketInefficiencySystem on
the same historical NBA data and compare their performance head-to-head.

This is the definitive answer to "does the market inefficiency approach actually
outperform a pure classifier?" — it trains both models on identical train/test
splits and measures accuracy, edge capture, ROI, Sharpe, drawdown, and more.

Architecture
────────────
  1. Load historical NBA data via NBADataLoader + FeatureEngineer
  2. Walk through chronological test periods (e.g. monthly, seasonal)
  3. For each period:
       a. Train RobustPredictionSystem  (classifier, home_win only)
       b. Train MarketInefficiencySystem (classifier + market error)
       c. Evaluate both on the same held-out test period
  4. Aggregate metrics across all periods
  5. Simulate flat-betting and Kelly-betting for both models
  6. Generate comparison report with deltas and statistical significance

Usage:
    from betting_intel.pipeline.backtest_comparison import BacktestComparison

    comparison = BacktestComparison()
    report = comparison.run()
    comparison.print_report(report)

    # Or via CLI:
    # python -m betting_intel.cli.main backtest compare
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ModelComparisonMetrics:
    """Performance metrics for a single model on a single test period."""

    # ── Identity ───────────────────────────────────────────────────
    model_name: str = ""                   # "classifier" or "market_inefficiency"
    period_label: str = ""                 # e.g. "2023-11", "2024-01..2024-04"

    # ── Sample counts ──────────────────────────────────────────────
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0

    # ── Classification metrics ──────────────────────────────────────
    accuracy: float = 0.0                  # Correct win/loss predictions
    brier_score: float = 0.5               # Lower is better (0=perfect, 0.25=no skill)
    log_loss: float = 0.0                  # Lower is better
    auc_roc: float = 0.5                   # Area under ROC curve
    precision: float = 0.0                 # TP / (TP + FP)
    recall: float = 0.0                    # TP / (TP + FN)
    f1_score: float = 0.0                  # Harmonic mean of precision & recall

    # ── Calibration metrics ─────────────────────────────────────────
    calibration_error: float = 0.0         # ECE (expected calibration error)
    calibration_slope: float = 1.0         # 1.0 = perfectly calibrated

    # ── Edge metrics ────────────────────────────────────────────────
    avg_predicted_edge: float = 0.0        # Mean absolute predicted edge
    actual_edge_correlation: float = 0.0   # Corr(predicted_edge, actual_outcome)
    edge_capture_rate: float = 0.0         # % of games where edge sign matches outcome
    top_edge_accuracy: float = 0.0         # Accuracy on top-25% edge games

    # ── Betting simulation: Flat $10 ────────────────────────────────
    flat_n_bets: int = 0
    flat_wins: int = 0
    flat_losses: int = 0
    flat_win_rate: float = 0.0
    flat_total_stake: float = 0.0
    flat_total_profit: float = 0.0
    flat_roi: float = 0.0
    flat_sharpe: float = 0.0
    flat_max_drawdown: float = 0.0

    # ── Betting simulation: Kelly (¼ Kelly) ─────────────────────────
    kelly_n_bets: int = 0
    kelly_wins: int = 0
    kelly_losses: int = 0
    kelly_win_rate: float = 0.0
    kelly_total_stake: float = 0.0
    kelly_total_profit: float = 0.0
    kelly_roi: float = 0.0
    kelly_sharpe: float = 0.0
    kelly_max_drawdown: float = 0.0
    kelly_final_bankroll: float = 0.0

    # ── Confusion matrix ────────────────────────────────────────────
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    # ── Raw predictions (for agreement analysis) ────────────────────
    _y_pred_raw: Optional[np.ndarray] = None  # Binary predictions (0/1)
    _y_true_raw: Optional[np.ndarray] = None  # Ground truth (0/1)


@dataclass
class ModelComparisonPeriod:
    """Comparison of both models on a single test period."""

    period_label: str = ""

    # Classifier (RobustPredictionSystem = pre-change)
    classifier: ModelComparisonMetrics = field(default_factory=ModelComparisonMetrics)

    # Market inefficiency (MarketInefficiencySystem = post-change)
    market_inefficiency: ModelComparisonMetrics = field(default_factory=ModelComparisonMetrics)

    # ── Δ (delta = market_inefficiency - classifier) ────────────────
    delta_accuracy: float = 0.0
    delta_brier: float = 0.0              # Positive = market_inefficiency better
    delta_flat_roi: float = 0.0
    delta_kelly_roi: float = 0.0
    delta_flat_sharpe: float = 0.0
    delta_edge_capture: float = 0.0
    delta_win_rate: float = 0.0

    # Meta
    n_games: int = 0
    home_win_rate: float = 0.0
    avg_market_prob: float = 0.0


@dataclass
class BacktestComparisonReport:
    """Aggregated comparison report across all test periods."""

    generated_at: str = ""
    total_n_train: int = 0
    total_n_test: int = 0
    n_periods: int = 0
    n_seasons: int = 0

    # ── Aggregate model metrics ─────────────────────────────────────
    classifier: ModelComparisonMetrics = field(default_factory=ModelComparisonMetrics)
    market_inefficiency: ModelComparisonMetrics = field(default_factory=ModelComparisonMetrics)

    # ── Aggregate deltas ────────────────────────────────────────────
    delta_accuracy: float = 0.0
    delta_brier: float = 0.0              # Positive = market_inefficiency better
    delta_flat_roi: float = 0.0
    delta_kelly_roi: float = 0.0
    delta_flat_sharpe: float = 0.0
    delta_kelly_sharpe: float = 0.0
    delta_edge_capture: float = 0.0
    delta_win_rate: float = 0.0
    delta_top_edge_accuracy: float = 0.0

    # ── Per-period breakdown ────────────────────────────────────────
    periods: list[ModelComparisonPeriod] = field(default_factory=list)

    # ── Statistical significance ────────────────────────────────────
    is_significant: bool = False
    p_value: float = 1.0
    confidence_level: str = "none"         # "none", "low", "medium", "high", "very_high"

    # ── Model agreement ─────────────────────────────────────────────
    agreement_rate: float = 0.0            # % of games where both models predict same winner
    disagreement_rate: float = 0.0         # % of games where they disagree
    classifier_won_rate: float = 0.0       # When they disagree, how often was classifier right
    market_won_rate: float = 0.0           # When they disagree, how often was market right

    # ── Metadata ────────────────────────────────────────────────────
    min_edge_threshold: float = 0.02
    kelly_fraction: float = 0.25
    initial_bankroll: float = 10_000.0
    total_elapsed_seconds: float = 0.0
    feature_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST COMPARISON
# ═══════════════════════════════════════════════════════════════════════════


class BacktestComparison:
    """
    Head-to-head comparison of RobustPredictionSystem vs MarketInefficiencySystem.

    Walks through historical NBA data chronologically, trains both models on
    identical train/test splits, and compares every aspect of their performance.

    Usage:
        comparison = BacktestComparison()
        report = comparison.run()
        comparison.print_report(report)
    """

    def __init__(
        self,
        n_folds: int = 5,
        min_train_samples: int = 200,
        min_test_samples: int = 50,
        min_edge_threshold: float = 0.02,
        kelly_fraction: float = 0.25,
        initial_bankroll: float = 10_000.0,
        random_state: int = 42,
    ):
        self.n_folds = n_folds
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        self.min_edge_threshold = min_edge_threshold
        self.kelly_fraction = kelly_fraction
        self.initial_bankroll = initial_bankroll
        self.random_state = random_state

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        n_test_periods: int = 5,      # Number of chronological test splits
        test_size_games: int = 100,    # Approx games per test period
        verbose: bool = True,
    ) -> BacktestComparisonReport:
        """
        Run the full comparison: train both models on each test period.

        Args:
            n_test_periods: Number of chronological test splits to create
            test_size_games: Approximate number of games per test period
            verbose: Print progress during execution

        Returns:
            BacktestComparisonReport with all metrics and deltas
        """
        start_time = time.time()

        if verbose:
            self._print_header()

        # ── Step 1: Load data and build features ──────────────────────
        if verbose:
            print("  📊  Loading historical NBA data...")

        df, features_df, feature_cols = self._load_and_build_features()

        if features_df is None or len(features_df) < self.min_train_samples + self.min_test_samples:
            raise ValueError(
                f"Not enough data: {len(features_df) if features_df is not None else 0} samples, "
                f"need ≥{self.min_train_samples + self.min_test_samples}"
            )

        n_total = len(features_df)
        feature_count = len(feature_cols)

        if verbose:
            print(f"  ✅  {n_total} games, {feature_count} features")

        # ── Step 2: Create chronological test splits ──────────────────
        periods = self._create_test_periods(
            features_df, n_test_periods, test_size_games
        )

        if verbose:
            print(f"  📅  {len(periods)} test periods created")

        # ── Step 3: For each period, train both models and evaluate ───
        all_metrics: list[ModelComparisonPeriod] = []

        for period_idx, (train_df, test_df) in enumerate(periods):
            label = self._period_label(train_df, test_df, period_idx, len(periods))

            if verbose:
                print(
                    f"\n  🔄  Period {period_idx + 1}/{len(periods)}: "
                    f"train={len(train_df)} test={len(test_df)}"
                )

            period_comparison = self._evaluate_period(
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
                period_label=label,
                verbose=verbose,
            )

            all_metrics.append(period_comparison)

            if verbose:
                self._print_period_summary(period_comparison)

        # ── Step 4: Aggregate across periods ──────────────────────────
        report = self._aggregate_results(all_metrics, feature_count, start_time)

        if verbose:
            self.print_report(report)

        return report

    def print_report(self, report: BacktestComparisonReport) -> None:
        """Print a formatted comparison report to console."""
        print(f"\n{'█' * 70}")
        print(f"  BACKTEST COMPARISON REPORT")
        print(f"  Pre-change:  RobustPredictionSystem (classifier, home_win only)")
        print(f"  Post-change: MarketInefficiencySystem (classifier + market error)")
        print(f"  Generated:   {report.generated_at}")
        print(f"{'█' * 70}")

        # ── Overview ────────────────────────────────────────────────
        print(f"\n{'─' * 35}  OVERVIEW  {'─' * 35}")
        print(f"  Total training samples:   {report.total_n_train:,}")
        print(f"  Total test samples:       {report.total_n_test:,}")
        print(f"  Test periods:             {report.n_periods}")
        print(f"  Features:                 {report.feature_count}")
        print(f"  Total time:               {report.total_elapsed_seconds:.1f}s")
        print(f"  Edge threshold:           {report.min_edge_threshold:.1%}")
        print(f"  Kelly fraction:           {report.kelly_fraction:.0%}")
        print(f"  Initial bankroll:         ${report.initial_bankroll:,.0f}")

        # ── Classification Metrics ──────────────────────────────────
        c = report.classifier
        m = report.market_inefficiency

        print(f"\n{'─' * 35}  CLASSIFICATION METRICS  {'─' * 35}")
        self._print_metric_row("Accuracy", c.accuracy, m.accuracy, report.delta_accuracy, "↑", pct=True)
        self._print_metric_row("Brier Score", c.brier_score, m.brier_score, report.delta_brier, "↓", pct=False)
        self._print_metric_row("Log Loss", c.log_loss, m.log_loss, c.log_loss - m.log_loss, "↓", pct=False)
        self._print_metric_row("AUC-ROC", c.auc_roc, m.auc_roc, m.auc_roc - c.auc_roc, "↑", pct=False)
        self._print_metric_row("Precision", c.precision, m.precision, m.precision - c.precision, "↑", pct=True)
        self._print_metric_row("Recall", c.recall, m.recall, m.recall - c.recall, "↑", pct=True)
        self._print_metric_row("F1 Score", c.f1_score, m.f1_score, m.f1_score - c.f1_score, "↑", pct=True)
        self._print_metric_row("Top-25% Edge Acc", c.top_edge_accuracy, m.top_edge_accuracy,
                               report.delta_top_edge_accuracy, "↑", pct=True)

        # ── Edge Metrics ────────────────────────────────────────────
        print(f"\n{'─' * 35}  EDGE METRICS  {'─' * 35}")
        self._print_metric_row("Edge Capture Rate", c.edge_capture_rate, m.edge_capture_rate,
                               report.delta_edge_capture, "↑", pct=True)
        self._print_metric_row("Avg |Predicted Edge|", c.avg_predicted_edge, m.avg_predicted_edge,
                               m.avg_predicted_edge - c.avg_predicted_edge, "~", pct=True)
        self._print_metric_row("Edge-Outcome Corr", c.actual_edge_correlation, m.actual_edge_correlation,
                               m.actual_edge_correlation - c.actual_edge_correlation, "↑", pct=False)

        # ── Flat Betting ────────────────────────────────────────────
        print(f"\n{'─' * 35}  FLAT BETTING ($10/bet)  {'─' * 35}")
        self._print_flat_row(c, "Classifier")
        self._print_flat_row(m, "Market Inefficiency")
        delta_roi = report.delta_flat_roi
        delta_str = f"{delta_roi:+.2%}" if abs(delta_roi) > 0.0001 else "0.00%"
        arrow = "✅" if delta_roi > 0 else ("❌" if delta_roi < 0 else "➖")
        print(f"  Δ ROI:              {arrow} {delta_str}")
        print(f"  Δ Sharpe:           {report.delta_flat_sharpe:+.3f}")

        # ── Kelly Betting ───────────────────────────────────────────
        print(f"\n{'─' * 35}  KELLY BETTING (¼ Kelly)  {'─' * 35}")
        self._print_kelly_row(c, "Classifier")
        self._print_kelly_row(m, "Market Inefficiency")
        delta_kelly_roi = report.delta_kelly_roi
        delta_kelly_str = f"{delta_kelly_roi:+.2%}" if abs(delta_kelly_roi) > 0.0001 else "0.00%"
        arrow_k = "✅" if delta_kelly_roi > 0 else ("❌" if delta_kelly_roi < 0 else "➖")
        print(f"  Δ ROI:              {arrow_k} {delta_kelly_str}")
        print(f"  Δ Sharpe:           {report.delta_kelly_sharpe:+.3f}")
        print(f"  Final Bankroll:")
        print(f"    Classifier:             ${c.kelly_final_bankroll:,.0f}")
        print(f"    Market Inefficiency:    ${m.kelly_final_bankroll:,.0f}")

        # ── Model Agreement ─────────────────────────────────────────
        print(f"\n{'─' * 35}  MODEL AGREEMENT  {'─' * 35}")
        print(f"  Agreement rate:     {report.agreement_rate:.1%}")
        print(f"  Disagreement rate:  {report.disagreement_rate:.1%}")
        print(f"  When they disagree:")
        print(f"    Classifier correct:  {report.classifier_won_rate:.1%}")
        print(f"    Market Inefficiency: {report.market_won_rate:.1%}")
        net = report.market_won_rate - report.classifier_won_rate
        print(f"    Net advantage:       {net:+.1%}")

        # ── Statistical Significance ────────────────────────────────
        print(f"\n{'─' * 35}  STATISTICAL SIGNIFICANCE  {'─' * 35}")
        print(f"  P-value:            {report.p_value:.4f}")
        print(f"  Significant:        {'✅ Yes (p < 0.05)' if report.is_significant else '❌ No (p >= 0.05)'}")
        print(f"  Confidence:         {report.confidence_level}")

        # ── Summary verdict ─────────────────────────────────────────
        print(f"\n{'─' * 35}  VERDICT  {'─' * 35}")
        verdict = self._generate_verdict(report)
        print(f"  {verdict}")

        print(f"\n{'█' * 70}\n")

    # ── Internal: Data Loading & Feature Engineering ─────────────────────

    def _load_and_build_features(
        self,
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], list[str]]:
        """
        Load historical NBA data and build features.

        Returns:
            Tuple of (raw_df, features_df, feature_cols)
        """
        try:
            from betting_intel.data.loader import NBADataLoader
            from betting_intel.data.features import FeatureEngineer

            loader = NBADataLoader()
            fe = FeatureEngineer()

            raw_df = loader.load_game_logs()
            if raw_df is None or raw_df.empty:
                logger.warning("No data loaded from NBADataLoader")
                return None, None, []

            # Add IS_HOME flag and rest days
            raw_df["IS_HOME"] = raw_df["MATCHUP"].fillna("").str.contains("vs.").astype(int)
            raw_df = loader.compute_rest_days(raw_df)

            # Build game-level dataset
            games_df = loader.build_game_dataset(raw_df)
            if games_df is None or games_df.empty:
                logger.warning("No games built from raw data")
                return None, None, []

            # Engineer features
            features_df = fe.build_all_features(games_df, raw_df)
            if features_df is None or features_df.empty:
                logger.warning("Feature engineering failed")
                return None, None, []

            # Add home_win target
            features_df["home_win"] = (features_df["point_diff"] > 0).astype(int)

            # Select feature columns
            feature_cols = fe.select_features(features_df)

            return raw_df, features_df, feature_cols

        except ImportError as e:
            logger.error(f"Import error in data loading: {e}")
            return None, None, []
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return None, None, []

    def _create_test_periods(
        self,
        features_df: pd.DataFrame,
        n_periods: int = 5,
        test_size_games: int = 100,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create chronological train/test splits.

        Each period uses ALL data before it as training.
        """
        n = len(features_df)
        min_required = self.min_train_samples + self.min_test_samples

        if n < min_required:
            return [(features_df, features_df)]

        # Use date-based splits if possible
        if "GAME_DATE" in features_df.columns:
            dates = sorted(features_df["GAME_DATE"].unique())
            if len(dates) >= n_periods:
                return self._create_date_splits(features_df, dates, n_periods)

        # Fallback: index-based splits
        periods = []
        step = max(test_size_games, self.min_test_samples)
        for i in range(n_periods):
            split_point = n - (n_periods - i) * step
            if split_point < self.min_train_samples:
                break
            train_df = features_df.iloc[:split_point].copy()
            test_df = features_df.iloc[split_point:split_point + step].copy()
            if len(test_df) < self.min_test_samples:
                test_df = features_df.iloc[split_point:].copy()
            if len(test_df) >= self.min_test_samples:
                periods.append((train_df, test_df))

        return periods

    def _create_date_splits(
        self,
        features_df: pd.DataFrame,
        dates: list,
        n_periods: int = 5,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Create splits based on dates (e.g. monthly or seasonal)."""
        n_dates = len(dates)
        games_per_period = max(1, n_dates // n_periods)

        periods = []
        for i in range(n_periods):
            start_idx = (n_periods - i - 1) * games_per_period
            end_idx = min(start_idx + games_per_period, n_dates)

            train_dates = dates[:start_idx]
            test_dates = dates[start_idx:end_idx]

            if not test_dates:
                continue

            train_df = features_df[features_df["GAME_DATE"].isin(train_dates)].copy()
            test_df = features_df[features_df["GAME_DATE"].isin(test_dates)].copy()

            if len(train_df) >= self.min_train_samples and len(test_df) >= self.min_test_samples:
                periods.append((train_df, test_df))

        return periods

    # ── Internal: Per-Period Evaluation ──────────────────────────────────

    def _evaluate_period(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        period_label: str = "",
        verbose: bool = True,
    ) -> ModelComparisonPeriod:
        """
        Train both models on train_df, evaluate on test_df.

        Returns a ModelComparisonPeriod with all metrics.
        """
        # ── Prepare data ──────────────────────────────────────────
        X_train = train_df[feature_cols].fillna(0).values.astype(np.float64)
        y_train = train_df["home_win"].values.astype(np.int32)
        X_test = test_df[feature_cols].fillna(0).values.astype(np.float64)
        y_test = test_df["home_win"].values.astype(np.int32)

        # Market probabilities: ELO-based or spread-based proxy
        market_train = self._get_market_probs(train_df)
        market_test = self._get_market_probs(test_df)

        # ── Train RobustPredictionSystem (classifier only) ─────────
        if verbose:
            print(f"     🏋  Training RobustPredictionSystem...", end=" ")

        t0 = time.time()
        try:
            from betting_intel.models.robust_ensemble import RobustPredictionSystem

            classifier = RobustPredictionSystem(
                calibrate=True,
                n_folds=min(self.n_folds, 3),
                min_train_samples=100,
                random_state=self.random_state,
            )
            classifier.fit(
                X_train, y_train,
                feature_names=feature_cols,
                verbose=False,
            )

            # Predict on test set
            classifier_probs = classifier.predict_proba(X_test)[:, 1]
            classifier_metrics = self._compute_metrics(
                model_name="classifier",
                y_true=y_test,
                y_prob=classifier_probs,
                market_probs=market_test,
                period_label=period_label,
                n_train=len(X_train),
                n_features=len(feature_cols),
            )
            print(f"OK ({time.time() - t0:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            classifier_metrics = ModelComparisonMetrics(
                model_name="classifier", period_label=period_label,
            )

        # ── Train MarketInefficiencySystem ─────────────────────────
        if verbose:
            print(f"     🧠  Training MarketInefficiencySystem...", end=" ")

        t0 = time.time()
        try:
            from betting_intel.models.robust_ensemble import MarketInefficiencySystem

            market_system = MarketInefficiencySystem(
                calibrate=True,
                n_folds=min(self.n_folds, 3),
                min_train_samples=100,
                random_state=self.random_state,
            )
            market_system.fit(
                X_train, y_train,
                market_probs=market_train,
                feature_names=feature_cols,
                verbose=False,
            )

            # Predict on test set using market_probs
            market_probs_result = market_system.predict_proba(
                X_test, market_probs=market_test
            )[:, 1]

            market_metrics = self._compute_metrics(
                model_name="market_inefficiency",
                y_true=y_test,
                y_prob=market_probs_result,
                market_probs=market_test,
                period_label=period_label,
                n_train=len(X_train),
                n_features=len(feature_cols),
            )
            print(f"OK ({time.time() - t0:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            market_metrics = ModelComparisonMetrics(
                model_name="market_inefficiency", period_label=period_label,
            )

        # ── Compute deltas ─────────────────────────────────────────
        period = ModelComparisonPeriod(
            period_label=period_label,
            classifier=classifier_metrics,
            market_inefficiency=market_metrics,
            delta_accuracy=market_metrics.accuracy - classifier_metrics.accuracy,
            delta_brier=classifier_metrics.brier_score - market_metrics.brier_score,
            delta_flat_roi=market_metrics.flat_roi - classifier_metrics.flat_roi,
            delta_kelly_roi=market_metrics.kelly_roi - classifier_metrics.kelly_roi,
            delta_flat_sharpe=market_metrics.flat_sharpe - classifier_metrics.flat_sharpe,
            delta_edge_capture=market_metrics.edge_capture_rate - classifier_metrics.edge_capture_rate,
            delta_win_rate=market_metrics.flat_win_rate - classifier_metrics.flat_win_rate,
            n_games=len(y_test),
            home_win_rate=float(np.mean(y_test)),
            avg_market_prob=float(np.mean(market_test)) if len(market_test) > 0 else 0.5,
        )

        return period

    def _compute_metrics(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        market_probs: np.ndarray,
        period_label: str,
        n_train: int,
        n_features: int,
    ) -> ModelComparisonMetrics:
        """Compute all classification, edge, and betting metrics."""
        from sklearn.metrics import (
            accuracy_score, brier_score_loss, log_loss,
            roc_auc_score, precision_score, recall_score, f1_score,
        )

        y_prob = np.clip(y_prob, 0.001, 0.999)
        y_pred = (y_prob >= 0.5).astype(int)
        n_test = len(y_true)

        # ── Basic classification metrics ───────────────────────────
        metrics = ModelComparisonMetrics(
            model_name=model_name,
            period_label=period_label,
            n_train=n_train,
            n_test=n_test,
            n_features=n_features,
        )

        if n_test == 0:
            return metrics

        metrics.accuracy = accuracy_score(y_true, y_pred)
        metrics.brier_score = brier_score_loss(y_true, y_prob)

        try:
            metrics.log_loss = log_loss(y_true, y_prob)
        except Exception:
            metrics.log_loss = 0.0

        try:
            if len(np.unique(y_true)) >= 2:
                metrics.auc_roc = roc_auc_score(y_true, y_prob)
            else:
                metrics.auc_roc = 0.5
        except Exception:
            metrics.auc_roc = 0.5

        try:
            metrics.precision = precision_score(y_true, y_pred, zero_division=0)
            metrics.recall = recall_score(y_true, y_pred, zero_division=0)
            metrics.f1_score = f1_score(y_true, y_pred, zero_division=0)
        except Exception:
            metrics.precision = 0.0
            metrics.recall = 0.0
            metrics.f1_score = 0.0

        # Confusion matrix
        metrics.tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        metrics.tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        metrics.fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        metrics.fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        # ── Edge metrics ───────────────────────────────────────────
        # Edge = model_prob - market_prob
        if len(market_probs) == n_test:
            edges = y_prob - market_probs

            metrics.avg_predicted_edge = float(np.mean(np.abs(edges)))

            # Edge capture: does predicted edge sign match outcome?
            # Positive edge = model says home more likely than market
            # If home wins (y_true=1), positive edge is correct
            predicted_sign = np.sign(edges)
            actual = y_true.astype(float) - 0.5  # +0.5 if home won, -0.5 if lost
            actual_sign = np.sign(actual)
            correct_edge = (predicted_sign == actual_sign) | (edges == 0)
            metrics.edge_capture_rate = float(np.mean(correct_edge))

            # Correlation between edge and actual outcome
            try:
                metrics.actual_edge_correlation = float(np.corrcoef(edges, y_true)[0, 1])
            except Exception:
                metrics.actual_edge_correlation = 0.0

                # Store raw predictions for agreement analysis
            metrics._y_pred_raw = y_pred.copy()
            metrics._y_true_raw = y_true.copy()

            # Top-25% edge accuracy
            abs_edges = np.abs(edges)
            if np.any(abs_edges > 0):
                threshold = np.percentile(abs_edges, 75)
                high_edge_mask = abs_edges >= threshold
                if np.sum(high_edge_mask) > 0:
                    metrics.top_edge_accuracy = accuracy_score(
                        y_true[high_edge_mask], y_pred[high_edge_mask]
                    )

            # ── Flat betting simulation ($10/bet on edge > threshold) ──
            edge_threshold = self.min_edge_threshold
            bet_mask = np.abs(edges) >= edge_threshold

            if np.sum(bet_mask) > 0:
                bet_probs = y_prob[bet_mask]
                bet_outcomes = y_true[bet_mask]
                bet_edges = edges[bet_mask]

                # Simulate bets: $10 flat, win = +$9.09 (at -110 odds), loss = -$10
                metrics.flat_n_bets = int(np.sum(bet_mask))
                metrics.flat_wins = int(np.sum(bet_outcomes == 1))
                metrics.flat_losses = metrics.flat_n_bets - metrics.flat_wins
                metrics.flat_win_rate = metrics.flat_wins / metrics.flat_n_bets if metrics.flat_n_bets > 0 else 0.0
                metrics.flat_total_stake = metrics.flat_n_bets * 10.0

                # Profit calculation (assuming -110 odds)
                profits = np.where(
                    bet_outcomes == 1,
                    9.09,   # Win $9.09 on $10 bet
                    -10.0   # Lose $10
                )
                metrics.flat_total_profit = float(np.sum(profits))

                metrics.flat_roi = (
                    metrics.flat_total_profit / metrics.flat_total_stake
                    if metrics.flat_total_stake > 0 else 0.0
                )

                # Sharpe: risk-adjusted return
                if metrics.flat_n_bets > 1:
                    avg_profit = float(np.mean(profits))
                    std_profit = float(np.std(profits)) if np.std(profits) > 0 else 1.0
                    metrics.flat_sharpe = (avg_profit / std_profit) * math.sqrt(metrics.flat_n_bets)
                else:
                    metrics.flat_sharpe = 0.0

                # Max drawdown
                cumulative = np.cumsum(profits)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = running_max - cumulative
                metrics.flat_max_drawdown = float(np.max(drawdowns))

                # ── Kelly betting simulation (¼ Kelly) ──────────────
                bankroll = self.initial_bankroll
                kelly_profits = []
                for prob, outcome, edge in zip(bet_probs, bet_outcomes, bet_edges):
                    # Odds: assume -110 → decimal = 1.909
                    decimal_odds = 1.909
                    # Kelly fraction = (edge * decimal_odds) / (decimal_odds - 1)
                    # For -110: (edge * 1.909) / 0.909
                    kelly_pct = (edge * decimal_odds) / (decimal_odds - 1)
                    kelly_pct = np.clip(kelly_pct, 0, 0.25) * self.kelly_fraction
                    stake = bankroll * kelly_pct
                    if stake < 1.0:
                        continue
                    if outcome == 1:
                        profit = stake * (decimal_odds - 1)
                        bankroll += profit
                        kelly_profits.append(profit)
                    else:
                        bankroll -= stake
                        kelly_profits.append(-stake)

                if kelly_profits:
                    metrics.kelly_n_bets = len(kelly_profits)
                    kelly_profits_arr = np.array(kelly_profits)
                    metrics.kelly_wins = int(np.sum(kelly_profits_arr > 0))
                    metrics.kelly_losses = metrics.kelly_n_bets - metrics.kelly_wins
                    metrics.kelly_win_rate = metrics.kelly_wins / metrics.kelly_n_bets if metrics.kelly_n_bets > 0 else 0.0
                    metrics.kelly_total_stake = float(np.sum(np.abs(kelly_profits_arr[kelly_profits_arr < 0]) * -1))
                    # Use the actual bankroll-weighted stakes
                    # Reconstruct the stakes from the Kelly loop
                    actual_total_stake = sum(
                        abs(p) for p in kelly_profits_arr if p < 0
                    ) + sum(
                        p / (decimal_odds - 1) for p in kelly_profits_arr if p > 0
                    )
                    metrics.kelly_total_stake = actual_total_stake
                    metrics.kelly_total_profit = float(np.sum(kelly_profits_arr))
                    metrics.kelly_roi = (
                        metrics.kelly_total_profit / actual_total_stake
                        if actual_total_stake > 0 else 0.0
                    )
                    metrics.kelly_final_bankroll = bankroll

                    if metrics.kelly_n_bets > 1:
                        avg_kp = float(np.mean(kelly_profits_arr))
                        std_kp = float(np.std(kelly_profits_arr)) if np.std(kelly_profits_arr) > 0 else 1.0
                        metrics.kelly_sharpe = (avg_kp / std_kp) * math.sqrt(metrics.kelly_n_bets)

                    # Kelly drawdown
                    kelly_cumsum = np.cumsum(kelly_profits_arr)
                    kelly_rmax = np.maximum.accumulate(kelly_cumsum)
                    kelly_dd = kelly_rmax - kelly_cumsum
                    metrics.kelly_max_drawdown = float(np.max(kelly_dd)) if len(kelly_dd) > 0 else 0.0

        return metrics

    # ── Internal: Market Probabilities ───────────────────────────────────

    def _get_market_probs(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get market-implied home win probabilities from the DataFrame.

        Priority:
          1. elo_home_prob (computed chronologically by FeatureEngineer)
          2. Spread-based from point_diff
          3. 0.5 default
        """
        if "elo_home_prob" in df.columns:
            probs = df["elo_home_prob"].fillna(0.5).values
            return np.clip(probs, 0.01, 0.99).astype(np.float64)

        if "point_diff" in df.columns:
            # Convert point_diff to implied probability
            from betting_intel.features.market_inefficiency import (
                margin_to_implied_prob,
            )
            probs = df["point_diff"].apply(
                lambda x: margin_to_implied_prob(float(x), home=True)
                if pd.notna(x) else 0.5
            ).values
            return np.clip(probs, 0.01, 0.99).astype(np.float64)

        return np.full(len(df), 0.5, dtype=np.float64)

    # ── Internal: Aggregation ───────────────────────────────────────────

    def _aggregate_results(
        self,
        periods: list[ModelComparisonPeriod],
        feature_count: int,
        start_time: float,
    ) -> BacktestComparisonReport:
        """Aggregate per-period results into a unified report."""
        n_periods = len(periods)
        if n_periods == 0:
            return BacktestComparisonReport(generated_at=datetime.now().isoformat())

        # ── Aggregate classifier metrics ───────────────────────────
        c_total = ModelComparisonMetrics(model_name="classifier")
        m_total = ModelComparisonMetrics(model_name="market_inefficiency")

        total_n_train = sum(p.classifier.n_train for p in periods)
        total_n_test = sum(p.classifier.n_test for p in periods)

        # Weighted averages (by test set size)
        for p in periods:
            weight = p.classifier.n_test / max(total_n_test, 1)
            c_total.accuracy += p.classifier.accuracy * weight
            c_total.brier_score += p.classifier.brier_score * weight
            c_total.log_loss += p.classifier.log_loss * weight
            c_total.auc_roc += p.classifier.auc_roc * weight
            c_total.precision += p.classifier.precision * weight
            c_total.recall += p.classifier.recall * weight
            c_total.f1_score += p.classifier.f1_score * weight
            c_total.edge_capture_rate += p.classifier.edge_capture_rate * weight
            c_total.avg_predicted_edge += p.classifier.avg_predicted_edge * weight
            c_total.actual_edge_correlation += p.classifier.actual_edge_correlation * weight
            c_total.top_edge_accuracy += p.classifier.top_edge_accuracy * weight

            m_weight = p.market_inefficiency.n_test / max(total_n_test, 1)
            m_total.accuracy += p.market_inefficiency.accuracy * m_weight
            m_total.brier_score += p.market_inefficiency.brier_score * m_weight
            m_total.log_loss += p.market_inefficiency.log_loss * m_weight
            m_total.auc_roc += p.market_inefficiency.auc_roc * m_weight
            m_total.precision += p.market_inefficiency.precision * m_weight
            m_total.recall += p.market_inefficiency.recall * m_weight
            m_total.f1_score += p.market_inefficiency.f1_score * m_weight
            m_total.edge_capture_rate += p.market_inefficiency.edge_capture_rate * m_weight
            m_total.avg_predicted_edge += p.market_inefficiency.avg_predicted_edge * m_weight
            m_total.actual_edge_correlation += p.market_inefficiency.actual_edge_correlation * m_weight
            m_total.top_edge_accuracy += p.market_inefficiency.top_edge_accuracy * m_weight

            # Summed metrics (not weighted)
            c_total.flat_n_bets += p.classifier.flat_n_bets
            c_total.flat_wins += p.classifier.flat_wins
            c_total.flat_losses += p.classifier.flat_losses
            c_total.flat_total_stake += p.classifier.flat_total_stake
            c_total.flat_total_profit += p.classifier.flat_total_profit

            m_total.flat_n_bets += p.market_inefficiency.flat_n_bets
            m_total.flat_wins += p.market_inefficiency.flat_wins
            m_total.flat_losses += p.market_inefficiency.flat_losses
            m_total.flat_total_stake += p.market_inefficiency.flat_total_stake
            m_total.flat_total_profit += p.market_inefficiency.flat_total_profit

            c_total.kelly_n_bets += p.classifier.kelly_n_bets
            c_total.kelly_wins += p.classifier.kelly_wins
            c_total.kelly_losses += p.classifier.kelly_losses
            c_total.kelly_total_stake += p.classifier.kelly_total_stake
            c_total.kelly_total_profit += p.classifier.kelly_total_profit
            c_total.kelly_final_bankroll += p.classifier.kelly_final_bankroll

            m_total.kelly_n_bets += p.market_inefficiency.kelly_n_bets
            m_total.kelly_wins += p.market_inefficiency.kelly_wins
            m_total.kelly_losses += p.market_inefficiency.kelly_losses
            m_total.kelly_total_stake += p.market_inefficiency.kelly_total_stake
            m_total.kelly_total_profit += p.market_inefficiency.kelly_total_profit
            m_total.kelly_final_bankroll += p.market_inefficiency.kelly_final_bankroll

            # Confusion matrices
            c_total.tp += p.classifier.tp
            c_total.tn += p.classifier.tn
            c_total.fp += p.classifier.fp
            c_total.fn += p.classifier.fn

            m_total.tp += p.market_inefficiency.tp
            m_total.tn += p.market_inefficiency.tn
            m_total.fp += p.market_inefficiency.fp
            m_total.fn += p.market_inefficiency.fn

        # Compute aggregate flat ROI
        if c_total.flat_total_stake > 0:
            c_total.flat_roi = c_total.flat_total_profit / c_total.flat_total_stake
        if m_total.flat_total_stake > 0:
            m_total.flat_roi = m_total.flat_total_profit / m_total.flat_total_stake

        # Compute aggregate Kelly ROI
        if c_total.kelly_total_stake > 0:
            c_total.kelly_roi = c_total.kelly_total_profit / c_total.kelly_total_stake
        if m_total.kelly_total_stake > 0:
            m_total.kelly_roi = m_total.kelly_total_profit / m_total.kelly_total_stake

        # Flat win rates (weighted)
        total_c_bets = c_total.flat_wins + c_total.flat_losses
        total_m_bets = m_total.flat_wins + m_total.flat_losses
        c_total.flat_win_rate = c_total.flat_wins / max(total_c_bets, 1)
        m_total.flat_win_rate = m_total.flat_wins / max(total_m_bets, 1)

        # Kelly win rates
        total_c_kelly = c_total.kelly_wins + c_total.kelly_losses
        total_m_kelly = m_total.kelly_wins + m_total.kelly_losses
        c_total.kelly_win_rate = c_total.kelly_wins / max(total_c_kelly, 1)
        m_total.kelly_win_rate = m_total.kelly_wins / max(total_m_kelly, 1)

        # ── Model agreement across all periods ─────────────────────
        agreement_total = 0
        disagreement_total = 0
        classifier_wins_on_disagree = 0
        market_wins_on_disagree = 0

        for p in periods:
            if not hasattr(p.classifier, '_y_pred_raw') or not hasattr(p.market_inefficiency, '_y_pred_raw'):
                continue
            c_preds = p.classifier._y_pred_raw
            m_preds = p.market_inefficiency._y_pred_raw
            actuals = p.classifier._y_true_raw

            if c_preds is None or m_preds is None:
                continue

            agree = (c_preds == m_preds)
            agreement_total += int(np.sum(agree))
            disagreement_total += int(np.sum(~agree))

            if np.sum(~agree) > 0:
                c_right = np.sum((c_preds == actuals) & (c_preds != m_preds))
                m_right = np.sum((m_preds == actuals) & (c_preds != m_preds))
                classifier_wins_on_disagree += int(c_right)
                market_wins_on_disagree += int(m_right)

        total_games = agreement_total + disagreement_total
        agreement_rate = agreement_total / max(total_games, 1)
        disagreement_rate = disagreement_total / max(total_games, 1)

        c_won_rate = classifier_wins_on_disagree / max(disagreement_total, 1)
        m_won_rate = market_wins_on_disagree / max(disagreement_total, 1)

        # ── Statistical significance (paired t-test on accuracy) ──
        accuracy_deltas = np.array([p.delta_accuracy for p in periods])
        if len(accuracy_deltas) >= 3:
            mean_delta = np.mean(accuracy_deltas)
            std_delta = np.std(accuracy_deltas, ddof=1) if len(accuracy_deltas) > 1 else 1.0
            t_stat = mean_delta / (std_delta / math.sqrt(len(accuracy_deltas)))
            # Approximate p-value from t-distribution (using normal for simplicity)
            from scipy.stats import norm
            p_value = float(2.0 * (1.0 - norm.cdf(abs(t_stat))))  # two-tailed
        else:
            p_value = 1.0

        is_significant = p_value < 0.05
        if is_significant and p_value < 0.001:
            confidence = "very_high"
        elif is_significant and p_value < 0.01:
            confidence = "high"
        elif is_significant:
            confidence = "medium"
        else:
            confidence = "none"

        # ── Build report ──────────────────────────────────────────
        report = BacktestComparisonReport(
            generated_at=datetime.now().isoformat(),
            total_n_train=total_n_train,
            total_n_test=total_n_test,
            n_periods=n_periods,
            n_seasons=max(1, n_periods // 2),
            classifier=c_total,
            market_inefficiency=m_total,
            delta_accuracy=m_total.accuracy - c_total.accuracy,
            delta_brier=c_total.brier_score - m_total.brier_score,
            delta_flat_roi=m_total.flat_roi - c_total.flat_roi,
            delta_kelly_roi=m_total.kelly_roi - c_total.kelly_roi,
            delta_flat_sharpe=m_total.flat_sharpe - c_total.flat_sharpe,
            delta_kelly_sharpe=m_total.kelly_sharpe - c_total.kelly_sharpe,
            delta_edge_capture=m_total.edge_capture_rate - c_total.edge_capture_rate,
            delta_win_rate=m_total.flat_win_rate - c_total.flat_win_rate,
            delta_top_edge_accuracy=m_total.top_edge_accuracy - c_total.top_edge_accuracy,
            periods=periods,
            is_significant=is_significant,
            p_value=round(p_value, 4),
            confidence_level=confidence,
            agreement_rate=agreement_rate,
            disagreement_rate=disagreement_rate,
            classifier_won_rate=c_won_rate,
            market_won_rate=m_won_rate,
            min_edge_threshold=self.min_edge_threshold,
            kelly_fraction=self.kelly_fraction,
            initial_bankroll=self.initial_bankroll,
            total_elapsed_seconds=round(time.time() - start_time, 1),
            feature_count=feature_count,
        )

        return report

    # ── Internal: Helpers ───────────────────────────────────────────────

    def _period_label(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame,
        idx: int, total: int,
    ) -> str:
        """Create a human-readable label for this period."""
        try:
            train_start = str(train_df["GAME_DATE"].iloc[0])[:10] if "GAME_DATE" in train_df.columns else "?"
            train_end = str(train_df["GAME_DATE"].iloc[-1])[:10] if "GAME_DATE" in train_df.columns else "?"
            test_start = str(test_df["GAME_DATE"].iloc[0])[:10] if "GAME_DATE" in test_df.columns else "?"
            test_end = str(test_df["GAME_DATE"].iloc[-1])[:10] if "GAME_DATE" in test_df.columns else "?"
            return f"T{idx + 1}: train={train_start}..{train_end} | test={test_start}..{test_end}"
        except Exception:
            return f"Period {idx + 1}/{total}"

    def _print_header(self) -> None:
        """Print the backtest comparison header."""
        print(f"\n{'█' * 70}")
        print(f"  BACKTEST COMPARISON")
        print(f"  Comparing RobustPredictionSystem vs MarketInefficiencySystem")
        print(f"  on historical NBA data with chronological walk-forward")
        print(f"{'█' * 70}")
        print()

    def _print_period_summary(self, period: ModelComparisonPeriod) -> None:
        """Print a one-line summary of a period's results."""
        c = period.classifier
        m = period.market_inefficiency
        d_acc = period.delta_accuracy
        d_brier = period.delta_brier
        d_roi = period.delta_flat_roi

        acc_str = f"acc: {c.accuracy:.1%}→{m.accuracy:.1%} ({d_acc:+.1%})"
        brier_str = f"brier: {c.brier_score:.4f}→{m.brier_score:.4f} ({d_brier:+.4f})"
        roi_str = f"roi: {c.flat_roi:.1%}→{m.flat_roi:.1%} ({d_roi:+.1%})"

        print(f"     {acc_str} | {brier_str} | {roi_str}")

    def _print_metric_row(
        self, label: str,
        old_val: float, new_val: float,
        delta: float, direction: str = "↑",
        pct: bool = False,
    ) -> None:
        """Print a formatted metric comparison row."""
        def fmt(v: float, p: bool) -> str:
            return f"{v:.2%}" if p else f"{v:.4f}"

        d_str = f"{delta:+.2%}" if pct else f"{delta:+.4f}"
        is_improvement = delta > 0
        if direction == "↓":
            # For lower-is-better metrics (Brier, log loss), delta = old - new,
            # so delta > 0 means new < old = improvement
            is_improvement = delta > 0
        else:
            # For higher-is-better metrics (accuracy, AUC, etc.), delta = new - old,
            # so delta > 0 means improvement
            is_improvement = delta > 0

        if abs(delta) < 0.0001:
            arrow = "➖"
        elif is_improvement:
            arrow = "✅"
        else:
            arrow = "❌"
        print(f"  {label:<25}  {fmt(old_val, pct):>8}  →  {fmt(new_val, pct):>8}  {arrow} Δ={d_str:>10}")

    def _print_flat_row(self, m: ModelComparisonMetrics, label: str) -> None:
        """Print flat betting metrics for one model."""
        print(f"  {label}:")
        print(f"    Bets:       {m.flat_n_bets:>5}  |  Wins: {m.flat_wins:>4}  Losses: {m.flat_losses:>4}")
        print(f"    Win Rate:   {m.flat_win_rate:.1%}")
        print(f"    Total Stake: ${m.flat_total_stake:,.0f}")
        profit_str = f"+${m.flat_total_profit:,.0f}" if m.flat_total_profit >= 0 else f"-${abs(m.flat_total_profit):,.0f}"
        print(f"    Profit:     {profit_str}")
        print(f"    ROI:        {m.flat_roi:.2%}")
        print(f"    Sharpe:     {m.flat_sharpe:.3f}")
        print(f"    Max DD:     ${m.flat_max_drawdown:,.0f}")

    def _print_kelly_row(self, m: ModelComparisonMetrics, label: str) -> None:
        """Print Kelly betting metrics for one model."""
        print(f"  {label}:")
        print(f"    Bets:       {m.kelly_n_bets:>5}  |  Wins: {m.kelly_wins:>4}  Losses: {m.kelly_losses:>4}")
        print(f"    Win Rate:   {m.kelly_win_rate:.1%}")
        print(f"    Total Stake: ${m.kelly_total_stake:,.0f}")
        profit_str = f"+${m.kelly_total_profit:,.0f}" if m.kelly_total_profit >= 0 else f"-${abs(m.kelly_total_profit):,.0f}"
        print(f"    Profit:     {profit_str}")
        print(f"    ROI:        {m.kelly_roi:.2%}")
        print(f"    Sharpe:     {m.kelly_sharpe:.3f}")
        print(f"    Max DD:     ${m.kelly_max_drawdown:,.0f}")

    def _generate_verdict(self, report: BacktestComparisonReport) -> str:
        """Generate a human-readable verdict."""
        parts = []

        # Accuracy
        if report.delta_accuracy > 0.02:
            parts.append(f"Market Inefficiency is clearly better: {report.delta_accuracy:+.1%} accuracy gain")
        elif report.delta_accuracy > 0.005:
            parts.append(f"Market Inefficiency shows modest accuracy gain ({report.delta_accuracy:+.1%})")
        elif report.delta_accuracy > -0.005:
            parts.append("No meaningful accuracy difference between the two")
        else:
            parts.append(f"Classifier outperforms on accuracy ({report.delta_accuracy:+.1%})")

        # Brier
        if report.delta_brier > 0.01:
            parts.append(f"and significantly better calibration (Brier {report.delta_brier:+.4f})")
        elif report.delta_brier > 0:
            parts.append(f"with slightly better calibration (Brier {report.delta_brier:+.4f})")
        else:
            parts.append(f"but classifier has better calibration (Brier {report.delta_brier:+.4f})")

        # ROI
        if report.delta_flat_roi > 0.05:
            parts.append(f"and much higher flat-betting ROI ({report.delta_flat_roi:+.1%})")
        elif report.delta_flat_roi > 0:
            parts.append(f"and better flat-betting ROI ({report.delta_flat_roi:+.1%})")
        else:
            parts.append(f"but classifier has better ROI ({report.delta_flat_roi:+.1%})")

        # Sharpe
        if report.delta_flat_sharpe > 0.2:
            parts.append("with superior risk-adjusted returns (Sharpe).")
        elif report.delta_flat_sharpe > 0:
            parts.append("with slightly better risk-adjusted returns.")
        else:
            parts.append("but lower risk-adjusted returns.")

        # Significance
        if report.is_significant:
            parts.append(f" Results are statistically significant (p={report.p_value:.4f}).")
        else:
            parts.append(" Results are NOT statistically significant (more data needed).")

        return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════


def run_backtest_comparison(
    n_periods: int = 5,
    test_size_games: int = 100,
    min_edge: float = 0.02,
    verbose: bool = True,
) -> BacktestComparisonReport:
    """
    One-call convenience: run the full backtest comparison.

    Args:
        n_periods: Number of chronological test periods
        test_size_games: Approximate games per test period
        min_edge: Minimum edge threshold for betting simulation
        verbose: Print progress

    Returns:
        BacktestComparisonReport
    """
    comparison = BacktestComparison(min_edge_threshold=min_edge)
    return comparison.run(
        n_test_periods=n_periods,
        test_size_games=test_size_games,
        verbose=verbose,
    )


__all__ = [
    "BacktestComparison",
    "BacktestComparisonReport",
    "ModelComparisonMetrics",
    "ModelComparisonPeriod",
    "run_backtest_comparison",
]
