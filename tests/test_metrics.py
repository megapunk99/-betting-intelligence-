"""
Tests for the backtesting metrics module.
"""
import numpy as np
import pandas as pd
import pytest

from betting_intel.backtesting.metrics import BacktestMetrics


class TestBacktestMetrics:
    """Tests for the BacktestMetrics static methods."""

    def test_compute_all_empty_df(self):
        metrics = BacktestMetrics.compute_all(None)
        assert metrics == {"error": "no_bets"}

    def test_compute_all_empty_dataframe(self):
        df = pd.DataFrame()
        metrics = BacktestMetrics.compute_all(df)
        assert "error" in metrics

    def test_wilson_ci_zero_total(self):
        lower, upper = BacktestMetrics._wilson_ci(0, 0)
        assert lower == 0
        assert upper == 0

    def test_wilson_ci_perfect(self):
        lower, upper = BacktestMetrics._wilson_ci(100, 100)
        assert lower > 0.95
        assert upper <= 1.0

    def test_wilson_ci_fifty(self):
        lower, upper = BacktestMetrics._wilson_ci(50, 100)
        assert lower < 0.5 < upper

    def test_longest_streak_empty(self):
        assert BacktestMetrics._longest_streak(pd.Series(dtype=bool)) == 0

    def test_longest_streak_all_true(self):
        s = pd.Series([True, True, True, True])
        assert BacktestMetrics._longest_streak(s) == 4

    def test_longest_streak_mixed(self):
        s = pd.Series([True, True, False, True, True, True, False])
        assert BacktestMetrics._longest_streak(s) == 3

    def test_longest_streak_all_false(self):
        s = pd.Series([False, False, False])
        assert BacktestMetrics._longest_streak(s) == 0

    def test_check_overfitting_single(self):
        assert not BacktestMetrics.check_overfitting([{"win_rate": 0.55}])

    def test_check_overfitting_few(self):
        assert not BacktestMetrics.check_overfitting([])

    def test_check_overfitting_detects(self):
        metrics_list = [
            {"win_rate": 0.59},
            {"win_rate": 0.60},
            {"win_rate": 0.59},
        ]
        assert BacktestMetrics.check_overfitting(metrics_list)

    def test_check_overfitting_no_false_positive(self):
        metrics_list = [
            {"win_rate": 0.53},
            {"win_rate": 0.55},
            {"win_rate": 0.57},
        ]
        assert not BacktestMetrics.check_overfitting(metrics_list)

    def test_format_report_error(self):
        report = BacktestMetrics.format_report({"error": "no_bets"})
        assert "No bets" in report


class TestBacktestMetricsIntegration:
    """Integration-level tests with realistic data."""

    @pytest.fixture
    def sample_bets(self):
        np.random.seed(42)
        n_bets = 200
        dates = pd.date_range("2023-01-01", periods=n_bets, freq="D")

        outcomes = np.random.choice(["WIN", "LOSS", "PUSH"], size=n_bets, p=[0.55, 0.40, 0.05])
        profits = []
        for o in outcomes:
            if o == "WIN":
                profits.append(1.0)
            elif o == "LOSS":
                profits.append(-1.0)
            else:
                profits.append(0.0)

        df = pd.DataFrame({
            "game_date": dates,
            "game_id": [f"GAME_{i:04d}" for i in range(n_bets)],
            "strategy": "momentum",
            "model": "Logistic",
            "bet_type": "SPREAD",
            "outcome": outcomes,
            "profit_units": profits,
            "edge_pct": np.random.uniform(-0.05, 0.10, size=n_bets),
        })
        return df

    def test_compute_all_with_realistic_data(self, sample_bets):
        metrics = BacktestMetrics.compute_all(sample_bets)
        assert metrics["total_bets"] == 200
        assert metrics["wins"] > metrics["losses"]
        assert metrics["total_profit_units"] > 0
        assert metrics["max_drawdown_units"] >= 0
        assert metrics["sharpe_ratio"] != 0
        assert metrics["recovery_factor"] > 0
        assert "monthly_avg" in metrics
        assert "longest_win_streak" in metrics
        assert "longest_loss_streak" in metrics

    def test_compute_all_profit_factor(self, sample_bets):
        metrics = BacktestMetrics.compute_all(sample_bets)
        assert metrics["profit_factor"] > 0
        assert metrics["profit_factor"] > 1.0  # Should be profitable

    def test_compute_all_edge_correlation(self, sample_bets):
        metrics = BacktestMetrics.compute_all(sample_bets)
        assert "avg_edge_pct" in metrics
        assert "edge_outcome_corr" in metrics
        assert -1.0 <= metrics["edge_outcome_corr"] <= 1.0

    def test_format_report_with_data(self, sample_bets):
        metrics = BacktestMetrics.compute_all(sample_bets)
        report = BacktestMetrics.format_report(metrics)
        assert "BACKTEST PERFORMANCE REPORT" in report
        assert "Total Bets" in report
        assert "Win Rate" in report
        assert "Sharpe" in report
        assert "Max Drawdown" in report

    def test_win_rate_ci_valid(self, sample_bets):
        metrics = BacktestMetrics.compute_all(sample_bets)
        lower = metrics["win_rate_ci_lower"]
        upper = metrics["win_rate_ci_upper"]
        assert 0 <= lower <= upper <= 1
        assert metrics["win_rate_ci_lower"] < metrics["win_rate"] < metrics["win_rate_ci_upper"]
