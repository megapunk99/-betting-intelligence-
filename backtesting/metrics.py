"""
Performance metrics for betting strategy evaluation.
All metrics assume units-based profit/loss tracking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats


class BacktestMetrics:
    """Computes comprehensive performance metrics for betting strategies."""

    @staticmethod
    def compute_all(bets_df: pd.DataFrame) -> Dict[str, float]:
        """Compute all metrics from a bets DataFrame."""
        if bets_df is None or len(bets_df) == 0:
            return {"error": "no_bets"}

        metrics = {}
        df = bets_df.copy()

        # Basic counts
        total = len(df)
        wins = len(df[df["outcome"] == "WIN"])
        losses = len(df[df["outcome"] == "LOSS"])
        pushes = len(df[df["outcome"] == "PUSH"])
        decided = wins + losses

        if decided == 0:
            return {"error": "no_decided_bets"}

        # Core metrics
        metrics["total_bets"] = total
        metrics["wins"] = wins
        metrics["losses"] = losses
        metrics["pushes"] = pushes
        metrics["win_rate"] = wins / decided
        metrics["total_profit_units"] = df["profit_units"].sum()
        metrics["roi_pct"] = (metrics["total_profit_units"] / total) * 100

        # Expected value
        metrics["avg_profit_per_bet"] = df["profit_units"].mean()

        # Variance and standard deviation
        metrics["profit_std"] = df["profit_units"].std()
        metrics["profit_var"] = df["profit_units"].var()

        # Confidence intervals for win rate (Wilson score)
        metrics["win_rate_ci_lower"], metrics["win_rate_ci_upper"] = (
            BacktestMetrics._wilson_ci(wins, decided)
        )

        # Test if win rate is significantly > 50% (assuming -110 odds)
        p_value = stats.binomtest(wins, decided, p=0.5).pvalue
        if not isinstance(p_value, float):
            p_value = float(p_value)
        metrics["p_value_gt_50pct"] = float(p_value)
        metrics["is_significant"] = p_value < 0.05

        # Implied edge at -110 (52.38% needed to break even)
        break_even = 52.38 / 100
        if metrics["win_rate"] > break_even:
            metrics["edge_over_vig"] = metrics["win_rate"] - break_even
        else:
            metrics["edge_over_vig"] = metrics["win_rate"] - break_even

        # Drawdown analysis
        df_sorted = df.sort_values("game_date").reset_index(drop=True)
        cumulative = df_sorted["profit_units"].cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max

        metrics["max_drawdown_units"] = abs(drawdown.min())
        metrics["max_drawdown_pct"] = (
            abs(drawdown.min()) / max(cumulative.max(), 1)
        )
        metrics["avg_drawdown"] = abs(drawdown.mean())

        # Recovery factor
        if metrics["max_drawdown_units"] > 0:
            metrics["recovery_factor"] = (
                metrics["total_profit_units"] / metrics["max_drawdown_units"]
            )
        else:
            metrics["recovery_factor"] = float("inf")

        # Profit factor
        gross_wins = wins * abs(df[df["outcome"] == "WIN"]["profit_units"].mean())
        gross_losses = abs(losses * df[df["outcome"] == "LOSS"]["profit_units"].mean())
        metrics["profit_factor"] = (
            gross_wins / max(gross_losses, 1)
        )

        # Sharpe ratio (annualized)
        if metrics["profit_std"] > 0:
            metrics["sharpe_ratio"] = (
                metrics["avg_profit_per_bet"] / metrics["profit_std"]
                * np.sqrt(82)  # ~82 games per NBA season
            )
        else:
            metrics["sharpe_ratio"] = 0

        # Sortino ratio (downside deviation only)
        downside = df[df["profit_units"] < 0]["profit_units"]
        if len(downside) > 0 and downside.std() > 0:
            metrics["sortino_ratio"] = (
                metrics["avg_profit_per_bet"] / downside.std()
                * np.sqrt(82)
            )
        else:
            metrics["sortino_ratio"] = 0

        # Win/loss streak analysis
        metrics["longest_win_streak"] = BacktestMetrics._longest_streak(
            df["outcome"] == "WIN"
        )
        metrics["longest_loss_streak"] = BacktestMetrics._longest_streak(
            df["outcome"] == "LOSS"
        )

        # Monthly breakdown
        df_sorted["month"] = pd.to_datetime(df_sorted["game_date"]).dt.to_period("M")
        monthly = df_sorted.groupby("month")["profit_units"].sum()
        metrics["monthly_avg"] = monthly.mean()
        metrics["monthly_std"] = monthly.std()
        metrics["best_month"] = monthly.max() if len(monthly) > 0 else 0
        metrics["worst_month"] = monthly.min() if len(monthly) > 0 else 0
        metrics["profitable_months"] = (monthly > 0).sum()
        metrics["total_months"] = len(monthly)

        # Edge analysis
        if "edge_pct" in df.columns:
            metrics["avg_edge_pct"] = df["edge_pct"].mean()
            metrics["median_edge_pct"] = df["edge_pct"].median()
            metrics["edge_std"] = df["edge_pct"].std()

            # Correlation between edge size and outcome
            edge_outcome_corr = df["edge_pct"].corr(
                (df["outcome"] == "WIN").astype(int)
            )
            metrics["edge_outcome_corr"] = (
                edge_outcome_corr if not np.isnan(edge_outcome_corr) else 0
            )

        return metrics

    @staticmethod
    def _wilson_ci(wins: int, total: int, z: float = 1.96) -> Tuple[float, float]:
        """Wilson score confidence interval for binomial proportion."""
        if total == 0:
            return (0, 0)
        p = wins / total
        denominator = 1 + z**2 / total
        centre = (p + z**2 / (2 * total)) / denominator
        margin = (z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))) / denominator
        return (centre - margin, centre + margin)

    @staticmethod
    def _longest_streak(condition: pd.Series) -> int:
        """Calculate longest consecutive True streak."""
        if len(condition) == 0:
            return 0
        streaks = (condition != condition.shift()).cumsum()
        streak_lengths = condition.groupby(streaks).transform("cumsum")
        return int(streak_lengths.max())

    @staticmethod
    def check_overfitting(metrics_list: List[Dict]) -> bool:
        """
        Heuristic check for overfitting across multiple strategies.
        If all strategies show similar results, likely overfitting.
        """
        if len(metrics_list) < 2:
            return False

        win_rates = [m.get("win_rate", 0) for m in metrics_list if "win_rate" in m]
        if len(win_rates) < 2:
            return False

        std_wr = np.std(win_rates)
        mean_wr = np.mean(win_rates)

        # If all win rates are suspiciously high and similar
        if mean_wr > 0.58 and std_wr < 0.02:
            return True
        return False

    @staticmethod
    def format_report(metrics: Dict) -> str:
        """Format metrics as a readable report."""
        if "error" in metrics:
            return f"No bets generated: {metrics['error']}"

        report = [
            "─" * 50,
            "BACKTEST PERFORMANCE REPORT",
            "─" * 50,
            f"Total Bets:     {metrics.get('total_bets', 0)}",
            f"Wins:           {metrics.get('wins', 0)}",
            f"Losses:         {metrics.get('losses', 0)}",
            f"Pushes:         {metrics.get('pushes', 0)}",
            f"Win Rate:       {metrics.get('win_rate', 0):.2%}",
            f"ROI:            {metrics.get('roi_pct', 0):.2f}%",
            f"Profit:         {metrics.get('total_profit_units', 0):.1f} units",
            "",
            f"Risk Metrics:",
            f"Max Drawdown:   {metrics.get('max_drawdown_units', 0):.1f} units",
            f"Sharpe Ratio:   {metrics.get('sharpe_ratio', 0):.2f}",
            f"Sortino Ratio:  {metrics.get('sortino_ratio', 0):.2f}",
            f"Profit Factor:  {metrics.get('profit_factor', 0):.2f}",
            f"Recovery:       {metrics.get('recovery_factor', 0):.2f}",
            "",
            f"Streaks:",
            f"Longest Win:    {metrics.get('longest_win_streak', 0)}",
            f"Longest Loss:   {metrics.get('longest_loss_streak', 0)}",
            "",
            f"Statistical Significance:",
            f"Win Rate CI:    {metrics.get('win_rate_ci_lower', 0):.1%} - {metrics.get('win_rate_ci_upper', 0):.1%}",
            f"P-value >50%:   {metrics.get('p_value_gt_50pct', 1):.4f}",
            f"Significant:    {'YES' if metrics.get('is_significant', False) else 'NO'}",
            "",
            f"Monthly:",
            f"Avg Month:      {metrics.get('monthly_avg', 0):.1f} units",
            f"Best Month:     {metrics.get('best_month', 0):.1f} units",
            f"Worst Month:    {metrics.get('worst_month', 0):.1f} units",
            f"Profitable:     {metrics.get('profitable_months', 0)}/{metrics.get('total_months', 0)} months",
            "─" * 50,
        ]
        return "\n".join(report)
