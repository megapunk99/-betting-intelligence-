"""
Monte Carlo simulation for betting strategy evaluation.
Simulates thousands of possible season outcomes to:
  - Estimate bankroll variance
  - Compute confidence intervals for win rate and profit
  - Assess risk of ruin
  - Compare strategy robustness
  - Detect overfitting (if simulated performance >> actual)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


@dataclass
class SimulationResult:
    """Results from a Monte Carlo simulation."""

    n_simulations: int
    median_profit: float
    mean_profit: float
    profit_ci_90: Tuple[float, float]  # 5th and 95th percentile
    profit_ci_95: Tuple[float, float]  # 2.5th and 97.5th percentile
    median_win_rate: float
    win_rate_ci_95: Tuple[float, float]
    median_roi: float
    roi_ci_95: Tuple[float, float]
    max_drawdown_ci_95: Tuple[float, float]
    risk_of_ruin: float  # Probability of losing >50% of bankroll
    probability_profit: float  # Probability of being profitable
    probability_beating_market: float  # Probability of >52.38% win rate
    percentile_10: float
    percentile_25: float
    percentile_75: float
    percentile_90: float
    all_outcomes: np.ndarray  # All simulated profit outcomes


class MonteCarloSimulator:
    """
    Monte Carlo simulation for betting strategies.

    Uses bootstrapped resampling of historical bet outcomes to
    generate thousands of possible future seasons. This provides
    a realistic estimate of strategy variance and risk.

    Key features:
      - Bootstrapped bet resampling (with replacement)
      - Win rate uncertainty estimation
      - Drawdown simulation
      - Risk of ruin calculation
      - Confidence intervals for all metrics
    """

    def __init__(self, n_simulations: int = 10000, seed: int = 42):
        """
        Args:
            n_simulations: Number of simulated seasons to run
            seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.result: Optional[SimulationResult] = None

    def simulate_from_bets(
        self,
        bets_df: pd.DataFrame,
        n_games_per_season: int = 500,
        initial_bankroll: float = 10000.0,
        stake_per_bet: float = 0.02,  # 2% fractional Kelly
        vigorish: float = 0.045,  # ~4.5% vig
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation from historical bet data.

        Args:
            bets_df: DataFrame with bet outcomes (must have 'profit_units' column)
            n_games_per_season: Number of bets per simulated season
            initial_bankroll: Starting bankroll for each simulation
            stake_per_bet: Fraction of bankroll to bet each time
            vigorish: Bookmaker vigorish

        Returns:
            SimulationResult with all aggregated metrics
        """
        if bets_df is None or len(bets_df) == 0:
            return self._empty_result()

        # Extract bet outcomes
        profits = bets_df["profit_units"].values
        outcomes = bets_df["outcome"].values
        win_mask = outcomes == "WIN"

        win_rate_historical = win_mask.mean()
        n_bets_available = len(profits)

        print(f"  Monte Carlo: {self.n_simulations:,} simulations from {n_bets_available} historical bets")
        print(f"  Historical win rate: {win_rate_historical:.1%}")

        # Run simulations
        all_profits = np.zeros(self.n_simulations)
        all_win_rates = np.zeros(self.n_simulations)
        all_rois = np.zeros(self.n_simulations)
        all_max_drawdowns = np.zeros(self.n_simulations)

        for sim_idx in range(self.n_simulations):
            # Bootstrap: sample with replacement from historical bets
            sampled_indices = self.rng.integers(0, n_bets_available, size=n_games_per_season)
            sampled_profits = profits[sampled_indices]
            sampled_outcomes = outcomes[sampled_indices]

            # Track bankroll
            bankroll = initial_bankroll
            peak_bankroll = initial_bankroll
            max_drawdown = 0.0
            total_invested = 0.0

            for profit, outcome in zip(sampled_profits, sampled_outcomes):
                # Dynamic stake sizing
                stake = bankroll * stake_per_bet
                if stake < 1.0:
                    break

                total_invested += stake

                if outcome == "WIN":
                    # At -110 odds, profit = stake * (100/110)
                    bankroll += stake * (1.0 / 1.1)  # simplified
                elif outcome == "LOSS":
                    bankroll -= stake

                # Track drawdown
                if bankroll > peak_bankroll:
                    peak_bankroll = bankroll
                dd = (peak_bankroll - bankroll) / peak_bankroll
                if dd > max_drawdown:
                    max_drawdown = dd

                # Stop if bankrupt
                if bankroll < initial_bankroll * 0.1:
                    break

            total_profit = bankroll - initial_bankroll
            sim_win_rate = np.mean(sampled_outcomes == "WIN")
            sim_roi = total_profit / max(total_invested, 1) * 100

            all_profits[sim_idx] = total_profit
            all_win_rates[sim_idx] = sim_win_rate
            all_rois[sim_idx] = sim_roi
            all_max_drawdowns[sim_idx] = max_drawdown

        # Compute statistics
        all_profits_sorted = np.sort(all_profits)
        all_win_rates_sorted = np.sort(all_win_rates)
        all_rois_sorted = np.sort(all_rois)
        all_dd_sorted = np.sort(all_max_drawdowns)

        def percentile(data, p):
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        result = SimulationResult(
            n_simulations=self.n_simulations,
            median_profit=np.median(all_profits),
            mean_profit=np.mean(all_profits),
            profit_ci_90=(percentile(all_profits_sorted, 5),
                          percentile(all_profits_sorted, 95)),
            profit_ci_95=(percentile(all_profits_sorted, 2.5),
                          percentile(all_profits_sorted, 97.5)),
            median_win_rate=np.median(all_win_rates),
            win_rate_ci_95=(percentile(all_win_rates_sorted, 2.5),
                            percentile(all_win_rates_sorted, 97.5)),
            median_roi=np.median(all_rois),
            roi_ci_95=(percentile(all_rois_sorted, 2.5),
                       percentile(all_rois_sorted, 97.5)),
            max_drawdown_ci_95=(percentile(all_dd_sorted, 2.5),
                                percentile(all_dd_sorted, 97.5)),
            risk_of_ruin=np.mean(all_profits < -initial_bankroll * 0.5),
            probability_profit=np.mean(all_profits > 0),
            probability_beating_market=np.mean(all_win_rates > 0.5238),
            percentile_10=percentile(all_profits_sorted, 10),
            percentile_25=percentile(all_profits_sorted, 25),
            percentile_75=percentile(all_profits_sorted, 75),
            percentile_90=percentile(all_profits_sorted, 90),
            all_outcomes=all_profits,
        )

        self.result = result
        return result

    def simulate_win_rate_only(
        self,
        n_bets: int,
        true_win_rate: float,
    ) -> Dict[str, float]:
        """
        Simple binomial simulation to understand win rate variance.
        Useful for determining if a strategy's win rate is statistically significant.

        Args:
            n_bets: Number of bets
            true_win_rate: Assumed true win rate

        Returns:
            Dictionary with confidence intervals for win rate
        """
        outcomes = self.rng.binomial(1, true_win_rate, size=(self.n_simulations, n_bets))
        simulated_wrs = outcomes.mean(axis=1)
        simulated_wrs.sort()

        def pct(p):
            idx = int(len(simulated_wrs) * p / 100)
            return simulated_wrs[min(idx, len(simulated_wrs) - 1)]

        break_even = 52.38  # -110 odds

        return {
            "n_simulations": self.n_simulations,
            "n_bets": n_bets,
            "assumed_win_rate": true_win_rate,
            "median_wr": np.median(simulated_wrs),
            "ci_80": (pct(10), pct(90)),
            "ci_90": (pct(5), pct(95)),
            "ci_95": (pct(2.5), pct(97.5)),
            "prob_profitable": np.mean(simulated_wrs > break_even / 100),
            "prob_losing": np.mean(simulated_wrs < 0.47),
        }

    def compare_strategies(
        self,
        strategy_results: Dict[str, pd.DataFrame],
        n_games_per_season: int = 500,
    ) -> pd.DataFrame:
        """
        Run Monte Carlo on multiple strategies and compare.

        Args:
            strategy_results: Dict of strategy_name -> bets_df
            n_games_per_season: Number of bets per simulated season

        Returns:
            DataFrame comparing strategy risk/return profiles
        """
        comparison = []

        for name, bets_df in strategy_results.items():
            if bets_df is None or len(bets_df) < 10:
                continue

            sim_result = self.simulate_from_bets(
                bets_df, n_games_per_season=n_games_per_season
            )

            comparison.append({
                "Strategy": name,
                "Historical Bets": len(bets_df),
                "Historical WR": (bets_df["outcome"] == "WIN").mean(),
                "Median Profit": f"${sim_result.median_profit:,.0f}",
                "Profit CI 95%": f"${sim_result.profit_ci_95[0]:,.0f} to ${sim_result.profit_ci_95[1]:,.0f}",
                "P(Profitable)": f"{sim_result.probability_profit:.1%}",
                "P(Beating Vig)": f"{sim_result.probability_beating_market:.1%}",
                "Risk of Ruin": f"{sim_result.risk_of_ruin:.1%}",
                "Median ROI": f"{sim_result.median_roi:.1f}%",
            })

        return pd.DataFrame(comparison)

    def _empty_result(self) -> SimulationResult:
        return SimulationResult(
            n_simulations=0,
            median_profit=0, mean_profit=0,
            profit_ci_90=(0, 0), profit_ci_95=(0, 0),
            median_win_rate=0, win_rate_ci_95=(0, 0),
            median_roi=0, roi_ci_95=(0, 0),
            max_drawdown_ci_95=(0, 0),
            risk_of_ruin=1.0, probability_profit=0,
            probability_beating_market=0,
            percentile_10=0, percentile_25=0, percentile_75=0, percentile_90=0,
            all_outcomes=np.array([]),
        )

    def format_report(self, result: Optional[SimulationResult] = None) -> str:
        """Format simulation results as a readable report."""
        r = result or self.result
        if r is None or r.n_simulations == 0:
            return "No simulation results available."

        def fmt_money(v):
            return f"${v:,.0f}" if abs(v) < 10000 else f"${v:,.0f}"

        lines = [
            "═" * 60,
            "MONTE CARLO SIMULATION REPORT",
            f"  {r.n_simulations:,} simulations | {self.seed=}",
            "═" * 60,
            "",
            "── Profit Analysis ──",
            f"  Median Profit:    {fmt_money(r.median_profit):>10s}",
            f"  Mean Profit:      {fmt_money(r.mean_profit):>10s}",
            f"  90% CI:           {fmt_money(r.profit_ci_90[0]):>10s} to {fmt_money(r.profit_ci_90[1]):>10s}",
            f"  95% CI:           {fmt_money(r.profit_ci_95[0]):>10s} to {fmt_money(r.profit_ci_95[1]):>10s}",
            f"  10th percentile:  {fmt_money(r.percentile_10):>10s}",
            f"  25th percentile:  {fmt_money(r.percentile_25):>10s}",
            f"  75th percentile:  {fmt_money(r.percentile_75):>10s}",
            f"  90th percentile:  {fmt_money(r.percentile_90):>10s}",
            "",
            "── Win Rate Analysis ──",
            f"  Median Win Rate:  {r.median_win_rate:.1%}",
            f"  95% CI:           {r.win_rate_ci_95[0]:.1%} to {r.win_rate_ci_95[1]:.1%}",
            f"  P(Beating Vig):   {r.probability_beating_market:.1%}",
            "",
            "── Risk Analysis ──",
            f"  P(Profitable):    {r.probability_profit:.1%}",
            f"  Risk of Ruin:     {r.risk_of_ruin:.1%}",
            f"  Median ROI:       {r.median_roi:.1f}%",
            f"  95% ROI CI:       {r.roi_ci_95[0]:.1f}% to {r.roi_ci_95[1]:.1f}%",
            f"  Max DD 95% CI:    {r.max_drawdown_ci_95[0]:.1%} to {r.max_drawdown_ci_95[1]:.1%}",
            "",
            "═" * 60,
        ]
        return "\n".join(lines)
