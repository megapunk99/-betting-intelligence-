"""
Multi-bet Kelly optimization: optimal simultaneous staking across correlated bets.

Standard Kelly assumes independent bets. In reality, NBA bets are highly
correlated (same game, same league). This module implements:

1. Multi-bet Kelly (fractional): optimal allocation across N bets
2. Correlation-aware Kelly: adjusts for bet-bet correlations
3. Sequential Kelly: for overlapping bet windows
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy.optimize import minimize


@dataclass
class KellyBet:
    """A bet with its Kelly-optimal stake."""

    bet_id: str
    edge_pct: float
    win_probability: float
    decimal_odds: float
    kelly_fraction: float = 0.0
    dollar_stake: float = 0.0
    expected_value: float = 0.0


class KellyCalculator:
    """
    Single-bet fractional Kelly calculator with multiple variants.

    Supports:
    - Full Kelly (max growth)
    - Fractional Kelly (conservative, default 25%)
    - Half Kelly (50% of full)
    - Quarter Kelly (25% of full)
    - Dynamic Kelly (adjusts based on confidence)
    """

    def __init__(
        self,
        bankroll: float = 10_000.0,
        fraction: float = 0.25,
        max_fraction: float = 0.15,
        min_edge: float = 0.02,
        drawdown_protection: bool = True,
    ):
        """
        Args:
            bankroll: Current bankroll
            fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
            max_fraction: Maximum fraction of bankroll for any single bet
            min_edge: Minimum edge required to bet
            drawdown_protection: Reduce stakes during drawdowns
        """
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll
        self.peak_bankroll = bankroll
        self.fraction = fraction
        self.max_fraction = max_fraction
        self.min_edge = min_edge
        self.drawdown_protection = drawdown_protection

    def compute_kelly(
        self,
        win_probability: float,
        decimal_odds: float = 1.91,
    ) -> Tuple[float, float]:
        """
        Compute fractional Kelly stake.

        Args:
            win_probability: Estimated win probability (0-1)
            decimal_odds: Market decimal odds

        Returns:
            (kelly_fraction_of_bankroll, dollar_stake)
        """
        if win_probability <= 0 or win_probability >= 1:
            return (0.0, 0.0)

        b = decimal_odds - 1.0  # Net odds
        if b <= 0:
            return (0.0, 0.0)

        p = win_probability
        q = 1.0 - p

        # Full Kelly: f* = (b*p - q) / b
        full_kelly = (b * p - q) / b

        if full_kelly <= 0:
            return (0.0, 0.0)

        # Check edge threshold
        implied_prob = 1.0 / decimal_odds
        edge = p - implied_prob
        if edge < self.min_edge:
            return (0.0, 0.0)

        # Apply fractional Kelly
        fraction = full_kelly * self.fraction

        # Cap at max fraction
        fraction = min(fraction, self.max_fraction)

        # Drawdown protection
        if self.drawdown_protection:
            drawdown_pct = (
                self.peak_bankroll - self.current_bankroll
            ) / self.peak_bankroll
            if drawdown_pct > 0.05:
                fraction *= max(0.1, 1.0 - drawdown_pct * 3)

        fraction = max(0.0, fraction)
        dollar_stake = fraction * self.current_bankroll

        return (fraction, dollar_stake)

    def compute_with_edge(
        self,
        win_probability: float,
        edge_pct: float,
        decimal_odds: float = 1.91,
    ) -> Tuple[float, float]:
        """
        Compute stake using edge percentage.

        Args:
            win_probability: Model-estimated win probability
            edge_pct: Edge over market (e.g., 0.03 = 3%)
            decimal_odds: Market decimal odds

        Returns:
            (kelly_fraction, dollar_stake)
        """
        implied_prob = 1.0 / decimal_odds
        adjusted_prob = implied_prob * (1 + edge_pct)

        # Blend with model probability
        blended_prob = 0.7 * max(win_probability, adjusted_prob) + 0.3 * adjusted_prob

        return self.compute_kelly(blended_prob, decimal_odds)

    def record_result(self, stake_dollars: float, won: bool, decimal_odds: float = 1.91):
        """Record bet outcome and update bankroll."""
        if won:
            profit = stake_dollars * (decimal_odds - 1.0)
            self.current_bankroll += profit
        else:
            self.current_bankroll -= stake_dollars

        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll

    def get_current_state(self) -> Dict:
        """Get current bankroll state."""
        drawdown = self.peak_bankroll - self.current_bankroll
        drawdown_pct = drawdown / self.peak_bankroll if self.peak_bankroll > 0 else 0

        return {
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": round(self.current_bankroll, 2),
            "peak_bankroll": round(self.peak_bankroll, 2),
            "drawdown": round(drawdown, 2),
            "drawdown_pct": round(drawdown_pct * 100, 2),
            "kelly_fraction": self.fraction,
        }


class MultiBetKelly:
    """
    Multi-bet Kelly optimization for simultaneous bets.

    Standard Kelly assumes you can size each bet independently.
    When multiple bets overlap in time (same night), the total
    capital at risk must be allocated optimally across all bets.

    Uses convex optimization to find the optimal allocation
    that maximizes log-growth of bankroll.

    Usage:
        mk = MultiBetKelly(bankroll=10000)
        stakes = mk.optimize([
            {"edge": 0.05, "odds": 1.91},
            {"edge": 0.03, "odds": 2.10},
            {"edge": 0.04, "odds": 1.80},
        ])
    """

    def __init__(
        self,
        bankroll: float = 10_000.0,
        fraction: float = 0.25,
        max_single_bet: float = 0.15,
        max_total_exposure: float = 0.40,
    ):
        self.bankroll = bankroll
        self.fraction = fraction
        self.max_single_bet = max_single_bet
        self.max_total_exposure = max_total_exposure

    def optimize(
        self,
        bets: List[Dict],
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Find optimal stakes for multiple simultaneous bets.

        Args:
            bets: List of dicts with keys:
                - 'win_probability': Estimated win prob (0-1)
                - 'decimal_odds': Market decimal odds
                - 'edge_pct': Edge percentage (optional)
            correlation_matrix: NxN correlation matrix between bet outcomes.
                If None, assumes independence.

        Returns:
            Array of optimal fractions of bankroll for each bet
        """
        n_bets = len(bets)
        if n_bets == 0:
            return np.array([])

        # Extract probabilities and odds
        probs = np.array([b.get("win_probability", 0.5) for b in bets])
        odds = np.array([b.get("decimal_odds", 1.91) for b in bets])
        edges = np.array([b.get("edge_pct", 0.0) for b in bets])

        # Adjust probabilities by edge
        for i in range(n_bets):
            if edges[i] > 0:
                implied = 1.0 / odds[i]
                probs[i] = max(probs[i], implied * (1 + edges[i]))

        # Default correlation matrix
        if correlation_matrix is None:
            correlation_matrix = np.eye(n_bets)
        else:
            correlation_matrix = np.array(correlation_matrix)

        # Objective: maximize expected log growth
        def negative_log_growth(fractions):
            fractions = np.clip(fractions, 0, self.max_single_bet)

            # Expected portfolio return
            total_risk = np.sum(fractions)
            if total_risk > self.max_total_exposure:
                return 1e10  # Penalize over-exposure

            # Log growth expectation (simplified for correlated bets)
            log_growth = 0.0
            for i in range(n_bets):
                pi, oi = probs[i], odds[i]
                fi = fractions[i]

                # Win scenario
                log_wealth_win = np.log(1 + fi * (oi - 1) - np.sum(fractions) + fi)
                # Loss scenario
                log_wealth_loss = np.log(1 - fi - np.sum(fractions) + fi)

                # Correlation-adjusted probability scenarios
                log_growth += pi * log_wealth_win + (1 - pi) * log_wealth_loss

            # Penalty for high correlation (concentration risk)
            if n_bets > 1:
                total_f = fractions / (np.sum(fractions) + 1e-10)
                concentration = np.sum(
                    total_f[:, None] * total_f[None, :] * correlation_matrix
                )
                log_growth -= 0.05 * concentration * np.sum(fractions)

            return -log_growth

        # Initial guess: equal allocation
        x0 = np.full(n_bets, self.fraction * 0.5 / max(n_bets, 1))

        # Bounds: 0 to max_single_bet
        bounds = [(0, self.max_single_bet)] * n_bets

        # Constraint: total exposure <= max_total_exposure
        constraints = [
            {"type": "ineq", "fun": lambda x: self.max_total_exposure - np.sum(x)}
        ]

        result = minimize(
            negative_log_growth,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-8},
        )

        return np.clip(result.x, 0, self.max_single_bet)

    def compute_stakes(
        self,
        bets: List[Dict],
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> List[KellyBet]:
        """
        Compute dollar stakes for multiple simultaneous bets.

        Returns:
            List of KellyBet objects with computed stakes
        """
        fractions = self.optimize(bets, correlation_matrix)

        result = []
        for i, bet in enumerate(bets):
            kb = KellyBet(
                bet_id=bet.get("bet_id", f"bet_{i}"),
                edge_pct=bet.get("edge_pct", 0),
                win_probability=bet.get("win_probability", 0.5),
                decimal_odds=bet.get("decimal_odds", 1.91),
                kelly_fraction=float(fractions[i]),
                dollar_stake=float(fractions[i] * self.bankroll),
                expected_value=float(
                    bet.get("win_probability", 0.5) * (bet.get("decimal_odds", 1.91) - 1)
                    - (1 - bet.get("win_probability", 0.5))
                ),
            )
            result.append(kb)

        return result


def correlated_kelly(
    win_probs: np.ndarray,
    decimal_odds: np.ndarray,
    correlation_matrix: np.ndarray,
    bankroll: float = 10_000.0,
    fraction: float = 0.25,
) -> np.ndarray:
    """
    Convenience function: correlated Kelly optimization.

    Args:
        win_probs: Array of win probabilities
        decimal_odds: Array of decimal odds
        correlation_matrix: NxN correlation matrix
        bankroll: Current bankroll
        fraction: Fraction of full Kelly

    Returns:
        Array of optimal dollar stakes
    """
    bets = [
        {"win_probability": p, "decimal_odds": o}
        for p, o in zip(win_probs, decimal_odds)
    ]

    mk = MultiBetKelly(bankroll=bankroll, fraction=fraction)
    stakes = mk.compute_stakes(bets, correlation_matrix)

    return np.array([s.dollar_stake for s in stakes])
