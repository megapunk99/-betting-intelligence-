"""
Bankroll management: Kelly criterion staking, drawdown protection,
and portfolio-level risk management.

Professional betting is 50% model, 50% staking discipline.
This module implements institutional-grade bankroll management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from betting_intel.config import INITIAL_BANKROLL, UNIT_SIZE, MAX_KELLY_FRACTION


@dataclass
class BetStake:
    """A staking decision for a single bet."""

    game_id: str
    strategy: str
    edge_pct: float
    probability: float
    kelly_fraction: float
    stake_units: float
    stake_dollars: float
    implied_odds: float
    expected_value: float


@dataclass
class BankrollSnapshot:
    """Snapshot of bankroll state at a point in time."""

    date: str
    bankroll: float
    total_bets: int
    winning_bets: int
    peak_bankroll: float
    current_drawdown: float
    current_kelly_multiplier: float


class BankrollManager:
    """
    Manages bankroll using fractional Kelly criterion.
    Includes drawdown protection and dynamic stake sizing.

    Kelly formula: f* = (bp - q) / b
    where:
        f* = fraction of bankroll to bet
        b = odds received on the bet (decimal odds - 1)
        p = probability of winning
        q = probability of losing (1 - p)
    """

    def __init__(
        self,
        initial_bankroll: float = INITIAL_BANKROLL,
        base_kelly_fraction: float = UNIT_SIZE,
        max_kelly_fraction: float = MAX_KELLY_FRACTION,
        drawdown_reduction: bool = True,
        max_consecutive_losses: int = 5,
    ):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.peak_bankroll = initial_bankroll
        self.base_kelly_fraction = base_kelly_fraction
        self.max_kelly_fraction = max_kelly_fraction
        self.drawdown_reduction = drawdown_reduction
        self.max_consecutive_losses = max_consecutive_losses

        self.bets_placed: List[BetStake] = []
        self.history: List[BankrollSnapshot] = []
        self.consecutive_losses = 0
        self.total_bets = 0
        self.winning_bets = 0

    def compute_kelly_stake(
        self,
        win_probability: float,
        decimal_odds: float = 1.91,  # -110 US odds
        edge_pct: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Compute optimal Kelly stake.

        Args:
            win_probability: Our estimated win probability (0-1)
            decimal_odds: Market decimal odds
            edge_pct: Our edge over the market

        Returns:
            (kelly_fraction, stake_units) where kelly_fraction is % of bankroll
        """
        # Implied probability from market odds
        implied_prob = 1.0 / decimal_odds

        # If we have edge_pct, use it to adjust win_probability
        if edge_pct > 0 and win_probability <= implied_prob:
            win_probability = implied_prob * (1 + edge_pct)

        # b in Kelly formula = decimal odds - 1
        b = decimal_odds - 1.0
        p = win_probability
        q = 1.0 - p

        # Full Kelly
        if b > 0:
            full_kelly = (b * p - q) / b
        else:
            full_kelly = 0.0

        # Apply fraction (conservative)
        fraction = full_kelly * self.base_kelly_fraction

        # Cap at max
        fraction = min(fraction, self.max_kelly_fraction)

        # Apply drawdown reduction if enabled
        if self.drawdown_reduction:
            drawdown_factor = self._get_drawdown_factor()
            fraction *= drawdown_factor

        # Apply consecutive loss reduction
        if self.consecutive_losses >= 3:
            fraction *= max(0.1, 1.0 - (self.consecutive_losses - 2) * 0.2)

        # Never bet negative Kelly
        fraction = max(0.0, fraction)

        # Compute dollar stake
        stake_dollars = fraction * self.current_bankroll

        # Round to reasonable values
        fraction = round(fraction, 4)
        stake_dollars = round(stake_dollars, 2)

        return fraction, stake_dollars

    def place_bet(
        self,
        game_id: str,
        strategy: str,
        win_probability: float,
        decimal_odds: float = 1.91,
        edge_pct: float = 0.0,
    ) -> Optional[BetStake]:
        """
        Place a bet with proper staking.

        Returns:
            BetStake if bet is placed, None if no bet (negative edge)
        """
        fraction, stake_dollars = self.compute_kelly_stake(
            win_probability, decimal_odds, edge_pct
        )

        if fraction <= 0 or stake_dollars < 1.0:
            return None

        ev = (win_probability * (decimal_odds - 1)) - ((1 - win_probability) * 1)

        bet = BetStake(
            game_id=game_id,
            strategy=strategy,
            edge_pct=edge_pct,
            probability=win_probability,
            kelly_fraction=fraction,
            stake_units=round(fraction / self.base_kelly_fraction, 2),
            stake_dollars=stake_dollars,
            implied_odds=decimal_odds,
            expected_value=ev,
        )

        self.bets_placed.append(bet)
        return bet

    def record_result(self, bet: BetStake, won: bool):
        """Record the result of a bet and update bankroll."""
        if won:
            profit = bet.stake_dollars * (bet.implied_odds - 1)
            self.current_bankroll += profit
            self.winning_bets += 1
            self.consecutive_losses = 0
        else:
            self.current_bankroll -= bet.stake_dollars
            self.consecutive_losses += 1

        self.total_bets += 1

        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll

    def take_snapshot(self, date: str) -> BankrollSnapshot:
        """Record bankroll state at a given date."""
        snapshot = BankrollSnapshot(
            date=date,
            bankroll=self.current_bankroll,
            total_bets=self.total_bets,
            winning_bets=self.winning_bets,
            peak_bankroll=self.peak_bankroll,
            current_drawdown=self.peak_bankroll - self.current_bankroll,
            current_kelly_multiplier=self._get_drawdown_factor(),
        )
        self.history.append(snapshot)
        return snapshot

    def _get_drawdown_factor(self) -> float:
        """Reduces exposure during drawdowns."""
        if self.peak_bankroll <= 0:
            return 1.0
        drawdown_pct = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll

        if drawdown_pct <= 0.05:
            return 1.0
        elif drawdown_pct <= 0.10:
            return 0.75
        elif drawdown_pct <= 0.20:
            return 0.50
        elif drawdown_pct <= 0.30:
            return 0.25
        else:
            return 0.0  # Stop betting

    def get_metrics(self) -> Dict[str, float]:
        """Get current bankroll performance metrics."""
        total_return = (
            (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll * 100
        )

        return {
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": round(self.current_bankroll, 2),
            "peak_bankroll": round(self.peak_bankroll, 2),
            "total_return_pct": round(total_return, 2),
            "total_bets": self.total_bets,
            "win_rate": self.winning_bets / max(self.total_bets, 1),
            "consecutive_losses": self.consecutive_losses,
            "drawdown": round(self.peak_bankroll - self.current_bankroll, 2),
            "drawdown_pct": round(
                (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll * 100,
                2
            ),
        }

    def format_summary(self) -> str:
        """Format bankroll summary."""
        m = self.get_metrics()
        return (
            f"Bankroll: ${m['current_bankroll']:,.2f} "
            f"(Return: {m['total_return_pct']:+.1f}%) | "
            f"Peak: ${m['peak_bankroll']:,.2f} | "
            f"Drawdown: {m['drawdown_pct']:.1f}% | "
            f"Bets: {m['total_bets']} | "
            f"Win Rate: {m['win_rate']:.1%}"
        )
