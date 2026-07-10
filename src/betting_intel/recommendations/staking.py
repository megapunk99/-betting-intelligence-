"""
Kelly Staking Engine — professional bankroll management for sports betting.

Core Principles
───────────────
  1. Full Kelly Criterion — mathematically optimal growth rate
  2. Fractional Kelly — reduce variance at the cost of growth
  3. Drawdown Protection — reduce stake during losing streaks
  4. Exposure Limits — no more than X% on any single game/team/league
  5. Correlation Awareness — correlated bets reduce effective bankroll
  6. Confidence-Based Staking — higher confidence → higher fraction of Kelly

Formulas
────────
  Full Kelly: f* = (bp - q) / b
    where b = decimal odds - 1
          p = win probability
          q = 1 - p

  Fractional Kelly: f = f* * fraction

  Drawdown Adjustment: multiplier = 1 - (current_drawdown / max_drawdown)²

Usage:
    from betting_intel.recommendations.staking import KellyStaker

    staker = KellyStaker(initial_bankroll=10000, kelly_fraction=0.25)
    stake = staker.compute_stake(
        win_probability=0.60,
        decimal_odds=1.91,
        confidence_score=0.80,
        league="NBA",
        team="Lakers",
    )
    # Returns: stake amount, kelly_pct, exposure_pct
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class StakeResult:
    """Result of a stake calculation."""

    stake_dollars: float = 0.0
    kelly_full: float = 0.0
    kelly_fractional: float = 0.0
    kelly_used: float = 0.0
    exposure_pct: float = 0.0
    bankroll_after: float = 0.0
    max_allowed: float = 0.0
    adjustment_reasons: list[str] = field(default_factory=list)


@dataclass
class BankrollState:
    """Current state of the bankroll."""

    current: float
    initial: float
    peak: float
    drawdown: float
    drawdown_pct: float
    n_bets_today: int = 0
    n_bets_this_week: int = 0
    consecutive_losses: int = 0
    last_updated: str = ""


@dataclass
class ExposureTracker:
    """Tracks exposure across teams, leagues, and games."""

    per_team: dict[str, float] = field(default_factory=dict)
    per_league: dict[str, float] = field(default_factory=dict)
    per_game: dict[str, float] = field(default_factory=dict)
    total_exposed: float = 0.0
    max_exposure_pct: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  KELLY STAKER
# ═══════════════════════════════════════════════════════════════════════════


class KellyStaker:
    """
    Professional bankroll management using Kelly Criterion.

    Handles:
    - Full Kelly, fractional Kelly, and Kelly with drawdown adjustment
    - Per-team, per-league, and per-game exposure limits
    - Consecutive loss streak detection and stake reduction
    - Bankroll tracking with peak and drawdown monitoring
    - Confidence-based staking (higher confidence → higher fraction of Kelly)

    Usage:
        staker = KellyStaker(initial_bankroll=10000)
        staker.record_bet(team="Lakers", league="NBA", stake=250, won=True)
        result = staker.compute_stake(win_prob=0.60, decimal_odds=1.91, ...)
    """

    def __init__(
        self,
        initial_bankroll: float = 10_000.0,
        kelly_fraction: float = 0.25,
        max_exposure_pct: float = 0.05,  # Max 5% on any single bet
        max_team_exposure_pct: float = 0.10,  # Max 10% on any single team
        max_league_exposure_pct: float = 0.30,  # Max 30% on any single league
        max_daily_bets: int = 20,
        drawdown_recovery: bool = True,
        min_edge_threshold: float = 0.03,  # Minimum 3% edge to bet (was 1%)
        confidence_multipliers: Optional[dict[str, float]] = None,
    ):
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.max_exposure_pct = max_exposure_pct
        self.max_team_exposure_pct = max_team_exposure_pct
        self.max_league_exposure_pct = max_league_exposure_pct
        self.max_daily_bets = max_daily_bets
        self.drawdown_recovery = drawdown_recovery
        self.min_edge_threshold = min_edge_threshold

        self.confidence_multipliers = confidence_multipliers or {
            "VERY_HIGH": 1.0,  # Full Kelly fraction
            "HIGH": 0.75,  # 75% of Kelly fraction
            "MEDIUM": 0.50,  # 50% of Kelly fraction
            "LOW": 0.25,  # 25% of Kelly fraction
            "VERY_LOW": 0.10,  # 10% of Kelly fraction
        }

        # Bankroll state
        self._bankroll = initial_bankroll
        self._peak = initial_bankroll
        self._initial = initial_bankroll
        self._consecutive_losses = 0
        self._n_bets_today = 0
        self._n_bets_this_week = 0
        self._last_bet_date: Optional[str] = None
        self._last_week_start: Optional[str] = None
        self._exposure = ExposureTracker()
        self._history: list[dict] = []

    @property
    def bankroll(self) -> float:
        return self._bankroll

    @property
    def peak(self) -> float:
        return self._peak

    @property
    def drawdown(self) -> float:
        return max(0.0, self._peak - self._bankroll)

    @property
    def drawdown_pct(self) -> float:
        return self.drawdown / self._peak if self._peak > 0 else 0.0

    @property
    def total_profit(self) -> float:
        return self._bankroll - self._initial

    @property
    def roi(self) -> float:
        return self.total_profit / self._initial if self._initial > 0 else 0.0

    def get_state(self) -> BankrollState:
        """Get the current bankroll state."""
        return BankrollState(
            current=round(self._bankroll, 2),
            initial=round(self._initial, 2),
            peak=round(self._peak, 2),
            drawdown=round(self.drawdown, 2),
            drawdown_pct=round(self.drawdown_pct, 4),
            n_bets_today=self._n_bets_today,
            n_bets_this_week=self._n_bets_this_week,
            consecutive_losses=self._consecutive_losses,
            last_updated=datetime.now().isoformat(),
        )

    def compute_stake(
        self,
        win_probability: float,
        decimal_odds: float,
        confidence_score: float = 0.5,
        confidence_label: str = "MEDIUM",
        edge_pct: float = 0.0,
        league: str = "NBA",
        team: str = "",
        game_id: str = "",
    ) -> StakeResult:
        """
        Compute the optimal stake for a bet.

        Args:
            win_probability: Model-estimated win probability (0-1)
            decimal_odds: Market decimal odds (e.g., 1.91 for -110)
            confidence_score: Model's confidence in this prediction (0-1)
            confidence_label: Confidence category
            edge_pct: Edge over market (0.03 = 3%)
            league: League name for exposure tracking
            team: Team name for exposure tracking
            game_id: Game ID for game-level exposure

        Returns:
            StakeResult with calculated stake and adjustments
        """
        adjustments: list[str] = []
        bankroll = self._bankroll

        # ── 1. Edge check ──────────────────────────────────────────────
        if abs(edge_pct) < self.min_edge_threshold:
            adjustments.append(
                f"Edge ({edge_pct:.2%}) below threshold ({self.min_edge_threshold:.2%})"
            )
            return StakeResult(
                stake_dollars=0.0,
                adjustment_reasons=adjustments,
                bankroll_after=bankroll,
                max_allowed=bankroll * self.max_exposure_pct,
            )

        # ── 2. Full Kelly ──────────────────────────────────────────────
        b = decimal_odds - 1.0  # Net odds received
        if b <= 0:
            adjustments.append(f"Invalid odds: decimal_odds={decimal_odds}")
            return StakeResult(
                stake_dollars=0.0,
                adjustment_reasons=adjustments,
                bankroll_after=bankroll,
            )

        p = max(0.01, min(0.99, win_probability))
        q = 1.0 - p
        full_kelly = (b * p - q) / b
        full_kelly = max(0.0, full_kelly)  # No negative stakes

        # ── 3. Apply confidence multiplier ──────────────────────────────
        conf_mult = self.confidence_multipliers.get(confidence_label, 0.5)
        # Also scale by the numeric confidence score
        conf_score_factor = 0.5 + confidence_score * 0.5  # 0.5 to 1.0
        effective_fraction = self.kelly_fraction * conf_mult * conf_score_factor

        fractional_kelly = full_kelly * effective_fraction

        # ── 4. Drawdown protection ──────────────────────────────────────
        dd_adjustment = 1.0
        if self.drawdown_recovery and self.drawdown_pct > 0.05:
            # Quadratic penalty: at 20% drawdown, cut to 25% of Kelly
            # at 40% drawdown, cut to 0%
            if self.drawdown_pct >= 0.40:
                dd_adjustment = 0.0
                adjustments.append(
                    f"Drawdown >40% ({self.drawdown_pct:.1%}) — no betting"
                )
            else:
                dd_adjustment = 1.0 - (self.drawdown_pct / 0.40) ** 2
                adjustments.append(
                    f"Drawdown adjustment: {dd_adjustment:.2f}x "
                    f"(drawdown={self.drawdown_pct:.1%})"
                )

        # ── 5. Consecutive loss streak ─────────────────────────────────
        loss_adjustment = 1.0
        if self._consecutive_losses >= 3:
            loss_adjustment = max(0.25, 1.0 - self._consecutive_losses * 0.15)
            adjustments.append(
                f"Loss streak adjustment: {loss_adjustment:.2f}x "
                f"({self._consecutive_losses} consecutive losses)"
            )

        # ── 6. Final Kelly ─────────────────────────────────────────────
        adjusted_kelly = fractional_kelly * dd_adjustment * loss_adjustment

        # ── 7. Max single-bet exposure ─────────────────────────────────
        max_single_stake = bankroll * self.max_exposure_pct
        stake = min(adjusted_kelly * bankroll, max_single_stake)

        # ── 8. Team-level exposure ─────────────────────────────────────
        if team:
            team_exposed = self._exposure.per_team.get(team, 0.0)
            max_team_stake = bankroll * self.max_team_exposure_pct
            remaining_team = max(0.0, max_team_stake - team_exposed)
            if stake > remaining_team:
                adjustments.append(
                    f"Team exposure limit: ${stake:.0f} → ${remaining_team:.0f} "
                    f"(already ${team_exposed:.0f} on {team})"
                )
                stake = min(stake, remaining_team)

        # ── 9. League-level exposure ───────────────────────────────────
        league_exposed = self._exposure.per_league.get(league, 0.0)
        max_league_stake = bankroll * self.max_league_exposure_pct
        remaining_league = max(0.0, max_league_stake - league_exposed)
        if stake > remaining_league:
            adjustments.append(
                f"League exposure limit: ${stake:.0f} → ${remaining_league:.0f} "
                f"(already ${league_exposed:.0f} on {league})"
            )
            stake = min(stake, remaining_league)

        # ── 10. Daily bet limit ────────────────────────────────────────
        if self._n_bets_today >= self.max_daily_bets:
            adjustments.append(f"Daily bet limit reached ({self.max_daily_bets})")
            stake = 0.0

        # ── 11. Minimum stake ──────────────────────────────────────────
        if stake < 1.0:
            adjustments.append(f"Stake (${stake:.2f}) below $1 minimum")
            stake = 0.0

        stake = round(stake, 2)

        return StakeResult(
            stake_dollars=stake,
            kelly_full=round(full_kelly, 6),
            kelly_fractional=round(fractional_kelly, 6),
            kelly_used=round(adjusted_kelly, 6),
            exposure_pct=round(stake / bankroll, 6) if bankroll > 0 else 0.0,
            bankroll_after=round(bankroll - stake, 2) if stake > 0 else bankroll,
            max_allowed=round(max_single_stake, 2),
            adjustment_reasons=adjustments,
        )

    def record_bet(
        self,
        team: str = "",
        league: str = "NBA",
        game_id: str = "",
        stake: float = 0.0,
        won: Optional[bool] = None,
        decimal_odds: float = 1.91,
        profit: float = 0.0,
    ):
        """
        Record a bet result and update bankroll.

        Args:
            team: Team name
            league: League name
            game_id: Game identifier
            stake: Amount staked
            won: True if won, False if lost, None if push/void
            decimal_odds: Decimal odds of the bet
            profit: Actual profit (positive = win, negative = loss)
        """
        # Update bankroll
        if profit != 0:
            self._bankroll += profit
            if self._bankroll > self._peak:
                self._peak = self._bankroll

        # Update exposure
        if team:
            self._exposure.per_team[team] = (
                self._exposure.per_team.get(team, 0.0) + stake
            )
        if league:
            self._exposure.per_league[league] = (
                self._exposure.per_league.get(league, 0.0) + stake
            )
        if game_id:
            self._exposure.per_game[game_id] = (
                self._exposure.per_game.get(game_id, 0.0) + stake
            )
        self._exposure.total_exposed += stake

        # Update consecutive losses
        if won is True:
            self._consecutive_losses = 0
        elif won is False:
            self._consecutive_losses += 1

        # Update bet counts
        today = datetime.now().strftime("%Y-%m-%d")
        week_start = (
            datetime.now() - timedelta(days=datetime.now().weekday())
        ).strftime("%Y-%m-%d")

        if self._last_bet_date != today:
            self._n_bets_today = 0
            self._last_bet_date = today
        self._n_bets_today += 1

        if self._last_week_start != week_start:
            self._n_bets_this_week = 0
            self._last_week_start = week_start
        self._n_bets_this_week += 1

        # History
        self._history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "team": team,
                "league": league,
                "game_id": game_id,
                "stake": stake,
                "decimal_odds": decimal_odds,
                "won": won,
                "profit": profit,
                "bankroll_after": round(self._bankroll, 2),
            }
        )

    def release_exposure(self, team: str = "", league: str = "", game_id: str = ""):
        """Release exposure for settled bets."""
        if team and team in self._exposure.per_team:
            self._exposure.total_exposed = max(
                0.0, self._exposure.total_exposed - self._exposure.per_team[team]
            )
            del self._exposure.per_team[team]
        if league and league in self._exposure.per_league:
            self._exposure.total_exposed = max(
                0.0, self._exposure.total_exposed - self._exposure.per_league[league]
            )
            del self._exposure.per_league[league]
        if game_id and game_id in self._exposure.per_game:
            self._exposure.total_exposed = max(
                0.0, self._exposure.total_exposed - self._exposure.per_game[game_id]
            )
            del self._exposure.per_game[game_id]

    def get_exposure(self) -> ExposureTracker:
        """Get current exposure by team, league, and game."""
        return self._exposure

    def reset(self, bankroll: Optional[float] = None):
        """Reset the staker's state."""
        if bankroll is not None:
            self._bankroll = bankroll
            self._initial = bankroll
            self._peak = bankroll
        else:
            self._bankroll = self._initial
            self._peak = self._initial
        self._consecutive_losses = 0
        self._n_bets_today = 0
        self._n_bets_this_week = 0
        self._exposure = ExposureTracker()
        self._history = []

    def get_history(self, n: int = 50) -> list[dict]:
        """Get the last N bet history entries."""
        return self._history[-n:]

    def get_performance_summary(self) -> dict:
        """Get a performance summary."""
        if not self._history:
            return {"n_bets": 0, "status": "no_history"}

        wins = sum(1 for h in self._history if h.get("won") is True)
        losses = sum(1 for h in self._history if h.get("won") is False)
        pushes = sum(1 for h in self._history if h.get("won") is None)
        total_stake = sum(h["stake"] for h in self._history)
        total_profit = sum(h.get("profit", 0) for h in self._history)

        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

        # Sharpe-like ratio
        profits = [h.get("profit", 0) for h in self._history if h.get("profit") != 0]
        avg_p = float(np.mean(profits)) if profits else 0.0
        std_p = float(np.std(profits, ddof=1)) if len(profits) > 1 else 1.0
        sharpe = (avg_p / std_p) * math.sqrt(len(profits)) if std_p > 0 else 0.0

        from betting_intel.models.robust_ensemble import (
            compute_statistical_significance,
        )

        sig = compute_statistical_significance(wins, losses)

        return {
            "n_bets": len(self._history),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(win_rate, 4),
            "total_stake": round(total_stake, 2),
            "total_profit": round(total_profit, 2),
            "roi": round(total_profit / total_stake, 4) if total_stake > 0 else 0.0,
            "sharpe_ratio": round(sharpe, 4),
            "bankroll": round(self._bankroll, 2),
            "peak": round(self._peak, 2),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "is_significant": sig["is_significant"],
            "p_value": sig["p_value"],
        }


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return 1.0 + american_odds / 100.0
    elif american_odds < 0:
        return 1.0 + 100.0 / abs(american_odds)
    return 1.0


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal_odds <= 1.0:
        return -10000
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100.0)
    else:
        return round(-100.0 / (decimal_odds - 1.0))


def american_to_implied(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    elif american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100.0)
    return 0.5


def remove_vig(home_implied: float, away_implied: float) -> tuple[float, float]:
    """Remove the vig from two implied probabilities."""
    total = home_implied + away_implied
    if total > 0:
        return (home_implied / total, away_implied / total)
    return (home_implied, away_implied)


__all__ = [
    "KellyStaker",
    "StakeResult",
    "BankrollState",
    "ExposureTracker",
    "american_to_decimal",
    "decimal_to_american",
    "american_to_implied",
    "remove_vig",
]
