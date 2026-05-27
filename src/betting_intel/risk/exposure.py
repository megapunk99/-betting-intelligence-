"""
Portfolio exposure management: track and limit risk across all active bets.

Professional betting requires strict exposure control:
- Max exposure per game (both sides)
- Max exposure per league
- Max exposure per bet type (O/U, spreads, moneylines)
- Max total exposure across all active bets
- Correlation-aware exposure limits
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date


@dataclass
class PositionLimit:
    """Defines a position limit constraint."""

    limit_type: str  # 'game', 'league', 'bet_type', 'total', 'team', 'correlation'
    entity: str  # e.g., 'NBA', 'BOS_vs_LAL', 'total_points'
    max_exposure: float  # In dollars
    max_stake_pct: float  # % of bankroll
    current_exposure: float = 0.0
    n_bets: int = 0


@dataclass
class ActiveBet:
    """A currently active (unresolved) bet."""

    bet_id: str
    game_id: str
    matchup: str
    league: str
    bet_type: str  # 'moneyline', 'spread', 'total', 'player_prop'
    side: str  # 'OVER', 'UNDER', 'Home', 'Away', 'Player Name'
    stake_dollars: float
    decimal_odds: float
    edge_pct: float
    win_probability: float
    placed_at: datetime = field(default_factory=datetime.now)
    team: Optional[str] = None
    player: Optional[str] = None


@dataclass
class ExposureReport:
    """Summary of current exposure."""

    total_exposure: float = 0.0
    n_active_bets: int = 0
    bankroll_pct: float = 0.0
    by_league: Dict[str, float] = field(default_factory=dict)
    by_game: Dict[str, float] = field(default_factory=dict)
    by_bet_type: Dict[str, float] = field(default_factory=dict)
    largest_bet: float = 0.0
    violations: List[str] = field(default_factory=list)


class ExposureManager:
    """
    Manages portfolio exposure across all active bets.

    Enforces position limits and provides real-time exposure monitoring.

    Usage:
        manager = ExposureManager(bankroll=10000)
        manager.set_limit('total', max_exposure=4000)
        manager.set_limit('league', 'NBA', max_exposure=3000)
        manager.add_bet(ActiveBet(...))
        report = manager.get_report()
    """

    def __init__(
        self,
        bankroll: float = 10_000.0,
        default_max_exposure_pct: float = 0.35,
        default_max_per_game_pct: float = 0.15,
        default_max_per_league_pct: float = 0.25,
    ):
        self.bankroll = bankroll
        self.default_max_exposure_pct = default_max_exposure_pct
        self.default_max_per_game_pct = default_max_per_game_pct
        self.default_max_per_league_pct = default_max_per_league_pct

        self.active_bets: List[ActiveBet] = []
        self.settled_bets: List[ActiveBet] = []
        self.limits: Dict[str, PositionLimit] = {}

        # Set default limits
        self._set_default_limits()

    def _set_default_limits(self):
        """Set default position limits."""
        self.limits["total"] = PositionLimit(
            limit_type="total",
            entity="all",
            max_exposure=self.bankroll * self.default_max_exposure_pct,
            max_stake_pct=self.default_max_exposure_pct,
        )

    def set_limit(
        self,
        limit_type: str,
        entity: str = "all",
        max_exposure: Optional[float] = None,
        max_stake_pct: Optional[float] = None,
    ):
        """Set a position limit."""
        key = f"{limit_type}:{entity}"
        if key in self.limits:
            limit = self.limits[key]
            if max_exposure is not None:
                limit.max_exposure = max_exposure
            if max_stake_pct is not None:
                limit.max_stake_pct = max_stake_pct
        else:
            pct = max_stake_pct or (
                self.default_max_per_game_pct
                if limit_type == "game"
                else self.default_max_per_league_pct
                if limit_type == "league"
                else 0.35
            )
            self.limits[key] = PositionLimit(
                limit_type=limit_type,
                entity=entity,
                max_exposure=max_exposure or (self.bankroll * pct),
                max_stake_pct=pct,
            )

    def add_bet(self, bet: ActiveBet) -> Tuple[bool, Optional[str]]:
        """
        Add a bet if it doesn't violate any limits.

        Returns:
            (accepted: bool, reason: Optional[str])
        """
        # Check total exposure
        if not self._check_limit("total", "all", bet.stake_dollars):
            return (False, "Total exposure limit exceeded")

        # Check league exposure
        league_key = f"league:{bet.league}"
        if league_key not in self.limits:
            self.set_limit("league", bet.league)
        if not self._check_limit("league", bet.league, bet.stake_dollars):
            return (False, f"League exposure limit exceeded for {bet.league}")

        # Check game exposure
        game_key = f"game:{bet.game_id}"
        if game_key not in self.limits:
            self.set_limit("game", bet.game_id)
        if not self._check_limit("game", bet.game_id, bet.stake_dollars):
            return (False, f"Game exposure limit exceeded for {bet.matchup}")

        # Check bet type exposure
        type_key = f"bet_type:{bet.bet_type}"
        if type_key not in self.limits:
            self.set_limit("bet_type", bet.bet_type)

        # Add bet
        self.active_bets.append(bet)
        self._update_limit("total", "all", bet.stake_dollars)
        self._update_limit("league", bet.league, bet.stake_dollars)
        self._update_limit("game", bet.game_id, bet.stake_dollars)

        return (True, None)

    def remove_bet(self, bet_id: str):
        """Remove a bet (when settled or cancelled)."""
        for i, bet in enumerate(self.active_bets):
            if bet.bet_id == bet_id:
                self.settled_bets.append(bet)
                self.active_bets.pop(i)
                self._remove_exposure(bet)
                return

    def _check_limit(self, limit_type: str, entity: str, additional_stake: float) -> bool:
        """Check if adding a bet would violate a limit."""
        key = f"{limit_type}:{entity}"
        if key not in self.limits:
            return True

        limit = self.limits[key]
        new_exposure = limit.current_exposure + additional_stake
        return new_exposure <= limit.max_exposure

    def _update_limit(self, limit_type: str, entity: str, stake: float):
        """Update limit tracking with new stake."""
        key = f"{limit_type}:{entity}"
        if key in self.limits:
            self.limits[key].current_exposure += stake
            self.limits[key].n_bets += 1

    def _remove_exposure(self, bet: ActiveBet):
        """Remove exposure from limits when a bet is settled."""
        for key in [
            "total:all",
            f"league:{bet.league}",
            f"game:{bet.game_id}",
        ]:
            if key in self.limits:
                self.limits[key].current_exposure -= bet.stake_dollars
                self.limits[key].n_bets -= 1

    def get_exposure_by_league(self) -> Dict[str, float]:
        """Get total exposure per league."""
        exposure = {}
        for bet in self.active_bets:
            exposure[bet.league] = exposure.get(bet.league, 0) + bet.stake_dollars
        return exposure

    def get_exposure_by_game(self) -> Dict[str, float]:
        """Get total exposure per game."""
        exposure = {}
        for bet in self.active_bets:
            key = f"{bet.matchup} ({bet.game_id})"
            exposure[key] = exposure.get(key, 0) + bet.stake_dollars
        return exposure

    def get_exposure_by_bet_type(self) -> Dict[str, float]:
        """Get total exposure by bet type."""
        exposure = {}
        for bet in self.active_bets:
            exposure[bet.bet_type] = exposure.get(bet.bet_type, 0) + bet.stake_dollars
        return exposure

    def get_report(self) -> ExposureReport:
        """Generate comprehensive exposure report."""
        report = ExposureReport()

        report.total_exposure = sum(b.stake_dollars for b in self.active_bets)
        report.n_active_bets = len(self.active_bets)
        report.bankroll_pct = report.total_exposure / self.bankroll if self.bankroll > 0 else 0
        report.by_league = self.get_exposure_by_league()
        report.by_game = self.get_exposure_by_game()
        report.by_bet_type = self.get_exposure_by_bet_type()
        report.largest_bet = max(
            (b.stake_dollars for b in self.active_bets), default=0.0
        )

        # Check all limits for violations
        for key, limit in self.limits.items():
            if limit.current_exposure > limit.max_exposure:
                report.violations.append(
                    f"LIMIT VIOLATION: {limit.limit_type}:{limit.entity} - "
                    f"${limit.current_exposure:.0f} / ${limit.max_exposure:.0f} "
                    f"({limit.current_exposure / limit.max_exposure * 100:.0f}%)"
                )

        # Check concentration: if any single bet > 20% of total exposure
        for bet in self.active_bets:
            if report.total_exposure > 0:
                pct = bet.stake_dollars / report.total_exposure
                if pct > 0.20:
                    report.violations.append(
                        f"CONCENTRATION: {bet.bet_id} is {pct:.0%} of total exposure"
                    )

        return report

    def format_report(self) -> str:
        """Format exposure report as readable text."""
        report = self.get_report()

        lines = [
            "=" * 60,
            "  PORTFOLIO EXPOSURE REPORT",
            "=" * 60,
            f"  Active Bets:    {report.n_active_bets}",
            f"  Total Exposure: ${report.total_exposure:,.0f} "
            f"({report.bankroll_pct:.1%} of bankroll)",
            f"  Largest Bet:    ${report.largest_bet:,.0f}",
            "",
            "  -- By League --",
        ]

        for league, exp in sorted(report.by_league.items(), key=lambda x: -x[1]):
            pct = exp / self.bankroll * 100
            lines.append(f"  {league:12s}: ${exp:>8,.0f} ({pct:.1f}%)")

        lines.extend(["", "  -- By Type --"])
        for btype, exp in sorted(report.by_bet_type.items(), key=lambda x: -x[1]):
            lines.append(f"  {btype:15s}: ${exp:>8,.0f}")

        if report.violations:
            lines.extend(["", "  !! VIOLATIONS !!"])
            for v in report.violations:
                lines.append(f"  {v}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_available_capacity(self, limit_type: str = "total", entity: str = "all") -> float:
        """Get remaining capacity for a given limit."""
        key = f"{limit_type}:{entity}"
        if key in self.limits:
            limit = self.limits[key]
            return max(0, limit.max_exposure - limit.current_exposure)
        return 0.0


class BetPortfolio:
    """
    Portfolio of betting strategies with allocation management.

    Allows multiple strategies/bankrolls with different risk profiles.

    Usage:
        portfolio = BetPortfolio(total_bankroll=10000)
        portfolio.add_strategy("aggressive", bankroll_pct=0.3, kelly_fraction=0.5)
        portfolio.add_strategy("conservative", bankroll_pct=0.7, kelly_fraction=0.15)
    """

    def __init__(self, total_bankroll: float = 10_000.0):
        self.total_bankroll = total_bankroll
        self.strategies: Dict[str, Dict] = {}
        self.exposure_manager = ExposureManager(bankroll=total_bankroll)

    def add_strategy(
        self,
        name: str,
        bankroll_pct: float = 1.0,
        kelly_fraction: float = 0.25,
        max_exposure_pct: float = 0.35,
    ):
        """Add a strategy with its own allocation."""
        strategy_bankroll = self.total_bankroll * bankroll_pct
        self.strategies[name] = {
            "name": name,
            "bankroll_pct": bankroll_pct,
            "strategy_bankroll": strategy_bankroll,
            "kelly_fraction": kelly_fraction,
            "max_exposure_pct": max_exposure_pct,
            "max_exposure": strategy_bankroll * max_exposure_pct,
            "current_exposure": 0.0,
            "bets": [],
        }

    def get_strategy_summary(self) -> str:
        """Get summary of all strategies."""
        lines = ["=" * 60, "  PORTFOLIO STRATEGY ALLOCATION", "=" * 60]
        lines.append(f"  Total Bankroll: ${self.total_bankroll:,.0f}")

        for name, strat in self.strategies.items():
            used_pct = (
                strat["current_exposure"] / strat["strategy_bankroll"] * 100
                if strat["strategy_bankroll"] > 0
                else 0
            )
            lines.extend([
                "",
                f"  {name}:",
                f"    Allocation:     {strat['bankroll_pct']:.0%} "
                f"(${strat['strategy_bankroll']:,.0f})",
                f"    Kelly Fraction: {strat['kelly_fraction']:.0%}",
                f"    Max Exposure:   ${strat['max_exposure']:,.0f}",
                f"    Current:        ${strat['current_exposure']:,.0f} ({used_pct:.0f}%)",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)
