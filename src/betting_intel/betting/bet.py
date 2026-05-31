"""
Unified Betting Engine — Phase 1.3 of the Professional Betting Intelligence Platform.

Wires Expected Value Engine results into Kelly staking to produce full
betting recommendations with:

    recommended_stake
    bankroll_percentage
    risk_level

Integrates with:
    - EV Engine (betting/ev.py) for edge and expected value
    - Kelly Calculator (risk/kelly.py) for stake sizing
    - Bankroll Manager (betting/bankroll.py) for drawdown protection

Every betting recommendation must include stake sizing.
"""

import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

from betting_intel.betting.ev import (
    ExpectedValueEngine, EVResult, GameEVResult,
    american_to_decimal, decimal_to_american,
)
from betting_intel.risk.kelly import KellyCalculator


class BetSide(Enum):
    HOME = "home"
    AWAY = "away"
    OVER = "over"
    UNDER = "under"


class MarketType(Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class StakeSizing:
    """Kelly-derived stake sizing for a single bet."""
    kelly_fraction: float = 0.0          # Fraction of full Kelly
    recommended_fraction: float = 0.0     # Fraction of bankroll to risk
    recommended_stake: float = 0.0        # Dollar amount
    bankroll_percentage: float = 0.0       # % of bankroll
    risk_level: str = "skip"
    max_stake: float = 0.0
    is_valid: bool = False


@dataclass
class BetRecommendation:
    """Complete betting recommendation with EV + Kelly sizing."""
    # Identification
    game_id: str = ""
    home_team: str = ""
    away_team: str = ""
    commence_time: str = ""

    # The bet
    bet_side: str = ""          # 'home', 'away', 'over', 'under'
    market_type: str = "moneyline"  # 'moneyline', 'spread', 'total'
    market_line: Optional[float] = None  # e.g. 218.5 for totals, -3.5 for spread

    # Odds
    odds_american: Optional[float] = None
    odds_decimal: Optional[float] = None

    # Model
    model_probability: float = 0.0
    model_name: str = ""

    # EV
    implied_probability: float = 0.0
    edge_percentage: float = 0.0
    expected_value: float = 0.0
    vig_free_probability: Optional[float] = None

    # Kelly Sizing
    stake: StakeSizing = field(default_factory=StakeSizing)

    # Decision
    is_actionable: bool = False
    recommendation: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @property
    def bet_label(self) -> str:
        side_label = self.bet_side.upper()
        if self.market_line is not None:
            line_str = f"{self.market_line:+.1f}" if self.market_type == "spread" else f"O/U {self.market_line:.1f}"
            return f"{self.away_team} @ {self.home_team} | {side_label} {line_str}"
        return f"{self.away_team} @ {self.home_team} | {side_label} ML"

    def summary(self) -> str:
        """Human-readable one-liner."""
        side = self.bet_side.upper()
        if self.odds_american:
            odds_str = f"{self.odds_american:+.0f}"
        elif self.odds_decimal:
            odds_str = f"{self.odds_decimal:.2f}"
        else:
            odds_str = "N/A"
        stake_str = f"${self.stake.recommended_stake:.0f}" if self.stake.is_valid else "no bet"
        return (
            f"{'✅' if self.is_actionable else '⛔'} {self.bet_label} "
            f"| Edge: {self.edge_percentage:.2%} "
            f"| EV: {self.expected_value:.2%} "
            f"| Stake: {stake_str} "
            f"| Risk: {self.stake.risk_level}"
        )


@dataclass
class DailyPortfolio:
    """A set of recommended bets for a single day."""
    date: str
    bets: List[BetRecommendation] = field(default_factory=list)
    total_risk: float = 0.0           # Total $ at risk
    total_bankroll_pct: float = 0.0   # Total % of bankroll at risk
    num_bets: int = 0
    num_actionable: int = 0

    def add(self, bet: BetRecommendation):
        self.bets.append(bet)
        if bet.is_actionable and bet.stake.is_valid:
            self.total_risk += bet.stake.recommended_stake
            self.total_bankroll_pct += bet.stake.bankroll_percentage
            self.num_actionable += 1
        self.num_bets = len(self.bets)

    def sort_by_edge(self) -> List[BetRecommendation]:
        return sorted(self.bets, key=lambda b: b.edge_percentage, reverse=True)

    def sort_by_ev(self) -> List[BetRecommendation]:
        return sorted(self.bets, key=lambda b: b.expected_value, reverse=True)

    def actionable(self) -> List[BetRecommendation]:
        return [b for b in self.bets if b.is_actionable]


# ═══════════════════════════════════════════════════════════════════════════
#  BETTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class BettingEngine:
    """
    Master betting engine that wires EV calculation -> Kelly staking -> recommendation.

    Supports:
    - Full Kelly, Half Kelly, Quarter Kelly (default: Quarter Kelly)
    - Multi-bet portfolio optimization
    - Drawdown protection via BankrollManager
    - Configurable thresholds

    Usage:
        engine = BettingEngine(bankroll=10000, kelly_fraction=0.25)
        bet = engine.create_bet(
            model_probability=0.62,
            odds_american=-110,
            home_team="Spurs",
            away_team="Knicks",
        )
        print(bet.summary())
    """

    def __init__(
        self,
        bankroll: float = 10_000.0,
        kelly_fraction: float = 0.25,      # Quarter Kelly default
        min_edge: float = 0.02,             # 2% minimum edge
        max_single_stake_pct: float = 0.15,  # Max 15% of bankroll per bet
        max_daily_stake_pct: float = 0.40,   # Max 40% of bankroll per day
        drawdown_protection: bool = True,
    ):
        self.ev_engine = ExpectedValueEngine(min_edge_threshold=min_edge)
        self.kelly = KellyCalculator(
            bankroll=bankroll,
            fraction=kelly_fraction,
            max_fraction=max_single_stake_pct,
            min_edge=min_edge,
            drawdown_protection=drawdown_protection,
        )
        self.max_daily_stake_pct = max_daily_stake_pct
        self.daily_portfolios: Dict[str, DailyPortfolio] = {}

    # ═══════════════════════════════════════════════════════════════════
    #  SINGLE BET CREATION
    # ═══════════════════════════════════════════════════════════════════

    def create_bet(
        self,
        model_probability: float,
        odds_american: Optional[float] = None,
        odds_decimal: Optional[float] = None,
        game_id: str = "",
        home_team: str = "",
        away_team: str = "",
        commence_time: str = "",
        bet_side: str = "home",
        market_type: str = "moneyline",
        market_line: Optional[float] = None,
        model_name: str = "",
        opponent_odds_american: Optional[float] = None,
    ) -> BetRecommendation:
        """
        Create a fully calculated betting recommendation.

        Steps:
        1. Calculate expected value from model probability and market odds
        2. If actionable, compute Kelly stake sizing
        3. Package into BetRecommendation
        """
        # Step 1: EV calculation
        ev_result = self.ev_engine.calculate(
            model_probability=model_probability,
            market_odds_american=odds_american,
            market_odds_decimal=odds_decimal,
            game_id=game_id,
            market_type=market_type,
            bet_side=bet_side,
            home_team=home_team,
            away_team=away_team,
            model_name=model_name,
            opponent_odds_american=opponent_odds_american,
        )

        # Determine decimal odds
        if odds_decimal is not None:
            dec_odds = odds_decimal
        elif odds_american is not None:
            dec_odds = american_to_decimal(odds_american)
        else:
            dec_odds = 1.91  # default

        # Determine american odds
        if odds_american is None:
            odds_american = decimal_to_american(dec_odds) if dec_odds != 1.91 else None

        # Step 2: Kelly stake sizing
        if ev_result.is_actionable and ev_result.expected_value > 0:
            kelly_fraction, dollar_stake = self.kelly.compute_kelly(
                win_probability=model_probability,
                decimal_odds=dec_odds,
            )

            risk_level = self._determine_risk_level(kelly_fraction, ev_result.edge_percentage)
            bankroll_pct = (dollar_stake / self.kelly.current_bankroll * 100) if self.kelly.current_bankroll > 0 else 0

            stake = StakeSizing(
                kelly_fraction=self.kelly.fraction,
                recommended_fraction=kelly_fraction,
                recommended_stake=round(dollar_stake, 2),
                bankroll_percentage=round(bankroll_pct, 2),
                risk_level=risk_level,
                max_stake=round(self.kelly.current_bankroll * self.kelly.max_fraction, 2),
                is_valid=dollar_stake > 0,
            )
        else:
            stake = StakeSizing(
                risk_level="skip",
                is_valid=False,
            )

        # Step 3: Package
        bet = BetRecommendation(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence_time,
            bet_side=bet_side,
            market_type=market_type,
            market_line=market_line,
            odds_american=odds_american if odds_american is not None and not (isinstance(odds_american, float) and math.isnan(odds_american)) else None,
            odds_decimal=round(dec_odds, 4),
            model_probability=round(model_probability, 4),
            model_name=model_name,
            implied_probability=round(ev_result.implied_probability, 4),
            edge_percentage=round(ev_result.edge_percentage, 4),
            expected_value=round(ev_result.expected_value, 4),
            vig_free_probability=ev_result.vig_free_probability,
            stake=stake,
            is_actionable=ev_result.is_actionable and stake.is_valid,
            recommendation=ev_result.recommendation,
        )

        return bet

    def create_moneyline_bet(
        self,
        model_home_prob: float,
        home_odds_american: float,
        away_odds_american: float,
        game_id: str = "",
        home_team: str = "",
        away_team: str = "",
        commence_time: str = "",
        model_name: str = "",
        preferred_side: str = "home",
    ) -> BetRecommendation:
        """
        Create a bet recommendation for a moneyline market.
        Automatically picks the side with the best edge.
        """
        home_ev, away_ev = self.ev_engine.calculate_moneyline(
            model_home_prob=model_home_prob,
            home_odds_american=home_odds_american,
            away_odds_american=away_odds_american,
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            model_name=model_name,
        )

        # Pick best actionable side, or the preferred side
        if home_ev.is_actionable and away_ev.is_actionable:
            best = home_ev if home_ev.expected_value >= away_ev.expected_value else away_ev
        elif home_ev.is_actionable:
            best = home_ev
        elif away_ev.is_actionable:
            best = away_ev
        else:
            best = home_ev if preferred_side == "home" else away_ev

        odds = best.market_odds_american
        prob = best.model_probability

        return self.create_bet(
            model_probability=prob,
            odds_american=odds,
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence_time,
            bet_side=best.bet_side,
            market_type="moneyline",
            model_name=model_name,
            opponent_odds_american=away_odds_american if best.bet_side == "home" else home_odds_american,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  DAILY PORTFOLIO
    # ═══════════════════════════════════════════════════════════════════

    def build_daily_portfolio(self, bets: List[BetRecommendation],
                              date: Optional[str] = None) -> DailyPortfolio:
        """
        Build a daily portfolio from a list of bets.
        Checks total exposure limits.
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        portfolio = DailyPortfolio(date=date)

        for bet in bets:
            portfolio.add(bet)

        # Check if total risk exceeds daily limit
        max_daily_risk = self.kelly.current_bankroll * self.max_daily_stake_pct
        if portfolio.total_risk > max_daily_risk:
            # Scale down all bets proportionally
            scale = max_daily_risk / portfolio.total_risk
            for bet in portfolio.bets:
                if bet.stake.is_valid:
                    bet.stake.recommended_stake = round(bet.stake.recommended_stake * scale, 2)
                    bet.stake.bankroll_percentage = round(bet.stake.bankroll_percentage * scale, 2)
                    bet.stake.risk_level = "reduced"
            portfolio.total_risk *= scale
            portfolio.total_bankroll_pct *= scale

        self.daily_portfolios[date] = portfolio
        return portfolio

    def _determine_risk_level(self, kelly_fraction: float, edge_pct: float) -> str:
        """Determine risk level based on Kelly fraction and edge."""
        if kelly_fraction <= 0:
            return "skip"
        if kelly_fraction > 0.10 or edge_pct > 0.10:
            return "extreme"
        if kelly_fraction > 0.05 or edge_pct > 0.05:
            return "high"
        if kelly_fraction > 0.02 or edge_pct > 0.03:
            return "medium"
        return "low"

    # ═══════════════════════════════════════════════════════════════════
    #  BANKROLL MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    @property
    def bankroll(self) -> float:
        return self.kelly.current_bankroll

    def record_result(self, stake_dollars: float, won: bool, decimal_odds: float = 1.91):
        """Record a bet result and update bankroll."""
        self.kelly.record_result(stake_dollars, won, decimal_odds)

    def get_state(self) -> Dict:
        """Get current betting state."""
        state = self.kelly.get_current_state()
        state["max_daily_stake_pct"] = self.max_daily_stake_pct
        state["daily_portfolios"] = len(self.daily_portfolios)
        return state

    def format_portfolio(self, portfolio: DailyPortfolio) -> str:
        """Format a daily portfolio for display."""
        lines = [
            f"{'=' * 60}",
            f"📋 DAILY PORTFOLIO — {portfolio.date}",
            f"{'=' * 60}",
            f"Bankroll: ${self.kelly.current_bankroll:,.2f}",
            f"Actionable bets: {portfolio.num_actionable}/{portfolio.num_bets}",
            f"Total risk: ${portfolio.total_risk:,.2f} ({portfolio.total_bankroll_pct:.1f}% of bankroll)",
        ]

        if portfolio.actionable():
            lines.append(f"\n{'─' * 60}")
            lines.append("RECOMMENDED BETS")
            lines.append(f"{'─' * 60}")
            for bet in sorted(portfolio.actionable(), key=lambda b: b.expected_value, reverse=True):
                lines.append(f"\n{bet.summary()}")

        return "\n".join(lines)
