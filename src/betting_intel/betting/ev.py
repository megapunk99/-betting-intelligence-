"""
Expected Value Engine — Phase 1.2 of the Professional Betting Intelligence Platform.

Every prediction must calculate betting value. No recommendation should be
generated without EV calculation.

calculate_expected_value()
Inputs:
    model_probability
    market_odds
Outputs:
    expected_value
    edge_percentage
    implied_probability

Only recommend bets when:
    EV > configurable_threshold
"""

from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import math


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

class BetSide(Enum):
    HOME = "home"
    AWAY = "away"
    OVER = "over"
    UNDER = "under"


class MarketType(Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"


@dataclass
class EVResult:
    """Complete expected value calculation for a single bet."""
    # Inputs
    game_id: str = ""
    market_type: str = "moneyline"
    bet_side: str = "home"
    model_probability: float = 0.0
    market_odds_american: Optional[float] = None
    market_odds_decimal: Optional[float] = None

    # Computed
    implied_probability: float = 0.0
    edge_percentage: float = 0.0
    expected_value: float = 0.0
    vig_free_probability: Optional[float] = None

    # Decision
    is_actionable: bool = False
    recommendation: str = ""

    # Metadata
    home_team: str = ""
    away_team: str = ""
    model_name: str = ""


@dataclass
class GameEVResult:
    """EV results for all sides/markets of a single game."""
    game_id: str
    home_team: str
    away_team: str
    commence_time: str

    # Per-side results
    home_ml: Optional[EVResult] = None
    away_ml: Optional[EVResult] = None
    over_total: Optional[EVResult] = None
    under_total: Optional[EVResult] = None
    home_spread: Optional[EVResult] = None
    away_spread: Optional[EVResult] = None

    # Best bets
    best_edge: Optional[EVResult] = None
    num_actionable: int = 0


# ═══════════════════════════════════════════════════════════════════════════
#  CONVERSION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds == 0:
        return 1.0
    if american_odds > 0:
        return 1.0 + (american_odds / 100.0)
    else:
        return 1.0 + (100.0 / abs(american_odds))


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal_odds <= 1.0:
        return float("nan")
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    else:
        return -100.0 / (decimal_odds - 1.0)


def american_to_implied_prob(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if american_odds == 0:
        return 1.0
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def compute_vig_free_prob(home_odds_american: float, away_odds_american: float) -> Tuple[float, float]:
    """
    Remove the vig from a two-way market to get true probabilities.

    Returns:
        (vig_free_home_prob, vig_free_away_prob)
    """
    home_imp = american_to_implied_prob(home_odds_american)
    away_imp = american_to_implied_prob(away_odds_american)
    total_imp = home_imp + away_imp
    if total_imp <= 0:
        return (0.5, 0.5)
    return (home_imp / total_imp, away_imp / total_imp)


def compute_vig_free_total(over_odds_american: float, under_odds_american: float) -> Tuple[float, float]:
    """Remove vig from over/under market."""
    return compute_vig_free_prob(over_odds_american, under_odds_american)


# ═══════════════════════════════════════════════════════════════════════════
#  EXPECTED VALUE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class ExpectedValueEngine:
    """
    Calculates expected value for betting decisions.

    Core formula:
        EV = (model_probability * potential_profit) - ((1 - model_probability) * stake)

    Edge = model_probability - implied_probability

    Only recommends bets when edge exceeds the configurable threshold.

    Usage:
        ev = ExpectedValueEngine(min_edge_threshold=0.02)
        result = ev.calculate(
            model_probability=0.62,
            market_odds_american=-110,
        )
        if result.is_actionable:
            print(f"Bet! EV={result.expected_value:.2%}, Edge={result.edge_percentage:.2%}")
    """

    def __init__(
        self,
        min_edge_threshold: float = 0.02,        # Minimum edge to recommend (2%)
        min_ev_threshold: float = 0.0,            # Minimum EV to recommend
        min_probability: float = 0.01,            # Minimum model probability
        max_probability: float = 0.99,            # Maximum model probability (safety)
        require_positive_ev: bool = True,         # Only recommend positive EV bets
    ):
        self.min_edge_threshold = min_edge_threshold
        self.min_ev_threshold = min_ev_threshold
        self.min_probability = min_probability
        self.max_probability = max_probability
        self.require_positive_ev = require_positive_ev

    def calculate(
        self,
        model_probability: float,
        market_odds_american: Optional[float] = None,
        market_odds_decimal: Optional[float] = None,
        game_id: str = "",
        market_type: str = "moneyline",
        bet_side: str = "home",
        home_team: str = "",
        away_team: str = "",
        model_name: str = "",
        opponent_odds_american: Optional[float] = None,
    ) -> EVResult:
        """
        Calculate expected value for a bet.

        Args:
            model_probability: Estimated probability (0.0 to 1.0)
            market_odds_american: Market odds in American format (e.g., -110)
            market_odds_decimal: Market odds in decimal format (e.g., 1.91)
            game_id: Optional game identifier
            market_type: 'moneyline', 'spread', 'total'
            bet_side: 'home', 'away', 'over', 'under'
            home_team: Optional home team name
            away_team: Optional away team name
            model_name: Optional model identifier
            opponent_odds_american: For moneyline, the odds of the other side
                                    (used for vig-free calculation)

        Returns:
            EVResult with all computed values
        """
        # Clamp probability to safe range
        model_probability = max(self.min_probability, min(self.max_probability, model_probability))

        # Determine decimal odds from whichever format is provided
        # American odds take precedence over decimal (more specific)
        decimal_odds: Optional[float] = None
        if market_odds_american is not None:
            decimal_odds = american_to_decimal(market_odds_american)
        elif market_odds_decimal is not None:
            decimal_odds = market_odds_decimal
        else:
            return EVResult(
                game_id=game_id,
                market_type=market_type,
                bet_side=bet_side,
                model_probability=model_probability,
                recommendation="ERROR: No odds provided",
            )

        # Compute implied probability from market odds
        if market_odds_american is not None:
            implied_prob = american_to_implied_prob(market_odds_american)
        else:
            implied_prob = decimal_to_implied_prob(decimal_odds)

        # Compute vig-free probability if opponent odds are available
        vig_free_prob = None
        if opponent_odds_american is not None and market_odds_american is not None:
            vf_home, vf_away = compute_vig_free_prob(
                market_odds_american if bet_side in ("home", "over") else opponent_odds_american,
                opponent_odds_american if bet_side in ("home", "over") else market_odds_american,
            )
            vig_free_prob = vf_home if bet_side in ("home", "over") else vf_away

        # Compute edge = model_prob - implied_prob
        edge_pct = model_probability - implied_prob

        # Compute expected value
        # For a $1 bet:
        # EV = (win_prob * profit) - (loss_prob * stake)
        # profit = decimal_odds - 1
        # stake = 1
        potential_profit = decimal_odds - 1.0
        ev = (model_probability * potential_profit) - ((1.0 - model_probability) * 1.0)

        # Determine if actionable
        is_actionable = True
        reasons = []

        if self.require_positive_ev and ev <= self.min_ev_threshold:
            is_actionable = False
            reasons.append(f"EV ({ev:.2%}) below threshold ({self.min_ev_threshold:.2%})")

        if edge_pct < self.min_edge_threshold:
            is_actionable = False
            reasons.append(f"Edge ({edge_pct:.2%}) below threshold ({self.min_edge_threshold:.2%})")

        if model_probability < self.min_probability or model_probability > self.max_probability:
            is_actionable = False
            reasons.append(f"Probability ({model_probability:.2%}) out of safe range")

        recommendation = "RECOMMEND" if is_actionable else "; ".join(reasons)

        return EVResult(
            game_id=game_id,
            market_type=market_type,
            bet_side=bet_side,
            model_probability=round(model_probability, 4),
            market_odds_american=market_odds_american,
            market_odds_decimal=round(decimal_odds, 4),
            implied_probability=round(implied_prob, 4),
            edge_percentage=round(edge_pct, 4),
            expected_value=round(ev, 4),
            vig_free_probability=round(vig_free_prob, 4) if vig_free_prob is not None else None,
            is_actionable=is_actionable,
            recommendation=recommendation,
            home_team=home_team,
            away_team=away_team,
            model_name=model_name,
        )

    def calculate_moneyline(
        self,
        model_home_prob: float,
        home_odds_american: float,
        away_odds_american: float,
        game_id: str = "",
        home_team: str = "",
        away_team: str = "",
        model_name: str = "",
    ) -> Tuple[EVResult, EVResult]:
        """
        Calculate EV for both sides of a moneyline market.

        Returns:
            (home_ev_result, away_ev_result)
        """
        home_ev = self.calculate(
            model_probability=model_home_prob,
            market_odds_american=home_odds_american,
            game_id=game_id,
            market_type="moneyline",
            bet_side="home",
            home_team=home_team,
            away_team=away_team,
            model_name=model_name,
            opponent_odds_american=away_odds_american,
        )

        away_ev = self.calculate(
            model_probability=1.0 - model_home_prob,
            market_odds_american=away_odds_american,
            game_id=game_id,
            market_type="moneyline",
            bet_side="away",
            home_team=home_team,
            away_team=away_team,
            model_name=model_name,
            opponent_odds_american=home_odds_american,
        )

        return (home_ev, away_ev)

    def calculate_total(
        self,
        model_over_prob: float,
        over_odds_american: float,
        under_odds_american: float,
        game_id: str = "",
        home_team: str = "",
        away_team: str = "",
        model_name: str = "",
    ) -> Tuple[EVResult, EVResult]:
        """
        Calculate EV for both sides of a totals market.

        Returns:
            (over_ev_result, under_ev_result)
        """
        over_ev = self.calculate(
            model_probability=model_over_prob,
            market_odds_american=over_odds_american,
            game_id=game_id,
            market_type="total",
            bet_side="over",
            home_team=home_team,
            away_team=away_team,
            model_name=model_name,
            opponent_odds_american=under_odds_american,
        )

        under_ev = self.calculate(
            model_probability=1.0 - model_over_prob,
            market_odds_american=under_odds_american,
            game_id=game_id,
            market_type="total",
            bet_side="under",
            home_team=home_team,
            away_team=away_team,
            model_name=model_name,
            opponent_odds_american=over_odds_american,
        )

        return (over_ev, under_ev)

    def analyze_game(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        commence_time: str,
        model_home_prob: float,
        model_total_over_prob: Optional[float] = None,
        model_home_cover_prob: Optional[float] = None,
        home_ml_odds: Optional[float] = None,
        away_ml_odds: Optional[float] = None,
        total_over_odds: Optional[float] = None,
        total_under_odds: Optional[float] = None,
        home_spread_odds: Optional[float] = None,
        away_spread_odds: Optional[float] = None,
        home_spread: Optional[float] = None,
        model_name: str = "",
    ) -> GameEVResult:
        """
        Full EV analysis for all markets in a single game.
        Identifies the best betting opportunity.
        """
        result = GameEVResult(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            commence_time=commence_time,
        )

        best_ev = -float("inf")

        # Moneyline analysis
        if home_ml_odds is not None and away_ml_odds is not None:
            home_ev, away_ev = self.calculate_moneyline(
                model_home_prob=model_home_prob,
                home_odds_american=home_ml_odds,
                away_odds_american=away_ml_odds,
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                model_name=model_name,
            )
            result.home_ml = home_ev
            result.away_ml = away_ev

            if home_ev.is_actionable and home_ev.expected_value > best_ev:
                best_ev = home_ev.expected_value
                result.best_edge = home_ev
            if away_ev.is_actionable and away_ev.expected_value > best_ev:
                best_ev = away_ev.expected_value
                result.best_edge = away_ev

        # Totals analysis
        if model_total_over_prob is not None and total_over_odds is not None and total_under_odds is not None:
            over_ev, under_ev = self.calculate_total(
                model_over_prob=model_total_over_prob,
                over_odds_american=total_over_odds,
                under_odds_american=total_under_odds,
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                model_name=model_name,
            )
            result.over_total = over_ev
            result.under_total = under_ev

            if over_ev.is_actionable and over_ev.expected_value > best_ev:
                best_ev = over_ev.expected_value
                result.best_edge = over_ev
            if under_ev.is_actionable and under_ev.expected_value > best_ev:
                best_ev = under_ev.expected_value
                result.best_edge = under_ev

        # Spread analysis
        if model_home_cover_prob is not None and home_spread_odds is not None and away_spread_odds is not None:
            home_spread_ev = self.calculate(
                model_probability=model_home_cover_prob,
                market_odds_american=home_spread_odds,
                game_id=game_id,
                market_type="spread",
                bet_side="home",
                home_team=home_team,
                away_team=away_team,
                model_name=model_name,
                opponent_odds_american=away_spread_odds,
            )
            result.home_spread = home_spread_ev

            away_spread_ev = self.calculate(
                model_probability=1.0 - model_home_cover_prob,
                market_odds_american=away_spread_odds,
                game_id=game_id,
                market_type="spread",
                bet_side="away",
                home_team=home_team,
                away_team=away_team,
                model_name=model_name,
                opponent_odds_american=home_spread_odds,
            )
            result.away_spread = away_spread_ev

            if home_spread_ev.is_actionable and home_spread_ev.expected_value > best_ev:
                best_ev = home_spread_ev.expected_value
                result.best_edge = home_spread_ev
            if away_spread_ev.is_actionable and away_spread_ev.expected_value > best_ev:
                best_ev = away_spread_ev.expected_value
                result.best_edge = away_spread_ev

        # Count actionable bets
        actionable_count = 0
        for ev_result in [
            result.home_ml, result.away_ml,
            result.over_total, result.under_total,
            result.home_spread, result.away_spread,
        ]:
            if ev_result and ev_result.is_actionable:
                actionable_count += 1
        result.num_actionable = actionable_count

        return result

    def format_ev_result(self, ev: EVResult) -> str:
        """Format an EV result for display."""
        if not ev:
            return "No EV data"

        icon = "✅" if ev.is_actionable else "⛔"
        odds_str = f"{ev.market_odds_american:+.0f}" if ev.market_odds_american else f"{ev.market_odds_decimal:.2f}"

        return (
            f"{icon} {ev.bet_side.upper()} ({ev.market_type})\n"
            f"   Model: {ev.model_probability:.1%} | Market: {ev.implied_probability:.1%}\n"
            f"   Odds: {odds_str} | Edge: {ev.edge_percentage:.2%} | EV: {ev.expected_value:.2%}\n"
            f"   {ev.recommendation}"
        )

    def format_game_analysis(self, game_ev: GameEVResult) -> str:
        """Full game analysis formatted for display."""
        lines = [
            f"{'=' * 60}",
            f"{game_ev.away_team} @ {game_ev.home_team}",
            f"{'=' * 60}",
        ]

        if game_ev.best_edge:
            lines.append(f"🎯 BEST BET: {game_ev.best_edge.bet_side.upper()} "
                         f"(Edge: {game_ev.best_edge.edge_percentage:.2%}, "
                         f"EV: {game_ev.best_edge.expected_value:.2%})")

        lines.append(f"\n📊 Actionable bets: {game_ev.num_actionable}")

        for label, ev in [
            ("ML Home", game_ev.home_ml),
            ("ML Away", game_ev.away_ml),
            ("Over", game_ev.over_total),
            ("Under", game_ev.under_total),
            ("Spread Home", game_ev.home_spread),
            ("Spread Away", game_ev.away_spread),
        ]:
            if ev:
                lines.append(f"\n{label}:")
                lines.append(f"  {self.format_ev_result(ev)}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def calculate_expected_value(
    model_probability: float,
    market_odds_american: Optional[float] = None,
    market_odds_decimal: Optional[float] = None,
    min_edge: float = 0.02,
) -> Dict:
    """
    Quick one-shot EV calculation.

    Args:
        model_probability: Your estimated probability (0-1)
        market_odds_american: Market odds in American format
        market_odds_decimal: Market odds in decimal format
        min_edge: Minimum edge threshold

    Returns:
        Dict with expected_value, edge_percentage, implied_probability,
        is_actionable, recommendation
    """
    engine = ExpectedValueEngine(min_edge_threshold=min_edge)
    result = engine.calculate(
        model_probability=model_probability,
        market_odds_american=market_odds_american,
        market_odds_decimal=market_odds_decimal,
    )
    return {
        "expected_value": result.expected_value,
        "edge_percentage": result.edge_percentage,
        "implied_probability": result.implied_probability,
        "model_probability": result.model_probability,
        "decimal_odds": result.market_odds_decimal,
        "is_actionable": result.is_actionable,
        "recommendation": result.recommendation,
    }


def edge_from_probabilities(
    model_probability: float,
    market_implied_probability: float,
) -> float:
    """Simple edge calculation: model_prob - market_implied_prob."""
    return model_probability - market_implied_probability
