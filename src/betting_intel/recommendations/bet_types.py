"""
All supported bet types for the recommendation engine.

Every bet type knows how to:
- Compute its edge vs market
- Estimate win probability
- Generate a readable description

We cover every major basketball betting market:
    Moneyline       -> Team A or Team B to win outright
    Spread          -> Team A -3.5 / Team B +3.5
    Total           -> Over/Under total points
    Team Total      -> Over/Under a specific team's points
    Quarter         -> Winner or total of 1st/2nd/3rd/4th quarter
    Half            -> Winner or total of 1st/2nd half
    Player Props    -> Points, rebounds, assists, PRA for a player
    Parlay          -> Multi-leg combination (auto-suggested)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class BetType(str, Enum):
    """All supported bet categories."""

    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL_POINTS = "total_points"
    TEAM_TOTAL = "team_total"
    FIRST_QUARTER_WINNER = "first_quarter_winner"
    FIRST_HALF_TOTAL = "first_half_total"
    PLAYER_POINTS = "player_points"
    PLAYER_REBOUNDS = "player_rebounds"
    PLAYER_ASSISTS = "player_assists"
    PLAYER_PRA = "player_pra"
    PARLAY = "parlay"

    def display_name(self) -> str:
        return {
            "moneyline": "Moneyline",
            "spread": "Point Spread",
            "total_points": "Total Points O/U",
            "team_total": "Team Total O/U",
            "first_quarter_winner": "1st Quarter Winner",
            "first_half_total": "1st Half Total",
            "player_points": "Player Points",
            "player_rebounds": "Player Rebounds",
            "player_assists": "Player Assists",
            "player_pra": "Player Pts + Reb + Ast",
            "parlay": "Parlay",
        }[self.value]

    def icon(self) -> str:
        return {
            "moneyline": "\U0001F3C6",
            "spread": "\U0001F4CA",
            "total_points": "\U0001F3AF",
            "team_total": "\U0001F4C8",
            "first_quarter_winner": "\U0001F525",
            "first_half_total": "\U0001F3C0",
            "player_points": "\U0001F4B0",
            "player_rebounds": "\U0001F4B0",
            "player_assists": "\U0001F4B0",
            "player_pra": "\U0001F4B0",
            "parlay": "\U0001F3B2",
        }[self.value]


class Confidence(str, Enum):
    """Confidence level for a bet suggestion."""

    VERY_HIGH = "VERY HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY LOW"

    def numeric(self) -> float:
        return {"VERY HIGH": 0.9, "HIGH": 0.75, "MEDIUM": 0.5, "LOW": 0.25, "VERY LOW": 0.1}[self.value]

    def is_clear(self) -> bool:
        """A 'clear' pick is one with high or very high confidence."""
        return self in (Confidence.VERY_HIGH, Confidence.HIGH)


@dataclass
class BetSuggestion:
    """
    A single betting suggestion with edge computation and reasoning.

    This is the universal container for ALL bet types.

    NOTE: All fields without defaults MUST come before fields with defaults
    (Python dataclass requirement). Required fields are grouped first.
    """

    # ═══ REQUIRED FIELDS (no defaults) ═════════════════════════════════

    # --- Identification ---
    game_id: str
    game_date: str
    matchup: str

    # --- The Bet ---
    bet_type: BetType
    bet_side: str  # e.g. "Spurs", "OVER", "Thunder -3.5", "Wemby o24.5pts"
    market_line: float  # The sportsbook line (if known, otherwise model-estimated)

    # --- Model Prediction ---
    predicted_value: float  # Model's prediction (win prob, total pts, margin, etc.)

    # --- Edge ---
    edge_pct: float  # Our edge over the market (0.03 = 3%)
    expected_value: float  # Expected value in units per dollar bet
    win_probability: float  # Model-estimated win probability (0-1)

    # ═══ OPTIONAL FIELDS (with defaults) ═══════════════════════════════

    # --- League & Labels ---
    league: str = "NBA"
    predicted_label: str = ""  # Human-readable prediction

    # --- Staking ---
    kelly_fraction: float = 0.0  # Fraction of bankroll to stake
    stake_dollars: float = 0.0  # Dollar amount ($10k bankroll assumed)
    stake_units: float = 0.0  # Units (1u = 1% of bankroll)

    # --- Confidence ---
    confidence: Confidence = Confidence.LOW
    confidence_reason: str = ""

    # --- Actionable Bet Instruction ---
    bet_action: str = ""  # e.g. "Bet $50 on Spurs -3.5"

    # --- Metadata ---
    reasoning: str = ""
    model_name: str = ""
    is_clear_pick: bool = False
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- Parlay support ---
    legs: list[BetSuggestion] = field(default_factory=list)

    @property
    def is_parlay(self) -> bool:
        return self.bet_type == BetType.PARLAY

    @property
    def combined_edge_pct(self) -> float:
        if self.is_parlay and self.legs:
            product = 1.0
            for leg in self.legs:
                product *= (1 + leg.edge_pct)
            return product - 1.0
        return self.edge_pct

    @property
    def action(self) -> str:
        """Generate the exact, actionable bet instruction."""
        if self.bet_action:
            return self.bet_action
        stake = f"${self.stake_dollars:.0f}" if self.stake_dollars > 0 else "$0"
        return f"{stake} on {self.bet_side}"

    def as_dict(self) -> dict:
        """Serialize to a plain dict for display / JSON export."""
        return {
            "game_id": self.game_id,
            "game_date": self.game_date,
            "matchup": self.matchup,
            "league": self.league,
            "bet_type": self.bet_type.value,
            "bet_type_display": self.bet_type.display_name(),
            "bet_side": self.bet_side,
            "action": self.action,
            "market_line": self.market_line,
            "predicted_value": round(self.predicted_value, 2),
            "predicted_label": self.predicted_label,
            "edge_pct": round(self.edge_pct, 4),
            "expected_value": round(self.expected_value, 4),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "stake_dollars": round(self.stake_dollars, 2),
            "stake_units": round(self.stake_units, 2),
            "confidence": self.confidence.value,
            "is_clear_pick": self.is_clear_pick,
            "reasoning": self.reasoning,
            "model_name": self.model_name,
            "tags": self.tags,
        }


# ── Convenience constructors for common bet types ──────────────────────────


def MoneylineBet(
    game_id: str,
    game_date: str,
    matchup: str,
    team: str,
    win_probability: float,
    market_implied_prob: float = 0.5,
    league: str = "NBA",
    **kwargs,
) -> BetSuggestion:
    """Create a moneyline bet suggestion."""
    edge = win_probability - market_implied_prob
    ev = (win_probability * 1.0) - ((1 - win_probability) * 1.0)
    label = f"{win_probability:.0%} win probability"

    return BetSuggestion(
        game_id=game_id,
        game_date=game_date,
        matchup=matchup,
        league=league,
        bet_type=BetType.MONEYLINE,
        bet_side=team,
        market_line=market_implied_prob,
        predicted_value=win_probability,
        predicted_label=label,
        edge_pct=edge,
        expected_value=ev,
        win_probability=win_probability,
        **kwargs,
    )


def SpreadBet(
    game_id: str,
    game_date: str,
    matchup: str,
    team: str,
    spread_line: float,
    predicted_margin: float,
    league: str = "NBA",
    **kwargs,
) -> BetSuggestion:
    """Create a spread bet suggestion."""
    # If team is favored (negative spread line), they need to win by more than the spread
    # If team is underdog (positive spread line), they need to not lose by more than the spread
    covers = (predicted_margin > spread_line) if spread_line < 0 else (predicted_margin > -spread_line)

    # Simple win probability estimate based on how far predicted margin is from spread
    diff = abs(predicted_margin - abs(spread_line))
    win_prob = min(0.5 + diff * 0.02, 0.85)

    market_implied = 0.5  # Spreads are typically ~50/50 (-110 both sides)
    edge = win_prob - market_implied
    ev = (win_prob * 0.91) - ((1 - win_prob) * 1.0)  # -110 odds

    side = f"{team} {'-' if spread_line < 0 else '+'}{abs(spread_line)}"
    label = f"Pred margin: {predicted_margin:+.1f}"

    return BetSuggestion(
        game_id=game_id,
        game_date=game_date,
        matchup=matchup,
        league=league,
        bet_type=BetType.SPREAD,
        bet_side=side,
        market_line=spread_line,
        predicted_value=predicted_margin,
        predicted_label=label,
        edge_pct=edge,
        expected_value=ev,
        win_probability=win_prob,
        **kwargs,
    )


def TotalBet(
    game_id: str,
    game_date: str,
    matchup: str,
    side: str,  # "OVER" or "UNDER"
    market_total: float,
    predicted_total: float,
    league: str = "NBA",
    **kwargs,
) -> BetSuggestion:
    """Create a total points over/under bet suggestion."""
    if side.upper() == "OVER":
        diff = predicted_total - market_total
        win_prob = min(0.5 + abs(diff) * 0.01, 0.75)
        label = f"Pred: {predicted_total:.0f}"
    else:
        diff = market_total - predicted_total
        win_prob = min(0.5 + abs(diff) * 0.01, 0.75)
        label = f"Pred: {predicted_total:.0f}"

    edge = win_prob - 0.5  # Market typically prices O/U at ~50/50
    ev = (win_prob * 0.91) - ((1 - win_prob) * 1.0)

    return BetSuggestion(
        game_id=game_id,
        game_date=game_date,
        matchup=matchup,
        league=league,
        bet_type=BetType.TOTAL_POINTS,
        bet_side=f"Total {side} {market_total:.0f}",
        market_line=market_total,
        predicted_value=predicted_total,
        predicted_label=label,
        edge_pct=edge,
        expected_value=ev,
        win_probability=win_prob,
        **kwargs,
    )


def TeamTotalBet(
    game_id: str,
    game_date: str,
    matchup: str,
    team: str,
    side: str,
    market_team_total: float,
    predicted_team_total: float,
    league: str = "NBA",
    **kwargs,
) -> BetSuggestion:
    """Create a team total over/under bet suggestion."""
    if side.upper() == "OVER":
        diff = predicted_team_total - market_team_total
        win_prob = min(0.5 + abs(diff) * 0.015, 0.72)
        label = f"Pred: {predicted_team_total:.0f}"
    else:
        diff = market_team_total - predicted_team_total
        win_prob = min(0.5 + abs(diff) * 0.015, 0.72)
        label = f"Pred: {predicted_team_total:.0f}"

    edge = win_prob - 0.5
    ev = (win_prob * 0.91) - ((1 - win_prob) * 1.0)

    return BetSuggestion(
        game_id=game_id,
        game_date=game_date,
        matchup=matchup,
        league=league,
        bet_type=BetType.TEAM_TOTAL,
        bet_side=f"{team} Team Total {side} {market_team_total:.0f}",
        market_line=market_team_total,
        predicted_value=predicted_team_total,
        predicted_label=label,
        edge_pct=edge,
        expected_value=ev,
        win_probability=win_prob,
        **kwargs,
    )


def QuarterBet(
    game_id: str,
    game_date: str,
    matchup: str,
    quarter: int,
    team: str,
    win_probability: float,
    market_implied_prob: float = 0.5,
    league: str = "NBA",
    **kwargs,
) -> BetSuggestion:
    """Create a quarter winner bet suggestion."""
    edge = win_probability - market_implied_prob
    ev = (win_probability * 1.0) - ((1 - win_probability) * 1.0)
    label = f"{win_probability:.0%} win probability"

    ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(quarter, f"{quarter}th")

    return BetSuggestion(
        game_id=game_id,
        game_date=game_date,
        matchup=matchup,
        league=league,
        bet_type=BetType.FIRST_QUARTER_WINNER,
        bet_side=f"{ordinal} Quarter - {team}",
        market_line=market_implied_prob,
        predicted_value=win_probability,
        predicted_label=label,
        edge_pct=edge,
        expected_value=ev,
        win_probability=win_probability,
        **kwargs,
    )


def HalfTotalBet(
    game_id: str,
    game_date: str,
    matchup: str,
    side: str,
    market_half_total: float,
    predicted_half_total: float,
    league: str = "NBA",
    **kwargs,
) -> BetSuggestion:
    """Create a first half total bet suggestion."""
    if side.upper() == "OVER":
        diff = predicted_half_total - market_half_total
        win_prob = min(0.5 + abs(diff) * 0.015, 0.72)
        label = f"Pred: {predicted_half_total:.0f}"
    else:
        diff = market_half_total - predicted_half_total
        win_prob = min(0.5 + abs(diff) * 0.015, 0.72)
        label = f"Pred: {predicted_half_total:.0f}"

    edge = win_prob - 0.5
    ev = (win_prob * 0.91) - ((1 - win_prob) * 1.0)

    return BetSuggestion(
        game_id=game_id,
        game_date=game_date,
        matchup=matchup,
        league=league,
        bet_type=BetType.FIRST_HALF_TOTAL,
        bet_side=f"1st Half {side} {market_half_total:.0f}",
        market_line=market_half_total,
        predicted_value=predicted_half_total,
        predicted_label=label,
        edge_pct=edge,
        expected_value=ev,
        win_probability=win_prob,
        **kwargs,
    )


def PlayerPropBet(
    game_id: str,
    game_date: str,
    matchup: str,
    player_name: str,
    prop_type: BetType,
    market_line: float,
    predicted_value: float,
    side: str = "OVER",
    league: str = "NBA",
    **kwargs,
) -> BetSuggestion:
    """Create a player prop bet suggestion."""
    if side.upper() == "OVER":
        diff = predicted_value - market_line
        win_prob = min(0.5 + abs(diff) * 0.03, 0.72)
        label = f"Pred: {predicted_value:.1f}"
    else:
        diff = market_line - predicted_value
        win_prob = min(0.5 + abs(diff) * 0.03, 0.72)
        label = f"Pred: {predicted_value:.1f}"

    edge = win_prob - 0.5
    ev = (win_prob * 0.91) - ((1 - win_prob) * 1.0)

    prop_display = prop_type.display_name().replace("Player ", "")
    side_display = f"o{market_line}" if side.upper() == "OVER" else f"u{market_line}"

    return BetSuggestion(
        game_id=game_id,
        game_date=game_date,
        matchup=matchup,
        league=league,
        bet_type=prop_type,
        bet_side=f"{player_name} {side_display} {prop_display}",
        market_line=market_line,
        predicted_value=predicted_value,
        predicted_label=label,
        edge_pct=edge,
        expected_value=ev,
        win_probability=win_prob,
        tags=["player_prop"],
        **kwargs,
    )


def ParlaySuggestion(
    legs: list[BetSuggestion],
    game_id: str = "parlay",
    game_date: str = "",
    matchup: str = "Multi-Game Parlay",
    league: str = "NBA",
    **kwargs,
) -> BetSuggestion:
    """Create a parlay from individual bet legs."""
    if not legs:
        raise ValueError("Parlay must have at least one leg")

    # Combined win probability (assuming independence)
    combined_prob = 1.0
    for leg in legs:
        combined_prob *= leg.win_probability

    # Parlay payout at -110 per leg
    leg_count = len(legs)
    parlay_payout = 1.0
    for _ in range(leg_count):
        parlay_payout *= 1.91  # -110 odds per leg
    parlay_payout -= 1.0  # Subtract stake

    edge = combined_prob * (1 + parlay_payout) - 1.0

    return BetSuggestion(
        game_id=game_id,
        game_date=game_date or (legs[0].game_date if legs else ""),
        matchup=matchup,
        league=league,
        bet_type=BetType.PARLAY,
        bet_side=f"{leg_count}-Leg Parlay",
        market_line=0,
        predicted_value=combined_prob,
        predicted_label=f"{combined_prob:.1%} to hit",
        edge_pct=edge,
        expected_value=edge,
        win_probability=combined_prob,
        legs=legs,
        tags=["parlay"],
        **kwargs,
    )
