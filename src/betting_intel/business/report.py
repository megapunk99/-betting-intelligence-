"""
Game Analysis Report — THE CORE PRODUCT.

This is what we SELL to customers. Every game gets the full treatment:
  Fair Spread, Market Spread, Spread Difference
  Fair Total, Market Total, Total Difference
  Fair Moneyline, Market Moneyline
  Expected Value, Recommended Bet Size, Confidence Score
  Key Reasons WHY the market is wrong

Output format is structured, professional, and designed for:
  - Direct display in Telegram/Discord
  - PDF report export
  - Web dashboard rendering
  - Mobile notification
"""

from __future__ import annotations

import json
import logging

import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import math
from dataclasses import dataclass
from typing import Optional


# ── Inline betting utilities (replaces deleted betting.ev, betting.bet modules) ──

def _american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal."""
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / abs(odds)


def _decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American."""
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    else:
        return -100.0 / (decimal - 1.0)


def _american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def _compute_vig_free_prob(home_odds: float, away_odds: float) -> tuple[float, float]:
    """Strip vig from two-way market odds."""
    home_prob = _american_to_implied_prob(home_odds)
    away_prob = _american_to_implied_prob(away_odds)
    total = home_prob + away_prob
    if total > 0:
        return (home_prob / total, away_prob / total)
    return (0.5, 0.5)


@dataclass
class _BetStake:
    recommended_stake: float = 0.0
    is_valid: bool = False


@dataclass
class _BetRecommendation:
    stake: _BetStake = _BetStake()
    expected_value: float = 0.0
    edge_percentage: float = 0.0
    is_actionable: bool = False


class _EVResult:
    """Result of an EV calculation."""
    expected_value: float = 0.0
    edge_percentage: float = 0.0
    is_actionable: bool = False


class _InlineExpectedValueEngine:
    """Minimal EV calculator — replaces deleted betting.ev.ExpectedValueEngine."""

    def __init__(self, min_edge_threshold: float = 0.02):
        self.min_edge_threshold = min_edge_threshold

    def calculate(
        self,
        model_probability: float,
        market_odds_american: float,
        opponent_odds_american: Optional[float] = None,
    ) -> _EVResult:
        decimal = _american_to_decimal(market_odds_american)
        market_implied = 1.0 / decimal
        edge = model_probability - market_implied
        expected_value = model_probability * (decimal - 1.0) - (1.0 - model_probability)
        result = _EVResult()
        result.expected_value = expected_value
        result.edge_percentage = edge
        result.is_actionable = edge >= self.min_edge_threshold
        return result


class _InlineBettingEngine:
    """Minimal betting engine — replaces deleted betting.bet.BettingEngine."""

    def __init__(self, bankroll: float = 10_000.0, kelly_fraction: float = 0.25, min_edge: float = 0.02):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge

    def create_bet(self, **kwargs) -> _BetRecommendation:
        model_prob = kwargs.get("model_probability", 0.5)
        odds_american = kwargs.get("odds_american", -110)
        decimal = _american_to_decimal(odds_american)

        # Full Kelly: f* = (bp - q) / b  where b = decimal - 1, q = 1-p
        b = decimal - 1.0
        if b <= 0 or model_prob <= 0 or model_prob >= 1:
            return _BetRecommendation(stake=_BetStake(), expected_value=0.0, edge_percentage=0.0, is_actionable=False)

        full_kelly = (b * model_prob - (1.0 - model_prob)) / b
        if full_kelly <= 0:
            return _BetRecommendation(stake=_BetStake(), expected_value=0.0, edge_percentage=0.0, is_actionable=False)

        frac = full_kelly * self.kelly_fraction
        frac = max(0.0, min(frac, 0.10))  # Cap at 10%
        stake_amount = round(frac * self.bankroll, 2)

        expected_value = model_prob * decimal - 1.0
        edge = model_prob - (1.0 / decimal) if decimal > 1 else 0.0

        return _BetRecommendation(
            stake=_BetStake(recommended_stake=stake_amount, is_valid=stake_amount >= 10),
            expected_value=expected_value,
            edge_percentage=edge,
            is_actionable=edge >= self.min_edge,
        )

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  THE PRODUCT: Game Analysis Report
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GameAnalysisReport:
    """
    Complete analysis for a single game.

    This is THE product we sell. Every field is computed from
    model predictions vs market data, with full transparency.
    """

    # Game info
    game_id: str = ""
    home_team: str = ""
    away_team: str = ""
    game_date: str = ""
    league: str = "NBA"
    commence_time: str = ""

    # ═══════════════════════════════════════════════════════════════════
    #  THE 11 FIELDS (from the framework)
    # ═══════════════════════════════════════════════════════════════════

    # 1. Fair Spread — What our model thinks the spread SHOULD be
    fair_spread: Optional[float] = None

    # 2. Market Spread — What the sportsbook is offering
    market_spread: Optional[float] = None

    # 3. Spread Difference — Our edge in spread points
    spread_difference: Optional[float] = None

    # 4. Fair Total — What our model thinks the total SHOULD be
    fair_total: Optional[float] = None

    # 5. Market Total — What the sportsbook is offering
    market_total: Optional[float] = None

    # 6. Total Difference — Our edge in total points
    total_difference: Optional[float] = None

    # 7. Fair Moneyline — Our model's fair odds (American format)
    fair_moneyline_home: Optional[float] = None
    fair_moneyline_away: Optional[float] = None

    # 8. Market Moneyline — What the sportsbook is offering
    market_moneyline_home: Optional[float] = None
    market_moneyline_away: Optional[float] = None

    # 9. Expected Value — The edge in percentage
    ev_home_ml: Optional[float] = None
    ev_away_ml: Optional[float] = None
    ev_over_total: Optional[float] = None
    ev_under_total: Optional[float] = None
    ev_home_spread: Optional[float] = None
    ev_away_spread: Optional[float] = None

    # 10. Recommended Bet Size — Dollar amount from Kelly staking
    recommended_bet_home_ml: Optional[float] = None
    recommended_bet_away_ml: Optional[float] = None
    recommended_bet_over: Optional[float] = None
    recommended_bet_under: Optional[float] = None
    recommended_bet_home_spread: Optional[float] = None
    recommended_bet_away_spread: Optional[float] = None

    # 11. Confidence Score (0-100)
    confidence_score_home_ml: float = 0.0
    confidence_score_away_ml: float = 0.0
    confidence_score_over: float = 0.0
    confidence_score_under: float = 0.0
    confidence_score_home_spread: float = 0.0
    confidence_score_away_spread: float = 0.0

    # Key Reasons — WHY the market is wrong
    key_reasons: list[str] = field(default_factory=list)

    # Best bet recommendation
    best_bet: Optional[dict] = None  # {side, type, edge, stake, reason}

    # Summary
    market_inefficiency_score: float = 0.0  # 0-100
    num_actionable_bets: int = 0
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Model information
    model_name: str = "Enhanced Ensemble v3.0"
    model_confidence: str = "MEDIUM"  # LOW, MEDIUM, HIGH, VERY_HIGH

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export / API."""
        return {
            "game": f"{self.away_team} @ {self.home_team}",
            "game_date": self.game_date,
            "league": self.league,

            "spread": {
                "fair": self.fair_spread,
                "market": self.market_spread,
                "difference": self.spread_difference,
            },
            "total": {
                "fair": self.fair_total,
                "market": self.market_total,
                "difference": self.total_difference,
            },
            "moneyline": {
                "home": {
                    "fair": self.fair_moneyline_home,
                    "market": self.market_moneyline_home,
                },
                "away": {
                    "fair": self.fair_moneyline_away,
                    "market": self.market_moneyline_away,
                },
            },
            "expected_value": {
                "home_ml": self.ev_home_ml,
                "away_ml": self.ev_away_ml,
                "over": self.ev_over_total,
                "under": self.ev_under_total,
                "home_spread": self.ev_home_spread,
                "away_spread": self.ev_away_spread,
            },
            "recommended_bets": {
                "home_ml": self.recommended_bet_home_ml,
                "away_ml": self.recommended_bet_away_ml,
                "over": self.recommended_bet_over,
                "under": self.recommended_bet_under,
                "home_spread": self.recommended_bet_home_spread,
                "away_spread": self.recommended_bet_away_spread,
            },
            "confidence": {
                "home_ml": self.confidence_score_home_ml,
                "away_ml": self.confidence_score_away_ml,
                "over": self.confidence_score_over,
                "under": self.confidence_score_under,
                "home_spread": self.confidence_score_home_spread,
                "away_spread": self.confidence_score_away_spread,
            },
            "best_bet": self.best_bet,
            "key_reasons": self.key_reasons,
            "market_inefficiency_score": self.market_inefficiency_score,
            "num_actionable_bets": self.num_actionable_bets,
            "model": {
                "name": self.model_name,
                "confidence": self.model_confidence,
            },
            "generated_at": self.generated_at,
        }

    def to_markdown(self) -> str:
        """Format as markdown — suitable for Telegram/Discord/email."""

        def fmt_odds(val):
            if val is None:
                return "N/A"
            return f"{val:+.0f}" if val < 0 or val > 0 else "EVEN"

        def fmt_edge(val):
            if val is None:
                return "N/A"
            return f"{val:.2%}"

        def fmt_stake(val):
            if val is None or val <= 0:
                return "PASS"
            return f"${val:.0f}"

        def fmt_spread_edge(val):
            if val is None:
                return ""
            return f"  Edge:   {val:+.1f} pts"

        def fmt_float(val):
            if val is None:
                return ""
            return f"{val:.1f}"

        spread_edge_line = fmt_spread_edge(self.spread_difference)
        total_fair_line = f"  Fair:   {fmt_float(self.fair_total)}" if self.fair_total is not None else ""
        total_market_line = f"  Market: {fmt_float(self.market_total)}" if self.market_total is not None else ""
        total_edge_line = f"  Edge:   {self.total_difference:+.1f} pts" if self.total_difference is not None else ""

        lines = [
            f"🎯 **{self.away_team} @ {self.home_team}**",
            f"📅 {self.game_date} · {self.league}",
            f"🤖 {self.model_name}",
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "**📊 GAME ANALYSIS REPORT**",
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            "**SPREAD**",
            f"  Fair:   {fmt_odds(self.fair_spread)}",
            f"  Market: {fmt_odds(self.market_spread)}",
            spread_edge_line,
            "",
            "**TOTAL**",
            total_fair_line,
            total_market_line,
            total_edge_line,
            "",
            "**MONEYLINE**",
            f"  {self.home_team}: Fair {fmt_odds(self.fair_moneyline_home)} | Market {fmt_odds(self.market_moneyline_home)}",
            f"  {self.away_team}: Fair {fmt_odds(self.fair_moneyline_away)} | Market {fmt_odds(self.market_moneyline_away)}",
            "",
            "**EXPECTED VALUE**",
            f"  {self.home_team} ML:  {fmt_edge(self.ev_home_ml)}",
            f"  {self.away_team} ML:  {fmt_edge(self.ev_away_ml)}",
            f"  Over:    {fmt_edge(self.ev_over_total)}",
            f"  Under:   {fmt_edge(self.ev_under_total)}",
            "",
            "**RECOMMENDED BETS**",
            f"  {self.home_team} ML:  {fmt_stake(self.recommended_bet_home_ml)}",
            f"  {self.away_team} ML:  {fmt_stake(self.recommended_bet_away_ml)}",
            f"  Over:    {fmt_stake(self.recommended_bet_over)}",
            f"  Under:   {fmt_stake(self.recommended_bet_under)}",
        ]

        if self.best_bet:
            lines.extend([
                "",
                f"⭐ **BEST BET**",
                f"  {self.best_bet.get('description', '')}",
                f"  Edge: {self.best_bet.get('edge', 0):.2%} | Stake: ${self.best_bet.get('stake', 0):.0f}",
            ])

        if self.key_reasons:
            lines.extend([
                "",
                "**🔍 WHY THE MARKET IS WRONG**",
            ])
            for i, reason in enumerate(self.key_reasons, 1):
                lines.append(f"  {i}. {reason}")

        lines.extend([
            "",
            f"**Confidence**: {self.confidence_score_home_ml:.0f}/100 · "
            f"**Inefficiency**: {self.market_inefficiency_score:.0f}/100",
            "",
            f"🕐 {self.generated_at}",
        ])

        return "\n".join(lines)

    def to_telegram(self) -> str:
        """Telegram-optimized format (shorter, emoji-rich)."""
        return self.to_markdown()


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT GENERATOR — The factory that produces our product
# ═══════════════════════════════════════════════════════════════════════════

class GameAnalysisGenerator:
    """
    Produces GameAnalysisReport from model predictions and market data.

    This is the ASSEMBLY LINE that takes raw predictions and converts them
    into the sellable product.

    Usage:
        generator = GameAnalysisGenerator(bankroll=10000)
        report = generator.analyze_game(
            home_team="Celtics",
            away_team="Lakers",
            model_home_win_prob=0.62,
            model_predicted_total=218.5,
            model_predicted_margin=4.5,
            home_ml_odds=-150,
            away_ml_odds=+130,
            market_total=214.5,
            home_spread=-3.5,
        )
        print(report.to_markdown())
    """

    def __init__(
        self,
        bankroll: float = 10_000.0,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.02,
        db_path: Optional[Path] = None,
    ):
        self.betting_engine = _InlineBettingEngine(
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            min_edge=min_edge,
        )
        self.ev_engine = _InlineExpectedValueEngine(min_edge_threshold=min_edge)
        self.db_path = db_path

    def analyze_game(
        self,
        home_team: str,
        away_team: str,
        game_date: str = "",
        league: str = "NBA",
        game_id: str = "",

        # Model predictions
        model_home_win_prob: float = 0.5,
        model_predicted_total: Optional[float] = None,
        model_predicted_margin: Optional[float] = None,

        # Market data
        home_ml_odds: Optional[float] = None,
        away_ml_odds: Optional[float] = None,
        market_total: Optional[float] = None,
        home_spread: Optional[float] = None,
        spread_odds_home: Optional[float] = None,
        spread_odds_away: Optional[float] = None,
        over_odds: Optional[float] = None,
        under_odds: Optional[float] = None,

        model_name: str = "Enhanced Ensemble v3.0",
    ) -> GameAnalysisReport:
        """Generate a complete GameAnalysisReport for a single game."""
        if not game_id:
            game_id = f"{home_team}_vs_{away_team}_{game_date}".replace(" ", "_")

        report = GameAnalysisReport(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            league=league,
            model_name=model_name,
        )

        key_reasons = []
        best_edge = 0.0
        best_side = None
        best_stake = 0.0
        best_desc = ""
        actionable_count = 0

        # ── Compute spread analysis ──────────────────────────────────
        if model_predicted_margin is not None:
            fair_spread = round(-model_predicted_margin * 2) / 2  # Round to nearest 0.5
            report.fair_spread = fair_spread
            report.market_spread = home_spread
            if home_spread is not None:
                report.spread_difference = round(fair_spread - home_spread, 1)

                # EV for spread
                if spread_odds_home and spread_odds_away:
                    home_cover_prob = self._estimate_cover_prob(model_predicted_margin, home_spread)
                    ev_home = self.ev_engine.calculate(
                        model_probability=home_cover_prob,
                        market_odds_american=spread_odds_home,
                    )
                    ev_away = self.ev_engine.calculate(
                        model_probability=1.0 - home_cover_prob,
                        market_odds_american=spread_odds_away,
                    )
                    report.ev_home_spread = ev_home.expected_value
                    report.ev_away_spread = ev_away.expected_value
                    report.confidence_score_home_spread = min(abs(ev_home.edge_percentage) * 1000, 95)
                    report.confidence_score_away_spread = min(abs(ev_away.edge_percentage) * 1000, 95)

                    # Kelly stake
                    if ev_home.is_actionable:
                        bet = self.betting_engine.create_bet(
                            model_probability=home_cover_prob,
                            odds_american=spread_odds_home,
                            home_team=home_team,
                            away_team=away_team,
                            bet_side="home",
                            market_type="spread",
                            market_line=home_spread,
                        )
                        report.recommended_bet_home_spread = bet.stake.recommended_stake if bet.stake.is_valid else 0
                        if bet.stake.recommended_stake > best_stake and bet.expected_value > best_edge:
                            best_edge = bet.expected_value
                            best_stake = bet.stake.recommended_stake
                            best_side = f"{home_team} {home_spread:+.1f}"
                            best_desc = f"{home_team} {home_spread:+.1f} — model predicts margin of {model_predicted_margin:+.1f}"
                            key_reasons.append(f"Spread mispricing: {home_team} {home_spread:+.1f} vs fair {fair_spread:+.1f} (model margin: {model_predicted_margin:+.1f})")
                            actionable_count += 1

                    if ev_away.is_actionable:
                        bet = self.betting_engine.create_bet(
                            model_probability=1.0 - home_cover_prob,
                            odds_american=spread_odds_away,
                            home_team=home_team,
                            away_team=away_team,
                            bet_side="away",
                            market_type="spread",
                            market_line=abs(home_spread) if home_spread else None,
                        )
                        report.recommended_bet_away_spread = bet.stake.recommended_stake if bet.stake.is_valid else 0
                        if bet.stake.recommended_stake > best_stake and bet.expected_value > best_edge:
                            best_edge = bet.expected_value
                            best_stake = bet.stake.recommended_stake
                            best_side = f"{away_team} {abs(home_spread):+.1f}" if home_spread else away_team
                            best_desc = f"{away_team} +{abs(home_spread):.1f}"
                            actionable_count += 1

        # ── Compute total analysis ───────────────────────────────────
        if model_predicted_total is not None and market_total is not None:
            report.fair_total = round(model_predicted_total, 1)
            report.market_total = market_total
            report.total_difference = round(model_predicted_total - market_total, 1)

            if over_odds and under_odds:
                # PROPER probability: sigmoid instead of crude min(0.5 + diff*0.01, 0.72)
                # The crude linear hack hard-clips at 0.72 and can cross 1.0 for large diffs
                # Sigmoid correctly maps (-inf, +inf) → (0, 1) with proper saturation
                import math
                diff = model_predicted_total - market_total
                # k=0.015 calibrated for total points (market is efficient)
                over_prob = 1.0 / (1.0 + math.exp(-0.015 * diff))
                over_prob = max(0.28, min(0.72, over_prob))

                ev_over = self.ev_engine.calculate(
                    model_probability=over_prob,
                    market_odds_american=over_odds,
                )
                ev_under = self.ev_engine.calculate(
                    model_probability=1.0 - over_prob,
                    market_odds_american=under_odds,
                )
                report.ev_over_total = ev_over.expected_value
                report.ev_under_total = ev_under.expected_value
                report.confidence_score_over = min(abs(ev_over.edge_percentage) * 1000, 95)
                report.confidence_score_under = min(abs(ev_under.edge_percentage) * 1000, 95)

                if ev_over.is_actionable:
                    bet = self.betting_engine.create_bet(
                        model_probability=over_prob,
                        odds_american=over_odds,
                        home_team=home_team,
                        away_team=away_team,
                        bet_side="over",
                        market_type="total",
                        market_line=market_total,
                    )
                    report.recommended_bet_over = bet.stake.recommended_stake if bet.stake.is_valid else 0
                    if bet.stake.recommended_stake > best_stake and bet.expected_value > best_edge:
                        best_edge = bet.expected_value
                        best_stake = bet.stake.recommended_stake
                        best_side = f"Over {market_total:.0f}"
                        best_desc = f"Over {market_total:.0f} — model projects {model_predicted_total:.1f}"
                        key_reasons.append(f"Total mispricing: Over {market_total:.0f} vs fair {model_predicted_total:.1f} (model projects higher scoring)")
                        actionable_count += 1

                if ev_under.is_actionable:
                    bet = self.betting_engine.create_bet(
                        model_probability=1.0 - over_prob,
                        odds_american=under_odds,
                        home_team=home_team,
                        away_team=away_team,
                        bet_side="under",
                        market_type="total",
                        market_line=market_total,
                    )
                    report.recommended_bet_under = bet.stake.recommended_stake if bet.stake.is_valid else 0
                    if bet.stake.recommended_stake > best_stake and bet.expected_value > best_edge:
                        best_edge = bet.expected_value
                        best_stake = bet.stake.recommended_stake
                        best_side = f"Under {market_total:.0f}"
                        best_desc = f"Under {market_total:.0f} — model projects {model_predicted_total:.1f}"
                        key_reasons.append(f"Total mispricing: Under {market_total:.0f} vs fair {model_predicted_total:.1f} (market overestimating scoring)")
                        actionable_count += 1

        # ── Compute moneyline analysis ───────────────────────────────
        if home_ml_odds and away_ml_odds:
            # Fair moneyline
            fair_home_prob = model_home_win_prob
            fair_away_prob = 1.0 - model_home_win_prob

            # Convert fair probability to American odds
            report.fair_moneyline_home = self._prob_to_american(fair_home_prob)
            report.fair_moneyline_away = self._prob_to_american(fair_away_prob)
            report.market_moneyline_home = home_ml_odds
            report.market_moneyline_away = away_ml_odds

            # EV for moneyline
            ev_home = self.ev_engine.calculate(
                model_probability=fair_home_prob,
                market_odds_american=home_ml_odds,
                opponent_odds_american=away_ml_odds,
            )
            ev_away = self.ev_engine.calculate(
                model_probability=fair_away_prob,
                market_odds_american=away_ml_odds,
                opponent_odds_american=home_ml_odds,
            )
            report.ev_home_ml = ev_home.expected_value
            report.ev_away_ml = ev_away.expected_value
            report.confidence_score_home_ml = min(abs(ev_home.edge_percentage) * 1000, 95)
            report.confidence_score_away_ml = min(abs(ev_away.edge_percentage) * 1000, 95)

            if ev_home.is_actionable:
                bet = self.betting_engine.create_bet(
                    model_probability=fair_home_prob,
                    odds_american=home_ml_odds,
                    home_team=home_team,
                    away_team=away_team,
                    bet_side="home",
                    market_type="moneyline",
                    opponent_odds_american=away_ml_odds,
                )
                report.recommended_bet_home_ml = bet.stake.recommended_stake if bet.stake.is_valid else 0
                if bet.stake.recommended_stake > best_stake and bet.expected_value > best_edge:
                    best_edge = bet.expected_value
                    best_stake = bet.stake.recommended_stake
                    best_side = f"{home_team} ML"
                    best_desc = f"{home_team} ML at {home_ml_odds:+.0f} — model win prob {fair_home_prob:.1%}"
                    key_reasons.append(f"Moneyline value: {home_team} at {home_ml_odds:+.0f} (fair: {report.fair_moneyline_home:+.0f}, model prob: {fair_home_prob:.1%})")
                    actionable_count += 1

            if ev_away.is_actionable:
                bet = self.betting_engine.create_bet(
                    model_probability=fair_away_prob,
                    odds_american=away_ml_odds,
                    home_team=home_team,
                    away_team=away_team,
                    bet_side="away",
                    market_type="moneyline",
                    opponent_odds_american=home_ml_odds,
                )
                report.recommended_bet_away_ml = bet.stake.recommended_stake if bet.stake.is_valid else 0
                if bet.stake.recommended_stake > best_stake and bet.expected_value > best_edge:
                    best_edge = bet.expected_value
                    best_stake = bet.stake.recommended_stake
                    best_side = f"{away_team} ML"
                    best_desc = f"{away_team} ML at {away_ml_odds:+.0f} — model win prob {fair_away_prob:.1%}"
                    key_reasons.append(f"Moneyline value: {away_team} at {away_ml_odds:+.0f} (fair: {report.fair_moneyline_away:+.0f}, model prob: {fair_away_prob:.1%})")
                    actionable_count += 1

        # ── Package the best bet ─────────────────────────────────────
        if best_side and best_stake > 0:
            report.best_bet = {
                "side": best_side,
                "description": best_desc,
                "edge": best_edge,
                "stake": best_stake,
            }

        report.key_reasons = key_reasons[:5]  # Max 5 reasons
        report.num_actionable_bets = actionable_count

        # Market inefficiency score (0-100)
        edges = [
            abs(e) for e in [
                report.ev_home_ml, report.ev_away_ml,
                report.ev_over_total, report.ev_under_total,
                report.ev_home_spread, report.ev_away_spread,
            ] if e is not None
        ]
        if edges:
            report.market_inefficiency_score = min(sum(edges) / len(edges) * 500, 100)

        # Model confidence
        if best_edge >= 0.08:
            report.model_confidence = "VERY_HIGH"
        elif best_edge >= 0.05:
            report.model_confidence = "HIGH"
        elif best_edge >= 0.03:
            report.model_confidence = "MEDIUM"
        else:
            report.model_confidence = "LOW"

        return report

    def _estimate_cover_prob(self, predicted_margin: float, spread: float) -> float:
        """Estimate probability of covering the spread given predicted margin."""
        if spread is None:
            return 0.5
        diff = predicted_margin + spread  # For negative spread, this is margin - spread
        # Simple logistic: if predicted margin beats the spread by X, higher prob
        z = diff / 6.0  # Standard deviation of NBA margins ~12 points
        prob = 1.0 / (1.0 + np.exp(-z))
        return max(0.25, min(0.75, prob))

    def _prob_to_american(self, prob: float) -> float:
        """Convert probability to American odds."""
        if prob <= 0 or prob >= 1:
            return 0.0
        if prob >= 0.5:
            denom = 1.0 - prob
            if denom < 0.001:
                return -10000.0
            return -100.0 * (prob / denom)
        else:
            denom = prob
            if denom < 0.001:
                return 10000.0
            return 100.0 * ((1.0 - prob) / denom)

    def generate_batch(self, game_data_list: list[dict]) -> list[GameAnalysisReport]:
        """Generate reports for multiple games at once."""
        return [self.analyze_game(**data) for data in game_data_list]


# ═══════════════════════════════════════════════════════════════════════════
#  DAILY BETTING CARD — THE PRODUCT WE DELIVER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DailyBettingCard:
    """
    A complete daily betting card — the product we deliver to subscribers.

    Contains:
      - Game analyses for all games today
      - Ranked picks by edge
      - Clear picks only (high confidence)
      - Total risk/exposure for the day
      - CLV tracking reference
    """

    date: str
    games: list[GameAnalysisReport] = field(default_factory=list)
    subscriber_tier: str = "free"  # free, basic, premium

    # Aggregated
    total_actionable_bets: int = 0
    total_recommended_stake: float = 0.0
    best_play: Optional[dict] = None

    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def nba_games(self) -> list[GameAnalysisReport]:
        return [g for g in self.games if g.league == "NBA"]

    @property
    def small_league_games(self) -> list[GameAnalysisReport]:
        return [g for g in self.games if g.league != "NBA"]

    def to_markdown(self) -> str:
        """Format the complete daily card as markdown."""

        lines = [
            "🎯 **DAILY BETTING CARD**",
            f"📅 {self.date}",
            f"👤 Tier: {self.subscriber_tier.upper()}",
            "━" * 35,
            "",
        ]

        # Best play header
        if self.best_play:
            lines.extend([
                f"⭐ **PLAY OF THE DAY**",
                f"  {self.best_play.get('description', '')}",
                f"  Edge: {self.best_play.get('edge', 0):.2%} | Stake: ${self.best_play.get('stake', 0):.0f}",
                "━" * 35,
                "",
            ])

        # Game analyses
        for i, game in enumerate(self.games, 1):
            lines.append(f"**GAME {i}: {game.away_team} @ {game.home_team}**")
            lines.append(f"  Model: {game.model_name} | Confidence: {game.model_confidence}")
            if game.best_bet:
                lines.append(f"  ⭐ {game.best_bet.get('description', '')}")
                lines.append(f"     Edge: {game.best_bet.get('edge', 0):.2%} | Stake: ${game.best_bet.get('stake', 0):.0f}")
            lines.append("")

        # Summary
        lines.extend([
            "━" * 35,
            f"**SUMMARY**",
            f"  Games: {len(self.games)}",
            f"  Actionable bets: {self.total_actionable_bets}",
            f"  Total stake: ${self.total_recommended_stake:.0f}",
            "",
            f"🕐 {self.generated_at}",
        ])

        return "\n".join(lines)

    def to_telegram(self) -> str:
        """Telegram-optimized format."""
        return self.to_markdown()
