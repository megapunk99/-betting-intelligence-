"""
+EV Scanner — identifies positive expected value betting opportunities.

Compares model predictions against real market odds from the OddsPoller
(which includes odds from multiple sportsbooks via TheOddsAPI). For every
game and every bet type, it:

  1. Computes no-vig (fair) probabilities from market odds
  2. Compares model-estimated probabilities against market-implied probabilities
  3. Flags opportunities where the model's probability > market-implied probability
  4. Ranks by expected value, edge percentage, and Kelly-optimal stake
  5. Tracks CLV (Closing Line Value) by comparing against historical snapshots

Key insight: If the model thinks an outcome has a 60% chance of happening
but the market (across all sportsbooks) prices it at 52%, that's an 8% edge —
a +EV opportunity.

Usage:
    scanner = PositiveEVScanner()

    # From OddsPoller snapshots
    odds_snapshots = poller.get_current_odds()
    opportunities = scanner.scan_odds_snapshots(odds_snapshots)

    # From RecommendationEngine model predictions
    opportunities = scanner.scan_with_model_predictions(
        odds_snapshots=poller.get_current_odds(),
        model_predictions=engine.generate_all_bets(),
    )

    # Rank and filter
    top_ev = scanner.rank_by_ev(opportunities)[:10]
    best_edges = scanner.filter_opportunities(opportunities, min_edge=0.05)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Optional
from enum import Enum

from betting_intel.recommendations.bet_types import BetSuggestion, BetType, Confidence

logger = logging.getLogger(__name__)


# ── Enums ───────────────────────────────────────────────────────────────────


class ScannerConfidence(str, Enum):
    """Confidence in a detected +EV opportunity."""

    VERY_HIGH = "VERY_HIGH"  # Multiple books agree, model confidence high
    HIGH = "HIGH"  # Model and market direction agree
    MEDIUM = "MEDIUM"  # Decent edge, reasonable confidence
    LOW = "LOW"  # Small edge, one-sided data
    SPECULATIVE = "SPECULATIVE"  # Very small edge or insufficient data

    def numeric(self) -> float:
        return {
            "VERY_HIGH": 0.90,
            "HIGH": 0.75,
            "MEDIUM": 0.50,
            "LOW": 0.25,
            "SPECULATIVE": 0.10,
        }[self.value]


class ScannerSource(str, Enum):
    """Source of the detected opportunity."""

    MODEL_VS_MARKET = "model_vs_market"  # Model prediction > market-implied prob
    MULTI_BOOK_ARB = "multi_book_arb"  # Discrepancy between sportsbooks
    MARKET_MISPRICING = "market_mispricing"  # Systematic market error detected
    LINE_MOVEMENT = "line_movement"  # Sharp line movement creates value



# ── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class EVOpportunity:
    """
    A detected positive expected value betting opportunity.

    This is the core output of the +EV Scanner — a structured package
    containing everything needed to decide whether to bet.
    """

    # Identification
    game_id: str
    league: str
    matchup: str
    game_date: str

    # The bet
    bet_side: str  # e.g. "Spurs", "OVER 214.5", "Thunder -3.5"
    market_line: float  # Current best available line
    sportsbook: str  # Which sportsbook has this line

    # Market data
    implied_probability: float  # Market-implied win probability (no-vig)
    best_odds_decimal: float  # Best decimal odds available

    # Model prediction
    model_probability: float  # Model-estimated win probability
    model_source: str  # Which model produced this prediction

    # Edge computation
    edge_pct: float  # model_prob - implied_prob
    expected_value: float  # EV per dollar wagered
    expected_value_pct: float  # EV as percentage of stake

    # Staking recommendation
    kelly_fraction: float  # Fraction of bankroll to stake
    recommended_stake_dollars: float  # Dollar amount (at $10k bankroll)

    # Quality
    confidence: ScannerConfidence
    n_sportsbooks: int  # How many books had this market
    consensus_agreement: float  # 0-1, how much books agree on the price

    # Team info (used for CLV direction heuristics)
    home_team: Optional[str] = None
    away_team: Optional[str] = None

    # CLV tracking
    opening_line: Optional[float] = None  # Earlier line for CLV computation
    clv_pct: Optional[float] = None  # Closing line value (positive = sharp)
    line_movement_direction: Optional[str] = None  # "toward", "away", "neutral"

    # Source
    source: ScannerSource = ScannerSource.MODEL_VS_MARKET
    tags: list[str] = field(default_factory=list)
    reasoning: str = ""
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def as_dict(self) -> dict:
        """Serialize to dict for API responses."""
        return {
            "game_id": self.game_id,
            "league": self.league,
            "matchup": self.matchup,
            "game_date": self.game_date,
            "bet_side": self.bet_side,
            "market_line": self.market_line,
            "sportsbook": self.sportsbook,
            "implied_probability": round(self.implied_probability, 4),
            "best_odds_decimal": self.best_odds_decimal,
            "model_probability": round(self.model_probability, 4),
            "model_source": self.model_source,
            "edge_pct": round(self.edge_pct, 4),
            "expected_value": round(self.expected_value, 4),
            "expected_value_pct": round(self.expected_value_pct, 4),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "recommended_stake_dollars": round(self.recommended_stake_dollars, 2),
            "confidence": self.confidence.value,
            "n_sportsbooks": self.n_sportsbooks,
            "consensus_agreement": round(self.consensus_agreement, 4),
            "opening_line": self.opening_line,
            "clv_pct": self.clv_pct,
            "line_movement_direction": self.line_movement_direction,
            "source": self.source.value,
            "tags": self.tags,
            "reasoning": self.reasoning,
            "detected_at": self.detected_at,
        }

    @property
    def action(self) -> str:
        """Generate a human-readable betting instruction."""
        stake = f"${self.recommended_stake_dollars:.0f}" if self.recommended_stake_dollars > 0 else "PASS"
        return f"{stake} on {self.bet_side} @ {self.best_odds_decimal:.2f} (EV: {self.expected_value_pct:.1%})"


@dataclass
class ScannerReport:
    """Summary report from the +EV Scanner."""

    total_opportunities: int = 0
    total_games_scanned: int = 0
    actionable_opportunities: int = 0  # Edge > 3%
    speculative_opportunities: int = 0
    average_edge_pct: float = 0.0
    best_edge_pct: float = 0.0
    total_kelly_exposure: float = 0.0  # Sum of recommended stakes
    by_league: dict[str, int] = field(default_factory=dict)
    by_bet_type: dict[str, int] = field(default_factory=dict)
    by_confidence: dict[str, int] = field(default_factory=dict)
    top_opportunities: list[EVOpportunity] = field(default_factory=list)
    scanned_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def as_dict(self) -> dict:
        return {
            "total_opportunities": self.total_opportunities,
            "total_games_scanned": self.total_games_scanned,
            "actionable_opportunities": self.actionable_opportunities,
            "speculative_opportunities": self.speculative_opportunities,
            "average_edge_pct": round(self.average_edge_pct, 4),
            "best_edge_pct": round(self.best_edge_pct, 4),
            "total_kelly_exposure": round(self.total_kelly_exposure, 2),
            "by_league": self.by_league,
            "by_bet_type": self.by_bet_type,
            "by_confidence": self.by_confidence,
            "top_opportunities": [o.as_dict() for o in self.top_opportunities[:5]],
            "scanned_at": self.scanned_at,
        }


# ── +EV Scanner ─────────────────────────────────────────────────────────────


class PositiveEVScanner:
    """
    Scans odds across sportsbooks and detects +EV betting opportunities.

    The scanner is the bridge between the raw odds data (from OddsPoller /
    TheOddsAPI) and the model predictions (from the RecommendationEngine).
    It performs the critical function of identifying when market prices
    deviate from model expectations — the core of +EV betting.

    Configuration thresholds:
      MIN_EDGE_PCT: Minimum edge to flag (default 1%)
      MIN_EV: Minimum expected value per dollar (default $0.01)
      MIN_SPORTSBOOKS: Minimum books with this market to trust consensus
      ACTIONABLE_EDGE: Edge above this = \"actionable\" opportunity
      KELLY_FRACTION: Fraction of full Kelly for staking (default 0.25)
      ASSUMED_BANKROLL: Bankroll assumption for stake calc (default $10k)
    """

    MIN_EDGE_PCT: float = 0.01  # 1% minimum edge to flag
    MIN_EV: float = 0.01  # Minimum EV per dollar
    MIN_SPORTSBOOKS: int = 1  # Minimum books with this market
    ACTIONABLE_EDGE: float = 0.03  # Edge > 3% = actionable
    KELLY_FRACTION: float = 0.25  # Quarter Kelly for stake sizing
    ASSUMED_BANKROLL: float = 10_000.0  # $10k bankroll assumption

    def __init__(
        self,
        min_edge_pct: float = MIN_EDGE_PCT,
        min_ev: float = MIN_EV,
        min_sportsbooks: int = MIN_SPORTSBOOKS,
        actionable_edge: float = ACTIONABLE_EDGE,
        kelly_fraction: float = KELLY_FRACTION,
        assumed_bankroll: float = ASSUMED_BANKROLL,
    ):
        self.min_edge = min_edge_pct
        self.min_ev = min_ev
        self.min_sportsbooks = min_sportsbooks
        self.actionable_edge = actionable_edge
        self.kelly_fraction = kelly_fraction
        self.bankroll = assumed_bankroll
        self._last_historical_snapshots: dict[str, list[dict]] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def scan_odds_snapshots(
        self,
        odds_snapshots: list[dict],
        model_predictions: Optional[list[BetSuggestion]] = None,
    ) -> ScannerReport:
        """
        Scan odds snapshots from the OddsPoller for +EV opportunities.

        Args:
            odds_snapshots: List of odds dicts from poller.get_current_odds()
            model_predictions: Optional list of BetSuggestion from engine
                              (provides model probabilities for edge calc)

        Returns:
            ScannerReport with all detected opportunities.
        """
        if not odds_snapshots:
            return ScannerReport()

        # Index model predictions by game/matchup for lookup
        model_index: dict[str, list[BetSuggestion]] = {}
        if model_predictions:
            for bet in model_predictions:
                key = bet.game_id or f"{bet.matchup}_{bet.league}"
                if key not in model_index:
                    model_index[key] = []
                model_index[key].append(bet)

        opportunities: list[EVOpportunity] = []
        games_scanned: set[str] = set()

        for snapshot in odds_snapshots:
            game_id = snapshot.get("game_id", "")
            league = snapshot.get("league", "NBA")
            home_team = snapshot.get("home_team", "")
            away_team = snapshot.get("away_team", "")
            game_date = snapshot.get("game_date", "")
            matchup = f"{away_team} @ {home_team}"
            games_scanned.add(game_id or matchup)

            # Look up model predictions for this game
            game_model_bets = model_index.get(game_id, [])
            if not game_model_bets:
                # Try matching by teams
                for key, bets in model_index.items():
                    if home_team.lower() in key.lower() or away_team.lower() in key.lower():
                        game_model_bets = bets
                        break

            # ── Moneyline Scan ─────────────────────────────────────────
            home_ml = snapshot.get("home_ml")
            away_ml = snapshot.get("away_ml")
            if home_ml and away_ml:
                # Compute no-vig probabilities
                no_vig_home, no_vig_away = self._compute_no_vig_probs(
                    home_ml, away_ml
                )

                # Find corresponding model predictions
                home_model_prob = None
                away_model_prob = None
                for bet in game_model_bets:
                    if bet.bet_type == BetType.MONEYLINE:
                        if bet.bet_side and home_team.lower() in bet.bet_side.lower():
                            home_model_prob = bet.win_probability
                        elif bet.bet_side and away_team.lower() in bet.bet_side.lower():
                            away_model_prob = bet.win_probability

                # Home moneyline
                if home_model_prob is not None:
                    opp = self._build_ml_opportunity(
                        game_id=game_id,
                        league=league,
                        matchup=matchup,
                        game_date=game_date,
                        team=home_team,
                        model_prob=home_model_prob,
                        implied_prob=no_vig_home,
                        decimal_odds=home_ml,
                        snapshot=snapshot,
                    )
                    if opp:
                        opportunities.append(opp)

                # Away moneyline
                if away_model_prob is not None:
                    opp = self._build_ml_opportunity(
                        game_id=game_id,
                        league=league,
                        matchup=matchup,
                        game_date=game_date,
                        team=away_team,
                        model_prob=away_model_prob,
                        implied_prob=no_vig_away,
                        decimal_odds=away_ml,
                        snapshot=snapshot,
                    )
                    if opp:
                        opportunities.append(opp)

            # ── Total Points Scan ──────────────────────────────────────
            total = snapshot.get("total")
            if total:
                # Estimate no-vig total (O/U typically 50/50 pricing)
                # Build opportunities if model made predictions
                for bet in game_model_bets:
                    if bet.bet_type == BetType.TOTAL_POINTS:
                        side = bet.bet_side  # e.g. "Total OVER 214.5"
                        opp = self._build_total_opportunity(
                            game_id=game_id,
                            league=league,
                            matchup=matchup,
                            game_date=game_date,
                            side="OVER" if "OVER" in side.upper() else "UNDER",
                            model_prob=bet.win_probability,
                            market_total=total,
                            snapshot=snapshot,
                        )
                        if opp:
                            opportunities.append(opp)

            # ── Spread Scan ────────────────────────────────────────────
            spread = snapshot.get("spread")
            spread_home = snapshot.get("spread_home") or snapshot.get("spread")
            if spread is not None:
                for bet in game_model_bets:
                    if bet.bet_type == BetType.SPREAD:
                        opp = self._build_spread_opportunity(
                            game_id=game_id,
                            league=league,
                            matchup=matchup,
                            game_date=game_date,
                            model_prob=bet.win_probability,
                            spread_line=bet.market_line,
                            snapshot=snapshot,
                        )
                        if opp:
                            opportunities.append(opp)

        # ── Post-processing ─────────────────────────────────────────────
        # Track CLV if historical data is available
        for opp in opportunities:
            self._compute_clv(opp)

        # Rank and filter
        opportunities = self._filter_and_rank(opportunities)

        # Build report
        report = self._build_report(opportunities, len(games_scanned))
        return report

    def scan_with_model_predictions(
        self,
        odds_snapshots: list[dict],
        model_predictions: list[BetSuggestion],
    ) -> ScannerReport:
        """
        Convenience wrapper — same as scan_odds_snapshots but requires
        model predictions (no fallback logic).
        """
        return self.scan_odds_snapshots(
            odds_snapshots=odds_snapshots,
            model_predictions=model_predictions,
        )

    def scan_from_poller(
        self,
        poller: Any,
        model_predictions: Optional[list[BetSuggestion]] = None,
    ) -> ScannerReport:
        """
        Convenience: scan directly from an OddsPoller instance.

        Args:
            poller: An OddsPoller instance (has get_current_odds method)
            model_predictions: Optional model predictions

        Returns:
            ScannerReport
        """
        odds = poller.get_current_odds()
        return self.scan_odds_snapshots(odds, model_predictions)

    # ── Utility Methods ─────────────────────────────────────────────────────

    def rank_by_ev(self, opportunities: list[EVOpportunity]) -> list[EVOpportunity]:
        """Sort opportunities by expected value descending."""
        return sorted(opportunities, key=lambda o: o.expected_value, reverse=True)

    def rank_by_edge(self, opportunities: list[EVOpportunity]) -> list[EVOpportunity]:
        """Sort opportunities by edge percentage descending."""
        return sorted(opportunities, key=lambda o: o.edge_pct, reverse=True)

    def filter_opportunities(
        self,
        opportunities: list[EVOpportunity],
        min_edge: Optional[float] = None,
        min_ev: Optional[float] = None,
        leagues: Optional[list[str]] = None,
        confidence_min: Optional[ScannerConfidence] = None,
    ) -> list[EVOpportunity]:
        """
        Filter opportunities by various criteria.

        Args:
            opportunities: List to filter
            min_edge: Minimum edge pct (defaults to self.min_edge)
            min_ev: Minimum EV per dollar (defaults to self.min_ev)
            leagues: Only include these leagues
            confidence_min: Minimum confidence level

        Returns:
            Filtered list
        """
        effective_min_edge = min_edge if min_edge is not None else self.min_edge
        effective_min_ev = min_ev if min_ev is not None else self.min_ev

        result = []
        for opp in opportunities:
            if opp.edge_pct < effective_min_edge:
                continue
            if opp.expected_value < effective_min_ev:
                continue
            if leagues and opp.league not in leagues:
                continue
            if confidence_min:
                # Compare by numeric value
                if opp.confidence.numeric() < confidence_min.numeric():
                    continue
            result.append(opp)

        return result

    def get_actionable(self, opportunities: list[EVOpportunity]) -> list[EVOpportunity]:
        """Get only actionable opportunities (edge > ACTIONABLE_EDGE)."""
        return self.filter_opportunities(
            opportunities, min_edge=self.actionable_edge
        )

    def record_historical_snapshot(self, odds_snapshots: list[dict]):
        """
        Store current odds as historical data for CLV tracking.

        Call this before games start to capture opening lines.
        The scanner will compare current odds against these to compute CLV.
        """
        now = time.time()
        for snap in odds_snapshots:
            game_id = snap.get("game_id", "")
            if game_id not in self._last_historical_snapshots:
                self._last_historical_snapshots[game_id] = []
            self._last_historical_snapshots[game_id].append({
                **snap,
                "_recorded_at": now,
            })

            # Keep only last 100 snapshots per game
            if len(self._last_historical_snapshots[game_id]) > 100:
                self._last_historical_snapshots[game_id] = (
                    self._last_historical_snapshots[game_id][-100:]
                )

    # ── Internal Methods ────────────────────────────────────────────────────

    @staticmethod
    def _compute_no_vig_probs(odds_a: float, odds_b: float) -> tuple[float, float]:
        """
        Remove the vigorish from a two-outcome market to get fair probabilities.

        Formula:
            implied_a = 1 / odds_a
            implied_b = 1 / odds_b
            vig = implied_a + implied_b - 1
            fair_a = implied_a / (1 + vig)
            fair_b = implied_b / (1 + vig)

        Args:
            odds_a: Decimal odds for outcome A
            odds_b: Decimal odds for outcome B

        Returns:
            (fair_prob_A, fair_prob_B) — sum to 1.0
        """
        if odds_a <= 1 or odds_b <= 1:
            return (0.5, 0.5)

        implied_a = 1.0 / odds_a
        implied_b = 1.0 / odds_b
        vig = implied_a + implied_b - 1.0

        if vig < 0:
            # Arbitrage opportunity (sum < 1)
            total = implied_a + implied_b
            return (implied_a / total, implied_b / total)

        if vig > 0.5:  # Sanity check — unrealistic vig
            return (0.5, 0.5)

        fair_a = implied_a / (1.0 + vig)
        fair_b = implied_b / (1.0 + vig)

        return (fair_a, fair_b)

    @staticmethod
    def _compute_kelly_fraction(
        win_prob: float,
        decimal_odds: float,
        fraction: float = 0.25,
        max_fraction: float = 0.10,
    ) -> float:
        """
        Compute fractional Kelly stake.

        Full Kelly: f* = (b*p - q) / b
        where b = decimal_odds - 1, p = win_prob, q = 1 - p
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        b = decimal_odds - 1.0
        if b <= 0:
            return 0.0

        p = win_prob
        q = 1.0 - p

        full_kelly = (b * p - q) / b
        if full_kelly <= 0:
            return 0.0

        result = full_kelly * fraction
        return max(0.0, min(result, max_fraction))

    @staticmethod
    def _compute_consensus_agreement(
        snapshot: dict, market_type: str = "h2h"
    ) -> tuple[float, int]:
        """
        Compute how much sportsbooks agree on a market.

        Args:
            snapshot: An odds snapshot dict
            market_type: "h2h", "spreads", "totals"

        Returns:
            (coefficient_of_variation, n_books)
            Lower CV = higher agreement. Returns (0, 0) if no data.
        """
        import statistics

        # In a real system, parse the per-bookmaker odds from the snapshot.
        # For now, use a simplified heuristic based on the spread between
        # home and away moneylines.
        home_ml = snapshot.get("home_ml")
        away_ml = snapshot.get("away_ml")
        if home_ml and away_ml:
            implied_home = 1.0 / home_ml
            implied_away = 1.0 / away_ml
            total_vig = implied_home + implied_away - 1.0
            # Lower vig = more efficient market = higher agreement
            agreement = max(0, 1.0 - total_vig * 5)  # Scale: 5% vig = 0.75
            return (agreement, 2)  # Assume at least 2 books

        return (0.0, 0)

    def _compute_clv(self, opp: EVOpportunity):
        """
        Compute Closing Line Value by comparing to historical snapshots.

        CLV is positive when the line moved in our favor after we identified
        the opportunity (i.e., our line was better than the closing line).
        """
        game_id = opp.game_id
        if game_id not in self._last_historical_snapshots:
            return

        historical = self._last_historical_snapshots[game_id]
        if not historical:
            return

        # Earliest snapshot = opening line
        opening = historical[0]
        opening_total = opening.get("total")
        opening_home_ml = opening.get("home_ml")

        if opening_total is not None and opp.market_line is not None:
            if "OVER" in opp.bet_side.upper() or "UNDER" in opp.bet_side.upper():
                opp.opening_line = opening_total
                diff = opp.market_line - opening_total
                opp.line_movement_direction = (
                    "toward" if (diff > 0 and "OVER" in opp.bet_side.upper()) or
                    (diff < 0 and "UNDER" in opp.bet_side.upper())
                    else "away"
                )
                opp.clv_pct = abs(diff) / opening_total if opening_total != 0 else None

        if opening_home_ml is not None and opp.market_line is not None:
            if opp.bet_side and opp.home_team and opp.home_team in opp.bet_side:
                opp.opening_line = opening_home_ml
                diff = opp.market_line - opening_home_ml
                opp.line_movement_direction = "toward" if diff < 0 else "away"
                opp.clv_pct = abs(diff) / opening_home_ml if opening_home_ml != 0 else None

    def _build_ml_opportunity(
        self,
        game_id: str,
        league: str,
        matchup: str,
        game_date: str,
        team: str,
        model_prob: float,
        implied_prob: float,
        decimal_odds: float,
        snapshot: dict,
    ) -> Optional[EVOpportunity]:
        """Build a moneyline EV opportunity if edge exceeds threshold."""
        edge = model_prob - implied_prob
        if edge < self.min_edge:
            return None

        ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)
        if ev < self.min_ev:
            return None

        kelly = self._compute_kelly_fraction(
            model_prob, decimal_odds, self.kelly_fraction
        )
        stake = kelly * self.bankroll
        consensus, n_books = self._compute_consensus_agreement(snapshot)

        confidence = self._determine_confidence(edge, n_books, model_prob)
        tags = self._generate_tags(edge, n_books, league)

        home_team = snapshot.get("home_team", "")
        away_team = snapshot.get("away_team", "")

        return EVOpportunity(
            game_id=game_id,
            league=league,
            matchup=matchup,
            game_date=game_date,
            bet_side=team,
            market_line=decimal_odds,
            sportsbook=f"Best of {n_books} books",
            implied_probability=implied_prob,
            best_odds_decimal=decimal_odds,
            model_probability=model_prob,
            model_source="RecommendationEngine",
            edge_pct=edge,
            expected_value=ev,
            expected_value_pct=ev / decimal_odds if decimal_odds else 0,
            kelly_fraction=kelly,
            recommended_stake_dollars=round(stake, 2),
            confidence=confidence,
            n_sportsbooks=n_books,
            consensus_agreement=consensus,
            home_team=home_team or None,
            away_team=away_team or None,
            tags=tags,
            reasoning=(
                f"{team} ML @ {decimal_odds:.2f}: model {model_prob:.1%} vs "
                f"market {implied_prob:.1%} (edge: {edge:.2%}, EV: ${ev:.2f}/$)"
            ),
            source=ScannerSource.MODEL_VS_MARKET,
        )

    def _build_total_opportunity(
        self,
        game_id: str,
        league: str,
        matchup: str,
        game_date: str,
        side: str,
        model_prob: float,
        market_total: float,
        snapshot: dict,
    ) -> Optional[EVOpportunity]:
        """Build a total points EV opportunity."""
        # Default market-implied probability: 50/50 for O/U
        implied_prob = 0.50
        edge = model_prob - implied_prob
        if edge < self.min_edge:
            return None

        # Estimate decimal odds: 1.91 (-110) standard for totals
        decimal_odds = 1.91

        ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)
        if ev < self.min_ev:
            return None

        kelly = self._compute_kelly_fraction(
            model_prob, decimal_odds, self.kelly_fraction
        )
        stake = kelly * self.bankroll
        consensus, n_books = self._compute_consensus_agreement(snapshot)
        confidence = self._determine_confidence(edge, n_books, model_prob)

        home_team = snapshot.get("home_team", "")
        away_team = snapshot.get("away_team", "")

        return EVOpportunity(
            game_id=game_id,
            league=league,
            matchup=matchup,
            game_date=game_date,
            bet_side=f"Total {side} {market_total:.0f}",
            market_line=market_total,
            sportsbook=f"Best of {n_books} books",
            implied_probability=implied_prob,
            best_odds_decimal=decimal_odds,
            model_probability=model_prob,
            model_source="RecommendationEngine",
            edge_pct=edge,
            expected_value=ev,
            expected_value_pct=ev / decimal_odds if decimal_odds else 0,
            kelly_fraction=kelly,
            recommended_stake_dollars=round(stake, 2),
            confidence=confidence,
            n_sportsbooks=n_books,
            consensus_agreement=consensus,
            home_team=home_team or None,
            away_team=away_team or None,
            tags=self._generate_tags(edge, n_books, league),
            reasoning=(
                f"Total {side} {market_total:.0f}: model {model_prob:.1%} vs "
                f"market {implied_prob:.1%} (edge: {edge:.2%})"
            ),
            source=ScannerSource.MODEL_VS_MARKET,
        )

    def _build_spread_opportunity(
        self,
        game_id: str,
        league: str,
        matchup: str,
        game_date: str,
        model_prob: float,
        spread_line: float,
        snapshot: dict,
    ) -> Optional[EVOpportunity]:
        """Build a spread EV opportunity."""
        implied_prob = 0.50  # Spreads typically priced at 50/50
        edge = model_prob - implied_prob
        if edge < self.min_edge:
            return None

        decimal_odds = 1.91  # Standard -110 pricing
        ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)
        if ev < self.min_ev:
            return None

        kelly = self._compute_kelly_fraction(
            model_prob, decimal_odds, self.kelly_fraction
        )
        stake = kelly * self.bankroll
        consensus, n_books = self._compute_consensus_agreement(snapshot)
        confidence = self._determine_confidence(edge, n_books, model_prob)

        # Determine which team is covered
        side_prefix = ""
        if spread_line < 0:
            side_prefix = snapshot.get("home_team", "")
        else:
            side_prefix = snapshot.get("away_team", "")

        home_team = snapshot.get("home_team", "")
        away_team = snapshot.get("away_team", "")

        return EVOpportunity(
            game_id=game_id,
            league=league,
            matchup=matchup,
            game_date=game_date,
            bet_side=f"{side_prefix} {spread_line:+.1f}".strip(),
            market_line=spread_line,
            sportsbook=f"Best of {n_books} books",
            implied_probability=implied_prob,
            best_odds_decimal=decimal_odds,
            model_probability=model_prob,
            model_source="RecommendationEngine",
            edge_pct=edge,
            expected_value=ev,
            expected_value_pct=ev / decimal_odds if decimal_odds else 0,
            kelly_fraction=kelly,
            recommended_stake_dollars=round(stake, 2),
            confidence=confidence,
            n_sportsbooks=n_books,
            consensus_agreement=consensus,
            home_team=home_team or None,
            away_team=away_team or None,
            tags=self._generate_tags(edge, n_books, league),
            reasoning=(
                f"Spread {spread_line:+.1f}: model {model_prob:.1%} vs "
                f"market {implied_prob:.1%} (edge: {edge:.2%})"
            ),
            source=ScannerSource.MODEL_VS_MARKET,
        )

    def _determine_confidence(
        self, edge: float, n_books: int, model_prob: float
    ) -> ScannerConfidence:
        """Determine confidence based on edge size, book consensus, and model certainty."""
        if edge >= 0.08 and n_books >= 3 and abs(model_prob - 0.5) > 0.1:
            return ScannerConfidence.VERY_HIGH
        elif edge >= 0.05 and n_books >= 2:
            return ScannerConfidence.HIGH
        elif edge >= 0.03 and n_books >= 1:
            return ScannerConfidence.MEDIUM
        elif edge >= 0.02:
            return ScannerConfidence.LOW
        else:
            return ScannerConfidence.SPECULATIVE

    def _generate_tags(self, edge: float, n_books: int, league: str) -> list[str]:
        """Generate descriptive tags for an opportunity."""
        tags = ["+ev"]
        if edge >= self.actionable_edge:
            tags.append("actionable")
        if edge >= 0.05:
            tags.append("high_edge")
        if n_books >= 3:
            tags.append("strong_consensus")
        if league != "NBA":
            tags.append("small_league")
            tags.append("inefficiency")
        return tags

    def _filter_and_rank(
        self, opportunities: list[EVOpportunity]
    ) -> list[EVOpportunity]:
        """Filter by thresholds and rank by EV descending."""
        # Filter
        result = [
            o
            for o in opportunities
            if o.edge_pct >= self.min_edge and o.expected_value >= self.min_ev
        ]
        # Rank by EV
        result.sort(key=lambda o: o.expected_value, reverse=True)
        return result

    def _build_report(
        self, opportunities: list[EVOpportunity], n_games: int
    ) -> ScannerReport:
        """Build a summary report from the filtered opportunities."""
        if not opportunities:
            return ScannerReport(total_games_scanned=n_games)

        actionable = [o for o in opportunities if o.edge_pct >= self.actionable_edge]
        speculative = [o for o in opportunities if o.edge_pct < self.actionable_edge]

        by_league: dict[str, int] = {}
        by_bet_type: dict[str, int] = {}
        by_confidence: dict[str, int] = {}
        total_stake = 0.0

        for opp in opportunities:
            by_league[opp.league] = by_league.get(opp.league, 0) + 1
            # Heuristic: infer bet type from bet_side
            if "OVER" in opp.bet_side.upper() or "UNDER" in opp.bet_side.upper():
                btype = "total_points"
            elif "+" in opp.bet_side or "-" in opp.bet_side:
                btype = "spread"
            else:
                btype = "moneyline"
            by_bet_type[btype] = by_bet_type.get(btype, 0) + 1
            by_confidence[opp.confidence.value] = (
                by_confidence.get(opp.confidence.value, 0) + 1
            )
            total_stake += opp.recommended_stake_dollars

        avg_edge = (
            sum(o.edge_pct for o in opportunities) / len(opportunities)
        )
        best_edge = max(o.edge_pct for o in opportunities)

        return ScannerReport(
            total_opportunities=len(opportunities),
            total_games_scanned=n_games,
            actionable_opportunities=len(actionable),
            speculative_opportunities=len(speculative),
            average_edge_pct=avg_edge,
            best_edge_pct=best_edge,
            total_kelly_exposure=total_stake,
            by_league=by_league,
            by_bet_type=by_bet_type,
            by_confidence=by_confidence,
            top_opportunities=opportunities[:10],
        )
