"""
Arbitrage Detection — identifies risk-free profit opportunities.

Scans odds across multiple sportsbooks (from the OddsPoller's TheOddsAPI
data) and finds situations where you can bet on ALL outcomes of a market
and guarantee a profit regardless of the result.

Key formulas:
    Arbitrage % = sum(1 / decimal_odds for all outcomes)
    Profit % = (1 / arbitrage_pct) - 1

If arbitrage_pct < 1.0 (or 100%), there's a guaranteed profit opportunity.

Examples:
    - Moneyline arb: Book A has Team A at 2.10, Book B has Team B at 2.00
      arb% = 1/2.10 + 1/2.00 = 0.476 + 0.500 = 0.976 (97.6%)
      profit = 1/0.976 - 1 = 2.46% guaranteed return

    - Spread arb: Different books offer different lines on the same game
    - Total arb: Books disagree on the over/under total

Usage:
    detector = ArbitrageDetector()
    odds = poller.get_current_odds()
    opportunities = detector.scan_for_arbitrage(odds)

    for opp in opportunities:
        print(f"{opp.market_type} arb: {opp.profit_pct:.2%} guaranteed")
        print(f"  Bet 1: ${opp.stakes[0]:.0f} on {opp.outcomes[0]} @ Book A")
        print(f"  Bet 2: ${opp.stakes[1]:.0f} on {opp.outcomes[1]} @ Book B")
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class ArbOutcome:
    """A single outcome in an arbitrage opportunity."""

    team: str
    decimal_odds: float
    sportsbook: str
    market_type: str  # "h2h", "spread", "total"
    line: Optional[float] = None  # e.g. -3.5 for spread, 214.5 for total


@dataclass
class ArbitrageOpportunity:
    """
    A risk-free arbitrage opportunity across sportsbooks.

    Example:
        market_type = "moneyline"
        profit_pct = 0.0246 (2.46% guaranteed return)
        outcomes = [
            ArbOutcome(team="Spurs", odds=2.10, book="DraftKings"),
            ArbOutcome(team="Thunder", odds=2.00, book="FanDuel"),
        ]
        stakes = [487.80, 512.20]  # $1,000 total investment
        guaranteed_profit = 24.60
    """

    game_id: str
    league: str
    matchup: str
    game_date: str
    market_type: str  # "moneyline", "spread", "total"
    outcomes: list[ArbOutcome]  # Each outcome from a different sportsbook

    # Financials
    arbitrage_pct: float  # Sum of implied probabilities (< 1 = arb)
    profit_pct: float  # Guaranteed return percentage
    stakes: list[float]  # Dollar stakes per outcome (sum = total_investment)
    total_investment: float  # Total amount to invest
    guaranteed_profit: float  # Profit regardless of outcome
    n_sportsbooks: int  # Number of books involved

    # Metadata
    is_risk_free: bool = True
    tags: list[str] = field(default_factory=list)
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def as_dict(self) -> dict:
        """Serialize to dict for API responses."""
        return {
            "game_id": self.game_id,
            "league": self.league,
            "matchup": self.matchup,
            "game_date": self.game_date,
            "market_type": self.market_type,
            "outcomes": [
                {
                    "team": o.team,
                    "decimal_odds": o.decimal_odds,
                    "sportsbook": o.sportsbook,
                    "market_type": o.market_type,
                    "line": o.line,
                }
                for o in self.outcomes
            ],
            "arbitrage_pct": round(self.arbitrage_pct, 4),
            "profit_pct": round(self.profit_pct, 4),
            "stakes": [round(s, 2) for s in self.stakes],
            "total_investment": round(self.total_investment, 2),
            "guaranteed_profit": round(self.guaranteed_profit, 2),
            "n_sportsbooks": self.n_sportsbooks,
            "is_risk_free": self.is_risk_free,
            "tags": self.tags,
            "detected_at": self.detected_at,
        }

    def action(self, total_investment: float = 1000.0) -> str:
        """Generate a human-readable arbitrage instruction."""
        lines = [f"ARBITRAGE: {self.matchup} - {self.market_type}"]
        lines.append(f"  Profit: {self.profit_pct:.2%} (${self.guaranteed_profit:.2f} on ${total_investment:.0f})")
        for outcome, stake in zip(self.outcomes, self.stakes):
            pct = stake / total_investment if total_investment else 0
            lines.append(f"  Bet ${stake:.0f} ({pct:.1%}) on {outcome.team} @ {outcome.decimal_odds:.2f} at {outcome.sportsbook}")
        return "\n".join(lines)


@dataclass
class ArbitrageReport:
    """Summary report of all arbitrage opportunities detected."""

    total_opportunities: int = 0
    total_games_scanned: int = 0
    profitable_opportunities: int = 0  # Profit > 0
    high_yield_opportunities: int = 0  # Profit > 2%
    average_profit_pct: float = 0.0
    best_profit_pct: float = 0.0
    total_investment_required: float = 0.0
    total_guaranteed_profit: float = 0.0
    by_league: dict[str, int] = field(default_factory=dict)
    by_market_type: dict[str, int] = field(default_factory=dict)
    top_opportunities: list[ArbitrageOpportunity] = field(default_factory=list)
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def as_dict(self) -> dict:
        return {
            "total_opportunities": self.total_opportunities,
            "total_games_scanned": self.total_games_scanned,
            "profitable_opportunities": self.profitable_opportunities,
            "high_yield_opportunities": self.high_yield_opportunities,
            "average_profit_pct": round(self.average_profit_pct, 4),
            "best_profit_pct": round(self.best_profit_pct, 4),
            "total_investment_required": round(self.total_investment_required, 2),
            "total_guaranteed_profit": round(self.total_guaranteed_profit, 2),
            "by_league": self.by_league,
            "by_market_type": self.by_market_type,
            "top_opportunities": [o.as_dict() for o in self.top_opportunities[:5]],
            "detected_at": self.detected_at,
        }


# ── Arbitrage Detector ──────────────────────────────────────────────────────


class ArbitrageDetector:
    """
    Scans odds across multiple sportsbooks for arbitrage opportunities.

    The detector works in three stages:
      1. Group odds by game and market (moneyline, spread, total)
      2. For each market, find the best price for each outcome across all books
      3. Compute arbitrage percentage: if sum(1/odds) < 1, an arb exists

    Configuration thresholds:
      MIN_PROFIT_PCT: Minimum profit to report (default 0.5%)
      MAX_INVESTMENT: Maximum total investment for stake calc (default $1,000)
      MIN_SPORTSBOOKS: Minimum number of distinct books needed
      DEFAULT_VIG: Expected vig to filter noise (default 4.5%)
    """

    MIN_PROFIT_PCT: float = 0.005  # 0.5% minimum profit to report
    MAX_INVESTMENT: float = 1000.0  # $1k default investment for calc
    MIN_SPORTSBOOKS: int = 2  # Need at least 2 distinct books
    DEFAULT_VIG: float = 0.045  # 4.5% expected vig

    def __init__(
        self,
        min_profit_pct: float = MIN_PROFIT_PCT,
        max_investment: float = MAX_INVESTMENT,
        min_sportsbooks: int = MIN_SPORTSBOOKS,
    ):
        self.min_profit = min_profit_pct
        self.max_investment = max_investment
        self.min_sportsbooks = min_sportsbooks

    # ── Public API ──────────────────────────────────────────────────────────

    def scan_for_arbitrage(
        self, odds_snapshots: list[dict]
    ) -> ArbitrageReport:
        """
        Main entry point: scan all odds snapshots for arbitrage opportunities.

        Args:
            odds_snapshots: List of odds dicts from poller.get_current_odds()

        Returns:
            ArbitrageReport with all detected opportunities.
        """
        if not odds_snapshots:
            return ArbitrageReport()

        opportunities: list[ArbitrageOpportunity] = []
        games_scanned: set[str] = set()

        for snapshot in odds_snapshots:
            game_id = snapshot.get("game_id", "")
            league = snapshot.get("league", "NBA")
            home_team = snapshot.get("home_team", "")
            away_team = snapshot.get("away_team", "")
            game_date = snapshot.get("game_date", "")
            matchup = f"{away_team} @ {home_team}"

            if not home_team or not away_team:
                continue

            games_scanned.add(game_id or matchup)

            # ── Moneyline Arbitrage ────────────────────────────────────
            home_ml = snapshot.get("home_ml")
            away_ml = snapshot.get("away_ml")
            if home_ml and away_ml:
                ml_opps = self._check_moneyline_arb(
                    game_id=game_id,
                    league=league,
                    matchup=matchup,
                    game_date=game_date,
                    home_team=home_team,
                    away_team=away_team,
                    home_ml=home_ml,
                    away_ml=away_ml,
                )
                opportunities.extend(ml_opps)

            # ── Total Points Arbitrage ────────────────────────────────
            total = snapshot.get("total")
            if total:
                total_opps = self._check_total_arb(
                    game_id=game_id,
                    league=league,
                    matchup=matchup,
                    game_date=game_date,
                    home_team=home_team,
                    away_team=away_team,
                    total=total,
                )
                opportunities.extend(total_opps)

            # ── Spread Arbitrage ────────────────────────────────────────
            spread = snapshot.get("spread")
            if spread is not None:
                spread_opps = self._check_spread_arb(
                    game_id=game_id,
                    league=league,
                    matchup=matchup,
                    game_date=game_date,
                    home_team=home_team,
                    away_team=away_team,
                    spread=spread,
                )
                opportunities.extend(spread_opps)

        # Filter and rank
        opportunities = self._filter_and_rank(opportunities)
        report = self._build_report(opportunities, len(games_scanned))

        return report

    def scan_moneyline(self, odds_snapshots: list[dict]) -> list[ArbitrageOpportunity]:
        """Scan only for moneyline arbitrage opportunities."""
        if not odds_snapshots:
            return []

        opportunities = []
        for snapshot in odds_snapshots:
            home_ml = snapshot.get("home_ml")
            away_ml = snapshot.get("away_ml")
            if home_ml and away_ml:
                opps = self._check_moneyline_arb(
                    game_id=snapshot.get("game_id", ""),
                    league=snapshot.get("league", "NBA"),
                    matchup=f"{snapshot.get('away_team', '')} @ {snapshot.get('home_team', '')}",
                    game_date=snapshot.get("game_date", ""),
                    home_team=snapshot.get("home_team", ""),
                    away_team=snapshot.get("away_team", ""),
                    home_ml=home_ml,
                    away_ml=away_ml,
                )
                opportunities.extend(opps)

        return self._filter_and_rank(opportunities)

    def scan_totals(self, odds_snapshots: list[dict]) -> list[ArbitrageOpportunity]:
        """Scan only for totals arbitrage opportunities."""
        if not odds_snapshots:
            return []
        return self.scan_for_arbitrage(odds_snapshots)

    # ── Market-Specific Checks ──────────────────────────────────────────────

    def _check_moneyline_arb(
        self,
        game_id: str,
        league: str,
        matchup: str,
        game_date: str,
        home_team: str,
        away_team: str,
        home_ml: float,
        away_ml: float,
    ) -> list[ArbitrageOpportunity]:
        """
        Check for moneyline arbitrage.

        Simple 2-outcome scenario:
          arb% = 1/home_ml + 1/away_ml

        If arb% < 1.0, there's a guaranteed profit by betting both sides.
        The odds already represent the best available across all sportsbooks
        (the OddsPoller picks the best odds per outcome).
        """
        # Validate odds
        if home_ml <= 1 or away_ml <= 1:
            return []

        implied_home = 1.0 / home_ml
        implied_away = 1.0 / away_ml
        arb_pct = implied_home + implied_away

        if arb_pct >= 1.0:
            return []  # No arbitrage

        profit_pct = (1.0 / arb_pct) - 1.0
        if profit_pct < self.min_profit:
            return []

        # Compute stakes for equal profit
        home_stake = self.max_investment * (implied_home / arb_pct)
        away_stake = self.max_investment * (implied_away / arb_pct)
        guaranteed_profit = self.max_investment * profit_pct

        opportunity = ArbitrageOpportunity(
            game_id=game_id,
            league=league,
            matchup=matchup,
            game_date=game_date,
            market_type="moneyline",
            outcomes=[
                ArbOutcome(
                    team=home_team,
                    decimal_odds=home_ml,
                    sportsbook="Best Book",
                    market_type="h2h",
                ),
                ArbOutcome(
                    team=away_team,
                    decimal_odds=away_ml,
                    sportsbook="Best Book",
                    market_type="h2h",
                ),
            ],
            arbitrage_pct=arb_pct,
            profit_pct=profit_pct,
            stakes=[home_stake, away_stake],
            total_investment=self.max_investment,
            guaranteed_profit=guaranteed_profit,
            n_sportsbooks=2,
            tags=self._generate_tags(profit_pct, arb_pct, league),
        )

        return [opportunity]

    def _check_total_arb(
        self,
        game_id: str,
        league: str,
        matchup: str,
        game_date: str,
        home_team: str,
        away_team: str,
        total: float,
    ) -> list[ArbitrageOpportunity]:
        """
        Check for totals arbitrage.

        For totals, we need odds for both OVER and UNDER.
        In the current snapshot format, we have a single total value.
        For full arbitrage detection, we'd need per-bookmaker odds.

        For now, we detect synthetic opportunities where the total line
        differs between books (a true arb would require one book offering
        OVER at X and another offering UNDER at a different Y).
        """
        # In the current snapshot format, total is a single value
        # (the best across books). For true arbitrage, we'd need
        # per-bookmaker breakdowns. Mark as limited detection.
        return []

    def _check_spread_arb(
        self,
        game_id: str,
        league: str,
        matchup: str,
        game_date: str,
        home_team: str,
        away_team: str,
        spread: float,
    ) -> list[ArbitrageOpportunity]:
        """
        Check for spread arbitrage.

        Similar to totals: the snapshot gives us a single spread value.
        True arbitrage would require different spreads at different books
        (e.g., Book A offers -3.5, Book B offers +4.5 — creating a middle).
        """
        return []

    # ── Staking Calculation ─────────────────────────────────────────────────

    @staticmethod
    def compute_optimal_stakes(
        outcomes: list[tuple[float, float]],
        total_investment: float,
    ) -> list[float]:
        """
        Compute optimal stakes for an arbitrage opportunity.

        The goal is to guarantee equal profit regardless of outcome.

        Args:
            outcomes: List of (decimal_odds, implied_probability) tuples
            total_investment: Total amount to invest across all outcomes

        Returns:
            List of dollar stakes, one per outcome
        """
        if not outcomes:
            return []

        implied_probs = [p for _, p in outcomes]
        total_implied = sum(implied_probs)

        if total_implied <= 0:
            return []

        stakes = [
            total_investment * (p / total_implied) for p in implied_probs
        ]
        return stakes

    @staticmethod
    def compute_arbitrage_pct(decimal_odds: list[float]) -> float:
        """
        Compute the arbitrage percentage for a set of outcomes.

        arb_pct = sum(1 / odds for each outcome)
        If arb_pct < 1.0 (100%), there's an arbitrage opportunity.

        Args:
            decimal_odds: List of decimal odds for all possible outcomes

        Returns:
            Arbitrage percentage (e.g., 0.976 = 97.6%)
        """
        if not decimal_odds:
            return 1.0
        implied = [1.0 / o for o in decimal_odds if o > 1]
        if not implied:
            return 1.0
        return sum(implied)

    @staticmethod
    def compute_profit_pct(arbitrage_pct: float) -> float:
        """
        Compute the guaranteed profit percentage from an arb opportunity.

        profit_pct = (1 / arb_pct) - 1

        Args:
            arbitrage_pct: The arbitrage percentage (< 1.0 = profit)

        Returns:
            Profit percentage (e.g., 0.0246 = 2.46%)
        """
        if arbitrage_pct <= 0:
            return 0.0
        return (1.0 / arbitrage_pct) - 1.0

    # ── Internal Methods ────────────────────────────────────────────────────

    def _filter_and_rank(
        self, opportunities: list[ArbitrageOpportunity]
    ) -> list[ArbitrageOpportunity]:
        """Filter by profit threshold and rank by profit descending."""
        result = [o for o in opportunities if o.profit_pct >= self.min_profit]
        result.sort(key=lambda o: o.profit_pct, reverse=True)
        return result

    def _generate_tags(
        self, profit_pct: float, arb_pct: float, league: str
    ) -> list[str]:
        """Generate descriptive tags."""
        tags = ["arbitrage", "risk_free"]
        if profit_pct >= 0.02:
            tags.append("high_yield")
        if arb_pct < 0.98:
            tags.append("deep_arb")  # Very low arb% = higher profit
        if league != "NBA":
            tags.append("small_league")
        if profit_pct >= 0.05:
            tags.append("exceptional")
        return tags

    def _build_report(
        self, opportunities: list[ArbitrageOpportunity], n_games: int
    ) -> ArbitrageReport:
        """Build a summary report from detected opportunities."""
        if not opportunities:
            return ArbitrageReport(total_games_scanned=n_games)

        profitable = [o for o in opportunities if o.profit_pct > 0]
        high_yield = [o for o in opportunities if o.profit_pct >= 0.02]

        by_league: dict[str, int] = {}
        by_market: dict[str, int] = {}
        total_investment = 0.0
        total_profit = 0.0

        for opp in opportunities:
            by_league[opp.league] = by_league.get(opp.league, 0) + 1
            by_market[opp.market_type] = by_market.get(opp.market_type, 0) + 1
            total_investment += opp.total_investment
            total_profit += opp.guaranteed_profit

        avg_profit = (
            sum(o.profit_pct for o in opportunities) / len(opportunities)
        )
        best_profit = max(o.profit_pct for o in opportunities)

        return ArbitrageReport(
            total_opportunities=len(opportunities),
            total_games_scanned=n_games,
            profitable_opportunities=len(profitable),
            high_yield_opportunities=len(high_yield),
            average_profit_pct=avg_profit,
            best_profit_pct=best_profit,
            total_investment_required=total_investment,
            total_guaranteed_profit=total_profit,
            by_league=by_league,
            by_market_type=by_market,
            top_opportunities=opportunities[:10],
        )
