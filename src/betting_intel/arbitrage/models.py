"""
Arbitrage data models — structured arbitrage opportunities.

Each ArbitrageOpportunity represents a guaranteed-profit situation
across two or more sportsbooks for the same game/matchup.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class ArbLeg:
    """One leg of an arbitrage opportunity — a single bet to place."""

    bookmaker: str          # Sportsbook key (e.g., "draftkings", "fanduel")
    team: str               # Which side to back (team name, "Over", "Under", "Draw")
    market: str             # "h2h", "spread", "total"
    point: Optional[float]  # For spreads/totals (e.g., -3.5, 225.5)
    price: int              # American odds (e.g., -110, +150)
    decimal_odds: float     # Decimal odds (e.g., 1.91, 2.50)
    stake_pct: float        # % of total stake to allocate to this leg (0-1)
    stake_dollars: float    # Dollar amount for a standard $1000 total stake

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArbitrageOpportunity:
    """A guaranteed-profit opportunity across sportsbooks."""

    id: str                         # Unique identifier
    game_id: str                    # TheOddsAPI game ID
    matchup: str                    # "Celtics @ Lakers"
    sport_key: str                  # "basketball_nba"
    league: str                     # "NBA"
    commence_time: str              # ISO 8601
    game_date: str                  # YYYY-MM-DD

    arb_type: str                   # "standard_2way", "three_way", "totals"
    legs: list[ArbLeg]              # The bets to place (2 or 3 legs)

    total_implied_prob: float       # Sum of implied probabilities (< 1.0 = arb)
    profit_pct: float               # Guaranteed profit % (e.g., 0.0315 = 3.15%)
    profit_per_1k: float            # $ profit per $1000 total stake
    n_books: int                    # Number of distinct books involved
    depth: int                      # Number of legs (2 for standard, 3 for soccer)

    # Timestamp
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["legs"] = [leg.to_dict() for leg in self.legs]
        return d

    @property
    def profit_grade(self) -> str:
        """Letter grade for the arbitrage opportunity."""
        pct = self.profit_pct
        if pct >= 0.10:
            return "A+"
        elif pct >= 0.05:
            return "A"
        elif pct >= 0.03:
            return "B"
        elif pct >= 0.02:
            return "C"
        elif pct >= 0.01:
            return "D"
        return "F"
