"""
ArbitrageDetector — finds arbitrage opportunities where combined implied
probability across outcomes is less than 100%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ArbitrageOutcome:
    """One outcome in an arbitrage opportunity."""
    team: str
    odds: int
    stake_pct: float


@dataclass
class ArbitrageOpportunity:
    """A single arbitrage opportunity."""
    game: str
    return_pct: float
    outcomes: List[ArbitrageOutcome] = field(default_factory=list)
    stakes: Dict[str, float] = field(default_factory=dict)


@dataclass
class ArbitrageReport:
    """Collection of arbitrage opportunities."""
    opportunities: List[ArbitrageOpportunity] = field(default_factory=list)


def _odds_to_decimal(american: int) -> float:
    """Convert American odds to decimal."""
    if american > 0:
        return 1.0 + american / 100.0
    elif american < 0:
        return 1.0 + 100.0 / abs(american)
    return 1.909


class ArbitrageDetector:
    """Detects arbitrage opportunities across betting markets."""

    MIN_RETURN_PCT = 0.01  # Minimum 1% return to report

    def scan_for_arbitrage(self, records: List[dict]) -> ArbitrageReport:
        """Scan game records for arbitrage opportunities.

        Checks for:
        1. Home vs Away moneyline arbitrage
        2. Over/Under total arbitrage

        Args:
            records: List of game dicts with odds data.

        Returns:
            ArbitrageReport with any opportunities found.
        """
        opportunities: List[ArbitrageOpportunity] = []

        for rec in records:
            home = rec.get("home_team", rec.get("team", ""))
            away = rec.get("away_team", rec.get("opponent", ""))
            game_str = f"{away} @ {home}" if away and home else home or "?"

            # ── Moneyline arbitrage ────────────────────────────────────
            home_odds = rec.get("home_ml_odds", rec.get("home_odds", None))
            away_odds = rec.get("away_ml_odds", rec.get("away_odds", None))

            if home_odds and away_odds and home_odds != away_odds:
                try:
                    home_dec = _odds_to_decimal(home_odds)
                    away_dec = _odds_to_decimal(away_odds)
                    implied_sum = 1.0 / home_dec + 1.0 / away_dec

                    if implied_sum < 1.0:
                        return_pct = (1.0 / implied_sum - 1.0) * 100
                        if return_pct >= self.MIN_RETURN_PCT:
                            # Calculate stakes for guaranteed profit
                            stake_home = 1.0 / home_dec / implied_sum
                            stake_away = 1.0 / away_dec / implied_sum

                            opp = ArbitrageOpportunity(
                                game=game_str,
                                return_pct=round(return_pct, 2),
                                outcomes=[
                                    ArbitrageOutcome(team=home, odds=home_odds, stake_pct=round(stake_home * 100, 1)),
                                    ArbitrageOutcome(team=away, odds=away_odds, stake_pct=round(stake_away * 100, 1)),
                                ],
                                stakes={
                                    home: round(stake_home, 4),
                                    away: round(stake_away, 4),
                                },
                            )
                            opportunities.append(opp)
                except (ValueError, TypeError, ZeroDivisionError):
                    pass

            # ── Over/Under arbitrage (less common, quick check) ───────
            over_odds = rec.get("over_odds", None)
            under_odds = rec.get("under_odds", None)
            if over_odds and under_odds and over_odds != under_odds:
                try:
                    over_dec = _odds_to_decimal(over_odds)
                    under_dec = _odds_to_decimal(under_odds)
                    implied_sum = 1.0 / over_dec + 1.0 / under_dec

                    if implied_sum < 1.0:
                        return_pct = (1.0 / implied_sum - 1.0) * 100
                        if return_pct >= self.MIN_RETURN_PCT:
                            stake_over = 1.0 / over_dec / implied_sum
                            stake_under = 1.0 / under_dec / implied_sum

                            opp = ArbitrageOpportunity(
                                game=game_str,
                                return_pct=round(return_pct, 2),
                                outcomes=[
                                    ArbitrageOutcome(team="OVER", odds=over_odds, stake_pct=round(stake_over * 100, 1)),
                                    ArbitrageOutcome(team="UNDER", odds=under_odds, stake_pct=round(stake_under * 100, 1)),
                                ],
                                stakes={
                                    "OVER": round(stake_over, 4),
                                    "UNDER": round(stake_under, 4),
                                },
                            )
                            opportunities.append(opp)
                except (ValueError, TypeError, ZeroDivisionError):
                    pass

        return ArbitrageReport(opportunities=opportunities)
