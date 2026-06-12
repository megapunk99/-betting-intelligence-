"""
PositiveEVScanner — finds +EV opportunities by comparing model probabilities
against market-implied probabilities from sportsbook odds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class EVOpportunity:
    """A single +EV betting opportunity."""

    game: str
    bet_type: str
    expected_value: float
    confidence: str
    source: str = "model_vs_market"
    edge_pct: float = 0.0
    model_prob: float = 0.0
    market_implied: float = 0.0
    odds: int = -110


@dataclass
class EVReport:
    """Collection of +EV opportunities."""
    opportunities: List[EVOpportunity] = field(default_factory=list)


def _odds_to_implied(odds: int) -> float:
    """Convert American odds to implied probability (0-1)."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 0.5


def _decimal_payout(odds: int) -> float:
    """Convert American odds to decimal payout multiplier."""
    if odds > 0:
        return 1.0 + odds / 100.0
    elif odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.909


class PositiveEVScanner:
    """Scans predictions and finds +EV opportunities vs market odds."""

    def scan_odds_snapshots(self, records: List[dict]) -> EVReport:
        """Scan a list of game records for +EV opportunities.

        Args:
            records: List of dicts, each representing a game with model
                    predictions and market odds.

        Returns:
            EVReport containing any +EV opportunities found.
        """
        opportunities: List[EVOpportunity] = []

        for rec in records:
            home = rec.get("home_team", rec.get("team", ""))
            away = rec.get("away_team", rec.get("opponent", ""))
            game_str = f"{away} @ {home}" if away and home else home or "?"

            # ── Total O/U EV ──────────────────────────────────────────
            predicted_total = rec.get("predicted_total", 0)
            market_total = rec.get("market_total", 0)
            if predicted_total and market_total:
                edge = (predicted_total - market_total) / market_total
                if abs(edge) > 0.01:
                    win_prob = 1.0 / (1.0 + math.exp(-edge * 5.0))
                    implied = 0.5
                    over_odds = rec.get("over_odds", -110)
                    payout = _decimal_payout(over_odds)
                    ev = win_prob * (payout - 1.0) - (1.0 - win_prob) * 1.0
                    if ev > 0:
                        side = "OVER" if edge > 0 else "UNDER"
                        opportunities.append(EVOpportunity(
                            game=game_str,
                            bet_type=f"total_{side}",
                            expected_value=ev,
                            confidence="HIGH" if ev > 0.05 else "MEDIUM",
                            edge_pct=abs(edge),
                            model_prob=win_prob,
                            market_implied=implied,
                            odds=over_odds,
                        ))

            # ── Moneyline EV ──────────────────────────────────────────
            for col_prefix, side_label in [("home_ml", "ML"), ("away_ml", "ML_opp")]:
                ml_odds = rec.get(f"{col_prefix}_odds", rec.get(col_prefix, None))
                if ml_odds and ml_odds != -110:
                    implied_prob = _odds_to_implied(ml_odds)
                    # Derive model win prob from prediction data
                    predicted_spread = rec.get("predicted_spread", None)
                    win_prob = None
                    if predicted_spread is not None:
                        try:
                            ps = float(predicted_spread)
                            if side_label == "ML":
                                win_prob = 1.0 / (1.0 + math.exp(-ps * 0.08))
                            else:
                                win_prob = 1.0 / (1.0 + math.exp(ps * 0.08))
                        except (ValueError, TypeError):
                            pass
                    if win_prob and win_prob > implied_prob:
                        payout = _decimal_payout(ml_odds)
                        ev = win_prob * (payout - 1.0) - (1.0 - win_prob) * 1.0
                        if ev > 0:
                            team = home if side_label == "ML" else away
                            opportunities.append(EVOpportunity(
                                game=game_str,
                                bet_type=f"moneyline_{team}",
                                expected_value=ev,
                                confidence="HIGH" if ev > 0.05 else "MEDIUM",
                                model_prob=win_prob,
                                market_implied=implied_prob,
                                odds=ml_odds,
                            ))

        return EVReport(opportunities=opportunities)
