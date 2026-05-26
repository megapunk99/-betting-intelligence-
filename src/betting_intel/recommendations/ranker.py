"""
Bet ranking engine — identifies "clear picks" and ranks all bets by edge.

A "Clear Pick" must satisfy ALL of:
  1. Edge >= 3%
  2. Win probability >= 55%
  3. HIGH or VERY HIGH confidence
  4. Kelly stake >= $10 (meaningful bet size)
  5. Not a parlay (individual bets only)

Usage:
    ranker = BetRanker()
    clear = ranker.get_clear_picks(all_bets)
    ranked = ranker.rank_bets(all_bets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from betting_intel.recommendations.bet_types import BetSuggestion, Confidence, BetType


@dataclass
class ClearPick:
    """
    A high-confidence betting recommendation that meets strict criteria.

    These are the bets the system is most confident about. In a real
    sportsbook scenario, these would be the first bets to place.
    """

    bet: BetSuggestion
    clear_score: float  # Composite score (0-100)
    reasons: list[str] = field(default_factory=list)
    risk_level: str = "MODERATE"  # CONSERVATIVE | MODERATE | AGGRESSIVE

    def as_dict(self) -> dict:
        return {
            **self.bet.as_dict(),
            "clear_score": self.clear_score,
            "reasons": self.reasons,
            "risk_level": self.risk_level,
        }


class BetRanker:
    """
    Ranks bets by edge, computes composite scores, and identifies
    "clear picks" — bets that meet strict confidence criteria.
    """

    # Thresholds for clear pick classification
    MIN_EDGE = 0.03  # 3% minimum edge
    MIN_WIN_PROB = 0.55  # 55% minimum win probability
    MIN_CONFIDENCE = Confidence.HIGH
    MIN_STAKE = 10.0  # $10 minimum stake

    def __init__(
        self,
        min_edge: float = MIN_EDGE,
        min_win_prob: float = MIN_WIN_PROB,
        min_confidence: Confidence = MIN_CONFIDENCE,
        min_stake: float = MIN_STAKE,
    ):
        self.min_edge = min_edge
        self.min_win_prob = min_win_prob
        self.min_confidence = min_confidence
        self.min_stake = min_stake

    def rank_bets(self, bets: list[BetSuggestion]) -> list[BetSuggestion]:
        """
        Rank and tag all bets. Computes:
          - Composite edge score
          - Clear pick status
          - Confidence tags
          - Risk level
        """
        for bet in bets:
            # Compute composite score (0-100)
            bet.metadata["composite_score"] = self._compute_score(bet)

            # Tag as clear pick if it meets criteria
            bet.is_clear_pick = self._is_clear_pick(bet)

            # Generate tags
            tags = []
            if bet.is_clear_pick:
                tags.append("clear_pick")
            if bet.edge_pct >= 0.05:
                tags.append("high_edge")
            if bet.win_probability >= 0.65:
                tags.append("high_prob")
            if bet.confidence in (Confidence.VERY_HIGH, Confidence.HIGH):
                tags.append("confident")
            if bet.league != "NBA":
                tags.append("small_league")
            if bet.bet_type in (BetType.PLAYER_POINTS, BetType.PLAYER_REBOUNDS, BetType.PLAYER_ASSISTS, BetType.PLAYER_PRA):
                tags.append("player_prop")
            bet.tags = tags

        # Sort by composite score descending
        bets.sort(key=lambda b: b.metadata.get("composite_score", 0), reverse=True)
        return bets

    def get_clear_picks(
        self,
        bets: list[BetSuggestion],
        min_edge: Optional[float] = None,
    ) -> list[ClearPick]:
        """
        Extract only the bets that qualify as "clear picks" — high-confidence,
        high-edge opportunities that are the system's strongest recommendations.

        Args:
            bets: Full list of bet suggestions
            min_edge: Override minimum edge threshold

        Returns:
            List of ClearPick objects, sorted by score
        """
        effective_min_edge = min_edge if min_edge is not None else self.min_edge

        clear: list[ClearPick] = []
        for bet in bets:
            if bet.edge_pct < effective_min_edge:
                continue
            if bet.win_probability < self.min_win_prob:
                continue
            if self._confidence_value(bet.confidence) < self._confidence_value(self.min_confidence):
                continue
            if bet.stake_dollars < self.min_stake:
                continue
            if bet.bet_type == BetType.PARLAY:
                continue

            score = self._compute_score(bet)
            reasons = self._generate_reasons(bet)
            risk = self._determine_risk(bet)

            clear.append(ClearPick(
                bet=bet,
                clear_score=score,
                reasons=reasons,
                risk_level=risk,
            ))

        clear.sort(key=lambda c: c.clear_score, reverse=True)
        return clear

    def get_summary(self, bets: list[BetSuggestion]) -> dict:
        """Get a summary of the ranked bets."""
        ranked = self.rank_bets(bets)
        clear = self.get_clear_picks(ranked)

        return {
            "total": len(ranked),
            "clear_picks": len(clear),
            "avg_edge": float(np.mean([b.edge_pct for b in ranked])) if ranked else 0,
            "avg_confidence": float(np.mean([self._confidence_value(b.confidence) for b in ranked])) if ranked else 0,
            "total_stake": sum(b.stake_dollars for b in ranked),
            "by_type": self._count_by(ranked, lambda b: b.bet_type.value),
            "by_confidence": self._count_by(ranked, lambda b: b.confidence.value),
            "clear_picks_detail": [c.as_dict() for c in clear],
        }

    # ── Internal Scoring ──────────────────────────────────────────────

    def _compute_score(self, bet: BetSuggestion) -> float:
        """
        Composite score (0-100) combining edge, probability, and confidence.

        Formula:
            edge_component   = min(edge_pct / 0.10, 1.0) * 40  (max 40 pts)
            prob_component   = (win_probability - 0.5) * 200   (max 40 pts)
            conf_component   = confidence_numeric * 20         (max 20 pts)

            Total = edge + prob + conf (max 100)
        """
        edge_score = min(bet.edge_pct / 0.10, 1.0) * 40
        prob_score = max(0, (bet.win_probability - 0.5) * 200)
        conf_score = self._confidence_value(bet.confidence) * 20

        return min(edge_score + prob_score + conf_score, 100)

    def _is_clear_pick(self, bet: BetSuggestion) -> bool:
        """Check if a bet qualifies as a clear pick."""
        if bet.edge_pct < self.min_edge:
            return False
        if bet.win_probability < self.min_win_prob:
            return False
        if self._confidence_value(bet.confidence) < self._confidence_value(self.min_confidence):
            return False
        if bet.stake_dollars < self.min_stake:
            return False
        if bet.bet_type == BetType.PARLAY:
            return False
        return True

    def _generate_reasons(self, bet: BetSuggestion) -> list[str]:
        """Generate human-readable reasons why this is a clear pick."""
        reasons = []
        if bet.edge_pct >= 0.05:
            reasons.append(f"Exceptionally high edge ({bet.edge_pct:.1%})")
        elif bet.edge_pct >= 0.03:
            reasons.append(f"Strong edge ({bet.edge_pct:.1%})")

        if bet.win_probability >= 0.65:
            reasons.append(f"High win probability ({bet.win_probability:.0%})")
        elif bet.win_probability >= 0.55:
            reasons.append(f"Positive win probability ({bet.win_probability:.0%})")

        if bet.confidence == Confidence.VERY_HIGH:
            reasons.append("Maximum confidence rating")
        elif bet.confidence == Confidence.HIGH:
            reasons.append("High confidence rating")

        if bet.model_name:
            reasons.append(f"Model signal: {bet.model_name}")

        if bet.bet_type in (BetType.MONEYLINE, BetType.FIRST_QUARTER_WINNER):
            reasons.append("Historically profitable bet type in backtesting")

        if bet.league != "NBA":
            reasons.append(f"Small-league market inefficiency ({bet.league})")

        return reasons[:4]  # Max 4 reasons

    def _determine_risk(self, bet: BetSuggestion) -> str:
        """Determine risk level for a clear pick."""
        if bet.edge_pct >= 0.06 and bet.win_probability >= 0.65:
            return "AGGRESSIVE"
        elif bet.edge_pct >= 0.04 and bet.win_probability >= 0.58:
            return "MODERATE"
        else:
            return "CONSERVATIVE"

    def _confidence_value(self, conf: Confidence) -> float:
        """Convert confidence enum to numeric value (0-1)."""
        mapping = {
            Confidence.VERY_HIGH: 1.0,
            Confidence.HIGH: 0.8,
            Confidence.MEDIUM: 0.5,
            Confidence.LOW: 0.25,
            Confidence.VERY_LOW: 0.1,
        }
        return mapping.get(conf, 0.5)

    def _count_by(self, bets: list[BetSuggestion], key_fn) -> dict:
        """Count bets by a key function."""
        counts = {}
        for b in bets:
            key = key_fn(b)
            counts[key] = counts.get(key, 0) + 1
        return counts
