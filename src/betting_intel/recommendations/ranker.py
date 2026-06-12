"""
BetRanker — ranks betting suggestions by edge and identifies clear picks.
"""

from __future__ import annotations

from typing import Any, List, Optional

from betting_intel.recommendations.bet_types import BetSuggestion, Confidence


class BetRanker:
    """Ranks bet suggestions and identifies high-confidence picks."""

    MIN_EDGE = 0.02  # Minimum edge (2%) to be considered a clear pick

    def rank_bets(self, bets: List[BetSuggestion]) -> List[BetSuggestion]:
        """Rank bets by absolute edge (descending)."""
        return sorted(bets, key=lambda b: abs(b.edge_pct), reverse=True)

    def get_clear_picks(
        self,
        bets: List[BetSuggestion],
        threshold: float = MIN_EDGE,
    ) -> List[BetSuggestion]:
        """Return only clear picks: high/very-high confidence with sufficient edge."""
        clear = []
        for b in bets:
            if b.is_clear_pick:
                clear.append(b)
            elif b.confidence.is_clear() and abs(b.edge_pct) >= threshold:
                b.is_clear_pick = True
                clear.append(b)
        return clear
