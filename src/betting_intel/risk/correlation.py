"""
Bet correlation tracking: measure and track correlations between bet outcomes.

Understanding bet correlations is critical for:
1. Avoiding over-concentration on correlated outcomes (e.g., same game overs + same team moneyline)
2. Computing multi-bet Kelly correctly
3. Portfolio risk management

Bets on the same game are inherently correlated.
Bets on different games in the same league have weak correlation.
Player props are strongly correlated with game totals.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


# Default correlation estimates based on empirical sports betting data
CORRELATION_ESTIMATES = {
    # Same game correlations
    ("moneyline", "spread", "same_game"): 0.85,  # ML and spread on same team
    ("moneyline", "total", "same_game"): 0.15,  # ML and total (weak)
    ("spread", "total", "same_game"): 0.10,  # Spread and total (weak)
    ("total_over", "total_under", "same_game"): -0.95,  # Opposite sides of same line
    ("moneyline_home", "moneyline_away", "same_game"): -0.98,  # Opposite sides
    ("spread_home", "spread_away", "same_game"): -0.95,  # Opposite sides
    ("player_points", "total_over", "same_game"): 0.30,  # Player points with game total
    ("player_points", "moneyline", "same_game"): 0.20,  # Player points with ML
    ("player_rebounds", "total_over", "same_game"): 0.15,  # Rebounds with total
    # Cross-game correlations
    ("nba", "nba", "diff_game_same_day"): 0.05,  # Different NBA games same day
    ("nba", "nba", "diff_game_diff_day"): 0.01,  # Different NBA games different days
    # Cross-league
    ("nba", "lnb_pro_b", "diff_game"): 0.0,  # Different leagues = independent
}


@dataclass
class BetCorrelationRecord:
    """Record of a tracked correlation between two bet types."""

    bet_type_a: str
    bet_type_b: str
    context: str  # 'same_game', 'same_league', 'cross_league'
    estimated_correlation: float
    n_observations: int = 0
    empirical_correlation: Optional[float] = None


class CorrelationMatrix:
    """
    N x N correlation matrix for a set of bets.

    Provides fast lookup of pairwise correlations.
    """

    def __init__(self, bet_ids: List[str]):
        self.bet_ids = bet_ids
        self.n = len(bet_ids)
        self._matrix = np.eye(self.n)

    def set_correlation(self, i: int, j: int, value: float):
        """Set correlation between bet i and bet j."""
        value = max(-1.0, min(1.0, value))
        self._matrix[i, j] = value
        self._matrix[j, i] = value

    def get_correlation(self, i: int, j: int) -> float:
        """Get correlation between bet i and bet j."""
        return self._matrix[i, j]

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    @property
    def is_positive_semidefinite(self) -> bool:
        """Check if matrix is valid (PSD)."""
        eigenvalues = np.linalg.eigvalsh(self._matrix)
        return float(eigenvalues.min()) >= -1e-8

    def make_positive_semidefinite(self):
        """
        Fix matrix to be positive semidefinite.
        Uses eigenvalue clipping (nearest PSD matrix).
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self._matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        self._matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return {
            "bet_ids": self.bet_ids,
            "matrix": self._matrix.tolist(),
        }


class BetCorrelationTracker:
    """
    Tracks correlations between bet outcomes.

    Uses a combination of:
    1. Empirical estimation from historical bet data
    2. Domain knowledge (correlation estimates above)
    3. Real-time tracking of concurrent bets

    Usage:
        tracker = BetCorrelationTracker()
        tracker.record_outcome(bet_id_1, won=True)
        tracker.record_outcome(bet_id_2, won=True)
        corr = tracker.get_correlation(bet_id_1, bet_id_2)
    """

    def __init__(self):
        # Historical bet outcomes stored by bet type and context
        self._outcomes: Dict[str, List[bool]] = defaultdict(list)
        self._bet_metadata: Dict[str, Dict] = {}
        self._pairwise_outcomes: Dict[Tuple[str, str], List[Tuple[bool, bool]]] = defaultdict(list)

    def register_bet(
        self,
        bet_id: str,
        bet_type: str,
        game_id: str,
        league: str = "NBA",
        team: Optional[str] = None,
        player: Optional[str] = None,
    ):
        """Register a bet for correlation tracking."""
        self._bet_metadata[bet_id] = {
            "bet_type": bet_type,
            "game_id": game_id,
            "league": league,
            "team": team,
            "player": player,
        }

    def record_outcome(self, bet_id: str, won: bool):
        """Record whether a bet won or lost."""
        if bet_id not in self._bet_metadata:
            return

        meta = self._bet_metadata[bet_id]
        bet_type = meta["bet_type"]

        self._outcomes[bet_id].append(won)

        # Update pairwise outcomes
        for other_id in self._outcomes:
            if other_id != bet_id and self._outcomes[other_id]:
                other_won = self._outcomes[other_id][-1]
                pair = tuple(sorted([bet_id, other_id]))
                self._pairwise_outcomes[pair].append((won, other_won))

    def get_correlation(
        self,
        bet_id_a: str,
        bet_id_b: str,
    ) -> float:
        """
        Get estimated correlation between two bets.

        Combines empirical data with domain knowledge estimates.
        """
        # Try empirical first
        pair = tuple(sorted([bet_id_a, bet_id_b]))
        if pair in self._pairwise_outcomes and len(self._pairwise_outcomes[pair]) >= 10:
            outcomes = self._pairwise_outcomes[pair]
            outcomes_a = np.array([o[0] for o in outcomes])
            outcomes_b = np.array([o[1] for o in outcomes])

            if np.std(outcomes_a) > 0 and np.std(outcomes_b) > 0:
                empirical_corr = float(np.corrcoef(outcomes_a, outcomes_b)[0, 1])
                if not np.isnan(empirical_corr):
                    return empirical_corr

        # Use domain knowledge estimates
        meta_a = self._bet_metadata.get(bet_id_a, {})
        meta_b = self._bet_metadata.get(bet_id_b, {})

        return self._estimate_domain_correlation(meta_a, meta_b)

    def _estimate_domain_correlation(self, meta_a: Dict, meta_b: Dict) -> float:
        """Estimate correlation based on domain knowledge."""
        if not meta_a or not meta_b:
            return 0.0

        same_game = meta_a.get("game_id") == meta_b.get("game_id")
        same_league = meta_a.get("league") == meta_b.get("league")
        type_a = meta_a.get("bet_type", "")
        type_b = meta_b.get("bet_type", "")

        if not same_league:
            return 0.0

        if same_game:
            # Check for opposite sides
            if type_a == "moneyline_home" and type_b == "moneyline_away":
                return -0.98
            if type_a == "spread_home" and type_b == "spread_away":
                return -0.95
            if type_a == "total_over" and type_b == "total_under":
                return -0.95

            # Player props with game totals
            if "player" in type_a and "total" in type_b:
                context = "same_game"
                key = (type_a, type_b, context)
                return CORRELATION_ESTIMATES.get(key, 0.20)

            # Same team ML + spread
            if (type_a == "moneyline" and meta_a.get("team") and 
                type_b == "spread" and meta_b.get("team") and
                meta_a["team"] == meta_b["team"]):
                return 0.85

            # General same-game correlation (small positive)
            return 0.10

        if same_league:
            # Same league, different games = weak correlation
            return 0.03

        return 0.0

    def get_correlation_matrix(self, bet_ids: List[str]) -> CorrelationMatrix:
        """Build correlation matrix for a list of bets."""
        matrix = CorrelationMatrix(bet_ids)

        for i in range(len(bet_ids)):
            for j in range(i + 1, len(bet_ids)):
                corr = self.get_correlation(bet_ids[i], bet_ids[j])
                matrix.set_correlation(i, j, corr)

        # Ensure PSD
        if not matrix.is_positive_semidefinite:
            matrix.make_positive_semidefinite()

        return matrix

    def get_summary(self) -> Dict:
        """Get summary of tracked correlations."""
        return {
            "n_bets_tracked": len(self._bet_metadata),
            "n_pairwise_observations": len(self._pairwise_outcomes),
            "pairs_with_empirical_data": sum(
                1 for v in self._pairwise_outcomes.values() if len(v) >= 10
            ),
        }


def estimate_game_correlations(
    game_id: str,
    bets: List[Dict],
) -> CorrelationMatrix:
    """
    Convenience function: estimate correlations between bets on the same game.

    Args:
        game_id: Game identifier
        bets: List of dicts with keys: bet_id, bet_type, team, player

    Returns:
        CorrelationMatrix for the given bets
    """
    tracker = BetCorrelationTracker()

    for bet in bets:
        tracker.register_bet(
            bet_id=bet["bet_id"],
            bet_type=bet.get("bet_type", "unknown"),
            game_id=game_id,
            league=bet.get("league", "NBA"),
            team=bet.get("team"),
            player=bet.get("player"),
        )

    bet_ids = [b["bet_id"] for b in bets]
    return tracker.get_correlation_matrix(bet_ids)
