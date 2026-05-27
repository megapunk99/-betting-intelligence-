"""
Betting recommendation engine: generates ALL possible bet types,
ranks them by edge, and identifies high-confidence "clear" picks.

Usage:
    from betting_intel.recommendations import RecommendationEngine

    engine = RecommendationEngine()
    bets = engine.generate_all_bets()
    clear_picks = engine.get_clear_picks(threshold=0.03)
    all_by_edge = engine.rank_by_edge()
"""

from betting_intel.recommendations.engine import RecommendationEngine
from betting_intel.recommendations.bet_types import (
    BetType,
    BetSuggestion,
    Confidence,
    MoneylineBet,
    SpreadBet,
    TotalBet,
    TeamTotalBet,
    QuarterBet,
    HalfTotalBet,
    PlayerPropBet,
    ParlaySuggestion,
)
from betting_intel.recommendations.ranker import BetRanker, ClearPick
from betting_intel.recommendations.validator import PreGameValidator, ValidationResult

__all__ = [
    "RecommendationEngine",
    "BetType",
    "BetSuggestion",
    "Confidence",
    "MoneylineBet",
    "SpreadBet",
    "TotalBet",
    "TeamTotalBet",
    "QuarterBet",
    "HalfTotalBet",
    "PlayerPropBet",
    "ParlaySuggestion",
    "BetRanker",
    "ClearPick",
    "PreGameValidator",
    "ValidationResult",
]
