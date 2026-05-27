"""
Betting recommendation engine: generates ALL possible bet types,
ranks them by edge, identifies high-confidence "clear" picks,
scans for +EV opportunities, and detects arbitrage across sportsbooks.

Usage:
    from betting_intel.recommendations import RecommendationEngine

    engine = RecommendationEngine()
    bets = engine.generate_all_bets()
    clear_picks = engine.get_clear_picks(threshold=0.03)

    # +EV scanning (requires odds from poller)
    scanner = PositiveEVScanner()
    ev_report = scanner.scan_odds_snapshots(poller.get_current_odds())

    # Arbitrage detection
    arb = ArbitrageDetector()
    arb_report = arb.scan_for_arbitrage(poller.get_current_odds())
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
from betting_intel.recommendations.ev_scanner import (
    PositiveEVScanner,
    EVOpportunity,
    ScannerReport,
    ScannerConfidence,
    ScannerSource,
)
from betting_intel.recommendations.arbitrage import (
    ArbitrageDetector,
    ArbitrageOpportunity,
    ArbitrageReport,
    ArbOutcome,
)

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
    "PositiveEVScanner",
    "EVOpportunity",
    "ScannerReport",
    "ScannerConfidence",
    "ScannerSource",
    "ArbitrageDetector",
    "ArbitrageOpportunity",
    "ArbitrageReport",
    "ArbOutcome",
]
