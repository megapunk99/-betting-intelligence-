"""Market Intelligence — Phase 3 of the Professional Betting Intelligence Platform.

Tracks market movements, detects steam moves, and compares model vs market.

Modules:
    movement: Market Movement Engine — opening/current/closing line tracking
    steam: Steam Move Detector — rapid + cross-book movement alerts
    comparison: Model vs Market Comparison — integrated edge display
"""

from betting_intel.market.movement import (
    MarketMovementEngine,
    MarketMovementRecord,
    LineMovement,
    MarketTrend,
)
from betting_intel.market.steam import (
    SteamMoveDetector,
    SteamAlert,
    SteamMoveType,
)
from betting_intel.market.comparison import (
    ModelMarketComparison,
    ComparisonResult,
    ComparisonAggregate,
)

__all__ = [
    "MarketMovementEngine",
    "MarketMovementRecord",
    "LineMovement",
    "MarketTrend",
    "SteamMoveDetector",
    "SteamAlert",
    "SteamMoveType",
    "ModelMarketComparison",
    "ComparisonResult",
    "ComparisonAggregate",
]
