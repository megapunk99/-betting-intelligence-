"""Risk management module: Kelly optimization, exposure management, correlation tracking."""

from betting_intel.risk.kelly import (
    KellyCalculator,
    MultiBetKelly,
    correlated_kelly,
)
from betting_intel.risk.exposure import (
    ExposureManager,
    BetPortfolio,
    PositionLimit,
)
from betting_intel.risk.correlation import (
    BetCorrelationTracker,
    CorrelationMatrix,
)

__all__ = [
    "KellyCalculator",
    "MultiBetKelly",
    "correlated_kelly",
    "ExposureManager",
    "BetPortfolio",
    "PositionLimit",
    "BetCorrelationTracker",
    "CorrelationMatrix",
]
