"""Validation module: cross-validation, calibration, overfitting detection."""

from betting_intel.validation.cross_validation import (
    TimeSeriesCrossValidator,
    ExpandingWindowCV,
    purged_walk_forward,
)
from betting_intel.validation.calibration import (
    ProbabilityCalibrator,
    PlattCalibrator,
    IsotonicCalibrator,
    BetaCalibrator,
)
from betting_intel.validation.overfitting import (
    OverfittingDetector,
    DeflatedSharpeRatio,
    ModelComparisonTest,
)

__all__ = [
    "TimeSeriesCrossValidator",
    "ExpandingWindowCV",
    "purged_walk_forward",
    "ProbabilityCalibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "BetaCalibrator",
    "OverfittingDetector",
    "DeflatedSharpeRatio",
    "ModelComparisonTest",
]
