"""
Pipeline package — modularized prediction pipeline components.

Split from the monolithic predict_tomorrow.py (root-level script)
into focused modules, each implementing a mixin class.

Usage:
    from betting_intel.pipeline import PredictionPipeline, main, parse_args
"""

from betting_intel.pipeline.pipeline import PredictionPipeline
from betting_intel.pipeline.cli import main, parse_args

__all__ = [
    "PredictionPipeline",
    "main",
    "parse_args",
]
