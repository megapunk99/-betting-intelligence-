"""
Arbitrage Detection — finds guaranteed-profit opportunities across sportsbooks.

Detects standard 2-way (h2h), three-way (soccer), and totals arbitrage
opportunities from raw TheOddsAPI odds data.
"""

from betting_intel.arbitrage.models import ArbLeg, ArbitrageOpportunity
from betting_intel.arbitrage.detector import detect_arbitrage

__all__ = [
    "ArbLeg",
    "ArbitrageOpportunity",
    "detect_arbitrage",
]
