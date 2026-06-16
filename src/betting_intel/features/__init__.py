"""Feature engineering — market inefficiency computation.

The FeatureStore (store.py) was dead code — the V2 versioned feature store
that was never connected to the live pipeline. Only market_inefficiency.py
is actively used by the MarketInefficiencySystem.
"""

from betting_intel.features.market_inefficiency import (
    MarketInefficiencyComputer,
    compute_market_inefficiency_targets,
    spread_to_implied_prob,
    margin_to_implied_prob,
    american_to_implied_prob,
    remove_vig,
)

__all__ = [
    "MarketInefficiencyComputer",
    "compute_market_inefficiency_targets",
    "spread_to_implied_prob",
    "margin_to_implied_prob",
    "american_to_implied_prob",
    "remove_vig",
]
