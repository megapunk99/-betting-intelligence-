"""Feature Store — market inefficiency computation.

All feature modules have been consolidated. The only active module
is market_inefficiency.py, which computes market inefficiency target
variables for the ML pipeline.

The FeatureStore (store.py) and FeatureBuilder (builder.py) were
dead code — the V2 versioned feature store that was never connected
to the live pipeline. Only market_inefficiency.py is actively used.
"""

from betting_intel.features.store import (
    FeatureStore,
    TeamFeatureRecord,
    ScheduleFeatureRecord,
    PlayerFeatureRecord,
    FeatureVersion,
)
from betting_intel.features.market_inefficiency import (
    MarketInefficiencyComputer,
    compute_market_inefficiency_targets,
    spread_to_implied_prob,
    margin_to_implied_prob,
    american_to_implied_prob,
    remove_vig,
)

__all__ = [
    "FeatureStore",
    "TeamFeatureRecord",
    "ScheduleFeatureRecord",
    "PlayerFeatureRecord",
    "FeatureVersion",
    "MarketInefficiencyComputer",
    "compute_market_inefficiency_targets",
    "spread_to_implied_prob",
    "margin_to_implied_prob",
    "american_to_implied_prob",
    "remove_vig",
]
