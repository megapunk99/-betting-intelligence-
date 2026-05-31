"""Feature Store — Phase 2.8 of the Professional Betting Intelligence Platform.

Versioned storage of team, schedule, and player features.

Store:
    Team Features: offensive_rating, defensive_rating, pace, net_rating
    Schedule Features: rest_days, back_to_back, travel_distance, home_away
    Player Features: injury_status, usage_rate, minutes

All features are versioned with timestamps.
"""

from betting_intel.features.store import (
    FeatureStore,
    TeamFeatureRecord,
    ScheduleFeatureRecord,
    PlayerFeatureRecord,
    FeatureVersion,
)
from betting_intel.features.builder import (
    FeatureBuilder,
    build_team_features,
    build_schedule_features,
    build_player_features,
)

__all__ = [
    "FeatureStore",
    "TeamFeatureRecord",
    "ScheduleFeatureRecord",
    "PlayerFeatureRecord",
    "FeatureVersion",
    "FeatureBuilder",
    "build_team_features",
    "build_schedule_features",
    "build_player_features",
]
