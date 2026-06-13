"""
Sport Configurations — NBA-only focus for the LivePredictionEngine.

Strategy:
  NBA is the only league we predict. Rationale:
  - 1,230 games/season with 30 teams = ample training data
  - High scoring = strong statistical signal
  - Public vs sharp money creates predictable market inefficiencies
  - Deep market with 10+ sportsbooks = stable consensus lines

Leagues removed:
  - NCAAB (362 teams, chaotic, low per-team data)
  - MLB (high variance, needs completely different modeling)
  - Tennis (individual sport, different dynamics)
  - MMA/Boxing (too few events for ML)
  - WNBA (user-excluded)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Sport Config Data Class ───────────────────────────────────────────────

@dataclass
class SportConfig:
    """Configuration for the NBA prediction engine."""

    # TheOddsAPI sport key (e.g. "basketball_nba")
    sport_key: str

    # Display names
    display_name: str          # e.g. "NBA" (short badge text)
    full_name: str             # e.g. "National Basketball Association"
    emoji: str = ""            # e.g. "🏀"

    # Market types this sport supports
    has_h2h: bool = True       # Moneyline (head-to-head)
    has_spreads: bool = False  # Point spreads
    has_totals: bool = False   # Over/under totals

    # Team name mapping: TheOddsAPI full name → short display name
    team_name_map: dict[str, str] = field(default_factory=dict)

    # Prediction strategy
    # "total" = over/under prediction (NBA)
    # "moneyline" = win probability
    prediction_strategy: str = "total"

    # Total range (NBA: 180-260)
    total_min: float = 180.0
    total_max: float = 260.0

    # Season months (Oct-Jun)
    season_start_month: int = 10
    season_end_month: int = 6

    # Markets to fetch from TheOddsAPI
    markets_to_fetch: list[str] = field(default_factory=lambda: ["h2h", "spreads", "totals"])

    @property
    def is_in_season(self) -> bool:
        """Check if the sport is currently in season (Oct-Jun, spans year boundary)."""
        now = datetime.now()
        month = now.month
        if self.season_start_month <= self.season_end_month:
            return self.season_start_month <= month <= self.season_end_month
        return month >= self.season_start_month or month <= self.season_end_month

    def get_short_name(self, full_name: str) -> str:
        """Convert a TheOddsAPI full name to a short display name."""
        if not full_name:
            return ""
        short = self.team_name_map.get(full_name)
        if short:
            return short
        parts = full_name.split()
        if len(parts) > 1:
            return parts[-1]
        return full_name


# ── NBA Definition ────────────────────────────────────────────────────────

# NBA — the only league we predict. Full season (Oct-Jun), 1,230 games.
NBA = SportConfig(
    sport_key="basketball_nba",
    display_name="NBA",
    full_name="National Basketball Association",
    emoji="🏀",
    has_h2h=True,
    has_spreads=True,
    has_totals=True,
    prediction_strategy="total",
    total_min=180.0,
    total_max=260.0,
    season_start_month=10,
    season_end_month=6,
    markets_to_fetch=["h2h", "spreads", "totals"],
)


# ── Master Lists ──────────────────────────────────────────────────────────

# All supported sports — only NBA
ALL_SPORTS: list[SportConfig] = [NBA]

# Sports that are currently in season
def get_active_sports() -> list[SportConfig]:
    return [s for s in ALL_SPORTS if s.is_in_season]

# TheOddsAPI sport_key → SportConfig lookup
SPORT_KEY_TO_CONFIG: dict[str, SportConfig] = {s.sport_key: s for s in ALL_SPORTS}

# Display name → SportConfig lookup (for UI filtering)
DISPLAY_NAME_TO_CONFIG: dict[str, SportConfig] = {s.display_name: s for s in ALL_SPORTS}


# ── Helper Functions ──────────────────────────────────────────────────────

def league_from_sport_key(sport_key: str) -> str:
    """Convert a TheOddsAPI sport key to a short league display name."""
    config = SPORT_KEY_TO_CONFIG.get(sport_key)
    if config:
        return config.display_name
    parts = sport_key.split("_")
    if len(parts) >= 2:
        return parts[-1].upper()
    return sport_key.upper()


def sport_key_to_group(sport_key: str) -> str:
    """Group sports by category for UI filtering. NBA → Basketball."""
    if "basketball" in sport_key:
        return "Basketball"
    return "Other"


# load_ncaab_team_map was removed during cleanup. NCAAB is not supported.
