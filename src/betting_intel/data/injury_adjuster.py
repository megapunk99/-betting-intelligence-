"""
Injury Adjuster — adjusts team-level feature projections based on player injuries.

Takes injury/availability data from PlayerInjuryFetcher (Odds API props) and
ESPNInjuryScraper, cross-references against actual player season stats from
PlayerStatsManager, and computes per-team adjustment factors to apply to the
feature vectors used for prediction.

How it works:
  1. Receives a list of missing players per team (from PlayerInjuryFetcher)
  2. Looks up each player's actual season PPG/MIN from PlayerStatsManager
  3. For each missing player, computes the "scoring loss":
     - STAR: 70% of PPG lost (hard to replace star production)
     - STARTER: 50% of PPG lost
     - ROTATION: 20% of PPG lost
     - BENCH: 5% of PPG lost
  4. Sums up the total PPG loss per team
  5. Returns adjusted feature overrides that reduce team scoring/pace/momentum

Usage:
    from betting_intel.data.injury_adjuster import InjuryAdjuster

    adjuster = InjuryAdjuster()
    adj = adjuster.compute_adjustment("NYK", ["Julius Randle"])
    # {'scoring_loss': 15.4, 'pace_loss': 0.5, 'momentum_loss': -0.05, ...}
"""

from __future__ import annotations

import logging
from typing import Optional

from betting_intel.data.player_injury import (
    GameInjuryData,
    PLAYER_DATABASE,
    ROLE_WEIGHT,
    TEAM_ABBR_TO_SHORT,
)
from betting_intel.data.player_stats import PlayerStatsManager

logger = logging.getLogger(__name__)

# ── Loss Factors per Role ─────────────────────────────────────────────────
# What fraction of a player's PPG is actually *lost* by the team when they're out.
# STAR (All-NBA): 70% — their production is hard to replace
# STARTER: 50% — significant but more replaceable
# ROTATION: 20% — bench depth is easier to replace
# BENCH: 10% — minimal impact
LOSS_FACTOR: dict[str, float] = {
    "STAR": 0.70,
    "STARTER": 0.50,
    "ROTATION": 0.20,
    "BENCH": 0.10,
}

# ── Feature Adjustment Multipliers ────────────────────────────────────────
# Maps missing-PPG loss to feature value adjustments.
#
# CRITICAL: We only adjust the PRIMARY scoring feature (avg_pts) and one
# defensive proxy (avg_pts_allowed). Older versions of this code had 10+
# feature entries (ema_pts, trend_pts, avg_margin, ema_margin, last_3_margin,
# pace, avg_pace, form_score, etc.) — ALL receiving the same scoring-loss
# signal scaled by different multipliers. This compounded the adjustment
# across too many correlated features, producing unrealistic 12-16 pt swings
# when the actual shift should be ~2-8 pts.
#
# The model's learned coefficients on avg_pts capture the total scoring
# impact across all correlated dimensions (EMAs, margins, pace, form).
# We trust the model to distribute the effect; we just need to nudge
# the primary input.
#
# Rationale:
# - avg_pts (0.50): Direct scoring. ~50% of weighted PPG loss shows up in
#   pts/100poss. STAR player out = ~15 PPG weighted × 0.50 = -7.5 on
#   avg_pts_10g, a meaningful but realistic shift.
# - avg_pts_allowed (-0.05): Losing a star slightly hurts defense (more
#   points allowed). Negative multiplier = -loss * (-0.05) = +0.05 * loss.
FEATURE_ADJUSTMENT: dict[str, float] = {
    "avg_pts": 0.50,            # Primary: ~50% of weighted PPG loss
    "avg_pts_allowed": -0.05,   # Defensive leakage: +5% of loss
}


class InjuryAdjuster:
    """
    Computes injury-adjusted feature overrides for upcoming games.

    Usage:
        adjuster = InjuryAdjuster()

        # For a single game with injury data
        adjustment = adjuster.compute_game_adjustment(
            home_team="NYK",
            away_team="SAS",
            home_missing=["Julius Randle (22 PPG, STAR)"],
            away_missing=["Jeremy Sochan (11 PPG, STARTER)", "Chris Paul (9 PPG, STARTER)"],
        )
        # Returns dict with 'home' and 'away' sub-dicts containing feature overrides

        # Apply to feature vector
        for col, adj_value in adjustment['home'].items():
            if col in features:
                features[col] += adj_value
    """

    def __init__(self, use_player_stats_db: bool = True):
        """
        Args:
            use_player_stats_db: If True, use actual season PPG from
                PlayerStatsManager. If False, fall back to hardcoded
                PLAYER_DATABASE (good for testing without fetching).
        """
        self._stats_manager: Optional[PlayerStatsManager] = None
        self.use_player_stats_db = use_player_stats_db

    @property
    def stats_manager(self) -> PlayerStatsManager:
        if self._stats_manager is None:
            self._stats_manager = PlayerStatsManager()
        return self._stats_manager

    def compute_game_adjustment(
        self,
        home_team: str,
        away_team: str,
        home_missing: Optional[list[str]] = None,
        away_missing: Optional[list[str]] = None,
        home_injury_data: Optional[dict] = None,
        away_injury_data: Optional[dict] = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute feature adjustments for a single game.

        Args:
            home_team: Home team abbreviation (e.g., "NYK")
            away_team: Away team abbreviation (e.g., "SAS")
            home_missing: List of missing player display names from
                PlayerInjuryFetcher (e.g., ["Julius Randle (22 PPG, STAR)"])
            away_missing: Same for away team.
            home_injury_data: Optional dict with 'missing_ppg_weighted' key.
            away_injury_data: Same for away team.

        Returns:
            Dict with 'home' and 'away' keys, each containing a dict of
            feature column → adjustment value.
        """
        result: dict[str, dict[str, float]] = {"home": {}, "away": {}}

        for side, team_abbr, missing_list, injury_dict in [
            ("home", home_team, home_missing or [], home_injury_data),
            ("away", away_team, away_missing or [], away_injury_data),
        ]:
            # Compute total scoring loss for this team
            scoring_loss = self._compute_team_scoring_loss(
                team_abbr, missing_list, injury_dict
            )

            if scoring_loss <= 0:
                continue

            # Convert scoring loss to feature adjustments
            adjustments = self._scoring_loss_to_features(scoring_loss, side)
            result[side] = adjustments

        return result

    def compute_game_adjustment_from_injury_data(
        self,
        injury_data: GameInjuryData,
    ) -> dict[str, dict[str, float]]:
        """
        Compute adjustments directly from a GameInjuryData object.

        Args:
            injury_data: GameInjuryData from PlayerInjuryFetcher.

        Returns:
            Dict with 'home' and 'away' feature adjustments.
        """
        home_missing = []
        away_missing = []

        if injury_data.home_impact:
            home_missing = injury_data.home_impact.missing_stars
            home_injury_data = {
                "missing_ppg_weighted": injury_data.home_impact.missing_ppg_weighted,
            }
        else:
            home_injury_data = None

        if injury_data.away_impact:
            away_missing = injury_data.away_impact.missing_stars
            away_injury_data = {
                "missing_ppg_weighted": injury_data.away_impact.missing_ppg_weighted,
            }
        else:
            away_injury_data = None

        # Extract team abbreviations
        home_abbr = injury_data.home_impact.team_abbr if injury_data.home_impact else "???"
        away_abbr = injury_data.away_impact.team_abbr if injury_data.away_impact else "???"

        return self.compute_game_adjustment(
            home_team=home_abbr if injury_data.home_impact else "???",
            away_team=away_abbr if injury_data.away_impact else "???",
            home_missing=home_missing,
            away_missing=away_missing,
            home_injury_data=home_injury_data,
            away_injury_data=away_injury_data,
        )

    def apply_adjustment(
        self,
        feature_row: dict[str, float],
        adjustment: dict[str, dict[str, float]],
        feature_cols: list[str],
    ) -> dict[str, float]:
        """
        Apply injury adjustments to a feature vector.

        Modifies the feature row in-place (returns a copy for safety).

        Args:
            feature_row: The feature dict for a game.
            adjustment: Output from compute_game_adjustment().
            feature_cols: List of feature column names to consider.

        Returns:
            Modified copy of feature_row with injury adjustments applied.
        """
        result = dict(feature_row)

        for side, adj_dict in adjustment.items():
            suffix = f"_{side}"
            # Also apply to _home/_away suffixed versions
            for col, adj_value in adj_dict.items():
                # Try direct match
                if col in feature_cols and col in result:
                    result[col] = result[col] + adj_value

                # Try suffixed version (e.g., 'avg_pts' -> 'avg_pts_home')
                suffixed = col + suffix
                if suffixed in feature_cols and suffixed in result:
                    result[suffixed] = result[suffixed] + adj_value

        return result

    # ── Internal Methods ────────────────────────────────────────────────

    def _compute_team_scoring_loss(
        self,
        team_abbr: str,
        missing_players: list[str],
        injury_dict: Optional[dict] = None,
    ) -> float:
        """
        Compute the total scoring loss (in PPG) for a team due to missing players.

        Priority:
        1. Use weighted PPG from PlayerStatsManager (dynamic, accurate)
        2. Fall back to PlayerInjuryFetcher's pre-computed missing_ppg_weighted
        3. Fall back to PLAYER_DATABASE hardcoded values

        Args:
            team_abbr: Team abbreviation.
            missing_players: List of missing player display names.
            injury_dict: Optional dict with 'missing_ppg_weighted' key.

        Returns:
            Total PPG loss for the team (e.g., 15.4 means ~15 points per game lost).
        """
        if not missing_players:
            return 0.0

        # Strategy 1: Use the pre-computed weighted PPG from PlayerInjuryFetcher
        if injury_dict and "missing_ppg_weighted" in injury_dict:
            weighted_ppg = injury_dict["missing_ppg_weighted"]
            return weighted_ppg

        # Strategy 2: Compute from PlayerStatsManager (actual season averages)
        if self.use_player_stats_db:
            total_loss = 0.0
            for display_name in missing_players:
                # Parse "Player Name (PPG, ROLE)" format
                name = display_name.split(" (")[0].strip()
                role = "STARTER"  # default
                for r in ["STAR", "STARTER", "ROTATION", "BENCH"]:
                    if r in display_name:
                        role = r
                        break

                # Get actual PPG from database
                ppg = self.stats_manager.get_player_ppg(name)

                if ppg == 0.0:
                    # Fall back to hardcoded PPG
                    player_info = PLAYER_DATABASE.get(name)
                    if player_info:
                        ppg = player_info.get("ppg", 0.0)

                if ppg > 0:
                    loss_factor = LOSS_FACTOR.get(role, 0.5)
                    total_loss += ppg * loss_factor

            return total_loss

        # Strategy 3: Use hardcoded PLAYER_DATABASE
        total_loss = 0.0
        for display_name in missing_players:
            name = display_name.split(" (")[0].strip()
            role = "STARTER"
            for r in ["STAR", "STARTER", "ROTATION", "BENCH"]:
                if r in display_name:
                    role = r
                    break

            player_info = PLAYER_DATABASE.get(name)
            if player_info:
                ppg = player_info.get("ppg", 0.0)
                loss_factor = LOSS_FACTOR.get(role, 0.5)
                total_loss += ppg * loss_factor

        return total_loss

    def _scoring_loss_to_features(
        self,
        scoring_loss: float,
        side: str,
    ) -> dict[str, float]:
        """
        Convert a total scoring loss (PPG) to per-feature adjustments.

        Each feature type gets a fraction of the loss based on FEATURE_ADJUSTMENT.

        Args:
            scoring_loss: Total PPG lost (e.g., 15.4)
            side: "home" or "away"

        Returns:
            Dict of feature name → adjustment value (negative = reduce feature).
        """
        adjustments: dict[str, float] = {}

        for feature_base, multiplier in FEATURE_ADJUSTMENT.items():
            adj_value = -scoring_loss * multiplier
            adjustments[feature_base] = adj_value

        return adjustments

    def describe_adjustment(
        self,
        adjustment: dict[str, dict[str, float]],
    ) -> list[str]:
        """
        Generate human-readable description of an adjustment.

        Args:
            adjustment: Output from compute_game_adjustment().

        Returns:
            List of display lines.
        """
        lines = []
        for side, adj_dict in adjustment.items():
            if not adj_dict:
                continue
            # Get the avg_pts adjustment as the main indicator
            pts_adj = adj_dict.get("avg_pts", 0.0)
            if pts_adj >= 0:
                continue  # No meaningful adjustment

            lines.append(
                f"  {side.title()} injury adjustment: ~{abs(pts_adj):.1f} PPG lost"
            )

            # Detail the key adjustments
            detail = []
            for col in ["avg_pts", "avg_pts_allowed"]:
                if col in adj_dict and adj_dict[col] != 0:
                    detail.append(f"{col}={adj_dict[col]:+.1f}")
            if detail:
                lines.append(f"    Adjustments: {', '.join(detail)}")

        return lines


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== InjuryAdjuster Quick Test ===\n")

    adjuster = InjuryAdjuster(use_player_stats_db=True)

    # Test 1: Knicks missing Randle
    print("Test 1: Knicks missing Julius Randle (STAR, 22 PPG)")
    adj = adjuster.compute_game_adjustment(
        home_team="NYK",
        away_team="SAS",
        home_missing=["Julius Randle (22 PPG, STAR)"],
        away_missing=[],
    )
    for line in adjuster.describe_adjustment(adj):
        print(line)
    print()

    # Test 2: Spurs missing Sochan and Paul
    print("Test 2: Spurs missing Jeremy Sochan + Chris Paul")
    adj2 = adjuster.compute_game_adjustment(
        home_team="SAS",
        away_team="NYK",
        home_missing=[
            "Jeremy Sochan (11 PPG, STARTER)",
            "Chris Paul (9 PPG, STARTER)",
        ],
        away_missing=[],
    )
    for line in adjuster.describe_adjustment(adj2):
        print(line)
    print()

    # Test 3: Both teams missing players
    print("Test 3: Both teams missing players")
    adj3 = adjuster.compute_game_adjustment(
        home_team="NYK",
        away_team="SAS",
        home_missing=["Julius Randle (22 PPG, STAR)", "Mitchell Robinson (6 PPG, ROTATION)"],
        away_missing=["Chris Paul (9 PPG, STARTER)"],
    )
    for line in adjuster.describe_adjustment(adj3):
        print(line)

    print("\nDone.")
    sys.exit(0)
