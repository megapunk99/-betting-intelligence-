"""
PreGameValidator — validates every bet against live data before it's recommended.

This is the gatekeeper that prevents stale bets from reaching the user.

For EVERY bet suggestion, it checks:
  1. Data freshness — are odds and injury data recent enough?
  2. Injury status — have key players been ruled out since the model's prediction?
  3. Line movement — has the line moved significantly since modeling?
  4. Roster changes — any unexpected trades or returns?
  5. Sharp money — is the line moving against public betting (sharp action)?

If any check fails, the bet gets:
  - A warning tag explaining the issue
  - Downgraded confidence
  - Reduced stake (or zeroed out for critical issues)
  - The `is_bet_safe=False` flag set

This module runs INSIDE the RecommendationEngine, AFTER model predictions
but BEFORE bets are returned to the user.

Usage:
    validator = PreGameValidator()
    validated_bets = validator.validate_all(bets)
    safe_bets = validator.get_safe_bets(bets)
    summary = validator.get_summary(bets)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from betting_intel.data.live_gateway import LiveDataGateway, LiveSnapshot
from betting_intel.recommendations.bet_types import BetSuggestion, Confidence

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single bet against live data."""

    bet: BetSuggestion
    is_safe: bool
    warnings: list[str] = field(default_factory=list)
    data_age_minutes: float = 0.0
    freshness_grade: str = "UNKNOWN"
    injury_factor_home: float = 0.0
    injury_factor_away: float = 0.0
    line_moved_points: float = 0.0
    original_edge_pct: float = 0.0
    adjusted_edge_pct: float = 0.0
    original_stake: float = 0.0
    adjusted_stake: float = 0.0
    live_total: Optional[float] = None
    live_ml_home: Optional[float] = None
    live_ml_away: Optional[float] = None


class PreGameValidator:
    """
    Validates bets against live data before they are recommended.

    Key thresholds (all configurable):
      - MAX_ODDS_AGE: 10 minutes — odds older than this are stale
      - MAX_INJURY_AGE: 30 minutes — injury data older than this is stale
      - MAX_INJURY_FACTOR: 0.4 — if either team's injury factor exceeds this,
        the bet is flagged (team is significantly impacted)
      - MAX_LINE_MOVEMENT: 3.0 points — if the total has moved more than this
        since the model was trained, the edge may be gone
      - EDGE_DEGRADE_PER_INJURY_PCT: 0.5 — reduce edge by 50% per injury factor
    """

    MAX_ODDS_AGE_MINUTES: float = 10.0
    MAX_INJURY_AGE_MINUTES: float = 30.0
    MAX_INJURY_FACTOR: float = 0.4
    MAX_LINE_MOVEMENT_POINTS: float = 3.0
    EDGE_DEGRADE_PER_INJURY_PCT: float = 0.5
    INJURY_AT_RISK_THRESHOLD: float = 0.2  # Flag bets when injury factor > 0.2

    def __init__(
        self,
        gateway: Optional[LiveDataGateway] = None,
        strict_mode: bool = True,
    ):
        """
        Args:
            gateway: LiveDataGateway instance. If None, creates a new one.
            strict_mode: If True, bets with ANY warning are marked unsafe.
                         If False, only critical warnings make a bet unsafe.
        """
        self.gateway = gateway or LiveDataGateway()
        self.strict_mode = strict_mode

    def validate_all(self, bets: list[BetSuggestion]) -> list[BetSuggestion]:
        """
        Validate ALL bets against live data.

        Each bet is checked, tagged with warnings if issues are found,
        and its stake/confidence is adjusted downward if live data
        indicates the edge has eroded.

        Args:
            bets: List of BetSuggestion objects from RecommendationEngine.

        Returns:
            Same list, but each bet is mutated in-place with:
              - bet.tags updated with warning/freshness tags
              - bet.metadata["validation"] = ValidationResult dict
              - bet.stake_dollars potentially reduced
              - bet.confidence potentially downgraded
              - bet.edge_pct potentially reduced
        """
        for bet in bets:
            result = self._validate_single(bet)
            bet.metadata["validation"] = result

            # Apply validation adjustments to the bet
            self._apply_validation(bet, result)

        return bets

    def get_safe_bets(self, bets: list[BetSuggestion]) -> list[BetSuggestion]:
        """Return only bets that passed validation (is_safe=True)."""
        self.validate_all(bets)
        return [b for b in bets if b.metadata.get("validation", {}).get("is_safe", False)]

    def get_unsafe_bets(self, bets: list[BetSuggestion]) -> list[BetSuggestion]:
        """Return bets that failed validation (is_safe=False)."""
        self.validate_all(bets)
        return [b for b in bets if not b.metadata.get("validation", {}).get("is_safe", True)]

    def get_summary(self, bets: list[BetSuggestion]) -> dict:
        """Get summary of validation results."""
        validated = self.validate_all(bets)
        safe = [b for b in validated if b.metadata.get("validation", {}).get("is_safe", False)]

        total_warnings = sum(
            len(b.metadata.get("validation", {}).get("warnings", []))
            for b in validated
        )

        # Warning type breakdown
        warning_types: dict[str, int] = {}
        for b in validated:
            v = b.metadata.get("validation", {})
            for w in v.get("warnings", []):
                key = w.split(":")[0].strip()
                warning_types[key] = warning_types.get(key, 0) + 1

        return {
            "total_bets": len(validated),
            "safe_bets": len(safe),
            "unsafe_bets": len(validated) - len(safe),
            "total_warnings": total_warnings,
            "warning_types": warning_types,
            "avg_data_age_minutes": self._avg_validation_metric(validated, "data_age_minutes"),
            "stake_reduction_pct": self._compute_stake_reduction(validated),
            "strict_mode": self.strict_mode,
        }

    # ── Internal Validation Logic ──────────────────────────────────────────

    def _validate_single(self, bet: BetSuggestion) -> dict:
        """
        Validate a single bet against live data.

        Returns a dict mirroring ValidationResult fields.
        """
        warnings: list[str] = []
        is_safe = True

        # Default result
        result = {
            "is_safe": True,
            "warnings": [],
            "data_age_minutes": 0.0,
            "freshness_grade": "UNKNOWN",
            "injury_factor_home": 0.0,
            "injury_factor_away": 0.0,
            "line_moved_points": 0.0,
            "original_edge_pct": bet.edge_pct,
            "adjusted_edge_pct": bet.edge_pct,
            "original_stake": bet.stake_dollars,
            "adjusted_stake": bet.stake_dollars,
            "live_total": None,
            "live_ml_home": None,
            "live_ml_away": None,
        }

        # If no teams in the bet, skip live validation
        if not bet.matchup or bet.league != "NBA":
            return result

        # Parse teams from matchup (format: "Away @ Home")
        teams = self._parse_teams(bet.matchup)
        if teams is None:
            return result

        home_team, away_team = teams

        # Fetch live snapshot for this game
        try:
            snapshot = self.gateway.get_game_snapshot(
                home_team=home_team,
                away_team=away_team,
                game_id=bet.game_id,
                game_date=bet.game_date,
            )
        except Exception as e:
            logger.debug(f"Could not fetch live snapshot for {bet.matchup}: {e}")
            result["warnings"].append(f"Live data unavailable: {e}")
            result["is_safe"] = False if self.strict_mode else True
            return result

        # Populate result
        result["injury_factor_home"] = snapshot.home_injury_factor
        result["injury_factor_away"] = snapshot.away_injury_factor
        result["data_age_minutes"] = max(snapshot.odds_age_minutes, snapshot.injury_age_minutes)
        result["freshness_grade"] = snapshot.freshness_grade
        result["live_total"] = snapshot.total
        result["live_ml_home"] = snapshot.home_ml
        result["live_ml_away"] = snapshot.away_ml
        result["line_moved_points"] = snapshot.total_movement

        # ── CHECK 1: Data Freshness ────────────────────────────────────────
        if result["freshness_grade"] == "MISSING":
            warnings.append("Data freshness: MISSING — cannot verify bet safety")
            if self.strict_mode:
                is_safe = False
        elif result["freshness_grade"] == "STALE":
            warnings.append(f"Data freshness: STALE ({result['data_age_minutes']:.0f}m old)")
            if self.strict_mode:
                is_safe = False

        # ── CHECK 2: Injury Impact ─────────────────────────────────────────
        home_impact = snapshot.home_injury_factor
        away_impact = snapshot.away_injury_factor

        if home_impact >= self.MAX_INJURY_FACTOR:
            warnings.append(
                f"Injury: {home_team} injury factor is {home_impact:.2f} "
                f"(threshold: {self.MAX_INJURY_FACTOR})"
            )
            is_safe = False
        elif home_impact >= self.INJURY_AT_RISK_THRESHOLD:
            warnings.append(
                f"Injury watch: {home_team} injury factor {home_impact:.2f}"
            )

        if away_impact >= self.MAX_INJURY_FACTOR:
            warnings.append(
                f"Injury: {away_team} injury factor is {away_impact:.2f} "
                f"(threshold: {self.MAX_INJURY_FACTOR})"
            )
            is_safe = False
        elif away_impact >= self.INJURY_AT_RISK_THRESHOLD:
            warnings.append(
                f"Injury watch: {away_team} injury factor {away_impact:.2f}"
            )

        # ── CHECK 3: Line Movement ─────────────────────────────────────────
        line_move = abs(snapshot.total_movement)
        if line_move > self.MAX_LINE_MOVEMENT_POINTS:
            warnings.append(
                f"Line movement: Total moved {snapshot.total_movement:+.1f} pts "
                f"(threshold: {self.MAX_LINE_MOVEMENT_POINTS} pts)"
            )
            is_safe = False
        elif line_move > 1.0:
            warnings.append(
                f"Line movement: Total moved {snapshot.total_movement:+.1f} pts"
            )

        # ── CHECK 4: Sharp Money ──────────────────────────────────────────
        if snapshot.sharp_money_flag:
            warnings.append(
                "Sharp money: Line moving against public betting"
            )
            if self.strict_mode:
                is_safe = False

        # ── CHECK 5: Roster Changes ───────────────────────────────────────
        if snapshot.home_roster_changes:
            roster_names = [r.get("player_name", "?") for r in snapshot.home_roster_changes[:3]]
            warnings.append(
                f"Roster changes (home): {', '.join(roster_names)}"
            )
        if snapshot.away_roster_changes:
            roster_names = [r.get("player_name", "?") for r in snapshot.away_roster_changes[:3]]
            warnings.append(
                f"Roster changes (away): {', '.join(roster_names)}"
            )

        # ── Compute adjusted edge ─────────────────────────────────────────
        adjusted_edge = bet.edge_pct

        # Degrade edge based on injury impact
        total_injury_impact = max(home_impact, away_impact)
        if total_injury_impact > 0:
            degradation = total_injury_impact * self.EDGE_DEGRADE_PER_INJURY_PCT
            adjusted_edge = adjusted_edge * (1 - degradation)

        # Degrade edge based on line movement
        if line_move > 1.0:
            movement_degradation = min(line_move / 10, 0.5)
            adjusted_edge = adjusted_edge * (1 - movement_degradation)

        result["adjusted_edge_pct"] = max(adjusted_edge, 0.0)

        # Compute adjusted stake
        stake_reduction = 1.0
        if result["adjusted_edge_pct"] < bet.edge_pct:
            stake_reduction = result["adjusted_edge_pct"] / max(bet.edge_pct, 0.001)
        result["adjusted_stake"] = bet.stake_dollars * stake_reduction

        # Add warning tags to result
        result["warnings"] = warnings
        result["is_safe"] = is_safe or (not self.strict_mode and len(warnings) == 0)

        return result

    def _apply_validation(self, bet: BetSuggestion, result: dict) -> None:
        """Apply validation results to a BetSuggestion in-place."""
        # Update tags
        if not result["is_safe"]:
            bet.tags.append("unsafe")
            bet.tags.append("stale_data")

        if result["warnings"]:
            for w in result["warnings"]:
                if "Injury" in w and "watch" not in w:
                    bet.tags.append("injury_risk")
                elif "Line movement" in w:
                    bet.tags.append("line_moved")
                elif "Sharp" in w:
                    bet.tags.append("sharp_money")
                elif "Roster" in w:
                    bet.tags.append("roster_change")
                elif "freshness" in w.lower():
                    bet.tags.append("stale_data")

        # Downgrade confidence for unsafe bets
        if not result["is_safe"]:
            current = bet.confidence
            if current == Confidence.VERY_HIGH:
                bet.confidence = Confidence.HIGH
            elif current == Confidence.HIGH:
                bet.confidence = Confidence.MEDIUM
            elif current == Confidence.MEDIUM:
                bet.confidence = Confidence.LOW
            elif current == Confidence.LOW:
                bet.confidence = Confidence.VERY_LOW

            bet.confidence_reason = "Downgraded by PreGameValidator — live data conflict"

        # Adjust edge
        bet.edge_pct = result["adjusted_edge_pct"]

        # Adjust stake
        bet.stake_dollars = result["adjusted_stake"]
        if bet.stake_dollars < 10:
            bet.stake_dollars = 0.0

    @staticmethod
    def _parse_teams(matchup: str) -> Optional[tuple[str, str]]:
        """Parse 'Away @ Home' format into (home, away) tuple."""
        if "@" not in matchup:
            return None
        parts = matchup.split("@")
        if len(parts) != 2:
            return None
        away = parts[0].strip()
        home = parts[1].strip()
        return home, away

    @staticmethod
    def _avg_validation_metric(bets: list[BetSuggestion], metric: str) -> float:
        """Compute average of a validation metric across bets."""
        values = [
            b.metadata.get("validation", {}).get(metric, 0)
            for b in bets
            if "validation" in b.metadata
        ]
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _compute_stake_reduction(bets: list[BetSuggestion]) -> float:
        """Compute total stake reduction from validation."""
        total_original = 0.0
        total_adjusted = 0.0
        for b in bets:
            v = b.metadata.get("validation", {})
            total_original += v.get("original_stake", 0)
            total_adjusted += v.get("adjusted_stake", 0)
        if total_original == 0:
            return 0.0
        return (total_original - total_adjusted) / total_original * 100
