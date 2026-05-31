"""
ESPN Injury Integrator — merges ESPN official injury status with prop-based detection.

Combines two data sources:
1. ESPNInjuryScraper → official injury status (OUT, Questionable, Probable, Day-To-Day)
2. PlayerInjuryFetcher → prop-based missing player detection

Provides cross-referenced injury data that shows:
- Which players are missing from props (likely injured)
- Their official ESPN injury status (if available)
- Teams with many injured players flagged by both sources
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from betting_intel.data.injury_scraper import ESPNInjuryScraper, InjuryRecord
from betting_intel.data.player_injury import (
    GameInjuryData,
    InjuryImpact,
    PLAYER_DATABASE,
    TEAM_ABBR_TO_SHORT,
)

logger = logging.getLogger(__name__)

# ── Data Structures ────────────────────────────────────────────────────────


@dataclass
class PlayerInjuryStatus:
    """
    Combined injury status for a single player from both data sources.

    Shows what the prop-based detection inferred vs what ESPN officially reports.
    """

    player_name: str
    team_abbr: str

    # Prop-based detection
    prop_detected_missing: bool = False
    prop_ppg: float = 0.0
    prop_role: str = ""

    # ESPN official status
    espn_status: Optional[str] = None          # e.g., "OUT", "Questionable", "Day-To-Day"
    espn_description: Optional[str] = None     # e.g., "Right Ankle Sprain"
    espn_date_updated: Optional[str] = None

    @property
    def display_status(self) -> str:
        """Combined display showing both sources."""
        if self.espn_status:
            status = f"[ESPN: {self.espn_status}]"
            if self.prop_detected_missing:
                return f"{status} [Props: confirmed missing]"
            return f"{status} [Props: has line]"
        if self.prop_detected_missing:
            return "[Props: no line — likely injured]"
        return "Active"


@dataclass
class TeamInjurySummary:
    """Per-team injury summary combining both data sources."""

    team_abbr: str
    team_short: str
    players: list[PlayerInjuryStatus] = field(default_factory=list)
    total_prop_missing: int = 0
    total_espn_injured: int = 0
    weighted_ppg_loss: float = 0.0


@dataclass
class MergedGameInjuryData:
    """
    Merged injury data for a single game from both sources.
    """

    game_id: str
    home_team: str
    away_team: str
    home_summary: Optional[TeamInjurySummary] = None
    away_summary: Optional[TeamInjurySummary] = None

    @property
    def has_any_injuries(self) -> bool:
        """Whether any injuries were detected by either source."""
        if self.home_summary and (self.home_summary.total_prop_missing > 0
                                  or self.home_summary.total_espn_injured > 0):
            return True
        if self.away_summary and (self.away_summary.total_prop_missing > 0
                                  or self.away_summary.total_espn_injured > 0):
            return True
        return False


# ── Integrator ─────────────────────────────────────────────────────────────


class ESPNInjuryIntegrator:
    """
    Merges ESPN official injury status with prop-based detection.

    Usage::

        integrator = ESPNInjuryIntegrator()
        merged_data = integrator.merge(prop_injury_data)
        for game in merged_data:
            print(game.home_summary.players)
    """

    def __init__(self, api_key: str = ""):
        self._scraper = ESPNInjuryScraper()
        self.api_key = api_key

    def merge(
        self,
        prop_injury_data: dict[str, GameInjuryData],
    ) -> dict[str, MergedGameInjuryData]:
        """
        Merge ESPN injury status into prop-based detection results.

        For each game in prop_injury_data, cross-references missing players
        against ESPN's official injury reports and vice versa.

        Args:
            prop_injury_data: Dict of game_id → GameInjuryData from
                            PlayerInjuryFetcher.

        Returns:
            Dict of game_id → MergedGameInjuryData.
        """
        if not prop_injury_data:
            logger.info("No prop injury data to merge with ESPN")
            return {}

        # Fetch all ESPN injury records
        espn_records = self._scraper.fetch_all()
        espn_by_team: dict[str, list[InjuryRecord]] = {}
        for rec in espn_records:
            if rec.team_abbr not in espn_by_team:
                espn_by_team[rec.team_abbr] = []
            espn_by_team[rec.team_abbr].append(rec)

        merged: dict[str, MergedGameInjuryData] = {}

        for game_id, gd in prop_injury_data.items():
            home_summary = self._merge_team(
                gd.home_impact, espn_by_team
            ) if gd.home_impact else None
            away_summary = self._merge_team(
                gd.away_impact, espn_by_team
            ) if gd.away_impact else None

            merged[game_id] = MergedGameInjuryData(
                game_id=game_id,
                home_team=gd.home_team,
                away_team=gd.away_team,
                home_summary=home_summary,
                away_summary=away_summary,
            )

        return merged

    def _merge_team(
        self,
        impact: InjuryImpact,
        espn_by_team: dict[str, list[InjuryRecord]],
    ) -> TeamInjurySummary:
        """
        Merge ESPN and prop data for a single team.

        Args:
            impact: InjuryImpact from PlayerInjuryFetcher.
            espn_by_team: ESPN records grouped by team abbreviation.

        Returns:
            TeamInjurySummary with combined player status info.
        """
        team_abbr = impact.team_abbr
        team_short = TEAM_ABBR_TO_SHORT.get(team_abbr, team_abbr)

        # ESPN records for this team
        team_espn_records = espn_by_team.get(team_abbr, [])
        espn_by_name: dict[str, InjuryRecord] = {
            r.player_name.lower(): r for r in team_espn_records
        }

        # Build per-player status combining both sources
        players: dict[str, PlayerInjuryStatus] = {}

        # Players detected as missing by props
        prop_missing_names = set()

        for star_str in impact.missing_stars:
            # Parse "Player Name (PPG, ROLE)"
            name = star_str.split(" (")[0].strip()
            prop_missing_names.add(name)

            pinfo = PLAYER_DATABASE.get(name, {})
            role = pinfo.get("role", "UNKNOWN")

            status = PlayerInjuryStatus(
                player_name=name,
                team_abbr=team_abbr,
                prop_detected_missing=True,
                prop_ppg=pinfo.get("ppg", 0.0),
                prop_role=role,
            )

            # Check ESPN status
            espn_rec = espn_by_name.get(name.lower())
            if espn_rec:
                status.espn_status = espn_rec.injury_status
                status.espn_description = espn_rec.injury_description
                status.espn_date_updated = espn_rec.date_updated

            players[name] = status

        # Add ESPN-only injured players (not caught by props)
        for espn_rec in team_espn_records:
            name = espn_rec.player_name
            if name not in players:
                # Check if player is in our database
                pinfo = PLAYER_DATABASE.get(name, {})
                status = PlayerInjuryStatus(
                    player_name=name,
                    team_abbr=team_abbr,
                    prop_detected_missing=False,
                    prop_ppg=pinfo.get("ppg", 0.0) if pinfo else 0.0,
                    prop_role=pinfo.get("role", "") if pinfo else "",
                    espn_status=espn_rec.injury_status,
                    espn_description=espn_rec.injury_description,
                    espn_date_updated=espn_rec.date_updated,
                )
                players[name] = status

        player_list = list(players.values())

        return TeamInjurySummary(
            team_abbr=team_abbr,
            team_short=team_short,
            players=player_list,
            total_prop_missing=len(prop_missing_names),
            total_espn_injured=len(team_espn_records),
            weighted_ppg_loss=impact.missing_ppg_weighted,
        )

    def get_display_lines(
        self,
        merged: MergedGameInjuryData,
    ) -> list[str]:
        """
        Generate human-readable display lines for merged injury data.

        Args:
            merged: MergedGameInjuryData from merge().

        Returns:
            List of display strings showing both sources.
        """
        lines = []

        if not merged.has_any_injuries:
            return lines

        lines.append(f"  {merged.away_team.split()[-1] if merged.away_team else '?'} @ "
                     f"{merged.home_team.split()[-1] if merged.home_team else '?'}:")
        lines.append(f"  {'─' * 60}")

        for side, summary in [("Home", merged.home_summary),
                               ("Away", merged.away_summary)]:
            if not summary or not summary.players:
                continue

            # ESPN-only players (not in our database)
            espn_only = [p for p in summary.players
                         if p.espn_status and not p.prop_detected_missing]
            # Prop-detected players
            prop_missing = [p for p in summary.players if p.prop_detected_missing]

            if prop_missing:
                lines.append(f"    {side} ({summary.team_short}) — Props: no line = likely out:")
                for p in prop_missing:
                    espn_tag = f"  [{p.espn_status}]" if p.espn_status else ""
                    desc_tag = f" — {p.espn_description}" if p.espn_description else ""
                    ppg_str = f" ({p.prop_ppg:.0f} PPG)" if p.prop_ppg > 0 else ""
                    lines.append(f"      • {p.player_name}{ppg_str}{espn_tag}{desc_tag}")

            if espn_only:
                lines.append(f"    {side} ({summary.team_short}) — ESPN official status:")
                for p in espn_only:
                    desc_tag = f" — {p.espn_description}" if p.espn_description else ""
                    ppg_str = f" ({p.prop_ppg:.0f} PPG)" if p.prop_ppg > 0 else ""
                    lines.append(f"      • {p.player_name}{ppg_str}   [{p.espn_status}]{desc_tag}")

            if summary.weighted_ppg_loss > 0:
                lines.append(
                    f"      Weighted PPG loss: ~{summary.weighted_ppg_loss:.1f}"
                )

        return lines
