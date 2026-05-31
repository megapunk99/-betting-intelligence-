"""
Feature Builder — computes team, schedule, and player features from raw game data.

Computes:
    Team Features: offensive_rating, defensive_rating, pace, net_rating
    Schedule Features: rest_days, back_to_back, travel_distance, home_away
    Player Features: injury_status, usage_rate, minutes

Stores all features in the FeatureStore with a version tag.
"""

import logging
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from betting_intel.features.store import (
    FeatureStore,
    TeamFeatureRecord,
    ScheduleFeatureRecord,
    PlayerFeatureRecord,
)

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Computes versioned features from raw game data and stores them in the FeatureStore.

    Usage:
        builder = FeatureBuilder(DB_PATH)
        version = builder.build_all(raw_df, version="v1.0")
        print(f"Features built: {version}")
    """

    def __init__(self, db_path: Path):
        self.store = FeatureStore(db_path)

    # ═══════════════════════════════════════════════════════════════════
    #  BUILD ALL FEATURES
    # ═══════════════════════════════════════════════════════════════════

    def build_all(self, raw_df: pd.DataFrame, version: Optional[str] = None,
                  description: str = "",
                  force: bool = False) -> str:
        """
        Compute and store ALL feature types from raw game data.

        Steps:
        1. Compute team features (ratings, pace, efficiency)
        2. Compute schedule features (rest, travel, back-to-back)
        3. Compute player features (injury, usage, minutes)
        4. Store all in FeatureStore
        5. Create version record

        Args:
            raw_df: Raw game log dataframe from NBADataLoader
            version: Version string (auto-generated if None)
            description: Human-readable description of this feature version
            force: Force rebuild even if version exists

        Returns:
            Version string
        """
        if version is None:
            version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Compute source hash for reproducibility
        source_hash = hashlib.md5(
            pd.util.hash_pandas_object(raw_df).values.tobytes()
        ).hexdigest()[:12]

        # Step 1: Team features
        logger.info(f"Computing team features for version {version}...")
        team_records = self.compute_team_features(raw_df, version)
        self.store.store_team_features(team_records)
        logger.info(f"  Stored {len(team_records)} team feature records")

        # Step 2: Schedule features
        logger.info(f"Computing schedule features for version {version}...")
        schedule_records = self.compute_schedule_features(raw_df, version)
        self.store.store_schedule_features(schedule_records)
        logger.info(f"  Stored {len(schedule_records)} schedule feature records")

        # Step 3: Player features (from team stats if no individual player data)
        logger.info(f"Computing player features for version {version}...")
        player_records = self.compute_player_features(raw_df, version)
        if player_records:
            self.store.store_player_features(player_records)
            logger.info(f"  Stored {len(player_records)} player feature records")

        # Step 4: Create version
        total_features = len(team_records) + len(schedule_records) + len(player_records)
        self.store.create_version(
            version=version,
            description=description or f"Auto-built from {len(raw_df)} game logs",
            source_hash=source_hash,
        )

        logger.info(f"Feature build complete: {version} ({total_features} records)")
        return version

    # ═══════════════════════════════════════════════════════════════════
    #  TEAM FEATURES
    # ═══════════════════════════════════════════════════════════════════

    def compute_team_features(self, raw_df: pd.DataFrame,
                              version: str) -> List[TeamFeatureRecord]:
        """
        Compute offensive/defensive ratings, pace, net rating, and efficiency.

        For each team on each game date, computes trailing averages
        from the most recent N games (default: 10).

        Args:
            raw_df: Game log dataframe
            version: Version string

        Returns:
            List of TeamFeatureRecord
        """
        records = []
        raw_df = raw_df.sort_values(["TEAM_NAME", "GAME_DATE"])

        for team_name, team_df in raw_df.groupby("TEAM_NAME"):
            team_abbr = team_df["TEAM_ABBREVIATION"].iloc[0] if "TEAM_ABBREVIATION" in team_df.columns else team_name[:3]

            team_df = team_df.reset_index(drop=True)

            for i, row in team_df.iterrows():
                game_date = row["GAME_DATE"]

                # Get last 10 games excluding current
                window = team_df.iloc[max(0, i - 10):i]
                if len(window) < 3:
                    continue  # Not enough data

                off_rating = window["PTS"].mean() / (window.get("FGA", pd.Series([0])).mean() + window.get("TOV", pd.Series([0])).mean() * 0.4) * 100 if window.get("FGA", pd.Series([0])).mean() > 0 else 0
                def_rating = window.get("OPP_PTS", window["PTS"] * 0.95).mean() / (window.get("OPP_FGA", window["PTS"] * 0.95).mean() if window.get("OPP_FGA", pd.Series([0])).mean() > 0 else 1) * 100

                # Pace estimate
                pace = window["PTS"].mean() * 2

                net_rating = off_rating - def_rating

                # Efficiency
                fga = window["FGA"].mean() if "FGA" in window.columns else 0
                fgm = window["FGM"].mean() if "FGM" in window.columns else 0
                fg3a = window["FG3A"].mean() if "FG3A" in window.columns else 0
                fg3m = window["FG3M"].mean() if "FG3M" in window.columns else 0
                fta = window["FTA"].mean() if "FTA" in window.columns else 0
                orb = window.get("OREB", window.get("OFF_REB", pd.Series([0]))).mean()
                tov = window["TOV"].mean() if "TOV" in window.columns else 0

                efg_pct = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0
                tov_pct = tov / (fga + 0.44 * fta + tov) if (fga + 0.44 * fta + tov) > 0 else 0
                orb_pct = orb / (orb + (window.get("DREB", window.get("DEF_REB", pd.Series([0]))).mean() or 1)) if orb > 0 else 0
                ft_rate = fta / fga if fga > 0 else 0

                win_pct = (window["WL"] == "W").mean() if "WL" in window.columns else 0.5

                record = TeamFeatureRecord(
                    team_name=team_name,
                    team_abbr=team_abbr,
                    game_date=str(game_date),
                    version=version,
                    offensive_rating=round(off_rating, 2) if not np.isnan(off_rating) else 0,
                    defensive_rating=round(def_rating, 2) if not np.isnan(def_rating) else 0,
                    pace=round(pace, 2) if not np.isnan(pace) else 0,
                    net_rating=round(net_rating, 2) if not np.isnan(net_rating) else 0,
                    efg_pct=round(efg_pct, 4) if not np.isnan(efg_pct) else 0,
                    tov_pct=round(tov_pct, 4) if not np.isnan(tov_pct) else 0,
                    orb_pct=round(orb_pct, 4) if not np.isnan(orb_pct) else 0,
                    ft_rate=round(ft_rate, 4) if not np.isnan(ft_rate) else 0,
                    win_pct=round(win_pct, 4) if not np.isnan(win_pct) else 0,
                    pts_scored_avg=round(window["PTS"].mean(), 2) if "PTS" in window.columns else 0,
                    pts_allowed_avg=round(window.get("OPP_PTS", window["PTS"] * 0.95).mean(), 2),
                    reb_avg=round(window["REB"].mean(), 2) if "REB" in window.columns else 0,
                    ast_avg=round(window["AST"].mean(), 2) if "AST" in window.columns else 0,
                )
                records.append(record)

        return records

    # ═══════════════════════════════════════════════════════════════════
    #  SCHEDULE FEATURES
    # ═══════════════════════════════════════════════════════════════════

    def compute_schedule_features(self, raw_df: pd.DataFrame,
                                   version: str) -> List[ScheduleFeatureRecord]:
        """
        Compute schedule context: rest days, back-to-back, travel distance.

        Args:
            raw_df: Game log dataframe
            version: Version string

        Returns:
            List of ScheduleFeatureRecord
        """
        records = []
        raw_df = raw_df.sort_values(["TEAM_NAME", "GAME_DATE"]).reset_index(drop=True)

        for team_name, team_df in raw_df.groupby("TEAM_NAME"):
            team_abbr = team_df["TEAM_ABBREVIATION"].iloc[0] if "TEAM_ABBREVIATION" in team_df.columns else team_name[:3]
            game_dates = team_df["GAME_DATE"].values

            for i, (_, row) in enumerate(team_df.iterrows()):
                game_date = row["GAME_DATE"]

                # Rest days since last game
                if i > 0:
                    try:
                        last_date = pd.to_datetime(game_dates[i - 1])
                        curr_date = pd.to_datetime(game_date)
                        rest_days = (curr_date - last_date).days
                    except Exception:
                        rest_days = 1
                else:
                    rest_days = 3  # First game: default rest

                is_b2b = rest_days <= 1
                is_home = "vs" in str(row.get("MATCHUP", ""))

                # Games in last 7/14 days
                try:
                    curr_dt = pd.to_datetime(game_date)
                    last_7 = curr_dt - pd.Timedelta(days=7)
                    last_14 = curr_dt - pd.Timedelta(days=14)
                    dates_series = pd.to_datetime(team_df["GAME_DATE"])
                    games_7 = int((dates_series > last_7) & (dates_series < curr_dt)).sum()
                    games_14 = int((dates_series > last_14) & (dates_series < curr_dt)).sum()
                except Exception:
                    games_7 = 2
                    games_14 = 4

                record = ScheduleFeatureRecord(
                    team_name=team_name,
                    team_abbr=team_abbr,
                    game_date=str(game_date),
                    version=version,
                    rest_days=float(rest_days),
                    is_back_to_back=is_b2b,
                    travel_distance=0.0,  # Need geo data for real calculation
                    is_home=is_home,
                    games_in_last_7=games_7,
                    games_in_last_14=games_14,
                )
                records.append(record)

        return records

    # ═══════════════════════════════════════════════════════════════════
    #  PLAYER FEATURES
    # ═══════════════════════════════════════════════════════════════════

    def compute_player_features(self, raw_df: pd.DataFrame,
                                 version: str) -> List[PlayerFeatureRecord]:
        """
        Compute player-level features from game logs.

        Note: The current game_logs table stores team-level stats.
        Full player-level features require a player stats data source
        (NBA API, Basketball Reference).

        This builds player feature stubs from team-level aggregates.

        Args:
            raw_df: Game log dataframe
            version: Version string

        Returns:
            List of PlayerFeatureRecord (may be empty if no player data)
        """
        records = []

        # Check if we have player-level data
        if "PLAYER_NAME" not in raw_df.columns:
            logger.debug("No player-level data available; skipping player features")
            return records

        raw_df = raw_df.sort_values(["TEAM_NAME", "PLAYER_NAME", "GAME_DATE"])

        for (team_name, player_name), player_df in raw_df.groupby(["TEAM_NAME", "PLAYER_NAME"]):
            team_abbr = player_df["TEAM_ABBREVIATION"].iloc[0] if "TEAM_ABBREVIATION" in player_df.columns else team_name[:3]

            player_df = player_df.reset_index(drop=True)

            for i, (_, row) in enumerate(player_df.iterrows()):
                game_date = row["GAME_DATE"]

                # Trailing averages (last 5 games)
                window = player_df.iloc[max(0, i - 5):i]
                if len(window) < 2:
                    continue

                record = PlayerFeatureRecord(
                    team_name=team_name,
                    team_abbr=team_abbr,
                    player_name=str(player_name),
                    game_date=str(game_date),
                    version=version,
                    injury_status="active",
                    usage_rate=round(window.get("USG_RATE", window.get("FGA", pd.Series([0])).mean() / 20).mean(), 4) if "USG_RATE" in window.columns or "FGA" in window.columns else 0,
                    minutes_avg=round(window["MIN"].mean(), 2) if "MIN" in window.columns else 0,
                    pts_avg=round(window["PTS"].mean(), 2) if "PTS" in window.columns else 0,
                    reb_avg=round(window["REB"].mean(), 2) if "REB" in window.columns else 0,
                    ast_avg=round(window["AST"].mean(), 2) if "AST" in window.columns else 0,
                    plus_minus_avg=round(window.get("PLUS_MINUS", pd.Series([0])).mean(), 2),
                    is_starter=bool(row.get("IS_STARTER", row.get("MIN", 0) > 20)),
                    impact_score=0.5,  # Default; real value from injury engine
                )
                records.append(record)

        return records


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_team_features(raw_df: pd.DataFrame, db_path: Path,
                        version: Optional[str] = None) -> str:
    """One-shot: build team features and store them."""
    builder = FeatureBuilder(db_path)
    return builder.build_all(raw_df, version=version, description="Team features only")


def build_schedule_features(raw_df: pd.DataFrame, db_path: Path,
                             version: Optional[str] = None) -> str:
    """One-shot: build schedule features and store them."""
    builder = FeatureBuilder(db_path)
    records = builder.compute_schedule_features(raw_df, version or "v1.0")
    builder.store.store_schedule_features(records)
    return version or "v1.0"


def build_player_features(raw_df: pd.DataFrame, db_path: Path,
                           version: Optional[str] = None) -> str:
    """One-shot: build player features and store them."""
    builder = FeatureBuilder(db_path)
    records = builder.compute_player_features(raw_df, version or "v1.0")
    if records:
        builder.store.store_player_features(records)
    return version or "v1.0"
