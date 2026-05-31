"""
Feature Store — persistent, versioned storage for team, schedule, and player features.

Each feature record is stored with a version tag and timestamp,
enabling full reproducibility and historical analysis.

Store:
    Team Features:
        - offensive_rating
        - defensive_rating
        - pace
        - net_rating

    Schedule Features:
        - rest_days
        - back_to_back
        - travel_distance
        - home_away

    Player Features:
        - injury_status
        - usage_rate
        - minutes

All features are versioned. Retrieval always includes the version metadata.
"""

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureVersion:
    """Version metadata for a feature set."""
    version: str                # e.g. "v1.0", "v2.1"
    created_at: str
    description: str = ""
    feature_count: int = 0
    source_hash: str = ""       # Hash of the data used to compute features
    model_compatible: bool = True


@dataclass
class TeamFeatureRecord:
    """A single team feature snapshot for a given date."""
    team_name: str
    team_abbr: str
    game_date: str
    version: str

    # Core ratings
    offensive_rating: float = 0.0
    defensive_rating: float = 0.0
    pace: float = 0.0
    net_rating: float = 0.0

    # Efficiency
    efg_pct: float = 0.0
    tov_pct: float = 0.0
    orb_pct: float = 0.0
    ft_rate: float = 0.0

    # Season stats
    win_pct: float = 0.0
    pts_scored_avg: float = 0.0
    pts_allowed_avg: float = 0.0
    reb_avg: float = 0.0
    ast_avg: float = 0.0

    # Metadata
    id: int = 0
    created_at: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    def feature_vector(self, keys: Optional[List[str]] = None) -> Dict[str, float]:
        """Extract numerical features as a dict."""
        base = {
            "offensive_rating": self.offensive_rating,
            "defensive_rating": self.defensive_rating,
            "pace": self.pace,
            "net_rating": self.net_rating,
            "efg_pct": self.efg_pct,
            "tov_pct": self.tov_pct,
            "orb_pct": self.orb_pct,
            "ft_rate": self.ft_rate,
            "win_pct": self.win_pct,
            "pts_scored_avg": self.pts_scored_avg,
            "pts_allowed_avg": self.pts_allowed_avg,
            "reb_avg": self.reb_avg,
            "ast_avg": self.ast_avg,
        }
        if keys:
            return {k: base.get(k, 0.0) for k in keys}
        return base


@dataclass
class ScheduleFeatureRecord:
    """Schedule context for a team on a given date."""
    team_name: str
    team_abbr: str
    game_date: str
    version: str

    rest_days: float = 0.0
    is_back_to_back: bool = False
    travel_distance: float = 0.0      # miles traveled since last game
    is_home: bool = True
    games_in_last_7: int = 0
    games_in_last_14: int = 0

    id: int = 0
    created_at: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    def feature_vector(self) -> Dict[str, float]:
        return {
            "rest_days": self.rest_days,
            "is_back_to_back": float(self.is_back_to_back),
            "travel_distance": self.travel_distance,
            "is_home": float(self.is_home),
            "games_in_last_7": float(self.games_in_last_7),
            "games_in_last_14": float(self.games_in_last_14),
        }


@dataclass
class PlayerFeatureRecord:
    """Player-level features for a given team on a given date."""
    team_name: str
    team_abbr: str
    player_name: str
    game_date: str
    version: str

    injury_status: str = "active"       # active, questionable, doubtful, out
    usage_rate: float = 0.0
    minutes_avg: float = 0.0
    pts_avg: float = 0.0
    reb_avg: float = 0.0
    ast_avg: float = 0.0
    plus_minus_avg: float = 0.0

    is_starter: bool = False
    impact_score: float = 0.0           # 0-1, how much this player matters

    id: int = 0
    created_at: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    def feature_vector(self) -> Dict[str, float]:
        return {
            "usage_rate": self.usage_rate,
            "minutes_avg": self.minutes_avg,
            "pts_avg": self.pts_avg,
            "reb_avg": self.reb_avg,
            "ast_avg": self.ast_avg,
            "plus_minus_avg": self.plus_minus_avg,
            "is_starter": float(self.is_starter),
            "impact_score": self.impact_score,
            "is_injured": float(self.injury_status != "active"),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE STORE
# ═══════════════════════════════════════════════════════════════════════════

class FeatureStore:
    """
    Versioned feature store for team, schedule, and player features.

    Design:
    - Each feature record is tagged with a version string
    - Versions enable reproducibility and rollback
    - Retrieval can filter by team, date range, or version

    Usage:
        store = FeatureStore(DB_PATH)

        # Store features
        store.store_team_features(team_records)
        store.store_schedule_features(schedule_records)
        store.store_player_features(player_records)

        # Retrieve
        team_feats = store.get_team_features("Spurs", "2026-06-04")
        schedule = store.get_schedule_features("Spurs", "2026-06-04")
        players = store.get_player_features("Spurs", "2026-06-04")

        # Versions
        current = store.get_current_version()
        versions = store.list_versions()
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Create feature store tables."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS feature_versions (
                    version         TEXT PRIMARY KEY,
                    created_at      TEXT NOT NULL,
                    description     TEXT DEFAULT '',
                    feature_count   INTEGER DEFAULT 0,
                    source_hash     TEXT DEFAULT '',
                    model_compatible INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS team_features (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_name       TEXT NOT NULL,
                    team_abbr       TEXT NOT NULL,
                    game_date       TEXT NOT NULL,
                    version         TEXT NOT NULL,
                    offensive_rating REAL DEFAULT 0,
                    defensive_rating REAL DEFAULT 0,
                    pace            REAL DEFAULT 0,
                    net_rating      REAL DEFAULT 0,
                    efg_pct         REAL DEFAULT 0,
                    tov_pct         REAL DEFAULT 0,
                    orb_pct         REAL DEFAULT 0,
                    ft_rate         REAL DEFAULT 0,
                    win_pct         REAL DEFAULT 0,
                    pts_scored_avg  REAL DEFAULT 0,
                    pts_allowed_avg REAL DEFAULT 0,
                    reb_avg         REAL DEFAULT 0,
                    ast_avg         REAL DEFAULT 0,
                    created_at      TEXT DEFAULT (datetime('now')),
                    UNIQUE(team_name, game_date, version)
                );

                CREATE TABLE IF NOT EXISTS schedule_features (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_name       TEXT NOT NULL,
                    team_abbr       TEXT NOT NULL,
                    game_date       TEXT NOT NULL,
                    version         TEXT NOT NULL,
                    rest_days       REAL DEFAULT 0,
                    is_back_to_back INTEGER DEFAULT 0,
                    travel_distance REAL DEFAULT 0,
                    is_home         INTEGER DEFAULT 1,
                    games_in_last_7 INTEGER DEFAULT 0,
                    games_in_last_14 INTEGER DEFAULT 0,
                    created_at      TEXT DEFAULT (datetime('now')),
                    UNIQUE(team_name, game_date, version)
                );

                CREATE TABLE IF NOT EXISTS player_features (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    team_name       TEXT NOT NULL,
                    team_abbr       TEXT NOT NULL,
                    player_name     TEXT NOT NULL,
                    game_date       TEXT NOT NULL,
                    version         TEXT NOT NULL,
                    injury_status   TEXT DEFAULT 'active',
                    usage_rate      REAL DEFAULT 0,
                    minutes_avg     REAL DEFAULT 0,
                    pts_avg         REAL DEFAULT 0,
                    reb_avg         REAL DEFAULT 0,
                    ast_avg         REAL DEFAULT 0,
                    plus_minus_avg  REAL DEFAULT 0,
                    is_starter      INTEGER DEFAULT 0,
                    impact_score    REAL DEFAULT 0,
                    created_at      TEXT DEFAULT (datetime('now')),
                    UNIQUE(team_name, player_name, game_date, version)
                );

                CREATE INDEX IF NOT EXISTS idx_team_feat_team ON team_features(team_name);
                CREATE INDEX IF NOT EXISTS idx_team_feat_date ON team_features(game_date);
                CREATE INDEX IF NOT EXISTS idx_team_feat_version ON team_features(version);

                CREATE INDEX IF NOT EXISTS idx_sched_feat_team ON schedule_features(team_name);
                CREATE INDEX IF NOT EXISTS idx_sched_feat_date ON schedule_features(game_date);

                CREATE INDEX IF NOT EXISTS idx_player_feat_team ON player_features(team_name);
                CREATE INDEX IF NOT EXISTS idx_player_feat_player ON player_features(player_name);
                CREATE INDEX IF NOT EXISTS idx_player_feat_date ON player_features(game_date);
            """)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ═══════════════════════════════════════════════════════════════════
    #  VERSION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def create_version(self, version: str, description: str = "",
                       source_hash: str = "") -> FeatureVersion:
        """Create a new feature version."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO feature_versions (version, created_at, description, source_hash) "
                "VALUES (?, ?, ?, ?)",
                (version, now, description, source_hash)
            )
            conn.commit()
        return FeatureVersion(
            version=version,
            created_at=now,
            description=description,
            source_hash=source_hash,
        )

    def get_current_version(self) -> Optional[str]:
        """Get the most recent version string."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT version FROM feature_versions ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return row["version"] if row else None

    def list_versions(self) -> List[FeatureVersion]:
        """List all feature versions."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feature_versions ORDER BY created_at DESC"
            ).fetchall()
        return [
            FeatureVersion(
                version=r["version"],
                created_at=r["created_at"],
                description=r["description"],
                source_hash=r["source_hash"],
            )
            for r in rows
        ]

    # ═══════════════════════════════════════════════════════════════════
    #  STORE FEATURES
    # ═══════════════════════════════════════════════════════════════════

    def store_team_features(self, records: List[TeamFeatureRecord]):
        """Store team feature records."""
        with self._connect() as conn:
            for r in records:
                conn.execute(
                    """INSERT OR REPLACE INTO team_features
                       (team_name, team_abbr, game_date, version,
                        offensive_rating, defensive_rating, pace, net_rating,
                        efg_pct, tov_pct, orb_pct, ft_rate,
                        win_pct, pts_scored_avg, pts_allowed_avg, reb_avg, ast_avg)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (r.team_name, r.team_abbr, r.game_date, r.version,
                     r.offensive_rating, r.defensive_rating, r.pace, r.net_rating,
                     r.efg_pct, r.tov_pct, r.orb_pct, r.ft_rate,
                     r.win_pct, r.pts_scored_avg, r.pts_allowed_avg, r.reb_avg, r.ast_avg)
                )
            conn.commit()

    def store_schedule_features(self, records: List[ScheduleFeatureRecord]):
        """Store schedule feature records."""
        with self._connect() as conn:
            for r in records:
                conn.execute(
                    """INSERT OR REPLACE INTO schedule_features
                       (team_name, team_abbr, game_date, version,
                        rest_days, is_back_to_back, travel_distance, is_home,
                        games_in_last_7, games_in_last_14)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (r.team_name, r.team_abbr, r.game_date, r.version,
                     r.rest_days, int(r.is_back_to_back), r.travel_distance, int(r.is_home),
                     r.games_in_last_7, r.games_in_last_14)
                )
            conn.commit()

    def store_player_features(self, records: List[PlayerFeatureRecord]):
        """Store player feature records."""
        with self._connect() as conn:
            for r in records:
                conn.execute(
                    """INSERT OR REPLACE INTO player_features
                       (team_name, team_abbr, player_name, game_date, version,
                        injury_status, usage_rate, minutes_avg,
                        pts_avg, reb_avg, ast_avg, plus_minus_avg,
                        is_starter, impact_score)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (r.team_name, r.team_abbr, r.player_name, r.game_date, r.version,
                     r.injury_status, r.usage_rate, r.minutes_avg,
                     r.pts_avg, r.reb_avg, r.ast_avg, r.plus_minus_avg,
                     int(r.is_starter), r.impact_score)
                )
            conn.commit()

    # ═══════════════════════════════════════════════════════════════════
    #  RETRIEVE FEATURES
    # ═══════════════════════════════════════════════════════════════════

    def get_team_features(self, team_name: str, game_date: str,
                          version: Optional[str] = None) -> Optional[TeamFeatureRecord]:
        """Get team features for a specific team and date."""
        version = version or self.get_current_version()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM team_features WHERE team_name = ? AND game_date = ? AND version = ?",
                (team_name, game_date, version)
            ).fetchone()
            if row:
                return self._row_to_team_record(row)
        return None

    def get_team_features_bulk(self, team_names: List[str], game_date: str,
                               version: Optional[str] = None) -> Dict[str, TeamFeatureRecord]:
        """Get team features for multiple teams on a given date."""
        version = version or self.get_current_version()
        results = {}
        with self._connect() as conn:
            placeholders = ",".join("?" * len(team_names))
            rows = conn.execute(
                f"SELECT * FROM team_features WHERE team_name IN ({placeholders}) AND game_date = ? AND version = ?",
                (*team_names, game_date, version)
            ).fetchall()
            for row in rows:
                results[row["team_name"]] = self._row_to_team_record(row)
        return results

    def get_schedule_features(self, team_name: str, game_date: str,
                              version: Optional[str] = None) -> Optional[ScheduleFeatureRecord]:
        """Get schedule features for a specific team and date."""
        version = version or self.get_current_version()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM schedule_features WHERE team_name = ? AND game_date = ? AND version = ?",
                (team_name, game_date, version)
            ).fetchone()
            if row:
                return self._row_to_schedule_record(row)
        return None

    def get_player_features(self, team_name: str, game_date: str,
                            version: Optional[str] = None) -> List[PlayerFeatureRecord]:
        """Get player features for a specific team and date."""
        version = version or self.get_current_version()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM player_features WHERE team_name = ? AND game_date = ? AND version = ?",
                (team_name, game_date, version)
            ).fetchall()
        return [self._row_to_player_record(r) for r in rows]

    def get_game_feature_vector(self, home_team: str, away_team: str,
                                game_date: str,
                                version: Optional[str] = None) -> Dict[str, float]:
        """Build a complete feature vector for a game from stored features."""
        version = version or self.get_current_version()
        features = {}

        home_team_feat = self.get_team_features(home_team, game_date, version)
        away_team_feat = self.get_team_features(away_team, game_date, version)
        home_sched = self.get_schedule_features(home_team, game_date, version)
        away_sched = self.get_schedule_features(away_team, game_date, version)

        # Team features
        if home_team_feat:
            for k, v in home_team_feat.feature_vector().items():
                features[f"home_{k}"] = v
        if away_team_feat:
            for k, v in away_team_feat.feature_vector().items():
                features[f"away_{k}"] = v

        # Schedule features
        if home_sched:
            for k, v in home_sched.feature_vector().items():
                features[f"home_{k}"] = features.get(f"home_{k}", 0.0) or v
        if away_sched:
            for k, v in away_sched.feature_vector().items():
                features[f"away_{k}"] = features.get(f"away_{k}", 0.0) or v

        # Differences
        if home_team_feat and away_team_feat:
            features["net_rating_diff"] = home_team_feat.net_rating - away_team_feat.net_rating
            features["pace_diff"] = home_team_feat.pace - away_team_feat.pace
            features["off_rating_diff"] = home_team_feat.offensive_rating - away_team_feat.offensive_rating
            features["def_rating_diff"] = home_team_feat.defensive_rating - away_team_feat.defensive_rating

        return features

    # ═══════════════════════════════════════════════════════════════════
    #  ROW CONVERTERS
    # ═══════════════════════════════════════════════════════════════════

    def _row_to_team_record(self, row) -> TeamFeatureRecord:
        return TeamFeatureRecord(
            id=row["id"],
            team_name=row["team_name"],
            team_abbr=row["team_abbr"],
            game_date=row["game_date"],
            version=row["version"],
            offensive_rating=row["offensive_rating"],
            defensive_rating=row["defensive_rating"],
            pace=row["pace"],
            net_rating=row["net_rating"],
            efg_pct=row["efg_pct"],
            tov_pct=row["tov_pct"],
            orb_pct=row["orb_pct"],
            ft_rate=row["ft_rate"],
            win_pct=row["win_pct"],
            pts_scored_avg=row["pts_scored_avg"],
            pts_allowed_avg=row["pts_allowed_avg"],
            reb_avg=row["reb_avg"],
            ast_avg=row["ast_avg"],
            created_at=row["created_at"],
        )

    def _row_to_schedule_record(self, row) -> ScheduleFeatureRecord:
        return ScheduleFeatureRecord(
            id=row["id"],
            team_name=row["team_name"],
            team_abbr=row["team_abbr"],
            game_date=row["game_date"],
            version=row["version"],
            rest_days=row["rest_days"],
            is_back_to_back=bool(row["is_back_to_back"]),
            travel_distance=row["travel_distance"],
            is_home=bool(row["is_home"]),
            games_in_last_7=row["games_in_last_7"],
            games_in_last_14=row["games_in_last_14"],
            created_at=row["created_at"],
        )

    def _row_to_player_record(self, row) -> PlayerFeatureRecord:
        return PlayerFeatureRecord(
            id=row["id"],
            team_name=row["team_name"],
            team_abbr=row["team_abbr"],
            player_name=row["player_name"],
            game_date=row["game_date"],
            version=row["version"],
            injury_status=row["injury_status"],
            usage_rate=row["usage_rate"],
            minutes_avg=row["minutes_avg"],
            pts_avg=row["pts_avg"],
            reb_avg=row["reb_avg"],
            ast_avg=row["ast_avg"],
            plus_minus_avg=row["plus_minus_avg"],
            is_starter=bool(row["is_starter"]),
            impact_score=row["impact_score"],
            created_at=row["created_at"],
        )

    # ═══════════════════════════════════════════════════════════════════
    #  STATISTICS
    # ═══════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._connect() as conn:
            return {
                "team_features": conn.execute("SELECT COUNT(*) FROM team_features").fetchone()[0],
                "schedule_features": conn.execute("SELECT COUNT(*) FROM schedule_features").fetchone()[0],
                "player_features": conn.execute("SELECT COUNT(*) FROM player_features").fetchone()[0],
                "versions": conn.execute("SELECT COUNT(*) FROM feature_versions").fetchone()[0],
                "teams_tracked": conn.execute("SELECT COUNT(DISTINCT team_name) FROM team_features").fetchone()[0],
                "dates_tracked": conn.execute("SELECT COUNT(DISTINCT game_date) FROM team_features").fetchone()[0],
            }
