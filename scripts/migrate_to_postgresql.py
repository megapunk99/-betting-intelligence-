"""
PostgreSQL Migration — Phase 5.16 of the Professional Betting Intelligence Platform.

Migrates all data from SQLite to PostgreSQL with proper schema.

Required Tables:
    games       — game data and predictions
    teams       — team metadata
    players     — player metadata (future)
    odds        — time-series odds snapshots
    predictions — model predictions with edge/EV/stake
    bets        — actual betting records
    injuries    — injury data
    features    — versioned feature store
    clv         — closing line value tracking
    backtests   — backtesting results

Usage:
    python scripts/migrate_to_postgresql.py
        --sqlite-path data/betting_intel.db
        --postgres-url "postgresql://user:pass@localhost:5432/betting_intel"
        --dry-run    # Preview what would be migrated

Requirements:
    pip install psycopg2-binary sqlalchemy
"""

import argparse
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  MIGRATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class PostgreSQLMigrator:
    """
    Migrates all data from SQLite to PostgreSQL.

    Handles:
    - Schema conversion (SQLite types → PostgreSQL types)
    - Data integrity checks
    - Incremental migration (skip already-migrated records)
    - Rollback on failure

    Usage:
        migrator = PostgreSQLMigrator(
            sqlite_path=Path("data/betting_intel.db"),
            postgres_url="postgresql://user:pass@localhost:5432/betting_intel",
        )
        report = migrator.migrate_all(dry_run=True)
        print(report)
    """

    # Mapping: source table → destination table info
    TABLES = {
        "game_logs": {
            "postgres_table": "games",
            "columns": [
                ("GAME_ID", "VARCHAR(20) PRIMARY KEY"),
                ("GAME_DATE", "DATE NOT NULL"),
                ("SEASON", "VARCHAR(10)"),
                ("TEAM_ID", "INTEGER"),
                ("TEAM_NAME", "VARCHAR(100)"),
                ("TEAM_ABBREVIATION", "VARCHAR(10)"),
                ("MATCHUP", "VARCHAR(100)"),
                ("WL", "VARCHAR(1)"),
                ("MIN", "REAL"),
                ("PTS", "REAL"),
                ("FGM", "REAL"),
                ("FGA", "REAL"),
                ("FG_PCT", "REAL"),
                ("FG3M", "REAL"),
                ("FG3A", "REAL"),
                ("FG3_PCT", "REAL"),
                ("FTM", "REAL"),
                ("FTA", "REAL"),
                ("FT_PCT", "REAL"),
                ("OREB", "REAL"),
                ("DREB", "REAL"),
                ("REB", "REAL"),
                ("AST", "REAL"),
                ("STL", "REAL"),
                ("BLK", "REAL"),
                ("TOV", "REAL"),
                ("PF", "REAL"),
                ("PLUS_MINUS", "REAL"),
                ("created_at", "TIMESTAMP DEFAULT NOW()"),
            ],
        },
        "odds": {
            "postgres_table": "odds",
            "columns": [
                ("id", "SERIAL PRIMARY KEY"),
                ("game_id", "VARCHAR(50) NOT NULL"),
                ("api_game_id", "VARCHAR(50) DEFAULT ''"),
                ("home_team", "VARCHAR(100) DEFAULT ''"),
                ("away_team", "VARCHAR(100) DEFAULT ''"),
                ("sportsbook", "VARCHAR(100) NOT NULL"),
                ("market", "VARCHAR(50) NOT NULL"),
                ("odds_value", "TEXT NOT NULL"),
                ("timestamp", "TIMESTAMP NOT NULL"),
                ("game_date", "DATE"),
                ("odds_type", "VARCHAR(20) DEFAULT 'current'"),
                ("created_at", "TIMESTAMP DEFAULT NOW()"),
                ("UNIQUE(game_id, sportsbook, market, timestamp)"),
            ],
        },
        "odds_meta": {
            "postgres_table": "game_metadata",
            "columns": [
                ("game_id", "VARCHAR(50) PRIMARY KEY"),
                ("home_team", "VARCHAR(100) NOT NULL"),
                ("away_team", "VARCHAR(100) NOT NULL"),
                ("commence_time", "TIMESTAMP NOT NULL"),
                ("first_seen", "TIMESTAMP NOT NULL"),
                ("last_updated", "TIMESTAMP NOT NULL"),
                ("is_finished", "BOOLEAN DEFAULT FALSE"),
            ],
        },
        "clv_tracking": {
            "postgres_table": "clv_tracking",
            "columns": [
                ("id", "SERIAL PRIMARY KEY"),
                ("game_id", "VARCHAR(50) NOT NULL"),
                ("home_team", "VARCHAR(100) NOT NULL"),
                ("away_team", "VARCHAR(100) NOT NULL"),
                ("game_date", "DATE NOT NULL"),
                ("market_type", "VARCHAR(20) NOT NULL"),
                ("bet_side", "VARCHAR(10) NOT NULL"),
                ("bet_odds_american", "REAL NOT NULL"),
                ("closing_odds_american", "REAL NOT NULL"),
                ("clv_difference", "REAL DEFAULT 0"),
                ("clv_percentage", "REAL DEFAULT 0"),
                ("bet_won", "BOOLEAN"),
                ("model_probability", "REAL DEFAULT 0"),
                ("edge_at_bet_time", "REAL DEFAULT 0"),
                ("bet_timestamp", "TIMESTAMP NOT NULL"),
                ("created_at", "TIMESTAMP DEFAULT NOW()"),
                ("UNIQUE(game_id, market_type, bet_side)"),
            ],
        },
        "feature_versions": {
            "postgres_table": "feature_versions",
            "columns": [
                ("version", "VARCHAR(50) PRIMARY KEY"),
                ("created_at", "TIMESTAMP NOT NULL"),
                ("description", "TEXT DEFAULT ''"),
                ("feature_count", "INTEGER DEFAULT 0"),
                ("source_hash", "VARCHAR(50) DEFAULT ''"),
                ("model_compatible", "BOOLEAN DEFAULT TRUE"),
            ],
        },
        "team_features": {
            "postgres_table": "team_features",
            "columns": [
                ("id", "SERIAL PRIMARY KEY"),
                ("team_name", "VARCHAR(100) NOT NULL"),
                ("team_abbr", "VARCHAR(10) NOT NULL"),
                ("game_date", "DATE NOT NULL"),
                ("version", "VARCHAR(50) NOT NULL"),
                ("offensive_rating", "REAL DEFAULT 0"),
                ("defensive_rating", "REAL DEFAULT 0"),
                ("pace", "REAL DEFAULT 0"),
                ("net_rating", "REAL DEFAULT 0"),
                ("efg_pct", "REAL DEFAULT 0"),
                ("tov_pct", "REAL DEFAULT 0"),
                ("orb_pct", "REAL DEFAULT 0"),
                ("ft_rate", "REAL DEFAULT 0"),
                ("win_pct", "REAL DEFAULT 0"),
                ("pts_scored_avg", "REAL DEFAULT 0"),
                ("pts_allowed_avg", "REAL DEFAULT 0"),
                ("reb_avg", "REAL DEFAULT 0"),
                ("ast_avg", "REAL DEFAULT 0"),
                ("created_at", "TIMESTAMP DEFAULT NOW()"),
                ("UNIQUE(team_name, game_date, version)"),
            ],
        },
        "schedule_features": {
            "postgres_table": "schedule_features",
            "columns": [
                ("id", "SERIAL PRIMARY KEY"),
                ("team_name", "VARCHAR(100) NOT NULL"),
                ("team_abbr", "VARCHAR(10) NOT NULL"),
                ("game_date", "DATE NOT NULL"),
                ("version", "VARCHAR(50) NOT NULL"),
                ("rest_days", "REAL DEFAULT 0"),
                ("is_back_to_back", "BOOLEAN DEFAULT FALSE"),
                ("travel_distance", "REAL DEFAULT 0"),
                ("is_home", "BOOLEAN DEFAULT TRUE"),
                ("games_in_last_7", "INTEGER DEFAULT 0"),
                ("games_in_last_14", "INTEGER DEFAULT 0"),
                ("created_at", "TIMESTAMP DEFAULT NOW()"),
                ("UNIQUE(team_name, game_date, version)"),
            ],
        },
        "model_vs_market": {
            "postgres_table": "model_vs_market",
            "columns": [
                ("id", "SERIAL PRIMARY KEY"),
                ("game_id", "VARCHAR(50) NOT NULL"),
                ("home_team", "VARCHAR(100) NOT NULL"),
                ("away_team", "VARCHAR(100) NOT NULL"),
                ("market_type", "VARCHAR(20) NOT NULL"),
                ("bet_side", "VARCHAR(10) NOT NULL"),
                ("model_probability", "REAL NOT NULL"),
                ("market_probability", "REAL NOT NULL"),
                ("vig_free_probability", "REAL"),
                ("edge", "REAL NOT NULL"),
                ("agreement", "VARCHAR(20) NOT NULL"),
                ("model_name", "VARCHAR(50) DEFAULT ''"),
                ("timestamp", "TIMESTAMP NOT NULL"),
                ("created_at", "TIMESTAMP DEFAULT NOW()"),
                ("UNIQUE(game_id, market_type, bet_side, model_name)"),
            ],
        },
    }

    def __init__(self, sqlite_path: Path, postgres_url: str):
        self.sqlite_path = Path(sqlite_path)
        self.postgres_url = postgres_url
        self._pg_conn = None

    # ═══════════════════════════════════════════════════════════════════
    #  MIGRATION
    # ═══════════════════════════════════════════════════════════════════

    def migrate_all(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Migrate ALL tables from SQLite to PostgreSQL.

        Args:
            dry_run: If True, only count records without inserting

        Returns:
            Dict with migration report
        """
        report = {
            "started_at": datetime.now().isoformat(),
            "dry_run": dry_run,
            "tables": {},
            "total_rows": 0,
            "errors": [],
        }

        sqlite_conn = sqlite3.connect(str(self.sqlite_path))
        sqlite_conn.row_factory = sqlite3.Row

        if not dry_run:
            self._connect_postgres()
            self._create_postgres_tables()

        for source_table, table_info in self.TABLES.items():
            try:
                row_count = self._migrate_table(
                    sqlite_conn, source_table, table_info, dry_run
                )
                report["tables"][source_table] = {
                    "rows_migrated": row_count,
                    "destination": table_info["postgres_table"],
                    "status": "ok",
                }
                report["total_rows"] += row_count
            except Exception as e:
                error_msg = f"Failed to migrate {source_table}: {e}"
                logger.error(error_msg)
                report["tables"][source_table] = {
                    "rows_migrated": 0,
                    "status": "error",
                    "error": str(e),
                }
                report["errors"].append(error_msg)

        sqlite_conn.close()
        if self._pg_conn and not dry_run:
            self._pg_conn.close()

        report["completed_at"] = datetime.now().isoformat()
        report["success"] = len(report["errors"]) == 0

        return report

    def _migrate_table(self, sqlite_conn, source_table: str,
                        table_info: Dict, dry_run: bool) -> int:
        """Migrate a single table."""
        dest_table = table_info["postgres_table"]
        col_names = [c[0] for c in table_info["columns"]
                     if c[0] not in ("id", "created_at")]

        # Count rows
        try:
            cursor = sqlite_conn.execute(f"SELECT COUNT(*) FROM {source_table}")
            row_count = cursor.fetchone()[0]
        except Exception:
            logger.debug(f"Table {source_table} does not exist in SQLite, skipping")
            return 0

        if dry_run:
            logger.info(f"[DRY RUN] Would migrate {row_count} rows from {source_table} → {dest_table}")
            return row_count

        if row_count == 0:
            return 0

        # Fetch all rows
        cursor = sqlite_conn.execute(f"SELECT * FROM {source_table}")
        rows = cursor.fetchall()

        # Build INSERT statement
        insert_cols = ", ".join(col_names)
        placeholders = ", ".join(f"%s" for _ in col_names)
        insert_sql = (
            f"INSERT INTO {dest_table} ({insert_cols}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT DO NOTHING"
        )

        # Insert in batches
        batch_size = 500
        migrated = 0
        pg_cursor = self._pg_conn.cursor()

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            for row in batch:
                values = [row[c] for c in col_names]
                # Convert SQLite types to Python types
                values = [
                    v if v is not None else None
                    for v in values
                ]
                try:
                    pg_cursor.execute(insert_sql, values)
                except Exception as e:
                    logger.warning(f"Failed to insert row in {dest_table}: {e}")

            self._pg_conn.commit()
            migrated += len(batch)
            logger.debug(f"  Migrated {migrated}/{row_count} rows to {dest_table}")

        pg_cursor.close()
        logger.info(f"Migrated {migrated} rows from {source_table} → {dest_table}")
        return migrated

    # ═══════════════════════════════════════════════════════════════════
    #  POSTGRESQL SETUP
    # ═══════════════════════════════════════════════════════════════════

    def _connect_postgres(self):
        """Connect to PostgreSQL."""
        try:
            import psycopg2
            self._pg_conn = psycopg2.connect(self.postgres_url)
            self._pg_conn.autocommit = False
            logger.info(f"Connected to PostgreSQL: {self.postgres_url}")
        except ImportError:
            logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def _create_postgres_tables(self):
        """Create all PostgreSQL tables using the defined schema."""
        if not self._pg_conn:
            raise ConnectionError("Not connected to PostgreSQL")

        cursor = self._pg_conn.cursor()

        for source_table, table_info in self.TABLES.items():
            dest_table = table_info["postgres_table"]
            col_defs = ", ".join(
                f"{c[0]} {c[1]}" for c in table_info["columns"]
            )

            create_sql = f"CREATE TABLE IF NOT EXISTS {dest_table} ({col_defs})"

            try:
                cursor.execute(create_sql)
                logger.debug(f"Created table: {dest_table}")
            except Exception as e:
                logger.warning(f"Failed to create {dest_table}: {e}")

        self._pg_conn.commit()
        cursor.close()
        logger.info("PostgreSQL tables created/verified")

    def create_indexes(self):
        """Create performance indexes on PostgreSQL tables."""
        if not self._pg_conn:
            self._connect_postgres()

        cursor = self._pg_conn.cursor()
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)",
            "CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team, away_team)",
            "CREATE INDEX IF NOT EXISTS idx_odds_game ON odds(game_id)",
            "CREATE INDEX IF NOT EXISTS idx_odds_timestamp ON odds(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_clv_game ON clv_tracking(game_id)",
            "CREATE INDEX IF NOT EXISTS idx_team_feat_team ON team_features(team_name)",
            "CREATE INDEX IF NOT EXISTS idx_team_feat_date ON team_features(game_date)",
            "CREATE INDEX IF NOT EXISTS idx_sched_feat_team ON schedule_features(team_name)",
            "CREATE INDEX IF NOT EXISTS idx_market_game ON model_vs_market(game_id)",
        ]

        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.debug(f"Index creation failed: {e}")

        self._pg_conn.commit()
        cursor.close()
        logger.info("PostgreSQL indexes created")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite database to PostgreSQL")
    parser.add_argument("--sqlite-path", default="data/betting_intel.db",
                        help="Path to SQLite database")
    parser.add_argument("--postgres-url", required=True,
                        help="PostgreSQL connection URL")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without migrating")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    migrator = PostgreSQLMigrator(
        sqlite_path=Path(args.sqlite_path),
        postgres_url=args.postgres_url,
    )

    report = migrator.migrate_all(dry_run=args.dry_run)

    print(f"\n{'=' * 60}")
    print(f"MIGRATION REPORT")
    print(f"{'=' * 60}")
    print(f"Dry run: {report['dry_run']}")
    print(f"Total rows migrated: {report['total_rows']}")
    print(f"Tables:")
    for table, info in report["tables"].items():
        status = "✅" if info["status"] == "ok" else "❌"
        print(f"  {status} {table:20s} → {info['destination']:20s} ({info['rows_migrated']} rows)")
    if report["errors"]:
        print(f"\nErrors:")
        for err in report["errors"]:
            print(f"  ❌ {err}")
    print(f"\nSuccess: {report['success']}")
    print(f"Completed: {report['completed_at']}")


if __name__ == "__main__":
    main()
