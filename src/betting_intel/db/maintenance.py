"""
Database maintenance: cleanup of stale/leftover data in SQLite tables.

Provides periodic maintenance routines for:
1. Pipeline run cleanup — remove old pipeline_runs beyond a retention window
2. Game data cleanup — purge very old game records (optional)
3. Bet record cleanup — purge old bet records (optional)
4. Overall health/vacuum — reclaim disk space after deletes

Each method logs what it does and is safe to call repeatedly — missing
tables are silently skipped.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DatabaseMaintenance:
    """Periodic database cleanup routines for stale/leftover data.

    Usage:
        maint = DatabaseMaintenance(db_path=Path("data/betting.db"))
        report = maint.cleanup_all(dry_run=True)
        # -> {"pipeline_runs": 12, "games": 0, ...}
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path

    # ── Public API ─────────────────────────────────────────────────────────

    def cleanup_all(
        self,
        pipeline_run_days: int = 90,
        game_days: int = 365,
        bet_days: int = 180,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Run ALL cleanup routines and return a summary dict.

        Args:
            pipeline_run_days: Keep pipeline runs newer than this many days.
            game_days:        Keep game records newer than this many days.
            bet_days:         Keep bet records newer than this many days.
            dry_run:          If True, only count what *would* be deleted
                              without actually deleting.

        Returns:
            Dict mapping table name -> number of rows that would be /
            were deleted.
        """
        results: dict[str, int] = {}

        results["pipeline_runs"] = self.cleanup_pipeline_runs(
            retention_days=pipeline_run_days, dry_run=dry_run
        )
        results["games"] = self.cleanup_games(
            retention_days=game_days, dry_run=dry_run
        )
        results["bets"] = self.cleanup_bets(
            retention_days=bet_days, dry_run=dry_run
        )

        if not dry_run:
            self.vacuum()

        total = sum(results.values())
        if total:
            logger.info(
                "Database cleanup complete",
                dry_run=dry_run,
                total_rows=total,
                details=results,
            )
        else:
            logger.debug("Database cleanup: nothing to clean")

        return results

    def cleanup_pipeline_runs(self, retention_days: int = 90, dry_run: bool = False) -> int:
        """Delete pipeline_runs older than ``retention_days``."""
        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(
                "SELECT COUNT(*) FROM pipeline_runs WHERE started_at < ?",
                (cutoff,),
            )
            count = cursor.fetchone()[0]

            if count and not dry_run:
                conn.execute(
                    "DELETE FROM pipeline_runs WHERE started_at < ?",
                    (cutoff,),
                )
                conn.commit()

            conn.close()

            if count:
                label = "would delete" if dry_run else "deleted"
                logger.info(f"Cleanup: {label} {count} old pipeline_runs")
            return count
        except Exception as exc:
            logger.debug(f"Could not cleanup pipeline_runs: {exc}")
            return 0

    def cleanup_games(self, retention_days: int = 365, dry_run: bool = False) -> int:
        """Delete game records older than ``retention_days``."""
        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(
                "SELECT COUNT(*) FROM games WHERE game_date < ?",
                (cutoff,),
            )
            count = cursor.fetchone()[0]

            if count and not dry_run:
                conn.execute(
                    "DELETE FROM games WHERE game_date < ?",
                    (cutoff,),
                )
                conn.commit()

            conn.close()

            if count:
                label = "would delete" if dry_run else "deleted"
                logger.info(f"Cleanup: {label} {count} old game records")
            return count
        except Exception as exc:
            logger.debug(f"Could not cleanup games: {exc}")
            return 0

    def cleanup_bets(self, retention_days: int = 180, dry_run: bool = False) -> int:
        """Delete bet records older than ``retention_days``."""
        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(
                "SELECT COUNT(*) FROM bets WHERE game_date < ?",
                (cutoff,),
            )
            count = cursor.fetchone()[0]

            if count and not dry_run:
                conn.execute(
                    "DELETE FROM bets WHERE game_date < ?",
                    (cutoff,),
                )
                conn.commit()

            conn.close()

            if count:
                label = "would delete" if dry_run else "deleted"
                logger.info(f"Cleanup: {label} {count} old bet records")
            return count
        except Exception as exc:
            logger.debug(f"Could not cleanup bets: {exc}")
            return 0

    def vacuum(self) -> bool:
        """Reclaim disk space after deletes (VACUUM).

        This is a no-op for in-memory databases.
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("VACUUM")
            conn.close()
            logger.debug("Database vacuumed")
            return True
        except Exception as exc:
            logger.debug(f"Could not vacuum database: {exc}")
            return False
