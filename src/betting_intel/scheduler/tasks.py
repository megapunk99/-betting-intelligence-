"""
Task scheduler for automated pipeline runs and data updates.
Can be run as a standalone process or integrated with cron/airflow.

Currently supports:
- Daily pipeline runs
- Weekly model retraining
- Periodic database cleanup
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

from betting_intel.config import settings
from betting_intel.services import logger, setup_logging


class TaskScheduler:
    """Simple task scheduler for periodic pipeline execution."""

    def __init__(self, interval_minutes: int = 60):
        self.interval = timedelta(minutes=interval_minutes)
        self.last_run: Optional[datetime] = None
        self._running = False

    def run_pipeline(self) -> dict:
        """Execute the full data pipeline."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        logger.info("Scheduled pipeline run starting", run_id=run_id)

        from betting_intel.main import BettingIntelligenceSystem

        start = time.time()
        system = BettingIntelligenceSystem()
        results = system.run_full_pipeline()
        duration = time.time() - start

        self.last_run = datetime.now()

        summary = {
            "run_id": run_id,
            "duration_seconds": round(duration, 2),
            "games_processed": results.get("games_data_shape", 0),
            "strategies_run": list(results.get("backtest_results", {}).keys()),
            "edges_detected": len(results.get("edge_signals", [])),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("Scheduled pipeline run completed", **summary)
        return summary

    def run_once(self):
        """Run the pipeline once and exit."""
        self.run_pipeline()

    def run_loop(self):
        """Run the pipeline on a loop with the configured interval."""
        self._running = True
        logger.info(
            "Task scheduler started",
            interval_minutes=self.interval.total_seconds() / 60,
        )

        while self._running:
            try:
                self.run_pipeline()
            except Exception as e:
                logger.error("Scheduled pipeline run failed", error=str(e))

            logger.info(
                "Next run scheduled",
                next_run=(datetime.now() + self.interval).isoformat(),
            )
            time.sleep(self.interval.total_seconds())

    def stop(self):
        """Stop the scheduler loop."""
        self._running = False
        logger.info("Task scheduler stopped")


def run_scheduler(interval_minutes: int = 60):
    """Entry point for running the scheduler."""
    from betting_intel.services.logging import setup_logging

    setup_logging(level="INFO")
    scheduler = TaskScheduler(interval_minutes=interval_minutes)

    try:
        scheduler.run_loop()
    except KeyboardInterrupt:
        scheduler.stop()
        logger.info("Scheduler terminated by user")


if __name__ == "__main__":
    run_scheduler()
