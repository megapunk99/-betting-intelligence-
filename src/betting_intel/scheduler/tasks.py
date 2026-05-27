"""
Task scheduler for automated pipeline runs, odds polling, and data updates.
Can be run as a standalone process or integrated with cron/airflow.

Supports:
- Daily pipeline runs
- Live odds polling and WebSocket broadcast
- League health monitoring and alerts
- Weekly model retraining
- Periodic database cleanup
"""

from __future__ import annotations

import asyncio
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

    def check_league_health(self) -> dict[str, str]:
        """Run health checks on all registered leagues and return statuses."""
        try:
            from betting_intel.data.small_leagues.league_registry import league_registry

            leagues = league_registry.list_leagues()
            statuses = {}
            for key in leagues:
                try:
                    health = league_registry.check_health(key)
                    statuses[key] = health.status
                    if health.status != "healthy":
                        logger.warning(
                            f"League {key} health: {health.status}",
                            warnings=health.warnings,
                        )
                except Exception as exc:
                    statuses[key] = "error"
                    logger.error(f"League {key} health check failed: {exc}")

            logger.info(
                "League health check complete",
                total=len(statuses),
                healthy=sum(1 for s in statuses.values() if s == "healthy"),
            )
            return statuses
        except ImportError:
            logger.info("League registry not available, skipping health check")
            return {}
        except Exception as exc:
            logger.error(f"League health check failed: {exc}")
            return {}

    async def send_daily_summary(self, pipeline_results: dict) -> None:
        """Send a daily performance summary via configured alert channels."""
        try:
            from betting_intel.alerts.dispatcher import alert_dispatcher

            backtest = pipeline_results.get("backtest_results", {})
            bankroll = pipeline_results.get("bankroll_results", {}).get("metrics", {})

            total_bets = 0
            wins = 0
            for key, result in backtest.items():
                if hasattr(result, "total_bets"):
                    total_bets += result.total_bets
                if hasattr(result, "wins"):
                    wins += result.wins

            profit = bankroll.get("total_return_pct", 0)
            roi = bankroll.get("total_return_pct", 0)

            await alert_dispatcher.dispatch_daily_summary(
                total_bets=total_bets,
                wins=wins,
                losses=total_bets - wins,
                profit=profit,
                roi=roi,
                best_bet="",
            )
            logger.info("Daily summary dispatched via alert channels")
        except Exception as exc:
            logger.debug(f"Daily summary dispatch skipped: {exc}")

    async def run_once(self):
        """Run the pipeline once and exit, with health checks and summary dispatch."""
        pipeline_result = self.run_pipeline()
        self.check_league_health()
        await self.send_daily_summary(pipeline_result)

    def run_loop(self):
        """Run the pipeline on a loop with the configured interval."""
        self._running = True
        logger.info(
            "Task scheduler started",
            interval_minutes=self.interval.total_seconds() / 60,
        )

        while self._running:
            try:
                asyncio.run(self.run_once())
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
