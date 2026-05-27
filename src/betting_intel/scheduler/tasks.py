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
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from betting_intel.services import logger, setup_logging


class TaskScheduler:
    """Simple task scheduler for periodic pipeline execution."""

    def __init__(self, interval_minutes: int = 60):
        self.interval = timedelta(minutes=interval_minutes)
        self.last_run: Optional[datetime] = None
        self._running = False

    def run_pipeline(self) -> dict:
        """Execute the full betting prediction pipeline via predict_tomorrow.py.

        Runs predict_tomorrow.py --scheduled as a subprocess and parses
        the JSON summary from its stdout. The results are compatible with
        send_daily_summary() for alert dispatch.
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        logger.info("Scheduled pipeline run starting", run_id=run_id)

        # Locate predict_tomorrow.py (project root / sibling of src)
        script_path = Path(__file__).resolve().parent.parent.parent.parent / "predict_tomorrow.py"
        if not script_path.exists():
            logger.error(f"predict_tomorrow.py not found at {script_path}")
            return {"run_id": run_id, "status": "error", "error": "script_not_found"}

        start = time.time()

        # Pass through environment with UTF-8 encoding for Windows Unicode support
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--scheduled"],
                capture_output=True,
                text=True,
                cwd=str(script_path.parent),
                env=env,
                timeout=600,  # 10 min max
            )
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            logger.error("Scheduled pipeline timed out after 10 min")
            return {
                "run_id": run_id,
                "status": "timeout",
                "duration_seconds": round(duration, 1),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as exc:
            duration = time.time() - start
            logger.error(f"Scheduled pipeline subprocess failed: {exc}")
            return {
                "run_id": run_id,
                "status": "error",
                "error": str(exc),
                "duration_seconds": round(duration, 1),
                "timestamp": datetime.now().isoformat(),
            }

        duration = time.time() - start

        # Parse the JSON summary from the scheduled marker line
        results = {}
        for line in result.stdout.splitlines():
            if line.startswith("##SCHEDULED_RESULT##"):
                try:
                    results = json.loads(line[len("##SCHEDULED_RESULT##"):])
                except json.JSONDecodeError as exc:
                    logger.error(f"Failed to parse scheduled result JSON: {exc}")
                    results = {"status": "parse_error", "error": str(exc)}
                break

        # Log stderr on subprocess failure for diagnostics
        if result.returncode != 0:
            logger.warning(
                "Subprocess returned non-zero exit code",
                returncode=result.returncode,
                stderr=result.stderr[-500:] if result.stderr else "",
            )

        # If parsing failed or no marker found, build from fallback
        if not results:
            logger.warning("No scheduled result marker found in stdout, building fallback summary")
            results = {
                "status": "partial",
                "stderr": result.stderr[-500:] if result.stderr else "",
                "returncode": result.returncode,
            }

        results["run_id"] = run_id
        results["duration_seconds"] = round(duration, 1)
        results["timestamp"] = datetime.now().isoformat()

        self.last_run = datetime.now()

        logger.info(
            "Scheduled pipeline run completed",
            run_id=run_id,
            status=results.get("status", "unknown"),
            duration=duration,
            games=results.get("games", 0),
            recommendations=results.get("recommendations", 0),
            clear_picks=results.get("clear_picks", 0),
        )
        return results

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
        """Send a daily performance summary via configured alert channels.

        Supports both the legacy BettingIntelligenceSystem result shape
        and the new predict_tomorrow.py --scheduled result shape.
        """
        try:
            from betting_intel.alerts.dispatcher import alert_dispatcher

            # Try new result shape (predict_tomorrow.py --scheduled)
            risk = pipeline_results.get("risk_assessment", {})
            sized_bets = risk.get("bets", [])
            total_bets = len(sized_bets)
            total_staked = sum(b.get("stake", 0) for b in sized_bets)
            bankroll_amt = risk.get("bankroll", pipeline_results.get("bankroll", 10000))

            # Fallback to legacy result shape (BettingIntelligenceSystem)
            if not total_bets:
                backtest = pipeline_results.get("backtest_results", {})
                bankroll = pipeline_results.get("bankroll_results", {}).get("metrics", {})
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
            else:
                # New result shape: actual outcomes unknown, send staking summary only
                summary_lines = [
                    f"📋 <b>SCHEDULED PIPELINE</b>",
                    f"━━━━━━━━━━━━━━━",
                    f"💰 Bankroll: <b>${bankroll_amt:,.0f}</b>",
                    f"📊 Recommendations: <b>{total_bets}</b>",
                    f"💵 Total Staked: <b>${total_staked:,.2f}</b>",
                    f"━━━━━━━━━━━━━━━",
                    f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
                ]
                # Send as a raw message (not a structured daily summary)
                await alert_dispatcher.dispatch_raw_message(
                    "\n".join(summary_lines)
                )

            logger.info(
                "Daily summary dispatched via alert channels",
                total_bets=total_bets,
                total_staked=round(total_staked, 2),
            )
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
