"""
Alert management API routes.

Wired to ResultsTracker.check_alerts() for real underperformance detection.
Provides:
  - Alert system health check
  - Trigger alert evaluation on demand
  - View current alert statistics
  - Read alerts from the JSONL log

The old Telegram/Discord dispatch system was removed during cleanup.
Check_alerts() still writes alerts to the JSONL file for external
monitoring (e.g., Logstash, Grafana, or custom scripts).
"""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Query

from betting_intel.analytics.tracker import ResultsTracker, ALERTS_LOG

router = APIRouter(prefix="/alerts", tags=["alerts"])


def _get_tracker() -> ResultsTracker:
    return ResultsTracker()


@router.get("/health")
async def alerts_health():
    """Get alert system health: how many strategies tracked, how many alerted."""
    try:
        tracker = _get_tracker()
        tracker.resolve_all()
        report = tracker.generate_report(window_days=30)
        alerts = tracker.check_alerts(report)

        return {
            "status": "ok",
            "n_strategies_tracked": len(report.strategies),
            "n_confident_strategies": len(report.confident_strategies),
            "n_active_alerts": len(alerts),
            "total_bets_resolved": report.total_bets,
            "overall_roi": round(report.overall_roi, 4),
            "alerts_log_path": str(ALERTS_LOG),
            "alerted_strategies": [
                {
                    "strategy_name": a.strategy_name,
                    "roi": round(a.roi, 4),
                    "n_bets": a.n_bets,
                    "total_profit": round(a.total_profit, 2),
                }
                for a in alerts
            ],
            "note": "Alert dispatch (Telegram/Discord) was removed during cleanup. "
            "Alerts are written to JSONL for external monitoring.",
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e),
            "note": "ResultsTracker encountered an error — check data/forward_test_results.json exists.",
        }


@router.post("/evaluate")
async def evaluate_alerts(
    window_days: int = Query(30, description="Trailing window in days", ge=7, le=365),
):
    """Run alert evaluation and return any triggered alerts."""
    try:
        tracker = _get_tracker()
        tracker.resolve_all()
        report = tracker.generate_report(window_days=window_days)
        alerts = tracker.check_alerts(report)

        return {
            "status": "ok",
            "n_evaluated": len(report.confident_strategies),
            "n_alerts_triggered": len(alerts),
            "window_days": window_days,
            "total_bets_resolved": report.total_bets,
            "alerts": [
                {
                    "strategy_name": a.strategy_name,
                    "model": a.model,
                    "league": a.league,
                    "bet_type": a.bet_type,
                    "n_bets": a.n_bets,
                    "wins": a.wins,
                    "losses": a.losses,
                    "win_rate": round(a.win_rate, 4),
                    "roi": round(a.roi, 4),
                    "total_profit": round(a.total_profit, 2),
                    "trailing_profit": round(sum(a.trailing_profits), 2),
                    "sharpe": a.sharpe,
                }
                for a in alerts
            ],
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e),
        }


@router.get("/stats")
async def alert_stats():
    """Get alert dispatch statistics from the alerts log."""
    try:
        alerts: list[dict] = []
        if ALERTS_LOG.exists():
            with open(ALERTS_LOG) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            alerts.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        # Count unique strategies that have been alerted
        unique_strategies = set(a.get("strategy_name", "") for a in alerts)
        cutoff = datetime.now().timestamp() - 86400 * 7
        recent_alerts = [
            a
            for a in alerts
            if a.get("timestamp")
            and datetime.fromisoformat(a["timestamp"]).timestamp() > cutoff
        ]

        return {
            "status": "ok",
            "n_total_alerts": len(alerts),
            "n_alerts_last_7d": len(recent_alerts),
            "n_unique_strategies_alerted": len(unique_strategies),
            "alerted_strategies": sorted(unique_strategies),
            "latest_alerts": alerts[-10:][::-1] if alerts else [],
            "alerts_log_path": str(ALERTS_LOG),
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": str(e),
            "alerts_log_path": str(ALERTS_LOG),
        }
