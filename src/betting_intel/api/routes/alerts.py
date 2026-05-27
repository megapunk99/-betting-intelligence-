"""Alert management API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from betting_intel.alerts.dispatcher import (
    AlertDispatcher,
    AlertConfig,
    BetAlert,
    alert_dispatcher,
)

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/health")
async def alerts_health():
    """Get alert system health and configuration status."""
    stats = alert_dispatcher.get_stats()
    return {
        "status": "active" if stats["channels_configured"] > 0 else "inactive",
        "channels": stats["channels_configured"],
        "total_dispatched": stats["total_dispatched"],
        "rate_limit_remaining": stats["rate_limit_remaining"],
        "channels_detail": stats["channels"],
        "config": {
            key: getattr(alert_dispatcher.config, key)
            for key in [
                "min_edge_pct", "min_confidence", "min_stake",
                "enable_live_movement_alerts", "enable_daily_summary",
                "max_alerts_per_hour",
            ]
        },
    }


@router.post("/test")
async def send_test_alert(
    channel: str = Query("telegram", description="Channel to send test alert to"),
    message: str = Query("Test alert from Betting Intelligence", description="Test message"),
):
    """Send a test alert to verify channel configuration."""
    sent = await alert_dispatcher.dispatch_raw_message(message, channel_name=channel)
    if not sent:
        raise HTTPException(
            status_code=404,
            detail=f"Channel '{channel}' not configured or message failed to send",
        )
    return {
        "status": "sent",
        "channel": channel,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/bet")
async def trigger_bet_alert(
    matchup: str,
    bet_type: str,
    edge_pct: float,
    confidence: float,
    stake: float,
    league: str = "NBA",
    reasoning: str = "",
    model: str = "",
):
    """Manually trigger a bet alert."""
    alert = BetAlert(
        game_id=f"{league}_{matchup}",
        matchup=matchup,
        bet_type=bet_type,
        edge_pct=edge_pct,
        confidence=confidence,
        stake=stake,
        league=league,
        reasoning=reasoning,
        model=model,
    )
    sent_to = await alert_dispatcher.dispatch_bet_alert(alert)
    return {
        "status": "dispatched" if sent_to else "filtered",
        "channels": sent_to,
        "alert": {
            "matchup": matchup,
            "bet_type": bet_type,
            "edge_pct": edge_pct,
            "stake": stake,
        },
    }


@router.get("/stats")
async def alert_stats():
    """Get alert dispatch statistics."""
    return alert_dispatcher.get_stats()


@router.post("/config")
async def update_alert_config(
    min_edge_pct: Optional[float] = None,
    min_confidence: Optional[float] = None,
    min_stake: Optional[float] = None,
    enable_live_movement_alerts: Optional[bool] = None,
    enable_daily_summary: Optional[bool] = None,
    enable_health_reports: Optional[bool] = None,
    max_alerts_per_hour: Optional[int] = None,
):
    """Update alert dispatch configuration."""
    if min_edge_pct is not None:
        alert_dispatcher.config.min_edge_pct = min_edge_pct
    if min_confidence is not None:
        alert_dispatcher.config.min_confidence = min_confidence
    if min_stake is not None:
        alert_dispatcher.config.min_stake = min_stake
    if enable_live_movement_alerts is not None:
        alert_dispatcher.config.enable_live_movement_alerts = enable_live_movement_alerts
    if enable_daily_summary is not None:
        alert_dispatcher.config.enable_daily_summary = enable_daily_summary
    if enable_health_reports is not None:
        alert_dispatcher.config.enable_health_reports = enable_health_reports
    if max_alerts_per_hour is not None:
        alert_dispatcher.config.max_alerts_per_hour = max_alerts_per_hour

    return {
        "status": "updated",
        "config": {
            "min_edge_pct": alert_dispatcher.config.min_edge_pct,
            "min_confidence": alert_dispatcher.config.min_confidence,
            "min_stake": alert_dispatcher.config.min_stake,
            "enable_live_movement_alerts": alert_dispatcher.config.enable_live_movement_alerts,
            "enable_daily_summary": alert_dispatcher.config.enable_daily_summary,
            "enable_health_reports": alert_dispatcher.config.enable_health_reports,
            "max_alerts_per_hour": alert_dispatcher.config.max_alerts_per_hour,
        },
    }
