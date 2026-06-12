"""
Alert management API routes — stub implementation.

The original alerts/ package (dispatcher, telegram, discord) was deleted
during a cleanup. This stub returns informative messages until the alert
system is re-created.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/health")
async def alerts_health():
    """Get alert system health and configuration status."""
    return {
        "status": "unavailable",
        "note": "Alert system was removed during package cleanup. Re-create betting_intel/alerts/ to re-enable.",
        "channels": 0,
        "total_dispatched": 0,
        "rate_limit_remaining": 0,
        "channels_detail": {},
        "config": {},
    }


@router.post("/test")
async def send_test_alert(channel: str = "telegram", message: str = "Test alert"):
    """Send a test alert — always returns unavailable."""
    return {
        "status": "unavailable",
        "channel": channel,
        "message": "Alert system unavailable — re-create betting_intel/alerts/ to re-enable.",
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
    """Manually trigger a bet alert — always returns filtered."""
    return {
        "status": "filtered",
        "channels": [],
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
    return {
        "channels_configured": 0,
        "total_dispatched": 0,
        "rate_limit_remaining": 0,
        "channels": {},
    }


@router.post("/config")
async def update_alert_config():
    """Update alert dispatch configuration — no-op."""
    return {"status": "unavailable", "config": {}}
