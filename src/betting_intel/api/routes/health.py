"""Health check and monitoring API endpoints."""

from __future__ import annotations

import time
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from betting_intel import __version__
from betting_intel.api.schemas import HealthResponse
from betting_intel.db.connection import db_manager
from betting_intel.models.persistence import model_registry

router = APIRouter(tags=["Health"])

# Track server start time
_server_start = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    db_healthy = db_manager.health_check()
    models = model_registry.list_models()

    return HealthResponse(
        status="ok" if db_healthy else "degraded",
        version=__version__,
        database="connected" if db_healthy else "disconnected",
        uptime_seconds=time.time() - _server_start,
        models_loaded=sum(m["total_versions"] for m in models),
    )


@router.get("/ready")
async def readiness():
    """Readiness probe for Kubernetes/Docker orchestration."""
    db_healthy = db_manager.health_check()
    if not db_healthy:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "database": "disconnected"},
        )
    return {"status": "ready"}


@router.get("/live")
async def liveness():
    """Liveness probe for Kubernetes/Docker orchestration."""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}
