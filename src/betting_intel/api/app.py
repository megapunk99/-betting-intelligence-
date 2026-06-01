"""
FastAPI application entry point — production-grade REST API.
Provides prediction, backtesting, edge detection, and monitoring endpoints.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from betting_intel import __version__
from betting_intel.api.routes import health, predict, backtest, alerts as alert_routes
from betting_intel.config import settings
from betting_intel.db.connection import db_manager
from betting_intel.services import logger, setup_logging
from betting_intel.alerts.dispatcher import alert_dispatcher
from betting_intel.alerts.telegram import TelegramBot
from betting_intel.alerts.discord import DiscordWebhook
from betting_intel.monitoring.metrics import metrics_endpoint

# WebSocket manager — initialized at app startup if live odds are enabled
_odds_ws_manager = None
_league_registry = None


def get_odds_ws_manager():
    """Get the global WebSocket odds manager instance."""
    return _odds_ws_manager


def get_league_registry():
    """Get the global league registry instance."""
    return _league_registry


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan: startup and shutdown."""
        logger.info("Starting API server", version=__version__)
        # Apply Alembic migrations to bring schema up to date
        db_manager.run_migrations()

        # ── Initialize League Registry ────────────────────────────────
        global _league_registry
        try:
            from betting_intel.data.small_leagues.league_registry import league_registry
            _league_registry = league_registry
            leagues = league_registry.list_leagues()
            logger.info(
                "League registry initialized",
                leagues_available=len(leagues),
                league_names=list(leagues.keys()),
            )
        except Exception as exc:
            logger.warning(f"League registry init skipped: {exc}")

        # ── Initialize Alert Channels ──────────────────────────────────
        _alert_senders = []
        try:
            if settings.enable_alerts:
                if settings.enable_telegram and settings.telegram_bot_token:
                    bot = TelegramBot(
                        token=settings.telegram_bot_token,
                        chat_id=settings.telegram_chat_id or None,
                    )
                    alert_dispatcher.add_channel("telegram", bot)
                    _alert_senders.append(bot)
                    logger.info("Telegram alert channel registered")

                if settings.enable_discord and settings.discord_webhook_url:
                    webhook = DiscordWebhook(
                        webhook_url=settings.discord_webhook_url,
                    )
                    alert_dispatcher.add_channel("discord", webhook)
                    _alert_senders.append(webhook)
                    logger.info("Discord alert channel registered")

                # Apply threshold config
                alert_dispatcher.config.min_edge_pct = settings.alert_min_edge_pct
                alert_dispatcher.config.min_confidence = settings.alert_min_confidence
                alert_dispatcher.config.min_stake = settings.alert_min_stake
                alert_dispatcher.config.rate_limit_seconds = settings.alert_rate_limit_seconds
        except Exception as exc:
            logger.warning(f"Alert channel init skipped: {exc}")

        # ── Initialize Live Odds Poller + WebSocket ───────────────────
        global _odds_ws_manager
        try:
            if settings.enable_live_odds and settings.odds_api_key and settings.odds_api_key != "your-api-key-here":
                from pathlib import Path
                from betting_intel.data.websocket_odds import OddsWebSocketManager

                _odds_ws_manager = OddsWebSocketManager(
                    poll_interval=settings.odds_poll_interval,
                    odds_api_key=settings.odds_api_key,
                    db_path=Path(settings.odds_snapshots_db),
                )
                await _odds_ws_manager.start()
                logger.info("Live odds WebSocket manager started")
            else:
                logger.info("Live odds disabled (set ENABLE_LIVE_ODDS=true and ODDS_API_KEY)")
        except Exception as exc:
            logger.warning(f"Live odds init failed: {exc}")
            _odds_ws_manager = None

        yield

        # ── Shutdown ───────────────────────────────────────────────────
        if _odds_ws_manager:
            await _odds_ws_manager.stop()
            logger.info("Live odds WebSocket manager stopped")

        # Close alert channel HTTP clients
        for sender in _alert_senders:
            try:
                await sender.close()
            except Exception as exc:
                logger.debug(f"Error closing alert sender {sender.__class__.__name__}: {exc}")

        logger.info("Shutting down API server")
        db_manager.close()

    app = FastAPI(
        title="Betting Intelligence API",
        description="Professional-grade betting intelligence for basketball market analysis",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────
    origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Middleware: Request timing ─────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time-MS"] = str(round(process_time * 1000, 2))
        return response

    # ── Global Error Handler ──────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            error=str(exc),
            path=str(request.url),
            method=request.method,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.log_level == "DEBUG" else "An unexpected error occurred",
                "code": "INTERNAL_ERROR",
            },
        )

    # ── Register Routers ─────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(backtest.router)
    app.include_router(alert_routes.router)

    # ── Prometheus Metrics Endpoint ──────────────────────────────────
    @app.get("/metrics")
    async def prometheus_metrics():
        """Expose Prometheus metrics for scraping by Prometheus server.

        Returns metrics in Prometheus text format with the correct
        Content-Type header (text/plain; version=0.0.4).
        """
        data, status_code, headers = metrics_endpoint()
        return Response(content=data, status_code=status_code, headers=headers)

    # ── WebSocket Endpoint ───────────────────────────────────────────
    @app.websocket("/ws/odds")
    async def odds_websocket(websocket: WebSocket):
        if _odds_ws_manager:
            await _odds_ws_manager.websocket_endpoint(websocket)
        else:
            await websocket.accept()
            await websocket.send_json({
                "type": "error",
                "message": "Live odds not enabled. Set ENABLE_LIVE_ODDS=true and ODDS_API_KEY.",
            })
            await websocket.close()

    # ── WebSocket Connection Tracking ─────────────────────────────────
    if _odds_ws_manager:
        @app.get("/ws/stats")
        async def websocket_stats():
            return {
                "active_connections": _odds_ws_manager.connection_manager.active_connections,
                "games_tracked": len(_odds_ws_manager.poller._last_snapshots),
                "polling": _odds_ws_manager.poller._running,
            }

    # ── League Registry Endpoint ─────────────────────────────────────
    @app.get("/leagues")
    async def list_leagues():
        """List all registered leagues and their health."""
        if _league_registry is None:
            return {"leagues": {}, "note": "League registry not initialized"}

        leagues = _league_registry.list_leagues()
        health_data = {}
        for key in leagues:
            try:
                status = _league_registry.check_health(key)
                health_data[key] = {
                    "name": status.league_name,
                    "status": status.status,
                    "total_games": status.total_games,
                    "games_last_24h": status.games_last_24h,
                    "freshness_grade": status.freshness_grade,
                    "is_available": status.is_available,
                }
            except Exception:
                health_data[key] = {"status": "error"}

        return {"leagues": health_data}

    return app


app = create_app()


def run():
    """Run the API server via uvicorn."""
    import uvicorn

    uvicorn.run(
        "betting_intel.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.api_log_level,
        reload=settings.log_level == "DEBUG",
    )


if __name__ == "__main__":
    run()
