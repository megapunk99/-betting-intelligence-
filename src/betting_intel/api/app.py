"""
FastAPI application entry point — production-grade REST API.
Provides prediction, backtesting, edge detection, and monitoring endpoints.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from betting_intel import __version__
from betting_intel.api.routes import health, predict, backtest
from betting_intel.config import settings
from betting_intel.db.connection import db_manager
from betting_intel.services import logger, setup_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan: startup and shutdown."""
        logger.info("Starting API server", version=__version__)
        db_manager.create_tables()
        yield
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
