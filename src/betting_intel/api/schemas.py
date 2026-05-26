"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Health ─────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    database: str
    uptime_seconds: float = 0.0
    models_loaded: int = 0
    last_pipeline_run: Optional[str] = None


# ── Prediction ─────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    home_team: str = Field(..., description="Home team name", min_length=1)
    away_team: str = Field(..., description="Away team name", min_length=1)
    game_date: Optional[str] = Field(None, description="Game date (YYYY-MM-DD)")


class PredictionResponse(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    predicted_total: float
    predicted_spread: float
    predicted_over_probability: float
    confidence: float
    model_version: str
    features_used: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Backtest ───────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    strategy: str = Field(
        ..., description="Strategy name: total_ridge, total_xgboost, spread, momentum, ensemble"
    )
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class BacktestResultSchema(BaseModel):
    strategy: str
    model: str
    total_bets: int
    wins: int
    losses: int
    win_rate: float
    total_profit: float
    roi_pct: float
    sharpe_ratio: float
    max_drawdown: float
    avg_edge: float
    is_significant: bool


class BacktestResponse(BaseModel):
    results: list[BacktestResultSchema]
    total_bets: int
    total_profit: float
    best_strategy: Optional[str] = None
    execution_time_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Edge Detection ─────────────────────────────────────────────────────────
class EdgeSignalSchema(BaseModel):
    strategy: str
    edge_type: str
    description: str
    avg_edge_pct: float
    sample_size: int
    win_rate: float
    expected_value: float
    confidence: float
    is_actionable: bool


class EdgeResponse(BaseModel):
    signals: list[EdgeSignalSchema]
    total_signals: int
    actionable_signals: int


# ── Bankroll ───────────────────────────────────────────────────────────────
class BankrollResponse(BaseModel):
    initial_bankroll: float
    current_bankroll: float
    peak_bankroll: float
    total_return_pct: float
    total_bets: int
    win_rate: float
    max_drawdown: float
    consecutive_losses: int


# ── Pipeline ───────────────────────────────────────────────────────────────
class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    games_processed: int
    bets_generated: int
    started_at: str
    completed_at: Optional[str] = None
    summary: Optional[str] = None


# ── Error ──────────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: str = "INTERNAL_ERROR"
