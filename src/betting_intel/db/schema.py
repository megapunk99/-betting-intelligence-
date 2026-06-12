"""Database schema: SQLAlchemy models for the betting intelligence system."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Game(Base):
    """Stores game data and predictions."""

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    game_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    season: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Teams
    home_team_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_team_name: Mapped[str] = mapped_column(String(100))
    away_team_name: Mapped[str] = mapped_column(String(100))
    home_team_abbr: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    away_team_abbr: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)

    # Actual results
    home_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_points: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    point_diff: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Features (JSON blob)
    features_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Predictions
    predicted_total: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prediction_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return f"<Game {self.game_id}: {self.home_team_name} vs {self.away_team_name}>"


class Bet(Base):
    """Stores individual betting records."""

    __tablename__ = "bets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(String(20), index=True)
    game_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    strategy: Mapped[str] = mapped_column(String(50), index=True)
    model: Mapped[str] = mapped_column(String(50))
    bet_type: Mapped[str] = mapped_column(String(20))
    matchup: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Prediction details
    predicted_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    edge_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Outcome
    outcome: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    profit_units: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Kelly staking
    kelly_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stake_dollars: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<Bet {self.game_id}: {self.strategy}/{self.bet_type} -> {self.outcome}>"


class ModelVersion(Base):
    """Tracks model versions for reproducibility."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[str] = mapped_column(String(20))
    parameters: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    feature_cols: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    artifact_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    training_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    training_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<ModelVersion {self.model_name}: v{self.version}>"


class PipelineRun(Base):
    """Tracks pipeline execution runs."""

    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(50), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    games_processed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bets_generated: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    errors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<PipelineRun {self.run_id}: {self.status}>"
