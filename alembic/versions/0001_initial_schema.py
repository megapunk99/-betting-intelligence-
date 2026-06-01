"""Initial schema — create all core tables.

Creates the 4 primary tables for the betting intelligence system:
  - games:         Game data and predictions
  - bets:          Individual betting records
  - model_versions:  Model version tracking for reproducibility
  - pipeline_runs:  Pipeline execution run tracking

Revision ID: 0001
Revises: None
Create Date: 2026-06-01

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables if they don't already exist (idempotent)."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing = inspector.get_table_names()

    # ── games table ──────────────────────────────────────────────────────
    if "games" not in existing:
        op.create_table(
            "games",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("game_id", sa.String(length=20), nullable=False),
            sa.Column("game_date", sa.DateTime(), nullable=False),
            sa.Column("season", sa.Integer(), nullable=True),
            sa.Column("home_team_id", sa.Integer(), nullable=True),
            sa.Column("home_team_name", sa.String(length=100), nullable=False),
            sa.Column("away_team_name", sa.String(length=100), nullable=False),
            sa.Column("home_team_abbr", sa.String(length=10), nullable=True),
            sa.Column("away_team_abbr", sa.String(length=10), nullable=True),
            sa.Column("home_score", sa.Float(), nullable=True),
            sa.Column("away_score", sa.Float(), nullable=True),
            sa.Column("total_points", sa.Float(), nullable=True),
            sa.Column("point_diff", sa.Float(), nullable=True),
            sa.Column("features_json", sa.Text(), nullable=True),
            sa.Column("predicted_total", sa.Float(), nullable=True),
            sa.Column("predicted_spread", sa.Float(), nullable=True),
            sa.Column("prediction_confidence", sa.Float(), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column(
                "updated_at", sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_games_game_id", "games", ["game_id"], unique=True)
        op.create_index("ix_games_game_date", "games", ["game_date"])

    # ── bets table ───────────────────────────────────────────────────────
    if "bets" not in existing:
        op.create_table(
            "bets",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("game_id", sa.String(length=20), nullable=False),
            sa.Column("game_date", sa.DateTime(), nullable=False),
            sa.Column("strategy", sa.String(length=50), nullable=False),
            sa.Column("model", sa.String(length=50), nullable=False),
            sa.Column("bet_type", sa.String(length=20), nullable=False),
            sa.Column("matchup", sa.String(length=200), nullable=True),
            sa.Column("predicted_value", sa.Float(), nullable=True),
            sa.Column("market_line", sa.Float(), nullable=True),
            sa.Column("actual_value", sa.Float(), nullable=True),
            sa.Column("edge_pct", sa.Float(), nullable=True),
            sa.Column("outcome", sa.String(length=10), nullable=True),
            sa.Column("profit_units", sa.Float(), nullable=True),
            sa.Column("kelly_fraction", sa.Float(), nullable=True),
            sa.Column("stake_dollars", sa.Float(), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_bets_game_id", "bets", ["game_id"])
        op.create_index("ix_bets_game_date", "bets", ["game_date"])
        op.create_index("ix_bets_strategy", "bets", ["strategy"])

    # ── model_versions table ─────────────────────────────────────────────
    if "model_versions" not in existing:
        op.create_table(
            "model_versions",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("model_name", sa.String(length=100), nullable=False),
            sa.Column("version", sa.String(length=20), nullable=False),
            sa.Column("parameters", sa.Text(), nullable=True),
            sa.Column("metrics_json", sa.Text(), nullable=True),
            sa.Column("feature_cols", sa.Text(), nullable=True),
            sa.Column("artifact_path", sa.String(length=500), nullable=True),
            sa.Column(
                "training_date", sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column("training_samples", sa.Integer(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_model_versions_model_name", "model_versions", ["model_name"])

    # ── pipeline_runs table ──────────────────────────────────────────────
    if "pipeline_runs" not in existing:
        op.create_table(
            "pipeline_runs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("run_id", sa.String(length=50), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="running"),
            sa.Column(
                "started_at", sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("games_processed", sa.Integer(), nullable=True),
            sa.Column("bets_generated", sa.Integer(), nullable=True),
            sa.Column("errors", sa.Text(), nullable=True),
            sa.Column("summary", sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_unique_constraint("uq_pipeline_runs_run_id", "pipeline_runs", ["run_id"])


def downgrade() -> None:
    """Roll back the initial migration — drop all tables."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing = inspector.get_table_names()

    for table in ("pipeline_runs", "model_versions", "bets", "games"):
        if table in existing:
            op.drop_table(table)
