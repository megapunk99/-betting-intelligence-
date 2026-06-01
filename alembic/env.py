"""
Alembic environment configuration — Betting Intelligence System.

Resolves the database URL from pydantic-settings (reads from .env / env vars)
and auto-discovers all SQLAlchemy models for autogenerate support.

Reference: https://alembic.sqlalchemy.org/en/latest/
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Ensure the project is importable ──────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
_src_dir = _project_root / "src"
for p in (_project_root, _src_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ── Alembic Config ────────────────────────────────────────────────────────
config = context.config

# Set up Python logging from the alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Import all models so Alembic's autogenerate can detect them ───────────
from betting_intel.db.schema import Base  # noqa: E402
from betting_intel.config import settings  # noqa: E402

# Import each model class to ensure they're registered on Base.metadata
from betting_intel.db.schema import Game, Bet, ModelVersion, PipelineRun  # noqa: E402, F401

# ── Resolve database URL from project settings ────────────────────────────
# This respects .env files and environment variables via pydantic-settings.
# If the URL was already set programmatically (e.g., by connection.py's
# run_migrations()), we respect that value and don't override it.
current_url = config.get_main_option("sqlalchemy.url", "")
if not current_url or current_url == "driver://user:pass@localhost/dbname":
    config.set_main_option("sqlalchemy.url", settings.database_url)

# MetaData to use for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine,
    emitting SQL as a script statt of executing it directly.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Creates an Engine and associates a connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
