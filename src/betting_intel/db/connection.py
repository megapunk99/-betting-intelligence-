"""Database connection management with engine pooling and Alembic migrations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from betting_intel.config import settings
import logging

logger = logging.getLogger(__name__)


def run_migrations(database_url: Optional[str] = None) -> bool:
    """
    Run Alembic migrations to bring the database schema up to date.

    Uses Alembic's Python API directly (no subprocess) to apply all
    pending migrations up to ``head``. The database URL is resolved from
    the project's pydantic-settings, which reads from .env / env vars.

    Args:
        database_url: Optional override for the database URL. If not
                      provided, the project settings DATABASE_URL is used.

    Returns:
        True if migrations completed successfully, False otherwise.
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    try:
        from alembic.config import Config
        from alembic import command

        alembic_cfg = Config(str(project_root / "alembic.ini"))
        alembic_cfg.set_main_option("script_location", str(project_root / "alembic"))

        if database_url:
            alembic_cfg.set_main_option("sqlalchemy.url", database_url)

        command.upgrade(alembic_cfg, "head")
        logger.info("Alembic migrations applied successfully")
        return True

    except Exception as exc:
        logger.error("Alembic migrations failed: %s", exc)
        return False


class DatabaseManager:
    """Manages database engine and sessions with connection pooling."""

    def __init__(self, database_url: Optional[str] = None):
        if database_url:
            self.database_url = database_url
        elif hasattr(settings, 'database_url') and getattr(settings, 'database_url', None):
            self.database_url = settings.database_url
        else:
            # Build SQLite URL from the nba_db_path setting
            db_path = getattr(settings, 'nba_db_path', './data/nba_data.db')
            self.database_url = f"sqlite:///{db_path}"
        self._engine = None
        self._session_factory = None

    @property
    def engine(self):
        if self._engine is None:
            is_sqlite = self.database_url.startswith("sqlite")
            connect_args = {}

            if is_sqlite:
                # SQLite's SingletonThreadPool doesn't support pool_size/max_overflow
                connect_args["check_same_thread"] = False
                self._engine = create_engine(
                    self.database_url,
                    connect_args=connect_args,
                    pool_pre_ping=True,
                    echo=False,
                )
            else:
                self._engine = create_engine(
                    self.database_url,
                    connect_args=connect_args,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                    echo=False,
                )

            logger.debug("Database engine created: %s", self.database_url)
        return self._engine

    @property
    def session_factory(self):
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autoflush=False,
                expire_on_commit=False,
            )
        return self._session_factory

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.session_factory()

    def create_tables(self):
        """
        Create all tables defined in the schema directly.

        .. deprecated::
            Use ``run_migrations()`` instead, which applies Alembic
            migrations for proper version-controlled schema management.
            This method is kept for backward compatibility.
        """
        from betting_intel.db.schema import Base

        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified (direct)")

    def run_migrations(self) -> bool:
        """
        Apply Alembic migrations to bring the schema to the latest version.

        Wrapper around the module-level ``run_migrations()`` that passes
        the manager's database URL.

        Returns:
            True if migrations completed successfully.
        """
        return run_migrations(database_url=self.database_url)

    def health_check(self) -> bool:
        """Check if database is reachable."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False

    def close(self):
        """Dispose of the engine."""
        if self._engine:
            self._engine.dispose()
            logger.debug("Database engine disposed")


# Global database manager instance
db_manager = DatabaseManager()
