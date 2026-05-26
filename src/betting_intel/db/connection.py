"""Database connection management with engine pooling."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from betting_intel.config import settings
from betting_intel.services import logger


class DatabaseManager:
    """Manages database engine and sessions with connection pooling."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        self._engine = None
        self._session_factory = None

    @property
    def engine(self):
        if self._engine is None:
            connect_args = {}
            # SQLite specific configuration
            if self.database_url.startswith("sqlite"):
                connect_args["check_same_thread"] = False

            self._engine = create_engine(
                self.database_url,
                connect_args=connect_args,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                echo=False,
            )
            logger.debug("Database engine created", url=self.database_url)
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
        """Create all tables defined in the schema."""
        from betting_intel.db.schema import Base

        Base.metadata.create_all(self.engine)
        logger.info("Database tables created/verified")

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
