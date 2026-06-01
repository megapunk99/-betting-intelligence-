#!/usr/bin/env python3
"""
Initialize the Betting Intelligence database via Alembic migrations.

Applies all pending Alembic migrations to bring the schema up to date.
Supports both SQLite (default) and PostgreSQL.

Usage:
    python scripts/init_db.py                       # Use default SQLite
    python scripts/init_db.py --url postgresql://... # Custom database URL
    python scripts/init_db.py --check                # Verify connectivity only
    python scripts/init_db.py --direct               # Old direct create_tables()

This is equivalent to:
    betting-intel db init

Run this after installing the package to set up your database.
"""

import argparse
import sys
from pathlib import Path


def init_db(
    database_url: str | None = None,
    check_only: bool = False,
    direct: bool = False,
) -> bool:
    """
    Initialize the database using Alembic migrations (default) or direct
    table creation (fallback).

    Args:
        database_url: Database connection URL. If None, uses the default
                      from settings (can be overridden via .env DATABASE_URL).
        check_only: If True, only check connectivity without creating tables.
        direct: If True, use direct create_tables() instead of Alembic.

    Returns:
        True if successful, False otherwise.
    """
    # Ensure the project root is on sys.path for imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    try:
        from betting_intel.config import settings
        from betting_intel.db.connection import DatabaseManager, run_migrations
    except ImportError as e:
        print(f"❌ Could not import betting_intel: {e}")
        print("   Make sure the package is installed:")
        print("     pip install -e .")
        print("   Or set PYTHONPATH to include the src/ directory.")
        return False

    # Determine database URL
    url = database_url or settings.database_url

    if check_only:
        db_manager = DatabaseManager(database_url=url)
        healthy = db_manager.health_check()
        if healthy:
            print(f"✅ Database connection OK ({url})")
        else:
            print(f"❌ Database connection FAILED ({url})")
        db_manager.close()
        return healthy

    if direct:
        # Fallback: direct table creation (no versioning)
        db_manager = DatabaseManager(database_url=url)

        # Ensure parent directory exists for SQLite
        if url.startswith("sqlite"):
            db_path_str = url.replace("sqlite:///", "").replace("sqlite://", "")
            db_path = Path(db_path_str)
            if not db_path.is_absolute():
                db_path = settings.project_root / db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"📁 Ensured directory exists: {db_path.parent}")

        try:
            db_manager.create_tables()
            print(f"✅ Database tables created successfully ({url})")

            from sqlalchemy import inspect
            inspector = inspect(db_manager.engine)
            tables = inspector.get_table_names()
            if tables:
                print(f"   Tables created: {', '.join(tables)}")
            else:
                print("   ⚠ No tables were created. Check your schema.")
            return True
        except Exception as e:
            print(f"❌ Failed to create tables: {e}")
            return False
        finally:
            db_manager.close()

    # Default: use Alembic migrations
    print("Running Alembic migrations...")
    success = run_migrations(database_url=url)
    if success:
        print(f"✅ Migrations applied. Database is up to date ({url})")
    else:
        print("❌ Migrations failed. Try --direct as fallback.")
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Initialize the Betting Intelligence database",
    )
    parser.add_argument(
        "--url",
        help="Database URL (default: from settings / .env DATABASE_URL)",
        default=None,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check connectivity, don't create tables",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use direct create_tables() instead of Alembic migrations",
    )
    parser.add_argument(
        "--env-file",
        help="Path to .env file to load (default: ./.env)",
        default=None,
    )
    args = parser.parse_args()

    # Load optional .env file
    if args.env_file:
        from dotenv import load_dotenv
        env_path = Path(args.env_file)
        if env_path.exists():
            load_dotenv(env_path)
            print(f"📄 Loaded environment from {env_path}")
        else:
            print(f"⚠ .env file not found: {env_path}")

    success = init_db(
        database_url=args.url,
        check_only=args.check,
        direct=args.direct,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
