"""
src/database/connection.py
Thread-safe database connection management, engine pooling, session scoping,
and schema initialization for SGP-II.
"""

import os
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, scoped_session, Session

from src.config import get_settings

logger = logging.getLogger(__name__)

# Base directory relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

class DatabaseManager:
    """
    Singleton-style thread-safe Database Connection Manager.
    Supports SQLite and PostgreSQL engines with appropriate connection options.
    """
    _instance: Optional["DatabaseManager"] = None

    def __init__(self, db_url: Optional[str] = None):
        settings = get_settings()
        self.db_url = db_url or settings.get_resolved_db_url()
        self.is_sqlite = self.db_url.startswith("sqlite")
        
        logger.info(f"Initializing DatabaseManager with URL: {self._sanitize_url(self.db_url)}")

        connect_args = {}
        if self.is_sqlite:
            # Prevent database locking issues on concurrent writes
            connect_args["timeout"] = 30.0
            connect_args["check_same_thread"] = False

        self.engine = create_engine(
            self.db_url,
            connect_args=connect_args,
            pool_pre_ping=True,
            echo=False
        )

        if self.is_sqlite:
            self._register_sqlite_pragmas(self.engine)

        self.session_factory = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self.ScopedSession = scoped_session(self.session_factory)

    @classmethod
    def get_instance(cls, db_url: Optional[str] = None) -> "DatabaseManager":
        """Returns the global DatabaseManager instance."""
        if cls._instance is None or (db_url and cls._instance.db_url != db_url):
            cls._instance = DatabaseManager(db_url)
        return cls._instance

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Hides credentials in connection logs."""
        if "@" in url:
            prefix = url.split("://")[0]
            host_part = url.split("@")[-1]
            return f"{prefix}://***:***@{host_part}"
        return url

    @staticmethod
    def _register_sqlite_pragmas(engine: Engine) -> None:
        """Enables WAL mode and busy timeout for SQLite connection stability."""
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA busy_timeout=5000;")
                cursor.execute("PRAGMA foreign_keys=ON;")
            except Exception as e:
                logger.warning(f"Could not set SQLite PRAGMAs: {e}")
            finally:
                cursor.close()

    def get_session(self) -> Session:
        """Creates and returns a scoped session."""
        return self.ScopedSession()

    def close(self) -> None:
        """Closes all sessions and disposes of engine."""
        self.ScopedSession.remove()
        self.engine.dispose()
        logger.info("DatabaseManager connections closed.")


def init_db(db_url: Optional[str] = None, schema_path: Optional[Path] = None) -> None:
    """
    Initializes database schema by executing schema.sql statements.
    
    Parameters
    ----------
    db_url : Optional[str]
        Optional custom connection string (e.g. for testing sqlite:///:memory:).
    schema_path : Optional[Path]
        Optional custom schema SQL file path.
    """
    db_manager = DatabaseManager.get_instance(db_url)
    target_schema = schema_path or DEFAULT_SCHEMA_PATH

    if not target_schema.exists():
        raise FileNotFoundError(f"Schema SQL file not found at: {target_schema}")

    logger.info(f"Applying database schema from {target_schema}...")
    with open(target_schema, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    # Split script into individual statements or use raw connection for executescript
    with db_manager.engine.begin() as conn:
        if db_manager.is_sqlite:
            # Raw sqlite3 executescript handles multi-statement schema.sql directly
            raw_conn = conn.connection
            raw_cursor = raw_conn.cursor()
            raw_cursor.executescript(schema_sql)
            raw_cursor.close()
        else:
            # PostgreSQL execution statement by statement
            statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
            for stmt in statements:
                conn.execute(text(stmt))

    logger.info("Database schema initialized successfully.")


@contextmanager
def get_db_session(db_url: Optional[str] = None) -> Generator[Session, None, None]:
    """
    Context manager for database sessions with automatic commit/rollback.
    Yields an active SQLAlchemy Session.
    """
    db_manager = DatabaseManager.get_instance(db_url)
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error, rolling back transaction: {e}")
        raise
    finally:
        session.close()
