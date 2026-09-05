"""SQLAlchemy engine/session setup."""
from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base


def _enable_sqlite_pragmas(dbapi_connection, _record) -> None:
    """WAL + busy timeout so API and 24/7 worker processes can share one
    SQLite file without tripping over each other's locks."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def create_db_engine(database_url: str) -> Engine:
    kwargs: dict = {"future": True}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if database_url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
    engine = create_engine(database_url, **kwargs)
    if database_url.startswith("sqlite"):
        event.listen(engine, "connect", _enable_sqlite_pragmas)
    return engine


def create_sessionmaker(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    import app.models  # noqa: F401  # ensure all ORM models are registered

    Base.metadata.create_all(engine)
