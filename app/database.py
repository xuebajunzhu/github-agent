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
    _add_missing_columns(engine)


def _add_missing_columns(engine: Engine) -> None:
    """Additive migration for pre-existing databases: create_all never alters
    tables that already exist."""
    from sqlalchemy import inspect, text

    migrations: dict[str, dict[str, str]] = {
        "projects": {
            "analysis_attempts": "ALTER TABLE projects ADD COLUMN analysis_attempts INTEGER NOT NULL DEFAULT 0",
            "license": "ALTER TABLE projects ADD COLUMN license VARCHAR(64)",
        },
    }
    # indexes for the queries that grow with traffic (see admin feedback /
    # digest aggregation and the language/star filtered listing)
    indexes: tuple[str, ...] = (
        "CREATE INDEX IF NOT EXISTS ix_query_history_created_at ON query_history (created_at)",
        "CREATE INDEX IF NOT EXISTS ix_feedback_created_at ON feedback (created_at)",
        "CREATE INDEX IF NOT EXISTS ix_projects_language ON projects (language)",
        "CREATE INDEX IF NOT EXISTS ix_projects_stars ON projects (stars)",
    )
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, columns in migrations.items():
            if table not in existing_tables:
                continue
            present = {column["name"] for column in inspector.get_columns(table)}
            for column_name, ddl in columns.items():
                if column_name not in present:
                    conn.execute(text(ddl))
        for ddl in indexes:
            table = ddl.split(" ON ")[1].split(" ")[0]
            if table in existing_tables:
                conn.execute(text(ddl))
