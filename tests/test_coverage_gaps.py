"""Tests for the coverage gaps found by the deep audit:
ChromaVectorStore (zero coverage), voice upload cap, rate-limiter proxy
logic, research-cache pruning, and the additive DB migration."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi import Request
from starlette.datastructures import Headers

from app.api.middleware import RateLimitMiddleware
from app.config import Settings
from app.database import create_db_engine, init_db
from app.vectorstore.chroma_store import ChromaVectorStore


# ----------------------------------------------------------------- Chroma


@pytest.fixture()
def chroma(tmp_path):
    return ChromaVectorStore(str(tmp_path / "chroma_test"), "unit_collection")


def test_chroma_roundtrip_and_cosine_order(chroma):
    chroma.upsert(
        ["a", "b"],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=["doc a", "doc b"],
        metadatas=[{"full_name": "a"}, {"full_name": "b"}],
    )
    assert chroma.count() == 2
    hits = chroma.query([1.0, 0.0, 0.0], top_k=2)
    assert hits[0][0] == "a"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-3)
    chroma.delete(["a"])
    assert chroma.count() == 1


def test_chroma_upsert_overwrites_by_id(chroma):
    chroma.upsert(["x"], [[1.0, 0.0]])
    chroma.upsert(["x"], [[0.0, 1.0]])
    assert chroma.count() == 1
    assert chroma.query([0.0, 1.0])[0][0] == "x"


def test_chroma_ensure_space_wipes_on_fingerprint_change(tmp_path):
    store = ChromaVectorStore(str(tmp_path / "chroma_space"), "space_collection")
    store.upsert(["a"], [[1.0, 0.0]])
    assert store.count() == 1
    # no stored fingerprint = unknown space: the conservative call is to wipe
    # (a no-op here, and a one-time rebuild on upgraded deployments)
    assert store.ensure_space("bge-m3:BAAI/bge-m3") is True
    store.upsert(["a"], [[1.0, 0.0]])
    assert store.ensure_space("trained-384") is True  # mismatch -> wiped
    assert store.count() == 0
    assert store.ensure_space("trained-384") is False  # adopted, idempotent


def test_chroma_existing_ids_reports_drift(chroma):
    chroma.upsert(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    assert chroma.existing_ids(["a", "zzz"]) == {"a"}


# ----------------------------------------------------------- voice upload cap


def test_voice_upload_size_cap_enforced(client):
    from tests.test_voice import FakeSTT

    client.app.state.services.stt = FakeSTT()
    client.app.state.services.settings.max_upload_mb = 1
    big = b"\x00" * (1024 * 1024 + 1)  # 1MB + 1 byte
    response = client.post(
        "/api/voice/transcribe",
        files={"file": ("big.wav", big, "audio/wav")},
    )
    assert response.status_code == 413
    small = client.post(
        "/api/voice/transcribe",
        files={"file": ("ok.wav", b"\x00\x00", "audio/wav")},
    )
    # small upload passes the size gate (then hits FakeSTT normally)
    assert small.status_code == 200


# ------------------------------------------------------- rate limiter proxy


def _request(client_host: str, headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": Headers(headers).raw,
            "client": (client_host, 12345),
        }
    )


def test_proxy_headers_honored_only_from_loopback():
    mw = RateLimitMiddleware(app=None, max_per_minute=10, trust_proxy_headers=True)
    loopback_spoofed = _request("127.0.0.1", {"cf-connecting-ip": "203.0.113.7"})
    assert mw._client_ip(loopback_spoofed) == "203.0.113.7"
    direct_spoofed = _request("203.0.113.9", {"cf-connecting-ip": "1.1.1.1"})
    assert mw._client_ip(direct_spoofed) == "203.0.113.9", (
        "a non-loopback caller must not be able to rotate limit buckets"
    )
    mw_off = RateLimitMiddleware(app=None, max_per_minute=10, trust_proxy_headers=False)
    assert mw_off._client_ip(loopback_spoofed) == "127.0.0.1"


def test_xff_first_hop_used_when_no_cf_header():
    mw = RateLimitMiddleware(app=None, max_per_minute=10, trust_proxy_headers=True)
    req = _request("127.0.0.1", {"x-forwarded-for": "198.51.100.5, 10.0.0.1"})
    assert mw._client_ip(req) == "198.51.100.5"


# ------------------------------------------------- research cache pruning


def test_patrol_prunes_expired_research_cache(tmp_path):
    from datetime import timedelta

    from app.models.research import ResearchCache
    from app.services.patrol_service import PatrolService
    from app.utils.timeutil import utcnow

    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'prune.db').as_posix()}",
        research_cache_retention_days=7,
    )
    engine = create_db_engine(settings.database_url)
    init_db(engine)
    session_factory = __import__(
        "app.database", fromlist=["create_sessionmaker"]
    ).create_sessionmaker(engine)
    with session_factory() as session:
        session.add(
            ResearchCache(
                query_hash="old", query="old", result_json="{}",
                created_at=utcnow() - timedelta(days=30),
            )
        )
        session.add(
            ResearchCache(
                query_hash="new", query="new", result_json="{}",
                created_at=utcnow(),
            )
        )
        session.commit()

    patrol = PatrolService(settings, session_factory, None, None, None)
    pruned = patrol._prune_query_history()

    assert pruned == 1
    with session_factory() as session:
        remaining = {row.query_hash for row in session.query(ResearchCache).all()}
    assert remaining == {"new"}


# ------------------------------------------------------- additive migration


def test_migration_upgrades_legacy_database(tmp_path):
    """A pre-existing DB with the OLD projects shape gains the new columns and
    the performance indexes without losing rows."""
    db = tmp_path / "legacy.db"
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE projects (id INTEGER PRIMARY KEY, github_id INTEGER, "
        "full_name TEXT, url TEXT, stars INTEGER)"
    )
    conn.execute(
        "INSERT INTO projects (github_id, full_name, url, stars) VALUES (1, 'acme/old', 'u', 5)"
    )
    conn.commit()
    conn.close()

    engine = create_db_engine(f"sqlite:///{db}")
    init_db(engine)

    from sqlalchemy import inspect

    inspector = inspect(engine)
    columns = {c["name"] for c in inspector.get_columns("projects")}
    assert {"analysis_attempts", "license"} <= columns
    index_names = {ix["name"] for ix in inspector.get_indexes("projects")}
    assert "ix_projects_stars" in index_names
    # legacy row survived
    with engine.connect() as c:
        assert c.exec_driver_sql("SELECT COUNT(*) FROM projects").scalar() == 1
