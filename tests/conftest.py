from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.utils.filelock as filelock
from app.config import Settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _isolated_lock_dir(tmp_path, monkeypatch):
    """Tests must never touch the live deployment's pipeline lock file: two
    concurrent pytest runs (or the running service) sharing logs/pipeline.lock
    made every pipeline test flake with 'another pipeline job is running'."""
    monkeypatch.setattr(filelock, "LOCK_DIR", tmp_path / "locks")


@pytest.fixture()
def settings(tmp_path) -> Settings:
    """Fully degraded (no network, no API keys) settings for deterministic tests."""
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        scheduler_enabled=False,
        llm_provider="rule",
        llm_fallback_providers=[],
        embedder_provider="tfidf",
        embedding_dimension=64,
        vector_store_backend="memory",
        admin_token="test-admin-token",
        secret_key="test-secret",
        github_client_id=None,
        github_client_secret=None,
        github_fetch_readme=False,
        asr_provider="none",
        tts_provider="none",
    )


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
