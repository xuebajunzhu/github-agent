"""Tests for the Redis-optional infrastructure layer (memory paths run
everywhere; Redis paths activate only when REDIS_URL points at a live server)."""
from __future__ import annotations

import time

from app.config import Settings
from app.infra import MemoryTTLCache, get_cache
from app.utils.filelock import RedisLock, job_lock


def test_memory_cache_roundtrip_and_expiry():
    cache = MemoryTTLCache()
    cache.put("k", {"a": 1}, ttl_seconds=60)
    assert cache.get("k") == {"a": 1}
    cache.put("short", "v", ttl_seconds=-1)  # already in the past
    assert cache.get("short") is None
    cache.delete("k")
    assert cache.get("k") is None


def test_get_cache_defaults_to_memory_without_redis():
    settings = Settings(_env_file=None, redis_url=None)
    assert isinstance(get_cache(settings), MemoryTTLCache)


def test_redis_lock_degrades_to_file_when_unreachable(tmp_path, monkeypatch):
    # port 1 is unreachable: acquire() must fall back to a file lock, not crash
    monkeypatch.setattr("app.utils.filelock.LOCK_DIR", tmp_path)
    lock = RedisLock("redis://127.0.0.1:1/0", "unittest")
    assert lock.acquire() is True, "unreachable redis must degrade to file lock"
    lock.release()


def test_job_lock_still_excludes_concurrent_holders(tmp_path, monkeypatch):
    monkeypatch.setattr("app.utils.filelock.LOCK_DIR", tmp_path)
    with job_lock("unit-x") as first:
        assert first is True
        with job_lock("unit-x") as second:
            assert second is False
    with job_lock("unit-x") as third:
        assert third is True
