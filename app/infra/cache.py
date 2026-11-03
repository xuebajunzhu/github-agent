"""TTL cache backends: in-memory default, Redis when REDIS_URL is set.

Used for research results and any future hot data. Values are JSON blobs;
TTL is enforced by the backend (lazy expiry in memory, native EXPIRE in
Redis, so restart-survival and cross-process sharing come for free when
Redis is enabled).
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Protocol

from loguru import logger

from app.config import Settings


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...

    def put(self, key: str, value: Any, ttl_seconds: int) -> None: ...

    def delete(self, key: str) -> None: ...


class MemoryTTLCache:
    """Thread-safe in-process TTL cache (the zero-dependency default)."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                self._data.pop(key, None)
                return None
            return value

    def put(self, key: str, value: Any, ttl_seconds: int) -> None:
        with self._lock:
            self._data[key] = (value, time.monotonic() + ttl_seconds)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


class RedisTTLCache:
    """Redis-backed cache (survives restarts, shared across processes)."""

    def __init__(self, redis_url: str):
        import redis

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Redis cache get failed: {exc}")
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    def put(self, key: str, value: Any, ttl_seconds: int) -> None:
        try:
            self._client.set(key, json.dumps(value, ensure_ascii=False), ex=max(1, ttl_seconds))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Redis cache put failed: {exc}")

    def delete(self, key: str) -> None:
        try:
            self._client.delete(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Redis cache delete failed: {exc}")


_cache: CacheBackend | None = None
_cache_lock = threading.Lock()


def get_cache(settings: Settings) -> CacheBackend:
    """Process-wide cache singleton; Redis only when REDIS_URL is configured."""
    global _cache
    with _cache_lock:
        if _cache is None:
            if settings.redis_url:
                try:
                    _cache = RedisTTLCache(settings.redis_url)
                    logger.info("Cache backend: redis")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Redis cache unavailable ({exc}); using in-memory cache.")
                    _cache = MemoryTTLCache()
            else:
                _cache = MemoryTTLCache()
                logger.info("Cache backend: in-memory")
        return _cache
