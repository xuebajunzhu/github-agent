"""Optional infrastructure backends: the Redis-optional layer.

Design rule (mirrors the embedder/vectorstore pattern): every piece of
infrastructure has an in-process default implementation that requires zero
extra services, and a Redis implementation that activates only when REDIS_URL
is configured. If Redis is configured but unreachable, callers degrade to the
in-process default with a warning -- a cache/lock/limiter outage must never
take the service down.
"""
from app.infra.cache import CacheBackend, MemoryTTLCache, RedisTTLCache, get_cache

__all__ = ["CacheBackend", "MemoryTTLCache", "RedisTTLCache", "get_cache"]
