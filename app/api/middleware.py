"""Per-client rate limiter: in-memory default, shared Redis sliding window
when REDIS_URL is configured.

The in-memory counters only work within one process; behind multiple
workers/boxes they fragment the budget. With Redis, the window is a sorted
set shared by every instance (ZADD + ZCARD sliding window), so the limit is
global and survives restarts. Any Redis failure degrades that request to the
in-memory path -- a limiter outage must never 500 the service.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

_LOOPBACK = {"127.0.0.1", "::1"}
# hard cap on tracked clients so spoofed/rotated IPs cannot grow the dict forever
_MAX_TRACKED_CLIENTS = 10000


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        max_per_minute: int = 120,
        trust_proxy_headers: bool = False,
        redis_url: str | None = None,
    ):
        super().__init__(app)
        self._max = max_per_minute
        self._trust_proxy_headers = trust_proxy_headers
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._last_sweep = time.monotonic()
        self._redis = None
        if redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(redis_url)
            except Exception as exc:  # noqa: BLE001
                self._redis = None
                print(f"[rate-limit] Redis unavailable ({exc}); using in-memory limiter.")

    def _client_ip(self, request: Request) -> str:
        client = request.client
        ip = client.host if client else "unknown"
        if not self._trust_proxy_headers or ip not in _LOOPBACK:
            return ip
        forwarded = request.headers.get("cf-connecting-ip")
        if not forwarded:
            xff = request.headers.get("x-forwarded-for")
            if xff:
                forwarded = xff.split(",", 1)[0].strip()
        return forwarded or ip

    async def _redis_allow(self, ip: str) -> bool:
        now = time.time()
        key = f"github-agent:rl:{ip}"
        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - 60.0)
        pipe.zadd(key, {f"{now}": now})
        pipe.zcard(key)
        pipe.expire(key, 120)
        count = (await pipe.execute())[2]
        return count <= self._max

    def _memory_allow(self, ip: str, now: float) -> bool:
        hits = self._hits[ip]
        while hits and now - hits[0] > 60.0:
            hits.popleft()
        if len(hits) >= self._max:
            return False
        hits.append(now)
        # opportunistic cleanup so idle/spoofed IPs do not accumulate forever
        if now - self._last_sweep > 300 or len(self._hits) > _MAX_TRACKED_CLIENTS:
            self._last_sweep = now
            stale = [k for k, stamps in self._hits.items() if not stamps or now - stamps[-1] > 600]
            for stale_ip in stale:
                self._hits.pop(stale_ip, None)
        return True

    async def dispatch(self, request: Request, call_next):
        ip = self._client_ip(request)
        now = time.monotonic()

        allowed = None
        if self._redis is not None:
            try:
                allowed = await self._redis_allow(ip)
            except Exception:  # noqa: BLE001  # degrade to memory on any redis failure
                allowed = None
        if allowed is None:
            allowed = self._memory_allow(ip, now)

        if not allowed:
            return JSONResponse(
                {"detail": "Rate limit exceeded, slow down."},
                status_code=429,
                headers={"Retry-After": "1"},
            )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Defense-in-depth response headers for a public-facing service.

    CSP pins every content origin to 'self' (the two HTML pages use inline
    scripts/styles, hence 'unsafe-inline' for those directives only — no
    external origin may load code or receive data). The remaining headers are
    the standard hardening set for an API that also serves HTML."""

    HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'"
        ),
    }

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for name, value in self.HEADERS.items():
            response.headers.setdefault(name, value)
        return response
