"""Simple in-memory per-client rate limiter for public deployment.

Behind a tunnel/reverse proxy every request arrives from the proxy's source
address, so all users would share one bucket. Set TRUST_PROXY_HEADERS=true to
rate-limit by the forwarded client IP instead -- forwarded headers are only
honored from loopback clients, so a direct caller cannot spoof them. Swap for
a token-bucket based on Redis when scaling beyond one box.
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
    def __init__(self, app, max_per_minute: int = 120, trust_proxy_headers: bool = False):
        super().__init__(app)
        self._max = max_per_minute
        self._trust_proxy_headers = trust_proxy_headers
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._last_sweep = time.monotonic()

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

    async def dispatch(self, request: Request, call_next):
        ip = self._client_ip(request)
        now = time.monotonic()

        hits = self._hits[ip]
        while hits and now - hits[0] > 60.0:
            hits.popleft()
        if len(hits) >= self._max:
            retry_after = 60 - (now - hits[0]) if hits else 60
            return JSONResponse(
                {"detail": "Rate limit exceeded, slow down."},
                status_code=429,
                headers={"Retry-After": str(max(1, int(retry_after)))},
            )
        hits.append(now)

        # opportunistic cleanup so idle/spoofed IPs do not accumulate forever
        if now - self._last_sweep > 300 or len(self._hits) > _MAX_TRACKED_CLIENTS:
            self._last_sweep = now
            stale = [ip for ip, stamps in self._hits.items() if not stamps or now - stamps[-1] > 600]
            for stale_ip in stale:
                self._hits.pop(stale_ip, None)

        return await call_next(request)
