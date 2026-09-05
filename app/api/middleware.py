"""Simple in-memory per-IP rate limiter for public deployment.

Good enough for a single-process demo deployment; swap for a token-bucket
based on Redis when scaling beyond one box.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_per_minute: int = 120):
        super().__init__(app)
        self._max = max_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._last_sweep = time.monotonic()

    async def dispatch(self, request: Request, call_next):
        client = request.client
        ip = client.host if client else "unknown"
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

        # opportunistic cleanup so idle IPs do not accumulate forever
        if now - self._last_sweep > 300:
            self._last_sweep = now
            stale = [ip for ip, stamps in self._hits.items() if not stamps or now - stamps[-1] > 600]
            for stale_ip in stale:
                self._hits.pop(stale_ip, None)

        return await call_next(request)
