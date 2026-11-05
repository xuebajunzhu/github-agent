"""Circuit breaker for the LLM analyzer chain (Design-for-Reliability).

A dead upstream (revoked key, exhausted quota, provider outage) used to cost
`llm_max_retries` backoff sleeps on EVERY call: the 401 storm of 2026-11-04
hammered the DeepSeek API hundreds of times with a revoked key while every
request paid the full retry tax before degrading to the fallback chain.

The breaker counts consecutive failures per analyzer; after `threshold` it
opens and `allow()` returns False for `cooldown_seconds`, after which a single
probe window lets one call through (half-open). Success resets everything.
Failures are reported by the caller — the breaker never raises.
"""
from __future__ import annotations

import time
from typing import Callable


class CircuitBreaker:
    def __init__(
        self,
        threshold: int = 3,
        cooldown_seconds: int = 300,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._threshold = max(1, threshold)
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        """May a call go through right now? While open, exactly one probe
        window per cooldown passes (the window extends on each probe, so
        concurrent callers do not all rush the dead upstream)."""
        if self._opened_at is None:
            return True
        if self._clock() - self._opened_at >= self._cooldown:
            # extend the window: this caller IS the probe
            self._opened_at = self._clock()
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold and self._opened_at is None:
            self._opened_at = self._clock()

    @property
    def open(self) -> bool:
        return self._opened_at is not None and not (
            self._clock() - self._opened_at >= self._cooldown
        )

    @property
    def failures(self) -> int:
        return self._failures
