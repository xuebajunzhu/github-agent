"""Rolling per-stage latency counters (Design-for-Verifiability).

Thread-safe sliding-window timers for the hot path (recommend: encode /
search / fusion / rerank). The admin surface snapshots p50/p95 so tuning
decisions — rerank on/off, hybrid weights — are made on data instead of
anecdotes (see scripts/eval_ranking.py for the offline counterpart).

Standalone module: wiring is one call site (`timers.record(stage, ms)`) and
one snapshot read, deliberately kept out of fast-moving files.
"""
from __future__ import annotations

import threading
from collections import deque
from statistics import mean
from typing import Dict, List


class StageTimers:
    def __init__(self, window: int = 200):
        self._window = window
        self._lock = threading.Lock()
        self._stages: Dict[str, deque] = {}

    def record(self, stage: str, ms: float) -> None:
        with self._lock:
            bucket = self._stages.setdefault(stage, deque(maxlen=self._window))
            bucket.append(float(ms))

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Per stage: count, avg, p50, p95 (over the sliding window)."""
        out: Dict[str, Dict[str, float]] = {}
        with self._lock:
            for stage, samples in self._stages.items():
                if not samples:
                    continue
                ordered = sorted(samples)
                count = len(ordered)

                def pct(p: float) -> float:
                    index = min(count - 1, max(0, round(p * (count - 1))))
                    return ordered[index]

                out[stage] = {
                    "count": count,
                    "avg_ms": round(mean(ordered), 1),
                    "p50_ms": round(pct(0.50), 1),
                    "p95_ms": round(pct(0.95), 1),
                }
        return out


def record_ms(timers, stage: str):
    """Context manager sugar: `with record_ms(timers, 'encode'): ...` — the
    stage is recorded with the elapsed wall time on exit (even on failure)."""
    return _RecordContext(timers, stage)


class _RecordContext:
    def __init__(self, timers, stage: str):
        import time

        self._timers = timers
        self._stage = stage
        self._start = time.perf_counter()

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        import time

        elapsed = (time.perf_counter() - self._start) * 1000
        self._timers.record(self._stage, elapsed)
        return None


# process-wide instance: the admin surface snapshots this, recommend records
# into it. Tests may construct their own StageTimers for isolation.
TIMERS = StageTimers()
