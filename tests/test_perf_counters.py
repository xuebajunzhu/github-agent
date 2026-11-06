"""StageTimers unit tests: windowing, percentiles, thread safety."""
from __future__ import annotations

import threading

import pytest

from app.services.perf_counters import StageTimers, record_ms


def test_snapshot_empty_is_safe():
    timers = StageTimers()
    assert timers.snapshot() == {}


def test_percentile_ordering_and_window_cap():
    timers = StageTimers(window=10)
    for value in range(1, 21):  # 1..20 ms, window keeps the last 10 (11..20)
        timers.record("s", float(value))
    snap = timers.snapshot()["s"]
    assert snap["count"] == 10  # window cap
    assert snap["p50_ms"] == 15.0  # round-half-even index into 11..20
    assert snap["p95_ms"] == 20.0
    assert snap["avg_ms"] == pytest.approx(15.5)


def test_record_ms_context_records_elapsed():
    timers = StageTimers()
    with record_ms(timers, "encode"):
        pass  # near-instant
    snap = timers.snapshot()
    assert snap["encode"]["count"] == 1
    assert snap["encode"]["avg_ms"] >= 0


def test_threaded_recording_never_loses_counts():
    timers = StageTimers(window=1000)

    def worker():
        for _ in range(100):
            timers.record("s", 1.0)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert timers.snapshot()["s"]["count"] == 500

