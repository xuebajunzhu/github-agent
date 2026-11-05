"""Circuit breaker tests: the 401-storm reliability guard."""
from __future__ import annotations

import pytest

from app.config import Settings
from app.llm.base import AnalysisError, AnalysisResult, LLMAnalyzer
from app.llm.circuit_breaker import CircuitBreaker
from app.services.analysis_service import AnalysisService


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_opens_after_threshold_and_blocks():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=3, cooldown_seconds=300, clock=clock)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.open
    assert breaker.allow() is False
    clock.advance(299)
    assert breaker.allow() is False
    clock.advance(2)  # cooldown elapsed
    assert breaker.allow() is True  # probe window


def test_probe_extends_window_so_no_second_rush():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=1, cooldown_seconds=300, clock=clock)
    breaker.record_failure()
    clock.advance(300)
    assert breaker.allow() is True  # probe
    assert breaker.allow() is False  # concurrent callers stay blocked


def test_success_resets_failure_count():
    clock = FakeClock()
    breaker = CircuitBreaker(threshold=3, cooldown_seconds=300, clock=clock)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.open is False  # count restarted, threshold not reached


class AlwaysFail(LLMAnalyzer):
    name = "dead"

    def __init__(self):
        self.calls = 0

    def available(self) -> bool:
        return True

    def analyze(self, project_data: dict) -> AnalysisResult:
        self.calls += 1
        raise AnalysisError("401 invalid key")

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        raise AnalysisError("401 invalid key")


class TrainedLike(LLMAnalyzer):
    """The offline fallback: no chat, analysis-only."""

    name = "trained"

    def __init__(self):
        self.analyzed = 0

    def available(self) -> bool:
        return True

    def analyze(self, project_data: dict) -> AnalysisResult:
        self.analyzed += 1
        from app.llm.base import AnalysisResult

        return AnalysisResult(summary="fallback", provider="trained")


def _service(llm: LLMAnalyzer, fallback: LLMAnalyzer | None = None) -> AnalysisService:
    settings = Settings(
        _env_file=None, llm_breaker_threshold=3, llm_breaker_cooldown_seconds=300
    )
    analyzers = [llm] + ([fallback] if fallback else [])
    return AnalysisService(analyzers, settings)


def test_breaker_stops_the_401_storm_and_fallback_answers():
    dead = AlwaysFail()
    fallback = TrainedLike()
    service = _service(dead, fallback)

    for round_index in range(4):
        result = service.analyze({"full_name": "acme/x"})
        assert result.provider == "trained"

    # 3 calls tripped the breaker; the 4th skipped the dead analyzer entirely
    assert dead.calls == 3, "the breaker must cut the retry storm at the threshold"
    assert fallback.analyzed == 4


def test_breaker_probes_after_cooldown():
    import time as time_module

    clock = FakeClock()
    dead = AlwaysFail()
    service = _service(dead)
    service._breakers["dead"] = CircuitBreaker(
        threshold=3, cooldown_seconds=300, clock=clock
    )
    for _ in range(3):
        with pytest.raises(AnalysisError):
            service.analyze({"full_name": "acme/x"})
    assert dead.calls == 3
    clock.advance(300)  # cooldown elapsed -> one probe allowed
    with pytest.raises(AnalysisError):
        service.analyze({"full_name": "acme/x"})
    assert dead.calls == 4  # exactly one probe, not a storm


def test_chat_path_uses_the_same_breaker():
    dead = AlwaysFail()
    service = _service(dead)
    for _ in range(3):
        with pytest.raises(AnalysisError):
            service.chat("system", "user")
    assert dead.calls == 3
    with pytest.raises(AnalysisError):
        service.chat("system", "user")
    assert dead.calls == 3, "open circuit must not spend a call"
