"""Analysis service: runs the analyzer chain with automatic degradation."""
from __future__ import annotations

from typing import Sequence

from loguru import logger

from app.config import Settings
from app.llm.base import AnalysisError, AnalysisResult, LLMAnalyzer
from app.llm.circuit_breaker import CircuitBreaker


class AnalysisService:
    def __init__(self, analyzers: Sequence[LLMAnalyzer], settings: Settings | None = None):
        self.analyzers = list(analyzers)
        self._settings = settings
        threshold = getattr(settings, "llm_breaker_threshold", 3) if settings else 3
        cooldown = getattr(settings, "llm_breaker_cooldown_seconds", 300) if settings else 300
        # Design-for-Reliability: a dead upstream (revoked key, exhausted quota)
        # opens after `threshold` consecutive failures and stops being called
        # for the cooldown window -- the fallback chain answers meanwhile.
        self._breakers: dict[str, CircuitBreaker] = {
            analyzer.name: CircuitBreaker(threshold=threshold, cooldown_seconds=cooldown)
            for analyzer in self.analyzers
        }

    def _breaker(self, analyzer: LLMAnalyzer) -> CircuitBreaker:
        return self._breakers.setdefault(
            analyzer.name,
            CircuitBreaker(
                threshold=getattr(self._settings, "llm_breaker_threshold", 3),
                cooldown_seconds=getattr(self._settings, "llm_breaker_cooldown_seconds", 300),
            ),
        )

    @property
    def provider_name(self) -> str:
        """Name of the first available analyzer, or 'none'."""
        for analyzer in self.analyzers:
            if analyzer.available():
                return analyzer.name
        return "none"

    def chat(self, system: str, user: str) -> str:
        """Free-form LLM chat walking the fallback chain; for general tasks
        (research planning, classification) rather than repo analysis."""
        last_error: Exception | None = None
        for analyzer in self.analyzers:
            if not analyzer.available():
                continue
            breaker = self._breaker(analyzer)
            if not breaker.allow():
                logger.warning(
                    f"Chat skipped '{analyzer.name}': circuit open "
                    f"({breaker.failures} consecutive failures)."
                )
                continue
            try:
                result = analyzer.chat(system, user)
            except NotImplementedError:
                continue  # analysis-only backends (trained / rule)
            except AnalysisError as exc:
                breaker.record_failure()
                last_error = exc
                logger.warning(f"Chat failed with '{analyzer.name}': {exc}")
                continue
            breaker.record_success()
            return result
        raise AnalysisError(f"No available LLM backend for chat: {last_error}")

    def analyze(self, project_data: dict) -> AnalysisResult:
        """Try analyzers in priority order; degrade on any failure."""
        available = [analyzer for analyzer in self.analyzers if analyzer.available()]
        if not available:
            raise AnalysisError("No LLM analyzer is available")
        errors: list[str] = []
        for index, analyzer in enumerate(available):
            breaker = self._breaker(analyzer)
            if not breaker.allow():
                logger.warning(
                    f"Analyzer '{analyzer.name}' skipped: circuit open "
                    f"({breaker.failures} consecutive failures)."
                )
                continue
            try:
                result = analyzer.analyze(project_data)
            except Exception as exc:  # noqa: BLE001  # always degrade, never crash the pipeline
                breaker.record_failure()
                errors.append(f"{analyzer.name}: {exc}")
                logger.warning(f"Analyzer '{analyzer.name}' failed: {exc}")
                continue
            breaker.record_success()
            if index > 0:
                logger.warning(
                    f"LLM analysis degraded: primary '{available[0].name}' unavailable, "
                    f"used '{analyzer.name}' instead."
                )
            return result
        raise AnalysisError("All analyzers failed: " + "; ".join(errors))
