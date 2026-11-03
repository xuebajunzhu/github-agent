"""Analysis service: runs the analyzer chain with automatic degradation."""
from __future__ import annotations

from typing import Sequence

from loguru import logger

from app.llm.base import AnalysisError, AnalysisResult, LLMAnalyzer


class AnalysisService:
    def __init__(self, analyzers: Sequence[LLMAnalyzer]):
        self.analyzers = list(analyzers)

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
            try:
                return analyzer.chat(system, user)
            except NotImplementedError:
                continue  # analysis-only backends (trained / rule)
            except AnalysisError as exc:
                last_error = exc
                logger.warning(f"Chat failed with '{analyzer.name}': {exc}")
        raise AnalysisError(f"No available LLM backend for chat: {last_error}")

    def analyze(self, project_data: dict) -> AnalysisResult:
        """Try analyzers in priority order; degrade on any failure."""
        available = [analyzer for analyzer in self.analyzers if analyzer.available()]
        if not available:
            raise AnalysisError("No LLM analyzer is available")
        errors: list[str] = []
        for index, analyzer in enumerate(available):
            try:
                result = analyzer.analyze(project_data)
            except Exception as exc:  # noqa: BLE001  # always degrade, never crash the pipeline
                errors.append(f"{analyzer.name}: {exc}")
                logger.warning(f"Analyzer '{analyzer.name}' failed: {exc}")
                continue
            if index > 0:
                logger.warning(
                    f"LLM analysis degraded: primary '{available[0].name}' unavailable, "
                    f"used '{analyzer.name}' instead."
                )
            return result
        raise AnalysisError("All analyzers failed: " + "; ".join(errors))
