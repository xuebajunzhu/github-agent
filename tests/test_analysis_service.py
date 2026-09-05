from __future__ import annotations

import pytest

from app.config import Settings
from app.llm.base import AnalysisError, AnalysisResult, LLMAnalyzer
from app.llm import build_analyzer_chain
from app.services.analysis_service import AnalysisService


class FailingAnalyzer(LLMAnalyzer):
    name = "failing"

    def analyze(self, project_data: dict) -> AnalysisResult:
        raise AnalysisError("boom")


class WorkingAnalyzer(LLMAnalyzer):
    name = "working"

    def analyze(self, project_data: dict) -> AnalysisResult:
        return AnalysisResult(summary="works", provider=self.name)


class UnavailableAnalyzer(LLMAnalyzer):
    name = "unavailable"

    def available(self) -> bool:
        return False

    def analyze(self, project_data: dict) -> AnalysisResult:
        raise AssertionError("should not be called")


def test_degrades_to_next_analyzer():
    service = AnalysisService([FailingAnalyzer(), WorkingAnalyzer()])
    result = service.analyze({"full_name": "x"})
    assert result.provider == "working"


def test_skips_unavailable_analyzers():
    service = AnalysisService([UnavailableAnalyzer(), WorkingAnalyzer()])
    assert service.provider_name == "working"
    result = service.analyze({"full_name": "x"})
    assert result.provider == "working"


def test_raises_when_all_fail():
    service = AnalysisService([FailingAnalyzer()])
    with pytest.raises(AnalysisError):
        service.analyze({"full_name": "x"})


def test_provider_name_none_when_nothing_available():
    service = AnalysisService([UnavailableAnalyzer()])
    assert service.provider_name == "none"


def test_build_analyzer_chain_deduplicates():
    settings = Settings(_env_file=None, llm_provider="rule", llm_fallback_providers=["rule"])
    chain = build_analyzer_chain(settings)
    assert len(chain) == 1
    assert chain[0].name == "rule"


def test_build_analyzer_chain_keeps_priority_order():
    settings = Settings(_env_file=None, llm_provider="rule", llm_fallback_providers=["openai"])
    chain = build_analyzer_chain(settings)
    assert [analyzer.name for analyzer in chain] == ["rule", "openai"]
