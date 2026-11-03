"""Tests for the LLM analyzer chain factory (provider wiring + dedup)."""
from __future__ import annotations

from app.config import Settings
from app.llm import build_analyzer_chain


def test_deepseek_primary_with_fallbacks():
    settings = Settings(
        _env_file=None,
        llm_provider="deepseek",
        llm_fallback_providers=["trained", "rule"],
        deepseek_api_key="test-key",
    )
    chain = build_analyzer_chain(settings)
    names = [analyzer.name for analyzer in chain]
    # trained is skipped gracefully when torch/models are absent in the test env
    assert names[0] == "deepseek"
    assert "rule" in names
    assert names.index("deepseek") < names.index("rule")


def test_deepseek_unavailable_without_key_falls_through():
    settings = Settings(
        _env_file=None,
        llm_provider="deepseek",
        llm_fallback_providers=["rule"],
        deepseek_api_key=None,
    )
    chain = build_analyzer_chain(settings)
    # unavailable primary still builds (skipped at call time), never crashes
    assert chain, "chain must never be empty"
    assert chain[-1].name == "rule"


def test_openai_and_deepseek_coexist_in_chain():
    """Dedup must be by provider name, not isinstance: DeepSeekAnalyzer
    subclasses OpenAIAnalyzer and must not be collapsed into it."""
    settings = Settings(
        _env_file=None,
        llm_provider="deepseek",
        llm_fallback_providers=["openai", "rule"],
        deepseek_api_key="test-key",
        openai_api_key="test-key-2",
    )
    chain = build_analyzer_chain(settings)
    names = [analyzer.name for analyzer in chain]
    assert "deepseek" in names
    assert "openai" in names
    assert names.index("deepseek") < names.index("openai")


def test_unknown_provider_is_skipped_not_fatal():
    settings = Settings(
        _env_file=None,
        llm_provider="no-such-provider",
        llm_fallback_providers=["rule"],
    )
    chain = build_analyzer_chain(settings)
    assert [analyzer.name for analyzer in chain] == ["rule"]
