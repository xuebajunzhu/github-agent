"""Pluggable LLM package with a factory for building the degradation chain."""
from app.config import Settings
from app.llm.base import (
    AnalysisError,
    AnalysisResult,
    LLMAnalyzer,
)
from app.llm.anthropic_llm import AnthropicAnalyzer
from app.llm.local_llm import LocalModelAnalyzer
from app.llm.openai_llm import OpenAIAnalyzer
from app.llm.rule_based import RuleBasedAnalyzer
from loguru import logger

__all__ = [
    "AnalysisError",
    "AnalysisResult",
    "LLMAnalyzer",
    "OpenAIAnalyzer",
    "AnthropicAnalyzer",
    "LocalModelAnalyzer",
    "RuleBasedAnalyzer",
    "build_analyzer_chain",
]


def _build(provider: str, settings: Settings) -> LLMAnalyzer:
    if provider == "openai":
        return OpenAIAnalyzer(settings)
    if provider == "anthropic":
        return AnthropicAnalyzer(settings)
    if provider == "trained":
        from app.llm.trained_analyzer import TrainedModelAnalyzer

        return TrainedModelAnalyzer(settings)
    if provider == "local":
        return LocalModelAnalyzer(settings)
    if provider == "rule":
        return RuleBasedAnalyzer()
    raise ValueError(f"Unknown LLM provider: {provider}")


def build_analyzer_chain(settings: Settings) -> list[LLMAnalyzer]:
    """Build analyzers in priority order: configured primary, then fallbacks.

    Unavailable/failing backends are skipped at call time; dedup keeps order.
    """
    chain: list[LLMAnalyzer] = []
    for provider in [settings.llm_provider, *settings.llm_fallback_providers]:
        try:
            analyzer = _build(provider, settings)
        except ValueError as exc:
            logger.warning(str(exc))
            continue
        except ImportError as exc:
            # e.g. LLM_PROVIDER=trained without torch installed -> skip gracefully
            logger.warning(f"LLM provider '{provider}' unavailable (missing dependency): {exc}")
            continue
        if not any(isinstance(analyzer, type(existing)) for existing in chain):
            chain.append(analyzer)
    if not chain:
        chain.append(RuleBasedAnalyzer())
    return chain
