"""Anthropic (Claude) backed analyzer."""
from __future__ import annotations

import time

from loguru import logger

from app.config import Settings
from app.llm.base import (
    LLMAnalyzer,
    AnalysisError,
    AnalysisResult,
    SYSTEM_PROMPT,
    build_analysis_prompt,
    build_rerank_prompt,
    extract_json,
    result_from_json,
)


class AnthropicAnalyzer(LLMAnalyzer):
    name = "anthropic"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None

    def available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(self._settings.anthropic_api_key)

    def _ensure_client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(
                api_key=self._settings.anthropic_api_key,
                timeout=self._settings.llm_timeout_seconds,
            )
        return self._client

    def _chat(self, system: str, user: str) -> str:
        client = self._ensure_client()
        last_error: Exception | None = None
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                response = client.messages.create(
                    model=self._settings.anthropic_model,
                    max_tokens=1024,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                content = "".join(
                    block.text for block in response.content if getattr(block, "type", "") == "text"
                )
                if not content:
                    raise AnalysisError("Anthropic returned an empty message")
                return content
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(f"Anthropic call failed (attempt {attempt + 1}): {exc}")
                if attempt < self._settings.llm_max_retries:
                    time.sleep(min(2**attempt, 8))
        raise AnalysisError(f"Anthropic call failed after retries: {last_error}")

    def analyze(self, project_data: dict) -> AnalysisResult:
        content = self._chat(SYSTEM_PROMPT, build_analysis_prompt(project_data))
        return result_from_json(extract_json(content), self.name)

    def rank_candidates(self, query: str, candidates: list[dict]) -> tuple[list[int], str] | None:
        try:
            content = self._chat(RERANK_SYSTEM_PROMPT, build_rerank_prompt(query, candidates))
            data = extract_json(content)
            ranking = [int(index) for index in data.get("ranking") or []]
            return ranking, str(data.get("reason") or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Anthropic rerank failed: {exc}")
            return None
