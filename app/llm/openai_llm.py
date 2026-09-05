"""OpenAI-backed analyzer (also works with any OpenAI-compatible endpoint)."""
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


class OpenAIAnalyzer(LLMAnalyzer):
    name = "openai"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None

    def available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(self._settings.openai_api_key)

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._settings.openai_api_key,
                base_url=self._settings.openai_base_url,
                timeout=self._settings.llm_timeout_seconds,
            )
        return self._client

    def _chat(self, system: str, user: str) -> str:
        client = self._ensure_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self._settings.openai_model,
                    messages=messages,
                    temperature=0.2,
                )
                content = response.choices[0].message.content
                if not content:
                    raise AnalysisError("OpenAI returned an empty message")
                return content
            except Exception as exc:  # noqa: BLE001  # degrade via AnalysisError
                last_error = exc
                logger.warning(f"OpenAI call failed (attempt {attempt + 1}): {exc}")
                if attempt < self._settings.llm_max_retries:
                    time.sleep(min(2**attempt, 8))
        raise AnalysisError(f"OpenAI call failed after retries: {last_error}")

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
            logger.warning(f"OpenAI rerank failed: {exc}")
            return None
