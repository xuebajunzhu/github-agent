"""Local small-model analyzer via Hugging Face transformers (degraded LLM path)."""
from __future__ import annotations

from loguru import logger

from app.config import Settings
from app.llm.base import (
    LLMAnalyzer,
    AnalysisError,
    AnalysisResult,
    SYSTEM_PROMPT,
    build_analysis_prompt,
    extract_json,
    result_from_json,
)


class LocalModelAnalyzer(LLMAnalyzer):
    """Runs a small text-generation model locally. Small models often produce
    malformed JSON; a parse failure raises AnalysisError so the chain degrades
    to the rule-based analyzer."""

    name = "local"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._generator = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_generator(self):
        if self._generator is None:
            from transformers import pipeline

            logger.info(f"Loading local model '{self._settings.local_model_path}' ...")
            self._generator = pipeline(
                "text-generation",
                model=self._settings.local_model_path,
                trust_remote_code=False,
            )
        return self._generator

    def analyze(self, project_data: dict) -> AnalysisResult:
        generator = self._ensure_generator()
        prompt = (
            f"{SYSTEM_PROMPT}\n\n{build_analysis_prompt(project_data)}\nJSON: "
        )
        try:
            outputs = generator(prompt, max_new_tokens=300, do_sample=False, return_full_text=False)
        except Exception as exc:  # noqa: BLE001
            raise AnalysisError(f"Local model inference failed: {exc}") from exc
        try:
            first = outputs[0] if isinstance(outputs, list) else outputs
            text = first["generated_text"] if isinstance(first, dict) else str(first)
        except Exception as exc:  # noqa: BLE001
            raise AnalysisError(f"Unexpected local model output format: {exc}") from exc
        return result_from_json(extract_json(text), self.name)
