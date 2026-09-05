"""Pluggable LLM analyzer base: prompts, lenient JSON parsing, and the ABC."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class AnalysisError(Exception):
    """Raised when an analyzer fails or returns unusable output."""


@dataclass
class AnalysisResult:
    summary: str
    use_cases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    provider: str = "unknown"


class LLMAnalyzer(ABC):
    """A single analysis backend. Implementations should raise AnalysisError on failure."""

    name: str = "base"

    @abstractmethod
    def analyze(self, project_data: dict) -> AnalysisResult:
        """Analyze repository metadata and return summary/use_cases/tags/dependencies."""

    def available(self) -> bool:
        """Whether this backend can currently be used (dependency + config checks)."""
        return True

    def rank_candidates(
        self, query: str, candidates: list[dict]
    ) -> tuple[list[int], str] | None:
        """Optional: reorder recommendation candidates for a user scenario.

        Returns (ranking indices best-first, reason) or None if unsupported.
        """
        return None


SYSTEM_PROMPT = (
    "You are an expert open-source software analyst. Given metadata about a GitHub "
    "repository, produce a strict JSON object with exactly these keys: "
    '{"summary": string, "use_cases": string[], "tags": string[], "dependencies": string[]}. '
    "'summary' is one or two concise sentences describing what the project does. "
    "'use_cases' are short scenario phrases (e.g. \"Web 开发\", \"自然语言处理\"). "
    "'tags' are lowercase technical keywords (e.g. \"python\", \"pytorch\"). "
    "'dependencies' are likely runtime dependencies or related ecosystem tools. "
    "Respond with JSON only, no markdown fences, no explanations."
)

RERANK_SYSTEM_PROMPT = (
    "You are a software recommendation assistant. The user describes a scenario; "
    "you rank candidate GitHub repositories by relevance. Respond with JSON only."
)


def build_analysis_prompt(project_data: dict) -> str:
    parts = [
        f"Repository: {project_data.get('full_name', 'unknown')}",
        f"Description: {project_data.get('description') or 'N/A'}",
        f"Primary language: {project_data.get('language') or 'unknown'}",
        f"Topics: {', '.join(project_data.get('topics') or []) or 'none'}",
    ]
    readme = (project_data.get("readme_excerpt") or "").strip()
    if readme:
        parts.append(f"README excerpt:\n{readme}")
    return "\n".join(parts)


def build_rerank_prompt(query: str, candidates: list[dict]) -> str:
    lines = [f"User scenario: {query}", "", "Candidate repositories:"]
    for index, candidate in enumerate(candidates):
        tags = ", ".join(candidate.get("tags") or []) or "none"
        summary = candidate.get("summary") or "N/A"
        lines.append(f"{index}. {candidate.get('full_name')}: {summary} (tags: {tags})")
    lines.append("")
    lines.append(
        'Return strict JSON: {"ranking": [candidate indices ordered by relevance, best first], '
        '"reason": "one short sentence explaining the top choice, in the same language as the user scenario"}'
    )
    return "\n".join(lines)


def extract_json(text: str) -> dict:
    """Extract the first JSON object from a model response, tolerating code fences."""
    if not text:
        raise AnalysisError("Empty response from model")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            language_tag = cleaned[:first_newline].strip().lower()
            if language_tag in {"json", "javascript", "js", ""}:
                cleaned = cleaned[first_newline + 1 :]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise AnalysisError(f"No JSON object found in response: {text[:200]!r}")
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Invalid JSON in model response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AnalysisError("Model response is not a JSON object")
    return parsed


def _str_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def result_from_json(data: dict, provider: str) -> AnalysisResult:
    summary = str(data.get("summary") or "").strip()
    if not summary:
        raise AnalysisError("Analyzer returned an empty summary")
    return AnalysisResult(
        summary=summary[:1000],
        use_cases=_str_list(data.get("use_cases"))[:10],
        tags=[tag.lower() for tag in _str_list(data.get("tags"))][:15],
        dependencies=_str_list(data.get("dependencies"))[:10],
        provider=provider,
    )
