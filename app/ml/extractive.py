"""Extractive summarization: README sentence scoring with zero generation.

Used when a repository has no description. Scores sentences by title/keyword
overlap plus a position prior, penalizing boilerplate lines. Deterministic and
CPU-free, which keeps the summary path of the trained analyzer reliable.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")

_BOILERPLATE = (
    "license", "copyright", "install", "installation", "getting started",
    "contributing", "badge", "build status", "table of contents", "documentation",
    "acknowledgement", "acknowledgment", "donation", "sponsor",
)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text or "")}


def extractive_summary(
    full_name: str,
    readme: str,
    max_chars: int = 220,
    max_sentences: int = 2,
) -> str:
    """Pick the most informative sentences of a README as the summary."""
    repo_tokens = tokenize(full_name.replace("/", " ").replace("-", " ").replace("_", " "))
    best: list[tuple[float, int, str]] = []
    for position, raw in enumerate(_SENTENCE_RE.split(readme or "")):
        sentence = raw.strip().lstrip("#").lstrip("*- ").strip()
        if len(sentence) < 20 or len(sentence) > 400:
            continue
        lowered = sentence.lower()
        if any(marker in lowered for marker in _BOILERPLATE):
            continue
        if sentence.count("|") >= 2 or sentence.count("![") >= 1 or sentence.startswith("<"):
            continue  # tables, images, html
        tokens = tokenize(sentence)
        if not tokens:
            continue
        overlap = len(tokens & repo_tokens) / max(len(tokens), 1)
        position_prior = 1.0 / (1 + position * 0.15)
        length_score = min(len(sentence) / 120.0, 1.0)
        score = overlap * 2.0 + position_prior + length_score * 0.5
        best.append((score, position, sentence))
    best.sort(key=lambda item: (-item[0], item[1]))
    picked = [sentence for _, _, sentence in best[:max_sentences]]
    summary = " ".join(picked).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."
    return summary
