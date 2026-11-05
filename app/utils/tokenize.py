"""Shared tokenizer: latin tokens + CJK character bigrams.

Chinese has no spaces, so CJK runs are split into character bigrams — a whole
Chinese sentence must not collapse into a single token. Used by the hashing
TF-IDF embedder and the BM25 hybrid retrieval channel so both agree on what a
term is.
"""
from __future__ import annotations

import re

_LATIN_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    tokens = _LATIN_RE.findall(text.lower())
    for match in _CJK_RUN_RE.finditer(text):
        run = match.group()
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(a + b for a, b in zip(run, run[1:]))
    return tokens
