"""Pure-Python BM25 index over the indexed project corpus.

The lexical channel of the hybrid retrieval: semantic (BGE-M3) search handles
paraphrases, BM25 handles exact keyword hits ("jq", "kubernetes", 专有名词) that
embeddings blur. At the current index size (hundreds of docs) a plain Python
implementation scores in microseconds; the corpus is rebuilt lazily when the
(done+embedded) population changes.
"""
from __future__ import annotations

import math
from collections import Counter

from app.utils.tokenize import tokenize


class BM25Index:
    """Okapi BM25 over pre-tokenized corpus texts, keyed by caller ids
    (project primary keys for the retrieval fusion)."""

    def __init__(self, ids: list[int | str], corpus: list[str]):
        if len(ids) != len(corpus):
            raise ValueError("ids and corpus must have the same length")
        self._ids = list(ids)
        doc_tokens = [tokenize(text) for text in corpus]
        self._doc_len = [len(tokens) for tokens in doc_tokens]
        self._N = len(doc_tokens)
        self._avgdl = (sum(self._doc_len) / self._N) if self._N else 0.0
        self._tf: list[Counter] = [Counter(tokens) for tokens in doc_tokens]
        self._df: dict[str, int] = Counter()
        for tf in self._tf:
            for term in tf:
                self._df[term] += 1
        # BM25+ style idf, always non-negative
        self._idf = {
            term: math.log(1.0 + (self._N - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }

    def search(
        self, query: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75
    ) -> list[tuple[int | str, float]]:
        if not self._N:
            return []
        scores: list[tuple[float, int]] = []
        query_tokens = tokenize(query)
        for doc_index in range(self._N):
            score = 0.0
            tf = self._tf[doc_index]
            if not tf:
                continue
            length_norm = k1 * (1 - b + b * self._doc_len[doc_index] / self._avgdl)
            for term in query_tokens:
                term_freq = tf.get(term)
                if not term_freq:
                    continue
                idf = self._idf.get(term, 0.0)
                score += idf * term_freq * (k1 + 1) / (term_freq + length_norm)
            if score > 0:
                scores.append((score, doc_index))
        scores.sort(key=lambda pair: -pair[0])
        return [(self._ids[doc_index], score) for score, doc_index in scores[:top_k]]
