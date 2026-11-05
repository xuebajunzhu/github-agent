"""Pure-Python hashing TF-IDF embedder: the offline, dependency-free fallback.

Tokens are hashed into a fixed number of buckets (stable across runs via
SHA-256, unlike Python's salted hash()) with sublinear term frequency and L2
normalization. Deterministic and instant, but quality is below real embedding
models -- enable it explicitly via EMBEDDER_PROVIDER=tfidf when no better
option is available.

Chinese text has no spaces, so CJK runs are split into character bigrams
(see app.utils.tokenize): without this a whole Chinese sentence collapses
into a single bucket and the fallback path returns near-random recommendations
for the project's primary query language.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter

from app.embedders.base import Embedder
from app.utils.tokenize import tokenize as _tokenize  # noqa: F401  # re-exported


class HashingTfidfEmbedder(Embedder):
    name = "tfidf"

    def __init__(self, dimension: int = 384):
        self.dimension = max(16, dimension)

    @property
    def space_id(self) -> str:
        return f"{self.name}-{self.dimension}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        counts = Counter(_tokenize(text))
        for token, count in counts.items():
            bucket = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
            index = bucket % self.dimension
            vector[index] += 1.0 + math.log(count)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector
