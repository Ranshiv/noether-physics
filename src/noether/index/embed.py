"""Vector representations for the semantic half of hybrid retrieval.

The default backend is a deterministic hashed bag of words with sublinear term
weighting -- no model download, no GPU, works offline, identical results on
every machine. It is genuinely weaker than a trained sentence encoder at
matching paraphrases, and `docs/DEFERRED.md` says so rather than implying
otherwise. It is here because BM25 does most of the work in a technical corpus
and a wrong-but-confident semantic score is worse than a modest honest one.

``Embedder`` is the seam: point ``NOETHER_EMBEDDER`` at another backend and the
retriever changes behaviour without changing shape.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Protocol

#: Vector width. Large enough that hash collisions are rare at corpus scale,
#: small enough that a few thousand chunks stay comfortably in memory.
DIMENSIONS = 2048

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Words carrying no discriminative weight in a physics corpus.
_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "then", "there", "these", "this", "to", "was", "were", "which", "with", "we", "our", "us"]
)


class Embedder(Protocol):
    """Anything that turns text into a fixed-width vector."""

    dimensions: int

    def encode(self, text: str) -> list[float]: ...


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed.

    LaTeX fragments tokenise badly by design: ``\\gamma_{\\rm n}`` becomes
    ``gamma``, ``rm``, ``n``. That is acceptable -- equations are retrieved
    through their surrounding prose, not through their own symbols.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _bucket(token: str, dimensions: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


class HashingEmbedder:
    """Deterministic hashed bag of words, L2-normalised.

    Sublinear term frequency (``1 + log tf``) keeps a word repeated twenty times
    from dominating a passage, which matters in papers where one symbol name
    recurs constantly.
    """

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        counts = Counter(tokenize(text))
        vector = [0.0] * self.dimensions
        if not counts:
            return vector

        for token, count in counts.items():
            vector[_bucket(token, self.dimensions)] += 1.0 + math.log(count)

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def encode_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors that are already normalised.

    Falls back to full normalisation when they are not, so a caller passing raw
    vectors gets a correct answer rather than a quietly inflated one.
    """
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    if abs(norm_a - 1.0) < 1e-9 and abs(norm_b - 1.0) < 1e-9:
        return dot
    return dot / (norm_a * norm_b)


def default_embedder() -> Embedder:
    return HashingEmbedder()
