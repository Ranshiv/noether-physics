"""Hybrid retrieval: BM25 and vector similarity, fused by reciprocal rank.

Reciprocal rank fusion rather than score blending, because BM25 scores and
cosine similarities live on incomparable scales and any weighting between them
would be a number invented to look principled. RRF only needs each retriever's
ordering, which is the part each one is actually good at.

Every hit carries the span it came from, so a retrieval result converts straight
into an anchor. That is the whole point of retrieving chunks rather than papers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ingest.document import Anchor, Span
from .embed import Embedder, cosine, default_embedder
from .store import Chunk, Store

#: RRF damping. 60 is the value from the original formulation and behaves well
#: when one retriever is much stronger than the other, which is our case.
RRF_K = 60


@dataclass
class Hit:
    """One retrieved passage, with the evidence for why it surfaced."""

    chunk: Chunk
    score: float
    #: Per-retriever rank, 1-based. Absent means that retriever did not return it.
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def paper_id(self) -> str:
        return self.chunk.paper_id

    @property
    def span(self) -> Span:
        return self.chunk.span

    def to_anchor(self) -> Anchor:
        """Convert directly into a citable anchor."""
        return Anchor(
            paper_id=self.chunk.paper_id,
            section_path=self.chunk.section_path,
            span=self.chunk.span,
            quote=self.chunk.text,
        )


class Retriever:
    """Hybrid search over the corpus."""

    def __init__(self, store: Store, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or default_embedder()
        self._vectors: dict[int, list[float]] | None = None

    # ---- vector half ---------------------------------------------------

    def _ensure_vectors(self) -> dict[int, list[float]]:
        """Embed the corpus lazily and keep it in memory.

        At the scale one researcher's library reaches, recomputing on demand is
        cheaper and far simpler than maintaining a persisted index that can fall
        out of step with the store.
        """
        if self._vectors is None:
            self._vectors = {
                chunk.chunk_id: self.embedder.encode(chunk.text)
                for chunk in self.store.chunks()
            }
        return self._vectors

    def invalidate(self) -> None:
        """Drop cached vectors. Call after ingesting new papers."""
        self._vectors = None

    def search_vector(self, query: str, limit: int = 20) -> list[tuple[Chunk, float]]:
        vectors = self._ensure_vectors()
        if not vectors:
            return []
        query_vector = self.embedder.encode(query)
        scored = [
            (chunk_id, cosine(query_vector, vector))
            for chunk_id, vector in vectors.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        hits: list[tuple[Chunk, float]] = []
        for chunk_id, score in scored[:limit]:
            if score <= 0.0:
                break
            chunk = self.store.chunk(chunk_id)
            if chunk:
                hits.append((chunk, score))
        return hits

    # ---- fusion --------------------------------------------------------

    def search(self, query: str, limit: int = 10, pool: int = 40) -> list[Hit]:
        """Run both retrievers over a wider pool, then fuse."""
        rankings = {
            "bm25": self.store.search_bm25(query, limit=pool),
            "vector": self.search_vector(query, limit=pool),
        }

        fused: dict[int, Hit] = {}
        for retriever, results in rankings.items():
            for rank, (chunk, _score) in enumerate(results, start=1):
                hit = fused.get(chunk.chunk_id)
                if hit is None:
                    hit = Hit(chunk=chunk, score=0.0)
                    fused[chunk.chunk_id] = hit
                hit.ranks[retriever] = rank
                hit.score += 1.0 / (RRF_K + rank)

        ordered = sorted(fused.values(), key=lambda h: h.score, reverse=True)
        return ordered[:limit]

    def search_papers(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Fuse to paper level, for 'which papers matter here' questions."""
        totals: dict[str, float] = {}
        for hit in self.search(query, limit=limit * 5):
            totals[hit.paper_id] = totals.get(hit.paper_id, 0.0) + hit.score
        return sorted(totals.items(), key=lambda item: item[1], reverse=True)[:limit]

    def expand_by_citation(self, paper_ids: list[str], limit: int = 5) -> list[str]:
        """Papers in the corpus that the seeds cite, or that cite the seeds.

        Citation adjacency finds work that shares no vocabulary with the query
        but sits directly in its lineage -- the papers a reader would reach next.
        """
        neighbours: dict[str, int] = {}
        for paper_id in paper_ids:
            cited_keys = {use["cite_key"] for use in self.store.citation_uses(paper_id)}
            for other in self.store.paper_ids():
                if other == paper_id or other in paper_ids:
                    continue
                other_keys = {ref["cite_key"] for ref in self.store.references(other)}
                shared = len(cited_keys & other_keys)
                if shared:
                    neighbours[other] = neighbours.get(other, 0) + shared
        return [p for p, _ in sorted(neighbours.items(), key=lambda i: i[1], reverse=True)[:limit]]
