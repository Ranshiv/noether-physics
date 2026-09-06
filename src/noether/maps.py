"""Literature maps and contradiction detection.

Contradiction detection here is deliberately modest, and it matters that the
limits are stated rather than implied. It finds passages that **discuss the same
quantity and disagree in direction** -- one reports an increase where another
reports a decrease, one an upper bound where another claims a detection. It does
not understand physics.

So every candidate is reported as *worth a human look*, never as a settled
conflict, and each one carries both spans so the reader can judge in seconds.
Two papers can disagree in wording and agree in fact -- different regimes,
different conventions, different definitions of the same symbol. Presenting that
as a contradiction would be its own kind of fabrication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .index.retrieve import Retriever
from .index.store import Chunk, Store

#: Direction words that oppose each other. A pair from opposite sides of one
#: row, discussing the same quantity, is what makes a candidate.
OPPOSING = [
    ({"increase", "increases", "increased", "increasing", "grows", "rises", "enhanced", "larger"},
     {"decrease", "decreases", "decreased", "decreasing", "falls", "drops", "suppressed", "smaller"}),
    ({"linear", "linearly"}, {"quadratic", "quadratically", "exponential", "exponentially"}),
    ({"observe", "observed", "detection", "detected", "measured"},
     {"upper bound", "no evidence", "null result", "constraint", "limit of"}),
    ({"agree", "agrees", "consistent", "confirms"},
     {"disagree", "disagrees", "inconsistent", "tension", "discrepancy", "contradicts"}),
    ({"markovian"}, {"non-markovian", "nonmarkovian"}),
]

_WORD_RE = re.compile(r"[a-z][a-z-]+")

#: Words too generic to establish that two passages discuss the same thing.
_STOP = frozenset(
    ["the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "have", "has", "been", "which", "their", "they", "our", "its", "can", "may", "not", "but", "all", "one", "two", "also", "than", "then", "when", "where", "while", "such", "these", "those", "there", "here", "into", "over", "under", "more", "most", "some", "any", "each"]
)


@dataclass
class Contradiction:
    """Two passages that may disagree. A prompt to look, not a verdict."""

    left: Chunk
    right: Chunk
    #: The shared subject matter that makes them comparable.
    shared_terms: list[str] = field(default_factory=list)
    #: The opposing direction words found.
    opposition: tuple[str, str] = ("", "")
    overlap: float = 0.0

    @property
    def papers(self) -> tuple[str, str]:
        return self.left.paper_id, self.right.paper_id

    def describe(self) -> str:
        return (
            f"{self.left.paper_id} says '{self.opposition[0]}' where "
            f"{self.right.paper_id} says '{self.opposition[1]}', both discussing "
            + ", ".join(self.shared_terms[:4])
        )


@dataclass
class Cluster:
    """A line of work: papers that share vocabulary and citations."""

    label: str
    paper_ids: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)


@dataclass
class LiteratureMap:
    """What the corpus looks like around a question."""

    query: str
    clusters: list[Cluster] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    #: paper_id -> the papers it shares references with, and how many.
    citation_links: dict[str, dict[str, int]] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{len(self.clusters)} lines of work, "
            f"{len(self.contradictions)} passages worth a second look"
        )


def _terms(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 3}


def find_contradictions(
    chunks: list[Chunk], min_overlap: float = 0.12, limit: int = 20
) -> list[Contradiction]:
    """Find passages that discuss the same thing and pull in opposite directions.

    Requires both a shared vocabulary (so the passages are about the same
    quantity) and an opposing pair (so they actually differ). Either alone
    produces noise: shared vocabulary is just topicality, and an opposing word
    pair across unrelated passages means nothing.
    """
    candidates: list[Contradiction] = []
    profiles = [(chunk, _terms(chunk.text), chunk.text.lower()) for chunk in chunks]

    for i, (left, left_terms, left_text) in enumerate(profiles):
        for right, right_terms, right_text in profiles[i + 1 :]:
            # Passages from one paper disagreeing with themselves is usually a
            # narrative contrast, not a contradiction between works.
            if left.paper_id == right.paper_id:
                continue

            shared = left_terms & right_terms
            union = left_terms | right_terms
            if not union:
                continue
            overlap = len(shared) / len(union)
            if overlap < min_overlap:
                continue

            opposition = _opposing_pair(left_text, right_text)
            if opposition is None:
                continue

            candidates.append(
                Contradiction(
                    left=left,
                    right=right,
                    shared_terms=sorted(shared)[:8],
                    opposition=opposition,
                    overlap=round(overlap, 3),
                )
            )

    candidates.sort(key=lambda c: c.overlap, reverse=True)
    return candidates[:limit]


def _opposing_pair(left: str, right: str) -> tuple[str, str] | None:
    """Return the first opposing word pair found across two passages."""
    for positive, negative in OPPOSING:
        left_hit = next((w for w in positive if w in left), None)
        right_hit = next((w for w in negative if w in right), None)
        if left_hit and right_hit:
            return (left_hit, right_hit)
        left_hit = next((w for w in negative if w in left), None)
        right_hit = next((w for w in positive if w in right), None)
        if left_hit and right_hit:
            return (left_hit, right_hit)
    return None


def cluster_papers(store: Store, paper_ids: list[str], max_clusters: int = 5) -> list[Cluster]:
    """Group papers into lines of work by shared vocabulary.

    Single-link agglomeration on title and abstract terms. Crude, and labelled as
    such: it separates obviously different topics and will not distinguish two
    neighbouring sub-fields.
    """
    profiles: dict[str, set[str]] = {}
    for paper_id in paper_ids:
        row = store.paper_row(paper_id)
        if row:
            profiles[paper_id] = _terms(f"{row['title']} {row['abstract']}")

    clusters: list[Cluster] = []
    unassigned = dict(profiles)

    while unassigned and len(clusters) < max_clusters:
        seed_id, seed_terms = unassigned.popitem()
        members = [seed_id]
        terms = set(seed_terms)

        for other_id, other_terms in list(unassigned.items()):
            union = terms | other_terms
            if union and len(terms & other_terms) / len(union) >= 0.10:
                members.append(other_id)
                terms |= other_terms
                del unassigned[other_id]

        common = sorted(seed_terms & terms, key=len, reverse=True)[:5] or sorted(seed_terms)[:5]
        clusters.append(Cluster(label=", ".join(common[:3]), paper_ids=members, terms=common))

    # Anything left over is its own line of work, not silently dropped.
    for paper_id in unassigned:
        clusters.append(Cluster(label=paper_id, paper_ids=[paper_id]))

    return clusters


def citation_links(store: Store, paper_ids: list[str]) -> dict[str, dict[str, int]]:
    """How many references each pair of papers shares."""
    keys = {p: {r["cite_key"] for r in store.references(p)} for p in paper_ids}
    links: dict[str, dict[str, int]] = {}
    for paper_id, own in keys.items():
        neighbours = {
            other: len(own & other_keys)
            for other, other_keys in keys.items()
            if other != paper_id and own & other_keys
        }
        if neighbours:
            links[paper_id] = dict(sorted(neighbours.items(), key=lambda i: i[1], reverse=True))
    return links


def build_map(
    store: Store, query: str, limit: int = 40, retriever: Retriever | None = None
) -> LiteratureMap:
    """Retrieve around a question, then cluster and look for disagreement."""
    retriever = retriever or Retriever(store)
    hits = retriever.search(query, limit=limit)
    chunks = [hit.chunk for hit in hits]
    paper_ids = list(dict.fromkeys(chunk.paper_id for chunk in chunks))

    return LiteratureMap(
        query=query,
        clusters=cluster_papers(store, paper_ids),
        contradictions=find_contradictions(chunks),
        citation_links=citation_links(store, paper_ids),
    )
