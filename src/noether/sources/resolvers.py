"""Identifier resolution against Crossref, OpenAlex and INSPIRE-HEP.

Guarantee 1 -- no unresolvable citation ever leaves the system -- is enforced
here. A reference is resolvable when some provider returns a record for it.

Two kinds of lookup, with very different confidence:

* **Identifier lookup** (DOI, arXiv id) is exact. Either the record exists or it
  does not, and a 404 is a definite negative.
* **Bibliographic lookup** from free text is a similarity judgement. Most
  physics references are free text with no DOI, so this is the common path, and
  it must return a *score* the caller can threshold rather than a bare boolean.
  A confident-looking wrong match is the failure mode we are trying to avoid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from ..config import Config
from ..netclient import NetClient
from .registry import get_source

#: Retained for identifier lookups only. Free-text matching moved to
#: noether.verify.matching, which corroborates year/author/volume/page rather
#: than comparing a whole reference string to a title.
MATCH_THRESHOLD = 0.82

_NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")


@dataclass
class Resolution:
    """The outcome of trying to resolve one reference."""

    identifier: str
    kind: str
    resolved: bool
    source: str | None = None
    title: str | None = None
    #: 1.0 for an exact identifier hit; a similarity score for a text match.
    confidence: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def trustworthy(self) -> bool:
        return self.resolved and self.confidence >= MATCH_THRESHOLD


def normalise_title(title: str) -> str:
    """Lowercase, strip punctuation and collapse space, for comparison only."""
    return re.sub(r"\s+", " ", _NON_WORD_RE.sub(" ", title.lower())).strip()


def title_similarity(a: str, b: str) -> float:
    """Similarity of two titles in [0, 1]."""
    left, right = normalise_title(a), normalise_title(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


class CrossrefClient:
    """DOI resolution and bibliographic search."""

    def __init__(self, net: NetClient | None = None, config: Config | None = None) -> None:
        self.config = config or (net.config if net else Config.from_env())
        self.net = net or NetClient(self.config)
        self.spec = get_source("crossref")
        self.net.limiter.configure("api.crossref.org", self.spec.min_interval_s)

    def _params(self, extra: dict[str, Any]) -> dict[str, Any]:
        # A mailto puts us in Crossref's faster, more reliable polite pool.
        if self.config.mailto:
            extra = {**extra, "mailto": self.config.mailto}
        return extra

    def resolve_doi(self, doi: str) -> Resolution:
        response = self.net.get(
            "crossref", f"{self.spec.base_url}/works/{doi}", params=self._params({})
        )
        if response.status == 404:
            return Resolution(doi, "doi", False, "crossref", reason="Crossref has no such DOI")
        if response.status != 200:
            return Resolution(
                doi, "doi", False, "crossref", reason=f"Crossref returned HTTP {response.status}"
            )
        message = _json(response.text).get("message", {})
        return Resolution(
            identifier=doi,
            kind="doi",
            resolved=True,
            source="crossref",
            title=_first(message.get("title")),
            confidence=1.0,
            payload={
                "year": _crossref_year(message),
                "container": _first(message.get("container-title")),
                "type": message.get("type"),
            },
        )

    def search(self, text: str, rows: int = 3) -> list[dict[str, Any]]:
        response = self.net.get(
            "crossref",
            f"{self.spec.base_url}/works",
            params=self._params({"query.bibliographic": text[:400], "rows": rows}),
        )
        if response.status != 200:
            return []
        return _json(response.text).get("message", {}).get("items", [])


class OpenAlexClient:
    """Fallback resolution and citation-graph metadata."""

    def __init__(self, net: NetClient | None = None, config: Config | None = None) -> None:
        self.config = config or (net.config if net else Config.from_env())
        self.net = net or NetClient(self.config)
        self.spec = get_source("openalex")
        self.net.limiter.configure("api.openalex.org", self.spec.min_interval_s)

    def _params(self, extra: dict[str, Any]) -> dict[str, Any]:
        if self.config.mailto:
            extra = {**extra, "mailto": self.config.mailto}
        return extra

    def resolve_doi(self, doi: str) -> Resolution:
        response = self.net.get(
            "openalex", f"{self.spec.base_url}/works/doi:{doi}", params=self._params({})
        )
        if response.status != 200:
            return Resolution(
                doi, "doi", False, "openalex", reason=f"OpenAlex returned HTTP {response.status}"
            )
        data = _json(response.text)
        return Resolution(
            identifier=doi,
            kind="doi",
            resolved=True,
            source="openalex",
            title=data.get("title"),
            confidence=1.0,
            payload={"year": data.get("publication_year"), "cited_by": data.get("cited_by_count")},
        )

    def search(self, text: str, rows: int = 3) -> list[dict[str, Any]]:
        response = self.net.get(
            "openalex",
            f"{self.spec.base_url}/works",
            params=self._params({"search": text[:350], "per-page": rows}),
        )
        if response.status != 200:
            return []
        return _json(response.text).get("results", [])


class InspireClient:
    """INSPIRE-HEP: the authoritative index for high-energy physics."""

    def __init__(self, net: NetClient | None = None, config: Config | None = None) -> None:
        self.config = config or (net.config if net else Config.from_env())
        self.net = net or NetClient(self.config)
        self.spec = get_source("inspire")
        self.net.limiter.configure("inspirehep.net", self.spec.min_interval_s)

    def search(self, query: str, size: int = 3) -> list[dict[str, Any]]:
        response = self.net.get(
            "inspire",
            f"{self.spec.base_url}/literature",
            params={
                "q": query,
                "size": size,
                # authors and publication_info carry the fields the matcher
                # corroborates on; without them every INSPIRE hit scores low.
                "fields": "titles,authors,arxiv_eprints,dois,publication_info",
            },
        )
        if response.status != 200:
            return []
        return _json(response.text).get("hits", {}).get("hits", [])

    def search_journal(self, journal: str, volume: str, page: str, size: int = 3) -> list[dict[str, Any]]:
        """Look a reference up by its journal coordinate.

        INSPIRE indexes physics literature by journal, volume and page, which is
        exactly the information a physics reference states even when it omits
        the title. This is a lookup, not a similarity search, so it answers the
        question a general bibliographic search cannot.
        """
        return self.search(f"j {journal},{volume},{page}", size=size)

    def resolve_arxiv(self, arxiv_id: str) -> Resolution:
        hits = self.search(f"arxiv:{arxiv_id}", size=1)
        if not hits:
            return Resolution(
                arxiv_id, "arxiv", False, "inspire", reason="INSPIRE has no record of this e-print"
            )
        metadata = hits[0].get("metadata", {})
        titles = metadata.get("titles") or [{}]
        return Resolution(
            identifier=arxiv_id,
            kind="arxiv",
            resolved=True,
            source="inspire",
            title=titles[0].get("title"),
            confidence=1.0,
            payload={"control_number": metadata.get("control_number")},
        )


def best_match(text: str, candidates: list[dict[str, Any]], extract: Any) -> tuple[float, dict[str, Any] | None]:
    """Pick the candidate whose title best matches ``text``.

    Returns the score alongside the record so the caller can apply its own
    threshold rather than inheriting a hidden one.
    """
    best_score = 0.0
    best_record: dict[str, Any] | None = None
    for candidate in candidates:
        title = extract(candidate)
        if not title:
            continue
        score = title_similarity(text, title)
        if score > best_score:
            best_score, best_record = score, candidate
    return best_score, best_record


def crossref_title(item: dict[str, Any]) -> str:
    return _first(item.get("title")) or ""


def openalex_title(item: dict[str, Any]) -> str:
    return item.get("title") or item.get("display_name") or ""


def inspire_title(item: dict[str, Any]) -> str:
    titles = item.get("metadata", {}).get("titles") or [{}]
    return titles[0].get("title", "")


def _first(value: Any) -> str | None:
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value else None


def _crossref_year(message: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        parts = message.get(key, {}).get("date-parts")
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def _json(text: str) -> dict[str, Any]:
    import json

    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}
