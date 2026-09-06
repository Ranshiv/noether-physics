"""Reference resolution -- the enforcement point for guarantee 1.

A reference is resolved by trying, in order of decreasing certainty:

1. **arXiv id** -- exact, and the most common identifier in physics.
2. **DOI** -- exact, via Crossref then OpenAlex.
3. **Free text** -- a similarity judgement against Crossref, OpenAlex and
   INSPIRE. This is the common case for ``thebibliography`` entries, and it is
   where a wrong answer does real damage, so a match below
   ``MATCH_THRESHOLD`` is reported as *unresolved* rather than as a weak hit.

Results are cached in the store, because an identifier that resolved once should
not cost another provider request.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..index.store import Store
from ..ingest.bibliography import BibEntry
from ..netclient import NetClient
from ..sources.arxiv import ArxivClient, ArxivError
from ..sources.resolvers import (
    CrossrefClient,
    InspireClient,
    OpenAlexClient,
    Resolution,
)
from .matching import (
    MATCH_THRESHOLD,
    best_candidate,
    extract_fields,
    from_crossref,
    from_inspire,
    from_openalex,
    title_similarity,
)


@dataclass
class AuditFinding:
    """One problem found in a reference list."""

    key: str
    #: unresolvable | unchecked | mismatch | cited-unread | no-identifier | undeclared
    kind: str
    detail: str
    resolution: Resolution | None = None

    @property
    def severity(self) -> str:
        """Only a positive finding is an error.

        ``unchecked`` is deliberately a warning. Offline mode does not consult
        any provider, so reporting its references as ``unresolvable`` would state
        that no record exists when in fact none was sought -- the same collapse
        of "did not check" into "failed" that the dimensional analysis is careful
        to avoid.
        """
        return "error" if self.kind == "unresolvable" else "warning"


class CitationResolver:
    """Resolves references, caching outcomes in the store."""

    def __init__(
        self,
        store: Store,
        net: NetClient | None = None,
        offline: bool = False,
    ) -> None:
        self.store = store
        self.net = net or NetClient()
        # Offline mode resolves only from cache. Used by tests and by anyone
        # working without a network; it never silently pretends to have checked.
        self.offline = offline
        self._arxiv: ArxivClient | None = None
        self._crossref: CrossrefClient | None = None
        self._openalex: OpenAlexClient | None = None
        self._inspire: InspireClient | None = None

    # Clients are built lazily so an offline run makes no connections at all.
    @property
    def arxiv(self) -> ArxivClient:
        if self._arxiv is None:
            self._arxiv = ArxivClient(self.net)
        return self._arxiv

    @property
    def crossref(self) -> CrossrefClient:
        if self._crossref is None:
            self._crossref = CrossrefClient(self.net)
        return self._crossref

    @property
    def openalex(self) -> OpenAlexClient:
        if self._openalex is None:
            self._openalex = OpenAlexClient(self.net)
        return self._openalex

    @property
    def inspire(self) -> InspireClient:
        if self._inspire is None:
            self._inspire = InspireClient(self.net)
        return self._inspire

    # ---- single reference ----------------------------------------------

    def resolve(self, entry: BibEntry, use_cache: bool = True) -> Resolution:
        identifier = entry.arxiv_id or entry.doi or entry.key

        if use_cache:
            cached = self.store.cached_resolution(identifier)
            if cached:
                return Resolution(
                    identifier=identifier,
                    kind=cached["kind"],
                    resolved=cached["resolved"],
                    source=cached["source"],
                    title=cached["title"],
                    confidence=float(cached["payload"].get("confidence", 1.0)),
                    payload=cached["payload"],
                    reason=cached["payload"].get("reason", ""),
                )

        if self.offline:
            return Resolution(
                identifier, "unknown", False, reason="offline: no cached resolution"
            )

        resolution = self._resolve_live(entry, identifier)
        self.store.record_resolution(
            identifier=identifier,
            kind=resolution.kind,
            resolved=resolution.resolved,
            source=resolution.source,
            title=resolution.title,
            payload={**resolution.payload, "confidence": resolution.confidence,
                     "reason": resolution.reason},
        )
        return resolution

    def _resolve_live(self, entry: BibEntry, identifier: str) -> Resolution:
        if entry.arxiv_id:
            try:
                meta = self.arxiv.get(entry.arxiv_id)
                return Resolution(
                    identifier, "arxiv", True, "arxiv", meta.title, 1.0,
                    {"year": meta.published.year if meta.published else None},
                )
            except ArxivError as exc:
                return Resolution(identifier, "arxiv", False, "arxiv", reason=str(exc))

        if entry.doi:
            resolution = self.crossref.resolve_doi(entry.doi)
            if resolution.resolved:
                return resolution
            fallback = self.openalex.resolve_doi(entry.doi)
            # Crossref's 404 is authoritative for a DOI; report its reason.
            return fallback if fallback.resolved else resolution

        return self._resolve_text(entry, identifier)

    def _resolve_text(self, entry: BibEntry, identifier: str) -> Resolution:
        """Match a free-text reference by corroborating the fields it carries.

        Not by title similarity: most physics references state no title, and an
        earlier title-only matcher resolved 3 of 37 references on a real paper.
        See :mod:`noether.verify.matching`.
        """
        text = entry.text or entry.raw
        if not text.strip():
            return Resolution(identifier, "text", False, reason="reference has no text to match")

        fields = extract_fields(text, known_title=entry.title)
        if fields.is_thin:
            return Resolution(
                identifier, "text", False,
                reason="reference states neither a year nor a recognisable author",
            )

        # A journal coordinate is an exact lookup, so try it first. It answers
        # the case general bibliographic search cannot: a reference that states
        # journal, volume and page but no title.
        if fields.journal and fields.volume and fields.page:
            try:
                hits = self.inspire.search_journal(fields.journal, fields.volume, fields.page)
            except Exception:
                hits = []
            if hits:
                candidate = from_inspire(hits[0])
                return Resolution(
                    identifier=identifier,
                    kind="text",
                    resolved=True,
                    source="inspire",
                    title=candidate.title,
                    confidence=0.95,
                    payload={"doi": candidate.doi,
                             "matched_by": f"journal coordinate {fields.journal},"
                                           f"{fields.volume},{fields.page}"},
                )

        attempts = (
            ("crossref", self.crossref.search, from_crossref),
            ("openalex", self.openalex.search, from_openalex),
            ("inspire", self.inspire.search, from_inspire),
        )

        best = Resolution(
            identifier, "text", False,
            reason=f"no provider corroborated above {MATCH_THRESHOLD:.2f}",
        )
        for source, search, adapt in attempts:
            try:
                records = search(text)
            except Exception:
                continue
            corroboration, candidate = best_candidate(fields, [adapt(r) for r in records])
            if candidate is None:
                continue
            if corroboration.score > best.confidence:
                best = Resolution(
                    identifier=identifier,
                    kind="text",
                    resolved=corroboration.score >= MATCH_THRESHOLD,
                    source=source,
                    title=candidate.title,
                    confidence=corroboration.score,
                    payload={"doi": candidate.doi, "corroboration": corroboration.explain()},
                    reason=(
                        ""
                        if corroboration.score >= MATCH_THRESHOLD
                        else f"best corroboration {corroboration.score:.2f} "
                             f"({corroboration.explain()}), below {MATCH_THRESHOLD:.2f}"
                    ),
                )
            if best.resolved:
                break
        return best

    # ---- whole reference lists -----------------------------------------

    def audit(
        self,
        entries: list[BibEntry],
        cited_keys: set[str] | None = None,
        used_keys: set[str] | None = None,
    ) -> list[AuditFinding]:
        """Check a reference list end to end.

        ``cited_keys`` are the keys the document actually cites; ``used_keys``
        those whose surrounding sentence shows real engagement. A key present in
        the bibliography but absent from the text is the 'cited but never read'
        pattern that citation audits exist to surface.
        """
        findings: list[AuditFinding] = []

        for entry in entries:
            resolution = self.resolve(entry)

            if not resolution.resolved:
                # Distinguish "no provider has this" from "we never asked".
                unchecked = resolution.reason.startswith("offline:")
                findings.append(
                    AuditFinding(
                        entry.key,
                        "unchecked" if unchecked else "unresolvable",
                        resolution.reason or "no provider could resolve this reference",
                        resolution,
                    )
                )
                continue

            # A resolved record whose title disagrees with what the author wrote
            # is the "right title, wrong edition/DOI" case.
            if entry.title and resolution.title:
                similarity = title_similarity(entry.title, resolution.title)
                if similarity < MATCH_THRESHOLD:
                    findings.append(
                        AuditFinding(
                            entry.key, "mismatch",
                            f"bibliography says {entry.title!r}; "
                            f"{resolution.source} resolves to {resolution.title!r}",
                            resolution,
                        )
                    )

            if not entry.has_identifier:
                findings.append(
                    AuditFinding(
                        entry.key, "no-identifier",
                        f"matched by text similarity ({resolution.confidence:.2f}), "
                        "not by DOI or arXiv id",
                        resolution,
                    )
                )

        if cited_keys is not None:
            declared = {e.key for e in entries}
            for key in sorted(declared - cited_keys):
                findings.append(
                    AuditFinding(key, "cited-unread", "in the bibliography but never cited in the text")
                )

        if used_keys is not None:
            for key in sorted((cited_keys or set()) - used_keys):
                findings.append(
                    AuditFinding(key, "cited-unread", "cited without any discussion in the surrounding text")
                )

        return findings
