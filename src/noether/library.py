"""The high-level API. One engine, two faces.

The CLI and the MCP server both call this module and nothing lower. That keeps
them two presentations of one behaviour rather than two implementations that
drift apart -- which matters more than usual here, because a guarantee enforced
on one path and not the other is not a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answer.pipeline import DraftClaim, gather_evidence, submit
from .answer.records import AnswerRecord, EvidencePack
from .compute.symbolic import parse_equation
from .compute.units import check_equation
from .config import Config
from .index.retrieve import Retriever
from .index.store import Store
from .ingest.bibliography import BibEntry
from .ingest.document import Document
from .ingest.latex import LatexIngestError, parse_document
from .netclient import NetClient
from .sources.arxiv import ArxivClient, ArxivError
from .verify.citations import AuditFinding, CitationResolver


@dataclass
class IngestResult:
    """What happened when a paper was taken into the corpus."""

    paper_id: str
    title: str
    ok: bool
    equations: int = 0
    numbered_equations: int = 0
    figures: int = 0
    citations: int = 0
    references: int = 0
    macros: int = 0
    chars: int = 0
    license: str | None = None
    error: str = ""

    def summary(self) -> str:
        if not self.ok:
            return f"{self.paper_id}: {self.error}"
        return (
            f"{self.paper_id}: {self.chars} chars, {self.numbered_equations} numbered "
            f"equations, {self.figures} figures, {self.references} references"
        )


class Library:
    """A researcher's corpus, and everything that can be done to it."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self.config.ensure_dirs()
        self.store = Store(self.config.db_path)
        self.net = NetClient(self.config)
        self._retriever: Retriever | None = None

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(self.store)
        return self._retriever

    # ---- ingest --------------------------------------------------------

    def ingest_arxiv(self, arxiv_id: str, refresh: bool = False) -> IngestResult:
        """Fetch, parse and store one arXiv paper."""
        client = ArxivClient(self.net, self.config)

        try:
            meta = client.get(arxiv_id)
        except ArxivError as exc:
            return IngestResult(arxiv_id, "", False, error=str(exc))

        paper_id = f"arXiv:{meta.arxiv_id}"
        if self.store.has_paper(paper_id) and not refresh:
            row = self.store.paper_row(paper_id) or {}
            return IngestResult(
                paper_id, row.get("title", meta.title), True,
                chars=len(row.get("text", "")), macros=row.get("macro_count", 0),
                error="already in corpus (pass refresh to re-ingest)",
            )

        unpacked = self.config.cache_dir / "unpacked" / meta.arxiv_id.replace("/", "_")
        try:
            client.unpack_source(meta.arxiv_id, unpacked)
        except ArxivError as exc:
            # A PDF-only submission is expected, not exceptional. Say which.
            return IngestResult(paper_id, meta.title, False, error=f"no LaTeX source: {exc}")

        try:
            document, references = parse_document(
                unpacked, paper_id, title=meta.title, license=meta.license
            )
        except LatexIngestError as exc:
            return IngestResult(paper_id, meta.title, False, error=str(exc))

        # arXiv metadata is authoritative for authors; the LaTeX author block is
        # full of affiliation markup that parses into noise.
        document.authors = meta.authors or document.authors
        self.store.add_document(document, references)
        self._retriever = None  # the corpus changed; drop cached vectors

        return IngestResult(
            paper_id=paper_id,
            title=document.title,
            ok=True,
            equations=len(document.equations),
            numbered_equations=sum(1 for e in document.equations if e.is_numbered),
            figures=len(document.figures),
            citations=len(document.citations),
            references=len(references),
            macros=document.macro_count,
            chars=len(document.text),
            license=meta.license,
        )

    def ingest_directory(self, root: Path, paper_id: str) -> IngestResult:
        """Ingest an already-unpacked LaTeX source tree."""
        try:
            document, references = parse_document(root, paper_id)
        except LatexIngestError as exc:
            return IngestResult(paper_id, "", False, error=str(exc))
        self.store.add_document(document, references)
        self._retriever = None
        return IngestResult(
            paper_id=paper_id, title=document.title, ok=True,
            equations=len(document.equations),
            numbered_equations=sum(1 for e in document.equations if e.is_numbered),
            figures=len(document.figures), citations=len(document.citations),
            references=len(references), macros=document.macro_count,
            chars=len(document.text),
        )

    # ---- search and answer ---------------------------------------------

    def search_arxiv(self, query: str, limit: int = 10) -> list[Any]:
        return ArxivClient(self.net, self.config).search(query, max_results=limit)

    def search_corpus(self, query: str, limit: int = 10) -> list[Any]:
        return self.retriever.search(query, limit=limit)

    def evidence(self, question: str, limit: int = 12) -> EvidencePack:
        return gather_evidence(self.store, question, limit=limit, retriever=self.retriever)

    def submit_answer(
        self,
        pack: EvidencePack,
        claims: list[DraftClaim],
        equations: list[dict[str, Any]] | None = None,
        references: dict[str, bool] | None = None,
    ) -> AnswerRecord:
        return submit(
            self.store, pack, claims, equations, references, self.config.answers_dir
        )

    # ---- equations -----------------------------------------------------

    def equations(self, paper_id: str) -> list[dict[str, Any]]:
        return self.store.equations(paper_id)

    def equation(self, paper_id: str, number: int) -> dict[str, Any] | None:
        """The equation the *authors* numbered ``number``."""
        return next(
            (e for e in self.store.equations(paper_id) if e["number"] == number), None
        )

    def check_equation(self, latex: str, units: dict[str, str] | None = None) -> dict[str, Any]:
        parsed = parse_equation(latex)
        check = check_equation(parsed, units)
        return {
            "latex": latex,
            "normalised": parsed.normalised,
            "parsed": parsed.ok,
            "error": parsed.error,
            "lhs": str(parsed.lhs) if parsed.lhs is not None else None,
            "rhs": str(parsed.rhs) if parsed.rhs is not None else None,
            "relation": parsed.relation,
            "symbols": parsed.symbols,
            "dimensions": check.verdict.value,
            "detail": check.detail,
            "hint": check.hint() if check.unknown_symbols else "",
            "assumed_units": check.assumed,
            "notes": parsed.notes,
        }

    # ---- audit ---------------------------------------------------------

    def audit_paper(self, paper_id: str, offline: bool = False) -> list[AuditFinding]:
        """Resolve every reference of a stored paper and report the problems."""
        rows = self.store.references(paper_id)
        entries = [
            BibEntry(
                key=r["cite_key"], raw=r["raw"], text=r["text"], doi=r["doi"],
                arxiv_id=r["arxiv_id"], year=r["year"], title=r["title"], origin=r["origin"],
            )
            for r in rows
        ]
        cited = {u["cite_key"] for u in self.store.citation_uses(paper_id)}
        resolver = CitationResolver(self.store, self.net, offline=offline)
        return resolver.audit(entries, cited_keys=cited)

    def audit_document(self, document: Document, references: list[BibEntry], offline: bool = False) -> list[AuditFinding]:
        resolver = CitationResolver(self.store, self.net, offline=offline)
        return resolver.audit(references, cited_keys=document.cited_keys())

    # ---- corpus --------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return self.store.stats()

    def papers(self) -> list[dict[str, Any]]:
        return [
            row
            for row in (self.store.paper_row(pid) for pid in self.store.paper_ids())
            if row
        ]

    def close(self) -> None:
        self.net.close()
