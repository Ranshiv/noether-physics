"""Chunking for retrieval.

Two rules, both consequences of guarantee 2:

* A chunk never crosses an equation boundary. Splitting mid-derivation produces
  a passage that reads as a claim but is half of one.
* A chunk's span is always a real range in the paper's text, so a retrieval hit
  converts directly into an anchor without any further lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ingest.document import Document, Span

#: Target chunk size in characters. Paragraph-scale: large enough to carry an
#: argument, small enough that a quote drawn from it is checkable by eye.
TARGET_CHARS = 900
MIN_CHARS = 120


@dataclass(frozen=True)
class DocChunk:
    """A retrievable passage with its exact position."""

    paper_id: str
    section_path: str
    kind: str
    span: Span
    text: str


def chunk_document(document: Document, target: int = TARGET_CHARS) -> list[DocChunk]:
    """Group a document's blocks into retrievable chunks.

    Consecutive prose blocks in the same section merge up to ``target``.
    Equations stand alone, which is what keeps a chunk from straddling one.
    """
    chunks: list[DocChunk] = []
    pending: list[tuple[Span, str]] = []
    pending_section = ""

    def flush() -> None:
        nonlocal pending, pending_section
        if not pending:
            return
        start = pending[0][0].start
        end = pending[-1][0].end
        text = document.text[start:end]
        if text.strip():
            chunks.append(
                DocChunk(document.paper_id, pending_section, "paragraph", Span(start, end), text)
            )
        pending = []

    for block in document.blocks:
        if block.kind == "equation":
            flush()
            chunks.append(
                DocChunk(
                    document.paper_id, block.section_path, "equation", block.span, block.text
                )
            )
            continue

        if pending and (
            block.section_path != pending_section
            or block.span.end - pending[0][0].start > target
        ):
            flush()

        if not pending:
            pending_section = block.section_path
        pending.append((block.span, block.text))

    flush()

    # An abstract is worth retrieving on its own even when short.
    return [c for c in chunks if len(c.text) >= MIN_CHARS or c.kind != "paragraph"]
