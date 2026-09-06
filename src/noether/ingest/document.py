"""The typed document model every ingested paper becomes.

Char offsets are the load-bearing detail. Guarantee 2 -- every claim anchored to
a verbatim span -- is only checkable if we can point at exact character ranges in
text we still hold. So every block records where it came from, and
``Document.quote()`` is the one way to turn an anchor back into text.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["paragraph", "equation", "theorem", "caption", "abstract"]


@dataclass(frozen=True)
class Span:
    """A character range in a document's flattened text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Anchor:
    """A citable pointer into one paper. This is what every claim carries."""

    paper_id: str
    section_path: str
    span: Span
    #: The exact text at that span when the anchor was created. Stored so a
    #: verifier can detect drift without refetching the source.
    quote: str

    def key(self) -> str:
        return f"{self.paper_id}#{self.span.start}-{self.span.end}"


@dataclass
class Equation:
    """A display equation, ideally with the number the authors gave it."""

    latex: str
    span: Span
    #: The LaTeX ``\label{}``, when present. This is how "equation 14" resolves
    #: to the equation the *authors* numbered 14 rather than the 14th we found.
    label: str | None = None
    #: Sequential number assigned by the document, 1-based, numbered equations only.
    number: int | None = None
    environment: str = "equation"
    section_path: str = ""

    @property
    def is_numbered(self) -> bool:
        return self.number is not None


@dataclass
class Figure:
    """A figure, its caption, and the image files the source shipped."""

    label: str | None
    caption: str
    span: Span
    #: Paths relative to the unpacked source root, from ``\includegraphics``.
    graphics: list[str] = field(default_factory=list)
    number: int | None = None
    section_path: str = ""


@dataclass
class CitationUse:
    r"""One ``\cite`` occurrence, with the sentence that surrounds it.

    The surrounding sentence is what makes "cited but unread" detectable and
    what lets a reader see *how* a work was used, not merely that it was listed.
    """

    key: str
    span: Span
    context: str
    section_path: str = ""


@dataclass
class Block:
    """A run of text with a known position."""

    kind: BlockKind
    text: str
    span: Span
    section_path: str = ""


@dataclass
class Section:
    """A section node. Nesting mirrors the paper's own structure."""

    title: str
    level: int
    span: Span
    path: str
    children: list[Section] = field(default_factory=list)

    def walk(self) -> Iterator[Section]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class Document:
    """One ingested paper.

    ``text`` is the flattened plain-text rendering that every span indexes into.
    It is the single source of truth for anchoring: if a quote is not literally
    present here, the claim carrying it does not ship.
    """

    paper_id: str
    title: str
    text: str
    #: Where the content came from: "latex" when we had arXiv source, "pdf" when
    #: we had to fall back to parsing. Readers should weight these differently.
    provenance: Literal["latex", "pdf"] = "latex"
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    citations: list[CitationUse] = field(default_factory=list)
    #: arXiv/journal licence string. Gates whether figures may be re-exported.
    license: str | None = None
    #: How many macros the source defined (newcommand/def). Reported because a
    #: high count is the usual reason equation LaTeX fails to reach SymPy intact.
    macro_count: int = 0

    # ---- anchoring -----------------------------------------------------

    def quote(self, span: Span) -> str:
        """Return the text at a span. The only sanctioned way to build a quote."""
        if span.end > len(self.text):
            raise ValueError(
                f"span [{span.start}, {span.end}) runs past the document ({len(self.text)} chars)"
            )
        return self.text[span.start : span.end]

    def anchor(self, span: Span, section_path: str = "") -> Anchor:
        return Anchor(
            paper_id=self.paper_id,
            section_path=section_path,
            span=span,
            quote=self.quote(span),
        )

    def verify_anchor(self, anchor: Anchor) -> bool:
        """True when the anchor's stored quote still matches the document.

        False means either the anchor was fabricated or the source changed --
        both of which must stop an answer from being emitted.
        """
        if anchor.paper_id != self.paper_id:
            return False
        try:
            return self.quote(anchor.span) == anchor.quote
        except ValueError:
            return False

    # ---- lookup --------------------------------------------------------

    def equation_by_number(self, number: int) -> Equation | None:
        return next((e for e in self.equations if e.number == number), None)

    def equation_by_label(self, label: str) -> Equation | None:
        return next((e for e in self.equations if e.label == label), None)

    def cited_keys(self) -> set[str]:
        return {c.key for c in self.citations}

    # ---- identity ------------------------------------------------------

    def content_hash(self) -> str:
        """SHA-256 over the flattened text, for corpus snapshots."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()
