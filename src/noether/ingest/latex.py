"""LaTeX source to :class:`Document`, preserving exact character spans.

The document text is built incrementally: each structural item is converted to
plain text, appended to a buffer, and its span recorded as it goes. That makes
every span exact by construction rather than by a fragile mapping back onto the
original source -- which matters because guarantee 2 rests entirely on a span
resolving to the text a human would read.

Equation numbering follows LaTeX's own rules, so "equation 14" resolves to the
equation *the authors* numbered 14: starred environments and rows marked
``\\nonumber`` do not advance the counter, and multi-row ``align`` bodies emit
one numbered equation per row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pylatexenc.latex2text import LatexNodes2Text

from .bibliography import BibEntry, parse_references
from .document import Block, CitationUse, Document, Equation, Figure, Section, Span
from .preprocess import find_matching_brace, preprocess

#: Environments that produce display math. Starred forms are unnumbered.
_MATH_ENVS = ("equation", "align", "eqnarray", "gather", "multline", "displaymath", "flalign")

#: Environments whose rows are numbered individually.
_MULTIROW_ENVS = frozenset({"align", "eqnarray", "gather", "flalign"})

_SECTION_LEVELS = {
    "part": 0, "chapter": 0, "section": 1, "subsection": 2,
    "subsubsection": 3, "paragraph": 4, "subparagraph": 5,
}

_THEOREM_ENVS = frozenset(
    {"theorem", "lemma", "proposition", "corollary", "definition", "remark", "proof", "conjecture"}
)

_DOC_RE = re.compile(r"\\begin\{document\}(.*?)(?:\\end\{document\}|\Z)", re.DOTALL)
_ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL)
_TITLE_RE = re.compile(r"\\title\s*(?:\[[^\]]*\])?\s*\{")
_AUTHOR_RE = re.compile(r"\\author\s*(?:\[[^\]]*\])?\s*\{")
_LABEL_RE = re.compile(r"\\label\s*\{([^}]+)\}")
_CITE_RE = re.compile(r"\\(?:cite|citep|citet|citealp|citeauthor|autocite)\*?\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_GRAPHICS_RE = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_CAPTION_RE = re.compile(r"\\caption\s*(?:\[[^\]]*\])?\s*\{")
_NONUMBER_RE = re.compile(r"\\(?:nonumber|notag)\b")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

#: Placeholder marking a citation's position through text conversion. Chosen to
#: survive LatexNodes2Text unchanged and never to occur in real prose.
_CITE_MARK = "ZqCITEq{}qZ"
_CITE_MARK_RE = re.compile(r"ZqCITEq(\d+)qZ")

#: Commands whose rendered output is metadata furniture, not paper content.
#: \maketitle in particular expands to "[NO 	itle GIVEN]", "[NO uthor
#: GIVEN]" and -- via 	oday -- THE CURRENT DATE. That last one is not merely
#: noise: it injects a false fact into the corpus and makes content_hash change
#: from one day to the next, which silently breaks reproducibility.
_FURNITURE_RE = re.compile(
    r"\\(?:maketitle|today|tableofcontents|listoffigures|listoftables)\b"
)

_ENV_START = re.compile(r"\\begin\{([A-Za-z*]+)\}")
_SECTION_RE = re.compile(r"\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\s*\{")


class LatexIngestError(RuntimeError):
    """The source could not be read as a paper."""


def _to_text(latex: str) -> str:
    """Render a LaTeX fragment as plain text."""
    try:
        return LatexNodes2Text(math_mode="text", keep_comments=False).latex_to_text(latex)
    except Exception:
        # pylatexenc raises a wide variety on malformed input; a degraded strip
        # is far better than refusing to ingest the paper at all.
        return re.sub(r"\\[A-Za-z@]+\s*", " ", latex)


def _normalise(text: str) -> str:
    """Collapse whitespace so spans line up with what a reader sees."""
    return re.sub(r"[ \t]*\n[ \t]*", "\n", re.sub(r"[ \t]+", " ", text)).strip()


def _braced_after(text: str, match: re.Match[str]) -> tuple[str, int]:
    """Content of the brace group the match ends on, and the index past it."""
    open_brace = match.end() - 1
    close = find_matching_brace(text, open_brace)
    if close == -1:
        return "", match.end()
    return text[open_brace + 1 : close], close + 1


def find_main_tex(root: Path) -> Path:
    """Pick the file that actually starts the document.

    An arXiv submission routinely ships several ``.tex`` files with no manifest
    saying which is the entry point, so we look for the one that declares the
    document rather than guessing by filename.
    """
    candidates = sorted(root.rglob("*.tex"))
    if not candidates:
        raise LatexIngestError(f"no .tex file under {root}")

    scored: list[tuple[int, int, Path]] = []
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        score = 0
        if "\\begin{document}" in text:
            score += 10
        if "\\documentclass" in text:
            score += 5
        if "\\title" in text:
            score += 2
        scored.append((score, len(text), path))

    if not scored:
        raise LatexIngestError(f"no readable .tex file under {root}")
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if scored[0][0] == 0:
        raise LatexIngestError(f"no file under {root} declares a document")
    return scored[0][2]


@dataclass
class _Builder:
    """Accumulates plain text and the spans that index into it."""

    text: str = ""

    def append(self, chunk: str, separator: str = "\n\n") -> Span:
        """Add a chunk, returning the span it occupies in the buffer."""
        if not chunk:
            return Span(len(self.text), len(self.text))
        prefix = separator if self.text else ""
        start = len(self.text) + len(prefix)
        self.text += prefix + chunk
        return Span(start, start + len(chunk))


class LatexParser:
    """Walks preprocessed LaTeX and produces a :class:`Document`."""

    def __init__(self, paper_id: str, license: str | None = None) -> None:
        self.paper_id = paper_id
        self.license = license
        self.builder = _Builder()
        self.equations: list[Equation] = []
        self.figures: list[Figure] = []
        self.citations: list[CitationUse] = []
        self.blocks: list[Block] = []
        self.sections: list[Section] = []
        self._stack: list[Section] = []
        self._eq_counter = 0
        self._fig_counter = 0

    # ---- section bookkeeping -------------------------------------------

    @property
    def section_path(self) -> str:
        return " > ".join(s.title for s in self._stack) if self._stack else ""

    def _push_section(self, title: str, level: int, span: Span) -> None:
        while self._stack and self._stack[-1].level >= level:
            self._stack.pop()
        parent_path = " > ".join(s.title for s in self._stack)
        section = Section(
            title=title,
            level=level,
            span=span,
            path=f"{parent_path} > {title}" if parent_path else title,
        )
        if self._stack:
            self._stack[-1].children.append(section)
        else:
            self.sections.append(section)
        self._stack.append(section)

    # ---- emitters ------------------------------------------------------

    def _emit_prose(self, latex: str) -> None:
        """Convert a run of prose into paragraphs, recording citation uses."""
        if not latex.strip():
            return

        # Mark citations before conversion so their position survives it.
        keys: list[list[str]] = []

        def mark(match: re.Match[str]) -> str:
            keys.append([k.strip() for k in match.group(1).split(",") if k.strip()])
            return _CITE_MARK.format(len(keys) - 1)

        marked = _CITE_RE.sub(mark, latex)
        marked = _LABEL_RE.sub("", marked)

        for chunk in re.split(r"\n\s*\n", marked):
            rendered = _normalise(_to_text(chunk))
            if not rendered.strip():
                continue

            # Strip markers, remembering where each sat in the final text.
            positions: list[tuple[int, list[str]]] = []
            clean_parts: list[str] = []
            cursor = 0
            offset = 0
            for match in _CITE_MARK_RE.finditer(rendered):
                clean_parts.append(rendered[cursor : match.start()])
                offset += match.start() - cursor
                index = int(match.group(1))
                if index < len(keys):
                    positions.append((offset, keys[index]))
                cursor = match.end()
            clean_parts.append(rendered[cursor:])
            clean = "".join(clean_parts).strip()
            if not clean:
                continue

            span = self.builder.append(clean)
            self.blocks.append(Block("paragraph", clean, span, self.section_path))

            for local_offset, cite_keys in positions:
                context = _sentence_at(clean, local_offset)
                for key in cite_keys:
                    self.citations.append(
                        CitationUse(
                            key=key,
                            span=Span(span.start, span.end),
                            context=context,
                            section_path=self.section_path,
                        )
                    )

    def _emit_equation(self, env: str, body: str) -> None:
        """Record display math, following LaTeX's numbering rules."""
        starred = env.endswith("*")
        base = env.rstrip("*")
        rows = _split_rows(body) if base in _MULTIROW_ENVS else [body]

        for row in rows:
            latex = row.strip()
            if not latex:
                continue
            label_match = _LABEL_RE.search(latex)
            label = label_match.group(1) if label_match else None
            numbered = not starred and not _NONUMBER_RE.search(latex)

            clean_latex = _normalise(_LABEL_RE.sub("", _NONUMBER_RE.sub("", latex)))
            if not clean_latex:
                continue

            if numbered:
                self._eq_counter += 1

            span = self.builder.append(clean_latex)
            self.equations.append(
                Equation(
                    latex=clean_latex,
                    span=span,
                    label=label,
                    number=self._eq_counter if numbered else None,
                    environment=base,
                    section_path=self.section_path,
                )
            )
            self.blocks.append(Block("equation", clean_latex, span, self.section_path))

    def _emit_figure(self, body: str) -> None:
        """Record a figure, its caption, and the graphics files it names."""
        caption = ""
        caption_match = _CAPTION_RE.search(body)
        if caption_match:
            raw, _ = _braced_after(body, caption_match)
            caption = _normalise(_to_text(_CITE_RE.sub("", _LABEL_RE.sub("", raw))))

        label_match = _LABEL_RE.search(body)
        graphics = [g.strip() for g in _GRAPHICS_RE.findall(body)]
        self._fig_counter += 1

        span = self.builder.append(caption) if caption else Span(len(self.builder.text), len(self.builder.text))
        if caption:
            self.blocks.append(Block("caption", caption, span, self.section_path))

        self.figures.append(
            Figure(
                label=label_match.group(1) if label_match else None,
                caption=caption,
                span=span,
                graphics=graphics,
                number=self._fig_counter,
                section_path=self.section_path,
            )
        )

    # ---- main walk -----------------------------------------------------

    def parse(self, body: str) -> None:
        """Walk the document body, emitting blocks in reading order."""
        cursor = 0
        while cursor < len(body):
            env_match = _ENV_START.search(body, cursor)
            sec_match = _SECTION_RE.search(body, cursor)

            candidates = [m for m in (env_match, sec_match) if m]
            if not candidates:
                self._emit_prose(body[cursor:])
                return
            match = min(candidates, key=lambda m: m.start())

            self._emit_prose(body[cursor : match.start()])

            if match is sec_match:
                title_raw, cursor = _braced_after(body, match)
                title = _normalise(_to_text(title_raw)) or "(untitled)"
                span = self.builder.append(title)
                self._push_section(title, _SECTION_LEVELS.get(match.group(1), 1), span)
                self.blocks.append(Block("paragraph", title, span, self.section_path))
                continue

            env = match.group(1)
            inner, cursor = _environment_body(body, match, env)
            base = env.rstrip("*")

            if base in _MATH_ENVS:
                self._emit_equation(env, inner)
            elif base in ("figure", "subfigure"):
                self._emit_figure(inner)
            elif base == "abstract":
                text = _normalise(_to_text(inner))
                if text:
                    span = self.builder.append(text)
                    self.blocks.append(Block("abstract", text, span, ""))
            elif base in _THEOREM_ENVS:
                text = _normalise(_to_text(_LABEL_RE.sub("", inner)))
                if text:
                    span = self.builder.append(text)
                    self.blocks.append(Block("theorem", text, span, self.section_path))
            elif base in ("thebibliography", "table", "tabular"):
                continue  # References are parsed separately; tables are Phase 7.
            else:
                self._emit_prose(inner)


def _environment_body(text: str, match: re.Match[str], env: str) -> tuple[str, int]:
    """Content of ``\\begin{env}...\\end{env}``, and the index past it."""
    end_marker = f"\\end{{{env}}}"
    end = text.find(end_marker, match.end())
    if end == -1:
        return text[match.end() :], len(text)
    return text[match.end() : end], end + len(end_marker)


def _split_rows(body: str) -> list[str]:
    r"""Split an align-style body on ``\\`` row separators."""
    return re.split(r"\\\\(?:\s*\[[^\]]*\])?", body)


def _sentence_at(text: str, offset: int) -> str:
    """The sentence containing ``offset``.

    This is what makes 'cited but never engaged with' detectable: it records how
    a work was used, not merely that it appeared in a list.
    """
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        if match.end() > offset:
            break
        start = match.end()
    end_match = _SENTENCE_END_RE.search(text, offset)
    end = end_match.start() if end_match else len(text)
    return text[start:end].strip()


def parse_document(
    root: Path,
    paper_id: str,
    title: str = "",
    license: str | None = None,
    main: Path | None = None,
) -> tuple[Document, list[BibEntry]]:
    """Ingest a LaTeX source tree into a Document and its references.

    ``main`` names the entry file explicitly. Auditing someone's own draft needs
    that: the directory may hold several papers, and guessing which one the user
    meant is not a decision this function should make.
    """
    main = main or find_main_tex(root)
    expanded, macros = preprocess(main, root)

    doc_match = _DOC_RE.search(expanded)
    body = doc_match.group(1) if doc_match else expanded
    body = _FURNITURE_RE.sub(" ", body)

    if not title:
        title_match = _TITLE_RE.search(expanded)
        if title_match:
            title = _normalise(_to_text(_braced_after(expanded, title_match)[0]))

    authors: list[str] = []
    for author_match in _AUTHOR_RE.finditer(expanded):
        rendered = _normalise(_to_text(_braced_after(expanded, author_match)[0]))
        authors.extend(a.strip() for a in re.split(r",| and ", rendered) if a.strip())

    abstract = ""
    abstract_match = _ABSTRACT_RE.search(expanded)
    if abstract_match:
        abstract = _normalise(_to_text(abstract_match.group(1)))

    parser = LatexParser(paper_id, license=license)
    parser.parse(body)

    bib_files = [
        path.read_text(encoding="utf-8", errors="replace") for path in sorted(root.rglob("*.bib"))
    ]
    references = parse_references(expanded, bib_files)

    document = Document(
        paper_id=paper_id,
        title=title or paper_id,
        text=parser.builder.text,
        provenance="latex",
        abstract=abstract,
        authors=authors,
        sections=parser.sections,
        blocks=parser.blocks,
        equations=parser.equations,
        figures=parser.figures,
        citations=parser.citations,
        license=license,
    )
    document.macro_count = len(macros)
    return document, references
