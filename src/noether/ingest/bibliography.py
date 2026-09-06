"""Reference-list parsing.

Physics papers overwhelmingly ship an inline ``thebibliography`` of free-text
``\\bibitem`` entries rather than a structured ``.bib`` -- the nEDM paper we
measured has 44 of them, none carrying a DOI. So the common case is prose that
must be mined for identifiers and then fuzzy-matched against Crossref or
INSPIRE. That is harder than parsing BibTeX, and it is the default path, not the
exception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .preprocess import find_matching_brace

_THEBIB_RE = re.compile(r"\\begin\{thebibliography\}(?:\{[^}]*\})?(.*?)\\end\{thebibliography\}", re.DOTALL)
_BIBITEM_RE = re.compile(r"\\bibitem\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")

#: Identifiers worth mining out of free-text references, most specific first.
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
_ARXIV_RE = re.compile(
    r"arxiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Za-z-]+)?/\d{7})", re.IGNORECASE
)
_YEAR_RE = re.compile(r"\((\d{4})\)|\b(19|20)\d{2}\b")

_BIBTEX_ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)
_BIBTEX_FIELD_RE = re.compile(r"(\w+)\s*=\s*", re.IGNORECASE)

#: LaTeX noise that carries no meaning once the reference is plain text.
_TEX_NOISE_RE = re.compile(r"\\(?:newblock|emph|textbf|textit|bibinfo|url|href|doi)\b")


@dataclass
class BibEntry:
    """One reference, however it was declared.

    ``raw`` is kept verbatim because it is the only thing a fuzzy matcher can
    work from when no identifier is present, and because a resolution result has
    to be auditable against what the authors actually wrote.
    """

    key: str
    raw: str
    text: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    year: int | None = None
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    journal: str | None = None
    #: "thebibliography" or "bibtex" -- resolution confidence differs by origin.
    origin: str = "thebibliography"

    @property
    def has_identifier(self) -> bool:
        """True when the entry carries something directly resolvable."""
        return bool(self.doi or self.arxiv_id)


def clean_tex(text: str) -> str:
    """Reduce a LaTeX reference string to readable plain text."""
    text = _TEX_NOISE_RE.sub(" ", text)
    text = re.sub(r"\\[A-Za-z@]+\s*", " ", text)
    text = text.replace("~", " ").replace("\\&", "&")
    text = re.sub(r"[{}]", "", text)
    return re.sub(r"\s+", " ", text).strip(" ,.;")


def mine_identifiers(entry: BibEntry) -> BibEntry:
    """Pull DOI, arXiv id and year out of a free-text reference."""
    if entry.doi is None:
        doi = _DOI_RE.search(entry.raw)
        if doi:
            # Trailing punctuation is part of the sentence, not the DOI.
            entry.doi = doi.group(0).rstrip(".,;)")
    if entry.arxiv_id is None:
        arxiv = _ARXIV_RE.search(entry.raw)
        if arxiv:
            entry.arxiv_id = arxiv.group(1)
    if entry.year is None:
        year = _YEAR_RE.search(entry.text or entry.raw)
        if year:
            found = year.group(1) or year.group(0)
            with_digits = re.search(r"\d{4}", found)
            if with_digits:
                entry.year = int(with_digits.group(0))
    return entry


def parse_thebibliography(text: str) -> list[BibEntry]:
    """Parse an inline ``thebibliography`` environment."""
    block = _THEBIB_RE.search(text)
    if not block:
        return []
    body = block.group(1)

    matches = list(_BIBITEM_RE.finditer(body))
    entries: list[BibEntry] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        raw = body[start:end].strip()
        entry = BibEntry(key=match.group(1).strip(), raw=raw, text=clean_tex(raw))
        entries.append(mine_identifiers(entry))
    return entries


def parse_bibtex(text: str) -> list[BibEntry]:
    """Parse a ``.bib`` file. Brace-aware, so nested braces in titles survive."""
    entries: list[BibEntry] = []
    for match in _BIBTEX_ENTRY_RE.finditer(text):
        open_brace = text.rfind("{", match.start(), match.end())
        close = find_matching_brace(text, open_brace)
        if close == -1:
            continue
        body = text[match.end() : close]
        entry = BibEntry(
            key=match.group(2),
            raw=text[match.start() : close + 1],
            origin="bibtex",
        )
        for field_name, value in _bibtex_fields(body):
            lowered = field_name.lower()
            cleaned = clean_tex(value)
            if lowered == "title":
                entry.title = cleaned
            elif lowered == "doi":
                entry.doi = cleaned
            elif lowered in ("journal", "journaltitle"):
                entry.journal = cleaned
            elif lowered == "year":
                digits = re.search(r"\d{4}", cleaned)
                entry.year = int(digits.group(0)) if digits else None
            elif lowered == "author":
                entry.authors = [a.strip() for a in re.split(r"\s+and\s+", cleaned) if a.strip()]
            elif lowered in ("eprint", "archiveprefix") and "arxiv" not in cleaned.lower():
                entry.arxiv_id = entry.arxiv_id or cleaned
        entry.text = " ".join(
            part for part in (", ".join(entry.authors), entry.title, entry.journal) if part
        )
        entries.append(mine_identifiers(entry))
    return entries


def _bibtex_fields(body: str) -> list[tuple[str, str]]:
    """Yield ``(name, value)`` for each BibTeX field in an entry body."""
    fields: list[tuple[str, str]] = []
    for match in _BIBTEX_FIELD_RE.finditer(body):
        index = match.end()
        while index < len(body) and body[index] in " \t\n":
            index += 1
        if index >= len(body):
            break
        if body[index] == "{":
            close = find_matching_brace(body, index)
            if close == -1:
                continue
            fields.append((match.group(1), body[index + 1 : close]))
        elif body[index] == '"':
            close = body.find('"', index + 1)
            if close == -1:
                continue
            fields.append((match.group(1), body[index + 1 : close]))
        else:
            end = body.find(",", index)
            fields.append((match.group(1), body[index : end if end != -1 else len(body)]))
    return fields


def parse_references(latex: str, bib_files: list[str] | None = None) -> list[BibEntry]:
    """Collect references from an inline bibliography and any ``.bib`` sources.

    Inline entries win on key collision: they are what the submitted paper
    actually rendered from.
    """
    entries: dict[str, BibEntry] = {}
    for text in bib_files or []:
        for entry in parse_bibtex(text):
            entries[entry.key] = entry
    for entry in parse_thebibliography(latex):
        entries[entry.key] = entry
    return list(entries.values())
