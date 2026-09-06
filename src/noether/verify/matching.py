"""Matching a free-text reference to a real record.

The first version of this scored a reference by comparing its whole text to a
candidate's **title**, and resolved 3 of 37 references on a real paper -- 8%.
The reason is plain once measured: a physics reference is written

    G. Luders. Mat.-fys. Medd. 28 (1954) 5.

There is no title in it at all. Many carry only author, journal, volume, year and
page. Comparing that string to a title is comparing two different things, and
the low scores were correct answers to the wrong question.

So matching works on the fields a physics reference actually carries. Each
candidate is *corroborated*: does the year agree, does the first author's surname
appear, do volume and page agree, and -- only when the reference states one --
does the title agree. Independent agreements compound; a single coincidence does
not carry a match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

#: Corroboration needed to accept a match. Reachable by year plus surname plus
#: one of volume/page, or by a strong title agreement plus a year.
MATCH_THRESHOLD = 0.70

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")
#: "124, 081803" and the equally common "92 (2015) 052008".
_VOLUME_PAGE_RE = re.compile(r"\b(\d{1,4})\s*[,:]\s*(\d{1,6})\b")
_VOLUME_YEAR_PAGE_RE = re.compile(
    r"\b(\d{1,4})\s*\(\d{4}\)\s*(\d{1,6})\b"
)
_SURNAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")

#: Words that look like surnames but are journal or structural noise.
_NOT_SURNAMES = frozenset(
    ["Phys", "Rev", "Lett", "Nucl", "Instrum", "Methods", "Journal", "Physics", "Review", "Letters", "Nature", "Science", "Annals", "Proceedings", "Conference", "Press", "University", "Springer", "Elsevier", "American", "Physical", "Society", "European", "Modern", "Reports", "Progress", "Advances", "Acta", "Communications", "Applied", "Optics", "Chemical", "Astronomy", "Astrophysics", "Data", "Nuclear", "Particle", "High", "Energy", "Instruments", "Methods", "Research", "Section", "Volume", "Edition", "Cambridge", "Oxford", "Academic", "Wiley", "Taylor", "Francis", "World", "Scientific"]
)


#: Abbreviation fragments mapped to the journal names INSPIRE indexes under.
#: Longest fragments first so "phys rev lett" is not matched as "phys rev".
JOURNAL_PATTERNS = [
    ("phys rev lett", "Phys.Rev.Lett."),
    ("phys rev a", "Phys.Rev.A"), ("phys rev b", "Phys.Rev.B"),
    ("phys rev c", "Phys.Rev.C"), ("phys rev d", "Phys.Rev.D"),
    ("phys rev e", "Phys.Rev.E"), ("phys rev x", "Phys.Rev.X"),
    ("rev mod phys", "Rev.Mod.Phys."),
    ("nucl phys b", "Nucl.Phys.B"), ("nucl phys a", "Nucl.Phys.A"),
    ("phys lett b", "Phys.Lett.B"), ("phys lett a", "Phys.Lett.A"),
    ("eur phys j c", "Eur.Phys.J.C"), ("eur phys j a", "Eur.Phys.J.A"),
    ("nucl instrum", "Nucl.Instrum.Meth.A"),
    ("j high energy phys", "JHEP"), ("jhep", "JHEP"),
    ("new j phys", "New J.Phys."), ("phys rev", "Phys.Rev."),
]


@dataclass
class ReferenceFields:
    """What could be mined out of a free-text reference."""

    year: int | None = None
    surnames: list[str] = field(default_factory=list)
    volume: str | None = None
    page: str | None = None
    title: str | None = None
    #: The original reference string, for signals that need the whole text.
    raw_text: str = ""
    #: Journal name, when recognisable. With volume and page this forms a
    #: coordinate INSPIRE can resolve exactly.
    journal: str | None = None

    @property
    def first_surname(self) -> str | None:
        return self.surnames[0] if self.surnames else None

    @property
    def is_thin(self) -> bool:
        """True when there is too little here to corroborate anything."""
        return self.year is None and not self.surnames


def extract_fields(text: str, known_title: str | None = None) -> ReferenceFields:
    """Mine year, surnames, volume and page out of a reference string."""
    fields = ReferenceFields(title=known_title, raw_text=text)

    years = _YEAR_RE.findall(text)
    if years:
        # The last year is usually the publication year; an earlier one tends to
        # belong to a series title or an edition note.
        fields.year = int(years[-1])

    volume_page = _VOLUME_PAGE_RE.search(text)
    if volume_page:
        fields.volume, fields.page = volume_page.group(1), volume_page.group(2)
    else:
        alternative = _VOLUME_YEAR_PAGE_RE.search(text)
        if alternative:
            fields.volume, fields.page = alternative.group(1), alternative.group(2)

    fields.surnames = [
        word for word in _SURNAME_RE.findall(text) if word not in _NOT_SURNAMES
    ][:6]
    fields.journal = _journal_name(text)
    return fields


def _journal_name(text: str) -> str | None:
    """Recognise a journal from its standard abbreviation."""
    # Punctuation becomes spaces, so "Phys. Rev. A" leaves doubled gaps that
    # would not match a single-spaced pattern. Collapse them.
    lowered = re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", text.lower()))
    for pattern, name in JOURNAL_PATTERNS:
        if pattern in lowered:
            return name
    return None


def _title_appears_in(title: str, text: str) -> bool:
    """True when a candidate's title is present, near-verbatim, in the reference."""
    needle = normalise_title(title)
    if len(needle) < 12:
        return False  # too short to be evidence of anything
    return needle in normalise_title(text)


def normalise_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", title.lower())).strip()


def title_similarity(a: str, b: str) -> float:
    left, right = normalise_title(a), normalise_title(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


@dataclass
class Corroboration:
    """How well a candidate agrees with a reference, and on what."""

    score: float
    agreements: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)

    def explain(self) -> str:
        parts = []
        if self.agreements:
            parts.append("agrees on " + ", ".join(self.agreements))
        if self.disagreements:
            parts.append("differs on " + ", ".join(self.disagreements))
        return "; ".join(parts) or "nothing to corroborate"


@dataclass
class Candidate:
    """A provider record, flattened to the fields we can compare."""

    title: str | None = None
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    volume: str | None = None
    page: str | None = None
    doi: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def corroborate(fields: ReferenceFields, candidate: Candidate) -> Corroboration:
    """Score a candidate against a reference's mined fields.

    Weights are chosen so that no single signal reaches the threshold alone. A
    year agreeing is weak -- thousands of papers share a year. Year plus the
    right first author plus the right volume is strong.
    """
    score = 0.0
    agreements: list[str] = []
    disagreements: list[str] = []

    if fields.year and candidate.year:
        if fields.year == candidate.year:
            score += 0.30
            agreements.append("year")
        elif abs(fields.year - candidate.year) == 1:
            # Preprint and journal years commonly differ by one.
            score += 0.15
            agreements.append("year within 1")
        else:
            score -= 0.25
            disagreements.append(f"year ({fields.year} vs {candidate.year})")

    if fields.first_surname and candidate.authors:
        author_text = " ".join(candidate.authors).lower()
        if fields.first_surname.lower() in author_text:
            score += 0.30
            agreements.append("first author")
        elif any(s.lower() in author_text for s in fields.surnames):
            score += 0.15
            agreements.append("an author")
        else:
            score -= 0.15
            disagreements.append("no shared author")

    if fields.volume and candidate.volume and fields.volume == str(candidate.volume):
        score += 0.20
        agreements.append("volume")
    if fields.page and candidate.page and str(candidate.page).startswith(fields.page):
        score += 0.20
        agreements.append("page")

    if fields.title and candidate.title:
        similarity = title_similarity(fields.title, candidate.title)
        if similarity >= 0.85:
            score += 0.40
            agreements.append("title")
        elif similarity >= 0.60:
            score += 0.20
            agreements.append("title (approximate)")
        else:
            score -= 0.10
            disagreements.append("title")
    elif candidate.title and fields.raw_text:
        # The reference states no title field, which is normal in physics -- but
        # the title is often sitting in the reference text unlabelled. A book
        # reference reading "Nuclear Magnetism: Order and Disorder, OUP (1982)"
        # names its title without any structure saying so.
        if _title_appears_in(candidate.title, fields.raw_text):
            score += 0.35
            agreements.append("title appears in the reference text")

    return Corroboration(max(0.0, min(1.0, score)), agreements, disagreements)


def best_candidate(
    fields: ReferenceFields, candidates: list[Candidate]
) -> tuple[Corroboration, Candidate | None]:
    """Pick the best-corroborated candidate."""
    best = Corroboration(0.0)
    chosen: Candidate | None = None
    for candidate in candidates:
        result = corroborate(fields, candidate)
        if result.score > best.score:
            best, chosen = result, candidate
    return best, chosen


# ---- provider adapters -------------------------------------------------

def from_crossref(item: dict[str, Any]) -> Candidate:
    titles = item.get("title") or []
    authors = [
        " ".join(filter(None, (a.get("given"), a.get("family"))))
        for a in item.get("author", [])
    ]
    year = None
    for key in ("published-print", "published-online", "issued", "created"):
        parts = item.get(key, {}).get("date-parts")
        if parts and parts[0] and parts[0][0]:
            year = int(parts[0][0])
            break
    return Candidate(
        title=titles[0] if titles else None,
        year=year,
        authors=authors,
        volume=item.get("volume"),
        page=(item.get("page") or "").split("-")[0] or None,
        doi=item.get("DOI"),
        raw=item,
    )


def from_openalex(item: dict[str, Any]) -> Candidate:
    authorships = item.get("authorships", [])
    biblio = item.get("biblio", {}) or {}
    return Candidate(
        title=item.get("title") or item.get("display_name"),
        year=item.get("publication_year"),
        authors=[a.get("author", {}).get("display_name", "") for a in authorships],
        volume=biblio.get("volume"),
        page=biblio.get("first_page"),
        doi=(item.get("doi") or "").replace("https://doi.org/", "") or None,
        raw=item,
    )


def from_inspire(hit: dict[str, Any]) -> Candidate:
    metadata = hit.get("metadata", {})
    titles = metadata.get("titles") or [{}]
    authors = [a.get("full_name", "") for a in metadata.get("authors", [])]
    publication = (metadata.get("publication_info") or [{}])[0]
    return Candidate(
        title=titles[0].get("title"),
        year=publication.get("year"),
        authors=authors,
        volume=publication.get("journal_volume"),
        page=publication.get("page_start"),
        doi=(metadata.get("dois") or [{}])[0].get("value"),
        raw=metadata,
    )
