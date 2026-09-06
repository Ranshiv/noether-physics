"""arXiv connector: metadata via the Atom API, LaTeX source via e-print.

The e-print endpoint is the reason this project is a physics tool rather than a
general one. It returns the actual submission -- LaTeX with figure files -- so
equations, their numbers, section structure and cite keys are read rather than
guessed. PDF parsing is a fallback for the minority of papers that have no
source, not the main path.

Bulk crawling arXiv is forbidden by their terms. We fetch one paper at a time,
behind the registry's 3-second interval, and cache permanently so any given
paper is downloaded exactly once.
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from ..config import Config
from ..netclient import NetClient
from .registry import get_source

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

#: 2007-and-later ids (0704.0001v2) and the older scheme. Old subclasses are
#: not uniformly two uppercase letters -- cond-mat.str-el, math.AG and
#: physics.flu-dyn all occur -- so the subclass accepts mixed case and hyphens.
_ID_RE = re.compile(
    r"^(?:arxiv:)?(?P<id>\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Za-z-]+)?/\d{7})(?:v(?P<version>\d+))?$",
    re.IGNORECASE,
)

_ABS_PREFIXES = (
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
    "https://arxiv.org/pdf/",
    "http://arxiv.org/pdf/",
)


class ArxivError(RuntimeError):
    """arXiv returned something we cannot use."""


def parse_id(raw: str) -> tuple[str, int | None]:
    """Normalise an arXiv identifier. Returns ``(id, version or None)``.

    Accepts ``2001.11966``, ``arXiv:2001.11966v2``, ``hep-th/9901001`` and the
    full abs/pdf URLs users paste out of a browser.
    """
    text = raw.strip()
    for prefix in _ABS_PREFIXES:
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.removesuffix(".pdf")

    match = _ID_RE.match(text)
    if not match:
        raise ArxivError(f"{raw!r} is not an arXiv identifier")
    version = match.group("version")
    return match.group("id"), int(version) if version else None


@dataclass
class PaperMeta:
    """What the Atom API tells us about one paper."""

    arxiv_id: str
    version: int | None
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: datetime | None = None
    updated: datetime | None = None
    doi: str | None = None
    journal_ref: str | None = None
    #: Licence URL. Guards figure re-export -- see answer/assemble.py.
    license: str | None = None
    comment: str | None = None
    links: dict[str, str] = field(default_factory=dict)

    @property
    def primary_category(self) -> str | None:
        return self.categories[0] if self.categories else None

    @property
    def versioned_id(self) -> str:
        return f"{self.arxiv_id}v{self.version}" if self.version else self.arxiv_id


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    # Atom hard-wraps abstracts at the source's column width; collapsing runs of
    # whitespace makes span offsets match what a reader actually sees.
    return re.sub(r"\s+", " ", node.text).strip()


def _timestamp(node: ET.Element | None) -> datetime | None:
    raw = _text(node)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def parse_entry(entry: ET.Element) -> PaperMeta:
    """Turn one Atom ``<entry>`` into a PaperMeta."""
    raw_id = _text(entry.find(f"{_ATOM}id"))
    arxiv_id, version = parse_id(raw_id.rsplit("/abs/", 1)[-1])

    links: dict[str, str] = {}
    for link in entry.findall(f"{_ATOM}link"):
        rel = link.get("title") or link.get("rel") or ""
        href = link.get("href")
        if href:
            links[rel] = href

    return PaperMeta(
        arxiv_id=arxiv_id,
        version=version,
        title=_text(entry.find(f"{_ATOM}title")),
        authors=[
            _text(a.find(f"{_ATOM}name"))
            for a in entry.findall(f"{_ATOM}author")
            if _text(a.find(f"{_ATOM}name"))
        ],
        abstract=_text(entry.find(f"{_ATOM}summary")),
        categories=[c.get("term", "") for c in entry.findall(f"{_ATOM}category") if c.get("term")],
        published=_timestamp(entry.find(f"{_ATOM}published")),
        updated=_timestamp(entry.find(f"{_ATOM}updated")),
        doi=_text(entry.find(f"{_ARXIV}doi")) or None,
        journal_ref=_text(entry.find(f"{_ARXIV}journal_ref")) or None,
        license=_text(entry.find(f"{_ATOM}rights")) or None,
        comment=_text(entry.find(f"{_ARXIV}comment")) or None,
        links=links,
    )


def parse_feed(xml: str) -> list[PaperMeta]:
    """Parse an Atom feed into papers, skipping arXiv's error entries."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ArxivError(f"arXiv returned unparseable XML: {exc}") from exc

    papers: list[PaperMeta] = []
    for entry in root.findall(f"{_ATOM}entry"):
        # A failed lookup comes back as a single entry whose id is the error URL.
        if "api/errors" in _text(entry.find(f"{_ATOM}id")):
            continue
        try:
            papers.append(parse_entry(entry))
        except ArxivError:
            continue
    return papers


class ArxivClient:
    """Metadata search and LaTeX source retrieval."""

    def __init__(self, client: NetClient | None = None, config: Config | None = None) -> None:
        self.config = config or (client.config if client else Config.from_env())
        self.net = client or NetClient(self.config)
        self.spec = get_source("arxiv")
        # Register arXiv's rate limit with the shared limiter.
        self.net.limiter.configure("export.arxiv.org", self.spec.min_interval_s)

    # ---- metadata ------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int = 20,
        start: int = 0,
        sort_by: str = "relevance",
    ) -> list[PaperMeta]:
        """Search arXiv using its own query syntax, e.g. ``cat:quant-ph AND ti:Rabi``."""
        response = self.net.get(
            "arxiv",
            self.spec.base_url,
            params={
                "search_query": query,
                "start": start,
                # arXiv caps a single page well below this; asking for more just
                # wastes a round trip.
                "max_results": min(max_results, 100),
                "sortBy": sort_by,
                "sortOrder": "descending",
            },
        )
        if response.status != 200:
            raise ArxivError(f"arXiv search returned HTTP {response.status}")
        return parse_feed(response.text)

    def get(self, arxiv_id: str) -> PaperMeta:
        """Fetch metadata for one paper."""
        ident, _ = parse_id(arxiv_id)
        response = self.net.get(
            "arxiv", self.spec.base_url, params={"id_list": ident, "max_results": 1}
        )
        papers = parse_feed(response.text)
        if not papers:
            raise ArxivError(f"arXiv has no record of {arxiv_id!r}")
        return papers[0]

    # ---- source --------------------------------------------------------

    def source_path(self, arxiv_id: str) -> Path:
        ident, _ = parse_id(arxiv_id)
        return self.config.cache_dir / "arxiv" / f"{ident.replace('/', '_')}.tar.gz"

    def fetch_source(self, arxiv_id: str, refresh: bool = False) -> bytes:
        """Download the e-print archive, or return the cached copy.

        Cached permanently: a submitted version is immutable, so refetching it
        would spend arXiv's bandwidth to obtain bytes we already hold.
        """
        path = self.source_path(arxiv_id)
        if path.exists() and not refresh:
            return path.read_bytes()

        ident, _ = parse_id(arxiv_id)
        url = self.spec.url("eprint_url") + ident
        response = self.net.get("arxiv", url)
        if response.status != 200:
            raise ArxivError(
                f"e-print fetch for {ident} returned HTTP {response.status}. "
                "Papers submitted as PDF-only have no source; use the PDF fallback."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return response.content

    def unpack_source(self, arxiv_id: str, dest: Path) -> Path:
        """Unpack an e-print into ``dest``. Returns the directory written.

        arXiv serves either a gzipped tarball or, for single-file submissions, a
        bare gzipped .tex -- both arrive without a distinguishing content type,
        so we try tar first and fall back.
        """
        blob = self.fetch_source(arxiv_id)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
                _safe_extract(tar, dest)
        except tarfile.ReadError:
            try:
                (dest / "main.tex").write_bytes(gzip.decompress(blob))
            except (OSError, EOFError) as exc:
                raise ArxivError(f"e-print for {arxiv_id} is neither tar.gz nor gzip") from exc
        return dest


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract, refusing members that would escape the destination.

    arXiv submissions are author-supplied archives. Treating them as trusted
    input is how a tool ends up writing outside its own data root.
    """
    root = dest.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            continue
        target = (root / member.name).resolve()
        if not target.is_relative_to(root):
            raise ArxivError(f"refusing path-traversing archive member {member.name!r}")
    tar.extractall(dest, filter="data")
