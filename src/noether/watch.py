"""Watch lists: new arXiv postings in a topic you follow.

A digest is only useful if it is honest about what it did not do. So a run
records which papers it saw, which it managed to ingest, and which it could not
(PDF-only submissions have no LaTeX source and are reported, never silently
dropped). Seen ids are remembered, so the same paper is never announced twice.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .library import Library


@dataclass
class Watch:
    """One saved query."""

    name: str
    query: str
    #: arXiv ids already reported, so a digest never repeats itself.
    seen: list[str] = field(default_factory=list)
    created_at: str = ""
    last_run: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DigestEntry:
    """One paper in a digest, and what happened to it."""

    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    ingested: bool = False
    note: str = ""


@dataclass
class Digest:
    """The result of running one watch."""

    name: str
    query: str
    entries: list[DigestEntry] = field(default_factory=list)
    ran_at: str = ""

    @property
    def ingested(self) -> list[DigestEntry]:
        return [e for e in self.entries if e.ingested]

    @property
    def skipped(self) -> list[DigestEntry]:
        return [e for e in self.entries if not e.ingested]

    def summary(self) -> str:
        if not self.entries:
            return f"{self.name}: nothing new"
        return (
            f"{self.name}: {len(self.entries)} new, "
            f"{len(self.ingested)} ingested, {len(self.skipped)} skipped"
        )


class WatchList:
    """Saved queries, stored as JSON under the data root."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Watch]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {name: Watch(**data) for name, data in raw.items()}

    def _save(self, watches: dict[str, Watch]) -> None:
        self.path.write_text(
            json.dumps({n: w.to_dict() for n, w in watches.items()}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def list(self) -> list[Watch]:
        return sorted(self._load().values(), key=lambda w: w.name)

    def add(self, name: str, query: str) -> Watch:
        watches = self._load()
        watch = Watch(
            name=name,
            query=query,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        watches[name] = watch
        self._save(watches)
        return watch

    def remove(self, name: str) -> bool:
        watches = self._load()
        if name not in watches:
            return False
        del watches[name]
        self._save(watches)
        return True

    def run(self, name: str, library: Library, limit: int = 10, ingest: bool = True) -> Digest:
        """Run one watch: search, report what is new, optionally ingest it."""
        watches = self._load()
        if name not in watches:
            raise KeyError(f"no watch named {name!r}")
        watch = watches[name]

        digest = Digest(
            name=name,
            query=watch.query,
            ran_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

        # Sorted by submission date: a watch is about what is new, not about
        # what best matches the words.
        papers = library.search_arxiv(watch.query, limit=limit)
        seen = set(watch.seen)

        for paper in papers:
            if paper.arxiv_id in seen:
                continue
            entry = DigestEntry(
                arxiv_id=paper.arxiv_id, title=paper.title, authors=paper.authors[:4]
            )
            if ingest:
                result = library.ingest_arxiv(paper.arxiv_id)
                entry.ingested = result.ok
                entry.note = result.error
            digest.entries.append(entry)
            watch.seen.append(paper.arxiv_id)

        watch.last_run = digest.ran_at
        watches[name] = watch
        self._save(watches)
        return digest
