"""SQLite corpus store.

Papers, their flattened text, chunks, equations, figures, citation uses and
reference entries all live here. FTS5 supplies BM25 without a second dependency.

The schema keeps ``chunks.start``/``end`` alongside every chunk because a
retrieval hit has to come back as a span into the paper's text, not as a loose
string -- otherwise nothing downstream can anchor a claim to it.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ingest.bibliography import BibEntry
from ..ingest.document import Document, Span

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id     TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    authors      TEXT NOT NULL DEFAULT '[]',
    abstract     TEXT NOT NULL DEFAULT '',
    text         TEXT NOT NULL,
    provenance   TEXT NOT NULL DEFAULT 'latex',
    license      TEXT,
    macro_count  INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    ingested_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id     TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    section_path TEXT NOT NULL DEFAULT '',
    kind         TEXT NOT NULL DEFAULT 'paragraph',
    start        INTEGER NOT NULL,
    end          INTEGER NOT NULL,
    text         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_paper ON chunks(paper_id);

CREATE TABLE IF NOT EXISTS equations (
    paper_id     TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    number       INTEGER,
    label        TEXT,
    latex        TEXT NOT NULL,
    environment  TEXT NOT NULL DEFAULT 'equation',
    section_path TEXT NOT NULL DEFAULT '',
    start        INTEGER NOT NULL,
    end          INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS equations_paper ON equations(paper_id);

CREATE TABLE IF NOT EXISTS figures (
    paper_id     TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    number       INTEGER,
    label        TEXT,
    caption      TEXT NOT NULL DEFAULT '',
    graphics     TEXT NOT NULL DEFAULT '[]',
    section_path TEXT NOT NULL DEFAULT '',
    start        INTEGER NOT NULL,
    end          INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS figures_paper ON figures(paper_id);

CREATE TABLE IF NOT EXISTS citation_uses (
    paper_id     TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    cite_key     TEXT NOT NULL,
    context      TEXT NOT NULL DEFAULT '',
    section_path TEXT NOT NULL DEFAULT '',
    start        INTEGER NOT NULL,
    end          INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS uses_paper ON citation_uses(paper_id);

CREATE TABLE IF NOT EXISTS references_ (
    paper_id  TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    cite_key  TEXT NOT NULL,
    raw       TEXT NOT NULL,
    text      TEXT NOT NULL DEFAULT '',
    doi       TEXT,
    arxiv_id  TEXT,
    year      INTEGER,
    title     TEXT,
    origin    TEXT NOT NULL DEFAULT 'thebibliography',
    PRIMARY KEY (paper_id, cite_key)
);

-- Resolution results are cached across sessions: an identifier that resolved
-- once should not cost another request to a provider.
CREATE TABLE IF NOT EXISTS resolutions (
    identifier  TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    resolved    INTEGER NOT NULL,
    source      TEXT,
    title       TEXT,
    payload     TEXT NOT NULL DEFAULT '{}',
    checked_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='chunk_id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.chunk_id, old.text);
END;
"""


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit, carrying enough to become an anchor."""

    chunk_id: int
    paper_id: str
    section_path: str
    kind: str
    start: int
    end: int
    text: str

    @property
    def span(self) -> Span:
        return Span(self.start, self.end)


class Store:
    """The corpus. One SQLite file, opened per operation."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- writing -------------------------------------------------------

    def add_document(
        self, document: Document, references: list[BibEntry] | None = None, chunks: list[Any] | None = None
    ) -> None:
        """Insert or replace a paper and everything derived from it."""
        from .chunk import chunk_document  # local import: chunking depends on the model

        pieces = chunks if chunks is not None else chunk_document(document)

        with self.connect() as conn:
            conn.execute("DELETE FROM papers WHERE paper_id = ?", (document.paper_id,))
            conn.execute(
                "INSERT INTO papers(paper_id, title, authors, abstract, text, provenance,"
                " license, macro_count, content_hash) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    document.paper_id,
                    document.title,
                    json.dumps(document.authors),
                    document.abstract,
                    document.text,
                    document.provenance,
                    document.license,
                    document.macro_count,
                    document.content_hash(),
                ),
            )
            conn.executemany(
                "INSERT INTO chunks(paper_id, section_path, kind, start, end, text)"
                " VALUES (?,?,?,?,?,?)",
                [
                    (document.paper_id, c.section_path, c.kind, c.span.start, c.span.end, c.text)
                    for c in pieces
                ],
            )
            conn.executemany(
                "INSERT INTO equations(paper_id, number, label, latex, environment,"
                " section_path, start, end) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (document.paper_id, e.number, e.label, e.latex, e.environment,
                     e.section_path, e.span.start, e.span.end)
                    for e in document.equations
                ],
            )
            conn.executemany(
                "INSERT INTO figures(paper_id, number, label, caption, graphics,"
                " section_path, start, end) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (document.paper_id, f.number, f.label, f.caption, json.dumps(f.graphics),
                     f.section_path, f.span.start, f.span.end)
                    for f in document.figures
                ],
            )
            conn.executemany(
                "INSERT INTO citation_uses(paper_id, cite_key, context, section_path, start, end)"
                " VALUES (?,?,?,?,?,?)",
                [
                    (document.paper_id, c.key, c.context, c.section_path, c.span.start, c.span.end)
                    for c in document.citations
                ],
            )
            conn.executemany(
                "INSERT OR REPLACE INTO references_(paper_id, cite_key, raw, text, doi,"
                " arxiv_id, year, title, origin) VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (document.paper_id, r.key, r.raw, r.text, r.doi, r.arxiv_id,
                     r.year, r.title, r.origin)
                    for r in (references or [])
                ],
            )

    def record_resolution(
        self,
        identifier: str,
        kind: str,
        resolved: bool,
        source: str | None = None,
        title: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO resolutions(identifier, kind, resolved, source, title,"
                " payload) VALUES (?,?,?,?,?,?)",
                (identifier, kind, int(resolved), source, title, json.dumps(payload or {})),
            )

    # ---- reading -------------------------------------------------------

    def has_paper(self, paper_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        return row is not None

    def paper_ids(self) -> list[str]:
        with self.connect() as conn:
            return [r["paper_id"] for r in conn.execute("SELECT paper_id FROM papers ORDER BY paper_id")]

    def paper_text(self, paper_id: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT text FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return row["text"] if row else None

    def paper_row(self, paper_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        return dict(row) if row else None

    def chunks(self, paper_id: str | None = None) -> list[Chunk]:
        query = "SELECT * FROM chunks"
        params: tuple[Any, ...] = ()
        if paper_id:
            query += " WHERE paper_id = ?"
            params = (paper_id,)
        with self.connect() as conn:
            rows = conn.execute(query + " ORDER BY chunk_id", params).fetchall()
        return [Chunk(**{k: r[k] for k in Chunk.__dataclass_fields__}) for r in rows]

    def chunk(self, chunk_id: int) -> Chunk | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return Chunk(**{k: row[k] for k in Chunk.__dataclass_fields__}) if row else None

    def equations(self, paper_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM equations WHERE paper_id = ? ORDER BY start", (paper_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def figures(self, paper_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM figures WHERE paper_id = ? ORDER BY number", (paper_id,)
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["graphics"] = json.loads(item["graphics"])
            out.append(item)
        return out

    def references(self, paper_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM references_ WHERE paper_id = ? ORDER BY cite_key", (paper_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def citation_uses(self, paper_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM citation_uses WHERE paper_id = ?", (paper_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def cached_resolution(self, identifier: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM resolutions WHERE identifier = ?", (identifier,)
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["resolved"] = bool(item["resolved"])
        item["payload"] = json.loads(item["payload"])
        return item

    def search_bm25(self, query: str, limit: int = 20) -> list[tuple[Chunk, float]]:
        """Full-text search. Returns chunks with a positive relevance score."""
        cleaned = _fts_query(query)
        if not cleaned:
            return []
        with self.connect() as conn:
            try:
                rows = conn.execute(
                    "SELECT c.*, bm25(chunks_fts) AS score FROM chunks_fts"
                    " JOIN chunks c ON c.chunk_id = chunks_fts.rowid"
                    " WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                    (cleaned, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                # A query that survives sanitising can still be invalid FTS5
                # syntax; an empty result is better than a crashed search.
                return []
        # bm25() returns more-negative for better matches; flip for sanity.
        return [
            (Chunk(**{k: r[k] for k in Chunk.__dataclass_fields__}), -float(r["score"]))
            for r in rows
        ]

    def stats(self) -> dict[str, int]:
        with self.connect() as conn:
            return {
                name: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for name, table in (
                    ("papers", "papers"), ("chunks", "chunks"), ("equations", "equations"),
                    ("figures", "figures"), ("citation_uses", "citation_uses"),
                    ("references", "references_"), ("resolutions", "resolutions"),
                )
            }


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    FTS5 treats a pile of punctuation as syntax, and physics queries are full of
    hyphens, slashes and carets. Quoting each word makes every token a literal.
    """
    words = [w for w in "".join(c if c.isalnum() else " " for c in query).split() if len(w) > 1]
    return " OR ".join(f'"{w}"' for w in words)
