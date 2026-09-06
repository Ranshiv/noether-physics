"""Answer records -- the artifact an answer actually is.

An answer here is not a string. It is a record on disk holding the claims, the
span each rests on, the references and whether they resolved, the equations with
their dimensional verdicts, the figures, and the corpus hash the whole thing was
computed against. That is what makes guarantee 4 -- reproducibility -- more than
an intention: a record either replays or it does not.

The shape is deliberately serialisable to plain JSON. A record that can only be
read back by the version of the code that wrote it is not a durable artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..ingest.document import Anchor, Span

RECORD_SCHEMA_VERSION = 1


@dataclass
class EvidenceItem:
    """One retrieved passage offered to whoever is drafting the answer."""

    evidence_id: str
    paper_id: str
    section_path: str
    quote: str
    start: int
    end: int
    kind: str = "paragraph"
    score: float = 0.0

    def to_anchor(self) -> Anchor:
        return Anchor(self.paper_id, self.section_path, Span(self.start, self.end), self.quote)


@dataclass
class EvidencePack:
    """What ``ask_grounded`` hands back: the raw material for an answer.

    Deliberately not a draft. The drafting is done by whatever model is calling
    us; our job is to supply passages that provably exist and then to judge what
    comes back against them.
    """

    question: str
    items: list[EvidenceItem] = field(default_factory=list)
    papers: dict[str, str] = field(default_factory=dict)
    corpus_hash: str = ""
    #: Instructions returned alongside the evidence, so a caller cannot claim
    #: not to have known the rules it will be judged by.
    contract: str = (
        "Every sentence must cite one or more evidence_id values. A sentence "
        "without one will be dropped. Do not paraphrase a quote into a stronger "
        "claim than it supports. If the evidence does not answer the question, "
        "say so rather than filling the gap."
    )

    def by_id(self, evidence_id: str) -> EvidenceItem | None:
        return next((i for i in self.items if i.evidence_id == evidence_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "corpus_hash": self.corpus_hash,
            "contract": self.contract,
            "papers": self.papers,
            "evidence": [asdict(i) for i in self.items],
        }


@dataclass
class RecordedClaim:
    """A claim that survived the gate, with the spans that carry it."""

    text: str
    anchors: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_anchors(cls, text: str, anchors: list[Anchor]) -> RecordedClaim:
        return cls(
            text=text,
            anchors=[
                {
                    "paper_id": a.paper_id,
                    "section_path": a.section_path,
                    "start": a.span.start,
                    "end": a.span.end,
                    "quote": a.quote,
                }
                for a in anchors
            ],
        )


@dataclass
class RecordedEquation:
    """An equation shown in an answer, with its verdicts attached."""

    latex: str
    paper_id: str = ""
    number: int | None = None
    parsed: bool = False
    dimensions: str = "unknown"
    detail: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class AnswerRecord:
    """A complete, replayable answer."""

    answer_id: str
    question: str
    claims: list[RecordedClaim] = field(default_factory=list)
    equations: list[RecordedEquation] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    references: dict[str, bool] = field(default_factory=dict)
    #: What the gate removed, and why. Always present, even when empty: a reader
    #: needs to see that the check ran and found nothing, not merely find silence.
    dropped: list[dict[str, str]] = field(default_factory=list)
    corpus_hash: str = ""
    created_at: str = ""
    schema_version: int = RECORD_SCHEMA_VERSION

    @property
    def resolved_count(self) -> int:
        return sum(1 for ok in self.references.values() if ok)

    @property
    def fully_resolved(self) -> bool:
        """Guarantee 1, restated as a property of the finished record."""
        return all(self.references.values())

    def summary(self) -> str:
        parts = [f"{len(self.claims)} claims"]
        if self.references:
            parts.append(f"{self.resolved_count}/{len(self.references)} references resolved")
        if self.equations:
            checked = sum(1 for e in self.equations if e.dimensions == "consistent")
            parts.append(f"{checked}/{len(self.equations)} equations dimensionally consistent")
        if self.dropped:
            parts.append(f"{len(self.dropped)} dropped")
        return ", ".join(parts)

    # ---- persistence ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.answer_id}.json"
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: Path) -> AnswerRecord:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("schema_version", 0)
        if version > RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"record schema v{version} is newer than this build understands "
                f"(v{RECORD_SCHEMA_VERSION}); upgrade rather than reading it partially"
            )
        return cls(
            answer_id=data["answer_id"],
            question=data["question"],
            claims=[RecordedClaim(**c) for c in data.get("claims", [])],
            equations=[RecordedEquation(**e) for e in data.get("equations", [])],
            figures=data.get("figures", []),
            references=data.get("references", {}),
            dropped=data.get("dropped", []),
            corpus_hash=data.get("corpus_hash", ""),
            created_at=data.get("created_at", ""),
            schema_version=version,
        )

    def content_hash(self) -> str:
        """Hash over the record's substance, excluding the timestamp.

        Two runs of the same question over the same corpus should hash the same,
        so a changed hash means changed content rather than a changed clock.
        """
        payload = self.to_dict()
        payload.pop("created_at", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def new_answer_id(question: str) -> str:
    """A short, stable-ish id: time-ordered, with a hash of the question."""
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:6]
    return f"ans_{stamp}_{digest}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
