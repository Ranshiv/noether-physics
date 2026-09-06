"""The answering pipeline: gather evidence, judge a draft, record the result.

The division of labour is the point. NOETHER does not write prose and does not
call a language model. It:

1. **gathers** passages that provably exist in the corpus (``gather_evidence``),
2. **judges** whatever claims come back against those passages (``submit``),
3. **records** what survived, what did not, and why.

Claude -- or any caller -- does the writing. That separation is what makes the
guarantees enforceable: the component that could hallucinate is not the
component that decides what ships.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..compute.symbolic import parse_equation
from ..compute.units import check_equation
from ..index.retrieve import Retriever
from ..index.store import Store
from ..verify.gate import Claim, Gate
from .records import (
    AnswerRecord,
    EvidenceItem,
    EvidencePack,
    RecordedClaim,
    RecordedEquation,
    new_answer_id,
    utc_now,
)


@dataclass
class DraftClaim:
    """A claim as submitted by a caller: text plus the evidence it cites."""

    text: str
    evidence_ids: list[str] = field(default_factory=list)


def corpus_hash(store: Store) -> str:
    """A hash over which papers are in the corpus and their content.

    Part of guarantee 4: an answer is only reproducible against the corpus it
    was computed from, so the record has to name that corpus exactly.
    """
    digest = hashlib.sha256()
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT paper_id, content_hash FROM papers ORDER BY paper_id"
        ).fetchall()
    for row in rows:
        digest.update(row["paper_id"].encode("utf-8"))
        digest.update(row["content_hash"].encode("utf-8"))
    return digest.hexdigest()


def gather_evidence(
    store: Store,
    question: str,
    limit: int = 12,
    retriever: Retriever | None = None,
) -> EvidencePack:
    """Retrieve passages relevant to a question, ready to be cited.

    Each item gets a short stable id so a caller can reference it precisely
    rather than by quoting text back at us -- quoting back is exactly where a
    paraphrase silently becomes a fabricated quote.
    """
    retriever = retriever or Retriever(store)
    hits = retriever.search(question, limit=limit)

    items: list[EvidenceItem] = []
    papers: dict[str, str] = {}
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        items.append(
            EvidenceItem(
                evidence_id=f"e{index}",
                paper_id=chunk.paper_id,
                section_path=chunk.section_path,
                quote=chunk.text,
                start=chunk.start,
                end=chunk.end,
                kind=chunk.kind,
                score=round(hit.score, 6),
            )
        )
        if chunk.paper_id not in papers:
            row = store.paper_row(chunk.paper_id)
            papers[chunk.paper_id] = row["title"] if row else chunk.paper_id

    return EvidencePack(
        question=question, items=items, papers=papers, corpus_hash=corpus_hash(store)
    )


def submit(
    store: Store,
    pack: EvidencePack,
    drafts: list[DraftClaim],
    equations: list[dict[str, Any]] | None = None,
    references: dict[str, bool] | None = None,
    answers_dir: Path | None = None,
) -> AnswerRecord:
    """Judge a draft against the evidence and produce a record.

    Claims citing evidence that does not exist, or whose quote no longer matches
    the corpus, are dropped by the gate and listed in ``record.dropped``. A thin
    answer is visibly thin; it is never padded to look complete.
    """
    gate = Gate(store)

    claims: list[Claim] = []
    unknown_ids: dict[str, str] = {}
    for draft in drafts:
        anchors = []
        for evidence_id in draft.evidence_ids:
            item = pack.by_id(evidence_id)
            if item is None:
                unknown_ids[draft.text] = evidence_id
                continue
            anchors.append(item.to_anchor())
        claims.append(Claim(text=draft.text, anchors=anchors))

    report = gate.check(claims, references=references)

    record = AnswerRecord(
        answer_id=new_answer_id(pack.question),
        question=pack.question,
        claims=[RecordedClaim.from_anchors(c.text, c.anchors) for c in report.accepted],
        references=dict(references or {}),
        corpus_hash=pack.corpus_hash,
        created_at=utc_now(),
    )

    for rejection in report.rejected:
        detail = rejection.detail
        if rejection.claim.text in unknown_ids:
            detail = f"cited evidence id {unknown_ids[rejection.claim.text]!r}, which was not offered"
        record.dropped.append(
            {"claim": rejection.claim.text, "reason": rejection.reason, "detail": detail}
        )
    for key, reason in report.dropped_references:
        record.dropped.append({"claim": "", "reason": "unresolved-reference", "detail": f"{key}: {reason}"})

    for entry in equations or []:
        record.equations.append(_check_equation_entry(entry))

    if answers_dir is not None:
        record.save(answers_dir)
    return record


def _check_equation_entry(entry: dict[str, Any]) -> RecordedEquation:
    """Parse and dimension-check one equation offered for an answer."""
    latex = entry.get("latex", "")
    parsed = parse_equation(latex)
    check = check_equation(parsed, entry.get("units"))
    return RecordedEquation(
        latex=latex,
        paper_id=entry.get("paper_id", ""),
        number=entry.get("number"),
        parsed=parsed.ok,
        dimensions=check.verdict.value,
        detail=check.detail or parsed.error,
        notes=list(parsed.notes),
    )


def render_markdown(record: AnswerRecord) -> str:
    """Render a record for a terminal or a document.

    Anchors are shown inline, because a citation the reader cannot check on the
    spot is only marginally better than no citation at all.
    """
    lines = [f"# {record.question}", ""]

    if not record.claims:
        lines += [
            "_No claim in this answer survived verification._",
            "",
            "That is a result, not an error: the corpus does not support an answer to",
            "this question. Ingest more papers, or ask something the corpus covers.",
            "",
        ]

    for index, claim in enumerate(record.claims, start=1):
        lines.append(f"{index}. {claim.text}")
        for anchor in claim.anchors:
            quote = anchor["quote"]
            trimmed = quote if len(quote) <= 160 else quote[:157] + "..."
            location = anchor["section_path"] or "no section"
            lines.append(f"   > {trimmed}")
            lines.append(
                f"   -- {anchor['paper_id']}, {location}, "
                f"chars {anchor['start']}-{anchor['end']}"
            )
        lines.append("")

    if record.equations:
        lines += ["## Equations", ""]
        for equation in record.equations:
            status = "parsed" if equation.parsed else "did not parse"
            lines.append(f"- `{equation.latex}`")
            lines.append(f"  {status}; dimensions: {equation.dimensions}")
            if equation.detail:
                lines.append(f"  {equation.detail}")
            for note in equation.notes:
                lines.append(f"  note: {note}")
        lines.append("")

    if record.references:
        resolved = record.resolved_count
        lines += ["## References", "", f"{resolved}/{len(record.references)} resolved"]
        for key, ok in sorted(record.references.items()):
            lines.append(f"- {'OK ' if ok else 'UNRESOLVED'} {key}")
        lines.append("")

    if record.dropped:
        lines += ["## Dropped by the gate", ""]
        for item in record.dropped:
            subject = item["claim"][:80] or item["detail"]
            lines.append(f"- [{item['reason']}] {subject}")
            if item["claim"] and item["detail"]:
                lines.append(f"  {item['detail']}")
        lines.append("")

    # The scope statement sits beside the claims, not buried in the docs:
    # an anchored claim is one whose evidence exists, not one shown to follow.
    lines += [
        "---",
        "Checked: every quote above is verbatim in the cited paper and every span resolves.",
        "NOT checked: whether each quote supports the claim drawn from it. A claim can",
        "contradict its own evidence and still appear here. Read the quotes.",
        "",
        f"corpus {record.corpus_hash[:12]} · {record.summary()}",
    ]
    return "\n".join(lines)
