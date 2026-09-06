"""Acceptance gates -- the measured numbers.

The project's claims are only worth what these report. Each gate writes its
result to ``research/results/`` with the code version and the corpus it ran
against, so a number in the README can always be traced to a run.

A gate that has not been run reports **unmeasured**. It never reports a default,
an estimate, or a previous run's figure, because an unmeasured gate presented as
passing is the same class of error as an unresolvable citation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .compute.symbolic import parse_equation
from .compute.units import check_equation


def research_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "research"


@dataclass
class GateResult:
    """One benchmark run."""

    gate: str
    measured: bool
    value: float | None = None
    threshold: float | None = None
    total: int = 0
    passed: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""
    noether_version: str = __version__
    measured_at: str = ""

    @property
    def meets_threshold(self) -> bool | None:
        """None when unmeasured -- which is not the same as False."""
        if not self.measured or self.value is None or self.threshold is None:
            return None
        return self.value >= self.threshold

    def summary(self) -> str:
        if not self.measured:
            return f"{self.gate}: UNMEASURED ({self.detail})"
        verdict = {True: "PASS", False: "FAIL", None: "no threshold"}[self.meets_threshold]
        return (
            f"{self.gate}: {self.value:.1%} ({self.passed}/{self.total}) "
            f"threshold {self.threshold:.0%} -> {verdict}"
        )

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or (research_dir() / "results")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.gate}.json"
        payload = asdict(self)
        payload["measured_at"] = self.measured_at or datetime.now(UTC).isoformat(
            timespec="seconds"
        )
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


def load_result(gate: str, directory: Path | None = None) -> GateResult:
    """Read a previous run, or an explicitly unmeasured result."""
    path = (directory or (research_dir() / "results")) / f"{gate}.json"
    if not path.exists():
        return GateResult(gate=gate, measured=False, detail="never run")
    return GateResult(**json.loads(path.read_text(encoding="utf-8")))


# ---- gate 1: equations -------------------------------------------------

def equation_benchmark_path() -> Path:
    return research_dir() / "benchmarks" / "equations.yaml"


def run_equation_gate(threshold: float = 0.90, path: Path | None = None) -> GateResult:
    """Fraction of labelled equations that parse *and* dimension-check correctly.

    An equation counts as passing only when the outcome matches the label. A
    deliberately wrong equation that we call consistent is a failure, exactly as
    a correct one we call inconsistent is -- both directions matter.
    """
    path = path or equation_benchmark_path()
    if not path.exists():
        return GateResult("equations", False, detail=f"no benchmark at {path}")

    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = spec.get("equations", [])
    if not cases:
        return GateResult("equations", False, detail="benchmark is empty")

    failures: list[dict[str, Any]] = []
    passed = 0

    for case in cases:
        latex = case["latex"]
        expect_parse = case.get("parses", True)
        expect_dimensions = case.get("dimensions")
        units = case.get("units")

        parsed = parse_equation(latex)
        check = check_equation(parsed, units)

        problems = []
        if parsed.ok != expect_parse:
            problems.append(
                f"parse: expected {expect_parse}, got {parsed.ok} ({parsed.error[:60]})"
            )
        if expect_dimensions and check.verdict.value != expect_dimensions:
            problems.append(f"dimensions: expected {expect_dimensions}, got {check.verdict.value}")

        if problems:
            failures.append({"latex": latex, "problems": problems})
        else:
            passed += 1

    return GateResult(
        gate="equations",
        measured=True,
        value=passed / len(cases),
        threshold=threshold,
        total=len(cases),
        passed=passed,
        failures=failures,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


# ---- gate 2: retrieval -------------------------------------------------

def retrieval_benchmark_path() -> Path:
    return research_dir() / "benchmarks" / "retrieval.yaml"


def run_retrieval_gate(library: Any, threshold: float = 0.80, path: Path | None = None) -> GateResult:
    """Fraction of questions whose known-relevant paper appears in the top k.

    Needs the benchmark's papers ingested first; if they are not, the gate says
    so rather than scoring against a corpus that cannot contain the answer.
    """
    path = path or retrieval_benchmark_path()
    if not path.exists():
        return GateResult("retrieval", False, detail=f"no benchmark at {path}")

    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = spec.get("questions", [])
    if not questions:
        return GateResult("retrieval", False, detail="benchmark is empty")

    present = set(library.store.paper_ids())
    required = {q["expect_paper"] for q in questions}
    missing = sorted(required - present)
    if missing:
        return GateResult(
            "retrieval", False,
            detail=f"corpus is missing {len(missing)} benchmark papers: {', '.join(missing[:3])}",
        )

    k = spec.get("top_k", 10)

    # A top_k at or above the corpus size returns everything, so every question
    # "passes" and the gate measures nothing. Refuse rather than report a
    # flattering number.
    if k >= len(present):
        return GateResult(
            "retrieval", False,
            detail=(
                f"top_k={k} with only {len(present)} papers in the corpus: every "
                "paper would be returned, so the result would be vacuous. Ingest "
                "more papers or lower top_k."
            ),
        )

    failures: list[dict[str, Any]] = []
    passed = 0

    for question in questions:
        hits = library.search_corpus(question["question"], limit=k)
        found = {hit.paper_id for hit in hits}
        if question["expect_paper"] in found:
            passed += 1
        else:
            failures.append(
                {"question": question["question"], "expected": question["expect_paper"],
                 "got": sorted(found)[:5]}
            )

    return GateResult(
        gate="retrieval", measured=True, value=passed / len(questions), threshold=threshold,
        total=len(questions), passed=passed, failures=failures,
        # The corpus size belongs with the number. 'Right paper in the top 3'
        # means something entirely different over 8 papers than over 60, and a
        # bare percentage invites the reader to forget that.
        detail=f"top-{k} of {len(present)} papers",
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


# ---- gate 3: citations -------------------------------------------------

def run_citation_gate(library: Any, paper_ids: list[str], threshold: float = 0.85) -> GateResult:
    """Resolution coverage: the fraction of references we can tie to a record.

    **This is not guarantee 1, and the threshold is deliberately not 1.0.**

    An earlier version set it to 1.0 on the reasoning that "nothing unresolvable
    is ever emitted". That conflated two different things. Guarantee 1 is
    enforced at the emission gate, which *drops* references that do not resolve;
    it does not require that every reference in every bibliography can be
    resolved in the first place.

    And many genuinely cannot. Measured on the nEDM letter, the residue was a
    1954 Danish journal, a 1955 Pauli chapter, a 1982 Oxford book, a proceedings
    contribution, and one reference carrying a volume typo in the published
    paper itself. Books, theses, pre-digital articles and private communications
    are simply not in Crossref, OpenAlex or INSPIRE. A gate demanding 100% would
    read FAIL forever while telling nobody anything.

    So this measures coverage, and the failures it lists are worth reading: they
    are the references a human has to check by hand.
    """
    if not paper_ids:
        return GateResult("citations", False, detail="no papers given")

    total = 0
    resolved = 0
    failures: list[dict[str, Any]] = []

    for paper_id in paper_ids:
        references = library.store.references(paper_id)
        if not references:
            continue
        findings = library.audit_paper(paper_id)
        unresolvable = {f.key for f in findings if f.kind == "unresolvable"}
        total += len(references)
        resolved += len(references) - len(unresolvable)
        failures.extend(
            {"paper_id": paper_id, "key": key,
             "detail": next(f.detail for f in findings if f.key == key)}
            for key in sorted(unresolvable)
        )

    if not total:
        return GateResult("citations", False, detail="no references in the given papers")

    return GateResult(
        gate="citations", measured=True, value=resolved / total, threshold=threshold,
        total=total, passed=resolved, failures=failures[:50],
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def status(directory: Path | None = None) -> list[GateResult]:
    """Every gate's last known result, unmeasured ones included."""
    return [load_result(gate, directory) for gate in ("equations", "retrieval", "citations")]
