"""The shape of an explanation.

A concept entry is **curated data, not generated text**. That is the whole
point. Textbook physics is settled, and writing it down once, carefully, is more
trustworthy than regenerating it on every request and hoping.

What the machine contributes is the part a machine is actually better at:

* every equation is parsed and dimension-checked before it is shown,
* every worked example is *recomputed*, and the stored answer is compared with
  the computed one, so a typo in the library surfaces as a mismatch,
* the linked simulation is really run.

So an explanation is assembled from checked parts rather than asserted. If a
worked example stops matching its own arithmetic, the reader is told.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkedExample:
    """A concrete calculation with real numbers."""

    question: str
    #: symbol -> (value, units). Fed straight into the evaluator.
    given: dict[str, tuple[float, str]] = field(default_factory=dict)
    #: The equation to evaluate, in LaTeX. Solved for its left-hand side.
    equation: str = ""
    steps: list[str] = field(default_factory=list)
    #: What the answer should come to, for checking the library against itself.
    expected: float | None = None
    units: str = ""
    #: Why the number is believable -- an order-of-magnitude anchor.
    sanity: str = ""

    def values(self) -> dict[str, float]:
        return {name: value for name, (value, _units) in self.given.items()}

    def unit_map(self) -> dict[str, str]:
        return {name: units for name, (_value, units) in self.given.items()}


@dataclass
class Concept:
    """One physics idea, explained."""

    key: str
    name: str
    #: Other names a user might ask for.
    aliases: list[str] = field(default_factory=list)
    domain: str = "classical"
    #: One sentence. What it is.
    definition: str = ""
    #: The analogy. Something the reader has physically experienced.
    everyday: str = ""
    #: Two or three paragraphs of real explanation.
    detail: list[str] = field(default_factory=list)
    #: (latex, what it says in words, unit declarations for checking)
    equations: list[tuple[str, str, dict[str, str]]] = field(default_factory=list)
    examples: list[WorkedExample] = field(default_factory=list)
    #: Model key in noether.sim, when one illustrates this.
    simulation: str | None = None
    simulation_parameters: dict[str, float] = field(default_factory=dict)
    #: Common misunderstandings, stated as the correction.
    misconceptions: list[str] = field(default_factory=list)
    #: arXiv query for further reading, and known landmark papers.
    reading_query: str = ""
    landmark_papers: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)

    def matches(self, term: str) -> bool:
        term = term.lower().strip()
        return term == self.key or term == self.name.lower() or term in [a.lower() for a in self.aliases]


@dataclass
class CheckedEquation:
    """An equation after it has been through the physics checks."""

    latex: str
    says: str
    parsed: bool
    dimensions: str
    detail: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class CheckedExample:
    """A worked example after recomputation."""

    example: WorkedExample
    computed: float | None = None
    matches_expected: bool | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.computed is not None and not self.error

    def discrepancy(self) -> str:
        if self.matches_expected is not False or self.example.expected is None:
            return ""
        return (
            f"library says {self.example.expected:g}, recomputation gives "
            f"{self.computed:g} — the library entry is wrong and should be fixed"
        )


@dataclass
class Explanation:
    """An assembled, checked explanation."""

    concept: Concept
    equations: list[CheckedEquation] = field(default_factory=list)
    examples: list[CheckedExample] = field(default_factory=list)
    simulation: Any = None
    #: Papers found in the local corpus, with their spans. May be empty.
    corpus_papers: list[dict[str, Any]] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)

    @property
    def all_checks_passed(self) -> bool:
        return (
            all(e.parsed for e in self.equations)
            and all(e.matches_expected is not False for e in self.examples)
        )

    def summary(self) -> str:
        parts = [f"{len(self.equations)} equations"]
        checked = sum(1 for e in self.equations if e.dimensions == "consistent")
        if checked:
            parts.append(f"{checked} dimensionally consistent")
        if self.examples:
            good = sum(1 for e in self.examples if e.matches_expected is not False)
            parts.append(f"{good}/{len(self.examples)} worked examples verified")
        if self.simulation is not None and getattr(self.simulation, "ok", False):
            parts.append("simulation ran")
        return " · ".join(parts)
