"""Assembling an explanation from checked parts.

Nothing here is generated. The prose comes from the curated library; this module
adds the checking a machine can do better than a person can do repeatedly:

* parse and dimension-check every equation,
* **recompute every worked example** and compare against the stored answer, so a
  typo in the library shows up as a mismatch rather than as a confident wrong
  number,
* run the linked simulation,
* look for the concept in the user's own corpus.

That last point is the honest one: if the corpus has nothing on the topic, the
explanation says so rather than implying it drew on papers it never read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..compute.run import evaluate
from ..compute.symbolic import parse_equation
from ..compute.units import check_equation
from .concepts import CheckedEquation, CheckedExample, Concept, Explanation, WorkedExample
from .library import find, search

#: How far a recomputed example may sit from its stored answer before the
#: library entry is reported as wrong. Loose enough for rounded constants in the
#: stored value, tight enough to catch a real error.
TOLERANCE = 5e-3


def check_equations(concept: Concept) -> list[CheckedEquation]:
    """Parse and dimension-check every equation in a concept."""
    checked: list[CheckedEquation] = []
    for latex, says, units in concept.equations:
        parsed = parse_equation(latex)
        # Same alignment the worked examples need: a library author writes T_1,
        # the parser emits T_{1}. Without this a fully-declared equation reports
        # "unknown", which under our own rules means "we did not check" -- the
        # worst possible answer for an equation we could have checked.
        verdict = check_equation(parsed, _align_names(parsed, units) if units else None)
        checked.append(
            CheckedEquation(
                latex=latex,
                says=says,
                parsed=parsed.ok,
                dimensions=verdict.verdict.value,
                detail=verdict.detail or parsed.error,
                notes=list(parsed.notes),
            )
        )
    return checked


def check_example(example: WorkedExample) -> CheckedExample:
    """Recompute a worked example and compare it with the stored answer."""
    result = CheckedExample(example=example)

    parsed = parse_equation(example.equation)
    if not parsed.ok:
        result.error = parsed.error or "equation did not parse"
        return result

    # The parser renders a subscript as T_{1}; a library author writes T_1.
    # Align the two rather than making every entry guess the convention.
    values = _align_names(parsed, example.values())
    units = _align_names(parsed, example.unit_map())

    evaluation = evaluate(parsed, values, units, require_dimensions=False)
    if not evaluation.ok:
        result.error = evaluation.error
        return result

    value = evaluation.value
    result.computed = float(value.real if isinstance(value, complex) else value)

    if example.expected is not None:
        scale = max(abs(example.expected), 1e-30)
        result.matches_expected = (
            abs(result.computed - example.expected) / scale <= TOLERANCE
        )
    return result


def _canonical(name: str) -> str:
    """Strip subscript braces so T_1 and T_{1} compare equal."""
    return name.replace("{", "").replace("}", "").replace(" ", "")


def _align_names(parsed: Any, supplied: dict[str, Any]) -> dict[str, Any]:
    """Rename supplied values onto the symbol names the parser actually produced."""
    symbols = {str(s) for s in getattr(parsed.expr, "free_symbols", set())}
    lookup = {_canonical(s): s for s in symbols}

    aligned: dict[str, Any] = {}
    for name, value in supplied.items():
        aligned[lookup.get(_canonical(name), name)] = value
    return aligned


def compose(
    term: str,
    library: Any = None,
    run_simulation: bool = True,
    figures_dir: Path | None = None,
) -> Explanation | None:
    """Build a checked explanation of ``term``, or None when it is unknown."""
    concept = find(term)
    if concept is None:
        return None

    explanation = Explanation(
        concept=concept,
        equations=check_equations(concept),
        examples=[check_example(e) for e in concept.examples],
    )

    if run_simulation and concept.simulation:
        from ..sim import engine

        result = engine.run(concept.simulation, concept.simulation_parameters)
        explanation.simulation = result
        if result.ok and figures_dir is not None:
            path = engine.plot(result, Path(figures_dir) / f"{concept.key}.png")
            if path:
                explanation.figures.append(str(path))

    # Only claim corpus support when the corpus actually has something.
    if library is not None:
        try:
            hits = library.search_corpus(f"{concept.name} {concept.definition}", limit=4)
        except Exception:
            hits = []
        explanation.corpus_papers = [
            {
                "paper_id": hit.paper_id,
                "section": hit.chunk.section_path,
                "start": hit.span.start,
                "end": hit.span.end,
                "quote": hit.chunk.text[:400],
            }
            for hit in hits
        ]

    return explanation


def suggestions(term: str) -> list[str]:
    """Concept names to offer when a lookup misses."""
    return [c.name for c in search(term)] or []


def render(explanation: Explanation, depth: str = "full") -> str:
    """Render an explanation as text.

    ``depth`` is "brief" (definition and analogy), "standard" (adds detail,
    equations, one example), or "full" (everything).
    """
    concept = explanation.concept
    lines = [concept.name, "=" * len(concept.name), "", concept.definition, ""]

    lines += ["In everyday terms", "-----------------", concept.everyday, ""]
    if depth == "brief":
        return "\n".join(lines)

    if concept.detail:
        lines += ["What is going on", "----------------"]
        lines += [paragraph + "\n" for paragraph in concept.detail]

    if explanation.equations:
        lines += ["The mathematics", "---------------"]
        for equation in explanation.equations:
            lines.append(f"  {equation.latex}")
            lines.append(f"    {equation.says}")
            status = "parsed" if equation.parsed else f"DID NOT PARSE - {equation.detail}"
            # Parentheses, not square brackets: Rich reads [...] as markup and
            # would swallow this line entirely.
            lines.append(f"    ({status}; dimensions: {equation.dimensions})")
            for note in equation.notes:
                lines.append(f"    note: {note}")
            lines.append("")

    examples = explanation.examples if depth == "full" else explanation.examples[:1]
    if examples:
        lines += ["Worked example", "--------------"]
        for checked in examples:
            example = checked.example
            lines.append(f"  {example.question}")
            for name, (value, units) in example.given.items():
                lines.append(f"    given {name} = {value:g} {units}")
            for step in example.steps:
                lines.append(f"    {step}")
            if checked.ok:
                lines.append(f"    => {checked.computed:.6g} {example.units}")
                if checked.matches_expected is False:
                    lines.append(f"    (!) {checked.discrepancy()}")
            else:
                lines.append(f"    (!) could not recompute: {checked.error}")
            if example.sanity:
                lines.append(f"    Sanity check: {example.sanity}")
            lines.append("")

    simulation = explanation.simulation
    if simulation is not None and getattr(simulation, "ok", False):
        lines += ["Simulation", "----------"]
        lines.append(f"  {simulation.summary()}")
        for key, value in list(simulation.derived.items())[:5]:
            if value is not None:
                lines.append(f"    {key} = {value}")
        for path in explanation.figures:
            lines.append(f"    plot: {path}")
        lines.append("")

    if depth == "full" and concept.misconceptions:
        lines += ["Commonly got wrong", "------------------"]
        lines += [f"  - {item}" for item in concept.misconceptions]
        lines.append("")

    if depth == "full" and simulation is not None and getattr(simulation, "caveats", None):
        lines += ["What the model leaves out", "-------------------------"]
        lines += [f"  - {item}" for item in simulation.caveats]
        lines.append("")

    if explanation.corpus_papers:
        lines += ["From your corpus", "----------------"]
        for paper in explanation.corpus_papers[:3]:
            lines.append(f"  {paper['paper_id']} (chars {paper['start']}-{paper['end']})")
            lines.append(f"    {paper['quote'][:180].strip()}")
        lines.append("")
    elif depth == "full":
        lines += [
            "From your corpus",
            "----------------",
            "  Nothing ingested on this topic. This explanation comes from the built-in",
            "  concept library, not from papers. To ground it in literature:",
            f"    noether search \"{concept.reading_query}\"",
            "",
        ]

    if concept.related:
        lines.append("Related: " + ", ".join(concept.related))
    return "\n".join(lines)
