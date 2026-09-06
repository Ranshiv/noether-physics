"""MCP server -- the interface Claude drives.

The division of labour is the whole design. The model reasons and writes; this
server retrieves, checks and records. The component that can hallucinate is not
the component that decides what ships.

That shape is visible in the tool set. There is no ``write_answer``. There is
``gather_evidence``, which hands back passages that provably exist, and
``submit_answer``, which judges what the model wrote against those same
passages and reports what it dropped. A model cannot talk its way past the gate,
because the gate re-reads the corpus rather than trusting the model's account
of it.

Run with::

    uv run python -m noether.mcp_server

and register it as a stdio MCP server in Claude Code or Claude Desktop.
"""

from __future__ import annotations

import json
from typing import Any

from .answer.pipeline import DraftClaim, render_markdown
from .library import Library
from .sources.arxiv import ArxivError

# The SDK renamed FastMCP to MCPServer in 2.x. Support both rather than pinning
# to one: a researcher's environment is not ours to dictate, and the tool
# definitions below are identical either way.
try:
    from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2
except ImportError:  # pragma: no cover - depends on the installed SDK
    try:
        from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x
    except ImportError as exc:
        raise SystemExit(
            "the MCP server needs the 'mcp' package: uv pip install -e '.[mcp]'"
        ) from exc

server = _Server("noether")

_library: Library | None = None


def library() -> Library:
    """One Library per process, opened on first use."""
    global _library
    if _library is None:
        _library = Library()
    return _library


@server.tool()
def search_arxiv(query: str, limit: int = 10) -> str:
    """Search arXiv. Query uses arXiv syntax, e.g. 'cat:quant-ph AND ti:Rabi'.

    Returns papers with their ids, titles, authors and licences. Use
    ``ingest_paper`` to bring one into the corpus before asking about it.
    """
    try:
        papers = library().search_arxiv(query, limit=limit)
    except ArxivError as exc:
        return f"arXiv error: {exc}"

    return json.dumps(
        [
            {
                "arxiv_id": p.arxiv_id,
                "version": p.version,
                "title": p.title,
                "authors": p.authors[:8],
                "category": p.primary_category,
                "doi": p.doi,
                "license": p.license,
                "abstract": p.abstract[:400],
            }
            for p in papers
        ],
        indent=2,
    )


@server.tool()
def ingest_paper(arxiv_id: str, refresh: bool = False) -> str:
    """Fetch a paper's LaTeX source and add it to the corpus.

    Reads the authors' actual LaTeX, so equations keep the numbers the authors
    gave them and citations keep their keys. Papers submitted as PDF only have
    no source and are reported as such rather than silently skipped.
    """
    result = library().ingest_arxiv(arxiv_id, refresh=refresh)
    return json.dumps(
        {
            "paper_id": result.paper_id,
            "title": result.title,
            "ok": result.ok,
            "error": result.error,
            "numbered_equations": result.numbered_equations,
            "figures": result.figures,
            "references": result.references,
            "macros_expanded": result.macros,
            "license": result.license,
        },
        indent=2,
    )


@server.tool()
def corpus_status() -> str:
    """List the papers currently ingested, with counts and sizes.

    Check this before answering a literature question: retrieval can only draw on
    what is here. An empty or thin corpus means the honest answer is to ingest
    more papers first, not to answer from memory.
    """
    lib = library()
    return json.dumps(
        {
            "stats": lib.stats(),
            "papers": [
                {"paper_id": r["paper_id"], "title": r["title"], "chars": len(r["text"])}
                for r in lib.papers()
            ],
        },
        indent=2,
    )


@server.tool()
def gather_evidence(question: str, limit: int = 12) -> str:
    """Retrieve citable passages for a question. **Call this before answering.**

    Each passage has an ``evidence_id``. To answer, write claims that cite those
    ids and pass them to ``submit_answer``. Do not quote the corpus from memory:
    a quote that is not literally present will be rejected, and paraphrasing one
    into a stronger statement is exactly the failure this tool exists to stop.

    If the returned passages do not answer the question, say so. An empty result
    means the corpus lacks the material -- ingest more papers first.
    """
    pack = library().evidence(question, limit=limit)
    return json.dumps(pack.to_dict(), indent=2)


@server.tool()
def submit_answer(
    question: str,
    claims: list[dict[str, Any]],
    equations: list[dict[str, Any]] | None = None,
    references: dict[str, bool] | None = None,
) -> str:
    """Submit drafted claims for verification, and record what survives.

    ``claims`` is a list of ``{"text": ..., "evidence_ids": [...]}``. Each claim
    is checked against the corpus: its cited spans must exist and their quoted
    text must match exactly. Claims that fail are **dropped** and listed in the
    result with the reason -- they are not softened or hedged into the answer.

    ``equations`` may hold ``{"latex": ..., "units": {...}}`` entries; each is
    parsed and dimensionally checked, and the verdict is recorded alongside it.

    Returns the rendered answer plus a record id. A thin answer is a real
    result: it means the evidence did not support more.
    """
    lib = library()
    pack = lib.evidence(question)
    drafts = [
        DraftClaim(text=c.get("text", ""), evidence_ids=list(c.get("evidence_ids", [])))
        for c in claims
    ]
    record = lib.submit_answer(pack, drafts, equations, references)
    return json.dumps(
        {
            "answer_id": record.answer_id,
            "summary": record.summary(),
            "accepted": len(record.claims),
            "dropped": record.dropped,
            "fully_resolved": record.fully_resolved,
            "markdown": render_markdown(record),
        },
        indent=2,
    )


@server.tool()
def list_equations(paper_id: str) -> str:
    """List a paper's equations, numbered as the authors numbered them.

    'Equation 14' means the equation the authors labelled 14, not the fourteenth
    one found -- starred environments and \\nonumber rows do not take a number.
    """
    return json.dumps(
        [
            {
                "number": row["number"],
                "label": row["label"],
                "latex": row["latex"],
                "section": row["section_path"],
            }
            for row in library().equations(paper_id)
        ],
        indent=2,
    )


@server.tool()
def check_equation(latex: str, units: dict[str, str] | None = None) -> str:
    """Parse an equation and check it dimensionally.

    ``units`` maps symbol names to units, e.g. ``{"E": "joule", "m": "kilogram"}``.
    Most physics symbols are context-dependent -- ``h`` is Planck's constant or a
    height, ``T`` a period or a temperature -- so an undeclared symbol yields
    ``unknown`` with a hint, never a guess. ``unknown`` is not a pass.
    """
    return json.dumps(library().check_equation(latex, units), indent=2)


@server.tool()
def run_equation(
    latex: str,
    values: dict[str, float],
    units: dict[str, str] | None = None,
) -> str:
    """Evaluate an equation numerically at the given values.

    Refuses to evaluate an equation already shown to be dimensionally
    inconsistent. Physical constants (hbar, c, k_B, ...) are supplied
    automatically; passing an explicit value overrides them, so natural units work.
    """
    from .compute.run import evaluate
    from .compute.symbolic import parse_equation

    parsed = parse_equation(latex)
    result = evaluate(parsed, values, units)
    return json.dumps(
        {
            "latex": latex,
            "value": result.value,
            "ok": result.ok,
            "error": result.error,
            "dimensions": result.dimensions.verdict.value if result.dimensions else "unknown",
            "notes": result.notes,
        },
        indent=2,
    )


@server.tool()
def audit_references(paper_id: str, offline: bool = False) -> str:
    """Resolve every reference of a stored paper and report the problems.

    Findings are: ``unresolvable`` (no provider has such a record -- the
    hallucinated-citation case), ``mismatch`` (resolves, but to something with a
    different title), ``no-identifier`` (matched only by text similarity), and
    ``cited-unread`` (listed but never cited in the text).
    """
    findings = library().audit_paper(paper_id, offline=offline)
    return json.dumps(
        [
            {
                "key": f.key,
                "kind": f.kind,
                "severity": f.severity,
                "detail": f.detail,
                "resolved_title": f.resolution.title if f.resolution else None,
                "confidence": f.resolution.confidence if f.resolution else None,
            }
            for f in findings
        ],
        indent=2,
    )


@server.tool()
def search_corpus(query: str, limit: int = 8) -> str:
    """Search inside ingested papers, returning passages with their spans."""
    return json.dumps(
        [
            {
                "paper_id": hit.paper_id,
                "section": hit.chunk.section_path,
                "start": hit.span.start,
                "end": hit.span.end,
                "text": hit.chunk.text,
                "score": round(hit.score, 6),
            }
            for hit in library().search_corpus(query, limit=limit)
        ],
        indent=2,
    )




@server.tool()
def list_concepts() -> str:
    """List the physics concepts the explainer knows, with their simulations.

    Use ``explain_concept`` for any of these. The explanations are curated and
    machine-checked, not generated, so they are a reliable base to build on.
    """
    from .explain.library import CONCEPTS

    return json.dumps(
        [
            {
                "key": c.key,
                "name": c.name,
                "domain": c.domain,
                "aliases": c.aliases,
                "definition": c.definition,
                "simulation": c.simulation,
            }
            for c in CONCEPTS
        ],
        indent=2,
    )


@server.tool()
def explain_concept(topic: str, run_simulation: bool = True) -> str:
    """Explain a physics concept, with an everyday analogy, checked equations,
    a recomputed worked example, and a live simulation.

    The prose is curated rather than generated. Every equation has been parsed
    and dimension-checked, every worked example recomputed against its stored
    answer, and the simulation actually run. Present this material as given; if
    a check failed it is reported, and you should pass that on rather than
    smoothing it over.

    ``corpus_papers`` is empty when nothing on the topic has been ingested. Say
    so rather than implying the explanation came from the literature.
    """
    from .explain.compose import compose, suggestions

    lib = library()
    explanation = compose(
        topic, library=lib, run_simulation=run_simulation,
        figures_dir=lib.config.figures_dir,
    )
    if explanation is None:
        return json.dumps(
            {"found": False, "topic": topic, "suggestions": suggestions(topic)}, indent=2
        )

    concept = explanation.concept
    simulation = explanation.simulation
    return json.dumps(
        {
            "found": True,
            "key": concept.key,
            "name": concept.name,
            "definition": concept.definition,
            "everyday_analogy": concept.everyday,
            "detail": concept.detail,
            "equations": [
                {
                    "latex": e.latex, "says": e.says, "parsed": e.parsed,
                    "dimensions": e.dimensions, "detail": e.detail, "notes": e.notes,
                }
                for e in explanation.equations
            ],
            "worked_examples": [
                {
                    "question": c.example.question,
                    "given": {k: {"value": v, "units": u} for k, (v, u) in c.example.given.items()},
                    "steps": c.example.steps,
                    "answer": c.computed,
                    "units": c.example.units,
                    "sanity_check": c.example.sanity,
                    "recomputation_matches": c.matches_expected,
                    "error": c.error,
                }
                for c in explanation.examples
            ],
            "misconceptions": concept.misconceptions,
            "simulation": simulation.to_dict() if simulation is not None else None,
            "figures": explanation.figures,
            "corpus_papers": explanation.corpus_papers,
            "further_reading_query": concept.reading_query,
            "related": concept.related,
            "all_checks_passed": explanation.all_checks_passed,
        },
        indent=2,
    )


@server.tool()
def list_simulations() -> str:
    """List every simulation model, with its parameters, defaults and bounds."""
    from .sim import engine

    return json.dumps(engine.available(), indent=2)


@server.tool()
def run_simulation(model: str, parameters: dict[str, float] | None = None) -> str:
    """Run a named physics simulation and return the full time series.

    Parameters are validated against declared bounds before anything runs, so an
    impossible request fails immediately with a reason. Every run reports a
    convergence check performed by halving the step size: if ``converged`` is
    false, say so rather than quoting the numbers as settled.

    ``caveats`` lists what the model genuinely does not capture. Pass those on.
    """
    from pathlib import Path

    from .sim import engine

    result = engine.run(model, parameters or {})
    if result.ok:
        engine.plot(result, Path(library().config.figures_dir) / f"{model}.png")
    return json.dumps(result.to_dict(), indent=2)


# Keep this block last. A @server.tool() defined below it would never be
# registered when the module is run as a script, because main() blocks in
# server.run() before execution reaches it.
def main() -> None:
    # Same reason as the CLI: the Windows console defaults to cp1252 and a
    # physics author's name is enough to crash it. json.dumps escapes non-ASCII
    # by default, but any traceback or log line written here would not.
    from .cli import _utf8_stdio

    _utf8_stdio()
    server.run()


if __name__ == "__main__":
    main()
