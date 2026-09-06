"""Command line entry point.

Commands are thin: parse arguments, call :mod:`noether.library`, render. Every
behaviour worth testing lives below this layer, so the CLI and the MCP server
stay two faces of one engine rather than two implementations of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from . import bench as benchmarks
from . import draft as draft_module
from . import export as export_module
from .answer.pipeline import DraftClaim, render_markdown
from .answer.records import AnswerRecord
from .compute.plots import plot_sweep
from .compute.run import sweep as run_sweep
from .compute.symbolic import parse_equation
from .config import Config
from .library import Library
from .maps import build_map
from .sources.arxiv import ArxivError
from .sources.registry import load_registry
from .watch import WatchList

app = typer.Typer(
    name="noether",
    help="A verifiable research-paper assistant for physics.",
    no_args_is_help=True,
    add_completion=False,
)
sources_app = typer.Typer(help="Inspect the configured data sources.")
app.add_typer(sources_app, name="sources")


def _utf8_stdio() -> None:
    """Force UTF-8 on stdout/stderr.

    The Windows console still defaults to cp1252, which cannot encode a great
    many physics author names -- Turkish dotted I, Polish crossed l, Scandinavian
    and Czech diacritics all appear routinely. Whether output renders must not
    depend on who wrote the paper.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass  # a redirected stream may refuse; rich escapes instead


_utf8_stdio()
console = Console()


# ---- basics ------------------------------------------------------------

@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"noether {__version__}")


@app.command()
def config() -> None:
    """Show where noether keeps its data."""
    cfg = Config.from_env()
    table = Table(show_header=False, box=None)
    for label, value in (
        ("data root", str(cfg.root)),
        ("cache", str(cfg.cache_dir)),
        ("database", str(cfg.db_path)),
        ("figures", str(cfg.figures_dir)),
        ("answers", str(cfg.answers_dir)),
        ("cassettes", f"{cfg.cassettes()}  (mode: {cfg.cassette_mode})"),
        ("root exists", "yes" if cfg.root.exists() else "no  (run: noether init)"),
    ):
        table.add_row(label, value)
    console.print(table)


@app.command()
def init() -> None:
    """Create the data root and its subdirectories."""
    cfg = Config.from_env()
    cfg.ensure_dirs()
    Library(cfg).close()
    console.print(f"[green]ready[/green]  {cfg.root}")


@sources_app.command("list")
def sources_list() -> None:
    """List every source the tool is allowed to call."""
    table = Table(title="Source registry")
    for column in ("key", "name", "rate", "auth", "status", "verified"):
        table.add_column(column)
    for key, spec in sorted(load_registry().items()):
        colour = "green" if spec.status == "contract-verified" else "yellow"
        table.add_row(
            key, spec.name, f"{spec.min_interval_s:g}s", spec.auth,
            f"[{colour}]{spec.status}[/{colour}]", spec.verified_on or "[dim]never[/dim]",
        )
    console.print(table)


# ---- corpus ------------------------------------------------------------

@app.command()
def search(
    query: str = typer.Argument(..., help="arXiv query, e.g. 'cat:quant-ph AND ti:Rabi'"),
    limit: int = typer.Option(10, "--limit", "-n"),
) -> None:
    """Search arXiv for papers."""
    library = Library()
    try:
        papers = library.search_arxiv(query, limit=limit)
    except ArxivError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not papers:
        console.print("[yellow]no matches[/yellow]")
        return

    table = Table(show_lines=True)
    table.add_column("arXiv", style="cyan", no_wrap=True)
    table.add_column("title")
    table.add_column("authors", max_width=26)
    table.add_column("cat", no_wrap=True)
    for paper in papers:
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += f" +{len(paper.authors) - 3}"
        table.add_row(paper.versioned_id, paper.title, authors, paper.primary_category or "")
    console.print(table)


@app.command()
def ingest(
    arxiv_ids: list[str] = typer.Argument(..., help="arXiv ids to add to the corpus"),
    refresh: bool = typer.Option(False, "--refresh", help="Re-ingest papers already stored."),
) -> None:
    """Fetch papers and add them to the corpus."""
    library = Library()
    for arxiv_id in arxiv_ids:
        result = library.ingest_arxiv(arxiv_id, refresh=refresh)
        colour = "green" if result.ok else "red"
        console.print(f"[{colour}]{'OK  ' if result.ok else 'FAIL'}[/{colour}] {result.summary()}")
        if result.ok and result.license:
            console.print(f"       licence: {result.license}")
    library.close()


@app.command()
def corpus() -> None:
    """Show what is in the corpus."""
    library = Library()
    stats = library.stats()
    console.print(
        "  ".join(f"[cyan]{v}[/cyan] {k}" for k, v in stats.items() if v or k == "papers")
    )
    papers = library.papers()
    if papers:
        table = Table(show_header=True)
        table.add_column("paper", style="cyan", no_wrap=True)
        table.add_column("title")
        table.add_column("chars", justify="right")
        for row in papers:
            table.add_row(row["paper_id"], row["title"][:64], str(len(row["text"])))
        console.print(table)
    library.close()


@app.command()
def find(
    query: str = typer.Argument(..., help="Free-text query over the ingested corpus"),
    limit: int = typer.Option(6, "--limit", "-n"),
) -> None:
    """Search inside ingested papers, returning citable passages."""
    library = Library()
    hits = library.search_corpus(query, limit=limit)
    if not hits:
        console.print("[yellow]nothing matched. Is the corpus empty? (noether corpus)[/yellow]")
        library.close()
        return
    for index, hit in enumerate(hits, start=1):
        text = hit.chunk.text.replace("\n", " ")
        console.print(f"[cyan]{index}.[/cyan] {hit.paper_id}  [dim]{hit.chunk.section_path}[/dim]")
        console.print(f"   {text[:220]}{'...' if len(text) > 220 else ''}")
        console.print(f"   [dim]chars {hit.span.start}-{hit.span.end} · rrf {hit.score:.4f} · {hit.ranks}[/dim]\n")
    library.close()


# ---- answering ---------------------------------------------------------

@app.command()
def evidence(
    question: str = typer.Argument(..., help="The question to gather evidence for"),
    limit: int = typer.Option(10, "--limit", "-n"),
    as_json: bool = typer.Option(False, "--json", help="Emit the pack as JSON."),
) -> None:
    """Gather citable evidence for a question.

    This is the first half of answering. The drafting is done by a model reading
    this pack; the gate then judges what it wrote against these same passages.
    """
    library = Library()
    pack = library.evidence(question, limit=limit)
    if as_json:
        console.print_json(json.dumps(pack.to_dict()))
        library.close()
        return

    console.print(f"[bold]{pack.question}[/bold]")
    console.print(f"[dim]corpus {pack.corpus_hash[:12]} · {len(pack.items)} passages[/dim]\n")
    for item in pack.items:
        console.print(f"[cyan]{item.evidence_id}[/cyan] {item.paper_id} [dim]{item.section_path}[/dim]")
        console.print(f"   {item.quote[:200].replace(chr(10), ' ')}...\n")
    console.print(f"[dim]{pack.contract}[/dim]")
    library.close()


@app.command()
def ask(
    question: str = typer.Argument(...),
    claims_file: Path = typer.Option(
        ..., "--claims", help="JSON file: [{\"text\": ..., \"evidence_ids\": [...]}]"
    ),
) -> None:
    """Judge drafted claims against the corpus and record the answer.

    The claims come from whatever wrote them; this command is the gate, not the
    author. Unanchored or unverifiable claims are dropped and reported.
    """
    library = Library()
    pack = library.evidence(question)
    drafts = [
        DraftClaim(text=item["text"], evidence_ids=item.get("evidence_ids", []))
        for item in json.loads(claims_file.read_text(encoding="utf-8"))
    ]
    record = library.submit_answer(pack, drafts)
    console.print(render_markdown(record))
    console.print(f"\n[dim]saved: {library.config.answers_dir / (record.answer_id + '.json')}[/dim]")
    library.close()


# ---- equations ---------------------------------------------------------

@app.command()
def equations(paper_id: str = typer.Argument(..., help="e.g. arXiv:2001.11966")) -> None:
    """List a paper's equations by the number its authors gave them."""
    library = Library()
    rows = library.equations(paper_id)
    if not rows:
        console.print(f"[yellow]no equations stored for {paper_id}[/yellow]")
        library.close()
        return
    table = Table(show_lines=True)
    table.add_column("eq", justify="right", no_wrap=True)
    table.add_column("label", no_wrap=True)
    table.add_column("latex")
    for row in rows:
        table.add_row(str(row["number"] or "-"), row["label"] or "", row["latex"][:88])
    console.print(table)
    library.close()


@app.command()
def eq(
    latex: str = typer.Argument(..., help="LaTeX, or 'arXiv:ID#N' for a stored equation"),
    units: str = typer.Option("", "--units", help="e.g. 'E=joule,m=kilogram,v=meter/second'"),
    vary: str = typer.Option("", "--vary", help="e.g. 'x=0..10' to sweep and plot"),
    at: str = typer.Option("", "--at", help="Fixed values, e.g. 'm=1,g=9.81'"),
) -> None:
    """Parse, dimension-check and optionally run an equation."""
    library = Library()
    source = latex

    if "#" in latex and latex.lower().startswith("arxiv:"):
        paper_id, _, number = latex.partition("#")
        row = library.equation(paper_id, int(number))
        if row is None:
            console.print(f"[red]{paper_id} has no equation numbered {number}[/red]")
            raise typer.Exit(1)
        source = row["latex"]
        console.print(f"[dim]{paper_id} equation {number}[/dim]")

    unit_map = _parse_pairs(units, as_float=False)
    result = library.check_equation(source, unit_map or None)

    console.print(f"[bold]{source}[/bold]")
    console.print(f"normalised : {result['normalised']}")
    if not result["parsed"]:
        console.print(f"[red]did not parse: {result['error']}[/red]")
        library.close()
        raise typer.Exit(1)
    if result["lhs"] is not None:
        operator = result["relation"] or "="
        console.print(f"parsed     : {result['lhs']}  {operator}  {result['rhs']}")
    console.print(f"symbols    : {', '.join(result['symbols']) or '(none)'}")

    verdict = result["dimensions"]
    colour = {"consistent": "green", "inconsistent": "red"}.get(verdict, "yellow")
    console.print(f"dimensions : [{colour}]{verdict}[/{colour}] {result['detail']}")
    if result["hint"]:
        console.print(result["hint"])
    for note in result["notes"]:
        console.print(f"[yellow]note[/yellow]       : {note}")

    if vary:
        _do_sweep(library, source, vary, at, unit_map)
    library.close()


def _do_sweep(library: Library, latex: str, vary: str, at: str, units: dict[str, str]) -> None:
    """Run a parameter sweep and write the plot."""
    variable, _, span = vary.partition("=")
    start_text, _, stop_text = span.partition("..")
    try:
        start, stop = float(start_text), float(stop_text)
    except ValueError:
        console.print(f"[red]--vary wants 'name=start..stop'; got {vary!r}[/red]")
        return

    parsed = parse_equation(latex)
    result = run_sweep(
        parsed, variable.strip(), start, stop,
        fixed=_parse_pairs(at, as_float=True), units=units or None,
    )
    if not result.finite:
        console.print("[red]no sweep point evaluated; supply the missing values with --at[/red]")
        return

    path = library.config.figures_dir / f"sweep_{variable.strip()}.png"
    figure = plot_sweep(result, path, equation=latex, xlabel=variable.strip())
    console.print(f"swept      : {figure.caption}")
    console.print(f"plot       : {figure.path}")


def _parse_pairs(text: str, as_float: bool) -> dict:
    """Parse 'a=1,b=2' into a dict."""
    out: dict = {}
    for chunk in (c for c in text.split(",") if c.strip()):
        key, _, value = chunk.partition("=")
        if not value:
            continue
        out[key.strip()] = float(value) if as_float else value.strip()
    return out


# ---- audit -------------------------------------------------------------

@app.command()
def audit(
    target: str = typer.Argument(..., help="A .tex draft, a directory, or a corpus paper id"),
    offline: bool = typer.Option(False, "--offline", help="Resolve from cache only."),
) -> None:
    """Resolve every reference and report the problems.

    Given a path, audits your own draft without ingesting it -- run this before
    you submit. Given a paper id, audits a paper already in the corpus.
    """
    library = Library()

    candidate = Path(target)
    if candidate.exists():
        result = draft_module.audit_draft(candidate, library.store, library.net, offline=offline)
        console.print(draft_module.render_audit(result))
        library.close()
        raise typer.Exit(result.exit_code)

    paper_id = target
    findings = library.audit_paper(paper_id, offline=offline)
    references = library.store.references(paper_id)

    errors = [f for f in findings if f.severity == "error"]
    console.print(f"[bold]{paper_id}[/bold]  {len(references)} references")
    console.print(
        f"[{'red' if errors else 'green'}]{len(references) - len(errors)}/{len(references)} "
        f"resolved[/{'red' if errors else 'green'}]\n"
    )
    if not findings:
        console.print("[green]no problems found[/green]")
        library.close()
        return

    for finding in findings:
        colour = "red" if finding.severity == "error" else "yellow"
        console.print(f"[{colour}]{finding.kind:16}[/{colour}] {finding.key}")
        console.print(f"                 {finding.detail[:100]}")
    library.close()


if __name__ == "__main__":
    app()


# ---- benchmarks --------------------------------------------------------

bench_app = typer.Typer(help="Run the acceptance gates and record the numbers.")
app.add_typer(bench_app, name="bench")


def _render_gate(result: benchmarks.GateResult) -> None:
    colour = {True: "green", False: "red", None: "yellow"}[result.meets_threshold]
    console.print(f"[{colour}]{result.summary()}[/{colour}]")
    for failure in result.failures[:8]:
        console.print(f"  [dim]{json.dumps(failure)[:150]}[/dim]")


@bench_app.command("equations")
def bench_equations() -> None:
    """Score equation parsing and dimensional analysis against the benchmark."""
    result = benchmarks.run_equation_gate()
    _render_gate(result)
    if result.measured:
        console.print(f"[dim]saved: {result.save()}[/dim]")


@bench_app.command("retrieval")
def bench_retrieval() -> None:
    """Score retrieval: is the known-relevant paper in the top k?"""
    library = Library()
    result = benchmarks.run_retrieval_gate(library)
    _render_gate(result)
    if result.measured:
        console.print(f"[dim]saved: {result.save()}[/dim]")
    library.close()


@bench_app.command("citations")
def bench_citations(
    paper_ids: list[str] = typer.Argument(None, help="Defaults to the whole corpus."),
) -> None:
    """Score reference resolution. Hits the network; expect it to be slow."""
    library = Library()
    targets = paper_ids or library.store.paper_ids()
    result = benchmarks.run_citation_gate(library, targets)
    _render_gate(result)
    if result.measured:
        console.print(f"[dim]saved: {result.save()}[/dim]")
    library.close()


@bench_app.command("status")
def bench_status() -> None:
    """Show every gate's last result. Unmeasured gates say so."""
    table = Table(title="Acceptance gates")
    for column in ("gate", "result", "threshold", "verdict", "measured"):
        table.add_column(column)
    for result in benchmarks.status():
        if not result.measured:
            table.add_row(result.gate, "[yellow]unmeasured[/yellow]", "-", "-", result.detail)
            continue
        verdict = {True: "[green]PASS[/green]", False: "[red]FAIL[/red]", None: "-"}[
            result.meets_threshold
        ]
        table.add_row(
            result.gate, f"{result.value:.1%} ({result.passed}/{result.total})",
            f"{result.threshold:.0%}",
            verdict,
            # The scope a number was measured at is part of the number.
            result.measured_at[:10] + (f"  {result.detail}" if result.detail else ""),
        )
    console.print(table)


# ---- export, watch, maps -----------------------------------------------

@app.command()
def export(
    answer_id: str = typer.Argument(..., help="An answer id, e.g. ans_20260905..."),
    fmt: str = typer.Option("markdown", "--format", "-f", help="markdown | latex | jupyter"),
    out: Path = typer.Option(None, "--out", "-o", help="Directory to write into."),
) -> None:
    """Export a recorded answer.

    Nothing is verified here -- the record already passed the gate. Export only
    renders what survived it.
    """
    library = Library()
    path = library.config.answers_dir / f"{answer_id}.json"
    if not path.exists():
        console.print(f"[red]no answer {answer_id}; look in {library.config.answers_dir}[/red]")
        library.close()
        raise typer.Exit(1)

    record = AnswerRecord.load(path)
    titles = {r["paper_id"]: r["title"] for r in library.papers()}
    try:
        written = export_module.write(record, out or library.config.answers_dir, fmt, titles)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        library.close()
        raise typer.Exit(1) from exc

    for item in written:
        console.print(f"[green]wrote[/green] {item}")
    library.close()


watch_app = typer.Typer(help="Follow a topic and get new arXiv postings.")
app.add_typer(watch_app, name="watch")


def _watch_list(library: Library) -> WatchList:
    return WatchList(library.config.root / "watches.json")


@watch_app.command("add")
def watch_add(name: str, query: str) -> None:
    """Save a query to follow."""
    library = Library()
    watch = _watch_list(library).add(name, query)
    console.print(f"[green]watching[/green] {watch.name}: {watch.query}")
    library.close()


@watch_app.command("list")
def watch_list_cmd() -> None:
    """List saved watches."""
    library = Library()
    watches = _watch_list(library).list()
    if not watches:
        console.print("[yellow]no watches yet (noether watch add <name> <query>)[/yellow]")
    else:
        table = Table()
        for column in ("name", "query", "seen", "last run"):
            table.add_column(column)
        for watch in watches:
            table.add_row(watch.name, watch.query[:52], str(len(watch.seen)), watch.last_run[:16] or "never")
        console.print(table)
    library.close()


@watch_app.command("remove")
def watch_remove(name: str) -> None:
    """Delete a watch."""
    library = Library()
    console.print("removed" if _watch_list(library).remove(name) else f"[yellow]no watch {name!r}[/yellow]")
    library.close()


@watch_app.command("run")
def watch_run(
    name: str,
    limit: int = typer.Option(10, "--limit", "-n"),
    ingest: bool = typer.Option(True, "--ingest/--no-ingest"),
) -> None:
    """Run a watch and print the digest of what is new."""
    library = Library()
    try:
        digest = _watch_list(library).run(name, library, limit=limit, ingest=ingest)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        library.close()
        raise typer.Exit(1) from exc

    console.print(f"[bold]{digest.summary()}[/bold]\n")
    for entry in digest.entries:
        mark = "[green]ingested[/green]" if entry.ingested else "[yellow]skipped [/yellow]"
        console.print(f"{mark} {entry.arxiv_id}  {entry.title[:62]}")
        if entry.note:
            console.print(f"           [dim]{entry.note[:90]}[/dim]")
    library.close()


@app.command()
def map(
    query: str = typer.Argument(..., help="A topic to map across the corpus"),
    limit: int = typer.Option(40, "--limit", "-n"),
) -> None:
    """Cluster the corpus around a topic and flag passages that may disagree."""
    library = Library()
    result = build_map(library.store, query, limit=limit, retriever=library.retriever)
    console.print(f"[bold]{result.query}[/bold]  —  {result.summary()}\n")

    for cluster in result.clusters:
        console.print(f"[cyan]{cluster.label[:60]}[/cyan]")
        for paper_id in cluster.paper_ids:
            console.print(f"   {paper_id}")
    console.print()

    if result.contradictions:
        console.print("[yellow]Worth a second look — these may disagree:[/yellow]")
        for item in result.contradictions[:6]:
            console.print(f"  {item.describe()}")
            console.print(f"    [dim]{item.left.text[:110].strip()}[/dim]")
            console.print(f"    [dim]{item.right.text[:110].strip()}[/dim]\n")
        console.print(
            "[dim]Flagged by wording, not by physics. Two papers can differ in "
            "words and agree in fact.[/dim]"
        )
    library.close()


# ---- explanation and simulation ----------------------------------------
# Registered from a sibling module to keep this file readable; the commands
# attach to the same app and behave as if defined here.
from .cli_explain import register as _register_explain

_register_explain(app, console, _parse_pairs)
