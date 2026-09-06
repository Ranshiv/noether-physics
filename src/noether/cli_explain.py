"""CLI commands for explanation and simulation.

Kept in their own module so ``cli.py`` stays readable; the commands register
onto the same Typer app, so they behave exactly as if they were defined there.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .explain.compose import compose, suggestions
from .explain.compose import render as render_explanation
from .explain.library import CONCEPTS
from .library import Library
from .sim import engine as sim_engine


def register(app: typer.Typer, console, parse_pairs) -> None:
    """Attach the explanation and simulation commands to the main app."""

    @app.command()
    def explain(
        topic: str = typer.Argument(..., help="A concept, e.g. 'decoherence' or 'resonance'"),
        depth: str = typer.Option("full", "--depth", "-d", help="brief | standard | full"),
        no_sim: bool = typer.Option(False, "--no-sim", help="Skip running the simulation."),
    ) -> None:
        """Explain a physics concept, with a worked example and a live simulation.

        The prose is curated, not generated. What the tool adds is checking: every
        equation is dimension-checked, every worked example is recomputed, and the
        simulation actually runs.
        """
        library = Library()
        explanation = compose(
            topic,
            library=library,
            run_simulation=not no_sim,
            figures_dir=library.config.figures_dir,
        )

        if explanation is None:
            console.print(f"[yellow]no concept matching {topic!r}[/yellow]")
            near = suggestions(topic)
            if near:
                console.print("Did you mean: " + ", ".join(near))
            else:
                console.print("Known concepts: " + ", ".join(c.key for c in CONCEPTS))
            library.close()
            raise typer.Exit(1)

        console.print(render_explanation(explanation, depth=depth))
        if not explanation.all_checks_passed:
            console.print(
                "[red]Some checks failed above; treat those lines as unverified.[/red]"
            )
        library.close()

    @app.command()
    def concepts() -> None:
        """List every concept the explainer knows."""
        from rich.table import Table

        table = Table(title="Concept library")
        for column in ("key", "name", "domain", "simulation"):
            table.add_column(column)
        for concept in CONCEPTS:
            table.add_row(concept.key, concept.name, concept.domain, concept.simulation or "-")
        console.print(table)

    @app.command()
    def simulate(
        model: str = typer.Argument(..., help="A model key, e.g. 'rabi' or 'pendulum'"),
        params: str = typer.Option("", "--params", "-p", help="e.g. 'detuning=3,duration=8'"),
        plot_path: Path = typer.Option(None, "--plot", help="Where to write the figure."),
        bloch: bool = typer.Option(False, "--bloch", help="Also draw the Bloch sphere."),
    ) -> None:
        """Run a physics simulation and report it with its convergence check."""
        from rich.table import Table

        library = Library()
        result = sim_engine.run(model, parse_pairs(params, as_float=True))

        if not result.ok:
            console.print(f"[red]{result.error}[/red]")
            library.close()
            raise typer.Exit(1)

        console.print(f"[bold]{result.model_name}[/bold]")
        console.print(result.solver)
        colour = {True: "green", False: "red", None: "yellow"}[result.convergence.converged]
        console.print(f"[{colour}]{result.convergence.describe()}[/{colour}]")
        console.print("")

        table = Table(show_header=True)
        for column in ("series", "units", "start", "end", "min", "max"):
            table.add_column(column)
        for item in result.series:
            low, high = item.extremes or (0.0, 0.0)
            table.add_row(
                item.name,
                item.units or "-",
                f"{item.values[0]:.4g}",
                f"{item.final:.4g}",
                f"{low:.4g}",
                f"{high:.4g}",
            )
        console.print(table)

        if result.derived:
            console.print("")
            console.print("[bold]Derived[/bold]")
            for key, value in result.derived.items():
                if value is not None:
                    console.print(f"  {key} = {value}")

        if result.caveats:
            console.print("")
            console.print("[bold]What this model leaves out[/bold]")
            for caveat in result.caveats:
                console.print(f"  [dim]- {caveat}[/dim]")

        target = plot_path or (library.config.figures_dir / f"{model}.png")
        written = sim_engine.plot(result, target)
        if written:
            console.print("")
            console.print(f"plot: {written}")
        if bloch:
            sphere = sim_engine.bloch_sphere(
                result, target.with_name(target.stem + "_bloch.png")
            )
            console.print(
                f"bloch: {sphere}" if sphere else "[dim]no Bloch data in this model[/dim]"
            )
        library.close()

    @app.command()
    def models() -> None:
        """List every simulation model, with its parameters."""
        for item in sim_engine.available():
            console.print(f"[cyan]{item['key']}[/cyan]  ({item['domain']})  {item['name']}")
            console.print(f"   {item['summary']}")
            console.print(f"   [dim]{item['everyday']}[/dim]")
            names = ", ".join(
                f"{p['name']}={p['default']:g}{p['units']}" for p in item["parameters"]
            )
            console.print(f"   params: {names}")
            console.print("")
