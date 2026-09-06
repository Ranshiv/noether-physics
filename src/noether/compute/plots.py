"""Plot rendering -- deterministic, never generated.

Every figure this project produces is drawn from data by matplotlib, or is a
file extracted from a real paper. Nothing is image-generated. A diffusion model
asked for a Feynman diagram yields non-conserving vertices and invented
particles that look convincing, which is precisely the failure this project
exists to eliminate.

So each plot carries its provenance in the figure itself: the equation, the
values held fixed, and any approximation the parse required. A plot that leaves
this system can always be traced back to what produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

# A non-interactive backend: this runs headless, in CLI and MCP contexts where
# no display exists and a GUI backend would hang rather than fail.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .run import Sweep


@dataclass
class Figure:
    """A rendered plot and the provenance that justifies it."""

    path: Path
    title: str
    caption: str
    #: "computed" here always; extracted figures carry their paper and licence.
    provenance: str = "computed"
    notes: tuple[str, ...] = ()


def plot_sweep(
    result: Sweep,
    path: Path,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    equation: str = "",
) -> Figure:
    """Render a parameter sweep.

    Points that did not evaluate are omitted rather than interpolated across:
    a gap in a curve is information, and joining over it invents data.
    """
    points = result.finite
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(7.0, 4.4), dpi=140)

    if points:
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        axes.plot(xs, ys, linewidth=1.8, color="#1f4e79")
    else:
        axes.text(
            0.5, 0.5, "no points evaluated", transform=axes.transAxes,
            ha="center", va="center", color="#b00020",
        )

    axes.set_xlabel(xlabel or result.variable)
    axes.set_ylabel(ylabel or "value")
    axes.set_title(title or (equation[:70] if equation else "parameter sweep"))
    axes.grid(True, alpha=0.25, linewidth=0.6)
    axes.spines[["top", "right"]].set_visible(False)

    footer = _footer(result)
    if footer:
        figure.text(0.01, 0.01, footer, fontsize=6.5, color="#555555", va="bottom")
        figure.subplots_adjust(bottom=0.22)

    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

    caption = (
        f"{result.variable} swept over "
        f"[{result.values[0]:.4g}, {result.values[-1]:.4g}], "
        f"{len(points)}/{len(result.values)} points evaluated"
    )
    return Figure(
        path=path,
        title=title or "parameter sweep",
        caption=caption,
        notes=tuple(result.notes),
    )


def _footer(result: Sweep) -> str:
    """Provenance line drawn onto the figure itself."""
    parts = []
    if result.substituted:
        parts.append("held: " + ", ".join(f"{k}={v:g}" for k, v in sorted(result.substituted.items())))
    if result.dimensions is not None:
        parts.append(f"dimensions: {result.dimensions.verdict.value}")
    if result.coverage < 1.0:
        parts.append(f"coverage: {result.coverage:.0%}")
    for note in result.notes:
        parts.append(f"note: {note}")
    return "  |  ".join(parts)


def plot_series(
    series: list[tuple[str, list[float], list[float]]],
    path: Path,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> Figure:
    """Overlay several named curves on shared axes.

    This is what a reproduction overlay uses: the paper's reported curve and our
    re-run of its stated equation, on the same axes, so agreement or divergence
    is visible rather than asserted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7.0, 4.4), dpi=140)

    for index, (label, xs, ys) in enumerate(series):
        axes.plot(xs, ys, linewidth=1.8, label=label, alpha=0.9,
                  linestyle=["-", "--", ":", "-."][index % 4])

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True, alpha=0.25, linewidth=0.6)
    axes.spines[["top", "right"]].set_visible(False)
    if len(series) > 1:
        axes.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

    return Figure(path=path, title=title or "series", caption=f"{len(series)} curves")


def describe(figure: Any) -> str:
    """One-line provenance string for a figure, for reports and exports."""
    if getattr(figure, "provenance", "computed") == "computed":
        base = f"computed: {figure.caption}"
    else:
        base = figure.provenance
    notes = getattr(figure, "notes", ())
    return base + ("  [" + "; ".join(notes) + "]" if notes else "")
