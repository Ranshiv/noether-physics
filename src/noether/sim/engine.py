"""Running a simulation, and drawing the result.

The entry point every caller uses. It validates parameters against the model's
declared bounds *before* starting, so an impossible request fails in
milliseconds with a reason instead of after a long run or an out-of-memory kill.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .models import all_models, get, load_all
from .result import SimulationResult

load_all()

__all__ = ["all_models", "available", "describe", "get", "plot", "run"]


def available() -> list[dict]:
    """Every model, with enough detail to choose one."""
    return [
        {
            "key": model.key,
            "name": model.name,
            "domain": model.domain,
            "summary": model.summary,
            "everyday": model.everyday,
            "equation": model.equation,
            "parameters": [
                {
                    "name": p.name, "default": p.default, "units": p.units,
                    "description": p.description, "min": p.minimum, "max": p.maximum,
                }
                for p in model.parameters
            ],
            "observables": model.observables,
        }
        for model in all_models()
    ]


def describe(key: str) -> dict:
    model = get(key)
    return next(item for item in available() if item["key"] == key) | {
        "caveats": model.caveats
    }


def run(key: str, parameters: dict[str, float] | None = None) -> SimulationResult:
    """Run a named model. Invalid parameters fail fast, with a reason."""
    try:
        model = get(key)
    except KeyError as exc:
        return SimulationResult(model_key=key, model_name=key, error=str(exc))

    resolved, problems = model.resolve(parameters)
    if problems:
        return SimulationResult(
            model_key=key, model_name=model.name, parameters=resolved,
            error="; ".join(problems),
        )

    if model.runner is None:  # pragma: no cover - registration guarantees this
        return SimulationResult(key, model.name, error="model has no runner")

    # Integer-typed knobs (grid points, Fock levels) must not arrive as floats.
    signature = inspect.signature(model.runner)
    call: dict[str, float | int] = {}
    for name in signature.parameters:
        value = resolved[name]
        call[name] = int(value) if name in ("points", "levels", "photons") else float(value)

    try:
        result = model.runner(**call)
    except RuntimeError as exc:
        return SimulationResult(key, model.name, parameters=resolved, error=str(exc))
    except Exception as exc:
        return SimulationResult(
            key, model.name, parameters=resolved,
            error=f"{type(exc).__name__}: {exc}"[:300],
        )

    if not result.caveats:
        result.caveats = list(model.caveats)
    return result


def plot(result: SimulationResult, path: Path, series: list[str] | None = None) -> Path | None:
    """Draw a run. Returns the path written, or None when there is nothing to draw.

    The convergence verdict is printed onto the figure, because a plot that
    travels without it invites exactly the trust it has not earned.
    """
    if not result.ok:
        return None

    chosen = [s for s in result.series if series is None or s.name in series]
    if not chosen:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(7.4, 4.4), dpi=140)
    for index, item in enumerate(chosen):
        axes.plot(
            result.times, item.values, linewidth=1.7,
            linestyle=["-", "--", ":", "-."][index % 4],
            label=f"{item.name}" + (f" [{item.units}]" if item.units else ""),
        )

    axes.set_xlabel("time [s]")
    axes.set_title(result.model_name)
    axes.grid(True, alpha=0.25, linewidth=0.6)
    axes.spines[["top", "right"]].set_visible(False)
    if len(chosen) > 1:
        axes.legend(frameon=False, fontsize=8)

    footer = " | ".join(
        [
            "computed",
            result.solver,
            result.convergence.describe(),
            ", ".join(f"{k}={v:g}" for k, v in list(result.parameters.items())[:4]),
        ]
    )
    figure.text(0.01, 0.005, footer[:190], fontsize=6.2, color="#555555", va="bottom")
    figure.subplots_adjust(bottom=0.2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

    result.figures.append(str(path))
    return path


def bloch_sphere(result: SimulationResult, path: Path) -> Path | None:
    """Draw the Bloch trajectory, when the run produced one."""
    x, y, z = (result.get(f"bloch_{axis}") for axis in "xyz")
    if not (x and y and z):
        return None

    try:
        import qutip
    except ImportError:  # pragma: no cover - optional dependency
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sphere = qutip.Bloch()
    sphere.add_points([x.values, y.values, z.values], meth="l")
    sphere.point_color = ["#1f4e79"]
    sphere.save(name=str(path))
    result.figures.append(str(path))
    return path
