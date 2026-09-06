"""What a simulation returns.

A number without its convergence evidence is a rumour, so every run carries the
grid it was solved on, the solver used, and a convergence check performed by
re-running at finer resolution and comparing. If the answer moves when the step
size halves, the run says so rather than reporting the first value it computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Series:
    """One named curve over the time (or parameter) grid."""

    name: str
    values: list[float]
    units: str = ""
    description: str = ""

    @property
    def final(self) -> float | None:
        return self.values[-1] if self.values else None

    @property
    def extremes(self) -> tuple[float, float] | None:
        return (min(self.values), max(self.values)) if self.values else None


@dataclass
class Convergence:
    """Evidence that the answer is the physics, not the grid.

    Computed by halving the step and comparing: ``max_drift`` is the largest
    absolute change in any observable. A converged run is one where refining the
    grid changes nothing you would report.
    """

    checked: bool = False
    refined_points: int = 0
    max_drift: float = 0.0
    tolerance: float = 1e-3
    note: str = ""

    @property
    def converged(self) -> bool | None:
        """None when not checked -- which is not the same as converged."""
        return None if not self.checked else self.max_drift <= self.tolerance

    def describe(self) -> str:
        if not self.checked:
            return f"convergence not checked ({self.note})" if self.note else "convergence not checked"
        verdict = "converged" if self.converged else "NOT CONVERGED"
        return (
            f"{verdict}: halving the step moved results by {self.max_drift:.2e} "
            f"(tolerance {self.tolerance:.0e})"
        )


@dataclass
class SimulationResult:
    """A completed run, with everything needed to judge it."""

    model_key: str
    model_name: str
    parameters: dict[str, float] = field(default_factory=dict)
    times: list[float] = field(default_factory=list)
    series: list[Series] = field(default_factory=list)
    solver: str = ""
    convergence: Convergence = field(default_factory=Convergence)
    caveats: list[str] = field(default_factory=list)
    #: Files written by the run (plots).
    figures: list[str] = field(default_factory=list)
    error: str = ""
    #: Extra scalars worth reporting: periods, rates, fitted lifetimes.
    derived: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.series)

    def get(self, name: str) -> Series | None:
        return next((s for s in self.series if s.name == name), None)

    def summary(self) -> str:
        if self.error:
            return f"{self.model_key}: {self.error}"
        parts = [
            f"{self.model_name}",
            f"{len(self.times)} points over t=[{self.times[0]:.4g}, {self.times[-1]:.4g}]"
            if self.times
            else "no grid",
            self.convergence.describe(),
        ]
        return " · ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_key,
            "name": self.model_name,
            "parameters": self.parameters,
            "solver": self.solver,
            "times": self.times,
            "series": [
                {"name": s.name, "units": s.units, "description": s.description, "values": s.values}
                for s in self.series
            ],
            "derived": self.derived,
            "convergence": {
                "checked": self.convergence.checked,
                "converged": self.convergence.converged,
                "max_drift": self.convergence.max_drift,
                "note": self.convergence.describe(),
            },
            "caveats": self.caveats,
            "figures": self.figures,
            "error": self.error,
        }
