"""The simulation registry: named physical systems with declared parameters.

A simulation is not a script you edit. It is a *named model* with typed,
documented, bounded parameters, so that "simulate a damped oscillator with
Q = 12" is a request the system can answer, check and refuse.

Bounds matter more than they look. A user asking for 40 qubits is not asking for
a 20-minute hang and an out-of-memory kill; they are asking for something this
machine cannot do, and the honest response is to say so before starting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Domain = Literal["classical", "quantum"]


@dataclass(frozen=True)
class Parameter:
    """One knob on a model, with the physics stated."""

    name: str
    default: float
    units: str
    description: str
    minimum: float | None = None
    maximum: float | None = None

    def validate(self, value: float) -> str | None:
        """Return a reason the value is unusable, or None."""
        if self.minimum is not None and value < self.minimum:
            return f"{self.name} = {value:g} is below the minimum {self.minimum:g} {self.units}"
        if self.maximum is not None and value > self.maximum:
            return (
                f"{self.name} = {value:g} exceeds the maximum {self.maximum:g} {self.units}. "
                "That limit exists because beyond it the run stops being tractable on a "
                "single machine, not because the physics stops."
            )
        return None


@dataclass
class Model:
    """A simulatable system."""

    key: str
    name: str
    domain: Domain
    summary: str
    #: What a reader would recognise this as, outside physics.
    everyday: str
    #: The governing equation, in LaTeX. Dimension-checked like any other.
    equation: str
    parameters: list[Parameter] = field(default_factory=list)
    #: Quantities the run reports, in order.
    observables: list[str] = field(default_factory=list)
    #: What the simulation genuinely does not capture.
    caveats: list[str] = field(default_factory=list)
    #: Populated by the domain modules at import time.
    runner: Callable[..., Any] | None = None

    def defaults(self) -> dict[str, float]:
        return {p.name: p.default for p in self.parameters}

    def parameter(self, name: str) -> Parameter | None:
        return next((p for p in self.parameters if p.name == name), None)

    def resolve(self, values: dict[str, float] | None) -> tuple[dict[str, float], list[str]]:
        """Merge caller values over defaults, returning any problems found."""
        resolved = self.defaults()
        problems: list[str] = []

        for name, value in (values or {}).items():
            parameter = self.parameter(name)
            if parameter is None:
                known = ", ".join(p.name for p in self.parameters)
                problems.append(f"{self.key} has no parameter {name!r}; it has: {known}")
                continue
            reason = parameter.validate(value)
            if reason:
                problems.append(reason)
                continue
            resolved[name] = value

        return resolved, problems


REGISTRY: dict[str, Model] = {}


def register(model: Model) -> Model:
    REGISTRY[model.key] = model
    return model


def get(key: str) -> Model:
    if key not in REGISTRY:
        raise KeyError(
            f"no model {key!r}. Available: " + ", ".join(sorted(REGISTRY))
        )
    return REGISTRY[key]


def all_models(domain: Domain | None = None) -> list[Model]:
    models = sorted(REGISTRY.values(), key=lambda m: (m.domain, m.key))
    return [m for m in models if domain is None or m.domain == domain]


def load_all() -> None:
    """Import the domain modules so they register their models."""
    from . import classical, quantum  # noqa: F401
