"""Numeric evaluation and parameter sweeps.

Turning a parsed equation into numbers is the last step of guarantee 3, and the
one where a plausible-looking answer is easiest to produce and hardest to
notice. Two rules follow from that:

* **Nothing runs that has not been dimensionally checked.** ``evaluate`` takes
  the check as an argument and refuses an INCONSISTENT equation outright.
* **Every result carries how it was produced** -- the values substituted, the
  units assumed, and whether the equation needed lossy normalisation to parse.
  A number without that context is not a result, it is a rumour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import sympy

from .symbolic import ParsedEquation
from .units import DimensionCheck, Verdict, check_equation

#: Functions a paper equation may legitimately call. Anything outside this set
#: is refused rather than resolved, so evaluation cannot reach arbitrary code.
SAFE_FUNCTIONS = {
    "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
    "exp", "log", "sqrt", "Abs", "re", "im", "conjugate", "factorial",
    "erf", "erfc", "gamma", "besselj", "bessely", "Min", "Max", "sign",
}


#: CODATA values, supplied automatically so an equation containing hbar or c
#: evaluates without the caller restating physics. Sourced from scipy.constants,
#: which tracks the current CODATA adjustment.
def _physical_constants() -> dict[str, float]:
    from scipy import constants

    return {
        "hbar": constants.hbar,
        "c": constants.c,
        "k_B": constants.k,
        "kB": constants.k,
        "N_A": constants.N_A,
        "epsilon_0": constants.epsilon_0,
        "mu_0": constants.mu_0,
        "G_N": constants.G,
    }


class EvaluationError(RuntimeError):
    """The equation could not be evaluated as asked."""


@dataclass
class Evaluation:
    """One numeric result, with everything needed to judge it."""

    value: float | complex | None
    substituted: dict[str, float] = field(default_factory=dict)
    #: The dimensional verdict at the time of evaluation.
    dimensions: DimensionCheck | None = None
    #: Approximations inherited from parsing, repeated here so a caller reading
    #: only the result still sees them.
    notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.error


@dataclass
class Sweep:
    """A parameter sweep: one variable varied, everything else held."""

    variable: str
    values: list[float]
    results: list[float | None]
    substituted: dict[str, float] = field(default_factory=dict)
    dimensions: DimensionCheck | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def finite(self) -> list[tuple[float, float]]:
        """Points that evaluated to a real, finite number."""
        return [
            (x, y)
            for x, y in zip(self.values, self.results)
            if y is not None and isinstance(y, (int, float)) and math.isfinite(y)
        ]

    @property
    def coverage(self) -> float:
        """Fraction of sweep points that produced a usable value."""
        return len(self.finite) / len(self.values) if self.values else 0.0


def _target(parsed: ParsedEquation) -> Any:
    """The expression to evaluate: the right-hand side of an equality."""
    if parsed.is_equality:
        return parsed.rhs
    return parsed.expr


def _check_safe(expression: Any) -> None:
    """Refuse expressions calling anything outside :data:`SAFE_FUNCTIONS`."""
    for node in sympy.preorder_traversal(expression):
        if isinstance(node, sympy.Function):
            name = type(node).__name__
            if name not in SAFE_FUNCTIONS:
                raise EvaluationError(f"refusing to evaluate unknown function {name!r}")


def evaluate(
    parsed: ParsedEquation,
    values: dict[str, float],
    units: dict[str, str] | None = None,
    require_dimensions: bool = True,
) -> Evaluation:
    """Evaluate an equation's right-hand side at the given values.

    With ``require_dimensions`` set, an equation whose dimensions are known to
    be inconsistent is refused: producing a number from an equation we have
    already shown to be wrong would be the most misleading thing we could do.
    """
    result = Evaluation(value=None, substituted=dict(values), notes=list(parsed.notes))

    if not parsed.ok:
        result.error = parsed.error or "equation did not parse"
        return result

    check = check_equation(parsed, units)
    result.dimensions = check
    if require_dimensions and check.verdict is Verdict.INCONSISTENT:
        result.error = f"refusing to evaluate: {check.detail}"
        return result

    expression = _target(parsed)
    if expression is None:
        result.error = "nothing to evaluate"
        return result

    try:
        _check_safe(expression)
    except EvaluationError as exc:
        result.error = str(exc)
        return result

    # Known constants fill themselves in; explicit values always win, so a
    # caller working in natural units can set hbar = 1 and be obeyed.
    resolved = {**_physical_constants(), **values}
    missing = sorted({str(s) for s in expression.free_symbols} - set(resolved))
    if missing:
        result.error = "no value supplied for: " + ", ".join(missing)
        return result

    try:
        substituted = expression.subs({sympy.Symbol(k): v for k, v in resolved.items()})
        numeric = complex(sympy.N(substituted))
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:200]
        return result

    result.value = numeric.real if abs(numeric.imag) < 1e-12 else numeric
    return result


def sweep(
    parsed: ParsedEquation,
    variable: str,
    start: float,
    stop: float,
    points: int = 200,
    fixed: dict[str, float] | None = None,
    units: dict[str, str] | None = None,
) -> Sweep:
    """Vary one symbol across a range, holding the rest fixed."""
    if points < 2:
        raise EvaluationError("a sweep needs at least two points")

    fixed = dict(fixed or {})
    step = (stop - start) / (points - 1)
    grid = [start + step * i for i in range(points)]

    check = check_equation(parsed, units)
    results: list[float | None] = []

    for value in grid:
        evaluation = evaluate(
            parsed, {**fixed, variable: value}, units, require_dimensions=False
        )
        results.append(
            evaluation.value.real if isinstance(evaluation.value, complex) else evaluation.value
        )

    return Sweep(
        variable=variable,
        values=grid,
        results=results,
        substituted=fixed,
        dimensions=check,
        notes=list(parsed.notes),
    )


def solve_for(parsed: ParsedEquation, symbol: str) -> list[Any]:
    """Solve an equality for one symbol. Returns every branch SymPy finds."""
    if not parsed.ok or not parsed.is_equality:
        return []
    try:
        return list(sympy.solve(sympy.Eq(parsed.lhs, parsed.rhs), sympy.Symbol(symbol)))
    except Exception:
        return []
