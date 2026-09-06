"""Dimensional analysis -- the second half of guarantee 3.

This is the cheapest high-value check in the system. Language models emit
dimensionally inconsistent equations constantly, and a human skimming an answer
will not notice a length added to a time. A machine notices every time, free,
before the claim is shown.

**Why almost nothing is assumed.** An early version of this module shipped a
table mapping ``h`` to joule-seconds, and promptly reported ``E = mv^2/2 + mgh``
-- correct physics -- as dimensionally inconsistent, because in that equation
``h`` is a height, not Planck's constant. Nearly every physics symbol is
context-dependent: ``T`` is period or temperature, ``k`` is wavenumber or
Boltzmann's constant, ``I`` is current or moment of inertia, ``p`` is momentum
or pressure. A confident wrong verdict is exactly as damaging as a fabricated
citation.

So only genuinely unambiguous constants are assumed. Everything else must be
declared by the caller, and until it is the verdict is ``unknown`` -- with a
message naming the plausible readings, so the user knows what to declare.

The check is three-valued on purpose. ``consistent`` and ``inconsistent`` are
verdicts; ``unknown`` means we did not check. Those three never collapse to two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pint
import sympy

#: Symbols with one universally accepted meaning. Nothing goes in here unless
#: reading it any other way would be perverse.
UNAMBIGUOUS: dict[str, str] = {
    "hbar": "joule * second",
    "c": "meter / second",
    "k_B": "joule / kelvin",
    "kB": "joule / kelvin",
    "N_A": "1 / mole",
    "epsilon_0": "farad / meter",
    "mu_0": "henry / meter",
    "G_N": "meter**3 / (kilogram * second**2)",
}

#: Symbols that are dimensionless whichever reading you take.
DIMENSIONLESS = frozenset({"n", "N", "i", "j", "pi", "theta", "phi", "varphi", "psi"})

#: Symbols with more than one standard reading, and what they might be. Used to
#: explain an "unknown" verdict instead of merely announcing it.
AMBIGUOUS: dict[str, tuple[str, ...]] = {
    "h": ("Planck constant (J s)", "height (m)"),
    "T": ("period (s)", "temperature (K)", "kinetic energy (J)"),
    "k": ("wavenumber (1/m)", "Boltzmann constant (J/K)", "spring constant (N/m)"),
    "I": ("current (A)", "moment of inertia (kg m^2)", "intensity (W/m^2)"),
    "p": ("momentum (kg m/s)", "pressure (Pa)"),
    "V": ("voltage (V)", "volume (m^3)", "potential energy (J)"),
    "A": ("area (m^2)", "current (A)", "vector potential (T m)"),
    "C": ("capacitance (F)", "heat capacity (J/K)"),
    "S": ("entropy (J/K)", "action (J s)"),
    "L": ("length (m)", "angular momentum (J s)", "inductance (H)"),
    "e": ("elementary charge (C)", "Euler's number (dimensionless)"),
    "g": ("gravitational acceleration (m/s^2)", "coupling constant", "g-factor"),
    "a": ("acceleration (m/s^2)", "lattice constant (m)", "scale factor"),
    "d": ("distance (m)", "dimension (dimensionless)"),
    "f": ("frequency (Hz)", "force (N)", "a function"),
    "m": ("mass (kg)", "magnetic quantum number (dimensionless)"),
    "R": ("resistance (ohm)", "gas constant (J/(mol K))", "radius (m)"),
    "E": ("energy (J)", "electric field (V/m)"),
    "B": ("magnetic field (T)", "a coefficient"),
    "mu": ("magnetic moment (J/T)", "reduced mass (kg)", "chemical potential (J)"),
    "beta": ("inverse temperature (1/J)", "velocity ratio v/c (dimensionless)"),
    "gamma": ("gyromagnetic ratio", "Lorentz factor (dimensionless)", "damping rate (1/s)"),
    "omega": ("angular frequency (rad/s)",),
    "lambda": ("wavelength (m)", "coupling constant", "eigenvalue"),
}


class Verdict(str, Enum):
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    UNKNOWN = "unknown"


@dataclass
class DimensionCheck:
    """The outcome of checking one equation."""

    verdict: Verdict
    detail: str = ""
    lhs_units: str | None = None
    rhs_units: str | None = None
    #: Symbols we had no dimension for. Non-empty means the verdict is UNKNOWN.
    unknown_symbols: list[str] = field(default_factory=list)
    #: What each symbol was taken to be, so a verdict is auditable.
    assumed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True only for a positive verdict. UNKNOWN is not a pass."""
        return self.verdict is Verdict.CONSISTENT

    def hint(self) -> str:
        """Guidance on what to declare to turn UNKNOWN into a verdict."""
        lines = []
        for symbol in self.unknown_symbols:
            readings = AMBIGUOUS.get(symbol)
            if readings:
                lines.append(f"  {symbol}: could be " + ", or ".join(readings))
            else:
                lines.append(f"  {symbol}: no standard reading; declare its units")
        return "\n".join(lines)

    def __str__(self) -> str:
        if self.verdict is Verdict.CONSISTENT:
            return f"dimensionally consistent ({self.lhs_units})"
        if self.verdict is Verdict.INCONSISTENT:
            return f"DIMENSIONALLY INCONSISTENT: {self.detail}"
        return f"not checked: {self.detail}"


_REGISTRY: pint.UnitRegistry | None = None


def registry() -> pint.UnitRegistry:
    """The shared unit registry. Quantities from different registries do not mix."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = pint.UnitRegistry()
    return _REGISTRY


def dimensions_for(symbol: str, declared: dict[str, str] | None = None) -> str | None:
    """Units for a symbol, or None when it must be declared.

    Caller declarations always win: in a paper's own context the author knows
    what ``h`` means and we do not.
    """
    if declared and symbol in declared:
        return declared[symbol]
    if symbol in UNAMBIGUOUS:
        return UNAMBIGUOUS[symbol]
    if symbol in DIMENSIONLESS:
        return "dimensionless"
    # f_{n} carries the dimensions of f: a subscript labels an instance of a
    # quantity, it does not change what the quantity is.
    base = symbol.split("_")[0].strip("{}")
    if base != symbol:
        return dimensions_for(base, declared)
    return None


def check_dimensions(
    lhs: Any,
    rhs: Any,
    units: dict[str, str] | None = None,
) -> DimensionCheck:
    """Compare the dimensions of two SymPy expressions.

    ``units`` maps symbol name to a Pint-parseable unit string. Symbols that are
    neither declared nor unambiguous make the verdict UNKNOWN -- deliberately,
    since guessing is how a correct equation gets called wrong.
    """
    if lhs is None or rhs is None:
        return DimensionCheck(Verdict.UNKNOWN, "equation has no two sides to compare")

    symbols = sorted({str(s) for s in lhs.free_symbols | rhs.free_symbols})
    assignments: dict[str, Any] = {}
    unknown: list[str] = []
    assumed: dict[str, str] = {}
    ureg = registry()

    for name in symbols:
        spec = dimensions_for(name, units)
        if spec is None:
            unknown.append(name)
            continue
        try:
            assignments[name] = ureg.Quantity(1.0, spec)
            assumed[name] = spec
        except Exception:
            unknown.append(name)

    if unknown:
        return DimensionCheck(
            Verdict.UNKNOWN,
            "units not declared for: " + ", ".join(unknown),
            unknown_symbols=unknown,
            assumed=assumed,
        )

    try:
        left = _evaluate(lhs, assignments)
        right = _evaluate(rhs, assignments)
    except pint.DimensionalityError as exc:
        # Pint raises here when a sum mixes dimensions -- a length added to a
        # time. That is a positive finding of inconsistency, not a failure to
        # check, and calling it "unknown" would hide a real error.
        return DimensionCheck(
            Verdict.INCONSISTENT, f"incompatible terms in a sum: {exc}", assumed=assumed
        )
    except Exception as exc:
        return DimensionCheck(
            Verdict.UNKNOWN, f"could not evaluate units: {type(exc).__name__}", assumed=assumed
        )

    left_units = _dimensionality(left)
    right_units = _dimensionality(right)

    # Compare the dimensionality objects rather than their rendered strings:
    # a float exponent prints as "[time] ** 2.0" and an int as "[time] ** 2",
    # so string comparison calls correct physics inconsistent.
    if _same_dimensions(left, right):
        return DimensionCheck(Verdict.CONSISTENT, "", left_units, right_units, assumed=assumed)
    return DimensionCheck(
        Verdict.INCONSISTENT,
        f"left side is {left_units}, right side is {right_units}",
        left_units,
        right_units,
        assumed=assumed,
    )


def _same_dimensions(left: Any, right: Any) -> bool:
    """True when two quantities share a dimensionality."""
    left_dim = getattr(left, "dimensionality", None)
    right_dim = getattr(right, "dimensionality", None)
    if left_dim is None or right_dim is None:
        return _dimensionality(left) == _dimensionality(right)
    return left_dim == right_dim


def _dimensionality(quantity: Any) -> str:
    return str(quantity.dimensionality) if hasattr(quantity, "dimensionality") else "dimensionless"


def _evaluate(expression: Any, assignments: dict[str, Any]) -> Any:
    """Evaluate a SymPy expression with Pint quantities substituted in."""
    ureg = registry()

    def walk(node: Any) -> Any:
        if node.is_Symbol:
            return assignments[str(node)]
        if node.is_Number:
            return ureg.Quantity(float(node), "dimensionless")
        if node.is_Add:
            # Addition is where inconsistency surfaces: pint raises when terms
            # do not share a dimension, which is exactly the check we want.
            total = walk(node.args[0])
            for term in node.args[1:]:
                total = total + walk(term)
            return total
        if node.is_Mul:
            product = walk(node.args[0])
            for factor in node.args[1:]:
                product = product * walk(factor)
            return product
        if node.is_Pow:
            base, exponent = node.args
            if not exponent.is_Number:
                # A symbolic exponent forces its base to be dimensionless.
                return ureg.Quantity(1.0, "dimensionless")
            power = float(exponent)
            return walk(base) ** (int(power) if power.is_integer() else power)
        if isinstance(node, sympy.Function):
            # sin, exp and log take dimensionless arguments and return the same.
            return ureg.Quantity(1.0, "dimensionless")
        return ureg.Quantity(1.0, "dimensionless")

    return walk(sympy.sympify(expression))


def check_equation(parsed: Any, units: dict[str, str] | None = None) -> DimensionCheck:
    """Check a :class:`~noether.compute.symbolic.ParsedEquation`."""
    if not getattr(parsed, "ok", False):
        return DimensionCheck(Verdict.UNKNOWN, "equation did not parse")
    if not getattr(parsed, "is_equality", False):
        return DimensionCheck(Verdict.UNKNOWN, "not an equality; nothing to compare")
    return check_dimensions(parsed.lhs, parsed.rhs, units)
