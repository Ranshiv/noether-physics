"""LaTeX equations to SymPy -- the first half of guarantee 3.

Paper LaTeX is not textbook LaTeX. It carries roman-type subscripts
(``\\gamma_{\\rm n}``), spacing commands, display sizing, and macros already
expanded into shapes no LaTeX grammar accepts. So an equation is normalised
first (see :mod:`.normalise`), parsed, and then the placeholders introduced
during normalisation are substituted back into the expression.

A failure here is a *result*, not an exception to swallow: "this equation did
not parse" is information a physicist can act on, whereas a silently mangled
expression is the thing this project exists to prevent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import sympy

from .normalise import (
    RELATIONS,
    Normalised,
    _Placeholders,
    blocking_environment,
    collapse_double_braces,
    fix_leading_minus,
    insert_explicit_multiplication,
    strip_decorators,
    strip_unknown_commands,
    substitute_derivatives,
    substitute_symbols,
)

#: Sizing and spacing commands that carry no mathematical content.
_DROP_COMMANDS = (
    r"\left", r"\right", r"\big", r"\Big", r"\bigg", r"\Bigg",
    r"\displaystyle", r"\textstyle", r"\scriptstyle", r"\limits",
    r"\nonumber", r"\notag", r"\qquad", r"\quad", r"\!", r"\,", r"\;", r"\:",
)

#: Type-style wrappers inside subscripts: \gamma_{\rm n} -> \gamma_{n}.
_TYPESTYLE_RE = re.compile(r"\\(?:rm|mathrm|text|mathbf|bf|it|mathit|textrm|mbox|operatorname)\s*")

_LABEL_RE = re.compile(r"\\label\s*\{[^}]*\}")
_TAG_RE = re.compile(r"\\tag\s*\{[^}]*\}")
_ALIGN_RE = re.compile(r"(?<!\\)&")

#: Names denoting a mathematical constant rather than a free symbol.
#: Leaving pi as a Symbol makes every sweep demand a value for it, and
#: makes dimensional analysis ask what units pi has.
MATH_CONSTANTS = {"pi": sympy.pi, "oo": sympy.oo}


@dataclass
class ParsedEquation:
    """The outcome of trying to parse one equation."""

    latex: str
    normalised: str
    expr: Any = None
    #: Present when the equation was an equality: the left and right sides.
    lhs: Any = None
    rhs: Any = None
    symbols: list[str] = field(default_factory=list)
    ok: bool = False
    error: str = ""
    #: Approximations made to get this through the grammar. A non-empty list
    #: means the parse succeeded but lost information; report it, never hide it.
    notes: list[str] = field(default_factory=list)
    #: Set when the statement was a relation rather than an equality, e.g.
    #: ">=" for an uncertainty relation. lhs/rhs still hold both sides.
    relation: str | None = None

    @property
    def is_equality(self) -> bool:
        return self.lhs is not None and self.rhs is not None

    @property
    def lossy(self) -> bool:
        """True when the parse required an information-losing substitution."""
        return bool(self.notes)

    def free_symbol_names(self) -> list[str]:
        if self.expr is None:
            return []
        return sorted(str(s) for s in self.expr.free_symbols)


def normalise(latex: str) -> Normalised:
    """Reduce paper LaTeX to grammar-safe form, keeping the placeholder map."""
    placeholders = _Placeholders()

    blocked = blocking_environment(latex)
    if blocked:
        return Normalised(
            text="",
            blocked=(
                f"contains a {blocked!r} environment, which has structure this "
                "scalar representation cannot carry"
            ),
        )

    text = _LABEL_RE.sub("", latex)
    text = _TAG_RE.sub("", text)
    text = _ALIGN_RE.sub("", text)          # alignment markers are layout only
    text = _TYPESTYLE_RE.sub("", text)      # \gamma_{\rm n} -> \gamma_{n}

    for command in _DROP_COMMANDS:
        text = text.replace(command, " ")

    # \dfrac and \tfrac are \frac with a size hint.
    text = re.sub(r"\\[dt]frac\b", r"\\frac", text)
    # A bare \over is TeX-era syntax no LaTeX grammar handles.
    text = text.replace(r"\over", "/")

    relation = next(
        (symbol for name, symbol in RELATIONS.items() if re.search(rf"[\\]{name}(?![A-Za-z])", text)),
        None,
    )
    for name in RELATIONS:
        text = re.sub(rf"[\\]{name}(?![A-Za-z])", "=", text)

    text = collapse_double_braces(text)
    text = strip_decorators(text)
    text, notes = substitute_derivatives(text, placeholders)
    text = substitute_symbols(text, placeholders)
    text, removed = strip_unknown_commands(text)
    if removed:
        notes.append("dropped unsupported commands: " + ", ".join(sorted(set(removed))))

    # Bare delimiters left over from removing sizing commands are not operators.
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # A trailing relational operator is a fragment, usually from a split row.
    text = re.sub(r"[=<>+\-*/^_,;&]+$", "", text).strip()
    text = insert_explicit_multiplication(text)

    if relation:
        notes.append(f"statement is a relation ({relation}), not an equality; "
                     "both sides are parsed but the relation is not modelled")

    return Normalised(
        text=fix_leading_minus(text),
        placeholders=placeholders.mapping,
        notes=notes,
        relation=relation,
    )


def normalise_latex(latex: str) -> str:
    """Normalised text alone, for callers that do not need the mapping."""
    return normalise(latex).text


def parse_equation(latex: str) -> ParsedEquation:
    """Parse one display equation into SymPy.

    Equalities are split so both sides are available: dimensional analysis needs
    to compare them, and a solver needs the right-hand side alone.
    """
    normalised = normalise(latex)
    result = ParsedEquation(
        latex=latex,
        normalised=normalised.text,
        notes=list(normalised.notes),
        relation=normalised.relation,
    )

    if normalised.blocked:
        result.error = normalised.blocked
        return result

    if not normalised.text:
        result.error = "equation is empty after normalisation"
        return result

    # Split on a top-level '=' so each side parses independently. A nested '='
    # inside braces (a subscripted sum, say) must not split the equation.
    sides = _split_top_level(normalised.text, "=")

    try:
        if len(sides) == 2 and all(s.strip() for s in sides):
            lhs = _parse_side(fix_leading_minus(sides[0].strip()), normalised.placeholders)
            rhs = _parse_side(fix_leading_minus(sides[1].strip()), normalised.placeholders)
            result.lhs, result.rhs = lhs, rhs
            result.expr = sympy.Eq(lhs, rhs, evaluate=False)
        else:
            result.expr = _parse_side(normalised.text, normalised.placeholders)
        result.ok = True
        result.symbols = result.free_symbol_names()
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200]

    return result


def _parse_side(latex: str, placeholders: dict[str, str]) -> Any:
    """Parse one side, then restore the real symbol names."""
    from sympy.parsing.latex import parse_latex

    errors: list[str] = []
    expression = None
    # The lark backend ships with SymPy and needs no ANTLR runtime, so a default
    # install has a working path; ANTLR is tried only as a fallback.
    for backend in ("lark", None):
        try:
            expression = parse_latex(latex, backend=backend) if backend else parse_latex(latex)
            break
        except Exception as exc:
            errors.append(f"{backend or 'antlr'}: {type(exc).__name__}")
    if expression is None:
        raise ValueError("; ".join(errors))

    expression = resolve_ambiguity(expression, latex)
    return restore_placeholders(expression, placeholders)


def resolve_ambiguity(expression: Any, latex: str) -> Any:
    """Pick a reading when the grammar returns an ambiguous parse tree.

    ``\tanh(x)`` is ambiguous to the grammar: it offers ``tanh(x)``,
    ``tan(h(x))`` and ``tan(h*x)``. The disambiguating fact is in the source --
    the author wrote the single command ``\tanh``, so the reading that keeps
    ``tanh`` as one function is the one they meant.

    A tree we cannot resolve is left as it is, so it fails downstream rather
    than being silently replaced by a guess.
    """
    children = getattr(expression, "children", None)
    if children is None or getattr(expression, "data", "") != "_ambig":
        return expression

    candidates = [c for c in children if isinstance(c, sympy.Basic)]
    if not candidates:
        return expression

    named = {m.group(1) for m in re.finditer(r"\\([A-Za-z]+)", latex)}

    def score(candidate: Any) -> tuple[int, int]:
        functions = {
            type(node).__name__
            for node in sympy.preorder_traversal(candidate)
            if isinstance(node, sympy.Function)
        }
        # Prefer readings whose functions were actually written as commands,
        # then those introducing the fewest undefined functions.
        matched = len(functions & named)
        undefined = sum(
            1
            for node in sympy.preorder_traversal(candidate)
            if isinstance(node, sympy.core.function.AppliedUndef)
        )
        return (-matched, undefined)

    return min(candidates, key=score)


def restore_placeholders(expression: Any, placeholders: dict[str, str]) -> Any:
    """Swap placeholder symbols back for the names they stand for.

    A placeholder surviving into a result would be a lie of a different shape,
    so this runs on every parsed expression rather than only on request.
    """
    if not placeholders:
        return expression
    substitutions = {
        sympy.Symbol(token): MATH_CONSTANTS.get(real, sympy.Symbol(real))
        for token, real in placeholders.items()
    }
    return expression.subs(substitutions)


def _split_top_level(text: str, separator: str) -> list[str]:
    """Split on ``separator`` only outside braces, brackets and parentheses."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    index = 0
    while index < len(text):
        ch = text[index]
        if ch == "\\" and index + 1 < len(text):
            current.append(text[index : index + 2])
            index += 2
            continue
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if ch == separator and depth == 0:
            # '==', '<=' and '>=' are operators, not equation splits.
            previous = text[index - 1] if index else ""
            following = text[index + 1] if index + 1 < len(text) else ""
            if previous not in "<>=!" and following != "=":
                parts.append("".join(current))
                current = []
                index += 1
                continue
        current.append(ch)
        index += 1
    parts.append("".join(current))
    return parts


def parse_all(equations: list[str]) -> list[ParsedEquation]:
    """Parse a batch, for benchmarking the parse rate over a corpus."""
    return [parse_equation(e) for e in equations]


def parse_rate(results: list[ParsedEquation]) -> float:
    """Fraction that parsed. The first component of acceptance gate 2."""
    return sum(1 for r in results if r.ok) / len(results) if results else 0.0
