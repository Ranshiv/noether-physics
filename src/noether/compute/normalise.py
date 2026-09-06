"""Normalising paper LaTeX into something SymPy's grammar accepts.

The rules here are not guesses. Each was measured by probing the parser directly
(``tests/test_symbolic.py::TestGrammarLimits`` pins the behaviour, so a SymPy
upgrade that fixes one shows up as a failing test).

The important discovery, and the reason this module exists at all: **the grammar
reads juxtaposed letters as multiplication.** Rewriting ``\\Omega`` to the plain
text ``Omega`` does not produce a symbol named Omega -- it produces the product
``O*m*e*g*a``. That parses cleanly and is silently, confidently wrong, which is
the single worst failure mode this project can have.

So every unsupported symbol becomes a **placeholder the grammar treats as one
token** (``X_{9001}``, in the subscripted form it does keep whole), and the real
name is substituted back into the SymPy expression after parsing. Nothing that
reaches a user is ever a letter-product standing in for a symbol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Symbol commands SymPy's LaTeX grammar rejects. Most lowercase Greek is
#: accepted natively and is deliberately absent -- but not all of it:
#: sigma, varsigma and iota are rejected, and sigma is far too common in
#: physics (cross sections, Pauli matrices, conductivity) to lose.
SYMBOL_NAMES = {
    r"\sigma": "sigma",
    r"\varsigma": "varsigma",
    r"\iota": "iota",
    r"\hbar": "hbar",
    r"\pi": "pi",
    r"\infty": "oo",
    r"\ell": "ell",
    r"\partial": "partial",
    r"\nabla": "nabla",
}

_UPPER_GREEK = (
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma",
    "Upsilon", "Phi", "Psi", "Omega",
)

#: Decorations that change how a symbol is drawn but not what it denotes.
_DECORATORS = ("vec", "hat", "tilde", "bar", "dot", "ddot", "mathcal", "mathbb", "boldsymbol")

#: Leibniz derivatives: \frac{d^2 x}{d t^2} and \frac{dx}{dt}.
_DERIV_RE = re.compile(
    r"\\frac\s*\{\s*\\?[d\u2202]\s*(?:\^\s*\{?(\d+)\}?)?\s*([A-Za-z\\][A-Za-z0-9_\\{}]*)\s*\}"
    r"\s*\{\s*\\?[d\u2202]\s*([A-Za-z\\][A-Za-z0-9_\\{}]*)\s*(?:\^\s*\{?(\d+)\}?)?\s*\}"
)

_DAGGER_RE = re.compile(r"([A-Za-z])\s*\^\s*\{?\s*\\dagger\s*\}?")
_DECORATOR_RE = re.compile(r"\\(" + "|".join(_DECORATORS) + r")\s*\{([^{}]*)\}")
_DOUBLE_BRACE_RE = re.compile(r"\{\{([^{}]*)\}\}")
_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")

#: Commands the grammar does understand and which must survive untouched.
KEPT_COMMANDS = frozenset(
    ["frac", "sqrt", "sum", "int", "prod", "lim", "log", "ln", "exp", "sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta", "kappa", "lambda", "mu", "nu", "xi", "rho", "varrho", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega", "cdot", "times", "pm", "mp"]
)


#: Environments carrying structure we cannot represent as a scalar expression.
#: Stripping them is worse than failing: removing egin leaves the literal
#: "{bmatrix}", which the grammar then reads as b*m*a*t*r*i*x -- a clean parse
#: of a meaningless product, reported as success.
UNSUPPORTED_ENVIRONMENTS = frozenset(
    ["matrix", "bmatrix", "pmatrix", "vmatrix", "Vmatrix", "smallmatrix", "array", "cases", "aligned", "split", "gathered", "subequations", "tabular"]
)

_ENV_RE = re.compile(r"\\begin\s*\{([A-Za-z*]+)\}")


def blocking_environment(text: str) -> str | None:
    """Name the first environment that makes this equation unrepresentable."""
    for match in _ENV_RE.finditer(text):
        name = match.group(1).rstrip("*")
        if name in UNSUPPORTED_ENVIRONMENTS:
            return name
    return None


@dataclass
class Normalised:
    """Normalised LaTeX plus everything needed to undo the placeholders."""

    text: str
    #: placeholder symbol name -> the real name it stands for.
    placeholders: dict[str, str] = field(default_factory=dict)
    #: Information-losing substitutions, reported to the user verbatim.
    notes: list[str] = field(default_factory=list)
    #: The relational operator, when the statement was not an equality.
    relation: str | None = None
    #: Set when the equation cannot be represented at all; parsing must fail.
    blocked: str = ""


#: Placeholder subscripts are letters only. The grammar accepts ``X_{aa}`` as a
#: single symbol but rejects ``X_{9001}`` and ``X_{q1}`` outright -- digits in a
#: subscript are a parse error, which is why the encoding below is base-26.
PLACEHOLDER_PREFIX = "X_{zz"


class _Placeholders:
    """Allocates tokens the LaTeX grammar keeps whole.

    ``X_{zzaa}`` works because a subscripted letter run parses as one symbol.
    The ``zz`` prefix keeps placeholders from colliding with a real subscript a
    paper might use.
    """

    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}
        self._by_name: dict[str, str] = {}
        self._next = 0

    @staticmethod
    def _encode(index: int) -> str:
        """Base-26 in lowercase letters, at least two characters."""
        letters = ""
        value = index
        while True:
            letters = chr(ord("a") + value % 26) + letters
            value //= 26
            if value == 0:
                break
        return letters.rjust(2, "a")

    def token(self, real_name: str) -> str:
        if real_name in self._by_name:
            return self._by_name[real_name]
        placeholder = f"{PLACEHOLDER_PREFIX}{self._encode(self._next)}}}"
        self._next += 1
        self.mapping[placeholder] = real_name
        self._by_name[real_name] = placeholder
        return placeholder


def collapse_double_braces(text: str) -> str:
    """``f_{{n}}`` is common after macro expansion and the grammar rejects it."""
    previous = None
    while previous != text:
        previous = text
        text = _DOUBLE_BRACE_RE.sub(r"{\1}", text)
    return text


def strip_decorators(text: str) -> str:
    r"""``\vec{B}`` and ``\hat{H}`` denote the same symbol as ``B`` and ``H``."""
    previous = None
    while previous != text:
        previous = text
        text = _DECORATOR_RE.sub(r"\2", text)
    return text


def substitute_derivatives(text: str, placeholders: _Placeholders) -> tuple[str, list[str]]:
    """Replace Leibniz derivatives with opaque symbols, reporting each.

    The grammar has no derivative production. The choice is between failing every
    differential equation and carrying the derivative as an opaque symbol; we
    carry it and say so, because 'parsed with d/dt as a symbol' is useful for
    structure and useless for calculus, and the caller must know which it has.
    """
    notes: list[str] = []

    def replace(match: re.Match[str]) -> str:
        order = match.group(1) or match.group(4) or "1"
        function = _identifier(match.group(2))
        variable = _identifier(match.group(3))
        suffix = order if order != "1" else ""
        name = f"d{suffix}_{function}_d{variable}{suffix}"
        power = f"^{order}" if order != "1" else ""
        notes.append(
            f"derivative d{power}{function}/d{variable}{power} carried as the opaque symbol {name}"
        )
        return f" {placeholders.token(name)} "

    return _DERIV_RE.sub(replace, text), notes


#: An optional subscript trailing a symbol: _x, _{n}, _{eff}.
_TRAILING_SUBSCRIPT = r"(?:\s*_\s*(\{[^{}]*\}|[A-Za-z0-9]))?"


def substitute_symbols(text: str, placeholders: _Placeholders) -> str:
    """Swap grammar-hostile symbol commands for whole-token placeholders.

    A trailing subscript is folded into the placeholder's name rather than left
    behind. A placeholder is itself a subscripted token, so leaving
    ``\\sigma_x`` as ``X_{zzaa}_x`` yields a double subscript the grammar
    rejects. Folding gives one symbol named ``sigma_x`` -- which is what the
    author wrote in the first place.
    """
    # Creation operators first: a^\\dagger is one symbol, not a power.
    text = _DAGGER_RE.sub(lambda m: f" {placeholders.token(m.group(1) + 'dag')} ", text)

    def replace(match: re.Match[str], base: str) -> str:
        subscript = match.group(1)
        if subscript:
            suffix = re.sub(r"[^A-Za-z0-9]", "", subscript.strip("{}"))
            base = f"{base}_{suffix}" if suffix else base
        return f" {placeholders.token(base)} "

    for name in _UPPER_GREEK:
        text = re.sub(
            rf"\\{name}(?![A-Za-z]){_TRAILING_SUBSCRIPT}",
            lambda m, n=name: replace(m, n),
            text,
        )
    for command, real in SYMBOL_NAMES.items():
        text = re.sub(
            re.escape(command) + r"(?![A-Za-z])" + _TRAILING_SUBSCRIPT,
            lambda m, r=real: replace(m, r),
            text,
        )
    return text


def strip_unknown_commands(text: str) -> tuple[str, list[str]]:
    """Drop commands the grammar does not know, recording what was removed."""
    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        if match.group(1) in KEPT_COMMANDS:
            return match.group(0)
        removed.append(match.group(1))
        return " "

    return _COMMAND_RE.sub(replace, text), removed


def fix_leading_minus(text: str) -> str:
    r"""``-\omega^2 x`` is rejected by the grammar; ``0 -\omega^2 x`` is not."""
    stripped = text.lstrip()
    return f"0 {stripped}" if stripped.startswith("-") else text


def _identifier(raw: str) -> str:
    """Reduce a LaTeX fragment to a bare identifier."""
    return re.sub(r"[^A-Za-z0-9]", "", raw.replace("\\", "")) or "f"


#: Commands that take an argument in parentheses and must NOT gain a \cdot
#: before it -- sin(x) is application, omega(x) is multiplication.
FUNCTION_COMMANDS = frozenset(
    ["sin", "cos", "tan", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan", "log", "ln", "exp", "lim", "sum", "int", "prod", "sqrt", "frac"]
)

#: How many braced arguments each command consumes. Its arguments belong to it
#: and must never be separated from it by an inserted \cdot -- doing that turns
#: rac{1}{2} into "one half times two", which parses and is wrong.
#: Relational commands. Dropping these silently turns an inequality into a
#: product -- an uncertainty relation would come back as an equation, which is
#: a different and false statement. They are recognised, not discarded.
RELATIONS = {
    "geq": ">=", "ge": ">=", "leq": "<=", "le": "<=", "neq": "!=", "ne": "!=",
    "gg": ">>", "ll": "<<", "approx": "~=", "sim": "~", "simeq": "~=",
    "propto": "propto", "equiv": "==",
}

ARGUMENT_ARITY = {"frac": 2, "sqrt": 1, "text": 1, "mathrm": 1, "overline": 1, "underline": 1}


_UNIT_RE = re.compile(
    r"""
      (?P<command>\\[A-Za-z]+)          # \omega, \frac, \sqrt
    | (?P<group>\{)                     # a braced group, matched by scanner
    | (?P<open>\()
    | (?P<close>\))
    | (?P<script>[\^_])
    | (?P<number>\d+(?:\.\d+)?)
    | (?P<name>[A-Za-z])
    | (?P<operator>[+\-*/=<>,;!])
    | (?P<space>\s+)
    """,
    re.VERBOSE,
)


def insert_explicit_multiplication(text: str) -> str:
    r"""Turn implicit juxtaposition into explicit ``\cdot``.

    SymPy's grammar cannot resolve ``\omega (a+1)`` -- it returns an *ambiguous
    parse tree* rather than an expression, because the juxtaposition could mean
    application or multiplication. It also rejects ``\omega^{2} x`` outright.
    Both are fixed by saying which we mean.

    We mean multiplication. In physics notation a Greek letter beside a
    parenthesis is a coefficient essentially always, and genuine function
    application is written with a name the grammar already knows
    (``\sin``, ``\log``) -- those are excluded via ``FUNCTION_COMMANDS``.
    This is a real semantic choice and it is stated here rather than buried.
    """
    units: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char == "{":
            close = _matching(text, index)
            units.append(text[index : close + 1])
            index = close + 1
            continue
        match = _UNIT_RE.match(text, index)
        if not match:
            units.append(char)
            index += 1
            continue
        units.append(match.group(0))
        index = match.end()

    return _assemble(units)


def _assemble(units: list[str]) -> str:
    r"""Join units, inserting \cdot between adjacent operands.

    Sub- and superscripts bind tightly to what precedes them, so ``X``, ``_``
    and ``{zzaa}`` must come back out as ``X_{zzaa}`` and never as
    ``X _ {zzaa}`` -- spacing a subscript apart from its base changes what the
    grammar sees into something it rejects.
    """
    out: list[str] = []
    attach_next = False
    pending_args = 0

    for unit in units:
        if unit in ("^", "_"):
            out.append(unit)          # binds to the previous unit
            attach_next = True
            continue
        if attach_next:
            out.append(unit)          # the script's argument
            attach_next = False
            continue
        if pending_args and unit.startswith("{"):
            # A command argument IS an expression, so its interior needs the
            # same treatment. A subscript group is a name part and never gets
            # this -- recursing into {zzaa} would shatter it into z*z*a*a.
            out.append("{" + insert_explicit_multiplication(unit[1:-1]) + "}")
            pending_args -= 1
            continue
        if unit.startswith("\\") and unit[1:] in ARGUMENT_ARITY:
            if out and _ends_operand(_last_operand(out)):
                out.append(r" \cdot ")
            elif out:
                out.append(" ")
            out.append(unit)
            pending_args = ARGUMENT_ARITY[unit[1:]]
            continue
        if out and _ends_operand(_last_operand(out)) and _starts_operand(unit):
            if not _is_application(_last_operand(out), unit):
                out.append(r" \cdot ")
            # else: application binds tight -- 	anh(x), never 	anh (x)
        elif out:
            out.append(" ")
        out.append(unit)

    return "".join(out).strip()


def _last_operand(out: list[str]) -> str:
    """The last unit that was not a script marker or separator."""
    for token in reversed(out):
        if token.strip() and token not in ("^", "_"):
            return token
    return ""


def _ends_operand(token: str) -> bool:
    if token in (r"\cdot", "^", "_") or token.startswith(("+", "-", "*", "/", "=", "<", ">", ",", ";")):
        return False
    if token.startswith("\\"):
        # A bare command name still needs its argument; \frac does not end one.
        return token[1:] not in FUNCTION_COMMANDS
    return bool(token) and (token[-1].isalnum() or token[-1] in "})")


def _starts_operand(token: str) -> bool:
    if token in ("^", "_", r"\cdot"):
        return False
    if token.startswith("\\"):
        return True
    return bool(token) and (token[0].isalnum() or token[0] in "({")


def _is_application(previous: str, following: str) -> bool:
    """True when the pair is function application, not multiplication."""
    return previous.startswith("\\") and previous[1:] in FUNCTION_COMMANDS


def _matching(text: str, open_index: int) -> int:
    """Index of the brace matching the one at ``open_index``."""
    depth = 0
    index = open_index
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            index += 2
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return len(text) - 1
