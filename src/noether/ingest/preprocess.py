"""LaTeX preprocessing: comments, ``\\input`` resolution, macro expansion.

All three are mandatory rather than nice to have, and each was forced by
measurement against a real paper (arXiv:2001.11966, recorded in docs/DEFERRED.md):

* A commented-out ``\\includegraphics`` points at a file that does not exist, so
  comments must go before anything reads structure.
* ``\\input`` appears both with and without the ``.tex`` extension in one document.
* 100 ``\\newcommand`` definitions live in a separate file. Without expansion the
  equation LaTeX never reaches SymPy in a parseable state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

#: How deep \input nesting may go before we assume a cycle.
MAX_INPUT_DEPTH = 12

#: How many expansion passes a macro body gets. Macros defined in terms of other
#: macros are common; unbounded expansion on a recursive definition is not.
MAX_MACRO_PASSES = 8

_INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")

_NEWCOMMAND_RE = re.compile(
    r"\\(?:new|renew|provide)command\*?\s*"
    r"(?:\{\\([A-Za-z@]+)\}|\\([A-Za-z@]+))"          # name, braced or bare
    r"\s*(?:\[(\d+)\])?"                                # arity
    r"\s*(?:\[([^\]]*)\])?"                             # optional-arg default
    r"\s*\{",                                           # body opens here
)

_DEF_RE = re.compile(r"\\def\s*\\([A-Za-z@]+)\s*\{")


def strip_comments(text: str) -> str:
    """Remove LaTeX comments, preserving escaped ``\\%``.

    A trailing ``%`` at end of line in LaTeX also suppresses the newline. We keep
    the newline: our spans index plain text a human reads, not a typeset box.
    """
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        result: list[str] = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and i + 1 < len(line):
                # An escaped character passes through whole, so \% is not a comment.
                result.append(line[i : i + 2])
                i += 2
                continue
            if ch == "%":
                # Keep the line terminator so line structure survives.
                trailing = "\n" if line.endswith("\n") else ""
                result.append(trailing)
                break
            result.append(ch)
            i += 1
        out.append("".join(result))
    return "".join(out)


def find_matching_brace(text: str, open_index: int) -> int:
    """Index of the ``}`` matching the ``{`` at ``open_index``, or -1.

    Brace counting must ignore escaped braces; a macro body containing ``\\{``
    otherwise terminates in the wrong place.
    """
    if open_index >= len(text) or text[open_index] != "{":
        return -1
    depth = 0
    i = open_index
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def read_source(path: Path) -> str:
    """Read a LaTeX file, trying the encodings arXiv submissions actually use.

    Decoding an older Latin-1 source as UTF-8 with ``errors="replace"`` turns
    every accented character into U+FFFD, so "Departement de Physique,
    Universite de Sherbrooke, Quebec" arrives full of replacement characters and
    is then indexed, quoted and cited in that state. Latin-1 decodes any byte
    sequence without raising, so it is the last resort rather than a guess.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return ""

    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def resolve_inputs(main: Path, root: Path, depth: int = 0, seen: set[Path] | None = None) -> str:
    """Inline ``\\input`` and ``\\include`` targets, depth-first.

    Targets appear with and without the ``.tex`` extension in the same document,
    so both are tried. A missing target is left as the literal command rather
    than raising: a paper that references a file arXiv did not ship should still
    ingest, minus that fragment.
    """
    seen = seen if seen is not None else set()
    text = read_source(main)
    if not text:
        return ""

    text = strip_comments(text)
    if depth >= MAX_INPUT_DEPTH:
        return text

    def substitute(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        for candidate in (root / target, root / f"{target}.tex"):
            resolved = candidate.resolve()
            if resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                return resolve_inputs(resolved, root, depth + 1, seen)
        return match.group(0)

    return _INPUT_RE.sub(substitute, text)


@dataclass
class Macro:
    """One user-defined command."""

    name: str
    arity: int
    body: str
    #: Default value for the first argument when declared optional.
    default: str | None = None


@dataclass
class MacroTable:
    """Collected ``\\newcommand`` / ``\\def`` definitions."""

    macros: dict[str, Macro] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.macros)

    def __contains__(self, name: str) -> bool:
        return name in self.macros

    def add(self, macro: Macro) -> None:
        self.macros[macro.name] = macro


def collect_macros(text: str) -> MacroTable:
    """Scan for macro definitions. Later definitions win, as in LaTeX."""
    table = MacroTable()

    for match in _NEWCOMMAND_RE.finditer(text):
        name = match.group(1) or match.group(2)
        open_brace = match.end() - 1
        close = find_matching_brace(text, open_brace)
        if close == -1:
            continue
        table.add(
            Macro(
                name=name,
                arity=int(match.group(3) or 0),
                body=text[open_brace + 1 : close],
                default=match.group(4),
            )
        )

    for match in _DEF_RE.finditer(text):
        open_brace = match.end() - 1
        close = find_matching_brace(text, open_brace)
        if close == -1:
            continue
        # \def with parameter text is rare in papers and hard to do correctly;
        # treating it as zero-arity is better than mangling the body.
        table.add(Macro(name=match.group(1), arity=0, body=text[open_brace + 1 : close]))

    return table


def strip_definitions(text: str) -> str:
    """Remove definition bodies once collected, so they are not read as content."""
    for pattern in (_NEWCOMMAND_RE, _DEF_RE):
        while True:
            match = pattern.search(text)
            if not match:
                break
            close = find_matching_brace(text, match.end() - 1)
            if close == -1:
                # Unbalanced definition: drop the header alone to make progress.
                text = text[: match.start()] + text[match.end() :]
                continue
            text = text[: match.start()] + text[close + 1 :]
    return text


def _read_argument(text: str, index: int) -> tuple[str, int]:
    """Read one macro argument starting at ``index``. Returns (value, next index).

    A braced group is taken whole; otherwise a single token, which is how LaTeX
    treats ``\\frac12``.
    """
    while index < len(text) and text[index] in " \t\n":
        index += 1
    if index >= len(text):
        return "", index
    if text[index] == "{":
        close = find_matching_brace(text, index)
        if close == -1:
            return "", index
        return text[index + 1 : close], close + 1
    if text[index] == "\\" and index + 1 < len(text):
        match = re.match(r"\\[A-Za-z@]+|\\.", text[index:])
        if match:
            return match.group(0), index + match.end()
    return text[index], index + 1


def expand_macros(text: str, table: MacroTable, passes: int = MAX_MACRO_PASSES) -> str:
    """Substitute user macros, repeatedly, until stable or out of passes.

    Bounded rather than fixpoint: a self-referential definition is a real thing
    to find in the wild and must not hang ingestion.
    """
    if not table.macros:
        return text

    # Longest names first so \alphaBeta is not clipped by a shorter \alpha.
    names = sorted(table.macros, key=len, reverse=True)
    pattern = re.compile(r"\\(" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z@])")

    for _ in range(passes):
        out: list[str] = []
        cursor = 0
        changed = False

        for match in pattern.finditer(text):
            if match.start() < cursor:
                continue
            macro = table.macros[match.group(1)]
            index = match.end()
            args: list[str] = []

            if macro.arity:
                # An optional first argument arrives in brackets, not braces.
                if macro.default is not None and index < len(text) and text[index] == "[":
                    end = text.find("]", index)
                    args.append(macro.default if end == -1 else text[index + 1 : end])
                    index = index + 1 if end == -1 else end + 1
                elif macro.default is not None:
                    args.append(macro.default)
                while len(args) < macro.arity:
                    value, index = _read_argument(text, index)
                    args.append(value)

            body = macro.body
            for position, value in enumerate(args, start=1):
                body = body.replace(f"#{position}", value)

            out.append(text[cursor : match.start()])
            out.append(body)
            cursor = index
            changed = True

        out.append(text[cursor:])
        text = "".join(out)
        if not changed:
            break

    return text


#: Journal abbreviation macros used across physics bibliographies. They are
#: defined in style files (aas_macros.sty and friends) that a submission may or
#: may not ship, and losing them costs the journal name -- which is most of what
#: a reference matcher has to work with when the reference states no title.
JOURNAL_MACROS = {
    "prl": "Phys. Rev. Lett.", "pra": "Phys. Rev. A", "prb": "Phys. Rev. B",
    "prc": "Phys. Rev. C", "prd": "Phys. Rev. D", "pre": "Phys. Rev. E",
    "prx": "Phys. Rev. X", "physrev": "Phys. Rev.", "rmp": "Rev. Mod. Phys.",
    "npb": "Nucl. Phys. B", "npa": "Nucl. Phys. A", "plb": "Phys. Lett. B",
    "jhep": "JHEP", "jcap": "JCAP", "epjc": "Eur. Phys. J. C",
    "epja": "Eur. Phys. J. A", "nima": "Nucl. Instrum. Methods A",
    "aj": "Astron. J.", "apj": "Astrophys. J.", "apjl": "Astrophys. J. Lett.",
    "aap": "Astron. Astrophys.", "mnras": "Mon. Not. R. Astron. Soc.",
    "araa": "Annu. Rev. Astron. Astrophys.", "pasp": "Publ. Astron. Soc. Pac.",
    "nat": "Nature", "science": "Science", "jpb": "J. Phys. B",
    "jap": "J. Appl. Phys.", "apl": "Appl. Phys. Lett.", "njp": "New J. Phys.",
}


def journal_macro_table() -> MacroTable:
    """A macro table of journal abbreviations, used only as a fallback."""
    table = MacroTable()
    for name, expansion in JOURNAL_MACROS.items():
        table.add(Macro(name=name, arity=0, body=expansion))
    return table


def preprocess(main: Path, root: Path) -> tuple[str, MacroTable]:
    """Full pipeline: inline inputs, collect macros, drop definitions, expand.

    Returns the expanded body and the macro table, the latter so callers can
    report how many definitions a paper carried.
    """
    text = resolve_inputs(main, root)

    # Style files carry definitions the document never \inputs. aas_macros.sty
    # is where \pra and \prl live, and dropping them costs the journal name
    # from every reference that used one.
    table = collect_macros(text)
    for style in sorted(root.rglob("*.sty")):
        try:
            for name, macro in collect_macros(style.read_text(encoding="utf-8", errors="replace")).macros.items():
                table.macros.setdefault(name, macro)
        except OSError:
            continue

    # Count only what the document itself defined: our journal fallbacks are
    # not the paper's macros, and reporting them as such is misleading.
    own_macros = MacroTable(macros=dict(table.macros))

    for name, macro in journal_macro_table().macros.items():
        table.macros.setdefault(name, macro)

    text = strip_definitions(text)
    return expand_macros(text, table), own_macros
