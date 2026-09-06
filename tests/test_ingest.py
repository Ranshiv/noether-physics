"""LaTeX preprocessing, bibliography parsing, and document construction.

Each class here pins one of the constraints measured from a real paper
(arXiv:2001.11966) and recorded in docs/DEFERRED.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from noether.ingest.bibliography import (
    clean_tex,
    parse_bibtex,
    parse_references,
    parse_thebibliography,
)
from noether.ingest.latex import find_main_tex, parse_document
from noether.ingest.preprocess import (
    MacroTable,
    collect_macros,
    expand_macros,
    find_matching_brace,
    resolve_inputs,
    strip_comments,
    strip_definitions,
)


class TestStripComments:
    def test_removes_trailing_comment(self) -> None:
        assert strip_comments("real % noise\n").strip() == "real"

    def test_preserves_escaped_percent(self) -> None:
        assert "100\\%" in strip_comments("100\\% efficiency\n")

    def test_drops_fully_commented_line(self) -> None:
        assert strip_comments("%\\bibitem{Ghost}\n").strip() == ""

    def test_commented_graphics_are_removed(self) -> None:
        """The source we measured had one pointing at a nonexistent file."""
        source = "%\t\\includegraphics{missing}\n\t\\includegraphics{real}\n"
        out = strip_comments(source)
        assert "missing" not in out
        assert "real" in out


class TestFindMatchingBrace:
    def test_simple(self) -> None:
        assert find_matching_brace("{abc}", 0) == 4

    def test_nested(self) -> None:
        assert find_matching_brace("{a{b}c}", 0) == 6

    def test_ignores_escaped_braces(self) -> None:
        text = "{a\\{b}"
        assert find_matching_brace(text, 0) == len(text) - 1

    def test_unbalanced_returns_minus_one(self) -> None:
        assert find_matching_brace("{abc", 0) == -1


class TestInputResolution:
    def test_resolves_with_and_without_extension(self, tmp_path: Path) -> None:
        """One real document used both forms."""
        (tmp_path / "packages.tex").write_text("PACKAGES", encoding="utf-8")
        (tmp_path / "macros.tex").write_text("MACROS", encoding="utf-8")
        main = tmp_path / "main.tex"
        main.write_text("\\input{packages}\n\\input{macros.tex}\n", encoding="utf-8")

        out = resolve_inputs(main, tmp_path)
        assert "PACKAGES" in out
        assert "MACROS" in out

    def test_missing_target_is_left_alone(self, tmp_path: Path) -> None:
        main = tmp_path / "main.tex"
        main.write_text("\\input{nope}\n", encoding="utf-8")
        assert "\\input{nope}" in resolve_inputs(main, tmp_path)

    def test_cycle_terminates(self, tmp_path: Path) -> None:
        a = tmp_path / "a.tex"
        a.write_text("A\\input{b}", encoding="utf-8")
        (tmp_path / "b.tex").write_text("B\\input{a}", encoding="utf-8")
        assert "A" in resolve_inputs(a, tmp_path)


class TestMacros:
    def test_zero_arity(self) -> None:
        table = collect_macros("\\newcommand{\\nEDM}{neutron EDM}")
        assert expand_macros("The \\nEDM result", table) == "The neutron EDM result"

    def test_with_arguments(self) -> None:
        table = collect_macros("\\newcommand{\\unit}[2]{#1\\,\\mathrm{#2}}")
        assert expand_macros("\\unit{5}{m}", table) == "5\\,\\mathrm{m}"

    def test_optional_argument_default(self) -> None:
        table = collect_macros("\\newcommand{\\g}[1][n]{\\gamma_{#1}}")
        assert expand_macros("\\g", table) == "\\gamma_{n}"
        assert expand_macros("\\g[Hg]", table) == "\\gamma_{Hg}"

    def test_longest_name_wins(self) -> None:
        """A shorter macro must not clip a longer one that shares its prefix."""
        table = collect_macros("\\newcommand{\\aa}{SHORT}\\newcommand{\\aab}{LONG}")
        assert expand_macros("\\aab", table) == "LONG"

    def test_def_form(self) -> None:
        table = collect_macros("\\def\\hbar{\\hslash}")
        assert "hslash" in expand_macros("\\hbar", table)

    def test_nested_macros_expand(self) -> None:
        table = collect_macros("\\newcommand{\\a}{\\b}\\newcommand{\\b}{done}")
        assert expand_macros("\\a", table) == "done"

    def test_recursive_definition_terminates(self) -> None:
        """A self-referential macro must not hang ingestion."""
        table = MacroTable()
        table.macros.update(collect_macros("\\newcommand{\\loop}{\\loop x}").macros)
        assert isinstance(expand_macros("\\loop", table, passes=3), str)

    def test_definitions_are_removed_from_body(self) -> None:
        out = strip_definitions("\\newcommand{\\x}{y}Body text")
        assert "newcommand" not in out
        assert "Body text" in out


class TestBibliography:
    BIB = r"""
\begin{thebibliography}{10}
\bibitem{Luders1954}
G.~Luders. \newblock Mat.-fys. Medd. \textbf{28} (1954) 5.
\bibitem{Abel2020}
C.~Abel et al. Phys. Rev. Lett. \textbf{124}, 081803 (2020),
doi:10.1103/PhysRevLett.124.081803, arXiv:2001.11966.
\end{thebibliography}
"""

    def test_extracts_every_entry(self) -> None:
        entries = parse_thebibliography(self.BIB)
        assert [e.key for e in entries] == ["Luders1954", "Abel2020"]

    def test_mines_doi_and_arxiv_id(self) -> None:
        entry = parse_thebibliography(self.BIB)[1]
        assert entry.doi == "10.1103/PhysRevLett.124.081803"
        assert entry.arxiv_id == "2001.11966"
        assert entry.year == 2020
        assert entry.has_identifier

    def test_entry_without_identifier_is_flagged(self) -> None:
        """The common case: free text a fuzzy matcher must resolve."""
        entry = parse_thebibliography(self.BIB)[0]
        assert not entry.has_identifier
        assert entry.raw

    def test_clean_tex_strips_markup(self) -> None:
        assert clean_tex("G.~Luders \\newblock \\textbf{28}") == "G. Luders 28"

    def test_bibtex_entry(self) -> None:
        entries = parse_bibtex(
            '@article{key1, title = {A {Nested} Title}, year = {2020}, '
            'author = {A. One and B. Two}, doi = {10.1000/x}}'
        )
        assert len(entries) == 1
        entry = entries[0]
        assert entry.title == "A Nested Title"
        assert entry.authors == ["A. One", "B. Two"]
        assert entry.doi == "10.1000/x"
        assert entry.origin == "bibtex"

    def test_inline_bibliography_wins_over_bibtex(self) -> None:
        merged = parse_references(self.BIB, ["@article{Abel2020, title={Stale}}"])
        entry = next(e for e in merged if e.key == "Abel2020")
        assert entry.origin == "thebibliography"


def _write_paper(root: Path, body: str, preamble: str = "") -> Path:
    (root / "main.tex").write_text(
        f"\\documentclass{{article}}\n{preamble}\n\\begin{{document}}\n{body}\n"
        f"\\end{{document}}\n",
        encoding="utf-8",
    )
    return root / "main.tex"


class TestFindMainTex:
    def test_picks_the_file_declaring_the_document(self, tmp_path: Path) -> None:
        (tmp_path / "macros.tex").write_text("\\newcommand{\\x}{y}" * 200, encoding="utf-8")
        _write_paper(tmp_path, "Body")
        assert find_main_tex(tmp_path).name == "main.tex"


class TestParseDocument:
    def test_equation_numbering_follows_latex(self, tmp_path: Path) -> None:
        _write_paper(
            tmp_path,
            "\\begin{equation}\\label{eq:one}E=mc^2\\end{equation}\n"
            "\\begin{equation*}x=1\\end{equation*}\n"
            "\\begin{equation}\\label{eq:two}F=ma\\end{equation}\n",
        )
        doc, _ = parse_document(tmp_path, "test")
        numbered = [e for e in doc.equations if e.is_numbered]
        assert [e.number for e in numbered] == [1, 2]
        # The starred equation must not advance the counter.
        assert doc.equation_by_number(2).label == "eq:two"

    def test_align_rows_are_numbered_individually(self, tmp_path: Path) -> None:
        _write_paper(tmp_path, "\\begin{align}a &= 1 \\\\ b &= 2\\end{align}")
        doc, _ = parse_document(tmp_path, "test")
        assert [e.number for e in doc.equations] == [1, 2]

    def test_nonumber_row_is_unnumbered(self, tmp_path: Path) -> None:
        _write_paper(tmp_path, "\\begin{align}a &= 1 \\nonumber \\\\ b &= 2\\end{align}")
        doc, _ = parse_document(tmp_path, "test")
        assert [e.number for e in doc.equations] == [None, 1]

    def test_figure_caption_and_graphics(self, tmp_path: Path) -> None:
        _write_paper(
            tmp_path,
            "\\begin{figure}\\includegraphics[width=0.6\\columnwidth]{Apparatus}"
            "\\caption{The apparatus.}\\label{fig:app}\\end{figure}",
        )
        doc, _ = parse_document(tmp_path, "test")
        figure = doc.figures[0]
        assert figure.graphics == ["Apparatus"]
        assert figure.caption == "The apparatus."
        assert figure.label == "fig:app"

    def test_citations_record_their_sentence(self, tmp_path: Path) -> None:
        _write_paper(tmp_path, "First sentence. We follow the method of \\cite{Smith2020} here.")
        doc, _ = parse_document(tmp_path, "test")
        use = doc.citations[0]
        assert use.key == "Smith2020"
        assert "method" in use.context
        assert "First sentence" not in use.context

    def test_multiple_keys_in_one_cite(self, tmp_path: Path) -> None:
        _write_paper(tmp_path, "As shown \\cite{A2020,B2021}.")
        doc, _ = parse_document(tmp_path, "test")
        assert doc.cited_keys() == {"A2020", "B2021"}

    def test_sections_nest(self, tmp_path: Path) -> None:
        _write_paper(tmp_path, "\\section{Method}Text.\\subsection{Detail}More.")
        doc, _ = parse_document(tmp_path, "test")
        assert doc.sections[0].title == "Method"
        assert doc.sections[0].children[0].title == "Detail"

    def test_letters_without_sections_still_ingest(self, tmp_path: Path) -> None:
        """PRL letters use no sectioning commands at all."""
        _write_paper(tmp_path, "Just prose, no sections anywhere.")
        doc, _ = parse_document(tmp_path, "test")
        assert doc.sections == []
        assert "prose" in doc.text

    def test_every_span_resolves_to_its_text(self, tmp_path: Path) -> None:
        """The property guarantee 2 rests on."""
        _write_paper(
            tmp_path,
            "\\section{S}Prose here.\\begin{equation}E=mc^2\\end{equation}More prose.",
        )
        doc, _ = parse_document(tmp_path, "test")
        assert doc.blocks
        for block in doc.blocks:
            assert doc.quote(block.span) == block.text
        for equation in doc.equations:
            assert doc.quote(equation.span) == equation.latex

    def test_anchors_built_from_blocks_verify(self, tmp_path: Path) -> None:
        _write_paper(tmp_path, "A claim worth anchoring.")
        doc, _ = parse_document(tmp_path, "test")
        anchor = doc.anchor(doc.blocks[0].span, doc.blocks[0].section_path)
        assert doc.verify_anchor(anchor)

    def test_macros_expand_before_equations_are_read(self, tmp_path: Path) -> None:
        _write_paper(
            tmp_path,
            "\\begin{equation}\\gn = 1\\end{equation}",
            preamble="\\newcommand{\\gn}{\\gamma_{\\rm n}}",
        )
        doc, _ = parse_document(tmp_path, "test")
        assert "gamma" in doc.equations[0].latex
        assert doc.macro_count == 1


@pytest.mark.live
class TestRealPaper:
    """Runs against a fetched e-print. Needs NOETHER_LIVE=1 and a warm cache."""

    def test_nedm_letter(self) -> None:
        root = Path.home() / "AppData/Local/noether/cache/unpacked/2001.11966"
        if not root.exists():
            pytest.skip("run: noether fetch 2001.11966 --unpack")
        doc, refs = parse_document(root, "arXiv:2001.11966")
        assert doc.macro_count > 90
        assert len([e for e in doc.equations if e.is_numbered]) == 7
        assert len([e for e in doc.equations if e.label]) == 5
        assert len(doc.figures) == 4
        # 44 bibitem lines in the source, 7 of them commented out.
        assert len(refs) == 37
        for block in doc.blocks:
            assert doc.quote(block.span) == block.text
