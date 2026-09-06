r"""Source hygiene: what must never reach the corpus text.

Both classes here come from a real observation. Reading a paper's ingested text
through the MCP server showed it beginning with::

    [NO \title GIVEN]
    [NO \author GIVEN]
    September 5, 2026
    ======================

    Département de Physique...   (rendered as D?partement)

Two distinct faults, one of them serious:

* ``\maketitle`` expands to those placeholders and, via ``\today``, to **the
  current date**. That is a fact the paper never stated, and because it changes
  daily it makes ``content_hash`` unstable — silently breaking the
  reproducibility guarantee.
* The source was Latin-1, decoded as UTF-8 with ``errors="replace"``, so every
  accented character became U+FFFD and was then indexed, retrieved and quotable
  in that state.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from noether.ingest.latex import parse_document
from noether.ingest.preprocess import read_source


def _current_years() -> set[str]:
    """Years \today could plausibly render, in any timezone.

    LaTeX renders the local date, and the test machine's timezone need not match
    UTC. On 31 December they differ, so both are checked rather than assuming.
    """
    now = datetime.datetime.now(datetime.UTC)
    return {
        str(now.year),
        str((now - datetime.timedelta(days=1)).year),
        str((now + datetime.timedelta(days=1)).year),
    }


def _write_paper(root: Path, body: str) -> Path:
    (root / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n" + body + "\n\\end{document}\n",
        encoding="utf-8",
    )
    return root / "main.tex"


class TestSourceEncoding:
    """arXiv sources are not all UTF-8, and mojibake gets indexed and quoted."""

    def test_utf8_source(self, tmp_path: Path) -> None:
        path = tmp_path / "u.tex"
        path.write_bytes("Université de Sherbrooke".encode())
        assert read_source(path) == "Université de Sherbrooke"

    def test_latin1_source_is_not_mangled(self, tmp_path: Path) -> None:
        """Decoding Latin-1 as UTF-8 with errors='replace' destroys every accent."""
        path = tmp_path / "l.tex"
        path.write_bytes("Département de Physique, Québec".encode("latin-1"))
        text = read_source(path)
        assert text == "Département de Physique, Québec"
        assert "\ufffd" not in text

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert read_source(tmp_path / "nope.tex") == ""

    def test_latin1_paper_ingests_with_accents_intact(self, tmp_path: Path) -> None:
        (tmp_path / "main.tex").write_bytes(
            (
                "\\documentclass{article}\n\\begin{document}\n"
                "Work done at the Département de Physique in Québec, Canada.\n"
                "\\end{document}\n"
            ).encode("latin-1")
        )
        document, _ = parse_document(tmp_path, "test")
        assert "Département" in document.text
        assert "\ufffd" not in document.text


class TestTitleFurniture:
    r"""``\maketitle`` renders placeholders and today's date into the body."""

    def _ingest(self, root: Path, body: str) -> str:
        _write_paper(root, body)
        document, _ = parse_document(root, "test")
        return document.text

    def test_maketitle_placeholders_are_removed(self, tmp_path: Path) -> None:
        text = self._ingest(tmp_path, "\\maketitle\nReal content follows here.")
        assert "[NO " not in text
        assert "Real content" in text

    def test_today_is_not_baked_into_the_corpus(self, tmp_path: Path) -> None:
        r"""The decisive one.

        ``\today`` injects a date the paper never stated, and makes
        ``content_hash`` change from one day to the next.
        """
        text = self._ingest(tmp_path, "\\maketitle\nSome physics.")
        assert not any(year in text for year in _current_years()), text[:120]

    def test_today_command_alone_is_stripped(self, tmp_path: Path) -> None:
        text = self._ingest(tmp_path, "Written on \\today by someone.")
        assert not any(year in text for year in _current_years()), text[:120]

    def test_content_hash_is_stable_across_runs(self, tmp_path: Path) -> None:
        """Guarantee 4 depends on this: same source, same hash."""
        _write_paper(tmp_path, "\\maketitle\nStable content.")
        first, _ = parse_document(tmp_path, "test")
        second, _ = parse_document(tmp_path, "test")
        assert first.content_hash() == second.content_hash()

    def test_real_content_survives_the_stripping(self, tmp_path: Path) -> None:
        """The filter must remove furniture without eating the paper."""
        text = self._ingest(
            tmp_path,
            "\\maketitle\n\\tableofcontents\nThe measured value was 2.5 GHz.",
        )
        assert "measured value was 2.5 GHz" in text
