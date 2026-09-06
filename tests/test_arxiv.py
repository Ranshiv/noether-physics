"""arXiv identifier handling and Atom feed parsing."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from noether.sources.arxiv import ArxivError, _safe_extract, parse_feed, parse_id


class TestParseId:
    @pytest.mark.parametrize(
        "raw,expected_id,expected_version",
        [
            ("2001.11966", "2001.11966", None),
            ("2001.11966v2", "2001.11966", 2),
            ("arXiv:2001.11966v13", "2001.11966", 13),
            ("hep-th/9901001", "hep-th/9901001", None),
            ("cond-mat.str-el/0512xxx".replace("xxx", "001"), "cond-mat.str-el/0512001", None),
            ("https://arxiv.org/abs/2001.11966", "2001.11966", None),
            ("https://arxiv.org/pdf/2001.11966v2.pdf", "2001.11966", 2),
            ("  2001.11966  ", "2001.11966", None),
        ],
    )
    def test_accepts_known_forms(self, raw: str, expected_id: str, expected_version: int | None) -> None:
        assert parse_id(raw) == (expected_id, expected_version)

    @pytest.mark.parametrize("raw", ["", "not-an-id", "12.34", "2001.119660000v"])
    def test_rejects_junk(self, raw: str) -> None:
        with pytest.raises(ArxivError):
            parse_id(raw)


class TestParseFeed:
    def test_reads_both_identifier_schemes(self, atom_feed: str) -> None:
        papers = parse_feed(atom_feed)
        assert [p.arxiv_id for p in papers] == ["2001.11966", "hep-th/9901001"]
        assert [p.version for p in papers] == [1, 2]

    def test_extracts_metadata(self, atom_feed: str) -> None:
        paper = parse_feed(atom_feed)[0]
        assert paper.title.startswith("Measurement of the permanent electric dipole")
        assert paper.authors[:2] == ["C. Abel", "S. Afach"]
        assert paper.doi == "10.1103/PhysRevLett.124.081803"
        assert paper.journal_ref == "Phys. Rev. Lett. 124, 081803 (2020)"
        assert paper.primary_category == "hep-ex"
        assert paper.categories == ["hep-ex", "nucl-ex"]

    def test_captures_licence(self, atom_feed: str) -> None:
        """Licence drives figure-export gating, so it must survive parsing."""
        papers = parse_feed(atom_feed)
        assert papers[0].license == "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"
        assert papers[1].license == "http://creativecommons.org/licenses/by/4.0/"

    def test_unwraps_hard_wrapped_abstracts(self, atom_feed: str) -> None:
        assert parse_feed(atom_feed)[1].abstract == "Abstract text wrapped across lines."

    def test_versioned_id(self, atom_feed: str) -> None:
        assert parse_feed(atom_feed)[0].versioned_id == "2001.11966v1"

    def test_unparseable_xml_raises(self) -> None:
        with pytest.raises(ArxivError, match="unparseable"):
            parse_feed("<feed><entry>")

    def test_empty_feed_is_empty_not_an_error(self) -> None:
        assert parse_feed('<feed xmlns="http://www.w3.org/2005/Atom"></feed>') == []


class TestSafeExtract:
    def _tar(self, name: str, tmp_path: Path) -> tarfile.TarFile:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name)
            info.size = 3
            tar.addfile(info, io.BytesIO(b"abc"))
        buf.seek(0)
        return tarfile.open(fileobj=buf, mode="r:gz")

    def test_extracts_normal_member(self, tmp_path: Path) -> None:
        with self._tar("main.tex", tmp_path) as tar:
            _safe_extract(tar, tmp_path / "out")
        assert (tmp_path / "out" / "main.tex").read_bytes() == b"abc"

    def test_refuses_path_traversal(self, tmp_path: Path) -> None:
        """arXiv submissions are author-supplied archives, not trusted input."""
        with (
            self._tar("../escaped.tex", tmp_path) as tar,
            pytest.raises(ArxivError, match="path-traversing"),
        ):
            _safe_extract(tar, tmp_path / "out")
