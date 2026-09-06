"""Draft auditing, export, watch lists and literature maps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from noether import export as export_module
from noether.answer.records import AnswerRecord, RecordedClaim, RecordedEquation
from noether.draft import audit_draft, render_audit
from noether.index.store import Store
from noether.maps import citation_links, cluster_papers, find_contradictions
from noether.watch import WatchList

from .test_index_and_gate import make_document

DRAFT = r"""
\documentclass{article}
\begin{document}
\title{A test draft}
We follow the method \cite{Real2020}. We also cite something undeclared
\cite{Missing2021}.
\begin{thebibliography}{9}
\bibitem{Real2020}
C. Abel et al. Phys. Rev. Lett. 124, 081803 (2020).
\bibitem{NeverCited}
S. Someone. Phys. Rev. D 92, 052008 (2015).
\end{thebibliography}
\end{document}
"""


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "corpus.sqlite")


@pytest.fixture
def draft_path(tmp_path: Path) -> Path:
    path = tmp_path / "paper.tex"
    path.write_text(DRAFT, encoding="utf-8")
    return path


class TestDraftAudit:
    def test_reads_a_draft_without_ingesting_it(self, draft_path: Path, store: Store) -> None:
        """A draft is the user's own file; it must not enter the corpus."""
        audit = audit_draft(draft_path, store, offline=True)
        assert audit.ok
        assert store.paper_ids() == []

    def test_finds_cite_keys_and_entries(self, draft_path: Path, store: Store) -> None:
        audit = audit_draft(draft_path, store, offline=True)
        assert audit.cite_keys == {"Real2020", "Missing2021"}
        assert {r.key for r in audit.references} == {"Real2020", "NeverCited"}

    def test_undeclared_key_is_reported(self, draft_path: Path, store: Store) -> None:
        """LaTeX only signals this as a bold [?], which authors miss."""
        audit = audit_draft(draft_path, store, offline=True)
        assert audit.undeclared_keys == {"Missing2021"}
        assert any(f.kind == "undeclared" for f in audit.findings)

    def test_uncited_entry_is_reported(self, draft_path: Path, store: Store) -> None:
        audit = audit_draft(draft_path, store, offline=True)
        assert any(f.kind == "cited-unread" and f.key == "NeverCited" for f in audit.findings)

    def test_offline_references_are_unchecked_not_unresolvable(
        self, draft_path: Path, store: Store
    ) -> None:
        """Offline mode consults no provider, so it cannot report a negative.

        Reporting every reference as 'unresolvable' would assert that no record
        exists when none was sought — the same collapse of "did not check" into
        "failed" that dimensional analysis avoids by keeping 'unknown' separate
        from 'inconsistent'.
        """
        audit = audit_draft(draft_path, store, offline=True)
        assert not audit.unresolvable, "offline mode must not claim a reference is missing"
        assert audit.unchecked, "offline references should be reported as unchecked"
        assert all(f.severity == "warning" for f in audit.unchecked)

    def test_unchecked_references_do_not_count_as_resolved(
        self, draft_path: Path, store: Store
    ) -> None:
        audit = audit_draft(draft_path, store, offline=True)
        assert audit.resolved_count == 0

    def test_exit_code_fails_on_a_risky_draft(self, draft_path: Path, store: Store) -> None:
        """The value of a pre-commit hook is entirely in this number."""
        assert audit_draft(draft_path, store, offline=True).exit_code == 1

    def test_exit_code_passes_a_clean_draft(self, tmp_path: Path, store: Store) -> None:
        clean = tmp_path / "clean.tex"
        clean.write_text(
            r"\documentclass{article}\begin{document}Prose only.\end{document}",
            encoding="utf-8",
        )
        audit = audit_draft(clean, store, offline=True)
        assert audit.undeclared_keys == set()
        assert audit.exit_code == 0

    def test_missing_file_is_reported_not_raised(self, tmp_path: Path, store: Store) -> None:
        audit = audit_draft(tmp_path / "nope.tex", store, offline=True)
        assert not audit.ok
        assert audit.error

    def test_render_lists_the_problems(self, draft_path: Path, store: Store) -> None:
        text = render_audit(audit_draft(draft_path, store, offline=True))
        assert "UNDECLARED" in text
        assert "Missing2021" in text


def _record() -> AnswerRecord:
    return AnswerRecord(
        answer_id="ans_test",
        question="How does the model work?",
        claims=[
            RecordedClaim(
                text="The model couples an atom to a cavity.",
                anchors=[{
                    "paper_id": "arXiv:0000.0001", "section_path": "I",
                    "start": 0, "end": 40, "quote": "The Jaynes-Cummings model couples an atom",
                }],
            )
        ],
        equations=[RecordedEquation(latex="E = m c^2", parsed=True, dimensions="consistent")],
        dropped=[{"claim": "Unsupported.", "reason": "unanchored", "detail": "no span"}],
        corpus_hash="abc123def456",
    )


class TestExport:
    def test_bibtex_covers_cited_papers(self) -> None:
        bib = export_module.to_bibtex(_record(), {"arXiv:0000.0001": "A model paper"})
        assert "@article{arxiv00000001," in bib
        assert "A model paper" in bib
        assert "eprint       = {0000.0001}" in bib

    def test_bibtex_excludes_uncited_papers(self) -> None:
        """An entry for a paper the answer did not use is the pattern we report on."""
        bib = export_module.to_bibtex(_record(), {"arXiv:9999.9999": "Unused"})
        assert "Unused" not in bib

    def test_latex_cites_match_the_bibtex_keys(self) -> None:
        record = _record()
        tex = export_module.to_latex(record)
        bib = export_module.to_bibtex(record)
        assert "\\cite{arxiv00000001}" in tex
        assert "@article{arxiv00000001," in bib

    def test_latex_records_what_was_dropped(self) -> None:
        """Dropping a claim silently would defeat the point of the gate."""
        tex = export_module.to_latex(_record())
        assert "Dropped by the verification gate" in tex
        assert "unanchored" in tex

    def test_latex_escapes_special_characters(self) -> None:
        record = _record()
        record.claims[0].text = "A 50% effect with x_1 & y"
        tex = export_module.to_latex(record)
        assert r"50\%" in tex and r"x\_1" in tex and r"\&" in tex

    def test_notebook_is_valid_json_with_code_cells(self) -> None:
        notebook = json.loads(export_module.to_notebook(_record()))
        assert notebook["nbformat"] == 4
        assert any(c["cell_type"] == "code" for c in notebook["cells"])

    def test_write_creates_both_latex_files(self, tmp_path: Path) -> None:
        written = export_module.write(_record(), tmp_path, "latex")
        assert {p.suffix for p in written} == {".tex", ".bib"}
        assert all(p.exists() for p in written)

    def test_unknown_format_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown format"):
            export_module.write(_record(), tmp_path, "pdf")


class TestWatchList:
    def test_add_and_list(self, tmp_path: Path) -> None:
        watches = WatchList(tmp_path / "w.json")
        watches.add("noise", "cat:quant-ph AND abs:decoherence")
        assert [w.name for w in watches.list()] == ["noise"]

    def test_persists_between_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "w.json"
        WatchList(path).add("noise", "q")
        assert len(WatchList(path).list()) == 1

    def test_remove(self, tmp_path: Path) -> None:
        watches = WatchList(tmp_path / "w.json")
        watches.add("noise", "q")
        assert watches.remove("noise") is True
        assert watches.remove("noise") is False

    def test_running_an_unknown_watch_raises(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError):
            WatchList(tmp_path / "w.json").run("nope", library=None)  # type: ignore[arg-type]


class TestMaps:
    def test_contradiction_needs_both_overlap_and_opposition(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "c.sqlite")
        store.add_document(make_document("arXiv:0000.0001"), [])
        chunks = store.chunks()
        # One paper cannot contradict itself here by construction.
        assert find_contradictions(chunks) == []

    def test_opposing_passages_are_flagged(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "c.sqlite")
        for paper_id, verb in (("arXiv:1111.1111", "increases"), ("arXiv:2222.2222", "decreases")):
            document = make_document(paper_id)
            # Must clear the chunker's MIN_CHARS floor, or nothing is indexed
            # and the test passes vacuously.
            document.text = (
                f"The measured dephasing rate {verb} with flux noise amplitude "
                "in the superconducting transmon qubit studied here, across the "
                "full range of applied flux bias and drive power we surveyed."
            )
            document.blocks[0].text = document.text
            document.blocks[0].span = type(document.blocks[0].span)(0, len(document.text))
            document.blocks = document.blocks[:1]
            document.equations = []
            store.add_document(document, [])

        found = find_contradictions(store.chunks(), min_overlap=0.05)
        assert found, "opposing direction words on a shared subject should be flagged"
        assert set(found[0].papers) == {"arXiv:1111.1111", "arXiv:2222.2222"}

    def test_clusters_cover_every_paper(self, tmp_path: Path) -> None:
        store = Store(tmp_path / "c.sqlite")
        ids = ["arXiv:1111.1111", "arXiv:2222.2222"]
        for paper_id in ids:
            store.add_document(make_document(paper_id), [])
        clustered = {p for c in cluster_papers(store, ids) for p in c.paper_ids}
        assert clustered == set(ids)

    def test_citation_links_counts_shared_references(self, tmp_path: Path) -> None:
        from noether.ingest.bibliography import BibEntry

        store = Store(tmp_path / "c.sqlite")
        shared = [BibEntry(key="Shared1963", raw="A shared reference")]
        for paper_id in ("arXiv:1111.1111", "arXiv:2222.2222"):
            store.add_document(make_document(paper_id), shared)
        links = citation_links(store, ["arXiv:1111.1111", "arXiv:2222.2222"])
        assert links["arXiv:1111.1111"]["arXiv:2222.2222"] == 1
