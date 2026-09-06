"""Corpus store, retrieval, and the emission gate -- guarantees 1 and 2."""

from __future__ import annotations

from pathlib import Path

import pytest

from noether.answer.pipeline import (
    DraftClaim,
    corpus_hash,
    gather_evidence,
    render_markdown,
    submit,
)
from noether.answer.records import AnswerRecord
from noether.index.chunk import chunk_document
from noether.index.embed import HashingEmbedder, cosine, tokenize
from noether.index.retrieve import Retriever
from noether.index.store import Store
from noether.ingest.bibliography import BibEntry
from noether.ingest.document import Anchor, Block, Document, Equation, Span
from noether.provenance import bundle as bundles
from noether.verify.gate import Claim, Gate


def make_document(paper_id: str = "arXiv:0000.0001") -> Document:
    text = (
        "The Jaynes-Cummings model couples a two-level atom to a single cavity mode. "
        "Vacuum Rabi oscillations occur at frequency 2g even with the field in its vacuum state.\n\n"
        "H = \\hbar g (a^\\dagger \\sigma_- + a \\sigma_+)\n\n"
        "Decoherence damps these oscillations at a rate set by the cavity linewidth kappa."
    )
    first = Span(0, 178)
    equation = Span(180, 224)
    third = Span(226, len(text))
    return Document(
        paper_id=paper_id,
        title="A model paper",
        text=text,
        blocks=[
            Block("paragraph", text[first.start:first.end], first, "I"),
            Block("equation", text[equation.start:equation.end], equation, "I"),
            Block("paragraph", text[third.start:third.end], third, "II"),
        ],
        equations=[Equation(latex=text[equation.start:equation.end], span=equation, number=1, label="eq:jc")],
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "corpus.sqlite")
    document = make_document()
    store.add_document(
        document,
        [BibEntry(key="Jaynes1963", raw="Jaynes and Cummings, Proc. IEEE 51, 89 (1963)")],
    )
    return store


class TestStore:
    def test_round_trip(self, store: Store) -> None:
        assert store.paper_ids() == ["arXiv:0000.0001"]
        assert store.has_paper("arXiv:0000.0001")
        assert "Jaynes-Cummings" in (store.paper_text("arXiv:0000.0001") or "")

    def test_equations_keep_author_numbering(self, store: Store) -> None:
        equations = store.equations("arXiv:0000.0001")
        assert equations[0]["number"] == 1
        assert equations[0]["label"] == "eq:jc"

    def test_chunk_spans_index_the_paper_text(self, store: Store) -> None:
        """The property every anchor depends on."""
        text = store.paper_text("arXiv:0000.0001") or ""
        for chunk in store.chunks():
            assert text[chunk.start : chunk.end] == chunk.text

    def test_bm25_finds_a_passage(self, store: Store) -> None:
        hits = store.search_bm25("vacuum Rabi oscillations")
        assert hits and hits[0][0].paper_id == "arXiv:0000.0001"

    def test_punctuation_heavy_query_does_not_crash(self, store: Store) -> None:
        """Physics queries are full of characters FTS5 treats as syntax."""
        assert isinstance(store.search_bm25('spin-1/2 "a^\\dagger" (n+1)^2'), list)

    def test_reingest_replaces_rather_than_duplicates(self, store: Store) -> None:
        store.add_document(make_document(), [])
        assert len(store.chunks()) == len(store.chunks("arXiv:0000.0001"))
        assert store.stats()["papers"] == 1

    def test_resolution_cache(self, store: Store) -> None:
        store.record_resolution("10.1000/x", "doi", True, "crossref", "A Title")
        cached = store.cached_resolution("10.1000/x")
        assert cached and cached["resolved"] is True and cached["title"] == "A Title"


class TestChunking:
    def test_equations_are_not_merged_into_prose(self) -> None:
        """A chunk straddling a derivation reads as a claim but is half of one."""
        chunks = chunk_document(make_document())
        kinds = [c.kind for c in chunks]
        assert "equation" in kinds

    def test_every_chunk_span_matches_its_text(self) -> None:
        document = make_document()
        for chunk in chunk_document(document):
            assert document.text[chunk.span.start : chunk.span.end] == chunk.text


class TestEmbedding:
    def test_tokenize_drops_stopwords(self) -> None:
        assert "the" not in tokenize("the cavity mode")

    def test_identical_text_scores_one(self) -> None:
        embedder = HashingEmbedder()
        vector = embedder.encode("vacuum Rabi oscillation")
        assert cosine(vector, vector) == pytest.approx(1.0)

    def test_unrelated_text_scores_low(self) -> None:
        embedder = HashingEmbedder()
        a = embedder.encode("quantum cavity electrodynamics")
        b = embedder.encode("sediment transport in rivers")
        assert cosine(a, b) < 0.2

    def test_empty_text_is_a_zero_vector(self) -> None:
        assert not any(HashingEmbedder().encode(""))


class TestRetrieval:
    def test_hybrid_search_returns_spans(self, store: Store) -> None:
        hits = Retriever(store).search("vacuum Rabi oscillations", limit=3)
        assert hits
        assert hits[0].span.end > hits[0].span.start

    def test_hit_converts_to_a_verifying_anchor(self, store: Store) -> None:
        """Retrieval must produce anchors directly, or the chain breaks."""
        hit = Retriever(store).search("cavity mode", limit=1)[0]
        gate = Gate(store)
        ok, reason = gate.check_anchor(hit.to_anchor())
        assert ok, reason

    def test_both_retrievers_contribute(self, store: Store) -> None:
        hits = Retriever(store).search("decoherence cavity linewidth", limit=5)
        assert any(h.ranks for h in hits)

    def test_empty_corpus_returns_nothing(self, tmp_path: Path) -> None:
        assert Retriever(Store(tmp_path / "empty.sqlite")).search("anything") == []


class TestGate:
    def test_accepts_a_real_anchor(self, store: Store) -> None:
        gate = Gate(store)
        text = store.paper_text("arXiv:0000.0001") or ""
        quote = text[0:80]
        claim = Claim("The model couples an atom to a cavity.", [Anchor("arXiv:0000.0001", "I", Span(0, 80), quote)])
        assert gate.check([claim]).accepted

    def test_rejects_a_fabricated_quote(self, store: Store) -> None:
        """The core failure this project exists to prevent."""
        gate = Gate(store)
        claim = Claim(
            "This paper proves supersymmetry.",
            [Anchor("arXiv:0000.0001", "I", Span(0, 80), "This paper proves supersymmetry at last.")],
        )
        report = gate.check([claim])
        assert not report.accepted
        assert report.rejected[0].reason == "anchor-failed"

    def test_rejects_an_unanchored_claim(self, store: Store) -> None:
        report = Gate(store).check([Claim("An assertion with no evidence.", [])])
        assert report.rejected[0].reason == "unanchored"

    def test_rejects_a_span_past_the_end(self, store: Store) -> None:
        gate = Gate(store)
        claim = Claim("x", [Anchor("arXiv:0000.0001", "", Span(0, 999999), "y" * 40)])
        assert not gate.check([claim]).accepted

    def test_rejects_a_quote_too_short_to_be_evidence(self, store: Store) -> None:
        """Any short string can be found by chance in a long paper."""
        gate = Gate(store)
        claim = Claim("x", [Anchor("arXiv:0000.0001", "", Span(0, 3), "The")])
        assert not gate.check([claim]).accepted

    def test_rejects_an_anchor_into_an_unknown_paper(self, store: Store) -> None:
        gate = Gate(store)
        claim = Claim("x", [Anchor("arXiv:9999.9999", "", Span(0, 40), "z" * 40)])
        report = gate.check([claim])
        assert "not in the corpus" in report.rejected[0].detail

    def test_stale_offsets_are_distinguished_from_fabrication(self, store: Store) -> None:
        """Different problems deserve different diagnostics."""
        gate = Gate(store)
        text = store.paper_text("arXiv:0000.0001") or ""
        quote = text[40:120]
        claim = Claim("x", [Anchor("arXiv:0000.0001", "", Span(0, 80), quote)])
        assert "stale" in gate.check([claim]).rejected[0].detail

    def test_locate_builds_a_correct_anchor(self, store: Store) -> None:
        gate = Gate(store)
        anchor = gate.locate("arXiv:0000.0001", "Vacuum Rabi oscillations occur at frequency 2g")
        assert anchor is not None
        assert gate.check_anchor(anchor)[0]

    def test_unresolved_reference_is_dropped(self, store: Store) -> None:
        """Guarantee 1, at the gate."""
        report = Gate(store).check([], references={"Ghost2020": False, "Real2019": True})
        assert report.dropped_references == [("Ghost2020", "reference could not be resolved")]

    def test_partial_anchors_keep_only_the_good_ones(self, store: Store) -> None:
        gate = Gate(store)
        text = store.paper_text("arXiv:0000.0001") or ""
        good = Anchor("arXiv:0000.0001", "I", Span(0, 80), text[0:80])
        bad = Anchor("arXiv:0000.0001", "I", Span(0, 80), "invented text " * 4)
        accepted = gate.check([Claim("A claim.", [good, bad])]).accepted
        assert len(accepted[0].anchors) == 1


class TestPipeline:
    def test_evidence_pack_ids_are_stable(self, store: Store) -> None:
        pack = gather_evidence(store, "vacuum Rabi oscillations")
        assert pack.items[0].evidence_id == "e1"
        assert pack.corpus_hash

    def test_submit_accepts_grounded_claims(self, store: Store) -> None:
        pack = gather_evidence(store, "vacuum Rabi oscillations")
        record = submit(store, pack, [DraftClaim("Oscillations occur.", [pack.items[0].evidence_id])])
        assert len(record.claims) == 1
        assert not record.dropped

    def test_submit_drops_a_claim_citing_a_nonexistent_id(self, store: Store) -> None:
        pack = gather_evidence(store, "cavity")
        record = submit(store, pack, [DraftClaim("Invented.", ["e999"])])
        assert not record.claims
        assert "was not offered" in record.dropped[0]["detail"]

    def test_equations_are_checked_on_submission(self, store: Store) -> None:
        pack = gather_evidence(store, "cavity")
        record = submit(
            store, pack, [],
            equations=[{"latex": r"E = m c", "units": {"E": "joule", "m": "kilogram"}}],
        )
        assert record.equations[0].dimensions == "inconsistent"

    def test_corpus_hash_tracks_content(self, store: Store, tmp_path: Path) -> None:
        before = corpus_hash(store)
        other = make_document("arXiv:0000.0002")
        store.add_document(other, [])
        assert corpus_hash(store) != before

    def test_render_says_so_when_nothing_survived(self, store: Store) -> None:
        pack = gather_evidence(store, "cavity")
        record = submit(store, pack, [DraftClaim("Unanchored.", [])])
        assert "did not support" in render_markdown(record) or "No claim" in render_markdown(record)

    def test_record_round_trips(self, store: Store, tmp_path: Path) -> None:
        pack = gather_evidence(store, "cavity")
        record = submit(store, pack, [DraftClaim("A claim.", [pack.items[0].evidence_id])])
        path = record.save(tmp_path)
        assert AnswerRecord.load(path).content_hash() == record.content_hash()


class TestBundles:
    def test_build_and_verify(self, store: Store) -> None:
        pack = gather_evidence(store, "cavity")
        record = submit(store, pack, [DraftClaim("A claim.", [pack.items[0].evidence_id])])
        bundle = bundles.build(record, store.stats())
        assert bundles.verify(bundle).ok

    def test_tampering_is_detected(self, store: Store) -> None:
        record = AnswerRecord(answer_id="a", question="q", corpus_hash="h")
        bundle = bundles.build(record, {})
        bundle.record["question"] = "a different question"
        result = bundles.verify(bundle)
        assert not result.ok
        assert "content hash mismatch" in result.problems[0]

    def test_corpus_drift_is_reported(self, store: Store) -> None:
        record = AnswerRecord(answer_id="a", question="q", corpus_hash="old")
        result = bundles.verify(bundles.build(record, {}), corpus_hash="new")
        assert not result.ok
        assert "corpus has changed" in result.problems[0]

    def test_round_trip(self, store: Store, tmp_path: Path) -> None:
        bundle = bundles.build(AnswerRecord(answer_id="a", question="q"), {})
        path = bundle.save(tmp_path / "b.json")
        assert bundles.verify(bundles.Bundle.load(path)).ok


class TestGateScope:
    """What the gate does NOT establish, pinned so it is never overstated.

    Found by driving the MCP server: a claim asserting a *nonzero* neutron EDM
    "at five sigma" was accepted while citing a passage reporting
    d_n = (0.0 +/- 1.1) x 10^-26 e cm -- a null result. The claim contradicted
    its own evidence and passed, because the quote was real.
    """

    def test_a_claim_contradicting_its_evidence_still_passes(self, store: Store) -> None:
        """Not a bug to fix here: entailment needs a model, which this layer excludes.

        The test exists so the limitation is visible in the suite rather than
        discovered by a user trusting an answer.
        """
        gate = Gate(store)
        text = store.paper_text("arXiv:0000.0001") or ""
        quote = text[0:90]
        claim = Claim(
            "This paper disproves the Jaynes-Cummings model entirely.",
            [Anchor("arXiv:0000.0001", "I", Span(0, 90), quote)],
        )
        report = gate.check([claim])
        assert report.accepted, "anchored claims pass regardless of whether they follow"

    def test_report_states_what_was_not_checked(self, store: Store) -> None:
        wording = Gate(store).check([]).what_was_checked()
        assert "NOT checked" in wording
        assert "contradict its own evidence" in wording

    def test_rendered_answer_carries_the_caveat(self, store: Store) -> None:
        """A reader sees the scope next to the claims, not buried in docs."""
        pack = gather_evidence(store, "vacuum Rabi oscillations")
        record = submit(store, pack, [DraftClaim("A claim.", [pack.items[0].evidence_id])])
        rendered = render_markdown(record)
        assert "NOT checked" in rendered
        assert "Read the quotes." in rendered
