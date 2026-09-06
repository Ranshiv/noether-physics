"""The document model, with emphasis on span anchoring.

Guarantee 2 says every claim is anchored to a verbatim span. These tests pin the
one behaviour that guarantee rests on: a fabricated or drifted anchor must fail
verification rather than pass quietly.
"""

from __future__ import annotations

import pytest

from noether.ingest.document import Anchor, Document, Equation, Span

TEXT = "The atom and field exchange a single excitation at frequency 2g."


@pytest.fixture
def doc() -> Document:
    return Document(
        paper_id="arXiv:1963.jc",
        title="Comparison of quantum and semiclassical radiation theories",
        text=TEXT,
        equations=[
            Equation(latex=r"H = \hbar g (a^\dagger \sigma_- + a \sigma_+)",
                     span=Span(0, 10), label="eq:jc", number=3),
            Equation(latex=r"\Omega = 2g\sqrt{n+1}", span=Span(10, 20), number=4),
        ],
    )


class TestSpan:
    def test_length(self) -> None:
        assert len(Span(4, 10)) == 6

    @pytest.mark.parametrize("start,end", [(-1, 5), (10, 4)])
    def test_rejects_invalid(self, start: int, end: int) -> None:
        with pytest.raises(ValueError):
            Span(start, end)


class TestQuoting:
    def test_quote_returns_exact_text(self, doc: Document) -> None:
        assert doc.quote(Span(0, 8)) == "The atom"

    def test_quote_past_end_raises(self, doc: Document) -> None:
        with pytest.raises(ValueError, match="runs past"):
            doc.quote(Span(0, len(TEXT) + 1))

    def test_anchor_carries_its_own_quote(self, doc: Document) -> None:
        anchor = doc.anchor(Span(4, 8), section_path="II")
        assert anchor.quote == "atom"
        assert anchor.section_path == "II"
        assert anchor.paper_id == doc.paper_id


class TestAnchorVerification:
    def test_honest_anchor_verifies(self, doc: Document) -> None:
        assert doc.verify_anchor(doc.anchor(Span(4, 8))) is True

    def test_fabricated_quote_fails(self, doc: Document) -> None:
        """The failure mode this whole project exists to catch."""
        fake = Anchor(doc.paper_id, "II", Span(4, 8), quote="proves supersymmetry")
        assert doc.verify_anchor(fake) is False

    def test_anchor_from_another_paper_fails(self, doc: Document) -> None:
        stolen = Anchor("arXiv:9999.99999", "I", Span(4, 8), quote="atom")
        assert doc.verify_anchor(stolen) is False

    def test_out_of_range_span_fails_without_raising(self, doc: Document) -> None:
        overrun = Anchor(doc.paper_id, "I", Span(0, len(TEXT) + 50), quote="whatever")
        assert doc.verify_anchor(overrun) is False


class TestEquationLookup:
    def test_by_author_number_not_discovery_order(self, doc: Document) -> None:
        """'Equation 3' means the one the authors numbered 3."""
        eq = doc.equation_by_number(3)
        assert eq is not None and eq.label == "eq:jc"

    def test_by_label(self, doc: Document) -> None:
        eq = doc.equation_by_label("eq:jc")
        assert eq is not None and eq.number == 3

    def test_missing_returns_none(self, doc: Document) -> None:
        assert doc.equation_by_number(99) is None
        assert doc.equation_by_label("eq:nope") is None

    def test_is_numbered(self, doc: Document) -> None:
        assert all(e.is_numbered for e in doc.equations)


def test_content_hash_is_stable_and_content_sensitive(doc: Document) -> None:
    first = doc.content_hash()
    assert first == doc.content_hash()
    doc.text += " Extra."
    assert doc.content_hash() != first
