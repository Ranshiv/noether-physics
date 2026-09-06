"""Reference matching -- guarantee 1's hardest case.

A title-only matcher resolved 3 of 37 references on a real paper. These tests
pin the reason and the fix: physics references usually state no title at all, so
matching corroborates the fields they *do* carry.
"""

from __future__ import annotations

from noether.verify.matching import (
    MATCH_THRESHOLD,
    Candidate,
    best_candidate,
    corroborate,
    extract_fields,
    from_crossref,
    from_inspire,
    from_openalex,
    title_similarity,
)


class TestExtractFields:
    def test_year_from_parentheses(self) -> None:
        fields = extract_fields("C. Abel et al. Phys. Rev. Lett. 124, 081803 (2020)")
        assert fields.year == 2020

    def test_volume_and_page(self) -> None:
        fields = extract_fields("Phys. Rev. Lett. 124, 081803 (2020)")
        assert fields.volume == "124"
        assert fields.page == "081803"

    def test_surname_extraction_skips_journal_words(self) -> None:
        """'Phys' and 'Review' are not authors."""
        fields = extract_fields("G. Luders. Physical Review Letters 28 (1954) 5.")
        assert "Luders" in fields.surnames
        assert "Review" not in fields.surnames
        assert "Physical" not in fields.surnames

    def test_reference_with_no_title_still_yields_fields(self) -> None:
        """The common physics case, and the one that broke the old matcher."""
        fields = extract_fields("G. Luders. Mat.-fys. Medd. 28 (1954) 5.")
        assert fields.year == 1954
        assert fields.first_surname == "Luders"
        assert fields.title is None
        assert not fields.is_thin

    def test_thin_reference_is_flagged(self) -> None:
        assert extract_fields("ibid.").is_thin

    def test_last_year_wins(self) -> None:
        """An earlier year usually belongs to a series or edition note."""
        assert extract_fields("Springer Series 1984, 3rd ed. (2011)").year == 2011


class TestCorroboration:
    FIELDS = extract_fields("C. Abel et al. Phys. Rev. Lett. 124, 081803 (2020)")

    def test_full_agreement_resolves(self) -> None:
        candidate = Candidate(
            title="Measurement of the permanent electric dipole moment of the neutron",
            year=2020, authors=["C. Abel", "S. Afach"], volume="124", page="081803",
        )
        assert corroborate(self.FIELDS, candidate).score >= MATCH_THRESHOLD

    def test_year_alone_is_not_enough(self) -> None:
        """Thousands of papers share a year; one weak signal must not match."""
        result = corroborate(self.FIELDS, Candidate(year=2020, authors=["Z. Nobody"]))
        assert result.score < MATCH_THRESHOLD

    def test_wrong_year_is_penalised(self) -> None:
        right = corroborate(self.FIELDS, Candidate(year=2020, authors=["C. Abel"]))
        wrong = corroborate(self.FIELDS, Candidate(year=1998, authors=["C. Abel"]))
        assert wrong.score < right.score

    def test_adjacent_year_is_tolerated(self) -> None:
        """Preprint and journal years routinely differ by one."""
        result = corroborate(self.FIELDS, Candidate(year=2019, authors=["C. Abel"]))
        assert "year within 1" in result.agreements

    def test_no_shared_author_is_penalised(self) -> None:
        result = corroborate(self.FIELDS, Candidate(year=2020, authors=["Q. Different"]))
        assert "no shared author" in result.disagreements

    def test_volume_and_page_compound(self) -> None:
        base = Candidate(year=2020, authors=["C. Abel"])
        with_locator = Candidate(year=2020, authors=["C. Abel"], volume="124", page="081803")
        assert corroborate(self.FIELDS, with_locator).score > corroborate(self.FIELDS, base).score

    def test_explanation_names_the_agreements(self) -> None:
        result = corroborate(self.FIELDS, Candidate(year=2020, authors=["C. Abel"], volume="124"))
        explanation = result.explain()
        assert "year" in explanation and "author" in explanation

    def test_best_candidate_picks_the_strongest(self) -> None:
        candidates = [
            Candidate(year=1999, authors=["X. Other"]),
            Candidate(year=2020, authors=["C. Abel"], volume="124", page="081803"),
        ]
        result, chosen = best_candidate(self.FIELDS, candidates)
        assert chosen is not None and chosen.year == 2020
        assert result.score >= MATCH_THRESHOLD

    def test_no_candidates_yields_nothing(self) -> None:
        result, chosen = best_candidate(self.FIELDS, [])
        assert chosen is None and result.score == 0.0


class TestAdapters:
    def test_crossref(self) -> None:
        candidate = from_crossref(
            {
                "title": ["A Paper"],
                "author": [{"given": "C.", "family": "Abel"}],
                "issued": {"date-parts": [[2020, 1, 31]]},
                "volume": "124", "page": "081803-081810", "DOI": "10.1103/x",
            }
        )
        assert candidate.title == "A Paper"
        assert candidate.year == 2020
        assert candidate.authors == ["C. Abel"]
        assert candidate.page == "081803"

    def test_openalex(self) -> None:
        candidate = from_openalex(
            {
                "title": "A Paper", "publication_year": 2020,
                "authorships": [{"author": {"display_name": "C. Abel"}}],
                "biblio": {"volume": "124", "first_page": "081803"},
                "doi": "https://doi.org/10.1103/x",
            }
        )
        assert candidate.year == 2020
        assert candidate.doi == "10.1103/x"
        assert candidate.volume == "124"

    def test_inspire(self) -> None:
        candidate = from_inspire(
            {
                "metadata": {
                    "titles": [{"title": "A Paper"}],
                    "authors": [{"full_name": "Abel, C."}],
                    "publication_info": [{"year": 2020, "journal_volume": "124", "page_start": "081803"}],
                    "dois": [{"value": "10.1103/x"}],
                }
            }
        )
        assert candidate.title == "A Paper"
        assert candidate.year == 2020
        assert "Abel" in candidate.authors[0]


def test_title_similarity() -> None:
    assert title_similarity("The Same Title", "the same title!") == 1.0
    assert title_similarity("Quantum optics", "Sediment transport") < 0.5
    assert title_similarity("", "anything") == 0.0
