"""The emission gate -- the single point every answer passes through.

Nothing reaches a user except through :meth:`Gate.check`. That is deliberate:
guarantees are only guarantees if there is exactly one door, and this is it.

The gate is uncompromising by design. A claim whose quote is not literally
present in the cited paper is **dropped**, not softened, not hedged, not
prefixed with "approximately". A citation that does not resolve is **removed**.
The caller receives a report of everything removed and why, so a thin answer is
visibly thin rather than quietly padded with unverifiable filler.

**What this gate does NOT establish.** It verifies that a quote is real and
that a reference resolves. It does *not* verify that the quote supports the
claim. Demonstrated live: a claim reading "the experiment detected a nonzero
neutron EDM at five sigma, confirming CP violation" was accepted while citing a
passage stating d_n = (0.0 +/- 1.1) x 10^-26 e cm -- a null result. The claim
contradicted its own evidence and passed, because the quote existed.

Closing that gap needs entailment checking, which needs a language model, which
is precisely the component this design keeps outside the verifier. So the
limitation is structural rather than an oversight, and every report says so in
:meth:`GateReport.what_was_checked` rather than letting "verified" be read as
more than it is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..index.store import Store
from ..ingest.document import Anchor, Span


@dataclass
class Claim:
    """One sentence an answer wants to make, with the evidence for it."""

    text: str
    anchors: list[Anchor] = field(default_factory=list)
    #: Free-form tag so a caller can trace a claim back to its own pipeline.
    origin: str = ""


@dataclass
class Rejection:
    """Why one claim did not survive."""

    claim: Claim
    reason: str
    detail: str = ""


@dataclass
class GateReport:
    """What the gate let through, and what it did not."""

    accepted: list[Claim] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    #: References dropped for failing to resolve.
    dropped_references: list[tuple[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when nothing was rejected. A thin answer can still pass."""
        return not self.rejected and not self.dropped_references

    @property
    def acceptance_rate(self) -> float:
        total = len(self.accepted) + len(self.rejected)
        return len(self.accepted) / total if total else 0.0

    def what_was_checked(self) -> str:
        """State the scope of the check, so nobody reads more into it.

        Printed alongside every accepted answer. An anchored claim is a claim
        whose evidence exists -- not a claim shown to follow from it.
        """
        return (
            "Checked: every quote is verbatim in the cited paper, every span "
            "resolves, and every reference was resolvable. NOT checked: whether "
            "each quote actually supports the claim made from it. A claim can "
            "contradict its own evidence and still pass. Read the quotes."
        )

    def summary(self) -> str:
        parts = [f"{len(self.accepted)} claims accepted"]
        if self.rejected:
            parts.append(f"{len(self.rejected)} dropped")
        if self.dropped_references:
            parts.append(f"{len(self.dropped_references)} references removed")
        return ", ".join(parts)


class Gate:
    """Validates claims against the corpus before anything is emitted."""

    #: An anchor quoting fewer characters than this is not evidence of anything;
    #: any short string can be found somewhere in a long paper by chance.
    MIN_QUOTE_CHARS = 24

    def __init__(self, store: Store) -> None:
        self.store = store
        self._texts: dict[str, str] = {}

    def _text(self, paper_id: str) -> str | None:
        if paper_id not in self._texts:
            text = self.store.paper_text(paper_id)
            if text is None:
                return None
            self._texts[paper_id] = text
        return self._texts[paper_id]

    # ---- anchor checking -----------------------------------------------

    def check_anchor(self, anchor: Anchor) -> tuple[bool, str]:
        """Verify one anchor against the stored paper. Returns (ok, reason)."""
        text = self._text(anchor.paper_id)
        if text is None:
            return False, f"paper {anchor.paper_id} is not in the corpus"

        if len(anchor.quote.strip()) < self.MIN_QUOTE_CHARS:
            return False, (
                f"quote is {len(anchor.quote.strip())} chars; "
                f"under {self.MIN_QUOTE_CHARS} it is not evidence"
            )

        if anchor.span.end > len(text):
            return False, (
                f"span [{anchor.span.start}, {anchor.span.end}) runs past the "
                f"paper ({len(text)} chars)"
            )

        actual = text[anchor.span.start : anchor.span.end]
        if actual != anchor.quote:
            # Distinguish a fabricated quote from a stale one: if the text
            # appears elsewhere the anchor drifted; if not, it was invented.
            if anchor.quote in text:
                return False, "span offsets are stale: the quote sits elsewhere in this paper"
            return False, "quote does not appear in this paper"

        return True, ""

    def locate(self, paper_id: str, quote: str) -> Anchor | None:
        """Build a verified anchor by finding ``quote`` in a paper.

        The honest way to construct an anchor when only the text is known: the
        offsets come from the corpus, never from a caller's assertion.
        """
        text = self._text(paper_id)
        if text is None or len(quote.strip()) < self.MIN_QUOTE_CHARS:
            return None
        index = text.find(quote)
        if index == -1:
            return None
        return Anchor(paper_id, "", Span(index, index + len(quote)), quote)

    # ---- the gate ------------------------------------------------------

    def check(
        self,
        claims: list[Claim],
        references: dict[str, bool] | None = None,
        require_anchor: bool = True,
    ) -> GateReport:
        """Filter claims down to those the corpus actually supports.

        ``references`` maps a citation key to whether it resolved. Unresolved
        keys are reported and removed, enforcing guarantee 1.
        """
        report = GateReport()

        for claim in claims:
            if require_anchor and not claim.anchors:
                report.rejected.append(Rejection(claim, "unanchored", "no supporting span"))
                continue

            failures = []
            verified = []
            for anchor in claim.anchors:
                ok, reason = self.check_anchor(anchor)
                (verified if ok else failures).append((anchor, reason))

            if require_anchor and not verified:
                detail = "; ".join(f"{a.paper_id}: {r}" for a, r in failures) or "no valid anchor"
                report.rejected.append(Rejection(claim, "anchor-failed", detail))
                continue

            # Keep only the anchors that checked out; a claim rides on evidence
            # that survived, never on evidence that did not.
            report.accepted.append(
                Claim(text=claim.text, anchors=[a for a, _ in verified], origin=claim.origin)
            )

        for key, resolved in (references or {}).items():
            if not resolved:
                report.dropped_references.append((key, "reference could not be resolved"))

        return report


def sentences_without_anchors(claims: list[Claim]) -> list[str]:
    """Convenience for reporting: which claim texts carry no evidence at all."""
    return [c.text for c in claims if not c.anchors]


def build_claims(
    pairs: list[tuple[str, str, str]],
    locate: Callable[[str, str], Anchor | None],
) -> list[Claim]:
    """Build claims from ``(text, paper_id, quote)`` triples.

    An entry whose quote cannot be located yields an *unanchored* claim rather
    than being silently discarded here -- the gate is what decides its fate, so
    the rejection appears in the report.
    """
    claims: list[Claim] = []
    for text, paper_id, quote in pairs:
        anchor = locate(paper_id, quote)
        claims.append(Claim(text=text, anchors=[anchor] if anchor else []))
    return claims
