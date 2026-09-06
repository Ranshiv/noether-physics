"""Grow the corpus to a size where the benchmarks mean something.

Eight papers is a demonstration. At that size "the right paper is in the top 3"
is nearly guaranteed, so the retrieval number flatters the retriever. This
script ingests a few dozen quant-ph papers across the topics the concept library
covers, which makes the same benchmark genuinely hard.

It reports every failure rather than skipping quietly: a PDF-only submission has
no LaTeX source and cannot be ingested at all, and knowing how often that
happens is itself a measurement worth having.

Run:  uv run python scripts/seed_corpus.py [target_count]
"""

from __future__ import annotations

import sys
import time
from collections import Counter

from noether.library import Library

#: Queries chosen to cover the concept library's topics plus the surrounding
#: literature, so retrieval is tested on a corpus with real near-neighbours
#: rather than on eight papers about unrelated things.
QUERIES = [
    'cat:quant-ph AND abs:"decoherence"',
    'cat:quant-ph AND abs:"dephasing"',
    'cat:quant-ph AND abs:"superconducting qubit"',
    'cat:quant-ph AND abs:"transmon"',
    'cat:quant-ph AND abs:"Jaynes-Cummings"',
    'cat:quant-ph AND abs:"cavity QED"',
    'cat:quant-ph AND abs:"Rabi oscillations"',
    'cat:quant-ph AND abs:"Lindblad"',
    'cat:quant-ph AND abs:"open quantum system"',
    'cat:quant-ph AND abs:"quantum error correction"',
    'cat:quant-ph AND abs:"entanglement"',
    'cat:quant-ph AND abs:"coherence time"',
    'cat:quant-ph AND abs:"master equation"',
    'cat:quant-ph AND abs:"circuit QED"',
]


def main(target: int = 50) -> int:
    library = Library()
    already = set(library.store.paper_ids())
    print(f"corpus starts with {len(already)} papers; target {target}\n", flush=True)

    # Collect candidates first so the ingest loop is one steady stream of
    # rate-limited fetches rather than interleaved searches.
    candidates: list[str] = []
    for query in QUERIES:
        try:
            for paper in library.search_arxiv(query, limit=8):
                if f"arXiv:{paper.arxiv_id}" not in already and paper.arxiv_id not in candidates:
                    candidates.append(paper.arxiv_id)
        except Exception as exc:  # noqa: BLE001 - one bad query must not stop the run
            print(f"  search failed for {query!r}: {type(exc).__name__}", flush=True)
        if len(candidates) >= target * 2:
            break

    print(f"{len(candidates)} candidates found; ingesting up to {target}\n", flush=True)

    outcomes: Counter[str] = Counter()
    started = time.monotonic()
    ingested = 0

    for arxiv_id in candidates:
        if ingested >= target:
            break
        result = library.ingest_arxiv(arxiv_id)
        if result.ok:
            ingested += 1
            outcomes["ok"] += 1
            print(
                f"  [{ingested:3}] {result.paper_id:22} {result.numbered_equations:3} eq  "
                f"{result.references:3} refs  {result.title[:44]}",
                flush=True,
            )
        else:
            reason = "no-latex-source" if "no LaTeX source" in result.error else "other"
            outcomes[reason] += 1
            print(f"        SKIP {arxiv_id:18} {result.error[:60]}", flush=True)

    elapsed = time.monotonic() - started
    stats = library.stats()
    print(
        f"\ndone in {elapsed / 60:.1f} min · {dict(outcomes)}"
        f"\ncorpus now: {stats['papers']} papers, {stats['chunks']} chunks, "
        f"{stats['equations']} equations, {stats['references']} references",
        flush=True,
    )
    library.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 50))
