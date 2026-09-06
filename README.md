# NOETHER

<!-- mcp-name: io.github.ranshiv/noether-physics -->

**A physics research assistant that shows its work.**

Explains physics concepts with everyday analogies, dimension-checked equations, worked examples it
recomputes, and simulations it actually runs. Answers questions from real arXiv papers with every
sentence anchored to a verbatim span. Audits your own draft's bibliography for references that don't
exist.

```bash
pip install noether-physics
noether explain decoherence
```

---

## Read this before you trust it

This is a research-integrity tool, so its limits matter more than its features.

**The verification gate checks that a quote is real. It does not check that the quote supports the
claim.** A claim can contradict its own citation and still pass. Demonstrated during development: a
claim asserting *"the experiment detected a nonzero neutron EDM at five sigma"* was accepted while
citing a passage reporting **d_n = 0.0 ± 1.1 × 10⁻²⁶ e·cm** — a null result. The quote was genuine,
so the gate let it through. Closing that gap needs entailment checking, which needs a language
model, and keeping the model outside the verifier is what makes the other guarantees meaningful. So
every answer this tool emits ends by saying so, and **you should read the quotes.**

**The reference resolver's false-positive rate is unmeasured.** It finds 86.5% of real references.
Nobody knows how often it would confidently match a *fabricated* one. For a tool whose first
promise concerns hallucinated citations, that is the most important number that does not yet exist.

**It is small.** 58 papers in the test corpus, 8 concepts in the explanation library, quantum
coverage limited to two-level systems and one cavity mode. No PDF fallback, so roughly 9% of
quant-ph papers (measured: 5 of 55) cannot be ingested at all.

[`docs/DEFERRED.md`](docs/DEFERRED.md) is the full account — every gap, with the measurement behind
it. It is the honest document; this README is the summary.

---

## Why it exists

2026 audits found a **17% phantom-citation rate** in AI-assisted survey papers — references
resolving to no digital object — and roughly **146,900 AI-generated fake citations** across 2.5M
papers in 2025. arXiv now bans authors who submit hallucinated references.

Physics has a structural advantage nobody exploits: **arXiv distributes the actual LaTeX source.**
Equations arrive with their `\label`s, citations with their `\cite` keys, sections with their real
structure — read, not guessed. Tools in other fields OCR PDFs because their domains have no source.
Physics does.

## What it does

**Explains.** Curated prose, not generated — then machine-checked. Every equation is parsed and
dimension-checked; every worked example is *recomputed* and compared against its stored answer, so a
typo in the library fails CI rather than teaching you a wrong number.

```bash
noether explain decoherence          # analogy, maths, worked example, live simulation
noether concepts                     # what the library covers
```

**Simulates.** Six models — pendulum, damped/driven oscillator, projectile with drag (SciPy); Rabi,
decoherence, Jaynes-Cummings (QuTiP). Every run halves the step size and reports the drift, and
where a closed form exists the numerics are compared against it automatically.

```bash
noether simulate rabi --params "detuning=6.28" --bloch
noether models
```

**Checks equations.** Undeclared symbols return `unknown`, never a guess — `h` is Planck's constant
*or* a height, `T` a period *or* a temperature. `unknown` is not a pass.

```bash
noether eq 'T = 2 \pi \sqrt{\frac{L}{g}}' --units 'T=second,L=meter,g=meter/second**2' --vary 'L=0.1..2' --at 'g=9.81'
```

**Reads papers.** Ingests LaTeX source, so "equation 14" means the equation *the authors* numbered 14.

```bash
noether ingest 2001.11966
noether equations arXiv:2001.11966
noether find "vacuum Rabi oscillations"
```

**Audits your draft.** Resolves every `\bibitem` against arXiv, Crossref, OpenAlex and INSPIRE.
Exits non-zero on unresolvable references or cite keys with no bibliography entry, so it works as a
pre-commit hook.

```bash
noether audit path/to/your-paper.tex
```

## Measured results

Run `noether bench status` yourself. As of the last run:

| gate | result | threshold | verdict |
|---|---|---|---|
| equations | 100% (31/31) | 90% | pass |
| retrieval | 83.3% (10/12) **at top-3 of 58 papers** | 80% | pass |
| citations | 86.5% (32/37) | 85% | pass |

Read these with their scope attached. "Right paper in the top 3" over 58 papers is a real task; over
8 it was nearly free, and an earlier run reported a meaningless 100% because `top_k` exceeded the
corpus size. The citation threshold is 85% rather than 100% because books, theses and pre-digital
articles are genuinely not in any index — the residual failures on the nEDM letter are a 1954 Danish
journal, a 1955 Pauli chapter and a 1982 Oxford book.

## Use it from Claude

```bash
pip install "noether-physics[mcp]"
claude mcp add --scope user noether -- python -m noether.mcp_server
```

14 tools. Claude reasons and writes; NOETHER retrieves, checks and records. There is deliberately no
`write_answer` tool — the component that can hallucinate is not the component that decides what
ships.

## Install

```bash
pip install noether-physics              # core
pip install "noether-physics[quantum]"   # + QuTiP simulations
pip install "noether-physics[mcp]"       # + MCP server
```

Python 3.12+. From source: `uv venv && uv pip install -e ".[dev,quantum,mcp]"`, then `uv run pytest`
(285 tests, no network — cassettes default to replay and a miss is an error, not a live fallback).

## Terms

arXiv forbids bulk crawling. This fetches one paper at a time behind a 3-second interval and caches
permanently, so a given paper is downloaded exactly once. Every source, its rate limit and its terms
are declared in `research/sources/source_registry.yaml`; a source not in that file is not called.

**No AI-generated imagery.** Every figure is extracted from a real paper or rendered
deterministically from code and data, and is tagged with which.

## Licence

MIT — see [LICENSE](LICENSE).

The name honours Emmy Noether. The PyPI distribution is `noether-physics` because `noether` belongs
to an unrelated [physical-units library](https://github.com/yunruse/noether); the command is
`noether` either way.
