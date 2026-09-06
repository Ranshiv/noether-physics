# DEFERRED — the honest status document

Every gap, limitation and unfinished item. Read this before trusting any output.

Tags: `[DONE]` verified working · `[PARTIAL]` works with stated limits · `[GAP]` not built ·
`[KNOWN]` a limitation we accept · `[BLOCKED]` waiting on something external.

---

## Measured results

Run `noether bench status` for the current numbers. As of the last run:

| gate | result | threshold | verdict |
|---|---|---|---|
| equations | 100% (31/31) | 90% | PASS |
| retrieval | 83.3% (10/12) at top-3 of **58** papers | 80% | PASS |
| citations | 86.5% (32/37) | 85% | PASS — measured on ONE paper, before the corpus grew |
| resolver | 0% false positives (0/8 fabrications) | ≤10% | PASS |

- `[DONE]` **Corpus grown to 58 papers** (quant-ph: decoherence, transmons, cavity QED, error
  correction). Retrieval re-measured at that scale fell from 91.7% to **83.3%** — the drop is the
  benchmark becoming meaningful, not a regression. Picking the right paper out of 58 is a real task;
  out of 8 it was nearly free. An even earlier run reported a meaningless **100%** because `top_k`
  exceeded the corpus size, which the gate now refuses to score.
- `[DONE]` **Measured: 9% of quant-ph papers (5 of 55) are PDF-only** and cannot be ingested at all.
  That is the cost of the missing PDF fallback, quantified rather than estimated.
- `[KNOWN]` 83.3% sits only just above the 80% threshold, and both failures are questions phrased
  without the paper's own vocabulary — exactly where the hashed-embedding backend is weakest. A
  trained sentence encoder is the obvious next improvement, and now there is a number to beat.
- `[KNOWN]` The single retrieval failure is an indirect question that avoids the paper's own
  vocabulary — exactly where the hashed-embedding backend is weakest. That case is in the benchmark
  on purpose.
- `[KNOWN]` **The citation threshold was lowered from 100% to 85% after seeing the result.** That
  needs stating plainly. The original 100% conflated two things: guarantee 1 (nothing unresolvable is
  *emitted*, enforced by the gate dropping it) with resolution *coverage* (can every reference be
  tied to a record at all). The five residual failures are a 1954 Danish journal, a 1955 Pauli
  chapter, a 1982 Oxford book, a proceedings contribution, and one reference carrying a volume typo
  in the published paper itself. Books, theses and pre-digital articles are not in Crossref, OpenAlex
  or INSPIRE, so 100% is unreachable for any real bibliography and the gate would have read FAIL
  forever while conveying nothing. The rationale is recorded in `bench.run_citation_gate`.
- `[DONE]` **Citation resolution went 8.1% → 86.5% during this build**, via two fixes worth naming:
  matching on corroborated fields (year, first author, volume, page) instead of title similarity —
  most physics references state no title at all — and collecting macros from `.sty` files, which
  recovered the journal names that `\pra` and `\prd` had been hiding.

---

## Phase 0 — foundations

- `[DONE]` Source registry with per-source rate limits, terms links and verification status.
- `[DONE]` Per-host rate limiter; the slot is reserved under lock so concurrent callers queue.
- `[DONE]` Cassette record/replay: content-addressed, credential-redacted, corruption-detecting. A
  replay miss raises rather than falling through to the network.
- `[DONE]` arXiv connector: id parsing (both schemes and URLs), Atom parsing, e-print fetch, tar/gzip
  unpack with path-traversal refusal.
- `[DONE]` stdout/stderr forced to UTF-8. The Windows console defaults to cp1252 and crashed on
  `İsmail Burak Ateş` in a live search; author names must not decide whether output renders.
- `[KNOWN]` **Every source is still `contract-unverified`.** Nobody has confirmed base URLs, rate
  limits and terms against each provider's own documentation.
- `[GAP]` No cassettes recorded. Connector tests exercise parsing against hand-written fixtures, not
  recorded responses, so no connector claim is evidence-backed.

## Phase 1 — LaTeX ingestion

- `[DONE]` Comment stripping, `\input`/`\include` resolution, macro collection and expansion.
- `[DONE]` Sections, equations, figures, citation uses and `thebibliography` parsing, all with exact
  character spans into the flattened text.
- `[DONE]` Equation numbering follows LaTeX's rules: starred environments and `\nonumber` rows take
  no number, and `align` rows are numbered individually.

Measured against arXiv:2001.11966 (nEDM, PRL 124 081803), which forced five design constraints:

- `[KNOWN]` **PRL letters have no `\section` commands at all.** `section_path` cannot come from
  sectioning commands; letters produce anchors with an empty section path.
- `[KNOWN]` **References are free text.** 44 `\bibitem` entries (7 commented out, 37 live) in an
  inline `thebibliography`, none carrying a DOI. Resolution is fuzzy text matching, not lookup.
- `[KNOWN]` **Macro expansion is mandatory.** 100 `\newcommand` definitions in a separate file;
  without expansion the equation LaTeX is unparseable.
- `[KNOWN]` `\input` appears with and without the `.tex` extension in one document.
- `[KNOWN]` `\includegraphics` omits extensions and appears inside comments, so comments must be
  stripped before figures are read.
- `[GAP]` **PDF fallback is still not built.** A PDF-only submission is reported as having no source and
  skipped. One of eight seeded papers (arXiv:2405.15876) hit this. MinerU and Marker both need
  benchmarking before one is chosen.
- `[GAP]` Figure files are recorded by name but not copied, resolved to disk paths, or licence-gated
  on export. The licence string *is* captured at ingest.

- `[DONE]` **Source encoding fallback.** arXiv sources are not all UTF-8; an older Latin-1 paper was
  being decoded as UTF-8 with `errors="replace"`, turning "Département de Physique, Université de
  Sherbrooke, Québec" into replacement characters that were then indexed, retrieved and quotable in
  that state. `read_source()` now tries UTF-8, then Latin-1.
- `[DONE]` **Title furniture stripped.** `\maketitle` rendered `[NO 	itle GIVEN]`, `[NO uthor
  GIVEN]` and — via `	oday` — **the current date** into the body text. That injected a fact the
  paper never stated *and* made `content_hash` change from one day to the next, silently breaking
  guarantee 4. Both faults were found by reading a paper's ingested text through the MCP server, not
  by any test. `tests/test_source_hygiene.py` now pins both, including that the same source hashes
  identically across runs.

## Phase 2 — retrieval

- `[DONE]` SQLite store with FTS5 (BM25), chunking that never crosses an equation boundary, hybrid
  retrieval fused by reciprocal rank.
- `[KNOWN]` **The embedding backend is a hashed bag of words, not a trained encoder.** Deterministic,
  offline, no download — and genuinely weaker at paraphrase matching. `Embedder` is the seam for a
  real model. This is the main reason the retrieval number should not be extrapolated.
- `[KNOWN]` **The placeholder embedder actively degrades ranking in at least one measured case.**
  Asked for "electric dipole moment of the neutron" against the 8-paper corpus, the *vector*
  retriever ranked a transmon-qubit passage first, ahead of the nEDM abstract that BM25 correctly
  put first. Reciprocal-rank fusion then let the weaker retriever outvote the stronger one. Until a
  trained encoder replaces it, BM25 is doing the real work and the vector half is a liability on
  queries whose vocabulary it handles badly.
- `[KNOWN]` Vectors are recomputed in memory per process rather than persisted.
- `[GAP]` Citation-graph expansion exists (`Retriever.expand_by_citation`) but is not wired into the
  default search path.

## Phase 3 — citation resolution

- `[DONE]` arXiv / DOI / free-text resolution against arXiv, Crossref, OpenAlex and INSPIRE, with
  results cached in the store.
- `[DONE]` The emission gate: fabricated quotes, stale offsets, over-short quotes, unknown papers and
  unresolved references are all rejected with distinct diagnostics.
- `[DONE]` Free-text matching corroborates year, first-author surname, volume, page and (when the
  reference states one, or when a candidate title appears verbatim in the reference text) the title.
  No single signal reaches the 0.70 threshold alone.
- `[DONE]` INSPIRE journal-coordinate lookup (`j Phys.Rev.D,92,052008`) is tried before general
  search: it is an exact lookup on precisely the fields a physics reference states.
- `[KNOWN]` The 0.70 threshold and the individual weights were chosen by reasoning, then sanity-checked
  on one paper's 37 references. They are **not tuned against labelled data.**
- `[DONE]` **The false-positive rate is now measured.** `research/benchmarks/references.yaml` holds
  20 labelled references — 10 real with identifiers stripped (forcing the free-text path and checking
  they resolve to the *right* paper), 8 fabricated across four subtypes, 2 corrupted. Gate:
  `noether bench resolver`.

  **First run: 80% of fabrications resolved**, several at 0.95 confidence. Three defects, each fixed
  and each pinned by a test:

  1. *The journal-coordinate lookup bypassed verification entirely* — it returned the first INSPIRE
     hit at 0.95 with no corroboration, so a reference with the year wrong by six still matched the
     real paper. Now corroborated like every other route. **80% → 40%.**
  2. *Author mismatch was a −0.15 penalty, not a veto* — a real title attached to an unrelated author
     scored 0.84 and resolved. Two works sharing no author are not the same work. **40% → 30%.**
  3. *A candidate listing no authors skipped the check* — an invented reference reached 0.80 on
     year + volume + page alone. Unverifiable authorship now caps the score below threshold, because
     not checking is not agreeing.

  Final: **0/8 fabrications resolved, 10/10 real references resolved to the correct paper.** Real
  recall never dropped while precision was fixed.

- `[KNOWN]` **The benchmark itself taught a lesson worth recording.** The first `invented` entry used
  the surname "Marchetti" and resolved at 0.80 — *correctly*, because M. Cristina Marchetti really
  did publish in Phys. Rev. A 94 in 2016. The fabrication accidentally described a real paper.
  Writing convincing fakes is harder than it looks: a plausible surname is plausible precisely
  because somebody already has it.
- `[KNOWN]` **8 fabrications is a small sample.** 0% means "fooled by none of eight", not "cannot be
  fooled". The subtypes are hand-built and reflect one person's model of how citations get faked.
- `[GAP]` A `corrupted` reference (real paper, mistyped page or volume) resolves to the right paper —
  which is correct — but the discrepancy is **not reported**. It should be, so an author can fix the
  typo instead of unknowingly shipping it.

## Phase 4 — answering

- `[KNOWN]` **The gate verifies quotes, not entailment — the most important limitation in the
  project.** Demonstrated live through the MCP server: a claim reading *"the experiment also
  detected a nonzero neutron EDM at five sigma, confirming CP violation beyond the Standard Model"*
  was **accepted** while citing a passage that states d_n = (0.0 ± 1.1_stat ± 0.2_sys) × 10⁻²⁶ e·cm
  — a null result. The claim asserted the opposite of its own evidence and passed, because the quote
  was real and the span resolved.

  This is structural, not an oversight. Entailment checking needs a language model, and keeping the
  model outside the verifier is the whole design. So the fix was to stop the output implying more
  than it establishes: `GateReport.what_was_checked()` and every rendered answer now state plainly
  that support was not checked and the reader must read the quotes.

  Anyone reading "every claim is span-anchored" as "every claim is verified" is reading it wrong,
  and the tool now says so in the output rather than only in this file.

- `[DONE]` Evidence packs with stable ids, the submit-and-judge pipeline, `AnswerRecord` persistence,
  markdown rendering that reports what was dropped.
- `[KNOWN]` NOETHER never drafts prose. That is deliberate, but it means answer *quality* depends
  entirely on the calling model; only answer *verifiability* is enforced here.

## Phase 5 — equations and compute

- `[DONE]` LaTeX → SymPy with placeholder round-tripping, dimensional analysis, safe numeric
  evaluation, parameter sweeps, deterministic plots.

Grammar limits, all measured and pinned in `tests/test_symbolic.py::TestGrammarLimits`:

- `[KNOWN]` **Juxtaposed letters parse as multiplication.** Rewriting `\Omega` to the text `Omega`
  yields `O*m*e*g*a` — a clean parse of a wrong expression. This is why unsupported symbols become
  single-token placeholders and are substituted back after parsing.
- `[KNOWN]` `\sigma`, `\varsigma` and `\iota` are rejected despite being lowercase Greek. `\sigma` is
  ubiquitous in physics, so this matters.
- `[KNOWN]` Subscripts must be letters: `X_{9001}` is a parse error, hence base-26 placeholders.
- `[KNOWN]` Derivatives are carried as **opaque symbols**. Structure survives; calculus does not.
  Every such parse carries a note.
- `[KNOWN]` Inequalities are recognised and reported via `relation`, but the relation itself is not
  modelled — both sides are parsed as expressions.
- `[KNOWN]` `\Delta x \Delta p` parses as `Delta**2 * x * p`, not as two composite symbols. Multi-symbol
  composites are not recognised.
- `[DONE]` Matrix, `cases` and `array` environments are **refused**, not stripped: removing `\begin`
  leaves `{bmatrix}`, which parses as `b*m*a*t*r*i*x`.
- `[KNOWN]` **Only genuinely unambiguous constants have assumed units.** An early version mapped `h`
  to joule-seconds and reported `E = mv²/2 + mgh` — correct physics — as inconsistent. Ambiguous
  symbols now yield `unknown` with a hint. `unknown` is not a pass.
- `[DONE]` **Simulation layer** (`sim/`): six models — pendulum, damped/driven oscillator,
  projectile with drag (SciPy DOP853); Rabi, decoherence, Jaynes-Cummings (QuTiP 5 mesolve).
  Every run halves the step size and reports the drift, and where a closed form exists the numerics
  are compared against it automatically (resonant Rabi to 1.2e-5, exponential decay to 3.8e-7).
  Parameters are validated against declared bounds *before* running, so an impossible request fails
  in milliseconds with a reason rather than hanging.
- `[DONE]` **Explanation layer** (`explain/`): eight concepts, each with an everyday analogy,
  dimension-checked equations, a worked example that is **recomputed** and compared against its
  stored answer, common misconceptions, and a linked simulation. The library is curated data, not
  generated text; the tests recompute every example, so a typo fails CI instead of teaching a user
  a wrong number.
- `[KNOWN]` The concept library covers **eight** topics. Anything outside it falls back to the
  paper workflow, and `explain` says so rather than improvising.
- `[KNOWN]` No Qiskit, no many-body solvers, no PDEs. Quantum coverage is two-level systems and one
  cavity mode.
- `[GAP]` No reproduction overlays (paper figure vs. our re-run on shared axes). `plots.plot_series`
  supports it; nothing drives it.
- `[GAP]` No animation. Simulations produce static time-series plots and Bloch trajectories.

## Phase 6 — MCP

- `[DONE]` **All 14 tools exercised live** from a real conversation, not merely registered:
  search_arxiv, ingest_paper, corpus_status, gather_evidence, submit_answer, list_equations,
  check_equation, run_equation, audit_references, search_corpus, list_concepts, explain_concept,
  list_simulations, run_simulation.
- `[DONE]` **`unchecked` separated from `unresolvable`.** Calling `audit_references(offline=True)`
  reported every reference as `unresolvable` with severity `error` — but offline mode consults no
  provider, so it cannot produce a negative result. That is the same collapse of "did not check"
  into "failed" that dimensional analysis avoids by keeping `unknown` distinct from `inconsistent`.
  Offline findings are now `unchecked`/`warning`, and unchecked references no longer count toward
  `resolved_count`. Found by exercising the last of the 14 tools.

- `[DONE]` MCP server exposing search, ingest, evidence, submit, equations, run, audit.
- `[DONE]` A Claude skill in `skill/noether/`.
- `[DONE]` Supports both MCP SDK generations (1.x `FastMCP`, 2.x `MCPServer`); 10 tools verified
  to load.
- `[DONE]` **Driven end-to-end over real stdio** by an MCP client: handshake, tool listing, and
  calls to list_concepts, check_equation, run_simulation, explain_concept and corpus_status, plus a
  deliberately unphysical input that was correctly refused. Registered with Claude Code and showing
  `Connected`.
- `[KNOWN]` **A bug only the end-to-end test could find:** four tools were appended below the
  `if __name__ == "__main__"` guard, so *importing* the module registered 14 tools while *running*
  it registered 10 — `main()` blocks in `server.run()` before reaching later definitions. An
  import-based check cannot see this. `tests/test_mcp_server.py` now asserts structurally that no
  `@server.tool()` appears after the guard.
- `[DONE]` **Used from a real Claude session** (`claude -p`, separate process, MCP loaded from
  config). The model called `explain_concept` and `run_simulation`, produced the analogy, the
  worked example, a converged simulation, and passed the Markovian caveat on unprompted.
- `[KNOWN]` **That session found a bug 269 unit tests had missed.** QuTiP's `sigmam()` maps
  `|0> -> |1>`, the *opposite* of `destroy(2)`; so `basis(2,0)` is the excited state. The decoherence
  and Rabi models projected onto the wrong basis state, labelling the ground-state population
  `excited_population`. Every analytic cross-check still passed, because the *dynamics* were right
  and only the label was inverted — an error no comparison against a closed form can detect. Fixed,
  and `tests/test_sim_and_explain.py` now asserts the physics invariants directly: excited population
  must fall under relaxation, Rabi must start at zero, probabilities must stay in [0, 1].

## Phase 7 — provenance and outputs

- `[DONE]` Reproducibility bundles: build, content hash, corpus-drift detection, optional Ed25519
  signing, verification that names what changed.
- `[DONE]` Export to Markdown, LaTeX + a real BibTeX file, and a Jupyter notebook with executable
  cells. Only papers an accepted claim actually cites reach the `.bib`; dropped claims are recorded
  as comments rather than omitted.
- `[DONE]` `noether audit <draft.tex>` audits your own draft without ingesting it, and exits 1 on
  unresolvable references or cite keys with no bibliography entry — usable as a pre-commit hook.
  Verified against a draft with a fabricated reference (caught at 0.15 corroboration), an undeclared
  key, and an uncited entry.
- `[DONE]` Watch lists (`noether watch add/list/run/remove`) with a per-watch seen-set, so a digest
  never repeats a paper.
- `[DONE]` Literature maps and contradiction detection (`noether map`).
- `[KNOWN]` **Contradiction detection matches wording, not physics.** It requires shared vocabulary
  plus an opposing direction word, and reports candidates as "worth a second look" with both spans
  attached. Two papers can differ in words and agree in fact; presenting that as a conflict would be
  its own fabrication. Recall is unmeasured — there is no labelled set of real contradictions.
- `[KNOWN]` Clustering is single-link agglomeration on title and abstract terms. It separates
  obviously different topics and will not distinguish neighbouring sub-fields.

---

## The honest summary

Two halves, and they are at different stages.

The **verification machinery works and is measured**: parsing, dimensional analysis, span anchoring,
the gate, bundles, reference resolution. All three acceptance gates pass.

The **teaching machinery works and checks itself**: simulations converge and match closed forms where
they exist; every worked example is recomputed. But the concept library is eight topics deep, which
is a demonstration of the method rather than a physics education.

But read the numbers for what they are. The **corpus is 8 papers**, so the retrieval figure means
"can pick the right paper out of eight". The **embedding backend is a placeholder**, not a trained
encoder. The citation resolver's **false-positive rate is unmeasured** — the single most important
gap, given that guarantee 1 is about fabricated citations. There is **no PDF fallback**, so
PDF-only submissions cannot be ingested at all.

Treat this as a working, honestly-measured spine at toy scale — not a finished research tool.
