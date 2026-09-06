---
name: noether
description: Explain physics and quantum-physics concepts with everyday analogies, checked equations, recomputed worked examples and live simulations; and answer questions from real papers with every claim anchored to a verbatim span and every citation resolved. Use for explaining a concept, running a physics simulation, literature questions, reading a specific arXiv paper's equations, auditing a bibliography for hallucinated references, or checking whether an equation is dimensionally consistent.
---

# noether

A verifiable research-paper assistant for physics. It reads arXiv **LaTeX source**, so equations
keep the numbers their authors gave them and citations keep their keys.

## The division of labour

**You reason and write. NOETHER retrieves, checks and records.** It has no `write_answer` tool by
design: the component that can hallucinate is not the component that decides what ships.

## Explaining a concept (start here for "what is X?")

1. `explain_concept("decoherence")` — returns a curated definition, an everyday analogy,
   dimension-checked equations, a worked example **recomputed** against its stored answer, a live
   simulation, and common misconceptions.
2. Present that material. It is checked; you do not need to re-derive it.
3. `list_concepts` shows what is covered. For anything outside it, fall back to the paper workflow
   below and say the explanation is not from the curated library.

Three honesty rules:

- If `all_checks_passed` is false, or a worked example has `recomputation_matches: false`, **say so**.
  A failed check means the library entry is wrong, not that it should be smoothed over.
- `corpus_papers` empty means nothing on this topic has been ingested. Do not imply the explanation
  came from the literature — offer `further_reading_query` instead.
- Repeat the `misconceptions` list. It is the part that changes what a reader believes.

## Running a simulation

`run_simulation("rabi", {"detuning": 6.28})` or via a concept's linked model. Every run reports:

- a **convergence check** from halving the step size — if `converged` is false, do not quote the
  numbers as settled;
- `caveats`, what the model genuinely does not capture (rotating-wave approximation, Markovian bath,
  Fock truncation). Pass these on;
- `derived` scalars — periods, rates, and where a closed form exists, the error against it.

Out-of-range parameters are refused before anything runs, with a reason. Relay the reason.

## Answering a question

1. `ingest_paper` the relevant papers if they are not in the corpus (`corpus_status` shows what is).
2. `gather_evidence(question)` — returns passages, each with an `evidence_id`.
3. Write claims that cite those ids.
4. `submit_answer(question, claims)` — the gate re-reads the corpus and judges each claim.

**Never quote a paper from memory.** A quote that is not literally present is rejected. So is a
paraphrase stretched into a stronger claim than the passage supports. Cite the `evidence_id` and let
the gate confirm the span.

If the evidence does not answer the question, say so. A thin answer is a real result; padding it
with unverifiable filler is the failure this tool exists to prevent.

## Equations

- `list_equations(paper_id)` — "equation 14" means the one the **authors** numbered 14.
- `check_equation(latex, units)` — parses and dimension-checks.
- `run_equation(latex, values, units)` — evaluates numerically.

Declare units. Most physics symbols are context-dependent: `h` is Planck's constant *or* a height,
`T` a period *or* a temperature, `k` a wavenumber *or* Boltzmann's constant. Undeclared symbols
return `unknown`, which is **not** a pass — it means the check did not run.

Parsing may report `notes`. A note means information was lost (a derivative carried as an opaque
symbol, an inequality not modelled). Repeat notes to the user; do not present a lossy parse as clean.

## Auditing a bibliography

`audit_references(paper_id)` resolves every reference and reports:

- `unresolvable` — no provider has such a record. This is the hallucinated-citation case, and arXiv
  now bans authors over it.
- `mismatch` — resolves, but to something with a different title (right paper, wrong edition or DOI).
- `no-identifier` — matched only by text similarity, so treat the match as provisional.
- `cited-unread` — in the bibliography but never cited in the text.

## Honesty rules

- Report what the gate dropped, not only what it accepted.
- `unknown` is never reported as `consistent`.
- Never claim a benchmark passes without `bench status` showing a measured result.
- Figures come from real papers or from computed plots. Never generate an image of a diagram.
