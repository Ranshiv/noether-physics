# Changelog

## 0.1.2

**Upgrade if you are on 0.1.0 or 0.1.1.** Those releases confirm fabricated
references, which is the specific failure this tool exists to prevent.

### Fixed — reference resolver accepted fabricated citations

A labelled benchmark (`research/benchmarks/references.yaml`, new in this
release) measured the resolver against references whose truth was known in
advance. The first run resolved **80% of deliberately fabricated references**,
several at 0.95 confidence — including one whose publication year was wrong by
six years, which still matched the real paper.

Three defects, each now covered by a test:

1. **The journal-coordinate lookup bypassed verification.** It returned the
   first INSPIRE hit at 0.95 confidence without checking whether that paper's
   year or authors matched the reference at all. A lookup that finds something
   is not a lookup that found the right thing. It now goes through the same
   corroboration as every other route.
2. **Author mismatch was a penalty, not a veto.** A −0.15 deduction let a
   matching title carry a reference whose named author appeared nowhere in the
   candidate: a real title attached to an unrelated author scored 0.84 and
   resolved. Two works that share no author are not the same work.
3. **A candidate listing no authors skipped the check entirely**, letting an
   entirely invented reference reach 0.80 on year, volume and page alone.
   Unverifiable authorship now caps the score below the acceptance threshold —
   not checking is not the same as agreeing.

After the fixes: **0 of 8 fabrications resolved, 10 of 10 real references
resolved to the correct paper.** Recall did not drop while precision was fixed.

### Added

- `noether bench resolver` — a fourth acceptance gate measuring the
  false-positive rate, broken down by fabrication subtype.
- `research/benchmarks/references.yaml` — 20 labelled references: 10 real with
  identifiers stripped (forcing the free-text path, and requiring resolution to
  the *correct* paper), 8 fabricated across four subtypes, 2 corrupted.

### Known limits of the new measurement

Eight fabrications is a small sample. **0% means "fooled by none of eight", not
"cannot be fooled."** The subtypes are hand-built and reflect one person's model
of how citations get faked.

Two notes from building the benchmark, both recorded in `docs/DEFERRED.md`:

- An early "invented" entry used the surname *Marchetti* and resolved at 0.80 —
  correctly. M. Cristina Marchetti really did publish in Phys. Rev. A 94 in
  2016, so the fabrication accidentally described a real paper. Writing
  convincing fakes is harder than it looks.
- A *corrupted* reference (real paper, mistyped page) resolves to the right
  paper, which is correct behaviour. But the discrepancy is not reported, so an
  author never learns about the typo. That remains open.

---

## 0.1.1

Corrected the MCP registry namespace to `io.github.Ranshiv/noether-physics`.
The registry namespace is case-sensitive and matches the GitHub username
exactly; the ownership marker is read from the published PyPI description, so
fixing the case required a new release rather than an edit.

## 0.1.0

Withdrawn — the ownership marker used the wrong namespace case. Use 0.1.2.
