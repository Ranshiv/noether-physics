# research/

Measured numbers and the artefacts behind them.

## Size policy

Only small tabular artefacts are tracked in git: labels, benchmark specs, metrics, review history.
Target under 50 MB each.

**Bulk artefacts stay outside the repo** — paper sources, figures, embeddings, model checkpoints —
under the data root (`$NOETHER_ROOT`, default OS data dir), referenced by manifest and checksum only.
`.gitignore` enforces this with negation rules; if you find yourself fighting it, the file probably
belongs in the data root.

## Layout

- `sources/source_registry.yaml` — every external API, its terms, rate limit and verification date.
  A source not in this file must not be called.
- `benchmarks/` — benchmark specs: the questions, the known-correct answers, the gate thresholds.
- `labels/` — externally-sourced ground truth, with the provenance of each label.
- `results/` — measured metrics with bootstrap CIs, each bound to a code revision and corpus hash.

An empty `results/` means the gates are unmeasured. That is the current state and it is reported as
such, never as passing.
