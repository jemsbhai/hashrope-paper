# Hashrope (IEEE ICTAI submission)

Working title: *Sub-Linear Context Routing and Dynamic State Management for
Agentic LLMs via Hash-Augmented Ropes* (title under revision).

**Target venue:** IEEE ICTAI. **Goal:** best-paper-quality submission — every
claim traceable to a logged, reproducible experiment; a runnable artifact;
honest baselines and uncertainty throughout.

This paper builds on the `hashrope` library (PyPI + crates.io), a BB[2/7]
weight-balanced rope with polynomial-hash metadata over Mersenne primes,
RepeatNode O(log q) repetition encoding, and a sliding-window rolling hash.
Library source lives in the sibling repo `hashrope/` (monorepo:
`packages/python`, `packages/rust`). This repo contains only the paper, its
experiments, and its figures — not the library itself.

## Repository map

```
CLAIMS.md            # Claims ledger — every paper claim ↔ experiment ↔ data. Read first.
LOGBOOK.md           # Append-only experimental logbook (plan before, results after).
findings.md          # Curated results summary + chronological raw-findings log.
README.md            # This file.
configs/             # Per-experiment configuration (frozen copies live in experiments/).
data/
  raw/               # Original inputs, never overwritten (real agent traces, genome data, …).
  processed/         # Derived/cleaned data.
  DATA_README.md     # Provenance for every dataset.
src/                 # Experiment + analysis code (chunkers, harnesses, stats, plotting).
scripts/             # Entry-point runners and figure generators.
experiments/         # Per-EXP outputs: frozen config, env snapshot, seeds, logs, results, figures.
figures/             # Publication-ready figures (regenerated from scripts/).
paper/               # LaTeX source (IEEE conference template).
old/                 # Prior-generation draft, notebooks, CSV suites, generated PDFs (reference only — see audit).
```

## Status

Workspace scaffolded 2026-06-10. The prior draft in `old/` was audited
(2026-06-09); see `CLAIMS.md` for which claims survived, were reframed, or were
retracted. Active work proceeds one repair at a time; current focus is
**repair #2: tokenizer-aligned ingestion** (claim S1).

## Reproduction

Reproduction instructions will be filled in as experiments land. Each EXP-XXX
directory under `experiments/` is self-contained: frozen config, environment
snapshot, seeds, and a runner invocation recorded in its logbook entry.

## Environment

Primary: Windows + PowerShell; 64 GB RAM; NVIDIA RTX 4090. GPU experiments
(serving-stack integration, ingestion) target the 4090 with a 7–8B model.
Per-experiment environment snapshots are committed under `experiments/`.
