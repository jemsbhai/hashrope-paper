# Findings

## Curated Summary

_Living results section, updated as the research picture clarifies. Written in
publication-ready prose, organized thematically, every number with an
uncertainty bound and an EXP-XXX reference. Empty until EXP-001 lands its first
supported result._

---

## Raw Findings Log

Chronological, append-only. Each entry ties a dated observation to an experiment.

### 2026-06-10 — Pre-experiment audit baseline (prior draft)

**Key result:** Audit of the prior draft (see `CLAIMS.md`) established the
starting point for the paper. Of the headline claims, two are false as stated
against their own data (L3 throughput: native wins 2.2×; S1 4.12× ingestion:
non-equivalent pipelines, 197/244 boundary divergence), and two are real effects
with incorrect framing (M1 memory: O(B·log N) not O(1); L4 ReAct: mostly delta
tokenization). T2 (Theorem 9 substr-hash) was repaired and shipped (Python 0.2.2).

This entry is the baseline against which corrected results will be reported. No
prior-draft number is carried into the paper without re-measurement under the
EXP-XXX protocol.
