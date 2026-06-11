# Findings

## Curated Summary

_Living results section, updated as the research picture clarifies. Written in
publication-ready prose, every number with an uncertainty bound and an EXP-XXX
reference._

### Tokenizer-aligned ingestion (S1; EXP-001, PILOT)

Snapping rope leaf boundaries to tokenizer-safe split points lets disjoint leaves be
tokenized in parallel while reconstructing the *exact* whole-string token IDs (0 boundary
divergences across byte-BPE, SentencePiece, and WordPiece families; EXP-001 Milestone A).
Given that identity guarantee, end-to-end ingestion (bytes → token IDs) decomposes into two
independent gains, measured on a non-tiled prose+code corpus (gpt2, t5-small; 7 reps; 95% CI):

- a **size-dependent single-thread gain** from chunking alone (no parallelism), 1.14×–1.52×
  as context grows 64 KB → 4 MB (gpt2), because per-leaf tokenization avoids the whole-string
  tokenizer's superlinear cost; and
- a **parallel gain** reaching ~6.5× at 16-way rayon threading (gpt2 4 MB 6.49× [6.4,6.7]), or
  up to **10.43× [10.3,10.5]** with 16-way multiprocessing on large contexts, where
  multiprocessing additionally parallelizes the Python token-object materialization that
  bottlenecks rayon's single-process gather.

The two backends show a clean, CI-separated crossover: rayon is faster for small contexts
(≤256 KB, where process/IPC overhead dominates) and multiprocessing is faster for large
contexts (≥1 MB). Both saturate by 16-way on the test CPU (32-logical hybrid Raptor Lake);
higher degrees regress. This identity-guaranteed result replaces the prior draft's retracted
4.12× ingestion claim (which compared pipelines producing different token streams).
_Pilot grade: single corpus realization (seed=42); ≥3-seed confirmatory run pending._

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

### 2026-06-11 — EXP-001: tokenizer-aligned ingestion speedup (timing, PILOT)

**Key result:** On provably identical token streams, ingestion decomposes into a ~1.1–1.5×
free single-thread chunking gain (grows with context length) plus parallel scaling to ~6.5×
(16-way rayon) or up to 10.43× [10.3,10.5] (16-way multiprocessing, large contexts). gpt2 +
t5-small, non-tiled corpus, 7 reps, 95% CI.

**Details:**
- chunk-only (rayon@1): gpt2 1.14× [1.10,1.15] @64KB → 1.52× [1.51,1.52] @4MB (baseline: serial whole-string)
- rayon peak @16: gpt2 4MB 6.49× [6.4,6.7]; t5 1MB 6.87× [6.6,7.0]
- MP @16, large ctx: gpt2 4MB 10.43× [10.3,10.5]; gpt2 1MB 7.85× [7.5,8.1]
- crossover: rayon wins ≤256KB (gpt2 256KB rayon 6.15× vs mp8 4.36×); MP wins ≥1MB (CI-separated for gpt2)

**Statistical tests:** None formal yet (pilot). Win/tie calls use 95% CI separation: gpt2 4MB
MP-vs-rayon non-overlapping ([10.3,10.5] vs [6.4,6.7]); t5 1MB overlapping → tie. Paired tests +
effect sizes deferred to confirmatory (multi-seed).

**Notes:** Surprising — multiprocessing beats rayon for large contexts, contradicting the initial
prediction; hypothesized parallelized Python token-object materialization (unprofiled). Cherry-pick
corrected: earlier-quoted 10.99× was the noisy mp32 max (CI [5.5,11.5]); mp32 oversubscribed and
excluded (gpt2 256KB mp32 = 0.77×). De-tiling reduced the chunk-only gain (1.6–1.9× → 1.1–1.5×).
Pilot grade: single corpus realization (seed=42); ≥3-seed confirmatory pending. Prior 4.12× retracted.
