# Experimental Logbook

Append-only. The plan (hypothesis, variables, protocol, environment) is written
**before** an experiment runs; results and interpretation are filled in after.
Never edit or delete a past entry — corrections go in a dated addendum. Every
update is its own git commit: `docs(logbook): EXP-XXX [planned|completed|failed]`.

Experiment IDs are assigned in order. Numbering note: the prior-draft figures in
`old/` used informal names (E-G2, E-D2, micro/macro suites); those are treated
as *pre-registration-less prior art* and are being re-run under proper protocol
with fresh EXP-XXX IDs. A prior figure does not count as a logged experiment.

---

## EXP-001: Tokenizer-aligned fat-leaf chunking — token-identity and ingestion speedup

**Date:** 2026-06-10 (planned); 2026-06-10 (correctness gate, in-progress)
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** in-progress

### Hypothesis
Snapping rope leaf boundaries to tokenizer-safe split points (rather than fixed
4 KB byte offsets) makes independent per-leaf tokenization produce a token
sequence **identical** to whole-string tokenization, for BPE-class tokenizers.
Given token identity, feeding disjoint leaves to parallel tokenizer workers
yields a real end-to-end ingestion speedup — hypothesized lower than the prior
draft's 4.12× (which compared non-equivalent pipelines), plausibly ~2–3×.

This directly addresses claim **S1** (currently RETRACTED) and supports a
corrected, honest version of it.

### Background / motivation
Audit finding (2026-06-09): the prior 4.12× result tokenized fixed 4 KB chunks
independently, diverging from whole-string tokenization at 197/244 boundaries
(+0.15% tokens on GPT-2 prose) and producing a padded 2D batch never recombined
into a single sequence. The speedup therefore compared pipelines with different
outputs. See `CLAIMS.md` row S1 and `old/hashrope_gpu_ingestion.pdf` (the figure
that bakes in the flaw).

### Independent variables
- Chunking strategy: {fixed 4 KB (prior, control), tokenizer-aligned (proposed)}
- Tokenizer: {GPT-2 BPE, and ≥1 other BPE family, e.g. Llama/SentencePiece} — to
  show the alignment rule is not GPT-2-specific
- Context length: logarithmic sweep (10⁴ … ~5×10⁷ chars)
- Tokenizer-worker parallelism: {1 (serial baseline), N cores}

### Dependent variables / metrics
- **Token identity** (primary correctness gate): exact equality of the leaf-wise
  concatenated token IDs vs whole-string token IDs. Boolean per (text, tokenizer),
  plus boundary-divergence count. This is a *gate*, not a performance metric:
  alignment must achieve 0 divergent boundaries or the approach is rejected.
- Tokens emitted (must equal whole-string count exactly under alignment)
- End-to-end ingestion latency (chars → token IDs ready), ms — warm, multi-run
- Speedup vs serial whole-string tokenization (with CI over replicates)

### Control conditions
- Same text corpus across strategies (real prose + code; **not** `"A"*N`, which
  is unrepresentative and would be O(1) under RepeatNode — see audit).
- Same tokenizer, same hardware, same worker count when isolating chunking.
- Baseline = whole-string single-thread tokenization (the honest "native" path,
  producing the canonical token sequence).

### Protocol
1. Implement `tokenizer_aligned_chunker` in `src/`: split near target leaf size
   but retreat to the nearest tokenizer-safe boundary (for byte-level BPE,
   a boundary that cannot sit inside a merge — e.g. a whitespace/pre-token edge).
2. Write the **token-identity proof obligation** as an automated test: for a
   battery of (text, tokenizer) pairs, assert leaf-wise concat == whole-string.
   Red first (fixed-4KB control fails it), green after (aligned passes).
3. Only once identity holds: measure serial vs parallel ingestion latency across
   the length sweep, ≥5 replicates, report median + 95% CI; record warm/cold.
4. Plot corrected speedup curve to replace `old/hashrope_gpu_ingestion.pdf`.

### Environment
- **Hardware:** [fill at run: laptop, 64 GB RAM, RTX 4090]
- **Software:** [fill: OS, Python, transformers/tokenizers versions]
- **Git commit:** [fill: clean SHA before run]
- **Config:** configs/exp_001_tokenizer_aligned.yaml (frozen copy in experiment dir)
- **Seeds:** [fill]

### Results

**Milestone A — token-identity correctness gate (Protocol steps 1–2). DONE 2026-06-10.**
Timing benchmark (Protocol steps 3–4) NOT yet run.

Implementation (`src/tokenizer_aligned.py`):
- `tokenizer_aligned_chunker(text, target_leaf_bytes, tokenizer)` — proposes cuts at a
  metaspace-aware "lone space flanked by non-whitespace" boundary (space leads the next
  leaf); identity-safe fallback (leaf grows) where no safe boundary exists in a window.
  Runtime path is a linear string scan; it does NOT tokenize the whole string.
- `token_identity_report` — offline gate: concat(per-leaf ids) vs whole-string ids.
- `fixed_byte_chunker` — the prior 4 KB behavior, kept as the control.

Red-first test (`tests/test_tokenizer_aligned.py`, 20 tests = 2 tokenizer families ×
3 leaf sizes, + UTF-8 safety + no-safe-boundary fallback):
- RED at stub: 14 failed / 6 passed (the fixed-4KB control correctly fails identity).
- GREEN after implementation: **20/20 passed** on the user's Windows machine.

Token identity (primary gate) — EXACT (0 divergent tokens) for BOTH tokenizer families:
- byte-level BPE = `gpt2`; SentencePiece = `t5-small`.
- Sandbox pre-validation on real corpus (738 KB prose + 103 KB code): aligned → 0
  divergences, 0 invalid-UTF-8 leaves, ideal leaf count (prose 181/181, code 26/26) for
  both families.
- Fixed-4KB control on the same corpus FAILS: first divergence ~token 1800; token-count
  delta +0.085% (prose) / +0.049% (code); **6 invalid-UTF-8 leaves** on prose (blind byte
  cuts bisected multibyte codepoints).

Chunking performance (sandbox, gpt2): linear, ~8 MB/s pure-Python (6.7 MB → ~1630 leaves
in ~0.8 s). Flagged for vectorization BEFORE the timing benchmark so chunking overhead does
not deflate the measured ingestion speedup.

### Observations

- The fixed-byte control surfaced claim **C3** live: naive byte cuts bisect multibyte
  codepoints, producing un-tokenizable leaves. The S1 tokenization flaw and the C3
  byte/char conflation share one root cause — cutting without tokenizer/UTF-8 awareness.
- Generality result: a single boundary predicate + a **global** (not per-leaf) verify gate
  covers both tokenizer families. An earlier attempt using per-leaf offset-boundary
  verification passed for byte-BPE but FAILED for SentencePiece (per-sequence leading
  metaspace + whitespace normalization corrupt the next leaf's opening) — fixed by the
  lone-space predicate + global verification.
- Speedup is NOT yet measured. Protocol steps 3–4 (serial whole-string vs parallel
  per-leaf ingestion latency over a length sweep, ≥5 replicates, median + 95% CI,
  warm/cold, corrected figure to replace `old/hashrope_gpu_ingestion.pdf`) remain.

### Interpretation

- The token-identity proof obligation is satisfied: leaf-wise tokenization can be made
  EXACTLY equal to whole-string tokenization for both major tokenizer families, with an
  always-correct fallback. S1's correctness precondition — that the two pipelines produce
  the same token stream — is now met and proven by test, converting S1 from
  "non-equivalent pipelines" to "provably equivalent token streams." C3 is addressed at
  the ingestion/chunking layer (HybridContext's own byte/char contract remains separate,
  still open under C3).
- The honest framing is stronger and more general than the prior GPT-2-only claim: a
  correctness theorem (gate) + a family-aware predicate (parallelism) + a fallback (safety).
- Claim movement: **S1 → IN-PROGRESS** (correctness gate passed; honest speedup pending).

### Artifacts
- Implementation: src/tokenizer_aligned.py
- Test (proof obligation): tests/test_tokenizer_aligned.py ; root conftest.py
- Results (timing): experiments/exp_001_tokenizer_aligned/results/ (see Addendum B)
- Figures: experiments/exp_001_tokenizer_aligned/figures/ (see Addendum B)

---

## EXP-001 — Addendum B (Milestone B): ingestion-speedup timing results

**Date:** 2026-06-11 ~05:00 (America/New_York)
**Researcher:** Muntaser Syed
**Status update:** in-progress → **S1 corrected form SUPPORTED at PILOT grade** (single
corpus realization; multi-seed confirmatory run still required before paper-final).
**Phase (per protocol §6a):** PILOT — point estimates with 95% CIs. NOT yet confirmatory.

Fills Protocol steps 3–4 (serial whole-string vs parallel per-leaf ingestion latency over a length
sweep). Numbers pulled from the logged JSON result files (bootstrap CIs from the harness), not from
console summaries. Append-only: nothing above is edited.

### Environment (filled from env block in result JSONs)
- **Hardware:** Intel64 Family 6 **Model 183** (Raptor Lake), `cpu_count=32` **logical** threads.
  NOTE: Intel **hybrid (P+E)** mobile part, NOT "16 physical + SMT" (an earlier verbal claim by
  Claude — corrected). Exact P/E split not in env block; stated cautiously. 64 GB RAM; RTX 4090
  present but **idle** (tokenization is CPU-only; `USE_TORCH=0`).
- **Software:** Windows-11-10.0.26200-SP0; Python 3.12.2; transformers 5.1.0; tokenizers 0.22.2.
- **Git commit:** [PENDING — run `git rev-parse HEAD`]. Flagged for fill before paper-final.
- **Seeds/reps:** corpus shuffle seed = 42 (fixed, single realization); reps = 7/cell; warmup
  discarded; bootstrap 95% CI. **No multi-seed yet** (see Limitations).
- **Data:** non-tiled (15 Gutenberg books + 40 Python-stdlib files, shuffled at 8 KB); provenance in
  data/DATA_README.md. ALL cells `corpus_tiled=False`.

### Results A — chunking-only (rayon@1, NO parallelism); speedup vs serial whole-string
Grows monotonically with length (whole-string superlinear penalty grows with N). Tight CIs.

| length | gpt2 | t5-small |
|---|---|---|
| 64 KB  | 1.14× [1.10,1.15] | 1.14× [1.14,1.15] |
| 256 KB | 1.35× [1.35,1.36] | 1.27× [1.27,1.30] |
| 1 MB   | 1.42× [1.38,1.43] | 1.31× [1.29,1.33] |
| 4 MB   | 1.52× [1.51,1.52] | 1.38× [1.37,1.40] |

### Results B — rayon parallel scaling (speedup vs serial)
Peaks at **16 threads**; 24/32 *regress* (hybrid CPU: E-cores / oversubscription degrade CPU-bound
BPE). gpt2 @ 4 MB: t1 1.52 → t2 2.60 → t4 4.37 → t8 6.22 → **t16 6.49 [6.4,6.7]** → t24 6.16 →
t32 5.08. Peak per cell:

| length | gpt2 peak | t5-small peak |
|---|---|---|
| 64 KB  | 4.21× [4.1,4.5] @16 | 3.63× [3.5,3.7] @8  |
| 256 KB | 6.15× [6.1,6.3] @16 | 5.35× [5.3,5.7] @8  |
| 1 MB   | 6.50× [6.3,6.6] @16 | 6.87× [6.6,7.0] @16 |
| 4 MB   | 6.49× [6.4,6.7] @16 | 6.33× [6.0,6.6] @16 |

Efficiency tracks granularity: 64 KB (16 leaves) parallelizes poorly; long contexts use cores well.

### Results C — rayon vs multiprocessing (lengths 256K/1M/4M; MP archive 045750Z)
**Clean CI-backed crossover:** rayon wins SMALL (MP pays process/IPC + pickle-back with few leaves);
MP wins LARGE (parallelizes the ~1M-object Python token materialization that bottlenecks rayon's
single main-process gather).

| cell | rayon peak | MP (clean, ≤16 workers) | winner (CI-backed) |
|---|---|---|---|
| gpt2 256 KB | **6.15× [6.1,6.3]** @16 | mp8 4.36× [4.1,4.4]; mp16 5.42× [3.9,6.0] | **rayon** |
| gpt2 1 MB   | 6.50× [6.3,6.6] @16 | **mp16 7.85× [7.5,8.1]** | **MP** (non-overlap) |
| gpt2 4 MB   | 6.49× [6.4,6.7] @16 | mp8 8.02× [8.0,8.2]; **mp16 10.43× [10.3,10.5]** | **MP** (decisive) |
| t5 256 KB   | **5.35× [5.3,5.7]** @8 | mp8 4.67× [4.6,4.7] | **rayon** |
| t5 1 MB     | 6.87× [6.6,7.0] @16 | mp16 6.89× [6.6,7.0] | **tie** (overlap) |
| t5 4 MB     | 6.33× [6.0,6.6] @16 | mp16 6.76× [6.6,6.8] | MP (marginal) |

### Observations (incl. corrections to prior verbal claims by Claude)
1. **Cherry-pick caught + corrected (§6c).** The "10.99×" headline Claude repeated is the **mp32**
   max, CI **[5.5,11.5]** — high-variance, NOT defensible. Clean number: **mp16 10.43× [10.3,10.5]**.
   The figure script's `best_speedup` reports the max, which masked this.
2. **mp32 oversubscribed** (>physical cores on a 24-core hybrid): high variance, sometimes
   catastrophic (**gpt2 256 KB mp32 = 0.77×**). **Excluded from all claims;** backfill drops it.
3. **Both backends saturate by 16.** rayon t24/t32 regress; MP >16 degrades. Colab NOT needed for
   CPU cores; reserved for GPU stage (S3).
4. **De-tiling cut chunk_x DOWNWARD** (1.6–1.9× tiled → 1.1–1.5× non-tiled). Tiled overstated it.

### Interpretation
- **S1 corrected form SUPPORTED (pilot).** Honest decomposition: ~1.1–1.5× free chunking + parallel
  to ~6.5× (rayon) / ~10.4× (MP, large contexts) at 16-way, on a provably identical token stream
  (Milestone A). Replaces retracted 4.12×. MP>rayon crossover is a genuine non-obvious result.
- **MP>rayon mechanism (hypothesis, unproven):** rayon parallelizes Rust tokenization but materializes
  Python token lists serially in the main process (GIL); for ~1M tokens that gather dominates. MP
  parallelizes materialization across processes, outweighing pickle-back IPC for large contexts.
  CONFIRM via a profile separating tokenize-time vs Python-materialize-time (deferred).
- **Required before paper-final (confirmatory, §6a/§6b):** (a) **multi-seed** — only ONE corpus
  realization (seed=42); need ≥3 shuffle seeds (this is the pilot→confirmatory gap; S1 stays PILOT
  until done); (b) record producing git SHA; (c) profile the MP>rayon mechanism; (d) a third
  tokenizer family for timing (identity already verified for more families in Milestone A).
- **Claim movement:** S1 IN-PROGRESS → **SUPPORTED (pilot)**; → SUPPORTED (confirmatory) after (a).

### Artifacts (canonical files this addendum is computed from)
- rayon@1 baseline (non-tiled): results/ingestion_rayon1_20260611T043517Z.json
- rayon@{2,4,8,16,24,32}: results/ingestion_rayon{N}_latest.json (non-tiled sweep, ~04:36–04:41Z)
- multiprocessing {2..32}: results/ingestion_rayon1_20260611T045750Z.json
- Figures: figures/exp001_speedup_vs_cores.{png,pdf}, figures/exp001_rayon_vs_mp.{png,pdf}
- Speedup CI derivation: serial_median / parallel_ci95 (serial CI <1% wide)

---

## EXP-001 — Addendum B-repro: reproducibility check (CRITICAL methodological finding)

**Date:** 2026-06-11 ~05:10 (America/New_York)
**Trigger:** Protocol §7c.3 — re-run a key result and confirm. The backfill run
(`rayon1_latest`, RAYON_NUM_THREADS=1, mp 2/4/8/16, same corpus seed=42) is an independent
repeat of the MP cells originally measured in archive 045750Z. Compared median ms across runs.

### Result — within-run 7-rep CIs UNDERESTIMATE cross-run variance
Comparing identical cells, two independent runs, same seed:

| cell | archived (045750Z) | backfill (latest) | cross-run |
|---|---|---|---|
| gpt2 4 MB mp16 | 10.43× (201 ms) | 10.97× (213 ms) | **disjoint** (~6%) |
| gpt2 4 MB mp8  | 8.02× (260 ms)  | 8.68× (269 ms)  | overlap |
| gpt2 1 MB mp16 | 7.85× (59 ms)   | 6.95× (77 ms)   | **disjoint** (~13%) |
| gpt2 256 KB mp16 | 5.42×         | **1.36×**        | **disjoint (4× swing)** |
| t5 256 KB mp16 | 1.83×           | 4.93×           | **disjoint** |
| t5 1 MB mp16   | 6.89×           | 6.86×           | overlap |
| t5 4 MB mp16   | 6.76×           | 7.07×           | overlap |

Also: gpt2 64 KB chunk-only (rayon@1) = 1.14× (043517Z) vs **0.87×** (backfill) — crosses below
1.0. The 64 KB "free chunking win" recorded in Addendum B is NOT reliably positive.

### Interpretation (tempers Addendum B; append-only, does not edit it)
- The 7-rep bootstrap CI captures rep-to-rep jitter but NOT across-run variance (process-pool
  spawn, OS scheduling, thermal). For the MP backend especially, cross-run drift is ~6–25% at
  large contexts and a 2–4× swing at small contexts. **The single-run CIs in Addendum B are too
  tight to be the paper's uncertainty.**
- What IS robust across runs (safe to claim): (a) the **direction** — tokenizer-aligned chunking
  helps and parallel ingestion helps; (b) **large-context magnitudes to ~1 sig fig** — rayon ~6.5×
  @16, MP ~10× @16 for gpt2 4 MB; (c) the **MP>rayon crossover at large contexts** (gpt2 4 MB MP
  >10× in BOTH runs vs rayon ~6.5×).
- What is NOT robust from one run (must NOT be stated as precise): exact multipliers; ALL ≤256 KB
  cells (chunk_x and MP both swing across 1.0–5×); the gpt2 64 KB "free win."
- **Action / claim adjustment:** S1 remains **SUPPORTED (pilot)** for the DIRECTION + 1-sig-fig
  large-context magnitudes ONLY. The confirmatory run is upgraded in scope: not just ≥3 corpus
  seeds, but ≥3 *independent process invocations per seed* (fresh interpreter + pool), reporting
  mean ± std ACROSS runs as the uncertainty — the within-run CI is retired as the error bar.
  Small contexts (≤64 KB) likely dropped from headline claims (overhead-dominated, unstable).
- This is a genuine finding, not a failure: it tells us the benchmark's true error model and
  prevents shipping over-precise multipliers a reviewer could fail to reproduce.
