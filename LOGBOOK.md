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

---

## EXP-001 — Addendum C (Milestone C): CONFIRMATORY run (planned)

**Date:** 2026-06-11 (planned, pre-run)
**Status:** planned → (results to follow)
**Motivated by:** Addendum B-repro — within-run CIs understate cross-run variance, so the
confirmatory error bar must be the spread across independent invocations and corpus seeds.

### Hypothesis
The DIRECTION established in pilot is stable under proper replication: at large contexts, (a)
tokenizer-aligned chunking gives a single-thread gain >1.0, (b) parallel ingestion gives a large
gain, and (c) multiprocessing beats rayon. Concretely we expect, across seeds x invocations,
rayon@16 mean ~6× and mp@16 mean ~8–11× for gpt2 at 4 MB, with mp>rayon in a clear majority of
paired runs (sign test). We do NOT pre-commit to exact multipliers (the pilot showed ~6–25% drift).

### Design
- **Replication unit = one independent process invocation** (fresh interpreter + fresh pool).
- Grid: seeds {42,43,44} x invocations {3} = **n=9 independent runs per cell**.
- Operating points (large contexts only, where the effect is real): chunk-only (rayon@1),
  rayon@16, mp@16; per invocation two bench subprocesses (P: rayon@16; M: rayon@1 + mp@16).
- Lengths {1 MB, 4 MB}; tokenizers {gpt2, t5-small}; reps=5 within each invocation (just to
  stabilise each invocation's point estimate — NOT the error bar).
- **Error bar = std across the 9 runs** (+ t-based 95% CI on the mean). Within-run bootstrap CI is
  retired. Corpus robustness shown via per-seed means. Paired **sign test** of mp@16 vs rayon@16
  (paired by seed,invocation); report wins/n and exact two-sided p.

### Independent / dependent variables
- IV: corpus seed, invocation index, backend (chunk/rayon/mp), context length, tokenizer.
- DV: speedup vs serial whole-string (per invocation), aggregated to mean ± std (n=9).

### Protocol (frozen command)
```
python scripts/exp001_confirm.py --seeds 42,43,44 --invocations 3 \
    --tokenizers gpt2 t5-small --lengths 1048576,4194304 --reps 5 --rayon-degree 16 --mp-degree 16
```
Harness: `scripts/exp001_confirm.py` (validated end-to-end on a 2x2 sandbox grid). Corpora
`data/raw/corpus_s{42,43,44}.txt` built via `prep_corpus.py --seed` (books cached: one download).
Outputs: `experiments/exp_001_tokenizer_aligned/results/confirm/{s<S>_i<inv>/, summary.json}`.
Env recorded per sub-run in each bench JSON; corpus SHA-256 (16) recorded in summary.json.

### Promotion criterion (what flips S1 pilot -> confirmatory)
S1 -> **SUPPORTED (confirmatory)** iff, for gpt2 at >=1 MB: (i) rayon@16 mean and mp@16 mean are
stable across seeds (per-seed means within ~1 std), (ii) mp>rayon sign test is significant
(>=8/9, p<=0.05), and (iii) the reported numbers are stated as mean ± std (n=9), never as the
within-run CI. Cells that fail (expected: small/marginal t5) are reported honestly as such, not
dropped silently. If the direction does NOT hold, S1 is downgraded and re-examined — the plan is
written before the data precisely so this is not retrofitted.

### Results (filled 2026-06-11, post-run; plan above unchanged)

**Run:** 3 seeds {42,43,44} x 3 invocations = n=9 independent process runs/cell; reps=5 within each.
**Corpora (SHA-256[:16]):** 42=fda6a43a3fe34184, 43=85ca58707bae1597, 44=dfd645be94bcddce.
**Artifacts:** results/confirm/summary.json + results/confirm/s{42,43,44}_i{0,1,2}/.
Speedup vs serial whole-string; mean +/- std across the 9 runs (NOT within-run CI):

| cell | chunk (rayon@1) | rayon@16 | mp@16 | mp>rayon (paired sign test) |
|---|---|---|---|---|
| gpt2 1 MB | 1.46 +/- 0.01 | 6.09 +/- 0.53 | 7.01 +/- 0.72 | **9/9, p=0.004** (median diff +0.87) |
| gpt2 4 MB | 1.50 +/- 0.02 | 6.62 +/- 0.37 | 7.59 +/- 0.41 | **9/9, p=0.004** (median diff +1.06) |
| t5 1 MB | 1.30 +/- 0.02 | 6.71 +/- 0.26 | 6.39 +/- 0.30 | 2/9, p=0.18 (n.s.; **rayon ahead**) |
| t5 4 MB | 1.36 +/- 0.02 | 6.51 +/- 0.33 | 7.16 +/- 0.40 | 9/9, p=0.004 (median diff +0.54) |

Per-seed means (corpus-robustness): all within ~1 std of the grand mean for every cell (e.g.
gpt2 4 MB rayon per-seed {6.90,6.46,6.51}, mp {7.69,7.64,7.46}). chunk-only per-seed spread <=0.01.

### Promotion-criterion evaluation (verbatim against the committed criterion)
- **gpt2 1 MB:** (i) rayon per-seed max-dev 0.49 < std 0.53; mp max-dev 0.43 < std 0.72 -> stable.
  (ii) 9/9, p=0.004. (iii) mean+/-std. **PASS.**
- **gpt2 4 MB:** (i) stable (devs << std). (ii) 9/9, p=0.004. (iii) mean+/-std. **PASS.**
- **Verdict: S1 -> SUPPORTED (confirmatory)** for gpt2 large-context. t5 reported honestly:
  t5 4 MB confirms MP>rayon (9/9); **t5 1 MB is an exception** (rayon 6.71 vs mp 6.39, 2/9, n.s.) --
  NOT dropped, reported as a regime where rayon and MP are comparable (rayon marginally ahead).

### Observations / honest corrections (this supersedes pilot magnitudes)
1. **Magnitude corrected DOWN; pilot was the artifact.** Confirmatory mp@16 gpt2 4 MB = 7.59 +/-
   0.41; all 9 runs in [7.16, 8.42]. The pilot single-runs (10.43, 10.97x) sit ABOVE every
   confirmatory run -- a systematic session-level gap, not noise. Hypothesis (not instrumented):
   under the sustained 18-run load, 16-core workloads (mp) throttle harder than the 1-core serial
   baseline, compressing the ratio; the pilot caught a cold-thermal peak. **The confirmatory
   steady-state supersedes the pilot peak** -- and steady-state is what a real server experiences,
   so it is the correct number to report. (This is the THIRD downward correction after de-tiling
   and the mp32 cherry-pick exclusion; each makes the claim more reproducible.)
2. **Effect size is modest, not dramatic.** MP beats rayon by ~13-16% (median diff +0.87 / +1.06
   for gpt2), NOT the ~60% the pilot 10-vs-6.5 implied. The WIN is highly reliable (9/9, p=0.004);
   the MARGIN is small. Both backends are 16-core so they throttle similarly -> the RELATIVE result
   is robust to the thermal effect even though absolute speedups compressed.
3. **Design caveat (conservative for MP):** within each invocation the M call (mp) runs AFTER the
   P call (rayon), i.e. on an already-warmed machine. If anything this DISADVANTAGES mp; mp wins
   gpt2 9/9 regardless. A future run could randomize P/M order or cool between calls.
4. **chunk-only is the most reproducible result:** 1.30-1.51x, std <=0.02, monotonic in length.
   The retracted 4.12x is replaced by: ~1.5x free (single-thread) + ~6.6x rayon / ~7.6x mp at
   16-way for gpt2 4 MB, all on a provably-identical token stream.
5. **Independence caveat:** the 9 runs are one session (shared thermal/background state), so std may
   understate cross-SESSION variance. Defensible as a controlled confirmatory; a cross-day repeat
   would further harden it (logged as optional future work, not blocking).

### Claim movement
S1: SUPPORTED (pilot) -> **SUPPORTED (confirmatory)** for gpt2 >=1 MB (direction + mean+/-std
magnitudes + significant paired MP>rayon). t5 1 MB MP-vs-rayon recorded as a comparable/rayon-
ahead regime (honest negative for the MP-crossover at that single cell).

---

## EXP-002: Flatten — in-order materialization vs midpoint re-split (claim S4)

**Date:** 2026-06-11 (executed 2026-06-11)
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** DONE (confirmatory)

### Hypothesis
The prior draft's flatten "tax" (claim S4, `4_serialization_tax.csv`: Hybrid ≈ 945 ms,
~constant in edit count K=1..100, vs Native growing 0.9→41 ms) is an IMPLEMENTATION
ARTIFACT, not an inherent O(N) serialization cost. The old `flatten_context`
(`old/hashrope1_exmain.py` L579) materializes a rope by recursively splitting at midpoints
down to ≤4 KB pieces. Where a midpoint lands inside a leaf (the generic, non-power-of-two
case), `rope_split` reconstructs `Leaf` objects, and every `Leaf.__init__` RECOMPUTES its
polynomial hash from scratch — re-hashing ~Θ(N) bytes through pure-Python per-byte modular
reduction (`mersenne_mod`). The library's in-order materializer `rope_to_bytes`
(`packages/python/hashrope/rope.py` L576 → `_collect_bytes`) reads each stored `Leaf.data`
once and joins — 0 splits, 0 leaf re-allocations, 0 hash recomputation. Expected ~2–3 orders
of magnitude faster, alignment-independent. S4 flips from "inherent tax / damage control" to
"the structure flattens in linear time with no recomputation."

### Background / recon + sandbox validation (2026-06-11)
- `rope_to_bytes`/`_collect_bytes` is ALREADY the correct O(N) in-order flatten; NO library
  change is required — the fix is to call it instead of re-splitting.
- `rope_from_bytes` returns a SINGLE `Leaf` (no auto-chunking), so the realistic rope must be
  built like the prior `HashRopeAdapter` (`old/hashrope1_exmain.py` L40): chunk into 4 KB
  leaves and merge BOTTOM-UP pairwise.
- ALIGNMENT is decisive: on a power-of-two leaf count every byte-midpoint hits a leaf boundary
  (`_split` returns existing children → 0 re-hash → no tax). Non-power-of-two counts misalign
  → the tax. The sweep therefore uses realistic DECIMAL sizes (e.g. N = 2,000,000 → 489 leaves),
  NOT binary 2 MiB (= 2^21 B = 512 leaves, which would hide the tax).
- Faithful sandbox reproduction (2 MB "C", bottom-up 4 KB build): broken 783 ms vs fixed
  1.25 ms = 627×; broken counts 511 splits / 1022 `Leaf` allocs / 2,092,364 `mersenne_mod`
  calls; cProfile attributes ~99% to `PolynomialHash.hash`. Confirms the mechanism is
  redundant hashing (NOT allocation churn or a clean N log N).

### Independent variables
- Flatten implementation: {broken = faithful `flatten_context` port (CONTROL); fixed =
  `rope_to_bytes`}
- Context size N (log sweep, DECIMAL / non-power-of-two leaf counts): {64,000; 256,000;
  1,000,000; 2,000,000; 4,000,000; 8,000,000; 16,000,000} bytes; **N = 2,000,000 is the
  reference point** (matches the prior 945 ms anchor)
- Leaf size fixed at 4096 B (leaf-size sensitivity is EXP-010)

### Dependent variables / metrics
- **Correctness (HARD gate, boolean):** fixed == original bytes AND == broken output,
  byte-identical, at every (size, seed)
- **Operation-count guard (instrumented via monkeypatch):** per flatten, # `rope_split` calls,
  # `Leaf` re-allocations, # `PolynomialHash.hash` recomputations. Expect fixed = 0/0/0;
  broken = Θ(#leaves) leaf re-allocs re-hashing ~Θ(N) bytes
- **Wall-clock latency (ms):** warm, interleaved broken/fixed, mean ± std across runs; CV
- **Speedup** = broken/fixed (mean ± std), per size
- **Empirical scaling exponent** (log-log slope of latency vs N) per arm — measured, not asserted

### Control conditions
- Same rope object flattened by both arms (paired)
- Real corpus (prose + code via `prep_corpus.py`); NOT `"A"*N`/`"C"*N` (degenerate). The re-hash
  mechanism is content-independent; real content guards against any content artifact.
- Identical bottom-up fat-leaf build for both arms
- Warm-up discarded; interleaved order; cool-downs; AC power; idle check before runs (per the
  TOML local-measurement discipline); fixed git SHA recorded
- Baseline = the broken `flatten_context` algorithm (reproduces the prior 945 ms-class cost)

### Protocol (TDD, red-first)
1. `src/flatten.py`: `make_hash`, `build_fat_leaf_rope` (bottom-up), `flatten_broken` (port,
   CONTROL), `flatten_fixed` (= `rope_to_bytes`, FIX).
2. `tests/test_flatten.py` (red-first, mirrors EXP-001): CONTROL test
   `test_broken_flatten_reproduces_rehash_tax` PASSES (tax real: splits/leaf/hash > 0); FIX
   tests (byte-identity, fixed==broken, zero-rehash guard) RED until `flatten_fixed` wired,
   then GREEN.
3. `scripts/exp002_bench.py`: build per size; time both arms (warm, interleaved, ≥3 corpus
   seeds × ≥3 independent process invocations); write `experiments/exp_002_flatten/results/`
   JSON (latest + timestamped) with env + corpus SHA-256 + git SHA; CV check.
4. `scripts/exp002_figure.py`: log-log latency vs N (both arms, fitted slopes) + speedup;
   replaces the prior serialization-tax figure.
5. Record Results/Observations/Interpretation here + findings.md; evaluate the promotion
   criterion verbatim; commit.

### Environment
- **Hardware:** laptop, Intel Core i9-14900HX (24c/32t hybrid P+E), 64 GB RAM, RTX 4090 Laptop
  (idle; CPU-only experiment), AC power
- **Software:** Windows 11 (10.0.26200), Python 3.12.2, hashrope 0.2.2
- **Git commit:** 47ea627 (source clean; result JSON records `dirty=True` = untracked smoke-run result files, not source changes)
- **Seeds:** corpus realization seeds {42, 43, 44} (≥3)

### Promotion criterion (verbatim, written before any data)
S4 → SUPPORTED iff, across ≥3 corpus seeds × ≥3 independent invocations:
(i)  [HARD] byte-identity holds at every (size, seed): fixed == original == broken,
     0 mismatches — else the experiment FAILS outright.
(ii) operation-count guard: `flatten_fixed` performs 0 `rope_split`, 0 `Leaf` re-allocations,
     and 0 hash recomputations; `flatten_broken` performs Θ(#leaves) `Leaf` re-allocations
     re-hashing ~Θ(N) bytes — i.e., the fix eliminates ALL redundant hashing.
(iii) wall-clock: at the N = 2,000,000 reference, fixed mean ≤ 5 ms AND speedup (broken/fixed)
     ≥ 100× (mean ± std, n ≥ 9); speedup ≥ 100× at every size ≥ 256 KB (no monotonicity
     assumed — both arms scale ~linearly in N).
(iv) scaling reported empirically (log-log slope per arm); fixed consistent with O(N)
     (slope ≈ [0.85, 1.2]); NO pre-asserted N log N — the mechanism is measured redundant hashing.
(v)  all numbers reported as mean ± std across runs (never within-run CI).
Failure of (ii)/(iii) → S4 stays REFRAMED and we investigate (no retrofit).
Honesty note: the multiplier is Python-amplified (pure-Python per-byte hashing); the conceptual
fix (read stored bytes, don't re-hash) holds in Rust too, where the absolute win is smaller.
Rust confirmation optional/deferred.

### Results
**Run 2026-06-11**, commit `47ea627` (clean source; the result JSON's `dirty=True` is the untracked result files written by the optional smoke run, not source changes). Windows 11 (10.0.26200), Python 3.12.2, hashrope **0.2.2**, i9-14900HX (32 logical). Corpus seeds {42,43,44}, SHA-256[:16] `fda6a43a` / `85ca5870` / `dfd645be` (the SAME real corpus as EXP-001). **n = 9 runs/cell** (3 seeds × 3 invocations), reps=5 within each invocation (median), 2 s cool-down, fresh subprocess per (seed,invocation). Results: `experiments/exp_002_flatten/results/exp002_flatten_latest.json` (+ `…_20260611T191246Z.json`).

| N (bytes) | leaves | broken ms (mean±std, CV) | fixed ms (mean±std, CV) | speedup (mean±std) | guard broken L/S/H | fixed |
|-----------|-------:|-------------------------|------------------------|--------------------|--------------------|-------|
| 64,000    | 16   | 11.23 ± 0.13 (1.2%)    | 0.003 ± 0.000 (7.2%)  | 3426.9 ± 219.8 | 30/15/30      | 0/0/0 |
| 256,000   | 63   | 48.88 ± 1.04 (2.1%)    | 0.050 ± 0.012 (23.4%) | 1082.1 ± 469.5 | 126/63/126    | 0/0/0 |
| 1,000,000 | 245  | 199.18 ± 4.48 (2.2%)   | 0.272 ± 0.016 (6.0%)  | 734.6 ± 39.3   | 510/255/510   | 0/0/0 |
| 2,000,000 | 489  | 437.36 ± 59.97 (13.7%) | 0.546 ± 0.016 (2.8%)  | 800.1 ± 101.3  | 1022/511/1022 | 0/0/0 |
| 4,000,000 | 977  | 865.01 ± 104.66 (12.1%)| 1.166 ± 0.053 (4.5%)  | 740.6 ± 65.5   | 2044/1023/2044| 0/0/0 |
| 8,000,000 | 1954 | 1692.57 ± 167.06 (9.9%)| 2.305 ± 0.092 (4.0%)  | 733.8 ± 55.0   | 4092/2047/4092| 0/0/0 |

(L/S/H = `Leaf` re-allocs / `rope_split` calls / `PolynomialHash.hash` calls, one instrumented pass per arm per size; mirrors `tests/test_flatten.py::_instrument`.) Empirical log-log slope (latency vs N): **broken 1.044**; fixed full-sweep 1.327; **fixed ≥1 MB 1.034**. Per-seed speedup means at N≥1M are within ~10% across seeds (2M: s42 891×, s43 746×, s44 764×; 8M: 769/716/716×) — no seed-specific artifact.

**Promotion-criterion evaluation (verbatim against the criterion above):**
- (i) byte-identity **[HARD]: PASS** — 0 mismatches across all 6 sizes × 3 seeds × 3 invocations (fixed == original == broken; the worker also re-checks identity after the rep loop, so persistence is confirmed).
- (ii) operation-count guard: **PASS** — `flatten_fixed` = 0/0/0 at every size; `flatten_broken` = Θ(#leaves) (L = H = 2·splits, scaling linearly with #leaves ≈ N/4096), re-hashing Θ(N) bytes. The fix eliminates ALL redundant hashing.
- (iii) wall-clock: **PASS** — at N=2,000,000 fixed 0.546 ms ≤ 5 ms AND speedup 800× ≥ 100×; speedup ≥ 100× at every size ≥ 256 KB (256K 1082×, 1M 735×, 2M 800×, 4M 741×, 8M 734×).
- (iv) scaling (descriptive, per the pre-committed decision — **non-gating**, since only (i)/(ii)/(iii) gate): broken slope **1.044** (linear; **not** N log N — consistent with "the mechanism is measured redundant hashing"). **Honest note on the [0.85,1.2] band:** the *fixed* full-sweep slope (1.327) lands ABOVE the band because the sub-ms small-N points (64K–256K fixed ≈ 3–50 µs) are timer-floor dominated; the ≥1 MB subset (above the floor) gives **1.034**, in-band. O(N) for fixed is established by the **guard** (0 re-hash / single pass), as pre-committed — not by the full-sweep slope fit. Reporting both slopes was pre-registered in the bench; the inflation pattern is exactly what the decision predicted.
- (v) mean ± std reported throughout (n=9); within-run CI retired.

**VERDICT: PASS** (gates (i) HARD, (ii), (iii) all hold). → S4 promoted REFRAMED → SUPPORTED.

### Observations
- The fix is mechanistically clean, not merely faster: the guard shows `rope_to_bytes` performs **literally zero** split / `Leaf` / hash operations, while the broken midpoint-resplit allocates and re-hashes Θ(#leaves) leaves (1022 allocs + 1022 hashes @2 M; 4092 + 4092 @8 M). The "tax" was redundant recomputation, exactly as the recon + sandbox predicted (sandbox @2 M had also reported 511 splits / 1022 allocs).
- **Broken is empirically O(N) linear (slope 1.044), NOT O(N log N)** as the prior-outline shorthand and the PROGRAM title state. Per-leaf cost is ~constant (~0.7–0.9 ms/leaf ≈ hashing one 4 KB leaf in pure Python), so total ∝ #leaves ∝ N. The earlier "N log N allocations" framing was imprecise; the cost is linear redundant re-hashing. (Title in PROGRAM left verbatim as the experiment's label; correction recorded in its note + CLAIMS S4.)
- Variance is asymmetric and informative: the **fixed** arm is rock-stable (CV 3–6% at N≥1M), the **broken** arm gets noisier with size (CV 10–14% at 2–8 M) — the allocation-churn / GC sensitivity the cross-run error model is built to capture. The verdict is robust to it: even at the worst cell (2 M, mean−1σ ≈ 700×) the speedup clears the 100× gate ~7×.
- The 256 K speedup band is wide (1082 ± 470) only because **fixed** @256 K (0.050 ms) sits near the timer floor (CV 23%); broken @256 K is tight (CV 2.1%). Not a stability concern for the claim.
- Provenance: corpus SHAs match EXP-001; `dirty=True` reflects the untracked smoke-run result files, source clean at 47ea627.

### Interpretation
S4 moves from "inherent serialization tax / damage control" to a **supported strength**: the rope materializes in linear time with **zero hash recomputation**, and the prior ~945 ms-class "tax" is fully accounted for as redundant per-leaf re-hashing introduced by midpoint re-splitting — removed by calling the in-order materializer the library already ships (`rope_to_bytes`). **No library change was required.**

**Scope / honesty (carried into the paper, per the pre-committed honesty note):** the headline ~730–800× is **Python-interpreter-amplified** — the broken arm's cost is pure-Python per-byte polynomial hashing, so the multiplier is a property of *this* implementation, not a language-independent constant. The transferable result is the **elimination of redundant work** (0 splits / 0 re-allocs / 0 re-hashing — the guard proves it), which holds in Rust with a smaller absolute constant. The paper states the mechanism (guard counts + linear broken scaling) as the claim and treats the ms multiplier as an implementation datum, not a fundamental bound. The absolute broken latency here (437 ms @2 M) differs from the historical 945 ms figure (different hardware/run); the *mechanism and its removal* are the claim, not a specific millisecond value. Rust confirmation remains optional/deferred (the conceptual fix is identical: read stored bytes, do not re-hash). **EXP-002 is closed.**

### Artifacts
- Implementation/control: src/flatten.py
- Tests (gate + guard): tests/test_flatten.py
- Bench: scripts/exp002_bench.py ; results experiments/exp_002_flatten/results/
- Figure: scripts/exp002_figure.py ; figures/exp002_flatten_latency.{png,pdf}


---

## EXP-005: Prefix-reuse identification — LCP via prefix-hash binary search (claim T3)

**Date:** 2026-06-11 (planned)
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned

### Hypothesis
The longest common prefix (LCP) of two ropes can be found in O(log² N) time via
binary search on prefix hashes (each step using `rope_substr_hash`, Theorem 9).
Given ropes A, B of comparable length ~N, binary search performs O(log N) steps,
each comparing a single prefix hash in O(k·log w) via the Theorem 9 fast path,
yielding O(log N · log w) = O(log² N). This is the **content-identity-query**
side of the unification thesis: the same persistent structure that gives O(log)
edits answers "how much prefix do these two contexts share?" in O(log² N),
enabling prefix-dedup for KV-cache reuse without materializing or scanning the
full text.

Prior informal evidence: ~28 ms flat from 500 KB to 16 MB (exact divergence byte
verified on 50+ random points). This experiment formalizes that under the
confirmatory protocol.

### Independent variables
- Rope size N (bytes): {64,000; 256,000; 1,000,000; 2,000,000; 4,000,000; 8,000,000}
- LCP fraction f (where the divergence byte is placed): {0.5} (primary timing
  condition), {0.0, 0.99} (secondary)
- Content: real corpus (3 seeds) alongside synthetic controls (flipped-byte pairs
  with algebraically known LCP)
- Corpus seed: {42, 43, 44}

### Dependent variables / metrics
- **Correctness (HARD gate):** `lcp_hash` returns the exact same position as
  `lcp_brute` for every (N, seed, f) — 0 mismatches.
- **Step-count guard:** number of `rope_substr_hash` calls per `lcp_hash`
  invocation ≤ ceil(log2(min(len_a, len_b))) + 1.
- **LCP-hash latency (ms):** warm, per (N, f), mean ± std across ≥9 runs.
- **Brute-force latency (ms):** comparison baseline (expected O(LCP_length), i.e.
  O(f·N) — linear in the prefix length, providing a scaling contrast).
- **Prefix-dedup workload (descriptive):** K=10 simulated "prompts" sharing a
  system-prompt prefix of known length; report LCP identification accuracy and
  total batch LCP time. (Subsumed by the correctness gate if gate (i) passes;
  included for the narrative connecting T3 to the prefix-reuse application.)

### Control conditions
- Brute-force byte-by-byte LCP (`lcp_brute`) as correctness oracle + timing
  baseline
- Synthetic control: pair with identical prefix + single flipped byte at position
  L → algebraically known LCP = L. Guarantees correctness is tested against a
  ground truth, not just against a second implementation.
- Real corpus (same files as EXP-001/002: `data/raw/corpus_s{42,43,44}.txt`,
  SHA-256[:16] 42=fda6a43a, 43=85ca5870, 44=dfd645be)
- Same hardware, AC power, idle baseline, warmup discarded, cool-downs

### Workload construction
For each (N, seed, f):
1. `A_bytes = corpus[:N]`; build rope A via bottom-up 4 KB fat-leaf merge
   (same builder as EXP-002).
2. `B_bytes = A_bytes` with byte at position `L = int(f * N)` flipped
   (XOR 0xFF; if the result is the same byte — can't happen with XOR 0xFF —
   use 0x01). This guarantees `LCP(A, B) = L` exactly.
3. Build rope B from B_bytes.
4. For f=0.0: flip at position 0 → LCP = 0.
   For f=0.5: flip at position N//2 → LCP = N//2.
   For f=0.99: flip at position int(0.99*N) → LCP = int(0.99*N).

Prefix-dedup workload: 10 "prompts" = shared_prefix (corpus[:P]) +
unique_suffix (10 disjoint corpus slices). Verify LCP correctly returns P
for all 45 pairs.

### Protocol (TDD, red-first)
1. Implement `lcp_hash(rope_a, rope_b, h)` in `src/lcp.py`: binary search
   on prefix length L ∈ [0, min(len_a, len_b)], comparing
   `rope_substr_hash(A, 0, L, h) == rope_substr_hash(B, 0, L, h)`. Also
   `lcp_brute(a: bytes, b: bytes) -> int`: byte-by-byte oracle. Include an
   instrumented variant (`lcp_hash_counted`) returning (lcp, num_hash_calls).
2. `tests/test_lcp.py` (red-first, mirrors EXP-001/002):
   - `test_lcp_correctness_real_corpus`: lcp_hash == lcp_brute on corpus
     slices with known flipped byte (RED at stub)
   - `test_lcp_correctness_synthetic`: algebraically known LCP (RED at stub)
   - `test_lcp_step_count`: hash queries ≤ ceil(log2(N)) + 1 (RED at stub)
   - `test_lcp_zero_prefix`: f=0.0 → returns 0
   - `test_lcp_full_match`: identical ropes → LCP = len
   - `test_lcp_empty_ropes`: edge cases (None, empty)
3. `scripts/exp005_bench.py`: orchestrator (spawns fresh subprocesses). Per
   (seed, invocation): load corpus, build rope pairs at each (N, f), time
   lcp_hash (reps=5→median) and lcp_brute (reps=1 at large N), record
   correctness + step count + latencies. Write
   `experiments/exp_005_lcp/results/` JSON (latest + timestamped) with env,
   corpus SHA-256[:16], git SHA, params, verbatim criterion evaluation.
4. `scripts/exp005_figure.py`: (a) log-log latency vs N (hash at f=0.5 vs
   brute at f=0.5, with fitted slopes); (b) step count vs N (should be
   ~ceil(log2 N)). Save .png (300 dpi) + .pdf to `figures/`.
5. Record Results/Observations/Interpretation here + findings.md; evaluate
   promotion criterion verbatim; update CLAIMS.md + PROGRAM.md; commit.

### Environment
- **Hardware:** laptop, Intel i9-14900HX (24c/32t hybrid P+E), 64 GB RAM,
  RTX 4090 (idle; CPU-only)
- **Software:** Windows 11 (10.0.26200), Python 3.12.2, hashrope 0.2.2
- **Git commit:** [fill before run — commit bench script first]
- **Seeds:** corpus {42, 43, 44}

### Promotion criterion (verbatim, written before any data)
T3 → **SUPPORTED** iff, across ≥3 corpus seeds × ≥3 independent invocations
(n ≥ 9):

(i)   **[HARD] Correctness:** `lcp_hash` returns the identical divergence
      position as `lcp_brute` for every (N, seed, f) tested — 0 mismatches.
      For synthetic controls with algebraically known LCP = L, both return
      exactly L.
(ii)  **Step-count guard:** `lcp_hash` performs ≤ ceil(log2(min(len_a,
      len_b))) + 1 calls to `rope_substr_hash` per invocation, at every
      (N, f).
(iii) **Wall-clock scaling:** at f=0.5, log-log slope of LCP-hash latency
      vs N ≤ 0.3 across the full size sweep {64K … 8M} (for reference:
      O(log² N) predicts slope ~0.14 over this range; O(N) = 1.0; the
      informal pilot showed ~flat).
(iv)  **Brute-force contrast (descriptive, non-gating):** brute-force
      latency at f=0.5 scales roughly linearly (slope ∈ [0.7, 1.3]),
      confirming the comparison baseline is O(N).
(v)   All numbers reported as mean ± std across runs (n ≥ 9); within-run
      CI retired.

Failure of (i) → experiment FAILS outright. Failure of (ii)/(iii) → T3
stays IN-PROGRESS, investigate (no retrofit). (iv) is descriptive.

**Honesty note:** LCP-hash latency in pure Python includes interpreter
overhead per hash query; the transferable claim is the O(log² N) **step
count** (guard-proven), with a smaller absolute constant in Rust. The
log-log slope test captures the scaling shape even with Python overhead
inflating the intercept.

### Results
[To be filled after run.]

### Observations
[To be filled after run.]

### Interpretation
[To be filled after run.]

### Artifacts
- Implementation: src/lcp.py
- Tests: tests/test_lcp.py
- Bench: scripts/exp005_bench.py
- Figure: scripts/exp005_figure.py
- Results: experiments/exp_005_lcp/results/
