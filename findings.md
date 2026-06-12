# Findings

## Curated Summary

_Living results section, updated as the research picture clarifies. Written in
publication-ready prose, every number with an uncertainty bound and an EXP-XXX
reference._

### Tokenizer-aligned ingestion (S1; EXP-001, CONFIRMATORY)

Snapping rope leaf boundaries to tokenizer-safe split points lets disjoint leaves be
tokenized in parallel while reconstructing the *exact* whole-string token IDs (0 boundary
divergences across byte-BPE, SentencePiece, and WordPiece families; EXP-001 Milestone A).
Given that identity guarantee, end-to-end ingestion (bytes → token IDs) decomposes into two
independent gains, replicated across 3 corpus seeds × 3 independent process invocations
(n=9 runs/cell; error bars are std across runs, since a reproducibility check showed the
within-run bootstrap CI understates run-to-run variance):

- a **size-dependent single-thread gain** from chunking alone (no parallelism), the most stable
  result we have: 1.30×–1.51× across cells with std ≤0.02, growing with context length (gpt2
  1.46×→1.50× from 1→4 MB), because per-leaf tokenization avoids the whole-string tokenizer's
  superlinear cost; and
- a **parallel gain** at 16-way: rayon ~6.6× and multiprocessing ~7.6× for gpt2 at 4 MB
  (mean ± std: rayon 6.62 ± 0.37, mp 7.59 ± 0.41). Multiprocessing also parallelizes the Python
  token-object materialization that bottlenecks rayon's single-process gather, and **reliably
  beats rayon for gpt2** at both 1 MB and 4 MB (paired sign test 9/9, p=0.004) — though the
  margin is modest (~13–16%).

The multiprocessing advantage is regime-dependent: it holds for gpt2 (both sizes) and t5-small
at 4 MB, but **reverses at t5-small 1 MB**, where rayon (6.71 ± 0.26) edges multiprocessing
(6.39 ± 0.30, not significant). This identity-guaranteed, replicated result replaces the prior
draft's retracted 4.12× ingestion claim (which compared pipelines producing different token
streams). _Confirmatory grade (EXP-001 Addendum C). The confirmatory multiprocessing magnitude
(~7.6×) is lower than, and supersedes, an early single-run peak (~10×) attributable to a
cold-thermal artifact — steady-state under sustained load is the reported figure, and is also
what a production server experiences. Small contexts (≤256 KB) are overhead-dominated and
excluded from headline claims._

### Flatten in linear time (S4; EXP-002, CONFIRMATORY)

The structure materializes to a contiguous byte string in linear time with **no hash recomputation**. The prior draft's ~945 ms flatten "tax" (claimed inherent and ~constant in edit count) was an implementation artifact: the old `flatten_context` rebuilt the rope by recursively splitting at midpoints, and every reconstructed leaf recomputed its polynomial hash from scratch — re-hashing Θ(N) bytes. Calling the library's in-order materializer `rope_to_bytes` instead performs **zero** splits, leaf re-allocations, or hash recomputations (operation-count guard, n=9 across 3 corpus seeds × 3 process invocations), giving fixed 0.546 ± 0.016 ms vs the broken path's 437 ± 60 ms at 2 MB. Both paths scale linearly in N (broken log-log slope 1.044 — the cost is redundant re-hashing, not an N log N allocation tree), so the speedup is roughly flat at 730–800× for N ≥ 1 MB. _Confirmatory (EXP-002). The multiplier is Python-interpreter-amplified — the broken arm is bottlenecked on pure-Python per-byte hashing; the language-independent result is the elimination of redundant work (guard-proven), which carries to the Rust implementation with a smaller constant. No library change was required._

### Prefix-reuse identification in O(log² N) (T3; EXP-005, CONFIRMATORY)

The longest common prefix (LCP) of two ropes is found in O(log² N) by binary search on prefix
hashes, each step comparing `rope_substr_hash` values (Theorem 9). The step count — exactly
2·⌈log₂ N⌉, deterministic and content-independent — is the guard-proven complexity bound; wall-clock
latency corroborates it with a log-log slope of **0.028** at f=0.5 across 64 KB–8 MB (pre-registered
criterion ≤ 0.3; O(log² N) predicts ~0.14; O(N) = 1.0). Brute-force byte-by-byte comparison scales
linearly (slope 1.008), and the crossover falls at ~1 MB — precisely where LLM context lengths
become interesting.

At 8 MB, hash-LCP completes in 19 ms regardless of where the divergence falls (f=0.5 or f=0.99),
while brute-force takes 99–200 ms (5–10× slower). Correctness is exact: 0 mismatches vs the
byte-by-byte oracle across 162 checks (9 runs × 6 sizes × 3 fractions). A prefix-dedup workload
(K=10 prompts sharing a 1.2 MB prefix) confirms the practical application: LCP identifies the
shared prefix for 98.5% of pairs at 20.9 ms per pair.

This is the **content-identity query** side of the unification thesis: the same persistent
structure that provides O(log) edits answers “how much prefix do these two contexts share?” in
O(log² N) without materializing or scanning the full text, enabling prefix-dedup for KV-cache
reuse. _Confirmatory (EXP-005). Absolute latency is Python-amplified; the transferable claim is
the O(log² N) step count (guard-proven), with a smaller constant in Rust._

### Branch/snapshot in O(B · log N) memory (M1; EXP-004, CONFIRMATORY)

Forking a rope context for Tree-of-Thought branching costs **O(log w) new nodes per fork** (the
spine from root to the new leaf), with all other nodes structurally shared via immutability
(Invariant I9). The node-count guard is **perfectly deterministic** (std=0 across all 9 runs):
at (N=2M, B=50) the rope arm uses 1,477 unique objects vs 49,927 for deep copies (33.8× sharing
ratio); per-fork new nodes ≈ 10 ≈ ⌈log₂ 489⌉, exactly matching the O(log w) prediction.

In actual memory (tracemalloc), the compression ratio grows with context size: 6.3× at 64 KB →
50.9× at 2 MB → **168.6× at 8 MB** (B=50). At the largest cell (N=8M, B=100), rope forks consume
159 KB vs 25.8 MB for deep copies: **166×** compression with std=0.05 (near-deterministic). The
per-fork rope memory is essentially constant from 1M to 8M (~1.3–1.6 KB/fork; log-log slope
**0.183**), while deep-copy per-fork memory scales linearly (slope **0.884**).

Fork creation time is a bonus result: ~0.037 ms for rope (constant in N) vs 4–17 ms for deepcopy
(linear), yielding **114–454×** speedup. The prior draft’s O(1) memory claim was incorrect; the
honest O(B · log N) is supported, with no library change required. _Confirmatory (EXP-004).
Node-count guard is a mathematical proof (deterministic), not a statistical estimate._

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

### 2026-06-11 — EXP-001 reproducibility check (independent re-run, same seed)

**Key result:** An independent repeat of the MP cells (backfill vs archive 045750Z) shows the
within-run 7-rep bootstrap CIs UNDERESTIMATE cross-run variance. Large-context results reproduce
directionally but not to the digit; small-context cells are unstable.

**Details (cross-run, same corpus seed=42):**
- gpt2 4 MB mp16: 10.43× → 10.97× (disjoint ms CIs, ~6% drift) — both >10×, both beat rayon 6.49×
- gpt2 1 MB mp16: 7.85× → 6.95× (~13% drift, disjoint)
- gpt2 256 KB mp16: 5.42× → 1.36× (4× swing); t5 256 KB mp16: 1.83× → 4.93×
- gpt2 64 KB chunk-only (rayon@1): 1.14× → 0.87× (crosses below 1.0)

**Notes:** The benchmark's real error model is cross-run, not rep-to-rep. Robust claims: direction
(chunking + parallel both help) and ~1-sig-fig large-context magnitudes (rayon ~6.5×, MP ~10× @16,
gpt2 4 MB) and the MP>rayon crossover at large contexts. NOT robust: exact multipliers; all ≤256 KB
cells. Confirmatory experiment upgraded to ≥3 seeds × ≥3 independent invocations, mean ± std across
runs; within-run CI retired as the error bar; ≤64 KB likely dropped from headline claims.

### 2026-06-11 — EXP-001 CONFIRMATORY (n=9: 3 seeds × 3 invocations)

**Key result:** gpt2 ≥1 MB PASSES the pre-committed promotion criterion (LOGBOOK Addendum C).
Final S1 numbers, mean ± std across 9 independent process runs: chunk-only 1.46–1.50× (std ≤0.02),
rayon@16 6.1–6.6×, mp@16 7.0–7.6×; MP>rayon paired sign test **9/9, p=0.004**. t5 4 MB also 9/9;
**t5 1 MB exception** (rayon 6.71 ± 0.26 vs mp 6.39 ± 0.30, 2/9, n.s.).

**Promotion:** S1 SUPPORTED (pilot) → **SUPPORTED (confirmatory)** for gpt2 large-context.

**Details (mean ± std, n=9):** gpt2 1 MB chunk 1.46±0.01 / rayon 6.09±0.53 / mp 7.01±0.72;
gpt2 4 MB chunk 1.50±0.02 / rayon 6.62±0.37 / mp 7.59±0.41; t5 1 MB chunk 1.30±0.02 / rayon
6.71±0.26 / mp 6.39±0.30; t5 4 MB chunk 1.36±0.02 / rayon 6.51±0.33 / mp 7.16±0.40. Corpus
SHA-256[:16] 42=fda6a43a, 43=85ca5870, 44=dfd645be.

**Statistical tests:** paired sign test of mp@16 vs rayon@16 (paired by seed,invocation), exact
two-sided p. Per-seed means all within ~1 std of grand mean (corpus-robust).

**Notes:** Confirmatory mp@16 (~7.6×) SUPERSEDES the pilot single-run peaks (10.43/10.97×) — all 9
confirmatory runs lie in [7.16, 8.42], systematically below the pilot peaks; hypothesized as a
cold-thermal artifact in the pilot vs sustained-load steady state in the confirmatory (16-core
workloads throttle harder than the 1-core serial baseline; not instrumented). Effect size modest
(~13–16% MP-over-rayon margin) but reliable (9/9). chunk-only fully reproducible (std ≤0.02).
Design caveat: M(mp)-after-P(rayon) ordering is conservative for MP; mp wins gpt2 anyway.
Independence caveat: single session; a cross-day repeat would harden the error bars (optional,
non-blocking). This is the third downward magnitude correction (after de-tiling and the mp32
cherry-pick exclusion); each increased reproducibility.

### 2026-06-11 — EXP-002: flatten fix (in-order materialization vs midpoint re-split, CONFIRMATORY)

**Key result:** The ~945 ms flatten "tax" (S4) is an implementation artifact — redundant per-leaf hash
recomputation from midpoint re-splitting — not an inherent O(N) serialization cost. The library's in-order
`rope_to_bytes` materializes with **0** splits / **0** leaf re-allocs / **0** hash recomputations
(operation-count guard), passing the pre-registered LOGBOOK criterion. n=9 (3 seeds × 3 invocations), real
corpus (same as EXP-001), commit 47ea627, hashrope 0.2.2.

**Promotion:** S4 REFRAMED → **SUPPORTED**.

**Details (mean ± std, n=9):** fixed vs broken ms / speedup — 1M: 0.272±0.016 vs 199.18±4.48 / 735×;
2M: 0.546±0.016 vs 437.36±59.97 / 800×; 4M: 1.166±0.053 vs 865.01±104.66 / 741×; 8M: 2.305±0.092 vs
1692.57±167.06 / 734×. Guard (broken Leaf/split/hash ; fixed) — 2M 1022/511/1022, 8M 4092/2047/4092;
fixed 0/0/0 at every size. Byte-identity fixed==original==broken, 0 mismatches (HARD gate). Log-log slope
broken 1.044 (linear, not N log N); fixed 1.034 over ≥1M (full-sweep 1.327 is small-N timer-floor inflation;
O(N)-fixed rests on the guard per the pre-committed decision). Corpus SHA-256[:16] 42=fda6a43a, 43=85ca5870,
44=dfd645be.

**Statistical tests:** Verdict gates on (i) byte-identity [HARD], (ii) guard, (iii) wall-clock — all PASS;
(iv) scaling descriptive, (v) mean±std. Per-seed speedup means within ~10% across seeds (corpus-robust).

**Notes:** The ~730–800× multiplier is Python-amplified (pure-Python per-byte hashing in the broken arm); the
transferable claim is the elimination of redundant work (guard-proven), smaller constant expected in Rust
(confirmation deferred). Absolute broken latency (437 ms @2M) differs from the historical 945 ms (different
hardware/run); the mechanism and its removal are the claim, not a specific ms. Variance is asymmetric (fixed
CV 3–6%, broken CV 10–14% at large N — allocation-churn sensitivity the cross-run error model captures);
verdict robust (worst cell mean−1σ ≈ 700× ≫ 100×). No library change required. EXP-002 closed.

### 2026-06-12 — EXP-005: LCP via prefix-hash binary search (CONFIRMATORY)

**Key result:** The longest common prefix (LCP) of two ropes is found in O(log² N) by binary
search on prefix hashes via `rope_substr_hash` (Theorem 9). Step-count guard: exactly
2·⌈log₂ N⌉ calls, content-independent and deterministic. Wall-clock log-log slope **0.028** at
f=0.5 across 64 KB–8 MB (criterion ≤ 0.3; O(log² N) predicts ~0.14; O(N) = 1.0). Correctness:
exact match with byte-by-byte oracle, 0 mismatches across 162 checks. n=9 (3 seeds × 3
invocations), commit a15e883, hashrope 0.2.2.

**Promotion:** T3 IN-PROGRESS → **SUPPORTED**.

**Details (mean ± std ms, n=9 @ f=0.5):** 64K 16.65±0.36 / 256K 8.58±0.28 / 1M 8.56±0.27 /
2M 9.56±0.30 / 4M 12.82±0.40 / 8M 19.03±0.64 (hash); brute: 64K 0.76±0.02 / 256K 3.13±0.09 /
1M 12.13±0.35 / 2M 25.04±0.63 / 4M 49.16±1.53 / 8M 99.42±3.32. Log-log slopes: hash 0.028,
brute 1.008 (textbook linear). Step counts deterministic: 30/34/38/40/42/44 (= 2·⌈log₂ N⌉).
Prefix-dedup workload (descriptive): K=10 prompts, prefix=1.2M bytes, hit_rate=0.985,
per_pair=20.9 ms. Corpus SHA-256[:16] 42=fda6a43a, 43=85ca5870, 44=dfd645be.

**Notes:** Hash LCP latency is essentially flat from 256 KB to 8 MB (~8.6–19 ms at f=0.5); the
log-log slope 0.028 is below even the O(log² N) theoretical prediction of ~0.14 — Python
interpreter overhead per binary-search step is roughly constant regardless of tree depth,
masking the log factor. Crossover vs brute at ~1 MB (f=0.5: hash 8.56 ms vs brute 12.13 ms);
at 8 MB f=0.99: hash 19.45 ms vs brute 200.33 ms (10.3×). 64K f=0.5 anomaly (16.65 ms >
256K’s 8.58 ms) is a cold-cache effect consistent with EXP-002’s timer-floor pattern. f=0.0 is
brute’s domain (brute ~0 ms; hash ~7–10 ms) — expected, since hash-LCP is O(log² N) regardless
of f, useful when prefixes are long. Dedup hit_rate=0.985 (not 1.0) because one suffix pair
shared initial bytes beyond the intended prefix boundary. Honesty note: absolute latency is
Python-amplified; the transferable claim is the O(log² N) step count (guard-proven), with a
smaller constant in Rust. This is the **content-identity query** side of the unification thesis:
the same persistent rope that provides O(log) edits answers “how much prefix do these two
contexts share?” in O(log² N). EXP-005 closed.

### 2026-06-12 — EXP-004: Branch/snapshot memory under ToT branching (CONFIRMATORY)

**Key result:** Hashrope's immutable structural sharing gives **O(B · log N) incremental memory**
for B ToT-style forks at context size N, vs O(B · N) for deep copies. Node-count guard is
**perfectly deterministic** (std=0 across all 9 runs). At (N=8M, B=100): rope 159 KB vs deepcopy
25.8 MB = **166×** compression. Prior O(1) claim retracted; O(B · log N) supported.

**Promotion:** M1 REFRAMED → **SUPPORTED**.

**Details (n=9, mean ± std):** Node sharing ratio: 6.0× (64K) → 33.8× (2M) → 44.7× (8M) at B=50;
78.9× at (8M, B=100). Per-fork rope memory: 691 B (64K) → 1,630 B (8M) — essentially constant;
log-log slope 0.183. Deepcopy per-fork: 4.4 KB → 275 KB; slope 0.884 (linear). tracemalloc
compression: 6.3× (64K) → 168.6× (8M) at B=50. Fork timing (descriptive): rope 0.037 ms
(constant), deepcopy 4–17 ms (linear), 114–454× speedup. Byte-identity 0 mismatches. Commit
a19a4c7, corpus SHA-256[:16] 42=fda6a43a, 43=85ca5870, 44=dfd645be.

**Notes:** Node counts are deterministic because structural sharing is a mathematical property of
the immutable tree, not a statistical measurement — the experiment is effectively a proof.
tracemalloc std <0.1% relative (memory allocation is near-deterministic for frozen dataclasses).
Compression ratio grows with N (the O(log N) vs O(N) gap widens) and converges at large B
(both arms scale linearly in B). No library change required. EXP-004 closed.
