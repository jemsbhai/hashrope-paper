# Claims Ledger

**The most important file in this repository.** Every empirical or theoretical
claim that appears in the paper must have a row here, and every row must name
the experiment (EXP-XXX) and data artifact that supports it. No claim ships in
the paper without a `SUPPORTED` row backed by a logged, reproducible experiment.

This ledger is maintained in lockstep with the paper. A claim cannot move to
`SUPPORTED` on assertion; it requires a logbook entry, a results file, and (for
comparative claims) a baseline and uncertainty bound.

The forward-looking experimental program (EXP-002 … EXP-016) — each experiment
mapped to the claim(s) it earns, with mode, hardware, status, and sequencing — is
registered in `PROGRAM.md`. Full per-experiment plans (hypothesis, IVs/DVs,
protocol, verbatim promotion criterion) are written into `LOGBOOK.md` at each
experiment's kickoff, before any code runs.

## Status vocabulary

- **SUPPORTED** — backed by a completed EXP-XXX with logged data, baseline, and
  uncertainty. Safe to state in the paper as written.
- **IN-PROGRESS** — experiment designed/running; claim provisional.
- **RETRACTED** — was claimed in the prior draft (`old/`) but the audit found it
  false, unsupported, or measuring something other than what it states. Must be
  removed from or rewritten in the paper. Kept here for traceability.
- **REFRAMED** — the underlying effect is real but the prior framing overclaimed
  or misattributed it; the row records the honest restatement we will support.
- **PLANNED** — intended claim with no experiment yet.

---

## Theoretical claims

| ID | Claim | Status | Evidence / Notes |
|----|-------|--------|------------------|
| T1 | `concat`, `split` are O(log w) on a BB[2/7]-balanced rope | SUPPORTED | Library + balance-termination proof (`BALANCE_TERMINATION_PROOF.md`, 9.2M-combination sweep). Carries over from the data-structures work. |
| T2 | `substr_hash` over a range is O(k·log w), not O(range size) (Theorem 9) | SUPPORTED | Python fix shipped 0.2.2; Rust had it since 0.2.2. Guard test `test_substr_hash_theorem9.py` (CountingHash) proves stored hashes are consulted. Bench: half-range query ~120ms→~1ms @500KB. |
| T3 | LCP via prefix-hash binary search is O(log² N) | IN-PROGRESS | Correctness verified (exact divergence byte on 50+ random points). Size-independence demonstrated (~28ms flat 500KB→16MB). Needs: formal write-up + multi-seed timing table with CIs as its own EXP. |
| T4 | RepeatNode encodes q repetitions with O(log q) hash via geometric accumulator φ | SUPPORTED | Library; φ is O(log q) doubling (theory proven). Empirical compression/throughput benchmark — RepeatNode vs naïve materialization on repetition-heavy workloads (repeated text blocks, log/JSON payloads, synthetic q-repetition) — to be (re-)established under protocol as **EXP-006**. |
| T5 | Lazy materialization gives O(t·log t) construction vs O(k·c_F·t) eager (Theorem 44) | PLANNED | Shipped in Rust 0.3.0. Claim needs an EXP isolating construction cost lazy-vs-eager on real workloads. |

## Asymptotic / memory claims

| ID | Claim | Status | Evidence / Notes |
|----|-------|--------|------------------|
| M1 | Structural sharing yields sub-linear peak memory under branching | REFRAMED | Prior draft said **O(1)** peak memory under ToT branching. Own data (`3_macro_memory_ram_mb.csv`) shows growth with branch count B (0.021→0.396 MB). Honest claim: **O(B·log N) incremental**, vs O(N·B) deep-copy. Still a strong result (486MB→0.40MB @50 forks/10M chars); just not O(1). |

## Latency / throughput claims

| ID | Claim | Status | Evidence / Notes |
|----|-------|--------|------------------|
| L1 | Mutation latency reduced >20× at extreme scale (≥50M chars) | IN-PROGRESS | Prior draft claimed >20×. Plausible but never backed by a multi-seed run with CIs; per-op rope cost swings 13× across the old CSVs with no warmup protocol. Re-measure cleanly (warm/cold split, replicates) before stating a multiplier. |
| L2 | Micro-latency hybrid tracks C-backend below threshold, detaches at 1MB | IN-PROGRESS | `1_micro_latency_crossover.csv` shows the crossover but also a persistent ~5× sub-threshold gap (UTF-8 decode/encode round-trip in HybridContext). Fix the round-trip, then the "tracks flawlessly" claim becomes defensible. |
| L3 | >10× request throughput vs contiguous strings | RETRACTED | `2_macro_throughput.csv`: native ~920 RPS vs hybrid ~420 at the tested 5M size — **native wins 2.2×**. Experiment is a serial loop labeled "concurrent." Must re-run with a real arrival process/concurrency at larger contexts, or drop the claim. |
| L4 | ReAct-loop per-step latency 470ms→0.6ms | REFRAMED | Real effect but mostly delta tokenization, which a list-of-token-tensors baseline also gets. Rope's distinct value = mid-context edits + verified prefix identity. Redesign EXP so that is what's measured; add token-cache baseline. |

## Systems / GPU claims

| ID | Claim | Status | Evidence / Notes |
|----|-------|--------|------------------|
| S1 | Parallel per-leaf tokenization of tokenizer-aligned leaves reproduces the whole-string token stream and accelerates ingestion | SUPPORTED (confirmatory) | **Correctness gate DONE (EXP-001, 2026-06-10):** the tokenizer-aligned chunker (`src/tokenizer_aligned.py`) makes leaf-wise tokenization EXACTLY equal whole-string tokenization — proven for byte-level BPE (gpt2) AND SentencePiece (t5-small) by `tests/test_tokenizer_aligned.py` (red-first, 20/20), with 0 invalid-UTF-8 leaves and an identity-safe fallback. This retires the original flaw (prior pipeline diverged at 197/244 boundaries and emitted a padded 2D batch never recombined). **Speedup DONE at pilot grade (EXP-001 Addendum B, 2026-06-11):** non-tiled corpus, gpt2+t5, 7 reps, 95% CI. Decomposes into a free single-thread chunking gain (gpt2 1.14×→1.52× as ctx 64KB→4MB) + parallel scaling to ~6.5× (16-way rayon; gpt2 4MB 6.49× [6.4,6.7]) or up to 10.43× [10.3,10.5] (16-way multiprocessing, large ctx). Clean CI-separated rayon/MP crossover (rayon wins ≤256KB, MP wins ≥1MB). Corrected figures `figures/exp001_speedup_vs_cores.{png,pdf}` + `exp001_rayon_vs_mp.{png,pdf}` replace `old/hashrope_gpu_ingestion.pdf`. **Confirmatory pending:** ≥3 corpus-shuffle seeds (currently single seed=42), producing git SHA, and a profile of the MP>rayon materialization mechanism before paper-final. (The earlier ~2–3× expectation was itself an underestimate; the noisy 10.99× mp32 max was excluded — see Addendum B.) **Reproducibility re-run (Addendum B-repro, 2026-06-11):** an independent same-seed repeat showed the within-run 7-rep CIs understate cross-run variance (~6–25% drift at large contexts, 2–4× swings ≤256 KB; gpt2 64 KB chunk-only crossed below 1.0). Only the **direction** and **~1-sig-fig large-context magnitudes** (rayon ~6.5×, MP ~10× @16, gpt2 4 MB) and the large-context MP>rayon crossover are robust; ≤256 KB cells excluded; confirmatory upgraded to ≥3 seeds × ≥3 independent invocations reporting mean±std across runs (within-run CI retired as the error bar). **CONFIRMATORY DONE (Addendum C, 2026-06-11):** 3 seeds x 3 invocations (n=9/cell), mean +/- std across runs, paired sign test. gpt2 PASSES the pre-committed criterion at >=1 MB: chunk 1.46-1.50x (std <=0.02), rayon@16 6.1-6.6x, mp@16 7.0-7.6x, MP>rayon **9/9 p=0.004** (4 MB: rayon 6.62+/-0.37, mp 7.59+/-0.41). t5 4 MB confirms MP>rayon (9/9); **t5 1 MB is an honest exception** (rayon 6.71+/-0.26 vs mp 6.39+/-0.30, 2/9, n.s.). KEY CORRECTION: confirmatory mp@16 (~7.6x) is LOWER than the pilot single-run peak (10.43/10.97x) -- pilot caught a cold-thermal peak; steady-state under sustained load is ~7.6x and supersedes it. MP's margin over rayon is modest (~13-16%) but reliable. **Final S1 statement = ~1.5x free single-thread chunking + ~6.6x rayon / ~7.6x mp at 16-way (gpt2 4 MB), provably-identical token stream.** |
| S2 | 4KB fat leaves are "physically isomorphic" to PagedAttention VRAM blocks | REFRAMED | Overclaim. 4KB byte chunks are not KV-block topology. Honest: "structurally congruent with paged layouts." |
| S3 | Application-layer prefix identification enables KV-cache reuse → lower TTFT / higher throughput / energy savings | REFRAMED (→ **headline energy claim, EXP-015**) | **Verified at code+data level (2026-06-11):** the prior "thermodynamics" suite (`run_a100_energy_benchmark`, `old/hashrope1_exmain.py` L1685; CSVs 14–17) compares a *Native* arm that recomputes a hardcoded ~560-tok shared prefix 64× against a *Hashrope* arm that computes it once + a plain torch KV broadcast (`broadcast_kv_cache` = tensor `.expand`, **not** a hashrope call); the *Hybrid* arm is the Hashrope arm + `time.sleep(0.001)` (L2263). **Hashrope is never invoked; the prefix is assumed, not identified.** The 62.7% "energy reduction" (`17_a100_energy_totals.csv`: 226.52 J→84.43 J) is the vLLM/SGLang prefix-caching effect on GPT-2-124M — single-run, sub-second, GPU-ramp-dominated, time-confounded (0.86 s vs 0.51 s) — misattributed to hashrope. **Honest headline (EXP-015):** hashrope *identifies* reusable prefixes in O(log² N); measure end-to-end TTFT / throughput / **energy (J)** on a real serving stack (vLLM or SGLang) + a 7–8B model on **Colab A100/H100**, isolating hashrope's contribution from the framework's built-in prefix caching. **Design confer before any cells** — isolation is the whole ballgame. |
| S4 | The prior flatten "tax" is an implementation artifact (midpoint re-splitting → Θ(#leaves) `Leaf` re-allocations re-hashing Θ(N) bytes), **not** an inherent serialization cost; in-order materialization (`rope_to_bytes`) eliminates it with **0** splits / **0** re-allocs / **0** hash recomputations | SUPPORTED | **EXP-002 (2026-06-11, commit 47ea627), confirmatory n=9 (3 seeds × 3 invocations), real corpus, hashrope 0.2.2.** Op-count guard: fixed `rope_to_bytes` = 0/0/0 at every size; broken midpoint-resplit = Θ(#leaves) (2M 1022 allocs/511 splits/1022 hashes; 8M 4092/2047/4092), re-hashing Θ(N) bytes — the redundant work (cProfile: ~99% in `PolynomialHash.hash`) is removed. Byte-identity fixed==original==broken, **0 mismatches** (HARD gate). Wall-clock @2M: fixed 0.546 ± 0.016 ms, speedup **800×** (≥100× at every size ≥256 KB; 730–800× for N≥1M, ±~10% across seeds). Scaling: broken log-log slope **1.044** (linear — **not** N log N); fixed 1.034 over ≥1M (O(N), guard-backed; full-sweep 1.327 is small-N timer-floor inflation). Results: `experiments/exp_002_flatten/results/exp002_flatten_latest.json`. **Caveat (carried to paper):** the ~730–800× multiplier is Python-interpreter-amplified (pure-Python per-byte hashing in the broken arm); the language-independent claim is the *elimination of redundant work* (guard-proven), with a smaller constant expected in Rust (confirmation deferred). Promoted from REFRAMED on the pre-registered LOGBOOK EXP-002 criterion. |
| S5 | Pure-functional 4KB allocations avoid stop-the-world GC pauses (p99) | IN-PROGRESS | `6_gc_tail_latency.csv` exists; needs re-run with replicates and a defined p99 protocol. Low-risk but currently single-shot. → **EXP-008**. |
| S6 | hashrope's O(log N) edit draws less CPU+DRAM energy per mutation than an O(N) contiguous copy | PLANNED | **EXP-016** (supporting; grounds the memory-bus motivation — DRAM ≈ 15 pJ/bit ≫ ALU ≈ 0.1 pJ/bit). **Relative, paired** joules/op via Intel RAPL (package domain; + DRAM domain if the part exposes it), read on Windows via LibreHardwareMonitor (admin/MSR). Laptop CPU throttles → cross-run error model (≥3 invocations, mean±std), interleaved paired design within one thermal envelope, fixed power plan, warmup discarded (user's prior settling/warmup protocol folded in at kickoff). Reuses the EXP-003 mutation workload (joules/op reported alongside ms/op). **NOT** a datacenter-energy claim — absolute datacenter energy is **S3/EXP-015** only; CPU-local, not Colab. |

## Correctness / safety claims (best-paper differentiators)

| ID | Claim | Status | Evidence / Notes |
|----|-------|--------|------------------|
| C1 | Hash values are byte-identical across the Python and Rust implementations | IN-PROGRESS | Cross-language consistency tests exist in both crates. Promote to an explicit artifact-evaluation claim with a reproducible cross-impl vector. |
| C2 | Polynomial hashing is collision-safe for routing under the stated model | PLANNED | Mersenne-prime polynomial hashes are **adversarially forgeable** with public params — a cache-poisoning vector in a multi-tenant router. Need: keyed hashing per tenant or verify-on-match, plus a collision-probability paragraph. This is a credibility item, not optional. |
| C3 | Indices are handled safely for multibyte (UTF-8) content | PLANNED | Current HybridContext conflates byte and character indices; a split can bisect a UTF-8 sequence. Define and test the byte-index contract (or add char-aware handling). **Partial progress (EXP-001):** the ingestion chunker is UTF-8-safe by construction (cuts on char boundaries; `test_aligned_never_bisects_utf8` green for both tokenizer families) — but HybridContext's own index contract is a separate code path, still open. |

---

## Audit provenance

Rows marked RETRACTED/REFRAMED come from the technical audit of the prior draft
(`old/hashrope-paper-outline.md` + the four CSV suites + two notebooks),
conducted 2026-06-09, cross-checked against the live `hashrope` library code.
The four claims directly contradicted by their own shipped CSV data: **L3, S1**
(false as stated) and **M1, L4** (real effect, wrong framing/attribution).

## Working rule

When drafting any sentence in the paper that asserts a number, a complexity, or
a comparison, find its row here first. If there is no `SUPPORTED` row, the
sentence does not go in — either run the experiment that earns the row, or cut
the sentence.
