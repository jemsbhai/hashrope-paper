# Experimental Program

The forward-looking registry of every experiment for the hashrope debut paper.
This file is the **roadmap**; `CLAIMS.md` is the authoritative claim ledger
(every experiment here earns one or more claim rows there); `LOGBOOK.md` is the
append-only execution record (the full plan for each experiment is written there
**at its kickoff**, before any code runs); `findings.md` carries results.

## Phase directive (current)

- **All experimental work is completed BEFORE any paper prose / LaTeX is written.**
  No writing in `paper/` until this program is done.
- **One experiment at a time.** The user picks the next experiment; Claude writes
  its LOGBOOK plan, gets sign-off, executes, validates, records, commits, and
  **waits**. Numbering is assigned in order; a prior-draft figure does not count
  as a logged experiment.
- This program **solidifies** (re-measures honestly), **adds**, **ablates**, and
  **expands** the prior work; every retained prior measurement is replaced by a
  rigorous EXP-XXX with a baseline and an uncertainty bound.

## Global error model & protocol (applies to every EXP)

Established empirically in EXP-001 (Addendum B-repro): for these benchmarks the
real uncertainty is **cross-run / cross-invocation**, not the within-run
bootstrap CI.

- **Confirmatory grade** = **≥3 corpus/workload seeds × ≥3 independent process
  invocations** (fresh interpreter/process), reported as **mean ± std across
  runs** (the within-run CI is retired as the error bar), plus a **paired test**
  (sign test / Wilcoxon) and an **effect size**.
- **Plan before data:** hypothesis + a **verbatim promotion criterion** are
  written in LOGBOOK before the run; the criterion is evaluated verbatim after.
  Never retrofit.
- **Integrity:** real data alongside synthetic controls; honest negatives
  reported, never dropped; identical algorithms across cross-language
  comparisons; report tight-CI numbers, never noisy maxima.
- **Phasing:** exploratory → pilot (point estimates + CIs) → confirmatory. Only
  confirmatory numbers enter the paper.
- **Files:** full rewrites or surgical edits only — never offset/append writes.
- **Hardware note:** the laptop is an Intel Raptor Lake hybrid part (32 logical
  threads, P+E), 64 GB RAM, RTX 4090, Windows + PowerShell. It throttles under
  sustained load — measurements must use the cross-run error model and a fixed
  power plan. Colab Pro (A100/H100) is reserved for the GPU energy stage.

## Master table

Legend — **Mode:** Sol = solidify, Add = new, Abl = ablation, Exp = expand.
**HW:** local-CPU / local-Rust / GPU-Colab.

| EXP | Claim(s) | Title | Mode | HW | Status | Key dependency / note |
|-----|----------|-------|------|----|--------|-----------------------|
| 001 | S1 (+C3 partial) | Tokenizer-aligned fat-leaf chunking: token-identity gate + ingestion speedup | Sol+Exp | local-CPU | **DONE (confirmatory)** | gpt2 ≥1 MB passed; t5 1 MB honest exception. Closed. |
| 002 | S4 | Flatten fix (in-order leaf walk + `b"".join`) vs broken O(N log N) re-split | Sol+Exp | local-CPU + Rust | **DONE (confirmatory)** | EXP-002 (2026-06-11, commit 47ea627): n=9 (3 seeds × 3 inv), real corpus. Guard: fixed `rope_to_bytes` = 0/0/0; broken = Θ(#leaves) re-hashing Θ(N) bytes. **Broken is empirically O(N) (slope 1.044), NOT N log N** — cost is redundant re-hashing, not tree depth. Byte-identity 0 mismatches; speedup 730–800× (N≥1M); fixed slope 1.034 (≥1M). ~500× expectation exceeded (Python-amplified; transferable claim = work elimination, smaller constant in Rust — confirmation deferred). S4 → SUPPORTED. Closed. |
| 003 | L1, L2 | Incremental-edit (mutation) latency + micro-latency crossover; kill the ~5× sub-threshold UTF-8 round-trip gap | Sol | local-CPU | planned | Needs the C3 fix (EXP-013). **Produces the canonical mutation workload reused by EXP-016** (ms/op + J/op on the same runs). |
| 004 | M1 | Branch/snapshot: peak memory under ToT branching (honest O(B·log N), not O(1)) + time-travel rollback cost | Sol+Exp | local-CPU | **DONE (confirmatory)** | EXP-004 (2026-06-12, commit a19a4c7): n=9, real corpus. Node-count guard DETERMINISTIC (std=0); sharing ratio 33.8× @(N=2M,B=50). tracemalloc compression 166× @(N=8M,B=100). Rope slope 0.183 (O(log N)); deepcopy 0.884 (O(N)). Fork time 0.037 ms vs 17 ms (454×). M1 → SUPPORTED. Closed. |
| 005 | T3 | Prefix-reuse identification: LCP O(log² N) multi-seed timing table (CIs) + prefix-dedup hit-rate | Add | local-CPU | **DONE (confirmatory)** | EXP-005 (2026-06-12, commit a15e883): n=9 (3 seeds × 3 inv), real corpus. Step-count guard 2·⌈log₂ N⌉, 0 violations. Hash slope 0.028 at f=0.5 (≤ 0.3). Correctness 0 mismatches. Crossover vs brute ~1 MB. **T3 → SUPPORTED.** Closed. |
| 006 | T4 | Repetition: RepeatNode O(log q) compression/throughput vs naïve materialization | Sol+Exp | local-CPU | **DONE (confirmatory)** | EXP-006 (2026-06-12, commit 23e8eb6): n=9, real corpus. Node-count guard DETERMINISTIC (std=0); compression 9,999.5× @(q=10000, unit=4KB). Construction slope: repeat 0.175 (O(log q)), naïve 1.041 (O(q)). Speedup 1,033,724× @q=10000. Memory 327 B vs 44 MB. T4 evidence enriched. Closed. |
| 007 | L3 (RETRACTED) | Request throughput under a real arrival process / concurrency at extreme N | Sol | local-CPU | planned | Honest re-measure; report even where native wins. Prior "concurrent" run was a serial loop. |
| 008 | S5 | GC tail-latency p99 with replicates + a defined p99 protocol | Sol | local-CPU | planned | Low-risk; currently single-shot. |
| 009 | L4 | Agentic workload: trace-driven replay + ReAct loop + **list-of-token-tensors cache baseline** + end-to-end generation | Sol+Exp | local-CPU (+opt GPU) | planned | Baseline isolates rope's distinct value (mid-context edits + verified prefix identity) from plain delta tokenization. |
| 010 | S2 (+leaf-size) | Leaf-size ablation (one variable) + peak-VRAM / OOM curve | Abl | local + GPU | planned | Also supports the "structurally congruent with paged layouts" reframing of S2. |
| 011 | C1 | Cross-language byte-identical hash vectors as a reproducible artifact-eval claim | Add | local-CPU + Rust | planned | Python ↔ Rust hash equality on a frozen vector set. |
| 012 | C2 | Hash security: keyed / verify-on-match hashing + collision & adversarial-forgeability analysis | Add | local; **touches lib** | planned | Best-paper differentiator. Current poly hash is forgeable with public params. |
| 013 | C3 | Define + test HybridContext's byte/char index contract (no UTF-8 bisection) | Add | local-CPU | planned | **Pairs with EXP-003** (the UTF-8 round-trip fix is the same code path). |
| 014 | T5 | Lazy vs eager construction cost (Theorem 44), Rust 0.3.0, on real workloads | Add | local-Rust | planned | Isolate O(t·log t) lazy vs O(k·c_F·t) eager construction. |
| 015 | S3 | **HEADLINE ENERGY:** real serving stack (vLLM/SGLang) + 7–8B model — does hashrope's O(log² N) prefix identification → KV reuse → lower TTFT / higher throughput / lower **energy**, **isolated** from the framework's built-in prefix caching | Add | **GPU-Colab (A100/H100)** | planned | **DESIGN CONFER BEFORE ANY CELLS.** Interactive, one cell at a time. Isolation methodology is the whole ballgame (the old suite's failure was unfalsifiability). |
| 016 | S6 | **SUPPORTING ENERGY:** CPU package (+DRAM if exposed) joules per mutation via Intel RAPL, read with LibreHardwareMonitor — hashrope O(log N) edit vs O(N) contiguous copy | Add | local-CPU | planned | Grounds the memory-bus motivation. **Reuses the EXP-003 workload.** Fold in the user's prior LibreHardwareMonitor settling/warmup protocol. LibreHardwareMonitor (admin/MSR) — confirm install at kickoff; WSL2 cannot read host RAPL. |
| 017 | B1 | **COMPETITIVE Leg 3:** prefix-identification — hashrope LCP vs SGLang RadixCache v0.1.17 (vendored byte-identical) + numpy flat C-speed floor, on controlled-L corpus sweep (1k–2M tokens) + ShareGPT/LMSYS real pairs (co-primary) + one-vs-many K-sweep | Add | local-CPU | **DONE (confirmatory)** | B1 SUPPORTED. Crossover L*≈571k tokens; 4.63× at L=2M (9/9 paired). Grid extended from 1M to 2M after initial run revealed crossover higher than pre-registered estimate (128k–256k). Criterion (iv) checkpoint revised from (512k,1M) to (2M). Honest negatives: radix wins real pairs and K-sweep. |
| 018 | B1 (ext) | **COMPETITIVE Leg 3+1:** serving-loop replay — per-request radix match+insert vs hashrope append+LCP on ShareGPT/LMSYS conversation streams | Add | local-CPU | planned | Framing C; shares the EXP-017 harness. Bundles incremental edit (Leg 1) with identification (Leg 3) — the unification systems number. Design confer at kickoff. |
| 019 | B3 | **COMPETITIVE Leg 2:** branch/snapshot — hashrope vs faithful PagedAttention block-table COW (Kwon et al. SOSP'23; structure §4.2, COW §4.4) on ToT branching: controlled context-size sweep (1k–1M tokens, **PRIMARY/headline**) + real gpt-oss-120b Game-of-24 traces (ecological) | Add | local-CPU | **DONE (Milestone A + B)** | EXP-019 Milestone A (2026-06-13, commit 5848110): n=9, real corpus. **B3 SUPPORTED (Milestone A).** Gated = branch-creation (Addendum A; bare fork has no crossover). Crossover N*=16k tok @ block 16; at N=1M all blocks 9/9 paired (p=0.004), branch-creation 169×/97×/43×/20× (b8/16/32/64), branching-memory compression 613×/306×/153×/68× (5 branches). Byte-identity 0 mismatches; guards deterministic. Honest boundaries: append PagedAttention wins ~18–35×; bare fork hashrope O(1) ~39 ns flat. Baseline `src/paged_attention_cow.py` (55 unit tests). **Milestone B** (real gpt-oss-120b Game-of-24 ToT traces, recorded+replayed) **CLOSED 2026-06-16**: n=9 ecological corroboration -- 0 mismatches; branch-creation 9/9 every block, 74.6x/38.9x/19.1x/9.6x (b8/16/32/64) at P=256k; marginal memory compression 17x-67x for {8,16,32} (b64 a disclosed 9.6x near-miss, not retrofitted); crossover N*=16k confirmed, block-dependent crossovers characterized. LOGBOOK Milestone B Results + 2026-06-16 (iii) clarification logged. EXP-019 closed. |
| 020 | B2 | **COMPETITIVE Leg 1:** incremental edit -- hashrope (Rust) vs Ropey 1.6.1 (vendored dev-dep, pinned + checksum) on ASCII buffers (1k-1M chars): churn-edit latency (regime A) + edit-with-content-identity (regime B, **headline**) | Add | local-Rust | **in-progress** | EXP-020 kickoff 2026-06-16. Rust-to-Rust; standalone (decoupled from EXP-003, spec reusable). Claim: same O(log N) edit class at bounded constant cost WHILE maintaining a verifiable whole-buffer fingerprint Ropey lacks; net-superior on edit+identity (hashrope O(1) maintained vs Ropey O(N) recompute, same PolynomialHash). Verbatim criterion in LOGBOOK: (i) HARD byte-identity + maintained-hash-vs-recompute 0 mismatches, (ii) HARD edit slope <=0.3, (iii) HARD regime-B r=1 9/9 + >=2x at N=1M monotone, (iv) edit-only constant C<=30x framing rule (else narrow to iii), (v) honest negatives incl persistent-arena memory growth vs in-place. Non-lazy arena (fingerprint cost included); arena-drop excluded (iter_batched PerIteration); ASCII char==byte; power-cache warmed. Red-first = correctness test (compile-red until ropey dev-dep added). |

## The energy narrative (two committed experiments, fixed hierarchy)

- **Headline = EXP-015 (S3), datacenter GPU energy, on Colab A100/H100.** The most
  impactful claim: hashrope *identifies* reusable prefixes in O(log² N) →
  the serving stack skips redundant forward passes → measurable end-to-end
  TTFT / throughput / **energy (J)** win, with hashrope's contribution **isolated
  from the framework's own prefix caching**. This replaces the prior draft's
  faked 62.7% "thermodynamics" figure (which never invoked hashrope; see
  `CLAIMS.md` S3 for the code+data provenance).
- **Supporting = EXP-016 (S6), CPU / memory-bus energy, on the laptop.** A
  **relative, paired** joules-per-mutation result (O(log N) edit vs O(N) copy)
  that turns the opening memory-bus physics argument (DRAM ≈ 15 pJ/bit ≫
  ALU ≈ 0.1 pJ/bit) from rhetoric into a measured number. **Not** a
  datacenter-energy claim — that is EXP-015's job alone. CPU-local, not Colab.

## Sequencing notes

- **Theory-led spine = Group {003, 004, 005, 006}** (the four context mechanisms
  — incremental edit, branch/snapshot, prefix-reuse identification, repetition —
  as one structure). EXP-005 is the conceptual heart and the on-ramp to EXP-015.
- **EXP-003 ↔ EXP-013** are coupled (shared UTF-8 code path) — do them adjacent.
- **EXP-003 → EXP-016**: EXP-003 produces the mutation workload EXP-016 reuses, so
  EXP-003 should precede EXP-016.
- **EXP-015 is the long pole and highest-risk.** Regardless of run order, do its
  **design confer early** so the isolation methodology is settled before cells.
- **Recommended first step: EXP-002** — local, fast, low-risk, converts a prior
  weakness into a ~500× strength, and re-establishes the
  plan→red→run→record→commit cadence. (Alternative substance-first start: EXP-005.)
- **Competitive-baseline layer (EXP-017+, started 2026-06-12):** head-to-head
  vs SOTA per-leg specialists on canonical workloads, layered ON TOP of the
  closed mechanism experiments (EXP-001…006 stay clean; no retrofitting).
  Order: EXP-017 (SGLang radix, Leg 3) → EXP-018 (serving-loop replay) →
  Ropey (Leg 1, Rust workstream, couples to EXP-003) → PagedAttention COW
  (Leg 2, Game of 24). Design principle: actual published implementations
  where extractable (vendored verbatim with provenance), faithful cited
  reimplementations otherwise — never toy baselines. Leg 4 (RepeatNode) has
  no direct SOTA competitor; the naive-materialization baseline (EXP-006)
  stands.

## Status key

`planned` → `in-progress` → `done`. Update this table and the matching `CLAIMS.md`
row whenever an experiment changes state; commit the change with the logbook
update.
