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
| 004 | M1 | Branch/snapshot: peak memory under ToT branching (honest O(B·log N), not O(1)) + time-travel rollback cost | Sol+Exp | local-CPU | planned | Structural-sharing measurement vs deep-copy O(N·B) baseline. |
| 005 | T3 | Prefix-reuse identification: LCP O(log² N) multi-seed timing table (CIs) + prefix-dedup hit-rate | Add | local-CPU | **DONE (confirmatory)** | EXP-005 (2026-06-12, commit a15e883): n=9 (3 seeds × 3 inv), real corpus. Step-count guard 2·⌈log₂ N⌉, 0 violations. Hash slope 0.028 at f=0.5 (≤ 0.3). Correctness 0 mismatches. Crossover vs brute ~1 MB. **T3 → SUPPORTED.** Closed. |
| 006 | T4 | Repetition: RepeatNode O(log q) compression/throughput vs naïve materialization | Sol+Exp | local-CPU | planned | Workloads: repeated text blocks, log/JSON payloads, synthetic q-repetition. |
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

## Status key

`planned` → `in-progress` → `done`. Update this table and the matching `CLAIMS.md`
row whenever an experiment changes state; commit the change with the logbook
update.
