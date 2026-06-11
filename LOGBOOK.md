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
- Results (timing, pending): experiments/exp_001_tokenizer_aligned/results/
- Figures (pending): experiments/exp_001_tokenizer_aligned/figures/
