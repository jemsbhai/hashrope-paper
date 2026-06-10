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

**Date:** 2026-06-10 (planned)
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** planned

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
_[after run]_

### Observations
_[after run]_

### Interpretation
_[after run]_

### Artifacts
- Results: experiments/exp_001_tokenizer_aligned/results/
- Figures: experiments/exp_001_tokenizer_aligned/figures/
