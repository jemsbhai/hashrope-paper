# Data Provenance

Every dataset used in the paper is documented here: what it is, where it came
from, how it was obtained, and its checksum. Raw data under `data/raw/` is never
overwritten. Checksums for all data files live in `data/checksums.sha256`.

This file is a stub; entries are added as datasets enter the pipeline.

## Datasets

| Name | Source | Obtained | Used by | Checksum |
|------|--------|----------|---------|----------|
| _(none yet)_ | | | | |

## Planned

- **Real agent edit traces** — instrumented runs from an actual agent harness
  (e.g. SWE-bench / OpenHands), to characterize the edit distribution
  (append vs mid-edit vs delete). Motivates the entire problem framing; replaces
  the prior draft's synthetic `random.seed(42)` "SWE-agent" trace.
- **Tokenization corpora** — real prose + source code for EXP-001 (explicitly
  not `"A"*N`).
- **Genome data (chr22, ~51 MB decompressed)** — for repetition/compression
  benchmarks, if the bio angle enters this paper.
