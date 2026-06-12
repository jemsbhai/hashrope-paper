# NOTICE — Vendored third-party code: SGLang RadixCache

This directory contains a **verbatim, unmodified** copy of the radix tree
implementation underlying **RadixAttention**, vendored from the SGLang project
for use as a competitive baseline (prefix-reuse identification, paper claim T3 /
experiment EXP-017).

## Provenance

| Field | Value |
|---|---|
| Upstream repository | https://github.com/sgl-project/sglang |
| Release tag | `v0.1.17` |
| Commit SHA | `e8a2327d523ce646edf400a2c6da647ca7d8c645` |
| Commit date | 2024-06-07 |
| Path at tag | `python/sglang/srt/managers/controller/radix_cache.py` |
| Retrieved | 2026-06-12 |
| Retrieval method | `git show v0.1.17:<path>` from a clone of the upstream repo; cross-verified byte-identical against `https://raw.githubusercontent.com/sgl-project/sglang/v0.1.17/python/sglang/srt/managers/controller/radix_cache.py` |
| SHA-256 of `radix_cache.py` | `42749c7cf0f3cbf42066dd273360730c8fe10e2a21983a49f102a500702ca71c` |
| License | Apache License 2.0 (repository-root `LICENSE` at tag `v0.1.17`; a copy is included in this directory). The file carries no per-file license header at this tag. |

## Why this version

Tag `v0.1.17` (2024-06-07) is the **paper-era release**: it was published one
day after arXiv:2312.07104 **v2** (2024-06-06), the revision whose content
matches the NeurIPS 2024 camera-ready. This is the canonical published
implementation of the algorithm described in:

> Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang,
> Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez,
> Clark Barrett, Ying Sheng.
> *SGLang: Efficient Execution of Structured Language Model Programs.*
> NeurIPS 2024. arXiv:2312.07104.

The diff of `radix_cache.py` between `v0.1.16` and `v0.1.17` is cosmetic
(file rename plus a `disable`-path tweak); the tree algorithm is identical,
so the baseline is not sensitive to the exact paper-era tag chosen.

## Modifications

**None.** The file is byte-identical to upstream (verify with the SHA-256
above; on Windows: `Get-FileHash radix_cache.py -Algorithm SHA256`).

All adaptations required to exercise this code standalone live in the
benchmark harness, **not** in this file:

- The tree is constructed as `RadixCache(None, None, False)` — the two SGLang
  memory-pool arguments are used only by `cache_req`, which the benchmark does
  not call. (Upstream's own `__main__` demo instantiates it the same way.)
- `match_prefix` concatenates node values with `torch.concat`, so the harness
  inserts keys with `torch.Tensor` values — exactly as SGLang itself does
  (`indices.clone()` in `cache_req`).

The benchmarked API surface is the tree algorithm: `insert`, `match_prefix`,
`evict`, and the lock-ref bookkeeping.
