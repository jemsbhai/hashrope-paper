#!/usr/bin/env python3
"""
EXP-015: Shared identification core (claim S3).

The host-side prefix-IDENTIFICATION layer, measured against a real serving
stack. This module is engine-agnostic and GPU-free: it builds the two request
regimes (R1 realistic streams, R2 long-shared-prefix synthetic), runs the three
identifier arms over them, and enforces the HARD cross-arm correctness gate
(LOGBOOK EXP-015 criterion (i)) including a non-vacuous negative control.

The GPU serving harness imports the *stream builders* from here so the request
schedule dispatched to SGLang is byte-identical across identifier arms -- the
construction that makes the GPU-energy result identification-invariant (H2).

Arms (reused VERBATIM from EXP-017 src.competitive -- no reimplementation):
  - radix    : vendored byte-identical SGLang RadixCache v0.1.17 (match_prefix).
               Lazy-imported; requires torch -- present on the cluster, optional
               in a CPU sandbox.
  - hashrope : LCP via prefix-hash binary search (Theorem 9 / EXP-005 machinery).
  - flat     : numpy C-speed array comparison (floor reference).
  - oracle   : plain-Python token loop (the correctness ground truth).

All arms operate on the common 4-byte-LE token encoding (src.competitive).

Regimes:
  R1  realistic streams -- ShareGPT/LMSYS multi-turn replay (src.realpairs),
      ~2k-token prefixes. hashrope is EXPECTED to lose identification latency to
      radix here (EXP-017 honest negative; reported in full, not the lede).
  R2  controlled long-shared-prefix synthetic -- one long cached context of L
      tokens, then short divergent queries each sharing exactly L. Straddles
      L*~571k to locate the live-stack identification crossover L*'.

This module does NOT touch a GPU, the serving engine, or NVML. It is fully
sandbox-validatable on CPU (oracle/hashrope/flat); the radix arm's three-way
agreement is exercised on the cluster smoke run (torch+sglang present there).
"""
from __future__ import annotations

import gc
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.competitive import (
    build_token_rope,
    flat_lcp_np,
    hashrope_lcp_tokens,
    oracle_lcp,
    radix_lcp,
)

TAIL_TOKENS = 1024            # divergent-tail length (matches EXP-017 convention)
TOKEN_DOMAIN = 1 << 32        # 4-byte LE encoding domain (uint32)


# ----------------------------------------------------------------------------
# Radix arm availability (lazy; requires torch)
# ----------------------------------------------------------------------------

def radix_available() -> bool:
    """True iff the vendored radix arm can be imported (torch present)."""
    try:
        import torch  # noqa: F401
        from third_party.sglang_radix_cache.radix_cache import RadixCache  # noqa: F401
        return True
    except Exception:
        return False


def _build_radix(cached_tokens: list[int]):
    """Insert cached tokens into a fresh vendored RadixCache; None if torch absent."""
    try:
        import torch
        from third_party.sglang_radix_cache.radix_cache import RadixCache
    except Exception:
        return None
    cache = RadixCache(None, None, False)
    if cached_tokens:
        cache.insert(list(cached_tokens), torch.arange(len(cached_tokens)))
    return cache


# ----------------------------------------------------------------------------
# Arm handles (prepared once per cached entry; reused across that entry's queries)
# ----------------------------------------------------------------------------

@dataclass
class ArmHandles:
    cached_tokens: list[int]
    rope_cached: Any           # hashrope Node | None
    h: Any                     # PolynomialHash
    arr_cached: np.ndarray
    radix_cache: Any           # vendored RadixCache | None
    radix_on: bool


def prepare(cached_tokens: list[int], with_radix: bool = True) -> ArmHandles:
    """Build per-arm cached structures once. radix only if requested AND torch present."""
    rope_cached, h = build_token_rope(cached_tokens)
    arr_cached = (np.array(cached_tokens, dtype=np.int64)
                  if cached_tokens else np.array([], dtype=np.int64))
    cache = _build_radix(cached_tokens) if with_radix else None
    return ArmHandles(
        cached_tokens=list(cached_tokens),
        rope_cached=rope_cached, h=h,
        arr_cached=arr_cached,
        radix_cache=cache, radix_on=cache is not None,
    )


# ----------------------------------------------------------------------------
# Identification: matched token-prefix length per arm
# ----------------------------------------------------------------------------

def identify(handles: ArmHandles, query_tokens: list[int]) -> dict[str, int | None]:
    """Each available arm's matched token-prefix length + the oracle.

    radix is None when the arm is unavailable (torch absent). The query rope is
    built with the cached entry's hash object so the two share an encoding.
    """
    rope_q, _ = build_token_rope(query_tokens, handles.h)
    arr_q = (np.array(query_tokens, dtype=np.int64)
             if query_tokens else np.array([], dtype=np.int64))
    return {
        "oracle": oracle_lcp(handles.cached_tokens, query_tokens),
        "hashrope": hashrope_lcp_tokens(handles.rope_cached, rope_q, handles.h),
        "flat": flat_lcp_np(handles.arr_cached, arr_q),
        "radix": (radix_lcp(handles.radix_cache, query_tokens)
                  if handles.radix_on else None),
    }


# ----------------------------------------------------------------------------
# Correctness gate (criterion i) + negative control
# ----------------------------------------------------------------------------

@dataclass
class GateResult:
    ok: bool
    n_checks: int
    mismatches: int
    arms_present: list[str]
    radix_on: bool
    detail: list[dict] = field(default_factory=list)


def correctness_gate(cached_tokens: list[int], queries: list[list[int]],
                     with_radix: bool = True, keep_detail: bool = False) -> GateResult:
    """Every available arm must equal the oracle for every query. 0 mismatches.

    The HARD criterion-(i) check, run per cached entry over its queries.
    """
    handles = prepare(cached_tokens, with_radix=with_radix)
    arms = ["hashrope", "flat"] + (["radix"] if handles.radix_on else [])
    mism = 0
    detail: list[dict] = []
    for q in queries:
        r = identify(handles, q)
        oracle = r["oracle"]
        bad = {a: r[a] for a in arms if r[a] != oracle}
        if bad:
            mism += 1
            if keep_detail:
                detail.append({"oracle": oracle, "disagree": bad, "query_len": len(q)})
    return GateResult(
        ok=(mism == 0), n_checks=len(queries), mismatches=mism,
        arms_present=["oracle"] + arms, radix_on=handles.radix_on, detail=detail,
    )


def negative_control(cached_tokens: list[int], query_tokens: list[int],
                     corrupt_pos: int | None = None, with_radix: bool = True) -> dict:
    """Corrupt ONE cached token before the true LCP; every available arm must
    report the shortened match. Proves the gate is non-vacuous (the arms truly
    compare content, they do not blindly return the query length).

    Returns {pos, true_lcp, results, arms_checked, detected}; detected is True
    iff every available arm reports matched length == corrupt_pos (< true_lcp).
    """
    true_lcp = oracle_lcp(cached_tokens, query_tokens)
    if true_lcp < 1:
        raise ValueError("negative_control requires true_lcp >= 1")
    pos = corrupt_pos if corrupt_pos is not None else true_lcp // 2
    if not (0 <= pos < true_lcp):
        raise ValueError(f"corrupt_pos {pos} must be in [0, true_lcp={true_lcp})")
    corrupted = list(cached_tokens)
    corrupted[pos] = (corrupted[pos] + 1) % TOKEN_DOMAIN
    handles = prepare(corrupted, with_radix=with_radix)
    r = identify(handles, query_tokens)
    arms = ["hashrope", "flat"] + (["radix"] if handles.radix_on else [])
    detected = all(r[a] == pos for a in arms)
    return {"pos": pos, "true_lcp": true_lcp, "results": r,
            "arms_checked": arms, "detected": detected}


# ----------------------------------------------------------------------------
# R2 stream: one long cached prefix (L tokens) + N short divergent queries
# ----------------------------------------------------------------------------

def build_r2_stream(corpus_tokens: list[int], L: int, n_queries: int,
                    seed: int, tail_tokens: int = TAIL_TOKENS) -> dict:
    """One cached context of EXACTLY L tokens; n_queries each = cached + a
    distinct divergent tail (oracle LCP == L for every query).

    Because the cached entry has length L, any query beginning with the same L
    tokens shares EXACTLY L (cached is exhausted at L), independent of its tail.
    Distinct tails vary the (LCP-irrelevant) continuation so the stream is not
    trivially repeated queries.
    """
    rng = random.Random(seed + L)
    need = L + tail_tokens
    if len(corpus_tokens) < need:
        raise ValueError(f"corpus has {len(corpus_tokens)} tokens, need >= {need} for L={L}")
    cached = list(corpus_tokens[:L])
    region_lo, region_hi = L, len(corpus_tokens) - tail_tokens
    queries: list[list[int]] = []
    for k in range(n_queries):
        if region_hi > region_lo:
            start = region_lo + rng.randrange(region_hi - region_lo)
        else:
            start = region_lo
        tail = list(corpus_tokens[start:start + tail_tokens])
        if not tail:  # corpus-exhausted edge case: synthesize a tail
            tail = [(cached[-1] + 1 + k + j) % (1 << 16) for j in range(tail_tokens)]
        queries.append(cached + tail)
    return {"regime": "R2", "L": L, "oracle_lcp": L,
            "cached_tokens": cached, "queries": queries, "n_queries": n_queries}


# ----------------------------------------------------------------------------
# R1 stream: realistic ShareGPT/LMSYS multi-turn replay (reuses src.realpairs)
# ----------------------------------------------------------------------------

def build_r1_stream(dataset_path: str, n_pairs: int, seed: int,
                    tokenizer_name: str = "gpt2") -> dict:
    """Realistic stream from a canonical conversation dataset (lazy import:
    needs transformers). Each item = (cached=turns[:t], query=turns[:t+1]) with
    the BPE-determined token-LCP as the oracle.

    tokenizer_name should be the SERVING model (EXP-015 tokenizes with the serving
    model's tokenizer throughout, so identifier IDs == served IDs); it defaults to
    gpt2 only for standalone/EXP-017-style use.

    Returns {regime, dataset, items:[{cached_tokens, query_tokens, oracle_lcp, conv_id}]}.
    """
    from src.realpairs import load_real_pairs  # lazy: transformers dependency
    pairs = load_real_pairs(dataset_path, n_pairs=n_pairs, seed=seed,
                            tokenizer_name=tokenizer_name)
    items = [{"cached_tokens": p["cached_tokens"],
              "query_tokens": p["query_tokens"],
              "oracle_lcp": p["oracle_token_lcp"],
              "conv_id": p["conv_id"]} for p in pairs]
    return {"regime": "R1", "dataset": Path(dataset_path).name, "items": items}


# ----------------------------------------------------------------------------
# Identification timing (EXP-017-consistent: prebuilt structures, warm + reps,
# gc disabled). Times the identification QUERY only; the query-structure build
# cost is reported separately for the serving-loop view. radix timed iff present.
# ----------------------------------------------------------------------------

def time_identify(handles: ArmHandles, query_tokens: list[int],
                  reps: int = 5) -> dict[str, float | None]:
    """Median identification latency (ms) per arm on prebuilt structures.

    Mirrors EXP-017: cached + query structures are built untimed; only the
    LCP/match query is timed. Returns hashrope/flat ms, radix ms (or None), and
    the untimed query-structure build cost (hashrope_query_build_ms).
    """
    t0 = time.perf_counter()
    rope_q, _ = build_token_rope(query_tokens, handles.h)
    hashrope_build_ms = (time.perf_counter() - t0) * 1e3
    arr_q = (np.array(query_tokens, dtype=np.int64)
             if query_tokens else np.array([], dtype=np.int64))

    # warm (discarded)
    hashrope_lcp_tokens(handles.rope_cached, rope_q, handles.h)
    flat_lcp_np(handles.arr_cached, arr_q)
    if handles.radix_on:
        radix_lcp(handles.radix_cache, query_tokens)

    gc.disable()
    h_ms = []
    for _ in range(reps):
        t = time.perf_counter()
        hashrope_lcp_tokens(handles.rope_cached, rope_q, handles.h)
        h_ms.append((time.perf_counter() - t) * 1e3)
    f_ms = []
    for _ in range(reps):
        t = time.perf_counter()
        flat_lcp_np(handles.arr_cached, arr_q)
        f_ms.append((time.perf_counter() - t) * 1e3)
    r_ms = None
    if handles.radix_on:
        rr = []
        for _ in range(reps):
            t = time.perf_counter()
            radix_lcp(handles.radix_cache, query_tokens)
            rr.append((time.perf_counter() - t) * 1e3)
        r_ms = statistics.median(rr)
    gc.enable()

    return {"hashrope_ms": statistics.median(h_ms),
            "flat_ms": statistics.median(f_ms),
            "radix_ms": r_ms,
            "hashrope_query_build_ms": hashrope_build_ms}


# ----------------------------------------------------------------------------
# Self-test (synthetic; CPU-only): exercises identify + gate + negative control.
# Run from the repo root: python -m src.exp015_identify
# ----------------------------------------------------------------------------

def _selftest() -> int:
    rng = random.Random(20260618)
    VOCAB = 50257  # gpt2-like domain (fits the 4B LE encoding)

    print("EXP-015 identification core -- self-test (synthetic, CPU)")
    print(f"  radix arm available (torch present): {radix_available()}")
    failures = 0

    base = [rng.randrange(VOCAB) for _ in range(200_000)]

    # --- R2 builder + correctness gate at several L ---
    for L in [1_000, 16_000, 64_000]:
        stream = build_r2_stream(base, L=L, n_queries=4, seed=42)
        assert stream["oracle_lcp"] == L
        gate = correctness_gate(stream["cached_tokens"], stream["queries"],
                                keep_detail=True)
        status = "PASS" if gate.ok else "FAIL"
        failures += 0 if gate.ok else 1
        print(f"  [R2 L={L:>7,}] gate {status}: checks={gate.n_checks} "
              f"mismatches={gate.mismatches} arms={gate.arms_present}")
        if gate.detail:
            print(f"      first disagreement: {gate.detail[0]}")

    # --- explicit LCP value check (cached is a proper prefix of query) ---
    stream = build_r2_stream(base, L=4_096, n_queries=1, seed=7)
    handles = prepare(stream["cached_tokens"])
    res = identify(handles, stream["queries"][0])
    val_ok = (res["oracle"] == 4_096 and res["hashrope"] == 4_096
              and res["flat"] == 4_096 and res["radix"] in (None, 4_096))
    print(f"  [value check L=4096] {res} -> {'PASS' if val_ok else 'FAIL'}")
    failures += 0 if val_ok else 1

    # --- negative control: corruption before the LCP must be detected ---
    nc = negative_control(stream["cached_tokens"], stream["queries"][0],
                          corrupt_pos=2_048)
    nc_ok = (nc["detected"] and nc["results"]["hashrope"] == 2_048
             and nc["results"]["flat"] == 2_048
             and nc["results"]["radix"] in (None, 2_048))
    print(f"  [neg control pos=2048] detected={nc['detected']} "
          f"results={nc['results']} -> {'PASS' if nc_ok else 'FAIL'}")
    failures += 0 if nc_ok else 1

    # --- a divergent (non-prefix) query: LCP < L ---
    cached = [rng.randrange(VOCAB) for _ in range(5_000)]
    q = list(cached)
    q[1_234] = (q[1_234] + 1) % TOKEN_DOMAIN  # first difference at index 1234
    handles = prepare(cached)
    res = identify(handles, q)
    div_ok = (res["oracle"] == 1_234 and res["hashrope"] == 1_234
              and res["flat"] == 1_234 and res["radix"] in (None, 1_234))
    print(f"  [divergent query] LCP={res} -> {'PASS' if div_ok else 'FAIL'}")
    failures += 0 if div_ok else 1

    # --- timing smoke (no magnitude assertions) ---
    print(f"  [timing smoke] {time_identify(handles, q, reps=3)}")

    print(f"\n  SELF-TEST {'PASSED' if failures == 0 else 'FAILED'} "
          f"({failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
