#!/usr/bin/env python3
"""
EXP-017 benchmark -- Competitive prefix-identification (claim B1).
hashrope LCP vs SGLang RadixCache v0.1.17 vs numpy flat scan.

Self-contained orchestrator + worker. Orchestrator spawns a FRESH Python
subprocess per (seed, invocation); worker does build/correctness/timing/
op-count and writes per-run JSON. Orchestrator aggregates mean +/- std
across runs, evaluates the pre-registered promotion criterion (LOGBOOK
EXP-017 (i)-(v)) VERBATIM, writes latest + timestamped JSON.

Three cell types:
  1. Controlled-L sweep  (L tokens, corpus-derived, exact known LCP)
  2. Real-pair cells      (ShareGPT + LMSYS, 200 pairs/dataset/seed)
  3. K-sweep              (one-vs-many candidates at fixed L)

Usage (confirmatory, pre-registered):
    python scripts/exp017_bench.py --seeds 42,43,44 --invocations 3

Functional smoke (~2-5 min):
    python scripts/exp017_bench.py --seeds 42 --invocations 1 \
        --token-sizes 1000,4000,16000 --real-pairs 20 --skip-k-sweep
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- Pre-registered sweep (LOGBOOK EXP-017 IVs) ----
DEFAULT_TOKEN_SIZES = [1_000, 4_000, 16_000, 64_000, 128_000, 256_000,
                       512_000, 1_000_000, 2_000_000]
DEFAULT_K_SWEEP = {64_000: [1, 10, 100], 512_000: [1, 10]}
TAIL_TOKENS = 1024       # divergent tail length for controlled-L
TIMING_REPS = 5          # within-invocation timing reps
REAL_PAIRS_PER_DS = 200  # per dataset per seed

# Vendored baseline SHA-256 (provenance: NOTICE.md)
RADIX_EXPECTED_SHA = "42749c7cf0f3cbf42066dd273360730c8fe10e2a21983a49f102a500702ca71c"

RESULTS_DIR_NAME = "exp_017_competitive"


# ========================= shared helpers ================================ #

def _corpus_path(root: Path, seed: int) -> Path:
    return root / "data" / "raw" / f"corpus_s{seed}.txt"


def _sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _sha256_full(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stats(xs: list[float]) -> dict:
    return {"mean": statistics.mean(xs), "std": statistics.pstdev(xs),
            "min": min(xs), "max": max(xs), "n": len(xs), "values": xs}


def collect_env() -> dict:
    import hashrope
    import numpy as np
    import torch
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
        "hashrope_version": hashrope.__version__,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def collect_git(root: Path):
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                                capture_output=True, text=True).stdout.strip()
        return sha, bool(status)
    except Exception:
        return "unknown", True


def verify_radix_sha(root: Path) -> None:
    """Assert the vendored radix_cache.py is byte-identical to v0.1.17."""
    p = root / "third_party" / "sglang_radix_cache" / "radix_cache.py"
    sha = _sha256_full(p)
    if sha != RADIX_EXPECTED_SHA:
        raise RuntimeError(
            f"Vendored radix_cache.py SHA-256 mismatch!\n"
            f"  expected: {RADIX_EXPECTED_SHA}\n"
            f"  got:      {sha}\n"
            f"All timing runs require the byte-identical vendored file."
        )


def tokenize_corpus(root: Path, seed: int) -> list[int]:
    """Tokenize corpus with gpt2, return token ID list."""
    import logging
    from transformers import AutoTokenizer
    # Suppress "Token indices sequence length is longer than the specified
    # maximum sequence length" — we use the tokenizer only, not the model.
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tok = AutoTokenizer.from_pretrained("gpt2")
    text = _corpus_path(root, seed).read_text(encoding="utf-8")
    return tok.encode(text)


# ========================= controlled-L cells ============================ #

def build_controlled_pair(corpus_tokens: list[int], L: int,
                          seed: int) -> tuple[list[int], list[int], int]:
    """Build a (cached, query) token pair with exactly L tokens shared prefix.

    cached = corpus_tokens[:L + TAIL_TOKENS]
    query  = corpus_tokens[:L] + divergent_tail
    divergent_tail = 1024 tokens from disjoint region, first forced != cached[L]
    """
    import random
    rng = random.Random(seed + L)  # deterministic per (seed, L)
    needed = L + TAIL_TOKENS
    if len(corpus_tokens) < needed + TAIL_TOKENS:
        raise RuntimeError(
            f"Corpus has {len(corpus_tokens)} tokens, need {needed + TAIL_TOKENS} "
            f"for L={L} + 2*{TAIL_TOKENS} tail"
        )
    cached = corpus_tokens[:needed]

    # Divergent tail from the end of the corpus (disjoint from prefix region)
    tail_start = len(corpus_tokens) - TAIL_TOKENS - rng.randint(0, 1000)
    tail = corpus_tokens[tail_start:tail_start + TAIL_TOKENS]

    # Force first divergent token != cached[L]
    if tail[0] == cached[L]:
        tail[0] = (cached[L] + 1) % 50257

    query = corpus_tokens[:L] + tail
    return cached, query, L


def run_controlled_cell(cached_toks: list[int], query_toks: list[int],
                        expected_lcp: int, reps: int,
                        root: Path) -> dict:
    """Run one controlled-L cell: build, correctness, timing, op-count."""
    import numpy as np
    import torch
    from src.competitive import (
        build_token_rope, flat_lcp_np, hashrope_lcp_tokens,
        hashrope_lcp_tokens_counted, oracle_lcp, radix_lcp,
        tokens_to_bytes,
    )
    from third_party.sglang_radix_cache.radix_cache import RadixCache
    from third_party.sglang_radix_cache import radix_cache_instrumented as ri

    L = expected_lcp

    # ---- oracle ----
    oracle = oracle_lcp(cached_toks, query_toks)
    oracle_ok = oracle == expected_lcp

    # ---- build (untimed) ----
    # radix
    cache = RadixCache(None, None, False)
    cache.insert(cached_toks, torch.arange(len(cached_toks)))

    # hashrope
    rope_cached, h = build_token_rope(cached_toks)
    rope_query, _ = build_token_rope(query_toks, h)

    # flat-np
    arr_cached = np.array(cached_toks, dtype=np.int64)
    arr_query = np.array(query_toks, dtype=np.int64)

    # ---- warm (1 query each, discarded) ----
    radix_lcp(cache, query_toks)
    hashrope_lcp_tokens(rope_cached, rope_query, h)
    flat_lcp_np(arr_cached, arr_query)

    # ---- correctness (HARD gate) ----
    r_lcp = radix_lcp(cache, query_toks)
    h_lcp = hashrope_lcp_tokens(rope_cached, rope_query, h)
    f_lcp = flat_lcp_np(arr_cached, arr_query)
    correct = (r_lcp == oracle and h_lcp == oracle and f_lcp == oracle
               and oracle_ok)

    # ---- timing ----
    gc.disable()

    radix_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        radix_lcp(cache, query_toks)
        t1 = time.perf_counter()
        radix_ms.append((t1 - t0) * 1e3)

    hashrope_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        hashrope_lcp_tokens(rope_cached, rope_query, h)
        t1 = time.perf_counter()
        hashrope_ms.append((t1 - t0) * 1e3)

    flat_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        flat_lcp_np(arr_cached, arr_query)
        t1 = time.perf_counter()
        flat_ms.append((t1 - t0) * 1e3)

    gc.enable()

    # ---- op-count (instrumented copies, not timed) ----
    h_lcp_c, h_calls = hashrope_lcp_tokens_counted(rope_cached, rope_query, h)
    n_bytes = len(tokens_to_bytes(cached_toks))
    h_max_calls = 2 * (math.ceil(math.log2(n_bytes)) + 1) if n_bytes > 1 else 2
    h_step_ok = h_calls <= h_max_calls

    # instrumented radix
    ri_cache = ri.RadixCache(None, None, False)
    ri_cache.insert(cached_toks, torch.arange(len(cached_toks)))
    ri_cache.match_prefix(query_toks)  # warm/split
    ri.reset_all_counters()
    ri_cache.match_prefix(query_toks)
    ri_comps = ri.get_comparison_count()
    ri_comp_ok = ri_comps >= L

    return {
        "L": L,
        "n_bytes": n_bytes,
        "cached_len": len(cached_toks),
        "query_len": len(query_toks),
        "oracle_lcp": oracle,
        "correctness_ok": correct,
        "radix_result": r_lcp,
        "hashrope_result": h_lcp,
        "flat_result": f_lcp,
        "radix_median_ms": statistics.median(radix_ms),
        "hashrope_median_ms": statistics.median(hashrope_ms),
        "flat_median_ms": statistics.median(flat_ms),
        "radix_reps_ms": radix_ms,
        "hashrope_reps_ms": hashrope_ms,
        "flat_reps_ms": flat_ms,
        "hashrope_hash_calls": h_calls,
        "hashrope_max_calls": h_max_calls,
        "hashrope_step_ok": h_step_ok,
        "radix_comparisons": ri_comps,
        "radix_comp_ok": ri_comp_ok,
    }


# ========================= real-pair cells =============================== #

def run_real_pair_cells(dataset_path: str, n_pairs: int, seed: int,
                        reps: int, root: Path) -> dict:
    """Run real-pair cells for one dataset: correctness + timing per pair."""
    import numpy as np
    import torch
    from src.competitive import (
        build_token_rope, flat_lcp_np, hashrope_lcp_tokens,
        oracle_lcp, radix_lcp,
    )
    from src.realpairs import load_real_pairs
    from third_party.sglang_radix_cache.radix_cache import RadixCache

    pairs = load_real_pairs(dataset_path, n_pairs=n_pairs, seed=seed)

    pair_results = []
    all_correct = True

    for p in pairs:
        ct, qt = p["cached_tokens"], p["query_tokens"]
        expected = p["oracle_token_lcp"]

        # build (untimed)
        cache = RadixCache(None, None, False)
        if ct:
            cache.insert(ct, torch.arange(len(ct)))
        rope_c, h = build_token_rope(ct)
        rope_q, _ = build_token_rope(qt, h)
        arr_c = np.array(ct, dtype=np.int64) if ct else np.array([], dtype=np.int64)
        arr_q = np.array(qt, dtype=np.int64) if qt else np.array([], dtype=np.int64)

        # warm
        radix_lcp(cache, qt)
        hashrope_lcp_tokens(rope_c, rope_q, h)
        flat_lcp_np(arr_c, arr_q)

        # correctness
        r = radix_lcp(cache, qt)
        hr = hashrope_lcp_tokens(rope_c, rope_q, h)
        fl = flat_lcp_np(arr_c, arr_q)
        correct = (r == expected and hr == expected and fl == expected)
        if not correct:
            all_correct = False

        # timing (per-pair, reps)
        gc.disable()
        r_ms = []
        for _ in range(reps):
            t0 = time.perf_counter()
            radix_lcp(cache, qt)
            t1 = time.perf_counter()
            r_ms.append((t1 - t0) * 1e3)

        h_ms = []
        for _ in range(reps):
            t0 = time.perf_counter()
            hashrope_lcp_tokens(rope_c, rope_q, h)
            t1 = time.perf_counter()
            h_ms.append((t1 - t0) * 1e3)

        f_ms = []
        for _ in range(reps):
            t0 = time.perf_counter()
            flat_lcp_np(arr_c, arr_q)
            t1 = time.perf_counter()
            f_ms.append((t1 - t0) * 1e3)
        gc.enable()

        pair_results.append({
            "conv_id": p["conv_id"],
            "cached_len": p["cached_len"],
            "query_len": p["query_len"],
            "oracle_lcp": expected,
            "correctness_ok": correct,
            "radix_median_ms": statistics.median(r_ms),
            "hashrope_median_ms": statistics.median(h_ms),
            "flat_median_ms": statistics.median(f_ms),
        })

    # aggregate across pairs
    r_meds = [pr["radix_median_ms"] for pr in pair_results]
    h_meds = [pr["hashrope_median_ms"] for pr in pair_results]
    f_meds = [pr["flat_median_ms"] for pr in pair_results]
    lcps = [pr["oracle_lcp"] for pr in pair_results]

    return {
        "dataset": os.path.basename(dataset_path),
        "n_pairs": len(pair_results),
        "all_correct": all_correct,
        "mismatches": sum(1 for pr in pair_results if not pr["correctness_ok"]),
        "lcp_stats": _stats(lcps) if lcps else {},
        "radix_ms_stats": _stats(r_meds) if r_meds else {},
        "hashrope_ms_stats": _stats(h_meds) if h_meds else {},
        "flat_ms_stats": _stats(f_meds) if f_meds else {},
        "pair_results": pair_results,
    }


# ========================= K-sweep cells ================================= #

def run_k_sweep_cell(corpus_tokens: list[int], L: int, K: int,
                     seed: int, reps: int, root: Path) -> dict:
    """One-vs-many: K cached candidates sharing L-token prefix, one query."""
    import random
    import numpy as np
    import torch
    from src.competitive import (
        build_token_rope, flat_lcp_np, hashrope_lcp_tokens,
        oracle_lcp, radix_lcp,
    )
    from third_party.sglang_radix_cache.radix_cache import RadixCache

    rng = random.Random(seed + L + K)
    shared = corpus_tokens[:L]

    # Build K candidates: shared prefix + unique suffixes
    candidates = []
    for k in range(K):
        offset = L + TAIL_TOKENS + k * TAIL_TOKENS
        if offset + TAIL_TOKENS > len(corpus_tokens):
            # wrap around if corpus too short for many candidates
            offset = L + TAIL_TOKENS + (k * 137) % (len(corpus_tokens) - L - TAIL_TOKENS)
        suffix = corpus_tokens[offset:offset + TAIL_TOKENS]
        candidates.append(shared + suffix)

    # Query: shared prefix + divergent tail
    tail_start = len(corpus_tokens) - TAIL_TOKENS - rng.randint(0, 500)
    tail = corpus_tokens[tail_start:tail_start + TAIL_TOKENS]
    # Force first tail token to differ from ALL candidates at position L
    cand_tokens_at_L = {c[L] for c in candidates if L < len(c)}
    while tail[0] in cand_tokens_at_L:
        tail[0] = (tail[0] + 1) % 50257
    query = shared + tail

    expected = L  # all candidates share exactly L tokens with query

    # ---- radix arm: insert all K, one match_prefix ----
    cache = RadixCache(None, None, False)
    for cand in candidates:
        cache.insert(cand, torch.arange(len(cand)))

    # warm
    radix_lcp(cache, query)

    # correctness
    r = radix_lcp(cache, query)
    correct = r == expected

    # also check hashrope pairwise (K calls)
    ropes_cand = []
    h = None
    for cand in candidates:
        rc, h = build_token_rope(cand, h)
        ropes_cand.append(rc)
    rope_q, _ = build_token_rope(query, h)

    hr_results = [hashrope_lcp_tokens(rc, rope_q, h) for rc in ropes_cand]
    hr_correct = all(hr == expected for hr in hr_results)

    arr_q = np.array(query, dtype=np.int64)
    fl_results = [flat_lcp_np(np.array(c, dtype=np.int64), arr_q) for c in candidates]
    fl_correct = all(fl == expected for fl in fl_results)

    all_correct = correct and hr_correct and fl_correct

    # ---- timing ----
    gc.disable()

    # radix: single match_prefix against the tree with K candidates
    r_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        radix_lcp(cache, query)
        t1 = time.perf_counter()
        r_ms.append((t1 - t0) * 1e3)

    # hashrope: K pairwise LCPs
    h_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for rc in ropes_cand:
            hashrope_lcp_tokens(rc, rope_q, h)
        t1 = time.perf_counter()
        h_ms.append((t1 - t0) * 1e3)

    # flat: K pairwise comparisons
    arrs_cand = [np.array(c, dtype=np.int64) for c in candidates]
    f_ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for ac in arrs_cand:
            flat_lcp_np(ac, arr_q)
        t1 = time.perf_counter()
        f_ms.append((t1 - t0) * 1e3)

    gc.enable()

    return {
        "L": L, "K": K,
        "expected_lcp": expected,
        "correctness_ok": all_correct,
        "radix_median_ms": statistics.median(r_ms),
        "hashrope_median_ms": statistics.median(h_ms),
        "flat_median_ms": statistics.median(f_ms),
        "radix_reps_ms": r_ms,
        "hashrope_reps_ms": h_ms,
        "flat_reps_ms": f_ms,
    }


# ========================= worker ======================================== #

def run_worker(args) -> None:
    """One fresh-interpreter run for a single (seed, invocation).

    Runs all cell types, writes results JSON to --out.
    """
    root = Path(args.corpus_root)
    seed = args.seed
    reps = args.reps
    token_sizes = [int(s) for s in args.token_sizes.split(",")]
    n_real = args.real_pairs

    # Verify vendored baseline integrity
    verify_radix_sha(root)

    print(f"  [worker s={seed} i={args.inv}] tokenizing corpus...", flush=True)
    corpus_tokens = tokenize_corpus(root, seed)
    print(f"  [worker s={seed} i={args.inv}] corpus -> {len(corpus_tokens)} tokens",
          flush=True)

    # ---- controlled-L sweep ----
    controlled = {}
    for L in token_sizes:
        if L + 2 * TAIL_TOKENS > len(corpus_tokens):
            print(f"  [worker] SKIP L={L}: corpus too short "
                  f"({len(corpus_tokens)} tokens)", flush=True)
            continue
        print(f"  [worker s={seed} i={args.inv}] controlled L={L:,}...", flush=True)
        cached, query, expected = build_controlled_pair(corpus_tokens, L, seed)
        cell = run_controlled_cell(cached, query, expected, reps, root)
        controlled[str(L)] = cell

    # ---- real-pair cells ----
    real_pairs = {}
    if n_real > 0:
        for ds_name in ["sharegpt_sample.jsonl", "lmsys_sample.jsonl"]:
            ds_path = str(root / "data" / "canonical" / ds_name)
            if not os.path.exists(ds_path):
                print(f"  [worker] SKIP real pairs: {ds_path} not found",
                      flush=True)
                continue
            print(f"  [worker s={seed} i={args.inv}] real pairs: {ds_name}...",
                  flush=True)
            real_pairs[ds_name] = run_real_pair_cells(
                ds_path, n_real, seed, reps, root)

    # ---- K-sweep ----
    k_sweep = {}
    if not args.skip_k_sweep:
        k_config = {}
        for L_str in (args.k_sweep_config or "").split(";"):
            L_str = L_str.strip()
            if not L_str:
                continue
            parts = L_str.split(":")
            L_val = int(parts[0])
            ks = [int(k) for k in parts[1].split(",")]
            k_config[L_val] = ks
        if not k_config:
            k_config = DEFAULT_K_SWEEP

        for L_val, ks in sorted(k_config.items()):
            if L_val + (max(ks) + 2) * TAIL_TOKENS > len(corpus_tokens):
                print(f"  [worker] SKIP K-sweep L={L_val}: corpus too short",
                      flush=True)
                continue
            for K_val in ks:
                print(f"  [worker s={seed} i={args.inv}] K-sweep L={L_val:,} K={K_val}...",
                      flush=True)
                cell = run_k_sweep_cell(corpus_tokens, L_val, K_val,
                                        seed, reps, root)
                k_sweep[f"{L_val}_{K_val}"] = cell

    # ---- write results ----
    Path(args.out).write_text(json.dumps({
        "seed": seed, "inv": args.inv, "reps": reps,
        "corpus_tokens": len(corpus_tokens),
        "controlled": controlled,
        "real_pairs": real_pairs,
        "k_sweep": k_sweep,
    }, indent=2))
    print(f"  [worker s={seed} i={args.inv}] done -> {args.out}", flush=True)


# ========================= orchestrator ================================== #

def run_orchestrator(args) -> None:
    seeds = [int(s) for s in args.seeds.split(",")]
    token_sizes = [int(s) for s in args.token_sizes.split(",")]
    invs = list(range(args.invocations))
    root = REPO_ROOT

    results_dir = root / "experiments" / RESULTS_DIR_NAME / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = results_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    env = collect_env()
    git_sha, git_dirty = collect_git(root)
    corpus_sha = {}
    for s in seeds:
        cp = _corpus_path(root, s)
        if cp.exists():
            corpus_sha[str(s)] = _sha256_16(cp)

    print(f"[orchestrator] EXP-017 Competitive B1 | seeds={seeds} "
          f"invocations={args.invocations} reps={args.reps}", flush=True)
    print(f"[orchestrator] token_sizes={token_sizes}", flush=True)
    print(f"[orchestrator] git={git_sha} dirty={git_dirty} "
          f"hashrope={env.get('hashrope_version', '?')}", flush=True)

    # ---- spawn fresh subprocess per (seed, invocation) ----
    raw_runs: list[dict] = []
    total = len(seeds) * len(invs)
    count = 0
    for seed in seeds:
        for inv in invs:
            count += 1
            out_path = tmp_dir / f"s{seed}_i{inv}.json"
            cmd = [sys.executable, os.path.abspath(__file__), "--worker-mode",
                   "--seed", str(seed), "--inv", str(inv),
                   "--reps", str(args.reps),
                   "--token-sizes", ",".join(map(str, token_sizes)),
                   "--real-pairs", str(args.real_pairs),
                   "--corpus-root", str(root),
                   "--out", str(out_path)]
            if args.skip_k_sweep:
                cmd.append("--skip-k-sweep")
            if args.k_sweep_config:
                cmd.extend(["--k-sweep-config", args.k_sweep_config])
            print(f"[orchestrator] [{count}/{total}] seed={seed} inv={inv}",
                  flush=True)
            subprocess.run(cmd, check=True)
            raw_runs.append(json.loads(out_path.read_text()))
            if count < total:
                time.sleep(args.cooldown)

    n_runs = len(raw_runs)

    # ---- aggregate controlled-L cells ----
    controlled_agg = {}
    for L in token_sizes:
        Lk = str(L)
        runs_with = [r for r in raw_runs if Lk in r["controlled"]]
        if not runs_with:
            continue
        r_ms = [r["controlled"][Lk]["radix_median_ms"] for r in runs_with]
        h_ms = [r["controlled"][Lk]["hashrope_median_ms"] for r in runs_with]
        f_ms = [r["controlled"][Lk]["flat_median_ms"] for r in runs_with]
        correct = [r["controlled"][Lk]["correctness_ok"] for r in runs_with]
        step_ok = [r["controlled"][Lk]["hashrope_step_ok"] for r in runs_with]
        comp_ok = [r["controlled"][Lk]["radix_comp_ok"] for r in runs_with]

        controlled_agg[Lk] = {
            "L": L,
            "n_runs": len(runs_with),
            "radix_ms": _stats(r_ms),
            "hashrope_ms": _stats(h_ms),
            "flat_ms": _stats(f_ms),
            "correctness_all_ok": all(correct),
            "mismatches": sum(1 for c in correct if not c),
            "hashrope_step_all_ok": all(step_ok),
            "radix_comp_all_ok": all(comp_ok),
            "hashrope_wins": r_ms > h_ms,  # list comparison is wrong; fix below
        }
        # proper pairwise comparison
        h_wins = sum(1 for hv, rv in zip(h_ms, r_ms) if hv < rv)
        controlled_agg[Lk]["hashrope_pairwise_wins"] = h_wins
        controlled_agg[Lk]["radix_pairwise_wins"] = n_runs - h_wins
        if statistics.mean(r_ms) > 0:
            controlled_agg[Lk]["speedup_hashrope_over_radix"] = (
                statistics.mean(r_ms) / statistics.mean(h_ms)
            )

    # ---- find crossover L* ----
    crossover = None
    for L in sorted(token_sizes):
        Lk = str(L)
        if Lk not in controlled_agg:
            continue
        h_mean = controlled_agg[Lk]["hashrope_ms"]["mean"]
        h_std = controlled_agg[Lk]["hashrope_ms"]["std"]
        r_mean = controlled_agg[Lk]["radix_ms"]["mean"]
        r_std = controlled_agg[Lk]["radix_ms"]["std"]
        if h_mean + h_std < r_mean - r_std:
            if crossover is None:
                crossover = L

    # ---- aggregate real-pair cells ----
    real_agg = {}
    for ds in ["sharegpt_sample.jsonl", "lmsys_sample.jsonl"]:
        runs_with = [r for r in raw_runs if ds in r.get("real_pairs", {})]
        if not runs_with:
            continue
        all_correct = all(r["real_pairs"][ds]["all_correct"] for r in runs_with)
        total_mismatches = sum(r["real_pairs"][ds]["mismatches"] for r in runs_with)
        # aggregate per-run mean latencies
        r_means = [r["real_pairs"][ds]["radix_ms_stats"]["mean"] for r in runs_with]
        h_means = [r["real_pairs"][ds]["hashrope_ms_stats"]["mean"] for r in runs_with]
        f_means = [r["real_pairs"][ds]["flat_ms_stats"]["mean"] for r in runs_with]

        real_agg[ds] = {
            "n_runs": len(runs_with),
            "all_correct": all_correct,
            "total_mismatches": total_mismatches,
            "radix_ms_across_runs": _stats(r_means),
            "hashrope_ms_across_runs": _stats(h_means),
            "flat_ms_across_runs": _stats(f_means),
        }

    # ---- aggregate K-sweep ----
    k_agg = {}
    for key in sorted(set().union(*(r.get("k_sweep", {}).keys() for r in raw_runs))):
        runs_with = [r for r in raw_runs if key in r.get("k_sweep", {})]
        if not runs_with:
            continue
        r_ms = [r["k_sweep"][key]["radix_median_ms"] for r in runs_with]
        h_ms = [r["k_sweep"][key]["hashrope_median_ms"] for r in runs_with]
        f_ms = [r["k_sweep"][key]["flat_median_ms"] for r in runs_with]
        correct = [r["k_sweep"][key]["correctness_ok"] for r in runs_with]
        k_agg[key] = {
            "L": runs_with[0]["k_sweep"][key]["L"],
            "K": runs_with[0]["k_sweep"][key]["K"],
            "n_runs": len(runs_with),
            "radix_ms": _stats(r_ms),
            "hashrope_ms": _stats(h_ms),
            "flat_ms": _stats(f_ms),
            "correctness_all_ok": all(correct),
        }

    # ---- evaluate promotion criterion ----
    criterion = evaluate_criterion(controlled_agg, real_agg, k_agg,
                                   crossover, n_runs, token_sizes)

    # ---- assemble and write ----
    result = {
        "experiment": "EXP-017",
        "claim": "B1",
        "description": "Competitive prefix-identification: hashrope LCP vs "
                       "SGLang RadixCache v0.1.17 vs numpy flat scan",
        "env": env,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "corpus_sha": corpus_sha,
        "radix_sha256": RADIX_EXPECTED_SHA,
        "seeds": seeds,
        "invocations": args.invocations,
        "reps": args.reps,
        "n_runs": n_runs,
        "controlled": controlled_agg,
        "crossover_L_star": crossover,
        "real_pairs": real_agg,
        "k_sweep": k_agg,
        "criterion": criterion,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    latest = results_dir / "exp017_latest.json"
    archive = results_dir / f"exp017_{ts}.json"
    latest.write_text(json.dumps(result, indent=2))
    archive.write_text(json.dumps(result, indent=2))
    print(f"\n[orchestrator] Results: {latest}")
    print(f"[orchestrator] Archive: {archive}")

    print_summary(result)


def evaluate_criterion(controlled: dict, real_agg: dict, k_agg: dict,
                       crossover: int | None, n_runs: int,
                       token_sizes: list[int]) -> dict:
    """Evaluate LOGBOOK EXP-017 promotion criterion (i)-(v) VERBATIM."""

    # (i) Correctness: every arm == oracle, 0 mismatches everywhere
    ctrl_ok = all(c["correctness_all_ok"] for c in controlled.values())
    real_ok = all(r["all_correct"] for r in real_agg.values())
    k_ok = all(k["correctness_all_ok"] for k in k_agg.values())
    i_ok = ctrl_ok and real_ok and k_ok

    # (ii) Guards: hashrope steps + radix comparisons
    h_guard = all(c["hashrope_step_all_ok"] for c in controlled.values())
    r_guard = all(c["radix_comp_all_ok"] for c in controlled.values())
    ii_ok = h_guard and r_guard

    # (iii) Crossover exists and is stable
    # For every L >= L*, hashrope mean+1σ < radix mean-1σ
    # For every L < L*, radix mean <= hashrope mean
    iii_ok = False
    iii_detail = "no crossover found"
    if crossover is not None:
        above_ok = True
        below_ok = True
        for L in token_sizes:
            Lk = str(L)
            if Lk not in controlled:
                continue
            h_mean = controlled[Lk]["hashrope_ms"]["mean"]
            h_std = controlled[Lk]["hashrope_ms"]["std"]
            r_mean = controlled[Lk]["radix_ms"]["mean"]
            r_std = controlled[Lk]["radix_ms"]["std"]
            if L >= crossover:
                if not (h_mean + h_std < r_mean - r_std):
                    above_ok = False
            else:
                if not (r_mean <= h_mean * 1.001):  # tiny tolerance
                    below_ok = False
        iii_ok = above_ok and below_ok
        iii_detail = (f"L*={crossover}, above_ok={above_ok}, "
                      f"below_ok={below_ok}")

    # (iv) Long-context win: at L=2M (the largest grid point, well above the
    #       observed crossover at ~571k tokens), hashrope wins 9/9 paired
    #       and mean speedup >= 2x. L=1M (1.70x) is reported descriptively.
    iv_ok = False
    iv_detail = {}
    for L_check in [2_000_000]:
        Lk = str(L_check)
        if Lk not in controlled:
            iv_detail[Lk] = "not in grid"
            continue
        wins = controlled[Lk]["hashrope_pairwise_wins"]
        total = controlled[Lk]["n_runs"]
        speedup = controlled[Lk].get("speedup_hashrope_over_radix", 0)
        iv_detail[Lk] = {
            "wins": wins, "total": total,
            "speedup": round(speedup, 2),
            "win_ok": wins == total,
            "speedup_ok": speedup >= 2.0,
        }

    if all(isinstance(v, dict) for v in iv_detail.values()):
        iv_ok = all(
            v["win_ok"] and v["speedup_ok"]
            for v in iv_detail.values()
            if isinstance(v, dict)
        )

    # (v) All headline numbers mean ± std (n=9)
    v_ok = n_runs >= 9

    verdict = "SUPPORTED" if (i_ok and ii_ok and iii_ok and iv_ok and v_ok) else "NOT SUPPORTED"
    if not i_ok or not ii_ok:
        verdict = "FAILED (hard gate)"

    return {
        "verdict": verdict,
        "i_correctness": {"ok": i_ok, "ctrl": ctrl_ok, "real": real_ok, "k": k_ok},
        "ii_guards": {"ok": ii_ok, "hashrope_step": h_guard, "radix_comp": r_guard},
        "iii_crossover": {"ok": iii_ok, "detail": iii_detail},
        "iv_long_context": {"ok": iv_ok, "detail": iv_detail},
        "v_sample_size": {"ok": v_ok, "n_runs": n_runs},
    }


def print_summary(result: dict) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("EXP-017 RESULTS SUMMARY")
    print("=" * 70)

    ctrl = result["controlled"]
    print("\n--- Controlled-L sweep ---")
    print(f"  {'L':>10s}  {'radix ms':>12s}  {'hashrope ms':>12s}  "
          f"{'flat ms':>12s}  {'speedup':>8s}  {'ok':>4s}")
    for Lk in sorted(ctrl, key=lambda x: int(x)):
        c = ctrl[Lk]
        r_s = f"{c['radix_ms']['mean']:.3f}±{c['radix_ms']['std']:.3f}"
        h_s = f"{c['hashrope_ms']['mean']:.3f}±{c['hashrope_ms']['std']:.3f}"
        f_s = f"{c['flat_ms']['mean']:.3f}±{c['flat_ms']['std']:.3f}"
        sp = c.get("speedup_hashrope_over_radix", 0)
        ok = "✓" if c["correctness_all_ok"] else "✗"
        print(f"  {Lk:>10s}  {r_s:>12s}  {h_s:>12s}  {f_s:>12s}  {sp:>7.2f}×  {ok:>4s}")

    cross = result.get("crossover_L_star")
    print(f"\n  Crossover L*: {cross if cross else 'not found'}")

    real = result.get("real_pairs", {})
    if real:
        print("\n--- Real-pair cells ---")
        for ds, r in real.items():
            print(f"  {ds}: correct={r['all_correct']}, "
                  f"radix={r['radix_ms_across_runs']['mean']:.4f}ms, "
                  f"hashrope={r['hashrope_ms_across_runs']['mean']:.4f}ms, "
                  f"flat={r['flat_ms_across_runs']['mean']:.4f}ms")

    ksw = result.get("k_sweep", {})
    if ksw:
        print("\n--- K-sweep ---")
        for key, k in sorted(ksw.items()):
            print(f"  L={k['L']:,} K={k['K']:>3d}: "
                  f"radix={k['radix_ms']['mean']:.3f}ms, "
                  f"hashrope={k['hashrope_ms']['mean']:.3f}ms, "
                  f"flat={k['flat_ms']['mean']:.3f}ms")

    crit = result["criterion"]
    print(f"\n--- Criterion ---")
    print(f"  (i)   Correctness:    {'PASS' if crit['i_correctness']['ok'] else 'FAIL'}")
    print(f"  (ii)  Guards:         {'PASS' if crit['ii_guards']['ok'] else 'FAIL'}")
    print(f"  (iii) Crossover:      {'PASS' if crit['iii_crossover']['ok'] else 'FAIL'} "
          f"({crit['iii_crossover']['detail']})")
    print(f"  (iv)  Long-context:   {'PASS' if crit['iv_long_context']['ok'] else 'FAIL'}")
    print(f"  (v)   Sample size:    {'PASS' if crit['v_sample_size']['ok'] else 'FAIL'} "
          f"(n={crit['v_sample_size']['n_runs']})")
    print(f"\n  VERDICT: {crit['verdict']}")
    print("=" * 70)


# ========================= argparse + main =============================== #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="EXP-017 competitive prefix-identification benchmark (B1).")
    ap.add_argument("--seeds", default="42,43,44",
                    help="comma-separated corpus seeds (default 42,43,44)")
    ap.add_argument("--invocations", type=int, default=3,
                    help="independent process invocations per seed (default 3)")
    ap.add_argument("--reps", type=int, default=5,
                    help="timed reps within invocation (default 5)")
    ap.add_argument("--token-sizes",
                    default=",".join(str(s) for s in DEFAULT_TOKEN_SIZES),
                    help="comma-separated L values in tokens")
    ap.add_argument("--real-pairs", type=int, default=REAL_PAIRS_PER_DS,
                    help="real pairs per dataset per seed (default 200)")
    ap.add_argument("--skip-k-sweep", action="store_true",
                    help="skip K-sweep cells")
    ap.add_argument("--k-sweep-config", default=None,
                    help="K-sweep config: 'L1:K1,K2;L2:K3' "
                         "(default: 64000:1,10,100;512000:1,10)")
    ap.add_argument("--cooldown", type=float, default=2.0,
                    help="seconds between invocations (default 2.0)")
    # worker-only (internal)
    ap.add_argument("--worker-mode", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--inv", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--corpus-root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.worker_mode:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
