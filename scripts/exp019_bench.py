#!/usr/bin/env python
"""
scripts/exp019_bench.py -- EXP-019 Milestone A bench (claim B3).

hashrope branch/snapshot vs a faithful PagedAttention block-table COW baseline
(Kwon et al., SOSP 2023, sec 4.2/4.4) on the controlled context-size sweep.

Orchestrator + worker (fresh subprocess per seed x invocation, EXP-004/017
template). Per (N, block size) the worker measures, and reports, ALL THREE
latency decompositions plus branching memory:

  * bare structural fork   -- hashrope O(1) root-share vs paged O(ceil(N/B)) table-copy   [descriptive]
  * branch-creation        -- fork + first divergent step                                  [GATED]
  * per-token append       -- hashrope O(log w) vs paged O(1) amortized                     [descriptive, boundary]
  * branching memory       -- hold `branch_count` live branches (EXP-004 style)             [GATED memory]

plus deterministic guards (hashrope per-branch new nodes <= ceil(log2 w)+3; paged
fork touches exactly ceil(N/B) block-table entries) and the HARD cross-arm
byte-identity check (hashrope == paged == oracle).

Gated metric is branch-creation, NOT bare fork -- see LOGBOOK EXP-019 Addendum A
(locked before this runs). Aggregation across the n=9 runs: mean +/- std, paired
sign test on branch-creation, verbatim criterion evaluation.

Base context = the seed's gpt2-tokenized corpus prefix (EXP-017 convention),
cached once per seed to data/canonical/exp019_base_tokens_s{seed}.npy (gitignored).
The divergent step and the per-branch memory steps are disjoint corpus slices that
follow the base. Latencies are pure-Python and interpreter-amplified; the
transferable claims are the deterministic structural guards and the O(log w) vs
O(N/B) scaling -- not the absolute constants.
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
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

from src.branch_bench import (
    expected_after_branch,
    hr_build,
    hr_fork,
    hr_append_token,
    hr_branch_create,
    hr_materialize,
    pa_build,
    pa_fork,
    pa_append_token,
    pa_branch_create,
    pa_materialize,
)
from src.flatten import make_hash
from src.memory import count_unique_nodes


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_INVOCATIONS = 3
DEFAULT_N_SWEEP = [1000, 4000, 16000, 64000, 256000, 1000000]
DEFAULT_BLOCK_SIZES = [8, 16, 32, 64]
DEFAULT_BRANCH_COUNTS = [5, 10, 25, 50]
DEFAULT_REPS = 5
DEFAULT_TARGET_S = 0.02
DEFAULT_STEP = 32          # divergent step length (tokens)
DEFAULT_APPEND_N = 256     # tokens appended for per-token append latency
REF_BLOCK = 16             # reference block size for the crossover criterion
MAX_K = 200_000            # inner-loop cap (bounds memory of cheap-op batches)

OUT_DIR = os.path.join(_REPO, "experiments", "exp_019_branch", "results")


# ---------------------------------------------------------------------------
# Token source
# ---------------------------------------------------------------------------

def base_tokens_path(seed: int) -> str:
    return os.path.join(_REPO, "data", "canonical", f"exp019_base_tokens_s{seed}.npy")


def corpus_path(seed: int) -> str:
    return os.path.join(_REPO, "data", "raw", f"corpus_s{seed}.txt")


def ensure_base_tokens(seed: int, needed: int) -> None:
    """Orchestrator-side: gpt2-tokenize the corpus prefix once and cache to .npy.
    Caches a generous margin so workers can slice base + disjoint steps."""
    p = base_tokens_path(seed)
    if os.path.exists(p) and len(np.load(p, mmap_mode="r")) >= needed:
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    with open(corpus_path(seed), "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < needed:
        raise RuntimeError(
            f"corpus s{seed} yields {len(ids)} tokens < needed {needed}; "
            f"use a larger corpus or smaller N."
        )
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, np.asarray(ids[:needed], dtype=np.int64))


def load_base_tokens(seed: int, needed: int) -> list[int]:
    arr = np.load(base_tokens_path(seed))
    if len(arr) < needed:
        raise RuntimeError(f"cached tokens s{seed}: {len(arr)} < needed {needed}")
    return arr[:needed].tolist()


def sha256_16(path: str) -> str:
    if not os.path.exists(path):
        return "NA"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _auto_time_rebind(call, reps: int, target_s: float, max_k: int) -> float:
    """Median per-op seconds for a cheap, side-effect-light op whose result is
    discarded by rebinding (no list-append overhead in the hot loop). `call` must
    not accumulate growing state across invocations."""
    t0 = time.perf_counter()
    n = 0
    while True:
        call()
        n += 1
        if (time.perf_counter() - t0) >= target_s or n >= max_k:
            break
    k = max(1, n)
    per = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(k):
            call()
        per.append((time.perf_counter() - t0) / k)
        gc.collect()
    return statistics.median(per)


def time_bare_fork_hr(hr_base, reps, target_s) -> float:
    return _auto_time_rebind(lambda: hr_fork(hr_base), reps, target_s, MAX_K)


def time_bare_fork_pa(pa_base, reps, target_s) -> float:
    # paged fork rebind-discards the child (freed immediately; no new blocks);
    # base block refcounts inflate harmlessly across the loop.
    return _auto_time_rebind(lambda: pa_fork(pa_base), reps, target_s, MAX_K)


def time_branch_create_hr(hr_base, step, h, reps, target_s) -> float:
    # immutable: results discarded, base shared, nothing leaks.
    return _auto_time_rebind(lambda: hr_branch_create(hr_base, step, h), reps, target_s, MAX_K)


def time_branch_create_pa(base_tokens, bs, step, reps, target_s) -> float:
    # paged branch-create COWs/allocates new blocks into the pool; rebuild a fresh
    # pool per batch (untimed) so the whole leak is discarded between batches.
    per = []
    for _ in range(reps):
        base = pa_build(base_tokens, bs)
        t0 = time.perf_counter()
        n = 0
        while True:
            pa_branch_create(base, step)
            n += 1
            if (time.perf_counter() - t0) >= target_s or n >= MAX_K:
                break
        per.append((time.perf_counter() - t0) / n)
        del base
        gc.collect()
    return statistics.median(per)


def time_append_hr(hr_base, h, n_app, reps) -> float:
    # immutable base is reused across reps (appends never mutate it).
    toks = list(range(1, n_app + 1))
    per = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = hr_base
        for t in toks:
            r = hr_append_token(r, t, h)
        per.append((time.perf_counter() - t0) / n_app)
        del r
        gc.collect()
    return statistics.median(per)


def time_append_pa(pa_base, n_app, reps) -> float:
    # fork the base (untimed, COW) per rep and append onto the fork; the base is
    # protected by copy-on-write, so it is reused across reps.
    toks = list(range(1, n_app + 1))
    per = []
    for _ in range(reps):
        child = pa_fork(pa_base)
        t0 = time.perf_counter()
        for t in toks:
            pa_append_token(child, t)
        per.append((time.perf_counter() - t0) / n_app)
        del child
        gc.collect()
    return statistics.median(per)


# ---------------------------------------------------------------------------
# Memory (EXP-004 style): hold `branch_count` live branches
# ---------------------------------------------------------------------------

def measure_memory(hr_base, base_tokens, bs, branch_count, steps, h) -> dict:
    # hashrope base is prebuilt and reused (immutable).
    base_unique = count_unique_nodes([hr_base])["unique_count"]
    gc.collect()
    tracemalloc.start()
    hr_branches = [hr_branch_create(hr_base, steps[i], h) for i in range(branch_count)]
    hr_cur, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    hr_unique = count_unique_nodes([hr_base] + hr_branches)["unique_count"]
    hr_new_nodes = hr_unique - base_unique
    del hr_branches
    gc.collect()

    # paged
    pa_base = pa_build(base_tokens, bs)
    gc.collect()
    tracemalloc.start()
    pa_branches = [pa_branch_create(pa_base, steps[i]) for i in range(branch_count)]
    pa_cur, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pa_table_entries = pa_base.num_blocks + sum(s.num_blocks for s in pa_branches)
    pa_live_blocks = pa_base.pool.num_live_blocks
    del pa_branches
    gc.collect()

    return {
        "N": len(base_tokens),
        "block_size": bs,
        "branch_count": branch_count,
        "hr_bytes": hr_cur,
        "pa_bytes": pa_cur,
        "compression": (pa_cur / hr_cur) if hr_cur > 0 else float("inf"),
        "hr_new_nodes": hr_new_nodes,
        "pa_table_entries": pa_table_entries,
        "pa_live_blocks": pa_live_blocks,
    }


# ---------------------------------------------------------------------------
# Worker: one (seed, invocation)
# ---------------------------------------------------------------------------

def run_worker(args) -> None:
    seed = args.seed
    n_sweep = args.n_sweep
    block_sizes = args.block_sizes
    branch_counts = args.branch_counts
    reps = args.reps
    target_s = args.target_s
    step_len = args.step
    append_n = args.append_n

    needed = max(n_sweep) + (max(branch_counts) + 1) * step_len + 16
    tokens = load_base_tokens(seed, needed)

    cells = []
    memory = []
    for N in n_sweep:
        base_tokens = tokens[:N]
        step = tokens[N:N + step_len]
        h = make_hash()
        hr_base, _h = hr_build(base_tokens, h)

        for bs in block_sizes:
            pa_base = pa_build(base_tokens, bs)

            # ---- correctness (HARD) ----
            oracle = expected_after_branch(base_tokens, step)
            hr_out = hr_materialize(hr_branch_create(hr_base, step, h))
            pa_out = pa_materialize(pa_branch_create(pa_base, step))
            mismatches = int(hr_out != oracle) + int(pa_out != oracle)

            # ---- guards (deterministic) ----
            base_unique = count_unique_nodes([hr_base])["unique_count"]
            one_branch = hr_branch_create(hr_base, step, h)
            after_unique = count_unique_nodes([hr_base, one_branch])["unique_count"]
            hr_new_nodes = after_unique - base_unique
            w = hr_base.weight if hr_base is not None else 1
            log_w = math.ceil(math.log2(max(w, 2)))
            hr_guard_ok = hr_new_nodes <= log_w + 3

            pa_fork_entries = pa_base.num_blocks
            pa_guard_expected = math.ceil(N / bs)
            pa_guard_ok = (pa_fork_entries == pa_guard_expected)

            # ---- latencies ----
            t_bare_hr = time_bare_fork_hr(hr_base, reps, target_s)
            t_bare_pa = time_bare_fork_pa(pa_base, reps, target_s)
            t_bc_hr = time_branch_create_hr(hr_base, step, h, reps, target_s)
            t_bc_pa = time_branch_create_pa(base_tokens, bs, step, reps, target_s)
            t_app_hr = time_append_hr(hr_base, h, append_n, reps)
            t_app_pa = time_append_pa(pa_base, append_n, reps)

            cells.append({
                "N": N,
                "block_size": bs,
                "mismatches": mismatches,
                "hr_new_nodes": hr_new_nodes,
                "log_w": log_w,
                "hr_guard_ok": hr_guard_ok,
                "pa_fork_entries": pa_fork_entries,
                "pa_guard_expected": pa_guard_expected,
                "pa_guard_ok": pa_guard_ok,
                "t_bare_fork_hr": t_bare_hr,
                "t_bare_fork_pa": t_bare_pa,
                "t_branch_create_hr": t_bc_hr,
                "t_branch_create_pa": t_bc_pa,
                "t_append_hr": t_app_hr,
                "t_append_pa": t_app_pa,
            })

            del pa_base
            gc.collect()

        # ---- memory cell: B distinct steps following the base ----
        max_bc = max(branch_counts)
        steps = [tokens[N + i * step_len:N + (i + 1) * step_len] for i in range(max_bc)]
        for bs in block_sizes:
            for bc in branch_counts:
                memory.append(measure_memory(hr_base, base_tokens, bs, bc, steps, h))

    out = {
        "experiment": "EXP-019",
        "milestone": "A",
        "seed": seed,
        "invocation": args.invocation,
        "step": step_len,
        "append_n": append_n,
        "reps": reps,
        "target_s": target_s,
        "n_sweep": n_sweep,
        "block_sizes": block_sizes,
        "branch_counts": branch_counts,
        "cells": cells,
        "memory": memory,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Orchestrator: spawn workers, aggregate, evaluate criterion
# ---------------------------------------------------------------------------

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "NA"


def hashrope_version() -> str:
    try:
        import hashrope
        return getattr(hashrope, "__version__", "NA")
    except Exception:
        return "NA"


def _mean_std(xs):
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def aggregate(runs, n_sweep, block_sizes, branch_counts):
    """runs: list of worker JSON dicts (the n=9). Returns aggregate + verdict."""
    # index per-run cells by (N, bs)
    def cell(run, N, bs):
        for c in run["cells"]:
            if c["N"] == N and c["block_size"] == bs:
                return c
        raise KeyError((N, bs))

    def mem(run, N, bs, bc):
        for m in run["memory"]:
            if m["N"] == N and m["block_size"] == bs and m["branch_count"] == bc:
                return m
        raise KeyError((N, bs, bc))

    agg_cells = []
    total_mismatches = 0
    all_guards_ok = True
    for N in n_sweep:
        for bs in block_sizes:
            cs = [cell(r, N, bs) for r in runs]
            total_mismatches += sum(c["mismatches"] for c in cs)
            for c in cs:
                all_guards_ok = all_guards_ok and c["hr_guard_ok"] and c["pa_guard_ok"]
            bc_hr = [c["t_branch_create_hr"] for c in cs]
            bc_pa = [c["t_branch_create_pa"] for c in cs]
            wins = sum(1 for a, b in zip(bc_hr, bc_pa) if a < b)
            m_hr, s_hr = _mean_std(bc_hr)
            m_pa, s_pa = _mean_std(bc_pa)
            speedup = (m_pa / m_hr) if (m_hr and m_hr > 0) else None
            hr_wins_band = (m_hr + s_hr) < (m_pa - s_pa)
            agg_cells.append({
                "N": N, "block_size": bs,
                "branch_create_hr_mean": m_hr, "branch_create_hr_std": s_hr,
                "branch_create_pa_mean": m_pa, "branch_create_pa_std": s_pa,
                "branch_create_speedup": speedup,
                "branch_create_sign_wins": wins, "n": len(cs),
                "hr_wins_band": hr_wins_band,
                "bare_fork_hr_mean": _mean_std([c["t_bare_fork_hr"] for c in cs])[0],
                "bare_fork_pa_mean": _mean_std([c["t_bare_fork_pa"] for c in cs])[0],
                "append_hr_mean": _mean_std([c["t_append_hr"] for c in cs])[0],
                "append_pa_mean": _mean_std([c["t_append_pa"] for c in cs])[0],
            })

    # memory aggregate
    agg_mem = []
    for N in n_sweep:
        for bs in block_sizes:
            for bc in branch_counts:
                ms = [mem(r, N, bs, bc) for r in runs]
                comp = [m["compression"] for m in ms]
                m_c, s_c = _mean_std(comp)
                agg_mem.append({
                    "N": N, "block_size": bs, "branch_count": bc,
                    "compression_mean": m_c, "compression_std": s_c,
                    "hr_bytes_mean": _mean_std([m["hr_bytes"] for m in ms])[0],
                    "pa_bytes_mean": _mean_std([m["pa_bytes"] for m in ms])[0],
                })

    # ---- verdict (verbatim criterion; gated metric = branch-creation) ----
    n_max = max(n_sweep)
    n = len(runs)

    # (i) HARD byte-identity
    crit_i = (total_mismatches == 0)
    # (ii) guards
    crit_ii = all_guards_ok

    # (iii) crossover at ref block 16: smallest N where hr wins the band, and
    #       pa <= hr for all smaller N (monotone crossover).
    ref_rows = sorted([c for c in agg_cells if c["block_size"] == REF_BLOCK], key=lambda c: c["N"])
    crossover_N = None
    for c in ref_rows:
        if c["hr_wins_band"]:
            crossover_N = c["N"]
            break
    below_ok = all((c["branch_create_pa_mean"] <= c["branch_create_hr_mean"])
                   for c in ref_rows if crossover_N is not None and c["N"] < crossover_N)
    crit_iii = (crossover_N is not None) and below_ok

    # (iv) at N=max: every block size -> 9/9 sign wins AND speedup>=2x (branch-create)
    #      AND branching-memory compression (branch_count=5) >= 10x.
    iv_latency = []
    for bs in block_sizes:
        row = next(c for c in agg_cells if c["N"] == n_max and c["block_size"] == bs)
        ok = (row["branch_create_sign_wins"] == n) and (row["branch_create_speedup"] is not None
                                                        and row["branch_create_speedup"] >= 2.0)
        iv_latency.append((bs, ok, row["branch_create_sign_wins"], row["branch_create_speedup"]))
    iv_memory = []
    for bs in block_sizes:
        mrow = next(m for m in agg_mem if m["N"] == n_max and m["block_size"] == bs and m["branch_count"] == 5)
        iv_memory.append((bs, (mrow["compression_mean"] is not None and mrow["compression_mean"] >= 10.0),
                          mrow["compression_mean"]))
    crit_iv = all(x[1] for x in iv_latency) and all(x[1] for x in iv_memory)

    supported = crit_i and crit_ii and crit_iii and crit_iv

    return {
        "aggregate_cells": agg_cells,
        "aggregate_memory": agg_mem,
        "verdict": {
            "n": n,
            "total_mismatches": total_mismatches,
            "crit_i_byte_identity": crit_i,
            "crit_ii_guards": crit_ii,
            "crit_iii_crossover_refblock": {"ok": crit_iii, "crossover_N": crossover_N,
                                            "ref_block": REF_BLOCK},
            "crit_iv_at_Nmax": {"ok": crit_iv, "N": n_max,
                                "latency": iv_latency, "memory_bc5": iv_memory},
            "B3_supported": supported,
        },
    }


def run_orchestrator(args) -> None:
    seeds = args.seeds
    invocations = args.invocations
    n_sweep = args.n_sweep
    block_sizes = args.block_sizes
    branch_counts = args.branch_counts

    os.makedirs(OUT_DIR, exist_ok=True)
    needed = max(n_sweep) + (max(branch_counts) + 1) * args.step + 16

    for seed in seeds:
        ensure_base_tokens(seed, needed)

    runs = []
    run_paths = []
    for seed in seeds:
        for inv in range(invocations):
            tag = f"s{seed}_i{inv}"
            run_out = os.path.join(OUT_DIR, f"exp019_run_{tag}.json")
            cmd = [
                sys.executable, os.path.abspath(__file__), "--worker",
                "--seed", str(seed), "--invocation", str(inv),
                "--n-sweep", ",".join(map(str, n_sweep)),
                "--block-sizes", ",".join(map(str, block_sizes)),
                "--branch-counts", ",".join(map(str, branch_counts)),
                "--reps", str(args.reps), "--target-s", str(args.target_s),
                "--step", str(args.step), "--append-n", str(args.append_n),
                "--out", run_out,
            ]
            print(f"[orch] {tag} ...", flush=True)
            subprocess.run(cmd, check=True)
            with open(run_out, "r", encoding="utf-8") as f:
                runs.append(json.load(f))
            run_paths.append(run_out)

    agg = aggregate(runs, n_sweep, block_sizes, branch_counts)

    env = {
        "git_sha": git_sha(),
        "hashrope_version": hashrope_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "corpus_sha16": {str(s): sha256_16(corpus_path(s)) for s in seeds},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary = {
        "experiment": "EXP-019",
        "milestone": "A",
        "claim": "B3",
        "gated_metric": "branch_create_latency (LOGBOOK Addendum A)",
        "seeds": seeds,
        "invocations": invocations,
        "n_runs": len(runs),
        "n_sweep": n_sweep,
        "block_sizes": block_sizes,
        "branch_counts": branch_counts,
        "ref_block": REF_BLOCK,
        "env": env,
        "run_files": [os.path.basename(p) for p in run_paths],
        **agg,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    latest = os.path.join(OUT_DIR, "exp019_milestoneA_latest.json")
    stamped = os.path.join(OUT_DIR, f"exp019_milestoneA_{ts}.json")
    for path in (latest, stamped):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    v = summary["verdict"]
    print("\n=== EXP-019 Milestone A verdict ===", flush=True)
    print(f"  n={v['n']}  mismatches={v['total_mismatches']}", flush=True)
    print(f"  (i)  byte-identity : {v['crit_i_byte_identity']}", flush=True)
    print(f"  (ii) guards        : {v['crit_ii_guards']}", flush=True)
    print(f"  (iii) crossover@b{REF_BLOCK}: {v['crit_iii_crossover_refblock']}", flush=True)
    print(f"  (iv) N={v['crit_iv_at_Nmax']['N']}    : {v['crit_iv_at_Nmax']['ok']}", flush=True)
    print(f"  B3 SUPPORTED       : {v['B3_supported']}", flush=True)
    print(f"  -> {latest}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EXP-019 Milestone A bench (claim B3).")
    p.add_argument("--worker", action="store_true", help="run a single (seed, invocation) worker")
    p.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    p.add_argument("--invocation", type=int, default=0)
    p.add_argument("--seeds", type=_int_list, default=DEFAULT_SEEDS)
    p.add_argument("--invocations", type=int, default=DEFAULT_INVOCATIONS)
    p.add_argument("--n-sweep", dest="n_sweep", type=_int_list, default=DEFAULT_N_SWEEP)
    p.add_argument("--block-sizes", dest="block_sizes", type=_int_list, default=DEFAULT_BLOCK_SIZES)
    p.add_argument("--branch-counts", dest="branch_counts", type=_int_list, default=DEFAULT_BRANCH_COUNTS)
    p.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p.add_argument("--target-s", dest="target_s", type=float, default=DEFAULT_TARGET_S)
    p.add_argument("--step", type=int, default=DEFAULT_STEP)
    p.add_argument("--append-n", dest="append_n", type=int, default=DEFAULT_APPEND_N)
    p.add_argument("--out", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.worker:
        if args.out is None:
            raise SystemExit("--worker requires --out")
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
