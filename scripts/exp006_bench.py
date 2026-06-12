#!/usr/bin/env python3
"""
EXP-006: RepeatNode O(log q) compression/throughput benchmark (claim T4).

Orchestrator+worker pattern.
Grid:
  q-sweep:    unit=4KB fixed,  q in {1,10,100,1000,10000}
  unit-sweep: q=1000 fixed,    unit in {128,1024,4096,16384}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev, median

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "experiments" / "exp_006_repeat" / "results"
DATA_DIR = REPO / "data" / "raw"

sys.path.insert(0, str(REPO))

import hashrope.rope as _rope
from hashrope import PolynomialHash
from src.flatten import build_fat_leaf_rope, make_hash
from src.memory import count_unique_nodes
from src.repeat_bench import build_repeat, build_naive_repeat

# Grid
DEFAULT_Q_VALUES = [1, 10, 100, 1_000, 10_000]
DEFAULT_Q_SWEEP_UNIT = 4096
DEFAULT_UNIT_VALUES = [128, 1024, 4096, 16384]
DEFAULT_UNIT_SWEEP_Q = 1000
REPS = 5  # timing reps per cell


def build_grid(q_values, q_sweep_unit, unit_values, unit_sweep_q):
    cells = set()
    for q in q_values:
        cells.add((q, q_sweep_unit))
    for u in unit_values:
        cells.add((unit_sweep_q, u))
    return sorted(cells)


def env_snapshot():
    import hashrope
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hashrope_version": getattr(hashrope, "__version__", "unknown"),
    }


def corpus_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def git_info(repo: Path) -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo),
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo),
            stderr=subprocess.DEVNULL).decode().strip())
        return {"sha": sha, "dirty": dirty}
    except Exception:
        return {"sha": "unknown", "dirty": None}


def run_worker(seed: int, inv: int, grid: list[tuple[int, int]]):
    corpus_path = DATA_DIR / f"corpus_s{seed}.txt"
    corpus = corpus_path.read_bytes()
    sha16 = corpus_sha(corpus_path)
    h = make_hash()

    results = []
    for q, unit_size in grid:
        if unit_size > len(corpus):
            print(f"  SKIP (q={q}, unit={unit_size}): "
                  f"unit {unit_size:,} > corpus {len(corpus):,}")
            continue

        unit_bytes = corpus[:unit_size]
        unit_rope = build_fat_leaf_rope(unit_bytes, h)
        unit_stats = count_unique_nodes([unit_rope])

        cell = {"q": q, "unit_size": unit_size,
                "unit_nodes": unit_stats["unique_count"]}

        # ---- RepeatNode arm ----
        # Warmup
        _ = build_repeat(unit_rope, q, h)

        # Timing
        times_r = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            r = build_repeat(unit_rope, q, h)
            t1 = time.perf_counter()
            times_r.append((t1 - t0) * 1000)

        # Memory
        tracemalloc.start()
        s0 = tracemalloc.take_snapshot()
        repeat_rope = build_repeat(unit_rope, q, h)
        s1 = tracemalloc.take_snapshot()
        repeat_delta = sum(s.size_diff for s in s1.compare_to(s0, "filename")
                          if s.size_diff > 0)
        tracemalloc.stop()

        repeat_stats = count_unique_nodes([repeat_rope])
        repeat_bytes = _rope.rope_to_bytes(repeat_rope)
        repeat_hash = _rope.rope_hash(repeat_rope)

        cell["repeat"] = {
            "unique_nodes": repeat_stats["unique_count"],
            "time_ms_median": median(times_r),
            "tracemalloc_bytes": repeat_delta,
            "hash": repeat_hash,
        }

        # ---- Naïve arm ----
        # Warmup (skip for very large materializations)
        if unit_size * q <= 50_000_000:
            _ = build_naive_repeat(unit_bytes, q, h)

        times_n = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            n = build_naive_repeat(unit_bytes, q, h)
            t1 = time.perf_counter()
            times_n.append((t1 - t0) * 1000)

        tracemalloc.start()
        s0 = tracemalloc.take_snapshot()
        naive_rope = build_naive_repeat(unit_bytes, q, h)
        s1 = tracemalloc.take_snapshot()
        naive_delta = sum(s.size_diff for s in s1.compare_to(s0, "filename")
                         if s.size_diff > 0)
        tracemalloc.stop()

        naive_stats = count_unique_nodes([naive_rope])
        naive_bytes = _rope.rope_to_bytes(naive_rope)
        naive_hash = _rope.rope_hash(naive_rope)

        cell["naive"] = {
            "unique_nodes": naive_stats["unique_count"],
            "time_ms_median": median(times_n),
            "tracemalloc_bytes": naive_delta,
            "hash": naive_hash,
        }

        # ---- Correctness ----
        expected = unit_bytes * q
        bytes_match = (repeat_bytes == expected and naive_bytes == expected)
        hash_match = (repeat_hash == naive_hash)
        cell["correctness"] = {
            "bytes_match": bytes_match,
            "hash_match": hash_match,
            "mismatch": not (bytes_match and hash_match),
        }

        # ---- Ratios ----
        cell["node_compression"] = (
            naive_stats["unique_count"] / repeat_stats["unique_count"]
            if repeat_stats["unique_count"] > 0 else float("inf"))
        cell["mem_compression"] = (
            naive_delta / repeat_delta
            if repeat_delta > 0 else float("inf"))
        cell["time_speedup"] = (
            median(times_n) / median(times_r)
            if median(times_r) > 0 else float("inf"))

        print(f"  s{seed} i{inv} q={q:>6} u={unit_size:>6}: "
              f"nodes {repeat_stats['unique_count']:>5}/{naive_stats['unique_count']:>8} "
              f"({cell['node_compression']:>8.1f}x)  "
              f"time {median(times_r):>8.3f}/{median(times_n):>10.1f} ms "
              f"({cell['time_speedup']:>8.1f}x)  "
              f"correct={'OK' if not cell['correctness']['mismatch'] else 'FAIL'}")

        del repeat_rope, naive_rope, repeat_bytes, naive_bytes
        results.append(cell)

    record = {
        "experiment": "exp006_repeat",
        "seed": seed, "invocation": inv,
        "corpus_sha256_16": sha16,
        "env": env_snapshot(),
        "git": git_info(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cells": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"exp006_s{seed}_i{inv}.json"
    out.write_text(json.dumps(record, indent=2, default=str))
    print(f"  -> {out}")
    return record


def aggregate(seeds, invocations):
    all_records = []
    for s in seeds:
        for i in range(invocations):
            p = RESULTS_DIR / f"exp006_s{s}_i{i}.json"
            if p.exists():
                all_records.append(json.loads(p.read_text()))

    if not all_records:
        print("ERROR: no worker results found")
        return None

    n_runs = len(all_records)
    print(f"\nAggregating {n_runs} worker results...")

    cell_data: dict[tuple[int, int], list[dict]] = {}
    for rec in all_records:
        for cell in rec["cells"]:
            key = (cell["q"], cell["unit_size"])
            cell_data.setdefault(key, []).append(cell)

    summary_cells = []
    total_mismatches = 0

    for (q, u), cells in sorted(cell_data.items()):
        n = len(cells)
        mismatches = sum(1 for c in cells if c["correctness"]["mismatch"])
        total_mismatches += mismatches

        r_nodes = [c["repeat"]["unique_nodes"] for c in cells]
        n_nodes = [c["naive"]["unique_nodes"] for c in cells]
        r_times = [c["repeat"]["time_ms_median"] for c in cells]
        n_times = [c["naive"]["time_ms_median"] for c in cells]
        node_comp = [c["node_compression"] for c in cells]
        time_sp = [c["time_speedup"] for c in cells]
        r_mem = [c["repeat"]["tracemalloc_bytes"] for c in cells]
        n_mem = [c["naive"]["tracemalloc_bytes"] for c in cells]

        sc = {
            "q": q, "unit_size": u, "n_runs": n,
            "unit_nodes": cells[0]["unit_nodes"],
            "repeat_nodes_mean": mean(r_nodes),
            "naive_nodes_mean": mean(n_nodes),
            "repeat_time_ms_mean": mean(r_times),
            "repeat_time_ms_std": stdev(r_times) if n > 1 else 0,
            "naive_time_ms_mean": mean(n_times),
            "naive_time_ms_std": stdev(n_times) if n > 1 else 0,
            "node_compression_mean": mean(node_comp),
            "time_speedup_mean": mean(time_sp),
            "time_speedup_std": stdev(time_sp) if n > 1 else 0,
            "repeat_mem_mean": mean(r_mem),
            "naive_mem_mean": mean(n_mem),
            "mismatches": mismatches,
        }
        summary_cells.append(sc)

    # ---- Criterion evaluation ----
    criterion = {}

    # (i) Correctness
    criterion["i_correctness"] = {
        "pass": total_mismatches == 0,
        "total_mismatches": total_mismatches,
    }

    # (ii) Node-count guard at (q=10000, unit=4096)
    c10k = [c for c in summary_cells if c["q"] == 10_000 and c["unit_size"] == 4096]
    if c10k:
        c = c10k[0]
        expected_repeat = c["unit_nodes"] + 1
        criterion["ii_node_guard"] = {
            "pass": (c["repeat_nodes_mean"] == expected_repeat
                     and c["node_compression_mean"] >= 100),
            "repeat_nodes": c["repeat_nodes_mean"],
            "expected": expected_repeat,
            "naive_nodes": c["naive_nodes_mean"],
            "compression": c["node_compression_mean"],
        }
    else:
        criterion["ii_node_guard"] = {"pass": False, "note": "cell missing"}

    # (iii) Construction-time scaling (q-sweep, unit=4096)
    q_sweep = [c for c in summary_cells if c["unit_size"] == 4096]
    q_sweep.sort(key=lambda c: c["q"])
    if len(q_sweep) >= 3:
        def log_log_slope(pairs):
            if len(pairs) < 2:
                return None
            lx = [math.log10(x) for x, _ in pairs]
            ly = [math.log10(y) for _, y in pairs]
            n = len(lx)
            mx, my = sum(lx)/n, sum(ly)/n
            num = sum((x-mx)*(y-my) for x, y in zip(lx, ly))
            den = sum((x-mx)**2 for x in lx)
            return num/den if den > 0 else 0

        # q=1 excluded from repeat slope: rope_repeat(node,1,h) returns the
        # input unchanged (no RepeatNode, no φ). It's a degenerate no-op.
        r_pairs = [(c["q"], c["repeat_time_ms_mean"]) for c in q_sweep
                    if c["q"] >= 10 and c["repeat_time_ms_mean"] > 0]
        n_pairs = [(c["q"], c["naive_time_ms_mean"]) for c in q_sweep
                    if c["q"] >= 1 and c["naive_time_ms_mean"] > 0]

        r_slope = log_log_slope(r_pairs)
        n_slope = log_log_slope(n_pairs)

        criterion["iii_scaling"] = {
            "pass": r_slope is not None and r_slope <= 0.3,
            "repeat_slope": r_slope,
            "naive_slope": n_slope,
            "naive_in_band": n_slope is not None and 0.7 <= n_slope <= 1.3,
        }
    else:
        criterion["iii_scaling"] = {"pass": False, "note": "too few q-sweep cells"}

    hard_pass = criterion["i_correctness"]["pass"]
    gates_pass = (hard_pass
                  and criterion.get("ii_node_guard", {}).get("pass", False)
                  and criterion.get("iii_scaling", {}).get("pass", False))
    criterion["verdict"] = "PASS" if gates_pass else "FAIL"
    criterion["hard_gate"] = "PASS" if hard_pass else "FAIL"

    summary = {
        "experiment": "exp006_repeat",
        "n_runs": n_runs, "seeds": seeds, "invocations": invocations,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": all_records[0]["env"],
        "git": all_records[0]["git"],
        "corpus_sha256_16": {str(r["seed"]): r["corpus_sha256_16"]
                             for r in all_records},
        "criterion_evaluation": criterion,
        "cells": summary_cells,
    }

    latest = RESULTS_DIR / "exp006_repeat_latest.json"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = RESULTS_DIR / f"exp006_repeat_{ts}.json"
    for p in [latest, archive]:
        p.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'='*70}")
    print(f"CRITERION EVALUATION")
    print(f"{'='*70}")
    print(f"  (i)   Correctness [HARD]:    {criterion['i_correctness']}")
    print(f"  (ii)  Node guard (q=10k):    {criterion.get('ii_node_guard', {})}")
    print(f"  (iii) Scaling (q-sweep):     {criterion.get('iii_scaling', {})}")
    print(f"  VERDICT: {criterion['verdict']}")
    print(f"\n  -> {latest}")
    print(f"  -> {archive}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="EXP-006 repeat benchmark")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inv", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--invocations", type=int, default=3)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Re-aggregate existing per-worker JSONs (no reruns)")
    args = parser.parse_args()

    grid = build_grid(DEFAULT_Q_VALUES, DEFAULT_Q_SWEEP_UNIT,
                      DEFAULT_UNIT_VALUES, DEFAULT_UNIT_SWEEP_Q)

    if args.worker:
        print(f"EXP-006 worker: seed={args.seed}, inv={args.inv}, "
              f"{len(grid)} cells")
        run_worker(args.seed, args.inv, grid)
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    invocations = args.invocations

    if not args.aggregate_only:
        print(f"EXP-006 orchestrator: seeds={seeds}, invocations={invocations}, "
              f"{len(grid)} cells/worker")

        for s in seeds:
            for i in range(invocations):
                print(f"\n--- Spawning worker seed={s} inv={i} ---")
                cmd = [sys.executable, __file__,
                       "--worker", "--seed", str(s), "--inv", str(i)]
                result = subprocess.run(cmd, cwd=str(REPO))
                if result.returncode != 0:
                    print(f"  WORKER FAILED (rc={result.returncode})")
                    sys.exit(1)

    aggregate(seeds, invocations)


if __name__ == "__main__":
    main()
