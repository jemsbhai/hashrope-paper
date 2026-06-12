#!/usr/bin/env python3
"""
EXP-004: Branch/snapshot memory benchmark (claim M1).

Orchestrator+worker pattern (mirrors EXP-001/002/005).
  orchestrator (default): spawns fresh subprocesses per (seed, invocation)
  worker (--worker):      measures one (seed, inv) and writes per-worker JSON

Grid:
  B-sweep:  N=2M fixed,  B in {1,2,5,10,25,50,100}
  N-sweep:  B=50 fixed,  N in {64K,256K,1M,2M,4M,8M}
  Extra:    (N=8M, B=100)  -- for criterion (iii)
"""
from __future__ import annotations

import argparse
import copy
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "experiments" / "exp_004_memory" / "results"
DATA_DIR = REPO / "data" / "raw"

sys.path.insert(0, str(REPO))

import hashrope.rope as _rope
from hashrope import PolynomialHash
from src.flatten import build_fat_leaf_rope, make_hash
from src.memory import count_unique_nodes, fork_rope, fork_deepcopy

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
DEFAULT_B_SWEEP_N = 2_000_000
DEFAULT_B_VALUES = [1, 2, 5, 10, 25, 50, 100]
DEFAULT_N_SWEEP_B = 50
DEFAULT_N_VALUES = [64_000, 256_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000]
EXTRA_CELLS = [(8_000_000, 100)]  # criterion (iii)
THOUGHT_BYTES = 128


def build_grid(n_values, b_values, n_sweep_b, b_sweep_n, extra):
    """Build deduplicated list of (N, B) cells."""
    cells = set()
    for b in b_values:
        cells.add((b_sweep_n, b))
    for n in n_values:
        cells.add((n, n_sweep_b))
    for n, b in extra:
        cells.add((n, b))
    return sorted(cells)


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------
def env_snapshot():
    import hashrope
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hashrope_version": getattr(hashrope, "__version__", "unknown"),
        "hashrope_file": getattr(hashrope, "__file__", "unknown"),
    }


def corpus_sha(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()[:16]


def git_info(repo: Path) -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo),
            stderr=subprocess.DEVNULL
        ).decode().strip())
        return {"sha": sha, "dirty": dirty}
    except Exception:
        return {"sha": "unknown", "dirty": None}


# ---------------------------------------------------------------------------
# Worker: measure one (seed, invocation)
# ---------------------------------------------------------------------------
def run_worker(seed: int, inv: int, grid: list[tuple[int, int]]):
    corpus_path = DATA_DIR / f"corpus_s{seed}.txt"
    corpus = corpus_path.read_bytes()
    sha16 = corpus_sha(corpus_path)
    h = make_hash()

    results = []
    for N, B in grid:
        # Ensure enough corpus for base + thoughts
        needed = N + B * THOUGHT_BYTES
        if needed > len(corpus):
            print(f"  SKIP (N={N}, B={B}): corpus too short ({len(corpus)} < {needed})")
            continue

        base_bytes = corpus[:N]
        thoughts = [corpus[N + i * THOUGHT_BYTES: N + (i + 1) * THOUGHT_BYTES]
                     for i in range(B)]

        # Build base rope (outside measurement)
        base = build_fat_leaf_rope(base_bytes, h)
        base_stats = count_unique_nodes([base])

        cell = {
            "N": N, "B": B, "thought_bytes": THOUGHT_BYTES,
            "base_weight": base.weight,
            "base_node_count": base_stats["unique_count"],
            "base_by_type": base_stats["by_type"],
        }

        # ---- Rope arm ----
        # Timing (individual forks)
        fork_times_rope = []
        for i in range(B):
            t0 = time.perf_counter()
            thought_leaf = _rope.Leaf(thoughts[i], h)
            _f = _rope.rope_concat(base, thought_leaf, h)
            t1 = time.perf_counter()
            fork_times_rope.append((t1 - t0) * 1000)

        # Memory (fresh tracemalloc)
        tracemalloc.start()
        snap0 = tracemalloc.take_snapshot()
        rope_forks = fork_rope(base, thoughts, h)
        snap1 = tracemalloc.take_snapshot()
        rope_delta = sum(
            s.size_diff for s in snap1.compare_to(snap0, "filename")
            if s.size_diff > 0
        )
        tracemalloc.stop()

        # Node count (with all forks alive)
        rope_node_stats = count_unique_nodes([base] + rope_forks)

        # Correctness
        rope_mismatches = 0
        for i, f in enumerate(rope_forks):
            if _rope.rope_to_bytes(f) != base_bytes + thoughts[i]:
                rope_mismatches += 1

        cell["rope"] = {
            "tracemalloc_delta_bytes": rope_delta,
            "per_fork_delta_bytes": rope_delta / max(B, 1),
            "unique_nodes": rope_node_stats["unique_count"],
            "by_type": rope_node_stats["by_type"],
            "total_reachable": rope_node_stats["total_reachable"],
            "correctness_mismatches": rope_mismatches,
            "per_fork_time_ms_median": median(fork_times_rope),
        }
        del rope_forks

        # ---- Deepcopy arm ----
        # Timing (individual forks)
        fork_times_dc = []
        for i in range(B):
            t0 = time.perf_counter()
            bc = copy.deepcopy(base)
            thought_leaf = _rope.Leaf(thoughts[i], h)
            _f = _rope.rope_concat(bc, thought_leaf, h)
            t1 = time.perf_counter()
            fork_times_dc.append((t1 - t0) * 1000)

        # Memory (fresh tracemalloc)
        tracemalloc.start()
        snap0 = tracemalloc.take_snapshot()
        dc_forks = fork_deepcopy(base, thoughts, h)
        snap1 = tracemalloc.take_snapshot()
        dc_delta = sum(
            s.size_diff for s in snap1.compare_to(snap0, "filename")
            if s.size_diff > 0
        )
        tracemalloc.stop()

        # Node count
        dc_node_stats = count_unique_nodes([base] + dc_forks)

        # Correctness
        dc_mismatches = 0
        for i, f in enumerate(dc_forks):
            if _rope.rope_to_bytes(f) != base_bytes + thoughts[i]:
                dc_mismatches += 1

        cell["deepcopy"] = {
            "tracemalloc_delta_bytes": dc_delta,
            "per_fork_delta_bytes": dc_delta / max(B, 1),
            "unique_nodes": dc_node_stats["unique_count"],
            "by_type": dc_node_stats["by_type"],
            "total_reachable": dc_node_stats["total_reachable"],
            "correctness_mismatches": dc_mismatches,
            "per_fork_time_ms_median": median(fork_times_dc),
        }
        del dc_forks

        # Ratios
        cell["compression_ratio_mem"] = (
            dc_delta / rope_delta if rope_delta > 0 else float("inf")
        )
        cell["sharing_ratio_nodes"] = (
            dc_node_stats["unique_count"] / rope_node_stats["unique_count"]
            if rope_node_stats["unique_count"] > 0 else float("inf")
        )

        print(f"  s{seed} i{inv} N={N:>10,} B={B:>3}: "
              f"rope={rope_delta:>12,}B  dc={dc_delta:>12,}B  "
              f"ratio={cell['compression_ratio_mem']:>8.1f}x  "
              f"nodes={rope_node_stats['unique_count']:>6}/{dc_node_stats['unique_count']:>6}  "
              f"correct={rope_mismatches}/{dc_mismatches}")

        results.append(cell)

    record = {
        "experiment": "exp004_memory",
        "seed": seed,
        "invocation": inv,
        "corpus_sha256_16": sha16,
        "env": env_snapshot(),
        "git": git_info(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cells": results,
    }

    # Write per-worker JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"exp004_s{seed}_i{inv}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"  -> {out}")
    return record


# ---------------------------------------------------------------------------
# Orchestrator: spawn workers, aggregate, evaluate criterion
# ---------------------------------------------------------------------------
def aggregate(seeds, invocations, grid):
    """Read all per-worker JSONs and produce summary."""
    all_records = []
    for s in seeds:
        for i in range(invocations):
            p = RESULTS_DIR / f"exp004_s{s}_i{i}.json"
            if p.exists():
                all_records.append(json.loads(p.read_text()))

    if not all_records:
        print("ERROR: no worker results found")
        return None

    n_runs = len(all_records)
    print(f"\nAggregating {n_runs} worker results...")

    # Build per-cell aggregation
    cell_data: dict[tuple[int, int], list[dict]] = {}
    for rec in all_records:
        for cell in rec["cells"]:
            key = (cell["N"], cell["B"])
            cell_data.setdefault(key, []).append(cell)

    summary_cells = []
    total_correctness_mismatches = 0

    for (N, B), cells in sorted(cell_data.items()):
        n = len(cells)
        rope_deltas = [c["rope"]["tracemalloc_delta_bytes"] for c in cells]
        dc_deltas = [c["deepcopy"]["tracemalloc_delta_bytes"] for c in cells]
        rope_nodes = [c["rope"]["unique_nodes"] for c in cells]
        dc_nodes = [c["deepcopy"]["unique_nodes"] for c in cells]
        ratios_mem = [c["compression_ratio_mem"] for c in cells]
        ratios_node = [c["sharing_ratio_nodes"] for c in cells]
        rope_mismatches = sum(c["rope"]["correctness_mismatches"] for c in cells)
        dc_mismatches = sum(c["deepcopy"]["correctness_mismatches"] for c in cells)
        total_correctness_mismatches += rope_mismatches + dc_mismatches
        rope_times = [c["rope"]["per_fork_time_ms_median"] for c in cells]
        dc_times = [c["deepcopy"]["per_fork_time_ms_median"] for c in cells]

        sc = {
            "N": N, "B": B, "n_runs": n,
            "base_weight": cells[0]["base_weight"],
            "base_node_count": cells[0]["base_node_count"],
            "rope_delta_mean": mean(rope_deltas),
            "rope_delta_std": stdev(rope_deltas) if n > 1 else 0,
            "dc_delta_mean": mean(dc_deltas),
            "dc_delta_std": stdev(dc_deltas) if n > 1 else 0,
            "rope_nodes_mean": mean(rope_nodes),
            "rope_nodes_std": stdev(rope_nodes) if n > 1 else 0,
            "dc_nodes_mean": mean(dc_nodes),
            "dc_nodes_std": stdev(dc_nodes) if n > 1 else 0,
            "compression_ratio_mem_mean": mean(ratios_mem),
            "compression_ratio_mem_std": stdev(ratios_mem) if n > 1 else 0,
            "sharing_ratio_nodes_mean": mean(ratios_node),
            "sharing_ratio_nodes_std": stdev(ratios_node) if n > 1 else 0,
            "rope_mismatches": rope_mismatches,
            "dc_mismatches": dc_mismatches,
            "rope_time_ms_mean": mean(rope_times),
            "dc_time_ms_mean": mean(dc_times),
            "rope_per_fork_delta_mean": mean(rope_deltas) / max(B, 1),
            "dc_per_fork_delta_mean": mean(dc_deltas) / max(B, 1),
        }
        summary_cells.append(sc)

    # ---- Evaluate promotion criterion ----
    criterion = {}

    # (i) HARD: byte-identity
    criterion["i_byte_identity"] = {
        "pass": total_correctness_mismatches == 0,
        "total_mismatches": total_correctness_mismatches,
    }

    # (ii) Structural sharing guard at (N=2M, B=50)
    c2m50 = [c for c in summary_cells if c["N"] == 2_000_000 and c["B"] == 50]
    if c2m50:
        c = c2m50[0]
        w_base = c["base_weight"]
        log_w = math.ceil(math.log2(max(w_base, 2)))
        max_rope_nodes = (2 * w_base) + 50 * (log_w + 3)
        criterion["ii_sharing_guard"] = {
            "pass": (c["rope_nodes_mean"] <= max_rope_nodes
                     and c["sharing_ratio_nodes_mean"] >= 10.0),
            "rope_nodes_mean": c["rope_nodes_mean"],
            "max_allowed": max_rope_nodes,
            "sharing_ratio_mean": c["sharing_ratio_nodes_mean"],
            "w_base": w_base, "log_w": log_w,
        }
    else:
        criterion["ii_sharing_guard"] = {"pass": False, "note": "cell (2M,50) missing"}

    # (iii) Memory at (N=8M, B=100)
    c8m100 = [c for c in summary_cells if c["N"] == 8_000_000 and c["B"] == 100]
    if c8m100:
        c = c8m100[0]
        criterion["iii_memory_compression"] = {
            "pass": c["compression_ratio_mem_mean"] >= 50.0,
            "ratio_mean": c["compression_ratio_mem_mean"],
            "ratio_std": c["compression_ratio_mem_std"],
        }
    else:
        criterion["iii_memory_compression"] = {"pass": False, "note": "cell (8M,100) missing"}

    # (iv) Scaling (descriptive, non-gating)
    n_sweep = [c for c in summary_cells if c["B"] == 50]
    n_sweep.sort(key=lambda c: c["N"])
    if len(n_sweep) >= 3:
        ns = [c["N"] for c in n_sweep]
        rope_pf = [c["rope_per_fork_delta_mean"] for c in n_sweep]
        dc_pf = [c["dc_per_fork_delta_mean"] for c in n_sweep]

        # Filter out zeros for log
        valid_rope = [(n, d) for n, d in zip(ns, rope_pf) if d > 0]
        valid_dc = [(n, d) for n, d in zip(ns, dc_pf) if d > 0]

        def log_log_slope(pairs):
            if len(pairs) < 2:
                return None
            log_n = [math.log10(n) for n, _ in pairs]
            log_d = [math.log10(d) for _, d in pairs]
            n_pts = len(log_n)
            x_mean = sum(log_n) / n_pts
            y_mean = sum(log_d) / n_pts
            num = sum((x - x_mean) * (y - y_mean)
                      for x, y in zip(log_n, log_d))
            den = sum((x - x_mean) ** 2 for x in log_n)
            return num / den if den > 0 else 0

        rope_slope = log_log_slope(valid_rope)
        dc_slope = log_log_slope(valid_dc)

        criterion["iv_scaling"] = {
            "descriptive": True,
            "rope_slope": rope_slope,
            "dc_slope": dc_slope,
            "rope_pass": rope_slope is not None and rope_slope <= 0.3,
            "dc_in_band": dc_slope is not None and 0.7 <= dc_slope <= 1.3,
        }
    else:
        criterion["iv_scaling"] = {"descriptive": True, "note": "too few N-sweep cells"}

    # Overall verdict
    hard_pass = criterion["i_byte_identity"]["pass"]
    gates_pass = (hard_pass
                  and criterion.get("ii_sharing_guard", {}).get("pass", False)
                  and criterion.get("iii_memory_compression", {}).get("pass", False))
    criterion["verdict"] = "PASS" if gates_pass else "FAIL"
    criterion["hard_gate"] = "PASS" if hard_pass else "FAIL"

    summary = {
        "experiment": "exp004_memory",
        "n_runs": n_runs,
        "seeds": seeds,
        "invocations": invocations,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": all_records[0]["env"],
        "git": all_records[0]["git"],
        "corpus_sha256_16": {str(r["seed"]): r["corpus_sha256_16"]
                             for r in all_records},
        "criterion_evaluation": criterion,
        "cells": summary_cells,
    }

    # Write summary
    latest = RESULTS_DIR / "exp004_memory_latest.json"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = RESULTS_DIR / f"exp004_memory_{ts}.json"
    for p in [latest, archive]:
        p.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*70}")
    print(f"CRITERION EVALUATION")
    print(f"{'='*70}")
    print(f"  (i)   Byte-identity [HARD]:  {criterion['i_byte_identity']}")
    print(f"  (ii)  Sharing guard (N=2M,B=50): {criterion.get('ii_sharing_guard', {})}")
    print(f"  (iii) Memory (N=8M,B=100):   {criterion.get('iii_memory_compression', {})}")
    print(f"  (iv)  Scaling (descriptive):  {criterion.get('iv_scaling', {})}")
    print(f"  VERDICT: {criterion['verdict']}")
    print(f"\n  -> {latest}")
    print(f"  -> {archive}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="EXP-004 memory benchmark")
    parser.add_argument("--worker", action="store_true", help="Worker mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inv", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="42,43,44",
                        help="Comma-separated seeds (orchestrator)")
    parser.add_argument("--invocations", type=int, default=3)
    args = parser.parse_args()

    grid = build_grid(
        DEFAULT_N_VALUES, DEFAULT_B_VALUES, DEFAULT_N_SWEEP_B,
        DEFAULT_B_SWEEP_N, EXTRA_CELLS,
    )

    if args.worker:
        print(f"EXP-004 worker: seed={args.seed}, inv={args.inv}, "
              f"{len(grid)} cells")
        run_worker(args.seed, args.inv, grid)
        return

    # Orchestrator
    seeds = [int(s) for s in args.seeds.split(",")]
    invocations = args.invocations
    print(f"EXP-004 orchestrator: seeds={seeds}, invocations={invocations}, "
          f"{len(grid)} cells/worker")

    for s in seeds:
        for i in range(invocations):
            print(f"\n--- Spawning worker seed={s} inv={i} ---")
            cmd = [
                sys.executable, __file__,
                "--worker", "--seed", str(s), "--inv", str(i),
            ]
            result = subprocess.run(cmd, cwd=str(REPO))
            if result.returncode != 0:
                print(f"  WORKER FAILED (returncode={result.returncode})")
                sys.exit(1)

    aggregate(seeds, invocations, grid)


if __name__ == "__main__":
    main()
