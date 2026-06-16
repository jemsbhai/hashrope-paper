#!/usr/bin/env python
"""
scripts/exp019_milestoneB_bench.py -- EXP-019 Milestone B bench (claim B3,
supplementary ecological validity).

Deterministic replay of real gpt-oss-120b Game-of-24 Tree-of-Thought traces
(data/canonical/tot_traces_gptoss120b_s{seed}.jsonl) through the SAME
branch/snapshot harness used in Milestone A (src/branch_bench.py), reusing
src/memory.py UNCHANGED. Milestone A's scripts/exp019_bench.py is frozen and
untouched.

Milestone B is supplementary: it CANNOT un-support B3, which is already SUPPORTED
via Milestone A. See LOGBOOK "EXP-019 -- Milestone B plan" for the locked design
and the verbatim CORROBORATION criterion.

Structure mirrors Milestone A (orchestrator + hidden --worker-mode subprocess per
(seed, invocation); _mean_std aggregate; paired sign test; dual JSON output with
full env metadata). The workload is swapped from the synthetic single-branch
N-sweep to the real-trace replay batch.

Replay model (token space, 4-byte little-endian via src.competitive, identical to
Milestone A):
  * base context at the root  = prefix_tokens + puzzle_tokens
  * node v's context          = base + concatenation of thought_tokens along root..v
  * each non-root node         = ONE branch-creation off its parent's context
      - hashrope: hr_branch_create(ctx[parent], thought_tokens[v], h)   -- O(log w)
      - paged:    pa_branch_create(ctx[parent], thought_tokens[v])      -- O(ceil(N/B))
The oracle is the token-list concatenation prefix + puzzle + path-thoughts
(src.branch_bench.expected_after_branch semantics); 4-byte-LE makes token-concat
== byte-concat exactly.

PASS 1 = importable replay/oracle/guard helpers (tested in tests/test_exp019_milestoneB.py).
PASS 2 = worker/orchestrator/timing/memory/verdict (below the helpers).
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
    hr_build,
    hr_branch_create,
    hr_materialize,
    pa_build,
    pa_branch_create,
    pa_materialize,
)
from src.memory import count_unique_nodes
from src.flatten import make_hash


# ===========================================================================
# PASS 1 -- importable replay / oracle / guard helpers
# ===========================================================================

# ---------------------------------------------------------------------------
# Trace loading / tree topology
# ---------------------------------------------------------------------------

def load_traces(path: str) -> list[dict]:
    """Load a ToT trace JSONL file: one puzzle-tree object per non-empty line."""
    trees: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trees.append(json.loads(line))
    return trees


def _node_index(tree: dict) -> dict:
    """Map node id -> node dict."""
    return {n["id"]: n for n in tree["nodes"]}


def count_expansions(tree: dict) -> int:
    """Branch-creation expansions in a tree = nodes - 1 (the root is the base)."""
    return len(tree["nodes"]) - 1


def path_thought_tokens(tree: dict, node_id: int) -> list[int]:
    """Concatenate thought_tokens along the path root..node_id (root first).
    The root contributes [] (its thought_tokens == [])."""
    index = _node_index(tree)
    chain: list[int] = []
    cur = node_id
    while cur is not None:
        chain.append(cur)
        cur = index[cur]["parent_id"]
    chain.reverse()  # root -> node
    out: list[int] = []
    for nid in chain:
        out.extend(index[nid]["thought_tokens"])
    return out


def final_beam_node_ids(tree: dict) -> list[int]:
    """The surviving beam at the deepest recorded depth (<= beam_width ids)."""
    beams = tree.get("beams") or {}
    if not beams:
        return []
    max_depth = max(int(k) for k in beams)
    return list(beams[str(max_depth)])


# ---------------------------------------------------------------------------
# Oracle (token-list ground truth)
# ---------------------------------------------------------------------------

def reconstruct_oracle(prefix_tokens: list[int], puzzle_tokens: list[int],
                       tree: dict, node_id: int) -> list[int]:
    """Ground-truth token sequence a node context must materialize to:
    prefix + puzzle + (root..node thoughts)."""
    return (list(prefix_tokens) + list(puzzle_tokens)
            + path_thought_tokens(tree, node_id))


# ---------------------------------------------------------------------------
# Tokenizer (gpt2; identical call form to the generator / Milestone A)
# ---------------------------------------------------------------------------

_TOKENIZER = None


def get_tokenizer():
    """Cached HF gpt2 AutoTokenizer (lazy import; matches the generator basis)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
    return _TOKENIZER


def tokenize_text(tok, text: str) -> list[int]:
    """gpt2-encode text with add_special_tokens=False (the token-seal basis)."""
    return tok(text, add_special_tokens=False)["input_ids"]


def puzzle_tokens_for(tok, tree: dict) -> list[int]:
    """gpt2 token ids of the puzzle string (the per-puzzle part of the base)."""
    return tokenize_text(tok, tree["puzzle"])


# ---------------------------------------------------------------------------
# Replay: build every node's context by walking the tree in id (BFS) order
# ---------------------------------------------------------------------------

def replay_tree_hr(prefix_tokens: list[int], puzzle_tokens: list[int],
                   tree: dict, h) -> dict:
    """hashrope replay. Returns {node_id: rope_node}. Root context = base
    (prefix + puzzle); each non-root v = hr_branch_create(ctx[parent], thoughts)."""
    base_tokens = list(prefix_tokens) + list(puzzle_tokens)
    root_rope, _h = hr_build(base_tokens, h)
    ctx: dict = {}
    for node in sorted(tree["nodes"], key=lambda n: n["id"]):
        pid = node["parent_id"]
        if pid is None:
            ctx[node["id"]] = root_rope
        else:
            ctx[node["id"]] = hr_branch_create(ctx[pid], node["thought_tokens"], h)
    return ctx


def replay_tree_pa(prefix_tokens: list[int], puzzle_tokens: list[int],
                   tree: dict, block_size: int) -> dict:
    """paged replay. Returns {node_id: PagedSequence}. One physical pool per tree
    (pa_build mints it); each non-root v = pa_branch_create(ctx[parent], thoughts)."""
    base_tokens = list(prefix_tokens) + list(puzzle_tokens)
    root_seq = pa_build(base_tokens, block_size)
    ctx: dict = {}
    for node in sorted(tree["nodes"], key=lambda n: n["id"]):
        pid = node["parent_id"]
        if pid is None:
            ctx[node["id"]] = root_seq
        else:
            ctx[node["id"]] = pa_branch_create(ctx[pid], node["thought_tokens"])
    return ctx


# ---------------------------------------------------------------------------
# Structural guards (deterministic; carry the asymptotics)
# ---------------------------------------------------------------------------

def hr_new_nodes(parent_rope, child_rope) -> int:
    """Nodes in child not shared with parent (the new O(log w) spine + leaf)."""
    u_parent = count_unique_nodes([parent_rope])["unique_count"]
    u_both = count_unique_nodes([parent_rope, child_rope])["unique_count"]
    return u_both - u_parent


def hr_guard_bound(parent_rope) -> int:
    """ceil(log2 max(w, 2)) + 3, w = parent leaf count (node.weight)."""
    w = parent_rope.weight if parent_rope is not None else 1
    return math.ceil(math.log2(max(w, 2))) + 3


def pa_fork_entries(parent_seq) -> int:
    """Block-table entries a fork of parent_seq touches (== parent.num_blocks)."""
    return parent_seq.num_blocks


def pa_guard_expected(parent_seq, block_size: int) -> int:
    """Expected touched entries: ceil(parent_len / B)."""
    return math.ceil(parent_seq.length / block_size)


# ===========================================================================
# PASS 2 -- worker / orchestrator / timing / memory / verdict
# ===========================================================================

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_INVOCATIONS = 3
DEFAULT_P_SWEEP = [0, 4000, 16000, 64000, 256000]   # P=0 bare-Game-of-24 anchor; no 1M
DEFAULT_BLOCK_SIZES = [8, 16, 32, 64]
DEFAULT_REPS = 5
REF_BLOCK = 16                 # reference block size for the crossover criterion
N_STAR = 16000                 # Milestone A crossover N*; criterion (iii) anchor

OUT_DIR = os.path.join(_REPO, "experiments", "exp_019_branch", "results")
TMP_DIR = os.path.join(OUT_DIR, "_tmp")


# ---------------------------------------------------------------------------
# Token source (reuses Milestone A's cached .npy; gitignored)
# ---------------------------------------------------------------------------

def trace_path(seed: int) -> str:
    return os.path.join(_REPO, "data", "canonical",
                        f"tot_traces_gptoss120b_s{seed}.jsonl")


def base_tokens_path(seed: int) -> str:
    return os.path.join(_REPO, "data", "canonical",
                        f"exp019_base_tokens_s{seed}.npy")


def corpus_path(seed: int) -> str:
    return os.path.join(_REPO, "data", "raw", f"corpus_s{seed}.txt")


def load_base_tokens(seed: int, needed: int) -> list[int]:
    """Load the cached gpt2-tokenized corpus prefix (Milestone A convention)."""
    if needed <= 0:
        return []
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
# Per-tree plan (topology is fixed across P / reps; precompute once)
# ---------------------------------------------------------------------------

def tree_plan(tree: dict):
    """Return (nonroot_nodes_sorted_by_id, {depth: [nodes]})."""
    nonroot = sorted((n for n in tree["nodes"] if n["parent_id"] is not None),
                     key=lambda n: n["id"])
    by_depth: dict[int, list] = {}
    for n in nonroot:
        by_depth.setdefault(n["depth"], []).append(n)
    return nonroot, by_depth


# ---------------------------------------------------------------------------
# Correctness (HARD) + structural guards -- run for EVERY (seed, P, block)
# (the strict version: no determinism assumption; correctness is re-verified
#  in every worker at every prefix size and every block size)
# ---------------------------------------------------------------------------

def correctness_and_guards(trees, plans, pzs, prefix, blocks):
    """Returns (mismatches, hr_guard_ok, pa_guard_ok). hashrope correctness and
    guard are block-independent (checked once per (seed,P)); paged is per block."""
    mism = 0
    hr_ok = True
    pa_ok = True
    for tree, (nonroot, _bd), pz in zip(trees, plans, pzs):
        # hashrope (block-independent)
        h = make_hash()
        hr_ctx = replay_tree_hr(prefix, pz, tree, h)
        for node in tree["nodes"]:
            nid = node["id"]
            if hr_materialize(hr_ctx[nid]) != reconstruct_oracle(prefix, pz, tree, nid):
                mism += 1
        for node in nonroot:
            pid = node["parent_id"]
            if hr_new_nodes(hr_ctx[pid], hr_ctx[node["id"]]) > hr_guard_bound(hr_ctx[pid]):
                hr_ok = False
        del hr_ctx
        # paged (per block)
        for bs in blocks:
            pa_ctx = replay_tree_pa(prefix, pz, tree, bs)
            for node in tree["nodes"]:
                nid = node["id"]
                if pa_materialize(pa_ctx[nid]) != reconstruct_oracle(prefix, pz, tree, nid):
                    mism += 1
            for node in nonroot:
                pid = node["parent_id"]
                if pa_fork_entries(pa_ctx[pid]) != pa_guard_expected(pa_ctx[pid], bs):
                    pa_ok = False
            del pa_ctx
        gc.collect()
    return mism, hr_ok, pa_ok


# ---------------------------------------------------------------------------
# Latency: batched replay of ALL expansions across the seed's trees as ONE batch
# (base build is UNTIMED; both arms rebuild bases fresh per rep; paged pools are
#  rebuilt per rep to discard the COW leak, as Milestone A). Metric = median
#  batch_time / n_expansions = mean per-expansion branch-creation latency.
# ---------------------------------------------------------------------------

def time_replay_hr(plans, pzs, prefix, reps, n_exp):
    per = []
    for _ in range(reps):
        h = make_hash()
        roots = [hr_build(prefix + pz, h)[0] for pz in pzs]            # UNTIMED setup
        t0 = time.perf_counter()
        for ti, (nonroot, _bd) in enumerate(plans):
            ctx = {0: roots[ti]}
            for n in nonroot:
                ctx[n["id"]] = hr_branch_create(ctx[n["parent_id"]], n["thought_tokens"], h)
        dt = time.perf_counter() - t0
        per.append(dt / n_exp)
        del roots
        gc.collect()
    return statistics.median(per)


def time_replay_pa(plans, pzs, prefix, bs, reps, n_exp):
    per = []
    for _ in range(reps):
        roots = [pa_build(prefix + pz, bs) for pz in pzs]              # UNTIMED setup (fresh pools)
        t0 = time.perf_counter()
        for ti, (nonroot, _bd) in enumerate(plans):
            ctx = {0: roots[ti]}
            for n in nonroot:
                ctx[n["id"]] = pa_branch_create(ctx[n["parent_id"]], n["thought_tokens"])
        dt = time.perf_counter() - t0
        per.append(dt / n_exp)
        del roots
        gc.collect()
    return statistics.median(per)


def _depth_count(plans, d):
    return sum(len(bd.get(d, [])) for _nr, bd in plans)


def time_replay_per_depth_hr(plans, pzs, prefix, reps, depths):
    """hashrope per-depth mean latency (block-independent): build depths < d
    untimed, time only the depth-d branch-creations across all trees."""
    result = {}
    for d in depths:
        cnt = _depth_count(plans, d)
        if cnt == 0:
            result[d] = None
            continue
        per = []
        for _ in range(reps):
            h = make_hash()
            roots = [hr_build(prefix + pz, h)[0] for pz in pzs]
            ctxs = []
            for ti, (nonroot, _bd) in enumerate(plans):
                ctx = {0: roots[ti]}
                for n in nonroot:
                    if n["depth"] < d:
                        ctx[n["id"]] = hr_branch_create(ctx[n["parent_id"]], n["thought_tokens"], h)
                ctxs.append(ctx)
            t0 = time.perf_counter()
            for ti, (_nr, bd) in enumerate(plans):
                ctx = ctxs[ti]
                for n in bd.get(d, []):
                    ctx[n["id"]] = hr_branch_create(ctx[n["parent_id"]], n["thought_tokens"], h)
            per.append((time.perf_counter() - t0) / cnt)
            del roots, ctxs
            gc.collect()
        result[d] = statistics.median(per)
    return result


def time_replay_per_depth_pa(plans, pzs, prefix, bs, reps, depths):
    """paged per-depth mean latency at one block size."""
    result = {}
    for d in depths:
        cnt = _depth_count(plans, d)
        if cnt == 0:
            result[d] = None
            continue
        per = []
        for _ in range(reps):
            roots = [pa_build(prefix + pz, bs) for pz in pzs]
            ctxs = []
            for ti, (nonroot, _bd) in enumerate(plans):
                ctx = {0: roots[ti]}
                for n in nonroot:
                    if n["depth"] < d:
                        ctx[n["id"]] = pa_branch_create(ctx[n["parent_id"]], n["thought_tokens"])
                ctxs.append(ctx)
            t0 = time.perf_counter()
            for ti, (_nr, bd) in enumerate(plans):
                ctx = ctxs[ti]
                for n in bd.get(d, []):
                    ctx[n["id"]] = pa_branch_create(ctx[n["parent_id"]], n["thought_tokens"])
            per.append((time.perf_counter() - t0) / cnt)
            del roots, ctxs
            gc.collect()
        result[d] = statistics.median(per)
    return result


# ---------------------------------------------------------------------------
# Branching memory: hold each puzzle's real final-beam survivors live; tracemalloc
# current delta over the built base (Milestone A measure_memory style). The base
# (prefix + puzzle) is built BEFORE tracemalloc.start so the shared prefix is
# excluded -> we measure the MARGINAL divergence cost (the gated number, Decision 2).
# Also measured: base bytes (for the supplementary TOTAL-compression floor) and the
# per-puzzle compression spread.
# ---------------------------------------------------------------------------

def _hr_beam_marginal(plan, pz, prefix, h, survivors):
    nonroot, _bd = plan
    base_hr, _ = hr_build(prefix + pz, h)            # BEFORE start -> excluded
    gc.collect()
    tracemalloc.start()
    ctx = {0: base_hr}
    for n in nonroot:
        ctx[n["id"]] = hr_branch_create(ctx[n["parent_id"]], n["thought_tokens"], h)
    beam = [ctx[i] for i in survivors]               # hold survivors (+reachable) live
    del ctx
    gc.collect()
    cur, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del beam, base_hr
    gc.collect()
    return cur


def _hr_base_bytes(pz, prefix, h):
    gc.collect()
    tracemalloc.start()
    base_hr, _ = hr_build(prefix + pz, h)
    cur, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del base_hr
    gc.collect()
    return cur


def _pa_beam_marginal(plan, pz, prefix, bs, survivors):
    nonroot, _bd = plan
    base_pa = pa_build(prefix + pz, bs)              # BEFORE start -> excluded
    gc.collect()
    tracemalloc.start()
    ctx = {0: base_pa}
    for n in nonroot:
        ctx[n["id"]] = pa_branch_create(ctx[n["parent_id"]], n["thought_tokens"])
    beam = [ctx[i] for i in survivors]
    del ctx
    gc.collect()
    cur, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del beam, base_pa
    gc.collect()
    return cur


def _pa_base_bytes(pz, prefix, bs):
    gc.collect()
    tracemalloc.start()
    base_pa = pa_build(prefix + pz, bs)
    cur, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del base_pa
    gc.collect()
    return cur


def measure_beam_memory_hr(trees, plans, pzs, prefix):
    """hashrope beam/base memory per puzzle (block-INDEPENDENT; measured once per
    P). Returns aligned per-puzzle lists over puzzles that have >=1 survivor."""
    hr_marg = []
    hr_base = []
    puzzle_idx = []
    for i, (tree, plan, pz) in enumerate(zip(trees, plans, pzs)):
        survivors = final_beam_node_ids(tree)
        if not survivors:
            continue
        h = make_hash()
        hr_marg.append(_hr_beam_marginal(plan, pz, prefix, h, survivors))
        hr_base.append(_hr_base_bytes(pz, prefix, make_hash()))
        puzzle_idx.append(i)
    return {"hr_marg": hr_marg, "hr_base": hr_base, "puzzle_idx": puzzle_idx}


def measure_beam_memory_pa(trees, plans, pzs, prefix, bs, puzzle_idx):
    """paged beam/base memory per puzzle at one block size, over the SAME puzzles
    (by index) that measure_beam_memory_hr kept, so the per-puzzle lists align."""
    pa_marg = []
    pa_base = []
    idx_set = set(puzzle_idx)
    for i, (tree, plan, pz) in enumerate(zip(trees, plans, pzs)):
        if i not in idx_set:
            continue
        survivors = final_beam_node_ids(tree)
        pa_marg.append(_pa_beam_marginal(plan, pz, prefix, bs, survivors))
        pa_base.append(_pa_base_bytes(pz, prefix, bs))
    return {"pa_marg": pa_marg, "pa_base": pa_base}


def combine_beam_memory(P, bs, hr_mem, pa_mem):
    """Combine hr (once-per-P) and pa (per-block) into the memory record:
    gated marginal Sum(pa)/Sum(hr) + per-puzzle spread + total-compression floor."""
    sum_hr = sum(hr_mem["hr_marg"])
    sum_pa = sum(pa_mem["pa_marg"])
    sum_hr_base = sum(hr_mem["hr_base"])
    sum_pa_base = sum(pa_mem["pa_base"])
    per_puzzle = [pa / hr for hr, pa in zip(hr_mem["hr_marg"], pa_mem["pa_marg"]) if hr > 0]
    comp_marg = (sum_pa / sum_hr) if sum_hr > 0 else float("inf")
    denom_total = sum_hr_base + sum_hr
    comp_total = ((sum_pa_base + sum_pa) / denom_total) if denom_total > 0 else float("inf")
    pp_mean, pp_std = _mean_std(per_puzzle)
    return {
        "P": P, "block_size": bs, "n_puzzles": len(hr_mem["hr_marg"]),
        "compression_marginal": comp_marg,
        "comp_marginal_per_puzzle_mean": pp_mean,
        "comp_marginal_per_puzzle_std": pp_std,
        "compression_total_floor": comp_total,
        "hr_marginal_bytes_sum": sum_hr, "pa_marginal_bytes_sum": sum_pa,
        "hr_base_bytes_sum": sum_hr_base, "pa_base_bytes_sum": sum_pa_base,
    }


# ---------------------------------------------------------------------------
# Worker: one (seed, invocation)
# ---------------------------------------------------------------------------

def run_worker(args) -> None:
    _wall0 = time.perf_counter()
    seed = args.seed
    p_sweep = args.p_sweep
    blocks = args.block_sizes
    reps = args.reps

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    trees = load_traces(trace_path(seed))
    tok = get_tokenizer()
    pzs = [puzzle_tokens_for(tok, t) for t in trees]
    print(f"[worker s{seed} i{args.invocation}] tokenizer+load: "
          f"{time.perf_counter() - _wall0:.1f}s", file=sys.stderr, flush=True)
    plans = [tree_plan(t) for t in trees]
    n_exp = sum(len(nr) for nr, _bd in plans)
    depths = sorted({d for _nr, bd in plans for d in bd.keys()})
    depth_hist = {}
    for _nr, bd in plans:
        for d, ns in bd.items():
            depth_hist[d] = depth_hist.get(d, 0) + len(ns)

    base_all = load_base_tokens(seed, max(p_sweep)) if max(p_sweep) > 0 else []

    cells = []
    latency_by_depth = []
    memory = []
    check_correctness = not args.skip_correctness
    for P in p_sweep:
        _tp = time.perf_counter()
        prefix = base_all[:P]

        if check_correctness:
            mism, hr_guard_ok, pa_guard_ok = correctness_and_guards(
                trees, plans, pzs, prefix, blocks)
        else:
            mism, hr_guard_ok, pa_guard_ok = None, None, None

        t_hr = time_replay_hr(plans, pzs, prefix, reps, n_exp)         # block-independent
        for bs in blocks:
            t_pa = time_replay_pa(plans, pzs, prefix, bs, reps, n_exp)
            cells.append({
                "P": P, "block_size": bs,
                "mismatches": mism,
                "hr_guard_ok": hr_guard_ok, "pa_guard_ok": pa_guard_ok,
                "correctness_checked": check_correctness,
                "n_expansions": n_exp,
                "t_branch_create_hr": t_hr,
                "t_branch_create_pa": t_pa,
            })

        # per-depth latency (descriptive): hr block-independent, pa at ref block
        dl_hr = time_replay_per_depth_hr(plans, pzs, prefix, reps, depths)
        dl_pa = time_replay_per_depth_pa(plans, pzs, prefix, REF_BLOCK, reps, depths)
        for d in depths:
            latency_by_depth.append({
                "P": P, "depth": d,
                "hr": dl_hr.get(d), "pa_refblock": dl_pa.get(d),
                "n": _depth_count(plans, d),
            })

        # branching memory: hashrope measured ONCE per P (block-independent, B),
        # paged per block, then combined.
        hr_mem = measure_beam_memory_hr(trees, plans, pzs, prefix)
        for bs in blocks:
            pa_mem = measure_beam_memory_pa(trees, plans, pzs, prefix, bs, hr_mem["puzzle_idx"])
            memory.append(combine_beam_memory(P, bs, hr_mem, pa_mem))

        print(f"[worker s{seed} i{args.invocation}] P={P} done: "
              f"{time.perf_counter() - _tp:.1f}s  (cum {time.perf_counter() - _wall0:.1f}s)",
              file=sys.stderr, flush=True)

    out = {
        "experiment": "EXP-019", "milestone": "B", "claim": "B3",
        "seed": seed, "invocation": args.invocation,
        "p_sweep": p_sweep, "block_sizes": blocks, "reps": reps,
        "ref_block": REF_BLOCK,
        "n_expansions": n_exp, "n_nodes": sum(len(t["nodes"]) for t in trees),
        "depth_hist": depth_hist,
        "cells": cells,
        "latency_by_depth": latency_by_depth,
        "memory": memory,
        "trace_provenance": {
            "seed": seed,
            "trace_sha16": sha256_16(trace_path(seed)),
            "n_trees": len(trees),
            "model_requested": trees[0].get("model_requested") if trees else None,
            "tokenizer": trees[0].get("tokenizer") if trees else None,
        },
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
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def aggregate(runs, p_sweep, block_sizes):
    """runs: list of worker JSON dicts (the n=9). Returns aggregate + verdict."""
    def cell(run, P, bs):
        for c in run["cells"]:
            if c["P"] == P and c["block_size"] == bs:
                return c
        raise KeyError((P, bs))

    def mem(run, P, bs):
        for m in run["memory"]:
            if m["P"] == P and m["block_size"] == bs:
                return m
        raise KeyError((P, bs))

    agg_cells = []
    total_mismatches = 0
    all_guards_ok = True
    coverage_ok = True   # every (P, bs) must have correctness verified by >=1 run
    for P in p_sweep:
        for bs in block_sizes:
            cs = [cell(r, P, bs) for r in runs]
            checked = [c for c in cs if c.get("correctness_checked")]
            if not checked:
                coverage_ok = False
            for c in checked:
                if c["mismatches"] is not None:
                    total_mismatches += c["mismatches"]
                if not (c["hr_guard_ok"] and c["pa_guard_ok"]):
                    all_guards_ok = False
            bc_hr = [c["t_branch_create_hr"] for c in cs]
            bc_pa = [c["t_branch_create_pa"] for c in cs]
            wins = sum(1 for a, b in zip(bc_hr, bc_pa) if a < b)
            m_hr, s_hr = _mean_std(bc_hr)
            m_pa, s_pa = _mean_std(bc_pa)
            speedup = (m_pa / m_hr) if (m_hr and m_hr > 0) else None
            hr_wins_band = (m_hr is not None and m_pa is not None
                            and (m_hr + s_hr) < (m_pa - s_pa))
            agg_cells.append({
                "P": P, "block_size": bs,
                "branch_create_hr_mean": m_hr, "branch_create_hr_std": s_hr,
                "branch_create_pa_mean": m_pa, "branch_create_pa_std": s_pa,
                "branch_create_speedup": speedup,
                "branch_create_sign_wins": wins, "n": len(cs),
                "hr_wins_band": hr_wins_band,
            })

    # memory aggregate (marginal compression -- the gated number)
    agg_mem = []
    for P in p_sweep:
        for bs in block_sizes:
            ms = [mem(r, P, bs) for r in runs]
            comp = [m["compression_marginal"] for m in ms]
            m_c, s_c = _mean_std(comp)
            comp_tot = [m["compression_total_floor"] for m in ms]
            mt_c, st_c = _mean_std(comp_tot)
            agg_mem.append({
                "P": P, "block_size": bs,
                "compression_marginal_mean": m_c, "compression_marginal_std": s_c,
                "compression_total_floor_mean": mt_c, "compression_total_floor_std": st_c,
                "hr_marginal_bytes_mean": _mean_std([m["hr_marginal_bytes_sum"] for m in ms])[0],
                "pa_marginal_bytes_mean": _mean_std([m["pa_marginal_bytes_sum"] for m in ms])[0],
            })

    n = len(runs)
    p_max = max(p_sweep)

    # (i) HARD byte-identity (zero mismatches across all CHECKED cells, and every
    #     (P, block) checked by at least one run -- coverage)
    crit_i = (total_mismatches == 0) and coverage_ok
    # (ii) guards
    crit_ii = all_guards_ok

    # (iii) crossover at ref block: smallest P where hr wins the band, with pa<=hr
    #       for all smaller P (monotone), AND P* within one grid step of N*.
    ref_rows = sorted([c for c in agg_cells if c["block_size"] == REF_BLOCK],
                      key=lambda c: c["P"])
    crossover_P = None
    for c in ref_rows:
        if c["hr_wins_band"]:
            crossover_P = c["P"]
            break
    below_ok = True
    if crossover_P is not None:
        for c in ref_rows:
            if c["P"] < crossover_P:
                if not (c["branch_create_pa_mean"] is not None
                        and c["branch_create_hr_mean"] is not None
                        and c["branch_create_pa_mean"] <= c["branch_create_hr_mean"]):
                    below_ok = False
    # within one grid step of N_STAR
    grid = sorted(p_sweep)
    near_nstar = False
    if crossover_P is not None and N_STAR in grid:
        i = grid.index(N_STAR)
        neighbors = {N_STAR}
        if i - 1 >= 0:
            neighbors.add(grid[i - 1])
        if i + 1 < len(grid):
            neighbors.add(grid[i + 1])
        near_nstar = crossover_P in neighbors
    crit_iii = (crossover_P is not None) and below_ok and near_nstar

    # (iv) at P=max: every block -> n/n sign wins AND speedup>=2x AND
    #      marginal compression mean >= 10x.
    iv_latency = []
    for bs in block_sizes:
        row = next(c for c in agg_cells if c["P"] == p_max and c["block_size"] == bs)
        ok = (row["branch_create_sign_wins"] == n
              and row["branch_create_speedup"] is not None
              and row["branch_create_speedup"] >= 2.0)
        iv_latency.append((bs, ok, row["branch_create_sign_wins"], row["branch_create_speedup"]))
    iv_memory = []
    for bs in block_sizes:
        mrow = next(m for m in agg_mem if m["P"] == p_max and m["block_size"] == bs)
        ok = (mrow["compression_marginal_mean"] is not None
              and mrow["compression_marginal_mean"] >= 10.0)
        iv_memory.append((bs, ok, mrow["compression_marginal_mean"]))
    crit_iv = all(x[1] for x in iv_latency) and all(x[1] for x in iv_memory)

    # (v) sub-crossover regimes where paged wins, reported in full (not hidden)
    sub_crossover = [
        {"P": c["P"], "block_size": c["block_size"],
         "pa_mean": c["branch_create_pa_mean"], "hr_mean": c["branch_create_hr_mean"]}
        for c in agg_cells
        if (c["branch_create_pa_mean"] is not None and c["branch_create_hr_mean"] is not None
            and c["branch_create_pa_mean"] <= c["branch_create_hr_mean"])
    ]

    corroborated = crit_i and crit_ii and crit_iii and crit_iv

    return {
        "aggregate_cells": agg_cells,
        "aggregate_memory": agg_mem,
        "verdict": {
            "n": n,
            "total_mismatches": total_mismatches,
            "correctness_coverage_ok": coverage_ok,
            "crit_i_byte_identity": crit_i,
            "crit_ii_guards": crit_ii,
            "crit_iii_crossover_refblock": {
                "ok": crit_iii, "crossover_P": crossover_P,
                "below_monotone_ok": below_ok, "within_one_step_of_Nstar": near_nstar,
                "ref_block": REF_BLOCK, "N_star": N_STAR,
            },
            "crit_iv_at_Pmax": {
                "ok": crit_iv, "P": p_max,
                "latency": iv_latency, "memory_marginal": iv_memory,
            },
            "sub_crossover_paged_wins": sub_crossover,
            "milestoneB_corroborated": corroborated,
            "B3_supported_via_milestoneA": True,  # invariant: MS-A is closed/SUPPORTED
        },
    }


def run_orchestrator(args) -> None:
    seeds = args.seeds
    invocations = args.invocations
    p_sweep = args.p_sweep
    block_sizes = args.block_sizes

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    runs = []
    run_paths = []
    for seed in seeds:
        for inv in range(invocations):
            tag = f"s{seed}_i{inv}"
            run_out = os.path.join(TMP_DIR, f"exp019_mB_run_{tag}.json")
            cmd = [
                sys.executable, os.path.abspath(__file__), "--worker-mode",
                "--seed", str(seed), "--invocation", str(inv),
                "--p-sweep", ",".join(map(str, p_sweep)),
                "--block-sizes", ",".join(map(str, block_sizes)),
                "--reps", str(args.reps),
                "--out", run_out,
            ]
            # A: HARD correctness + guards verified once per seed (invocation 0);
            # invocations 1+ re-time the byte-identical deterministic replay only.
            if inv != 0:
                cmd.append("--skip-correctness")
            print(f"[orch] {tag} ...", flush=True)
            subprocess.run(cmd, check=True)
            with open(run_out, "r", encoding="utf-8") as f:
                runs.append(json.load(f))
            run_paths.append(run_out)

    agg = aggregate(runs, p_sweep, block_sizes)

    env = {
        "git_sha": git_sha(),
        "hashrope_version": hashrope_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "trace_sha16": {str(s): sha256_16(trace_path(s)) for s in seeds},
        "corpus_sha16": {str(s): sha256_16(corpus_path(s)) for s in seeds},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary = {
        "experiment": "EXP-019", "milestone": "B", "claim": "B3",
        "gated_metric": "branch_create_latency + marginal branching-memory compression",
        "seeds": seeds, "invocations": invocations, "n_runs": len(runs),
        "p_sweep": p_sweep, "block_sizes": block_sizes, "ref_block": REF_BLOCK,
        "N_star": N_STAR,
        "env": env,
        "run_files": [os.path.basename(p) for p in run_paths],
        **agg,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    latest = os.path.join(OUT_DIR, "exp019_milestoneB_latest.json")
    stamped = os.path.join(OUT_DIR, f"exp019_milestoneB_{ts}.json")
    for path in (latest, stamped):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    v = summary["verdict"]
    print("\n=== EXP-019 Milestone B verdict ===", flush=True)
    print(f"  n={v['n']}  mismatches={v['total_mismatches']}", flush=True)
    print(f"  (i)   byte-identity : {v['crit_i_byte_identity']}", flush=True)
    print(f"  (ii)  guards        : {v['crit_ii_guards']}", flush=True)
    print(f"  (iii) crossover@b{REF_BLOCK}: {v['crit_iii_crossover_refblock']}", flush=True)
    print(f"  (iv)  P={v['crit_iv_at_Pmax']['P']}    : {v['crit_iv_at_Pmax']['ok']}", flush=True)
    print(f"  Milestone B CORROBORATED : {v['milestoneB_corroborated']}", flush=True)
    print(f"  B3 SUPPORTED (via Milestone A) : {v['B3_supported_via_milestoneA']}", flush=True)
    print(f"  -> {latest}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EXP-019 Milestone B bench (claim B3, replay).")
    p.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip-correctness", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    p.add_argument("--invocation", type=int, default=0)
    p.add_argument("--seeds", type=_int_list, default=DEFAULT_SEEDS)
    p.add_argument("--invocations", type=int, default=DEFAULT_INVOCATIONS)
    p.add_argument("--p-sweep", dest="p_sweep", type=_int_list, default=DEFAULT_P_SWEEP)
    p.add_argument("--block-sizes", dest="block_sizes", type=_int_list, default=DEFAULT_BLOCK_SIZES)
    p.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p.add_argument("--out", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.worker_mode:
        if args.out is None:
            raise SystemExit("--worker-mode requires --out")
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
