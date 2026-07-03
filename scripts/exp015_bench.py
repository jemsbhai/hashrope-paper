#!/usr/bin/env python3
"""
EXP-015 driver (claim S3) -- two-layer energy/identification bench on a real
vLLM serving stack. Orchestrator + worker (one fresh process per cell).

Decision (LOGBOOK EXP-015, confirmed 2026-06-18): FULL RELAUNCH per invocation
on both tracks -- the pre-registered "fresh process invocations" stands. Each
cell launches its own vLLM engine, runs one steady-state window, tears down.

GPU allocation (regime-dependent, pre-registered):
  R1 (realistic streams, ~2k-token prefixes): single-GPU cells run 4-way in
     PARALLEL across the GPUs (each worker pinned via CUDA_VISIBLE_DEVICES +
     CUDA_DEVICE_ORDER=PCI_BUS_ID; NVML measures that worker's own physical
     index; never summed across GPUs) to cut wall-clock.
  R2 (long-shared-prefix synthetic, L straddling ~571k): cells run SEQUENTIALLY
     with tensor parallelism tp=len(gpus) for KV capacity past ~400k tokens;
     energy summed over the TP working group.

Run ONE regime per SLURM job (both fit well under a 24 h cap).

Two orthogonal measurement tracks per cell, sharing the request stream:
  Track A (engine): cache {OFF, ON} -> GPU Joules/token, TTFT, throughput,
     cached_tokens. The cache axis is the ONLY thing that changes the GPU work.
  Track B (identification, CPU, cache-independent): the three identifier arms
     {radix, hashrope, flat} -> matched-prefix length (correctness gate, i) +
     identification latency (cost regimes, iv). Timed strictly OUTSIDE the GPU
     energy window (confound control).

This driver writes RAW per-cell + combined JSON only. Aggregation + the verbatim
promotion-criterion evaluation are a SEPARATE analysis step (run on our machine
after the collaborator returns the artifacts) -- never inline, never retrofit.

Validatable off-GPU: stream building, Track B (oracle/hashrope/flat), the R1
parallel scheduler, the raw JSON schema, arg-parsing, and the --dry-run cell
path all run on CPU. --synthetic bypasses the tokenizer for pipeline validation.
The vLLM/NVML execution is the collaborator's smoke gate.

Usage (confirmatory, per regime):
    python scripts/exp015_bench.py --regime R1 --gpus 0,1,2,3 \
        --model Qwen/Qwen2.5-7B-Instruct-1M --seeds 42,43,44 --invocations 3
    python scripts/exp015_bench.py --regime R2 --gpus 0,1,2,3 \
        --model Qwen/Qwen2.5-7B-Instruct-1M --seeds 42,43,44 --invocations 3

Smoke (de-risks the binding constraint on the real 1M model before the grid):
    python scripts/exp015_bench.py --regime R1 --smoke --gpus 0 \
        --model Qwen/Qwen2.5-7B-Instruct-1M
    python scripts/exp015_bench.py --regime R2 --smoke --gpus 0,1,2,3 \
        --model Qwen/Qwen2.5-7B-Instruct-1M

Off-GPU pipeline check (no engine, no tokenizer):
    python scripts/exp015_bench.py --regime R1 --smoke --dry-run --synthetic --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pre-registered defaults (LOGBOOK EXP-015 IVs)
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_R2_L = [131_072, 262_144, 393_216, 524_288, 1_000_000]  # ~128k..1M (top=1e6 = Qwen2.5-1M usable ceiling), straddles ~571k
SMOKE_R2_L = [1_000_000]  # smoke the binding constraint: 1M model at its near-ceiling L
DEFAULT_R1_DATASETS = ["sharegpt_sample.jsonl", "lmsys_sample.jsonl"]
TAIL_TOKENS = 1024
TIMING_REPS = 5
RESULTS_DIR_NAME = "exp_015_energy"


# ========================= shared helpers ================================ #

def _stats(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {}
    return {"mean": statistics.mean(xs), "std": statistics.pstdev(xs),
            "min": min(xs), "max": max(xs), "n": len(xs), "values": xs}


def _sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def collect_env() -> dict:
    env = {"platform": platform.platform(), "python": platform.python_version(),
           "timestamp": datetime.now(timezone.utc).isoformat()}
    try:
        import hashrope
        env["hashrope_version"] = hashrope.__version__
    except Exception:
        env["hashrope_version"] = "unavailable"
    try:
        import numpy as np
        env["numpy_version"] = np.__version__
    except Exception:
        pass
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
    except Exception:
        env["torch_version"] = "unavailable"
    try:
        from src.exp015_serving import engine_name, engine_version
        env["engine"] = engine_name()
        env["engine_version"] = engine_version()
    except Exception:
        env["engine"] = "unknown"
        env["engine_version"] = "unavailable"
    return env


def collect_git(root: Path):
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                               capture_output=True, text=True).stdout.strip()
        return sha or "unknown", bool(dirty)
    except Exception:
        return "unknown", True


def corpus_path(root: Path, seed: int) -> Path:
    return root / "data" / "raw" / f"corpus_s{seed}.txt"


def _tokenize_with(model_path: str, text: str) -> list[int]:
    """Tokenize text with the SERVING MODEL's tokenizer (EXP-015 uses the serving
    model's tokenizer throughout so identifier IDs == served IDs). Lazy import."""
    import logging
    from transformers import AutoTokenizer
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tok.encode(text)


# ========================= stream building =============================== #

def build_stream(regime: str, args, seed: int, L: int | None) -> dict:
    """Build the request stream for one cell. Reuses the identification core's
    R1/R2 builders. --synthetic bypasses the tokenizer (pipeline validation)."""
    from src.exp015_identify import build_r2_stream

    if regime == "R2":
        n_queries = args.r2_queries
        if args.synthetic:
            rng = random.Random(seed)
            need = (L or 0) + TAIL_TOKENS + n_queries * TAIL_TOKENS + TAIL_TOKENS
            toks = [rng.randrange(args.vocab) for _ in range(need)]
        else:
            toks = _tokenize_with(args.model, corpus_path(REPO_ROOT, seed).read_text(encoding="utf-8"))
        stream = build_r2_stream(toks, L=L, n_queries=n_queries, seed=seed)
        return stream

    # R1
    n_pairs = args.r1_pairs
    if args.synthetic:
        rng = random.Random(seed)
        items = []
        for k in range(n_pairs):
            clen = rng.randint(256, 2048)
            cached = [rng.randrange(args.vocab) for _ in range(clen)]
            tail = [rng.randrange(args.vocab) for _ in range(rng.randint(8, 64))]
            query = cached + tail            # query extends cached -> oracle LCP == clen
            items.append({"cached_tokens": cached, "query_tokens": query,
                          "oracle_lcp": clen, "conv_id": f"syn{seed}_{k}"})
        return {"regime": "R1", "dataset": "synthetic", "items": items}

    from src.exp015_identify import build_r1_stream
    items = []
    for ds in args.r1_datasets:
        ds_path = REPO_ROOT / "data" / "canonical" / ds
        if not ds_path.exists():
            continue
        s = build_r1_stream(str(ds_path), n_pairs=n_pairs, seed=seed,
                            tokenizer_name=args.model)
        items.extend(s["items"])
    return {"regime": "R1", "dataset": ",".join(args.r1_datasets), "items": items}


def _groups(stream: dict) -> list[tuple[list[int], list[list[int]]]]:
    """Normalise a stream to [(cached, [queries])] groups for identification."""
    if stream["regime"] == "R2":
        return [(stream["cached_tokens"], stream["queries"])]
    return [(it["cached_tokens"], [it["query_tokens"]]) for it in stream["items"]]


# ========================= Track B: identification ======================= #

def run_track_b(stream: dict) -> dict:
    """Three-arm identification over the stream: correctness gate (i) + latency
    (iv). Cache-independent, CPU, OUTSIDE any GPU energy window."""
    from src.exp015_identify import (correctness_gate, identify, prepare,
                                      radix_available, time_identify)

    arms = {"hashrope": {"matched": [], "lat_ms": []},
            "flat": {"matched": [], "lat_ms": []},
            "radix": {"matched": [], "lat_ms": []}}
    oracle_lens: list[int] = []
    n_checks = 0
    mismatches = 0

    for cached, queries in _groups(stream):
        gate = correctness_gate(cached, queries, keep_detail=True)
        n_checks += gate.n_checks
        mismatches += gate.mismatches
        handles = prepare(cached)
        for q in queries:
            r = identify(handles, q)
            oracle_lens.append(r["oracle"])
            for a in ("hashrope", "flat", "radix"):
                if r[a] is not None:
                    arms[a]["matched"].append(r[a])
            t = time_identify(handles, q, reps=TIMING_REPS)
            arms["hashrope"]["lat_ms"].append(t["hashrope_ms"])
            arms["flat"]["lat_ms"].append(t["flat_ms"])
            if t["radix_ms"] is not None:
                arms["radix"]["lat_ms"].append(t["radix_ms"])

    out = {"radix_available": radix_available(),
           "gate": {"ok": mismatches == 0, "n_checks": n_checks,
                    "mismatches": mismatches},
           "oracle_lcp": _stats([float(x) for x in oracle_lens]),
           "arms": {}}
    for a, d in arms.items():
        if d["lat_ms"] or d["matched"]:
            out["arms"][a] = {"lat_ms": _stats(d["lat_ms"]),
                              "matched_lcp": _stats([float(x) for x in d["matched"]])}
    return out


# ========================= Track A: engine ============================== #

def run_track_a(stream: dict, regime: str, cache_on: bool,
                nvml_indices: list[int], tp: int, args) -> dict:
    """Serve the stream under one cache setting; measure GPU energy (NVML window),
    TTFT, throughput, cached_tokens. Skipped (graceful) off-GPU or in --dry-run."""
    from src.exp015_energy_probe import GpuEnergyProbe, energy_path, gpu_metadata
    from src.exp015_serving import (VllmServingArm, engine_available,
                                    engine_name, engine_version)

    if args.dry_run or not engine_available():
        return {"skipped": True,
                "reason": "dry_run" if args.dry_run else "engine_or_gpu_unavailable",
                "engine": engine_name(), "engine_version": engine_version(),
                "energy_path": energy_path(nvml_indices)}

    if regime == "R2":
        cached_entries = [stream["cached_tokens"]]
        queries = stream["queries"]
    else:
        cached_entries = [it["cached_tokens"] for it in stream["items"]]
        queries = [it["query_tokens"] for it in stream["items"]]

    long_ctx_extra = ({"enable_chunked_prefill": True, "max_num_batched_tokens": 131072,
                       "enforce_eager": True, "max_num_seqs": 1}
                      if regime == "R2" else None)  # Qwen2.5-1M recipe; held constant across cache axis (R2 only)
    arm = VllmServingArm(args.model, cache_on=cache_on, tp_size=tp,
                        mem_fraction_static=args.mem_fraction_static,
                        context_length=args.context_length,
                        random_seed=args.seed_for_engine,
                        extra=long_ctx_extra)
    try:
        warm = queries[0][:min(len(queries[0]), 256)] if queries else [1, 2, 3]
        arm.warmup(warm, rounds=2, max_new_tokens=args.max_new_tokens)

        # Populate the cache (untimed setup, OUTSIDE the energy window; identical
        # serves for OFF and ON so only the cache flag differs).
        for ce in cached_entries:
            arm._generate(ce, max_new_tokens=1)

        # TTFT pass (per query; max_new_tokens=1 isolates prefill). Separate from
        # the energy window.
        ttft_ms, cached_per_req = [], []
        for q in queries:
            r = arm.ttft(q)
            ttft_ms.append(r["ttft_s"] * 1e3)
            cached_per_req.append(r["cached_tokens"])

        # Energy + throughput window (steady-state batch).
        gm = gpu_metadata(nvml_indices)
        probe = GpuEnergyProbe(nvml_indices)
        with probe:
            bt = arm.generate_batch(queries, max_new_tokens=args.max_new_tokens)
        er = probe.result()

        out_tok = bt.total_completion_tokens
        jpt = (er.total_joules / out_tok) if (er.total_joules and out_tok) else None
        jpr = (er.total_joules / bt.n) if (er.total_joules and bt.n) else None
        thr_tok = (out_tok / bt.wall_s) if bt.wall_s > 0 else None
        thr_req = (bt.n / bt.wall_s) if bt.wall_s > 0 else None

        return {"skipped": False, "energy_path": er.path, "gpu_metadata": gm,
                "per_gpu_joules": {str(k): v for k, v in er.per_gpu_joules.items()},
                "total_joules": er.total_joules, "window_wall_s": er.wall_s,
                "n_requests": bt.n, "total_completion_tokens": out_tok,
                "total_cached_tokens": bt.total_cached_tokens,
                "j_per_token": jpt, "j_per_request": jpr,
                "throughput_tok_s": thr_tok, "throughput_req_s": thr_req,
                "ttft_ms": _stats(ttft_ms), "cached_tokens_per_req": cached_per_req,
                "n_power_samples": er.n_samples, "engine_kwargs": arm.kwargs,
                "engine": engine_name(), "engine_version": engine_version()}
    finally:
        arm.shutdown()


# ========================= worker (one cell) ============================= #

def cell_id(cell: dict) -> str:
    parts = [cell["regime"], "on" if cell["cache_on"] else "off",
             f"s{cell['seed']}", f"i{cell['inv']}"]
    if cell.get("L") is not None:
        parts.append(f"L{cell['L']}")
    return "_".join(parts)


def run_worker(args) -> None:
    cell = json.loads(args.cell_json)
    regime = cell["regime"]
    seed = cell["seed"]
    args.seed_for_engine = seed * 1000 + cell["inv"]
    stream = build_stream(regime, args, seed, cell.get("L"))
    n_items = stream["n_queries"] if regime == "R2" else len(stream["items"])
    if n_items == 0:
        print("\n".join([
            "", "=" * 72,
            f"EXP-015 ABORT ({cell_id(cell)}) -- request stream is EMPTY (n_items=0).",
            "  No requests to serve or identify, so this cell would record nothing.",
            "  For R1 this usually means the conversation datasets were not found at",
            f"  {REPO_ROOT / 'data' / 'canonical'}",
            f"  (expected: {', '.join(args.r1_datasets)}).",
            "  Commit/push the dataset files and `git pull` on the node, or pass",
            "  --synthetic for a no-data pipeline check. Aborting.",
            "=" * 72,
        ]), flush=True)
        sys.exit(1)

    track_b = run_track_b(stream) if cell.get("run_identification", True) else None
    track_a = run_track_a(stream, regime, cell["cache_on"],
                          cell["nvml_indices"], cell["tp"], args)

    record = {"cell_id": cell_id(cell), "regime": regime,
              "cache_on": cell["cache_on"], "seed": seed, "inv": cell["inv"],
              "L": cell.get("L"), "nvml_indices": cell["nvml_indices"],
              "tp": cell["tp"], "n_items": n_items,
              "dataset": stream.get("dataset"),
              "track_b": track_b, "track_a": track_a}
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"  [worker {cell_id(cell)}] done -> {args.out}", flush=True)


# ========================= orchestrator ================================= #

def build_cells(args) -> list[dict]:
    seeds = args.seeds
    invs = list(range(args.invocations))
    caches = [False, True]
    cells: list[dict] = []
    if args.regime == "R2":
        Ls = args.r2_L
        for cache in caches:
            for seed in seeds:
                for inv in invs:
                    for L in Ls:
                        cells.append({"regime": "R2", "cache_on": cache, "seed": seed,
                                      "inv": inv, "L": L,
                                      "run_identification": (cache is True)})
    else:
        for cache in caches:
            for seed in seeds:
                for inv in invs:
                    cells.append({"regime": "R1", "cache_on": cache, "seed": seed,
                                  "inv": inv, "L": None,
                                  "run_identification": (cache is True)})
    return cells


def _worker_cmd(args, cell: dict, gpus: list[int], tp: int, out: Path) -> list[int]:
    c = dict(cell)
    c["nvml_indices"] = gpus
    c["tp"] = tp
    cmd = [sys.executable, os.path.abspath(__file__), "--worker-mode",
           "--cell-json", json.dumps(c), "--out", str(out),
           "--regime", args.regime, "--model", args.model,
           "--r1-pairs", str(args.r1_pairs), "--r2-queries", str(args.r2_queries),
           "--max-new-tokens", str(args.max_new_tokens),
           "--mem-fraction-static", str(args.mem_fraction_static),
           "--vocab", str(args.vocab),
           "--r1-datasets", ",".join(args.r1_datasets)]
    if args.context_length is not None:
        cmd += ["--context-length", str(args.context_length)]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.synthetic:
        cmd.append("--synthetic")
    return cmd


def _worker_env(gpus: list[int]) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def _collect(outpath: Path, cell: dict) -> dict:
    if outpath.exists():
        try:
            return json.loads(outpath.read_text())
        except Exception:
            pass
    return {"cell_id": cell_id(cell), "error": "no/invalid output"}


def schedule_parallel(args, cells, gpus, tmp_dir, poll=0.5) -> list[dict]:
    """R1: each cell single-GPU; run up to len(gpus) concurrently."""
    free = list(gpus)
    pending = list(cells)
    running: dict[int, tuple] = {}
    results: list[dict] = []
    total = len(cells)
    done = 0
    while pending or running:
        while free and pending:
            gpu = free.pop(0)
            cell = pending.pop(0)
            out = tmp_dir / f"{cell_id(cell)}.json"
            if out.exists() and not args.force:               # resumability
                results.append(_collect(out, cell)); done += 1
                free.append(gpu)
                print(f"[orch] [{done}/{total}] SKIP (done) {cell_id(cell)}", flush=True)
                continue
            cmd = _worker_cmd(args, cell, [gpu], tp=1, out=out)
            p = subprocess.Popen(cmd, env=_worker_env([gpu]))
            running[gpu] = (p, cell, out)
            print(f"[orch] launch {cell_id(cell)} on GPU {gpu}", flush=True)
        time.sleep(poll)
        for gpu in [g for g, (p, _, _) in running.items() if p.poll() is not None]:
            p, cell, out = running.pop(gpu)
            if p.returncode != 0:
                results.append({"cell_id": cell_id(cell), "error": f"worker rc={p.returncode}"})
            else:
                results.append(_collect(out, cell))
            done += 1
            free.append(gpu)
            print(f"[orch] [{done}/{total}] done {cell_id(cell)} (rc={p.returncode})", flush=True)
            time.sleep(args.cooldown)
    return results


def schedule_sequential(args, cells, gpus, tmp_dir) -> list[dict]:
    """R2: each cell uses all GPUs (tp=len(gpus)); run one at a time."""
    results: list[dict] = []
    total = len(cells)
    for i, cell in enumerate(cells, 1):
        out = tmp_dir / f"{cell_id(cell)}.json"
        if out.exists() and not args.force:
            results.append(_collect(out, cell))
            print(f"[orch] [{i}/{total}] SKIP (done) {cell_id(cell)}", flush=True)
            continue
        cmd = _worker_cmd(args, cell, list(gpus), tp=len(gpus), out=out)
        print(f"[orch] [{i}/{total}] launch {cell_id(cell)} tp={len(gpus)}", flush=True)
        cp = subprocess.run(cmd, env=_worker_env(list(gpus)), check=False)
        if cp.returncode != 0:
            results.append({"cell_id": cell_id(cell), "error": f"worker rc={cp.returncode}"})
        else:
            results.append(_collect(out, cell))
        time.sleep(args.cooldown)
    return results


def _preflight_gpu(args) -> None:
    """Fail LOUDLY before any cell if the engine track cannot run. Without this,
    a torch/CUDA mismatch (torch.cuda.is_available()==False) silently skips Track
    A in every cell and still 'completes' with empty energy data (see the
    2026-06-19 first run). Pass --dry-run to intentionally skip the engine."""
    if args.dry_run:
        return
    from src.exp015_serving import engine_available, engine_name, engine_version
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        ndev = torch.cuda.device_count() if cuda_ok else 0
        tver = torch.__version__
    except Exception as e:
        cuda_ok, ndev, tver = False, 0, f"import-failed ({e})"
    if not engine_available():
        msg = [
            "", "=" * 72,
            "EXP-015 PREFLIGHT FAILED -- the GPU engine track cannot run.",
            f"  {engine_name()} importable+CUDA: {engine_available()} (version {engine_version()})",
            f"  torch: {tver} | torch.cuda.is_available(): {cuda_ok} | device_count: {ndev}",
            "  -> Track A (energy/TTFT/throughput) would be SKIPPED in every cell,",
            "     producing an empty-but-'completed' run. Aborting instead.",
            "  Likely cause: torch installed for a CUDA version the node driver does",
            "  not support. Reinstall torch matching `nvidia-smi` CUDA, e.g.:",
            "     pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121",
            "  then verify: python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.device_count())\"",
            "  (must print True and the GPU count). Re-run --smoke before sbatch.",
            "  To run WITHOUT the engine (identification only), pass --dry-run.",
            "=" * 72,
        ]
        print("\n".join(msg), flush=True)
        sys.exit(1)


def run_orchestrator(args) -> None:
    _preflight_gpu(args)
    gpus = args.gpus
    results_dir = REPO_ROOT / "experiments" / RESULTS_DIR_NAME / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = results_dir / f"_tmp_{args.regime}{'_smoke' if args.smoke else ''}"
    tmp_dir.mkdir(exist_ok=True)

    cells = build_cells(args)
    env = collect_env()
    git_sha, git_dirty = collect_git(REPO_ROOT)
    corpus_sha = {}
    if not args.synthetic and args.regime == "R2":
        for s in args.seeds:
            cp = corpus_path(REPO_ROOT, s)
            if cp.exists():
                corpus_sha[str(s)] = _sha256_16(cp)

    print(f"[orch] EXP-015 {args.regime} | gpus={gpus} model={args.model} "
          f"seeds={args.seeds} invocations={args.invocations} cells={len(cells)}", flush=True)
    print(f"[orch] git={git_sha} dirty={git_dirty} dry_run={args.dry_run} "
          f"synthetic={args.synthetic}", flush=True)

    if args.regime == "R1":
        cell_results = schedule_parallel(args, cells, gpus, tmp_dir)
    else:
        cell_results = schedule_sequential(args, cells, gpus, tmp_dir)

    raw = {"experiment": "EXP-015", "claim": "S3", "regime": args.regime,
           "description": "Two-layer energy/identification on a real vLLM stack: "
                          "engine cache {OFF,ON} x identifier {radix,hashrope,flat}",
           "env": env, "git_sha": git_sha, "git_dirty": git_dirty,
           "provenance": {"model": args.model, "engine": env.get("engine"),
                          "engine_version": env.get("engine_version"),
                          "hashrope_version": env.get("hashrope_version"),
                          "corpus_sha16": corpus_sha, "synthetic": args.synthetic,
                          "dry_run": args.dry_run, "max_new_tokens": args.max_new_tokens,
                          "r2_L": args.r2_L if args.regime == "R2" else None,
                          "r1_datasets": args.r1_datasets if args.regime == "R1" else None},
           "seeds": args.seeds, "invocations": args.invocations, "gpus": gpus,
           "n_cells": len(cells),
           "completed_cells": [r.get("cell_id") for r in cell_results if "error" not in r],
           "cells": cell_results}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "_smoke" if args.smoke else ""  # isolate smoke outputs from the grid (resume-collision guard)
    latest = results_dir / f"exp015_{args.regime}{tag}_raw_latest.json"
    archive = results_dir / f"exp015_{args.regime}{tag}_raw_{ts}.json"
    latest.write_text(json.dumps(raw, indent=2))
    archive.write_text(json.dumps(raw, indent=2))
    n_ok = len(raw["completed_cells"])
    print(f"\n[orch] cells completed: {n_ok}/{len(cells)}")
    print(f"[orch] RAW results: {latest}")
    print(f"[orch] archive:     {archive}")
    print("[orch] NOTE: aggregation + verbatim criterion are the separate analysis step.")


# ========================= argparse + main ============================== #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EXP-015 energy/identification bench (S3).")
    ap.add_argument("--regime", choices=["R1", "R2"], required=True)
    ap.add_argument("--gpus", default="0,1,2,3", help="physical NVML indices available")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--invocations", type=int, default=3)
    ap.add_argument("--r1-pairs", type=int, default=80, help="pairs/dataset/seed (R1); 80 keeps the cached footprint ~0.42x single-GPU KV capacity to avoid the prefix-cache eviction cliff")
    ap.add_argument("--r1-datasets", default=",".join(DEFAULT_R1_DATASETS))
    ap.add_argument("--r2-queries", type=int, default=8, help="divergent queries per L (R2)")
    ap.add_argument("--r2-L", default=None, help="comma-separated L sweep (R2)")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--context-length", type=int, default=None,
                    help="engine context length; R2 default = max(L)+slack")
    ap.add_argument("--vocab", type=int, default=50257, help="synthetic-token vocab size")
    ap.add_argument("--cooldown", type=float, default=3.0, help="seconds between cell launches")
    ap.add_argument("--smoke", action="store_true", help="tiny: 1 seed, 1 inv, few requests")
    ap.add_argument("--dry-run", action="store_true", help="skip the engine track (Track A)")
    ap.add_argument("--synthetic", action="store_true", help="bypass tokenizer (synthetic tokens)")
    ap.add_argument("--force", action="store_true", help="re-run completed cells")
    # worker-only (internal)
    ap.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cell-json", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    return ap


def _post_parse(args):
    args.gpus = [int(g) for g in str(args.gpus).split(",") if g != ""]
    args.seeds = [int(s) for s in str(args.seeds).split(",") if s != ""]
    args.r1_datasets = [d for d in str(args.r1_datasets).split(",") if d]
    if args.r2_L is not None:
        args.r2_L = [int(x) for x in str(args.r2_L).split(",") if x]
    else:
        args.r2_L = list(DEFAULT_R2_L)
    if args.smoke:
        args.seeds = args.seeds[:1]
        args.invocations = 1
        args.r1_pairs = min(args.r1_pairs, 8)
        args.r2_queries = min(args.r2_queries, 4)
        args.r2_L = SMOKE_R2_L
    if args.regime == "R2" and args.context_length is None:
        args.context_length = max(args.r2_L) + TAIL_TOKENS + args.max_new_tokens + 64
    if args.regime == "R1" and args.context_length is None:
        args.context_length = 32768  # mirror Instruct-era derived max_model_len; bound KV on the 1M model
    args.seed_for_engine = 0
    return args


def main() -> None:
    args = _post_parse(build_parser().parse_args())
    if args.worker_mode:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
