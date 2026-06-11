#!/usr/bin/env python3
"""
EXP-002 timing benchmark -- flatten: in-order materialization (FIXED) vs
midpoint re-split (BROKEN control). Claim S4.

Self-contained orchestrator + worker. The orchestrator spawns a FRESH Python
subprocess per (seed, invocation) so each timing run is a clean interpreter --
the EXP-001 error model: the real uncertainty is CROSS-INVOCATION, not the
within-run jitter. It aggregates mean +/- std ACROSS the seeds x invocations
runs, runs an operation-count guard, evaluates the pre-registered promotion
criterion (LOGBOOK EXP-002 (i)-(v)) VERBATIM, and writes results JSON
(latest + timestamped) with env + corpus SHA-256 + git SHA.

Reads the real corpus data/raw/corpus_s{seed}.txt sliced to N bytes. A byte
slice is fine here because the gate is byte-IDENTITY, not UTF-8 validity (the
rope stores raw bytes). NOT "A"*N (degenerate; O(1) under RepeatNode).

Sweep (LOGBOOK EXP-002 IVs) capped at 8 MB: 16 MB pre-registered point dropped
because the corpus is ~11.5 MiB and 16 MB would require tiling. Recorded as a
deviation in the result JSON and the logbook addendum. N=2,000,000 is the
reference anchor (matches the prior 945 ms point).

Verdict gates on (i) HARD byte-identity, (ii) operation-count guard, and
(iii) wall-clock -- per LOGBOOK "Failure of (ii)/(iii) -> S4 stays REFRAMED" and
"(i) ... else the experiment FAILS outright". (iv) scaling is REPORTED
descriptively (broken & fixed log-log slopes); O(N) for FIXED rests on the guard
(0 re-hash / single pass), not on a wall-clock slope fit -- the recorded decision,
since sub-ms fixed points are timer-floor dominated. (v) mean +/- std is the
reporting format throughout.

Usage (confirmatory, pre-registered):
    python scripts/exp002_bench.py --seeds 42,43,44 --invocations 3 --reps 5

Optional functional smoke (~1 min, fewer runs/sizes):
    python scripts/exp002_bench.py --seeds 42 --invocations 1 --reps 2 \
        --sizes 64000,256000,2000000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pre-registered sweep (LOGBOOK EXP-002 IVs), capped at 8 MB (see module docstring).
DEFAULT_SIZES = [64_000, 256_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000]
REFERENCE_N = 2_000_000
DEFAULT_LEAF_BYTES = 4096

# Promotion-criterion thresholds (LOGBOOK EXP-002, verbatim):
REF_FIXED_MAX_MS = 5.0          # (iii) fixed mean <= 5 ms at N = 2,000,000
SPEEDUP_MIN = 100.0             # (iii) speedup >= 100x at ref AND every size >= 256 KB
SPEEDUP_SIZE_FLOOR = 256_000    # (iii) "every size >= 256 KB"


# ----------------------------- shared helpers ------------------------------ #

def _corpus_path(root: Path, seed: int) -> Path:
    return root / "data" / "raw" / f"corpus_s{seed}.txt"


def _load_corpus_bytes(root: Path, seed: int, nbytes: int) -> bytes:
    p = _corpus_path(root, seed)
    if not p.exists():
        raise SystemExit(f"corpus not found: {p}")
    data = p.read_bytes()
    if len(data) < nbytes:
        raise SystemExit(
            f"corpus {p} has {len(data):,} bytes < requested {nbytes:,}; "
            f"reduce --sizes or rebuild the corpus."
        )
    return data[:nbytes]


def _sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _n_leaves(nbytes: int, leaf_bytes: int) -> int:
    return math.ceil(nbytes / leaf_bytes)


def _stats(xs: list[float]) -> dict:
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {"mean": m, "std": sd, "n": len(xs),
            "cv": (sd / m) if m else None, "all": xs}


def _linfit_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    return (n * sxy - sx * sy) / denom if denom else float("nan")


# ------------------------------- worker role ------------------------------- #

def run_worker(args) -> None:
    """One fresh-interpreter timing run for a single (seed, invocation).

    Builds each size once (NOT timed; the rope is persistent so both arms may be
    timed repeatedly against the same object), checks the HARD byte-identity gate,
    discards a warmup, then times both arms interleaved for `reps` and records the
    median ms per arm. Writes per-size results to --out as JSON.
    """
    from src.flatten import (build_fat_leaf_rope, flatten_broken,
                             flatten_fixed, make_hash)

    sizes = [int(s) for s in args.sizes.split(",")]
    root = Path(args.corpus_root)
    leaf_bytes = args.leaf_bytes
    reps = args.reps

    per_size: dict[str, dict] = {}
    for n in sizes:
        data = _load_corpus_bytes(root, args.seed, n)
        h = make_hash()
        rope = build_fat_leaf_rope(data, h, leaf_bytes)        # build NOT timed

        # (i) HARD byte-identity gate: fixed == original == broken
        out_fixed = flatten_fixed(rope)
        out_broken = flatten_broken(rope, h, leaf_bytes)
        identity_ok = (out_fixed == data) and (out_broken == data)

        # warmup (discarded)
        flatten_broken(rope, h, leaf_bytes)
        flatten_fixed(rope)

        broken_ms: list[float] = []
        fixed_ms: list[float] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            flatten_broken(rope, h, leaf_bytes)
            t1 = time.perf_counter()
            flatten_fixed(rope)
            t2 = time.perf_counter()
            broken_ms.append((t1 - t0) * 1e3)
            fixed_ms.append((t2 - t1) * 1e3)

        # paranoia: confirm no mutation across reps (persistence invariant)
        post_ok = (flatten_fixed(rope) == data)

        per_size[str(n)] = {
            "broken_median_ms": statistics.median(broken_ms),
            "fixed_median_ms": statistics.median(fixed_ms),
            "broken_reps_ms": broken_ms,
            "fixed_reps_ms": fixed_ms,
            "identity_ok": bool(identity_ok and post_ok),
        }

    Path(args.out).write_text(json.dumps({
        "seed": args.seed, "inv": args.inv, "reps": reps,
        "leaf_bytes": leaf_bytes, "per_size": per_size,
    }, indent=2))


# --------------------------- operation-count guard ------------------------- #

def run_guard(sizes: list[int], root: Path, seed: int, leaf_bytes: int) -> dict:
    """Instrument Leaf.__init__ / rope_split / PolynomialHash.hash and count them
    for ONE pass of each arm per size (criterion ii). Not timed. Mirrors
    tests/test_flatten.py::_instrument. Counts are seed/content-independent, so a
    single seed suffices for the guard.
    """
    import hashrope.rope as _rope
    from hashrope import PolynomialHash
    from src.flatten import (build_fat_leaf_rope, flatten_broken,
                             flatten_fixed, make_hash)

    out: dict[str, dict] = {}
    for n in sizes:
        data = _load_corpus_bytes(root, seed, n)
        h = make_hash()
        rope = build_fat_leaf_rope(data, h, leaf_bytes)    # build BEFORE instrument

        counts: dict[str, dict] = {}
        for arm in ("broken", "fixed"):
            c = {"leaf": 0, "split": 0, "hash": 0}
            orig_leaf = _rope.Leaf.__init__
            orig_split = _rope.rope_split
            orig_hash = PolynomialHash.hash

            def counting_leaf(self, d, hh, _o=orig_leaf, _c=c):
                _c["leaf"] += 1
                return _o(self, d, hh)

            def counting_split(*a, _o=orig_split, _c=c, **k):
                _c["split"] += 1
                return _o(*a, **k)

            def counting_hash(self, d, _o=orig_hash, _c=c):
                _c["hash"] += 1
                return _o(self, d)

            _rope.Leaf.__init__ = counting_leaf
            _rope.rope_split = counting_split
            PolynomialHash.hash = counting_hash
            try:
                if arm == "broken":
                    flatten_broken(rope, h, leaf_bytes)
                else:
                    flatten_fixed(rope)
            finally:
                _rope.Leaf.__init__ = orig_leaf
                _rope.rope_split = orig_split
                PolynomialHash.hash = orig_hash
            counts[arm] = c

        out[str(n)] = {"guard": counts, "leaves": _n_leaves(n, leaf_bytes)}
    return out


# ----------------------------- provenance ---------------------------------- #

def collect_env() -> dict:
    try:
        import hashrope
        hv = getattr(hashrope, "__version__", "unknown")
        hf = getattr(hashrope, "__file__", "unknown")
    except Exception as e:                                   # pragma: no cover
        hv, hf = f"import-error: {e}", "unknown"
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hashrope_version": hv,
        "hashrope_file": hf,
    }


def collect_git(root: Path):
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                                capture_output=True, text=True, check=True).stdout.strip()
        return sha, (len(status) > 0)
    except Exception:
        return None, None


# ----------------------------- aggregation --------------------------------- #

def compute_slopes(sizes: list[int], per_size_agg: dict) -> dict:
    """log-log least-squares slope of mean latency vs N, per arm (full sweep),
    plus a fixed-arm slope over the >=1 MB subset (above the timer floor)."""
    xs = [math.log(n) for n in sizes]
    out: dict[str, float] = {}
    for arm in ("broken_ms", "fixed_ms"):
        ys = [math.log(per_size_agg[str(n)][arm]["mean"]) for n in sizes]
        out[arm] = _linfit_slope(xs, ys)
    big = [n for n in sizes if n >= 1_000_000]
    if len(big) >= 2:
        xb = [math.log(n) for n in big]
        yb = [math.log(per_size_agg[str(n)]["fixed_ms"]["mean"]) for n in big]
        out["fixed_ms_ge1M"] = _linfit_slope(xb, yb)
    return out


def evaluate_criterion(sizes: list[int], per_size_agg: dict, slopes: dict) -> dict:
    sk_ref = str(REFERENCE_N)

    # (i) byte-identity at every (size, seed) -- HARD
    i_mismatch = sum(per_size_agg[str(n)]["identity_mismatches"] for n in sizes)
    i_pass = (i_mismatch == 0)

    # (ii) guard: fixed 0/0/0 ; broken Theta(#leaves) leaf re-allocs re-hashing
    ii_detail: dict[str, dict] = {}
    ii_pass = True
    for n in sizes:
        g = per_size_agg[str(n)]["guard"]
        fixed_zero = (g["fixed"]["split"] == 0 and g["fixed"]["leaf"] == 0
                      and g["fixed"]["hash"] == 0)
        broken_pos = (g["broken"]["leaf"] > 0 and g["broken"]["hash"] > 0)
        ii_detail[str(n)] = {"fixed_zero": fixed_zero, "broken_positive": broken_pos,
                             "fixed": g["fixed"], "broken": g["broken"]}
        ii_pass = ii_pass and fixed_zero and broken_pos

    # (iii) wall-clock: at ref fixed mean <=5 ms AND speedup>=100x; speedup>=100x
    #       at every size >= 256 KB
    ref_fixed_mean = per_size_agg[sk_ref]["fixed_ms"]["mean"] if sk_ref in per_size_agg else None
    ref_speedup = per_size_agg[sk_ref]["speedup"]["mean"] if sk_ref in per_size_agg else None
    if ref_fixed_mean is None:
        iii_ref_pass = None
    else:
        iii_ref_pass = (ref_fixed_mean <= REF_FIXED_MAX_MS) and (ref_speedup >= SPEEDUP_MIN)
    iii_sizes: dict[str, dict] = {}
    iii_all_pass = True
    for n in sizes:
        if n >= SPEEDUP_SIZE_FLOOR:
            sp = per_size_agg[str(n)]["speedup"]["mean"]
            ok = sp >= SPEEDUP_MIN
            iii_sizes[str(n)] = {"speedup_mean": sp, "pass": ok}
            iii_all_pass = iii_all_pass and ok
    iii_pass = bool(iii_ref_pass) and iii_all_pass

    # (iv) DESCRIPTIVE: report slopes; O(N)-fixed rests on the guard (recorded decision)
    iv = {
        "broken_slope_loglog": slopes.get("broken_ms"),
        "fixed_slope_loglog_fullsweep": slopes.get("fixed_ms"),
        "fixed_slope_loglog_ge1M": slopes.get("fixed_ms_ge1M"),
        "fixed_oN_basis": ("operation-count guard (0 re-hash, single pass); fixed "
                           "wall-clock slope is descriptive only -- sub-ms points "
                           "are timer-floor dominated (recorded decision)."),
        "gating": "descriptive / non-gating",
    }

    # (v) mean +/- std reporting format -- structural
    v_pass = True

    verdict = "PASS" if (i_pass and ii_pass and iii_pass and v_pass) else "FAIL"
    return {
        "i_byte_identity": {"pass": i_pass, "total_mismatches": i_mismatch, "gate": "HARD"},
        "ii_operation_guard": {"pass": ii_pass, "per_size": ii_detail},
        "iii_wall_clock": {
            "pass": iii_pass,
            "reference_n": REFERENCE_N,
            "ref_fixed_mean_ms": ref_fixed_mean, "ref_fixed_max_ms": REF_FIXED_MAX_MS,
            "ref_speedup": ref_speedup, "speedup_min": SPEEDUP_MIN,
            "per_size_ge_256k": iii_sizes,
        },
        "iv_scaling_descriptive": iv,
        "v_mean_std_reported": v_pass,
        "verdict": verdict,
        "gates_note": ("Verdict gates on (i) HARD, (ii), (iii) per LOGBOOK "
                       "'Failure of (ii)/(iii) -> S4 stays REFRAMED' and (i) HARD. "
                       "(iv) descriptive; (v) format."),
    }


def run_orchestrator(args) -> None:
    seeds = [int(s) for s in args.seeds.split(",")]
    sizes = [int(s) for s in args.sizes.split(",")]
    invs = list(range(args.invocations))
    leaf_bytes = args.leaf_bytes
    root = REPO_ROOT

    results_dir = root / "experiments" / "exp_002_flatten" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = results_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    env = collect_env()
    git_sha, git_dirty = collect_git(root)
    corpus_sha = {str(s): _sha256_16(_corpus_path(root, s)) for s in seeds}

    print(f"[orchestrator] EXP-002 flatten | seeds={seeds} invocations={args.invocations} "
          f"reps={args.reps} sizes={sizes}", flush=True)
    print(f"[orchestrator] git={git_sha} dirty={git_dirty} "
          f"hashrope={env['hashrope_version']}", flush=True)

    # ---- spawn fresh subprocess per (seed, invocation) ----
    raw_runs: list[dict] = []
    last = (seeds[-1], invs[-1])
    for seed in seeds:
        for inv in invs:
            out_path = tmp_dir / f"s{seed}_i{inv}.json"
            cmd = [sys.executable, os.path.abspath(__file__), "--worker-mode",
                   "--seed", str(seed), "--inv", str(inv),
                   "--reps", str(args.reps), "--sizes", ",".join(map(str, sizes)),
                   "--leaf-bytes", str(leaf_bytes), "--corpus-root", str(root),
                   "--out", str(out_path)]
            print(f"[orchestrator] seed={seed} inv={inv} -> spawning worker", flush=True)
            subprocess.run(cmd, check=True)
            raw_runs.append(json.loads(out_path.read_text()))
            if (seed, inv) != last:
                time.sleep(args.cooldown)               # cool-down between invocations

    n_runs = len(raw_runs)

    # ---- aggregate across runs (n = seeds x invocations) ----
    per_size_agg: dict[str, dict] = {}
    per_seed_means: dict[str, dict] = {}
    for n in sizes:
        sk = str(n)
        broken = [r["per_size"][sk]["broken_median_ms"] for r in raw_runs]
        fixed = [r["per_size"][sk]["fixed_median_ms"] for r in raw_runs]
        identity = [r["per_size"][sk]["identity_ok"] for r in raw_runs]
        speedup = [b / f for b, f in zip(broken, fixed)]
        per_size_agg[sk] = {
            "broken_ms": _stats(broken),
            "fixed_ms": _stats(fixed),
            "speedup": _stats(speedup),
            "identity_ok_all": all(identity),
            "identity_mismatches": sum(1 for x in identity if not x),
            "leaves": _n_leaves(n, leaf_bytes),
        }
        ps: dict[str, dict] = {}
        for seed in seeds:
            b = [r["per_size"][sk]["broken_median_ms"] for r in raw_runs if r["seed"] == seed]
            f = [r["per_size"][sk]["fixed_median_ms"] for r in raw_runs if r["seed"] == seed]
            sp = [bb / ff for bb, ff in zip(b, f)]
            ps[str(seed)] = {"broken_ms_mean": statistics.fmean(b),
                             "fixed_ms_mean": statistics.fmean(f),
                             "speedup_mean": statistics.fmean(sp)}
        per_seed_means[sk] = ps

    # ---- operation-count guard (one clean in-process pass per size) ----
    guard = run_guard(sizes, root, seeds[0], leaf_bytes)
    for sk, g in guard.items():
        per_size_agg[sk]["guard"] = g["guard"]

    # ---- empirical scaling slopes ----
    slopes = compute_slopes(sizes, per_size_agg)

    # ---- evaluate promotion criterion VERBATIM ----
    criterion = evaluate_criterion(sizes, per_size_agg, slopes)

    # ---- write JSON (latest + timestamped) ----
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "experiment": "EXP-002",
        "claim": "S4",
        "title": "Flatten: in-order materialization (fixed) vs midpoint re-split (broken control)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "env": env,
        "params": {
            "seeds": seeds, "invocations": args.invocations,
            "n_runs_per_cell": n_runs, "reps": args.reps,
            "sizes": sizes, "reference_n": REFERENCE_N,
            "leaf_bytes": leaf_bytes, "cooldown_s": args.cooldown,
            "within_invocation_estimator": "median_of_reps",
            "error_bar": ("std across runs (n=seeds*invocations); within-run CI "
                          "retired per the EXP-001 cross-run error model"),
            "sweep_note": ("16 MB pre-registered IV dropped: corpus ~11.5 MiB; 16 MB "
                           "would require tiling. Recorded deviation."),
        },
        "corpus_sha256_16": corpus_sha,
        "per_size": per_size_agg,
        "per_seed_means": per_seed_means,
        "scaling_slopes_loglog": slopes,
        "promotion_criterion": criterion,
    }
    (results_dir / "exp002_flatten_latest.json").write_text(json.dumps(result, indent=2))
    (results_dir / f"exp002_flatten_{ts}.json").write_text(json.dumps(result, indent=2))

    print_summary(result)
    print(f"\n[orchestrator] wrote: {results_dir / 'exp002_flatten_latest.json'}")
    print(f"[orchestrator] wrote: {results_dir / ('exp002_flatten_' + ts + '.json')}")


# ------------------------------- reporting --------------------------------- #

def print_summary(result: dict) -> None:
    p = result["per_size"]
    sizes = result["params"]["sizes"]
    n = result["params"]["n_runs_per_cell"]
    print("\n" + "=" * 78)
    print(f"EXP-002 flatten -- S4    (n={n} runs/cell = "
          f"{len(result['params']['seeds'])} seeds x {result['params']['invocations']} inv)")
    print("=" * 78)
    hdr = f"{'N (bytes)':>12} {'leaves':>7} {'broken ms':>20} {'fixed ms':>18} {'speedup':>16} {'id':>3}"
    print(hdr)
    print("-" * 78)
    for sz in sizes:
        s = p[str(sz)]
        b, f, sp = s["broken_ms"], s["fixed_ms"], s["speedup"]
        ident = "ok" if s["identity_ok_all"] else "BAD"
        print(f"{sz:>12,} {s['leaves']:>7} "
              f"{b['mean']:>11.2f} +/- {b['std']:>5.2f} "
              f"{f['mean']:>9.3f} +/- {f['std']:>5.3f} "
              f"{sp['mean']:>9.1f} +/- {sp['std']:>4.1f} {ident:>3}")
    sl = result["scaling_slopes_loglog"]
    print("-" * 78)
    print(f"log-log slope: broken={sl.get('broken_ms'):.3f}  "
          f"fixed(full)={sl.get('fixed_ms'):.3f}  "
          f"fixed(>=1M)={sl.get('fixed_ms_ge1M', float('nan')):.3f}  "
          f"(fixed slope descriptive; O(N) via guard)")
    c = result["promotion_criterion"]
    iii = c["iii_wall_clock"]
    print("-" * 78)
    print("PROMOTION CRITERION (verbatim, gates on i/ii/iii):")
    print(f"  (i)   byte-identity HARD          : "
          f"{'PASS' if c['i_byte_identity']['pass'] else 'FAIL'} "
          f"(mismatches={c['i_byte_identity']['total_mismatches']})")
    print(f"  (ii)  operation-count guard       : "
          f"{'PASS' if c['ii_operation_guard']['pass'] else 'FAIL'} "
          f"(fixed 0/0/0; broken>0)")
    print(f"  (iii) wall-clock @N={REFERENCE_N:,}   : "
          f"{'PASS' if iii['pass'] else 'FAIL'} "
          f"(fixed {iii['ref_fixed_mean_ms']:.3f}<=5 ms; "
          f"speedup {iii['ref_speedup']:.1f}>=100x; all>=256K hold)")
    print(f"  (iv)  scaling                     : descriptive (reported)")
    print(f"  (v)   mean+/-std                  : reported")
    print("=" * 78)
    print(f"  VERDICT: {c['verdict']}   ->  "
          f"{'S4 -> SUPPORTED (pending logbook/CLAIMS update)' if c['verdict']=='PASS' else 'S4 stays REFRAMED; investigate (no retrofit)'}")
    print("=" * 78)


# --------------------------------- main ------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EXP-002 flatten timing + guard benchmark (S4).")
    ap.add_argument("--seeds", default="42,43,44",
                    help="comma-separated corpus seeds (default 42,43,44)")
    ap.add_argument("--invocations", type=int, default=3,
                    help="independent process invocations per seed (default 3)")
    ap.add_argument("--reps", type=int, default=5,
                    help="timed reps per arm within an invocation (default 5)")
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                    help="comma-separated context sizes in bytes")
    ap.add_argument("--leaf-bytes", type=int, default=DEFAULT_LEAF_BYTES,
                    help="fat-leaf size in bytes (default 4096; leaf-size sweep is EXP-010)")
    ap.add_argument("--cooldown", type=float, default=2.0,
                    help="seconds to sleep between invocations (default 2.0)")
    # worker-only (internal)
    ap.add_argument("--worker-mode", dest="worker_mode", action="store_true",
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
