#!/usr/bin/env python3
"""
EXP-005 timing benchmark -- LCP via prefix-hash binary search. Claim T3.

Self-contained orchestrator + worker. The orchestrator spawns a FRESH Python
subprocess per (seed, invocation) so each timing run is a clean interpreter --
the EXP-001 error model: the real uncertainty is CROSS-INVOCATION, not the
within-run jitter. It aggregates mean +/- std ACROSS the seeds x invocations
runs, evaluates the pre-registered promotion criterion (LOGBOOK EXP-005
(i)-(v)) VERBATIM, and writes results JSON (latest + timestamped) with env +
corpus SHA-256 + git SHA.

Reads the real corpus data/raw/corpus_s{seed}.txt sliced to N bytes. Pairs are
built by flipping a single byte at position L = int(f * N) to create a known
LCP = L.

Usage (confirmatory, pre-registered):
    python scripts/exp005_bench.py --seeds 42,43,44 --invocations 3 --reps 5

Optional functional smoke (~1 min):
    python scripts/exp005_bench.py --seeds 42 --invocations 1 --reps 2 \
        --sizes 64000,256000,1000000
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

# Pre-registered sweep (LOGBOOK EXP-005 IVs).
DEFAULT_SIZES = [64_000, 256_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000]
FRACTIONS = [0.0, 0.5, 0.99]
PRIMARY_F = 0.5  # the fraction used for the scaling criterion
DEFAULT_LEAF_BYTES = 4096

# Promotion-criterion thresholds (LOGBOOK EXP-005, verbatim):
MAX_SLOPE = 0.3  # (iii) log-log slope of hash latency vs N at f=0.5

# Prefix-dedup workload params (descriptive).
DEDUP_K = 10          # number of simulated prompts
DEDUP_PREFIX_FRAC = 0.3  # fraction of N used as shared prefix


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


def _make_pair_bytes(data: bytes, f: float) -> tuple[bytes, bytes, int]:
    """Return (a_bytes, b_bytes, expected_lcp). b has byte at L=int(f*N) flipped."""
    n = len(data)
    lcp_expected = int(f * n)
    b = bytearray(data)
    if lcp_expected < n:
        b[lcp_expected] ^= 0xFF
    return data, bytes(b), lcp_expected


# ------------------------------- worker role ------------------------------- #

def run_worker(args) -> None:
    """One fresh-interpreter timing run for a single (seed, invocation).

    Builds rope pairs, checks correctness, times lcp_hash and lcp_brute,
    records step count. Writes per-(size,f) results to --out as JSON.
    """
    from src.flatten import build_fat_leaf_rope, make_hash
    from src.lcp import lcp_brute, lcp_hash, lcp_hash_counted

    sizes = [int(s) for s in args.sizes.split(",")]
    fractions = [float(f) for f in args.fractions.split(",")]
    root = Path(args.corpus_root)
    reps = args.reps

    per_cell: dict[str, dict] = {}
    for n in sizes:
        corpus_data = _load_corpus_bytes(root, args.seed, n)
        h = make_hash()

        for f in fractions:
            a_bytes, b_bytes, expected_lcp = _make_pair_bytes(corpus_data, f)
            rope_a = build_fat_leaf_rope(a_bytes, h)
            rope_b = build_fat_leaf_rope(b_bytes, h)

            # correctness gate
            brute_result = lcp_brute(a_bytes, b_bytes)
            hash_result = lcp_hash(rope_a, rope_b, h)
            correctness_ok = (brute_result == expected_lcp and
                              hash_result == expected_lcp and
                              hash_result == brute_result)

            # step count
            _, step_count = lcp_hash_counted(rope_a, rope_b, h)
            max_steps = 2 * (math.ceil(math.log2(n)) + 1)
            step_ok = step_count <= max_steps

            # warmup (discarded)
            lcp_hash(rope_a, rope_b, h)
            if f > 0:
                lcp_brute(a_bytes, b_bytes)

            # timing: lcp_hash
            hash_ms_list: list[float] = []
            for _ in range(reps):
                t0 = time.perf_counter()
                lcp_hash(rope_a, rope_b, h)
                t1 = time.perf_counter()
                hash_ms_list.append((t1 - t0) * 1e3)

            # timing: lcp_brute (reps=1 for large N with large f to avoid
            # multi-minute brute runs; descriptive only)
            brute_reps = 1 if (n >= 4_000_000 and f >= 0.5) else min(reps, 3)
            brute_ms_list: list[float] = []
            for _ in range(brute_reps):
                t0 = time.perf_counter()
                lcp_brute(a_bytes, b_bytes)
                t1 = time.perf_counter()
                brute_ms_list.append((t1 - t0) * 1e3)

            cell_key = f"{n}_{f}"
            per_cell[cell_key] = {
                "n": n,
                "f": f,
                "expected_lcp": expected_lcp,
                "hash_result": hash_result,
                "brute_result": brute_result,
                "correctness_ok": correctness_ok,
                "step_count": step_count,
                "max_steps_allowed": max_steps,
                "step_ok": step_ok,
                "hash_median_ms": statistics.median(hash_ms_list),
                "hash_reps_ms": hash_ms_list,
                "brute_median_ms": statistics.median(brute_ms_list),
                "brute_reps_ms": brute_ms_list,
            }

    # prefix-dedup workload (descriptive)
    dedup = _run_dedup_workload(root, args.seed, sizes[-1], reps)

    Path(args.out).write_text(json.dumps({
        "seed": args.seed, "inv": args.inv, "reps": reps,
        "per_cell": per_cell,
        "dedup_workload": dedup,
    }, indent=2))


def _run_dedup_workload(root: Path, seed: int, max_n: int, reps: int) -> dict:
    """K=10 prompts sharing a prefix; LCP all 45 pairs. Descriptive."""
    from src.flatten import build_fat_leaf_rope, make_hash
    from src.lcp import lcp_hash

    # Use half the max size to leave room for suffixes
    n = min(max_n, 4_000_000)
    corpus = _load_corpus_bytes(root, seed, n)
    h = make_hash()
    prefix_len = int(DEDUP_PREFIX_FRAC * n)
    suffix_len = 4096  # each prompt gets a unique 4 KB suffix
    prompt_len = prefix_len + suffix_len

    # need K * suffix_len + prefix_len bytes of corpus
    needed = prefix_len + DEDUP_K * suffix_len
    if len(corpus) < needed:
        return {"skipped": True, "reason": f"corpus too small: need {needed}"}

    prefix = corpus[:prefix_len]
    ropes = []
    for i in range(DEDUP_K):
        suffix_start = prefix_len + i * suffix_len
        suffix = corpus[suffix_start:suffix_start + suffix_len]
        prompt_bytes = prefix + suffix
        ropes.append(build_fat_leaf_rope(prompt_bytes, h))

    # LCP all K*(K-1)/2 pairs
    pairs_correct = 0
    pairs_total = 0
    t0 = time.perf_counter()
    for i in range(DEDUP_K):
        for j in range(i + 1, DEDUP_K):
            lcp_val = lcp_hash(ropes[i], ropes[j], h)
            pairs_total += 1
            if lcp_val == prefix_len:
                pairs_correct += 1
    t1 = time.perf_counter()

    return {
        "skipped": False,
        "k": DEDUP_K,
        "prefix_len": prefix_len,
        "prompt_len": prompt_len,
        "pairs_total": pairs_total,
        "pairs_correct": pairs_correct,
        "hit_rate": pairs_correct / pairs_total if pairs_total else 0.0,
        "total_ms": (t1 - t0) * 1e3,
        "per_pair_ms": (t1 - t0) * 1e3 / pairs_total if pairs_total else 0.0,
    }


# ----------------------------- provenance ---------------------------------- #

def collect_env() -> dict:
    try:
        import hashrope
        hv = getattr(hashrope, "__version__", "unknown")
        hf = getattr(hashrope, "__file__", "unknown")
    except Exception as e:
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

def compute_slopes(sizes: list[int], per_size_f: dict, f: float) -> dict:
    """log-log slope of hash and brute latency vs N at a given fraction."""
    out: dict[str, float] = {}
    xs = [math.log(n) for n in sizes]
    for arm in ("hash_ms", "brute_ms"):
        ys = []
        for n in sizes:
            key = f"{n}_{f}"
            ys.append(math.log(per_size_f[key][arm]["mean"]))
        out[arm] = _linfit_slope(xs, ys)
    return out


def evaluate_criterion(sizes: list[int], per_size_f: dict,
                       slopes: dict, n_runs: int) -> dict:
    """Evaluate promotion criterion VERBATIM from LOGBOOK EXP-005."""
    fractions = sorted({v["f"] for v in per_size_f.values()})

    # (i) HARD correctness: 0 mismatches across all (size, f, run)
    i_mismatches = sum(1 for v in per_size_f.values()
                       if not v.get("correctness_all_ok", True))
    i_pass = (i_mismatches == 0)

    # (ii) step-count guard: all within bound
    ii_violations = sum(1 for v in per_size_f.values()
                        if not v.get("step_all_ok", True))
    ii_pass = (ii_violations == 0)

    # (iii) wall-clock scaling: log-log slope of hash at f=0.5 <= 0.3
    hash_slope = slopes.get("hash_ms", float("nan"))
    iii_pass = (hash_slope <= MAX_SLOPE)

    # (iv) brute-force slope (descriptive, non-gating)
    brute_slope = slopes.get("brute_ms", float("nan"))

    # (v) format: mean +/- std
    v_pass = True

    verdict = "PASS" if (i_pass and ii_pass and iii_pass) else "FAIL"
    return {
        "i_correctness": {
            "pass": i_pass, "total_mismatches": i_mismatches, "gate": "HARD",
        },
        "ii_step_count": {
            "pass": ii_pass, "violations": ii_violations,
        },
        "iii_wall_clock_scaling": {
            "pass": iii_pass,
            "hash_slope_loglog": hash_slope,
            "max_slope": MAX_SLOPE,
            "reference": ("O(log^2 N) predicts slope ~0.14 over 64KB-8MB; "
                          "O(N) = 1.0; informal pilot showed ~flat"),
        },
        "iv_brute_slope_descriptive": {
            "brute_slope_loglog": brute_slope,
            "expected": "[0.7, 1.3] (linear in LCP length at f=0.5)",
            "gating": "descriptive / non-gating",
        },
        "v_mean_std_reported": v_pass,
        "verdict": verdict,
        "gates_note": ("Verdict gates on (i) HARD, (ii), (iii) per LOGBOOK. "
                       "(iv) descriptive; (v) format."),
    }


# ----------------------------- orchestrator -------------------------------- #

def run_orchestrator(args) -> None:
    seeds = [int(s) for s in args.seeds.split(",")]
    sizes = [int(s) for s in args.sizes.split(",")]
    fractions = [float(f) for f in args.fractions.split(",")]
    invs = list(range(args.invocations))
    root = REPO_ROOT

    results_dir = root / "experiments" / "exp_005_lcp" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = results_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    env = collect_env()
    git_sha, git_dirty = collect_git(root)
    corpus_sha = {str(s): _sha256_16(_corpus_path(root, s)) for s in seeds}

    print(f"[orchestrator] EXP-005 LCP | seeds={seeds} invocations={args.invocations} "
          f"reps={args.reps} sizes={sizes} fractions={fractions}", flush=True)
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
                   "--reps", str(args.reps),
                   "--sizes", ",".join(map(str, sizes)),
                   "--fractions", ",".join(str(f) for f in fractions),
                   "--corpus-root", str(root),
                   "--out", str(out_path)]
            print(f"[orchestrator] seed={seed} inv={inv} -> spawning worker",
                  flush=True)
            subprocess.run(cmd, check=True)
            raw_runs.append(json.loads(out_path.read_text()))
            if (seed, inv) != last:
                time.sleep(args.cooldown)

    n_runs = len(raw_runs)

    # ---- aggregate across runs ----
    per_size_f_agg: dict[str, dict] = {}
    for n in sizes:
        for f in fractions:
            ck = f"{n}_{f}"
            hash_ms = [r["per_cell"][ck]["hash_median_ms"] for r in raw_runs]
            brute_ms = [r["per_cell"][ck]["brute_median_ms"] for r in raw_runs]
            correct = [r["per_cell"][ck]["correctness_ok"] for r in raw_runs]
            steps = [r["per_cell"][ck]["step_count"] for r in raw_runs]
            step_ok = [r["per_cell"][ck]["step_ok"] for r in raw_runs]
            expected = raw_runs[0]["per_cell"][ck]["expected_lcp"]
            max_steps = raw_runs[0]["per_cell"][ck]["max_steps_allowed"]

            per_size_f_agg[ck] = {
                "n": n, "f": f,
                "expected_lcp": expected,
                "hash_ms": _stats(hash_ms),
                "brute_ms": _stats(brute_ms),
                "correctness_all_ok": all(correct),
                "correctness_mismatches": sum(1 for c in correct if not c),
                "step_counts": steps,
                "step_count_max_seen": max(steps),
                "max_steps_allowed": max_steps,
                "step_all_ok": all(step_ok),
            }

    # ---- slopes at primary fraction f=0.5 ----
    slopes = compute_slopes(sizes, per_size_f_agg, PRIMARY_F)

    # ---- dedup workload aggregate (descriptive) ----
    dedup_runs = [r.get("dedup_workload", {}) for r in raw_runs
                  if not r.get("dedup_workload", {}).get("skipped", True)]
    dedup_agg: dict = {}
    if dedup_runs:
        hr = [d["hit_rate"] for d in dedup_runs]
        pp = [d["per_pair_ms"] for d in dedup_runs]
        dedup_agg = {
            "hit_rate": _stats(hr),
            "per_pair_ms": _stats(pp),
            "k": dedup_runs[0]["k"],
            "prefix_len": dedup_runs[0]["prefix_len"],
            "pairs_total": dedup_runs[0]["pairs_total"],
        }

    # ---- evaluate promotion criterion ----
    criterion = evaluate_criterion(sizes, per_size_f_agg, slopes, n_runs)

    # ---- write JSON ----
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "experiment": "EXP-005",
        "claim": "T3",
        "title": "LCP via prefix-hash binary search: O(log^2 N) timing + correctness",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha,
        "git_dirty": git_dirty,
        "env": env,
        "params": {
            "seeds": seeds, "invocations": args.invocations,
            "n_runs_per_cell": n_runs, "reps": args.reps,
            "sizes": sizes, "fractions": fractions,
            "primary_fraction": PRIMARY_F,
            "cooldown_s": args.cooldown,
            "within_invocation_estimator": "median_of_reps",
            "error_bar": ("std across runs (n=seeds*invocations); within-run CI "
                          "retired per the EXP-001 cross-run error model"),
        },
        "corpus_sha256_16": corpus_sha,
        "per_size_fraction": per_size_f_agg,
        "scaling_slopes_loglog_f05": slopes,
        "dedup_workload": dedup_agg,
        "promotion_criterion": criterion,
    }
    (results_dir / "exp005_lcp_latest.json").write_text(
        json.dumps(result, indent=2))
    (results_dir / f"exp005_lcp_{ts}.json").write_text(
        json.dumps(result, indent=2))

    print_summary(result)
    print(f"\n[orchestrator] wrote: {results_dir / 'exp005_lcp_latest.json'}")
    print(f"[orchestrator] wrote: {results_dir / ('exp005_lcp_' + ts + '.json')}")


# ------------------------------- reporting --------------------------------- #

def print_summary(result: dict) -> None:
    p = result["per_size_fraction"]
    sizes = result["params"]["sizes"]
    fracs = result["params"]["fractions"]
    n = result["params"]["n_runs_per_cell"]

    print("\n" + "=" * 90)
    print(f"EXP-005 LCP -- T3    (n={n} runs/cell = "
          f"{len(result['params']['seeds'])} seeds x "
          f"{result['params']['invocations']} inv)")
    print("=" * 90)

    # timing table per fraction
    for f in fracs:
        print(f"\n--- f={f} ---")
        hdr = (f"{'N':>12} {'LCP':>10} {'hash ms':>20} "
               f"{'brute ms':>20} {'steps':>6} {'ok':>3}")
        print(hdr)
        print("-" * 90)
        for sz in sizes:
            ck = f"{sz}_{f}"
            s = p[ck]
            hm, bm = s["hash_ms"], s["brute_ms"]
            print(f"{sz:>12,} {s['expected_lcp']:>10,} "
                  f"{hm['mean']:>11.3f} +/- {hm['std']:>6.3f} "
                  f"{bm['mean']:>11.3f} +/- {bm['std']:>6.3f} "
                  f"{s['step_count_max_seen']:>6} "
                  f"{'ok' if s['correctness_all_ok'] and s['step_all_ok'] else 'BAD':>3}")

    sl = result["scaling_slopes_loglog_f05"]
    print("\n" + "-" * 90)
    print(f"log-log slope @ f={PRIMARY_F}: hash={sl.get('hash_ms', float('nan')):.4f}  "
          f"brute={sl.get('brute_ms', float('nan')):.4f}")

    # dedup workload
    dd = result.get("dedup_workload", {})
    if dd and "hit_rate" in dd:
        print(f"dedup workload: K={dd['k']} prefix={dd['prefix_len']:,}  "
              f"hit_rate={dd['hit_rate']['mean']:.3f}  "
              f"per_pair={dd['per_pair_ms']['mean']:.2f} ms")

    c = result["promotion_criterion"]
    print("-" * 90)
    print("PROMOTION CRITERION (verbatim, gates on i/ii/iii):")
    print(f"  (i)   correctness HARD             : "
          f"{'PASS' if c['i_correctness']['pass'] else 'FAIL'} "
          f"(mismatches={c['i_correctness']['total_mismatches']})")
    print(f"  (ii)  step-count guard             : "
          f"{'PASS' if c['ii_step_count']['pass'] else 'FAIL'} "
          f"(violations={c['ii_step_count']['violations']})")
    iii = c["iii_wall_clock_scaling"]
    print(f"  (iii) wall-clock slope @ f={PRIMARY_F}    : "
          f"{'PASS' if iii['pass'] else 'FAIL'} "
          f"(slope={iii['hash_slope_loglog']:.4f} <= {iii['max_slope']})")
    print(f"  (iv)  brute slope (descriptive)    : "
          f"{c['iv_brute_slope_descriptive']['brute_slope_loglog']:.4f}")
    print(f"  (v)   mean+/-std                   : reported")
    print("=" * 90)
    print(f"  VERDICT: {c['verdict']}   ->  "
          f"{'T3 -> SUPPORTED' if c['verdict'] == 'PASS' else 'T3 stays IN-PROGRESS; investigate'}")
    print("=" * 90)


# --------------------------------- main ------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="EXP-005 LCP timing + step-count benchmark (T3).")
    ap.add_argument("--seeds", default="42,43,44",
                    help="comma-separated corpus seeds (default 42,43,44)")
    ap.add_argument("--invocations", type=int, default=3,
                    help="independent process invocations per seed (default 3)")
    ap.add_argument("--reps", type=int, default=5,
                    help="timed reps for lcp_hash within an invocation (default 5)")
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                    help="comma-separated context sizes in bytes")
    ap.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS),
                    help="comma-separated LCP fractions (default 0.0,0.5,0.99)")
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
