#!/usr/bin/env python3
"""EXP-020 (claim B2) -- cross-run bench driver.

Runs the criterion edit bench (benches/bench_b2_edit.rs in the downstream scratch
project) over the locked error model -- >=3 seeds x >=3 process invocations (n>=9) --
harvests criterion's per-benchmark point estimates, runs the (deterministic) memory
probe once per seed, and writes a raw results JSON with full environment metadata.

The within-run criterion confidence interval is NOT the error bar (retired in EXP-001);
the error bar is mean +/- std ACROSS the n>=9 independent process invocations, computed
in the separate analysis step. This driver only collects.

Tiering (decision (b)): tier 1 = 1k..1M (pre-registered N=1M headline + the (ii) slope);
tier 2 = 3M,10M (scale extension). Run tier 1 first, then tier 2.

Nothing here touches the canonical hashrope crate; cargo links it as a read-only path dep.

Usage (run from anywhere; results land next to this script):
    python exp020_bench_driver.py --tier 1
    python exp020_bench_driver.py --tier 2

Env overrides (defaults target the real Windows layout / full protocol):
    HASHROPE_B2_SCRATCH   path to the scratch crate (default: E:\\...\\hashrope_b2_scratch)
    HASHROPE_B2_RESULTS   results dir            (default: <this dir>/results)
    HASHROPE_B2_SEEDS     comma list             (default: 1,2,3)
    HASHROPE_B2_INVOCATIONS invocations per seed (default: 3)
Flags:
    --sizes "1000,3000"        override the tier's size sweep (used for fast validation)
    --criterion-args "..."     args passed through to criterion (default: --warm-up-time 1)
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRATCH_DIR = Path(
    os.environ.get("HASHROPE_B2_SCRATCH", r"E:\data\code\claudecode\hashrope_b2_scratch")
)
RESULTS_DIR = Path(
    os.environ.get("HASHROPE_B2_RESULTS", str(Path(__file__).resolve().parent / "results"))
)
BENCH = "bench_b2_edit"
SEEDS = [int(x) for x in os.environ.get("HASHROPE_B2_SEEDS", "1,2,3").split(",")]
INVOCATIONS = int(os.environ.get("HASHROPE_B2_INVOCATIONS", "3"))

# Locked closure batch counts (must match bench_b2_edit.rs).
K_A = 200
K_B = 50

TIERS = {
    "1": [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000],
    "2": [3_000_000, 10_000_000],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cmd_stdout(args):
    try:
        return subprocess.run(
            args, cwd=SCRATCH_DIR, capture_output=True, text=True
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<err: {exc}>"


def ropey_from_lock():
    lock = SCRATCH_DIR / "Cargo.lock"
    if not lock.exists():
        return None
    m = re.search(
        r'name = "ropey"\nversion = "([^"]+)"\nsource[^\n]*\nchecksum = "([^"]+)"',
        lock.read_text(),
    )
    return {"version": m.group(1), "checksum": m.group(2)} if m else None


def env_metadata(tier, sizes):
    bench_path = SCRATCH_DIR / "benches" / f"{BENCH}.rs"
    return {
        "experiment": "EXP-020",
        "claim": "B2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "python": sys.version.split()[0],
        "rustc": _cmd_stdout(["rustc", "--version"]),
        "cargo": _cmd_stdout(["cargo", "--version"]),
        "scratch_dir": str(SCRATCH_DIR),
        "bench_sha256": sha256(bench_path) if bench_path.exists() else None,
        "ropey": ropey_from_lock(),
        "K_A": K_A,
        "K_B": K_B,
        "tier": tier,
        "sizes": sizes,
        "seeds": SEEDS,
        "invocations": INVOCATIONS,
    }


def run_bench(seed, sizes, criterion_args):
    env = dict(os.environ)
    env["HASHROPE_B2_SEED"] = str(seed)
    env["HASHROPE_B2_SIZES"] = ",".join(str(s) for s in sizes)
    env.pop("HASHROPE_B2_MEMPROBE", None)
    cmd = ["cargo", "bench", "--bench", BENCH, "--", *criterion_args]
    print(f"    $ HASHROPE_B2_SEED={seed} cargo bench --bench {BENCH} -- {' '.join(criterion_args)}",
          flush=True)
    r = subprocess.run(cmd, cwd=SCRATCH_DIR, env=env)
    if r.returncode != 0:
        raise SystemExit(f"cargo bench failed (seed {seed}, returncode {r.returncode})")


def harvest(sizes):
    """Read criterion's just-written new/ estimates for the sizes in this tier."""
    crit = SCRATCH_DIR / "target" / "criterion"
    sizeset = set(sizes)
    recs = []
    for bjson in crit.glob("*/*/*/new/benchmark.json"):
        try:
            bm = json.loads(bjson.read_text())
            est = json.loads((bjson.parent / "estimates.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        try:
            n = int(bm["value_str"])
        except (KeyError, ValueError):
            continue
        if n not in sizeset:
            continue  # ignore stale dirs from the other tier
        k = bm.get("throughput", {}).get("Elements", 1) or 1
        mean_ns = est["mean"]["point_estimate"]
        recs.append(
            {
                "group": bm["group_id"],
                "arm": bm["function_id"],
                "n": n,
                "k": k,
                "mean_ns_total": mean_ns,
                "per_unit_ns": mean_ns / k,
                "std_err_ns_total": est["mean"]["standard_error"],
            }
        )
    return recs


def run_memprobe(seed, sizes):
    env = dict(os.environ)
    env["HASHROPE_B2_SEED"] = str(seed)
    env["HASHROPE_B2_SIZES"] = ",".join(str(s) for s in sizes)
    env["HASHROPE_B2_MEMPROBE"] = "1"
    r = subprocess.run(
        ["cargo", "bench", "--bench", BENCH],
        cwd=SCRATCH_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise SystemExit(f"memprobe failed (seed {seed}, returncode {r.returncode})")
    rows = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("n,"):
            continue
        parts = line.split(",")
        if len(parts) == 5:
            try:
                rows.append(
                    {
                        "seed": seed,
                        "n": int(parts[0]),
                        "nodes_initial": int(parts[1]),
                        "nodes_after": int(parts[2]),
                        "delta": int(parts[3]),
                        "nodes_per_op": float(parts[4]),
                    }
                )
            except ValueError:
                continue
    return rows


def summarize(data):
    import statistics as st

    agg = {}
    for rec in data["timing"]:
        agg.setdefault((rec["group"], rec["arm"], rec["n"]), []).append(rec["per_unit_ns"])

    def ms(group, arm, n):
        vals = agg.get((group, arm, n))
        if not vals:
            return None
        return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))

    ns_vals = sorted({rec["n"] for rec in data["timing"]})
    print("\n=== regime B r=1: cross-run per-cycle (ns) + hashrope speedup (eyeball; not the analysis) ===")
    print(f"{'n':>11} {'hashrope_ns':>14} {'ropey_ns':>16} {'speedup':>9} {'runs':>5}")
    for n in ns_vals:
        h = ms("b2_regimeB_r1", "hashrope", n)
        rp = ms("b2_regimeB_r1", "ropey", n)
        if h and rp:
            print(f"{n:>11} {h[0]:>14.1f} {rp[0]:>16.1f} {rp[0]/h[0]:>8.2f}x {h[2]:>5}")
    print("\n=== regime A: cross-run per-edit (ns) + constant factor C = hashrope/ropey ===")
    print(f"{'n':>11} {'hashrope_ns':>14} {'ropey_ns':>14} {'C':>9} {'runs':>5}")
    for n in ns_vals:
        h = ms("b2_regimeA_edit", "hashrope", n)
        rp = ms("b2_regimeA_edit", "ropey", n)
        if h and rp:
            print(f"{n:>11} {h[0]:>14.1f} {rp[0]:>14.1f} {h[0]/rp[0]:>8.2f}x {h[2]:>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=sorted(TIERS), required=True)
    ap.add_argument("--sizes", default="", help="override the tier size sweep (validation)")
    ap.add_argument("--criterion-args", default="--warm-up-time 1",
                    help="args passed through to criterion")
    args = ap.parse_args()

    sizes = (
        [int(x) for x in args.sizes.split(",") if x.strip()]
        if args.sizes.strip()
        else TIERS[args.tier]
    )
    criterion_args = args.criterion_args.split() if args.criterion_args.strip() else []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_latest = RESULTS_DIR / f"exp020_tier{args.tier}_raw_latest.json"

    if raw_latest.exists():
        data = json.loads(raw_latest.read_text())
        print(f"[resume] loaded {raw_latest}")
    else:
        data = {"meta": env_metadata(args.tier, sizes), "timing": [], "memprobe": []}

    completed = {tuple(c) for c in data.get("completed_cells", [])}

    def save():
        data["meta"]["last_write_utc"] = datetime.now(timezone.utc).isoformat()
        data["completed_cells"] = sorted([list(c) for c in completed])
        raw_latest.write_text(json.dumps(data, indent=2))

    print(f"== EXP-020 tier {args.tier}: seeds {SEEDS} x {INVOCATIONS} invocations, sizes {sizes} ==")

    for seed in SEEDS:
        for inv in range(1, INVOCATIONS + 1):
            if (seed, inv) in completed:
                print(f"[skip] timing seed {seed} inv {inv}")
                continue
            print(f"[run ] timing seed {seed} inv {inv}")
            run_bench(seed, sizes, criterion_args)
            for rec in harvest(sizes):
                rec["seed"] = seed
                rec["invocation"] = inv
                data["timing"].append(rec)
            completed.add((seed, inv))
            save()

    mp_seeds_done = {rec["seed"] for rec in data["memprobe"]}
    for seed in SEEDS:
        if seed in mp_seeds_done:
            print(f"[skip] memprobe seed {seed}")
            continue
        print(f"[run ] memprobe seed {seed}")
        data["memprobe"].extend(run_memprobe(seed, sizes))
        save()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap = RESULTS_DIR / f"exp020_tier{args.tier}_raw_{ts}.json"
    snap.write_text(json.dumps(data, indent=2))

    summarize(data)
    n_cells = len(completed)
    print(f"\n[done] {n_cells} timing cells, {len(set(r['seed'] for r in data['memprobe']))} memprobe seeds")
    print(f"       {raw_latest}")
    print(f"       {snap}")


if __name__ == "__main__":
    main()
