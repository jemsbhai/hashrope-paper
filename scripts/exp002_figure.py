#!/usr/bin/env python3
"""
EXP-002 figure -- flatten: in-order materialization (FIXED) vs midpoint re-split
(BROKEN). Reproducible from the logged confirmatory data: reads
experiments/exp_002_flatten/results/exp002_flatten_latest.json and renders a
two-panel figure (replaces the prior serialization-tax figure):

 (a) log-log flatten latency vs N, both arms, mean +/- std (n=9), with fitted
     power-law lines. The BROKEN fit is over the full sweep (slope ~1.04, linear
     -- the cost is redundant re-hashing, NOT N log N). The FIXED fit is drawn
     over N >= 1 MB only -- the region above the perf_counter timer floor; the
     sub-ms small-N fixed points (3-50 us) are timer-floor dominated and the
     full-sweep fixed slope is inflated by them (O(N)-fixed rests on the
     operation-count guard, per the pre-registered decision).
 (b) speedup (broken/fixed) vs N (log-y), mean +/- std, with the pre-registered
     100x promotion threshold. Stable ~730-800x plateau for N >= 1 MB; smaller-N
     speedups are timer-floor-inflated.

Saves figures/exp002_flatten_latency.{png,pdf}.

Usage:
    python scripts/exp002_figure.py
    python scripts/exp002_figure.py --result <path-to-json> --dpi 300
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT = (REPO_ROOT / "experiments" / "exp_002_flatten" / "results"
                  / "exp002_flatten_latest.json")
OUT_DIR = REPO_ROOT / "figures"

C_BROKEN = "#c1121f"   # red
C_FIXED = "#0353a4"    # blue
C_SPEED = "#2a9d8f"    # teal
C_THRESH = "#6c757d"   # grey


def _lsq(xs, ys):
    """least-squares slope+intercept of ys on xs."""
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    m = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b = (sy - m * sx) / n
    return m, b


def _fmt_bytes(x, _pos):
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="EXP-002 flatten figure (S4).")
    ap.add_argument("--result", default=str(DEFAULT_RESULT))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        raise SystemExit(f"result JSON not found: {result_path}\n"
                         f"run scripts/exp002_bench.py first.")
    d = json.loads(result_path.read_text())

    sizes = d["params"]["sizes"]
    ps = d["per_size"]
    broken_mean = [ps[str(n)]["broken_ms"]["mean"] for n in sizes]
    broken_std = [ps[str(n)]["broken_ms"]["std"] for n in sizes]
    fixed_mean = [ps[str(n)]["fixed_ms"]["mean"] for n in sizes]
    fixed_std = [ps[str(n)]["fixed_ms"]["std"] for n in sizes]
    sp_mean = [ps[str(n)]["speedup"]["mean"] for n in sizes]
    sp_std = [ps[str(n)]["speedup"]["std"] for n in sizes]
    slopes = d["scaling_slopes_loglog"]
    n_runs = d["params"]["n_runs_per_cell"]
    ref_n = d["params"]["reference_n"]

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    })
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # ---- Panel (a): latency vs N (log-log), data markers + power-law fits ----
    axL.errorbar(sizes, broken_mean, yerr=broken_std, fmt="o", color=C_BROKEN,
                 capsize=3, ms=6, lw=1.4, label="broken (midpoint re-split)", zorder=4)
    axL.errorbar(sizes, fixed_mean, yerr=fixed_std, fmt="s", color=C_FIXED,
                 capsize=3, ms=6, lw=1.4, label="fixed (rope_to_bytes)", zorder=4)
    axL.set_xscale("log")
    axL.set_yscale("log")

    xs_log = [math.log(n) for n in sizes]
    mb, bb = _lsq(xs_log, [math.log(v) for v in broken_mean])
    axL.plot(sizes, [math.exp(mb * x + bb) for x in xs_log], "--",
             color=C_BROKEN, lw=1.1, alpha=0.6, zorder=3)

    big = [n for n in sizes if n >= 1_000_000]
    xb_log = [math.log(n) for n in big]
    mf, bf = _lsq(xb_log, [math.log(ps[str(n)]["fixed_ms"]["mean"]) for n in big])
    axL.plot(big, [math.exp(mf * x + bf) for x in xb_log], "--",
             color=C_FIXED, lw=1.1, alpha=0.6, zorder=3)

    axL.set_xlabel("context size N (bytes)")
    axL.set_ylabel("flatten latency (ms)")
    axL.set_title(f"(a)  latency vs N   (mean \u00b1 std, n={n_runs})", fontsize=11.5)
    axL.legend(loc="upper left", frameon=False, fontsize=9.5)
    axL.xaxis.set_major_formatter(FuncFormatter(_fmt_bytes))
    axL.text(0.97, 0.30, f"broken slope \u2248 {slopes['broken_ms']:.2f}  (linear)",
             transform=axL.transAxes, ha="right", color=C_BROKEN, fontsize=9.5)
    axL.text(0.97, 0.13, f"fixed slope \u2248 {mf:.2f}  (N \u2265 1 MB)",
             transform=axL.transAxes, ha="right", color=C_FIXED, fontsize=9.5)

    # ---- Panel (b): speedup vs N (log-y) ----
    axR.errorbar(sizes, sp_mean, yerr=sp_std, fmt="D-", color=C_SPEED,
                 capsize=3, ms=6, lw=1.6, zorder=4, label="speedup (broken / fixed)")
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.axhline(100, color=C_THRESH, ls=":", lw=1.5, zorder=2)
    axR.text(0.5, 0.07, "pre-registered 100\u00d7 threshold", transform=axR.transAxes,
             color=C_THRESH, fontsize=9, ha="center")
    axR.set_xlabel("context size N (bytes)")
    axR.set_ylabel("speedup (\u00d7)")
    axR.set_title("(b)  speedup vs N", fontsize=11.5)
    axR.xaxis.set_major_formatter(FuncFormatter(_fmt_bytes))
    axR.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f"{y:.0f}"))
    axR.legend(loc="upper right", frameon=False, fontsize=9.5)
    # annotate the stable large-N regime
    ref_speedup = ps[str(ref_n)]["speedup"]["mean"]
    axR.annotate("~730\u2013800\u00d7 plateau (N \u2265 1 MB);\nsmaller N timer-floor-inflated",
                 xy=(ref_n, ref_speedup), xytext=(0.06, 0.26), textcoords="axes fraction",
                 fontsize=8.6, color="#264653",
                 arrowprops=dict(arrowstyle="->", color="#264653", lw=0.9))

    fig.suptitle("EXP-002  flatten: in-order materialization eliminates the re-hash tax "
                 "(0 splits / 0 re-allocs / 0 re-hashing)",
                 fontsize=12.0, y=1.01)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "exp002_flatten_latency.png"
    pdf = OUT_DIR / "exp002_flatten_latency.pdf"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
