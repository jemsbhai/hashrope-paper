#!/usr/bin/env python3
"""
EXP-019 figures (claim B3) -- hashrope branch/snapshot vs a faithful
PagedAttention block-table COW baseline. Reproducible from the logged
confirmatory data: reads
experiments/exp_019_branch/results/exp019_milestoneA_latest.json (n=9) and
renders three figures.

hashrope has no block size (it shares an immutable root and diverges through
rope nodes), so its branch-creation and per-token append latencies are
block-size-independent -- one hashrope curve, four PagedAttention curves (one
per block size {8,16,32,64}). The hashrope curve is the per-N mean across the
four cells; the band is the spread of those four cell means (visualizing the
block-independence). PagedAttention curves carry mean +/- std over the n=9 runs.

Figures (saved to figures/):
  1. exp019_branch_create_latency.{png,pdf}
        branch-creation latency vs N (log-log). The GATED metric. hashrope is
        ~flat (O(log w)); PagedAttention is linear (O(ceil(N/B))). The crossover
        at the reference block size (16) is marked at N* (= 16k from the run).
  2. exp019_branch_memory_compression.{png,pdf}
        branching-memory compression (PagedAttention / hashrope) vs N at
        branch_count = 5 (the ToT beam), one curve per block size, with the
        pre-registered 10x threshold line.
  3. exp019_boundary.{png,pdf}
        the honesty panel: (a) per-token append -- PagedAttention wins (O(1)
        amortized) -- the pre-registered boundary; (b) bare fork -- hashrope is
        O(1) root-share (~tens of ns, flat) while PagedAttention copies the
        block table (linear). Bare fork has no crossover, which is exactly why
        branch-creation -- not bare fork -- is the gated metric (Addendum A).

Usage:
    python scripts/exp019_figure.py
    python scripts/exp019_figure.py --result <path-to-json> --dpi 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT = (REPO_ROOT / "experiments" / "exp_019_branch" / "results"
                  / "exp019_milestoneA_latest.json")
OUT_DIR = REPO_ROOT / "figures"

US = 1e6  # seconds -> microseconds

C_HR = "#0353a4"        # hashrope (our method): blue
# PagedAttention, by block size (dark = small block = more entries = crosses earlier)
PA_COLORS = {8: "#7f0000", 16: "#c1121f", 32: "#e85d04", 64: "#f4a261"}
C_THRESH = "#6c757d"    # threshold grey
C_BAND = "#0353a4"      # hashrope band


def _fmt_tokens(x, _pos):
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def _fmt_log(y, _pos):
    if y >= 1000:
        return f"{y:.0f}"
    if y >= 1:
        return f"{y:.0f}"
    if y >= 0.01:
        return f"{y:.2f}"
    return f"{y:g}"


def main() -> None:
    ap = argparse.ArgumentParser(description="EXP-019 figures (claim B3).")
    ap.add_argument("--result", default=str(DEFAULT_RESULT))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        raise SystemExit(f"result JSON not found: {result_path}\n"
                         f"run scripts/exp019_bench.py first.")
    d = json.loads(result_path.read_text())

    sizes = d["n_sweep"]
    block_sizes = d["block_sizes"]
    ref_block = d["ref_block"]
    n = d["n_runs"]
    cells = {(c["N"], c["block_size"]): c for c in d["aggregate_cells"]}
    memo = {(m["N"], m["block_size"], m["branch_count"]): m for m in d["aggregate_memory"]}
    crossover_N = d["verdict"]["crit_iii_crossover_refblock"]["crossover_N"]

    def pa_series(metric, bs):
        return [cells[(N, bs)][metric] * US for N in sizes]

    def pa_std_series(metric, bs):
        return [cells[(N, bs)][metric] * US for N in sizes]

    def hr_mean_series(metric):
        return [mean(cells[(N, bs)][metric] for bs in block_sizes) * US for N in sizes]

    def hr_band(metric):
        lo = [min(cells[(N, bs)][metric] for bs in block_sizes) * US for N in sizes]
        hi = [max(cells[(N, bs)][metric] for bs in block_sizes) * US for N in sizes]
        return lo, hi

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Figure 1 -- branch-creation latency vs N (GATED)
    # =====================================================================
    fig1, ax = plt.subplots(figsize=(7.2, 5.0))
    hr_y = hr_mean_series("branch_create_hr_mean")
    lo, hi = hr_band("branch_create_hr_mean")
    ax.fill_between(sizes, lo, hi, color=C_BAND, alpha=0.15, zorder=2)
    ax.plot(sizes, hr_y, "s-", color=C_HR, ms=6, lw=2.0, zorder=6,
            label="hashrope (block-size-independent)")
    for bs in block_sizes:
        ax.errorbar(sizes, pa_series("branch_create_pa_mean", bs),
                    yerr=pa_std_series("branch_create_pa_std", bs),
                    fmt="o--", color=PA_COLORS[bs], ms=5, lw=1.3, capsize=2.5,
                    zorder=4, label=f"PagedAttention (block {bs})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if crossover_N is not None:
        ax.axvline(crossover_N, color=C_THRESH, ls=":", lw=1.4, zorder=3)
        ax.text(crossover_N * 1.08, ax.get_ylim()[1] * 0.30,
                f"crossover N* = {crossover_N//1000}K\n(reference block {ref_block})",
                color=C_THRESH, fontsize=9, va="top")
    ax.set_xlabel("context size N (tokens)")
    ax.set_ylabel("branch-creation latency (\u00b5s)")
    ax.set_title(f"branch-creation: fork + first divergent step   (mean \u00b1 std, n={n})",
                 fontsize=11.5)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_tokens))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_log))
    ax.legend(loc="upper left", frameon=False, fontsize=9.2)
    ax.text(0.985, 0.05,
            "hashrope O(log w)  vs  PagedAttention O(\u2308N/B\u2309)",
            transform=ax.transAxes, ha="right", fontsize=9, color="#264653")
    fig1.suptitle("EXP-019  branch/snapshot creation latency vs PagedAttention COW",
                  fontsize=12.0, y=0.98)
    fig1.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"exp019_branch_create_latency.{ext}"
        fig1.savefig(p, dpi=args.dpi if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {p}")

    # =====================================================================
    # Figure 2 -- branching-memory compression vs N (GATED memory)
    # =====================================================================
    fig2, ax = plt.subplots(figsize=(7.2, 5.0))
    for bs in block_sizes:
        comp = [memo[(N, bs, 5)]["compression_mean"] for N in sizes]
        cstd = [memo[(N, bs, 5)]["compression_std"] for N in sizes]
        ax.errorbar(sizes, comp, yerr=cstd, fmt="o-", color=PA_COLORS[bs],
                    ms=5, lw=1.6, capsize=2.5, zorder=4,
                    label=f"PagedAttention block {bs}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(10, color=C_THRESH, ls=":", lw=1.5, zorder=3)
    ax.text(sizes[0], 11, "pre-registered 10\u00d7 threshold",
            color=C_THRESH, fontsize=9, va="bottom")
    ax.set_xlabel("context size N (tokens)")
    ax.set_ylabel("branching-memory compression  (\u00d7)")
    ax.set_title("branching memory: 5 live branches (ToT beam)   (mean \u00b1 std, n=%d)" % n,
                 fontsize=11.5)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_tokens))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_log))
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    big = max(sizes)
    ax.text(0.985, 0.06,
            f"at N=1M: {memo[(big,8,5)]['compression_mean']:.0f}\u00d7 (b8) "
            f"\u2192 {memo[(big,64,5)]['compression_mean']:.0f}\u00d7 (b64)",
            transform=ax.transAxes, ha="right", fontsize=9, color="#264653")
    fig2.suptitle("EXP-019  branching-memory compression vs PagedAttention COW",
                  fontsize=12.0, y=0.98)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"exp019_branch_memory_compression.{ext}"
        fig2.savefig(p, dpi=args.dpi if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {p}")

    # =====================================================================
    # Figure 3 -- boundary panel: append (PA wins) + bare fork (hr O(1))
    # =====================================================================
    fig3, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # (a) per-token append
    hr_app = hr_mean_series("append_hr_mean")
    pa_app = hr_mean_series("append_pa_mean")  # PA append is ~block-independent too
    axA.plot(sizes, hr_app, "s-", color=C_HR, ms=6, lw=2.0, zorder=5,
             label="hashrope  O(log w)")
    axA.plot(sizes, pa_app, "o-", color=PA_COLORS[16], ms=6, lw=2.0, zorder=5,
             label="PagedAttention  O(1) amortized")
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel("context size N (tokens)")
    axA.set_ylabel("per-token append latency (\u00b5s)")
    axA.set_title("(a)  per-token append \u2014 PagedAttention wins (boundary)", fontsize=11)
    axA.xaxis.set_major_formatter(FuncFormatter(_fmt_tokens))
    axA.yaxis.set_major_formatter(FuncFormatter(_fmt_log))
    axA.legend(loc="center left", frameon=False, fontsize=9.5)
    axA.text(0.985, 0.06, "pre-registered boundary\n(reported in full)",
             transform=axA.transAxes, ha="right", fontsize=8.8, color="#264653")

    # (b) bare fork
    hr_fork = hr_mean_series("bare_fork_hr_mean")
    axB.plot(sizes, hr_fork, "s-", color=C_HR, ms=6, lw=2.0, zorder=6,
             label="hashrope  O(1) root-share")
    for bs in block_sizes:
        axB.plot(sizes, pa_series("bare_fork_pa_mean", bs), "o--",
                 color=PA_COLORS[bs], ms=5, lw=1.3, zorder=4,
                 label=f"PagedAttention block {bs}")
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel("context size N (tokens)")
    axB.set_ylabel("bare fork latency (\u00b5s)")
    axB.set_title("(b)  bare fork \u2014 hashrope O(1), no crossover", fontsize=11)
    axB.xaxis.set_major_formatter(FuncFormatter(_fmt_tokens))
    axB.yaxis.set_major_formatter(FuncFormatter(_fmt_log))
    axB.legend(loc="upper left", frameon=False, fontsize=8.8)
    axB.text(0.985, 0.06, "divergence deferred \u2192\nwhy branch-creation is gated",
             transform=axB.transAxes, ha="right", fontsize=8.8, color="#264653")

    fig3.suptitle("EXP-019  honest boundaries: append (PagedAttention wins) and bare fork "
                  "(hashrope O(1), deferred divergence)", fontsize=11.6, y=1.0)
    fig3.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"exp019_boundary.{ext}"
        fig3.savefig(p, dpi=args.dpi if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
