#!/usr/bin/env python
"""
scripts/fig_exp019_milestoneB.py -- EXP-019 Milestone B figures (claim B3, replay).

Reads the n=9 aggregate (exp019_milestoneB_latest.json) and renders three figures:
  1. exp019_milestoneB_crossover        -- branch-creation latency hashrope vs
     PagedAttention (one paged line per block; hashrope is block-independent) vs P,
     log-log, +/-1 sigma bands, block-dependent crossovers; reference-block N* marked.
  2. exp019_milestoneB_compression       -- marginal beam-memory compression vs P,
     one line per block, with the 10x reference line.
  3. exp019_milestoneB_speedup_by_block  -- branch-creation speedup at P=256k per block.

Win-first, honest: every paged win (sub-crossover) and the disclosed b64 9.6x memory
point are visible, not hidden.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(_REPO, "experiments", "exp_019_branch", "results",
                               "exp019_milestoneB_latest.json")
DEFAULT_OUTDIR = os.path.join(_REPO, "figures")

BLOCK_COLORS = {8: "#1b9e77", 16: "#d95f02", 32: "#7570b3", 64: "#e7298a"}
HR_COLOR = "#222222"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cells_by(d):
    by = {}
    for c in d["aggregate_cells"]:
        by[(c["P"], c["block_size"])] = c
    return by


def _mem_by(d):
    by = {}
    for m in d["aggregate_memory"]:
        by[(m["P"], m["block_size"])] = m
    return by


def fig_crossover(d, outdir):
    cells = _cells_by(d)
    p_sweep = [p for p in d["p_sweep"] if p > 0]   # log axis: drop P=0 (shown in caption)
    blocks = d["block_sizes"]
    n_star = d.get("N_star", 16000)
    ref = d.get("ref_block", 16)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # hashrope is block-independent -> single line (use ref-block cells; identical across b)
    hr_mean = [cells[(p, ref)]["branch_create_hr_mean"] * 1e6 for p in p_sweep]
    hr_std = [cells[(p, ref)]["branch_create_hr_std"] * 1e6 for p in p_sweep]
    ax.plot(p_sweep, hr_mean, "o-", color=HR_COLOR, lw=2.4, ms=6,
            label="hashrope (O(log w), block-independent)", zorder=5)
    ax.fill_between(p_sweep,
                    [m - s for m, s in zip(hr_mean, hr_std)],
                    [m + s for m, s in zip(hr_mean, hr_std)],
                    color=HR_COLOR, alpha=0.15, zorder=4)

    for bs in blocks:
        pa_mean = [cells[(p, bs)]["branch_create_pa_mean"] * 1e6 for p in p_sweep]
        pa_std = [cells[(p, bs)]["branch_create_pa_std"] * 1e6 for p in p_sweep]
        ax.plot(p_sweep, pa_mean, "s--", color=BLOCK_COLORS[bs], lw=1.6, ms=4.5,
                label=f"PagedAttention block={bs}")
        ax.fill_between(p_sweep,
                        [m - s for m, s in zip(pa_mean, pa_std)],
                        [m + s for m, s in zip(pa_mean, pa_std)],
                        color=BLOCK_COLORS[bs], alpha=0.10)

    ax.axvline(n_star, color="0.4", ls=":", lw=1.2)
    ax.annotate(f"N* = {n_star//1000}k\n(ref block {ref})", xy=(n_star, ax.get_ylim()[0]),
                xytext=(n_star * 1.1, hr_mean[0] * 0.6), color="0.3", fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("prefix size P (tokens)")
    ax.set_ylabel("branch-creation latency (microseconds)")
    ax.set_title("EXP-019 Milestone B: branch-creation latency on real Game-of-24 ToT traces\n"
                 "hashrope flat (O(log w)); PagedAttention linear (O(ceil(N/B))); "
                 "block-dependent crossover", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    fig.text(0.5, 0.005,
             "Sub-crossover (P=0, bare puzzles, ~5-token base): PagedAttention wins every "
             "block (speedup 0.30-0.40x) -- reported in full.",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, outdir, "exp019_milestoneB_crossover")


def fig_compression(d, outdir):
    mem = _mem_by(d)
    p_sweep = [p for p in d["p_sweep"] if p > 0]
    blocks = d["block_sizes"]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for bs in blocks:
        c_mean = [mem[(p, bs)]["compression_marginal_mean"] for p in p_sweep]
        c_std = [mem[(p, bs)]["compression_marginal_std"] for p in p_sweep]
        ax.errorbar(p_sweep, c_mean, yerr=c_std, fmt="o-", color=BLOCK_COLORS[bs],
                    lw=1.8, ms=5, capsize=2.5, label=f"block={bs}")

    ax.axhline(10, color="0.4", ls=":", lw=1.3)
    ax.text(p_sweep[-1], 10.5, "10x criterion", ha="right", va="bottom",
            color="0.3", fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("prefix size P (tokens)")
    ax.set_ylabel("beam-memory compression  (paged / hashrope)")
    ax.set_title("EXP-019 Milestone B: beam-memory compression on real ToT beams (<=5 survivors)\n"
                 "17x-67x across deployment-standard blocks {8,16,32} at P=256k", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, title="paged block size")
    fig.text(0.5, 0.005,
             "At P=256k: b8 67x, b16 34x, b32 17x, b64 9.6x (disclosed near-miss of the 10x bar; "
             "block-size sensitivity tail).",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, outdir, "exp019_milestoneB_compression")


def fig_speedup_by_block(d, outdir):
    cells = _cells_by(d)
    blocks = d["block_sizes"]
    p_max = max(d["p_sweep"])

    speedups = [cells[(p_max, bs)]["branch_create_speedup"] for bs in blocks]
    colors = [BLOCK_COLORS[bs] for bs in blocks]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xs = range(len(blocks))
    bars = ax.bar(xs, speedups, color=colors, width=0.62, edgecolor="0.2", linewidth=0.6)
    for x, s in zip(xs, speedups):
        ax.text(x, s * 1.02, f"{s:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(1, color="0.5", ls="-", lw=0.8)
    ax.axhline(2, color="0.4", ls=":", lw=1.1)
    ax.text(len(blocks) - 0.55, 2.15, "2x criterion", ha="right", va="bottom",
            color="0.3", fontsize=9)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"block={bs}" for bs in blocks])
    ax.set_yscale("log")
    ax.set_ylabel("branch-creation speedup at P=256k")
    ax.set_title("EXP-019 Milestone B: branch-creation speedup at P=256k\n"
                 "9/9 paired wins at every block (n=9)", fontsize=10.5)
    fig.tight_layout()
    _save(fig, outdir, "exp019_milestoneB_speedup_by_block")


def _save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf")


def main():
    ap = argparse.ArgumentParser(description="EXP-019 Milestone B figures.")
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    d = _load(args.results)
    fig_crossover(d, args.outdir)
    fig_compression(d, args.outdir)
    fig_speedup_by_block(d, args.outdir)
    print(f"figures -> {args.outdir}")


if __name__ == "__main__":
    main()
