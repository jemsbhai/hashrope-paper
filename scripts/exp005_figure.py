#!/usr/bin/env python3
"""
EXP-005 figure -- LCP timing (log-log) + step count.

Reads from experiments/exp_005_lcp/results/exp005_lcp_latest.json.
Produces figures/exp005_lcp_latency.{png,pdf} and
        figures/exp005_lcp_steps.{png,pdf}.

Panel (a): log-log latency vs N at f=0.5 — hash (flat) vs brute (linear).
Panel (b): step count vs N — observed vs theoretical ceil(log2(N)).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT = REPO_ROOT / "experiments" / "exp_005_lcp" / "results" / "exp005_lcp_latest.json"
FIG_DIR = REPO_ROOT / "figures"
PRIMARY_F = 0.5


def load_result():
    if not RESULT.exists():
        print(f"Result file not found: {RESULT}", file=sys.stderr)
        sys.exit(1)
    return json.loads(RESULT.read_text())


def make_latency_figure(result: dict) -> None:
    """Log-log latency vs N at f=0.5: hash vs brute."""
    sizes = result["params"]["sizes"]
    p = result["per_size_fraction"]

    hash_mean = [p[f"{n}_{PRIMARY_F}"]["hash_ms"]["mean"] for n in sizes]
    hash_std = [p[f"{n}_{PRIMARY_F}"]["hash_ms"]["std"] for n in sizes]
    brute_mean = [p[f"{n}_{PRIMARY_F}"]["brute_ms"]["mean"] for n in sizes]
    brute_std = [p[f"{n}_{PRIMARY_F}"]["brute_ms"]["std"] for n in sizes]

    slopes = result["scaling_slopes_loglog_f05"]
    hash_slope = slopes["hash_ms"]
    brute_slope = slopes["brute_ms"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(sizes, hash_mean, yerr=hash_std, fmt="o-",
                color="#2563eb", linewidth=2, markersize=6, capsize=4,
                label=f"hash-LCP (slope {hash_slope:.3f})")
    ax.errorbar(sizes, brute_mean, yerr=brute_std, fmt="s--",
                color="#dc2626", linewidth=2, markersize=6, capsize=4,
                label=f"brute-force (slope {brute_slope:.3f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rope size N (bytes)", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("LCP latency vs rope size (f = 0.5, n = 9 runs)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    # annotate crossover
    for i, n in enumerate(sizes):
        if hash_mean[i] < brute_mean[i] and (i == 0 or hash_mean[i-1] >= brute_mean[i-1]):
            ax.annotate(f"crossover ~{n/1e6:.0f} MB",
                        xy=(n, hash_mean[i]), fontsize=9, color="#666",
                        xytext=(n * 2, hash_mean[i] * 2.5),
                        arrowprops=dict(arrowstyle="->", color="#666"))
            break

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "exp005_lcp_latency.png", dpi=300)
    fig.savefig(FIG_DIR / "exp005_lcp_latency.pdf")
    plt.close(fig)
    print(f"[figure] wrote: {FIG_DIR / 'exp005_lcp_latency.png'}")
    print(f"[figure] wrote: {FIG_DIR / 'exp005_lcp_latency.pdf'}")


def make_steps_figure(result: dict) -> None:
    """Step count vs N: observed vs theoretical."""
    sizes = result["params"]["sizes"]
    p = result["per_size_fraction"]

    observed = [p[f"{n}_{PRIMARY_F}"]["step_count_max_seen"] for n in sizes]
    theoretical = [2 * (math.ceil(math.log2(n)) + 1) for n in sizes]
    log2_n = [2 * math.ceil(math.log2(n)) for n in sizes]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, observed, "o-", color="#2563eb", linewidth=2,
            markersize=7, label="observed (max across runs)")
    ax.plot(sizes, log2_n, "x--", color="#9333ea", linewidth=1.5,
            markersize=7, label=r"$2 \cdot \lceil \log_2 N \rceil$")
    ax.plot(sizes, theoretical, "^:", color="#dc2626", linewidth=1.5,
            markersize=6, label=r"$2 \cdot (\lceil \log_2 N \rceil + 1)$ (bound)")

    ax.set_xscale("log")
    ax.set_xlabel("Rope size N (bytes)", fontsize=12)
    ax.set_ylabel("rope_substr_hash calls", fontsize=12)
    ax.set_title("LCP step count vs rope size (f = 0.5)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp005_lcp_steps.png", dpi=300)
    fig.savefig(FIG_DIR / "exp005_lcp_steps.pdf")
    plt.close(fig)
    print(f"[figure] wrote: {FIG_DIR / 'exp005_lcp_steps.png'}")
    print(f"[figure] wrote: {FIG_DIR / 'exp005_lcp_steps.pdf'}")


def main():
    result = load_result()
    make_latency_figure(result)
    make_steps_figure(result)


if __name__ == "__main__":
    main()
