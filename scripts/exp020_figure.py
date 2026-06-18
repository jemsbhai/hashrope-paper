#!/usr/bin/env python3
"""
EXP-020 figures (claim B2) -- hashrope (Rust) incremental edit vs Ropey 1.6.1,
reproducible from the logged confirmatory analysis. Reads
experiments/exp_020_edit/results/exp020_analysis_latest.json (n=9, crate 0.3.1,
tier-1 1k-1M + tier-2 3M-10M) and renders three figures.

Both arms are Rust (no interpreter amplification). hashrope maintains a
whole-buffer polynomial fingerprint on every edit (non-lazy arena); Ropey does
not, so for the content-identity query Ropey recomputes hashrope's OWN
PolynomialHash over its materialized bytes -- maintained-O(1) vs recomputed-O(N),
apples-to-apples.

Figures (saved to figures/):
  1. exp020_edit_identity_latency.{png,pdf}
        HEADLINE. Regime B r=1 per-cycle latency vs N (log-log): hashrope reads
        the fingerprint in O(1) (+ O(log N) edit) and stays ~flat; Ropey
        recomputes in O(N) and climbs linearly. The gap is the differentiator
        win -- 369.7x at 1M, 2901.8x at 10M, 9/9 paired.
  2. exp020_edit_class.{png,pdf}
        Regime A per-edit latency vs N (log-log): the two curves are parallel --
        hashrope edits in the SAME O(log N) class as Ropey (slopes 0.20 vs 0.21).
        The bounded raw-edit constant C = hashrope/Ropey is annotated (peaks
        ~22x then declines to ~14x as both go log N).
  3. exp020_boundary.{png,pdf}
        Honesty panel. (a) regime-B speedup vs N for query:edit ratios
        r in {1, 0.1, 0.01}: r=1 wins at every N (monotone), but with fewer
        queries the crossover N*_id moves right (r=0.1 at 10k, r=0.01 at 300k).
        (b) persistent-arena retained nodes per op vs N: ~O(log N) dead-node
        growth (+2.7 nodes per doubling) -- the flip side of the EXP-004
        branch/snapshot win.

Usage:
    python scripts/exp020_figure.py
    python scripts/exp020_figure.py --result <path-to-json> --dpi 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT = (REPO_ROOT / "experiments" / "exp_020_edit" / "results"
                  / "exp020_analysis_latest.json")
OUT_DIR = REPO_ROOT / "figures"

NS_US = 1e-3   # nanoseconds -> microseconds

C_HR = "#0353a4"        # hashrope (our method): blue
C_ROPEY = "#c1121f"     # Ropey baseline: red
C_THRESH = "#6c757d"    # parity / threshold grey
C_MEM = "#7b2cbf"       # persistent-arena cost: purple
# regime-B r-sweep (decreasing query density)
R_COLORS = {"r1": "#0353a4", "r0.1": "#e85d04", "r0.01": "#6a994e"}
R_LABEL = {"r1": "r = 1 (1 query : 1 edit)",
           "r0.1": "r = 0.1 (1 : 10)",
           "r0.01": "r = 0.01 (1 : 100)"}


def _fmt_size(x, _pos):
    if x >= 1e6:
        return f"{x/1e6:.0f}M"
    if x >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def _fmt_log(y, _pos):
    if y >= 1:
        return f"{y:.0f}"
    if y >= 0.01:
        return f"{y:.2f}"
    return f"{y:g}"


def _sizes(per_n):
    return sorted(int(k) for k in per_n)


def main() -> None:
    ap = argparse.ArgumentParser(description="EXP-020 figures (claim B2).")
    ap.add_argument("--result", default=str(DEFAULT_RESULT))
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    result_path = Path(args.result)
    if not result_path.exists():
        raise SystemExit(f"result JSON not found: {result_path}\n"
                         f"run scripts/exp020_analysis.py first.")
    d = json.loads(result_path.read_text())

    regA = d["regimeA"]
    regB = d["regimeB"]
    mem = d["memory"]
    n = regA["per_n"][next(iter(regA["per_n"]))]["hashrope_ns"]["n"]

    def A(metric, N):  # regime-A per_n accessor (mean, sd) in ns
        c = regA["per_n"][str(N)][metric]
        return c["mean"], c["sd"]

    def Bv(rlabel, metric, N):
        c = regB[rlabel]["per_n"][str(N)][metric]
        return c["mean"], c["sd"]

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
    # Figure 1 -- HEADLINE: regime B r=1 per-cycle latency vs N
    # =====================================================================
    sizes = _sizes(regB["r1"]["per_n"])
    hr = [Bv("r1", "hashrope_ns", N)[0] * NS_US for N in sizes]
    hr_sd = [Bv("r1", "hashrope_ns", N)[1] * NS_US for N in sizes]
    rp = [Bv("r1", "ropey_ns", N)[0] * NS_US for N in sizes]
    rp_sd = [Bv("r1", "ropey_ns", N)[1] * NS_US for N in sizes]

    fig1, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.errorbar(sizes, rp, yerr=rp_sd, fmt="o--", color=C_ROPEY, ms=5, lw=1.6,
                capsize=2.5, zorder=4, label="Ropey 1.6.1  (recompute, O(N))")
    ax.errorbar(sizes, hr, yerr=hr_sd, fmt="s-", color=C_HR, ms=6, lw=2.0,
                capsize=2.5, zorder=6, label="hashrope  (maintained read, O(1) + log-N edit)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # speedup callouts at 1M and 10M
    for N in (1_000_000, 10_000_000):
        if N in sizes:
            sp = Bv("r1", "speedup_ropey_over_hashrope", N)[0]
            yr = Bv("r1", "ropey_ns", N)[0] * NS_US
            yh = Bv("r1", "hashrope_ns", N)[0] * NS_US
            ax.annotate(f"{sp:.0f}x", xy=(N, (yr * yh) ** 0.5),
                        color=C_ROPEY, fontsize=10.5, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.75))
    ax.set_xlabel("buffer size N (chars)")
    ax.set_ylabel("per-cycle latency (\u00b5s)")
    ax.set_title(f"regime B (r=1): edit + whole-buffer content-identity query   "
                 f"(mean \u00b1 std, n={n})", fontsize=11)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_size))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_log))
    ax.legend(loc="upper left", frameon=False, fontsize=9.4)
    ax.text(0.985, 0.06,
            "9/9 paired sign wins at every N\nhashrope O(1) read  vs  Ropey O(N) recompute",
            transform=ax.transAxes, ha="right", fontsize=9, color="#264653")
    fig1.suptitle("EXP-020  edit + identity: hashrope vs Ropey 1.6.1  "
                  "(369.7x @1M \u2192 2901.8x @10M)", fontsize=11.8, y=0.98)
    fig1.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"exp020_edit_identity_latency.{ext}"
        fig1.savefig(p, dpi=args.dpi if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {p}")

    # =====================================================================
    # Figure 2 -- regime A per-edit latency vs N (same O(log N) class)
    # =====================================================================
    hr_e = [A("hashrope_ns", N)[0] * NS_US for N in sizes]
    hr_e_sd = [A("hashrope_ns", N)[1] * NS_US for N in sizes]
    rp_e = [A("ropey_ns", N)[0] * NS_US for N in sizes]
    rp_e_sd = [A("ropey_ns", N)[1] * NS_US for N in sizes]
    s_hr = regA["slope"]["hashrope"]["tier12_headline_means_slope"]
    s_rp = regA["slope"]["ropey"]["tier12_headline_means_slope"]
    C_1m = regA["per_n"]["1000000"]["C_hashrope_over_ropey"]["mean"]
    C_peak = max(regA["per_n"][str(N)]["C_hashrope_over_ropey"]["mean"] for N in sizes)
    C_10m = regA["per_n"][str(max(sizes))]["C_hashrope_over_ropey"]["mean"]

    fig2, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.errorbar(sizes, hr_e, yerr=hr_e_sd, fmt="s-", color=C_HR, ms=6, lw=2.0,
                capsize=2.5, zorder=6,
                label=f"hashrope  (slope {s_hr:.2f}, + fingerprint upkeep)")
    ax.errorbar(sizes, rp_e, yerr=rp_e_sd, fmt="o--", color=C_ROPEY, ms=5, lw=1.6,
                capsize=2.5, zorder=4, label=f"Ropey 1.6.1  (slope {s_rp:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("buffer size N (chars)")
    ax.set_ylabel("per-edit latency (\u00b5s)")
    ax.set_title(f"regime A: churn-edit latency   (mean \u00b1 std, n={n})", fontsize=11)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_size))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_log))
    ax.legend(loc="upper left", frameon=False, fontsize=9.4)
    ax.text(0.985, 0.06,
            f"same O(log N) class (slopes {s_hr:.2f} vs {s_rp:.2f})\n"
            f"raw-edit C = hashrope/Ropey: peaks {C_peak:.0f}x \u2192 {C_10m:.0f}x @10M "
            f"({C_1m:.0f}x @1M)",
            transform=ax.transAxes, ha="right", fontsize=9, color="#264653")
    fig2.suptitle("EXP-020  raw-edit cost: same log class, bounded (declining) constant factor",
                  fontsize=11.8, y=0.98)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"exp020_edit_class.{ext}"
        fig2.savefig(p, dpi=args.dpi if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {p}")

    # =====================================================================
    # Figure 3 -- honesty panel: (a) speedup vs r crossover  (b) arena growth
    # =====================================================================
    fig3, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # (a) regime-B speedup vs N for each r, with parity + N*_id crossovers
    for rlabel in ("r1", "r0.1", "r0.01"):
        rs = _sizes(regB[rlabel]["per_n"])
        sp = [Bv(rlabel, "speedup_ropey_over_hashrope", N)[0] for N in rs]
        axA.plot(rs, sp, "o-", color=R_COLORS[rlabel], ms=5, lw=1.8, zorder=5,
                 label=R_LABEL[rlabel])
        nstar = regB[rlabel]["Nstar_id"]
        if nstar is not None and nstar > min(rs):
            axA.axvline(nstar, color=R_COLORS[rlabel], ls=":", lw=1.2, zorder=3)
    axA.axhline(1.0, color=C_THRESH, ls="--", lw=1.3, zorder=2)
    axA.text(min(sizes), 1.15, "parity (hashrope = Ropey)", color=C_THRESH,
             fontsize=8.6, va="bottom")
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel("buffer size N (chars)")
    axA.set_ylabel("speedup  Ropey / hashrope  (\u00d7)")
    axA.set_title("(a)  edit+identity speedup vs query density\n"
                  "crossover N*_id moves right as queries thin", fontsize=10.5)
    axA.xaxis.set_major_formatter(FuncFormatter(_fmt_size))
    axA.yaxis.set_major_formatter(FuncFormatter(_fmt_log))
    axA.legend(loc="upper left", frameon=False, fontsize=8.8)
    axA.text(0.985, 0.05, "N*_id: r=1 < 1k, r=0.1 = 10k, r=0.01 = 300k\n(honest negative for r<1)",
             transform=axA.transAxes, ha="right", fontsize=8.4, color="#264653")

    # (b) persistent-arena retained nodes per op vs N
    msz = _sizes(mem["per_n"])
    npo = [mem["per_n"][str(N)]["nodes_per_op"]["mean"] for N in msz]
    npo_sd = [mem["per_n"][str(N)]["nodes_per_op"]["sd"] for N in msz]
    slope_dbl = mem.get("nodes_per_op_vs_log2N", {}).get("slope_per_doubling")
    axB.errorbar(msz, npo, yerr=npo_sd, fmt="D-", color=C_MEM, ms=5, lw=1.8,
                 capsize=2.5, zorder=5, label="retained nodes / op")
    axB.set_xscale("log")
    axB.set_xlabel("buffer size N (chars)")
    axB.set_ylabel("persistent-arena retained nodes / op")
    axB.set_title("(b)  persistent-arena growth under linear churn\n"
                  "~O(log N) dead-node retention", fontsize=10.5)
    axB.xaxis.set_major_formatter(FuncFormatter(_fmt_size))
    axB.legend(loc="upper left", frameon=False, fontsize=9.0)
    if slope_dbl is not None:
        axB.text(0.985, 0.06,
                 f"+{slope_dbl:.1f} nodes per doubling of N\n"
                 f"flip side of the EXP-004 branch win",
                 transform=axB.transAxes, ha="right", fontsize=8.6, color="#264653")
    fig3.suptitle("EXP-020  honest boundaries: query-density crossover and persistent-arena cost",
                  fontsize=11.6, y=1.0)
    fig3.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"exp020_boundary.{ext}"
        fig3.savefig(p, dpi=args.dpi if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
