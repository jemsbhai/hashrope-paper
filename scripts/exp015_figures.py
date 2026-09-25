#!/usr/bin/env python3
"""EXP-015 figures (claim S3): generated SOLELY from the analysis summary JSON.

No hardcoded result numbers anywhere: every plotted value, error bar, and
annotation (crossover bracket, win multiple, sign-test p, energy ratios,
serving-wall position, provenance footer) is read from
experiments/exp_015_energy/results/exp015_analysis_summary_latest.json,
which is itself derived from the committed raw grid JSONs. Regenerating the
summary regenerates the figures.

Outputs (both .png at 300 dpi and vector .pdf):
  figures/exp015_identification_latency_vs_L.{png,pdf}   <- the money figure
      Track B identification latency vs L, log-log: hashrope is flat across L
      (the O(log) signature), the vendored SGLang radix matcher grows, the
      numpy flat scan grows ~linearly (slope ~1 on log-log). The crossover
      bracket (L*') is shaded; the above-crossover point is annotated with
      the win multiple and the paired sign test.
  figures/exp015_caching_energy_vs_L.{png,pdf}
      Track A engine prefix-cache effect vs L (attributed to the ENGINE,
      explicitly not hashrope): top panel OFF/ON energy ratio, bottom panel
      TTFT OFF vs ON (log y). The dense-attention serving wall (Track A
      skipped) is shaded.

Optional:
  --exp017-crossover N   draw a dashed vertical line at the EXP-017
                         CPU-model crossover prediction (tokens) on the
                         identification figure. Off by default; pass the
                         EXP-017 value explicitly so its provenance lives in
                         the invocation, not in this script.
  --no-footer            omit the provenance footer line.
  --selftest             render from a synthetic summary into a temp dir and
                         assert outputs exist; no repo files touched.

Usage (from repo root):
  python scripts/exp015_figures.py
  python scripts/exp015_figures.py --exp017-crossover 571000
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# Okabe-Ito colorblind-safe palette
C_HASHROPE = "#0072B2"  # blue
C_RADIX = "#D55E00"     # vermillion
C_FLAT = "#009E73"      # green
C_OFF = "#D55E00"
C_ON = "#0072B2"
C_SHADE = "#BBBBBB"


def _short_tokens(x, _pos=None):
    if x >= 1_000_000:
        v = x / 1_000_000
        return f"{v:g}M"
    if x >= 1_000:
        return f"{x/1000:g}k"
    return f"{x:g}"


def load_summary(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# data extraction from the summary (single source of truth)
# --------------------------------------------------------------------------

def track_b_series(summary):
    by_L = summary["r2"]["track_b_by_L"]
    Ls = sorted(int(k) for k in by_L)
    out = {"L": Ls, "arms": {}, "sign": {}}
    for arm in ("hashrope", "radix", "flat"):
        means, stds = [], []
        for L in Ls:
            st = by_L[str(L)]["arms"][arm]["lat_ms"]
            means.append(st["mean"])
            stds.append(st["std"])
        out["arms"][arm] = {"mean": means, "std": stds}
    for L in Ls:
        out["sign"][L] = by_L[str(L)].get("sign_hashrope_beats_radix")
    return out


def crossover_bracket(tb):
    """(L_low, L_high): largest L with radix faster, smallest with hashrope
    <= radix. Same rule as the analyzer's clause (iv)."""
    L_low = L_high = None
    for i, L in enumerate(tb["L"]):
        h = tb["arms"]["hashrope"]["mean"][i]
        r = tb["arms"]["radix"]["mean"][i]
        if h <= r:
            if L_high is None or L < L_high:
                L_high = L
        else:
            if L_low is None or L > L_low:
                L_low = L
    return L_low, L_high


def track_a_series(summary):
    by_L = summary["r2"]["track_a_by_L"]
    Ls = sorted(int(k) for k in by_L)
    ratio_m = [by_L[str(L)]["ratio_j"]["mean"] for L in Ls]
    ratio_s = [by_L[str(L)]["ratio_j"]["std"] for L in Ls]
    ttft_off = [by_L[str(L)]["ttft_ms_off"]["mean"] for L in Ls]
    ttft_off_s = [by_L[str(L)]["ttft_ms_off"]["std"] for L in Ls]
    ttft_on = [by_L[str(L)]["ttft_ms_on"]["mean"] for L in Ls]
    ttft_on_s = [by_L[str(L)]["ttft_ms_on"]["std"] for L in Ls]
    n = by_L[str(Ls[0])]["n_pairs"] if Ls else 0
    skip_Ls = sorted({s["L"] for s in summary["r2"]["track_a_skips"]})
    skip_reason = (summary["r2"]["track_a_skips"][0]["reason"]
                   if summary["r2"]["track_a_skips"] else None)
    return {"L": Ls, "ratio_mean": ratio_m, "ratio_std": ratio_s,
            "ttft_off": ttft_off, "ttft_off_std": ttft_off_s,
            "ttft_on": ttft_on, "ttft_on_std": ttft_on_s,
            "n_pairs": n, "skip_Ls": skip_Ls, "skip_reason": skip_reason}


def footer_text(summary):
    p2 = summary["provenance"]["r2"]
    seeds = p2.get("seeds") or []
    inv = p2.get("invocations")
    n = len(seeds) * (inv or 0)
    return (f"EXP-015, n={n} ({len(seeds)} seeds x {inv} invocations), "
            f"mean +/- std; {p2.get('model')}, "
            f"vLLM {p2.get('engine_version')}, git {str(p2.get('git_sha'))[:7]}")


# --------------------------------------------------------------------------
# figure 1: identification latency vs L (money figure)
# --------------------------------------------------------------------------

def fig_identification(summary, out_stem: Path, exp017_crossover=None,
                       footer=True):
    tb = track_b_series(summary)
    Ls = tb["L"]
    L_low, L_high = crossover_bracket(tb)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    series = [
        ("hashrope (persistent hash rope, O(log) query)", "hashrope",
         C_HASHROPE, "o", "-"),
        ("vendored SGLang radix matcher", "radix", C_RADIX, "s", "-"),
        ("numpy flat scan (O(L), no persistence)", "flat", C_FLAT, "^", "--"),
    ]
    for label, arm, color, marker, ls in series:
        ax.errorbar(Ls, tb["arms"][arm]["mean"], yerr=tb["arms"][arm]["std"],
                    color=color, marker=marker, ls=ls, lw=1.6, ms=5,
                    capsize=3, label=label, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(Ls)
    ax.xaxis.set_major_formatter(FuncFormatter(_short_tokens))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("shared-prefix length L (tokens)")
    ax.set_ylabel("identification latency (ms)")
    ax.grid(True, which="major", ls=":", lw=0.6, alpha=0.6)

    # crossover bracket shading + annotation
    if L_low is not None and L_high is not None:
        ax.axvspan(L_low, L_high, color=C_SHADE, alpha=0.25, zorder=1)
        ymin, _ymax = ax.get_ylim()
        ax.text((L_low * L_high) ** 0.5, ymin * 1.6,
                "live-stack crossover L*'\nbracketed here",
                ha="center", va="bottom", fontsize=8, color="#444444")
        i_hi = Ls.index(L_high)
        h = tb["arms"]["hashrope"]["mean"][i_hi]
        r = tb["arms"]["radix"]["mean"][i_hi]
        s = tb["sign"].get(L_high) or {}
        mult = r / h if h else float("nan")
        ax.annotate(
            f"hashrope {mult:.2f}x faster\n"
            f"sign {s.get('wins')}/{s.get('n')}, p={s.get('p_one_sided'):.3f}",
            xy=(L_high, h), xytext=(0.98, 0.38),
            textcoords="axes fraction", ha="right", va="top", fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"),
        )
    if exp017_crossover:
        ax.axvline(exp017_crossover, color="#555555", ls="--", lw=1.0,
                   zorder=2)
        ax.text(exp017_crossover, ax.get_ylim()[1] * 0.55,
                " EXP-017 CPU-model\n prediction",
                fontsize=7.5, color="#555555", ha="left", va="top",
                rotation=90)

    ax.legend(loc="upper left", fontsize=8, frameon=True)
    if footer:
        fig.text(0.99, 0.005, footer_text(summary), ha="right", va="bottom",
                 fontsize=6.5, color="#777777")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix("." + ext), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 2: engine caching energy vs L (attributed to the engine)
# --------------------------------------------------------------------------

def fig_caching_energy(summary, out_stem: Path, footer=True):
    ta = track_a_series(summary)
    Ls = ta["L"]
    all_Ls = sorted(set(Ls) | set(ta["skip_Ls"]))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]})

    # top: OFF/ON energy ratio
    ax1.errorbar(Ls, ta["ratio_mean"], yerr=ta["ratio_std"], color=C_ON,
                 marker="o", ls="-", lw=1.6, ms=5, capsize=3, zorder=3)
    for L, m in zip(Ls, ta["ratio_mean"]):
        ax1.annotate(f"{m:.1f}x", xy=(L, m), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=8)
    ax1.set_ylabel("GPU energy ratio\ncache-OFF / cache-ON (J/token)")
    ax1.grid(True, which="major", ls=":", lw=0.6, alpha=0.6)
    # camera-ready: 15% headroom so the top value label clears the panel title
    ax1.set_ylim(0, max(m + s for m, s in zip(ta["ratio_mean"], ta["ratio_std"])) * 1.15)

    # bottom: TTFT OFF vs ON
    ax2.errorbar(Ls, ta["ttft_off"], yerr=ta["ttft_off_std"], color=C_OFF,
                 marker="s", ls="-", lw=1.6, ms=5, capsize=3,
                 label="cache OFF (full prefill)", zorder=3)
    ax2.errorbar(Ls, ta["ttft_on"], yerr=ta["ttft_on_std"], color=C_ON,
                 marker="o", ls="-", lw=1.6, ms=5, capsize=3,
                 label="cache ON", zorder=3)
    ax2.set_yscale("log")
    ax2.set_ylabel("TTFT (ms)")
    ax2.set_xlabel("shared-prefix length L (tokens)")
    ax2.grid(True, which="major", ls=":", lw=0.6, alpha=0.6)
    ax2.legend(loc="center left", fontsize=8, frameon=True)

    # serving wall shading on both panels
    if ta["skip_Ls"] and Ls:
        wall_lo = max(Ls)
        wall_hi = max(ta["skip_Ls"])
        for ax in (ax1, ax2):
            ax.axvspan(wall_lo, wall_hi, color=C_SHADE, alpha=0.25, zorder=1)
        ax1.text((wall_lo * wall_hi) ** 0.5, ax1.get_ylim()[1] * 0.5,
                 "dense-attention\nserving wall\n(engine cannot prefill;\n"
                 "Track A skipped)",
                 ha="center", va="center", fontsize=7.5, color="#444444")

    ax2.set_xscale("log")
    ax2.set_xticks(all_Ls)
    ax2.xaxis.set_major_formatter(FuncFormatter(_short_tokens))
    ax2.xaxis.set_minor_formatter(NullFormatter())

    ax1.set_title("Engine prefix-cache effect (vLLM's caching, "
                  "not hashrope's)", fontsize=9)
    if footer:
        fig.text(0.99, 0.005, footer_text(summary), ha="right", va="bottom",
                 fontsize=6.5, color="#777777")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix("." + ext), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _synth_summary():
    def stat(m, s, n=9):
        return {"mean": m, "std": s, "min": m - s, "max": m + s, "n": n}

    def arm(m, s):
        return {"lat_ms": stat(m, s), "lat_ms_cellmeans": [m] * 9}

    tbl, tal = {}, {}
    for L, hr, rx, fl in ((4096, 46.0, 10.0, 0.05), (8192, 44.0, 53.0, 0.6)):
        tbl[str(L)] = {
            "L": L, "n_cells": 9,
            "arms": {"hashrope": arm(hr, 0.5), "radix": arm(rx, 2.0),
                     "flat": arm(fl, 0.01)},
            "sign_hashrope_beats_radix": {
                "n": 9, "wins": 9 if hr <= rx else 0,
                "losses": 0 if hr <= rx else 9, "ties": 0,
                "p_one_sided": 0.001953125 if hr <= rx else 1.0},
        }
    tal["4096"] = {"L": 4096, "n_pairs": 9,
                   "ratio_j": stat(70.0, 1.0),
                   "ttft_ms_off": stat(7800.0, 10.0),
                   "ttft_ms_on": stat(220.0, 3.0)}
    return {
        "experiment": "EXP-015", "claim": "S3",
        "provenance": {"r2": {"model": "synthetic", "engine_version": "x",
                              "git_sha": "f" * 40, "seeds": [42, 43, 44],
                              "invocations": 3}},
        "r2": {"track_b_by_L": tbl, "track_a_by_L": tal,
               "track_a_skips": [{"L": 8192, "cache_on": False, "seed": 42,
                                  "inv": 0,
                                  "reason":
                                  "engine_serve_infeasible_dense_attention",
                                  "detail": "synthetic"}]},
    }


def selftest():
    print("SELFTEST: rendering figures from a synthetic summary ...")
    s = _synth_summary()
    tb = track_b_series(s)
    lo, hi = crossover_bracket(tb)
    assert (lo, hi) == (4096, 8192), f"bracket wrong: {(lo, hi)}"
    print(f"  [ok] crossover bracket = ({lo}, {hi}]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fig_identification(s, td / "f1", exp017_crossover=6000)
        fig_caching_energy(s, td / "f2")
        for name in ("f1.png", "f1.pdf", "f2.png", "f2.pdf"):
            p = td / name
            assert p.exists() and p.stat().st_size > 1000, f"missing {name}"
            print(f"  [ok] {name} rendered ({p.stat().st_size:,} bytes)")
    print("SELFTEST PASS")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="EXP-015 figures (claim S3)")
    ap.add_argument("--summary",
                    default="experiments/exp_015_energy/results/"
                            "exp015_analysis_summary_latest.json")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--exp017-crossover", type=int, default=None,
                    help="EXP-017 CPU-model crossover prediction (tokens); "
                         "drawn as a dashed line when provided")
    ap.add_argument("--no-footer", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    summary = load_summary(args.summary)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    footer = not args.no_footer

    f1 = out / "exp015_identification_latency_vs_L"
    fig_identification(summary, f1, exp017_crossover=args.exp017_crossover,
                       footer=footer)
    print(f"wrote {f1}.png / .pdf")

    f2 = out / "exp015_caching_energy_vs_L"
    fig_caching_energy(summary, f2, footer=footer)
    print(f"wrote {f2}.png / .pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
