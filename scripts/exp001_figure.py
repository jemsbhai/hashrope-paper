"""EXP-001 analysis + figure: speedup-vs-cores curve and chunking/parallelism decomposition.

Reads ingestion_rayon<N>_latest.json files (one per RAYON_NUM_THREADS) from a results dir,
plots total speedup vs thread count per (tokenizer, context-length) cell, and prints a
decomposition table separating the single-thread chunking effect from parallel scaling.

    python scripts/exp001_figure.py --results-dir experiments/exp_001_tokenizer_aligned/results --out-dir figures
"""
import os
import re
import sys
import glob
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TAG = re.compile(r"ingestion_rayon([0-9]+)_latest\.json$")


def load(results_dir):
    """-> ({(tokenizer, bytes): {threads: speedup}}, cpu_count, {cell: tiled})"""
    cells, tiled = {}, {}
    cpu = None
    for path in sorted(glob.glob(os.path.join(results_dir, "ingestion_rayon*_latest.json"))):
        m = _TAG.search(os.path.basename(path))
        if not m:
            continue  # skip the 'all' tag; the curve needs explicit thread counts
        threads = int(m.group(1))
        d = json.load(open(path, encoding="utf-8"))
        cpu = cpu or d.get("env", {}).get("cpu_count")
        for r in d["results"]:
            key = (r["tokenizer"], r["corpus_bytes"])
            cells.setdefault(key, {})[threads] = r["measurements"]["rayon_batch"]["speedup_vs_serial"]
            tiled[key] = r.get("corpus_tiled", False)
    return cells, cpu, tiled


def decompose(cells):
    rows = []
    for (tok, nbytes), sp in sorted(cells.items()):
        threads = sorted(sp)
        base = sp.get(1)  # chunking-only factor (no parallelism)
        peak_t = max(threads, key=lambda t: sp[t])
        peak = sp[peak_t]
        par = (peak / base) if base else float("nan")
        eff = (par / peak_t) if base else float("nan")
        rows.append({"tokenizer": tok, "bytes": nbytes, "chunking_x": base,
                     "peak_x": peak, "peak_threads": peak_t,
                     "parallel_x": par, "efficiency": eff})
    return rows


def figure(cells, cpu, out_path):
    plt.figure(figsize=(8, 5.5))
    all_threads = sorted({t for sp in cells.values() for t in sp})
    for (tok, nbytes), sp in sorted(cells.items()):
        xs = sorted(sp)
        plt.plot(xs, [sp[t] for t in xs], marker="o", linewidth=1.8,
                 label=f"{tok} @ {nbytes // 1024}KB")
    ones = [sp[1] for sp in cells.values() if 1 in sp]
    if ones:
        base = sum(ones) / len(ones)
        plt.plot(all_threads, [base] * len(all_threads), "k--", alpha=0.55, linewidth=1,
                 label=f"chunking-only (~{base:.2f}x, no parallelism)")
    plt.xscale("log", base=2)
    plt.xticks(all_threads, [str(t) for t in all_threads])
    plt.xlabel("rayon threads (CPU cores used)")
    plt.ylabel("ingestion speedup vs serial whole-string")
    title = "EXP-001: tokenizer-aligned parallel tokenization"
    if cpu:
        title += f"  (host cpu_count={cpu})"
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path.replace(".png", ".pdf"))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/exp_001_tokenizer_aligned/results")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    cells, cpu, tiled = load(args.results_dir)
    if not cells:
        print("No ingestion_rayon<N>_latest.json files found in", args.results_dir)
        sys.exit(1)
    rows = decompose(cells)
    os.makedirs(args.out_dir, exist_ok=True)
    fig = figure(cells, cpu, os.path.join(args.out_dir, "exp001_speedup_vs_cores.png"))

    print(f"\nhost cpu_count = {cpu}\n")
    print(f"{'tokenizer':<12}{'KB':>6}{'tiled':>7}{'chunk_x':>9}{'peak_x':>8}{'@thr':>6}{'parallel_x':>12}{'eff':>7}")
    for r in rows:
        t = tiled[(r["tokenizer"], r["bytes"])]
        print(f"{r['tokenizer']:<12}{r['bytes'] // 1024:>6}{str(t):>7}"
              f"{r['chunking_x']:>9.2f}{r['peak_x']:>8.2f}{r['peak_threads']:>6}"
              f"{r['parallel_x']:>12.2f}{r['efficiency'] * 100:>6.0f}%")
    print(f"\nwrote {fig} (+ .pdf)")


if __name__ == "__main__":
    main()
