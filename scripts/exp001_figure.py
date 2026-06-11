"""EXP-001 analysis + figures: rayon-batch vs multiprocessing scaling, with the
chunking-vs-parallelism decomposition.

Reads ingestion_rayon<N>_latest.json files (one per RAYON_NUM_THREADS). The rayon curve
comes from each file's rayon_batch at thread count N; the multiprocessing curve comes from
any file carrying mp_<W> measurements (run once with --mp-workers). Produces:
  - exp001_speedup_vs_cores.png : rayon scaling across (tokenizer, length) cells
  - exp001_rayon_vs_mp.png      : rayon vs MP at the largest context (if MP data present)
and prints a table with chunking factor and both peaks side by side.

    python scripts/exp001_figure.py
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
    rayon, mp, tiled = {}, {}, {}
    cpu = None
    for path in sorted(glob.glob(os.path.join(results_dir, "ingestion_rayon*_latest.json"))):
        m = _TAG.search(os.path.basename(path))
        if not m:
            continue  # skip the 'all' tag; curves need explicit thread counts
        threads = int(m.group(1))
        d = json.load(open(path, encoding="utf-8"))
        cpu = cpu or d.get("env", {}).get("cpu_count")
        for r in d["results"]:
            key = (r["tokenizer"], r["corpus_bytes"])
            meas = r["measurements"]
            rayon.setdefault(key, {})[threads] = meas["rayon_batch"]["speedup_vs_serial"]
            tiled[key] = r.get("corpus_tiled", False)
            for k, v in meas.items():
                if k.startswith("mp_"):
                    mp.setdefault(key, {})[int(k[3:])] = v["speedup_vs_serial"]
    return rayon, mp, cpu, tiled


def scaling_fig(rayon, cpu, out):
    plt.figure(figsize=(8, 5.5))
    allt = sorted({t for sp in rayon.values() for t in sp})
    for (tok, nb), sp in sorted(rayon.items()):
        xs = sorted(sp)
        plt.plot(xs, [sp[t] for t in xs], marker="o", linewidth=1.7, label=f"{tok} @ {nb // 1024}KB")
    ones = [sp[1] for sp in rayon.values() if 1 in sp]
    if ones:
        base = sum(ones) / len(ones)
        plt.plot(allt, [base] * len(allt), "k--", alpha=0.55, linewidth=1,
                 label=f"chunking-only (~{base:.2f}x)")
    plt.xscale("log", base=2)
    plt.xticks(allt, [str(t) for t in allt])
    plt.xlabel("rayon threads (cores)")
    plt.ylabel("speedup vs serial whole-string")
    plt.title(f"EXP-001: tokenizer-aligned parallel tokenization (cpu_count={cpu})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace(".png", ".pdf"))


def compare_fig(rayon, mp, cpu, out):
    if not mp:
        return False
    L = sorted({b for (_, b) in rayon})[-1]
    plt.figure(figsize=(8, 5.5))
    for (tok, nb) in sorted(rayon):
        if nb != L:
            continue
        xs = sorted(rayon[(tok, nb)])
        plt.plot(xs, [rayon[(tok, nb)][t] for t in xs], marker="o", linewidth=1.8, label=f"{tok} rayon")
        if (tok, nb) in mp:
            xm = sorted(mp[(tok, nb)])
            plt.plot(xm, [mp[(tok, nb)][w] for w in xm], marker="s", linestyle="--", linewidth=1.8,
                     label=f"{tok} multiproc")
    plt.xscale("log", base=2)
    xt = sorted({t for (tok, nb) in rayon if nb == L for t in rayon[(tok, nb)]} |
                {w for (tok, nb) in mp if nb == L for w in mp[(tok, nb)]})
    plt.xticks(xt, [str(t) for t in xt])
    plt.xlabel("workers / threads (cores)")
    plt.ylabel("speedup vs serial whole-string")
    plt.title(f"EXP-001: rayon vs multiprocessing @ {L // 1024}KB (cpu_count={cpu})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace(".png", ".pdf"))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/exp_001_tokenizer_aligned/results")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    rayon, mp, cpu, tiled = load(args.results_dir)
    if not rayon:
        print("No ingestion_rayon<N>_latest.json files found in", args.results_dir)
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    scaling_fig(rayon, cpu, os.path.join(args.out_dir, "exp001_speedup_vs_cores.png"))
    has_mp = compare_fig(rayon, mp, cpu, os.path.join(args.out_dir, "exp001_rayon_vs_mp.png"))

    print(f"\ncpu_count = {cpu}\n")
    hdr = f"{'tokenizer':<11}{'KB':>6}{'tiled':>7}{'chunk_x':>9}{'rayon_pk':>10}{'@thr':>6}"
    if mp:
        hdr += f"{'mp_pk':>9}{'@wrk':>6}{'winner':>8}"
    print(hdr)
    for (tok, nb) in sorted(rayon):
        sp = rayon[(tok, nb)]
        base = sp.get(1, float("nan"))
        rt = max(sp, key=lambda t: sp[t]); rpk = sp[rt]
        line = f"{tok:<11}{nb // 1024:>6}{str(tiled[(tok, nb)]):>7}{base:>9.2f}{rpk:>10.2f}{rt:>6}"
        if mp and (tok, nb) in mp:
            mw = max(mp[(tok, nb)], key=lambda w: mp[(tok, nb)][w]); mpk = mp[(tok, nb)][mw]
            line += f"{mpk:>9.2f}{mw:>6}{('mp' if mpk > rpk else 'rayon'):>8}"
        print(line)
    print(f"\nwrote figures (scaling{' + rayon_vs_mp' if has_mp else ''})")


if __name__ == "__main__":
    main()
