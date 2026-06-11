"""EXP-001 confirmatory figure: grouped bars (chunk / rayon / multiprocessing) with std error
bars across the seeds x invocations grid, annotated with the paired mp>rayon sign-test result.

Reads results/confirm/summary.json (produced by exp001_confirm.py). Error bars are std across the
n independent process invocations -- the confirmatory error model -- NOT the within-run bootstrap CI.

    python scripts/exp001_confirm_figure.py
"""
import os
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="experiments/exp_001_tokenizer_aligned/results/confirm/summary.json")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    d = json.load(open(args.summary, encoding="utf-8"))
    cells = list(d["cells"])
    metrics = ["chunk", "rayon", "mp"]
    RD, MD = d["rayon_degree"], d["mp_degree"]
    labels = {"chunk": "chunk-only (1 thread)", "rayon": f"rayon @ {RD}", "mp": f"multiproc @ {MD}"}
    x = np.arange(len(cells))
    w = 0.26

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, m in enumerate(metrics):
        means = [d["cells"][c][m]["mean"] for c in cells]
        stds = [d["cells"][c][m]["std"] for c in cells]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3, label=labels[m])

    # annotate mp bars with the paired sign-test outcome
    for j, c in enumerate(cells):
        st = d["cells"][c]["mp_minus_rayon"]["sign_test"]
        mm = d["cells"][c]["mp"]
        if st["n"]:
            sig = st["p_two_sided"] <= 0.05 and st["pos"] > st["neg"]
            txt = f"{st['pos']}/{st['n']}" + ("*" if sig else " n.s.")
            ax.text(x[j] + w, mm["mean"] + mm["std"] + 0.15, txt, ha="center", fontsize=8,
                    color="black" if sig else "gray")

    ax.axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.set_xticks(x)

    def lab(c):
        tok, b = c.split("@")
        return f"{tok}\n{int(b) // 1024 // 1024} MB"

    ax.set_xticklabels([lab(c) for c in cells], fontsize=9)
    ax.set_ylabel(f"speedup vs serial whole-string  (mean +/- std, n={d['n_per_cell']})")
    ax.set_title("EXP-001 confirmatory: ingestion speedup  (3 seeds x 3 invocations)\n"
                 "mp>rayon paired sign-test wins annotated (* = p<=0.05)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "exp001_confirmatory.png")
    plt.savefig(out, dpi=150)
    plt.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
