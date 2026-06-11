"""EXP-001 CONFIRMATORY harness: cross-invocation x cross-seed error model for the ingestion speedup.

The reproducibility check (LOGBOOK Addendum B-repro) showed the within-run 7-rep bootstrap CI
UNDERESTIMATES run-to-run variance. So the confirmatory error bar is the spread ACROSS independent
process invocations and across corpus realizations -- not within one run.

For each corpus seed (>=3) and each independent invocation (>=3, fresh interpreter + fresh pool),
two bench subprocesses are launched per (seed, invocation):
  P : RAYON_NUM_THREADS=<rayon_degree>, no MP        -> rayon@degree speedup (and serial baseline)
  M : RAYON_NUM_THREADS=1, --mp-workers <mp_degree>  -> rayon@1 (chunk-only) + mp@degree speedup
Each invocation contributes ONE point estimate per metric; we aggregate mean +/- std across the
seeds x invocations grid (n = seeds * invocations), report a t-based 95% CI on the mean, and run a
paired sign test of mp@degree vs rayon@degree (paired by seed,invocation). Books are cached by
prep_corpus.py, so the >=3 corpora cost one download.

    python scripts/exp001_confirm.py --seeds 42,43,44 --invocations 3 `
        --tokenizers gpt2 t5-small --lengths 1048576,4194304 --reps 5 --rayon-degree 16 --mp-degree 16
"""
import os
import sys
import json
import math
import hashlib
import subprocess
import argparse
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
BENCH = os.path.join(REPO, "scripts", "exp001_bench.py")
PREP = os.path.join(REPO, "scripts", "prep_corpus.py")

# two-sided t critical values at 95% by df; df>30 -> normal approx
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31,
        9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13,
        16: 2.12, 18: 2.10, 20: 2.09, 25: 2.06, 30: 2.04}


def _tcrit(df):
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    if df > 30:
        return 1.96
    ks = [k for k in _T95 if k <= df]
    return _T95[max(ks)] if ks else 12.71


def _stats(xs):
    n = len(xs)
    if n == 0:
        return None
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    ci = _tcrit(n - 1) * sd / math.sqrt(n) if n > 1 else 0.0
    return {"mean": m, "std": sd, "ci95_halfwidth": ci, "min": min(xs), "max": max(xs), "n": n}


def _sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "n": 0, "p_two_sided": 1.0}
    k = max(pos, neg)
    p = 2.0 * sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return {"pos": pos, "neg": neg, "n": n, "p_two_sided": min(1.0, p)}


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def _run_bench(corpus, lengths, reps, toks, rayon_threads, mp_workers, out_dir):
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(rayon_threads)
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [PY, BENCH, "--tokenizers", *toks, "--corpus", corpus, "--lengths", lengths,
           "--reps", str(reps), "--out-dir", out_dir]
    if mp_workers:
        cmd += ["--mp-workers", mp_workers]
    subprocess.run(cmd, env=env, check=True)
    return json.load(open(os.path.join(out_dir, f"ingestion_rayon{rayon_threads}_latest.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--invocations", type=int, default=3)
    ap.add_argument("--tokenizers", nargs="+", default=["gpt2", "t5-small"])
    ap.add_argument("--lengths", default="1048576,4194304")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rayon-degree", type=int, default=16)
    ap.add_argument("--mp-degree", type=int, default=16)
    ap.add_argument("--books", type=int, default=15)
    ap.add_argument("--code-files", type=int, default=40)
    ap.add_argument("--corpus-dir", default="data/raw")
    ap.add_argument("--confirm-dir", default="experiments/exp_001_tokenizer_aligned/results/confirm")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x]
    lengths = [int(x) for x in args.lengths.split(",") if x]
    RD, MD = args.rayon_degree, args.mp_degree
    os.makedirs(args.confirm_dir, exist_ok=True)

    # 1. one corpus per seed (reuse if present; books cached so reshuffle is cheap)
    corpus_sha = {}
    for S in seeds:
        cp = os.path.join(args.corpus_dir, f"corpus_s{S}.txt")
        if not os.path.exists(cp):
            print(f"[prep] building corpus seed={S} -> {cp}", flush=True)
            subprocess.run([PY, PREP, "--seed", str(S), "--books", str(args.books),
                            "--code-files", str(args.code_files), "--out", cp,
                            "--readme", os.path.join(args.confirm_dir, f"readme_s{S}.md")], check=True)
        corpus_sha[S] = _sha(cp)
        print(f"[prep] seed={S} sha256[:16]={corpus_sha[S]}", flush=True)

    # 2. runs: data[(tok,bytes,metric)] -> [{seed,inv,speedup}]
    data = {}

    def add(tok, b, metric, S, inv, sp):
        data.setdefault((tok, b, metric), []).append({"seed": S, "inv": inv, "speedup": sp})

    for S in seeds:
        cp = os.path.join(args.corpus_dir, f"corpus_s{S}.txt")
        for inv in range(args.invocations):
            od = os.path.join(args.confirm_dir, f"s{S}_i{inv}")
            os.makedirs(od, exist_ok=True)
            print(f"\n=== seed={S} inv={inv} : P (rayon@{RD}) ===", flush=True)
            jp = _run_bench(cp, args.lengths, args.reps, args.tokenizers, RD, "", od)
            for r in jp["results"]:
                add(r["tokenizer"], r["corpus_bytes"], "rayon", S, inv,
                    r["measurements"]["rayon_batch"]["speedup_vs_serial"])
            print(f"=== seed={S} inv={inv} : M (rayon@1 chunk + mp@{MD}) ===", flush=True)
            jm = _run_bench(cp, args.lengths, args.reps, args.tokenizers, 1, str(MD), od)
            for r in jm["results"]:
                add(r["tokenizer"], r["corpus_bytes"], "chunk", S, inv,
                    r["measurements"]["rayon_batch"]["speedup_vs_serial"])
                add(r["tokenizer"], r["corpus_bytes"], "mp", S, inv,
                    r["measurements"][f"mp_{MD}"]["speedup_vs_serial"])

    # 3. aggregate
    summary = {"experiment": "EXP-001-confirmatory", "seeds": seeds,
               "invocations": args.invocations, "n_per_cell": len(seeds) * args.invocations,
               "rayon_degree": RD, "mp_degree": MD, "lengths": lengths,
               "tokenizers": args.tokenizers, "reps_per_invocation": args.reps,
               "corpus_sha256_16": corpus_sha, "cells": {}}
    for tok in args.tokenizers:
        for L in lengths:
            cell = {}
            for metric in ("chunk", "rayon", "mp"):
                vals = [d["speedup"] for d in data.get((tok, L, metric), [])]
                st = _stats(vals)
                if st:
                    st["per_seed_mean"] = {
                        S: round(statistics.mean([d["speedup"] for d in data[(tok, L, metric)] if d["seed"] == S]), 3)
                        for S in seeds if any(d["seed"] == S for d in data[(tok, L, metric)])}
                    cell[metric] = st
            rmap = {(d["seed"], d["inv"]): d["speedup"] for d in data.get((tok, L, "rayon"), [])}
            mmap = {(d["seed"], d["inv"]): d["speedup"] for d in data.get((tok, L, "mp"), [])}
            diffs = [mmap[k] - rmap[k] for k in rmap if k in mmap]
            cell["mp_minus_rayon"] = {"median_diff": (statistics.median(diffs) if diffs else None),
                                      "sign_test": _sign_test(diffs)}
            summary["cells"][f"{tok}@{L}"] = cell

    sp = os.path.join(args.confirm_dir, "summary.json")
    json.dump(summary, open(sp, "w"), indent=2)

    # 4. print table
    print(f"\n{'=' * 78}\nCONFIRMATORY SUMMARY  (n={len(seeds) * args.invocations} runs/cell ="
          f" {len(seeds)} seeds x {args.invocations} invocations; mean +/- std)")
    print(f"{'cell':<20}{'chunk':>16}{'rayon@' + str(RD):>16}{'mp@' + str(MD):>16}   mp>rayon (sign test)")
    for tok in args.tokenizers:
        for L in lengths:
            c = summary["cells"][f"{tok}@{L}"]

            def fmt(m):
                s = c.get(m)
                return f"{s['mean']:.2f}+/-{s['std']:.2f}" if s else "-"
            sg = c["mp_minus_rayon"]["sign_test"]
            print(f"{tok + ' ' + str(L // 1024) + 'K':<20}{fmt('chunk'):>16}{fmt('rayon'):>16}"
                  f"{fmt('mp'):>16}   {sg['pos']}/{sg['n']}  p={sg['p_two_sided']:.3f}")
    print(f"\nwrote {sp}")


if __name__ == "__main__":
    main()
