#!/usr/bin/env python3
"""EXP-020 (claim B2) -- analysis over the n>=9 raw results (crate 0.3.1).

Reads the tier-1 and tier-2 raw JSONs produced by exp020_bench_driver.py and computes
every number the promotion criterion needs, as mean +/- SD across the n>=9 independent
process invocations (the locked error model; the within-run criterion CI was retired in
EXP-001). Stdlib only -- no numpy/scipy -- so it runs anywhere the driver does.

Outputs: <results>/exp020_analysis_latest.json (+ a timestamped snapshot) and a console
summary with an explicit PASS/FAIL line for each HARD criterion.

Pairing note: one `cargo bench` invocation measures BOTH arms and ALL sizes, so hashrope
vs ropey at a given (seed, invocation, N) is a genuine pair (used for the sign test, the
paired speedup, and C). Tier-1 and tier-2 are SEPARATE invocations, so no run spans tiers:
the tier-1-only regime-A slope is fit per-run (mean +/- SD over 9 runs); the headline
tier-1+2 slope (10k..10M) is fit on the per-N means, and is labelled as such.

Usage:
    python exp020_analysis.py
    python exp020_analysis.py --tier1 <path> --tier2 <path> --out <dir>
"""

import argparse
import json
import math
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEF_RESULTS = HERE / "results"

R_GROUPS = {"r1": "b2_regimeB_r1", "r0.1": "b2_regimeB_r0_1", "r0.01": "b2_regimeB_r0_01"}
EDIT_GROUP = "b2_regimeA_edit"
SLOPE_BAR = 0.3          # criterion (ii)
SPEEDUP_BAR = 2.0        # criterion (iii)
C_BAR = 30.0             # criterion (iv) framing threshold


# ---------- small stats helpers ----------

def mean_sd(vals):
    """mean, sample SD (n-1), n. SD=0 when n<2."""
    n = len(vals)
    m = st.mean(vals)
    s = st.stdev(vals) if n > 1 else 0.0
    return {"mean": m, "sd": s, "n": n}


def linreg(xs, ys):
    """Least-squares y = a + b*x; returns slope b, intercept a, r2."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    b = sxy / sxx
    a = my - b * mx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else 1.0
    return b, a, r2


def loglog_slope_of_means(per_n_mean):
    """Slope of log10(mean) vs log10(N) over the given {N: mean}. Returns (slope, r2, Ns)."""
    Ns = sorted(per_n_mean)
    xs = [math.log10(n) for n in Ns]
    ys = [math.log10(per_n_mean[n]) for n in Ns]
    b, _, r2 = linreg(xs, ys)
    return b, r2, Ns


def sign_test_two_sided(wins, n):
    """Exact two-sided sign test p-value for `wins` successes in `n` Bernoulli(0.5)."""
    def tail_ge(k):
        return sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    def tail_le(k):
        return sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = 2.0 * min(tail_ge(wins), tail_le(wins))
    return min(1.0, p)


# ---------- load + index ----------

def load_raw(p):
    return json.loads(Path(p).read_text())


def index_timing(records):
    """(group, arm, n) -> {(seed, invocation): per_unit_ns}."""
    idx = {}
    for r in records:
        idx.setdefault((r["group"], r["arm"], r["n"]), {})[(r["seed"], r["invocation"])] = r["per_unit_ns"]
    return idx


def paired(idx, group, n):
    """Return aligned (hashrope, ropey) lists over the (seed,inv) keys present in BOTH arms."""
    h = idx.get((group, "hashrope", n), {})
    r = idx.get((group, "ropey", n), {})
    keys = sorted(set(h) & set(r))
    return [h[k] for k in keys], [r[k] for k in keys], keys


# ---------- main analysis ----------

def analyse(tier1, tier2):
    meta = {
        "tier1_meta": tier1["meta"],
        "tier2_meta": tier2["meta"],
        "error_model": "mean +/- sample SD (n-1) across n>=9 process invocations; paired sign test for regime B",
    }
    timing = tier1["timing"] + tier2["timing"]
    memprobe = tier1["memprobe"] + tier2["memprobe"]
    idx = index_timing(timing)
    all_Ns = sorted({n for (_, _, n) in idx})
    t1_Ns = sorted(set(tier1["meta"]["sizes"]))
    big_Ns = [n for n in all_Ns if n >= 10_000]

    out = {"meta": meta, "regimeA": {}, "regimeB": {}, "memory": {}, "verdict": {}}

    # ---- regime A: per-edit mean+/-SD per arm; C = hashrope/ropey paired ----
    regA = {"per_n": {}}
    for n in all_Ns:
        h, r, _ = paired(idx, EDIT_GROUP, n)
        C = [hi / ri for hi, ri in zip(h, r)]
        regA["per_n"][n] = {
            "hashrope_ns": mean_sd(h),
            "ropey_ns": mean_sd(r),
            "C_hashrope_over_ropey": mean_sd(C),
        }
    # slopes (criterion ii). hashrope is the gate; ropey reported alongside.
    regA["slope"] = {}
    for arm in ("hashrope", "ropey"):
        means_big = {n: regA["per_n"][n][f"{arm}_ns"]["mean"] for n in big_Ns}
        # tier-1-only, per-run (coherent within one invocation), mean+/-SD over runs
        t1_big = [n for n in t1_Ns if n >= 10_000]
        per_run_slopes = []
        # gather, for each (seed,inv) present across all t1_big for this arm, a slope
        run_keys = None
        arm_by_run = {}
        for n in t1_big:
            d = idx.get((EDIT_GROUP, arm, n), {})
            arm_by_run[n] = d
            ks = set(d)
            run_keys = ks if run_keys is None else (run_keys & ks)
        for k in sorted(run_keys or []):
            xs = [math.log10(n) for n in t1_big]
            ys = [math.log10(arm_by_run[n][k]) for n in t1_big]
            b, _, _ = linreg(xs, ys)
            per_run_slopes.append(b)
        t1_mean_slope, t1_r2, _ = loglog_slope_of_means({n: regA["per_n"][n][f"{arm}_ns"]["mean"] for n in t1_big})
        # tier-1+2 headline, slope of per-N means over 10k..10M
        t12_slope, t12_r2, t12_Ns = loglog_slope_of_means(means_big)
        regA["slope"][arm] = {
            "tier1_only_Nrange": [t1_big[0], t1_big[-1]],
            "tier1_only_per_run_slope": mean_sd(per_run_slopes),
            "tier1_only_means_slope": t1_mean_slope,
            "tier1_only_means_r2": t1_r2,
            "tier12_headline_Nrange": [t12_Ns[0], t12_Ns[-1]],
            "tier12_headline_means_slope": t12_slope,
            "tier12_headline_means_r2": t12_r2,
        }

    out["regimeA"] = regA

    # ---- regime B: per r, per N -- mean+/-SD both arms, paired speedup, sign test ----
    regB = {}
    for rlabel, group in R_GROUPS.items():
        per_n = {}
        for n in all_Ns:
            h, r, _ = paired(idx, group, n)
            if not h:
                continue
            speedup = [ri / hi for hi, ri in zip(h, r)]          # ropey/hashrope (>1 = hashrope faster)
            wins = sum(1 for hi, ri in zip(h, r) if hi < ri)     # hashrope-faster count
            per_n[n] = {
                "hashrope_ns": mean_sd(h),
                "ropey_ns": mean_sd(r),
                "speedup_ropey_over_hashrope": mean_sd(speedup),
                "sign_wins": wins,
                "sign_n": len(h),
                "sign_p_two_sided": sign_test_two_sided(wins, len(h)),
            }
        # crossover N*_id: smallest N where hashrope mean per-cycle < ropey mean per-cycle
        nstar = None
        for n in sorted(per_n):
            if per_n[n]["hashrope_ns"]["mean"] < per_n[n]["ropey_ns"]["mean"]:
                nstar = n
                break
        # monotonicity of the mean speedup across N
        ordered = [per_n[n]["speedup_ropey_over_hashrope"]["mean"] for n in sorted(per_n)]
        mono = all(b > a for a, b in zip(ordered, ordered[1:]))
        # per-cycle log-log slopes (mechanism): ropey ~O(N) query dominates, hashrope flat-ish
        h_slope, h_r2, _ = loglog_slope_of_means(
            {n: per_n[n]["hashrope_ns"]["mean"] for n in per_n if n >= 10_000})
        r_slope, r_r2, _ = loglog_slope_of_means(
            {n: per_n[n]["ropey_ns"]["mean"] for n in per_n if n >= 10_000})
        regB[rlabel] = {
            "per_n": per_n,
            "Nstar_id": nstar,
            "speedup_monotone_increasing": mono,
            "percycle_slope_hashrope_N>=10k": {"slope": h_slope, "r2": h_r2},
            "percycle_slope_ropey_N>=10k": {"slope": r_slope, "r2": r_r2},
        }
    out["regimeB"] = regB

    # ---- memory probe: nodes/op vs N (mean+/-SD over seeds), growth vs log2 N ----
    mem = {"per_n": {}}
    by_n = {}
    for rec in memprobe:
        by_n.setdefault(rec["n"], []).append(rec)
    for n in sorted(by_n):
        recs = by_n[n]
        mem["per_n"][n] = {
            "nodes_initial": mean_sd([x["nodes_initial"] for x in recs]),
            "nodes_after": mean_sd([x["nodes_after"] for x in recs]),
            "nodes_per_op": mean_sd([x["nodes_per_op"] for x in recs]),
        }
    npo_means = {n: mem["per_n"][n]["nodes_per_op"]["mean"] for n in mem["per_n"]}
    if len(npo_means) >= 2:
        xs = [math.log2(n) for n in sorted(npo_means)]
        ys = [npo_means[n] for n in sorted(npo_means)]
        b, a, r2 = linreg(xs, ys)
        mem["nodes_per_op_vs_log2N"] = {"slope_per_doubling": b, "intercept": a, "r2": r2}
    out["memory"] = mem

    # ---- verdict against HARD criteria ----
    hs = regA["slope"]["hashrope"]
    ii_t12 = hs["tier12_headline_means_slope"]
    ii_t1 = hs["tier1_only_per_run_slope"]["mean"]
    crit_ii = (ii_t12 <= SLOPE_BAR) and (ii_t1 <= SLOPE_BAR)

    r1 = regB["r1"]["per_n"]
    n1m = 1_000_000
    iii_sign = r1[n1m]["sign_wins"] == r1[n1m]["sign_n"]
    iii_speed = r1[n1m]["speedup_ropey_over_hashrope"]["mean"] >= SPEEDUP_BAR
    iii_mono = regB["r1"]["speedup_monotone_increasing"]
    crit_iii = iii_sign and iii_speed and iii_mono

    C1m = regA["per_n"][n1m]["C_hashrope_over_ropey"]["mean"]
    iv_competitive = C1m <= C_BAR

    out["verdict"] = {
        "criterion_ii_subLinearEdit_PASS": crit_ii,
        "criterion_ii_detail": {"tier12_slope": ii_t12, "tier1_per_run_slope_mean": ii_t1, "bar": SLOPE_BAR},
        "criterion_iii_differentiatorWin_PASS": crit_iii,
        "criterion_iii_detail": {
            "at_N": n1m,
            "sign_wins": r1[n1m]["sign_wins"], "sign_n": r1[n1m]["sign_n"],
            "sign_p_two_sided": r1[n1m]["sign_p_two_sided"],
            "speedup_mean": r1[n1m]["speedup_ropey_over_hashrope"]["mean"],
            "speedup_bar": SPEEDUP_BAR, "monotone_increasing": iii_mono,
        },
        "criterion_iv_editConstant": {
            "C_at_1M_mean": C1m, "bar": C_BAR,
            "competitive_clause_stands": iv_competitive,
        },
    }
    return out


# ---------- console report ----------

def fmt(ms, unit=""):
    return f"{ms['mean']:.1f}+/-{ms['sd']:.1f}{unit}"


def report(out):
    regA, regB, mem, v = out["regimeA"], out["regimeB"], out["memory"], out["verdict"]
    Ns = sorted(int(n) for n in regA["per_n"])

    print("\n================ EXP-020 / B2 analysis (crate 0.3.1, n>=9) ================")

    print("\n-- Regime A: per-edit latency (ns), mean +/- SD (n=9); C = hashrope/ropey (paired) --")
    print(f"{'N':>11} {'hashrope':>18} {'ropey':>16} {'C':>14}")
    for n in Ns:
        a = regA["per_n"][n]
        print(f"{n:>11} {fmt(a['hashrope_ns']):>18} {fmt(a['ropey_ns']):>16} "
              f"{a['C_hashrope_over_ropey']['mean']:>10.2f}x+/-{a['C_hashrope_over_ropey']['sd']:.2f}")
    hs = regA["slope"]["hashrope"]
    rs = regA["slope"]["ropey"]
    print(f"\n  regime-A log-log slope (criterion ii, bar <= {SLOPE_BAR}):")
    print(f"    hashrope tier-1 only [{hs['tier1_only_Nrange'][0]}..{hs['tier1_only_Nrange'][1]}]: "
          f"per-run {hs['tier1_only_per_run_slope']['mean']:.3f}+/-{hs['tier1_only_per_run_slope']['sd']:.3f}"
          f" (means fit {hs['tier1_only_means_slope']:.3f}, R2={hs['tier1_only_means_r2']:.4f})")
    print(f"    hashrope HEADLINE tier-1+2 [{hs['tier12_headline_Nrange'][0]}..{hs['tier12_headline_Nrange'][1]}]: "
          f"{hs['tier12_headline_means_slope']:.3f} (R2={hs['tier12_headline_means_r2']:.4f})")
    print(f"    ropey   HEADLINE tier-1+2: {rs['tier12_headline_means_slope']:.3f} "
          f"(R2={rs['tier12_headline_means_r2']:.4f})  [context: both O(log N)-class]")

    print("\n-- Regime B r=1: per-cycle latency (ns), mean +/- SD; speedup ropey/hashrope (paired); sign test --")
    print(f"{'N':>11} {'hashrope':>16} {'ropey':>20} {'speedup':>16} {'sign':>6} {'p2':>10}")
    for n in sorted(int(x) for x in regB["r1"]["per_n"]):
        b = regB["r1"]["per_n"][n]
        print(f"{n:>11} {fmt(b['hashrope_ns']):>16} {fmt(b['ropey_ns']):>20} "
              f"{b['speedup_ropey_over_hashrope']['mean']:>9.1f}x+/-{b['speedup_ropey_over_hashrope']['sd']:.1f} "
              f"{b['sign_wins']:>3}/{b['sign_n']:<2} {b['sign_p_two_sided']:>10.4g}")
    print(f"  monotone-increasing speedup across N: {regB['r1']['speedup_monotone_increasing']}")
    print(f"  per-cycle slope N>=10k: hashrope {regB['r1']['percycle_slope_hashrope_N>=10k']['slope']:.3f}, "
          f"ropey {regB['r1']['percycle_slope_ropey_N>=10k']['slope']:.3f}  "
          f"(ropey ~1 => O(N) query; hashrope small => O(1) query + log edits)")

    print("\n-- Regime B crossover N*_id by query:edit ratio r (honest negative for r<1) --")
    for rlabel in ("r1", "r0.1", "r0.01"):
        rr = regB[rlabel]
        ns = rr["Nstar_id"]
        ns_txt = "< smallest tested" if ns == Ns[0] else (str(ns) if ns else "> largest tested (no crossover)")
        print(f"    r={rlabel:<5} N*_id = {ns_txt}")

    print("\n-- Memory (persistent arena): nodes/op, mean +/- SD over seeds (the EXP-004 flip side) --")
    print(f"{'N':>11} {'nodes_initial':>16} {'nodes_after':>18} {'nodes_per_op':>16}")
    for n in sorted(int(x) for x in mem["per_n"]):
        mm = mem["per_n"][n]
        print(f"{n:>11} {mm['nodes_initial']['mean']:>16.0f} {mm['nodes_after']['mean']:>18.0f} "
              f"{fmt(mm['nodes_per_op']):>16}")
    if "nodes_per_op_vs_log2N" in mem:
        g = mem["nodes_per_op_vs_log2N"]
        print(f"  nodes/op vs log2(N): +{g['slope_per_doubling']:.1f} nodes per doubling (R2={g['r2']:.4f}) "
              f"-> retention ~O(log N) per op")

    print("\n================ VERDICT (HARD criteria) ================")
    print(f"  (ii)  sub-linear edit class:  {'PASS' if v['criterion_ii_subLinearEdit_PASS'] else 'FAIL'} "
          f"(tier1+2 slope {v['criterion_ii_detail']['tier12_slope']:.3f}, "
          f"tier1 per-run {v['criterion_ii_detail']['tier1_per_run_slope_mean']:.3f}; bar <= {v['criterion_ii_detail']['bar']})")
    d3 = v["criterion_iii_detail"]
    print(f"  (iii) differentiator win@1M:  {'PASS' if v['criterion_iii_differentiatorWin_PASS'] else 'FAIL'} "
          f"(sign {d3['sign_wins']}/{d3['sign_n']}, p2={d3['sign_p_two_sided']:.4g}, "
          f"speedup {d3['speedup_mean']:.1f}x >= {d3['speedup_bar']}x, monotone={d3['monotone_increasing']})")
    d4 = v["criterion_iv_editConstant"]
    print(f"  (iv)  edit constant C@1M:     C={d4['C_at_1M_mean']:.2f}x "
          f"(<= {d4['bar']}x => competitive clause {'STANDS' if d4['competitive_clause_stands'] else 'NARROWS to (iii)'})")
    print("  (i)/(v)/(vi): correctness harness green (separate); negatives above; all numbers mean +/- SD.")
    print("=========================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1", default=str(DEF_RESULTS / "exp020_tier1_raw_latest.json"))
    ap.add_argument("--tier2", default=str(DEF_RESULTS / "exp020_tier2_raw_latest.json"))
    ap.add_argument("--out", default=str(DEF_RESULTS))
    args = ap.parse_args()

    t1, t2 = load_raw(args.tier1), load_raw(args.tier2)
    out = analyse(t1, t2)
    report(out)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "exp020_analysis_latest.json").write_text(json.dumps(out, indent=2))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (outdir / f"exp020_analysis_{ts}.json").write_text(json.dumps(out, indent=2))
    print(f"[written] {outdir / 'exp020_analysis_latest.json'}")
    print(f"[written] {outdir / ('exp020_analysis_' + ts + '.json')}")


if __name__ == "__main__":
    main()
