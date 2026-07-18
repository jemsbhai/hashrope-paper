#!/usr/bin/env python3
"""EXP-015 analysis (claim S3): aggregation + paired tests + verbatim criterion evaluation.

Pure CPU, stdlib-only. No GPU, no engine, no numpy. Reads the raw grid JSONs
(exp015_R1_raw_latest.json, exp015_R2_raw_latest.json) and the negative-control
JSONs (exp015_negative_control_{R1,R2}.json), then:

  1. Aggregates n>=9 (3 seeds x 3 invocations) mean +/- std for every metric:
     R1 caching ratio; R2 Track A OFF/ON energy + TTFT per L; R1/R2 Track B
     identification latencies per arm.
  2. Computes Delta(ON-OFF) with exact paired sign tests (criterion iii) and
     hashrope-vs-radix paired sign tests per L (criterion iv).
  3. Evaluates the pre-registered promotion criterion (i)-(vi) + contingent F6
     (verbatim text in LOGBOOK.md, "Promotion criterion (verbatim; ...)"),
     applying the SETTLED operationalizations recorded in the LOGBOOK
     deviation entries:
       (i)   gate exactness + non-vacuity via corrupted-cached-entry controls;
       (ii)  GPU-energy neutrality is BY CONSTRUCTION (one engine run per
             cell; the identifier never feeds the engine, so no per-arm
             engine energies exist to difference) -- report the single
             cache-ON J/token + its run-to-run band; never fabricate
             per-arm engine energies;
       (iii) engine caching Delta(ON-OFF), attributed to the ENGINE;
       (iv)  R2 identification win beyond the live-stack crossover L*'
             (bracketed) + R1 honest negative;
       (v)   honest negatives in full;
       (vi)  n>=9 mean+/-std, paired sign tests, clocks + env recorded;
       F6    NOT triggered (structural: no per-arm end-to-end engine metrics
             exist under the one-engine-run-per-cell design).
  4. Prints a clear S3 verdict and writes a machine-readable summary JSON
     (single source of truth for figures + governance files).

Conventions (printed in the report header):
  - std        : sample standard deviation (ddof=1) across the n>=9
                 (seed, invocation) cells; per-cell request-level spread is
                 reported by the grid itself inside each cell.
  - sign test  : exact one-sided binomial, ties dropped,
                 p = P(X >= wins | n, 0.5).
  - 95% band   : mean +/- t(0.975, df=n-1) * std (run-to-run band).
  - J/token    : total_joules / total COMPLETION tokens. Absolute values are
                 prefill-dominated (inflated); the OFF/ON RATIO and the DELTA
                 are the meaningful quantities (stated wherever reported).

Self-test:
  python exp015_analysis.py --selftest
  Builds synthetic R1/R2 grids + negative-control files in both observed
  schemas, runs the full pipeline, asserts expected clause statuses, the
  crossover bracket, exact sign-test p-values, and that an injected gate
  mismatch flips (i) and the verdict to BLOCKED.

Usage on real results (from repo root):
  python scripts/exp015_analysis.py \
      --r1 experiments/exp_015_energy/results/exp015_R1_raw_latest.json \
      --r2 experiments/exp_015_energy/results/exp015_R2_raw_latest.json \
      --negctl-r1 experiments/exp_015_energy/results/exp015_negative_control_R1.json \
      --negctl-r2 experiments/exp_015_energy/results/exp015_negative_control_R2.json \
      --out-dir experiments/exp_015_energy/results
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# small stats helpers
# --------------------------------------------------------------------------

# two-sided 95% t critical values, df = 1..30 (df > 30 -> 1.96)
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t95(n: int) -> float:
    df = n - 1
    if df < 1:
        return float("nan")
    return _T95.get(df, 1.96)


def agg(xs):
    """mean/std/min/max/n over a list. std = sample std (ddof=1), 0.0 if n<2."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    m = sum(xs) / n
    s = statistics.stdev(xs) if n >= 2 else 0.0
    return {"mean": m, "std": s, "min": min(xs), "max": max(xs), "n": n}


def band95(a):
    """95% run-to-run band (t-based) from an agg() dict; None if n<2."""
    if a["n"] < 2 or a["mean"] is None:
        return None
    h = t95(a["n"]) * a["std"]
    return {"lo": a["mean"] - h, "hi": a["mean"] + h, "half_width": h}


def sign_test(deltas, positive_is_win: bool):
    """Exact one-sided paired sign test. Ties (delta == 0) are dropped.

    positive_is_win=True  -> a 'win' is delta > 0.
    positive_is_win=False -> a 'win' is delta < 0.
    Returns {n, wins, losses, ties, p_one_sided}.
    p = P(X >= wins | Binomial(n, 0.5)).
    """
    ties = sum(1 for d in deltas if d == 0)
    eff = [d for d in deltas if d != 0]
    n = len(eff)
    if positive_is_win:
        wins = sum(1 for d in eff if d > 0)
    else:
        wins = sum(1 for d in eff if d < 0)
    if n == 0:
        p = None
    else:
        p = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
    return {"n": n, "wins": wins, "losses": n - wins, "ties": ties,
            "p_one_sided": p}


def fnum(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if x != x:  # NaN
            return "nan"
        return f"{x:,.{nd}f}"
    return f"{x:,}"


def pfmt(p):
    if p is None:
        return "n/a"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# loading + cell extraction
# --------------------------------------------------------------------------

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _track_a_ok(cell):
    ta = cell.get("track_a")
    return ta is not None and not ta.get("skipped")


def extract_grid(doc):
    """Index a raw grid doc. Returns dict with:
       a_cells  : {(L, cache_on, seed, inv): track_a}   (non-skipped only)
       a_skips  : list of {L, cache_on, seed, inv, reason, detail}
       b_cells  : {(L, seed, inv): track_b}             (whichever cells carry it)
       gate     : {"checks": int, "mismatches": int, "cells": int}
       L_values : sorted list (R1 -> [None])
    """
    a_cells, a_skips, b_cells = {}, [], {}
    checks = mm = ncell = 0
    for c in doc.get("cells", []):
        L = c.get("L")
        key_a = (L, bool(c["cache_on"]), c["seed"], c["inv"])
        ta = c.get("track_a")
        if ta is not None:
            if ta.get("skipped"):
                a_skips.append({"L": L, "cache_on": bool(c["cache_on"]),
                                "seed": c["seed"], "inv": c["inv"],
                                "reason": ta.get("reason"),
                                "detail": ta.get("detail")})
            else:
                a_cells[key_a] = ta
        tb = c.get("track_b")
        if tb:
            b_cells[(L, c["seed"], c["inv"], bool(c["cache_on"]))] = tb
            g = tb.get("gate") or {}
            checks += int(g.get("n_checks", 0))
            mm += int(g.get("mismatches", 0))
            ncell += 1
    Ls = sorted({k[0] for k in a_cells} |
                {s["L"] for s in a_skips} |
                {k[0] for k in b_cells}, key=lambda x: (x is not None, x))
    return {"a_cells": a_cells, "a_skips": a_skips, "b_cells": b_cells,
            "gate": {"checks": checks, "mismatches": mm, "cells": ncell},
            "L_values": Ls}


# --------------------------------------------------------------------------
# Track A analysis (engine energy / TTFT / throughput), paired OFF vs ON
# --------------------------------------------------------------------------

def analyze_track_a(grid, L):
    """Paired OFF/ON analysis at one L (L=None for R1). Returns None if no
    (seed, inv) pair has both OFF and ON non-skipped Track A."""
    a = grid["a_cells"]
    si = sorted({(k[2], k[3]) for k in a if k[0] == L})
    pairs = []
    for (seed, inv) in si:
        off = a.get((L, False, seed, inv))
        on = a.get((L, True, seed, inv))
        if not (off and on):
            continue
        jt_off = off.get("j_per_token")
        jt_on = on.get("j_per_token")
        tj_off = off.get("total_joules")
        tj_on = on.get("total_joules")
        tt_off = (off.get("ttft_ms") or {}).get("mean")
        tt_on = (on.get("ttft_ms") or {}).get("mean")
        th_off = off.get("throughput_tok_s")
        th_on = on.get("throughput_tok_s")
        pairs.append({
            "seed": seed, "inv": inv,
            "j_per_token_off": jt_off, "j_per_token_on": jt_on,
            "total_joules_off": tj_off, "total_joules_on": tj_on,
            "ratio_j": (jt_off / jt_on) if (jt_off and jt_on) else None,
            "delta_j_per_token": (jt_on - jt_off)
                if (jt_off is not None and jt_on is not None) else None,
            "ttft_ms_off": tt_off, "ttft_ms_on": tt_on,
            "delta_ttft_ms": (tt_on - tt_off)
                if (tt_off is not None and tt_on is not None) else None,
            "thr_tok_s_off": th_off, "thr_tok_s_on": th_on,
            "cached_tokens_on": on.get("total_cached_tokens"),
        })
    if not pairs:
        return None

    ratios = [p["ratio_j"] for p in pairs]
    d_j = [p["delta_j_per_token"] for p in pairs if p["delta_j_per_token"] is not None]
    d_tt = [p["delta_ttft_ms"] for p in pairs if p["delta_ttft_ms"] is not None]
    per_seed = {}
    for s in sorted({p["seed"] for p in pairs}):
        per_seed[str(s)] = agg([p["ratio_j"] for p in pairs if p["seed"] == s])

    on_j = agg([p["j_per_token_on"] for p in pairs])
    res = {
        "L": L,
        "n_pairs": len(pairs),
        "pairs": pairs,
        "ratio_j": agg(ratios),
        "ratio_j_per_seed": per_seed,
        "j_per_token_off": agg([p["j_per_token_off"] for p in pairs]),
        "j_per_token_on": on_j,
        "j_per_token_on_band95": band95(on_j),
        "total_joules_off": agg([p["total_joules_off"] for p in pairs]),
        "total_joules_on": agg([p["total_joules_on"] for p in pairs]),
        "delta_j_per_token": agg(d_j),
        "sign_delta_j": sign_test(d_j, positive_is_win=False),  # win: ON < OFF
        "ttft_ms_off": agg([p["ttft_ms_off"] for p in pairs]),
        "ttft_ms_on": agg([p["ttft_ms_on"] for p in pairs]),
        "delta_ttft_ms": agg(d_tt),
        "sign_delta_ttft": sign_test(d_tt, positive_is_win=False),  # win: ON < OFF
        "thr_tok_s_off": agg([p["thr_tok_s_off"] for p in pairs]),
        "thr_tok_s_on": agg([p["thr_tok_s_on"] for p in pairs]),
        # regression checks for clause (v): ON worse than OFF
        "n_ttft_regressions": sum(
            1 for p in pairs
            if p["ttft_ms_off"] is not None and p["ttft_ms_on"] is not None
            and p["ttft_ms_on"] > p["ttft_ms_off"]),
        "n_thr_regressions": sum(
            1 for p in pairs
            if p["thr_tok_s_off"] is not None and p["thr_tok_s_on"] is not None
            and p["thr_tok_s_on"] < p["thr_tok_s_off"]),
    }
    cached = [p["cached_tokens_on"] for p in pairs]
    res["cached_token_counter_available"] = any(
        (c is not None and c > 0) for c in cached)
    res["cached_tokens_on_values"] = cached
    return res


# --------------------------------------------------------------------------
# Track B analysis (identification arms), per L, cells that carry track_b
# --------------------------------------------------------------------------

def analyze_track_b(grid, L):
    cells = [(k, v) for k, v in grid["b_cells"].items() if k[0] == L]
    if not cells:
        return None
    cells.sort(key=lambda kv: (kv[0][1], kv[0][2]))
    arm_names = set()
    for _k, tb in cells:
        arm_names |= set((tb.get("arms") or {}).keys())
    arms = {}
    for a in sorted(arm_names):
        per_cell = []
        for _k, tb in cells:
            d = (tb.get("arms") or {}).get(a)
            if d:
                per_cell.append((d.get("lat_ms") or {}).get("mean"))
        arms[a] = {"lat_ms_cellmeans": per_cell, "lat_ms": agg(per_cell)}

    res = {
        "L": L,
        "n_cells": len(cells),
        "arms": arms,
        "oracle_lcp": agg([(tb.get("oracle_lcp") or {}).get("mean")
                           for _k, tb in cells]),
        "gate_checks": sum((tb.get("gate") or {}).get("n_checks", 0)
                           for _k, tb in cells),
        "gate_mismatches": sum((tb.get("gate") or {}).get("mismatches", 0)
                               for _k, tb in cells),
        "radix_available": all(tb.get("radix_available", False)
                               for _k, tb in cells),
    }
    hr = arms.get("hashrope", {}).get("lat_ms_cellmeans", [])
    rx = arms.get("radix", {}).get("lat_ms_cellmeans", [])
    if hr and rx and len(hr) == len(rx):
        d = [r - h for h, r in zip(hr, rx)]  # >0 -> hashrope faster
        res["delta_radix_minus_hashrope_ms"] = agg(d)
        res["sign_hashrope_beats_radix"] = sign_test(d, positive_is_win=True)
        res["hashrope_over_radix_ratio"] = (
            arms["hashrope"]["lat_ms"]["mean"] / arms["radix"]["lat_ms"]["mean"]
            if arms["radix"]["lat_ms"]["mean"] else None)
    return res


# --------------------------------------------------------------------------
# negative controls (two observed schemas)
# --------------------------------------------------------------------------

def analyze_negctl(doc):
    """R1 schema: rows[] with int 'controls' + 'all_detected' per row.
       R2 schema: rows[] with list 'controls', each control has 'detected'.
       Returns totals + per-row summaries + grid_gate_context passthrough."""
    total = detected = 0
    rows_out = []
    for r in doc.get("rows", []):
        c = r.get("controls")
        if isinstance(c, int):
            n = c
            det = n if r.get("all_detected") else None
            if det is None:
                # fall back to counting the sample if full detail absent
                det = sum(1 for s in r.get("sample", []) if s.get("detected"))
            rows_out.append({"key": r.get("dataset") or r.get("L"),
                             "n": n, "detected": det})
            total += n
            detected += det
        elif isinstance(c, list):
            n = len(c)
            det = sum(1 for x in c if x.get("detected"))
            rows_out.append({"key": r.get("dataset") or r.get("L"),
                             "n": n, "detected": det})
            total += n
            detected += det
    return {
        "total": total,
        "detected": detected,
        "all_detected": bool(doc.get("all_detected", detected == total and total > 0)),
        "rows": rows_out,
        "grid_gate_context": doc.get("grid_gate_context"),
        "env": doc.get("env"),
        "seed": doc.get("seed"),
    }


# --------------------------------------------------------------------------
# criterion (i)-(vi) + F6 evaluation
# --------------------------------------------------------------------------

HARD = {"i", "ii", "iii"}


def evaluate_criterion(r1a, r1b, r2a_by_L, r2b_by_L, r2_skips, nc1, nc2,
                       gate_total, env_check):
    clauses = []

    # ---- (i) exactness + non-vacuity --------------------------------------
    nc_total = nc1["total"] + nc2["total"]
    nc_det = nc1["detected"] + nc2["detected"]
    gate_ok = gate_total["mismatches"] == 0 and gate_total["checks"] > 0
    nc_ok = (nc_det == nc_total and nc_total > 0
             and nc1["all_detected"] and nc2["all_detected"])
    ctx_ok = True
    for nc in (nc1, nc2):
        ctx = nc.get("grid_gate_context") or {}
        if ctx and ctx.get("grid_gate_checks") not in (None, gate_total["checks"]):
            ctx_ok = False
    status = "PASS" if (gate_ok and nc_ok) else "BLOCKED"
    detail = (f"gate {gate_total['mismatches']}/{gate_total['checks']} mismatches "
              f"(R1 {gate_total['r1_mismatches']}/{gate_total['r1_checks']}, "
              f"R2 {gate_total['r2_mismatches']}/{gate_total['r2_checks']}); "
              f"non-vacuous via {nc_det}/{nc_total} corrupted-cached-entry "
              f"controls detected (R1 {nc1['detected']}/{nc1['total']}, "
              f"R2 {nc2['detected']}/{nc2['total']})")
    if not ctx_ok:
        detail += ("; WARNING: negative-control grid_gate_context checks "
                   "disagree with the computed gate total")
        status = "BLOCKED"
    clauses.append({"clause": "i", "name": "exactness (HARD)",
                    "status": status, "detail": detail})

    # ---- (ii) GPU-energy neutrality: BY CONSTRUCTION ----------------------
    bands = []
    for lab, ta in [("R1", r1a)] + [(f"R2 L={L}", r2a_by_L.get(L))
                                    for L in sorted(r2a_by_L)]:
        if not ta:
            continue
        b = ta["j_per_token_on_band95"]
        thr = 0.02 * ta["j_per_token_on"]["mean"] if ta["j_per_token_on"]["mean"] else None
        bands.append(
            f"{lab}: cache-ON J/token {fnum(ta['j_per_token_on']['mean'])} "
            f"+/- {fnum(ta['j_per_token_on']['std'])} (n={ta['j_per_token_on']['n']}, "
            f"95% band +/- {fnum(b['half_width']) if b else 'n/a'}; "
            f"delta_E threshold 2% = {fnum(thr)})")
    detail = ("BY CONSTRUCTION: one engine run per cell; the identifier never "
              "feeds the engine, so |J/token(hashrope) - J/token(radix)| is "
              "identically 0 (identical dispatched GPU work) and lies trivially "
              "within the 2% delta_E margin and any run-to-run band. No per-arm "
              "engine energies exist; none are fabricated. "
              + " | ".join(bands)
              + " | NOTE: absolute J/token is prefill-dominated (denominator = "
                "completion tokens); the OFF/ON ratio and delta are the "
                "meaningful quantities.")
    clauses.append({"clause": "ii",
                    "name": "GPU-energy neutrality of identity layer (HARD)",
                    "status": "PASS", "detail": detail})

    # ---- (iii) framework caching energy, measured + attributed ------------
    parts = []
    reduction_everywhere = True
    for lab, ta in [("R1", r1a)] + [(f"R2 L={L}", r2a_by_L.get(L))
                                    for L in sorted(r2a_by_L)]:
        if not ta:
            continue
        sj, st = ta["sign_delta_j"], ta["sign_delta_ttft"]
        red = (sj["wins"] == sj["n"] and sj["n"] > 0)
        reduction_everywhere &= red
        parts.append(
            f"{lab}: Delta(ON-OFF) J/token {fnum(ta['delta_j_per_token']['mean'])} "
            f"+/- {fnum(ta['delta_j_per_token']['std'])} "
            f"(ratio OFF/ON {fnum(ta['ratio_j']['mean'],2)}x +/- "
            f"{fnum(ta['ratio_j']['std'],2)}, n={ta['n_pairs']}; "
            f"sign ON<OFF {sj['wins']}/{sj['n']}, p={pfmt(sj['p_one_sided'])}); "
            f"Delta TTFT {fnum(ta['delta_ttft_ms']['mean'],1)} ms "
            f"(sign {st['wins']}/{st['n']}, p={pfmt(st['p_one_sided'])})")
    if r2_skips:
        Ls = sorted({s['L'] for s in r2_skips})
        parts.append(f"R2 L in {Ls}: Track A SKIPPED "
                     f"({r2_skips[0]['reason']}; {len(r2_skips)} cells) -- "
                     "no engine energy at these L by design")
    detail = ("ATTRIBUTION: the ON-OFF delta is the ENGINE's prefix-cache "
              "effect, explicitly NOT hashrope's. " + " | ".join(parts))
    status = "PASS" if parts else "BLOCKED"
    if parts and not reduction_everywhere:
        detail += (" | NOTE: at least one config shows no reduction; per the "
                   "verbatim clause this re-characterizes the framework and "
                   "does not block S3.")
    clauses.append({"clause": "iii",
                    "name": "framework caching energy, attributed (HARD)",
                    "status": status, "detail": detail})

    # ---- (iv) identification regimes: R2 win + R1 honest negative ---------
    Ls = sorted(r2b_by_L)
    radix_faster, hashrope_faster = [], []
    for L in Ls:
        tb = r2b_by_L[L]
        hm = tb["arms"].get("hashrope", {}).get("lat_ms", {}).get("mean")
        rm = tb["arms"].get("radix", {}).get("lat_ms", {}).get("mean")
        if hm is None or rm is None:
            continue
        (hashrope_faster if hm <= rm else radix_faster).append(L)
    L_low = max(radix_faster) if radix_faster else None
    L_high = min(hashrope_faster) if hashrope_faster else None
    win_ok = False
    win_txt = "no L with hashrope <= radix found"
    if L_high is not None:
        tb = r2b_by_L[L_high]
        s = tb.get("sign_hashrope_beats_radix") or {}
        hm = tb["arms"]["hashrope"]["lat_ms"]
        rm = tb["arms"]["radix"]["lat_ms"]
        win_ok = (s.get("wins") == s.get("n") and s.get("n", 0) >= 9
                  and (s.get("p_one_sided") or 1) < 0.05)
        mult = (rm["mean"] / hm["mean"]) if hm["mean"] else None
        win_txt = (f"crossover L*' bracketed in ({fnum(L_low,0)}, {fnum(L_high,0)}]: "
                   f"radix still faster at L={fnum(L_low,0)}, hashrope faster at "
                   f"L={fnum(L_high,0)} ({fnum(hm['mean'],2)} +/- {fnum(hm['std'],2)} ms "
                   f"vs radix {fnum(rm['mean'],2)} +/- {fnum(rm['std'],2)} ms, "
                   f"hashrope {fnum(mult,2)}x faster; "
                   f"paired sign {s.get('wins')}/{s.get('n')}, "
                   f"p={pfmt(s.get('p_one_sided'))})")
    # overhead as a fraction of prefill TTFT (reference = largest servable L)
    frac_txt = "no Track A reference available"
    ref_L = max(r2a_by_L) if r2a_by_L else None
    if L_high is not None and ref_L is not None:
        hr_ms = r2b_by_L[L_high]["arms"]["hashrope"]["lat_ms"]["mean"]
        ta = r2a_by_L[ref_L]
        off_ms = ta["ttft_ms_off"]["mean"]
        on_ms = ta["ttft_ms_on"]["mean"]
        frac_txt = (
            f"hashrope identification at L={fnum(L_high,0)} = {fnum(hr_ms,2)} ms "
            f"= {hr_ms/off_ms*100:.3f}% of full-prefill (cache-OFF) TTFT and "
            f"{hr_ms/on_ms*100:.2f}% of cache-ON TTFT at L={fnum(ref_L,0)} "
            f"(largest dense-servable L; conservative reference -- TTFT grows "
            f"superlinearly in L, so true fractions at L={fnum(L_high,0)} "
            f"would be smaller)")
    # R1 honest negative
    r1_txt = "R1 Track B missing"
    if r1b:
        h = r1b["arms"].get("hashrope", {}).get("lat_ms", {})
        r = r1b["arms"].get("radix", {}).get("lat_ms", {})
        f = r1b["arms"].get("flat", {}).get("lat_ms", {})
        r1_txt = (f"R1 HONEST NEGATIVE (~{fnum(r1b['oracle_lcp']['mean'],0)}-token "
                  f"median-scale prefixes): radix {fnum(r['mean'],3)} ms < "
                  f"hashrope {fnum(h['mean'],2)} ms (flat {fnum(f['mean'],3)} ms); "
                  "reported in full, not the lede")
    status = "PASS" if (win_ok and r1b) else "CONDITIONAL"
    clauses.append({"clause": "iv",
                    "name": "identification-cost regimes (R2 win + R1 negative)",
                    "status": status,
                    "detail": win_txt + " | " + frac_txt + " | " + r1_txt})

    # ---- (v) honest negatives in full -------------------------------------
    regs = []
    for lab, ta in [("R1", r1a)] + [(f"R2 L={L}", r2a_by_L.get(L))
                                    for L in sorted(r2a_by_L)]:
        if not ta:
            continue
        regs.append((lab, ta["n_ttft_regressions"], ta["n_thr_regressions"],
                     ta["n_pairs"]))
    reg_txt = "; ".join(f"{lab}: {t} TTFT / {r} throughput regressions "
                        f"(of {n} pairs)"
                        for lab, t, r, n in regs) if regs else "no Track A pairs"
    flat_note = ""
    if r2b_by_L:
        Lmax = max(r2b_by_L)
        fl = r2b_by_L[Lmax]["arms"].get("flat", {}).get("lat_ms", {})
        if fl.get("mean") is not None:
            flat_note = (f" | numpy-flat is fastest at every L "
                         f"({fnum(fl['mean'],3)} ms at L={fnum(Lmax,0)}): a "
                         "C-vectorized O(L) scan with no persistence/edits/"
                         "branching; the functional comparison is vs radix "
                         "(the production engine matcher); hashrope's edge is "
                         "the unification, not single-query LCP wall-clock")
    detail = ("Disclosed: (a) R1 identification cost -- see clause (iv); "
              "(b) cache-ON regression check from data: " + reg_txt + "; "
              "(c) (ii) neutrality is by-construction, not a measured per-arm "
              "difference; (d) conditional-on-use scope: an upstream identifier "
              "is redundant inside a single engine, valuable in multi-instance "
              "routing / cross-request dedup / opaque-blob contexts"
              + flat_note)
    clauses.append({"clause": "v", "name": "honest negatives in full",
                    "status": "PASS", "detail": detail})

    # ---- (vi) n>=9, paired tests, clocks + env ----------------------------
    n_bad = []
    for lab, ta in [("R1 Track A", r1a)] + [(f"R2 L={L} Track A", r2a_by_L.get(L))
                                            for L in sorted(r2a_by_L)]:
        if ta and ta["n_pairs"] < 9:
            n_bad.append(f"{lab} n={ta['n_pairs']}")
    for lab, tb in [("R1 Track B", r1b)] + [(f"R2 L={L} Track B", r2b_by_L.get(L))
                                            for L in sorted(r2b_by_L)]:
        if tb and tb["n_cells"] < 9:
            n_bad.append(f"{lab} n={tb['n_cells']}")
    ok = not n_bad and env_check["clocks_recorded_cells"] == env_check["track_a_cells"] \
        and env_check["env_blocks_present"]
    detail = (f"all aggregates n>=9: {'YES' if not n_bad else 'NO (' + ', '.join(n_bad) + ')'}; "
              f"paired sign tests computed throughout; GPU clocks recorded in "
              f"{env_check['clocks_recorded_cells']}/{env_check['track_a_cells']} "
              f"non-skipped Track A cells; env blocks (python/torch/vllm/"
              f"hashrope versions, git sha, seeds) present in both grid files: "
              f"{'YES' if env_check['env_blocks_present'] else 'NO'}")
    clauses.append({"clause": "vi", "name": "statistics + environment",
                    "status": "PASS" if ok else "CONDITIONAL",
                    "detail": detail})

    # ---- F6 contingent win -------------------------------------------------
    clauses.append({
        "clause": "F6", "name": "contingent per-arm end-to-end GPU win",
        "status": "NOT TRIGGERED",
        "detail": ("Structural: the one-engine-run-per-cell design produces no "
                   "per-arm end-to-end engine metrics (TTFT/throughput/J-token "
                   "per identifier arm), so the F6 precondition (paired 9/9, "
                   "p<0.05, effect >= 10% on an engine metric) cannot arise. "
                   "S3 stands on (i)-(vi).")})

    # ---- verdict -----------------------------------------------------------
    by = {c["clause"]: c["status"] for c in clauses}
    if any(by[c] == "BLOCKED" for c in ("i", "ii", "iii")):
        verdict = "S3 BLOCKED -- hard clause failed; bug hunt, no retrofit"
    elif all(by[c] == "PASS" for c in ("i", "ii", "iii", "iv", "v", "vi")):
        verdict = ("S3 SUPPORTED (real-stack validation + energy-transparency "
                   "+ caching attribution); F6 not triggered")
    else:
        weak = [c for c in ("i", "ii", "iii", "iv", "v", "vi") if by[c] != "PASS"]
        verdict = f"S3 CONDITIONAL -- clauses not fully met: {', '.join(weak)}"
    return clauses, verdict


# --------------------------------------------------------------------------
# environment / provenance checks
# --------------------------------------------------------------------------

def env_check_of(r1_doc, r2_doc):
    track_a_cells = clocks = 0
    for doc in (r1_doc, r2_doc):
        for c in doc.get("cells", []):
            if _track_a_ok(c):
                track_a_cells += 1
                gm = (c["track_a"].get("gpu_metadata") or [])
                if gm and gm[0].get("sm_clock_mhz") is not None \
                        and gm[0].get("mem_clock_mhz") is not None:
                    clocks += 1
    env_ok = all(
        isinstance(doc.get("env"), dict)
        and doc["env"].get("engine_version")
        and doc.get("git_sha")
        and doc.get("seeds")
        for doc in (r1_doc, r2_doc))
    return {"track_a_cells": track_a_cells,
            "clocks_recorded_cells": clocks,
            "env_blocks_present": env_ok}


# --------------------------------------------------------------------------
# report printing
# --------------------------------------------------------------------------

def hr(ch="-", n=78):
    print(ch * n)


def print_track_a(label, ta):
    if not ta:
        print(f"{label}: no paired Track A data")
        return
    print(f"{label} (n={ta['n_pairs']} paired seed x invocation cells)")
    print(f"  J/token   OFF {fnum(ta['j_per_token_off']['mean'])} +/- "
          f"{fnum(ta['j_per_token_off']['std'])}   ON {fnum(ta['j_per_token_on']['mean'])} "
          f"+/- {fnum(ta['j_per_token_on']['std'])}   "
          f"ratio OFF/ON {fnum(ta['ratio_j']['mean'],2)}x +/- {fnum(ta['ratio_j']['std'],2)} "
          f"(min {fnum(ta['ratio_j']['min'],2)}, max {fnum(ta['ratio_j']['max'],2)})")
    if ta.get("ratio_j_per_seed"):
        per = ", ".join(f"s{k} {fnum(v['mean'],2)}x"
                        for k, v in ta["ratio_j_per_seed"].items())
        print(f"  per-seed ratio means: {per}")
    print(f"  total J   OFF {fnum(ta['total_joules_off']['mean'],0)}   "
          f"ON {fnum(ta['total_joules_on']['mean'],0)}")
    sj = ta["sign_delta_j"]
    print(f"  Delta(ON-OFF) J/token {fnum(ta['delta_j_per_token']['mean'])} +/- "
          f"{fnum(ta['delta_j_per_token']['std'])}   sign ON<OFF {sj['wins']}/{sj['n']} "
          f"p={pfmt(sj['p_one_sided'])}")
    st = ta["sign_delta_ttft"]
    print(f"  TTFT ms   OFF {fnum(ta['ttft_ms_off']['mean'],1)} +/- "
          f"{fnum(ta['ttft_ms_off']['std'],1)}   ON {fnum(ta['ttft_ms_on']['mean'],1)} +/- "
          f"{fnum(ta['ttft_ms_on']['std'],1)}   sign ON<OFF {st['wins']}/{st['n']} "
          f"p={pfmt(st['p_one_sided'])}")
    if not ta["cached_token_counter_available"]:
        print("  cached-token counter: UNAVAILABLE in this engine path "
              "(vLLM offline API returns None/0); Layer-B<->engine reuse tie "
              "carried by the TTFT/energy delta + the correctness gate")
    else:
        print(f"  cached tokens (ON cells): {ta['cached_tokens_on_values']}")


def print_track_b(label, tb):
    if not tb:
        print(f"{label}: no Track B data")
        return
    print(f"{label} (n={tb['n_cells']} cells; oracle prefix mean "
          f"{fnum(tb['oracle_lcp']['mean'],0)} tokens; gate "
          f"{tb['gate_mismatches']}/{tb['gate_checks']} mismatches)")
    for a in sorted(tb["arms"]):
        st = tb["arms"][a]["lat_ms"]
        print(f"  {a:<9} {fnum(st['mean'],3)} +/- {fnum(st['std'],3)} ms")
    if "sign_hashrope_beats_radix" in tb:
        s = tb["sign_hashrope_beats_radix"]
        rat = tb.get("hashrope_over_radix_ratio")
        if rat is not None and rat <= 1:
            who = f"hashrope {fnum(1.0/rat,2)}x faster"
        elif rat is not None:
            who = f"radix {fnum(rat,2)}x faster"
        else:
            who = "n/a"
        print(f"  hashrope/radix ratio {fnum(rat,2)}x ({who}); paired sign "
              f"hashrope-beats-radix {s['wins']}/{s['n']} p={pfmt(s['p_one_sided'])}")


def run_analysis(r1_doc, r2_doc, nc1_doc, nc2_doc, inputs_meta=None,
                 quiet=False):
    g1 = extract_grid(r1_doc)
    g2 = extract_grid(r2_doc)

    r1a = analyze_track_a(g1, None)
    r1b = analyze_track_b(g1, None)
    r2a_by_L, r2b_by_L = {}, {}
    for L in g2["L_values"]:
        if L is None:
            continue
        ta = analyze_track_a(g2, L)
        if ta:
            r2a_by_L[L] = ta
        tb = analyze_track_b(g2, L)
        if tb:
            r2b_by_L[L] = tb

    nc1 = analyze_negctl(nc1_doc)
    nc2 = analyze_negctl(nc2_doc)

    gate_total = {
        "checks": g1["gate"]["checks"] + g2["gate"]["checks"],
        "mismatches": g1["gate"]["mismatches"] + g2["gate"]["mismatches"],
        "r1_checks": g1["gate"]["checks"], "r1_mismatches": g1["gate"]["mismatches"],
        "r2_checks": g2["gate"]["checks"], "r2_mismatches": g2["gate"]["mismatches"],
    }
    envc = env_check_of(r1_doc, r2_doc)
    clauses, verdict = evaluate_criterion(
        r1a, r1b, r2a_by_L, r2b_by_L, g2["a_skips"], nc1, nc2, gate_total, envc)

    if not quiet:
        hr("=")
        print("EXP-015 ANALYSIS (claim S3)")
        print(f"generated {_dt.datetime.now(_dt.timezone.utc).isoformat()}")
        print("conventions: std = sample (ddof=1) across (seed,invocation) "
              "cells; sign test = exact one-sided binomial, ties dropped; "
              "95% band = t-based; J/token denominator = completion tokens "
              "(absolute values prefill-dominated; ratio/delta meaningful)")
        for lab, doc in (("R1", r1_doc), ("R2", r2_doc)):
            print(f"{lab}: model={doc['provenance'].get('model')}  "
                  f"engine={doc['env'].get('engine')} "
                  f"{doc['env'].get('engine_version')}  git={doc.get('git_sha','')[:7]} "
                  f"dirty={doc.get('git_dirty')}  seeds={doc.get('seeds')} x "
                  f"inv={doc.get('invocations')}  cells="
                  f"{len(doc.get('completed_cells', doc.get('cells', [])))}/"
                  f"{doc.get('n_cells')}")
        hr("=")
        print("\n--- TRACK A: engine caching energy (the ENGINE's effect) ---")
        print_track_a("R1 (ShareGPT/LMSYS realistic streams)", r1a)
        for L in sorted(r2a_by_L):
            print()
            print_track_a(f"R2 L={L:,}", r2a_by_L[L])
        skips = g2["a_skips"]
        if skips:
            Ls = sorted({s['L'] for s in skips})
            print(f"\nR2 L in {Ls}: Track A SKIPPED in {len(skips)} cells -- "
                  f"{skips[0]['reason']}")
            print(f"  detail: {skips[0]['detail']}")
        print("\n--- TRACK B: identification (CPU, engine-independent) ---")
        print_track_b("R1", r1b)
        for L in sorted(r2b_by_L):
            print()
            print_track_b(f"R2 L={L:,}", r2b_by_L[L])
        print("\n--- NEGATIVE CONTROLS (corrupted cached entry) ---")
        print(f"R1: {nc1['detected']}/{nc1['total']} detected; "
              f"R2: {nc2['detected']}/{nc2['total']} detected; "
              f"TOTAL {nc1['detected']+nc2['detected']}/{nc1['total']+nc2['total']}")
        print("\n--- PROMOTION CRITERION (verbatim clauses, settled "
              "operationalizations) ---")
        for c in clauses:
            hr()
            print(f"({c['clause']}) {c['name']}: {c['status']}")
            print(f"    {c['detail']}")
        hr("=")
        print(f"VERDICT: {verdict}")
        hr("=")

    summary = {
        "experiment": "EXP-015",
        "claim": "S3",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "inputs": inputs_meta or {},
        "conventions": {
            "std": "sample std (ddof=1) across (seed, invocation) cells",
            "sign_test": "exact one-sided binomial, ties dropped",
            "band95": "t-based, df = n-1",
            "j_per_token": "total_joules / total completion tokens "
                           "(prefill-dominated; ratio/delta meaningful)",
        },
        "provenance": {
            "r1": {"model": r1_doc["provenance"].get("model"),
                   "git_sha": r1_doc.get("git_sha"),
                   "git_dirty": r1_doc.get("git_dirty"),
                   "engine_version": r1_doc["env"].get("engine_version"),
                   "seeds": r1_doc.get("seeds"),
                   "invocations": r1_doc.get("invocations"),
                   "corpus_sha16": r1_doc["provenance"].get("corpus_sha16")},
            "r2": {"model": r2_doc["provenance"].get("model"),
                   "git_sha": r2_doc.get("git_sha"),
                   "git_dirty": r2_doc.get("git_dirty"),
                   "engine_version": r2_doc["env"].get("engine_version"),
                   "seeds": r2_doc.get("seeds"),
                   "invocations": r2_doc.get("invocations"),
                   "corpus_sha16": r2_doc["provenance"].get("corpus_sha16")},
        },
        "gate_total": gate_total,
        "r1": {"track_a": r1a, "track_b": r1b},
        "r2": {"track_a_by_L": {str(k): v for k, v in r2a_by_L.items()},
               "track_b_by_L": {str(k): v for k, v in r2b_by_L.items()},
               "track_a_skips": g2["a_skips"]},
        "negative_controls": {"r1": nc1, "r2": nc2},
        "env_check": envc,
        "criterion": clauses,
        "verdict": verdict,
    }
    return summary


# --------------------------------------------------------------------------
# self-test: synthetic grids in the exact observed schemas
# --------------------------------------------------------------------------

def _mkstats(vals):
    a = agg(vals)
    return {"mean": a["mean"], "std": a["std"] or 0.0, "min": a["min"],
            "max": a["max"], "n": a["n"], "values": vals}


def _mk_track_a(j_total, n_req, comp_tok, ttft_mean, cached=0, seed=0):
    ttfts = [ttft_mean * (1 + 0.02 * ((i * 7 + seed) % 5 - 2)) for i in range(n_req)]
    wall = max(ttfts) / 1000.0 + 1.0
    return {
        "skipped": False,
        "energy_path": "counter",
        "gpu_metadata": [{"index": 0, "driver": "575.51.03",
                          "name": "SYNTHETIC-GPU", "sm_clock_mhz": 1275,
                          "mem_clock_mhz": 1215, "power_limit_w": 400.0,
                          "energy_counter_supported": True}],
        "per_gpu_joules": {"0": j_total},
        "total_joules": j_total,
        "window_wall_s": wall,
        "n_requests": n_req,
        "total_completion_tokens": comp_tok,
        "total_cached_tokens": cached,
        "j_per_token": j_total / comp_tok,
        "j_per_request": j_total / n_req,
        "throughput_tok_s": comp_tok / wall,
        "throughput_req_s": n_req / wall,
        "ttft_ms": _mkstats(ttfts),
        "cached_tokens_per_req": [None] * n_req,
        "n_power_samples": {},
        "engine_kwargs": {"model": "synthetic", "enable_prefix_caching": True},
        "engine": "vllm", "engine_version": "0.8.5.post1",
    }


def _mk_track_b(n_q, lcp, hr_ms, rx_ms, fl_ms, jit=0.0):
    def vals(base):
        return [base * (1 + jit * ((i % 3) - 1)) for i in range(n_q)]
    arms = {}
    for name, base in (("hashrope", hr_ms), ("radix", rx_ms), ("flat", fl_ms)):
        arms[name] = {"lat_ms": _mkstats(vals(base)),
                      "matched_lcp": _mkstats([float(lcp)] * n_q)}
    return {"radix_available": True,
            "gate": {"ok": True, "n_checks": n_q, "mismatches": 0},
            "oracle_lcp": _mkstats([float(lcp)] * n_q),
            "arms": arms}


def build_synthetic():
    seeds, invs = [42, 43, 44], [0, 1, 2]
    hdr = {
        "experiment": "EXP-015", "claim": "S3",
        "env": {"platform": "synthetic", "python": "3.x",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "hashrope_version": "0.2.2", "numpy_version": "x",
                "torch_version": "x", "cuda_available": False,
                "engine": "vllm", "engine_version": "0.8.5.post1"},
        "git_sha": "f" * 40, "git_dirty": False,
        "seeds": seeds, "invocations": 3, "gpus": [0],
    }
    # R1: OFF ~ 22x ON; track_b in ON cells only; radix wins
    r1 = dict(hdr, regime="R1", description="synthetic R1",
              provenance={"model": "synthetic", "corpus_sha16": {},
                          "r1_datasets": ["a", "b"], "r2_L": None},
              n_cells=18, completed_cells=[], cells=[])
    for s in seeds:
        for i in invs:
            base_on = 0.10 * (1 + 0.03 * (s - 43) + 0.01 * i)
            base_off = base_on * (22.0 + 0.5 * (s - 43) + 0.2 * i)
            comp = 1280
            for cache_on, jt in ((False, base_off), (True, base_on)):
                cell = {"cell_id": f"R1_{'on' if cache_on else 'off'}_s{s}_i{i}",
                        "regime": "R1", "cache_on": cache_on, "seed": s,
                        "inv": i, "L": None, "nvml_indices": [0], "tp": 1,
                        "n_items": 160,
                        "dataset": "a,b",
                        "track_a": _mk_track_a(jt * comp, 160, comp,
                                               60.0 if not cache_on else 20.0,
                                               seed=s),
                        "track_b": _mk_track_b(160, 640, 17.0 + 0.1 * i,
                                               0.04, 0.005, jit=0.05)
                        if cache_on else None}
                r1["cells"].append(cell)
                r1["completed_cells"].append(cell["cell_id"])
    # R2: two L; low L Track A present + radix wins B; high L Track A skipped
    # + hashrope wins B 9/9
    r2 = dict(hdr, regime="R2", description="synthetic R2",
              provenance={"model": "synthetic",
                          "corpus_sha16": {"42": "x", "43": "y", "44": "z"},
                          "r1_datasets": None, "r2_L": [4096, 8192]},
              n_cells=36, completed_cells=[], cells=[])
    for L, skipped in ((4096, False), (8192, True)):
        for s in seeds:
            for i in invs:
                on_j = 0.5 * (1 + 0.02 * (s - 43))
                off_j = on_j * (70.0 + s - 43 + 0.3 * i)
                comp = 64
                for cache_on in (False, True):
                    if skipped:
                        ta = {"skipped": True,
                              "reason": "engine_serve_infeasible_dense_attention",
                              "detail": "synthetic skip", "track_a_max_L": 4096,
                              "L": L, "engine": "vllm",
                              "engine_version": "0.8.5.post1",
                              "energy_path": "counter"}
                    else:
                        ta = _mk_track_a((off_j if not cache_on else on_j) * comp,
                                         8, comp,
                                         7800.0 if not cache_on else 220.0,
                                         seed=s)
                    tb = None
                    if cache_on:
                        if L == 4096:
                            tb = _mk_track_b(8, L, 46.0 + 0.1 * i, 10.0, 0.05,
                                             jit=0.03)
                        else:
                            tb = _mk_track_b(8, L, 44.0 + 0.1 * i, 53.0, 0.6,
                                             jit=0.03)
                    cell = {"cell_id": f"R2_{'on' if cache_on else 'off'}_s{s}_i{i}_L{L}",
                            "regime": "R2", "cache_on": cache_on, "seed": s,
                            "inv": i, "L": L, "nvml_indices": [0, 1, 2, 3],
                            "tp": 4, "n_items": 8, "dataset": "synthetic",
                            "track_a": ta, "track_b": tb}
                    r2["cells"].append(cell)
                    r2["completed_cells"].append(cell["cell_id"])
    # negative controls in both observed schemas
    nc1 = {"experiment": "EXP-015", "claim": "S3", "regime": "R1", "seed": 42,
           "model": "synthetic", "purpose": "negctl",
           "rows": [{"dataset": "a", "sha16": "s", "pairs_tested": 15,
                     "controls": 45, "all_detected": True, "sample": []},
                    {"dataset": "b", "sha16": "t", "pairs_tested": 15,
                     "controls": 45, "all_detected": True, "sample": []}],
           "all_detected": True, "env": {},
           "grid_gate_context": {"grid_gate_mismatches": 0,
                                 "grid_gate_checks": 1584,
                                 "interpretation": "synthetic"}}
    nc2 = {"experiment": "EXP-015", "claim": "S3", "purpose": "negctl",
           "model": "synthetic", "corpus_sha16": {}, "regime": "R2",
           "seed": 42, "L_sweep": [4096, 8192],
           "rows": [{"L": L, "seed": 42, "gate_ok": True, "gate_checks": 8,
                     "gate_mismatches": 0,
                     "arms_present": ["oracle", "hashrope", "flat", "radix"],
                     "controls": [
                         {"where": w, "pos": p, "true_lcp": L,
                          "arms_checked": ["hashrope", "flat", "radix"],
                          "results": {"oracle": p, "hashrope": p, "flat": p,
                                      "radix": p},
                          "detected": True}
                         for w, p in (("first", 0), ("mid", L // 2),
                                      ("last", L - 1))]}
                    for L in (4096, 8192)],
           "all_detected": True, "all_gates_ok": True, "env": {},
           "grid_gate_context": {"grid_gate_mismatches": 0,
                                 "grid_gate_checks": 1584,
                                 "interpretation": "synthetic"}}
    return r1, r2, nc1, nc2


def selftest():
    print("SELFTEST: building synthetic grids in the observed schemas ...")
    r1, r2, nc1, nc2 = build_synthetic()
    s = run_analysis(r1, r2, nc1, nc2, quiet=True)

    def check(name, cond):
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            raise AssertionError(name)

    by = {c["clause"]: c["status"] for c in s["criterion"]}
    # gate: R1 9 ON cells x 160 + R2 4096 9 x 8 + R2 8192 9 x 8 = 1584? no:
    # R1 1440 + R2 (9+9)*8 = 144 -> 1584. Recompute independently:
    expect_checks = 9 * 160 + 2 * 9 * 8
    check(f"gate checks == {expect_checks}",
          s["gate_total"]["checks"] == expect_checks)
    check("gate mismatches == 0", s["gate_total"]["mismatches"] == 0)
    check("(i) PASS", by["i"] == "PASS")
    check("(ii) PASS", by["ii"] == "PASS")
    check("(iii) PASS", by["iii"] == "PASS")
    check("(iv) PASS", by["iv"] == "PASS")
    check("(v) PASS", by["v"] == "PASS")
    check("(vi) PASS", by["vi"] == "PASS")
    check("F6 NOT TRIGGERED", by["F6"] == "NOT TRIGGERED")
    check("verdict SUPPORTED", s["verdict"].startswith("S3 SUPPORTED"))
    # crossover bracket appears in (iv) detail
    d4 = next(c for c in s["criterion"] if c["clause"] == "iv")["detail"]
    check("crossover bracket (4,096, 8,192] reported",
          "(4,096, 8,192]" in d4)
    tb_hi = s["r2"]["track_b_by_L"]["8192"]
    st = tb_hi["sign_hashrope_beats_radix"]
    check("1M-analog sign test 9/9", st["wins"] == 9 and st["n"] == 9)
    check("1M-analog p == 0.5^9",
          abs(st["p_one_sided"] - 0.5 ** 9) < 1e-12)
    tb_lo = s["r2"]["track_b_by_L"]["4096"]
    st_lo = tb_lo["sign_hashrope_beats_radix"]
    check("low-L radix wins 9/9 (hashrope wins 0/9)",
          st_lo["wins"] == 0 and st_lo["n"] == 9)
    r1ta = s["r1"]["track_a"]
    check("R1 ratio ~22x", abs(r1ta["ratio_j"]["mean"] - 22.0) < 2.0)
    check("R1 Delta sign 9/9 ON<OFF",
          r1ta["sign_delta_j"]["wins"] == 9)
    n1 = s["negative_controls"]["r1"]
    n2 = s["negative_controls"]["r2"]
    check("negctl R1 90/90", n1["total"] == 90 and n1["detected"] == 90)
    check("negctl R2 6/6", n2["total"] == 6 and n2["detected"] == 6)
    check("cached-token counter reported UNAVAILABLE",
          r1ta["cached_token_counter_available"] is False)
    check("R2 skip recorded with reason",
          s["r2"]["track_a_skips"]
          and s["r2"]["track_a_skips"][0]["reason"]
          == "engine_serve_infeasible_dense_attention")
    check("env check clocks complete",
          s["env_check"]["clocks_recorded_cells"]
          == s["env_check"]["track_a_cells"])

    # poisoned run: inject one gate mismatch -> (i) BLOCKED, verdict BLOCKED
    r1p = copy.deepcopy(r1)
    for c in r1p["cells"]:
        if c.get("track_b"):
            c["track_b"]["gate"]["mismatches"] = 1
            c["track_b"]["gate"]["ok"] = False
            break
    sp = run_analysis(r1p, r2, nc1, nc2, quiet=True)
    byp = {c["clause"]: c["status"] for c in sp["criterion"]}
    check("poisoned: (i) BLOCKED", byp["i"] == "BLOCKED")
    check("poisoned: verdict BLOCKED", sp["verdict"].startswith("S3 BLOCKED"))

    # poisoned negctl: one undetected control -> (i) BLOCKED
    nc2p = copy.deepcopy(nc2)
    nc2p["rows"][0]["controls"][0]["detected"] = False
    nc2p["all_detected"] = False
    sp2 = run_analysis(r1, r2, nc1, nc2p, quiet=True)
    byp2 = {c["clause"]: c["status"] for c in sp2["criterion"]}
    check("poisoned negctl: (i) BLOCKED", byp2["i"] == "BLOCKED")

    print("SELFTEST PASS")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="EXP-015 analysis (claim S3)")
    d = "experiments/exp_015_energy/results"
    ap.add_argument("--r1", default=f"{d}/exp015_R1_raw_latest.json")
    ap.add_argument("--r2", default=f"{d}/exp015_R2_raw_latest.json")
    ap.add_argument("--negctl-r1", default=f"{d}/exp015_negative_control_R1.json")
    ap.add_argument("--negctl-r2", default=f"{d}/exp015_negative_control_R2.json")
    ap.add_argument("--out-dir", default=d)
    ap.add_argument("--no-write", action="store_true",
                    help="print only; do not write the summary JSON")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic-grid self-test and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0

    inputs_meta = {}
    for name, p in (("r1", args.r1), ("r2", args.r2),
                    ("negctl_r1", args.negctl_r1),
                    ("negctl_r2", args.negctl_r2)):
        inputs_meta[name] = {"path": str(p), "sha256": sha256_of(Path(p))}

    r1_doc = load_json(args.r1)
    r2_doc = load_json(args.r2)
    nc1_doc = load_json(args.negctl_r1)
    nc2_doc = load_json(args.negctl_r2)

    summary = run_analysis(r1_doc, r2_doc, nc1_doc, nc2_doc,
                           inputs_meta=inputs_meta)

    if not args.no_write:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        latest = out / "exp015_analysis_summary_latest.json"
        stamped = out / f"exp015_analysis_summary_{ts}Z.json"
        blob = json.dumps(summary, indent=1, sort_keys=False)
        latest.write_text(blob + "\n", encoding="utf-8", newline="\n")
        stamped.write_text(blob + "\n", encoding="utf-8", newline="\n")
        print(f"\nwrote {latest}")
        print(f"wrote {stamped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
