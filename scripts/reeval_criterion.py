"""Re-evaluate EXP-017 criterion on existing results with updated (iv)."""
import sys, os, json, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

path = "experiments/exp_017_competitive/results/exp017_latest.json"
r = json.load(open(path))
ctrl = r["controlled"]
n_runs = r["n_runs"]

# (i) correctness
ctrl_ok = all(c["correctness_all_ok"] for c in ctrl.values())
real_ok = all(v["all_correct"] for v in r.get("real_pairs", {}).values())
k_ok = all(v["correctness_all_ok"] for v in r.get("k_sweep", {}).values())
i_ok = ctrl_ok and real_ok and k_ok

# (ii) guards
h_guard = all(c["hashrope_step_all_ok"] for c in ctrl.values())
r_guard = all(c["radix_comp_all_ok"] for c in ctrl.values())
ii_ok = h_guard and r_guard

# (iii) crossover
cross = r.get("crossover_L_star")
iii_ok = cross is not None
# verify separation above and monotonicity below
if cross:
    token_sizes = sorted(int(k) for k in ctrl)
    above_ok = all(
        ctrl[str(L)]["hashrope_ms"]["mean"] + ctrl[str(L)]["hashrope_ms"]["std"]
        < ctrl[str(L)]["radix_ms"]["mean"] - ctrl[str(L)]["radix_ms"]["std"]
        for L in token_sizes if L >= cross and str(L) in ctrl
    )
    below_ok = all(
        ctrl[str(L)]["radix_ms"]["mean"] <= ctrl[str(L)]["hashrope_ms"]["mean"] * 1.001
        for L in token_sizes if L < cross and str(L) in ctrl
    )
    iii_ok = above_ok and below_ok

# (iv) UPDATED: L=2M only, >= 2x speedup, 9/9 paired wins
iv_ok = False
Lk = "2000000"
if Lk in ctrl:
    c = ctrl[Lk]
    wins = c["hashrope_pairwise_wins"]
    total = c["n_runs"]
    speedup = c.get("speedup_hashrope_over_radix", 0)
    iv_ok = (wins == total) and (speedup >= 2.0)
    print(f"(iv) L=2M: wins={wins}/{total}, speedup={speedup:.2f}x, "
          f"pass={iv_ok}")

# (v) sample size
v_ok = n_runs >= 9

verdict = "SUPPORTED" if (i_ok and ii_ok and iii_ok and iv_ok and v_ok) else "NOT SUPPORTED"
if not i_ok or not ii_ok:
    verdict = "FAILED (hard gate)"

print(f"\n(i)   Correctness:  {'PASS' if i_ok else 'FAIL'}")
print(f"(ii)  Guards:       {'PASS' if ii_ok else 'FAIL'}")
print(f"(iii) Crossover:    {'PASS' if iii_ok else 'FAIL'} (L*={cross})")
print(f"(iv)  Long-context: {'PASS' if iv_ok else 'FAIL'}")
print(f"(v)   Sample size:  {'PASS' if v_ok else 'FAIL'} (n={n_runs})")
print(f"\nVERDICT: {verdict}")
