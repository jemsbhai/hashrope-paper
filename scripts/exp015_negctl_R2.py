#!/usr/bin/env python3
"""EXP-015 criterion (i) negative control -- non-vacuity of the correctness gate.

Corrupts ONE cached token before the true LCP; every arm (hashrope, flat, radix)
must report the SHORTENED match (== corrupt_pos), proving the arms genuinely
compare content rather than blindly returning the query length.

Real workloads: the byte-identical seed-42 corpus the R2 grid consumed
(sha256[:16]=fda6a43a3fe34184, verified against the grid provenance) and the
same Qwen2.5-7B-Instruct-1M tokenizer.
"""
import sys, json, time, hashlib
sys.path.insert(0, '/home/claude/negctl')
from src.exp015_identify import (build_r2_stream, negative_control,
                                 correctness_gate, radix_available)

MODEL = "Qwen/Qwen2.5-7B-Instruct-1M"
LS = [131072, 262144, 393216, 524288, 1000000]
SEED = 42

print("radix_available:", radix_available(), flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
text = open('/mnt/user-data/uploads/corpus_s42.txt', encoding='utf-8').read()
sha = hashlib.sha256(open('/mnt/user-data/uploads/corpus_s42.txt','rb').read()).hexdigest()[:16]
t0 = time.time()
toks = tok(text, add_special_tokens=False)["input_ids"]
print(f"corpus sha16={sha} tokens={len(toks):,} (tokenized in {time.time()-t0:.0f}s)", flush=True)

results = []
for L in LS:
    stream = build_r2_stream(toks, L=L, n_queries=8, seed=SEED)
    cached, queries = stream["cached_tokens"], stream["queries"]
    q = queries[0]
    # sanity: uncorrupted gate must pass (0 mismatches) on this stream
    g = correctness_gate(cached, queries, with_radix=True)
    row = {"L": L, "seed": SEED, "gate_ok": g.ok, "gate_checks": g.n_checks,
           "gate_mismatches": g.mismatches, "arms_present": g.arms_present,
           "controls": []}
    for label, pos in [("first", 0), ("mid", L // 2), ("last", L - 1)]:
        nc = negative_control(cached, q, corrupt_pos=pos, with_radix=True)
        ok = nc["detected"]
        row["controls"].append({"where": label, "pos": nc["pos"],
                                "true_lcp": nc["true_lcp"],
                                "arms_checked": nc["arms_checked"],
                                "results": nc["results"], "detected": ok})
        print(f"  L={L:>9} corrupt@{label:<5} pos={nc['pos']:>9} true_lcp={nc['true_lcp']:>9} "
              f"-> {dict((k,v) for k,v in nc['results'].items())} detected={ok}", flush=True)
    results.append(row)

out = {"experiment": "EXP-015", "claim": "S3",
       "purpose": "criterion (i) non-vacuity: corrupted-cached-entry negative control",
       "model": MODEL, "corpus_sha16": {str(SEED): sha},
       "regime": "R2", "seed": SEED, "L_sweep": LS, "rows": results,
       "all_detected": all(c["detected"] for r in results for c in r["controls"]),
       "all_gates_ok": all(r["gate_ok"] for r in results)}
json.dump(out, open('/home/claude/negctl/exp015_negative_control_R2.json','w'), indent=2)
print("\nALL CONTROLS DETECTED:", out["all_detected"])
print("ALL UNCORRUPTED GATES OK:", out["all_gates_ok"])
