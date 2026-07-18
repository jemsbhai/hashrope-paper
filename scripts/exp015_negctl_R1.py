import sys, json, hashlib
sys.path.insert(0,'/home/claude/negctl')
from src.exp015_identify import build_r1_stream, negative_control, correctness_gate

MODEL="Qwen/Qwen2.5-7B-Instruct-1M"
rows=[]
for ds in ["sharegpt_sample.jsonl","lmsys_sample.jsonl"]:
    p=f"/mnt/user-data/uploads/{ds}"
    sha=hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
    s=build_r1_stream(p, n_pairs=80, seed=42, tokenizer_name=MODEL)
    items=s["items"]
    # gate over the whole real stream
    tested=0; detected_all=True; ctl=[]
    for it in items:
        cached,q = it["cached_tokens"], it["query_tokens"]
        true_lcp = it["oracle_lcp"]
        if true_lcp < 2:   # need >=1 corruptible position strictly before the LCP
            continue
        for label,pos in [("first",0),("mid",true_lcp//2),("last",true_lcp-1)]:
            nc=negative_control(cached,q,corrupt_pos=pos,with_radix=True)
            ok=nc["detected"]; detected_all &= ok
            ctl.append({"where":label,"pos":nc["pos"],"true_lcp":nc["true_lcp"],
                        "results":nc["results"],"detected":ok,"arms":nc["arms_checked"]})
        tested+=1
        if tested>=15: break   # 15 real pairs x 3 positions = 45 controls per dataset
    rows.append({"dataset":ds,"sha16":sha,"pairs_tested":tested,
                 "controls":len(ctl),"all_detected":detected_all,
                 "sample":ctl[:3]})
    print(f"{ds}: sha16={sha} pairs_tested={tested} controls={len(ctl)} all_detected={detected_all}", flush=True)

out={"experiment":"EXP-015","claim":"S3","regime":"R1","seed":42,"model":MODEL,
     "purpose":"criterion (i) non-vacuity: corrupted-cached-entry negative control (real ShareGPT/LMSYS streams)",
     "rows":rows,"all_detected":all(r["all_detected"] for r in rows)}
json.dump(out,open('/home/claude/negctl/exp015_negative_control_R1.json','w'),indent=2)
print("\nR1 ALL CONTROLS DETECTED:", out["all_detected"])
