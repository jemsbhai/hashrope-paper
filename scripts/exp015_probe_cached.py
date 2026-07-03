#!/usr/bin/env python3
"""EXP-015 diagnostic: locate where vLLM reports prefix-cache hits on THIS build.

The serving adapter reads RequestOutput.num_cached_tokens, which returns 0 on the
vLLM 0.8.5.post1 offline path even when prefix caching is demonstrably active
(the R1 Instruct run showed a ~20x cache-ON energy drop with num_cached_tokens==0
on every cell). This probe serves the SAME prompt twice with
enable_prefix_caching=True -- the second serve is a guaranteed full cache hit --
and dumps every cache/token/metric field of the RequestOutput, its metrics, and
the CompletionOutput, so we can pin the correct attribute and patch
parse_request_metrics in src/exp015_serving.py.

The cached-token field name is a vLLM-VERSION property, not model-specific, so
this uses a tiny model and runs in ~1 minute on one GPU.

Run (folded into the smoke, or standalone):
    python scripts/exp015_probe_cached.py [MODEL]
"""
import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
KEYS = ("cach", "prefix", "num", "token", "metric", "block", "hit")


def _interesting(name: str) -> bool:
    low = name.lower()
    return (not name.startswith("__")) and any(k in low for k in KEYS)


def dump(tag, obj, indent="  "):
    print(f"{indent}{tag}: type={type(obj).__name__}")
    seen = set()
    # instance dict first (explicit attrs), then dir() (adds properties)
    src = list(getattr(obj, "__dict__", {}).keys()) + dir(obj)
    for a in src:
        if a in seen or not _interesting(a):
            continue
        seen.add(a)
        try:
            v = getattr(obj, a)
        except Exception as e:
            print(f"{indent}  .{a} = <err {e}>")
            continue
        if callable(v):
            continue
        if isinstance(v, (list, tuple)) and len(v) > 8:
            v = f"{type(v).__name__}(len={len(v)}) head={list(v)[:8]}"
        print(f"{indent}  .{a} = {v}")


def main():
    import vllm
    from vllm import LLM, SamplingParams
    print(f"[probe] vllm {getattr(vllm, '__version__', '?')}  model={MODEL}")

    llm = LLM(model=MODEL, enable_prefix_caching=True, dtype="float16",
              gpu_memory_utilization=0.5, max_model_len=4096,
              enforce_eager=True, max_num_seqs=1)

    # 256 ids -> spans many KV blocks (block size <= 32), so a hit is unambiguous
    ids = list(range(10, 10 + 256))
    sp = SamplingParams(max_tokens=4, temperature=0)

    for label in ("SERVE 1 (cache MISS expected)", "SERVE 2 (cache HIT expected)"):
        print("\n" + "=" * 72 + f"\n{label}\n" + "=" * 72)
        outs = llm.generate({"prompt_token_ids": list(ids)}, sp)
        o = outs[0]
        dump("RequestOutput", o)
        m = getattr(o, "metrics", None)
        if m is not None:
            dump("RequestOutput.metrics", m, indent="    ")
        co = getattr(o, "outputs", None)
        if co:
            dump("CompletionOutput[0]", co[0], indent="    ")
        print(f"  [adapter currently reads] num_cached_tokens = "
              f"{getattr(o, 'num_cached_tokens', '<absent>')}")

    print("\n[probe] done. Report the SERVE 2 dump: the nonzero cache field "
          "(expect ~256) is the fix target for parse_request_metrics.")


if __name__ == "__main__":
    main()
