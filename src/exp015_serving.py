#!/usr/bin/env python3
"""
EXP-015: vLLM serving adapter (claim S3, Layer A -- the engine).

Engine backend for the S3 prefix-caching energy experiment. Switched from SGLang
to vLLM on 2026-06-21 after SGLang's prebuilt sgl-kernel wheels proved ABI-
incompatible with the cluster's torch 2.6.0+cu124 (undefined-symbol / missing
qserve op). The original SGLang adapter is preserved verbatim in
src/exp015_serving_sglang.py; to revert, copy it back over this file.

Uses the vLLM OFFLINE `LLM` class (in-process; no HTTP server, no port -- the
NVML energy window wraps the generate calls directly with no network confound).
This matches the previous SGLang offline-Engine execution model exactly.

The engine is the Layer-A {cache OFF, cache ON} mechanism:
  cache ON  : enable_prefix_caching=True  (automatic prefix/KV reuse).
  cache OFF : enable_prefix_caching=False.

Requests are sent as token ids (vLLM TokensPrompt: {"prompt_token_ids": ids}), so
the served sequence is byte-identical to what the Layer-B identifier arms see --
no re-tokenization drift. vLLM's RequestOutput reports `num_cached_tokens` (prompt
tokens served from the prefix cache), which:
  - directly confirms cache OFF (num_cached_tokens == 0) vs ON (> 0 on shared
    prefixes), and
  - cross-checks Layer B: the identifier's matched-prefix length should equal the
    engine's reported cached tokens under cache ON (a real-stack tie between the
    two layers, on top of criterion (i)).

GPU-free degradation: with no vLLM/CUDA (e.g. a CPU sandbox) engine_available()
is False; the driver then runs the Layer-B identification track only (dry-run),
and the engine track is cluster-gated. The pure helpers (engine-kwarg build,
request-metric parsing) are unit-tested here and are vLLM-independent.

Public names engine_available / engine_version / ServingArm are the stable
interface; sglang_available / sglang_version / SglServingArm remain as aliases so
the bench (scripts/exp015_bench.py) imports unchanged across the engine switch.

NOTE (cluster-smoke-gated): LLM(**kwargs) and llm.generate({"prompt_token_ids":
...}, SamplingParams(...)) follow the documented vLLM offline API. If the
installed vLLM version differs, the collaborator's smoke run is where any
signature drift surfaces; the vLLM calls are isolated in VllmServingArm so only
that class needs adjusting. Pin a vLLM built for torch 2.6/cu124 (the 0.8.x line)
so `pip install vllm` does not drag torch back to a CUDA-13 build.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Availability + version (vLLM-independent guards)
# ---------------------------------------------------------------------------

def engine_available() -> bool:
    """True iff vLLM imports AND a CUDA device is visible (engine needs a GPU)."""
    try:
        import vllm  # noqa: F401
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def engine_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unavailable"


def engine_name() -> str:
    return "vllm"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no vLLM needed)
# ---------------------------------------------------------------------------

def build_engine_kwargs(model_path: str, cache_on: bool, tp_size: int = 1,
                        mem_fraction_static: float = 0.85,
                        context_length: int | None = None,
                        dtype: str = "float16", random_seed: int = 0,
                        extra: dict | None = None) -> dict:
    """Construct vllm.LLM(**kwargs). cache OFF <=> enable_prefix_caching=False.

    Note the signature mirrors the SGLang adapter (mem_fraction_static, tp_size,
    context_length, random_seed) so the bench passes the same arguments; they map
    onto vLLM's gpu_memory_utilization / tensor_parallel_size / max_model_len /
    seed respectively.
    """
    kw: dict[str, Any] = {
        "model": model_path,
        "enable_prefix_caching": bool(cache_on),
        "tensor_parallel_size": tp_size,
        "gpu_memory_utilization": mem_fraction_static,
        "dtype": dtype,
        "seed": random_seed,
    }
    if context_length is not None:
        kw["max_model_len"] = context_length
    if extra:
        kw.update(extra)
    return kw


def parse_request_metrics(output: Any) -> dict:
    """Extract the fields we record from one vLLM RequestOutput.

    Duck-typed and tolerant of missing attributes (returns None) so the schema is
    stable across vLLM versions. Reads: prompt_token_ids (len -> prompt_tokens),
    outputs[0].token_ids (len -> completion_tokens), outputs[0].finish_reason,
    num_cached_tokens (-> cached_tokens), and metrics.{arrival_time,finished_time}
    (-> e2e_latency).
    """
    keys = ("prompt_tokens", "completion_tokens", "cached_tokens",
            "e2e_latency", "finish_reason")
    if output is None:
        return {k: None for k in keys}

    prompt_ids = getattr(output, "prompt_token_ids", None)
    prompt_tokens = len(prompt_ids) if prompt_ids is not None else None

    outs = getattr(output, "outputs", None)
    first = outs[0] if outs else None
    comp_ids = getattr(first, "token_ids", None) if first is not None else None
    completion_tokens = len(comp_ids) if comp_ids is not None else None
    finish_reason = getattr(first, "finish_reason", None) if first is not None else None

    cached_tokens = getattr(output, "num_cached_tokens", None)

    e2e_latency = None
    m = getattr(output, "metrics", None)
    if m is not None:
        a = getattr(m, "arrival_time", None)
        f = getattr(m, "finished_time", None)
        if a is not None and f is not None:
            e2e_latency = f - a

    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens, "e2e_latency": e2e_latency,
            "finish_reason": finish_reason}


def _as_list(out: Any) -> list:
    """vLLM generate() returns a list[RequestOutput]; normalise defensively."""
    if isinstance(out, list):
        return out
    if out is None:
        return []
    return [out]


# ---------------------------------------------------------------------------
# Serving arm (offline LLM wrapper) -- construction requires vLLM + GPU
# ---------------------------------------------------------------------------

@dataclass
class BatchTiming:
    wall_s: float
    n: int
    total_completion_tokens: int
    total_prompt_tokens: int
    total_cached_tokens: int
    records: list[dict] = field(default_factory=list)


class VllmServingArm:
    """Offline vLLM LLM for one cache setting. Construct on the cluster only.

    Method surface mirrors the SGLang adapter exactly (_generate / ttft /
    generate_batch / warmup / shutdown) so the bench is engine-agnostic.

    Usage:
        arm = VllmServingArm("Qwen/Qwen2.5-7B-Instruct", cache_on=True, tp_size=1)
        arm.warmup(sample_input_ids)
        rec = arm.ttft(input_ids)                 # single-request, max_new_tokens=1
        bt  = arm.generate_batch(list_input_ids)  # throughput / energy window body
        arm.shutdown()
    """

    def __init__(self, model_path: str, cache_on: bool, tp_size: int = 1,
                 mem_fraction_static: float = 0.85, context_length: int | None = None,
                 dtype: str = "float16", random_seed: int = 0, extra: dict | None = None):
        from vllm import LLM  # cluster-only import
        self.cache_on = cache_on
        self.kwargs = build_engine_kwargs(
            model_path, cache_on, tp_size, mem_fraction_static,
            context_length, dtype, random_seed, extra)
        self.llm = LLM(**self.kwargs)

    def _sampling(self, max_new_tokens: int, temperature: float):
        from vllm import SamplingParams
        return SamplingParams(max_tokens=max_new_tokens, temperature=temperature)

    def _generate(self, input_ids, max_new_tokens: int, temperature: float = 0.0):
        """Single-prompt generate (token ids). Returns list[RequestOutput] (len 1).
        Mirrors the SGLang adapter's _generate signature; the bench calls this for
        the untimed cache-populate step."""
        sp = self._sampling(max_new_tokens, temperature)
        return self.llm.generate({"prompt_token_ids": list(input_ids)}, sp)

    def _generate_batch(self, list_input_ids, max_new_tokens: int, temperature: float = 0.0):
        sp = self._sampling(max_new_tokens, temperature)
        prompts = [{"prompt_token_ids": list(x)} for x in list_input_ids]
        return self.llm.generate(prompts, sp)

    def ttft(self, input_ids: list[int], temperature: float = 0.0) -> dict:
        """Time to first token, isolated to prefill via max_new_tokens=1."""
        t0 = time.perf_counter()
        out = self._generate(input_ids, max_new_tokens=1, temperature=temperature)
        wall = time.perf_counter() - t0
        recs = _as_list(out)
        rec = parse_request_metrics(recs[0]) if recs else parse_request_metrics(None)
        rec["ttft_s"] = wall
        return rec

    def generate_batch(self, list_input_ids: list[list[int]], max_new_tokens: int = 8,
                       temperature: float = 0.0) -> BatchTiming:
        """Serve a batch; body of the steady-state energy/throughput window."""
        t0 = time.perf_counter()
        out = self._generate_batch(list_input_ids, max_new_tokens=max_new_tokens,
                                   temperature=temperature)
        wall = time.perf_counter() - t0
        recs = [parse_request_metrics(o) for o in _as_list(out)]
        def _sum(key):
            return sum(r[key] or 0 for r in recs)
        return BatchTiming(
            wall_s=wall, n=len(recs),
            total_completion_tokens=_sum("completion_tokens"),
            total_prompt_tokens=_sum("prompt_tokens"),
            total_cached_tokens=_sum("cached_tokens"),
            records=recs,
        )

    def warmup(self, sample_input_ids: list[int], rounds: int = 2,
               max_new_tokens: int = 8) -> None:
        """Discarded warm rounds to leave the GPU + engine at steady state."""
        for _ in range(rounds):
            self._generate(sample_input_ids, max_new_tokens=max_new_tokens)

    def shutdown(self) -> None:
        """vLLM offline LLM has no explicit shutdown; drop refs and free VRAM."""
        try:
            import gc
            import torch
            if hasattr(self, "llm"):
                del self.llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Backward-compatible aliases (so the bench imports unchanged across the switch)
# ---------------------------------------------------------------------------

sglang_available = engine_available
sglang_version = engine_version
SglServingArm = VllmServingArm
parse_meta_info = parse_request_metrics


# ---------------------------------------------------------------------------
# Self-test (vLLM-free): validates the pure helpers + availability guard.
# Run from the repo root: python -m src.exp015_serving
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import types
    failures = 0
    print("EXP-015 vLLM serving adapter -- self-test (vLLM-free)")
    print(f"  engine_available (vllm + CUDA): {engine_available()}")
    print(f"  engine_version: {engine_version()} | engine_name: {engine_name()}")

    # cache_on=True -> enable_prefix_caching=True; tp/context/mem wired through
    on = build_engine_kwargs("Qwen/Qwen2.5-7B-Instruct", cache_on=True,
                             tp_size=1, context_length=8192)
    off = build_engine_kwargs("Qwen/Qwen2.5-7B-Instruct", cache_on=False,
                              tp_size=4, mem_fraction_static=0.85)
    ok = (on["enable_prefix_caching"] is True and off["enable_prefix_caching"] is False
          and on["tensor_parallel_size"] == 1 and off["tensor_parallel_size"] == 4
          and on["max_model_len"] == 8192 and "max_model_len" not in off
          and on["gpu_memory_utilization"] == 0.85 and on["model"].endswith("7B-Instruct")
          and on["dtype"] == "float16" and on["seed"] == 0)
    print(f"  [engine kwargs] ON.prefix={on['enable_prefix_caching']} "
          f"OFF.prefix={off['enable_prefix_caching']} tp(ON/OFF)={on['tensor_parallel_size']}/{off['tensor_parallel_size']} "
          f"-> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # request-metric parsing on a synthetic RequestOutput-like object
    synthetic = types.SimpleNamespace(
        prompt_token_ids=[1, 2, 3, 4, 5],
        num_cached_tokens=2,
        outputs=[types.SimpleNamespace(token_ids=[101, 102], finish_reason="length")],
        metrics=types.SimpleNamespace(arrival_time=10.0, finished_time=10.264),
    )
    p = parse_request_metrics(synthetic)
    ok = (p["prompt_tokens"] == 5 and p["completion_tokens"] == 2
          and p["cached_tokens"] == 2 and abs(p["e2e_latency"] - 0.264) < 1e-9
          and p["finish_reason"] == "length")
    print(f"  [parse metrics] {p} -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # tolerant of missing attributes / None
    bare = types.SimpleNamespace()
    pb = parse_request_metrics(bare)
    ok = (parse_request_metrics(None)["cached_tokens"] is None
          and pb["prompt_tokens"] is None and pb["completion_tokens"] is None
          and pb["cached_tokens"] is None and pb["e2e_latency"] is None)
    print(f"  [parse metrics missing] -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # _as_list normalisation (list vs single vs None)
    ok = (_as_list([synthetic, synthetic]) == [synthetic, synthetic]
          and _as_list(synthetic) == [synthetic] and _as_list(None) == [])
    print(f"  [_as_list] -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # aliases resolve to the vLLM implementations (bench imports unchanged)
    ok = (sglang_available is engine_available and sglang_version is engine_version
          and SglServingArm is VllmServingArm and parse_meta_info is parse_request_metrics)
    print(f"  [back-compat aliases] -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    print(f"\n  SELF-TEST {'PASSED' if failures == 0 else 'FAILED'} ({failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
