#!/usr/bin/env python3
"""
EXP-015: SGLang serving adapter (claim S3, Layer A -- the engine).

Thin wrapper over the SGLang OFFLINE Engine (in-process; no HTTP server, so the
NVML energy window wraps the generate calls directly with no network confound).
The engine is the Layer-A {cache OFF, cache ON} mechanism:

  cache ON  : default RadixAttention prefix caching.
  cache OFF : disable_radix_cache=True (verified current; mutually exclusive with
              enable_hierarchical_cache).

Requests are sent as `input_ids` (the exact token streams from the identification
core), so the served sequence is byte-identical to what the Layer-B identifier
arms see -- no re-tokenization drift. The response `meta_info` carries
`cached_tokens` (prompt tokens served from cache), which:
  - directly confirms cache OFF (cached_tokens == 0) vs ON (> 0 on shared prefixes), and
  - cross-checks Layer B: the identifier's matched-prefix length should equal the
    engine's reported cached_tokens under cache ON (a real-stack tie between the
    two layers, on top of criterion (i)).

GPU-free degradation: with no SGLang/CUDA (e.g. a CPU sandbox) sglang_available()
is False; the driver then runs the Layer-B identification track only (dry-run),
and the engine track is cluster-gated. The pure helpers (engine-kwarg build,
meta_info parsing) are unit-tested here and are SGLang-independent.

NOTE (cluster-smoke-gated): Engine.generate(input_ids=...) and Engine(**server_args)
follow the documented SGLang API. If the installed SGLang version differs, the
collaborator's smoke run is where any signature drift surfaces; the SGLang calls
are isolated in SglServingArm so only that class needs adjusting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Availability + version (SGLang-independent guards)
# ---------------------------------------------------------------------------

def sglang_available() -> bool:
    """True iff SGLang imports AND a CUDA device is visible (engine needs a GPU)."""
    try:
        import sglang  # noqa: F401
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def sglang_version() -> str:
    try:
        import sglang
        return getattr(sglang, "__version__", "unknown")
    except Exception:
        return "unavailable"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no SGLang needed)
# ---------------------------------------------------------------------------

def build_engine_kwargs(model_path: str, cache_on: bool, tp_size: int = 1,
                        mem_fraction_static: float = 0.85,
                        context_length: int | None = None,
                        dtype: str = "float16", random_seed: int = 0,
                        extra: dict | None = None) -> dict:
    """Construct sgl.Engine(**kwargs). cache OFF <=> disable_radix_cache=True."""
    kw: dict[str, Any] = {
        "model_path": model_path,
        "disable_radix_cache": (not cache_on),
        "tp_size": tp_size,
        "mem_fraction_static": mem_fraction_static,
        "dtype": dtype,
        "random_seed": random_seed,
    }
    if context_length is not None:
        kw["context_length"] = context_length
    if extra:
        kw.update(extra)
    return kw


def parse_meta_info(output: Any) -> dict:
    """Extract the fields we record from one SGLang generate output dict.

    Tolerant of missing keys (returns None) so the schema is stable across
    SGLang versions. Expected meta_info keys: prompt_tokens, completion_tokens,
    cached_tokens, e2e_latency, finish_reason.
    """
    mi = output.get("meta_info", {}) if isinstance(output, dict) else {}
    return {
        "prompt_tokens": mi.get("prompt_tokens"),
        "completion_tokens": mi.get("completion_tokens"),
        "cached_tokens": mi.get("cached_tokens"),
        "e2e_latency": mi.get("e2e_latency"),
        "finish_reason": mi.get("finish_reason"),
    }


def _as_list(out: Any) -> list:
    """Normalise a generate() return (dict for single, list for batch) to a list."""
    if isinstance(out, list):
        return out
    if out is None:
        return []
    return [out]


# ---------------------------------------------------------------------------
# Serving arm (offline Engine wrapper) -- construction requires SGLang + GPU
# ---------------------------------------------------------------------------

@dataclass
class BatchTiming:
    wall_s: float
    n: int
    total_completion_tokens: int
    total_prompt_tokens: int
    total_cached_tokens: int
    records: list[dict] = field(default_factory=list)


class SglServingArm:
    """Offline SGLang Engine for one cache setting. Construct on the cluster only.

    Usage:
        arm = SglServingArm("Qwen/Qwen2.5-7B-Instruct", cache_on=True, tp_size=1)
        arm.warmup(sample_input_ids)
        rec = arm.ttft(input_ids)                 # single-request, max_new_tokens=1
        bt  = arm.generate_batch(list_input_ids)  # throughput / energy window body
        arm.shutdown()
    """

    def __init__(self, model_path: str, cache_on: bool, tp_size: int = 1,
                 mem_fraction_static: float = 0.85, context_length: int | None = None,
                 dtype: str = "float16", random_seed: int = 0, extra: dict | None = None):
        import sglang as sgl  # cluster-only import
        self.cache_on = cache_on
        self.kwargs = build_engine_kwargs(
            model_path, cache_on, tp_size, mem_fraction_static,
            context_length, dtype, random_seed, extra)
        self.engine = sgl.Engine(**self.kwargs)

    def _generate(self, input_ids, max_new_tokens: int, temperature: float = 0.0):
        sp = {"max_new_tokens": max_new_tokens, "temperature": temperature}
        return self.engine.generate(input_ids=input_ids, sampling_params=sp)

    def ttft(self, input_ids: list[int], temperature: float = 0.0) -> dict:
        """Time to first token, isolated to prefill via max_new_tokens=1."""
        t0 = time.perf_counter()
        out = self._generate(input_ids, max_new_tokens=1, temperature=temperature)
        wall = time.perf_counter() - t0
        recs = _as_list(out)
        rec = parse_meta_info(recs[0]) if recs else parse_meta_info(None)
        rec["ttft_s"] = wall
        return rec

    def generate_batch(self, list_input_ids: list[list[int]], max_new_tokens: int = 8,
                       temperature: float = 0.0) -> BatchTiming:
        """Serve a batch; body of the steady-state energy/throughput window."""
        t0 = time.perf_counter()
        out = self._generate(list_input_ids, max_new_tokens=max_new_tokens,
                             temperature=temperature)
        wall = time.perf_counter() - t0
        recs = [parse_meta_info(o) for o in _as_list(out)]
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
        try:
            self.engine.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Self-test (SGLang-free): validates the pure helpers + availability guard.
# Run from the repo root: python -m src.exp015_serving
# ---------------------------------------------------------------------------

def _selftest() -> int:
    failures = 0
    print("EXP-015 serving adapter -- self-test (SGLang-free)")
    print(f"  sglang_available (sglang + CUDA): {sglang_available()}")
    print(f"  sglang_version: {sglang_version()}")

    # cache_on=True -> disable_radix_cache=False; tp/context wired through
    on = build_engine_kwargs("Qwen/Qwen2.5-7B-Instruct", cache_on=True,
                             tp_size=1, context_length=8192)
    off = build_engine_kwargs("Qwen/Qwen2.5-7B-Instruct", cache_on=False,
                              tp_size=4)
    ok = (on["disable_radix_cache"] is False and off["disable_radix_cache"] is True
          and on["tp_size"] == 1 and off["tp_size"] == 4
          and on["context_length"] == 8192 and "context_length" not in off
          and on["dtype"] == "float16")
    print(f"  [engine kwargs] ON.disable={on['disable_radix_cache']} "
          f"OFF.disable={off['disable_radix_cache']} tp(ON/OFF)={on['tp_size']}/{off['tp_size']} "
          f"-> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # meta_info parsing on a synthetic output mimicking the documented shape
    synthetic = {
        "text": " Paris.",
        "output_ids": [12095, 13],
        "meta_info": {"prompt_tokens": 5, "completion_tokens": 2,
                      "cached_tokens": 2, "e2e_latency": 0.264,
                      "finish_reason": {"type": "length", "length": 2}},
    }
    p = parse_meta_info(synthetic)
    ok = (p["prompt_tokens"] == 5 and p["completion_tokens"] == 2
          and p["cached_tokens"] == 2 and abs(p["e2e_latency"] - 0.264) < 1e-9)
    print(f"  [parse meta_info] {p} -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # tolerant of missing meta_info / None
    ok = (parse_meta_info({})["cached_tokens"] is None
          and parse_meta_info(None)["completion_tokens"] is None)
    print(f"  [parse meta_info missing] -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # _as_list normalisation (single dict vs batch list vs None)
    ok = (_as_list(synthetic) == [synthetic]
          and _as_list([synthetic, synthetic]) == [synthetic, synthetic]
          and _as_list(None) == [])
    print(f"  [_as_list] -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    print(f"\n  SELF-TEST {'PASSED' if failures == 0 else 'FAILED'} ({failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
