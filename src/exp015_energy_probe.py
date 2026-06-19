#!/usr/bin/env python3
"""
EXP-015: GPU energy probe (claim S3, energy DV).

NVML-based GPU energy measurement, isolated by GPU index, built to the rigor the
prior "thermodynamics" suite lacked (it was single-run, sub-second, GPU-ramp-
dominated, and time-confounded; see CLAIMS.md S3).

Energy paths (auto-selected; the live path is recorded in the results JSON):
  1. counter  (preferred) -- nvmlDeviceGetTotalEnergyConsumption, a hardware
     cumulative mJ counter (A100 supports it). Read at window start/end;
     difference = exact Joules, no integration error.
  2. power_integration (fallback) -- high-frequency nvmlDeviceGetPowerUsage (mW)
     sampled on a background thread and trapezoidally integrated over wall time.
     Used ONLY when the counter is unavailable.

Isolation: the probe measures a caller-specified LIST of GPU indices (R1: one
card; R2: the tensor-parallel working group). Per-GPU Joules are reported AND
summed over the working set -- idle / other-work GPUs are never included.

GPU-free degradation: with no NVML/driver (e.g. a CPU sandbox) available() is
False and the probe runs in 'unavailable' mode -- it returns None energy without
crashing, so the end-to-end pipeline (build stream -> serve -> record) can be
dry-run validated off-GPU. Real energy numbers require the cluster.

The pure measurement math (trapezoidal integration, counter delta) is unit-
tested in _selftest() and is GPU-independent.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import pynvml  # provided by the `nvidia-ml-py` package
    _HAS_PYNVML = True
except Exception:
    pynvml = None
    _HAS_PYNVML = False


# ---------------------------------------------------------------------------
# Pure measurement math (GPU-independent; unit-tested)
# ---------------------------------------------------------------------------

def trapezoid_joules(times_s: list[float], powers_w: list[float]) -> float:
    """Trapezoidal integral of power(W) over time(s) -> energy(J).

    times_s must be non-decreasing and the same length as powers_w (>= 2
    samples). Returns 0.0 for fewer than 2 samples.
    """
    n = min(len(times_s), len(powers_w))
    if n < 2:
        return 0.0
    e = 0.0
    for i in range(1, n):
        dt = times_s[i] - times_s[i - 1]
        if dt < 0:
            raise ValueError("times_s must be non-decreasing")
        e += 0.5 * (powers_w[i] + powers_w[i - 1]) * dt
    return e


def counter_delta_joules(start_mj: int, end_mj: int) -> float:
    """Energy from two cumulative mJ counter reads -> Joules.

    The mJ counter is a 64-bit cumulative value, monotonic in practice; a
    backwards reading is treated as an error rather than silently yielding
    negative energy.
    """
    d = end_mj - start_mj
    if d < 0:
        raise ValueError(f"energy counter went backwards: {start_mj} -> {end_mj}")
    return d / 1000.0


# ---------------------------------------------------------------------------
# NVML availability + per-GPU metadata
# ---------------------------------------------------------------------------

_nvml_inited = False


def _ensure_nvml() -> bool:
    global _nvml_inited
    if not _HAS_PYNVML:
        return False
    if _nvml_inited:
        return True
    try:
        pynvml.nvmlInit()
        _nvml_inited = True
        return True
    except Exception:
        return False


def available() -> bool:
    """True iff NVML initialises (driver + GPU present)."""
    return _ensure_nvml()


def _counter_supported(handle) -> bool:
    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return True
    except Exception:
        return False


def _decode(x) -> str:
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def gpu_metadata(gpu_indices: list[int]) -> list[dict]:
    """Per-GPU static metadata for the results env block. Empty list if no NVML."""
    if not _ensure_nvml():
        return []
    try:
        driver = _decode(pynvml.nvmlSystemGetDriverVersion())
    except Exception:
        driver = "unknown"
    out = []
    for i in gpu_indices:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
        except Exception:
            out.append({"index": i, "error": "handle unavailable"})
            continue
        rec: dict[str, Any] = {"index": i, "driver": driver}
        try:
            rec["name"] = _decode(pynvml.nvmlDeviceGetName(h))
        except Exception:
            rec["name"] = "unknown"
        try:
            rec["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            rec["mem_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
        except Exception:
            pass
        try:
            rec["power_limit_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
        except Exception:
            try:
                rec["power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
            except Exception:
                pass
        rec["energy_counter_supported"] = _counter_supported(h)
        out.append(rec)
    return out


def energy_path(gpu_indices: list[int], force_power_integration: bool = False) -> str:
    """Which path will be used: 'counter' | 'power_integration' | 'unavailable'."""
    if not _ensure_nvml():
        return "unavailable"
    if force_power_integration:
        return "power_integration"
    try:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices]
        if all(_counter_supported(h) for h in handles):
            return "counter"
    except Exception:
        return "unavailable"
    return "power_integration"


# ---------------------------------------------------------------------------
# Energy probe (context manager) over a window, per GPU index + summed
# ---------------------------------------------------------------------------

@dataclass
class EnergyResult:
    path: str                                  # counter | power_integration | unavailable
    per_gpu_joules: dict[int, float | None]
    total_joules: float | None
    wall_s: float
    n_samples: dict[int, int] = field(default_factory=dict)  # power-integration only


class GpuEnergyProbe:
    """Context manager measuring GPU energy over the `with` block.

    Usage:
        probe = GpuEnergyProbe([0])
        with probe:
            ...   # steady-state work on GPU 0
        res = probe.result()          # EnergyResult
    """

    def __init__(self, gpu_indices: list[int], sample_hz: float = 100.0,
                 force_power_integration: bool = False):
        self.gpu_indices = list(gpu_indices)
        self.sample_hz = sample_hz
        self.path = energy_path(self.gpu_indices, force_power_integration)
        self._handles: dict[int, Any] = {}
        self._start_mj: dict[int, int] = {}
        self._end_mj: dict[int, int] = {}
        self._samples: dict[int, list[tuple[float, float]]] = {i: [] for i in self.gpu_indices}
        self._sampler: threading.Thread | None = None
        self._stop = threading.Event()
        self._t0 = 0.0
        self._t1 = 0.0
        self._result: EnergyResult | None = None
        if self.path != "unavailable":
            for i in self.gpu_indices:
                self._handles[i] = pynvml.nvmlDeviceGetHandleByIndex(i)

    def _sample_loop(self):
        period = 1.0 / self.sample_hz
        while not self._stop.is_set():
            t = time.perf_counter()
            for i, h in self._handles.items():
                try:
                    mw = pynvml.nvmlDeviceGetPowerUsage(h)
                    self._samples[i].append((t, mw / 1000.0))  # W
                except Exception:
                    pass
            elapsed = time.perf_counter() - t
            if elapsed < period:
                self._stop.wait(period - elapsed)

    def __enter__(self):
        self._t0 = time.perf_counter()
        if self.path == "counter":
            for i, h in self._handles.items():
                self._start_mj[i] = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        elif self.path == "power_integration":
            self._stop.clear()
            self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
            self._sampler.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.path == "counter":
            for i, h in self._handles.items():
                self._end_mj[i] = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        elif self.path == "power_integration":
            self._stop.set()
            if self._sampler is not None:
                self._sampler.join(timeout=2.0)
        self._t1 = time.perf_counter()
        self._finalize()
        return False  # never suppress exceptions

    def _finalize(self):
        wall = self._t1 - self._t0
        per_gpu: dict[int, float | None] = {}
        n_samples: dict[int, int] = {}
        if self.path == "counter":
            for i in self.gpu_indices:
                per_gpu[i] = counter_delta_joules(self._start_mj[i], self._end_mj[i])
        elif self.path == "power_integration":
            for i in self.gpu_indices:
                ts = [t for (t, _w) in self._samples[i]]
                ws = [w for (_t, w) in self._samples[i]]
                per_gpu[i] = trapezoid_joules(ts, ws)
                n_samples[i] = len(ts)
        else:  # unavailable
            for i in self.gpu_indices:
                per_gpu[i] = None
        total = (None if self.path == "unavailable"
                 else sum(v for v in per_gpu.values() if v is not None))
        self._result = EnergyResult(path=self.path, per_gpu_joules=per_gpu,
                                    total_joules=total, wall_s=wall, n_samples=n_samples)

    def result(self) -> EnergyResult:
        if self._result is None:
            raise RuntimeError("probe.result() called before the `with` block completed")
        return self._result


# ---------------------------------------------------------------------------
# Self-test (GPU-free): validates the pure math + unavailable-mode + path logic.
# Run from the repo root: python -m src.exp015_energy_probe
# ---------------------------------------------------------------------------

def _selftest() -> int:
    failures = 0
    print("EXP-015 energy probe -- self-test (GPU-free)")
    print(f"  pynvml importable: {_HAS_PYNVML}")
    print(f"  NVML available (GPU present): {available()}")

    # constant 100 W for 2.0 s -> 200 J
    e = trapezoid_joules([0.0, 0.5, 1.0, 1.5, 2.0], [100.0] * 5)
    ok = abs(e - 200.0) < 1e-9
    print(f"  [trapezoid const] 100W x 2s = {e:.6f} J (expect 200) -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # linear ramp 0->200 W over 2 s -> mean 100 W -> 200 J
    e = trapezoid_joules([0.0, 1.0, 2.0], [0.0, 100.0, 200.0])
    ok = abs(e - 200.0) < 1e-9
    print(f"  [trapezoid ramp]  0->200W over 2s = {e:.6f} J (expect 200) -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # degenerate sample counts -> 0.0
    ok = trapezoid_joules([], []) == 0.0 and trapezoid_joules([1.0], [50.0]) == 0.0
    print(f"  [trapezoid <2 samples] -> 0.0 -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # counter delta: 1000 mJ -> 4000 mJ = 3 J
    e = counter_delta_joules(1000, 4000)
    ok = abs(e - 3.0) < 1e-12
    print(f"  [counter delta] 1000->4000 mJ = {e} J (expect 3) -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # counter going backwards must raise
    try:
        counter_delta_joules(5000, 1000)
        print("  [counter backwards] -> FAIL (no raise)")
        failures += 1
    except ValueError:
        print("  [counter backwards] -> PASS (raised)")

    # unavailable-mode probe does not crash; returns None energy
    probe = GpuEnergyProbe([0])
    with probe:
        _ = sum(range(1000))  # CPU stand-in for 'work'
    res = probe.result()
    ok = (res.path == "unavailable" and res.total_joules is None
          and res.per_gpu_joules.get(0) is None and res.wall_s >= 0)
    print(f"  [unavailable-mode] path={res.path} total={res.total_joules} "
          f"wall={res.wall_s:.4f}s -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # path selection is one of the three known strings
    p = energy_path([0])
    ok = p in ("counter", "power_integration", "unavailable")
    print(f"  [energy_path] -> {p} -> {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    print(f"\n  SELF-TEST {'PASSED' if failures == 0 else 'FAILED'} ({failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
