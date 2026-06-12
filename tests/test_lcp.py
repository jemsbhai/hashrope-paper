"""
EXP-005 -- LCP correctness GATE + step-count guard (no timing).

Red-first contract:
  * test_lcp_correctness_real_corpus  -> RED (stub raises NotImplementedError)
  * test_lcp_correctness_synthetic    -> RED
  * test_lcp_step_count               -> RED
  * test_lcp_zero_prefix              -> RED
  * test_lcp_full_match               -> RED
  * test_lcp_empty_ropes              -> RED

Mechanism: lcp_hash performs binary search over prefix length L ∈ [0, N],
comparing rope_substr_hash(A, 0, L, h) vs rope_substr_hash(B, 0, L, h) at
each step. O(log N) steps × O(log w) per hash = O(log² N). Correctness is
verified against lcp_brute (byte-by-byte oracle) and against algebraically
known LCP values (single byte flipped at position L → LCP = L exactly).

Step-count guard: the binary search must perform ≤ ceil(log2(min(len_a,
len_b))) + 1 calls to rope_substr_hash (2 calls per step — one per rope —
so total calls ≤ 2 * (ceil(log2(N)) + 1)).

Unit-test sizes are small (10 KB – 50 KB). Paper-grade timing uses the
downloaded corpus and the full size sweep (scripts/exp005_bench.py).
"""
import math
import random

import pytest
import hashrope.rope as _rope
from hashrope import PolynomialHash, rope_substr_hash

from src.flatten import build_fat_leaf_rope, make_hash
from src.lcp import lcp_brute, lcp_hash, lcp_hash_counted

SIZES = [10_000, 50_000]
SEEDS = [42, 43]
FRACTIONS = [0.0, 0.5, 0.99]


def _corpus(nbytes: int, seed: int) -> bytes:
    """Deterministic varied bytes."""
    return random.Random(seed).randbytes(nbytes)


def _make_pair(data: bytes, f: float, h: PolynomialHash):
    """Build rope pair (A, B) with known LCP = int(f * len(data)).

    B is a copy of data with byte at position L flipped (XOR 0xFF).
    For f=0.0, L=0 → first byte differs → LCP = 0.
    For f=1.0, no flip → LCP = len(data).
    """
    n = len(data)
    lcp_expected = int(f * n)
    a_bytes = data
    b_bytes = bytearray(data)
    if lcp_expected < n:
        b_bytes[lcp_expected] ^= 0xFF  # guarantees difference
    b_bytes = bytes(b_bytes)
    rope_a = build_fat_leaf_rope(a_bytes, h)
    rope_b = build_fat_leaf_rope(b_bytes, h)
    return rope_a, rope_b, a_bytes, b_bytes, lcp_expected


# ── Correctness gate (real corpus) ──────────────────────────────────────


@pytest.mark.parametrize("nbytes", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("f", FRACTIONS)
def test_lcp_correctness_real_corpus(nbytes, seed, f):
    """lcp_hash must return the exact same position as lcp_brute."""
    h = make_hash()
    data = _corpus(nbytes, seed)
    rope_a, rope_b, a_bytes, b_bytes, lcp_expected = _make_pair(data, f, h)

    brute = lcp_brute(a_bytes, b_bytes)
    hashed = lcp_hash(rope_a, rope_b, h)

    assert brute == lcp_expected, (
        f"lcp_brute={brute} != expected={lcp_expected} (f={f}, N={nbytes})"
    )
    assert hashed == lcp_expected, (
        f"lcp_hash={hashed} != expected={lcp_expected} (f={f}, N={nbytes})"
    )
    assert hashed == brute, (
        f"lcp_hash={hashed} != lcp_brute={brute}"
    )


# ── Correctness gate (synthetic control) ────────────────────────────────


@pytest.mark.parametrize("lcp_len", [0, 1, 100, 4095, 4096, 4097, 10_000])
def test_lcp_correctness_synthetic(lcp_len):
    """Algebraically known LCP on uniform-byte data with single flip."""
    h = make_hash()
    n = max(lcp_len + 1, 1024)  # ensure rope is at least 1 KB
    a_bytes = bytes([0xAB] * n)
    b_bytes = bytearray(a_bytes)
    if lcp_len < n:
        b_bytes[lcp_len] ^= 0xFF
    b_bytes = bytes(b_bytes)

    rope_a = build_fat_leaf_rope(a_bytes, h)
    rope_b = build_fat_leaf_rope(b_bytes, h)

    assert lcp_brute(a_bytes, b_bytes) == lcp_len
    assert lcp_hash(rope_a, rope_b, h) == lcp_len


# ── Step-count guard ────────────────────────────────────────────────────


@pytest.mark.parametrize("nbytes", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_lcp_step_count(nbytes, seed):
    """Hash queries ≤ 2 * (ceil(log2(N)) + 1) -- two ropes per binary-search step."""
    h = make_hash()
    data = _corpus(nbytes, seed)
    rope_a, rope_b, _, _, _ = _make_pair(data, 0.5, h)

    n = min(nbytes, nbytes)  # both same length here
    max_steps = math.ceil(math.log2(n)) + 1
    max_calls = 2 * max_steps  # two rope_substr_hash calls per step

    lcp_val, num_calls = lcp_hash_counted(rope_a, rope_b, h)
    assert num_calls <= max_calls, (
        f"step-count guard violated: {num_calls} calls > {max_calls} "
        f"(ceil(log2({n}))={math.ceil(math.log2(n))}, +1={max_steps})"
    )


# ── Edge cases ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("nbytes", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_lcp_zero_prefix(nbytes, seed):
    """f=0.0: first byte differs → LCP = 0."""
    h = make_hash()
    data = _corpus(nbytes, seed)
    rope_a, rope_b, a_bytes, b_bytes, _ = _make_pair(data, 0.0, h)
    assert lcp_brute(a_bytes, b_bytes) == 0
    assert lcp_hash(rope_a, rope_b, h) == 0


@pytest.mark.parametrize("nbytes", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_lcp_full_match(nbytes, seed):
    """Identical ropes → LCP = len(data)."""
    h = make_hash()
    data = _corpus(nbytes, seed)
    rope_a = build_fat_leaf_rope(data, h)
    rope_b = build_fat_leaf_rope(data, h)
    assert lcp_brute(data, data) == nbytes
    assert lcp_hash(rope_a, rope_b, h) == nbytes


def test_lcp_empty_ropes():
    """Empty / None ropes → LCP = 0."""
    h = make_hash()
    assert lcp_brute(b"", b"") == 0
    assert lcp_brute(b"abc", b"") == 0
    assert lcp_brute(b"", b"abc") == 0
    assert lcp_hash(None, None, h) == 0
    rope = build_fat_leaf_rope(b"hello", h)
    assert lcp_hash(rope, None, h) == 0
    assert lcp_hash(None, rope, h) == 0
