"""
EXP-006 -- RepeatNode correctness GATE + node-count guard.

Red-first contract:
  * test_repeat_correctness   -> RED at stub; GREEN when bytes + hash match.
  * test_repeat_node_count    -> RED at stub; GREEN when repeat = unit_nodes + 1.
  * test_naive_node_count     -> RED at stub; GREEN when naïve ≥ q nodes.
  * test_repeat_faster        -> RED at stub; GREEN when repeat beats naïve.
  * test_q1_identity          -> RED at stub; GREEN when q=1 returns the unit itself.

Authored, deterministic, offline fixtures. The paper-grade benchmark uses
the documented downloaded corpus (scripts/exp006_bench.py).
"""
import random
import time

import pytest
import hashrope.rope as _rope
from hashrope import PolynomialHash

from src.flatten import build_fat_leaf_rope, make_hash
from src.memory import count_unique_nodes
from src.repeat_bench import build_repeat, build_naive_repeat

UNIT_SIZE = 4096
Q_SMALL = 10
Q_LARGE = 1000
SEED = 42


def _corpus(nbytes: int, seed: int) -> bytes:
    return random.Random(seed).randbytes(nbytes)


@pytest.fixture
def h():
    return make_hash()


@pytest.fixture
def corpus():
    return _corpus(UNIT_SIZE * 2, SEED)


@pytest.fixture
def unit_bytes(corpus):
    return corpus[:UNIT_SIZE]


@pytest.fixture
def unit_rope(unit_bytes, h):
    return build_fat_leaf_rope(unit_bytes, h)


# ---- Correctness gate (HARD) ----

class TestRepeatCorrectness:
    """RepeatNode and naïve produce identical bytes and hash."""

    def test_bytes_identity(self, unit_bytes, unit_rope, h):
        repeat = build_repeat(unit_rope, Q_SMALL, h)
        naive = build_naive_repeat(unit_bytes, Q_SMALL, h)
        expected = unit_bytes * Q_SMALL

        assert _rope.rope_to_bytes(repeat) == expected, "RepeatNode bytes mismatch"
        assert _rope.rope_to_bytes(naive) == expected, "Naïve bytes mismatch"

    def test_hash_identity(self, unit_bytes, unit_rope, h):
        repeat = build_repeat(unit_rope, Q_SMALL, h)
        naive = build_naive_repeat(unit_bytes, Q_SMALL, h)

        assert _rope.rope_hash(repeat) == _rope.rope_hash(naive), (
            f"Hash mismatch: repeat={_rope.rope_hash(repeat)}, "
            f"naive={_rope.rope_hash(naive)}"
        )

    def test_large_q(self, unit_bytes, unit_rope, h):
        repeat = build_repeat(unit_rope, Q_LARGE, h)
        naive = build_naive_repeat(unit_bytes, Q_LARGE, h)

        assert _rope.rope_hash(repeat) == _rope.rope_hash(naive)
        assert _rope.rope_len(repeat) == _rope.rope_len(naive) == UNIT_SIZE * Q_LARGE


# ---- Node-count guard ----

class TestNodeCount:
    """RepeatNode = unit_nodes + 1; naïve = O(q) nodes."""

    def test_repeat_node_count(self, unit_rope, h):
        unit_stats = count_unique_nodes([unit_rope])
        repeat = build_repeat(unit_rope, Q_LARGE, h)
        repeat_stats = count_unique_nodes([repeat])

        # RepeatNode adds exactly 1 node to the unit subtree
        assert repeat_stats["unique_count"] == unit_stats["unique_count"] + 1, (
            f"Expected {unit_stats['unique_count'] + 1}, "
            f"got {repeat_stats['unique_count']}"
        )

    def test_naive_node_count(self, unit_bytes, h):
        naive = build_naive_repeat(unit_bytes, Q_LARGE, h)
        naive_stats = count_unique_nodes([naive])

        # Naïve: ~2 * ceil(Q_LARGE * UNIT_SIZE / 4096) nodes
        # At unit=4KB, q=1000: 1000 leaves + ~999 internals = ~1999
        assert naive_stats["unique_count"] >= Q_LARGE, (
            f"Naïve should have ≥ {Q_LARGE} nodes, got {naive_stats['unique_count']}"
        )

    def test_compression_ratio(self, unit_bytes, unit_rope, h):
        repeat = build_repeat(unit_rope, Q_LARGE, h)
        naive = build_naive_repeat(unit_bytes, Q_LARGE, h)

        r_stats = count_unique_nodes([repeat])
        n_stats = count_unique_nodes([naive])

        ratio = n_stats["unique_count"] / r_stats["unique_count"]
        # At q=1000, unit=4KB: ~1999 / 2 ≈ 1000×
        assert ratio >= 100, f"Compression ratio too low: {ratio:.1f}×"


# ---- Timing ----

class TestTiming:
    """RepeatNode construction is faster than naïve."""

    def test_repeat_faster(self, unit_bytes, unit_rope, h):
        # Warmup
        _ = build_repeat(unit_rope, Q_LARGE, h)
        _ = build_naive_repeat(unit_bytes, Q_LARGE, h)

        # Time repeat
        t0 = time.perf_counter()
        for _ in range(5):
            build_repeat(unit_rope, Q_LARGE, h)
        t_repeat = time.perf_counter() - t0

        # Time naïve
        t0 = time.perf_counter()
        for _ in range(5):
            build_naive_repeat(unit_bytes, Q_LARGE, h)
        t_naive = time.perf_counter() - t0

        assert t_repeat < t_naive, (
            f"RepeatNode should be faster: repeat={t_repeat*1000:.1f}ms, "
            f"naive={t_naive*1000:.1f}ms"
        )


# ---- Edge cases ----

class TestEdgeCases:
    """q=1 returns the unit itself; q=0 returns None."""

    def test_q1_returns_unit(self, unit_rope, h):
        result = build_repeat(unit_rope, 1, h)
        assert result is unit_rope, "q=1 should return the unit itself"

    def test_q0_returns_none(self, unit_rope, h):
        result = build_repeat(unit_rope, 0, h)
        assert result is None, "q=0 should return None"
