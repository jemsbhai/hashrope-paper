"""
EXP-004 -- branch/snapshot correctness GATE + structural sharing guard.

Red-first contract:
  * test_fork_byte_identity       -> RED at stub (fork_rope raises NotImplementedError);
    GREEN when implemented. Every fork materializes to base_bytes + thought_bytes.
  * test_rope_shares_nodes        -> RED at stub (count_unique_nodes raises);
    GREEN when sharing is confirmed. Rope forks share interior nodes.
  * test_deepcopy_no_sharing      -> RED at stub; GREEN when deepcopy duplication
    is confirmed. Every node is duplicated.
  * test_tracemalloc_rope_smaller -> RED at stub; GREEN when rope memory < deepcopy.
  * test_sharing_ratio_grows      -> RED at stub; GREEN when ratio ≥ 3× at B=50.

Authored, deterministic, offline fixtures. The paper-grade memory benchmark uses
the documented downloaded corpus (scripts/exp004_bench.py), not these unit fixtures.
"""
import math
import random
import tracemalloc

import pytest
import hashrope.rope as _rope
from hashrope import PolynomialHash

from src.flatten import build_fat_leaf_rope, make_hash
from src.memory import count_unique_nodes, fork_rope, fork_deepcopy

# Small sizes for fast unit tests (mechanism is size-independent)
CORPUS_SIZE = 64_000       # 16 leaves @ 4 KB -- small enough for fast tests
B_SMALL = 10               # number of forks for basic tests
B_LARGE = 50               # for the sharing-ratio test
THOUGHT_BYTES = 128
SEED = 42


def _corpus(nbytes: int, seed: int) -> bytes:
    """Deterministic varied bytes."""
    return random.Random(seed).randbytes(nbytes)


def _thoughts(corpus: bytes, n: int, offset: int, size: int = THOUGHT_BYTES) -> list[bytes]:
    """Extract n deterministic thought slices from corpus starting at offset."""
    return [corpus[offset + i * size: offset + (i + 1) * size] for i in range(n)]


@pytest.fixture
def h():
    return make_hash()


@pytest.fixture
def corpus():
    # Extra bytes beyond CORPUS_SIZE for thought slices
    return _corpus(CORPUS_SIZE + B_LARGE * THOUGHT_BYTES + 1024, SEED)


@pytest.fixture
def base_rope(corpus, h):
    return build_fat_leaf_rope(corpus[:CORPUS_SIZE], h)


@pytest.fixture
def base_bytes(corpus):
    return corpus[:CORPUS_SIZE]


@pytest.fixture
def thoughts_small(corpus):
    return _thoughts(corpus, B_SMALL, CORPUS_SIZE)


@pytest.fixture
def thoughts_large(corpus):
    return _thoughts(corpus, B_LARGE, CORPUS_SIZE)


# ---- Correctness gate (HARD) ----

class TestForkByteIdentity:
    """Every fork materializes to base_bytes + thought_bytes."""

    def test_rope_forks(self, base_rope, base_bytes, thoughts_small, h):
        forks = fork_rope(base_rope, thoughts_small, h)
        assert len(forks) == B_SMALL
        for i, fork in enumerate(forks):
            expected = base_bytes + thoughts_small[i]
            actual = _rope.rope_to_bytes(fork)
            assert actual == expected, f"Rope fork {i} content mismatch"

    def test_deepcopy_forks(self, base_rope, base_bytes, thoughts_small, h):
        forks = fork_deepcopy(base_rope, thoughts_small, h)
        assert len(forks) == B_SMALL
        for i, fork in enumerate(forks):
            expected = base_bytes + thoughts_small[i]
            actual = _rope.rope_to_bytes(fork)
            assert actual == expected, f"Deepcopy fork {i} content mismatch"


# ---- Structural sharing guard ----

class TestNodeSharing:
    """Rope forks share interior nodes; deepcopy forks do not."""

    def test_rope_shares_nodes(self, base_rope, thoughts_small, h):
        forks = fork_rope(base_rope, thoughts_small, h)
        # Count unique nodes across base + all forks
        all_roots = [base_rope] + forks
        stats = count_unique_nodes(all_roots)
        unique = stats["unique_count"]

        # Count base-only unique nodes
        base_stats = count_unique_nodes([base_rope])
        base_unique = base_stats["unique_count"]

        # Rope sharing: unique should be base + O(B * log w) new nodes
        # NOT base * B (which would mean no sharing)
        # With 16 leaves (w=31 nodes), log2(31) ≈ 5, so B=10 forks add ~60 nodes
        # Total should be ~31 + 60 = ~91, NOT 31 * 10 = 310
        assert unique < B_SMALL * base_unique, (
            f"Rope forks should share nodes: unique={unique}, "
            f"B * base_unique={B_SMALL * base_unique}"
        )
        # Stronger: new nodes per fork should be O(log w)
        w = base_rope.weight
        log_w = math.ceil(math.log2(max(w, 2)))
        max_new_per_fork = log_w + 3  # log(w) spine + 1 new leaf + margin
        max_total = base_unique + B_SMALL * max_new_per_fork
        assert unique <= max_total, (
            f"Too many unique nodes: {unique} > {max_total} "
            f"(base={base_unique}, B={B_SMALL}, log_w={log_w})"
        )

    def test_deepcopy_no_sharing(self, base_rope, thoughts_small, h):
        forks = fork_deepcopy(base_rope, thoughts_small, h)
        # Count unique nodes across base + all forks
        all_roots = [base_rope] + forks
        stats = count_unique_nodes(all_roots)
        unique = stats["unique_count"]

        base_stats = count_unique_nodes([base_rope])
        base_unique = base_stats["unique_count"]

        # Deepcopy: each fork has its own full copy + new nodes
        # unique should be ≥ B * base_unique (every node duplicated)
        assert unique >= B_SMALL * base_unique, (
            f"Deepcopy should duplicate all nodes: unique={unique}, "
            f"B * base_unique={B_SMALL * base_unique}"
        )


# ---- Memory measurement ----

class TestTracemalloc:
    """Rope forks use less memory than deepcopy forks."""

    def test_rope_smaller_than_deepcopy(self, corpus, h):
        base_data = corpus[:CORPUS_SIZE]
        thoughts = _thoughts(corpus, B_SMALL, CORPUS_SIZE)

        # Measure rope arm
        base_rope = build_fat_leaf_rope(base_data, h)
        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()
        rope_forks = fork_rope(base_rope, thoughts, h)
        snap_after = tracemalloc.take_snapshot()
        rope_delta = sum(
            s.size for s in snap_after.compare_to(snap_before, "lineno")
            if s.size_diff > 0
        )
        # Keep forks alive until after snapshot
        assert len(rope_forks) == B_SMALL
        tracemalloc.stop()

        # Measure deepcopy arm (fresh tracemalloc)
        base_rope2 = build_fat_leaf_rope(base_data, h)
        tracemalloc.start()
        snap_before2 = tracemalloc.take_snapshot()
        dc_forks = fork_deepcopy(base_rope2, thoughts, h)
        snap_after2 = tracemalloc.take_snapshot()
        dc_delta = sum(
            s.size for s in snap_after2.compare_to(snap_before2, "lineno")
            if s.size_diff > 0
        )
        assert len(dc_forks) == B_SMALL
        tracemalloc.stop()

        assert rope_delta < dc_delta, (
            f"Rope should use less memory: rope={rope_delta}, deepcopy={dc_delta}"
        )


# ---- Sharing ratio at scale ----

class TestSharingRatio:
    """At B=50, sharing ratio is significant (≥ 3× even at small N).

    The confirmatory benchmark (LOGBOOK criterion) checks ≥ 10× at N=2M.
    At unit-test scale (N=64K, w=31), the theoretical maximum is w/log(w) ≈ 6×.
    """

    def test_sharing_ratio_large_b(self, base_rope, thoughts_large, h):
        rope_forks = fork_rope(base_rope, thoughts_large, h)
        dc_forks = fork_deepcopy(base_rope, thoughts_large, h)

        rope_stats = count_unique_nodes([base_rope] + rope_forks)
        dc_stats = count_unique_nodes([base_rope] + dc_forks)

        ratio = dc_stats["unique_count"] / rope_stats["unique_count"]
        # At small N (64K), ratio ≈ w/log(w) ≈ 6×; confirmatory ≥ 10× is at N=2M
        assert ratio >= 3.0, (
            f"Sharing ratio too low: {ratio:.1f}× "
            f"(rope={rope_stats['unique_count']}, dc={dc_stats['unique_count']})"
        )


# ---- count_unique_nodes sanity ----

class TestCountUniqueNodes:
    """Basic sanity for the node-counting utility."""

    def test_single_leaf(self, h):
        leaf = _rope.Leaf(b"hello", h)
        stats = count_unique_nodes([leaf])
        assert stats["unique_count"] == 1
        assert stats["by_type"]["Leaf"] == 1

    def test_same_root_twice(self, base_rope):
        """Passing the same root twice should not double-count."""
        stats_once = count_unique_nodes([base_rope])
        stats_twice = count_unique_nodes([base_rope, base_rope])
        assert stats_once["unique_count"] == stats_twice["unique_count"]
