"""
EXP-002 -- flatten correctness GATE + redundant-hash guard (no timing).

Red-first contract (mirrors EXP-001's control/target split):
  * test_broken_flatten_reproduces_rehash_tax -> PASSES now: the CONTROL proves
    the problem is real -- the prior midpoint re-split flatten re-allocates leaves
    and recomputes hashes (re-hashing the context).
  * test_fixed_flatten_byte_identity           -> RED until flatten_fixed is
    implemented; then GREEN.
  * test_fixed_equals_broken                    -> RED until implemented; GREEN.
  * test_fixed_flatten_zero_rehash              -> RED until implemented; GREEN
    (the fix performs 0 splits, 0 leaf re-allocations, 0 hash recomputations).

Mechanism (validated; see EXP-002 LOGBOOK): on a bottom-up 4 KB-leaf rope with a
non-power-of-two leaf count, byte-midpoints misalign with leaf boundaries, so the
broken flatten reconstructs Leaf objects whose __init__ recomputes the polynomial
hash. The fix (rope_to_bytes) reads the already-stored bytes. Sizes are chosen
non-power-of-two on purpose; power-of-two leaf counts would align and hide the tax.

Authored, deterministic, offline fixtures (seeded random bytes; the flatten
mechanism is content-independent -- hashing cost is per byte regardless of
content). The paper-grade timing benchmark uses the documented downloaded corpus
(scripts/exp002_bench.py), not these unit fixtures.
"""
import random

import pytest
import hashrope.rope as _rope
from hashrope import PolynomialHash

from src.flatten import build_fat_leaf_rope, flatten_broken, flatten_fixed, make_hash

SIZES = [250_000, 500_000]   # 62 / 123 leaves @ 4 KB -- non-power-of-two on purpose
SEEDS = [42, 43]


def _corpus(nbytes: int, seed: int) -> bytes:
    """Deterministic varied bytes of length nbytes (content-independent mechanism)."""
    return random.Random(seed).randbytes(nbytes)


class _Counter:
    __slots__ = ("leaf", "split", "hash")

    def __init__(self):
        self.leaf = 0
        self.split = 0
        self.hash = 0


def _instrument(monkeypatch) -> _Counter:
    """Count Leaf re-allocations, rope_split calls, and hash recomputations."""
    c = _Counter()
    orig_leaf = _rope.Leaf.__init__
    orig_split = _rope.rope_split
    orig_hash = PolynomialHash.hash

    def counting_leaf(self, data, h):
        c.leaf += 1
        return orig_leaf(self, data, h)

    def counting_split(*a, **k):
        c.split += 1
        return orig_split(*a, **k)

    def counting_hash(self, data):
        c.hash += 1
        return orig_hash(self, data)

    monkeypatch.setattr(_rope.Leaf, "__init__", counting_leaf)
    monkeypatch.setattr(_rope, "rope_split", counting_split)
    monkeypatch.setattr(PolynomialHash, "hash", counting_hash)
    return c


@pytest.mark.parametrize("nbytes", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_broken_flatten_reproduces_rehash_tax(nbytes, seed, monkeypatch):
    """CONTROL: the broken flatten must re-allocate leaves and re-hash (tax is real)."""
    h = make_hash()
    data = _corpus(nbytes, seed)
    rope = build_fat_leaf_rope(data, h)          # build BEFORE instrumenting
    c = _instrument(monkeypatch)
    out = flatten_broken(rope, h)
    assert out == data, "control flatten must still be byte-correct"
    assert c.split > 0 and c.leaf > 0 and c.hash > 0, (
        f"tax not reproduced (rope shape aligned?): "
        f"counts={{'split': {c.split}, 'leaf': {c.leaf}, 'hash': {c.hash}}}"
    )


@pytest.mark.parametrize("nbytes", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_fixed_flatten_byte_identity(nbytes, seed):
    """GREEN TARGET: the fix must reproduce the original bytes exactly."""
    h = make_hash()
    data = _corpus(nbytes, seed)
    rope = build_fat_leaf_rope(data, h)
    assert flatten_fixed(rope) == data


@pytest.mark.parametrize("nbytes", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_fixed_equals_broken(nbytes, seed):
    """The fix must be byte-identical to the (correct) control output."""
    h = make_hash()
    data = _corpus(nbytes, seed)
    rope = build_fat_leaf_rope(data, h)
    assert flatten_fixed(rope) == flatten_broken(rope, h)


@pytest.mark.parametrize("nbytes", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_fixed_flatten_zero_rehash(nbytes, seed, monkeypatch):
    """GUARD: the fix performs 0 splits, 0 leaf re-allocations, 0 hash recomputations."""
    h = make_hash()
    data = _corpus(nbytes, seed)
    rope = build_fat_leaf_rope(data, h)          # build BEFORE instrumenting
    c = _instrument(monkeypatch)
    out = flatten_fixed(rope)
    assert out == data
    assert c.split == 0 and c.leaf == 0 and c.hash == 0, (
        f"the fix performed redundant work: "
        f"counts={{'split': {c.split}, 'leaf': {c.leaf}, 'hash': {c.hash}}}"
    )


def test_single_leaf_rope_no_tax(monkeypatch):
    """Edge: a rope smaller than one leaf flattens with 0 splits in the control."""
    h = make_hash()
    data = b"x" * 1000                            # < 4 KB -> single leaf
    rope = build_fat_leaf_rope(data, h)
    c = _instrument(monkeypatch)
    out = flatten_broken(rope, h)
    assert out == data and c.split == 0
