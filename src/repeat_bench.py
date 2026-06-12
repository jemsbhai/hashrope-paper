"""
EXP-006: RepeatNode O(log q) compression/throughput vs naïve materialization.

build_repeat
    rope_repeat(unit_rope, q, h) — 1 new RepeatNode, O(log q) for Φ.

build_naive_repeat
    build_fat_leaf_rope(unit_bytes * q, h) — O(q) leaves + internals.

Both produce byte-identical content with identical polynomial hashes.
count_unique_nodes reused from src.memory.
"""
from __future__ import annotations

import hashrope.rope as _rope
from hashrope import PolynomialHash
from src.flatten import build_fat_leaf_rope


def build_repeat(unit_rope, q: int, h: PolynomialHash):
    """RepeatNode arm: O(1) node, O(log q) hash."""
    return _rope.rope_repeat(unit_rope, q, h)


def build_naive_repeat(unit_bytes: bytes, q: int, h: PolynomialHash):
    """Naïve arm: materialize unit_bytes * q as a flat rope."""
    return build_fat_leaf_rope(unit_bytes * q, h)
