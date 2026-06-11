"""
EXP-002: flatten -- in-order materialization vs midpoint re-split (claim S4).

Reproduces the prior draft's flatten "tax" and provides the fix, on a faithfully
reconstructed rope shape.

build_fat_leaf_rope
    Builds a balanced rope from bytes using <=leaf_bytes (4 KB) leaves and a
    BOTTOM-UP pairwise merge -- EXACTLY the prior HashRopeAdapter construction
    (old/hashrope1_exmain.py L40). This shape matters: rope_from_bytes alone
    returns a single Leaf (no flatten cost), and on a *power-of-two* leaf count
    every byte-midpoint aligns to a leaf boundary and the tax vanishes. Realistic
    (non-power-of-two) leaf counts misalign, which is what triggers the tax.

flatten_broken  (CONTROL -- reproduces the prior ~945 ms tax)
    Faithful port of the prior flatten_context (old/hashrope1_exmain.py L579):
    materialize by recursively splitting at midpoints down to <=leaf_bytes pieces.
    Where a midpoint lands inside a leaf, rope_split reconstructs Leaf objects, and
    every Leaf.__init__ RECOMPUTES its polynomial hash from scratch -- re-hashing
    ~Theta(N) bytes through pure-Python per-byte modular reduction (mersenne_mod).
    Profiled: ~99% of the time is PolynomialHash.hash (sandbox: 783 ms @ 2 MB).

flatten_fixed  (FIX)
    The library's in-order materializer, rope_to_bytes: read each stored Leaf.data
    once and join. 0 splits, 0 Leaf re-allocations, 0 hash recomputation. O(N),
    alignment-independent. The flatten cost is not inherent.

References to hashrope.rope.{rope_split, rope_concat, rope_to_bytes, Leaf} go
through the module object so tests can monkeypatch and count operations.
"""
from __future__ import annotations

import hashrope.rope as _rope
from hashrope import PolynomialHash

DEFAULT_LEAF_BYTES = 4096


def make_hash() -> PolynomialHash:
    """Shared hash config (MERSENNE_61, base 131), matching the prior adapter."""
    return PolynomialHash()


def build_fat_leaf_rope(data: bytes, h: PolynomialHash,
                        leaf_bytes: int = DEFAULT_LEAF_BYTES):
    """Balanced <=leaf_bytes-leaf rope via bottom-up pairwise merge (HashRopeAdapter).

    Returns None for empty input.
    """
    if len(data) == 0:
        return None
    nodes = [_rope.rope_from_bytes(data[i:i + leaf_bytes], h)
             for i in range(0, len(data), leaf_bytes)]
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                nxt.append(_rope.rope_concat(nodes[i], nodes[i + 1], h))
            else:
                nxt.append(nodes[i])
        nodes = nxt
    return nodes[0]


def flatten_broken(node, h: PolynomialHash,
                   leaf_bytes: int = DEFAULT_LEAF_BYTES) -> bytes:
    """CONTROL: faithful port of the prior flatten_context (midpoint re-split).

    Reproduces the tax: midpoint splits that land inside a leaf reconstruct Leaf
    objects, each recomputing its polynomial hash (re-hashing ~Theta(N) bytes).
    """
    if node is None:
        return b""
    parts: list[bytes] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        length = _rope.rope_len(cur)
        if length <= leaf_bytes:
            parts.append(_rope.rope_to_bytes(cur))
        else:
            left, right = _rope.rope_split(cur, length // 2, h)
            if right is not None:
                stack.append(right)
            if left is not None:
                stack.append(left)
            if left is None and right is None:
                parts.append(_rope.rope_to_bytes(cur))
    return b"".join(parts)


def flatten_fixed(node) -> bytes:
    """FIX: in-order materialization (the library's rope_to_bytes).

    Reads each stored Leaf.data once and joins; 0 splits, 0 Leaf re-allocations,
    0 hash recomputation. O(N), alignment-independent.
    """
    return _rope.rope_to_bytes(node)
