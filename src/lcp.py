"""
EXP-005: Longest Common Prefix via prefix-hash binary search (claim T3).

lcp_brute
    Byte-by-byte comparison oracle. O(LCP_length). Used as correctness
    ground truth, not for performance claims.

lcp_hash
    Binary search on prefix length using rope_substr_hash (Theorem 9).
    At each step, compare H(A[0..L)) vs H(B[0..L)). O(log N) steps ×
    O(log w) per hash query = O(log² N). Returns the length of the
    longest common prefix.

lcp_hash_counted
    Instrumented variant returning (lcp_length, num_hash_calls) so the
    step-count guard can verify ≤ 2 * (ceil(log2(min(len_a, len_b))) + 1).

Reuses build_fat_leaf_rope / make_hash from src.flatten.
"""
from __future__ import annotations

import math

from hashrope import PolynomialHash, rope_substr_hash, rope_len
from hashrope.rope import Node


def lcp_brute(a: bytes, b: bytes) -> int:
    """Byte-by-byte LCP oracle. O(min(len(a), len(b))) worst case."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def lcp_hash(rope_a: Node, rope_b: Node, h: PolynomialHash) -> int:
    """LCP via binary search on prefix hashes. O(log² N)."""
    lcp, _ = _lcp_binary_search(rope_a, rope_b, h, count=False)
    return lcp


def lcp_hash_counted(
    rope_a: Node, rope_b: Node, h: PolynomialHash
) -> tuple[int, int]:
    """Instrumented LCP returning (lcp_length, num_hash_calls)."""
    return _lcp_binary_search(rope_a, rope_b, h, count=True)


def _lcp_binary_search(
    rope_a: Node,
    rope_b: Node,
    h: PolynomialHash,
    *,
    count: bool,
) -> tuple[int, int]:
    """Core binary search. Returns (lcp_length, num_hash_calls).

    Binary search on L ∈ [0, N] where N = min(len_a, len_b).
    Invariant: lo is the largest confirmed-equal prefix length,
    hi is the smallest confirmed-different length (or N+1 if none yet).
    """
    len_a = rope_len(rope_a) if rope_a is not None else 0
    len_b = rope_len(rope_b) if rope_b is not None else 0
    n = min(len_a, len_b)
    if n == 0:
        return 0, 0

    num_calls = 0
    lo = 0   # last confirmed-equal prefix length (0 = nothing confirmed)
    hi = n   # search space upper bound (inclusive — could be all equal)

    while lo < hi:
        mid = lo + (hi - lo + 1) // 2  # bias high to make progress
        h_a = rope_substr_hash(rope_a, 0, mid, h)
        h_b = rope_substr_hash(rope_b, 0, mid, h)
        num_calls += 2
        if h_a == h_b:
            lo = mid      # prefixes match up to mid
        else:
            hi = mid - 1  # divergence before mid

    return lo, num_calls
