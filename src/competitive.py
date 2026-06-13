"""
EXP-017: Competitive prefix-identification primitives (claim B1).

Common-currency layer (4 B LE token encoding) and arm wrappers for the
competitive comparison: hashrope LCP vs SGLang RadixCache vs numpy flat scan.

tokens_to_bytes / bytes_to_tokens
    Fixed-width 4-byte little-endian encoding. Token-LCP = floor(byte-LCP / 4),
    exact: if two streams first differ at token i, bytes < 4i are identical and
    at least one byte in [4i, 4i+4) differs.

build_token_rope
    Wraps build_fat_leaf_rope + make_hash on the 4 B LE byte stream.

radix_lcp
    Wrapper around RadixCache.match_prefix returning the matched token count.

hashrope_lcp_tokens / hashrope_lcp_tokens_counted
    Byte-level LCP via prefix-hash binary search (EXP-005 machinery),
    converted to token-level LCP via floor division.

flat_lcp_np
    C-speed numpy array comparison (floor reference, not a deployable structure).

oracle_lcp
    Plain Python loop -- the token-level correctness ground truth.

Reuses:
    src.flatten.build_fat_leaf_rope, src.flatten.make_hash
    src.lcp.lcp_hash, src.lcp.lcp_hash_counted
"""
from __future__ import annotations

import struct

import numpy as np

from hashrope import PolynomialHash
from hashrope.rope import Node

from src.flatten import build_fat_leaf_rope, make_hash
from src.lcp import lcp_hash, lcp_hash_counted


# ---------------------------------------------------------------------------
# Common-currency encoding: 4-byte little-endian per token
# ---------------------------------------------------------------------------

def tokens_to_bytes(tokens: list[int]) -> bytes:
    """Convert a list of token IDs to fixed-width 4-byte LE bytes."""
    return struct.pack(f"<{len(tokens)}I", *tokens)


def bytes_to_tokens(data: bytes) -> list[int]:
    """Inverse of tokens_to_bytes. len(data) must be divisible by 4."""
    if len(data) == 0:
        return []
    if len(data) % 4 != 0:
        raise ValueError(
            f"data length {len(data)} is not divisible by 4"
        )
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def token_lcp_from_byte_lcp(byte_lcp: int) -> int:
    """Convert a byte-level LCP to a token-level LCP: floor(byte_lcp / 4)."""
    return byte_lcp // 4


# ---------------------------------------------------------------------------
# Arm builders
# ---------------------------------------------------------------------------

def build_token_rope(tokens: list[int], h: PolynomialHash | None = None):
    """Build a hashrope from token IDs (4 B LE encoding).

    Returns (rope_node, hash_obj). rope_node is None for empty input.
    """
    if h is None:
        h = make_hash()
    data = tokens_to_bytes(tokens)
    if len(data) == 0:
        return None, h
    rope = build_fat_leaf_rope(data, h)
    return rope, h


# ---------------------------------------------------------------------------
# Arm query wrappers
# ---------------------------------------------------------------------------

def radix_lcp(cache, key: list[int]) -> int:
    """Query RadixCache.match_prefix and return the matched token count."""
    value, _last_node = cache.match_prefix(key)
    return len(value)


def hashrope_lcp_tokens(
    rope_a: Node | None, rope_b: Node | None, h: PolynomialHash
) -> int:
    """hashrope LCP on token-encoded ropes, returning token-level LCP."""
    if rope_a is None or rope_b is None:
        return 0
    byte_lcp = lcp_hash(rope_a, rope_b, h)
    return token_lcp_from_byte_lcp(byte_lcp)


def hashrope_lcp_tokens_counted(
    rope_a: Node | None, rope_b: Node | None, h: PolynomialHash
) -> tuple[int, int]:
    """Instrumented hashrope LCP returning (token_lcp, num_hash_calls)."""
    if rope_a is None or rope_b is None:
        return 0, 0
    byte_lcp, num_calls = lcp_hash_counted(rope_a, rope_b, h)
    return token_lcp_from_byte_lcp(byte_lcp), num_calls


def flat_lcp_np(arr_a: np.ndarray, arr_b: np.ndarray) -> int:
    """C-speed numpy flat array LCP on int32/int64 token arrays."""
    n = min(len(arr_a), len(arr_b))
    if n == 0:
        return 0
    neq = arr_a[:n] != arr_b[:n]
    idx = np.argmax(neq)
    # argmax returns 0 both when the first element differs and when all equal
    if idx == 0 and not neq[0]:
        return n  # all equal up to min length
    return int(idx)


def oracle_lcp(a: list[int], b: list[int]) -> int:
    """Plain Python loop oracle for token-level LCP. The ground truth."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n
