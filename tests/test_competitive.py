"""
Tests for src/competitive.py (EXP-017, claim B1).

TDD red phase: these tests are written BEFORE the implementation.
All must FAIL against the NotImplementedError stubs, then PASS once
implemented. Run from the paper-repo root:

    python -m pytest tests/test_competitive.py -v
"""
from __future__ import annotations

import math
import struct

import numpy as np
import pytest
import torch

from src.competitive import (
    bytes_to_tokens,
    build_token_rope,
    flat_lcp_np,
    hashrope_lcp_tokens,
    hashrope_lcp_tokens_counted,
    oracle_lcp,
    radix_lcp,
    token_lcp_from_byte_lcp,
    tokens_to_bytes,
)

# We import RadixCache directly here so we can build caches for radix_lcp tests
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from third_party.sglang_radix_cache.radix_cache import RadixCache


# ============================================================
# Section 1: Common-currency encoding (4 B LE)
# ============================================================


class TestTokensToBytes:
    """tokens_to_bytes: list[int] -> bytes (4-byte little-endian per token)."""

    def test_roundtrip_small(self):
        """tokens_to_bytes -> bytes_to_tokens is identity for small lists."""
        tokens = [0, 1, 255, 256, 50256, 2**31 - 1]
        assert bytes_to_tokens(tokens_to_bytes(tokens)) == tokens

    def test_roundtrip_empty(self):
        assert bytes_to_tokens(tokens_to_bytes([])) == []

    def test_known_encoding(self):
        """Verify exact LE byte layout for known values."""
        result = tokens_to_bytes([0x04030201])
        assert result == b"\x01\x02\x03\x04"

    def test_single_token_length(self):
        """Each token -> exactly 4 bytes."""
        for n in [1, 10, 100]:
            assert len(tokens_to_bytes(list(range(n)))) == 4 * n

    def test_encoding_is_little_endian(self):
        """Token 1 -> b'\\x01\\x00\\x00\\x00'."""
        assert tokens_to_bytes([1]) == b"\x01\x00\x00\x00"

    def test_zero_token(self):
        assert tokens_to_bytes([0]) == b"\x00\x00\x00\x00"


class TestBytesToTokens:
    """bytes_to_tokens: bytes -> list[int], inverse of tokens_to_bytes."""

    def test_rejects_non_divisible_length(self):
        with pytest.raises((ValueError, struct.error)):
            bytes_to_tokens(b"\x01\x02\x03")

    def test_empty(self):
        assert bytes_to_tokens(b"") == []


class TestTokenLcpFromByteLcp:
    """token_lcp_from_byte_lcp: floor(byte_lcp / 4)."""

    def test_exact_boundary(self):
        """byte_lcp = 4k -> token_lcp = k."""
        for k in [0, 1, 10, 100, 1000]:
            assert token_lcp_from_byte_lcp(4 * k) == k

    def test_within_token_group(self):
        """byte_lcp in (4k, 4k+4) -> token_lcp = k (floor division)."""
        for k in [0, 1, 5, 50]:
            for offset in [1, 2, 3]:
                assert token_lcp_from_byte_lcp(4 * k + offset) == k

    def test_encoding_alignment_property(self):
        """If two token streams first differ at token i, then:
        - bytes [0, 4i) are identical
        - at least one byte in [4i, 4i+4) differs
        So floor(byte_lcp / 4) == token_lcp exactly.

        Verify by construction.
        """
        for diverge_at in [0, 1, 5, 37]:
            shared = list(range(100, 100 + diverge_at))
            a = shared + [999] + list(range(200, 250))
            b = shared + [888] + list(range(300, 350))
            ba = tokens_to_bytes(a)
            bb = tokens_to_bytes(b)
            # Byte-level LCP
            byte_lcp = 0
            for x, y in zip(ba, bb):
                if x != y:
                    break
                byte_lcp += 1
            assert token_lcp_from_byte_lcp(byte_lcp) == diverge_at


# ============================================================
# Section 2: Oracle (ground truth)
# ============================================================


class TestOracleLcp:
    """oracle_lcp: plain Python loop, the correctness ground truth."""

    def test_identical(self):
        assert oracle_lcp([1, 2, 3], [1, 2, 3]) == 3

    def test_empty(self):
        assert oracle_lcp([], [1, 2]) == 0
        assert oracle_lcp([1, 2], []) == 0
        assert oracle_lcp([], []) == 0

    def test_diverge_at_0(self):
        assert oracle_lcp([1, 2, 3], [9, 2, 3]) == 0

    def test_diverge_at_mid(self):
        assert oracle_lcp([1, 2, 3, 4], [1, 2, 9, 4]) == 2

    def test_prefix_of_other(self):
        assert oracle_lcp([1, 2], [1, 2, 3, 4]) == 2
        assert oracle_lcp([1, 2, 3, 4], [1, 2]) == 2


# ============================================================
# Section 3: Flat numpy arm
# ============================================================


class TestFlatLcpNp:
    """flat_lcp_np: C-speed numpy comparison."""

    def test_identical(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        assert flat_lcp_np(a, a.copy()) == 3

    def test_empty(self):
        a = np.array([], dtype=np.int64)
        assert flat_lcp_np(a, a) == 0

    def test_diverge_mid(self):
        a = np.array([10, 20, 30, 40], dtype=np.int64)
        b = np.array([10, 20, 99, 40], dtype=np.int64)
        assert flat_lcp_np(a, b) == 2

    def test_different_lengths(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        assert flat_lcp_np(a, b) == 3


# ============================================================
# Section 4: Radix arm
# ============================================================


class TestRadixLcp:
    """radix_lcp: wrapper around RadixCache.match_prefix -> token count."""

    def test_full_match(self):
        cache = RadixCache(None, None, False)
        cached = [10, 20, 30, 40, 50]
        cache.insert(cached, torch.arange(len(cached)))
        assert radix_lcp(cache, cached) == 5

    def test_partial_match(self):
        cache = RadixCache(None, None, False)
        cached = [10, 20, 30, 40, 50]
        cache.insert(cached, torch.arange(len(cached)))
        query = [10, 20, 30, 99, 88]
        assert radix_lcp(cache, query) == 3

    def test_no_match(self):
        cache = RadixCache(None, None, False)
        cache.insert([10, 20, 30], torch.arange(3))
        assert radix_lcp(cache, [99, 88, 77]) == 0

    def test_empty_query(self):
        cache = RadixCache(None, None, False)
        cache.insert([1, 2, 3], torch.arange(3))
        assert radix_lcp(cache, []) == 0


# ============================================================
# Section 5: hashrope arm
# ============================================================


class TestHashropeLcpTokens:
    """hashrope_lcp_tokens: byte-LCP -> token-LCP via floor division."""

    def test_identical(self):
        tokens = [100, 200, 300]
        rope_a, h = build_token_rope(tokens)
        rope_b, _ = build_token_rope(tokens, h)
        assert hashrope_lcp_tokens(rope_a, rope_b, h) == 3

    def test_diverge_mid(self):
        a = [100, 200, 300, 400]
        b = [100, 200, 999, 400]
        rope_a, h = build_token_rope(a)
        rope_b, _ = build_token_rope(b, h)
        assert hashrope_lcp_tokens(rope_a, rope_b, h) == 2

    def test_empty(self):
        rope_a, h = build_token_rope([])
        rope_b, _ = build_token_rope([], h)
        assert hashrope_lcp_tokens(rope_a, rope_b, h) == 0


class TestHashropeLcpTokensCounted:
    """hashrope_lcp_tokens_counted: returns (token_lcp, num_hash_calls)."""

    def test_returns_tuple(self):
        tokens = [100, 200, 300]
        rope_a, h = build_token_rope(tokens)
        rope_b, _ = build_token_rope(tokens, h)
        result = hashrope_lcp_tokens_counted(rope_a, rope_b, h)
        assert isinstance(result, tuple)
        assert len(result) == 2
        lcp, calls = result
        assert lcp == 3

    def test_guard_step_count(self):
        """Hash calls <= 2 * ceil(log2(N_bytes)) per the EXP-005 guard."""
        tokens = list(range(1000))
        rope_a, h = build_token_rope(tokens)
        rope_b, _ = build_token_rope(tokens, h)
        lcp, calls = hashrope_lcp_tokens_counted(rope_a, rope_b, h)
        n_bytes = len(tokens) * 4
        max_calls = 2 * (math.ceil(math.log2(n_bytes)) + 1)
        assert calls <= max_calls, (
            f"hash calls {calls} exceeds guard 2*ceil(log2({n_bytes}))+2 = {max_calls}"
        )


# ============================================================
# Section 6: All-arms agreement (the core correctness tests)
# ============================================================


class TestAllArmsAgree:
    """All four arms must agree with oracle_lcp on every case."""

    @staticmethod
    def _check_all_arms(a: list[int], b: list[int]):
        """Run all arms and assert they agree with oracle."""
        expected = oracle_lcp(a, b)

        # radix arm: insert 'a' as cached, query with 'b'
        cache = RadixCache(None, None, False)
        if a:
            cache.insert(a, torch.arange(len(a)))
        assert radix_lcp(cache, b) == expected, f"radix: {radix_lcp(cache, b)} != {expected}"

        # hashrope arm
        rope_a, h = build_token_rope(a)
        rope_b, _ = build_token_rope(b, h)
        hr_result = hashrope_lcp_tokens(rope_a, rope_b, h)
        assert hr_result == expected, f"hashrope: {hr_result} != {expected}"

        # flat-np arm
        arr_a = np.array(a, dtype=np.int64) if a else np.array([], dtype=np.int64)
        arr_b = np.array(b, dtype=np.int64) if b else np.array([], dtype=np.int64)
        np_result = flat_lcp_np(arr_a, arr_b)
        assert np_result == expected, f"flat_np: {np_result} != {expected}"

    def test_diverge_at_0(self):
        self._check_all_arms([1, 2, 3, 4, 5], [9, 2, 3, 4, 5])

    def test_diverge_at_1(self):
        self._check_all_arms([1, 2, 3, 4, 5], [1, 9, 3, 4, 5])

    def test_diverge_at_mid(self):
        self._check_all_arms([1, 2, 3, 4, 5], [1, 2, 9, 4, 5])

    def test_full_prefix(self):
        self._check_all_arms([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])

    def test_prefix_of_other(self):
        self._check_all_arms([1, 2], [1, 2, 3, 4, 5])
        self._check_all_arms([1, 2, 3, 4, 5], [1, 2])

    def test_single_token_match(self):
        self._check_all_arms([42], [42])

    def test_single_token_mismatch(self):
        self._check_all_arms([42], [99])

    def test_longer_sequence(self):
        """100-token sequence diverging at token 73."""
        shared = list(range(1000, 1073))
        a = shared + [9999] + list(range(2000, 2026))
        b = shared + [8888] + list(range(3000, 3026))
        self._check_all_arms(a, b)

    def test_empty_both(self):
        """Both empty -> LCP = 0."""
        # Note: radix with empty cache + empty query -> 0
        expected = oracle_lcp([], [])
        assert expected == 0
        # flat-np
        a = np.array([], dtype=np.int64)
        assert flat_lcp_np(a, a) == 0
        # hashrope
        rope_a, h = build_token_rope([])
        rope_b, _ = build_token_rope([], h)
        assert hashrope_lcp_tokens(rope_a, rope_b, h) == 0

    def test_empty_one(self):
        """One empty, one non-empty -> LCP = 0."""
        self._check_all_arms([], [1, 2, 3])
