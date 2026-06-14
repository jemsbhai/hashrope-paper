"""
EXP-019 (claim B3) -- branch/snapshot cross-arm correctness GATE.

The crux of B3's credibility: hashrope and the PagedAttention block-table COW
baseline must produce BYTE-IDENTICAL branched contexts on identical workloads
(criterion (i), HARD). This file pins that, plus the hashrope-side O(log w)
per-branch node guard (criterion (ii)).

Both arms operate in token space; the oracle is a plain Python token list. The
hashrope arm round-trips through the 4-byte-LE rope encoding (EXP-017 convention)
and back; the paged arm round-trips through the block table.

Red-first contract: RED at the branch_bench stub (NotImplementedError), GREEN
once src/branch_bench.py is implemented. Deterministic, offline.
"""
import math
import random

import pytest

from src.flatten import make_hash
from src.memory import count_unique_nodes
from src.branch_bench import (
    expected_after_branch,
    hr_build,
    hr_fork,
    hr_append_token,
    hr_append_step,
    hr_branch_create,
    hr_materialize,
    pa_build,
    pa_fork,
    pa_append_token,
    pa_append_step,
    pa_branch_create,
    pa_materialize,
)

SEED = 42
VOCAB = 50257                      # gpt2 vocab (plausible token-id range)
BLOCK_SIZES = [8, 16, 32, 64]      # the EXP-019 block-size sweep
TOKENS_PER_LEAF = 1024             # 4096-byte leaf / 4 bytes-per-token


def _tokens(n: int, seed: int = SEED) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(0, VOCAB) for _ in range(n)]


# ---------------------------------------------------------------------------
# hashrope arm round-trip (token <-> 4B-LE rope <-> token)
# ---------------------------------------------------------------------------

class TestHashropeArmRoundtrip:
    @pytest.mark.parametrize("n", [0, 1, 50, 1024, 3000])
    def test_build_materialize(self, n):
        toks = _tokens(n)
        rope, _h = hr_build(toks)
        assert hr_materialize(rope) == toks

    @pytest.mark.parametrize("n", [1, 50, 3000])
    @pytest.mark.parametrize("step_len", [1, 32])
    def test_branch_roundtrip(self, n, step_len):
        base = _tokens(n)
        step = _tokens(step_len, seed=SEED + 1)
        h = make_hash()
        rope, _h = hr_build(base, h)
        branch = hr_branch_create(rope, step, h)
        assert hr_materialize(branch) == expected_after_branch(base, step)
        # immutability: the base context is untouched by the branch
        assert hr_materialize(rope) == base

    def test_append_token(self):
        base = _tokens(50)
        h = make_hash()
        rope, _h = hr_build(base, h)
        r = hr_append_token(rope, 7, h)
        assert hr_materialize(r) == base + [7]
        assert hr_materialize(rope) == base          # parent untouched


# ---------------------------------------------------------------------------
# hashrope per-branch node guard (criterion ii, hashrope side)
# ---------------------------------------------------------------------------

class TestHashropeNodeGuard:
    def test_branches_share_nodes(self):
        """B branches off a multi-leaf base share interior nodes; new nodes per
        branch are O(log w), not O(base)."""
        B = 10
        base = _tokens(5 * TOKENS_PER_LEAF)          # ~5 leaves
        h = make_hash()
        rope, _h = hr_build(base, h)
        branches = [hr_branch_create(rope, [1000 + i], h) for i in range(B)]

        unique = count_unique_nodes([rope] + branches)["unique_count"]
        base_unique = count_unique_nodes([rope])["unique_count"]

        # sharing: far fewer than B independent copies
        assert unique < B * base_unique
        # O(log w) new nodes per branch (loose, weight-based bound, as in EXP-004)
        w = rope.weight
        log_w = math.ceil(math.log2(max(w, 2)))
        assert unique <= base_unique + B * (log_w + 3)


# ---------------------------------------------------------------------------
# CROSS-ARM byte-identity (criterion i, HARD) -- the crux
# ---------------------------------------------------------------------------

class TestCrossArmByteIdentity:
    @pytest.mark.parametrize("B", BLOCK_SIZES)
    @pytest.mark.parametrize("n", [50, 2000])
    @pytest.mark.parametrize("step_len", [1, 32])
    def test_branch_create_matches_oracle(self, B, n, step_len):
        base = _tokens(n)
        step = _tokens(step_len, seed=SEED + 2)
        oracle = expected_after_branch(base, step)

        # hashrope arm
        h = make_hash()
        rope, _h = hr_build(base, h)
        hr_out = hr_materialize(hr_branch_create(rope, step, h))

        # paged arm
        seq = pa_build(base, B)
        pa_out = pa_materialize(pa_branch_create(seq, step))

        assert hr_out == oracle, "hashrope branch != oracle"
        assert pa_out == oracle, "paged branch != oracle"
        # hence hr_out == pa_out (cross-arm byte-identity)

    @pytest.mark.parametrize("B", BLOCK_SIZES)
    def test_base_unchanged_after_branch_both_arms(self, B):
        base = _tokens(2000)
        step = _tokens(32, seed=SEED + 3)

        h = make_hash()
        rope, _h = hr_build(base, h)
        hr_branch_create(rope, step, h)
        assert hr_materialize(rope) == base

        seq = pa_build(base, B)
        pa_branch_create(seq, step)              # branch off a fork, not seq itself
        assert pa_materialize(seq) == base


class TestCrossArmParallelSampling:
    @pytest.mark.parametrize("B", BLOCK_SIZES)
    def test_two_divergent_branches(self, B):
        """Fig-8 cross-arm: two branches off one base diverge identically on both
        arms."""
        base = _tokens(40)
        step_a = [111, 112, 113]
        step_b = [221, 222, 223]
        oracle_a = expected_after_branch(base, step_a)
        oracle_b = expected_after_branch(base, step_b)

        h = make_hash()
        rope, _h = hr_build(base, h)
        hr_a = hr_materialize(hr_branch_create(rope, step_a, h))
        hr_b = hr_materialize(hr_branch_create(rope, step_b, h))

        seq = pa_build(base, B)
        pa_a = pa_materialize(pa_branch_create(seq, step_a))
        pa_b = pa_materialize(pa_branch_create(seq, step_b))

        assert hr_a == pa_a == oracle_a
        assert hr_b == pa_b == oracle_b


# ---------------------------------------------------------------------------
# Randomized cross-arm tree (strong correctness): hashrope == paged == oracle
# ---------------------------------------------------------------------------

class TestCrossArmRandomTree:
    @pytest.mark.parametrize("B", [8, 16])
    def test_random_fork_append_tree(self, B):
        """Drive the SAME random sequence of fork/append-token ops on the hashrope
        arm, the paged arm, and an oracle; all three must materialize identically."""
        rng = random.Random(SEED + B)
        h = make_hash()

        base = _tokens(rng.randrange(1, 60))
        hr_root, _h = hr_build(base, h)
        hr_seqs = [hr_root]
        pa_seqs = [pa_build(base, B)]
        oracles = [list(base)]

        for _ in range(150):
            i = rng.randrange(len(hr_seqs))
            if rng.random() < 0.4:                          # fork sequence i
                hr_seqs.append(hr_fork(hr_seqs[i]))
                pa_seqs.append(pa_fork(pa_seqs[i]))
                oracles.append(list(oracles[i]))
            else:                                            # append a token to i
                tok = rng.randrange(0, VOCAB)
                hr_seqs[i] = hr_append_token(hr_seqs[i], tok, h)
                pa_append_token(pa_seqs[i], tok)
                oracles[i].append(tok)

        for hr_s, pa_s, o in zip(hr_seqs, pa_seqs, oracles):
            assert hr_materialize(hr_s) == o
            assert pa_materialize(pa_s) == o
