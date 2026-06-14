"""
src/branch_bench.py

EXP-019 (claim B3) branch/snapshot harness: the hashrope arm + uniform wrappers
over the PagedAttention COW baseline (src/paged_attention_cow.py), plus the
token-level oracle. Both arms operate in TOKEN space; the hashrope arm encodes
tokens as fixed-width 4-byte little-endian (the EXP-017 convention, reusing
src.competitive.tokens_to_bytes / bytes_to_tokens) so the comparison is
apples-to-apples with EXP-017 and with the byte-level rope machinery (EXP-002/004).

The two arms expose the same atomic operations so the bench can time each
decomposition on identical workloads and report all of them:

  * bare structural fork
      - hashrope: share the immutable root -- O(1); divergence cost is DEFERRED
        to the first append (which creates the O(log w) spine). hr_fork returns
        the same node; appends never mutate it, so the parent is untouched.
      - paged:    copy the per-sequence block table + incref each shared physical
        block -- O(ceil(N/B)) (sec 4.4).
  * append one token
      - hashrope: rope_concat(rope, Leaf(one token)) -> new root -- O(log w).
      - paged:    append_token -- O(1) amortized (in-place / new block / one-block COW).
  * branch-create (fork + append a whole step) -- the operationally meaningful unit:
      - hashrope: rope_concat(base, Leaf(step bytes)) -- O(log w).
      - paged:    fork + append the step tokens.
  * materialize -> token list (for byte/token-identity vs the oracle).

Reuses:
    src.competitive.{tokens_to_bytes, bytes_to_tokens, build_token_rope}
    src.flatten.{build_fat_leaf_rope, make_hash}
    src.memory.count_unique_nodes
    src.paged_attention_cow (the baseline arm)
"""
from __future__ import annotations

import hashrope.rope as _rope
from hashrope import PolynomialHash
from hashrope.rope import Node

from src.competitive import tokens_to_bytes, bytes_to_tokens, build_token_rope
import src.paged_attention_cow as cow


# ---------------------------------------------------------------------------
# Oracle (token-level ground truth)
# ---------------------------------------------------------------------------

def expected_after_branch(base_tokens: list[int], step_tokens: list[int]) -> list[int]:
    """The token sequence a branch must materialize to: base followed by step."""
    return list(base_tokens) + list(step_tokens)


# ---------------------------------------------------------------------------
# hashrope arm (token space, 4-byte-LE encoding; immutable structural sharing)
# ---------------------------------------------------------------------------

def hr_build(tokens: list[int], h: PolynomialHash | None = None):
    """Build a hashrope base context from token ids (4 B LE). Returns
    (rope_node | None, hash_obj)."""
    return build_token_rope(tokens, h)


def hr_fork(rope: Node | None) -> Node | None:
    """Bare structural fork: share the immutable root (O(1)). Returns the same
    node; subsequent appends create new nodes and leave this one untouched."""
    return rope


def hr_append_step(rope: Node | None, step_tokens: list[int], h: PolynomialHash) -> Node:
    """Append a whole reasoning step as a single leaf -> new root. O(log w)."""
    if not step_tokens:
        return rope
    step_leaf = _rope.Leaf(tokens_to_bytes(step_tokens), h)
    if rope is None:
        return step_leaf
    return _rope.rope_concat(rope, step_leaf, h)


def hr_append_token(rope: Node | None, token: int, h: PolynomialHash) -> Node:
    """Append one token (4 B LE leaf) -> new root. O(log w)."""
    return hr_append_step(rope, [token], h)


def hr_branch_create(base: Node | None, step_tokens: list[int], h: PolynomialHash) -> Node:
    """Create a diverged branch = fork + append the step. O(log w)."""
    return hr_append_step(hr_fork(base), step_tokens, h)


def hr_materialize(rope: Node | None) -> list[int]:
    """Materialize a hashrope context back to its token-id list ([] for None)."""
    if rope is None:
        return []
    return bytes_to_tokens(_rope.rope_to_bytes(rope))


# ---------------------------------------------------------------------------
# PagedAttention COW arm (uniform wrappers over src.paged_attention_cow)
# ---------------------------------------------------------------------------

def pa_build(tokens: list[int], block_size: int) -> "cow.PagedSequence":
    """Build a paged base context (fresh pool of block_size-token blocks). The
    pool is reachable as the returned sequence's `.pool`."""
    pool = cow.PhysicalBlockPool(block_size)
    return cow.build_sequence(pool, tokens)


def pa_fork(seq: "cow.PagedSequence") -> "cow.PagedSequence":
    """Bare structural fork: copy the block table + incref shared blocks. O(N/B)."""
    return cow.fork(seq)


def pa_append_token(seq: "cow.PagedSequence", token: int) -> dict:
    """Append one token (in-place / new block / one-block COW). Returns op-stats."""
    return cow.append_token(seq, token)


def pa_append_step(seq: "cow.PagedSequence", step_tokens: list[int]) -> None:
    """Append a whole step, one token at a time."""
    for tok in step_tokens:
        cow.append_token(seq, tok)


def pa_branch_create(base_seq: "cow.PagedSequence", step_tokens: list[int]) -> "cow.PagedSequence":
    """Create a diverged branch = fork + append the step. O(N/B) + O(step)."""
    child = cow.fork(base_seq)
    pa_append_step(child, step_tokens)
    return child


def pa_materialize(seq: "cow.PagedSequence") -> list[int]:
    """Materialize a paged context back to its token-id list."""
    return cow.materialize(seq)
