#!/usr/bin/env python
"""
scripts/exp019_milestoneB_bench.py -- EXP-019 Milestone B bench (claim B3,
supplementary ecological validity).

Deterministic replay of real gpt-oss-120b Game-of-24 Tree-of-Thought traces
(data/canonical/tot_traces_gptoss120b_s{seed}.jsonl) through the SAME
branch/snapshot harness used in Milestone A (src/branch_bench.py), reusing
src/memory.py UNCHANGED. Milestone A's scripts/exp019_bench.py is frozen and
untouched.

Milestone B is supplementary: it CANNOT un-support B3, which is already SUPPORTED
via Milestone A. See LOGBOOK "EXP-019 -- Milestone B plan" for the locked design
and the verbatim CORROBORATION criterion.

PASS 1 (this cut): the importable replay / oracle / guard helpers exercised by the
red-first tests in tests/test_exp019_milestoneB.py. The orchestrator / worker /
timing / memory / verdict machinery is added in PASS 2 once these helpers are green.

Replay model (token space, 4-byte little-endian via src.competitive, identical to
Milestone A):
  * base context at the root  = prefix_tokens + puzzle_tokens
  * node v's context          = base + concatenation of thought_tokens along root..v
  * each non-root node         = ONE branch-creation off its parent's context
      - hashrope: hr_branch_create(ctx[parent], thought_tokens[v], h)
      - paged:    pa_branch_create(ctx[parent], thought_tokens[v])
The oracle is the token-list concatenation prefix + puzzle + path-thoughts
(src.branch_bench.expected_after_branch semantics); 4-byte-LE makes token-concat
== byte-concat exactly.
"""
from __future__ import annotations

import json
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.branch_bench import (
    hr_build,
    hr_branch_create,
    pa_build,
    pa_branch_create,
)
from src.memory import count_unique_nodes


# ---------------------------------------------------------------------------
# Trace loading / tree topology
# ---------------------------------------------------------------------------

def load_traces(path: str) -> list[dict]:
    """Load a ToT trace JSONL file: one puzzle-tree object per non-empty line."""
    trees: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trees.append(json.loads(line))
    return trees


def _node_index(tree: dict) -> dict:
    """Map node id -> node dict."""
    return {n["id"]: n for n in tree["nodes"]}


def count_expansions(tree: dict) -> int:
    """Branch-creation expansions in a tree = nodes - 1 (the root is the base)."""
    return len(tree["nodes"]) - 1


def path_thought_tokens(tree: dict, node_id: int) -> list[int]:
    """Concatenate thought_tokens along the path root..node_id (root first).
    The root contributes [] (its thought_tokens == [])."""
    index = _node_index(tree)
    chain: list[int] = []
    cur = node_id
    while cur is not None:
        chain.append(cur)
        cur = index[cur]["parent_id"]
    chain.reverse()  # root -> node
    out: list[int] = []
    for nid in chain:
        out.extend(index[nid]["thought_tokens"])
    return out


def final_beam_node_ids(tree: dict) -> list[int]:
    """The surviving beam at the deepest recorded depth (<= beam_width ids)."""
    beams = tree.get("beams") or {}
    if not beams:
        return []
    max_depth = max(int(k) for k in beams)
    return list(beams[str(max_depth)])


# ---------------------------------------------------------------------------
# Oracle (token-list ground truth)
# ---------------------------------------------------------------------------

def reconstruct_oracle(prefix_tokens: list[int], puzzle_tokens: list[int],
                       tree: dict, node_id: int) -> list[int]:
    """Ground-truth token sequence a node context must materialize to:
    prefix + puzzle + (root..node thoughts)."""
    return (list(prefix_tokens) + list(puzzle_tokens)
            + path_thought_tokens(tree, node_id))


# ---------------------------------------------------------------------------
# Tokenizer (gpt2; identical call form to the generator / Milestone A)
# ---------------------------------------------------------------------------

_TOKENIZER = None


def get_tokenizer():
    """Cached HF gpt2 AutoTokenizer (lazy import; matches the generator basis)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
    return _TOKENIZER


def tokenize_text(tok, text: str) -> list[int]:
    """gpt2-encode text with add_special_tokens=False (the token-seal basis)."""
    return tok(text, add_special_tokens=False)["input_ids"]


def puzzle_tokens_for(tok, tree: dict) -> list[int]:
    """gpt2 token ids of the puzzle string (the per-puzzle part of the base)."""
    return tokenize_text(tok, tree["puzzle"])


# ---------------------------------------------------------------------------
# Replay: build every node's context by walking the tree in id (BFS) order
# ---------------------------------------------------------------------------

def replay_tree_hr(prefix_tokens: list[int], puzzle_tokens: list[int],
                   tree: dict, h) -> dict:
    """hashrope replay. Returns {node_id: rope_node}. Root context = base
    (prefix + puzzle); each non-root v = hr_branch_create(ctx[parent], thoughts)."""
    base_tokens = list(prefix_tokens) + list(puzzle_tokens)
    root_rope, _h = hr_build(base_tokens, h)
    ctx: dict = {}
    for node in sorted(tree["nodes"], key=lambda n: n["id"]):
        pid = node["parent_id"]
        if pid is None:
            ctx[node["id"]] = root_rope
        else:
            ctx[node["id"]] = hr_branch_create(ctx[pid], node["thought_tokens"], h)
    return ctx


def replay_tree_pa(prefix_tokens: list[int], puzzle_tokens: list[int],
                   tree: dict, block_size: int) -> dict:
    """paged replay. Returns {node_id: PagedSequence}. One physical pool per tree
    (pa_build mints it); each non-root v = pa_branch_create(ctx[parent], thoughts)."""
    base_tokens = list(prefix_tokens) + list(puzzle_tokens)
    root_seq = pa_build(base_tokens, block_size)
    ctx: dict = {}
    for node in sorted(tree["nodes"], key=lambda n: n["id"]):
        pid = node["parent_id"]
        if pid is None:
            ctx[node["id"]] = root_seq
        else:
            ctx[node["id"]] = pa_branch_create(ctx[pid], node["thought_tokens"])
    return ctx


# ---------------------------------------------------------------------------
# Structural guards (deterministic; carry the asymptotics)
# ---------------------------------------------------------------------------

def hr_new_nodes(parent_rope, child_rope) -> int:
    """Nodes in child not shared with parent (the new O(log w) spine + leaf)."""
    u_parent = count_unique_nodes([parent_rope])["unique_count"]
    u_both = count_unique_nodes([parent_rope, child_rope])["unique_count"]
    return u_both - u_parent


def hr_guard_bound(parent_rope) -> int:
    """ceil(log2 max(w, 2)) + 3, w = parent leaf count (node.weight)."""
    w = parent_rope.weight if parent_rope is not None else 1
    return math.ceil(math.log2(max(w, 2))) + 3


def pa_fork_entries(parent_seq) -> int:
    """Block-table entries a fork of parent_seq touches (== parent.num_blocks)."""
    return parent_seq.num_blocks


def pa_guard_expected(parent_seq, block_size: int) -> int:
    """Expected touched entries: ceil(parent_len / B)."""
    return math.ceil(parent_seq.length / block_size)
