"""
tests/test_exp019_milestoneB.py

EXP-019 Milestone B (supplementary ecological validity for claim B3): red-first
tests for the deterministic replay of real gpt-oss-120b Game-of-24 Tree-of-Thought
traces through the branch/snapshot harness (scripts/exp019_milestoneB_bench.py).

These tests are authored BEFORE the bench implementation and must FAIL first:
importing the not-yet-written bench module errors at collection time, so every
test below is red. After the bench is implemented they must all pass. The heavy
P=256k correctness is proven in the n=9 confirmatory run, NOT here; unit tests use
small synthetic prefixes (P in {0, 64}) over the REAL trace topology + REAL
thought-token streams so they stay fast while exercising the real shapes.

Mapping to the locked LOGBOOK CORROBORATION criterion:
  (i)   HARD byte-identity  -> test_byte_identity_hr / _pa / test_cross_arm_agreement
  (ii)  structural guards    -> test_hr_guard / test_pa_guard
  basis integrity            -> test_token_seal_revalidation   (hardening #1)
  determinism (var ~ 0)      -> test_replay_determinism
  beam memory (structural)   -> test_beam_memory_sharing
  n=9 is real                -> test_n9_distinctness
  workload integrity         -> test_expansion_count
  oracle logic               -> test_oracle_path_walk

Note: replay correctness holds for ANY prefix/puzzle token list, so the unit
tests use synthetic prefix/puzzle tokens (real values would only change constants,
not identity). The REAL corpus prefix + real gpt2 puzzle tokenization are used in
the confirmatory bench, where ecological validity matters; the tokenizer/seal path
is validated separately in test_token_seal_revalidation.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# RED-FIRST: this import fails (ModuleNotFoundError) until the bench is authored,
# which makes the whole module error at collection -> every test below is red.
import exp019_milestoneB_bench as mb

# Reused UNCHANGED; imported AFTER mb so mb's absence is the first failure.
from src.branch_bench import hr_materialize, pa_materialize
from src.memory import count_unique_nodes
from src.flatten import make_hash


# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

DATA = _REPO / "data" / "canonical"
SEEDS = (42, 43, 44)
RANKS = list(range(901, 926))            # 901..925 inclusive (25 puzzles/seed)
BLOCKS = (8, 16, 32, 64)
SMALL_PS = (0, 64)                       # synthetic prefix sizes for unit speed
REF_BLOCK = 16

# Validated workload totals (continuation prompt; 75/75 puzzles).
EXPECTED_TOTAL_NODES = 7943
EXPECTED_TOTAL_EXPANSIONS = 7868         # sum(nodes - 1) over the 75 trees
EXPECTED_DEPTH_HIST = {1: 1979, 2: 4676, 3: 1213}   # expansions by child depth


def trace_path(seed: int) -> str:
    return str(DATA / f"tot_traces_gptoss120b_s{seed}.jsonl")


def _synth_prefix(p: int) -> list[int]:
    # Distinct-ish synthetic prefix tokens; any valid u32 list works for identity.
    return list(range(1000, 1000 + p))


def _synth_puzzle() -> list[int]:
    return [2, 3, 5, 7]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def traces_by_seed():
    return {s: mb.load_traces(trace_path(s)) for s in SEEDS}


@pytest.fixture(scope="module")
def tokenizer():
    return mb.get_tokenizer()


# ---------------------------------------------------------------------------
# oracle logic
# ---------------------------------------------------------------------------

def test_oracle_path_walk():
    """reconstruct_oracle = prefix + puzzle + (root..v thoughts), on a hand tree."""
    tree = {
        "puzzle_id": 999,
        "puzzle": "1 2 3 4",
        "nodes": [
            {"id": 0, "parent_id": None, "depth": 0, "thought_text": "",
             "thought_tokens": []},
            {"id": 1, "parent_id": 0, "depth": 1, "thought_text": "a",
             "thought_tokens": [10, 11]},
            {"id": 2, "parent_id": 1, "depth": 2, "thought_text": "b",
             "thought_tokens": [20]},
            {"id": 3, "parent_id": 0, "depth": 1, "thought_text": "c",
             "thought_tokens": [30, 31, 32]},
        ],
        "beams": {"1": [1, 3], "2": [2]},
        "beam_width": 5,
    }
    prefix = [100, 101]
    puzzle = [1, 2, 3, 4]

    assert mb.path_thought_tokens(tree, 0) == []
    assert mb.path_thought_tokens(tree, 1) == [10, 11]
    assert mb.path_thought_tokens(tree, 2) == [10, 11, 20]
    assert mb.path_thought_tokens(tree, 3) == [30, 31, 32]

    assert mb.reconstruct_oracle(prefix, puzzle, tree, 0) == prefix + puzzle
    assert mb.reconstruct_oracle(prefix, puzzle, tree, 2) == \
        prefix + puzzle + [10, 11, 20]
    assert mb.reconstruct_oracle(prefix, puzzle, tree, 3) == \
        prefix + puzzle + [30, 31, 32]


# ---------------------------------------------------------------------------
# basis integrity (hardening #1)
# ---------------------------------------------------------------------------

def test_token_seal_revalidation(traces_by_seed, tokenizer):
    """Re-tokenizing stored thought_text with the run-env gpt2 reproduces the
    stored thought_tokens (the seal), proving the tokenizer basis at replay time
    matches generation -- so live puzzle tokenization is on the identical basis."""
    checked = 0
    for tree in traces_by_seed[42][:3]:
        for node in tree["nodes"]:
            if node["parent_id"] is None:
                continue
            text = node["thought_text"]
            if text == "":
                continue
            assert mb.tokenize_text(tokenizer, text) == node["thought_tokens"], (
                f"seal mismatch puzzle {tree['puzzle_id']} node {node['id']}"
            )
            checked += 1
    assert checked > 0, "no non-root thoughts checked"


# ---------------------------------------------------------------------------
# (i) HARD byte-identity
# ---------------------------------------------------------------------------

def test_byte_identity_hr(traces_by_seed):
    """hashrope arm: every reconstructed node context == oracle, 0 mismatches."""
    mism = 0
    for tree in traces_by_seed[42]:
        pz = _synth_puzzle()
        for p in SMALL_PS:
            prefix = _synth_prefix(p)
            h = make_hash()
            ctx = mb.replay_tree_hr(prefix, pz, tree, h)
            for node in tree["nodes"]:
                got = hr_materialize(ctx[node["id"]])
                exp = mb.reconstruct_oracle(prefix, pz, tree, node["id"])
                mism += int(got != exp)
    assert mism == 0


def test_byte_identity_pa(traces_by_seed):
    """paged arm: every node context == oracle for all block sizes, 0 mismatches."""
    mism = 0
    for tree in traces_by_seed[42]:
        pz = _synth_puzzle()
        for p in SMALL_PS:
            prefix = _synth_prefix(p)
            for bs in BLOCKS:
                ctx = mb.replay_tree_pa(prefix, pz, tree, bs)
                for node in tree["nodes"]:
                    got = pa_materialize(ctx[node["id"]])
                    exp = mb.reconstruct_oracle(prefix, pz, tree, node["id"])
                    mism += int(got != exp)
    assert mism == 0


def test_cross_arm_agreement(traces_by_seed):
    """hashrope and paged reconstruct byte-identical contexts for every node."""
    diff = 0
    for tree in traces_by_seed[42]:
        pz = _synth_puzzle()
        for p in SMALL_PS:
            prefix = _synth_prefix(p)
            h = make_hash()
            hr_ctx = mb.replay_tree_hr(prefix, pz, tree, h)
            pa_ctx = mb.replay_tree_pa(prefix, pz, tree, REF_BLOCK)
            for node in tree["nodes"]:
                diff += int(hr_materialize(hr_ctx[node["id"]])
                            != pa_materialize(pa_ctx[node["id"]]))
    assert diff == 0


# ---------------------------------------------------------------------------
# (ii) structural guards
# ---------------------------------------------------------------------------

def test_hr_guard(traces_by_seed):
    """Per expansion, hashrope new nodes <= ceil(log2 max(parent.weight,2)) + 3."""
    for tree in traces_by_seed[42]:
        pz = _synth_puzzle()
        for p in SMALL_PS:
            prefix = _synth_prefix(p)
            h = make_hash()
            ctx = mb.replay_tree_hr(prefix, pz, tree, h)
            for node in tree["nodes"]:
                pid = node["parent_id"]
                if pid is None:
                    continue
                new_nodes = mb.hr_new_nodes(ctx[pid], ctx[node["id"]])
                bound = mb.hr_guard_bound(ctx[pid])
                assert new_nodes <= bound, (
                    f"hr guard puzzle {tree['puzzle_id']} node {node['id']}: "
                    f"{new_nodes} > {bound}"
                )


def test_pa_guard(traces_by_seed):
    """Per expansion, paged fork touches exactly ceil(parent_len / B) entries."""
    for tree in traces_by_seed[42]:
        pz = _synth_puzzle()
        for p in SMALL_PS:
            prefix = _synth_prefix(p)
            for bs in BLOCKS:
                ctx = mb.replay_tree_pa(prefix, pz, tree, bs)
                for node in tree["nodes"]:
                    pid = node["parent_id"]
                    if pid is None:
                        continue
                    parent = ctx[pid]
                    assert mb.pa_fork_entries(parent) == mb.pa_guard_expected(parent, bs), (
                        f"pa guard puzzle {tree['puzzle_id']} node {node['id']} B={bs}"
                    )


# ---------------------------------------------------------------------------
# determinism: invocation variance ~ 0 on structure
# ---------------------------------------------------------------------------

def test_replay_determinism(traces_by_seed):
    """Two independent replays of one tree yield byte-identical materializations
    and identical guard counts (substantiates ~0 structure/memory variance)."""
    tree = traces_by_seed[42][0]
    pz = _synth_puzzle()
    prefix = _synth_prefix(64)

    h1 = make_hash()
    h2 = make_hash()
    a = mb.replay_tree_hr(prefix, pz, tree, h1)
    b = mb.replay_tree_hr(prefix, pz, tree, h2)
    for node in tree["nodes"]:
        assert hr_materialize(a[node["id"]]) == hr_materialize(b[node["id"]])
        pid = node["parent_id"]
        if pid is not None:
            assert mb.hr_new_nodes(a[pid], a[node["id"]]) == \
                mb.hr_new_nodes(b[pid], b[node["id"]])

    c = mb.replay_tree_pa(prefix, pz, tree, REF_BLOCK)
    d = mb.replay_tree_pa(prefix, pz, tree, REF_BLOCK)
    for node in tree["nodes"]:
        assert pa_materialize(c[node["id"]]) == pa_materialize(d[node["id"]])
        assert c[node["id"]].num_blocks == d[node["id"]].num_blocks


# ---------------------------------------------------------------------------
# beam memory (structural)
# ---------------------------------------------------------------------------

def test_beam_memory_sharing(traces_by_seed):
    """final_beam_node_ids returns <= beam_width survivors; the hashrope beam
    contexts share the common base (structural sharing), the mechanism behind the
    branching-memory compression measured in the confirmatory run."""
    tree = next((t for t in traces_by_seed[42]
                 if len(mb.final_beam_node_ids(t)) >= 2), None)
    assert tree is not None, "no s42 tree has >=2 final-beam survivors"

    beam_ids = mb.final_beam_node_ids(tree)
    bw = tree.get("beam_width", 5)
    assert 1 <= len(beam_ids) <= bw
    node_ids = {n["id"] for n in tree["nodes"]}
    assert all(i in node_ids for i in beam_ids)

    prefix = _synth_prefix(256)
    pz = _synth_puzzle()
    h = make_hash()
    ctx = mb.replay_tree_hr(prefix, pz, tree, h)
    beam_ctx = [ctx[i] for i in beam_ids]

    combined = count_unique_nodes(beam_ctx)["unique_count"]
    individual_sum = sum(count_unique_nodes([c])["unique_count"] for c in beam_ctx)
    assert combined < individual_sum, "expected structural sharing across beam"


# ---------------------------------------------------------------------------
# n=9 is real
# ---------------------------------------------------------------------------

def _tree_signature(tree) -> tuple:
    nodes = sorted(tree["nodes"], key=lambda n: n["id"])
    return tuple((n["id"], n["parent_id"], tuple(n["thought_tokens"])) for n in nodes)


def test_n9_distinctness(traces_by_seed):
    """Each rank 901..925 yields three pairwise-distinct trees across the seeds,
    so the n=9 error model (3 seeds x 3 invocations) is real, not triple-counted."""
    by_rank = {r: {} for r in RANKS}
    for s in SEEDS:
        for tree in traces_by_seed[s]:
            by_rank[tree["puzzle_id"]][s] = tree

    for r in RANKS:
        assert set(by_rank[r].keys()) == set(SEEDS), f"rank {r} missing a seed"
        sigs = [_tree_signature(by_rank[r][s]) for s in SEEDS]
        assert len(set(sigs)) == 3, f"rank {r}: trees not all distinct across seeds"


# ---------------------------------------------------------------------------
# workload integrity
# ---------------------------------------------------------------------------

def test_expansion_count(traces_by_seed):
    """Per-seed and total node/expansion counts and the per-depth expansion
    histogram match the validated workload (7943 nodes, 7868 expansions)."""
    total_nodes = 0
    total_exp = 0
    depth_hist = Counter()
    for s in SEEDS:
        trees = traces_by_seed[s]
        ranks = sorted(t["puzzle_id"] for t in trees)
        assert ranks == RANKS, f"seed {s} ranks {ranks[:3]}... != 901..925"
        for tree in trees:
            total_nodes += len(tree["nodes"])
            total_exp += mb.count_expansions(tree)
            for node in tree["nodes"]:
                if node["parent_id"] is not None:
                    depth_hist[node["depth"]] += 1

    assert total_nodes == EXPECTED_TOTAL_NODES, total_nodes
    assert total_exp == EXPECTED_TOTAL_EXPANSIONS, total_exp
    assert dict(depth_hist) == EXPECTED_DEPTH_HIST, dict(depth_hist)
