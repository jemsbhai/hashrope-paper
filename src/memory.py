"""
EXP-004: Branch/snapshot — peak memory under ToT branching (claim M1).

Measures structural sharing under immutable rope forking vs deep-copy baseline.

count_unique_nodes
    Tree-walk one or more rope roots, collect unique Python id() values.
    Returns {total_reachable, unique_count, by_type: {Leaf, Internal, RepeatNode}}.
    The structural sharing guard: rope forks share interior nodes; deepcopy forks
    do not.

fork_rope
    Simulate B ToT branches by appending a unique "thought" (128-byte slice) to
    the base rope via rope_concat. All forks share the base tree's nodes.

fork_deepcopy
    Same logical operation but deep-copies the base rope before each concat.
    Every node is duplicated — no sharing.

References to hashrope.rope go through the module object so tests can
monkeypatch if needed.
"""
from __future__ import annotations

import copy
from typing import Any

import hashrope.rope as _rope
from hashrope import PolynomialHash


def count_unique_nodes(roots: list) -> dict[str, Any]:
    """Tree-walk all roots, return unique node statistics.

    Returns:
        {
            "total_reachable": int,   # sum of all node visits (with duplicates)
            "unique_count": int,      # number of distinct id() values
            "by_type": {"Leaf": int, "Internal": int, "RepeatNode": int},
        }
    """
    seen: set[int] = set()
    total_visits = 0
    by_type: dict[str, int] = {"Leaf": 0, "Internal": 0, "RepeatNode": 0}

    def walk(node) -> None:
        nonlocal total_visits
        if node is None:
            return
        total_visits += 1
        nid = id(node)
        if nid in seen:
            return  # already counted — structural sharing!
        seen.add(nid)
        type_name = type(node).__name__
        by_type[type_name] = by_type.get(type_name, 0) + 1

        if isinstance(node, _rope.Internal):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, _rope.RepeatNode):
            walk(node.child)
        # Leaf: no children

    for root in roots:
        walk(root)

    return {
        "total_reachable": total_visits,
        "unique_count": len(seen),
        "by_type": dict(by_type),
    }


def fork_rope(
    base, thoughts: list[bytes], h: PolynomialHash
) -> list:
    """Create B forks by appending each thought to the base rope (structural sharing).

    Each fork = rope_concat(base, Leaf(thought_i), h). All forks share the base
    tree's nodes; the only new allocations are the O(log w) Internal spine per
    fork plus the new thought Leaf.

    Args:
        base: the base rope root node
        thoughts: list of B byte strings, each appended as a new Leaf
        h: PolynomialHash instance

    Returns:
        list of B fork root nodes
    """
    forks = []
    for thought in thoughts:
        thought_leaf = _rope.Leaf(thought, h)
        fork = _rope.rope_concat(base, thought_leaf, h)
        forks.append(fork)
    return forks


def fork_deepcopy(
    base, thoughts: list[bytes], h: PolynomialHash
) -> list:
    """Create B forks by deep-copying the base rope then appending each thought.

    Each fork = rope_concat(deepcopy(base), Leaf(thought_i), h). Every node is
    duplicated — no structural sharing between forks.

    Args:
        base: the base rope root node
        thoughts: list of B byte strings, each appended as a new Leaf
        h: PolynomialHash instance

    Returns:
        list of B fork root nodes
    """
    forks = []
    for thought in thoughts:
        base_copy = copy.deepcopy(base)
        thought_leaf = _rope.Leaf(thought, h)
        fork = _rope.rope_concat(base_copy, thought_leaf, h)
        forks.append(fork)
    return forks
