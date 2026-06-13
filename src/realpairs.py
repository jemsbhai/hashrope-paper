"""
EXP-017: Deterministic real-pair extraction from ShareGPT / LMSYS datasets.

Each pair = (cached_tokens, query_tokens) where:
  - query  = tokenize(render(turns[:t+1]))   -- the arriving request
  - cached = tokenize(render(turns[:t]))      -- what's already in the cache
  - t drawn from [2, n_turns-1] per conversation, seeded

The token-level shared prefix is whatever BPE yields near the turn junction —
identical input to all arms, so internally consistent.

Rendering: f"{role}: {content}\\n\\n" concatenated (role names verbatim from
the dataset: "human"/"gpt" for ShareGPT, "user"/"assistant" for LMSYS).

Usage:
    from src.realpairs import load_real_pairs
    pairs = load_real_pairs("data/canonical/sharegpt_sample.jsonl", n_pairs=200, seed=42)
"""
from __future__ import annotations

import json
import random
from typing import Any


def load_conversations(path: str, min_turns: int = 4) -> list[dict]:
    """Load JSONL conversations, filtering by minimum turn count.

    Returns a list of dicts, each with keys: id, turns, n_turns.
    """
    convs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["n_turns"] >= min_turns:
                convs.append(rec)
    return convs


def render_turns(turns: list[dict[str, str]]) -> str:
    """Render a turn list as f"{role}: {content}\\n\\n" concatenated.

    Role names are used verbatim from the dataset (no normalization).
    """
    parts = []
    for turn in turns:
        parts.append(f"{turn['role']}: {turn['content']}\n\n")
    return "".join(parts)


def extract_text_pairs(
    conversations: list[dict],
    n_pairs: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Extract deterministic (cached_text, query_text) pairs.

    For each selected conversation, draws t in [2, n_turns-1] (seeded),
    then:
      query_text  = render(turns[:t+1])
      cached_text = render(turns[:t])

    Returns up to n_pairs dicts with: conv_id, split_turn, n_turns,
    cached_text, query_text.
    """
    rng = random.Random(seed)
    # Deterministic shuffle of qualifying conversations
    indices = list(range(len(conversations)))
    rng.shuffle(indices)
    selected = indices[:n_pairs]

    pairs = []
    for idx in selected:
        conv = conversations[idx]
        n_turns = conv["n_turns"]
        # t in [2, n_turns-1] inclusive
        t = rng.randint(2, n_turns - 1)
        turns = conv["turns"]
        cached_text = render_turns(turns[:t])
        query_text = render_turns(turns[:t + 1])
        pairs.append({
            "conv_id": conv["id"],
            "split_turn": t,
            "n_turns": n_turns,
            "cached_text": cached_text,
            "query_text": query_text,
        })
    return pairs


def tokenize_pairs(
    text_pairs: list[dict[str, Any]],
    tokenizer,
) -> list[dict[str, Any]]:
    """Tokenize cached_text and query_text, compute oracle token-LCP.

    Adds: cached_tokens, query_tokens, oracle_token_lcp, cached_len, query_len.
    """
    result = []
    for pair in text_pairs:
        cached_ids = tokenizer.encode(pair["cached_text"])
        query_ids = tokenizer.encode(pair["query_text"])
        # Oracle token-LCP: plain loop
        lcp = 0
        n = min(len(cached_ids), len(query_ids))
        for i in range(n):
            if cached_ids[i] != query_ids[i]:
                break
            lcp += 1
        else:
            lcp = n
        result.append({
            **pair,
            "cached_tokens": cached_ids,
            "query_tokens": query_ids,
            "oracle_token_lcp": lcp,
            "cached_len": len(cached_ids),
            "query_len": len(query_ids),
        })
    return result


def load_real_pairs(
    dataset_path: str,
    n_pairs: int = 200,
    seed: int = 42,
    tokenizer_name: str = "gpt2",
    min_turns: int = 4,
) -> list[dict[str, Any]]:
    """End-to-end: load dataset -> extract pairs -> tokenize.

    Returns list of dicts with all fields from extract_text_pairs +
    cached_tokens, query_tokens, oracle_token_lcp, cached_len, query_len.

    The tokenizer is loaded once via transformers.AutoTokenizer.
    """
    from transformers import AutoTokenizer

    convs = load_conversations(dataset_path, min_turns=min_turns)
    if len(convs) < n_pairs:
        import warnings
        warnings.warn(
            f"Only {len(convs)} conversations with >= {min_turns} turns "
            f"in {dataset_path} (requested {n_pairs}). Using all."
        )
    text_pairs = extract_text_pairs(convs, n_pairs, seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenize_pairs(text_pairs, tokenizer)
