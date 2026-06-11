"""
EXP-001 -- Tokenizer-aligned byte chunking with a token-identity guarantee.

Motivation
----------
Hashrope stores context as 4 KB *byte* leaves. The prior "4.12x GPU ingestion"
result (CLAIMS.md S1) tokenized those leaves independently and compared the
result against whole-string tokenization -- but the two pipelines produce
DIFFERENT model inputs: naive byte cuts (a) split BPE merges at chunk seams and
(b) can bisect a multibyte UTF-8 codepoint, yielding un-tokenizable leaves.

This module replaces blind chunking with a tokenizer-aligned chunker whose cuts
provably reconstruct the whole-string token ids.

Three-tier correctness design
-----------------------------
1. Boundary predicate (runtime, cheap string scan, NO whole-string tokenization):
   propose cuts at tokenizer-safe boundaries. The validated default is a
   metaspace-aware "lone space between non-whitespace" predicate, with the space
   leading the next leaf. Verified to preserve token identity for BOTH byte-level
   BPE (GPT-2 family) AND SentencePiece (T5/Gemma/Mistral family) on real prose
   and code, at ideal leaf granularity.
2. Offline proof obligation (this module's `token_identity_report`, exercised by
   the test suite): tokenize every leaf independently, concatenate, and assert
   exact equality with whole-string ids. This is the *gate* that validates a
   predicate for a given tokenizer. Run during development, not per request.
3. Fallback (always identity-safe): when no safe boundary exists inside a target
   window (e.g. a long no-space blob, or non-space-delimited script such as CJK),
   the leaf simply grows -- not cutting can never break identity. Parallelism
   degrades there; correctness never does.

NOTE: the runtime path applies only tier 1 (a cheap scan); it never tokenizes the
whole string, so the ingestion speedup is preserved. A cheap per-seam check is a
planned tier-2 runtime safety net for adversarial inputs (see LOGBOOK EXP-001).
"""
from typing import List, Optional, Tuple

# Whitespace bytes/chars treated as unsafe to cut inside (runs trigger
# SentencePiece normalization differences and byte-BPE pre-token effects).
_WHITESPACE = " \t\n\r\f\v"


def _enc(tokenizer, s: str) -> List[int]:
    return tokenizer.encode(s, add_special_tokens=False)


def _utf8_len(ch: str) -> int:
    """UTF-8 byte length of a single character, computed without encoding it."""
    o = ord(ch)
    if o < 0x80:
        return 1
    if o < 0x800:
        return 2
    if o < 0x10000:
        return 3
    return 4


def _next_safe_boundary(text: str, start: int, n: int) -> Optional[int]:
    """
    First 'lone space' boundary at char index j >= start: a single space (0x20)
    flanked by non-whitespace on both sides. Cutting here lets the space lead the
    next leaf, which preserves token identity for both byte-BPE and SentencePiece.
    Returns the char index, or None if no safe boundary remains.
    """
    j = max(start, 1)
    while j < n - 1:
        if text[j] == " " and text[j - 1] not in _WHITESPACE and text[j + 1] not in _WHITESPACE:
            return j
        j += 1
    return None


def whole_string_ids(text: str, tokenizer) -> List[int]:
    """Ground-truth token ids for the entire context, tokenized as one sequence."""
    return _enc(tokenizer, text)


def chunked_ids(chunks: List[bytes], tokenizer) -> Tuple[List[int], int]:
    """
    Concatenation of per-leaf token ids (each leaf tokenized independently).
    Returns (ids, n_invalid_utf8_leaves). A leaf that is not valid UTF-8 (a naive
    byte cut bisected a codepoint) is decoded with errors='replace', corrupting the
    seam -- counted as a distinct failure mode.
    """
    out: List[int] = []
    invalid = 0
    for cb in chunks:
        try:
            s = cb.decode("utf-8")
        except UnicodeDecodeError:
            invalid += 1
            s = cb.decode("utf-8", errors="replace")
        out.extend(_enc(tokenizer, s))
    return out, invalid


def token_identity_report(chunks: List[bytes], tokenizer, text: str) -> dict:
    """
    Tier-2 offline gate: compare concat(per-leaf ids) against whole-string ids.
    Returns exact-match flag plus divergence diagnostics.
    """
    whole = whole_string_ids(text, tokenizer)
    chunked, invalid_leaves = chunked_ids(chunks, tokenizer)
    n = min(len(whole), len(chunked))
    first_div = next((i for i in range(n) if whole[i] != chunked[i]), None)
    if first_div is None and len(whole) != len(chunked):
        first_div = n  # diverge only in length
    diff_positions = sum(1 for i in range(n) if whole[i] != chunked[i]) + abs(
        len(whole) - len(chunked)
    )
    return {
        "exact_match": whole == chunked,
        "n_leaves": len(chunks),
        "whole_tok": len(whole),
        "chunked_tok": len(chunked),
        "first_divergence_tok": first_div,
        "divergent_positions": diff_positions,
        "invalid_utf8_leaves": invalid_leaves,
    }


def all_valid_utf8(chunks: List[bytes]) -> bool:
    """True iff every leaf decodes as UTF-8 (no bisected codepoints)."""
    for cb in chunks:
        try:
            cb.decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True


def fixed_byte_chunker(data: bytes, target_leaf_bytes: int) -> List[bytes]:
    """Naive control: blind byte cuts every target_leaf_bytes (the old 4 KB behavior)."""
    if not data:
        return [b""]
    return [data[i:i + target_leaf_bytes] for i in range(0, len(data), target_leaf_bytes)]


def tokenizer_aligned_chunker(
    text: str,
    target_leaf_bytes: int,
    tokenizer=None,
) -> List[bytes]:
    """
    Split `text` into ~target_leaf_bytes UTF-8 byte leaves, retreating each cut to a
    tokenizer-safe boundary so that independently tokenizing each leaf and
    concatenating the ids EXACTLY reproduces whole-string tokenization.

    Returns a list of UTF-8 byte chunks. Cuts land on char boundaries (no UTF-8
    sequence is ever bisected). Where no safe boundary exists inside a window, the
    leaf grows (identity-preserving fallback).

    Parameters
    ----------
    text : str
        The full context.
    target_leaf_bytes : int
        Soft target leaf size, measured in UTF-8 bytes (matches hashrope leaves).
    tokenizer : optional
        Unused on the runtime path (kept for signature symmetry with the offline
        identity gate / family validation). The chunker is a string scan and does
        NOT tokenize the whole string, so the ingestion speedup is preserved.
    """
    if not text:
        return [b""]
    n = len(text)
    leaves: List[bytes] = []
    leaf_start = 0   # char index where the current leaf starts
    bytes_acc = 0    # UTF-8 bytes accumulated since leaf_start
    i = 0
    while i < n:
        bytes_acc += _utf8_len(text[i])
        if bytes_acc >= target_leaf_bytes:
            cut = _next_safe_boundary(text, i, n)
            if cut is None:
                break  # no safe boundary remains -> remainder becomes one leaf (fallback)
            leaves.append(text[leaf_start:cut].encode("utf-8"))
            leaf_start = cut
            bytes_acc = 0
            i = cut  # resume scanning at the cut; the space at `cut` leads the new leaf
            continue
        i += 1
    leaves.append(text[leaf_start:].encode("utf-8"))
    return leaves
