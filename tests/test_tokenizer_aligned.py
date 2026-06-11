"""
EXP-001 -- token-identity proof obligation (correctness GATE, no timing).

Red-first contract:
  * test_fixed_chunker_breaks_token_identity   -> PASSES now (the control proves
    the problem is real: blind byte chunking diverges from whole-string ids).
  * test_aligned_preserves_token_identity       -> RED until tokenizer_aligned_chunker
    is implemented; then GREEN.
  * test_aligned_never_bisects_utf8             -> RED until implemented; then GREEN.
  * test_fallback_preserves_identity_no_space   -> RED until implemented; then GREEN.

Coverage: byte-level BPE (gpt2) AND SentencePiece (t5-small), three leaf sizes,
multibyte text (curly quotes, accents, em dash, CJK, emoji), code with newlines,
and a no-safe-boundary fallback case.

Requires: transformers, tokenizers, sentencepiece (t5). First run downloads the
gpt2 and t5-small tokenizers from the HF hub. These are authored, deterministic,
OFFLINE fixtures -- the paper-grade ingestion benchmark uses a documented
downloaded corpus (see data/DATA_README.md), not these unit fixtures.
"""
import hashlib

import pytest

pytest.importorskip("transformers")
from transformers import AutoTokenizer

from src.tokenizer_aligned import (
    fixed_byte_chunker,
    tokenizer_aligned_chunker,
    token_identity_report,
    all_valid_utf8,
)

# --- deterministic offline corpus (varied natural prose + multibyte + code) ---
PROSE = (
    "The harbor town woke slowly under a sky the color of wet slate. Fishermen hauled "
    "their nets across the cobbles while gulls argued over the morning's discards. In the "
    "bakery on Mercer Street, the ovens had been burning since four, and the smell of rye "
    "drifted as far as the customs house. A young clerk named Aldous counted ledgers he did "
    "not understand, hoping the numbers would resolve themselves before his supervisor "
    "returned from the capital. Across the square, an old cartographer traced a coastline he "
    "had never sailed, certain that the river bent west where the official maps insisted it "
    "ran straight. Children chased a wooden hoop between the market stalls, and a violinist "
    "tuned an instrument that had survived three owners and two wars. By noon the fog would "
    "lift, the tide would turn, and the day's small commerce would resume its ancient rhythm. "
    "None of them suspected that the letter arriving on the afternoon packet would change the "
    "fortunes of the entire province. The merchant guild had quarreled for years over the "
    "tariff on imported salt, and the dispute had finally reached a court three hundred miles "
    "inland. Whatever the judges decided, the price of bread and the wages of dockhands would "
    "be tested before the season ended."
)
MULTIBYTE = (
    "Curly quotes \u201cyes\u201d, caf\u00e9 na\u00efve r\u00e9sum\u00e9 \u2014 an em dash; "
    "\u4f60\u597d\u4e16\u754c greetings; party \U0001F389 then done."
)
CODE = (
    "def merge_sorted(left, right):\n"
    "    result = []\n"
    "    i = j = 0\n"
    "    while i < len(left) and j < len(right):\n"
    "        if left[i] <= right[j]:\n"
    "            result.append(left[i]); i += 1\n"
    "        else:\n"
    "            result.append(right[j]); j += 1\n"
    "    result.extend(left[i:])\n"
    "    result.extend(right[j:])\n"
    "    return result\n"
)
CORPUS = PROSE + "\n\n" + MULTIBYTE + "\n\n" + CODE

# byte-level BPE and SentencePiece families
TOKENIZER_IDS = {"gpt2": "gpt2", "t5-small": "t5-small"}
TARGET_SIZES = [96, 192, 384]  # bytes; small so many leaves form -> stringent


@pytest.fixture(scope="module", params=list(TOKENIZER_IDS))
def tokenizer(request):
    return AutoTokenizer.from_pretrained(TOKENIZER_IDS[request.param], use_fast=True)


@pytest.mark.parametrize("target", TARGET_SIZES)
def test_fixed_chunker_breaks_token_identity(tokenizer, target):
    """RED CONTROL: blind byte chunking must NOT preserve token identity."""
    chunks = fixed_byte_chunker(CORPUS.encode("utf-8"), target)
    rep = token_identity_report(chunks, tokenizer, CORPUS)
    assert rep["exact_match"] is False, (
        "fixed byte chunker unexpectedly preserved identity -- the control is broken; "
        "the experiment cannot demonstrate the problem it claims to fix"
    )


@pytest.mark.parametrize("target", TARGET_SIZES)
def test_aligned_preserves_token_identity(tokenizer, target):
    """GREEN TARGET: aligned chunking must reproduce whole-string ids EXACTLY."""
    chunks = tokenizer_aligned_chunker(CORPUS, target, tokenizer)
    rep = token_identity_report(chunks, tokenizer, CORPUS)
    assert rep["exact_match"] is True, (
        f"token identity broken at target={target}: first divergence at token "
        f"{rep['first_divergence_tok']}, {rep['divergent_positions']} divergent positions "
        f"(whole={rep['whole_tok']} tok, chunked={rep['chunked_tok']} tok)"
    )


@pytest.mark.parametrize("target", TARGET_SIZES)
def test_aligned_never_bisects_utf8(tokenizer, target):
    """Aligned leaves must always be valid UTF-8 (no bisected codepoints)."""
    chunks = tokenizer_aligned_chunker(CORPUS, target, tokenizer)
    assert all_valid_utf8(chunks), "aligned chunker produced an invalid-UTF-8 leaf"


def test_fallback_preserves_identity_no_space(tokenizer):
    """No lone-space boundary inside the blob -> fallback grows the leaf; identity holds."""
    blob = "".join(hashlib.sha256(f"exp001-{i}".encode()).hexdigest() for i in range(20))
    text = (
        "The payload begins here and continues for a while before "
        + blob
        + " and then the trailer resumes after the blob ends."
    )
    chunks = tokenizer_aligned_chunker(text, 128, tokenizer)
    rep = token_identity_report(chunks, tokenizer, text)
    assert rep["exact_match"] is True, (
        f"fallback path broke identity: first divergence at token {rep['first_divergence_tok']}"
    )
