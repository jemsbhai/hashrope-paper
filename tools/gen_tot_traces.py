#!/usr/bin/env python
"""
tools/gen_tot_traces.py -- EXP-019 Milestone B: real Tree-of-Thought trace generator.

Runs a Yao-et-al.-style ToT-BFS controller (propose + value) over Game-of-24
puzzles using gpt-oss-120b via Ollama, and RECORDS the full branching tree per
puzzle to a versioned JSONL artifact. The recorded traces are later REPLAYED
(deterministically) through the EXP-019 branch/snapshot harness
(scripts/exp019_milestoneB_bench.py); LLM nondeterminism is sealed into this
artifact at generation time, so replay needs no model and no tokenizer.

Design (locked in LOGBOOK "EXP-019 -- Milestone B plan", plan-before-data):
  * Controller: BFS, beam b (default 5), depth D (default 3); 4 -> 3 -> 2 -> 1
    numbers. At each frontier node a single PROPOSE call lists candidate moves;
    each non-terminal candidate gets n_eval VALUE calls (default 1, value_temp>0)
    scored sure/likely/impossible; the global top-b survive. Depth-D candidates
    are terminal (1 number): no value call, checked for == 24.
  * State tracking trusts the LLM's parsed `remaining` numbers (the ToT
    scratchpad), NOT exact-Fraction recomputation -- intermediate states hold
    rounded decimals the model prints. Exact `fractions.Fraction` is used ONLY
    for the terminal solution check. Game-of-24 arithmetic correctness does NOT
    enter the gated byte-identity replay; solve-rate is advisory.
  * Tokenization: thought_text -> thought_tokens via the SAME call as Milestone A
    (transformers.AutoTokenizer.from_pretrained("gpt2"), add_special_tokens=False),
    sealing the tokens into the artifact for byte-identical replay.
  * Robustness: per-puzzle checkpoint/resume (a completed puzzle is appended as
    one JSONL line and skipped on restart); retry-with-backoff on Ollama errors.

Response shape (confirmed via tools/smoke_ollama.py probe): gpt-oss returns the
final answer in message.content and its reasoning in a separate message.thinking;
the parser reads message.content only.

Run (PowerShell) -- ONE seed per invocation (run 3x: 42, 43, 44):
    python tools/gen_tot_traces.py --seed 42
    python tools/gen_tot_traces.py --seed 42 --rank-start 901 --rank-count 25
Offline dry-run of the tree logic with ZERO LLM calls (in-process fake model):
    python tools/gen_tot_traces.py --seed 42 --rank-count 2 --mock
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from fractions import Fraction

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROPOSE_SYS = (
    "You are solving the Game of 24. You are given some numbers. You may pick two "
    "of them and combine them with +, -, *, or / to make a new number, replacing "
    "the two chosen numbers with the result. The goal is to eventually reach "
    "exactly 24, using each of the original numbers exactly once."
)


def propose_user(numbers_str: str) -> str:
    return (
        f"Current numbers: {numbers_str}\n\n"
        "Propose all sensible next steps. Give one step per line in exactly this "
        "form:\n"
        "a op b = c (remaining: <the numbers left>)\n"
        "For example: 4 * 6 = 24 (remaining: 24)"
    )


VALUE_SYS = (
    "You are evaluating intermediate states in the Game of 24, where the goal is "
    "to reach exactly 24 using the given numbers."
)


def value_user(numbers_str: str) -> str:
    return (
        f"Current numbers: {numbers_str}\n\n"
        "Can these numbers still be combined to reach exactly 24? "
        "Answer with exactly one word: sure, likely, or impossible."
    )


VALUE_SCORE = {"sure": 2, "likely": 1, "impossible": 0}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_NUM = r"-?\d+(?:\.\d+)?"
_MOVE_RE = re.compile(
    rf"^\s*({_NUM})\s*([+\-*/])\s*({_NUM})\s*=\s*({_NUM})\s*"
    rf"\(\s*remaining\s*:\s*([^)]*)\)\s*$"
)
_VALUE_RE = re.compile(r"\b(sure|likely|impossible)\b", re.IGNORECASE)


def parse_numbers(s: str) -> list[str]:
    """Split a 'remaining' string into number tokens (whitespace/comma separated)."""
    return [t for t in re.split(r"[\s,]+", s.strip()) if t]


def canonical_thought(a: str, op: str, b: str, c: str, remaining: list[str]) -> str:
    """Deterministic canonical move line (whitespace normalized; content preserved)."""
    return f"{a} {op} {b} = {c} (remaining: {' '.join(remaining)})"


def parse_propose(content: str) -> list[dict]:
    """Parse PROPOSE content into a de-duplicated list of candidate moves.

    Each candidate: {thought, remaining}. Lines that do not match the move grammar
    (prose, headers, blanks) are dropped. Duplicate canonical lines are removed,
    preserving first-occurrence order.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for line in content.splitlines():
        m = _MOVE_RE.match(line)
        if not m:
            continue
        a, op, b, c, rem = m.groups()
        remaining = parse_numbers(rem)
        thought = canonical_thought(a, op, b, c, remaining)
        if thought in seen:
            continue
        seen.add(thought)
        out.append({"thought": thought, "remaining": remaining})
    return out


def parse_value(content: str) -> tuple[str, int]:
    """Map VALUE content to (label, score). Unparsed -> ('unparsed', 0)."""
    m = _VALUE_RE.search(content or "")
    if not m:
        return "unparsed", 0
    label = m.group(1).lower()
    return label, VALUE_SCORE[label]


def is_24(numbers: list[str]) -> bool:
    """Exact terminal solution check: a single number equal to 24."""
    if len(numbers) != 1:
        return False
    tok = numbers[0]
    try:
        if Fraction(tok) == 24:
            return True
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return abs(float(tok) - 24.0) < 1e-4
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Ollama client (real) and in-process fake (--mock)
# ---------------------------------------------------------------------------

class OllamaError(RuntimeError):
    pass


def call_ollama(host, model, system, user, temperature, num_predict, timeout,
                max_retries, seed):
    """POST /api/chat (stream=false), return message.content. Retry w/ backoff."""
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict,
                    "seed": seed},
    }
    data = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            return obj.get("message", {}).get("content", "")
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
            last = e
            if attempt < max_retries - 1:
                back = 2.0 * (2 ** attempt)
                sys.stderr.write(
                    f"  [retry {attempt + 1}/{max_retries}] {type(e).__name__}: {e} "
                    f"-- backoff {back:.0f}s\n")
                sys.stderr.flush()
                time.sleep(back)
    raise OllamaError(f"Ollama call failed after {max_retries} attempts: {last}")


def _enumerate_moves(numbers: list[str]) -> list[dict]:
    """Deterministic Game-of-24 move enumeration for the in-process fake model.
    Mirrors the real model's exhaustive propose (probe-observed)."""
    fr = []
    for t in numbers:
        try:
            fr.append((t, Fraction(t)))
        except (ValueError, ZeroDivisionError):
            return []
    out = []
    n = len(fr)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            (sa, a), (sb, b) = fr[i], fr[j]
            rest_idx = [k for k in range(n) if k not in (i, j)]
            for op in ("+", "-", "*", "/"):
                if op == "/" and b == 0:
                    continue
                c = {"+": a + b, "-": a - b, "*": a * b,
                     "/": (a / b if b != 0 else None)}[op]
                if c is None:
                    continue
                cs = str(c.numerator) if c.denominator == 1 else f"{float(c):.10g}"
                remaining = [cs] + [fr[k][0] for k in rest_idx]
                out.append({"a": sa, "op": op, "b": sb, "c": cs,
                            "remaining": remaining})
    # commutative dedup (+/*) and direction handling are left in deliberately:
    # the real model also lists both orders; downstream dedup is by canonical line
    return out


def fake_propose(numbers: list[str]) -> str:
    lines = []
    for mv in _enumerate_moves(numbers):
        lines.append(canonical_thought(mv["a"], mv["op"], mv["b"], mv["c"],
                                        mv["remaining"]))
    return "\n".join(lines)


def fake_value(numbers: list[str], seed: int) -> str:
    h = (hash(("v", tuple(numbers), seed)) % 3)
    return ["impossible", "likely", "sure"][h]


# ---------------------------------------------------------------------------
# Tokenizer (gpt2, Milestone-A call form)
# ---------------------------------------------------------------------------

class Gpt2Tok:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("gpt2")

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]


# ---------------------------------------------------------------------------
# Tree controller
# ---------------------------------------------------------------------------

def build_tree(puzzle_numbers, rank, seed, model, host, endpoint,
               propose_temp, value_temp, n_eval, beam, max_depth,
               num_predict, timeout, max_retries, tok, mock):
    """Run ToT-BFS for one puzzle; return the recorded tree dict."""
    nodes = []
    calls = {"propose": 0, "value": 0}

    def new_node(parent_id, depth, thought, state):
        nid = len(nodes)
        toks = tok.encode(thought) if thought else []
        nodes.append({
            "id": nid, "parent_id": parent_id, "depth": depth,
            "thought_text": thought, "thought_tokens": toks,
            "state_numbers": state,
            "value_label": None, "value_score": None,
            "in_beam": False, "is_terminal": False, "is_solution": False,
        })
        return nodes[nid]

    def do_propose(numbers_str):
        calls["propose"] += 1
        if mock:
            return fake_propose(parse_numbers(numbers_str))
        return call_ollama(host, model, PROPOSE_SYS, propose_user(numbers_str),
                           propose_temp, num_predict, timeout, max_retries, seed)

    def do_value(numbers_str):
        # n_eval samples; average score; majority-ish label (highest-scoring seen)
        labels, scores = [], []
        for _ in range(n_eval):
            calls["value"] += 1
            if mock:
                content = fake_value(parse_numbers(numbers_str), seed)
            else:
                content = call_ollama(host, model, VALUE_SYS,
                                      value_user(numbers_str), value_temp,
                                      num_predict, timeout, max_retries, seed)
            lab, sc = parse_value(content)
            labels.append(lab)
            scores.append(sc)
        avg = sum(scores) / len(scores)
        best = labels[scores.index(max(scores))]
        return best, avg

    root = new_node(None, 0, "", list(puzzle_numbers))
    root["in_beam"] = True
    beams = {0: [root["id"]]}
    frontier = [root]
    gen_order = 0

    for d in range(1, max_depth + 1):
        candidates = []
        for parent in frontier:
            content = do_propose(" ".join(parent["state_numbers"]))
            for cand in parse_propose(content):
                node = new_node(parent["id"], d, cand["thought"], cand["remaining"])
                node["_order"] = gen_order
                gen_order += 1
                terminal = (len(node["state_numbers"]) <= 1) or (d >= max_depth)
                node["is_terminal"] = terminal
                candidates.append(node)

        if d >= max_depth:
            # terminal layer: solution check, no value, no further expansion
            for c in candidates:
                c["is_terminal"] = True
                c["is_solution"] = is_24(c["state_numbers"])
            beams[d] = [c["id"] for c in candidates if c["is_solution"]]
            frontier = []
            break

        # non-terminal layer: value all expandable candidates, then global beam
        expandable = [c for c in candidates if not c["is_terminal"]]
        for c in candidates:
            if c["is_terminal"]:
                c["is_solution"] = is_24(c["state_numbers"])
        for c in expandable:
            lab, sc = do_value(" ".join(c["state_numbers"]))
            c["value_label"], c["value_score"] = lab, sc
        kept = sorted(expandable, key=lambda c: (-c["value_score"], c["_order"]))[:beam]
        kept_ids = {c["id"] for c in kept}
        for c in candidates:
            c["in_beam"] = c["id"] in kept_ids
        beams[d] = [c["id"] for c in kept]
        frontier = kept

    for n in nodes:
        n.pop("_order", None)

    # first solution path (root -> solution), if any
    solved = any(n["is_solution"] for n in nodes)
    solution_path = None
    if solved:
        sol = next(n for n in nodes if n["is_solution"])
        path = []
        cur = sol
        while cur is not None:
            path.append(cur["id"])
            cur = nodes[cur["parent_id"]] if cur["parent_id"] is not None else None
        solution_path = list(reversed(path))

    return {
        "puzzle_id": rank,
        "puzzle": " ".join(puzzle_numbers),
        "seed": seed,
        "model_requested": model,
        "endpoint": endpoint,
        "sampling": {"propose_temp": propose_temp, "value_temp": value_temp,
                     "n_eval": n_eval},
        "beam_width": beam,
        "max_depth": max_depth,
        "tokenizer": "gpt2",
        "token_encoding": "hf_autotokenizer_gpt2_add_special_tokens_false",
        "nodes": nodes,
        "beams": {str(k): v for k, v in beams.items()},
        "solved": solved,
        "solution_path": solution_path,
        "n_llm_calls": calls,
        "gen_timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Corpus / checkpoint
# ---------------------------------------------------------------------------

def load_puzzles(corpus_jsonl, rank_start, rank_count):
    out = []
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rank_start <= rec["rank"] < rank_start + rank_count:
                out.append((rec["rank"], rec["puzzles"].split()))
    out.sort(key=lambda x: x[0])
    return out


def completed_ranks(out_path):
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["puzzle_id"])
            except (ValueError, KeyError):
                continue
    return done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="EXP-019 Milestone B ToT trace generator.")
    p.add_argument("--seed", type=int, required=True,
                   help="trace seed (run 42, 43, 44 in separate invocations)")
    p.add_argument("--rank-start", type=int, default=901)
    p.add_argument("--rank-count", type=int, default=25)
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--model", default="gpt-oss:120b-cloud")
    p.add_argument("--propose-temp", type=float, default=0.7)
    p.add_argument("--value-temp", type=float, default=0.7)
    p.add_argument("--n-eval", type=int, default=1)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--num-predict", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--corpus-jsonl", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--mock", action="store_true",
                   help="use an in-process deterministic fake model (no network)")
    return p


def main():
    args = build_parser().parse_args()
    corpus = args.corpus_jsonl or os.path.join(
        _REPO, "data", "canonical", "game_of_24.jsonl")
    out_path = args.out or os.path.join(
        _REPO, "data", "canonical", f"tot_traces_gptoss120b_s{args.seed}.jsonl")
    endpoint = "/api/chat"

    puzzles = load_puzzles(corpus, args.rank_start, args.rank_count)
    done = completed_ranks(out_path)
    todo = [(r, n) for (r, n) in puzzles if r not in done]

    print(f"seed={args.seed} model={args.model} mock={args.mock}")
    print(f"puzzles ranks [{args.rank_start}, {args.rank_start + args.rank_count}) "
          f"= {len(puzzles)} total, {len(done)} already done, {len(todo)} to do")
    print(f"out={out_path}")

    tok = Gpt2Tok()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    t_start = time.perf_counter()
    for i, (rank, numbers) in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            tree = build_tree(
                numbers, rank, args.seed, args.model, args.host, endpoint,
                args.propose_temp, args.value_temp, args.n_eval, args.beam,
                args.max_depth, args.num_predict, args.timeout, args.max_retries,
                tok, args.mock)
        except OllamaError as e:
            sys.stderr.write(
                f"\nABORT at rank {rank}: {e}\n"
                f"Current puzzle NOT written; {i - 1} puzzle(s) saved this run. "
                f"Re-run the same command to resume from rank {rank}.\n")
            sys.exit(1)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(tree, ensure_ascii=False) + "\n")
            f.flush()
        dt = time.perf_counter() - t0
        print(f"  [{i}/{len(todo)}] rank {rank} ({tree['puzzle']}): "
              f"{len(tree['nodes'])} nodes, solved={tree['solved']}, "
              f"calls={tree['n_llm_calls']}, {dt:.1f}s", flush=True)

    total = time.perf_counter() - t_start
    print(f"done: {len(todo)} puzzles in {total:.1f}s "
          f"({total / max(len(todo), 1):.1f}s/puzzle)")


if __name__ == "__main__":
    main()
