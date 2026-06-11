"""
EXP-001 ingestion benchmark.

Decomposed, identity-gated measurement of context tokenization:

    serial_whole   -- tokenize the whole string as one sequence (baseline, 1 effective thread)
    rayon_batch    -- batch-tokenize tokenizer-aligned leaves (rust/rayon threads across the batch)
    mp_<N>         -- tokenize leaves across N OS processes (rayon disabled inside workers)

Why decomposed: tokenizers have a superlinear component on long contiguous strings, so
chunking into short leaves speeds up tokenization EVEN ON ONE CORE. Reporting only
"serial vs parallel" would misattribute that chunking win to parallelism. We therefore
record serial_whole, rayon_batch at the process's RAYON_NUM_THREADS (sweep it across runs
to vary cores), and an in-process multiprocessing sweep.

Every parallel result asserts exact token identity against whole-string ids before timing
is trusted. Output: latest + timestamped JSON with full environment metadata.

Run (the rayon thread sweep is per-process -- invoke once per thread count; PowerShell):
    $env:RAYON_NUM_THREADS=1; python scripts/exp001_bench.py --tokenizers gpt2 t5-small `
        --corpus data/raw/corpus.txt --lengths 65536,262144,1048576,4194304 --reps 7 --mp-workers 2,4,8
    $env:RAYON_NUM_THREADS=4; python scripts/exp001_bench.py ...   (etc; results tagged by thread count)
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import sys
import json
import time
import math
import random
import argparse
import platform
import statistics
import datetime
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tokenizer_aligned import tokenizer_aligned_chunker  # noqa: E402


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


# ----------------------------- statistics -----------------------------
def _timed(fn, reps, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t)
    ts_sorted = sorted(ts)
    med = statistics.median(ts_sorted)
    boot = []
    for _ in range(2000):
        s = [ts_sorted[random.randrange(len(ts_sorted))] for _ in ts_sorted]
        boot.append(statistics.median(s))
    boot.sort()
    lo = boot[int(0.025 * len(boot))]
    hi = boot[int(0.975 * len(boot))]
    return {
        "median_ms": med * 1000.0,
        "ci95_ms": [lo * 1000.0, hi * 1000.0],
        "min_ms": ts_sorted[0] * 1000.0,
        "reps": reps,
    }, r


# ----------------------------- pipelines -----------------------------
def _leaves_str(text, leaf_bytes):
    """Chunk to tokenizer-aligned byte leaves, decode to str for the tokenizer."""
    return [b.decode("utf-8") for b in tokenizer_aligned_chunker(text, leaf_bytes)]


def _rayon_batch(tok, leaves):
    enc = tok(leaves, add_special_tokens=False)["input_ids"]
    return [i for seq in enc for i in seq]


# multiprocessing worker state (one tokenizer per worker, rayon disabled)
_MP_TOK = None


def _mp_init(hid):
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import transformers
    global _MP_TOK
    _MP_TOK = transformers.AutoTokenizer.from_pretrained(hid, use_fast=True)


def _mp_work(group):
    return [i for s in group for i in _MP_TOK.encode(s, add_special_tokens=False)]


def _contiguous_groups(leaves, n):
    k = math.ceil(len(leaves) / n)
    return [leaves[i:i + k] for i in range(0, len(leaves), k)]


def _mp_run(hid, leaves, workers):
    groups = _contiguous_groups(leaves, workers)
    with Pool(workers, initializer=_mp_init, initargs=(hid,)) as p:
        parts = p.map(_mp_work, groups)
    return [i for part in parts for i in part]


# ----------------------------- measurement -----------------------------
def measure(hid, text, leaf_bytes, reps, mp_workers):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hid, use_fast=True)
    leaves = _leaves_str(text, leaf_bytes)
    whole = tok.encode(text, add_special_tokens=False)

    res = {
        "tokenizer": hid,
        "corpus_bytes": len(text.encode("utf-8")),
        "n_leaves": len(leaves),
        "n_tokens": len(whole),
        "leaf_bytes": leaf_bytes,
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "default(all)"),
        "measurements": {},
    }

    s, _ = _timed(lambda: tok.encode(text, add_special_tokens=False), reps)
    res["measurements"]["serial_whole"] = s

    b, ids = _timed(lambda: _rayon_batch(tok, leaves), reps)
    b["identity"] = (ids == whole)
    b["speedup_vs_serial"] = s["median_ms"] / b["median_ms"]
    res["measurements"]["rayon_batch"] = b
    if not b["identity"]:
        raise AssertionError(f"IDENTITY FAIL rayon_batch tokenizer={hid} len={res['corpus_bytes']}")

    for n in mp_workers:
        m, ids = _timed(lambda n=n: _mp_run(hid, leaves, n), reps)
        m["identity"] = (ids == whole)
        m["speedup_vs_serial"] = s["median_ms"] / m["median_ms"]
        res["measurements"][f"mp_{n}"] = m
        if not m["identity"]:
            raise AssertionError(f"IDENTITY FAIL mp_{n} tokenizer={hid} len={res['corpus_bytes']}")
    return res


def _capture_env():
    env = {
        "platform": platform.platform(),
        "processor": platform.processor() or "n/a",
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS", "unset"),
    }
    try:
        import transformers, tokenizers
        env["transformers"] = transformers.__version__
        env["tokenizers"] = tokenizers.__version__
    except Exception:
        pass
    return env


def _load_corpus(path, length_bytes):
    """Return ~length_bytes of real text (char-boundary clean); tile if corpus shorter."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        full = f.read()
    full_bytes = full.encode("utf-8")
    tiled = False
    if len(full_bytes) < length_bytes:
        reps = math.ceil(length_bytes / max(1, len(full_bytes)))
        full_bytes = (full * reps).encode("utf-8")
        tiled = True
    sliced = full_bytes[:length_bytes].decode("utf-8", errors="ignore")
    return sliced, tiled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", nargs="+", default=["gpt2"])
    ap.add_argument("--corpus", required=True, help="path to a UTF-8 text corpus")
    ap.add_argument("--lengths", default="65536,262144,1048576",
                    help="comma-separated byte lengths to sweep")
    ap.add_argument("--leaf-bytes", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--mp-workers", default="2,4",
                    help="comma-separated multiprocessing worker counts (empty to skip)")
    ap.add_argument("--out-dir", default="experiments/exp_001_tokenizer_aligned/results")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x]
    mp_workers = [int(x) for x in args.mp_workers.split(",") if x]

    out = {
        "experiment": "EXP-001-ingestion",
        "timestamp_utc": _utc_now().isoformat(),
        "env": _capture_env(),
        "params": {"leaf_bytes": args.leaf_bytes, "reps": args.reps,
                   "lengths": lengths, "mp_workers": mp_workers},
        "results": [],
    }
    for hid in args.tokenizers:
        for L in lengths:
            text, tiled = _load_corpus(args.corpus, L)
            r = measure(hid, text, args.leaf_bytes, args.reps, mp_workers)
            r["corpus_tiled"] = tiled
            out["results"].append(r)
            best = max(
                (m.get("speedup_vs_serial", 0) for k, m in r["measurements"].items() if k != "serial_whole"),
                default=0,
            )
            print(f"{hid:<16} {L:>9}B  leaves={r['n_leaves']:>5}  best_speedup={best:4.2f}x  "
                  f"(rayon@{r['rayon_num_threads']}){'  [TILED]' if tiled else ''}")

    os.makedirs(args.out_dir, exist_ok=True)
    threads_tag = os.environ.get("RAYON_NUM_THREADS", "all")
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    archival = os.path.join(args.out_dir, f"ingestion_rayon{threads_tag}_{stamp}.json")
    latest = os.path.join(args.out_dir, f"ingestion_rayon{threads_tag}_latest.json")
    for p in (archival, latest):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    print(f"\nwrote {archival}\nwrote {latest}")


if __name__ == "__main__":
    main()
