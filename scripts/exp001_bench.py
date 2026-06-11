"""EXP-001 ingestion benchmark (decomposed serial / rayon-batch / multiprocessing, identity-gated).

serial_whole : tokenize the whole string as one sequence (baseline, 1 effective thread)
rayon_batch  : batch-tokenize tokenizer-aligned leaves (rust/rayon threads across the batch)
mp_<N>       : tokenize leaves across N OS processes (rayon disabled in workers; warm pool)

Decomposed because tokenizers are superlinear on long contiguous strings, so chunking into
short leaves speeds up tokenization even on ONE core -- reporting only "serial vs parallel"
would misattribute that chunking win to parallelism. The MP pool is created once per
worker-count and only the dispatch+gather is timed (a real server keeps workers warm; per-call
spawn would unfairly penalize MP). Every parallel result asserts exact token identity before
its timing is trusted.

Run (PowerShell; HF_HUB_OFFLINE silences hub chatter once tokenizers are cached; rayon thread
count is swept per-process via RAYON_NUM_THREADS):
    $env:HF_HUB_OFFLINE=1
    python scripts/exp001_bench.py --tokenizers gpt2 t5-small --corpus data/raw/corpus.txt `
        --lengths 262144,1048576,4194304 --reps 7 --mp-workers 2,4,8
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("USE_TORCH", "0")  # tokenizer-only; skip torch import (kills pynvml + slow worker spawn)

import warnings
warnings.filterwarnings("ignore")
import logging
for _n in ("huggingface_hub", "transformers", "tokenizers", "filelock", "urllib3"):
    logging.getLogger(_n).setLevel(logging.ERROR)
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
except Exception:
    pass

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
    return {"median_ms": med * 1000.0,
            "ci95_ms": [boot[int(0.025 * len(boot))] * 1000.0, boot[int(0.975 * len(boot))] * 1000.0],
            "min_ms": ts_sorted[0] * 1000.0, "reps": reps}, r


def _leaves_str(text, leaf_bytes):
    return [b.decode("utf-8") for b in tokenizer_aligned_chunker(text, leaf_bytes)]


def _rayon_batch(tok, leaves):
    enc = tok(leaves, add_special_tokens=False)["input_ids"]
    return [i for seq in enc for i in seq]


_MP_TOK = None


def _mp_init(hid):
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["USE_TORCH"] = "0"
    import warnings as _w
    _w.filterwarnings("ignore")
    import logging as _l
    for _n in ("huggingface_hub", "transformers", "tokenizers", "filelock", "urllib3"):
        _l.getLogger(_n).setLevel(_l.ERROR)
    import transformers
    try:
        from transformers.utils import logging as _hl
        _hl.set_verbosity_error()
    except Exception:
        pass
    global _MP_TOK
    _MP_TOK = transformers.AutoTokenizer.from_pretrained(hid, use_fast=True)


def _mp_work(group):
    return [i for s in group for i in _MP_TOK.encode(s, add_special_tokens=False)]


def _contiguous_groups(leaves, n):
    k = math.ceil(len(leaves) / n)
    return [leaves[i:i + k] for i in range(0, len(leaves), k)]


def measure(hid, text, leaf_bytes, reps, mp_workers):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hid, use_fast=True)
    leaves = _leaves_str(text, leaf_bytes)
    whole = tok.encode(text, add_special_tokens=False)

    res = {"tokenizer": hid, "corpus_bytes": len(text.encode("utf-8")),
           "n_leaves": len(leaves), "n_tokens": len(whole), "leaf_bytes": leaf_bytes,
           "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "default(all)"),
           "measurements": {}}

    s, _ = _timed(lambda: tok.encode(text, add_special_tokens=False), reps)
    res["measurements"]["serial_whole"] = s

    b, ids = _timed(lambda: _rayon_batch(tok, leaves), reps)
    b["identity"] = (ids == whole)
    b["speedup_vs_serial"] = s["median_ms"] / b["median_ms"]
    res["measurements"]["rayon_batch"] = b
    if not b["identity"]:
        raise AssertionError(f"IDENTITY FAIL rayon_batch tokenizer={hid} len={res['corpus_bytes']}")

    # multiprocessing: warm/persistent pool per worker-count; time only the dispatch+gather
    for n in mp_workers:
        groups = _contiguous_groups(leaves, n)
        pool = Pool(n, initializer=_mp_init, initargs=(hid,))
        try:
            def mp_map(pool=pool, groups=groups):
                parts = pool.map(_mp_work, groups)
                return [i for part in parts for i in part]
            m, ids = _timed(mp_map, reps)  # pool warm; _timed warmup primes it
        finally:
            pool.close()
            pool.join()
        m["identity"] = (ids == whole)
        m["speedup_vs_serial"] = s["median_ms"] / m["median_ms"]
        res["measurements"][f"mp_{n}"] = m
        if not m["identity"]:
            raise AssertionError(f"IDENTITY FAIL mp_{n} tokenizer={hid} len={res['corpus_bytes']}")
    return res


def _capture_env():
    env = {"platform": platform.platform(), "processor": platform.processor() or "n/a",
           "cpu_count": os.cpu_count(), "python": platform.python_version(),
           "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS", "unset")}
    try:
        import transformers, tokenizers
        env["transformers"] = transformers.__version__
        env["tokenizers"] = tokenizers.__version__
    except Exception:
        pass
    return env


def _load_corpus(path, length_bytes):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        full = f.read()
    fb = full.encode("utf-8")
    tiled = False
    if len(fb) < length_bytes:
        fb = (full * math.ceil(length_bytes / max(1, len(fb)))).encode("utf-8")
        tiled = True
    return fb[:length_bytes].decode("utf-8", errors="ignore"), tiled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", nargs="+", default=["gpt2"])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--lengths", default="65536,262144,1048576")
    ap.add_argument("--leaf-bytes", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--mp-workers", default="",
                    help="comma-separated worker counts e.g. 2,4,8; omit to skip multiprocessing")
    ap.add_argument("--out-dir", default="experiments/exp_001_tokenizer_aligned/results")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x]
    mp_workers = [int(x) for x in args.mp_workers.split(",") if x]

    out = {"experiment": "EXP-001-ingestion", "timestamp_utc": _utc_now().isoformat(),
           "env": _capture_env(),
           "params": {"leaf_bytes": args.leaf_bytes, "reps": args.reps,
                      "lengths": lengths, "mp_workers": mp_workers}, "results": []}
    for hid in args.tokenizers:
        for L in lengths:
            text, tiled = _load_corpus(args.corpus, L)
            r = measure(hid, text, args.leaf_bytes, args.reps, mp_workers)
            r["corpus_tiled"] = tiled
            out["results"].append(r)
            best = max((m.get("speedup_vs_serial", 0) for k, m in r["measurements"].items()
                        if k != "serial_whole"), default=0)
            print(f"{hid:<16} {L:>9}B  leaves={r['n_leaves']:>5}  best_speedup={best:4.2f}x  "
                  f"(rayon@{r['rayon_num_threads']}){'  [TILED]' if tiled else ''}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.environ.get("RAYON_NUM_THREADS", "all")
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    for p in (os.path.join(args.out_dir, f"ingestion_rayon{tag}_{stamp}.json"),
              os.path.join(args.out_dir, f"ingestion_rayon{tag}_latest.json")):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    print(f"\nwrote results to {args.out_dir} (tag rayon{tag})", flush=True)


if __name__ == "__main__":
    main()
