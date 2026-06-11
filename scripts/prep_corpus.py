"""Build a non-tiled, diverse prose+code benchmark corpus for EXP-001.

Prose: several public-domain Project Gutenberg books (fetched at runtime).
Code:  a deterministic slice of the local Python standard library (.py, PSF-licensed).
Sources are split into ~8KB pieces and shuffled (fixed seed) so EVERY length-slice of
the corpus is a representative prose+code mix. The corpus is git-ignored; provenance is
written to data/DATA_README.md.

    python scripts/prep_corpus.py            # full: 15 books + 40 stdlib files (~8-12 MB)
    python scripts/prep_corpus.py --books 3 --code-files 8   # quick
"""
import os
import re
import glob
import random
import argparse
import datetime
import urllib.request

BOOKS = {
    1342: "Austen - Pride and Prejudice", 84: "Shelley - Frankenstein",
    1661: "Doyle - Adventures of Sherlock Holmes", 2701: "Melville - Moby Dick",
    11: "Carroll - Alice in Wonderland", 1232: "Machiavelli - The Prince",
    98: "Dickens - A Tale of Two Cities", 76: "Twain - Huckleberry Finn",
    1080: "Swift - A Modest Proposal", 2542: "Ibsen - A Doll's House",
    345: "Stoker - Dracula", 1400: "Dickens - Great Expectations",
    174: "Wilde - The Picture of Dorian Gray", 5200: "Kafka - Metamorphosis",
    2600: "Tolstoy - War and Peace",
}
_START = re.compile(r"\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG.*?\*\*\*", re.S)
_END = re.compile(r"\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG", re.S)


def fetch_book(bid, cache_dir):
    # Cache raw books so multi-seed runs fetch once and only reshuffle.
    cache_path = os.path.join(cache_dir, f"book_{bid}.txt")
    if os.path.exists(cache_path):
        return open(cache_path, encoding="utf-8", errors="replace").read()
    urls = [f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt",
            f"https://www.gutenberg.org/files/{bid}/{bid}.txt",
            f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            txt = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
            m1, m2 = _START.search(txt), _END.search(txt)
            if m1 and m2:
                txt = txt[m1.end():m2.start()]
            txt = txt.strip()
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(txt)
            return txt
        except Exception:
            continue
    return None


def chunkify(text, piece=8192):
    return [text[i:i + piece] for i in range(0, len(text), piece)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", type=int, default=len(BOOKS))
    ap.add_argument("--code-files", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42,
                    help="shuffle seed; vary across multi-seed confirmatory runs (books fetched once via cache)")
    ap.add_argument("--cache-dir", default="data/raw/cache",
                    help="raw books cached here so different seeds reshuffle the same downloaded text")
    ap.add_argument("--out", default="data/raw/corpus.txt")
    ap.add_argument("--readme", default="data/DATA_README.md")
    args = ap.parse_args()

    prose_sources, code_sources, pieces = [], [], []
    for bid in list(BOOKS)[:args.books]:
        t = fetch_book(bid, args.cache_dir)
        if t:
            prose_sources.append((bid, BOOKS[bid], len(t.encode("utf-8"))))
            pieces += chunkify(t)
            print(f"  fetched {bid}: {BOOKS[bid]} ({len(t.encode('utf-8')) // 1024} KB)")
        else:
            print(f"  SKIP {bid}: fetch failed")

    libdir = os.path.dirname(os.__file__)
    for cf in sorted(glob.glob(os.path.join(libdir, "*.py")))[:args.code_files]:
        try:
            t = open(cf, encoding="utf-8", errors="replace").read()
            code_sources.append((os.path.basename(cf), len(t.encode("utf-8"))))
            pieces += chunkify(t)
        except Exception:
            pass

    random.seed(args.seed)
    random.shuffle(pieces)
    corpus = "\n".join(pieces)
    cb = corpus.encode("utf-8")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(corpus)

    with open(args.readme, "w", encoding="utf-8") as f:
        f.write(f"# EXP-001 corpus provenance\n\nBuilt {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        f.write(f"Total: {len(cb):,} bytes ({len(cb) / 1e6:.2f} MB). Shuffled at 8KB granularity, seed={args.seed}.\n")
        f.write(f"Code from local stdlib: {libdir}\n\n## Prose (Project Gutenberg, public domain)\n")
        for bid, title, nb in prose_sources:
            f.write(f"- [{bid}] {title} ({nb // 1024} KB) - gutenberg.org/ebooks/{bid}\n")
        f.write(f"\n## Code (Python stdlib, PSF license), {len(code_sources)} files\n")
        for name, nb in code_sources:
            f.write(f"- {name} ({nb // 1024} KB)\n")

    print(f"\ncorpus: {len(cb):,} bytes ({len(cb) / 1e6:.2f} MB)  "
          f"[{len(prose_sources)} books + {len(code_sources)} code files, {len(pieces)} shuffled pieces]")
    print(f"wrote {args.out} and {args.readme}")


if __name__ == "__main__":
    main()
