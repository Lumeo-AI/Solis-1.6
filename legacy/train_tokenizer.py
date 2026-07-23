"""Train the Solis BPE vocabulary on our own corpus.

The merges are learned here and nowhere else — no vocabulary is imported from
another model. Training runs on a sample of the corpus rather than all of it:
BPE merge order is decided by relative pair frequencies, which stabilise long
before the whole corpus has been read, so sampling costs nothing in quality and
saves a great deal of time.

Run:
    python data/train_tokenizer.py                    # 16k vocab
    python data/train_tokenizer.py --vocab-size 32768 --sample-mb 200

Writes: checkpoints/tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solis.tokenizer import SolisTokenizer, SPECIAL_END  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def iter_corpus_text(path: Path, max_bytes: int):
    """Yield message text from a JSONL corpus, stopping after `max_bytes`."""
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            seen += len(line)
            if seen > max_bytes:
                return
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for m in row.get("messages", []):
                yield m["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--sample-mb", type=int, default=150,
                    help="how much of the corpus to read for merge statistics")
    ap.add_argument("--min-frequency", type=int, default=2)
    args = ap.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"missing {args.corpus} — run data/build_corpus.py first")

    corpus_mb = args.corpus.stat().st_size / 1e6
    sample_mb = min(args.sample_mb, corpus_mb)
    print(f"corpus {corpus_mb:,.0f} MB, sampling {sample_mb:,.0f} MB")
    print(f"target vocab {args.vocab_size:,} "
          f"({args.vocab_size - SPECIAL_END:,} merges to learn)")

    t0 = time.time()
    tok = SolisTokenizer.train(
        iter_corpus_text(args.corpus, int(sample_mb * 1e6)),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    tok.save(args.out)
    print(f"trained in {time.time() - t0:,.0f}s -> {args.out}")

    # Report the compression we actually bought. This number is the whole
    # justification for having a tokenizer at all: it is the factor by which
    # every sequence — and therefore the cost of every layer — shrinks.
    plain = SolisTokenizer(merges=[])
    samples = list(iter_corpus_text(args.corpus, 400_000))
    text = "\n".join(samples)
    n_bytes = len(plain.encode(text))
    n_tokens = len(tok.encode(text))
    print(f"\nvocab size:  {tok.vocab_size:,}")
    print(f"compression: {n_bytes:,} byte-tokens -> {n_tokens:,} BPE tokens "
          f"({n_bytes / max(n_tokens, 1):.2f}x)")
    print(f"a {2048}-token context now holds ~"
          f"{int(2048 * n_bytes / max(n_tokens, 1)):,} characters")


if __name__ == "__main__":
    main()
