"""Tokenise the corpus and pack it into flat binary arrays for training.

Two files come out per split:

    <split>_tokens.bin   uint16/uint32 token ids, one long stream
    <split>_mask.bin     uint8, 1 where the token should be learned

The mask is what keeps this honest supervised training: only assistant tokens
carry loss, so the model is never rewarded for predicting the user's words.

Conversations are concatenated into one stream and cut into fixed-length
blocks. Packing this way wastes no compute on padding — every position in every
batch is a real token — and each conversation starts with `<|bos|>`, so the
model still gets a clear signal about where a document begins.

Run:
    python data/prepare.py
    python data/prepare.py --seq-len 2048 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solis.tokenizer import SolisTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

_TOK: SolisTokenizer | None = None


def _init_worker(tokenizer_path: str):
    global _TOK
    _TOK = SolisTokenizer.load(tokenizer_path)


def _encode_lines(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    assert _TOK is not None
    ids_out: list[int] = []
    mask_out: list[int] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        msgs = row.get("messages")
        if not msgs:
            continue
        ids, mask = _TOK.encode_chat_supervised(msgs)
        ids_out.extend(ids)
        mask_out.extend(mask)
    return (np.asarray(ids_out, dtype=np.int64),
            np.asarray(mask_out, dtype=np.uint8))


def chunked(path: Path, size: int):
    buf: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            if len(buf) >= size:
                yield buf
                buf = []
    if buf:
        yield buf


def prepare_split(corpus: Path, out_dir: Path, split: str,
                  tokenizer_path: Path, dtype, workers: int,
                  lines_per_chunk: int = 20_000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / f"{split}_tokens.bin"
    mask_path = out_dir / f"{split}_mask.bin"

    t0 = time.time()
    n_tokens = 0
    n_supervised = 0

    with tok_path.open("wb") as ft, mask_path.open("wb") as fm:
        if workers > 1:
            with Pool(workers, initializer=_init_worker,
                      initargs=(str(tokenizer_path),)) as pool:
                for ids, mask in pool.imap(
                        _encode_lines, chunked(corpus, lines_per_chunk)):
                    ft.write(ids.astype(dtype).tobytes())
                    fm.write(mask.tobytes())
                    n_tokens += ids.size
                    n_supervised += int(mask.sum())
                    if n_tokens and (n_tokens // 10_000_000) != \
                            ((n_tokens - ids.size) // 10_000_000):
                        rate = n_tokens / max(time.time() - t0, 1e-9)
                        print(f"  {split}: {n_tokens / 1e6:,.0f}M tokens "
                              f"({rate / 1e6:.2f}M tok/s)")
        else:
            _init_worker(str(tokenizer_path))
            for lines in chunked(corpus, lines_per_chunk):
                ids, mask = _encode_lines(lines)
                ft.write(ids.astype(dtype).tobytes())
                fm.write(mask.tobytes())
                n_tokens += ids.size
                n_supervised += int(mask.sum())

    dt = time.time() - t0
    print(f"  {split}: {n_tokens:,} tokens "
          f"({n_supervised:,} supervised, "
          f"{100 * n_supervised / max(n_tokens, 1):.1f}%) in {dt:,.0f}s")
    return {"tokens": n_tokens, "supervised": n_supervised,
            "tokens_file": tok_path.name, "mask_file": mask_path.name}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus.jsonl")
    ap.add_argument("--val-corpus", type=Path, default=None,
                    help="defaults to <corpus>.val.jsonl")
    ap.add_argument("--tokenizer", type=Path,
                    default=ROOT / "checkpoints" / "tokenizer.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "packed")
    ap.add_argument("--seq-len", type=int, default=2048,
                    help="recorded in meta.json; training reads blocks of this size")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.tokenizer.exists():
        raise SystemExit(
            f"missing {args.tokenizer} — run data/train_tokenizer.py first")
    if not args.corpus.exists():
        raise SystemExit(
            f"missing {args.corpus} — run data/build_corpus.py first")

    val_corpus = args.val_corpus or args.corpus.with_suffix(".val.jsonl")

    tok = SolisTokenizer.load(args.tokenizer)
    dtype = np.uint16 if tok.vocab_size < 2 ** 16 else np.uint32
    print(f"tokenizer vocab {tok.vocab_size:,} -> storing ids as {dtype.__name__}")

    meta = {
        "vocab_size": tok.vocab_size,
        "dtype": dtype.__name__,
        "seq_len": args.seq_len,
        "splits": {},
    }
    meta["splits"]["train"] = prepare_split(
        args.corpus, args.out_dir, "train", args.tokenizer, dtype, args.workers)
    if val_corpus.exists():
        meta["splits"]["val"] = prepare_split(
            val_corpus, args.out_dir, "val", args.tokenizer, dtype, 1)
    else:
        print(f"note: no validation corpus at {val_corpus}")

    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2),
                                            encoding="utf-8")
    total = meta["splits"]["train"]["tokens"]
    print(f"\npacked -> {args.out_dir}")
    print(f"train tokens: {total:,}  ({total / 1e6:,.1f}M)")


if __name__ == "__main__":
    main()
