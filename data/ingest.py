"""Ingest real-world datasets into the Solis corpus format.

`build_corpus.py` produces clean, procedurally generated conversations — good
for wiring the pipeline up, but the model can only ever learn the task shapes we
hand-authored. To train on real data, this script converts existing datasets
(Hugging Face datasets, or local JSONL / plain text) into the same
`{"messages": [...]}` JSONL that `prepare.py` already understands, with a
held-out validation split.

It normalises the common chat schemas seen in the wild:

  * ``messages``: [{role, content}, ...]            (already our shape)
  * ``system`` / ``user`` / ``assistant`` columns   (one turn per row)
  * ``conversations``: [{from, value}, ...]          (ShareGPT style)
  * ``prompt`` / ``response`` (or ``completion``)    (instruction pairs)
  * a single ``text`` column                          (raw pretraining text)

Examples:

    # From the Hugging Face cache / hub, several sources into one corpus:
    python data/ingest.py \
        --hf oyildirim/cyberstrike-sft-120k \
        --hf SkywardNomad92/pentest-findings-v2 \
        --hf AlicanKiraz0/cybersecurity-dataset-fenrir-v2.1 \
        --out data/corpus.jsonl

    # From local files:
    python data/ingest.py --jsonl mydata.jsonl --text notes.txt --out data/corpus.jsonl

The output is written exactly where `prepare.py` expects it, so the rest of the
pipeline is unchanged:

    python data/ingest.py --hf oyildirim/cyberstrike-sft-120k --out data/corpus.jsonl
    python data/train_tokenizer.py
    python data/prepare.py
    # then train.py — NOT run here.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterator, Optional

HERE = Path(__file__).resolve().parent

VALID_ROLES = {"system", "user", "assistant", "tool"}
# ShareGPT-style "from" values -> our roles.
FROM_MAP = {
    "system": "system", "human": "user", "user": "user", "prompter": "user",
    "gpt": "assistant", "assistant": "assistant", "bot": "assistant",
    "tool": "tool", "function": "tool", "observation": "tool",
}


# --------------------------------------------------------------------------- #
# Schema normalisation
# --------------------------------------------------------------------------- #
def _clean(msgs: list[dict]) -> Optional[list[dict]]:
    """Validate and tidy a message list; return None if unusable."""
    out = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        if role not in VALID_ROLES or not isinstance(content, str):
            return None
        content = content.strip()
        if content:
            out.append({"role": role, "content": content})
    # Need at least one user turn and one assistant turn to be trainable.
    roles = {m["role"] for m in out}
    if "assistant" not in roles or not out:
        return None
    return out


def normalise_row(row: dict) -> Optional[list[dict]]:
    """Map one dataset row (any supported schema) to a message list."""
    # 1. Already our shape.
    if isinstance(row.get("messages"), list):
        norm = []
        for m in row["messages"]:
            role = m.get("role") or m.get("from")
            role = FROM_MAP.get(role, role)
            content = m.get("content")
            if content is None:
                content = m.get("value")
            norm.append({"role": role, "content": content})
        return _clean(norm)

    # 2. ShareGPT-style conversations.
    conv = row.get("conversations") or row.get("conversation")
    if isinstance(conv, list):
        norm = [{"role": FROM_MAP.get(t.get("from"), t.get("from")),
                 "content": t.get("value")} for t in conv]
        return _clean(norm)

    # 3. Split system/user/assistant columns.
    if "assistant" in row and ("user" in row or "prompt" in row
                               or "instruction" in row):
        msgs = []
        if row.get("system"):
            msgs.append({"role": "system", "content": row["system"]})
        user = row.get("user") or row.get("prompt") or row.get("instruction")
        # Some instruction sets carry an extra input field.
        if row.get("input"):
            user = f"{user}\n\n{row['input']}"
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": row["assistant"]})
        return _clean(msgs)

    # 4. prompt / response (or completion) pairs.
    prompt = row.get("prompt") or row.get("instruction") or row.get("question")
    response = row.get("response") or row.get("completion") or row.get("answer")
    if prompt and response:
        return _clean([{"role": "user", "content": prompt},
                       {"role": "assistant", "content": response}])

    # 5. Raw text -> a single assistant turn, so pretraining-style corpora still
    #    contribute language modelling signal (the packer supervises assistant
    #    content, so this text is learned).
    text = row.get("text") or row.get("content")
    if isinstance(text, str) and text.strip():
        return [{"role": "assistant", "content": text.strip()}]

    return None


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def iter_hf(name: str, split: str, streaming: bool,
            text_field: Optional[str]) -> Iterator[list[dict]]:
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=streaming)
    for row in ds:
        if text_field:
            row = {"text": row.get(text_field, "")}
        msgs = normalise_row(row)
        if msgs:
            yield msgs


def iter_jsonl(path: Path) -> Iterator[list[dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = normalise_row(row)
            if msgs:
                yield msgs


def iter_textfile(path: Path, chunk_chars: int) -> Iterator[list[dict]]:
    """Split a plain-text file into chunks, each a single assistant turn."""
    buf: list[str] = []
    size = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for para in f.read().split("\n\n"):
            para = para.strip()
            if not para:
                continue
            buf.append(para)
            size += len(para)
            if size >= chunk_chars:
                yield [{"role": "assistant", "content": "\n\n".join(buf)}]
                buf, size = [], 0
    if buf:
        yield [{"role": "assistant", "content": "\n\n".join(buf)}]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", action="append", default=[],
                    help="Hugging Face dataset id (repeatable)")
    ap.add_argument("--jsonl", action="append", default=[], type=Path,
                    help="local JSONL file (repeatable)")
    ap.add_argument("--text", action="append", default=[], type=Path,
                    help="local plain-text file (repeatable)")
    ap.add_argument("--split", default="train", help="HF split to read")
    ap.add_argument("--hf-text-field", default=None,
                    help="treat this HF column as raw text")
    ap.add_argument("--streaming", action="store_true",
                    help="stream HF datasets instead of loading fully")
    ap.add_argument("--max-per-source", type=int, default=None,
                    help="cap conversations taken from each source")
    ap.add_argument("--text-chunk-chars", type=int, default=2000)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--min-chars", type=int, default=1,
                    help="drop conversations whose total content is shorter")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=HERE / "corpus.jsonl")
    args = ap.parse_args()

    if not (args.hf or args.jsonl or args.text):
        raise SystemExit("give at least one --hf, --jsonl or --text source")

    rng = random.Random(args.seed)
    out_train = args.out
    out_val = args.out.with_suffix(".val.jsonl")
    out_train.parent.mkdir(parents=True, exist_ok=True)

    def sources() -> Iterator[tuple[str, Iterator[list[dict]]]]:
        for name in args.hf:
            yield f"hf:{name}", iter_hf(name, args.split, args.streaming,
                                        args.hf_text_field)
        for p in args.jsonl:
            yield f"jsonl:{p.name}", iter_jsonl(p)
        for p in args.text:
            yield f"text:{p.name}", iter_textfile(p, args.text_chunk_chars)

    kept = 0
    per_source: dict[str, int] = {}
    ft = out_train.open("w", encoding="utf-8")
    fv = out_val.open("w", encoding="utf-8")
    try:
        for label, it in sources():
            n = 0
            for msgs in it:
                if args.max_per_source and n >= args.max_per_source:
                    break
                total_chars = sum(len(m["content"]) for m in msgs)
                if total_chars < args.min_chars:
                    continue
                line = json.dumps({"messages": msgs}, ensure_ascii=False)
                # Deterministic per-conversation val assignment.
                (fv if rng.random() < args.val_fraction else ft).write(line + "\n")
                n += 1
                kept += 1
                if kept % 100_000 == 0:
                    print(f"  {kept:,} conversations written")
            per_source[label] = n
            print(f"  {label}: kept {n:,}")
    finally:
        ft.close()
        fv.close()

    print(f"\ntotal: {kept:,} conversations")
    for label, n in per_source.items():
        print(f"  {label:<50} {n:>12,}")
    print(f"train -> {out_train}")
    print(f"val   -> {out_val}")
    if kept == 0:
        print("\nWARNING: nothing was ingested — check the source schemas.")
        sys.exit(1)


if __name__ == "__main__":
    main()
