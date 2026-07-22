"""Byte-level BPE tokenizer for Solis — trained from scratch on our own corpus.

The original Solis tokenizer emitted one token per UTF-8 byte. That is honest
but expensive: it spends roughly four tokens on an average English word, so a
2048-token context holds only ~500 words and every layer pays for four times the
sequence length it needs to. Training a BPE vocabulary on our own text is the
single largest efficiency win available to a small model, and it costs nothing
in provenance — the merges below are learned from the same corpus the model
trains on, not lifted from another project's vocab file.

Design:
  * ids 0-255            raw bytes, always present, so *nothing* is unencodable
  * ids 256-`SPECIAL_END` special tokens for chat structure
  * ids above that       learned merges, in rank order

Round-tripping is exact for arbitrary bytes, including invalid UTF-8.
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import regex as re


# --------------------------------------------------------------------------- #
# Special tokens
# --------------------------------------------------------------------------- #
# Sixteen slots are reserved above the byte range. Only some are used today;
# the spares mean a future capability (tool calls, reasoning traces) can be
# added without shifting every learned merge id and invalidating checkpoints.
SPECIAL_TOKENS: list[str] = [
    "<|pad|>",        # 256
    "<|bos|>",        # 257
    "<|eos|>",        # 258 - end of a turn
    "<|system|>",     # 259
    "<|user|>",       # 260
    "<|assistant|>",  # 261
    "<|tool|>",       # 262
    "<|tool_result|>",  # 263
    "<|think|>",      # 264
    "<|/think|>",     # 265
    "<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>",
    "<|reserved_3|>", "<|reserved_4|>", "<|reserved_5|>",
]
SPECIAL_BASE = 256
SPECIAL_END = SPECIAL_BASE + len(SPECIAL_TOKENS)  # first id available for merges

SPECIAL_IDS = {tok: SPECIAL_BASE + i for i, tok in enumerate(SPECIAL_TOKENS)}
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_IDS.items()}

PAD = SPECIAL_IDS["<|pad|>"]
BOS = SPECIAL_IDS["<|bos|>"]
EOS = SPECIAL_IDS["<|eos|>"]
SYSTEM = SPECIAL_IDS["<|system|>"]
USER = SPECIAL_IDS["<|user|>"]
ASSISTANT = SPECIAL_IDS["<|assistant|>"]

# Pre-tokenisation pattern. Splitting on this before merging keeps BPE from
# learning merges that straddle word/punctuation boundaries, which is what makes
# a small vocabulary generalise instead of memorising phrases.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)
_SPLIT_RE = re.compile(SPLIT_PATTERN)


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
class SolisTokenizer:
    """Byte-level BPE with a byte fallback and a chat template."""

    def __init__(self, merges: Sequence[tuple[int, int]] | None = None):
        # merges[i] is the pair that becomes id SPECIAL_END + i.
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self._rank = {pair: i for i, pair in enumerate(self.merges)}
        self._rebuild_vocab()

    # -- vocabulary -------------------------------------------------------- #
    def _rebuild_vocab(self) -> None:
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for tok, tid in SPECIAL_IDS.items():
            vocab[tid] = tok.encode("utf-8")
        for i, (a, b) in enumerate(self.merges):
            vocab[SPECIAL_END + i] = vocab[a] + vocab[b]
        self.vocab = vocab
        self._encode_chunk_cached = lru_cache(maxsize=200_000)(self._encode_chunk)

    @property
    def vocab_size(self) -> int:
        return SPECIAL_END + len(self.merges)

    # -- encoding ---------------------------------------------------------- #
    def _encode_chunk(self, chunk: bytes) -> tuple[int, ...]:
        """Apply merges to one pre-token, lowest rank first."""
        ids = list(chunk)
        if len(ids) < 2:
            return tuple(ids)
        while True:
            # Find the pair with the lowest merge rank present in `ids`.
            best_rank = None
            best_pos = -1
            for i in range(len(ids) - 1):
                r = self._rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pos = r, i
            if best_rank is None:
                break
            new_id = SPECIAL_END + best_rank
            ids[best_pos:best_pos + 2] = [new_id]
            if len(ids) < 2:
                break
        return tuple(ids)

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        ids: list[int] = [BOS] if add_bos else []
        for chunk in _SPLIT_RE.findall(text):
            ids.extend(self._encode_chunk_cached(chunk.encode("utf-8")))
        if add_eos:
            ids.append(EOS)
        return ids

    def encode_batch(self, texts: Iterable[str], **kw) -> list[list[int]]:
        return [self.encode(t, **kw) for t in texts]

    # -- decoding ---------------------------------------------------------- #
    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        parts: list[bytes] = []
        for i in ids:
            if i in ID_TO_SPECIAL:
                if skip_special:
                    continue
                parts.append(self.vocab[i])
            else:
                piece = self.vocab.get(int(i))
                if piece is not None:
                    parts.append(piece)
        return b"".join(parts).decode("utf-8", errors="replace")

    def decode_bytes(self, ids: Iterable[int], skip_special: bool = True) -> bytes:
        """Raw bytes, so a streaming caller can hold back partial UTF-8."""
        parts: list[bytes] = []
        for i in ids:
            if i in ID_TO_SPECIAL:
                if not skip_special:
                    parts.append(self.vocab[i])
                continue
            piece = self.vocab.get(int(i))
            if piece is not None:
                parts.append(piece)
        return b"".join(parts)

    # -- chat template ----------------------------------------------------- #
    def encode_chat(self, messages: Sequence[dict],
                    add_generation_prompt: bool = True) -> list[int]:
        """Render a conversation into the exact format Solis is trained on:

            <|bos|> <|system|> ... <|eos|> <|user|> ... <|eos|> <|assistant|> ... <|eos|>

        With `add_generation_prompt`, the sequence ends on a bare
        `<|assistant|>` so the model's next token starts its reply.
        """
        role_tok = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT,
                    "tool": SPECIAL_IDS["<|tool|>"]}
        ids: list[int] = [BOS]
        for m in messages:
            marker = role_tok.get(m["role"], USER)
            ids.append(marker)
            ids.extend(self.encode(m["content"]))
            ids.append(EOS)
        if add_generation_prompt:
            ids.append(ASSISTANT)
        return ids

    def encode_chat_supervised(self, messages: Sequence[dict]
                               ) -> tuple[list[int], list[int]]:
        """Same layout, plus a 0/1 mask marking the tokens worth learning.

        Only assistant content (and the `<|eos|>` that closes it) is supervised;
        the model is never trained to predict the user's words.
        """
        role_tok = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT,
                    "tool": SPECIAL_IDS["<|tool|>"]}
        ids: list[int] = [BOS]
        mask: list[int] = [0]
        for m in messages:
            marker = role_tok.get(m["role"], USER)
            body = self.encode(m["content"]) + [EOS]
            ids.append(marker)
            mask.append(0)
            ids.extend(body)
            mask.extend([1 if m["role"] == "assistant" else 0] * len(body))
        return ids, mask

    # -- persistence ------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": 1,
            "special_tokens": SPECIAL_TOKENS,
            "special_base": SPECIAL_BASE,
            "split_pattern": SPLIT_PATTERN,
            "merges": [list(m) for m in self.merges],
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SolisTokenizer":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        if blob.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError(
                "tokenizer file was built with a different special-token set; "
                "retrain it or restore the matching solis/tokenizer.py"
            )
        return cls(merges=[tuple(m) for m in blob["merges"]])

    # -- training ---------------------------------------------------------- #
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 32768,
              min_frequency: int = 2, verbose: bool = True) -> "SolisTokenizer":
        """Learn merges from `texts`.

        Standard byte-pair encoding, with two optimisations that make it
        practical in pure Python: text is collapsed to a (pre-token -> count)
        table first, and an inverted index maps each pair to the words
        containing it so a merge only touches the words it affects.
        """
        n_merges = vocab_size - SPECIAL_END
        if n_merges <= 0:
            raise ValueError(f"vocab_size must exceed {SPECIAL_END}")

        # 1. Collapse the corpus to unique pre-tokens with counts.
        word_counts: dict[bytes, int] = defaultdict(int)
        n_chunks = 0
        for text in texts:
            for chunk in _SPLIT_RE.findall(text):
                word_counts[chunk.encode("utf-8")] += 1
                n_chunks += 1
        if verbose:
            print(f"  corpus: {n_chunks:,} pre-tokens, "
                  f"{len(word_counts):,} unique")

        # 2. Represent each unique word as a mutable list of ids.
        words: list[list[int]] = []
        counts: list[int] = []
        for w, c in word_counts.items():
            if len(w) >= 2:
                words.append(list(w))
                counts.append(c)

        # 3. Pair statistics + inverted index (pair -> word indices).
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, ids in enumerate(words):
            c = counts[wi]
            for a, b in zip(ids, ids[1:]):
                pair_counts[(a, b)] += c
                pair_where[(a, b)].add(wi)

        merges: list[tuple[int, int]] = []
        for step in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            if pair_counts[best] < min_frequency:
                break
            new_id = SPECIAL_END + len(merges)
            merges.append(best)

            a, b = best
            affected = list(pair_where[best])
            for wi in affected:
                ids = words[wi]
                c = counts[wi]
                # Remove this word's old pair contributions.
                for p in zip(ids, ids[1:]):
                    pair_counts[p] -= c
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    pair_where[p].discard(wi)
                # Rewrite the word with the merge applied.
                out: list[int] = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                words[wi] = out
                # Add the new pair contributions.
                for p in zip(out, out[1:]):
                    pair_counts[p] += c
                    pair_where[p].add(wi)
            pair_counts.pop(best, None)
            pair_where.pop(best, None)

            if verbose and (step + 1) % 2000 == 0:
                print(f"  merge {step + 1:,}/{n_merges:,}")

        if verbose:
            print(f"  learned {len(merges):,} merges "
                  f"-> vocab {SPECIAL_END + len(merges):,}")
        return cls(merges=merges)


def load_default(path: str | Path = "checkpoints/tokenizer.json") -> SolisTokenizer:
    """Load the trained tokenizer, falling back to pure bytes if absent.

    A merge-less tokenizer is still fully functional — it just encodes one token
    per byte, which is exactly the behaviour of the original Solis tokenizer.
    """
    p = Path(path)
    if p.exists():
        return SolisTokenizer.load(p)
    return SolisTokenizer(merges=[])
