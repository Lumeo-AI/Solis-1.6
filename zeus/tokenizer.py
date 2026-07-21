"""Byte-level tokenizer for Zeus.

There is no learned vocabulary and no external dependency — every UTF-8 byte is
a token (0-255), plus a handful of special tokens for chat structure. This keeps
Zeus genuinely "from scratch": it learns language directly from raw bytes.
"""

from __future__ import annotations

from typing import List

# Special token ids live just above the 256 byte values.
BOS = 256   # beginning of sequence
EOS = 257   # end of sequence / end of a turn
USER = 258  # start of a user turn
ASST = 259  # start of an assistant turn

SPECIAL_IDS = {BOS, EOS, USER, ASST}


class ByteTokenizer:
    vocab_size = 260

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [BOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: List[int]) -> str:
        # Drop special tokens, reassemble the byte stream, decode leniently.
        raw = bytes(i for i in ids if i < 256)
        return raw.decode("utf-8", errors="replace")

    def encode_chat(self, messages: List[dict]) -> List[int]:
        """Render a chat transcript into the token format Zeus is trained on:

            <BOS> <USER> ...bytes... <EOS> <ASST> ...bytes... <EOS> ...
        """
        ids: List[int] = [BOS]
        for m in messages:
            role = m["role"]
            marker = USER if role == "user" else ASST
            ids.append(marker)
            ids.extend(self.encode(m["content"]))
            ids.append(EOS)
        # Prime the model to speak as the assistant.
        ids.append(ASST)
        return ids
