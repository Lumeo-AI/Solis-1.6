"""Checks for real-world data ingestion (data/ingest.py).

The normaliser is the load-bearing part: if it silently drops or mangles a
schema, training quietly runs on less (or worse) data than intended. These
tests pin the mapping for every schema the ingester claims to support.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data.ingest as ingest  # noqa: E402


def test_messages_schema_passthrough():
    row = {"messages": [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"}]}
    out = ingest.normalise_row(row)
    assert out == row["messages"]
    print("messages schema passes through")


def test_split_columns_schema():
    out = ingest.normalise_row({"system": "S", "user": "U", "assistant": "A"})
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    print("system/user/assistant columns map correctly")


def test_instruction_input_concatenated():
    out = ingest.normalise_row(
        {"instruction": "Summarise", "input": "the text", "assistant": "ok"})
    user = [m for m in out if m["role"] == "user"][0]["content"]
    assert "Summarise" in user and "the text" in user
    print("instruction + input are combined into the user turn")


def test_sharegpt_conversations_schema():
    out = ingest.normalise_row({"conversations": [
        {"from": "human", "value": "U"},
        {"from": "gpt", "value": "A"}]})
    assert [m["role"] for m in out] == ["user", "assistant"]
    print("ShareGPT conversations map correctly")


def test_prompt_response_schema():
    out = ingest.normalise_row({"prompt": "U", "response": "A"})
    assert out == [{"role": "user", "content": "U"},
                   {"role": "assistant", "content": "A"}]
    print("prompt/response pairs map correctly")


def test_raw_text_becomes_assistant_turn():
    out = ingest.normalise_row({"text": "raw passage"})
    assert out == [{"role": "assistant", "content": "raw passage"}]
    print("raw text becomes a single assistant turn")


def test_unusable_rows_are_dropped():
    # No assistant content -> not trainable.
    assert ingest.normalise_row({"user": "hi"}) is None
    # Empty everything.
    assert ingest.normalise_row({"text": "   "}) is None
    assert ingest.normalise_row({}) is None
    print("unusable rows are dropped rather than mangled")


def test_jsonl_roundtrip(tmp_path=Path("checkpoints/_ingest_test")):
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "src.jsonl"
    src.write_text(
        json.dumps({"prompt": "Q1", "response": "A1"}) + "\n" +
        json.dumps({"conversations": [{"from": "human", "value": "Q2"},
                                      {"from": "gpt", "value": "A2"}]}) + "\n" +
        json.dumps({"user": "no assistant here"}) + "\n",  # dropped
        encoding="utf-8")
    rows = list(ingest.iter_jsonl(src))
    assert len(rows) == 2, rows
    assert rows[0][-1]["content"] == "A1"
    assert rows[1][-1]["content"] == "A2"
    src.unlink()
    print("iter_jsonl keeps trainable rows and drops the rest")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nall {len(fns)} ingest checks passed")
