"""Tool / function calling for Solis.

An agentic front end (or an MCP client) is useless without this: the harness
owns the tools — read a file, run a command, look something up — and the model's
only job is to *ask* for them. That ask travels
as OpenAI-shaped `tool_calls` on the response.

Qwen's chat template emits calls in the Hermes style:

    <tool_call>
    {"name": "read_file", "arguments": {"path": "src/main.py"}}
    </tool_call>

so the round trip is: hand `tools` to the chat template (transformers renders
them into the system turn), generate, then parse any `<tool_call>` blocks back
out and return them as `tool_calls`. This module owns both ends of that plus the
tolerant parsing that keeps a near-miss usable.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

# Primary form. Non-greedy body, DOTALL so a pretty-printed object works.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Fallback: a fenced JSON object, for when the model drops the tags.
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class ParsedCall:
    """One tool call the model asked for."""

    name: str
    arguments: dict
    raw: str = ""
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")

    def to_openai(self) -> dict:
        """OpenAI `tool_calls` entry. Arguments are a JSON *string* there."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                # OpenAI clients expect a string here and json.loads it
                # themselves; sending an object breaks strict parsers.
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


def _load_call(s: str) -> Optional[dict]:
    """Parse one candidate object into {name, arguments}, or None."""
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None
    args: Any = obj.get("arguments", obj.get("parameters", {}))
    # Some models emit the arguments already stringified.
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return {"name": obj["name"], "arguments": args if isinstance(args, dict) else {}}


def parse_tool_calls(text: str) -> list[ParsedCall]:
    """Extract every tool call from a finished generation.

    Tagged `<tool_call>` blocks win. If there are none, we accept a fenced JSON
    object that looks like a call — models drift on the wrapper and a near-miss
    is better recovered than silently dropped.
    """
    calls: list[ParsedCall] = []
    for m in _TOOL_CALL_RE.finditer(text):
        obj = _load_call(m.group(1))
        if obj:
            calls.append(ParsedCall(obj["name"], obj["arguments"], m.group(0)))
    if calls:
        return calls
    for m in _FENCE_RE.finditer(text):
        obj = _load_call(m.group(1))
        if obj:
            calls.append(ParsedCall(obj["name"], obj["arguments"], m.group(0)))
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove tool-call blocks, leaving whatever prose surrounded them."""
    out = _TOOL_CALL_RE.sub("", text)
    # An unterminated block means generation was cut mid-call; drop the tail
    # rather than surfacing half a JSON object as content.
    if "<tool_call>" in out:
        out = out.split("<tool_call>", 1)[0]
    return out.strip()


def normalise_tools(tools: Optional[list[dict]]) -> Optional[list[dict]]:
    """Coerce incoming tool specs into the shape chat templates expect.

    Accepts both the OpenAI wrapper (`{"type": "function", "function": {...}}`)
    and a bare function object, and drops anything unusable rather than letting
    a malformed entry blow up template rendering.
    """
    if not tools:
        return None
    out: list[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if t.get("type") == "function" else t
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object",
                                                       "properties": {}},
            },
        })
    return out or None


class ToolCallGate:
    """Streaming filter that withholds `<tool_call>` blocks from the text stream.

    A tool call is structured output, not prose — a client streaming content
    deltas must not receive raw JSON mid-sentence. Tags can be split across
    chunk boundaries, so text is released immediately except for a trailing run
    that is still a viable prefix of the opening tag.
    """

    _OPEN, _CLOSE = "<tool_call>", "</tool_call>"

    def __init__(self) -> None:
        self.buf = ""
        self.in_call = False
        self.captured: list[str] = []   # raw bodies, for parsing at the end

    @staticmethod
    def _partial_len(s: str, tag: str) -> int:
        """Length of the longest proper prefix of `tag` that `s` ends with."""
        for k in range(min(len(s), len(tag) - 1), 0, -1):
            if s.endswith(tag[:k]):
                return k
        return 0

    def push(self, chunk: str) -> str:
        if not chunk:
            return ""
        self.buf += chunk
        out: list[str] = []
        while True:
            if self.in_call:
                idx = self.buf.find(self._CLOSE)
                if idx == -1:
                    break                      # keep buffering the call body
                self.captured.append(self.buf[:idx])
                self.buf = self.buf[idx + len(self._CLOSE):]
                self.in_call = False
                continue
            idx = self.buf.find(self._OPEN)
            if idx == -1:
                hold = self._partial_len(self.buf, self._OPEN)
                cut = len(self.buf) - hold
                if cut > 0:
                    out.append(self.buf[:cut])
                    self.buf = self.buf[cut:]
                break
            out.append(self.buf[:idx])
            self.buf = self.buf[idx + len(self._OPEN):]
            self.in_call = True
        return "".join(out)

    def flush(self) -> str:
        """Release any held-back prose once generation has finished."""
        if self.in_call:
            self.buf = ""      # truncated mid-call: emit nothing
            return ""
        tail, self.buf = self.buf, ""
        return tail

    def calls(self) -> list[ParsedCall]:
        """Tool calls captured during the stream."""
        out: list[ParsedCall] = []
        for body in self.captured:
            obj = _load_call(body.strip())
            if obj:
                out.append(ParsedCall(obj["name"], obj["arguments"], body))
        return out
