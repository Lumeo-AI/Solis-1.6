"""The Solis 1.9 text engine — loads a Qwen3 model and generates as Solis.

This is the core of the rebuilt Solis: instead of a from-scratch network, we
load a strong open base model through `transformers`, optionally in 4-bit so the
larger variants fit a 16 GB card, optionally with a Solis LoRA adapter on top,
and generate with the Solis identity applied.

Streaming uses `TextIteratorStreamer`, so the server can forward tokens as they
are produced exactly as before.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from .registry import Variant, get_variant
from .identity import build_system_prompt, strip_identity_leak
from .toolcall import (ParsedCall, normalise_tools, parse_tool_calls,
                       strip_tool_calls)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    seed: Optional[int] = None
    # Qwen3 hybrid reasoning. OFF by default: thinking mode spends hundreds of
    # tokens on a <think> block before answering, which is the difference
    # between "instant" and "why is it still going" in day-to-day chat. Turn it
    # on per-request for hard problems.
    thinking: bool = False
    # Qwen's own recommended sampling differs per mode; when a caller leaves the
    # defaults alone we apply the right preset for the mode being used.
    THINKING_PRESET = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    NON_THINKING_PRESET = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove a completed `<think>...</think>` block from a finished answer."""
    out = _THINK_RE.sub("", text)
    # An unterminated block (hit the token limit mid-thought) leaves a dangling
    # opener; drop everything from it rather than showing the scratchpad.
    if "<think>" in out:
        out = out.split("<think>", 1)[0]
    return out.strip()


class _ThinkGate:
    """Streaming filter that withholds `<think>` content, chunk by chunk.

    Chunks arrive at arbitrary boundaries, so a tag can be split across two of
    them ("<thi" + "nk>"). Text is released immediately except for a trailing
    run that is still a viable prefix of a tag — that is held until the next
    chunk resolves it. `flush()` releases whatever is left when the stream ends.
    """

    _OPEN, _CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buf = ""
        self.in_think = False

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
            if self.in_think:
                idx = self.buf.find(self._CLOSE)
                if idx == -1:
                    # Discard thought content, but keep a possible partial tag.
                    hold = self._partial_len(self.buf, self._CLOSE)
                    self.buf = self.buf[len(self.buf) - hold:] if hold else ""
                    break
                self.buf = self.buf[idx + len(self._CLOSE):]
                self.in_think = False
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
            self.in_think = True
        return "".join(out)

    def flush(self) -> str:
        """Release any held-back tail once generation has finished."""
        if self.in_think:
            self.buf = ""
            return ""
        tail, self.buf = self.buf, ""
        return tail


@dataclass
class ChatResult:
    """A finished non-streaming turn, with any tool calls split out."""

    content: str
    tool_calls: list[ParsedCall]

    @property
    def finish_reason(self) -> str:
        return "tool_calls" if self.tool_calls else "stop"


def _should_4bit(variant: Variant, device: str, override: Optional[bool]) -> bool:
    if override is not None:
        return override
    if device != "cuda":
        return False
    return variant.recommended_load == "nf4"


class SolisEngine:
    """A loaded Solis 1.9 text model."""

    def __init__(self, model, tokenizer, variant: Variant, device: str,
                 quantized: bool, adapter: Optional[str]):
        self.model = model
        self.tokenizer = tokenizer
        self.variant = variant
        self.device = device
        self.quantized = quantized
        self.adapter = adapter
        self._lock = threading.Lock()

    # -- loading ---------------------------------------------------------- #
    @classmethod
    def load(cls, variant_name: str = "solis-1.9",
             load_in_4bit: Optional[bool] = None,
             adapter_path: Optional[str] = None,
             dtype: str = "bfloat16",
             device_map: str = "auto") -> "SolisEngine":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        variant = get_variant(variant_name)
        device = _pick_device()
        want_4bit = _should_4bit(variant, device, load_in_4bit)

        torch_dtype = getattr(torch, dtype) if device == "cuda" else torch.float32
        kwargs: dict = {"torch_dtype": torch_dtype}

        if want_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = device_map
        elif device == "cuda":
            kwargs["device_map"] = device_map

        print(f"loading {variant.name}  <-  {variant.base_repo}"
              f"{'  [4-bit nf4]' if want_4bit else f'  [{dtype}]'}")
        tok = AutoTokenizer.from_pretrained(variant.base_repo)
        model = AutoModelForCausalLM.from_pretrained(variant.base_repo, **kwargs)

        if adapter_path:
            from peft import PeftModel
            print(f"applying Solis adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)

        if device != "cuda":
            model = model.to(device)
        model.eval()
        return cls(model, tok, variant, device, want_4bit, adapter_path)

    # -- prompt ----------------------------------------------------------- #
    def _render(self, messages: list[dict], thinking: bool = False,
                tools: Optional[list[dict]] = None) -> str:
        """Apply the chat template with the Solis system prompt injected.

        Qwen3's template takes an `enable_thinking` flag that switches the model
        between its reasoning and direct-answer modes, and a `tools` list it
        renders into the prompt itself — that is what teaches the model the
        `<tool_call>` grammar it was trained on, so we never hand-roll one.
        Older templates accept neither kwarg, so we degrade gracefully.
        """
        msgs = [dict(m) for m in messages]
        # Pull out an existing system message (if any) and rebuild it as Solis.
        system = None
        if msgs and msgs[0].get("role") == "system":
            system = msgs.pop(0).get("content")
        msgs.insert(0, {"role": "system",
                        "content": build_system_prompt(system)})
        base: dict = {"tokenize": False, "add_generation_prompt": True}
        # Most specific first; drop one unsupported kwarg at a time rather than
        # failing the request because a template is older than we assumed.
        attempts: list[dict] = []
        if tools:
            attempts.append({**base, "enable_thinking": thinking, "tools": tools})
            attempts.append({**base, "tools": tools})
        attempts.append({**base, "enable_thinking": thinking})
        attempts.append(base)
        for kw in attempts:
            try:
                return self.tokenizer.apply_chat_template(msgs, **kw)
            except (TypeError, ValueError):
                continue
        return self.tokenizer.apply_chat_template(msgs, **base)

    def _inputs(self, messages: list[dict], thinking: bool = False,
                tools: Optional[list[dict]] = None):
        text = self._render(messages, thinking, tools)
        enc = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in enc.items()}

    # -- tool-aware turn --------------------------------------------------- #
    def chat(self, messages: list[dict], cfg: GenerationConfig,
             tools: Optional[list[dict]] = None) -> "ChatResult":
        """One complete turn, with any tool calls parsed out of the text.

        This is the entry point the agent loop uses (`solis/tools.py`): it needs
        the whole turn before it can tell whether the model asked for a tool, so
        this path is deliberately non-streaming.
        """
        tools = normalise_tools(tools)
        inputs = self._inputs(messages, cfg.thinking, tools)
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        with self._lock, torch.inference_mode():
            out = self.model.generate(**inputs, **self._gen_kwargs(cfg))
        new = out[0, inputs["input_ids"].shape[1]:]
        text = strip_think(self.tokenizer.decode(new, skip_special_tokens=True))
        calls = parse_tool_calls(text) if tools else []
        content = strip_tool_calls(text) if calls else text
        return ChatResult(strip_identity_leak(content), calls)

    # -- generation ------------------------------------------------------- #
    def generate(self, messages: list[dict], cfg: GenerationConfig) -> str:
        inputs = self._inputs(messages, cfg.thinking)
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        with self._lock, torch.inference_mode():
            out = self.model.generate(
                **inputs, **self._gen_kwargs(cfg))
        new = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return strip_identity_leak(strip_think(text))

    def stream(self, messages: list[dict],
               cfg: GenerationConfig) -> Iterator[str]:
        """Yield decoded text chunks as they are generated.

        A `<think>` block (thinking mode) is withheld from the stream — callers
        get the answer, not the scratchpad.
        """
        from transformers import TextIteratorStreamer

        inputs = self._inputs(messages, cfg.thinking)
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        kwargs = {**inputs, **self._gen_kwargs(cfg), "streamer": streamer}

        def worker():
            with self._lock, torch.inference_mode():
                self.model.generate(**kwargs)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        gate = _ThinkGate()
        for chunk in streamer:
            visible = gate.push(chunk)
            if visible:
                yield strip_identity_leak(visible)
        tail = gate.flush()
        if tail:
            yield strip_identity_leak(tail)
        thread.join()

    def _gen_kwargs(self, cfg: GenerationConfig) -> dict:
        do_sample = cfg.temperature > 0
        kw = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": cfg.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id
            or self.tokenizer.eos_token_id,
        }
        if do_sample:
            # Thinking and non-thinking modes want different sampling; Qwen's
            # published presets avoid the repetition/degradation each mode is
            # prone to under the other's settings.
            preset = (GenerationConfig.THINKING_PRESET if cfg.thinking
                      else GenerationConfig.NON_THINKING_PRESET)
            kw.update(temperature=cfg.temperature or preset["temperature"],
                      top_p=cfg.top_p, top_k=cfg.top_k)
        return kw

    # -- introspection ---------------------------------------------------- #
    def info(self) -> dict:
        vram = None
        if self.device == "cuda":
            vram = {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1024 ** 3, 2),
            }
        return {
            "model": self.variant.name,
            "base_model": self.variant.base_repo,
            "attribution": self.variant.license_note,
            "params_b": self.variant.params_b,
            "context": self.variant.context,
            "quantized_4bit": self.quantized,
            "adapter": self.adapter,
            "device": self.device,
            "vram": vram,
        }
