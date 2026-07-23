"""The Solis 1.9 text engine — loads a Qwen2.5 model and generates as Solis.

This is the core of the rebuilt Solis: instead of a from-scratch network, we
load a strong open base model through `transformers`, optionally in 4-bit so the
larger variants fit a 16 GB card, optionally with a Solis LoRA adapter on top,
and generate with the Solis identity applied.

Streaming uses `TextIteratorStreamer`, so the server can forward tokens as they
are produced exactly as before.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from .registry import Variant, get_variant
from .identity import build_system_prompt, strip_identity_leak


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 20
    repetition_penalty: float = 1.05
    seed: Optional[int] = None


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    def load(cls, variant_name: str = "solis-1.9-small",
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
    def _render(self, messages: list[dict]) -> str:
        """Apply the chat template with the Solis system prompt injected."""
        msgs = [dict(m) for m in messages]
        # Pull out an existing system message (if any) and rebuild it as Solis.
        system = None
        if msgs and msgs[0].get("role") == "system":
            system = msgs.pop(0).get("content")
        msgs.insert(0, {"role": "system",
                        "content": build_system_prompt(system)})
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    def _inputs(self, messages: list[dict]):
        text = self._render(messages)
        enc = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in enc.items()}

    # -- generation ------------------------------------------------------- #
    def generate(self, messages: list[dict], cfg: GenerationConfig) -> str:
        inputs = self._inputs(messages)
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        with self._lock, torch.inference_mode():
            out = self.model.generate(
                **inputs, **self._gen_kwargs(cfg))
        new = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return strip_identity_leak(text)

    def stream(self, messages: list[dict],
               cfg: GenerationConfig) -> Iterator[str]:
        """Yield decoded text chunks as they are generated."""
        from transformers import TextIteratorStreamer

        inputs = self._inputs(messages)
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
        for chunk in streamer:
            if chunk:
                yield strip_identity_leak(chunk)
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
            kw.update(temperature=cfg.temperature, top_p=cfg.top_p,
                      top_k=cfg.top_k)
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
