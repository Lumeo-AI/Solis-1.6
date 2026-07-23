"""Image analysis for Solis 1.9 — vision-language via Qwen2.5-VL.

Unlike the text engine, a vision model needs both the image and the text in the
same forward pass, so this is its own model rather than a front end. It loads
Qwen2.5-VL (lazy, on first use — not cached by default) and answers questions
about images while presenting as Solis.

Held separate from the text engine so a text-only deployment never pays the
vision model's memory, and so the two can run on different variants/precisions.
"""

from __future__ import annotations

import base64
import io
from typing import Optional, Union

from .registry import Variant, get_variant
from .identity import build_system_prompt


class VisionEngine:
    def __init__(self, model, processor, variant: Variant, device: str,
                 quantized: bool):
        self.model = model
        self.processor = processor
        self.variant = variant
        self.device = device
        self.quantized = quantized

    @classmethod
    def load(cls, variant_name: str = "solis-1.9-vision",
             load_in_4bit: Optional[bool] = None,
             dtype: str = "bfloat16") -> "VisionEngine":
        import torch
        from transformers import AutoProcessor

        variant = get_variant(variant_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        want_4bit = (variant.recommended_load == "nf4"
                     if load_in_4bit is None else load_in_4bit) and device == "cuda"
        torch_dtype = getattr(torch, dtype) if device == "cuda" else torch.float32

        kwargs: dict = {"torch_dtype": torch_dtype}
        if want_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = "auto"
        elif device == "cuda":
            kwargs["device_map"] = "auto"

        # Qwen2.5-VL uses a dedicated model class.
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as VLModel

        print(f"loading {variant.name}  <-  {variant.base_repo}"
              f"{'  [4-bit nf4]' if want_4bit else f'  [{dtype}]'}")
        processor = AutoProcessor.from_pretrained(variant.base_repo)
        model = VLModel.from_pretrained(variant.base_repo, **kwargs)
        if device != "cuda":
            model = model.to(device)
        model.eval()
        return cls(model, processor, variant, device, want_4bit)

    def describe(self, images: list, prompt: str, system: Optional[str] = None,
                 max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Answer `prompt` about one or more images (PIL images or sources)."""
        import torch

        pil = [_to_pil(im) for im in images]
        content = [{"type": "image", "image": im} for im in pil]
        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": build_system_prompt(system)},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=pil, padding=True,
                                return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None)
        trimmed = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()

    def info(self) -> dict:
        return {
            "model": self.variant.name,
            "base_model": self.variant.base_repo,
            "attribution": self.variant.license_note,
            "quantized_4bit": self.quantized,
            "device": self.device,
        }


def _to_pil(source: Union[str, bytes, "object"]):
    """Coerce a data-URI / base64 / path / bytes / PIL image into a PIL image."""
    from PIL import Image

    if hasattr(source, "convert"):        # already a PIL image
        return source.convert("RGB")
    if isinstance(source, str):
        if source.startswith("data:"):
            raw = base64.b64decode(source.split(",", 1)[1])
        elif _looks_base64(source):
            raw = base64.b64decode(source)
        else:
            return Image.open(source).convert("RGB")
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(io.BytesIO(source)).convert("RGB")


def _looks_base64(s: str) -> bool:
    if len(s) < 64 or any(c.isspace() for c in s[:64]):
        return False
    import string
    ok = set(string.ascii_letters + string.digits + "+/=")
    return all(c in ok for c in s[:64])
