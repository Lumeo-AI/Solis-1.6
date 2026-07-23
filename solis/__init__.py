"""Solis 1.9 — a branded, fine-tuned build of open foundation models (Qwen3),
with image and voice analysis and a deferred image-generation hook."""

from .registry import (
    Variant, Modality, Deployment, get_variant, variants,
    ALL_VARIANTS, DEFAULT_TEXT, DEFAULT_VISION, DEFAULT_VOICE,
)
from .identity import ASSISTANT_NAME, PRODUCT_VERSION, build_system_prompt

__all__ = [
    "Variant", "Modality", "Deployment", "get_variant", "variants",
    "ALL_VARIANTS", "DEFAULT_TEXT", "DEFAULT_VISION", "DEFAULT_VOICE",
    "ASSISTANT_NAME", "PRODUCT_VERSION", "build_system_prompt",
]
__version__ = "1.9.0"
