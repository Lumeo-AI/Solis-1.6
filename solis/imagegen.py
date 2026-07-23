"""Image generation for Solis 1.9 — capability hook (deferred).

Per the current scope, the *slot* exists so the server and API know Solis can be
asked to draw, but no diffusion model is wired in yet. When enabled, this would
load SDXL (or Flux) through `diffusers` and return a generated image.

`available()` tells callers whether it's live; `generate()` raises a clear,
actionable error until the dependency and a model are in place, rather than
pretending to work.
"""

from __future__ import annotations

from typing import Optional

from .registry import get_variant

_NOT_WIRED = (
    "Image generation is not wired up yet — it's a deferred capability. To "
    "enable it: `pip install diffusers`, then implement ImageGenerator.load() "
    "to build a diffusers pipeline for the solis-1.9-draw variant (SDXL). The "
    "registry slot and this interface are ready for it."
)


def available() -> bool:
    """True only once a diffusion backend is actually installed and wired."""
    try:
        import diffusers  # noqa: F401
    except ImportError:
        return False
    # The dependency alone isn't enough — load() is still a stub below.
    return False


class ImageGenerator:
    """Interface a future diffusion backend fills in."""

    def __init__(self, pipe, variant):
        self.pipe = pipe
        self.variant = variant

    @classmethod
    def load(cls, variant_name: str = "solis-1.9-draw") -> "ImageGenerator":
        get_variant(variant_name)  # validate the name exists
        raise NotImplementedError(_NOT_WIRED)

    def generate(self, prompt: str, negative_prompt: Optional[str] = None,
                 width: int = 1024, height: int = 1024, steps: int = 30,
                 seed: Optional[int] = None) -> bytes:
        raise NotImplementedError(_NOT_WIRED)


def status() -> dict:
    return {
        "capability": "image_gen",
        "wired": available(),
        "variant": "solis-1.9-draw",
        "note": _NOT_WIRED if not available() else "ready",
    }
