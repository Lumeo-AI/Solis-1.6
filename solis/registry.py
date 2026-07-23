"""The Solis 1.9 model ladder.

Solis 1.9 is not trained from scratch. Each variant is a specific open
foundation model — the Qwen2.5 family — served (and, where we fine-tune,
adapted) under the Solis brand. This registry is the single source of truth for
which Solis name maps to which base model, what it needs to run, and where it is
meant to run.

Keeping this as data (not scattered constants) means the server, the fine-tune
scripts, and the docs all agree on the ladder, and adding a variant is a one-line
change.

Attribution: these are Qwen2.5 models (Alibaba Cloud). Most sizes are Apache-2.0;
Qwen2.5-3B and -72B carry the separate Qwen license. `license_note` records the
per-variant obligation, and the model card surfaces it — rebranding to Solis does
not remove the requirement to say what it is built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Modality(str, Enum):
    TEXT = "text"       # a chat/instruct LLM
    VISION = "vision"   # image analysis (vision-language model)
    VOICE = "voice"     # speech understanding / transcription
    IMAGE_GEN = "image_gen"  # image generation (deferred)


class Deployment(str, Enum):
    LOCAL = "local"     # runs on a 16 GB consumer card (often via 4-bit)
    CLOUD = "cloud"     # needs a datacentre GPU / multi-GPU


# Bytes/param for weight storage at a given load precision.
_BYTES = {"bf16": 2.0, "fp16": 2.0, "int8": 1.0, "nf4": 0.55}


@dataclass(frozen=True)
class Variant:
    """One Solis 1.9 variant and the base model behind it."""

    name: str                 # Solis-facing name, e.g. "solis-1.9-small"
    base_repo: str            # Hugging Face repo of the base model
    modality: Modality
    params_b: float           # billions of parameters
    context: int              # native context window (tokens)
    deployment: Deployment
    aliases: tuple[str, ...] = ()
    license: str = "apache-2.0"
    license_note: str = "Built on Qwen2.5 (Alibaba Cloud), Apache-2.0."
    notes: str = ""
    recommended_load: str = "bf16"   # bf16 | nf4 (4-bit) — the default we serve

    # -- memory ---------------------------------------------------------- #
    def weight_gb(self, precision: str | None = None) -> float:
        p = precision or self.recommended_load
        return round(self.params_b * 1e9 * _BYTES[p] / 1024 ** 3, 1)

    def serving_vram_gb(self, precision: str | None = None,
                        ctx: int | None = None) -> float:
        """Rough VRAM to serve this variant: weights + a KV-cache/activation
        allowance. Deliberately generous so 'fits 16 GB' is not optimistic."""
        p = precision or self.recommended_load
        ctx = ctx or min(self.context, 8192)
        weights = self.weight_gb(p)
        # KV cache + activations scale with size and context; a simple, safe
        # linear allowance keeps this honest without pretending to be exact.
        overhead = 1.0 + (self.params_b * 0.12) + (ctx / 8192) * 1.2
        return round(weights + overhead, 1)

    def fits_16gb(self, precision: str | None = None) -> bool:
        return self.serving_vram_gb(precision) <= 16.0


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #
# Text LLMs — the core Solis chat models, small (runs at home) to flagship
# (cloud). Names are the product; base_repo is what actually loads.
TEXT_VARIANTS: list[Variant] = [
    Variant("solis-1.9-nano", "Qwen/Qwen2.5-1.5B-Instruct", Modality.TEXT,
            1.5, 32768, Deployment.LOCAL, aliases=("nano",),
            notes="Fast, light; runs on almost anything, even CPU."),
    Variant("solis-1.9-mini", "Qwen/Qwen2.5-3B-Instruct", Modality.TEXT,
            3.1, 32768, Deployment.LOCAL, aliases=("mini",),
            license="qwen", license_note="Built on Qwen2.5-3B (Alibaba Cloud), "
            "Qwen license — see the model card.",
            notes="Comfortable on a 16 GB card in bf16."),
    Variant("solis-1.9-small", "Qwen/Qwen2.5-7B-Instruct", Modality.TEXT,
            7.6, 32768, Deployment.LOCAL, aliases=("small",),
            recommended_load="nf4",
            notes="16 GB card in 4-bit (~6 GB); bf16 needs ~20 GB."),
    Variant("solis-1.9-base", "Qwen/Qwen2.5-14B-Instruct", Modality.TEXT,
            14.7, 32768, Deployment.CLOUD, aliases=("base",),
            recommended_load="nf4",
            notes="4-bit is tight on 16 GB (~10 GB); comfortable in the cloud."),
    Variant("solis-1.9", "Qwen/Qwen2.5-32B-Instruct", Modality.TEXT,
            32.5, 32768, Deployment.CLOUD, aliases=("flagship",),
            notes="Flagship. Cloud GPU (A100/H100)."),
    Variant("solis-1.9-max", "Qwen/Qwen2.5-72B-Instruct", Modality.TEXT,
            72.7, 32768, Deployment.CLOUD, aliases=("max",),
            license="qwen", license_note="Built on Qwen2.5-72B (Alibaba Cloud), "
            "Qwen license — see the model card.",
            notes="Largest. Multi-GPU cloud."),
]

# Vision-language — image analysis. Qwen2.5-VL understands images (and video)
# alongside text.
VISION_VARIANTS: list[Variant] = [
    Variant("solis-1.9-vision-mini", "Qwen/Qwen2.5-VL-3B-Instruct",
            Modality.VISION, 3.7, 32768, Deployment.LOCAL,
            aliases=("vision-mini",), license="qwen",
            license_note="Built on Qwen2.5-VL-3B (Alibaba Cloud), Qwen license.",
            notes="Image analysis on a 16 GB card."),
    Variant("solis-1.9-vision", "Qwen/Qwen2.5-VL-7B-Instruct",
            Modality.VISION, 8.3, 32768, Deployment.LOCAL,
            aliases=("vision", "vision-small"), recommended_load="nf4",
            notes="Image analysis; 16 GB in 4-bit."),
    Variant("solis-1.9-vision-max", "Qwen/Qwen2.5-VL-32B-Instruct",
            Modality.VISION, 33.0, 32768, Deployment.CLOUD,
            aliases=("vision-max",),
            notes="High-fidelity image analysis, cloud."),
]

# Voice — speech understanding. Whisper transcribes; the text then flows into a
# Solis text variant. (Qwen2-Audio is an alternative for richer audio reasoning;
# it can be added here later.)
VOICE_VARIANTS: list[Variant] = [
    Variant("solis-1.9-voice", "openai/whisper-large-v3-turbo",
            Modality.VOICE, 0.8, 0, Deployment.LOCAL, aliases=("voice",),
            license="mit", license_note="Built on OpenAI Whisper (MIT).",
            notes="Speech-to-text front end for voice input."),
    Variant("solis-1.9-voice-fast", "distil-whisper/distil-large-v3",
            Modality.VOICE, 0.75, 0, Deployment.LOCAL, aliases=("voice-fast",),
            license="mit", license_note="Built on Distil-Whisper (MIT).",
            notes="Faster, lighter transcription."),
]

# Image generation — DEFERRED. Registered so the ladder and the API know the
# slot exists; no diffusion model is wired yet (see solis/imagegen.py).
IMAGE_GEN_VARIANTS: list[Variant] = [
    Variant("solis-1.9-draw", "stabilityai/stable-diffusion-xl-base-1.0",
            Modality.IMAGE_GEN, 3.5, 0, Deployment.LOCAL, aliases=("draw",),
            license="openrail++",
            license_note="Would be built on SDXL (CreativeML OpenRAIL++-M).",
            notes="NOT WIRED YET — capability hook only. Needs `diffusers`."),
]

ALL_VARIANTS: list[Variant] = (
    TEXT_VARIANTS + VISION_VARIANTS + VOICE_VARIANTS + IMAGE_GEN_VARIANTS)

_BY_NAME: dict[str, Variant] = {}
for _v in ALL_VARIANTS:
    _BY_NAME[_v.name] = _v
    for _a in _v.aliases:
        _BY_NAME[_a] = _v
    # Bare base-repo lookups are handy too.
    _BY_NAME.setdefault(_v.base_repo, _v)

DEFAULT_TEXT = "solis-1.9-small"
DEFAULT_VISION = "solis-1.9-vision"
DEFAULT_VOICE = "solis-1.9-voice"


def get_variant(name: str) -> Variant:
    """Resolve a Solis name, alias, or base repo to a Variant."""
    if name in _BY_NAME:
        return _BY_NAME[name]
    raise KeyError(
        f"unknown Solis variant {name!r}. Known: "
        f"{sorted({v.name for v in ALL_VARIANTS})}")


def variants(modality: Modality | None = None,
             deployment: Deployment | None = None) -> list[Variant]:
    out = ALL_VARIANTS
    if modality is not None:
        out = [v for v in out if v.modality == modality]
    if deployment is not None:
        out = [v for v in out if v.deployment == deployment]
    return out


def _fmt(v: Variant) -> str:
    vram = v.serving_vram_gb()
    fit = "16GB" if v.fits_16gb() else "cloud"
    return (f"{v.name:<24} {v.base_repo:<34} {v.params_b:5.1f}B "
            f"{v.recommended_load:>4}  ~{vram:4.1f}GB  {fit}")


def print_ladder() -> None:
    for title, group in [("TEXT", TEXT_VARIANTS), ("VISION", VISION_VARIANTS),
                         ("VOICE", VOICE_VARIANTS),
                         ("IMAGE GEN (deferred)", IMAGE_GEN_VARIANTS)]:
        print(f"\n{title}")
        print("-" * 92)
        for v in group:
            print("  " + _fmt(v))


if __name__ == "__main__":  # pragma: no cover
    print("Solis 1.9 — model ladder (base: Qwen2.5 / Whisper / SDXL)")
    print_ladder()
