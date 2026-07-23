"""The Solis 1.9 model ladder.

Solis 1.9 is not trained from scratch. Each variant is a specific open
foundation model — the Qwen3 family — served (and, where we fine-tune, adapted)
under the Solis brand. This registry is the single source of truth for which
Solis name maps to which base model, what it needs to run, what it needs to
*train*, and where each is meant to live.

Keeping this as data (not scattered constants) means the server, the fine-tune
scripts, and the docs all agree on the ladder, and adding a variant is a one-line
change. If Qwen ships a new size, add a row — nothing else needs to know.

The 16 GB rule
--------------
Two different budgets matter and they are not the same number:

  * **serving** — 4-bit weights + KV cache + activations. Fits 16 GB well below
    ~10 GB of weights.
  * **training** (QLoRA) — the same 4-bit weights *stay resident* while LoRA
    adapters, their gradients, a paged 8-bit optimizer and checkpointed
    activations pile on top. Budget roughly weights + 5 GB at seq 2048.

For a Mixture-of-Experts model, memory is driven by **total** parameters, not
active ones: every expert must be resident even though only a few fire per
token. `Qwen3.6-35B-A3B` is only 3B-active but still needs ~19 GB of 4-bit
weights — which is why it is CLOUD here despite the small active count.

Attribution: these are Qwen3 models (Alibaba Cloud/Group), Apache-2.0.
`license_note` records the per-variant obligation and the model card surfaces
it — rebranding to Solis does not remove the requirement to say what it is
built on.
"""

from __future__ import annotations

from dataclasses import dataclass
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

# Headroom a QLoRA run needs on top of the frozen 4-bit weights: LoRA params and
# their grads, paged optimizer state, and checkpointed activations. Scales mildly
# with sequence length; this is the seq-2048 figure and is deliberately generous.
_QLORA_OVERHEAD_GB = 5.0


@dataclass(frozen=True)
class Variant:
    """One Solis 1.9 variant and the base model behind it."""

    name: str                 # Solis-facing name, e.g. "solis-1.9-base"
    base_repo: str            # Hugging Face repo of the base model
    modality: Modality
    params_b: float           # billions of parameters (TOTAL, incl. all experts)
    context: int              # native context window (tokens)
    deployment: Deployment
    aliases: tuple[str, ...] = ()
    license: str = "apache-2.0"
    license_note: str = "Built on Qwen3 (Alibaba Group), Apache-2.0."
    notes: str = ""
    recommended_load: str = "bf16"   # bf16 | nf4 (4-bit) — the default we serve
    active_b: float | None = None    # billions active per token (MoE only)
    thinking: bool = False           # supports Qwen3 hybrid thinking mode

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
        # MoE KV scales with active size, so use it when we have it.
        kv_scale = self.active_b or self.params_b
        overhead = 1.0 + (kv_scale * 0.12) + (ctx / 8192) * 1.2
        return round(weights + overhead, 1)

    def training_vram_gb(self, seq_len: int = 2048) -> float:
        """Rough VRAM for a 4-bit QLoRA fine-tune of this variant."""
        weights = self.weight_gb("nf4")
        return round(weights + _QLORA_OVERHEAD_GB * (seq_len / 2048), 1)

    def fits_16gb(self, precision: str | None = None) -> bool:
        return self.serving_vram_gb(precision) <= 16.0

    def trainable_16gb(self, seq_len: int = 2048) -> bool:
        return self.training_vram_gb(seq_len) <= 16.0

    def trainable_24gb(self, seq_len: int = 2048) -> bool:
        return self.training_vram_gb(seq_len) <= 24.0


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #
# Text LLMs — the core Solis chat models. `solis-1.9` (Qwen3-8B) is the default:
# it is the largest Qwen3 that QLoRA-trains inside 16 GB *and* still decodes fast
# enough to feel instant in day-to-day chat.
#
# Qwen3 dense sizes: 0.6B, 1.7B, 4B, 8B, 14B, 32B.
# Qwen3.6 (2026) ships 27B dense and 35B-A3B MoE — both cloud-side here.
TEXT_VARIANTS: list[Variant] = [
    Variant("solis-1.9-nano", "Qwen/Qwen3-1.7B", Modality.TEXT,
            1.7, 32768, Deployment.LOCAL, aliases=("nano",), thinking=True,
            notes="Very fast; runs on almost anything, even CPU."),
    Variant("solis-1.9-mini", "Qwen/Qwen3-4B", Modality.TEXT,
            4.0, 32768, Deployment.LOCAL, aliases=("mini",), thinking=True,
            notes="Comfortable on a 16 GB card in bf16; trains easily."),

    # ---- THE DEFAULT ---------------------------------------------------- #
    # 4-bit weights ~4.5 GB. QLoRA fits 16 GB with room to spare, and decoding
    # is quick enough that chat feels instant. This is the "actually useful and
    # blazing fast on your card" pick.
    Variant("solis-1.9", "Qwen/Qwen3-8B", Modality.TEXT,
            8.2, 131072, Deployment.LOCAL,
            aliases=("small", "8b", "default"), recommended_load="nf4",
            thinking=True,
            notes="DEFAULT. Largest Qwen3 that QLoRA-trains on 16 GB and still "
                  "decodes fast. 128K context."),

    # ---- THE STRETCH ---------------------------------------------------- #
    # Trains on a rented 24 GB card, still SERVES on 16 GB at 4-bit (~8 GB).
    Variant("solis-1.9-base", "Qwen/Qwen3-14B", Modality.TEXT,
            14.8, 131072, Deployment.LOCAL, aliases=("base", "14b"),
            recommended_load="nf4", thinking=True,
            notes="Serves on 16 GB in 4-bit; train it on a rented 24 GB GPU "
                  "(tight but possible on 16 GB at seq 1024)."),

    Variant("solis-1.9-large", "Qwen/Qwen3-32B", Modality.TEXT,
            32.8, 131072, Deployment.CLOUD, aliases=("large", "32b"),
            recommended_load="nf4", thinking=True,
            notes="Cloud. 4-bit weights ~17 GB — over the 16 GB serving line."),

    # Qwen3.6 generation. Both are cloud-side on a 16 GB budget.
    Variant("solis-1.9-max", "Qwen/Qwen3.6-27B", Modality.TEXT,
            27.0, 262144, Deployment.CLOUD, aliases=("max", "27b"),
            recommended_load="nf4",
            license_note="Built on Qwen3.6 (Alibaba Group), Apache-2.0.",
            notes="Qwen3.6 dense flagship, 256K context. ~15 GB at 4-bit — "
                  "serving is marginal on 16 GB, training needs the cloud."),
    Variant("solis-1.9-moe", "Qwen/Qwen3.6-35B-A3B", Modality.TEXT,
            35.0, 262144, Deployment.CLOUD, aliases=("moe", "35b-a3b"),
            recommended_load="nf4", active_b=3.0,
            license_note="Built on Qwen3.6 (Alibaba Group), Apache-2.0.",
            notes="35B total / 3B active, 256K ctx. Only 3B fires per token, "
                  "but ALL experts stay resident: ~19 GB at 4-bit. Cloud only — "
                  "it cannot serve or train on a 16 GB card."),
]

# Vision-language — image analysis.
VISION_VARIANTS: list[Variant] = [
    Variant("solis-1.9-vision-mini", "Qwen/Qwen3-VL-4B-Instruct",
            Modality.VISION, 4.2, 131072, Deployment.LOCAL,
            aliases=("vision-mini",),
            license_note="Built on Qwen3-VL (Alibaba Group), Apache-2.0.",
            notes="Image analysis on a 16 GB card."),
    Variant("solis-1.9-vision", "Qwen/Qwen3-VL-8B-Instruct",
            Modality.VISION, 8.4, 131072, Deployment.LOCAL,
            aliases=("vision", "vision-small"), recommended_load="nf4",
            license_note="Built on Qwen3-VL (Alibaba Group), Apache-2.0.",
            notes="Image analysis; comfortable on 16 GB in 4-bit."),
    Variant("solis-1.9-vision-max", "Qwen/Qwen3-VL-32B-Instruct",
            Modality.VISION, 33.0, 131072, Deployment.CLOUD,
            aliases=("vision-max",),
            license_note="Built on Qwen3-VL (Alibaba Group), Apache-2.0.",
            notes="High-fidelity image analysis, cloud."),
]

# Voice — speech understanding. Whisper transcribes; the text then flows into a
# Solis text variant.
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

DEFAULT_TEXT = "solis-1.9"          # Qwen3-8B
DEFAULT_VISION = "solis-1.9-vision"
DEFAULT_VOICE = "solis-1.9-voice"

# What the local fine-tune script reaches for by default, and its fallback.
DEFAULT_TRAIN = "solis-1.9"         # Qwen3-8B  — trains on 16 GB
STRETCH_TRAIN = "solis-1.9-base"    # Qwen3-14B — trains on a rented 24 GB


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
    serve = v.serving_vram_gb()
    fit = "16GB" if v.fits_16gb() else "cloud"
    if v.modality is Modality.TEXT:
        train = v.training_vram_gb()
        tr = ("16GB" if v.trainable_16gb() else
              "24GB" if v.trainable_24gb() else "cloud")
        size = f"{v.params_b:5.1f}B" + (f"/A{v.active_b:g}B" if v.active_b else "")
        return (f"{v.name:<22} {v.base_repo:<28} {size:<12} "
                f"{v.recommended_load:>4}  serve ~{serve:4.1f}GB {fit:<5} "
                f"train ~{train:4.1f}GB {tr}")
    return (f"{v.name:<22} {v.base_repo:<28} {v.params_b:5.1f}B "
            f"{v.recommended_load:>4}  serve ~{serve:4.1f}GB {fit}")


def print_ladder() -> None:
    for title, group in [("TEXT", TEXT_VARIANTS), ("VISION", VISION_VARIANTS),
                         ("VOICE", VOICE_VARIANTS),
                         ("IMAGE GEN (deferred)", IMAGE_GEN_VARIANTS)]:
        print(f"\n{title}")
        print("-" * 116)
        for v in group:
            print("  " + _fmt(v))


if __name__ == "__main__":  # pragma: no cover
    print("Solis 1.9 — model ladder (base: Qwen3 / Qwen3-VL / Whisper / SDXL)")
    print_ladder()
    print(f"\ndefault serve/train: {DEFAULT_TEXT}  "
          f"(stretch: {STRETCH_TRAIN} on a rented 24 GB GPU)")
