# Solis 1.9 — Model Card

Solis 1.9 is a **branded, fine-tuned build of open foundation models**, not a
model trained from scratch. Each Solis variant is a specific base model served
(and optionally LoRA-fine-tuned) under the Solis identity.

## Attribution & licensing

Solis 1.9 is **built on Qwen3** (Alibaba Group) for its text and vision
variants, **OpenAI Whisper** for voice, and (when enabled) **SDXL** for image
generation. Rebranding to "Solis" does not remove these upstream obligations —
they are disclosed here and in `/health` as the licenses require.

| Solis variant | Base model | License |
| --- | --- | --- |
| solis-1.9-nano | Qwen/Qwen3-1.7B | Apache-2.0 |
| solis-1.9-mini | Qwen/Qwen3-4B | Apache-2.0 |
| solis-1.9 *(default)* | Qwen/Qwen3-8B | Apache-2.0 |
| solis-1.9-base | Qwen/Qwen3-14B | Apache-2.0 |
| solis-1.9-large | Qwen/Qwen3-32B | Apache-2.0 |
| solis-1.9-max | Qwen/Qwen3.6-27B | Apache-2.0 |
| solis-1.9-moe | Qwen/Qwen3.6-35B-A3B | Apache-2.0 |
| solis-1.9-vision(-mini/-max) | Qwen/Qwen3-VL-{4B,8B,32B}-Instruct | Apache-2.0 |
| solis-1.9-voice(-fast) | openai/whisper-large-v3-turbo / distil-whisper | MIT |
| solis-1.9-draw *(deferred)* | stabilityai/stable-diffusion-xl-base-1.0 | OpenRAIL++-M |

You must comply with each upstream license when distributing Solis. The Apache-2.0
and Qwen licenses permit derivative/rebranded use with attribution; keep this
model card (or equivalent notice) with any distribution.

## What Solis 1.9 is

- **Text**: a Qwen3 instruct model, served with the Solis identity (hardcoded
  in `solis/identity.py`), and optionally a Solis LoRA adapter
  (`finetune/lora_finetune.py`) that bakes identity and domain behaviour in.
  Qwen3's hybrid thinking mode is exposed per request and off by default.
- **Image analysis**: Qwen3-VL answers questions about images.
- **Voice analysis**: Whisper transcribes speech; the transcript flows into the
  text model.
- **Image generation**: a declared capability slot, **not yet wired** (needs
  `diffusers` + a diffusion model).

## Intended use

General assistant use — chat, reasoning, coding help, and analysis of user-
provided images and audio. Fine-tune with your own data to specialise it.

## Limitations & responsible use

- Inherits the base models' limitations: it can be confidently wrong, has a
  training cutoff, and has no inherent access to real-time information.
- Fine-tuning changes behaviour; evaluate any Solis adapter before relying on it.
- Voice transcription accuracy depends on audio quality and language.
- The Solis identity is applied by code (and, if fine-tuned, by adapter); it
  does not change the underlying model's knowledge or license.

## How it was built

No pretraining. Variants map to published base checkpoints (see the table).
Optional supervised fine-tuning uses QLoRA (4-bit base + low-rank adapters),
run through Unsloth where available, on datasets normalised by
`data/ingest.py`. See the repository README for exact commands.

Sizing note: memory for a Mixture-of-Experts variant is set by **total**
parameters, not active ones — every expert stays resident. `solis-1.9-moe`
(Qwen3.6-35B-A3B) activates only 3B per token but still needs ~19-21 GB of
4-bit weights, so it is a cloud variant despite the small active count.
