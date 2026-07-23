# Solis 1.9 — Model Card

Solis 1.9 is a **branded, fine-tuned build of open foundation models**, not a
model trained from scratch. Each Solis variant is a specific base model served
(and optionally LoRA-fine-tuned) under the Solis identity.

## Attribution & licensing

Solis 1.9 is **built on Qwen2.5** (Alibaba Cloud) for its text and vision
variants, **OpenAI Whisper** for voice, and (when enabled) **SDXL** for image
generation. Rebranding to "Solis" does not remove these upstream obligations —
they are disclosed here and in `/health` as the licenses require.

| Solis variant | Base model | License |
| --- | --- | --- |
| solis-1.9-nano | Qwen/Qwen2.5-1.5B-Instruct | Apache-2.0 |
| solis-1.9-mini | Qwen/Qwen2.5-3B-Instruct | Qwen License |
| solis-1.9-small | Qwen/Qwen2.5-7B-Instruct | Apache-2.0 |
| solis-1.9-base | Qwen/Qwen2.5-14B-Instruct | Apache-2.0 |
| solis-1.9 (flagship) | Qwen/Qwen2.5-32B-Instruct | Apache-2.0 |
| solis-1.9-max | Qwen/Qwen2.5-72B-Instruct | Qwen License |
| solis-1.9-vision(-mini/-max) | Qwen/Qwen2.5-VL-{3B,7B,32B}-Instruct | Qwen License |
| solis-1.9-voice(-fast) | openai/whisper-large-v3-turbo / distil-whisper | MIT |
| solis-1.9-draw *(deferred)* | stabilityai/stable-diffusion-xl-base-1.0 | OpenRAIL++-M |

You must comply with each upstream license when distributing Solis. The Apache-2.0
and Qwen licenses permit derivative/rebranded use with attribution; keep this
model card (or equivalent notice) with any distribution.

## What Solis 1.9 is

- **Text**: a Qwen2.5 instruct model, served with the Solis identity (hardcoded
  in `solis/identity.py`), and optionally a Solis LoRA adapter
  (`finetune/lora_finetune.py`) that bakes identity and domain behaviour in.
- **Image analysis**: Qwen2.5-VL answers questions about images.
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
Optional supervised fine-tuning uses QLoRA (4-bit base + low-rank adapters) on
datasets normalised by `data/ingest.py`. See the repository README for exact
commands.
