# Solis 1.9 ☀️

A branded, **fine-tuned multimodal assistant** built on open foundation models.
Solis 1.9 loads a strong base model (the Qwen2.5 family), serves it under the
Solis identity, understands **images** and **voice**, and can be **fine-tuned**
with your own data via QLoRA. Image generation is a declared, deferred
capability.

> Solis 1.9 is a rebrand/fine-tune, **not** a from-scratch model. It is built on
> Qwen2.5 (text + vision), Whisper (voice), and SDXL (image gen, when enabled).
> See [MODEL_CARD.md](MODEL_CARD.md) for attribution and licensing — required by
> the upstream licenses. The earlier from-scratch Solis lives in [`legacy/`](legacy/).

The Python package is `solis`; the product version is 1.9.

## The ladder — one brand, many sizes

`python -m solis.registry` prints this; VRAM is the serving estimate.

```
TEXT
  solis-1.9-nano    Qwen2.5-1.5B-Instruct    1.5B  bf16   ~5 GB    runs anywhere
  solis-1.9-mini    Qwen2.5-3B-Instruct      3.1B  bf16   ~8 GB    16 GB card
  solis-1.9-small   Qwen2.5-7B-Instruct      7.6B  nf4    ~7 GB    16 GB card (4-bit)
  solis-1.9-base    Qwen2.5-14B-Instruct    14.7B  nf4   ~12 GB    16 GB (tight) / cloud
  solis-1.9         Qwen2.5-32B-Instruct    32.5B  bf16  ~67 GB    cloud (flagship)
  solis-1.9-max     Qwen2.5-72B-Instruct    72.7B  bf16 ~146 GB    cloud
VISION   solis-1.9-vision(-mini/-max)  →  Qwen2.5-VL-{3B,7B,32B}
VOICE    solis-1.9-voice(-fast)        →  Whisper large-v3-turbo / distil
IMAGE GEN  solis-1.9-draw  →  SDXL   (deferred — capability hook only)
```

The default served model is `solis-1.9-small` (Qwen2.5-7B in 4-bit) — it fits a
16 GB card in ~5–7 GB and is a genuinely capable assistant.

## Quickstart

```bash
pip install -r requirements.txt

# serve (loads Qwen2.5-7B in 4-bit as Solis; ~5 GB VRAM)
python serve.py
```

Serves at **http://localhost:8000** — chat on `POST /v1/chat/completions`
(OpenAI-compatible; base URL `http://localhost:8000/v1`).

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Who are you?"}]}'
```

Pick a different variant with `SOLIS_MODEL`:

```bash
SOLIS_MODEL=solis-1.9-mini python serve.py   # 3B, bf16, lighter
SOLIS_MODEL=solis-1.9-nano python serve.py    # 1.5B, even CPU-ok
```

## Image and voice

Send OpenAI-style content parts; the server routes images to Qwen2.5-VL and
audio to Whisper, folds their readings into the conversation, and the text model
answers. The vision/voice models load lazily on first use (they download once).

```jsonc
{ "messages": [{ "role": "user", "content": [
    { "type": "text", "text": "What's in this picture?" },
    { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } },
    { "type": "input_audio", "input_audio": { "data": "<base64 wav>", "format": "wav" } }
]}]}
```

`GET /health` reports which capabilities are loaded and the Qwen attribution.
`GET /v1/models` lists the Solis 1.9 variants.

Environment: `SOLIS_MODEL`, `SOLIS_ADAPTER` (a Solis LoRA), `SOLIS_4BIT`,
`SOLIS_VISION`, `SOLIS_VOICE`, `SOLIS_LOAD_VISION/VOICE=1` (preload), `PORT`.

## Fine-tuning — make it *your* Solis

QLoRA trains low-rank adapters on a frozen 4-bit base, so a 7B fine-tunes on a
16 GB card. Data can be any Hugging Face dataset or local JSONL — rows are
normalised by `data/ingest.py` (messages / conversations / prompt-response /
question-answer / raw text) and rendered with the base chat template. The Solis
identity is trained in by default.

```bash
python finetune/lora_finetune.py \
    --model solis-1.9-small \
    --hf teknium/OpenHermes-2.5 --max-samples 20000 \
    --output checkpoints/solis-small-lora

# serve the fine-tune
SOLIS_MODEL=solis-1.9-small SOLIS_ADAPTER=checkpoints/solis-small-lora python serve.py
```

To fine-tune the largest model that fits a 16 GB card on a streamed mix of
massive datasets (with automatic 14B→7B fallback), use the launcher:

```bash
nohup bash finetune/train_local.sh > logs/finetune.log 2>&1 &
tail -f logs/finetune.log
```

The adapter is tens of MB. On a big GPU, drop `--model` to a larger variant and
add flash-attention + `--packing` for speed.

## Image generation (deferred)

`solis/imagegen.py` defines the capability and the registry reserves
`solis-1.9-draw` (SDXL), but no diffusion model is wired yet — `/health` reports
`image_generation.wired: false`. Enabling it means `pip install diffusers` and
implementing `ImageGenerator.load()`; the interface is ready.

## Layout

```
solis/             the Solis 1.9 package
  registry.py      the variant ladder (Solis name -> base model)
  identity.py      Solis branding / system prompt (hardcoded identity)
  engine.py        Qwen2.5 text engine (4-bit, LoRA, streaming)
  vision.py        Qwen2.5-VL image analysis
  voice.py         Whisper speech-to-text
  imagegen.py      image-generation hook (deferred)
serve.py           OpenAI-compatible multimodal server
finetune/          QLoRA fine-tuning -> Solis adapters
  lora_finetune.py    the fine-tuner
  train_local.sh      largest-that-fits launcher (streamed massive data)
data/ingest.py     dataset normaliser (shared)
deploy/            RunPod scripts (from the earlier from-scratch training)
legacy/            the archived from-scratch Solis (v1.0–1.6)
MODEL_CARD.md      attribution, licensing, limitations
```

## Attribution

Solis 1.9 is built on Qwen2.5 (Alibaba Cloud), OpenAI Whisper, and — when
enabled — SDXL. Keep [MODEL_CARD.md](MODEL_CARD.md) with any distribution; the
upstream licenses require the attribution it contains.
