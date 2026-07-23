# Solis 1.9 ☀️

A branded, **fine-tuned multimodal assistant** built on open foundation models.
Solis 1.9 loads a strong base model (the **Qwen3** family), serves it under the
Solis identity, understands **images** and **voice**, and can be **fine-tuned**
with your own data via QLoRA on a single 16 GB card. Image generation is a
declared, deferred capability.

> Solis 1.9 is a rebrand/fine-tune, **not** a from-scratch model. It is built on
> Qwen3 (text), Qwen3-VL (vision), Whisper (voice), and SDXL (image gen, when
> enabled). See [MODEL_CARD.md](MODEL_CARD.md) for attribution and licensing —
> required by the upstream licenses.

The Python package is `solis`; the product version is 1.9.

## The ladder — one brand, many sizes

`python -m solis.registry` prints this live, with both budgets.

```
TEXT                                    serve      train (QLoRA)
  solis-1.9-nano    Qwen3-1.7B    1.7B  ~5.6 GB    16 GB
  solis-1.9-mini    Qwen3-4B      4.0B  ~10 GB     16 GB
  solis-1.9         Qwen3-8B      8.2B  ~7.4 GB    16 GB   <- DEFAULT
  solis-1.9-base    Qwen3-14B    14.8B  ~12 GB     24 GB   <- stretch
  solis-1.9-large   Qwen3-32B    32.8B  cloud      cloud
  solis-1.9-max     Qwen3.6-27B  27.0B  cloud      cloud
  solis-1.9-moe     Qwen3.6-35B-A3B     cloud      cloud
VISION   solis-1.9-vision(-mini/-max)  →  Qwen3-VL-{4B,8B,32B}
VOICE    solis-1.9-voice(-fast)        →  Whisper large-v3-turbo / distil
IMAGE GEN  solis-1.9-draw  →  SDXL   (deferred — capability hook only)
```

**The default is `solis-1.9` (Qwen3-8B in 4-bit).** It is the largest Qwen3 that
QLoRA-trains inside 16 GB *and* still decodes fast enough that chat feels
instant — roughly 4.5 GB of weights, leaving plenty of the card for context.

### Why not something bigger?

Serving and training are two different budgets, and for Mixture-of-Experts models
memory follows **total** parameters, not active ones — every expert stays
resident even though only a few fire per token. `Qwen3.6-35B-A3B` is only
3B-active but still needs ~19–21 GB of 4-bit weights, so it can neither train
*nor serve* on a 16 GB card. `Qwen3.6-27B` and `Qwen3-32B` are over the line too.
Training a model you could not then run at home would be wasted money, so the
ladder stops at 14B for local use.

## Quickstart

```bash
pip install -r requirements.txt

# serve (loads Qwen3-8B in 4-bit as Solis; ~5 GB VRAM)
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
SOLIS_MODEL=solis-1.9-mini python serve.py   # Qwen3-4B, bf16, lighter
SOLIS_MODEL=solis-1.9-nano python serve.py   # Qwen3-1.7B, even CPU-ok
SOLIS_MODEL=solis-1.9-base python serve.py   # Qwen3-14B in 4-bit (~12 GB)
```

### Thinking mode

Qwen3 has a hybrid reasoning mode. It is **off by default** — a `<think>` block
costs hundreds of tokens before the answer starts, which is the difference
between instant and sluggish in everyday chat. Turn it on per request:

```jsonc
{ "messages": [...], "thinking": true }
```

The reasoning block is withheld from the response either way: you get the
answer, not the scratchpad.

## Image and voice

Send OpenAI-style content parts; the server routes images to Qwen3-VL and
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
`SOLIS_VISION`, `SOLIS_VOICE`, `SOLIS_LOAD_VISION/VOICE=1` (preload),
`SOLIS_MCP_CONFIG`, `SOLIS_NO_WEB`, `SOLIS_SEARCH_PROVIDER`,
`SOLIS_MAX_TOOL_ROUNDS`, `SOLIS_CORS_ORIGINS`, `PORT`.

## Web search and MCP

Solis can look things up instead of guessing past its training cutoff, and can
reach any MCP server you configure. Both are **on by default**; the server runs
those calls itself between turns and feeds the results back, then answers.

**Web search** needs no API key — it falls back to DuckDuckGo. Set one of
`BRAVE_API_KEY`, `TAVILY_API_KEY` or `SERPER_API_KEY` for better results, or
force a provider with `SOLIS_SEARCH_PROVIDER`. Built-in tools:

- `web_search(query, max_results)` — anything newer than the base model
- `fetch_page(url)` — pull a page down to readable text

**MCP** — point `SOLIS_MCP_CONFIG` at a JSON file in the standard `mcpServers`
shape (see [mcp.example.json](mcp.example.json)); an existing config drops
straight in. Both **stdio** (subprocess) and **streamable-HTTP** servers work,
using only the standard library — no extra dependencies. Tools are namespaced
`<server>__<tool>`.

```bash
SOLIS_MCP_CONFIG=mcp.json python serve.py
```

```bash
# what the server can execute itself
curl http://localhost:8000/v1/tools

# run one directly, to check a wiring
curl -X POST http://localhost:8000/v1/tools/call \
  -H 'Content-Type: application/json' \
  -d '{"name":"web_search","arguments":{"query":"who won the 2026 world cup"}}'

# search without going through the model
curl -X POST http://localhost:8000/v1/search \
  -H 'Content-Type: application/json' -d '{"query":"qwen3 release notes"}'
```

`POST /v1/chat` also emits `tool_call` / `tool_result` SSE events so a client can
show what the model looked up. Set `"use_tools": false` on a request for a pure
offline answer, or `SOLIS_NO_WEB=1` to disable the web tools entirely.

A broken MCP config disables MCP with a warning rather than taking the server
down, and one failed server never takes out the others.

> Web pages are untrusted input: a fetched page can contain text that tries to
> instruct the model. Treat search and fetch results as data, not commands.

## Fine-tuning — make it *your* Solis

QLoRA trains low-rank adapters on a frozen 4-bit base, so Qwen3-8B fine-tunes
comfortably on a 16 GB card. Data can be any Hugging Face dataset or local
JSONL — rows are normalised by `data/ingest.py` (messages / conversations /
prompt-response / question-answer / raw text) and rendered with the base chat
template. The Solis identity is trained in by default.

**Unsloth is the default backend** (~2x faster, ~50% less VRAM than stock PEFT);
it is what keeps 8B roomy on 16 GB and puts 14B inside a rented 24 GB card. If
Unsloth is not installed the fine-tuner falls back to plain PEFT automatically —
`--backend peft` forces it.

```bash
python finetune/lora_finetune.py \
    --model solis-1.9 \
    --hf Open-Orca/OpenOrca --hf teknium/OpenHermes-2.5 \
    --max-samples 20000 \
    --output checkpoints/solis-lora

# serve the fine-tune
SOLIS_MODEL=solis-1.9 SOLIS_ADAPTER=checkpoints/solis-lora python serve.py
```

### Training on Windows

Both native Windows and WSL2 work; **WSL2 is the smoother path**.

```powershell
# WSL2 (recommended) — the normal Linux install, CUDA passes through
wsl --install
# then inside WSL, exactly as on Linux:
pip install -r requirements.txt
```

Native Windows works too — Unsloth ships Windows support, but its kernels need
Triton, which is a separate Windows build (`triton-windows`, already in
`requirements.txt` behind a platform marker). Install **PyTorch with CUDA
first**, then the rest; Unsloth's Windows install is version-sensitive across
Python/CUDA/torch/triton.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Two Windows-specific gotchas at 16 GB:

- **The desktop eats VRAM.** The Windows compositor and browser take ~0.5-1.5 GB
  before you start. At 12.6/16 GB for the 14B that is the difference between
  fitting and not — close things, or run headless/WSL2.
- **If Unsloth fails to import**, the fine-tuner silently falls back to plain
  PEFT, which will OOM on 14B. Check the startup line says `backend=unsloth`,
  not `backend=peft`.

*Serving* on Windows needs none of this — `transformers` + `bitsandbytes` have
Windows wheels, so `python serve.py` works natively.

The launcher trains the largest model for your card on a streamed mix of
OpenOrca, OpenHermes, Tulu-3, Orca-Math and Glaive-Code, with an automatic
fallback if VRAM is tight:

```bash
# 16 GB card -> Qwen3-8B
nohup bash finetune/train_local.sh > logs/finetune.log 2>&1 &

# rented 24 GB card -> Qwen3-14B (the adapter still serves on your 16 GB)
TIER=stretch nohup bash finetune/train_local.sh > logs/finetune.log 2>&1 &
tail -f logs/finetune.log
```

The adapter is tens of MB — that is all you download back from a rented GPU.
Add `--merge` to also write merged 16-bit weights, and flash-attention +
`--packing` for more speed.

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
  engine.py        Qwen3 text engine (4-bit, LoRA, thinking mode, streaming)
  toolcall.py      tool-call parsing + the streaming gate
  websearch.py     web search providers + page fetching (stdlib only)
  mcp.py           MCP client (stdio + streamable HTTP, stdlib only)
  tools.py         the tool bus + agent loop
  vision.py        Qwen3-VL image analysis
  voice.py         Whisper speech-to-text
  imagegen.py      image-generation hook (deferred)
serve.py           OpenAI-compatible multimodal server
finetune/          QLoRA fine-tuning -> Solis adapters
  lora_finetune.py    the fine-tuner
  train_local.sh      largest-that-fits launcher (Unsloth, streamed data)
data/ingest.py     dataset normaliser (shared)
deploy/            RunPod scripts (from the earlier from-scratch training)
legacy/            the archived from-scratch Solis (v1.0–1.6)
MODEL_CARD.md      attribution, licensing, limitations
```

## Attribution

Solis 1.9 is built on Qwen3 (Alibaba Group), OpenAI Whisper, and — when
enabled — SDXL. Keep [MODEL_CARD.md](MODEL_CARD.md) with any distribution; the
upstream licenses require the attribution it contains.
