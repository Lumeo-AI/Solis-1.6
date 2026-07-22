# Solis ☀️

A **sparse Mixture-of-Experts** language model built entirely from scratch — no
base model, no pretrained weights, no borrowed vocabulary. Solis learns its own
tokenizer, its own experts, and its own router, starting from random numbers.

The 1.0 family is designed against one hard constraint: **every preset must
serve inside 16 GB of VRAM**, weights, KV cache, and activation workspace
included, while keeping *active* parameters low so decoding stays fast.

```
preset      total   active    ctx   weights     kv    act    peak   fits 16GB
------------------------------------------------------------------------------
mini         335M     123M   2048     0.62G  0.03G  0.07G   1.62G        yes
small       1.79B     489M   4096     3.33G  0.09G  0.22G   4.55G        yes
base        4.48B     974M   8192     8.35G  0.22G  0.63G  10.09G        yes
flagship    6.33B    1.40B   8192    11.80G  0.47G  0.72G  13.88G        yes
```

`python -m solis.config` prints this table; `python bench.py` measures it for
real and tells you where the estimate is wrong.

## Architecture

Decoder-only transformer (`solis/model.py`):

- **Byte-level BPE tokenizer** (`solis/tokenizer.py`) trained on our own corpus.
  256 byte tokens are always present as a fallback, so nothing is ever
  unencodable, and round-tripping is exact for arbitrary bytes.
- **RMSNorm** pre-normalisation, computed in fp32 regardless of autocast dtype.
- **Grouped-query attention** — fewer KV heads than query heads, which is what
  makes the KV cache small enough to serve long context in 16 GB.
- **QK-norm** — RMSNorm on queries and keys before the dot product. Removes the
  attention-logit blowup that otherwise makes small models diverge.
- **Sliding-window attention** on most layers, with every 4th layer left global,
  so attention cost grows linearly in context but information still travels the
  full sequence.
- **Rotary position embeddings**, with a scaling factor for serving beyond the
  trained context.
- **Mixture-of-Experts FFN** — one *shared* expert that runs for every token,
  plus a top-k routed set of specialists. The shared path is what keeps a sparse
  model coherent at small scale: general-purpose computation lives there instead
  of being relearned by every expert.
- **Aux-loss-free load balancing** — a per-expert bias steers routing toward
  under-used experts without adding gradient noise, backed by a small classic
  auxiliary loss and a router z-loss.
- The first layers stay **dense**; early layers do broad low-level work that
  every token needs, so routing them wastes capacity.
- Weight-tied embedding / LM head, and residual branches scaled by `1/sqrt(2L)`
  at init so the residual stream does not grow with depth.

### Two implementation details that are easy to get wrong

**MoE dispatch runs as three batched matmuls**, not a Python loop over experts.
Tokens are sorted by expert and packed into a fixed `(n_experts, capacity, dim)`
buffer. The naive version — slicing the token stream per expert and looping —
issues `3 x n_experts` small GEMMs per layer *and* needs a host synchronisation
to learn the slice boundaries.

**Capacity limiting is training-only.** During training a fixed capacity keeps
every tensor shape static; overflow pairs are routed to a scratch row whose
output is zeroed, so they contribute nothing. At inference capacity is the
largest actual group, which costs one device-to-host read per layer but
guarantees a token's output never depends on which other tokens shared its
batch. Without that split, a prefill disagrees with the same tokens decoded one
at a time — `tests/test_model.py` checks exactly this.
<<<<<<< HEAD

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 1. get a corpus — real-world data (recommended) OR the procedural fallback
python data/ingest.py --hf <dataset-id> --out data/corpus.jsonl
#   ...or: python data/build_corpus.py

python data/train_tokenizer.py   # 2. learn the BPE vocabulary
python data/prepare.py           # 3. tokenise + pack to .bin
python train.py                  # 4. train  -> checkpoints/solis-mini.pt

python eval.py                   # task accuracy + validation perplexity
python bench.py                  # real VRAM and throughput
uvicorn serve:app --port 8000    # serve
```

Run the tests with `python tests/test_model.py` and
`python tests/test_multimodal.py`.

## Training on real-world data

`data/ingest.py` converts existing datasets into the corpus format the rest of
the pipeline reads, so training is not limited to the built-in procedural text.
It normalises the common chat schemas — `messages`, split
`system`/`user`/`assistant` columns, ShareGPT `conversations`,
`prompt`/`response` pairs, and raw `text` — from Hugging Face datasets or local
JSONL / text files, and writes a train/val split.

```bash
python data/ingest.py \
    --hf oyildirim/cyberstrike-sft-120k \
    --hf SkywardNomad92/pentest-findings-v2 \
    --out data/corpus.jsonl
python data/train_tokenizer.py && python data/prepare.py
python train.py --preset mini
```

`train.py` uses fp32 master weights with bf16 autocast, gradient accumulation,
gradient checkpointing (on by default — a top-k MoE stores `k` copies of every
intermediate, so recomputation is worth much more here than in a dense model),
cosine decay after warmup, and no weight decay on norms or embeddings. The
micro-batch is chosen automatically from free VRAM, and an OOM anywhere in the
step halves it and retries rather than losing the run.

```bash
python train.py --preset mini --max-minutes 90
python train.py --resume
python train.py --preset small --adam8bit    # 8-bit optimizer states
```

`mini` and `small` train on a single 16 GB card. `base` and `flagship` are
inference-only at that size — training them needs roughly 55 GB and 77 GB of
optimizer state respectively, which is a multi-GPU or offloaded job.

## Image and voice input

Solis accepts pictures and audio clips alongside text. Each media item is run
through a small from-scratch encoder — a ViT for images, a log-mel + conv +
transformer stack for audio (`solis/multimodal.py`) — projected to the model's
width, and spliced into the token stream in place of an `<|image|>` or
`<|audio|>` placeholder. From the transformer's side they are just more
positions, so attention, the KV cache and streaming generation are unchanged.

The server speaks the OpenAI multimodal content-parts format:

```jsonc
{ "messages": [{ "role": "user", "content": [
    { "type": "text", "text": "what is in this picture?" },
    { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } },
    { "type": "input_audio", "input_audio": { "data": "<base64 wav>", "format": "wav" } }
]}]}
```

Enable the encoders with `SOLIS_ENABLE_VISION=1` / `SOLIS_ENABLE_AUDIO=1`, or by
loading a checkpoint that carries trained encoders. WAV audio is decoded with no
extra dependencies; other formats need a torchaudio backend.

> **The encoders ship untrained.** The full path works end to end — you can post
> an image or a clip today — but until the encoders and their projectors are
> trained on paired data the model *accepts* media without *understanding* it.
> `/health` reports `modalities.encoders_trained: false` so this is never
> ambiguous. Training them is a separate job from `train.py`, which optimises
> the language model only.

## The corpus

There are two ways to feed Solis, and they trade off provenance against
capability:

**Real-world data (`data/ingest.py`)** is the recommended path and what the
model is intended to train on. It pulls existing datasets — Hugging Face,
local JSONL, or plain text — normalises their schema, and produces the packed
corpus. This is where broad vocabulary and real knowledge come from; BPE no
longer saturates early because the text has genuine lexical diversity.

**The procedural generator (`data/build_corpus.py`)** writes everything from
templates — nothing scraped, nothing copyrighted. It is ideal for standing the
pipeline up and for teaching **in-context work** (answer from the given passage,
transform this text, follow this format), but a model trained only on it has
bounded vocabulary (~5k unique pre-tokens) and no world knowledge. Use it as a
fallback or to supplement real data with clean task-format examples.

Either way the output is `data/corpus.jsonl` + `data/corpus.val.jsonl`, and
`data/prepare.py` consumes any JSONL with a `messages` field.

## Serving API

`POST /v1/chat` — streams **Server-Sent Events**:

```jsonc
// request
{ "messages": [{ "role": "user", "content": "who are you?" }],
  "max_tokens": 256, "temperature": 0.8 }

// stream
data: {"type":"token","text":"I"}
data: {"type":"token","text":" am"}
...
data: {"type":"done","usage":{...},"timing":{"tokens_per_second":42.1}}
```

`POST /v1/chat/completions` — OpenAI-compatible, streaming or not, so existing
clients work unchanged.

`GET /health` — device, dtype, parameter counts, live VRAM.

Streaming is byte-aware: a BPE token can end mid-UTF-8-character, so partial
bytes are held back until they form a complete character.

Environment: `SOLIS_CKPT`, `SOLIS_TOKENIZER`, `SOLIS_DEVICE`, `SOLIS_DTYPE`,
`SOLIS_MAX_TOKENS`, `SOLIS_CORS_ORIGINS`, `PORT`.

## Website data

The companion site lives on a separate machine, not in this repo. The numbers
it needs come from the measurement scripts here: `python eval.py` writes
`results/eval.json` (task accuracy, validation perplexity) and `python bench.py`
writes `results/bench.json` (real VRAM and throughput per preset). Every Solis
number in those files is measured on the machine that produced it.
=======
>>>>>>> 0bc936d61329966639da9dc8f967c059ea2f1d3c
