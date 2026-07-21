# Zeus ⚡

A **Mixture-of-Experts (MoE)** language model built entirely from scratch — no
base model, no pretrained weights, no external vocabulary. Zeus reads and writes
raw bytes and routes every token through a sparse set of expert networks.

## Architecture

Decoder-only transformer (`zeus/model.py`):

- **Byte-level tokenizer** (`zeus/tokenizer.py`) — 256 byte tokens + a few
  special tokens for chat structure (`<BOS>`, `<EOS>`, `<USER>`, `<ASST>`).
- **RMSNorm** pre-normalization.
- **Rotary positional embeddings (RoPE)**.
- **Causal multi-head self-attention** (via `scaled_dot_product_attention`).
- **Mixture-of-Experts FFN**: a learned router picks the **top-2 of 8** SwiGLU
  experts per token, with a **load-balancing auxiliary loss** and a **router
  z-loss** for stable training.
- Weight-tied embedding / LM head.

Default config (`zeus/config.py`): dim 384, 6 layers, 6 heads, 8 experts,
top-2 routing, 512-token context — about 46M total parameters (only a fraction
active per token thanks to sparse routing).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python data/build_corpus.py   # generate the synthetic chat corpus
python train.py               # train from scratch -> checkpoints/zeus.pt
uvicorn serve:app --port 8000 # serve the chat endpoint
```

> The bundled corpus is small and synthetic, so Zeus is a real but *tiny* model:
> it learns the chat format and its training data, not general knowledge. Training
> takes a few minutes on Apple-silicon (MPS) or CPU. Scale up `data/build_corpus.py`,
> `ZeusConfig`, and the step count in `train.py` for a bigger model.

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
data: {"type":"done","usage":{"prompt_tokens":18,"completion_tokens":42,"total_tokens":60}}
```

`GET /health` — device, checkpoint status, parameter count.

## The website

The companion chat UI lives in [`../zeuswebsite`](../zeuswebsite). It points at
this server via the `ZEUS_ENDPOINT` environment variable and enforces a
100k-tokens-per-week-per-user limit behind Clerk authentication.
