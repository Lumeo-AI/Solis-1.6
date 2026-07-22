"""FastAPI inference server for Solis.

Run:
    uvicorn serve:app --host 0.0.0.0 --port 8000
    python serve.py                      # same thing, loads the model eagerly

Environment:
    SOLIS_CKPT      checkpoint path      (default checkpoints/solis-mini-best.pt)
    SOLIS_TOKENIZER tokenizer path       (default checkpoints/tokenizer.json)
    SOLIS_DEVICE    cuda | cpu           (default: cuda when available)
    SOLIS_DTYPE     bfloat16 | float16 | float32
    SOLIS_MAX_TOKENS hard cap on completion length (default 1024)
    PORT            listen port          (default 8000)

Endpoints:
    GET  /health                  device, VRAM, parameter counts
    GET  /v1/models               OpenAI-style model listing
    POST /v1/chat                 Server-Sent Events (the website's endpoint)
    POST /v1/chat/completions     OpenAI-compatible, streaming or not

Streaming is byte-aware: a BPE token can end mid-UTF-8-character, so partial
bytes are held back until they form a complete character rather than being
emitted as replacement characters.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from queue import Queue
from typing import List, Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from solis.config import SolisConfig, get_config
from solis.model import Solis, generate_stream
from solis.tokenizer import SolisTokenizer, EOS

ROOT = Path(__file__).resolve().parent
CKPT = Path(os.environ.get("SOLIS_CKPT",
                           ROOT / "checkpoints" / "solis-mini-best.pt"))
TOKENIZER = Path(os.environ.get("SOLIS_TOKENIZER",
                                ROOT / "checkpoints" / "tokenizer.json"))
MAX_TOKENS = int(os.environ.get("SOLIS_MAX_TOKENS", "1024"))

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
           "float32": torch.float32}


def _pick_device() -> str:
    env = os.environ.get("SOLIS_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()
DTYPE = _DTYPES[os.environ.get(
    "SOLIS_DTYPE", "bfloat16" if DEVICE == "cuda" else "float32")]

app = FastAPI(title="Solis", version="1.0.0",
              description="Sparse mixture-of-experts language model")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SOLIS_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"], allow_headers=["*"],
)

_model: Optional[Solis] = None
_tok: Optional[SolisTokenizer] = None
_cfg: Optional[SolisConfig] = None
_loaded_from = "uninitialised"
# One model, one CUDA stream: generation is serialised. Batching concurrent
# requests would need a paged KV cache, which is out of scope for this server.
_lock = threading.Lock()


def get_tokenizer() -> SolisTokenizer:
    global _tok
    if _tok is None:
        if TOKENIZER.exists():
            _tok = SolisTokenizer.load(TOKENIZER)
        else:
            print(f"WARNING: no tokenizer at {TOKENIZER}; "
                  "falling back to raw bytes")
            _tok = SolisTokenizer(merges=[])
    return _tok


def get_model() -> Solis:
    global _model, _cfg, _loaded_from
    if _model is not None:
        return _model
    if CKPT.exists():
        blob = torch.load(CKPT, map_location="cpu", weights_only=False)
        cfg = SolisConfig.from_dict(blob["config"])
        model = Solis(cfg)
        model.load_state_dict(blob["model"])
        step = (blob.get("meta") or {}).get("step", "?")
        _loaded_from = f"{CKPT.name} (step {step})"
        print(f"loaded checkpoint {CKPT} — {cfg.name}, "
              f"{model.num_params():,} params")
    else:
        # Still serve, so the endpoint can be wired up before training finishes.
        # Output will be noise, and /health says so.
        cfg = get_config("mini")
        cfg.vocab_size = get_tokenizer().vocab_size
        model = Solis(cfg)
        _loaded_from = "RANDOM WEIGHTS (no checkpoint found)"
        print(f"WARNING: no checkpoint at {CKPT}; serving random weights")
    _cfg = cfg
    _model = model.to(device=DEVICE, dtype=DTYPE).eval()
    return _model


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.05
    seed: Optional[int] = None


class OpenAIChatRequest(BaseModel):
    model: str = "solis-1.0-mini"
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    stream: bool = False
    seed: Optional[int] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


# --------------------------------------------------------------------------- #
# Generation plumbing
# --------------------------------------------------------------------------- #
def _prepare_prompt(messages: List[ChatMessage]) -> torch.Tensor:
    tok = get_tokenizer()
    model = get_model()
    ids = tok.encode_chat([m.model_dump() for m in messages])
    limit = model._max_cache_len()
    if len(ids) >= limit - 16:
        raise HTTPException(
            status_code=413,
            detail=f"prompt is {len(ids)} tokens; the context window is {limit}",
        )
    return torch.tensor([ids], device=DEVICE), len(ids)


class _ByteStreamer:
    """Accumulates token bytes and releases complete UTF-8 characters only."""

    def __init__(self, tok: SolisTokenizer):
        self.tok = tok
        self.buf = bytearray()

    def push(self, token_id: int) -> str:
        piece = self.tok.decode_bytes([token_id])
        if not piece:
            return ""
        self.buf.extend(piece)
        # Trim back to the last complete character boundary.
        for cut in range(len(self.buf), max(-1, len(self.buf) - 4), -1):
            try:
                text = self.buf[:cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
            del self.buf[:cut]
            return text
        return ""

    def flush(self) -> str:
        if not self.buf:
            return ""
        text = self.buf.decode("utf-8", errors="replace")
        self.buf.clear()
        return text


def _token_stream(prompt: torch.Tensor, req, max_new: int):
    """Run generation on a worker thread, yielding token ids as they appear."""
    model = get_model()
    q: Queue = Queue()

    def worker():
        try:
            with _lock:
                generate_stream(
                    model, prompt, max_new_tokens=max_new,
                    temperature=req.temperature,
                    top_k=getattr(req, "top_k", 50),
                    top_p=req.top_p,
                    min_p=getattr(req, "min_p", 0.0),
                    repetition_penalty=getattr(req, "repetition_penalty", 1.0),
                    eos_id=EOS, stream_cb=q.put, seed=req.seed,
                )
        except Exception as exc:  # surface errors instead of hanging the stream
            q.put(exc)
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    model = get_model()
    cfg = _cfg
    info = {
        "status": "ok",
        "model": cfg.name,
        "checkpoint": _loaded_from,
        "trained": CKPT.exists(),
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "params_total": model.num_params(),
        "params_active_per_token": cfg.n_active_params,
        "context_window": model._max_cache_len(),
        "vocab_size": cfg.vocab_size,
        "tokenizer_vocab": get_tokenizer().vocab_size,
        "experts": f"{cfg.n_experts} routed (top-{cfg.n_experts_per_tok})"
                   f" + {cfg.n_shared_experts} shared",
    }
    if DEVICE == "cuda":
        free, total = torch.cuda.mem_get_info()
        info["vram"] = {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024 ** 3, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
            "free_gb": round(free / 1024 ** 3, 2),
            "total_gb": round(total / 1024 ** 3, 2),
        }
    return info


@app.get("/v1/models")
def list_models():
    cfg = _cfg or get_config("mini")
    get_model()
    return {"object": "list", "data": [{
        "id": _cfg.name, "object": "model", "owned_by": "solis",
        "created": 0,
    }]}


@app.post("/v1/chat")
def chat(req: ChatRequest):
    """Server-Sent Events stream — the endpoint the website consumes."""
    prompt, prompt_tokens = _prepare_prompt(req.messages)
    max_new = max(1, min(req.max_tokens, MAX_TOKENS))
    tok = get_tokenizer()

    def event_stream():
        streamer = _ByteStreamer(tok)
        completion_tokens = 0
        t0 = time.time()
        first_token_at: Optional[float] = None
        try:
            for token_id in _token_stream(prompt, req, max_new):
                completion_tokens += 1
                if first_token_at is None:
                    first_token_at = time.time()
                text = streamer.push(token_id)
                if text:
                    yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
            tail = streamer.flush()
            if tail:
                yield f"data: {json.dumps({'type': 'token', 'text': tail})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        elapsed = time.time() - t0
        done = {
            "type": "done",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "timing": {
                "total_seconds": round(elapsed, 3),
                "time_to_first_token": round((first_token_at or t0) - t0, 3),
                "tokens_per_second": round(completion_tokens / max(elapsed, 1e-9), 1),
            },
        }
        yield f"data: {json.dumps(done)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/v1/chat/completions")
def openai_chat(req: OpenAIChatRequest):
    """OpenAI-compatible endpoint, so existing clients work unchanged."""
    prompt, prompt_tokens = _prepare_prompt(req.messages)
    max_new = max(1, min(req.max_tokens, MAX_TOKENS))
    tok = get_tokenizer()
    created = int(time.time())
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model_name = _cfg.name if _cfg else req.model

    if not req.stream:
        streamer = _ByteStreamer(tok)
        parts, n = [], 0
        for token_id in _token_stream(prompt, req, max_new):
            n += 1
            parts.append(streamer.push(token_id))
        parts.append(streamer.flush())
        text = "".join(parts)
        return {
            "id": resp_id, "object": "chat.completion", "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop" if n < max_new else "length",
            }],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                      "total_tokens": prompt_tokens + n},
        }

    def event_stream():
        streamer = _ByteStreamer(tok)
        base = {"id": resp_id, "object": "chat.completion.chunk",
                "created": created, "model": model_name}
        yield ("data: " + json.dumps({
            **base,
            "choices": [{"index": 0, "delta": {"role": "assistant"},
                         "finish_reason": None}]}) + "\n\n")
        n = 0
        for token_id in _token_stream(prompt, req, max_new):
            n += 1
            text = streamer.push(token_id)
            if text:
                yield ("data: " + json.dumps({
                    **base,
                    "choices": [{"index": 0, "delta": {"content": text},
                                 "finish_reason": None}]}) + "\n\n")
        tail = streamer.flush()
        if tail:
            yield ("data: " + json.dumps({
                **base,
                "choices": [{"index": 0, "delta": {"content": tail},
                             "finish_reason": None}]}) + "\n\n")
        yield ("data: " + json.dumps({
            **base,
            "choices": [{"index": 0, "delta": {},
                         "finish_reason": "stop" if n < max_new else "length"}]})
            + "\n\n")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    get_model()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
