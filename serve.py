"""FastAPI inference server for Zeus.

Exposes a streaming chat endpoint that the website talks to. Set ZEUS_ENDPOINT
in the website to this server's /v1/chat URL.

Run:
    uvicorn serve:app --host 0.0.0.0 --port 8000
    #  (or: python serve.py)

Endpoints:
    GET  /health          -> {"status": "ok", ...}
    POST /v1/chat         -> Server-Sent Events stream of tokens + usage
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from queue import Queue
from typing import List, Optional

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from zeus.config import ZeusConfig
from zeus.model import Zeus
from zeus.tokenizer import ByteTokenizer, EOS

ROOT = Path(__file__).parent
CKPT = Path(os.environ.get("ZEUS_CKPT", ROOT / "checkpoints" / "zeus.pt"))

app = FastAPI(title="Zeus", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")
_tok = ByteTokenizer()
_model: Optional[Zeus] = None
_lock = threading.Lock()  # single-model, serialise generation


def get_model() -> Zeus:
    global _model
    if _model is None:
        if CKPT.exists():
            blob = torch.load(CKPT, map_location=_device)
            cfg = ZeusConfig.from_dict(blob["config"])
            model = Zeus(cfg)
            model.load_state_dict(blob["model"])
            print(f"loaded trained checkpoint from {CKPT}")
        else:
            # No checkpoint yet — serve a randomly-initialised Zeus so the
            # endpoint still works end-to-end (output will be gibberish).
            print("WARNING: no checkpoint found, serving random weights")
            model = Zeus(ZeusConfig())
        _model = model.to(_device).eval()
    return _model


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95


@app.get("/health")
def health():
    return {"status": "ok", "device": _device, "checkpoint": CKPT.exists(),
            "params_millions": round(get_model().num_params() / 1e6, 2)}


@app.post("/v1/chat")
def chat(req: ChatRequest):
    model = get_model()
    prompt_ids = _tok.encode_chat([m.model_dump() for m in req.messages])
    prompt_tokens = len(prompt_ids)
    max_new = max(1, min(req.max_tokens, 1024))

    def event_stream():
        q: Queue = Queue()

        def worker():
            idx = torch.tensor([prompt_ids], device=_device)
            with _lock:
                model.generate(
                    idx, max_new_tokens=max_new,
                    temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
                    eos_id=EOS, stream_cb=lambda t: q.put(t),
                )
            q.put(None)  # sentinel

        threading.Thread(target=worker, daemon=True).start()

        completion_tokens = 0
        pending = bytearray()
        while True:
            tok = q.get()
            if tok is None:
                break
            completion_tokens += 1
            if tok >= 256:  # special token (e.g. EOS)
                continue
            pending.append(tok)
            # Emit only on a valid UTF-8 boundary to avoid splitting a char.
            try:
                text = pending.decode("utf-8")
                pending.clear()
            except UnicodeDecodeError:
                continue
            yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        yield f"data: {json.dumps({'type': 'done', 'usage': usage})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    get_model()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
