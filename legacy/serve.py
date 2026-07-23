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
    SOLIS_MCP_CONFIG path to an mcpServers JSON file — enables tool calling
    SOLIS_MCP_MAX_ROUNDS max tool round-trips per request (default 5)
    SOLIS_MCP_TIMEOUT    per-call MCP timeout in seconds (default 30)
    PORT            listen port          (default 8000)

Endpoints:
    GET  /health                  device, VRAM, parameter counts, MCP status
    GET  /v1/models               OpenAI-style model listing
    GET  /v1/mcp/tools            tools exposed by the configured MCP servers
    POST /v1/mcp/call             invoke one MCP tool directly (debugging)
    POST /v1/chat                 Server-Sent Events (the website's endpoint)
    POST /v1/chat/completions     OpenAI-compatible, streaming or not

Tool calling: when SOLIS_MCP_CONFIG points at MCP servers, the chat endpoints
let the model request those tools; each call is executed server-side and its
result fed back for another turn, until the model answers in plain text. The
SSE endpoint emits `tool_call` / `tool_result` events as the loop runs.

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
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from solis.config import SolisConfig, get_config
from solis.model import generate_stream
from solis.multimodal import SolisMM, DEFAULT_VISION, DEFAULT_AUDIO
from solis import preprocess
from solis.tokenizer import SolisTokenizer, EOS
from solis.mcp import (MCPManager, MCPError, render_tools_prompt,
                       parse_tool_calls, ParsedCall)


def _envflag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


ROOT = Path(__file__).resolve().parent
CKPT = Path(os.environ.get("SOLIS_CKPT",
                           ROOT / "checkpoints" / "solis-mini-best.pt"))
TOKENIZER = Path(os.environ.get("SOLIS_TOKENIZER",
                                ROOT / "checkpoints" / "tokenizer.json"))
MAX_TOKENS = int(os.environ.get("SOLIS_MAX_TOKENS", "1024"))
# Media acceptance. A checkpoint that carries trained encoders turns these on
# automatically; the env vars force them on for a text-only checkpoint, which
# lets the endpoint accept images/audio even though untrained encoders cannot
# yet interpret them (/health flags this as encoders_trained: false).
ENABLE_VISION = _envflag("SOLIS_ENABLE_VISION")
ENABLE_AUDIO = _envflag("SOLIS_ENABLE_AUDIO")
# MCP tool calling. Point SOLIS_MCP_CONFIG at a JSON file using the standard
# `mcpServers` shape and the endpoints will let the model call those tools,
# executing them server-side between generation turns. Absent config, every
# endpoint behaves exactly as before.
MAX_TOOL_ROUNDS = int(os.environ.get("SOLIS_MCP_MAX_ROUNDS", "5"))

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

app = FastAPI(title="Solis", version="1.1.0",
              description="Sparse mixture-of-experts language model "
                          "with image and voice input")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SOLIS_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"], allow_headers=["*"],
)

_model: Optional[SolisMM] = None
_tok: Optional[SolisTokenizer] = None
_cfg: Optional[SolisConfig] = None
_loaded_from = "uninitialised"
_encoders_trained = False
_mcp: Optional[MCPManager] = None
_mcp_init = False
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


def get_mcp() -> Optional[MCPManager]:
    """Lazily build and connect the MCP manager from $SOLIS_MCP_CONFIG.

    Connection is attempted once; a failure to load the config disables tools
    for the process rather than crashing the server, and per-server failures are
    isolated inside the manager (see MCPManager.connect).
    """
    global _mcp, _mcp_init
    if _mcp_init:
        return _mcp
    _mcp_init = True
    try:
        _mcp = MCPManager.from_env("SOLIS_MCP_CONFIG")
    except Exception as exc:
        print(f"WARNING: MCP disabled — {exc}")
        _mcp = None
        return None
    if _mcp is not None:
        status = _mcp.connect()
        for name, err in status.items():
            if err:
                print(f"WARNING: MCP server {name!r} failed: {err}")
            else:
                print(f"MCP server {name!r} connected")
    return _mcp


def get_model() -> SolisMM:
    global _model, _cfg, _loaded_from, _encoders_trained
    if _model is not None:
        return _model

    if CKPT.exists():
        blob = torch.load(CKPT, map_location="cpu", weights_only=False)
        cfg = SolisConfig.from_dict(blob["config"])
        vision_cfg, audio_cfg = SolisMM.config_from_dict(blob.get("modality"))
        # Env flags can add encoders to a text-only checkpoint.
        if vision_cfg is None and ENABLE_VISION:
            vision_cfg = DEFAULT_VISION
        if audio_cfg is None and ENABLE_AUDIO:
            audio_cfg = DEFAULT_AUDIO

        model = SolisMM(cfg, vision=vision_cfg, audio=audio_cfg)
        model.lm.load_state_dict(blob["model"])
        # Trained encoder weights, if this checkpoint has them.
        if blob.get("modality_state"):
            model.load_state_dict(blob["modality_state"], strict=False)
            _encoders_trained = True
        step = (blob.get("meta") or {}).get("step", "?")
        _loaded_from = f"{CKPT.name} (step {step})"
        print(f"loaded checkpoint {CKPT} — {cfg.name}, "
              f"{model.num_params():,} params "
              f"(encoders: {model.encoder_params():,})")
    else:
        # Still serve, so the endpoint can be wired up before training finishes.
        # Output will be noise, and /health says so.
        cfg = get_config("mini")
        cfg.vocab_size = get_tokenizer().vocab_size
        vision_cfg = DEFAULT_VISION if ENABLE_VISION else None
        audio_cfg = DEFAULT_AUDIO if ENABLE_AUDIO else None
        model = SolisMM(cfg, vision=vision_cfg, audio=audio_cfg)
        _loaded_from = "RANDOM WEIGHTS (no checkpoint found)"
        print(f"WARNING: no checkpoint at {CKPT}; serving random weights")

    if model.vision is not None or model.audio is not None:
        if not _encoders_trained:
            print("NOTE: media encoders are UNTRAINED — the endpoint accepts "
                  "images/audio but cannot yet interpret them.")
    _cfg = cfg
    _model = model.to(device=DEVICE, dtype=DTYPE).eval()
    return _model


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
# `content` is either a plain string or a list of parts. Parts follow the
# OpenAI multimodal shape so existing clients can talk to Solis unchanged:
#   {"type": "text", "text": "..."}
#   {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
#   {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
ContentType = Union[str, List[dict]]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: ContentType


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.05
    seed: Optional[int] = None
    # Tool calling. When MCP servers are configured, `use_tools` lets the model
    # request them and executes the calls between turns; a client can set it
    # false to force a plain text answer.
    use_tools: bool = True
    max_tool_rounds: Optional[int] = None


class OpenAIChatRequest(BaseModel):
    model: str = "solis-1.0-mini"
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    stream: bool = False
    seed: Optional[int] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    use_tools: bool = True
    max_tool_rounds: Optional[int] = None


class MCPCallRequest(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Generation plumbing
# --------------------------------------------------------------------------- #
@dataclass
class PreparedInput:
    ids: torch.Tensor           # (1, T) token ids incl. media placeholders
    media: list                 # ordered slot list from the tokenizer
    images: Optional[torch.Tensor]   # (n_images, 3, H, W) or None
    audios: Optional[list]      # list of (n_mels, frames) tensors or None
    prompt_tokens: int          # length AFTER media expansion


def _normalise_content(role: str, content: ContentType, model: SolisMM) -> dict:
    """Turn one message's content into the tokenizer's part format, decoding
    and preprocessing any media into tensors as we go."""
    if isinstance(content, str):
        return {"role": role, "content": content, "_images": [], "_audios": []}

    parts, images, audios = [], [], []
    for p in content:
        kind = p.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": p.get("text", "")})
        elif kind in ("image_url", "image"):
            if not model.supports_image:
                raise HTTPException(400, "this model has no vision encoder; "
                                    "start the server with SOLIS_ENABLE_VISION=1 "
                                    "or load a checkpoint with a vision encoder")
            url = p.get("image_url", {}).get("url") if kind == "image_url" \
                else p.get("data") or p.get("url")
            if not url:
                raise HTTPException(400, "image part is missing its data/url")
            images.append(preprocess.load_image(url, model.vision_cfg))
            parts.append({"type": "image"})
        elif kind in ("input_audio", "audio"):
            if not model.supports_audio:
                raise HTTPException(400, "this model has no audio encoder; "
                                    "start the server with SOLIS_ENABLE_AUDIO=1 "
                                    "or load a checkpoint with an audio encoder")
            blob = p.get("input_audio", {}).get("data") if kind == "input_audio" \
                else p.get("data") or p.get("url")
            if not blob:
                raise HTTPException(400, "audio part is missing its data")
            audios.append(preprocess.load_audio(blob, model.audio_cfg))
            parts.append({"type": "audio"})
        else:
            raise HTTPException(400, f"unknown content part type {kind!r}")
    return {"role": role, "content": parts, "_images": images, "_audios": audios}


def _prepare_input(messages: List[ChatMessage]) -> PreparedInput:
    tok = get_tokenizer()
    model = get_model()

    norm = [_normalise_content(m.role, m.content, model) for m in messages]
    images: list = []
    audios: list = []
    for m in norm:
        images.extend(m.pop("_images"))
        audios.extend(m.pop("_audios"))

    enc = tok.encode_chat_multimodal(norm)
    ids = torch.tensor([enc["ids"]], device=DEVICE)

    img_tensor = (torch.stack(images).to(DEVICE) if images else None)
    aud_tensors = ([a.to(DEVICE) for a in audios] if audios else None)

    # The real (post-expansion) length decides whether we fit the context.
    expanded = model.expand_ids(ids, enc["media"], img_tensor, aud_tensors) \
        if enc["media"] else ids
    limit = model.lm._max_cache_len()
    if expanded.shape[1] >= limit - 16:
        raise HTTPException(
            status_code=413,
            detail=f"prompt expands to {expanded.shape[1]} tokens "
                   f"(text + media); the context window is {limit}",
        )
    return PreparedInput(ids, enc["media"], img_tensor, aud_tensors,
                         expanded.shape[1])


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


def _token_stream(prepared: PreparedInput, req, max_new: int):
    """Run generation on a worker thread, yielding token ids as they appear.

    Media (when present) is encoded once during prefill; decoding then proceeds
    on plain token ids, so streaming is identical to the text-only path.
    """
    model = get_model()
    q: Queue = Queue()

    def worker():
        try:
            with _lock:
                model.generate(
                    prepared.ids, media=prepared.media,
                    images=prepared.images, audios=prepared.audios,
                    max_new_tokens=max_new,
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
# Tool calling (MCP)
# --------------------------------------------------------------------------- #
# The generation loop above streams token ids. When tools are in play we need a
# turn's *whole* text before we can tell whether it is a tool call, so these run
# generation to completion and inspect the result. A turn is generated once and
# either handed back as the final answer or parsed for calls — never twice.
def _tools_active(use_tools: bool) -> list:
    """The tools available for this request, or [] when tools are off/unset."""
    if not use_tools:
        return []
    mcp = get_mcp()
    if mcp is None:
        return []
    try:
        return mcp.tools()
    except Exception as exc:
        print(f"WARNING: could not list MCP tools: {exc}")
        return []


def _inject_tools(messages: List[ChatMessage], tools: list) -> List[ChatMessage]:
    """Prepend a system message teaching the model this session's tools.

    If the conversation already opens with a system message we extend it, so the
    caller's own instructions survive.
    """
    fragment = render_tools_prompt(tools)
    if not fragment:
        return list(messages)
    out = [ChatMessage(**m.model_dump()) for m in messages]
    if out and out[0].role == "system" and isinstance(out[0].content, str):
        out[0] = ChatMessage(role="system",
                             content=out[0].content.rstrip() + "\n\n" + fragment)
    else:
        out.insert(0, ChatMessage(role="system", content=fragment))
    return out


def _format_tool_result(name: str, payload: str, is_error: bool) -> str:
    tag = "error" if is_error else "result"
    return f"{name} {tag}:\n{payload}"


def _generate_message(messages: List[ChatMessage], req, max_new: int) -> str:
    """Run one full generation turn and return its decoded text."""
    prepared = _prepare_input(messages)
    tok = get_tokenizer()
    streamer = _ByteStreamer(tok)
    parts: list[str] = []
    for token_id in _token_stream(prepared, req, max_new):
        parts.append(streamer.push(token_id))
    parts.append(streamer.flush())
    return "".join(parts)


def _run_tool_rounds(messages: List[ChatMessage], req, max_new: int,
                     tools: list, on_event: Optional[callable] = None
                     ) -> tuple[List[ChatMessage], str]:
    """Drive the agent loop until the model answers without calling a tool.

    Returns the (possibly extended) message list and the final assistant text.
    `on_event`, if given, is called with dicts describing each tool call and its
    result so an SSE endpoint can surface progress live.
    """
    mcp = get_mcp()
    rounds = req.max_tool_rounds if req.max_tool_rounds is not None \
        else MAX_TOOL_ROUNDS
    work = _inject_tools(messages, tools)

    for _ in range(max(0, rounds)):
        text = _generate_message(work, req, max_new)
        calls: list[ParsedCall] = parse_tool_calls(text)
        if not calls:
            return work, text
        # Record the model's tool-request turn verbatim, then service each call.
        work.append(ChatMessage(role="assistant", content=text))
        for call in calls:
            if on_event:
                on_event({"type": "tool_call", "name": call.name,
                          "arguments": call.arguments})
            try:
                result = mcp.call(call.name, call.arguments)
                payload = result.text or "(the tool returned no output)"
                is_error = result.is_error
            except Exception as exc:
                payload, is_error = f"{exc}", True
            if on_event:
                on_event({"type": "tool_result", "name": call.name,
                          "is_error": is_error, "content": payload})
            work.append(ChatMessage(
                role="tool",
                content=_format_tool_result(call.name, payload, is_error)))

    # Rounds exhausted: take one last turn, tools no longer offered as an option.
    final = _generate_message(work, req, max_new)
    # Strip any dangling tool-call syntax so the user sees prose, not a call.
    for call in parse_tool_calls(final):
        final = final.replace(call.raw, "").strip()
    return work, final


def _chunk_text(text: str, size: int = 8):
    """Yield a buffered final answer in small slices, to keep SSE incremental."""
    for i in range(0, len(text), size):
        yield text[i:i + size]


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
        "context_window": model.lm._max_cache_len(),
        "vocab_size": cfg.vocab_size,
        "tokenizer_vocab": get_tokenizer().vocab_size,
        "experts": f"{cfg.n_experts} routed (top-{cfg.n_experts_per_tok})"
                   f" + {cfg.n_shared_experts} shared",
        "modalities": {
            "text": True,
            "image": model.supports_image,
            "audio": model.supports_audio,
            "encoder_params": model.encoder_params(),
            # The honest flag: with an untrained encoder the endpoint accepts
            # media but does not understand it.
            "encoders_trained": _encoders_trained,
        },
    }
    mcp = get_mcp()
    if mcp is not None:
        servers = mcp.status()
        info["mcp"] = {
            "enabled": True,
            "servers": servers,
            "tool_count": sum(len(s["tools"]) for s in servers
                              if s["connected"]),
        }
    else:
        info["mcp"] = {"enabled": False}
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


@app.get("/v1/mcp/tools")
def mcp_tools():
    """List the tools exposed by every connected MCP server (OpenAI-shaped)."""
    mcp = get_mcp()
    if mcp is None:
        return {"object": "list", "data": [], "enabled": False}
    tools = mcp.tools()
    return {
        "object": "list",
        "enabled": True,
        "data": [t.to_openai() for t in tools],
        "servers": mcp.status(),
    }


@app.post("/v1/mcp/call")
def mcp_call(req: MCPCallRequest):
    """Invoke one MCP tool directly — handy for debugging a server wiring."""
    mcp = get_mcp()
    if mcp is None:
        raise HTTPException(400, "MCP is not configured (set SOLIS_MCP_CONFIG)")
    try:
        result = mcp.call(req.name, req.arguments)
    except MCPError as exc:
        raise HTTPException(400, str(exc))
    return {"name": req.name, "is_error": result.is_error,
            "content": result.text, "raw": result.raw}


@app.post("/v1/chat")
def chat(req: ChatRequest):
    """Server-Sent Events stream — the endpoint the website consumes."""
    tools = _tools_active(req.use_tools)
    max_new = max(1, min(req.max_tokens, MAX_TOKENS))
    tok = get_tokenizer()

    # --- Tool path: run the agent loop, streaming progress + the final text. ---
    if tools:
        def tool_stream():
            t0 = time.time()
            events: list[dict] = []
            try:
                _, final = _run_tool_rounds(
                    req.messages, req, max_new, tools, on_event=events.append)
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                return
            # `events` were collected during the (synchronous) loop; replay them
            # before the answer so a client sees which tools ran.
            for ev in events:
                yield f"data: {json.dumps(ev)}\n\n"
            for piece in _chunk_text(final):
                yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"
            done = {
                "type": "done",
                "usage": {"completion_tokens": len(final)},
                "timing": {"total_seconds": round(time.time() - t0, 3)},
                "tools_used": [e["name"] for e in events
                               if e["type"] == "tool_call"],
            }
            yield f"data: {json.dumps(done)}\n\n"

        return StreamingResponse(tool_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    # --- Plain path: unchanged true token-by-token streaming. ---
    prepared = _prepare_input(req.messages)
    prompt_tokens = prepared.prompt_tokens

    def event_stream():
        streamer = _ByteStreamer(tok)
        completion_tokens = 0
        t0 = time.time()
        first_token_at: Optional[float] = None
        try:
            for token_id in _token_stream(prepared, req, max_new):
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
    max_new = max(1, min(req.max_tokens, MAX_TOKENS))
    tok = get_tokenizer()
    created = int(time.time())
    resp_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    model_name = _cfg.name if _cfg else req.model
    tools = _tools_active(req.use_tools)

    # --- Tool path: MCP calls are executed server-side; the client gets the
    # final answer (streamed as content chunks) just like a plain completion. ---
    if tools:
        events: list[dict] = []
        _, text = _run_tool_rounds(req.messages, req, max_new, tools,
                                   on_event=events.append)
        tools_used = [e["name"] for e in events if e["type"] == "tool_call"]
        if not req.stream:
            return {
                "id": resp_id, "object": "chat.completion", "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": len(text)},
                "solis_tools_used": tools_used,
            }

        def tool_completion_stream():
            base = {"id": resp_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name}
            yield ("data: " + json.dumps({
                **base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                     "finish_reason": None}]}) + "\n\n")
            for piece in _chunk_text(text):
                yield ("data: " + json.dumps({
                    **base, "choices": [{"index": 0, "delta": {"content": piece},
                                         "finish_reason": None}]}) + "\n\n")
            yield ("data: " + json.dumps({
                **base, "choices": [{"index": 0, "delta": {},
                                     "finish_reason": "stop"}]}) + "\n\n")
            yield "data: [DONE]\n\n"

        return StreamingResponse(tool_completion_stream(),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    prepared = _prepare_input(req.messages)
    prompt_tokens = prepared.prompt_tokens

    if not req.stream:
        streamer = _ByteStreamer(tok)
        parts, n = [], 0
        for token_id in _token_stream(prepared, req, max_new):
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
        for token_id in _token_stream(prepared, req, max_new):
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
