"""Solis 1.9 inference server — OpenAI-compatible, multimodal.

Serves a Solis 1.9 text variant (a Qwen3 model under the hood) and, on demand,
routes image and voice input to the vision and voice capabilities. Image
generation is a declared-but-deferred capability.

    SOLIS_MODEL      text variant to load   (default solis-1.9, 4-bit Qwen3-8B)
    SOLIS_ADAPTER    optional LoRA adapter path (a Solis fine-tune)
    SOLIS_4BIT       force 4-bit: 1/0       (default: the variant's recommendation)
    SOLIS_VISION     vision variant         (default solis-1.9-vision, lazy)
    SOLIS_VOICE      voice variant          (default solis-1.9-voice, lazy)
    SOLIS_LOAD_VISION / SOLIS_LOAD_VOICE  = 1 to preload at startup
    PORT             listen port            (default 8000)

Endpoints:
    GET  /health                 loaded models, capabilities, VRAM
    GET  /v1/models              Solis 1.9 variants
    POST /v1/chat                SSE stream (website endpoint)
    POST /v1/chat/completions    OpenAI-compatible (stream or not)

Vision/voice models are lazy: they load on first use so a text-only deployment
never pays their memory. Set SOLIS_LOAD_*=1 to warm them at startup.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from solis import __version__, variants, Modality
from solis.engine import SolisEngine, GenerationConfig
from solis.identity import ASSISTANT_NAME, PRODUCT_VERSION
from solis import imagegen
from solis.tools import ToolBus, run_agent_loop, MAX_TOOL_ROUNDS
from solis import websearch


def _flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


TEXT_MODEL = os.environ.get("SOLIS_MODEL", "solis-1.9")
ADAPTER = os.environ.get("SOLIS_ADAPTER") or None
VISION_MODEL = os.environ.get("SOLIS_VISION", "solis-1.9-vision")
VOICE_MODEL = os.environ.get("SOLIS_VOICE", "solis-1.9-voice")
FORCE_4BIT: Optional[bool] = (None if "SOLIS_4BIT" not in os.environ
                              else _flag("SOLIS_4BIT"))

app = FastAPI(title="Solis 1.9", version=__version__,
              description="Branded multimodal assistant on Qwen3 "
                          "(text + image + voice).")
app.add_middleware(CORSMiddleware,
                   allow_origins=os.environ.get("SOLIS_CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

# Lazily-populated singletons.
_engine: Optional[SolisEngine] = None
_vision = None
_voice = None
_bus: Optional[ToolBus] = None


def engine() -> SolisEngine:
    global _engine
    if _engine is None:
        _engine = SolisEngine.load(TEXT_MODEL, load_in_4bit=FORCE_4BIT,
                                   adapter_path=ADAPTER)
    return _engine


def bus() -> ToolBus:
    """The tool bus: built-in web search/fetch plus any MCP servers.

    Built lazily and once, so MCP subprocesses start on first use rather than at
    import time (which would make `--reload` spawn duplicates).
    """
    global _bus
    if _bus is None:
        _bus = ToolBus.from_env()
    return _bus


def vision():
    global _vision
    if _vision is None:
        from solis.vision import VisionEngine
        _vision = VisionEngine.load(VISION_MODEL)
    return _vision


def voice():
    global _voice
    if _voice is None:
        from solis.voice import VoiceTranscriber
        _voice = VoiceTranscriber.load(VOICE_MODEL)
    return _voice


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[dict]]


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    seed: Optional[int] = None
    # Qwen3 hybrid reasoning. Off by default so replies come back immediately;
    # set true for hard problems and the model reasons first (the <think> block
    # is withheld from the response — callers get the answer, not the notes).
    thinking: bool = False
    # Server-side tools (web search, page fetch, MCP). On by default, so the
    # model can look things up instead of guessing past its training cutoff.
    use_tools: bool = True
    max_tool_rounds: Optional[int] = None
    tools: Optional[List[dict]] = None


class OpenAIChatRequest(ChatRequest):
    model: str = TEXT_MODEL
    stream: bool = False


# --------------------------------------------------------------------------- #
# Content handling — resolve media parts into text the LLM can consume
# --------------------------------------------------------------------------- #
def _resolve_messages(messages: List[ChatMessage]) -> List[dict]:
    """Turn OpenAI content-parts into plain text messages.

    Images are analysed by the vision model and voice clips transcribed, and
    their results are folded into the text so any Solis text variant can reason
    over them. (A future path could hand images straight to the VL model for the
    final answer; folding keeps one consistent chat model in charge.)
    """
    out: List[dict] = []
    for m in messages:
        if isinstance(m.content, str):
            out.append({"role": m.role, "content": m.content})
            continue

        text_bits: List[str] = []
        images: List[str] = []
        for part in m.content:
            kind = part.get("type")
            if kind == "text":
                text_bits.append(part.get("text", ""))
            elif kind in ("image_url", "image"):
                url = (part.get("image_url", {}).get("url") if kind == "image_url"
                       else part.get("data") or part.get("url"))
                if not url:
                    raise HTTPException(400, "image part missing data/url")
                images.append(url)
            elif kind in ("input_audio", "audio"):
                blob = (part.get("input_audio", {}).get("data")
                        if kind == "input_audio"
                        else part.get("data") or part.get("url"))
                if not blob:
                    raise HTTPException(400, "audio part missing data")
                transcript = voice().transcribe(blob)
                text_bits.append(f"[voice transcript] {transcript}")
            else:
                raise HTTPException(400, f"unknown content part type {kind!r}")

        if images:
            question = " ".join(t for t in text_bits if t).strip() \
                or "Describe the image(s)."
            analysis = vision().describe(images, question)
            # Give the text model the vision model's reading of the image.
            text_bits = [f"[image analysis] {analysis}"]
        out.append({"role": m.role, "content": "\n".join(t for t in text_bits if t)})
    return out


def _cfg(req: ChatRequest) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=req.max_tokens, temperature=req.temperature,
        top_p=req.top_p, top_k=req.top_k,
        repetition_penalty=req.repetition_penalty, seed=req.seed,
        thinking=req.thinking)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    info = {
        "status": "ok",
        "product": f"{ASSISTANT_NAME} {PRODUCT_VERSION}",
        "text_model": TEXT_MODEL,
        "text_loaded": _engine is not None,
        "capabilities": {
            "text": True,
            "image_analysis": {"variant": VISION_MODEL,
                               "loaded": _vision is not None},
            "voice_analysis": {"variant": VOICE_MODEL,
                               "loaded": _voice is not None},
            "image_generation": imagegen.status(),
            "tools": bus().status(),
        },
    }
    if _engine is not None:
        info["engine"] = _engine.info()
    return info


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [
        {"id": v.name, "object": "model", "owned_by": "solis",
         "base_model": v.base_repo, "modality": v.modality.value,
         "deployment": v.deployment.value}
        for v in variants()]}


@app.get("/v1/tools")
def list_tools():
    """Every tool the server can execute itself — built-ins plus MCP."""
    tb = bus()
    return {"object": "list", "data": tb.server_tools(), "status": tb.status()}


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict = {}


@app.post("/v1/tools/call")
def call_tool(req: ToolCallRequest):
    """Invoke one server-owned tool directly — handy for checking a wiring."""
    tb = bus()
    if not tb.owns(req.name, None):
        raise HTTPException(400, f"{req.name!r} is not a server-owned tool; "
                                 "see GET /v1/tools")
    ex = tb.execute(req.name, req.arguments)
    return {"name": ex.name, "owner": ex.owner, "is_error": ex.is_error,
            "content": ex.content}


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


@app.post("/v1/search")
def search(req: SearchRequest):
    """Run a web search directly, without going through the model."""
    try:
        results = websearch.search(req.query, req.max_results)
    except websearch.SearchError as exc:
        raise HTTPException(400, str(exc))
    return {"provider": websearch.active_provider(), "query": req.query,
            "results": [r.__dict__ for r in results]}


def _chunks(text: str, size: int = 24):
    """Slice a buffered answer so SSE stays incremental.

    The agent loop must see a whole turn before it knows whether it contains a
    tool call, so tool-path answers are buffered and re-chunked here rather than
    streamed token-by-token.
    """
    for i in range(0, len(text), size):
        yield text[i:i + size]


@app.post("/v1/chat")
def chat(req: ChatRequest):
    msgs = _resolve_messages(req.messages)
    cfg = _cfg(req)

    # --- Tool path: the model may search the web / call MCP; we run those
    # between turns and stream progress, then the answer. ---
    if req.use_tools:
        def tool_stream():
            t0 = time.time()
            events: list[dict] = []
            try:
                content, pending, _ = run_agent_loop(
                    engine(), msgs, cfg, bus(), client_tools=req.tools,
                    max_rounds=(req.max_tool_rounds
                                if req.max_tool_rounds is not None
                                else MAX_TOOL_ROUNDS),
                    on_event=events.append)
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                return
            for ev in events:
                yield f"data: {json.dumps(ev)}\n\n"
            for piece in _chunks(content):
                yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"
            yield ("data: " + json.dumps({
                "type": "done",
                "tools_used": [e["name"] for e in events
                               if e["type"] == "tool_call"],
                "timing": {"total_seconds": round(time.time() - t0, 3),
                           "chars": len(content)}}) + "\n\n")

        return StreamingResponse(tool_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    def event_stream():
        t0 = time.time()
        n_chars = 0
        try:
            for chunk in engine().stream(msgs, cfg):
                n_chars += len(chunk)
                yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return
        dt = time.time() - t0
        yield ("data: " + json.dumps({
            "type": "done",
            "timing": {"total_seconds": round(dt, 3), "chars": n_chars}}) + "\n\n")

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/v1/chat/completions")
def openai_chat(req: OpenAIChatRequest):
    msgs = _resolve_messages(req.messages)
    cfg = _cfg(req)
    created = int(time.time())
    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if req.use_tools:
        content, pending, executions = run_agent_loop(
            engine(), msgs, cfg, bus(), client_tools=req.tools,
            max_rounds=(req.max_tool_rounds if req.max_tool_rounds is not None
                        else MAX_TOOL_ROUNDS))
        message: dict = {"role": "assistant", "content": content or None}
        if pending:
            message["tool_calls"] = [c.to_openai() for c in pending]
        payload = {
            "id": rid, "object": "chat.completion", "created": created,
            "model": TEXT_MODEL,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if pending else "stop"}],
            "solis_tools_used": [e.name for e in executions],
        }
        if not req.stream:
            return payload

        def tool_completion_stream():
            base = {"id": rid, "object": "chat.completion.chunk",
                    "created": created, "model": TEXT_MODEL}
            yield ("data: " + json.dumps({**base, "choices": [
                {"index": 0, "delta": {"role": "assistant"},
                 "finish_reason": None}]}) + "\n\n")
            for piece in _chunks(content):
                yield ("data: " + json.dumps({**base, "choices": [
                    {"index": 0, "delta": {"content": piece},
                     "finish_reason": None}]}) + "\n\n")
            if pending:
                yield ("data: " + json.dumps({**base, "choices": [
                    {"index": 0, "delta": {"tool_calls": [
                        {**c.to_openai(), "index": i}
                        for i, c in enumerate(pending)]},
                     "finish_reason": None}]}) + "\n\n")
            yield ("data: " + json.dumps({**base, "choices": [
                {"index": 0, "delta": {},
                 "finish_reason": "tool_calls" if pending else "stop"}]})
                + "\n\n")
            yield "data: [DONE]\n\n"

        return StreamingResponse(tool_completion_stream(),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    if not req.stream:
        text = engine().generate(msgs, cfg)
        return {
            "id": rid, "object": "chat.completion", "created": created,
            "model": TEXT_MODEL,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
        }

    def event_stream():
        base = {"id": rid, "object": "chat.completion.chunk",
                "created": created, "model": TEXT_MODEL}
        yield ("data: " + json.dumps({**base, "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            + "\n\n")
        for chunk in engine().stream(msgs, cfg):
            yield ("data: " + json.dumps({**base, "choices": [
                {"index": 0, "delta": {"content": chunk}, "finish_reason": None}]})
                + "\n\n")
        yield ("data: " + json.dumps({**base, "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    engine()  # eager-load the text model
    if _flag("SOLIS_LOAD_VISION"):
        vision()
    if _flag("SOLIS_LOAD_VOICE"):
        voice()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
