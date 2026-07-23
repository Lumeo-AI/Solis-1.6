"""Solis 1.9 inference server — OpenAI-compatible, multimodal.

Serves a Solis 1.9 text variant (a Qwen2.5 model under the hood) and, on demand,
routes image and voice input to the vision and voice capabilities. Image
generation is a declared-but-deferred capability.

    SOLIS_MODEL      text variant to load   (default solis-1.9-small, 4-bit 7B)
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


def _flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


TEXT_MODEL = os.environ.get("SOLIS_MODEL", "solis-1.9-small")
ADAPTER = os.environ.get("SOLIS_ADAPTER") or None
VISION_MODEL = os.environ.get("SOLIS_VISION", "solis-1.9-vision")
VOICE_MODEL = os.environ.get("SOLIS_VOICE", "solis-1.9-voice")
FORCE_4BIT: Optional[bool] = (None if "SOLIS_4BIT" not in os.environ
                              else _flag("SOLIS_4BIT"))

app = FastAPI(title="Solis 1.9", version=__version__,
              description="Branded multimodal assistant on Qwen2.5 "
                          "(text + image + voice).")
app.add_middleware(CORSMiddleware,
                   allow_origins=os.environ.get("SOLIS_CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

# Lazily-populated singletons.
_engine: Optional[SolisEngine] = None
_vision = None
_voice = None


def engine() -> SolisEngine:
    global _engine
    if _engine is None:
        _engine = SolisEngine.load(TEXT_MODEL, load_in_4bit=FORCE_4BIT,
                                   adapter_path=ADAPTER)
    return _engine


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
    top_p: float = 0.9
    top_k: int = 20
    repetition_penalty: float = 1.05
    seed: Optional[int] = None


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
        repetition_penalty=req.repetition_penalty, seed=req.seed)


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


@app.post("/v1/chat")
def chat(req: ChatRequest):
    msgs = _resolve_messages(req.messages)
    cfg = _cfg(req)

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
