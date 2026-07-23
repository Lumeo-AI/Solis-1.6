"""Voice analysis for Solis 1.9 — speech to text via Whisper.

Voice input is handled as a front end: the clip is transcribed, and the
transcript flows into the text engine as the user's message. This keeps the
architecture simple and lets any Solis text variant answer spoken questions.

The Whisper model is lazy-loaded on first use (it is not cached by default, so
it downloads once, ~1.5 GB for large-v3-turbo).
"""

from __future__ import annotations

import io
from typing import Optional, Union

from .registry import get_variant


class VoiceTranscriber:
    def __init__(self, pipe, model_name: str):
        self.pipe = pipe
        self.model_name = model_name

    @classmethod
    def load(cls, variant_name: str = "solis-1.9-voice",
             device: Optional[str] = None) -> "VoiceTranscriber":
        import torch
        from transformers import pipeline

        variant = get_variant(variant_name)
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if (isinstance(device, int) and device >= 0) \
            else torch.float32
        print(f"loading {variant.name}  <-  {variant.base_repo}")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=variant.base_repo,
            torch_dtype=dtype,
            device=device,
        )
        return cls(pipe, variant.base_repo)

    def transcribe(self, audio: Union[bytes, str], language: Optional[str] = None
                   ) -> str:
        """Transcribe audio (raw bytes, path, or data URI) to text."""
        data = _to_array(audio)
        kwargs = {"return_timestamps": False}
        if language:
            kwargs["generate_kwargs"] = {"language": language}
        out = self.pipe(data, **kwargs)
        return (out.get("text") or "").strip()


def _to_array(audio: Union[bytes, str]):
    """Decode audio into the {'array','sampling_rate'} the ASR pipeline wants.

    WAV is decoded dependency-free; other formats need a torchaudio backend.
    """
    import base64
    import numpy as np

    if isinstance(audio, str):
        if audio.startswith("data:"):
            audio = audio.split(",", 1)[1]
            raw = base64.b64decode(audio)
        elif _looks_base64(audio):
            raw = base64.b64decode(audio)
        else:
            with open(audio, "rb") as f:
                raw = f.read()
    else:
        raw = audio

    if raw[:4] == b"RIFF":
        arr, sr = _decode_wav(raw)
    else:
        import torchaudio
        wav, sr = torchaudio.load(io.BytesIO(raw))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        arr = wav.squeeze(0).numpy()
    return {"array": arr, "sampling_rate": sr}


def _decode_wav(raw: bytes):
    import wave
    import numpy as np
    with wave.open(io.BytesIO(raw), "rb") as w:
        ch, width, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    if width == 2:
        a = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    elif width == 4:
        a = np.frombuffer(frames, dtype="<i4").astype("float32") / 2147483648.0
    elif width == 1:
        a = (np.frombuffer(frames, dtype=np.uint8).astype("float32") - 128) / 128
    else:
        raise ValueError(f"unsupported WAV width {width*8}-bit")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def _looks_base64(s: str) -> bool:
    if len(s) < 64 or any(c.isspace() for c in s[:64]):
        return False
    import string
    ok = set(string.ascii_letters + string.digits + "+/=")
    return all(c in ok for c in s[:64])
