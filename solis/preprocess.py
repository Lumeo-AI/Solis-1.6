"""Turn raw media into the tensors the encoders expect.

Kept separate from the encoders so the server (and any offline data pipeline)
can prepare inputs without importing torch modules it does not need, and so the
exact preprocessing is written down in one place — it has to match between
training and serving or the encoders see a different distribution than they
were trained on.

Images  -> (3, H, W) float tensor, resized and normalised.
Audio   -> (n_mels, frames) log-mel spectrogram at the encoder's sample rate.
"""

from __future__ import annotations

import base64
import io
from typing import Union

import torch

from .multimodal import AudioConfig, VisionConfig

# ImageNet channel statistics — a reasonable default normalisation for a vision
# stack trained from scratch on natural images.
_IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


def _decode_data_uri(data: str) -> bytes:
    """Accept a data: URI, a bare base64 string, or return raw bytes."""
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def load_image(source: Union[str, bytes], cfg: VisionConfig) -> torch.Tensor:
    """Load an image from bytes / base64 / data-URI / path -> (3, H, W) float.

    Uses Pillow for decoding, which handles PNG/JPEG/WebP/GIF uniformly.
    """
    from PIL import Image

    if isinstance(source, bytes):
        raw = source
    elif source.startswith("data:") or _looks_base64(source):
        raw = _decode_data_uri(source)
    else:
        with open(source, "rb") as f:
            raw = f.read()

    import numpy as np
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((cfg.image_size, cfg.image_size), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0     # (H, W, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def _looks_base64(s: str) -> bool:
    # Long, no whitespace, base64 alphabet — almost certainly encoded bytes and
    # not a filesystem path.
    if len(s) < 64 or any(c.isspace() for c in s[:64]):
        return False
    import string
    allowed = set(string.ascii_letters + string.digits + "+/=")
    return all(c in allowed for c in s[:64])


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def _decode_wav_pcm(raw: bytes) -> tuple[torch.Tensor, int]:
    """Decode a PCM WAV from bytes using only the standard library.

    Kept dependency-free on purpose: some torchaudio builds route decoding
    through an optional codec package that may not be installed, and WAV is the
    format a browser's MediaRecorder / a curl upload most reliably produces.
    Returns (waveform (1, N) float in [-1, 1], sample_rate).
    """
    import wave

    with wave.open(io.BytesIO(raw), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())

    import numpy as np
    if sampwidth == 2:
        data = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    elif sampwidth == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype("float32")
                - 128.0) / 128.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype="<i4").astype("float32") / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width {sampwidth * 8} bit")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return torch.from_numpy(data.copy()).unsqueeze(0), sr


def load_audio(source: Union[str, bytes], cfg: AudioConfig) -> torch.Tensor:
    """Load audio -> (n_mels, frames) log-mel spectrogram.

    Decoded and resampled to `cfg.sample_rate` and mixed to mono. WAV is decoded
    with the standard library so no optional codec package is required; other
    containers fall back to torchaudio if a backend is available.
    """
    import torchaudio.transforms as T

    if isinstance(source, bytes):
        raw = source
    elif source.startswith("data:") or _looks_base64(source):
        raw = _decode_data_uri(source)
    else:
        with open(source, "rb") as f:
            raw = f.read()

    if raw[:4] == b"RIFF":
        wav, sr = _decode_wav_pcm(raw)
    else:
        try:
            import torchaudio
            wav, sr = torchaudio.load(io.BytesIO(raw))
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
        except Exception as exc:
            raise ValueError(
                "could not decode this audio; WAV (PCM) is supported without "
                "extra dependencies, other formats need a torchaudio backend "
                f"(install torchcodec or soundfile). Underlying error: {exc}"
            ) from exc

    if sr != cfg.sample_rate:
        wav = T.Resample(sr, cfg.sample_rate)(wav)

    mel = T.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, n_mels=cfg.n_mels,
    )(wav)                                     # (1, n_mels, frames)
    # Log compression with a floor, the standard front end for speech models.
    log_mel = torch.log(mel.clamp(min=1e-10))
    return log_mel.squeeze(0)


def waveform_to_mel(wav: torch.Tensor, sr: int, cfg: AudioConfig) -> torch.Tensor:
    """Same mel front end, for callers that already hold a decoded waveform."""
    import torchaudio.transforms as T
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != cfg.sample_rate:
        wav = T.Resample(sr, cfg.sample_rate)(wav)
    mel = T.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, n_mels=cfg.n_mels)(wav)
    return torch.log(mel.clamp(min=1e-10)).squeeze(0)
