"""Image and voice input for Solis.

Solis is a text model at heart. This module bolts two small, from-scratch
encoders onto it and projects their output into the language model's embedding
space, so a conversation can contain pictures and audio clips alongside text.

The design is the now-standard "soft token" one: each media item is encoded to
a short sequence of vectors, those vectors are projected to the LM width, and
they are spliced into the token stream in place of a single `<|image|>` or
`<|audio|>` placeholder. From the transformer's point of view they are just
more positions in the sequence — attention, RoPE, the KV cache and generation
all work unchanged.

  image  ──▶ ViT patch encoder ──▶ linear projector ─┐
  audio  ──▶ log-mel + conv + transformer ─▶ projector┼─▶ spliced into the
  text   ──▶ token embeddings ────────────────────────┘   LM input sequence

**These encoders are randomly initialised and untrained.** The plumbing is
complete and correct — you can feed an image or a clip end to end today — but
until the encoders and projectors are trained on paired data the model will not
*understand* the media, only accept it. Training is deliberately out of scope
here; `data/ingest.py` and `train.py` are where that would happen.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SolisConfig
from .model import RMSNorm, Solis, generate_stream


# --------------------------------------------------------------------------- #
# Encoder configuration
# --------------------------------------------------------------------------- #
@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 14          # 224/14 = 16x16 = 256 patches
    in_channels: int = 3
    dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    # Patches are pooled to this many tokens before entering the LM, so a
    # picture costs a fixed, modest slice of context instead of 256 positions.
    n_output_tokens: int = 64
    dropout: float = 0.0

    @property
    def n_patches(self) -> int:
        g = self.image_size // self.patch_size
        return g * g

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160          # 10 ms frames
    dim: int = 384
    n_layers: int = 4
    n_heads: int = 6
    mlp_ratio: float = 4.0
    conv_stride: int = 2           # two conv layers -> 4x time downsampling
    # Hard cap on soft tokens per clip, so a long recording cannot blow the
    # context window. ~30 s of audio at 40 ms/token.
    max_output_tokens: int = 256
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #
def sinusoidal_positions(seq_len: int, dim: int, device, dtype) -> torch.Tensor:
    """Standard fixed sinusoidal position table, shape (1, seq_len, dim)."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)[None, :]
    freq = torch.exp(-torch.log(torch.tensor(10000.0)) * i / dim)
    ang = pos * freq
    out = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
    out[:, 0::2] = torch.sin(ang)
    out[:, 1::2] = torch.cos(ang[:, : out[:, 1::2].shape[1]])
    return out.to(dtype)[None]


class EncoderBlock(nn.Module):
    """A plain pre-norm bidirectional transformer block for the encoders.

    Bidirectional (no causal mask): an image patch or an audio frame should see
    the whole clip, not just what came before it.
    """

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = RMSNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Vision
# --------------------------------------------------------------------------- #
class VisionEncoder(nn.Module):
    """ViT-style patch encoder producing a fixed number of soft tokens."""

    def __init__(self, cfg: VisionConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(cfg.in_channels, cfg.dim,
                               kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patches, cfg.dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            EncoderBlock(cfg.dim, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        # Project each retained token to the LM width.
        self.proj = nn.Sequential(
            nn.Linear(cfg.dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """pixels: (B, C, H, W) -> (B, n_output_tokens, out_dim)."""
        x = self.patch(pixels)                       # (B, dim, gh, gw)
        B, D, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)             # (B, n_patches, dim)
        x = x + self.pos[:, :x.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # Pool the patch grid down to a fixed token budget with adaptive
        # average pooling over the 2-D grid, then project.
        x = x.transpose(1, 2).reshape(B, D, gh, gw)
        side = int(self.cfg.n_output_tokens ** 0.5)
        x = F.adaptive_avg_pool2d(x, (side, side))   # (B, dim, side, side)
        x = x.flatten(2).transpose(1, 2)             # (B, side*side, dim)
        return self.proj(x)


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
class AudioEncoder(nn.Module):
    """Log-mel → conv subsampling → transformer → projector."""

    def __init__(self, cfg: AudioConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        # Two strided convs over (mel, time): downsample time by conv_stride^2
        # and lift the mel axis into the encoder width.
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.n_mels, cfg.dim, kernel_size=3,
                      stride=cfg.conv_stride, padding=1),
            nn.GELU(),
            nn.Conv1d(cfg.dim, cfg.dim, kernel_size=3,
                      stride=cfg.conv_stride, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            EncoderBlock(cfg.dim, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.proj = nn.Sequential(
            nn.Linear(cfg.dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T) log-mel spectrogram -> (B, T', out_dim)."""
        x = self.conv(mel)                           # (B, dim, T')
        x = x.transpose(1, 2)                        # (B, T', dim)
        if x.shape[1] > self.cfg.max_output_tokens:
            # Uniformly subsample frames rather than truncating, so a long clip
            # is summarised across its whole duration.
            idx = torch.linspace(0, x.shape[1] - 1, self.cfg.max_output_tokens,
                                 device=x.device).long()
            x = x[:, idx]
        # Sinusoidal positions: MultiheadAttention is permutation-equivariant, so
        # without this the frame order — which is the whole point of audio —
        # would be invisible to the encoder.
        x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.proj(x)


# --------------------------------------------------------------------------- #
# Multimodal wrapper
# --------------------------------------------------------------------------- #
class SolisMM(nn.Module):
    """A Solis language model plus optional vision and audio encoders.

    The language model is unchanged and its checkpoint loads straight in; the
    encoders are additive. If neither encoder is configured this is just Solis
    with a thin wrapper, so text-only checkpoints keep working.
    """

    def __init__(self, cfg: SolisConfig,
                 vision: Optional[VisionConfig] = None,
                 audio: Optional[AudioConfig] = None):
        super().__init__()
        self.cfg = cfg
        self.lm = Solis(cfg)
        self.vision_cfg = vision
        self.audio_cfg = audio
        self.vision = VisionEncoder(vision, cfg.dim) if vision else None
        self.audio = AudioEncoder(audio, cfg.dim) if audio else None

    # -- introspection ---------------------------------------------------- #
    @property
    def supports_image(self) -> bool:
        return self.vision is not None

    @property
    def supports_audio(self) -> bool:
        return self.audio is not None

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encoder_params(self) -> int:
        n = 0
        if self.vision:
            n += sum(p.numel() for p in self.vision.parameters())
        if self.audio:
            n += sum(p.numel() for p in self.audio.parameters())
        return n

    # -- embedding assembly ----------------------------------------------- #
    def build_inputs_embeds(self, ids: torch.Tensor, media: list[dict],
                            images: Optional[torch.Tensor] = None,
                            audios: Optional[list[torch.Tensor]] = None
                            ) -> torch.Tensor:
        """Token embeddings with each media placeholder replaced in place by its
        encoder output.

        `ids` is (1, T) and includes one placeholder id per media item. `media`
        is the ordered slot list from `SolisTokenizer.encode_chat_multimodal`.
        Because encoders emit several vectors per item, the returned sequence is
        longer than `ids`; the matching expanded id sequence is returned by
        `expand_ids` so the two stay aligned.
        """
        emb = self.lm.tok_emb(ids)                   # (1, T, dim)
        pieces: list[torch.Tensor] = []
        cursor = 0
        img_i = 0
        aud_i = 0
        for slot in media:
            j = slot["index"]
            pieces.append(emb[:, cursor:j])          # text/other tokens before
            if slot["kind"] == "image":
                if self.vision is None:
                    raise ValueError("model has no vision encoder")
                feats = self.vision(images[img_i:img_i + 1].to(emb.dtype))
                img_i += 1
            else:
                if self.audio is None:
                    raise ValueError("model has no audio encoder")
                mel = audios[aud_i].unsqueeze(0).to(emb.dtype)
                feats = self.audio(mel)
                aud_i += 1
            pieces.append(feats)                     # media soft tokens
            cursor = j + 1                           # skip the placeholder
        pieces.append(emb[:, cursor:])
        return torch.cat(pieces, dim=1)

    def expand_ids(self, ids: torch.Tensor, media: list[dict],
                   images=None, audios=None) -> torch.Tensor:
        """The id sequence stretched to match `build_inputs_embeds`, with each
        placeholder repeated to the width of its encoder output. Used so that
        decoding bookkeeping (positions, repetition penalty) lines up with the
        embedded sequence."""
        row = ids[0].tolist()
        out: list[int] = []
        cursor = 0
        img_i = aud_i = 0
        for slot in media:
            j = slot["index"]
            out.extend(row[cursor:j])
            if slot["kind"] == "image":
                n = self.vision(images[img_i:img_i + 1].to(
                    self.lm.tok_emb.weight.dtype)).shape[1] if images is not None \
                    else self.vision_cfg.n_output_tokens
                out.extend([row[j]] * n)
                img_i += 1
            else:
                out.extend([row[j]] * self._audio_token_count(audios, aud_i))
                aud_i += 1
            cursor = j + 1
        out.extend(row[cursor:])
        return torch.tensor([out], device=ids.device, dtype=ids.dtype)

    def _audio_token_count(self, audios, i: int) -> int:
        if audios is None:
            return 1
        with torch.inference_mode():
            return self.audio(audios[i].unsqueeze(0).to(
                self.lm.tok_emb.weight.dtype)).shape[1]

    # -- generation -------------------------------------------------------- #
    @torch.inference_mode()
    def generate(self, ids: torch.Tensor, media: Optional[list[dict]] = None,
                 images: Optional[torch.Tensor] = None,
                 audios: Optional[list[torch.Tensor]] = None, **kw):
        """Generate a continuation, optionally conditioned on media."""
        if not media:
            return generate_stream(self.lm, ids, **kw)
        embeds = self.build_inputs_embeds(ids, media, images, audios)
        expanded = self.expand_ids(ids, media, images, audios)
        return generate_stream(self.lm, expanded, inputs_embeds=embeds, **kw)

    # -- persistence ------------------------------------------------------- #
    def modality_config(self) -> dict:
        return {
            "vision": asdict(self.vision_cfg) if self.vision_cfg else None,
            "audio": asdict(self.audio_cfg) if self.audio_cfg else None,
        }

    @staticmethod
    def config_from_dict(d: Optional[dict]):
        if not d:
            return None, None
        v = VisionConfig(**{f.name: d["vision"][f.name]
                            for f in fields(VisionConfig)
                            if d.get("vision")}) if d.get("vision") else None
        a = AudioConfig(**{f.name: d["audio"][f.name]
                           for f in fields(AudioConfig)
                           if d.get("audio")}) if d.get("audio") else None
        return v, a


# Default encoder presets, sized so both encoders together add well under a
# gigabyte of weights and never threaten the 16 GB serving budget.
DEFAULT_VISION = VisionConfig()
DEFAULT_AUDIO = AudioConfig()
