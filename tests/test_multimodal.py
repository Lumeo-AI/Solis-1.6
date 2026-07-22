"""Checks for the image + audio input path.

These verify the *plumbing* — that media embeddings are spliced into the right
positions, that the sequence lengths line up, and that generation runs end to
end. They do not (and cannot) check that the model understands the media: the
encoders are randomly initialised and untrained by design.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solis.config import SolisConfig  # noqa: E402
from solis.multimodal import (  # noqa: E402
    SolisMM, VisionConfig, AudioConfig, sinusoidal_positions)
from solis.tokenizer import SolisTokenizer, IMAGE, AUDIO  # noqa: E402


def tiny_lm() -> SolisConfig:
    return SolisConfig(
        name="test", vocab_size=512, dim=64, n_layers=3, n_heads=4,
        n_kv_heads=2, head_dim=16, max_seq_len=512, n_experts=4,
        n_experts_per_tok=2, n_shared_experts=1, expert_hidden=32,
        dense_layers=1, dense_hidden=64, dtype="float32")


def tiny_vision() -> VisionConfig:
    return VisionConfig(image_size=32, patch_size=8, dim=48, n_layers=2,
                        n_heads=4, n_output_tokens=4)


def tiny_audio() -> AudioConfig:
    return AudioConfig(n_mels=32, dim=48, n_layers=2, n_heads=4,
                       max_output_tokens=16)


def build() -> SolisMM:
    torch.manual_seed(0)
    return SolisMM(tiny_lm(), vision=tiny_vision(), audio=tiny_audio()).eval()


def test_encoders_report_capabilities():
    mm = build()
    assert mm.supports_image and mm.supports_audio
    text_only = SolisMM(tiny_lm())
    assert not text_only.supports_image and not text_only.supports_audio
    print(f"capabilities reported | encoder params {mm.encoder_params():,}")


def test_vision_encoder_output_shape():
    mm = build()
    vcfg = mm.vision_cfg
    px = torch.randn(1, 3, vcfg.image_size, vcfg.image_size)
    out = mm.vision(px)
    assert out.shape == (1, vcfg.n_output_tokens, mm.cfg.dim), out.shape
    print(f"vision encoder: image -> {tuple(out.shape)}")


def test_audio_encoder_output_shape():
    mm = build()
    acfg = mm.audio_cfg
    mel = torch.randn(1, acfg.n_mels, 400)
    out = mm.audio(mel)
    assert out.shape[0] == 1 and out.shape[2] == mm.cfg.dim
    assert out.shape[1] <= acfg.max_output_tokens
    print(f"audio encoder: mel(400 frames) -> {tuple(out.shape)}")


def test_inputs_embeds_splice_lengths_match():
    """The spliced embedding sequence and the expanded id sequence must be the
    same length, or decoding positions drift."""
    mm = build()
    tok = SolisTokenizer(merges=[])
    enc = tok.encode_chat_multimodal([
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image"},
            {"type": "text", "text": "and this sound?"},
            {"type": "audio"},
        ]},
    ])
    ids = torch.tensor([enc["ids"]])
    images = torch.randn(1, 3, mm.vision_cfg.image_size, mm.vision_cfg.image_size)
    mel = torch.randn(mm.audio_cfg.n_mels, 300)

    embeds = mm.build_inputs_embeds(ids, enc["media"], images, [mel])
    expanded = mm.expand_ids(ids, enc["media"], images, [mel])
    assert embeds.shape[1] == expanded.shape[1], (embeds.shape, expanded.shape)

    # One image contributes n_output_tokens, replacing one placeholder.
    grew = embeds.shape[1] - ids.shape[1]
    n_img = mm.vision_cfg.n_output_tokens - 1
    assert grew >= n_img
    # The placeholder ids survive in the expanded sequence at full width.
    assert (expanded[0] == IMAGE).sum() == mm.vision_cfg.n_output_tokens
    assert (expanded[0] == AUDIO).sum() >= 1
    print(f"splice lengths align | ids {ids.shape[1]} -> embeds "
          f"{embeds.shape[1]}")


def test_text_only_path_unaffected():
    mm = build()
    tok = SolisTokenizer(merges=[])
    enc = tok.encode_chat_multimodal([{"role": "user", "content": "hello"}])
    assert enc["media"] == []
    ids = torch.tensor([enc["ids"]])
    out = mm.generate(ids, media=[], max_new_tokens=5, temperature=0.0)
    assert out.shape[1] == ids.shape[1] + 5
    print("text-only generation through the wrapper is unaffected")


def test_multimodal_generation_runs():
    mm = build()
    tok = SolisTokenizer(merges=[])
    enc = tok.encode_chat_multimodal([
        {"role": "user", "content": [
            {"type": "text", "text": "describe:"},
            {"type": "image"},
        ]},
    ])
    ids = torch.tensor([enc["ids"]])
    images = torch.randn(1, 3, mm.vision_cfg.image_size, mm.vision_cfg.image_size)
    seen: list[int] = []
    out = mm.generate(ids, media=enc["media"], images=images,
                      max_new_tokens=6, temperature=0.0, stream_cb=seen.append)
    assert len(seen) == 6
    print(f"multimodal generation ran | produced {len(seen)} tokens")


def test_lm_checkpoint_loads_into_wrapper():
    """A text-only Solis state_dict must load into SolisMM.lm unchanged, so
    existing checkpoints keep working when the encoders are added."""
    from solis.model import Solis
    cfg = tiny_lm()
    lm = Solis(cfg)
    mm = SolisMM(cfg, vision=tiny_vision())
    missing, unexpected = mm.lm.load_state_dict(lm.state_dict(), strict=True)
    assert not missing and not unexpected
    print("text-only checkpoint loads into the multimodal wrapper's LM")


def test_sinusoidal_positions_shape():
    p = sinusoidal_positions(10, 48, torch.device("cpu"), torch.float32)
    assert p.shape == (1, 10, 48)
    assert torch.isfinite(p).all()
    print("sinusoidal position table well-formed")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nall {len(fns)} multimodal checks passed")
