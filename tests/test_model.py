"""Correctness checks for the Solis stack.

Run: python -m pytest tests/ -q     (or: python tests/test_model.py)

The one that matters most is `test_kv_cache_matches_full_forward`: incremental
decoding and a full forward pass must produce identical logits, otherwise the
served model is quietly a different model from the trained one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solis.config import SolisConfig, PRESETS  # noqa: E402
from solis.model import Solis, KVCache, _forward_hidden, generate_stream  # noqa: E402
from solis.tokenizer import SolisTokenizer, BOS, EOS, ASSISTANT  # noqa: E402


def tiny_cfg(**kw) -> SolisConfig:
    base = dict(
        name="test", vocab_size=512, dim=64, n_layers=4, n_heads=4,
        n_kv_heads=2, head_dim=16, max_seq_len=64, n_experts=4,
        n_experts_per_tok=2, n_shared_experts=1, expert_hidden=32,
        dense_layers=1, dense_hidden=64, dtype="float32",
    )
    base.update(kw)
    return SolisConfig(**base)


def test_forward_shapes():
    cfg = tiny_cfg()
    m = Solis(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, out = m(x, targets=x, return_logits=True)
    assert logits.shape == (2, 16, cfg.vocab_size)
    loss, ce, aux = out
    assert torch.isfinite(loss)
    print("forward shapes ok | loss", round(loss.item(), 3))


def test_masked_loss_ignores_unsupervised_positions():
    """The sparse-loss path must give the same number as a dense computation
    over the same positions — this is the optimisation most likely to be
    silently wrong."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = Solis(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = x.clone()
    y[:, ::2] = -100  # supervise every other position

    with torch.no_grad():
        _, (loss, ce, _) = m(x, targets=y)
        dense_logits, _ = m(x, targets=y, return_logits=True)
        reference = torch.nn.functional.cross_entropy(
            dense_logits.reshape(-1, cfg.vocab_size).float(),
            y.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(ce, reference, rtol=1e-5, atol=1e-5)
    print("sparse masked loss matches dense reference")


def test_loss_sum_reduction_is_token_weighted():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = Solis(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = x.clone()
    y[:, ::2] = -100
    n = int((y != -100).sum())
    with torch.no_grad():
        _, (_, ce_mean, _) = m(x, targets=y, loss_reduction="mean")
        _, (_, ce_sum, _) = m(x, targets=y, loss_reduction="sum")
    torch.testing.assert_close(ce_sum / n, ce_mean, rtol=1e-4, atol=1e-4)
    print("sum reduction divides back to the mean")


def test_param_count_matches_config():
    """config.param_breakdown() is what the VRAM budget is built on, so it has
    to agree with the parameters the module actually allocates."""
    for name, preset in PRESETS.items():
        cfg = SolisConfig.from_dict({**preset.to_dict(),
                                     "dim": 64, "n_layers": 4, "n_heads": 4,
                                     "n_kv_heads": 2, "head_dim": 16,
                                     "vocab_size": 512, "expert_hidden": 32,
                                     "dense_hidden": 64, "dtype": "float32"})
        m = Solis(cfg)
        predicted = cfg.n_params
        actual = sum(p.numel() for p in m.parameters())
        assert predicted == actual, (
            f"{name}: config says {predicted:,}, module has {actual:,}")
    print("param accounting matches module for all presets")


def test_kv_cache_matches_full_forward():
    """Decoding token-by-token with the cache must equal one full forward."""
    torch.manual_seed(0)
    cfg = tiny_cfg(sliding_window=0, sliding_window_pattern=0)
    m = Solis(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))

    with torch.inference_mode():
        full = m.norm(_stack_no_cache(m, x))

        cache = KVCache(cfg, 1, 32, x.device, torch.float32)
        # Prefill all but the last token, then feed the last one alone.
        _forward_hidden(m, x[:, :-1], cache, offset=0)
        step = _forward_hidden(m, x[:, -1:], cache, offset=cache.pos)

    torch.testing.assert_close(step[:, -1], full[:, -1], rtol=1e-4, atol=1e-4)
    print("kv cache matches full forward")


def test_kv_cache_matches_with_sliding_window():
    torch.manual_seed(0)
    cfg = tiny_cfg(sliding_window=6, sliding_window_pattern=2)
    m = Solis(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 20))

    with torch.inference_mode():
        full = m.norm(_stack_no_cache(m, x))
        cache = KVCache(cfg, 1, 40, x.device, torch.float32)
        _forward_hidden(m, x[:, :-1], cache, offset=0)
        step = _forward_hidden(m, x[:, -1:], cache, offset=cache.pos)

    torch.testing.assert_close(step[:, -1], full[:, -1], rtol=1e-4, atol=1e-4)
    print("kv cache matches full forward (sliding window)")


def _stack_no_cache(m: Solis, idx):
    x = m.tok_emb(idx)
    cos, sin = m._rope_cache(idx.shape[1], x.device)
    for block in m.blocks:
        x, _aux = block(x, cos, sin, None, 0)
    return x


def test_moe_routing_covers_all_tokens():
    """Every token must come out of the MoE with a contribution — a dispatch
    bug that drops rows shows up as exact zeros here."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = Solis(cfg).train()
    x = torch.randn(2, 16, cfg.dim)
    moe = m.blocks[-1].ffn
    out, aux = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(aux)
    assert (out.abs().sum(-1) > 0).all(), "some tokens produced no expert output"
    print("moe dispatch covers every token")


def test_inference_output_is_batch_independent():
    """A sequence must get the same answer whether it is decoded alone or as
    part of a longer prefill.

    Capacity-limited MoE dispatch breaks this if it is left enabled at
    inference: whether a token's expert slot survives depends on how many other
    tokens in the same batch chose that expert.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg(n_experts=4, n_experts_per_tok=2, capacity_factor=1.0)
    m = Solis(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 48))

    with torch.inference_mode():
        long_form, _ = m(x)                       # all 48 tokens at once
        short, _ = m(x[:, :8])                    # first 8 only
        cache = KVCache(cfg, 1, 64, x.device, torch.float32)
        _forward_hidden(m, x[:, :7], cache, offset=0)
        stepped = m.lm_head(_forward_hidden(m, x[:, 7:8], cache, offset=cache.pos))

    torch.testing.assert_close(short[:, -1], stepped[:, -1], rtol=1e-4, atol=1e-4)
    assert long_form.shape[1] == 1
    print("inference output does not depend on batch composition")


def test_training_capacity_drops_are_bounded():
    """With balanced routing the fixed training capacity should drop very
    little. A high drop rate means capacity_factor needs raising."""
    torch.manual_seed(0)
    cfg = tiny_cfg(capacity_factor=1.25)
    m = Solis(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    y = x.clone()
    m(x, targets=y)
    rate = m.router_drop_rate()
    assert 0.0 <= rate < 0.5, f"drop rate {rate:.3f} is implausibly high"
    print(f"training capacity drop rate {rate * 100:.1f}%")


def test_forward_does_not_mutate_router_state():
    """Routing must be identical across repeated forwards of the same input.

    If forward mutates the balancing bias, gradient checkpointing recomputes
    with different routing, the per-expert slices change size, and backward
    dies on a shape mismatch. This is that regression test.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = Solis(cfg).train()
    moe = m.blocks[-1].ffn
    x = torch.randn(2, 32, cfg.dim)
    before = moe.expert_bias.clone()
    a, _ = moe(x)
    mid = moe.expert_bias.clone()
    b, _ = moe(x)
    torch.testing.assert_close(before, mid)
    torch.testing.assert_close(a, b)
    print("forward leaves router state untouched")


def test_load_balancing_bias_moves_on_update():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = Solis(cfg).train()
    moe = m.blocks[-1].ffn
    before = moe.expert_bias.clone()
    for _ in range(20):
        moe(torch.randn(2, 32, cfg.dim))
        m.update_router_bias()
    assert not torch.equal(before, moe.expert_bias), "balancing bias never updated"
    print("load-balancing bias updates when applied after a step")


def test_grad_checkpointing_matches_plain_backward():
    """Checkpointed and non-checkpointed training must produce the same
    gradients — this is what actually broke in practice."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = Solis(cfg).train()
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = x.clone()
    y[:, ::3] = -100

    def grads(checkpointing: bool):
        for p in m.parameters():
            p.grad = None
        m.enable_grad_checkpointing(checkpointing)
        torch.manual_seed(1)
        _, (loss, _, _) = m(x, targets=y)
        loss.backward()
        return {n: p.grad.clone() for n, p in m.named_parameters()
                if p.grad is not None}

    plain = grads(False)
    ckpt = grads(True)
    assert plain.keys() == ckpt.keys()
    for name in plain:
        torch.testing.assert_close(plain[name], ckpt[name], rtol=2e-3, atol=2e-4,
                                   msg=lambda s, n=name: f"{n}: {s}")
    m.enable_grad_checkpointing(False)
    print(f"gradient checkpointing matches plain backward "
          f"({len(plain)} tensors)")


def test_generation_runs_and_stops_on_eos():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = Solis(cfg).eval()
    prompt = torch.tensor([[1, 2, 3, 4]])
    seen: list[int] = []
    out = generate_stream(m, prompt, max_new_tokens=8, temperature=0.8,
                          stream_cb=seen.append, seed=0)
    assert out.shape[1] <= 4 + 8
    assert len(seen) == out.shape[1] - 4
    print(f"generation ok | produced {len(seen)} tokens")


def test_greedy_generation_is_deterministic():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = Solis(cfg).eval()
    prompt = torch.tensor([[1, 2, 3, 4]])
    a = generate_stream(m, prompt, max_new_tokens=10, temperature=0.0)
    b = generate_stream(m, prompt, max_new_tokens=10, temperature=0.0)
    torch.testing.assert_close(a, b)
    print("greedy decoding is deterministic")


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def test_tokenizer_roundtrip_without_merges():
    tok = SolisTokenizer(merges=[])
    for s in ["hello world", "", "héllo — ünicode ✨", "line\nbreak\ttab",
              "日本語のテキスト", "emoji 🌞 solis"]:
        assert tok.decode(tok.encode(s)) == s, s
    print("byte-fallback tokenizer round-trips exactly")


def test_tokenizer_roundtrip_with_merges():
    corpus = ["the quick brown fox jumps over the lazy dog. " * 20,
              "solis is a mixture of experts model. " * 20,
              "hello world, hello solis, hello experts. " * 20]
    tok = SolisTokenizer.train(corpus, vocab_size=400, verbose=False)
    assert tok.vocab_size <= 400
    for s in ["the quick brown fox", "solis is a mixture of experts model.",
              "unseen text with ünicode ✨ and 日本語"]:
        assert tok.decode(tok.encode(s)) == s, s
    # Merges must actually shorten the sequence.
    plain = SolisTokenizer(merges=[])
    long = "the quick brown fox jumps over the lazy dog. " * 5
    assert len(tok.encode(long)) < len(plain.encode(long))
    print(f"trained tokenizer round-trips | vocab {tok.vocab_size} | "
          f"{len(plain.encode(long))} bytes -> {len(tok.encode(long))} tokens")


def test_chat_template_supervision_mask():
    tok = SolisTokenizer(merges=[])
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    ids, mask = tok.encode_chat_supervised(msgs)
    assert len(ids) == len(mask)
    assert ids[0] == BOS
    # No user token is ever supervised.
    user_span = ids.index(ASSISTANT)
    assert sum(mask[:user_span]) == 0
    # The assistant's reply and its EOS are.
    assert sum(mask[user_span:]) == len(tok.encode("hello")) + 1
    print("chat supervision mask covers assistant tokens only")


def test_tokenizer_save_load(tmp_path=Path("checkpoints/_test_tok.json")):
    tok = SolisTokenizer.train(["abababab " * 50, "cdcdcdcd " * 50],
                               vocab_size=300, verbose=False)
    tok.save(tmp_path)
    back = SolisTokenizer.load(tmp_path)
    assert back.merges == tok.merges
    assert back.encode("abababab") == tok.encode("abababab")
    tmp_path.unlink()
    print("tokenizer save/load round-trips")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nall {len(fns)} checks passed")
