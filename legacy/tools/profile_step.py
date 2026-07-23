"""Isolate where a training step's time actually goes.

Runs the same model under a few configurations and times a full
forward+backward, so a slowdown can be attributed instead of guessed at.

Run: python tools/profile_step.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solis.config import get_config
from solis.model import Solis


def time_step(cfg, micro: int, grad_ckpt: bool, steps: int = 6,
              label: str = "", compile_model: bool = False) -> float:
    torch.manual_seed(0)
    model = Solis(cfg).cuda()
    model.enable_grad_checkpointing(grad_ckpt)
    model.train()
    if compile_model:
        model = torch.compile(model)
        steps = max(steps, 8)  # compilation dominates the first steps
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    x = torch.randint(0, cfg.vocab_size, (micro, cfg.max_seq_len), device="cuda")
    y = x.clone()
    y[:, ::3] = -100  # mimic the ~1/3 supervised density of chat data

    warmup = 4 if compile_model else 2
    for i in range(steps):
        if i == warmup:  # skip warmup + autotune (+ compilation)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, out = model(x, targets=y)
        out[0].backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = (steps - warmup) * micro * cfg.max_seq_len
    tok_s = n / dt
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"  {label:<44} {tok_s / 1e3:7.1f}k tok/s   peak {peak:5.2f} GB")

    del model, opt, x, y
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return tok_s


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"device: {torch.cuda.get_device_name(0)}\n")

    base = get_config("mini")
    micro = 8

    print(f"micro-batch {micro} x {base.max_seq_len} tokens\n")

    full_causal = dict(sliding_window=0, sliding_window_pattern=0)
    variants = [
        ("sliding window + checkpointing", dict(), True, False),
        ("full causal attention", full_causal, True, False),
        ("full causal, dense FFN only (no MoE)",
         {**full_causal, "dense_layers": base.n_layers}, True, False),
        ("full causal + torch.compile", full_causal, True, True),
        ("sliding window + torch.compile", dict(), True, True),
    ]

    for label, overrides, ckpt, comp in variants:
        cfg = type(base).from_dict({**base.to_dict(), **overrides})
        try:
            time_step(cfg, micro, ckpt, label=label, compile_model=comp)
        except torch.OutOfMemoryError:
            print(f"  {label:<44} OOM")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception as exc:
            print(f"  {label:<44} failed: {type(exc).__name__}: "
                  f"{str(exc)[:70]}")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
