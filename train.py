"""Train Solis from scratch.

    python data/build_corpus.py          # 1. generate the corpus
    python data/train_tokenizer.py       # 2. learn the BPE vocabulary
    python data/prepare.py               # 3. tokenise + pack to .bin
    python train.py                      # 4. train

Everything starts from random weights. There is no base model, no pretrained
checkpoint, and no imported vocabulary anywhere in this pipeline.

Recipe notes, since the choices here are load-bearing:

  * **Mixed precision, fp32 master weights.** Parameters and gradients live in
    fp32; the forward/backward runs under bf16 autocast. Keeping master weights
    in fp32 costs memory but removes a whole class of silent divergence, and at
    this model size the memory is affordable.
  * **Gradient accumulation** decouples the batch size that fits in VRAM from
    the batch size the optimiser sees. The tokens-per-step target is what
    actually matters for the loss curve; micro-batch size is just a memory knob.
  * **No weight decay on norms or embeddings.** Decaying them pulls the residual
    scale toward zero and hurts small models noticeably.
  * **Warmup then cosine.** The router is the fragile part early on — it can
    collapse onto one expert if the learning rate starts high — so warmup is
    longer than a dense model of this size would need.

Useful flags:
    --preset small          train a different size from the family
    --max-minutes 90        stop cleanly on a wall-clock budget
    --resume                continue from the last checkpoint
    --compile               torch.compile the model
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from solis.config import SolisConfig, get_config
from solis.model import Solis
from solis.tokenizer import SolisTokenizer, PAD

ROOT = Path(__file__).resolve().parent
DEFAULT_PACKED = ROOT / "data" / "packed"
DEFAULT_CKPT_DIR = ROOT / "checkpoints"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class PackedDataset:
    """Random fixed-length windows over the packed token stream.

    Sampling random offsets (rather than walking the file in order) means every
    step sees a fresh mix of conversations, which matters here because the
    corpus is generated in task-type order within each conversation.
    """

    def __init__(self, packed_dir: Path, split: str, seq_len: int):
        meta = json.loads((packed_dir / "meta.json").read_text(encoding="utf-8"))
        if split not in meta["splits"]:
            raise SystemExit(f"split {split!r} not in {packed_dir}/meta.json")
        info = meta["splits"][split]
        dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32
        self.tokens = np.memmap(packed_dir / info["tokens_file"],
                                dtype=dtype, mode="r")
        self.mask = np.memmap(packed_dir / info["mask_file"],
                              dtype=np.uint8, mode="r")
        n = min(self.tokens.size, self.mask.size)
        self.tokens = self.tokens[:n]
        self.mask = self.mask[:n]
        self.seq_len = seq_len
        self.vocab_size = meta["vocab_size"]
        self.n_tokens = n
        if n <= seq_len + 1:
            raise SystemExit(
                f"split {split!r} has only {n:,} tokens, need > {seq_len + 1:,}")

    def batch(self, batch_size: int, rng: np.random.Generator, device):
        T = self.seq_len
        starts = rng.integers(0, self.n_tokens - T - 1, size=batch_size)
        x = np.stack([self.tokens[s:s + T] for s in starts]).astype(np.int64)
        y = np.stack([self.tokens[s + 1:s + T + 1] for s in starts]).astype(np.int64)
        m = np.stack([self.mask[s + 1:s + T + 1] for s in starts])
        y[m == 0] = -100  # only assistant tokens carry loss
        xt = torch.from_numpy(x).pin_memory().to(device, non_blocking=True)
        yt = torch.from_numpy(y).pin_memory().to(device, non_blocking=True)
        return xt, yt

    def sequential_batches(self, batch_size: int, n_batches: int, device):
        """Deterministic batches for evaluation — the same windows every time,
        so successive val numbers are actually comparable."""
        T = self.seq_len
        stride = max(1, (self.n_tokens - T - 1) // max(n_batches * batch_size, 1))
        pos = 0
        for _ in range(n_batches):
            starts = [min(pos + i * stride, self.n_tokens - T - 2)
                      for i in range(batch_size)]
            pos += batch_size * stride
            x = np.stack([self.tokens[s:s + T] for s in starts]).astype(np.int64)
            y = np.stack([self.tokens[s + 1:s + T + 1] for s in starts]).astype(np.int64)
            m = np.stack([self.mask[s + 1:s + T + 1] for s in starts])
            y[m == 0] = -100
            yield (torch.from_numpy(x).to(device),
                   torch.from_numpy(y).to(device))


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #
def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float,
                    betas=(0.9, 0.95), eps=1e-8, use_8bit=False):
    """Decay matmul weights only; leave norms, biases and embeddings alone."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "emb" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay)
    n_plain = sum(p.numel() for p in no_decay)
    print(f"optimizer: {len(decay)} decayed tensors ({n_decay:,} params), "
          f"{len(no_decay)} undecayed ({n_plain:,} params)")

    if use_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas, eps=eps)
        except Exception as exc:  # pragma: no cover
            print(f"8-bit optimizer unavailable ({exc}); using fused AdamW")
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=True)


def lr_at(step: int, total: int, warmup: int, lr_max: float,
          lr_min_ratio: float = 0.1) -> float:
    if step < warmup:
        return lr_max * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, prog)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return lr_max * (lr_min_ratio + (1 - lr_min_ratio) * cos)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, data: PackedDataset, batch_size: int, n_batches: int,
             device, amp_dtype) -> dict:
    """Loss over held-out data, counting only supervised positions.

    Reported as a token-weighted mean rather than a mean of per-batch means, so
    batches with few assistant tokens don't distort the number.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in data.sequential_batches(batch_size, n_batches, device):
        n = int((y != -100).sum())
        if n == 0:
            continue
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
            _, out = model(x, targets=y, loss_reduction="sum")
        # out = (loss, ce, aux); at eval the aux term is zero.
        _, ce_sum, _ = out
        total_loss += float(ce_sum)
        total_tokens += n
    model.train()
    if total_tokens == 0:
        return {"loss": float("nan"), "ppl": float("nan"), "tokens": 0}
    loss = total_loss / total_tokens
    return {"loss": loss, "ppl": math.exp(min(loss, 20)), "tokens": total_tokens}


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model, opt, cfg: SolisConfig, step: int,
                    tokens_seen: int, best_val: float, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val": best_val,
        "meta": meta,
    }, tmp)
    tmp.replace(path)  # atomic: a killed job never leaves a half-written file


def save_weights_only(path: Path, model, cfg: SolisConfig, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "config": asdict(cfg),
                "meta": meta}, tmp)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="mini")
    ap.add_argument("--packed", type=Path, default=DEFAULT_PACKED)
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop cleanly once this much wall clock has passed")
    ap.add_argument("--tokens-per-step", type=int, default=131072,
                    help="optimiser batch size in tokens; micro-batch and "
                         "accumulation are derived from it")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="override the auto-chosen micro-batch size")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--compile", action="store_true")
    # On by default: a top-k MoE stores k copies of every intermediate, so
    # recomputing is worth far more here than it would be in a dense model.
    ap.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                    action="store_false", default=True)
    ap.add_argument("--adam8bit", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = (torch.bfloat16 if device == "cuda"
                 and torch.cuda.is_bf16_supported() else torch.float32)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- config ---------------------------------------------------------- #
    cfg = get_config(args.preset)
    meta_path = args.packed / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"missing {meta_path} — run data/prepare.py first")
    packed_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # The vocab is whatever the tokenizer actually learned, rounded up to a
    # multiple of 128. Padding to a multiple of 128 makes the embedding and
    # output GEMMs land on tensor-core-friendly shapes; the spare ids are never
    # emitted because they are never in the training data.
    real_vocab = packed_meta["vocab_size"]
    cfg.vocab_size = int(math.ceil(real_vocab / 128) * 128)
    if args.seq_len:
        cfg.max_seq_len = args.seq_len

    # ---- data ------------------------------------------------------------ #
    train_data = PackedDataset(args.packed, "train", cfg.max_seq_len)
    val_data = (PackedDataset(args.packed, "val", cfg.max_seq_len)
                if "val" in packed_meta["splits"] else None)

    # ---- batch sizing ---------------------------------------------------- #
    micro = args.micro_batch or auto_micro_batch(cfg, device,
                                                 args.grad_checkpointing)
    tokens_per_micro = micro * cfg.max_seq_len
    accum = max(1, round(args.tokens_per_step / tokens_per_micro))
    tokens_per_step = micro * cfg.max_seq_len * accum

    # ---- model ----------------------------------------------------------- #
    model = Solis(cfg).to(device)
    if args.grad_checkpointing:
        model.enable_grad_checkpointing(True)
    raw_model = model
    if args.compile:
        print("compiling model (first steps will be slow)...")
        model = torch.compile(model)

    breakdown = cfg.param_breakdown()
    print(f"\n{'=' * 72}")
    print(f"Solis {cfg.name}")
    print(f"{'=' * 72}")
    print(f"  total params      {breakdown['total']:>14,}")
    print(f"  active per token  {breakdown['active_per_token']:>14,} "
          f"({100 * breakdown['active_per_token'] / breakdown['total']:.0f}%)")
    print(f"  layers x dim      {cfg.n_layers} x {cfg.dim}")
    print(f"  heads (q/kv)      {cfg.n_heads}/{cfg.n_kv_heads}  "
          f"head_dim {cfg.head_dim}")
    print(f"  experts           {cfg.n_experts} routed (top-{cfg.n_experts_per_tok})"
          f" + {cfg.n_shared_experts} shared")
    print(f"  context           {cfg.max_seq_len}")
    print(f"  vocab             {cfg.vocab_size:,} "
          f"(tokenizer learned {real_vocab:,})")
    print(f"  device            {device} / {amp_dtype}")
    print(f"  train tokens      {train_data.n_tokens:,}")
    print(f"  batch             {micro} x {cfg.max_seq_len} x {accum} accum "
          f"= {tokens_per_step:,} tokens/step")
    print(f"  planned           {args.steps:,} steps "
          f"= {args.steps * tokens_per_step / 1e6:,.0f}M tokens "
          f"({args.steps * tokens_per_step / train_data.n_tokens:.2f} epochs)")
    print(f"{'=' * 72}\n")

    opt = build_optimizer(raw_model, args.lr, args.weight_decay,
                          use_8bit=args.adam8bit)

    # ---- resume ---------------------------------------------------------- #
    ckpt_path = args.ckpt_dir / f"solis-{args.preset}.pt"
    start_step, tokens_seen, best_val = 0, 0, float("inf")
    if args.resume and ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        start_step = blob["step"]
        tokens_seen = blob.get("tokens_seen", 0)
        best_val = blob.get("best_val", float("inf"))
        print(f"resumed from {ckpt_path} at step {start_step:,}\n")

    # ---- graceful stop --------------------------------------------------- #
    stop = {"flag": False}

    def _handle(signum, frame):
        print("\ninterrupt received — finishing this step and saving")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)

    # ---- train ----------------------------------------------------------- #
    model.train()
    t_start = time.time()
    t_log = t_start
    running = {"loss": 0.0, "ce": 0.0, "aux": 0.0, "n": 0}
    history: list[dict] = []
    deadline = (t_start + args.max_minutes * 60) if args.max_minutes else None

    for step in range(start_step, args.steps):
        lr = lr_at(step, args.steps, args.warmup, args.lr, args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr

        # Run the accumulation loop, halving the micro-batch and starting the
        # step over if the card runs out. An estimate can be wrong; losing a
        # long run to it should not be possible.
        while True:
            try:
                opt.zero_grad(set_to_none=True)
                step_loss = step_ce = step_aux = 0.0
                for _ in range(accum):
                    x, y = train_data.batch(micro, rng, device)
                    with torch.autocast("cuda", dtype=amp_dtype,
                                        enabled=device == "cuda"):
                        _, out = model(x, targets=y)
                    loss, ce, aux = out
                    (loss / accum).backward()
                    step_loss += float(loss.detach()) / accum
                    step_ce += float(ce) / accum
                    step_aux += float(aux) / accum
                break
            except torch.OutOfMemoryError:
                if micro <= 1:
                    raise
                opt.zero_grad(set_to_none=True)
                del x, y
                torch.cuda.empty_cache()
                micro = max(1, micro // 2)
                accum = max(1, round(args.tokens_per_step
                                     / (micro * cfg.max_seq_len)))
                tokens_per_step = micro * cfg.max_seq_len * accum
                print(f"  OOM at step {step + 1}: micro-batch -> {micro}, "
                      f"accum -> {accum} ({tokens_per_step:,} tokens/step)")

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(),
                                                   args.grad_clip)
        opt.step()
        # Load balancing is applied here, once per optimiser step, rather than
        # inside the forward pass — see Solis.update_router_bias.
        raw_model.update_router_bias()
        tokens_seen += tokens_per_step

        running["loss"] += step_loss
        running["ce"] += step_ce
        running["aux"] += step_aux
        running["n"] += 1

        # ---- logging ----------------------------------------------------- #
        if (step + 1) % args.log_every == 0:
            now = time.time()
            dt = now - t_log
            t_log = now
            n = max(running["n"], 1)
            tok_s = args.log_every * tokens_per_step / max(dt, 1e-9)
            mem = (torch.cuda.max_memory_allocated() / 1024 ** 3
                   if device == "cuda" else 0.0)
            # Model FLOPs utilisation, using active (not total) parameters —
            # sparse layers genuinely do not run, so counting them would
            # flatter the number.
            flops = 6 * cfg.n_active_params * tok_s
            print(f"step {step + 1:>6,}/{args.steps:,} | "
                  f"loss {running['loss'] / n:6.3f} | "
                  f"ce {running['ce'] / n:6.3f} | "
                  f"aux {running['aux'] / n:5.3f} | "
                  f"lr {lr:.2e} | gnorm {float(grad_norm):5.2f} | "
                  f"{tok_s / 1e3:6.1f}k tok/s | "
                  f"{flops / 1e12:5.1f} TFLOP/s | "
                  f"{mem:4.1f}GB | "
                  f"{(now - t_start) / 60:5.1f}m")
            history.append({
                "step": step + 1, "loss": running["loss"] / n,
                "ce": running["ce"] / n, "aux": running["aux"] / n,
                "lr": lr, "tokens_seen": tokens_seen,
                "tokens_per_sec": tok_s, "peak_gb": mem,
                "minutes": (now - t_start) / 60,
            })
            running = {"loss": 0.0, "ce": 0.0, "aux": 0.0, "n": 0}

        # ---- eval -------------------------------------------------------- #
        is_last = (step + 1) == args.steps
        if val_data is not None and ((step + 1) % args.eval_every == 0 or is_last):
            ev = evaluate(model, val_data, micro, args.eval_batches, device,
                          amp_dtype)
            flag = ""
            if ev["loss"] < best_val:
                best_val = ev["loss"]
                save_weights_only(
                    args.ckpt_dir / f"solis-{args.preset}-best.pt", raw_model,
                    cfg, {"step": step + 1, "val_loss": best_val,
                          "tokens_seen": tokens_seen})
                flag = "  <- best"
            print(f"  eval @ {step + 1:,}: val loss {ev['loss']:.4f} | "
                  f"ppl {ev['ppl']:.2f} | {ev['tokens']:,} supervised tokens"
                  f"{flag}")
            history.append({"step": step + 1, "val_loss": ev["loss"],
                            "val_ppl": ev["ppl"]})

        # ---- checkpoint -------------------------------------------------- #
        if (step + 1) % args.save_every == 0 or is_last:
            save_checkpoint(ckpt_path, raw_model, opt, cfg, step + 1,
                            tokens_seen, best_val,
                            {"preset": args.preset, "lr": args.lr,
                             "tokens_per_step": tokens_per_step})

        if stop["flag"] or (deadline and time.time() > deadline):
            reason = "interrupt" if stop["flag"] else "time budget"
            print(f"\nstopping early ({reason}) at step {step + 1:,}")
            save_checkpoint(ckpt_path, raw_model, opt, cfg, step + 1,
                            tokens_seen, best_val,
                            {"preset": args.preset, "stopped": reason})
            break

    total_min = (time.time() - t_start) / 60
    print(f"\ntrained {tokens_seen:,} tokens in {total_min:.1f} minutes")
    print(f"checkpoint: {ckpt_path}")
    print(f"best val loss: {best_val:.4f}  (ppl {math.exp(min(best_val, 20)):.2f})")

    hist_path = args.ckpt_dir / f"solis-{args.preset}-history.json"
    hist_path.write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"history: {hist_path}")

    # ---- a look at what it learned --------------------------------------- #
    sample_prompts = [
        "who are you?",
        "What is 24 + 17?",
        "Sort these words alphabetically: heron otter badger falcon",
    ]
    tok_path = args.ckpt_dir / "tokenizer.json"
    if tok_path.exists():
        tok = SolisTokenizer.load(tok_path)
        raw_model.eval()
        print(f"\n{'-' * 72}\nsamples\n{'-' * 72}")
        for p in sample_prompts:
            ids = tok.encode_chat([{"role": "user", "content": p}])
            out = raw_model.generate(
                torch.tensor([ids], device=device), max_new_tokens=64,
                temperature=0.7, top_p=0.9,
                eos_id=__import__("solis.tokenizer", fromlist=["EOS"]).EOS)
            reply = tok.decode(out[0, len(ids):].tolist())
            print(f"  > {p}\n    {reply}\n")


def activation_bytes_per_token(cfg: SolisConfig, grad_checkpointing: bool) -> float:
    """Activation memory held for the backward pass, per token.

    The MoE term is the one that matters and the one that is easy to get wrong.
    Routing top-k materialises k copies of each token's hidden state *and* k
    copies of every SwiGLU intermediate, so a top-4 layer stores roughly an
    order of magnitude more than a dense layer of the same width. That is what
    makes the naive micro-batch estimate blow up.
    """
    c = cfg.dim
    h = cfg.expert_hidden
    k = cfg.n_experts_per_tok + cfg.n_shared_experts

    if grad_checkpointing:
        # Only each block's input is kept; everything else is recomputed. Peak
        # transiently holds one block's worth on top of that.
        stored = cfg.n_layers * c
        recompute_peak = k * (2 * c + 4 * h) + 12 * c
        return (stored + recompute_peak) * 2.0

    attn_per_layer = 12 * c                       # q,k,v,o,resid,norms
    moe_per_layer = k * (2 * c + 4 * h)           # dispatch + SwiGLU intermediates
    dense_per_layer = 4 * cfg.dense_hidden
    total = (cfg.n_layers * attn_per_layer
             + cfg.n_moe_layers * moe_per_layer
             + cfg.dense_layers * dense_per_layer)
    return total * 2.0


def auto_micro_batch(cfg: SolisConfig, device: str,
                     grad_checkpointing: bool) -> int:
    """Pick a micro-batch that fits the card, from free VRAM and the config.

    Deliberately conservative — an OOM twenty minutes into a run costs far more
    than a slightly smaller batch does.
    """
    if device != "cuda":
        return 1
    free, _total = torch.cuda.mem_get_info()
    free_gb = free / 1024 ** 3
    # fp32 weights + fp32 grads + AdamW's two fp32 moments = 16 bytes/param.
    fixed = cfg.n_params * 16 / 1024 ** 3
    budget = max(0.5, free_gb - fixed - 2.0)  # 2GB for workspace/fragmentation

    per_token = activation_bytes_per_token(cfg, grad_checkpointing)
    # Loss memory scales with vocab, not depth. Only supervised positions reach
    # the vocab head (see Solis.forward), which is roughly a third of them on
    # chat data, but assume half and count logits + fp32 copy + softmax + grad.
    per_token += 0.5 * cfg.vocab_size * (2 + 4 + 4 + 4)
    per_seq = per_token * cfg.max_seq_len / 1024 ** 3

    # A 30% margin on an analytic estimate is cheap insurance; the OOM backoff
    # in the training loop is the real safety net.
    n = max(1, min(int(0.7 * budget / max(per_seq, 1e-9)), 64))
    print(f"auto micro-batch: {n}  (free {free_gb:.1f}GB, "
          f"model+opt {fixed:.1f}GB, {per_seq * 1024:.0f}MB/sequence, "
          f"checkpointing {'on' if grad_checkpointing else 'off'})")
    return n


if __name__ == "__main__":
    main()
