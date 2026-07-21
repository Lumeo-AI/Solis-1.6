"""Train Zeus from scratch on the bundled chat corpus.

Usage:
    python data/build_corpus.py      # generate data/corpus.jsonl (once)
    python train.py                  # train, writes checkpoints/zeus.pt

This is a small run designed to finish in a few minutes on an Apple-silicon Mac
(MPS) or CPU. It is enough for Zeus to learn the chat format and its corpus, not
to become a general assistant.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from zeus.config import ZeusConfig
from zeus.model import Zeus
from zeus.tokenizer import ByteTokenizer

ROOT = Path(__file__).parent
CKPT = ROOT / "checkpoints" / "zeus.pt"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_examples(tok: ByteTokenizer, max_seq_len: int):
    """Tokenise each conversation; mask the loss so we only learn assistant tokens."""
    path = ROOT / "data" / "corpus.jsonl"
    if not path.exists():
        raise SystemExit("Missing data/corpus.jsonl — run: python data/build_corpus.py")

    from zeus.tokenizer import BOS, EOS, USER, ASST
    examples = []
    for line in path.read_text().splitlines():
        msgs = json.loads(line)["messages"]
        ids, mask = [BOS], [0]
        for m in msgs:
            marker = USER if m["role"] == "user" else ASST
            body = tok.encode(m["content"]) + [EOS]
            ids.append(marker); mask.append(0)
            ids.extend(body)
            # Learn to produce assistant content + its EOS; ignore user tokens.
            learn = 1 if m["role"] == "assistant" else 0
            mask.extend([learn] * len(body))
        # Cap training length for speed; conversations are short anyway.
        train_cap = min(max_seq_len, 256)
        ids = ids[:train_cap]
        mask = mask[:train_cap]
        if len(ids) > 8:
            examples.append((ids, mask))
    return examples


def make_batch(examples, batch_size, max_seq_len, device):
    import random
    batch = random.sample(examples, min(batch_size, len(examples)))
    maxlen = max(len(ids) for ids, _ in batch)
    from zeus.tokenizer import EOS
    x = torch.full((len(batch), maxlen), EOS, dtype=torch.long)
    y = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    for i, (ids, mask) in enumerate(batch):
        t = torch.tensor(ids)
        x[i, : len(ids)] = t
        # target[j] is token j+1; only supervise where mask==1
        for j in range(len(ids) - 1):
            if mask[j + 1] == 1:
                y[i, j] = ids[j + 1]
    return x.to(device), y.to(device)


def main():
    device = pick_device()
    print(f"device: {device}")
    torch.manual_seed(1337)

    cfg = ZeusConfig()
    tok = ByteTokenizer()
    model = Zeus(cfg).to(device)
    print(f"Zeus: {model.num_params()/1e6:.2f}M params | "
          f"{cfg.n_experts} experts, top-{cfg.n_experts_per_tok}")

    examples = load_examples(tok, cfg.max_seq_len)
    print(f"loaded {len(examples)} training conversations")

    import os
    steps = int(os.environ.get("ZEUS_STEPS", "600"))
    batch_size = int(os.environ.get("ZEUS_BATCH", "24"))
    warmup = 60
    lr_max = 3e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(step):
        if step < warmup:
            return lr_max * step / warmup
        prog = (step - warmup) / max(1, steps - warmup)
        return lr_max * 0.1 + 0.5 * lr_max * 0.9 * (1 + math.cos(math.pi * prog))

    model.train()
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = make_batch(examples, batch_size, cfg.max_seq_len, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            dt = time.time() - t0
            print(f"step {step:4d}/{steps} | loss {loss.item():.3f} | "
                  f"lr {lr_at(step):.1e} | {dt:5.1f}s")

    CKPT.parent.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, CKPT)
    print(f"saved checkpoint -> {CKPT}")

    # Quick sanity sample.
    from zeus.tokenizer import ASST, EOS
    ids = tok.encode_chat([{"role": "user", "content": "who are you?"}])
    idx = torch.tensor([ids], device=device)
    out = model.generate(idx, max_new_tokens=120, eos_id=EOS, temperature=0.7)
    print("SAMPLE >>", tok.decode(out[0, len(ids):].tolist()))


if __name__ == "__main__":
    main()
