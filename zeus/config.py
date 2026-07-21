"""Model + training configuration for Zeus."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json


@dataclass
class ZeusConfig:
    # --- vocabulary ---
    # Byte-level tokenizer: 256 raw bytes + a few special tokens. No external
    # vocab file, no pretrained anything — the model learns text from bytes up.
    vocab_size: int = 260  # 256 bytes + BOS + EOS + PAD + USER/ASST markers folded in below

    # --- transformer shape ---
    dim: int = 384          # residual stream width
    n_layers: int = 6       # number of transformer blocks
    n_heads: int = 6        # attention heads (dim must be divisible by n_heads)
    max_seq_len: int = 512  # context window

    # --- mixture of experts ---
    n_experts: int = 8      # experts per MoE layer
    n_experts_per_tok: int = 2   # top-k routing
    expert_hidden_mult: float = 2.0  # SwiGLU hidden = dim * mult (per expert)
    aux_loss_coef: float = 0.01      # load-balancing loss weight
    router_z_loss_coef: float = 1e-3 # router logit stabilisation

    # --- regularisation ---
    dropout: float = 0.0
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        return self.dim // self.n_heads

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ZeusConfig":
        fields = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**fields)
