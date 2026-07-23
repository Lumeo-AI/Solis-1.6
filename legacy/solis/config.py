"""Model + training configuration for Solis.

Solis is a sparse Mixture-of-Experts decoder. The design goal for the 1.0 family
is a hard constraint: **the whole thing must serve inside 16 GB of VRAM** —
weights, KV cache, and activation workspace included — while keeping the number
of *active* parameters per token low so decoding stays fast.

Every preset below is checked against that budget by `vram_report()`, which is
an analytic model of the memory a config actually needs. `python -m solis.config`
prints the table.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import json


# Bytes per parameter for the supported weight dtypes.
DTYPE_BYTES = {
    "float32": 4.0,
    "bfloat16": 2.0,
    "float16": 2.0,
    "int8": 1.0,
    "nf4": 0.5625,  # 4-bit weight + fp16 absmax scale per 64-element block
}


@dataclass
class SolisConfig:
    """Shape of a Solis model.

    Defaults describe `solis-1.0-mini`, the preset small enough to *train* on a
    single 16 GB consumer card. The larger presets in `PRESETS` share this
    schema and are what the 16 GB *serving* budget is designed around.
    """

    name: str = "solis-1.0-mini"

    # --- vocabulary -------------------------------------------------------
    # Byte-level BPE trained from scratch on our own corpus. No external vocab
    # file, no borrowed tokenizer. 256 byte tokens are always present as a
    # fallback so nothing is ever unencodable.
    vocab_size: int = 32768

    # --- transformer shape ------------------------------------------------
    dim: int = 768           # residual stream width
    n_layers: int = 16       # transformer blocks
    n_heads: int = 12        # query heads
    n_kv_heads: int = 4      # key/value heads (GQA; must divide n_heads)
    head_dim: int = 64       # per-head width; dim need not equal n_heads*head_dim
    max_seq_len: int = 2048  # trained context window

    # --- mixture of experts ----------------------------------------------
    # `n_shared_experts` experts run for *every* token (they carry the general
    # -purpose computation), while the router picks `n_experts_per_tok` of the
    # `n_experts` specialists. This split is what keeps a sparse model coherent
    # at small scale: the shared path never has to be re-learned by each expert.
    n_experts: int = 16
    n_experts_per_tok: int = 4
    n_shared_experts: int = 1
    expert_hidden: int = 512       # SwiGLU inner width per expert
    dense_layers: int = 1          # first N layers use a plain dense FFN
    dense_hidden: int = 2048       # SwiGLU inner width for those dense layers

    # Dispatch capacity per expert, as a multiple of the even share
    # (tokens * top_k / n_experts). Headroom above 1.0 absorbs the routing
    # imbalance that always exists batch to batch; pairs beyond it are dropped.
    # Raising it costs memory, lowering it costs dropped tokens.
    capacity_factor: float = 1.25

    # Router regularisation.
    aux_loss_coef: float = 0.01       # load balancing across experts
    router_z_loss_coef: float = 1e-3  # keeps router logits from exploding
    router_bias_update_rate: float = 1e-3  # aux-loss-free bias correction
    norm_topk_prob: bool = True

    # --- attention details -------------------------------------------------
    qk_norm: bool = True             # RMSNorm on q and k; major stability win
    attn_logit_softcap: float = 0.0  # 0 disables; >0 caps logits at +/- value
    # Sliding-window attention trades a little quality for attention cost that
    # grows linearly rather than quadratically in context. It is not free in
    # PyTorch: an explicit mask drops `scaled_dot_product_attention` off its
    # fused flash kernel onto the slower memory-efficient one, measured here at
    # about a 10% end-to-end penalty. That is a clear loss at short context and
    # a clear win at long context, so the short-context preset leaves it off.
    sliding_window: int = 0          # 0 = full attention on every layer
    sliding_window_pattern: int = 0  # every Nth layer is full attention

    # --- positions ---------------------------------------------------------
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 1.0   # >1 extends context at inference

    # --- normalisation / regularisation -----------------------------------
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    init_std: float = 0.02
    logit_softcap: float = 0.0

    # --- runtime -----------------------------------------------------------
    dtype: str = "bfloat16"

    # ------------------------------------------------------------------ #
    # Derived quantities
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        if self.n_experts_per_tok > self.n_experts:
            raise ValueError("n_experts_per_tok cannot exceed n_experts")
        if self.dense_layers > self.n_layers:
            raise ValueError("dense_layers cannot exceed n_layers")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"unknown dtype {self.dtype!r}")

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def n_moe_layers(self) -> int:
        return self.n_layers - self.dense_layers

    def is_sliding(self, layer_idx: int) -> bool:
        """True if this layer uses a sliding-window mask."""
        if self.sliding_window <= 0:
            return False
        if self.sliding_window_pattern <= 0:
            return True
        # Every Nth layer stays global so information can still travel far.
        return (layer_idx + 1) % self.sliding_window_pattern != 0

    # ------------------------------------------------------------------ #
    # Parameter accounting
    # ------------------------------------------------------------------ #
    def param_breakdown(self) -> dict[str, int]:
        """Exact parameter counts, derived from the shapes in model.py."""
        d = self.dim
        embed = self.vocab_size * d
        head = 0 if self.tie_embeddings else self.vocab_size * d

        # Attention: q/k/v/o projections (+ QK norm weights).
        attn = d * self.q_dim + d * self.kv_dim + d * self.kv_dim + self.q_dim * d
        if self.qk_norm:
            attn += 2 * self.head_dim
        attn_all = attn * self.n_layers

        # SwiGLU block = gate + up + down.
        def swiglu(hidden: int) -> int:
            return 3 * d * hidden

        dense_ffn = swiglu(self.dense_hidden) * self.dense_layers

        routed = swiglu(self.expert_hidden) * self.n_experts
        shared = swiglu(self.expert_hidden) * self.n_shared_experts
        # The router is a single bias-free projection. Its load-balancing bias
        # is a buffer, not a parameter — it is updated by a rule, not a gradient.
        router = d * self.n_experts
        moe_all = (routed + shared + router) * self.n_moe_layers

        # RMSNorms: 2 per block + 1 final.
        norms = d * (2 * self.n_layers + 1)

        total = embed + head + attn_all + dense_ffn + moe_all + norms

        # Active parameters: what actually runs for a single token.
        active_routed = swiglu(self.expert_hidden) * self.n_experts_per_tok
        active_moe = (active_routed + shared + router) * self.n_moe_layers
        active = embed + head + attn_all + dense_ffn + active_moe + norms

        return {
            "embedding": embed,
            "lm_head": head,
            "attention": attn_all,
            "dense_ffn": dense_ffn,
            "moe": moe_all,
            "norms": norms,
            "total": total,
            "active_per_token": active,
        }

    @property
    def n_params(self) -> int:
        return self.param_breakdown()["total"]

    @property
    def n_active_params(self) -> int:
        return self.param_breakdown()["active_per_token"]

    # ------------------------------------------------------------------ #
    # Memory accounting
    # ------------------------------------------------------------------ #
    def kv_cache_bytes(self, seq_len: int | None = None, batch: int = 1,
                       dtype: str | None = None) -> int:
        """KV cache size. GQA is what makes this cheap: we store `n_kv_heads`
        key/value pairs per layer, not `n_heads`."""
        seq_len = seq_len or self.max_seq_len
        per_byte = DTYPE_BYTES[dtype or self.dtype]
        # 2 (K and V) * layers * kv_heads * head_dim
        per_token = 2 * self.n_layers * self.n_kv_heads * self.head_dim
        return int(per_token * seq_len * batch * per_byte)

    def weight_bytes(self, dtype: str | None = None) -> int:
        return int(self.n_params * DTYPE_BYTES[dtype or self.dtype])

    def activation_bytes(self, seq_len: int | None = None, batch: int = 1,
                         dtype: str | None = None,
                         full_logits: bool = False) -> int:
        """Rough peak transient workspace for a forward pass.

        `full_logits=False` matches the serving path in model.py: prefill runs
        the stack over the whole prompt but only projects the *last* hidden
        state through the vocab head, so the (seq x vocab) logits tensor — which
        would otherwise dominate everything else here — never exists. Training
        needs every position, hence the flag.
        """
        seq_len = seq_len or self.max_seq_len
        per_byte = DTYPE_BYTES[dtype or self.dtype]
        logit_positions = seq_len if full_logits else 1
        logits = batch * logit_positions * self.vocab_size * 4  # fp32
        # MoE dispatch materialises top-k copies of the hidden state.
        moe_ws = (batch * seq_len * self.dim
                  * (self.n_experts_per_tok + self.n_shared_experts)
                  * 3 * per_byte)
        resid = batch * seq_len * self.dim * per_byte * 8
        return int(logits + moe_ws + resid)

    def training_bytes(self, seq_len: int | None = None, batch: int = 1,
                       optimizer: str = "adamw",
                       grad_checkpointing: bool = True) -> int:
        """Memory needed to *train* this config (weights + grads + optimizer +
        activations). Training is far heavier than serving — this is why the
        larger presets are inference-only on a 16 GB card."""
        seq_len = seq_len or self.max_seq_len
        n = self.n_params
        # Master weights and gradients.
        weights = n * DTYPE_BYTES[self.dtype]
        grads = n * DTYPE_BYTES[self.dtype]
        opt = {"adamw": 8.0, "adamw8bit": 2.0, "sgd": 4.0}[optimizer] * n
        # Stored activations for backward.
        per_layer = batch * seq_len * self.dim * DTYPE_BYTES[self.dtype]
        acts = per_layer * (2 if grad_checkpointing else 24) * self.n_layers
        acts += batch * seq_len * self.vocab_size * 4 * 2  # logits + grad
        return int(weights + grads + opt + acts)

    def vram_report(self, seq_len: int | None = None, batch: int = 1,
                    dtype: str | None = None, budget_gb: float = 16.0) -> dict:
        """Everything needed to serve this config, in GB, versus the budget."""
        seq_len = seq_len or self.max_seq_len
        dtype = dtype or self.dtype
        gb = 1024 ** 3
        w = self.weight_bytes(dtype) / gb
        kv = self.kv_cache_bytes(seq_len, batch, dtype) / gb
        act = self.activation_bytes(seq_len, batch, dtype) / gb
        # CUDA context, cuBLAS workspaces, allocator fragmentation.
        overhead = 0.9
        total = w + kv + act + overhead
        return {
            "name": self.name,
            "dtype": dtype,
            "seq_len": seq_len,
            "batch": batch,
            "params_total": self.n_params,
            "params_active": self.n_active_params,
            "weights_gb": round(w, 2),
            "kv_cache_gb": round(kv, 2),
            "activations_gb": round(act, 2),
            "runtime_overhead_gb": overhead,
            "total_gb": round(total, 2),
            "budget_gb": budget_gb,
            "headroom_gb": round(budget_gb - total, 2),
            "fits": total <= budget_gb,
        }

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "SolisConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
# The 1.0 family
# --------------------------------------------------------------------------- #
# Each preset is a point on the same curve: raise total parameters (which buys
# capacity) while holding active parameters low (which buys speed), until the
# weights fill the 16 GB card. `flagship` is the largest config that still
# leaves room for an 8k-token KV cache and activation workspace.

PRESETS: dict[str, SolisConfig] = {
    # Trainable end to end on a single 16 GB card.
    "mini": SolisConfig(
        name="solis-1.0-mini",
        vocab_size=32768,
        dim=768, n_layers=16, n_heads=12, n_kv_heads=4, head_dim=64,
        max_seq_len=2048,
        n_experts=16, n_experts_per_tok=4, n_shared_experts=1,
        expert_hidden=512, dense_layers=1, dense_hidden=2048,
        # Full attention: at 2048 tokens a 1024 window saves little and costs
        # the flash kernel. See the note on `sliding_window` above.
        sliding_window=0, sliding_window_pattern=0,
    ),
    # Fine-tunable with LoRA / 8-bit optimizers on 16 GB.
    "small": SolisConfig(
        name="solis-1.0-small",
        vocab_size=32768,
        dim=1280, n_layers=24, n_heads=20, n_kv_heads=4, head_dim=64,
        max_seq_len=4096,
        n_experts=24, n_experts_per_tok=4, n_shared_experts=1,
        expert_hidden=768, dense_layers=2, dense_hidden=3584,
        sliding_window=2048, sliding_window_pattern=4,
    ),
    # Comfortable bf16 serving with long context.
    "base": SolisConfig(
        name="solis-1.0-base",
        vocab_size=49152,
        dim=1792, n_layers=28, n_heads=28, n_kv_heads=4, head_dim=64,
        max_seq_len=8192,
        n_experts=32, n_experts_per_tok=4, n_shared_experts=1,
        expert_hidden=896, dense_layers=2, dense_hidden=4864,
        sliding_window=4096, sliding_window_pattern=4,
    ),
    # The 16 GB flagship: as much total capacity as the card holds in bf16,
    # with active parameters kept near 1.5B so it decodes like a small model.
    "flagship": SolisConfig(
        name="solis-1.0",
        vocab_size=65536,
        dim=2048, n_layers=30, n_heads=32, n_kv_heads=8, head_dim=64,
        max_seq_len=8192,
        n_experts=32, n_experts_per_tok=4, n_shared_experts=1,
        expert_hidden=1024, dense_layers=2, dense_hidden=5632,
        sliding_window=4096, sliding_window_pattern=4,
    ),
}

DEFAULT_PRESET = "mini"


def get_config(name: str = DEFAULT_PRESET) -> SolisConfig:
    """Look up a preset by short name ('mini') or full name ('solis-1.0-mini')."""
    if name in PRESETS:
        return PRESETS[name]
    for cfg in PRESETS.values():
        if cfg.name == name:
            return cfg
    raise KeyError(
        f"unknown preset {name!r}; choose from "
        f"{sorted(PRESETS)} or {[c.name for c in PRESETS.values()]}"
    )


def _fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(n)


def print_family(budget_gb: float = 16.0) -> None:
    """Print the family table with the VRAM budget check."""
    head = (f"{'preset':<10} {'total':>8} {'active':>8} {'ctx':>6} "
            f"{'weights':>8} {'kv':>7} {'act':>7} {'peak':>7} {'fits 16GB':>10}")
    print(head)
    print("-" * len(head))
    for key, cfg in PRESETS.items():
        r = cfg.vram_report(budget_gb=budget_gb)
        print(f"{key:<10} {_fmt(r['params_total']):>8} "
              f"{_fmt(r['params_active']):>8} {cfg.max_seq_len:>6} "
              f"{r['weights_gb']:>7.2f}G {r['kv_cache_gb']:>6.2f}G "
              f"{r['activations_gb']:>6.2f}G {r['total_gb']:>6.2f}G "
              f"{('yes' if r['fits'] else 'NO'):>10}")
    print()
    print("Training memory (AdamW, grad checkpointing, batch 1):")
    for key, cfg in PRESETS.items():
        tb = cfg.training_bytes(batch=1) / 1024 ** 3
        tb8 = cfg.training_bytes(batch=1, optimizer="adamw8bit") / 1024 ** 3
        note = "trainable on 16GB" if tb8 < 14 else "needs multi-GPU / offload"
        print(f"  {key:<10} adamw {tb:6.1f}G | adamw8bit {tb8:6.1f}G   {note}")


if __name__ == "__main__":  # pragma: no cover
    print_family()
