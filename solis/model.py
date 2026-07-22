"""Solis: a decoder-only sparse Mixture-of-Experts transformer, from scratch.

Nothing here is pretrained and nothing is imported from another model — weights
start random and the whole stack is defined in this file.

What is in the block:

  * **RMSNorm** pre-normalisation, plus a post-norm on each residual branch.
  * **Grouped-query attention (GQA)** — fewer key/value heads than query heads,
    which is what makes the KV cache small enough to serve long context in 16 GB.
  * **QK-norm** — RMSNorm applied to queries and keys before the dot product.
    Removes the attention-logit blowup that otherwise makes small models
    diverge at high learning rates.
  * **Sliding-window attention** on most layers, with every Nth layer left
    global, so cost grows linearly in context while information can still
    travel the full sequence.
  * **Mixture-of-Experts FFN** — a shared expert that runs for every token plus
    a top-k routed set of specialists. Dispatch is done by sorting tokens by
    expert and running one GEMM per expert, not by masking in a Python loop.
  * **Aux-loss-free load balancing** — a per-expert bias nudges routing toward
    under-used experts without adding gradient noise to the main objective,
    backed by a small classic auxiliary loss and a router z-loss.

Generation uses an incremental KV cache, so decoding a token is O(1) in the
number of tokens already generated rather than a full re-forward.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .config import SolisConfig


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in fp32 regardless of the autocast dtype; this is cheap and
        # is the difference between a stable and an unstable bf16 run.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


# --------------------------------------------------------------------------- #
# Rotary position embeddings
# --------------------------------------------------------------------------- #
def build_rope_cache(seq_len: int, head_dim: int, theta: float,
                     device, scaling_factor: float = 1.0):
    """Precompute cos/sin tables.

    `scaling_factor` > 1 stretches positions (NTK-free linear interpolation),
    which lets a model trained at `max_seq_len` be served at a longer context.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float()
                                / head_dim))
    t = torch.arange(seq_len, device=device).float()
    if scaling_factor != 1.0:
        t = t / scaling_factor
    freqs = torch.outer(t, inv_freq)              # (seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (seq, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               offset: int = 0) -> torch.Tensor:
    """x: (batch, heads, seq, head_dim). `offset` is the number of cached
    positions already consumed, so incremental decoding rotates correctly."""
    seq = x.shape[2]
    c = cos[offset:offset + seq].to(x.dtype)[None, None]
    s = sin[offset:offset + seq].to(x.dtype)[None, None]
    return x * c + rotate_half(x) * s


# --------------------------------------------------------------------------- #
# KV cache
# --------------------------------------------------------------------------- #
class KVCache:
    """Per-layer key/value cache for incremental decoding.

    Preallocated to `max_seq_len` so decoding never reallocates. When a
    sliding-window layer overruns its window the oldest entries are simply
    never attended to (the mask handles it) — we keep them so global layers,
    which share the same cache, still see the full history.
    """

    def __init__(self, cfg: SolisConfig, batch: int, max_seq_len: int,
                 device, dtype):
        self.max_seq_len = max_seq_len
        self.pos = 0
        shape = (batch, cfg.n_kv_heads, max_seq_len, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype)
                  for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype)
                  for _ in range(cfg.n_layers)]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Append `k`/`v` for one layer and return the full cached history."""
        t = k.shape[2]
        end = self.pos + t
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: {end} > {self.max_seq_len}. "
                "Increase max_seq_len or trim the conversation."
            )
        self.k[layer][:, :, self.pos:end] = k
        self.v[layer][:, :, self.pos:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n: int):
        self.pos += n

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


def build_attn_mask(q_len: int, kv_len: int, window: int, device) -> Optional[torch.Tensor]:
    """Boolean mask (True = attend) of shape (q_len, kv_len).

    Returns None when plain causal masking is enough, so the caller can use the
    fused `is_causal` fast path instead of materialising a mask.
    """
    if window <= 0 and q_len == kv_len:
        return None  # caller uses is_causal=True
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)[:, None]
    k_pos = torch.arange(kv_len, device=device)[None, :]
    mask = k_pos <= q_pos
    if window > 0:
        mask &= (q_pos - k_pos) < window
    return mask


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, cfg: SolisConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.window = cfg.sliding_window if cfg.is_sliding(layer_idx) else 0
        self.softcap = cfg.attn_logit_softcap
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(cfg.dim, cfg.q_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.kv_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.kv_dim, bias=False)
        self.wo = nn.Linear(cfg.q_dim, cfg.dim, bias=False)

        if cfg.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        else:
            self.q_norm = self.k_norm = None

        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache: Optional[KVCache] = None, offset: int = 0):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        kv_len = k.shape[2]

        # Expand KV heads to match query heads (GQA).
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if self.softcap > 0:
            # Soft-capping needs the explicit logits, so no fused kernel here.
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = torch.tanh(attn / self.softcap) * self.softcap
            mask = build_attn_mask(T, kv_len, self.window, x.device)
            if mask is None:
                mask = build_attn_mask(T, kv_len, 0, x.device)
                if mask is None:  # T == kv_len, plain causal
                    mask = torch.ones(T, kv_len, dtype=torch.bool,
                                      device=x.device).tril()
            attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn.float(), dim=-1).to(q.dtype)
            out = torch.matmul(attn, v)
        else:
            mask = build_attn_mask(T, kv_len, self.window, x.device)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                is_causal=(mask is None),
                dropout_p=self.dropout if self.training else 0.0,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# --------------------------------------------------------------------------- #
# Feed-forward
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    """Dense SwiGLU feed-forward, used by the first few layers.

    Early layers do broad, low-level work that every token needs, so routing
    them wastes capacity — they stay dense.
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, dim, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GroupedExperts(nn.Module):
    """`n_experts` SwiGLU experts stored as stacked 3-D weight tensors.

    Keeping the experts in one tensor rather than a `ModuleList` of `Linear`s is
    what lets the whole layer run as three batched matmuls. The alternative —
    slicing the token stream per expert and looping — issues `3 * n_experts`
    small GEMMs per layer and needs a host synchronisation to learn the slice
    boundaries, which on this model measured several times slower end to end.
    """

    def __init__(self, n_experts: int, dim: int, hidden: int):
        super().__init__()
        self.n_experts = n_experts
        self.dim = dim
        self.hidden = hidden
        self.w1 = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.w3 = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.w2 = nn.Parameter(torch.empty(n_experts, hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n_experts, capacity, dim) -> same shape."""
        h = F.silu(torch.bmm(x, self.w1)) * torch.bmm(x, self.w3)
        return torch.bmm(h, self.w2)


class MoE(nn.Module):
    """Shared expert + top-k routed experts."""

    def __init__(self, cfg: SolisConfig):
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_experts_per_tok
        self.n_shared = cfg.n_shared_experts
        self.norm_topk = cfg.norm_topk_prob

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        # Selection-only bias for aux-loss-free load balancing. It shifts which
        # experts win the top-k, but never scales their output, so it changes
        # routing without distorting the forward computation.
        self.register_buffer("expert_bias", torch.zeros(cfg.n_experts))
        self.register_buffer("expert_load", torch.zeros(cfg.n_experts))

        self.experts = GroupedExperts(cfg.n_experts, cfg.dim, cfg.expert_hidden)
        self.shared = (SwiGLU(cfg.dim, cfg.expert_hidden * cfg.n_shared_experts)
                       if cfg.n_shared_experts > 0 else None)

        # Most recent per-expert load, recorded by forward and consumed by
        # `update_router_bias` after the optimiser step. Deliberately plain
        # attributes, not buffers: forward must not mutate module state.
        self._last_load: Optional[torch.Tensor] = None
        self._last_drop_rate: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        N = flat.shape[0]

        logits = self.router(flat.float())            # (N, E) — router in fp32
        probs = F.softmax(logits, dim=-1)

        # Selection uses the balancing bias; the weights used to combine expert
        # outputs come from the unbiased probabilities.
        sel = probs + self.expert_bias
        _, topk_idx = sel.topk(self.top_k, dim=-1)     # (N, k)
        topk_w = probs.gather(-1, topk_idx)
        if self.norm_topk:
            topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-9)

        # ---- dispatch ------------------------------------------------------
        # Every (token, slot) pair is placed into a buffer of shape
        # (n_experts, capacity, dim), so the whole layer runs as three batched
        # matmuls. Pairs past an expert's capacity are routed to a trailing
        # scratch row whose output is zeroed, so they contribute nothing rather
        # than corrupting another token's slot.
        E, k = self.n_experts, self.top_k
        n_pairs = N * k

        flat_expert = topk_idx.reshape(-1)                     # (N*k,)
        order = torch.argsort(flat_expert, stable=True)
        sorted_expert = flat_expert[order]
        token_idx = order.div(k, rounding_mode="floor")        # (N*k,)
        weights = topk_w.reshape(-1)[order]

        counts = torch.bincount(sorted_expert, minlength=E)
        starts = torch.cumsum(counts, 0) - counts              # (E,)
        rank = (torch.arange(n_pairs, device=x.device)
                - starts[sorted_expert])                       # position in group

        if self.training:
            # A fixed capacity keeps every shape independent of how the router
            # happened to split this batch, so no count is ever read back to
            # the host and the step never stalls on a sync.
            cap = max(1, int(self.cfg.capacity_factor * n_pairs / E))
        else:
            # At inference, capacity is the largest actual group. This costs one
            # device-to-host read per layer but guarantees nothing is dropped —
            # without it a token's output would depend on how many other tokens
            # happened to share its batch, and a prefill would disagree with the
            # same tokens decoded one at a time.
            cap = max(1, int(counts.max()))

        overflow = rank >= cap
        slot = torch.where(overflow, torch.full_like(rank, E * cap),
                           sorted_expert * cap + rank)

        wdtype = self.experts.w1.dtype
        buf = flat.new_zeros((E * cap + 1, C), dtype=wdtype)
        buf[slot] = flat[token_idx].to(wdtype)

        expert_out = self.experts(buf[:E * cap].view(E, cap, C))
        # The scratch row contributes zero, which is what makes dropped pairs
        # harmless instead of wrong.
        expert_out = torch.cat(
            [expert_out.reshape(E * cap, C),
             expert_out.new_zeros((1, C))], dim=0)

        contrib = expert_out[slot] * weights.unsqueeze(-1).to(expert_out.dtype)
        out = torch.zeros_like(flat, dtype=expert_out.dtype)
        out.index_add_(0, token_idx, contrib)

        if self.shared is not None:
            out = out + self.shared(flat)

        out = out.reshape(B, T, C).to(x.dtype)

        # ---- balancing signals --------------------------------------------
        if not self.training:
            return out, out.new_zeros(())

        with torch.no_grad():
            # Recorded, not applied. Applying it here would change routing
            # between the forward pass and its recomputation under gradient
            # checkpointing, which silently changes what the layer computes.
            self._last_load = counts.float() / max(n_pairs, 1)
            self._last_drop_rate = overflow.float().mean()

        # Classic Switch-style auxiliary loss, kept small — it is a backstop for
        # the bias rule, not the primary balancing mechanism. Returned rather
        # than stashed on the module, so that when this block is recomputed by
        # gradient checkpointing the loss stays attached to the live graph.
        frac = torch.zeros(self.n_experts, device=x.device, dtype=probs.dtype)
        frac.index_add_(0, flat_expert,
                        torch.ones_like(flat_expert, dtype=probs.dtype))
        frac = frac / max(N * self.top_k, 1)
        aux = self.n_experts * (frac * probs.mean(0)).sum()
        z_loss = logits.logsumexp(dim=-1).pow(2).mean()
        aux_total = (self.cfg.aux_loss_coef * aux
                     + self.cfg.router_z_loss_coef * z_loss)
        return out, aux_total.to(out.dtype)

    @torch.no_grad()
    def apply_bias_update(self) -> None:
        """Nudge routing toward under-used experts.

        Called once per optimiser step, never inside forward. Sign-only updates
        keep this from fighting the main gradient, and because the bias affects
        selection but not the mixing weights, it rebalances load without
        distorting what the layer computes.
        """
        if self._last_load is None:
            return
        target = 1.0 / self.n_experts
        self.expert_bias.add_(
            torch.sign(target - self._last_load) * self.cfg.router_bias_update_rate)
        self.expert_load.mul_(0.99).add_(self._last_load, alpha=0.01)


# --------------------------------------------------------------------------- #
# Block
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, cfg: SolisConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.is_moe = layer_idx >= cfg.dense_layers
        self.ffn = MoE(cfg) if self.is_moe else SwiGLU(cfg.dim, cfg.dense_hidden)

    def forward(self, x, cos, sin, cache=None, offset=0):
        """Returns (hidden, aux_loss). The aux term is returned rather than
        stored so it survives gradient checkpointing."""
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, offset)
        if self.is_moe:
            ffn_out, aux = self.ffn(self.ffn_norm(x))
        else:
            ffn_out, aux = self.ffn(self.ffn_norm(x)), x.new_zeros(())
        return x + ffn_out, aux


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Solis(nn.Module):
    def __init__(self, cfg: SolisConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.grad_checkpointing = False
        self._rope: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._rope_len = 0

        self.apply(self._init_weights)
        # Scale the output projection of every residual branch by 1/sqrt(2L).
        # Without this the residual stream grows with depth and deep models need
        # a much lower learning rate to stay stable.
        scale = (2 * cfg.n_layers) ** -0.5
        for block in self.blocks:
            block.attn.wo.weight.data.mul_(scale)
            if block.is_moe:
                block.ffn.experts.w2.data.mul_(scale)
                if block.ffn.shared is not None:
                    block.ffn.shared.w2.weight.data.mul_(scale)
            else:
                block.ffn.w2.weight.data.mul_(scale)

    # -- init ------------------------------------------------------------- #
    def _init_weights(self, m):
        std = self.cfg.init_std
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, GroupedExperts):
            for w in (m.w1, m.w3, m.w2):
                nn.init.normal_(w, mean=0.0, std=std)

    # -- bookkeeping ------------------------------------------------------- #
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def num_active_params(self) -> int:
        return self.cfg.n_active_params

    def enable_grad_checkpointing(self, enabled: bool = True):
        self.grad_checkpointing = enabled

    def update_router_bias(self) -> None:
        """Apply one load-balancing bias update per MoE layer.

        Call this after `optimizer.step()`. It is separate from forward so that
        routing is identical across gradient accumulation micro-steps and
        across gradient-checkpoint recomputation.
        """
        for block in self.blocks:
            if block.is_moe:
                block.ffn.apply_bias_update()

    def router_drop_rate(self) -> float:
        """Fraction of (token, expert) pairs that exceeded expert capacity on
        the last forward pass. Should sit near zero once load balancing has
        settled; a persistently high value means `capacity_factor` is too low."""
        rates = [b.ffn._last_drop_rate for b in self.blocks
                 if b.is_moe and b.ffn._last_drop_rate is not None]
        return float(torch.stack(rates).mean()) if rates else 0.0

    def expert_utilisation(self) -> torch.Tensor:
        """Per-layer expert load, (n_moe_layers, n_experts). A healthy run keeps
        every row close to uniform; a row collapsing onto a few columns means
        the router has stopped using most of the model."""
        rows = [b.ffn.expert_load for b in self.blocks if b.is_moe]
        return torch.stack(rows) if rows else torch.empty(0)

    # -- rope -------------------------------------------------------------- #
    def _rope_cache(self, seq_len: int, device):
        if self._rope is None or self._rope_len < seq_len \
                or self._rope[0].device != device:
            need = max(seq_len, self.cfg.max_seq_len)
            self._rope = build_rope_cache(
                need, self.cfg.head_dim, self.cfg.rope_theta, device,
                self.cfg.rope_scaling_factor,
            )
            self._rope_len = need
        return self._rope

    # -- forward ----------------------------------------------------------- #
    def forward(self, idx, targets=None, cache: Optional[KVCache] = None,
                offset: int = 0, loss_reduction: str = "mean",
                return_logits: bool = False):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope_cache(offset + T, x.device)

        aux_total = x.new_zeros(())
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x, aux = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, cache, offset, use_reentrant=False)
            else:
                x, aux = block(x, cos, sin, cache, offset)
            aux_total = aux_total + aux

        if cache is not None:
            cache.advance(T)

        x = self.norm(x)

        if targets is None:
            # Only the last position matters when decoding — skip the rest of
            # the vocab projection, which is the single biggest tensor here.
            return self.lm_head(x[:, -1:, :]), None

        # Supervised chat data masks out every user token, so typically only a
        # third of positions carry loss. Selecting them *before* the vocab
        # projection makes the largest tensor in the graph — and the fp32
        # softmax over it — three times smaller, at no cost in correctness.
        flat_x = x.reshape(-1, x.size(-1))
        flat_y = targets.reshape(-1)
        keep = flat_y != -100
        sel_x = flat_x[keep]
        sel_y = flat_y[keep]

        if sel_y.numel() == 0:
            zero = x.sum() * 0.0
            return (None, (zero, zero.detach(), zero.detach()))

        logits = self.lm_head(sel_x)
        if self.cfg.logit_softcap > 0:
            cap = self.cfg.logit_softcap
            logits = torch.tanh(logits / cap) * cap

        ce = F.cross_entropy(logits.float(), sel_y, reduction=loss_reduction)
        loss = ce + aux_total

        out_logits = self.lm_head(x) if return_logits else logits
        return out_logits, (loss, ce.detach(), aux_total.detach())

    # -- generation -------------------------------------------------------- #
    def _max_cache_len(self) -> int:
        """Longest context we will serve, after any RoPE scaling."""
        return int(self.cfg.max_seq_len * max(1.0, self.cfg.rope_scaling_factor))

    def generate(self, idx, **kw):
        """Sample a continuation. See `generate_stream` for the arguments."""
        return generate_stream(self, idx, **kw)


def _forward_hidden(model: "Solis", idx, cache, offset):
    """Run the stack and return the final normed hidden states."""
    x = model.tok_emb(idx)
    cos, sin = model._rope_cache(offset + idx.shape[1], x.device)
    for block in model.blocks:
        x, _aux = block(x, cos, sin, cache, offset)
    cache.advance(idx.shape[1])
    return model.norm(x)


def sample_token(logits, history, temperature=0.8, top_k=50, top_p=0.95,
                 min_p=0.0, repetition_penalty=1.0, generator=None):
    """Sample one token id from `logits` (shape (1, vocab))."""
    logits = logits.float()

    if repetition_penalty and repetition_penalty != 1.0 and history.numel() > 0:
        seen = torch.unique(history)
        vals = logits[0, seen]
        # Divide positive logits, multiply negative ones — the standard
        # formulation, which penalises regardless of sign.
        logits[0, seen] = torch.where(vals > 0, vals / repetition_penalty,
                                      vals * repetition_penalty)

    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k:
        k = min(int(top_k), logits.size(-1))
        kth = logits.topk(k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = F.softmax(logits, dim=-1)

    if min_p and min_p > 0:
        # Keep tokens at least `min_p` as likely as the mode. Scales with the
        # model's confidence, unlike a fixed top-p.
        thresh = probs.max(dim=-1, keepdim=True).values * min_p
        probs = torch.where(probs < thresh, torch.zeros_like(probs), probs)
        probs = probs / probs.sum(-1, keepdim=True).clamp(min=1e-9)

    if top_p and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = sorted_probs.cumsum(dim=-1)
        drop = cum - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True).clamp(min=1e-9)
        choice = torch.multinomial(sorted_probs, 1, generator=generator)
        return sorted_idx.gather(-1, choice)

    return torch.multinomial(probs, 1, generator=generator)


@torch.inference_mode()
def generate_stream(model: "Solis", idx: torch.Tensor, max_new_tokens: int = 256,
                    temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95,
                    min_p: float = 0.0, repetition_penalty: float = 1.05,
                    eos_id: Optional[int] = None, stop_ids: tuple = (),
                    stream_cb=None, seed: Optional[int] = None):
    """Prefill + incremental decode with a KV cache.

    Yields nothing; calls `stream_cb(token_id)` per token and returns the full
    id tensor. Kept as a module-level function so the server can call it
    without holding a method reference to a compiled module.
    """
    model.eval()
    device = idx.device
    B, T = idx.shape
    if B != 1:
        raise ValueError("generate_stream handles one sequence at a time")

    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    cache_len = min(T + max_new_tokens, model._max_cache_len())
    if T >= cache_len:
        raise ValueError(
            f"prompt of {T} tokens leaves no room in a {cache_len}-token context"
        )
    cache = KVCache(model.cfg, B, cache_len, device, model.tok_emb.weight.dtype)

    # Prefill.
    h = _forward_hidden(model, idx, cache, offset=0)
    logits = model.lm_head(h[:, -1, :])

    produced = []
    for _ in range(max_new_tokens):
        next_id = sample_token(
            logits, idx, temperature=temperature, top_k=top_k, top_p=top_p,
            min_p=min_p, repetition_penalty=repetition_penalty, generator=gen,
        )
        tok = int(next_id.item())
        idx = torch.cat((idx, next_id), dim=1)
        produced.append(tok)
        if stream_cb is not None:
            stream_cb(tok)
        if (eos_id is not None and tok == eos_id) or tok in stop_ids:
            break
        if cache.pos >= cache_len:
            break
        h = _forward_hidden(model, next_id, cache, offset=cache.pos)
        logits = model.lm_head(h[:, -1, :])

    return idx
