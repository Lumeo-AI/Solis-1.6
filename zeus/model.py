"""Zeus: a decoder-only Mixture-of-Experts transformer, from scratch.

Components implemented here:
  * RMSNorm
  * Rotary positional embeddings (RoPE)
  * Causal multi-head self-attention
  * A Mixture-of-Experts feed-forward layer with top-k routing, SwiGLU experts,
    a load-balancing auxiliary loss and a router z-loss.

Nothing is pretrained — weights are randomly initialised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ZeusConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq, head_dim/2)
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]
    return cos.to(dtype), sin.to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, heads, seq, head_dim)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    seq = x.shape[2]
    cos, sin = cos[:, :, :seq, :], sin[:, :, :seq, :]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.stack((rot1, rot2), dim=-1).flatten(-2)
    return out


class Attention(nn.Module):
    def __init__(self, cfg: ZeusConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class Expert(nn.Module):
    """A single SwiGLU feed-forward expert."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, dim, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """Top-k routed mixture of experts.

    Tokens are routed to `n_experts_per_tok` experts by a learned router. We
    accumulate a load-balancing auxiliary loss (encourages uniform expert usage)
    and a router z-loss (keeps router logits from exploding).
    """

    def __init__(self, cfg: ZeusConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_experts_per_tok
        self.aux_loss_coef = cfg.aux_loss_coef
        self.z_loss_coef = cfg.router_z_loss_coef
        hidden = int(cfg.dim * cfg.expert_hidden_mult)
        self.gate = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(cfg.dim, hidden) for _ in range(cfg.n_experts))
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)            # (N, C), N = B*T
        logits = self.gate(x_flat)           # (N, E)

        # Router z-loss: penalise large logits (log-sum-exp of router outputs).
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()

        probs = F.softmax(logits, dim=-1)    # (N, E)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)  # (N, k)
        topk_probs = topk_probs / (topk_probs.sum(-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(x_flat)
        # Dispatch each token to its selected experts.
        for slot in range(self.top_k):
            idx = topk_idx[:, slot]          # (N,)
            weight = topk_probs[:, slot].unsqueeze(-1)  # (N, 1)
            for e in range(self.n_experts):
                mask = idx == e
                if mask.any():
                    out[mask] += weight[mask] * self.experts[e](x_flat[mask])

        # Load-balancing loss (Switch/GShard style): fraction of tokens routed to
        # each expert times mean router probability for that expert.
        with torch.no_grad():
            one_hot = F.one_hot(topk_idx, self.n_experts).float().sum(1)  # (N, E)
            tokens_per_expert = one_hot.mean(0)  # f_i
        prob_per_expert = probs.mean(0)          # P_i
        aux_loss = self.n_experts * (tokens_per_expert * prob_per_expert).sum()
        self.last_aux_loss = self.aux_loss_coef * aux_loss + self.z_loss_coef * z_loss

        return out.reshape(B, T, C)


class Block(nn.Module):
    def __init__(self, cfg: ZeusConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.moe_norm = RMSNorm(cfg.dim)
        self.moe = MoE(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.moe(self.moe_norm(x))
        return x


class Zeus(nn.Module):
    def __init__(self, cfg: ZeusConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        # Weight tying.
        self.lm_head.weight = self.tok_emb.weight

        self._rope = None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _rope_cache(self, seq_len, device, dtype):
        if self._rope is None or self._rope[0].shape[2] < seq_len or self._rope[0].device != device:
            self._rope = build_rope_cache(
                max(seq_len, self.cfg.max_seq_len), self.cfg.head_dim,
                self.cfg.rope_theta, device, dtype,
            )
        return self._rope

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos, sin = self._rope_cache(T, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100,
            )
            aux = sum(b.moe.last_aux_loss for b in self.blocks)
            loss = ce + aux
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9, top_k=40, top_p=0.95,
                 eos_id=None, stream_cb=None):
        """Autoregressive sampling. Optionally stream ids via stream_cb(id)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)

            if top_p is not None and top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_probs[remove] = 0.0
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                next_sorted = torch.multinomial(sorted_probs, 1)
                next_id = sorted_idx.gather(-1, next_sorted)
            else:
                next_id = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_id), dim=1)
            tok = next_id.item()
            if stream_cb is not None:
                stream_cb(tok)
            if eos_id is not None and tok == eos_id:
                break
        return idx
