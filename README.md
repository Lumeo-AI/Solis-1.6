# Solis ☀️

A **sparse Mixture-of-Experts** language model built entirely from scratch — no
base model, no pretrained weights, no borrowed vocabulary. Solis learns its own
tokenizer, its own experts, and its own router, starting from random numbers.

The 1.0 family is designed against one hard constraint: **every preset must
serve inside 16 GB of VRAM**, weights, KV cache, and activation workspace
included, while keeping *active* parameters low so decoding stays fast.

```
preset      total   active    ctx   weights     kv    act    peak   fits 16GB
------------------------------------------------------------------------------
mini         335M     123M   2048     0.62G  0.03G  0.07G   1.62G        yes
small       1.79B     489M   4096     3.33G  0.09G  0.22G   4.55G        yes
base        4.48B     974M   8192     8.35G  0.22G  0.63G  10.09G        yes
flagship    6.33B    1.40B   8192    11.80G  0.47G  0.72G  13.88G        yes
```

`python -m solis.config` prints this table; `python bench.py` measures it for
real and tells you where the estimate is wrong.

## Architecture

Decoder-only transformer (`solis/model.py`):

- **Byte-level BPE tokenizer** (`solis/tokenizer.py`) trained on our own corpus.
  256 byte tokens are always present as a fallback, so nothing is ever
  unencodable, and round-tripping is exact for arbitrary bytes.
- **RMSNorm** pre-normalisation, computed in fp32 regardless of autocast dtype.
- **Grouped-query attention** — fewer KV heads than query heads, which is what
  makes the KV cache small enough to serve long context in 16 GB.
- **QK-norm** — RMSNorm on queries and keys before the dot product. Removes the
  attention-logit blowup that otherwise makes small models diverge.
- **Sliding-window attention** on most layers, with every 4th layer left global,
  so attention cost grows linearly in context but information still travels the
  full sequence.
- **Rotary position embeddings**, with a scaling factor for serving beyond the
  trained context.
- **Mixture-of-Experts FFN** — one *shared* expert that runs for every token,
  plus a top-k routed set of specialists. The shared path is what keeps a sparse
  model coherent at small scale: general-purpose computation lives there instead
  of being relearned by every expert.
- **Aux-loss-free load balancing** — a per-expert bias steers routing toward
  under-used experts without adding gradient noise, backed by a small classic
  auxiliary loss and a router z-loss.
- The first layers stay **dense**; early layers do broad low-level work that
  every token needs, so routing them wastes capacity.
- Weight-tied embedding / LM head, and residual branches scaled by `1/sqrt(2L)`
  at init so the residual stream does not grow with depth.

### Two implementation details that are easy to get wrong

**MoE dispatch runs as three batched matmuls**, not a Python loop over experts.
Tokens are sorted by expert and packed into a fixed `(n_experts, capacity, dim)`
buffer. The naive version — slicing the token stream per expert and looping —
issues `3 x n_experts` small GEMMs per layer *and* needs a host synchronisation
to learn the slice boundaries.

**Capacity limiting is training-only.** During training a fixed capacity keeps
every tensor shape static; overflow pairs are routed to a scratch row whose
output is zeroed, so they contribute nothing. At inference capacity is the
largest actual group, which costs one device-to-host read per layer but
guarantees a token's output never depends on which other tokens shared its
batch. Without that split, a prefill disagrees with the same tokens decoded one
at a time — `tests/test_model.py` checks exactly this.
