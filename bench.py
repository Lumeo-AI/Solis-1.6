"""Measure Solis's real memory and speed on this machine.

The 16 GB budget in `solis/config.py` is an analytic model. This script checks
it against reality: it builds each preset, runs a prefill and a decode, and
reports what the allocator actually held. Where the two disagree, reality wins
and the model in config.py should be corrected.

Presets too large to instantiate on the current card are reported as skipped
rather than silently omitted.

Run:
    python bench.py                       # every preset that fits
    python bench.py --preset mini --seq-len 2048
    python bench.py --json results/bench.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from solis.config import PRESETS, SolisConfig, get_config
from solis.model import Solis, KVCache, _forward_hidden, generate_stream

ROOT = Path(__file__).resolve().parent
GB = 1024 ** 3


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def bench_config(cfg: SolisConfig, device: str, dtype: torch.dtype,
                 prompt_len: int, decode_tokens: int, warmup: int = 3) -> dict:
    reset_memory()
    free_before, total = (torch.cuda.mem_get_info() if device == "cuda"
                          else (0, 0))

    try:
        model = Solis(cfg).to(device=device, dtype=dtype).eval()
    except torch.OutOfMemoryError:
        reset_memory()
        return {"name": cfg.name, "skipped": "out of memory building the model"}

    weights_gb = (torch.cuda.memory_allocated() / GB if device == "cuda"
                  else cfg.weight_bytes() / GB)

    idx = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)
    result: dict = {
        "name": cfg.name,
        "params_total": cfg.n_params,
        "params_active": cfg.n_active_params,
        "dtype": str(dtype).replace("torch.", ""),
        "prompt_len": prompt_len,
        "decode_tokens": decode_tokens,
        "weights_gb": round(weights_gb, 3),
    }

    try:
        with torch.inference_mode():
            # ---- prefill ------------------------------------------------- #
            for _ in range(warmup):
                cache = KVCache(cfg, 1, prompt_len + decode_tokens + 8,
                                device, dtype)
                _forward_hidden(model, idx, cache, offset=0)
            if device == "cuda":
                torch.cuda.synchronize()
            reset_memory()

            t0 = time.perf_counter()
            cache = KVCache(cfg, 1, prompt_len + decode_tokens + 8, device, dtype)
            _forward_hidden(model, idx, cache, offset=0)
            if device == "cuda":
                torch.cuda.synchronize()
            prefill_s = time.perf_counter() - t0
            prefill_peak = (torch.cuda.max_memory_allocated() / GB
                            if device == "cuda" else 0.0)

            # ---- decode --------------------------------------------------- #
            reset_memory()
            t0 = time.perf_counter()
            out = generate_stream(model, idx, max_new_tokens=decode_tokens,
                                  temperature=0.0, top_k=0, top_p=1.0,
                                  repetition_penalty=1.0)
            if device == "cuda":
                torch.cuda.synchronize()
            decode_s = time.perf_counter() - t0
            n_decoded = out.shape[1] - prompt_len
            decode_peak = (torch.cuda.max_memory_allocated() / GB
                           if device == "cuda" else 0.0)

        result.update({
            "prefill_tokens_per_sec": round(prompt_len / max(prefill_s, 1e-9), 1),
            "prefill_seconds": round(prefill_s, 4),
            "decode_tokens_per_sec": round(n_decoded / max(decode_s, 1e-9), 2),
            "decode_seconds": round(decode_s, 3),
            "kv_cache_gb": round(cache.nbytes() / GB, 3),
            "peak_prefill_gb": round(prefill_peak, 3),
            "peak_decode_gb": round(decode_peak, 3),
            "measured_peak_gb": round(max(prefill_peak, decode_peak), 3),
        })

        predicted = cfg.vram_report(seq_len=prompt_len + decode_tokens)
        result["predicted_peak_gb"] = predicted["total_gb"]
        # config.py adds a fixed allowance for the CUDA context and allocator
        # overhead, which memory_allocated() does not see; compare like for like.
        result["predicted_peak_gb_excl_overhead"] = round(
            predicted["total_gb"] - predicted["runtime_overhead_gb"], 3)
        result["fits_16gb"] = predicted["total_gb"] <= 16.0

    except torch.OutOfMemoryError:
        result["skipped"] = "out of memory during the benchmark"

    del model
    reset_memory()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=None,
                    help="one preset; default is every preset")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--json", type=Path, default=ROOT / "results" / "bench.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        gpu = props.name
        vram = props.total_memory / GB
        print(f"device: {gpu}  ({vram:.1f} GB VRAM)")
        free, _ = torch.cuda.mem_get_info()
        print(f"free before benchmark: {free / GB:.1f} GB "
              f"(other processes hold the rest)")
    else:
        gpu, vram = "cpu", 0.0
        print("device: cpu — memory numbers will not be meaningful")

    names = [args.preset] if args.preset else list(PRESETS)
    rows = []
    for name in names:
        cfg = get_config(name)
        print(f"\n--- {cfg.name} "
              f"({cfg.n_params / 1e9:.2f}B total / "
              f"{cfg.n_active_params / 1e9:.2f}B active) ---")
        row = bench_config(cfg, device, dtype, args.prompt_len,
                           args.decode_tokens)
        row["preset"] = name
        rows.append(row)
        if "skipped" in row:
            print(f"  skipped: {row['skipped']}")
            continue
        print(f"  weights           {row['weights_gb']:6.2f} GB")
        print(f"  kv cache          {row['kv_cache_gb']:6.2f} GB")
        print(f"  measured peak     {row['measured_peak_gb']:6.2f} GB")
        print(f"  predicted peak    {row['predicted_peak_gb']:6.2f} GB "
              f"(config.py, incl. {0.9} GB runtime allowance)")
        print(f"  prefill           {row['prefill_tokens_per_sec']:8,.0f} tok/s")
        print(f"  decode            {row['decode_tokens_per_sec']:8,.1f} tok/s")

    print(f"\n{'=' * 76}")
    print(f"{'preset':<10} {'total':>8} {'active':>8} {'peak GB':>9} "
          f"{'prefill':>11} {'decode':>10}  {'16GB':>5}")
    print("-" * 76)
    for r in rows:
        if "skipped" in r:
            print(f"{r['preset']:<10} {'—':>8} {'—':>8} {'—':>9} "
                  f"{'—':>11} {'—':>10}  {'—':>5}   {r['skipped']}")
            continue
        print(f"{r['preset']:<10} {r['params_total'] / 1e9:7.2f}B "
              f"{r['params_active'] / 1e9:7.2f}B "
              f"{r['measured_peak_gb']:8.2f} "
              f"{r['prefill_tokens_per_sec']:10,.0f} "
              f"{r['decode_tokens_per_sec']:9.1f}  "
              f"{'yes' if r['fits_16gb'] else 'NO':>5}")
    print("=" * 76)

    payload = {
        "gpu": gpu,
        "vram_total_gb": round(vram, 2),
        "dtype": args.dtype,
        "prompt_len": args.prompt_len,
        "decode_tokens": args.decode_tokens,
        "torch": torch.__version__,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
