"""Measure what Solis can actually do.

Two kinds of number come out of here, and they answer different questions:

  * **Validation loss / perplexity** — how well the model predicts held-out
    text. Comparable across checkpoints of the same tokenizer, and meaningless
    across different ones (a model with a coarser vocabulary has fewer, harder
    predictions to make, so its perplexity is not on the same scale).

  * **Task accuracy** — the model generates an answer and it is checked against
    ground truth, per task category. This is the honest measure: it is exact
    match on freshly generated problems the model has never seen, using a seed
    disjoint from the training corpus.

Every number this prints is measured here, now, on this machine. Nothing is
copied from a paper or another model's reported results.

Run:
    python eval.py
    python eval.py --checkpoint checkpoints/solis-mini-best.pt --n-per-task 100
    python eval.py --json results/eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solis.config import SolisConfig
from solis.model import Solis, generate_stream
from solis.tokenizer import SolisTokenizer, EOS

import data.build_corpus as corpus

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Task suites, generated fresh with a held-out seed
# --------------------------------------------------------------------------- #
TASK_SUITES = {
    "passage_qa": corpus.t_passage_qa,
    "passage_multi_qa": corpus.t_passage_multi_qa,
    "summarize": corpus.t_summarize,
    "extract_json": corpus.t_extract_json,
    "text_transform": corpus.t_text_transform,
    "list_format": corpus.t_list_format,
    "arithmetic": corpus.t_arithmetic,
    "word_problem": corpus.t_word_problem,
    "sequence": corpus.t_sequence,
    "code": corpus.t_code,
    "logic": corpus.t_logic,
    "calendar": corpus.t_calendar,
    "units": corpus.t_units,
    "table": corpus.t_table,
    "rewrite": corpus.t_rewrite,
    "spelling": corpus.t_spelling,
    "grammar": corpus.t_grammar,
    "definition": corpus.t_definition,
    "identity": corpus.t_identity,
    "calibration": corpus.t_calibration,
}

# Task types where any of several phrasings is correct, so exact string match
# would understate the model. These are scored on content words instead.
FUZZY_TASKS = {"identity", "calibration", "definition", "capability",
               "smalltalk", "summarize"}


def normalise(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def score(task: str, predicted: str, expected: str) -> float:
    p, e = normalise(predicted), normalise(expected)
    if task in FUZZY_TASKS:
        # Token overlap (F1), which is the standard way to score answers whose
        # exact wording is not the thing being tested.
        pt, et = set(p.split()), set(e.split())
        if not pt or not et:
            return float(p == e)
        overlap = len(pt & et)
        if overlap == 0:
            return 0.0
        prec, rec = overlap / len(pt), overlap / len(et)
        return 2 * prec * rec / (prec + rec)
    return float(p == e)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_model(path: Path, device: str, dtype: torch.dtype):
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path} — train first")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SolisConfig.from_dict(blob["config"])
    model = Solis(cfg)
    model.load_state_dict(blob["model"])
    model = model.to(device=device, dtype=dtype).eval()
    return model, cfg, blob.get("meta", {})


# --------------------------------------------------------------------------- #
# Validation loss
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validation_loss(model, cfg, packed_dir: Path, device, batch_size=4,
                    n_batches=50) -> dict:
    sys.path.insert(0, str(ROOT))
    from train import PackedDataset

    meta_path = packed_dir / "meta.json"
    if not meta_path.exists():
        return {"available": False}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "val" not in meta["splits"]:
        return {"available": False}

    data = PackedDataset(packed_dir, "val", cfg.max_seq_len)
    total, n_tok = 0.0, 0
    for x, y in data.sequential_batches(batch_size, n_batches, device):
        n = int((y != -100).sum())
        if n == 0:
            continue
        _, out = model(x, targets=y, loss_reduction="sum")
        total += float(out[1])
        n_tok += n
    if n_tok == 0:
        return {"available": False}
    loss = total / n_tok
    return {"available": True, "loss": round(loss, 4),
            "perplexity": round(math.exp(min(loss, 20)), 3),
            "tokens_scored": n_tok}


# --------------------------------------------------------------------------- #
# Task accuracy
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_task_suite(model, tok, device, n_per_task: int, seed: int,
                   max_new_tokens: int, verbose: bool) -> dict:
    results = {}
    examples_shown = []
    for name, fn in TASK_SUITES.items():
        # A seed far from the corpus seed (7) so these problems are new.
        rng = random.Random(seed + hash(name) % 10_000)
        total, n = 0.0, 0
        t0 = time.time()
        for i in range(n_per_task):
            try:
                q, expected = fn(rng)
            except Exception:
                continue
            ids = tok.encode_chat([{"role": "user", "content": q}])
            if len(ids) > model._max_cache_len() - max_new_tokens - 8:
                continue
            out = generate_stream(
                model, torch.tensor([ids], device=device),
                max_new_tokens=max_new_tokens, temperature=0.0,
                top_k=0, top_p=1.0, repetition_penalty=1.0, eos_id=EOS,
            )
            predicted = tok.decode(out[0, len(ids):].tolist())
            s = score(name, predicted, expected)
            total += s
            n += 1
            if verbose and i == 0:
                examples_shown.append({
                    "task": name, "prompt": q, "expected": expected,
                    "predicted": predicted, "score": round(s, 3)})
        acc = total / max(n, 1)
        results[name] = {"score": round(acc, 4), "n": n,
                         "seconds": round(time.time() - t0, 1)}
        print(f"  {name:<18} {acc * 100:5.1f}%   (n={n})")
    return results, examples_shown


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "checkpoints" / "solis-mini-best.pt")
    ap.add_argument("--tokenizer", type=Path,
                    default=ROOT / "checkpoints" / "tokenizer.json")
    ap.add_argument("--packed", type=Path, default=ROOT / "data" / "packed")
    ap.add_argument("--n-per-task", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=999_983,
                    help="held out from the corpus seed")
    ap.add_argument("--json", type=Path, default=ROOT / "results" / "eval.json")
    ap.add_argument("--no-examples", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model, cfg, meta = load_model(args.checkpoint, device, dtype)
    tok = SolisTokenizer.load(args.tokenizer)

    print(f"\n{'=' * 68}")
    print(f"Solis evaluation — {cfg.name}")
    print(f"{'=' * 68}")
    print(f"  checkpoint     {args.checkpoint.name}")
    print(f"  trained steps  {meta.get('step', '?')}")
    print(f"  params         {model.num_params():,} total / "
          f"{cfg.n_active_params:,} active")
    print(f"  device         {device} / {dtype}")
    print()

    print("validation loss:")
    vl = validation_loss(model, cfg, args.packed, device)
    if vl["available"]:
        print(f"  loss {vl['loss']}  perplexity {vl['perplexity']}  "
              f"({vl['tokens_scored']:,} tokens)")
    else:
        print("  (no packed validation split found)")

    print(f"\ntask accuracy (greedy, {args.n_per_task} fresh problems per task, "
          f"seed {args.seed}):")
    tasks, examples = run_task_suite(model, tok, device, args.n_per_task,
                                     args.seed, args.max_new_tokens,
                                     not args.no_examples)

    exact = [k for k in tasks if k not in FUZZY_TASKS]
    fuzzy = [k for k in tasks if k in FUZZY_TASKS]
    macro = sum(tasks[k]["score"] for k in tasks) / max(len(tasks), 1)
    macro_exact = sum(tasks[k]["score"] for k in exact) / max(len(exact), 1)
    macro_fuzzy = sum(tasks[k]["score"] for k in fuzzy) / max(len(fuzzy), 1)

    print(f"\n{'-' * 68}")
    print(f"  macro average (all tasks)        {macro * 100:5.1f}%")
    print(f"  exact-match tasks ({len(exact):>2})           {macro_exact * 100:5.1f}%")
    print(f"  open-ended tasks  ({len(fuzzy):>2}, token F1) {macro_fuzzy * 100:5.1f}%")
    print(f"{'-' * 68}")

    if examples and not args.no_examples:
        print("\nsample generations:")
        for ex in examples[:8]:
            prompt = ex["prompt"].replace("\n", " ")[:88]
            print(f"\n  [{ex['task']}] {prompt}")
            print(f"    expected:  {ex['expected'][:88]}")
            print(f"    predicted: {ex['predicted'][:88]}")

    payload = {
        "model": cfg.name,
        "checkpoint": str(args.checkpoint.name),
        "trained_steps": meta.get("step"),
        "params_total": model.num_params(),
        "params_active": cfg.n_active_params,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "dtype": str(dtype).replace("torch.", ""),
        "eval_seed": args.seed,
        "n_per_task": args.n_per_task,
        "validation": vl,
        "tasks": tasks,
        "macro_average": round(macro, 4),
        "macro_exact_match": round(macro_exact, 4),
        "macro_open_ended_f1": round(macro_fuzzy, 4),
        "examples": examples,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
