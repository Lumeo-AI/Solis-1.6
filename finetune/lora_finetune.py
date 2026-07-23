"""QLoRA fine-tuning — turn a Qwen3 base into a Solis fine-tune.

This is what makes Solis 1.9 "very fine-tuned" rather than a plain rebrand: it
trains low-rank adapters on top of a frozen 4-bit base, so an 8B model
fine-tunes comfortably on a 16 GB card and 14B fits a rented 24 GB one, at a few
percent of full-fine-tune cost. The result is a small adapter (tens of MB) the
server loads with `SOLIS_ADAPTER=...`.

Two backends, same CLI:

  * **unsloth** (default when installed) — fused kernels and a custom
    gradient-checkpointing path. Roughly 2x faster and ~50% less VRAM than
    stock PEFT, which is what moves 14B from "won't fit" to "fits". Unsloth
    must be imported before transformers, which `_import_unsloth()` handles.
  * **peft** — the portable path. Works anywhere, needs more memory.

Data can be any Hugging Face dataset or local JSONL; rows are normalised with
the same `data/ingest.py` logic (messages / conversations / prompt-response /
question-answer / raw text) and rendered with the base model's chat template.

    # the default: Qwen3-8B on a 16 GB card, Unsloth, OpenOrca + friends
    python finetune/lora_finetune.py \
        --model solis-1.9 \
        --hf Open-Orca/OpenOrca --hf teknium/OpenHermes-2.5 \
        --max-samples 20000 \
        --output checkpoints/solis-lora

    # then serve it:
    SOLIS_MODEL=solis-1.9 SOLIS_ADAPTER=checkpoints/solis-lora python serve.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _import_unsloth():
    """Import unsloth first so its patches land before transformers loads.

    Unsloth rewrites parts of transformers/peft at import time; importing it
    after them silently loses most of the speed and memory win, so this is done
    before anything else touches torch.
    """
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    return FastLanguageModel


def _unsloth_available() -> bool:
    return importlib.util.find_spec("unsloth") is not None


# Unsloth publishes pre-quantised 4-bit mirrors that download ~4x smaller and
# skip on-the-fly quantisation. Fall back to the stock repo when there is no
# mirror for a model.
UNSLOTH_MIRRORS = {
    "Qwen/Qwen3-1.7B": "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    "Qwen/Qwen3-4B": "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "Qwen/Qwen3-8B": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "Qwen/Qwen3-14B": "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "Qwen/Qwen3-32B": "unsloth/Qwen3-32B-unsloth-bnb-4bit",
}

# The projections that matter for a Qwen-style transformer.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_rows(args) -> list[list[dict]]:
    """Collect normalised conversations from the requested sources."""
    import data.ingest as ingest

    rows: list[list[dict]] = []

    def take(iterator, cap):
        n = 0
        for msgs in iterator:
            if cap and n >= cap:
                break
            rows.append(msgs)
            n += 1
        return n

    for name in args.hf:
        from datasets import load_dataset
        ds = load_dataset(name, split=args.split, streaming=True)
        n = take((m for m in (ingest.normalise_row(r) for r in ds) if m),
                 args.max_samples)
        print(f"  {name}: {n:,}")
    for p in args.jsonl:
        n = take(ingest.iter_jsonl(Path(p)), args.max_samples)
        print(f"  {p}: {n:,}")

    if not rows:
        raise SystemExit("no data — pass --hf and/or --jsonl")
    return rows


def build_dataset(rows: list[list[dict]], tokenizer, inject_identity: bool):
    """Render each conversation to a single training string with the chat
    template. Optionally prepend the Solis system prompt so identity is trained
    into the adapter, not just prompted at serving time.

    Qwen3's template takes `enable_thinking`; we render with it off so the
    adapter learns direct answers (the mode day-to-day chat actually uses).
    """
    from datasets import Dataset
    from solis.identity import DEFAULT_SYSTEM_PROMPT

    texts = []
    for msgs in rows:
        if inject_identity and (not msgs or msgs[0].get("role") != "system"):
            msgs = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + msgs
        try:
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                enable_thinking=False))
        except TypeError:
            try:
                texts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False))
            except Exception:
                continue
        except Exception:
            continue
    if not texts:
        raise SystemExit("every conversation failed to render — check the data")
    print(f"  rendered {len(texts):,} training sequences")
    return Dataset.from_dict({"text": texts})


# --------------------------------------------------------------------------- #
# Model — Unsloth path
# --------------------------------------------------------------------------- #
def load_unsloth(base_repo: str, args):
    FastLanguageModel = _import_unsloth()

    repo = base_repo
    if args.use_4bit and not args.no_mirror:
        repo = UNSLOTH_MIRRORS.get(base_repo, base_repo)
        if repo != base_repo:
            print(f"  using Unsloth 4-bit mirror: {repo}")

    model, tok = FastLanguageModel.from_pretrained(
        model_name=repo,
        max_seq_length=args.seq_len,
        dtype=None,                 # auto-detect (bf16 on modern cards)
        load_in_4bit=args.use_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=LORA_TARGETS,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,   # 0 is the optimised path
        bias="none",
        # Unsloth's own checkpointing: the single biggest VRAM saving here, and
        # what lets 14B train on 24 GB (and 8B leave headroom on 16 GB).
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=args.rslora,
    )
    return model, tok


# --------------------------------------------------------------------------- #
# Model — portable PEFT path
# --------------------------------------------------------------------------- #
def load_peft(base_repo: str, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {"dtype": torch.bfloat16, "device_map": "auto"}
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base_repo, **kwargs)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if args.use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
        use_rslora=args.rslora))
    return model, tok


def report_trainable(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {trainable/1e6:.1f}M "
          f"({100*trainable/max(total,1):.2f}% of {total/1e9:.2f}B)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="solis-1.9",
                    help="Solis variant name or a base HF repo "
                         "(default solis-1.9 = Qwen3-8B)")
    ap.add_argument("--backend", choices=("auto", "unsloth", "peft"),
                    default="auto",
                    help="training backend; auto prefers unsloth when installed")
    ap.add_argument("--hf", action="append", default=[],
                    help="Hugging Face dataset id (repeatable)")
    ap.add_argument("--jsonl", action="append", default=[],
                    help="local JSONL in messages format (repeatable)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-samples", type=int, default=20000,
                    help="cap conversations per source")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--no-4bit", dest="use_4bit", action="store_false",
                    default=True, help="full-precision base (needs a big GPU)")
    ap.add_argument("--no-mirror", action="store_true",
                    help="don't substitute Unsloth's pre-quantised repos")
    ap.add_argument("--no-identity", dest="inject_identity",
                    action="store_false", default=True,
                    help="don't prepend the Solis system prompt to training data")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0,
                    help="0 keeps Unsloth's fused path (recommended)")
    ap.add_argument("--rslora", action="store_true",
                    help="rank-stabilised LoRA (helps at higher r)")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--resume", action="store_true",
                    help="resume from the last checkpoint in --output")
    ap.add_argument("--packing", action="store_true",
                    help="pack multiple samples per sequence (faster; needs "
                         "flash-attention to avoid cross-contamination)")
    ap.add_argument("--merge", action="store_true",
                    help="also save merged 16-bit weights next to the adapter")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Resolve a Solis variant name to its base repo (or accept a raw repo).
    from solis.registry import get_variant
    try:
        variant = get_variant(args.model)
        base_repo = variant.base_repo
        if variant.modality.value == "text":
            need = variant.training_vram_gb(args.seq_len)
            print(f"{variant.name}: ~{need:.1f} GB to QLoRA at seq {args.seq_len}"
                  f"  (serve ~{variant.serving_vram_gb():.1f} GB)")
            if not variant.trainable_24gb(args.seq_len):
                print("  WARNING: this needs more than a 24 GB GPU. Pick a "
                      "smaller variant or lower --seq-len.")
    except KeyError:
        base_repo = args.model

    backend = args.backend
    if backend == "auto":
        backend = "unsloth" if _unsloth_available() else "peft"
    if backend == "unsloth" and not _unsloth_available():
        raise SystemExit("--backend unsloth but unsloth is not installed:\n"
                         "  pip install unsloth")
    print(f"fine-tuning base: {base_repo}  "
          f"(backend={backend}, 4-bit={args.use_4bit})")

    print("loading base model:")
    if backend == "unsloth":
        try:
            model, tok = load_unsloth(base_repo, args)
        except ImportError as exc:
            # The classic Windows case: unsloth is installed (so find_spec found
            # it) but its Triton kernels will not import. Auto-selected backends
            # degrade; an explicitly requested one fails loudly instead of
            # quietly training with different memory characteristics.
            if args.backend == "unsloth":
                raise
            print(f"WARNING: unsloth is installed but failed to import: {exc}")
            print("         Falling back to --backend peft. This uses more "
                  "VRAM and may OOM on a 14B at 16 GB.")
            print("         On Windows, `pip install triton-windows` usually "
                  "fixes it.")
            backend = "peft"
            model, tok = load_peft(base_repo, args)
    else:
        model, tok = load_peft(base_repo, args)
    print(f"  backend in use: {backend}")
    report_trainable(model)

    print("loading data:")
    rows = load_rows(args)
    print("preparing dataset:")
    dataset = build_dataset(rows, tok, args.inject_identity)

    import torch
    from trl import SFTTrainer, SFTConfig
    sft = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        max_length=args.seq_len,
        packing=args.packing,
        dataset_text_field="text",
        # Unsloth installs its own checkpointing in get_peft_model; letting the
        # trainer enable it again would double-wrap the model.
        gradient_checkpointing=(backend != "unsloth"),
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(model=model, args=sft, train_dataset=dataset,
                         processing_class=tok)

    print("training:")
    trainer.train(resume_from_checkpoint=args.resume or None)

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tok.save_pretrained(str(args.output))
    print(f"\nSolis adapter saved -> {args.output}")

    if args.merge:
        merged = Path(str(args.output) + "-merged")
        print(f"merging adapter into 16-bit weights -> {merged}")
        if backend == "unsloth":
            model.save_pretrained_merged(str(merged), tok,
                                         save_method="merged_16bit")
        else:
            model.merge_and_unload().save_pretrained(str(merged))
            tok.save_pretrained(str(merged))

    print("serve it with:")
    print(f"  SOLIS_MODEL={args.model} SOLIS_ADAPTER={args.output} python serve.py")


if __name__ == "__main__":
    main()
