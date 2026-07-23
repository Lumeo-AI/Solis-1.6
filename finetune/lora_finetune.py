"""QLoRA fine-tuning — turn a Qwen2.5 base into a Solis fine-tune.

This is what makes Solis 1.9 "very fine-tuned" rather than a plain rebrand: it
trains low-rank adapters on top of a frozen 4-bit base, so a 7B model fine-tunes
on a 16 GB card and larger models fine-tune in the cloud, at a few percent of
full-fine-tune cost. The result is a small adapter (tens of MB) the server loads
with `SOLIS_ADAPTER=...`.

Data can be any Hugging Face dataset or local JSONL; rows are normalised with the
same `data/ingest.py` logic (messages / conversations / prompt-response /
question-answer / raw text) and rendered with the base model's chat template.

    # bake Solis identity + a data mix into a 7B adapter, on a 16 GB card
    python finetune/lora_finetune.py \
        --model solis-1.9-small \
        --hf teknium/OpenHermes-2.5 --max-samples 20000 \
        --output checkpoints/solis-small-lora

    # then serve it:
    SOLIS_MODEL=solis-1.9-small SOLIS_ADAPTER=checkpoints/solis-small-lora python serve_solis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solis.registry import get_variant  # noqa: E402
from solis.identity import DEFAULT_SYSTEM_PROMPT  # noqa: E402
import data.ingest as ingest  # noqa: E402


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_rows(args) -> list[list[dict]]:
    """Collect normalised conversations from the requested sources."""
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


def build_dataset(rows: list[list[dict]], tokenizer, seq_len: int,
                  inject_identity: bool):
    """Render each conversation to a single training string with the chat
    template. Optionally prepend the Solis system prompt so identity is trained
    into the adapter, not just prompted at serving time."""
    from datasets import Dataset

    texts = []
    for msgs in rows:
        if inject_identity and (not msgs or msgs[0].get("role") != "system"):
            msgs = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + msgs
        try:
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False))
        except Exception:
            continue
    print(f"  rendered {len(texts):,} training sequences")
    return Dataset.from_dict({"text": texts})


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_base(base_repo: str, use_4bit: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {"dtype": torch.bfloat16, "device_map": "auto"}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base_repo, **kwargs)
    return model, tok


def attach_lora(model, use_4bit: bool, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM",
        # The projections that matter for a Qwen-style transformer.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA r={r} alpha={alpha} | trainable {trainable/1e6:.1f}M "
          f"({100*trainable/total:.2f}% of {total/1e9:.2f}B)")
    return model


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="solis-1.9-small",
                    help="Solis variant name or a base HF repo")
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
    ap.add_argument("--no-identity", dest="inject_identity",
                    action="store_false", default=True,
                    help="don't prepend the Solis system prompt to training data")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--packing", action="store_true",
                    help="pack multiple samples per sequence (faster, but needs "
                         "flash-attention to avoid cross-contamination — only "
                         "enable if flash_attention_2 is installed)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Resolve a Solis variant name to its base repo (or accept a raw repo).
    try:
        base_repo = get_variant(args.model).base_repo
    except KeyError:
        base_repo = args.model
    print(f"fine-tuning base: {base_repo}  (4-bit QLoRA={args.use_4bit})")

    print("loading data:")
    rows = load_rows(args)
    print("loading base model:")
    model, tok = load_base(base_repo, args.use_4bit)
    model = attach_lora(model, args.use_4bit, args.lora_r, args.lora_alpha,
                        args.lora_dropout)

    print("preparing dataset:")
    dataset = build_dataset(rows, tok, args.seq_len, args.inject_identity)

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
        # Packing needs flash-attention to keep samples from bleeding into each
        # other; off by default so training is correct everywhere.
        packing=args.packing,
        dataset_text_field="text",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(model=model, args=sft, train_dataset=dataset,
                         processing_class=tok)

    print("training:")
    trainer.train()

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tok.save_pretrained(str(args.output))
    print(f"\nSolis adapter saved -> {args.output}")
    print("serve it with:")
    print(f"  SOLIS_MODEL={args.model} SOLIS_ADAPTER={args.output} "
          "python serve_solis.py")


if __name__ == "__main__":
    main()
