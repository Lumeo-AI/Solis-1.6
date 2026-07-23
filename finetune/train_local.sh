#!/usr/bin/env bash
# Fine-tune the largest Solis 1.9 that fits this 16 GB card, on a streamed mix of
# massive general-purpose datasets, with a Solis identity baked in. Bounded so it
# completes (checkpointing lets you --resume for more), and it falls back from
# 14B to 7B automatically if VRAM is too tight.
#
#   bash finetune/train_solis_local.sh
#
# Runs long — launch it detached:
#   nohup bash finetune/train_solis_local.sh > logs/solis_finetune.log 2>&1 &
#   tail -f logs/solis_finetune.log
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TOKENIZERS_PARALLELISM=false

# Massive, diverse, streamed (no full download) — chat, instructions, reasoning,
# math, code. Capped per source so the run is bounded and materialises fast.
DATASETS=(
  --hf teknium/OpenHermes-2.5
  --hf allenai/tulu-3-sft-mixture
  --hf Open-Orca/OpenOrca
  --hf microsoft/orca-math-word-problems-200k
  --hf glaiveai/glaive-code-assistant
)
MAX_PER_SOURCE="${MAX_PER_SOURCE:-15000}"
MAX_STEPS="${MAX_STEPS:-4000}"
SAVE_STEPS="${SAVE_STEPS:-250}"

run() {  # model  seq_len  output
  python finetune/lora_finetune.py \
    --model "$1" "${DATASETS[@]}" \
    --max-samples "$MAX_PER_SOURCE" --seq-len "$2" \
    --batch-size 1 --grad-accum 16 --max-steps "$MAX_STEPS" \
    --lora-r 16 --lora-alpha 32 --save-steps "$SAVE_STEPS" \
    --output "$3"
}

echo "############################################################"
echo "# Attempting LARGEST trainable: Qwen2.5-14B (solis-1.9-base)"
echo "#   downloads ~29 GB on first load, then 4-bit QLoRA"
echo "############################################################"
if run solis-1.9-base 1024 checkpoints/solis-base-lora; then
  echo ">>> 14B fine-tune complete -> checkpoints/solis-base-lora"
  echo "SERVE_MODEL=solis-1.9-base"
  echo "SERVE_ADAPTER=checkpoints/solis-base-lora"
else
  rc=$?
  echo ">>> 14B run failed (exit $rc) — most likely VRAM. Falling back to 7B."
  # Free anything the failed attempt left on the GPU before retrying.
  python - <<'PY' 2>/dev/null || true
import torch, gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
PY
  if run solis-1.9-small 2048 checkpoints/solis-small-lora; then
    echo ">>> 7B fine-tune complete -> checkpoints/solis-small-lora"
    echo "SERVE_MODEL=solis-1.9-small"
    echo "SERVE_ADAPTER=checkpoints/solis-small-lora"
  else
    echo ">>> 7B run also failed — see the log above."
    exit 1
  fi
fi
echo "DONE. Serve with:  SOLIS_MODEL=<model> SOLIS_ADAPTER=<adapter> python serve_solis.py"
