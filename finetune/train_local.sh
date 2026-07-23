#!/usr/bin/env bash
# Fine-tune the largest Solis 1.9 that fits your card, on a streamed mix of
# large general-purpose datasets, with the Solis identity baked in.
#
#   bash finetune/train_local.sh              # 16 GB card -> Qwen3-8B
#   TIER=stretch bash finetune/train_local.sh # 24 GB card -> Qwen3-14B
#
# Runs long — launch it detached:
#   nohup bash finetune/train_local.sh > logs/solis_finetune.log 2>&1 &
#   tail -f logs/solis_finetune.log
#
# Why these sizes:
#   Qwen3-8B  ~4.5 GB of 4-bit weights. QLoRA fits 16 GB with headroom, and it
#             decodes fast enough that chat feels instant. This is the default.
#   Qwen3-14B ~8 GB of 4-bit weights. Train on a rented 24 GB GPU; the adapter
#             then SERVES fine on your 16 GB card. Better answers, still quick.
# Anything larger (Qwen3-32B, Qwen3.6-27B, Qwen3.6-35B-A3B) cannot serve on
# 16 GB, so training it would produce a model you could not run at home.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TOKENIZERS_PARALLELISM=false

# Massive, diverse, streamed (no full download) — chat, instructions, reasoning,
# math, code. Capped per source so the run is bounded and starts fast.
DATASETS=(
  --hf Open-Orca/OpenOrca
  --hf teknium/OpenHermes-2.5
  --hf allenai/tulu-3-sft-mixture
  --hf microsoft/orca-math-word-problems-200k
  --hf glaiveai/glaive-code-assistant
)
MAX_PER_SOURCE="${MAX_PER_SOURCE:-15000}"
MAX_STEPS="${MAX_STEPS:-4000}"
SAVE_STEPS="${SAVE_STEPS:-250}"
TIER="${TIER:-local}"

run() {  # model  seq_len  batch  grad_accum  output
  python finetune/lora_finetune.py \
    --model "$1" "${DATASETS[@]}" \
    --max-samples "$MAX_PER_SOURCE" --seq-len "$2" \
    --batch-size "$3" --grad-accum "$4" --max-steps "$MAX_STEPS" \
    --lora-r 16 --lora-alpha 32 --save-steps "$SAVE_STEPS" \
    --output "$5"
}

if [ "$TIER" = "stretch" ]; then
  echo "############################################################"
  echo "# 24 GB tier: Qwen3-14B (solis-1.9-base) via Unsloth QLoRA"
  echo "#   trains on the rented card; serves on your 16 GB card"
  echo "############################################################"
  if run solis-1.9-base 2048 1 16 checkpoints/solis-base-lora; then
    echo ">>> 14B fine-tune complete -> checkpoints/solis-base-lora"
    echo "SERVE_MODEL=solis-1.9-base"
    echo "SERVE_ADAPTER=checkpoints/solis-base-lora"
    exit 0
  fi
  echo ">>> 14B run failed — retrying at seq 1024."
  if run solis-1.9-base 1024 1 16 checkpoints/solis-base-lora; then
    echo ">>> 14B (seq 1024) complete -> checkpoints/solis-base-lora"
    exit 0
  fi
  echo ">>> 14B failed at both sequence lengths — see the log."
  exit 1
fi

echo "############################################################"
echo "# 16 GB tier: Qwen3-8B (solis-1.9) via Unsloth QLoRA"
echo "#   downloads ~6 GB (Unsloth 4-bit mirror), then trains"
echo "############################################################"
if run solis-1.9 2048 2 8 checkpoints/solis-lora; then
  echo ">>> 8B fine-tune complete -> checkpoints/solis-lora"
  echo "SERVE_MODEL=solis-1.9"
  echo "SERVE_ADAPTER=checkpoints/solis-lora"
else
  echo ">>> 8B run failed — most likely VRAM. Freeing the GPU and falling back."
  python - <<'PY' 2>/dev/null || true
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
PY
  if run solis-1.9-mini 2048 2 8 checkpoints/solis-mini-lora; then
    echo ">>> 4B fine-tune complete -> checkpoints/solis-mini-lora"
    echo "SERVE_MODEL=solis-1.9-mini"
    echo "SERVE_ADAPTER=checkpoints/solis-mini-lora"
  else
    echo ">>> 4B run also failed — see the log above."
    exit 1
  fi
fi
echo "DONE. Serve with:  SOLIS_MODEL=<model> SOLIS_ADAPTER=<adapter> python serve.py"
