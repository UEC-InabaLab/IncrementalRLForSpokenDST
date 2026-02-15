#!/usr/bin/env bash
# =============================================================================
# Incremental DST - Inference with ms-swift
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}"
CHECKPOINT="${CHECKPOINT:-}"  # e.g., output/sft_incremental_dst/checkpoint-xxx
VAL_DATA="${VAL_DATA:-data/dapo_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/inference_results}"

# ---------------------------------------------------------------------------
# Build model args
# ---------------------------------------------------------------------------
MODEL_ARGS="--model ${MODEL_PATH}"
if [ -n "${CHECKPOINT}" ]; then
    MODEL_ARGS="${MODEL_ARGS} --adapters ${CHECKPOINT}"
fi

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
echo "[INFO] Running inference..."
echo "  Model:      ${MODEL_PATH}"
echo "  Checkpoint: ${CHECKPOINT:-base model}"
echo "  Data:       ${VAL_DATA}"
echo "  Output:     ${OUTPUT_DIR}"

uv run swift infer \
    ${MODEL_ARGS} \
    --torch_dtype bfloat16 \
    --val_dataset "${VAL_DATA}" \
    --max_new_tokens 1024 \
    --result_path "${OUTPUT_DIR}/predictions.jsonl" \
    --attn_impl flash_attn

echo "[INFO] Inference completed."
echo "[INFO] Running evaluation..."

uv run python scripts/eval.py \
    --input "${OUTPUT_DIR}/predictions.jsonl" \
    --output "${OUTPUT_DIR}/metrics.json"

echo "[INFO] Done."
