#!/usr/bin/env bash
# =============================================================================
# Incremental DST - Predicted-History Inference with vLLM + Evaluation
#
# Uses model's own predicted transcript/state for subsequent turns (cascading).
#
# Usage:
#   bash scripts/infer/infer_predicted.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Model & Checkpoint
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}"
# ADAPTER="${ADAPTER:-output/sft_incremental_dst/v6-20260212-172415/checkpoint-4800}"
# ADAPTER="${ADAPTER:-output/grpo_incremental_dst/v8-20260213-162756/checkpoint-1200}"
ADAPTER="${ADAPTER:-output/grpo_incremental_dst_no_transcript/v0-20260217-231248/checkpoint-3400}"
# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
VAL_DATA="${VAL_DATA:-data/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/vllm_inference_results/predicted}"
AUDIO_BASE_DIR="${AUDIO_BASE_DIR:-/shrdlu/users/higuchi/dst/audio_flamingo}"

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# Verify GPU availability
if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi failed. No GPU available on this node." >&2
    exit 1
fi
echo "[INFO] GPU check passed:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# Inference parameters
# ---------------------------------------------------------------------------
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# ---------------------------------------------------------------------------
# Build args
# ---------------------------------------------------------------------------
INFER_ARGS=(
    --mode predicted
    --model "${MODEL_PATH}"
    --input "${VAL_DATA}"
    --output "${OUTPUT_DIR}/predictions.jsonl"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

if [ -n "${ADAPTER}" ]; then
    INFER_ARGS+=(--adapter "${ADAPTER}")
fi

if [ -n "${AUDIO_BASE_DIR}" ]; then
    INFER_ARGS+=(--audio-base-dir "${AUDIO_BASE_DIR}")
fi

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
echo "[INFO] Running vLLM inference (predicted-history mode)..."
echo "  Model:      ${MODEL_PATH}"
echo "  Adapter:    ${ADAPTER:-none}"
echo "  Data:       ${VAL_DATA}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  TP size:    ${TENSOR_PARALLEL_SIZE}"
echo "  GPUs:       ${CUDA_VISIBLE_DEVICES}"

mkdir -p "${OUTPUT_DIR}" logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/infer_predicted_${TIMESTAMP}.log"

nohup bash -c '
uv run python scripts/infer/infer.py '"$(printf ' %q' "${INFER_ARGS[@]}")"'
echo "[INFO] Inference completed."
echo "[INFO] Running evaluation..."
uv run python scripts/eval/eval.py \
    --input "'"${OUTPUT_DIR}"'/predictions.jsonl" \
    --output "'"${OUTPUT_DIR}"'/metrics.json"
echo "[INFO] Done. Results in '"${OUTPUT_DIR}"'/"
' > "${LOG_FILE}" 2>&1 &

echo "[INFO] Started in background (PID: $!)"
echo "[INFO] Log: tail -f ${LOG_FILE}"
