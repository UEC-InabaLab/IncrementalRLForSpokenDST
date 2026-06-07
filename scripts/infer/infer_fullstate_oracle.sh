#!/usr/bin/env bash
# =============================================================================
# Full-State DST - Oracle Inference with vLLM + Evaluation
#
# Uses ground truth dialogue history for each turn.
#
# Usage:
#   bash scripts/infer/infer_fullstate_oracle.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Model & Checkpoint
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}"
ADAPTER="${ADAPTER:-}"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
VAL_DATA="${VAL_DATA:-data/fullstate_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/fullstate_vllm_inference_results/oracle}"
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
    --mode oracle
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
echo "[INFO] Running full-state vLLM inference (oracle mode)..."
echo "  Model:      ${MODEL_PATH}"
echo "  Adapter:    ${ADAPTER:-none}"
echo "  Data:       ${VAL_DATA}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  TP size:    ${TENSOR_PARALLEL_SIZE}"
echo "  GPUs:       ${CUDA_VISIBLE_DEVICES}"

mkdir -p "${OUTPUT_DIR}" logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/infer_fullstate_oracle_${TIMESTAMP}.log"

nohup bash -c '
uv run python scripts/infer/infer_fullstate.py '"$(printf ' %q' "${INFER_ARGS[@]}")"'
echo "[INFO] Inference completed."
echo "[INFO] Running evaluation..."
uv run python scripts/eval/eval_fullstate.py \
    --input "'"${OUTPUT_DIR}"'/predictions.jsonl" \
    --output "'"${OUTPUT_DIR}"'/metrics.json"
echo "[INFO] Done. Results in '"${OUTPUT_DIR}"'/"
' > "${LOG_FILE}" 2>&1 &

echo "[INFO] Started in background (PID: $!)"
echo "[INFO] Log: tail -f ${LOG_FILE}"
