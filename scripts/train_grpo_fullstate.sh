#!/usr/bin/env bash
# =============================================================================
# Full-State DST - GRPO Training with ms-swift
# Model: Qwen2.5-Omni-7B (optionally starting from SFT checkpoint)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}"
# Set SFT_CHECKPOINT to use a LoRA checkpoint as the starting point
SFT_CHECKPOINT="${SFT_CHECKPOINT:-}"

TRAIN_DATA="${TRAIN_DATA:-data/fullstate_dapo_train.jsonl}"
VAL_DATA_FULL="${VAL_DATA_FULL:-data/fullstate_dapo_val.jsonl}"
VAL_DATA="${VAL_DATA:-data/fullstate_dapo_val_small.jsonl}"
VAL_SAMPLE_N="${VAL_SAMPLE_N:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-output/grpo_fullstate_dst}"
PLUGIN_PATH="${PLUGIN_PATH:-src/swift_plugin/dapo_reward.py}"

NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-8}"
NUM_INFER_GPUS="${NUM_INFER_GPUS:-2}"
TOTAL_GPUS="${TOTAL_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
BETA="${BETA:-0.02}"
NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-qwenomni-grpo-fullstate}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/grpo_fullstate_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# Prepare data if needed
# ---------------------------------------------------------------------------
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "[INFO] Converting training data to full-state GRPO format..."
    uv run python scripts/prepare_fullstate_data.py \
        --input data/dapo_train.jsonl \
        --output "${TRAIN_DATA}" \
        --format grpo
fi

if [ ! -f "${VAL_DATA_FULL}" ]; then
    echo "[INFO] Converting validation data to full-state GRPO format..."
    uv run python scripts/prepare_fullstate_data.py \
        --input data/dapo_val.jsonl \
        --output "${VAL_DATA_FULL}" \
        --format grpo
fi

if [ ! -f "${VAL_DATA}" ]; then
    echo "[INFO] Sampling ${VAL_SAMPLE_N} val examples from ${VAL_DATA_FULL}..."
    uv run python scripts/sample_val.py \
        --input "${VAL_DATA_FULL}" --output "${VAL_DATA}" --n "${VAL_SAMPLE_N}"
fi

# ---------------------------------------------------------------------------
# Build model args
# ---------------------------------------------------------------------------
MODEL_ARGS="--model ${MODEL_PATH}"
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "[INFO] Resuming from checkpoint: ${RESUME_CHECKPOINT}"
elif [ -n "${SFT_CHECKPOINT}" ]; then
    MODEL_ARGS="${MODEL_ARGS} --adapters ${SFT_CHECKPOINT}"
    echo "[INFO] Starting from SFT checkpoint: ${SFT_CHECKPOINT}"
fi

# ---------------------------------------------------------------------------
# Run GRPO
# ---------------------------------------------------------------------------
export WANDB_PROJECT
echo "[INFO] Starting full-state GRPO training..."
echo "  Model:        ${MODEL_PATH}"
echo "  Train data:   ${TRAIN_DATA}"
echo "  Plugin:       ${PLUGIN_PATH}"
echo "  Reward:       dst_fullstate"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Train GPUs:   ${NUM_TRAIN_GPUS}"
echo "  Generations:  ${NUM_GENERATIONS}"
echo "  Log:          ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TOTAL_GPUS - 1))) \
nohup uv run torchrun --nproc_per_node=${NUM_TRAIN_GPUS} \
    $(uv run python -c "import swift; print(swift.__path__[0])")/cli/rlhf.py \
    --rlhf_type grpo \
    ${MODEL_ARGS} \
    --train_type lora \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant true \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --torch_dtype bfloat16 \
    --dataset "${TRAIN_DATA}" \
    --val_dataset "${VAL_DATA}" \
    --external_plugins "${PLUGIN_PATH}" \
    --reward_funcs dst_fullstate \
    --reward_weights 1.0 \
    --num_generations ${NUM_GENERATIONS} \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --temperature ${TEMPERATURE} \
    --beta ${BETA} \
    --num_iterations ${NUM_ITERATIONS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit -1 \
    --logging_steps 5 \
    --report_to wandb \
    --output_dir "${OUTPUT_DIR}" \
    --freeze_vit true \
    --freeze_aligner true \
    --attn_impl flash_attn \
    --use_vllm false \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    ${RESUME_CHECKPOINT:+--resume_from_checkpoint "${RESUME_CHECKPOINT}"} \
    > "${LOG_FILE}" 2>&1 &

echo "[INFO] Full-state GRPO training started in background (PID: $!)"
echo "[INFO] Log: tail -f ${LOG_FILE}"
