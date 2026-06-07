#!/usr/bin/env bash
# =============================================================================
# Full-State DST - SFT Training with ms-swift
# Model: Qwen2.5-Omni-7B
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}"
TRAIN_DATA="${TRAIN_DATA:-data/fullstate_sft_train.jsonl}"
VAL_DATA="${VAL_DATA:-data/fullstate_sft_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/sft_fullstate_dst}"

NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
WANDB_PROJECT="${WANDB_PROJECT:-qwenomni-sft-fullstate}"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/sft_fullstate_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# Step 1: Convert data if SFT files don't exist
# ---------------------------------------------------------------------------
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "[INFO] Converting training data to full-state SFT format..."
    uv run python scripts/train/prepare_fullstate_data.py \
        --input data/train.jsonl \
        --output "${TRAIN_DATA}" \
        --format grpo
    # Convert GRPO -> SFT format
    uv run python scripts/train/convert_to_sft.py \
        --input "${TRAIN_DATA}" \
        --output "${TRAIN_DATA}.tmp"
    mv "${TRAIN_DATA}.tmp" "${TRAIN_DATA}"
fi

if [ ! -f "${VAL_DATA}" ]; then
    echo "[INFO] Converting validation data to full-state SFT format..."
    uv run python scripts/train/prepare_fullstate_data.py \
        --input data/val.jsonl \
        --output "${VAL_DATA}" \
        --format grpo
    # Convert GRPO -> SFT format
    uv run python scripts/train/convert_to_sft.py \
        --input "${VAL_DATA}" \
        --output "${VAL_DATA}.tmp"
    mv "${VAL_DATA}.tmp" "${VAL_DATA}"
fi

# ---------------------------------------------------------------------------
# Step 2: Run SFT
# ---------------------------------------------------------------------------
export WANDB_PROJECT
echo "[INFO] Starting full-state SFT training..."
echo "  Model:       ${MODEL_PATH}"
echo "  Train data:  ${TRAIN_DATA}"
echo "  Val data:    ${VAL_DATA}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  GPUs:        ${NUM_GPUS}"
echo "  LoRA rank:   ${LORA_RANK}"
echo "  Log:         ${LOG_FILE}"

nohup uv run torchrun --nproc_per_node=${NUM_GPUS} \
    $(uv run python -c "import swift; print(swift.__path__[0])")/cli/sft.py \
    --model "${MODEL_PATH}" \
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
    --max_length ${MAX_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --report_to wandb \
    --output_dir "${OUTPUT_DIR}" \
    --freeze_vit true \
    --freeze_aligner true \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    > "${LOG_FILE}" 2>&1 &

echo "[INFO] Full-state SFT training started in background (PID: $!)"
echo "[INFO] Log: tail -f ${LOG_FILE}"
