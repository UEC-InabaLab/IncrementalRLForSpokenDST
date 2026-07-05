#!/usr/bin/env bash
# =============================================================================
# Kimi-Audio SFT for incremental Spoken DST — wrapper around MoonshotAI's
# official fine-tuning code.
#
# Kimi-Audio is NOT registered in ms-swift and has no AutoModel class, so it
# cannot use this repo's train_sft.sh. MoonshotAI ships full-parameter DeepSpeed
# fine-tuning in their repo's `finetune_codes/` directory. This wrapper:
#   1. converts our GRPO-format JSONL to their `conversation` schema, and
#   2. launches their finetune script against it.
#
# PREREQUISITES (do these once, manually):
#   - git clone https://github.com/MoonshotAI/Kimi-Audio and set KIMIA_REPO to it
#   - install their requirements in a DEDICATED env (their modeling_kimia.py
#     needs a transformers version < 4.52.4; see MoonshotAI/Kimi-Audio#109)
#   - pre-extract semantic codes if their pipeline requires it
#     (finetune_codes/extract_semantic_codes.py)
#
# STATUS: NOT run end-to-end here (needs GPU, weights, and their repo). Their
# finetune entrypoint / flag names may drift — treat KIMIA_FINETUNE_SCRIPT and
# the launch line below as the spot to reconcile against their current README.
# Their fine-tuning path is full-parameter (LoRA not confirmed).
# =============================================================================
set -euo pipefail

KIMIA_REPO="${KIMIA_REPO:?Set KIMIA_REPO to your cloned MoonshotAI/Kimi-Audio checkout}"
MODEL_PATH="${MODEL_PATH:-moonshotai/Kimi-Audio-7B-Instruct}"

GRPO_TRAIN_DATA="${GRPO_TRAIN_DATA:-data/train.jsonl}"
GRPO_VAL_DATA="${GRPO_VAL_DATA:-data/val.jsonl}"
AUDIO_BASE_DIR="${AUDIO_BASE_DIR:-}"
KIMIA_TRAIN_DATA="${KIMIA_TRAIN_DATA:-data/kimia/train.jsonl}"
KIMIA_VAL_DATA="${KIMIA_VAL_DATA:-data/kimia/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/kimia_sft_incremental_dst}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIO_ARG=()
[ -n "${AUDIO_BASE_DIR}" ] && AUDIO_ARG=(--audio-base-dir "${AUDIO_BASE_DIR}")

# ---------------------------------------------------------------------------
# Step 1: Convert data to Kimi-Audio's fine-tuning schema
# ---------------------------------------------------------------------------
if [ ! -f "${KIMIA_TRAIN_DATA}" ]; then
    echo "[INFO] Converting training data to Kimi-Audio format..."
    python "${HERE}/convert_to_kimia_format.py" \
        --input "${GRPO_TRAIN_DATA}" --output "${KIMIA_TRAIN_DATA}" "${AUDIO_ARG[@]}"
fi
if [ -f "${GRPO_VAL_DATA}" ] && [ ! -f "${KIMIA_VAL_DATA}" ]; then
    echo "[INFO] Converting validation data to Kimi-Audio format..."
    python "${HERE}/convert_to_kimia_format.py" \
        --input "${GRPO_VAL_DATA}" --output "${KIMIA_VAL_DATA}" "${AUDIO_ARG[@]}"
fi

# ---------------------------------------------------------------------------
# Step 2: Launch MoonshotAI's fine-tuning script
# ---------------------------------------------------------------------------
# Reconcile the entrypoint and flags with the current finetune_codes/README.
KIMIA_FINETUNE_SCRIPT="${KIMIA_FINETUNE_SCRIPT:-${KIMIA_REPO}/finetune_codes/finetune_ds.sh}"

echo "[INFO] Launching Kimi-Audio fine-tuning:"
echo "  Repo:        ${KIMIA_REPO}"
echo "  Model:       ${MODEL_PATH}"
echo "  Train data:  ${KIMIA_TRAIN_DATA}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Script:      ${KIMIA_FINETUNE_SCRIPT}"

MODEL_PATH="${MODEL_PATH}" \
TRAIN_DATA="${KIMIA_TRAIN_DATA}" \
VAL_DATA="${KIMIA_VAL_DATA}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${KIMIA_FINETUNE_SCRIPT}"
