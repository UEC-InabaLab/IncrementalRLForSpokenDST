"""GRPO for Audio Flamingo 3 on incremental Spoken DST (EXPERIMENTAL SPIKE).

This wires TRL's GRPOTrainer to this repo's existing reward
(src/reward.py :: DSTRewardIncremental, which is already a plain callable with
the TRL reward-function signature `(completions, **dataset_columns) -> list[float]`
and imports fine without ms-swift installed).

STATUS — UNVERIFIED SPIKE. TRL's GRPO multimodal support is documented as "not
guaranteed to work with all VLMs", and audio models are not an advertised path
at all. Before investing, run a tiny rollout (a few prompts, num_generations=2)
and confirm: (a) generations are non-empty and well-formed, (b) the reward
values vary sensibly. If TRL cannot drive AF3's audio inputs through rollout,
fall back to a hand-rolled rollout loop: generate with model.generate(), score
completions with DSTRewardIncremental, and apply a GRPO/policy-gradient update
manually.

Data: this repo's GRPO-format JSONL. Each row is exposed to TRL with a "prompt"
(AF3 user-only conversation) plus the passthrough columns the reward needs
(solution / belief_state / prev_belief_state).

Usage (example, start SMALL):
    SFT_ADAPTER=output/af3_sft_dstc11 \
    GRPO_TRAIN_DATA=data/dstc11/train.jsonl \
    AUDIO_BASE_DIR=data/audio_dstc11/tts_verbatim/train \
    OUTPUT_DIR=output/af3_grpo_dstc11 \
    python scripts/train/audio_flamingo3/train_grpo.py
"""

import os
import sys

import torch
from datasets import Dataset
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

# reward.py (repo root / src) — plain callable, no ms-swift needed at import.
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward import DSTRewardIncremental  # noqa: E402
from af3_data import grpo_to_af3_conversation, load_grpo_jsonl  # noqa: E402


MODEL_ID = os.environ.get("MODEL_PATH", "nvidia/audio-flamingo-3-hf")
SFT_ADAPTER = os.environ.get("SFT_ADAPTER") or None
GRPO_TRAIN_DATA = os.environ.get("GRPO_TRAIN_DATA", "data/train.jsonl")
AUDIO_BASE_DIR = os.environ.get("AUDIO_BASE_DIR") or None
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/af3_grpo_incremental_dst")

NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "8"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-6"))
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "1024"))
BETA = float(os.environ.get("BETA", "0.02"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))


def build_dataset() -> Dataset:
    rows = []
    for s in load_grpo_jsonl(GRPO_TRAIN_DATA):
        rows.append({
            "prompt": grpo_to_af3_conversation(s, AUDIO_BASE_DIR, include_solution=False),
            # Passthrough columns consumed by DSTRewardIncremental(**kwargs).
            "solution": s.get("solution", ""),
            "belief_state": s.get("belief_state", "{}"),
            "prev_belief_state": s.get("prev_belief_state", "{}"),
        })
    return Dataset.from_list(rows)


def main() -> None:
    # Imported here so the caveat above is read before hitting any TRL API drift.
    from trl import GRPOConfig, GRPOTrainer

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    )
    if SFT_ADAPTER:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=True)

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        beta=BETA,
        temperature=TEMPERATURE,
        per_device_train_batch_size=int(os.environ.get("BATCH_SIZE", "2")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "8")),
        num_train_epochs=float(os.environ.get("NUM_EPOCHS", "1")),
        logging_steps=5,
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        report_to=os.environ.get("REPORT_TO", "wandb"),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[DSTRewardIncremental()],
        args=config,
        train_dataset=build_dataset(),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()
