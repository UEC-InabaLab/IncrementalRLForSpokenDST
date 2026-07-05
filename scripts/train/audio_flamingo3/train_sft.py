"""SFT for Audio Flamingo 3 on incremental Spoken DST, via HF Trainer + PEFT LoRA.

Audio Flamingo 3 is not registered in ms-swift, so it cannot use this repo's
`train_sft.sh` (which drives the ms-swift CLI). Instead this standalone script
fine-tunes `nvidia/audio-flamingo-3-hf` — a standard transformers
`PreTrainedModel` — with LoRA on the language-model projections while keeping
the Whisper-style audio encoder frozen (LoRA freezes all non-adapter params by
default, so freezing the encoder is automatic given the target modules below).

Data: this repo's GRPO-format JSONL (produced by scripts/train/prepare_data*.py),
converted to AF3 chat conversations by af3_data.grpo_to_af3_conversation.

The loss contract follows the official HF model card example:
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False,
        return_dict=True, output_labels=True)
    loss = model(**inputs).loss

STATUS: written against the documented AF3 / transformers / PEFT APIs but NOT
yet run end-to-end (requires a GPU + the model weights, unavailable in the dev
environment). Treat the first real run as a smoke test: start with a handful of
samples and confirm loss decreases before a full run.

Usage (example):
    GRPO_TRAIN_DATA=data/dstc11/train.jsonl \
    GRPO_VAL_DATA=data/dstc11/val.jsonl \
    AUDIO_BASE_DIR=data/audio_dstc11/tts_verbatim/train \
    OUTPUT_DIR=output/af3_sft_dstc11 \
    python scripts/train/audio_flamingo3/train_sft.py
"""

import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AudioFlamingo3ForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

# Sibling import (same pattern as prepare_data_dstc11.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from af3_data import grpo_to_af3_conversation, load_grpo_jsonl  # noqa: E402


MODEL_ID = os.environ.get("MODEL_PATH", "nvidia/audio-flamingo-3-hf")
GRPO_TRAIN_DATA = os.environ.get("GRPO_TRAIN_DATA", "data/train.jsonl")
GRPO_VAL_DATA = os.environ.get("GRPO_VAL_DATA", "data/val.jsonl")
AUDIO_BASE_DIR = os.environ.get("AUDIO_BASE_DIR") or None
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/af3_sft_incremental_dst")

LORA_RANK = int(os.environ.get("LORA_RANK", "64"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "128"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))

# LoRA targets: Qwen2 LLM backbone attention + MLP projections. The audio
# encoder / multimodal projector are left out, so they stay frozen.
LORA_TARGET_MODULES = os.environ.get(
    "LORA_TARGET_MODULES",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
).split(",")


@dataclass
class AF3Collator:
    processor: Any

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        conversations = [
            grpo_to_af3_conversation(s, AUDIO_BASE_DIR, include_solution=True)
            for s in samples
        ]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            output_labels=True,
            return_tensors="pt",
            padding=True,
        )
        return inputs


def main() -> None:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_samples = load_grpo_jsonl(GRPO_TRAIN_DATA)
    train_ds = Dataset.from_list(train_samples)
    eval_ds = None
    if os.path.exists(GRPO_VAL_DATA):
        eval_ds = Dataset.from_list(load_grpo_jsonl(GRPO_VAL_DATA))

    # Keep raw dict rows; the collator builds tensors per batch.
    train_ds = train_ds.with_format("python")
    if eval_ds is not None:
        eval_ds = eval_ds.with_format("python")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=200,
        bf16=True,
        gradient_checkpointing=True,
        report_to=os.environ.get("REPORT_TO", "wandb"),
        remove_unused_columns=False,  # collator needs the raw JSONL columns
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=AF3Collator(processor),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
