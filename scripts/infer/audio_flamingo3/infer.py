"""Batch inference for Audio Flamingo 3 on incremental Spoken DST (oracle mode).

Audio Flamingo 3 is not in ms-swift, so it does not go through this repo's
scripts/infer/infer.py. This standalone script loads
`nvidia/audio-flamingo-3-hf` (optionally with a LoRA adapter from the AF3 SFT
script) via transformers and writes a predictions JSONL compatible with
scripts/eval/eval.py.

Only ORACLE mode is implemented here: each turn is decoded independently using
the ground-truth dialogue history already present in the test JSONL's user
message. Predicted/cascading mode (feeding the model's own transcript forward,
with ground-truth system text injected as in scripts/infer/infer.py's
run_predicted) is a TODO — the sys_text / opening_user_text fields carried by
the data are already there to support it when added.

STATUS: written against the documented AF3 / transformers APIs but NOT yet run
(needs GPU + weights). Smoke-test on a few samples first.

Usage (example):
    ADAPTER=output/af3_sft_dstc11 \
    AUDIO_BASE_DIR=data/audio_dstc11/tts_verbatim/test \
    python scripts/infer/audio_flamingo3/infer.py \
        --input data/dstc11/test.jsonl \
        --output output/af3_predictions.jsonl
"""

import argparse
import json
import os
import sys

import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

# Reuse the SFT-side conversation builder (sibling package).
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "train", "audio_flamingo3",
    ),
)
from af3_data import grpo_to_af3_conversation, load_grpo_jsonl  # noqa: E402


def _metadata(sample: dict) -> dict:
    def to_json_str(val) -> str:
        if isinstance(val, dict):
            return json.dumps(val, ensure_ascii=False)
        return val if isinstance(val, str) else "{}"

    return {
        "belief_state": to_json_str(sample.get("belief_state", "{}")),
        # Oracle mode: state fed to the model is the gold previous state.
        "input_belief_state": to_json_str(sample.get("prev_belief_state", "{}")),
        "dialogue_id": sample.get("dialogue_id", "unknown"),
        "turn_idx": sample.get("turn_idx", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Test GRPO-format JSONL")
    parser.add_argument("--output", required=True, help="Predictions JSONL for eval.py")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "nvidia/audio-flamingo-3-hf"))
    parser.add_argument("--adapter", default=os.environ.get("ADAPTER") or None)
    parser.add_argument("--audio-base-dir", default=os.environ.get("AUDIO_BASE_DIR") or None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    data = load_grpo_jsonl(args.input)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results = []
    for start in range(0, len(data), args.batch_size):
        batch = data[start:start + args.batch_size]
        conversations = [
            grpo_to_af3_conversation(s, args.audio_base_dir, include_solution=False)
            for s in batch
        ]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(model.device, dtype=model.dtype)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        gen = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(gen, skip_special_tokens=True)

        for sample, prediction in zip(batch, decoded):
            results.append({
                "prediction": prediction,
                "solution": sample.get("solution", ""),
                **_metadata(sample),
            })

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} predictions → {args.output}")


if __name__ == "__main__":
    main()
