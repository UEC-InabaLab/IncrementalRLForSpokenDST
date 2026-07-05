"""Batch inference for Kimi-Audio on incremental Spoken DST (oracle mode).

Kimi-Audio is not in ms-swift/vLLM's standard path here, so it uses MoonshotAI's
own `kimia_infer` library. This script builds each prompt from our GRPO-format
JSONL (reusing the training-side conversation builder, without the assistant
turn) and writes a predictions JSONL compatible with scripts/eval/eval.py.

We want text-only output (the `<transcript>`/`<answer>` string); Kimi-Audio can
also emit audio via a parallel head, which this task never needs — so generation
is configured for text output only (see `output_type` below; reconcile with the
current kimia_infer API).

STATUS: written against MoonshotAI/Kimi-Audio's documented `KimiAudio` API but
NOT run here (needs GPU + weights + their env, transformers < 4.52.4). The
generate() return signature and sampling-param names may differ by version —
verify against their README before a real run. Only ORACLE mode is implemented;
predicted/cascading mode is a TODO (sys_text/opening_user_text are carried in
the data to support it).

Usage (example, in the Kimi-Audio env):
    AUDIO_BASE_DIR=data/audio_dstc11/tts_verbatim/test \
    python scripts/infer/kimi_audio/infer.py \
        --input data/dstc11/test.jsonl \
        --output output/kimia_predictions.jsonl \
        --model moonshotai/Kimi-Audio-7B-Instruct
"""

import argparse
import json
import os
import sys

# Reuse the conversation builder from the training-side converter.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "train", "kimi_audio",
    ),
)
from convert_to_kimia_format import grpo_to_kimia_conversation  # noqa: E402


def _load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _metadata(sample: dict) -> dict:
    def to_json_str(val) -> str:
        if isinstance(val, dict):
            return json.dumps(val, ensure_ascii=False)
        return val if isinstance(val, str) else "{}"

    return {
        "belief_state": to_json_str(sample.get("belief_state", "{}")),
        "input_belief_state": to_json_str(sample.get("prev_belief_state", "{}")),
        "dialogue_id": sample.get("dialogue_id", "unknown"),
        "turn_idx": sample.get("turn_idx", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Test GRPO-format JSONL")
    parser.add_argument("--output", required=True, help="Predictions JSONL for eval.py")
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "moonshotai/Kimi-Audio-7B-Instruct"))
    parser.add_argument("--audio-base-dir", default=os.environ.get("AUDIO_BASE_DIR") or None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    from kimia_infer.api.kimia import KimiAudio

    model = KimiAudio(model_path=args.model, load_detokenizer=False)

    # Text-only sampling; reconcile these names with the current kimia_infer API.
    sampling_params = {
        "audio_temperature": 0.0,
        "audio_top_k": 5,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "max_new_tokens": args.max_new_tokens,
    }

    data = _load_jsonl(args.input)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results = []
    for sample in data:
        messages = grpo_to_kimia_conversation(
            sample, args.audio_base_dir, include_solution=False
        )
        # Text-out understanding task: we only keep the decoded text.
        _wav, text = model.generate(
            messages, **sampling_params, output_type="text"
        )
        results.append({
            "prediction": text,
            "solution": sample.get("solution", ""),
            **_metadata(sample),
        })

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} predictions → {args.output}")


if __name__ == "__main__":
    main()
