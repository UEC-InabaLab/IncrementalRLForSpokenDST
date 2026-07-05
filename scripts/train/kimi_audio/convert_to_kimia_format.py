"""Convert this project's GRPO-format JSONL into Kimi-Audio's fine-tuning schema.

MoonshotAI's official fine-tuning code (`finetune_codes/` in their repo) expects
JSONL where each line is:

    {"task_type": "understanding",
     "conversation": [
        {"role": "user",      "message_type": "text",  "content": "<text>"},
        {"role": "user",      "message_type": "audio", "content": "/abs/path.wav"},
        {"role": "assistant", "message_type": "text",  "content": "<target text>"}
     ]}

Our GRPO-format sample has a system message, a user message whose string content
holds the dialogue-history text plus one `<audio>` placeholder per entry in
`audios`, and a `solution` string (the gold `<transcript>`/`<answer>`).

Mapping decisions:
  - Kimi-Audio's schema has no explicit system role, so the system prompt is
    folded into the leading text user turn (same choice as the AF3 adapter).
  - The `<audio>` placeholder is removed from the text turn; each referenced
    wav becomes its own `message_type: audio` user turn, in order.
  - `task_type` is "understanding" (text-out task; Kimi-Audio's audio-generation
    head is not exercised — see the multi-model plan's Kimi-Audio caveats).

Pure Python (no torch) so it can be unit-tested without the model.

Usage:
    python scripts/train/kimi_audio/convert_to_kimia_format.py \
        --input data/dstc11/train.jsonl \
        --output data/kimia/train.jsonl \
        --audio-base-dir data/audio_dstc11/tts_verbatim/train
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

AUDIO_PLACEHOLDER = "<audio>"


def resolve_audio_path(path: str, audio_base_dir: Optional[str]) -> str:
    if audio_base_dir and not os.path.isabs(path):
        path = os.path.join(audio_base_dir, path)
    return str(Path(path).resolve())


def grpo_to_kimia_conversation(
    sample: dict[str, Any],
    audio_base_dir: Optional[str] = None,
    include_solution: bool = True,
) -> list[dict[str, str]]:
    """Build a Kimi-Audio `conversation` list from one GRPO-format sample."""
    system_text = ""
    user_content = ""
    for msg in sample["messages"]:
        if msg["role"] == "system" and isinstance(msg["content"], str):
            system_text = msg["content"]
        elif msg["role"] == "user" and isinstance(msg["content"], str):
            user_content = msg["content"]

    if system_text:
        user_content = f"{system_text}\n\n{user_content}"

    # Drop the audio placeholder(s); audio becomes separate turns below.
    text_content = user_content.replace(AUDIO_PLACEHOLDER, "").rstrip()

    conversation: list[dict[str, str]] = [
        {"role": "user", "message_type": "text", "content": text_content}
    ]
    for audio in sample.get("audios", []):
        conversation.append({
            "role": "user",
            "message_type": "audio",
            "content": resolve_audio_path(audio, audio_base_dir),
        })

    if include_solution:
        conversation.append({
            "role": "assistant",
            "message_type": "text",
            "content": sample.get("solution", ""),
        })

    return conversation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="GRPO-format JSONL")
    parser.add_argument("--output", required=True, help="Kimi-Audio finetune JSONL")
    parser.add_argument("--audio-base-dir", default=None)
    parser.add_argument(
        "--no-solution",
        action="store_true",
        help="Omit the assistant turn (e.g. to build inference prompts)",
    )
    args = parser.parse_args()

    count = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            record = {
                "task_type": "understanding",
                "conversation": grpo_to_kimia_conversation(
                    sample, args.audio_base_dir, include_solution=not args.no_solution
                ),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} samples: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
