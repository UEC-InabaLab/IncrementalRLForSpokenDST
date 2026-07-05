"""Convert this project's GRPO-format JSONL into Audio Flamingo 3 chat
conversations for SFT / inference.

Audio Flamingo 3 (nvidia/audio-flamingo-3-hf) is a standard transformers
`PreTrainedModel` whose processor consumes chat conversations of the form:

    [{"role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "audio", "path": "/abs/path.wav"},
    ]},
     {"role": "assistant", "content": [{"type": "text", "text": "..."}]}]  # SFT only

Our GRPO-format sample (see scripts/train/prepare_data*.py) looks like:

    {"messages": [
        {"role": "system", "content": "<system prompt>"},
        {"role": "user", "content": "[Dialogue History]\\n...\\n[New Audio]\\n<audio>"}],
     "audios": ["mul0012_3.wav"],
     "solution": "<transcript>\\nUser: ...\\n</transcript>\\n<answer>...</answer>",
     ...}

The user content is a plain string with one `<audio>` placeholder per entry
in `audios`. This module splits that string on `<audio>` and interleaves the
resolved audio paths to build AF3 content parts.

NOTE: AF3's backbone is Qwen2, whose chat template supports a system role,
but to be robust across template versions we fold the system prompt into the
leading text of the first user turn rather than relying on a separate system
message. This keeps behaviour identical to how the ms-swift pipeline presents
the prompt (system text, then the dialogue-history user text, then audio).

This module is pure Python (no torch/transformers) so it can be unit-tested
without the model or GPU.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

AUDIO_PLACEHOLDER = "<audio>"


def resolve_audio_path(path: str, audio_base_dir: Optional[str]) -> str:
    if audio_base_dir and not os.path.isabs(path):
        path = os.path.join(audio_base_dir, path)
    return str(Path(path).resolve())


def _split_user_content_to_parts(
    content: str, audios: list[str], audio_base_dir: Optional[str]
) -> list[dict[str, Any]]:
    """Turn a GRPO user-content string with <audio> placeholders into AF3
    content parts, interleaving resolved audio paths."""
    segments = content.split(AUDIO_PLACEHOLDER)
    parts: list[dict[str, Any]] = []
    audio_idx = 0
    for i, segment in enumerate(segments):
        if segment:
            parts.append({"type": "text", "text": segment})
        # A placeholder sat between this segment and the next one.
        if i < len(segments) - 1 and audio_idx < len(audios):
            parts.append({
                "type": "audio",
                "path": resolve_audio_path(audios[audio_idx], audio_base_dir),
            })
            audio_idx += 1
    return parts


def grpo_to_af3_conversation(
    sample: dict[str, Any],
    audio_base_dir: Optional[str] = None,
    include_solution: bool = False,
) -> list[dict[str, Any]]:
    """Build an AF3 chat `conversation` from one GRPO-format sample.

    If `include_solution` is True (SFT), an assistant turn carrying the gold
    `solution` text is appended; otherwise (inference) it is omitted.
    """
    system_text = ""
    user_content = ""
    for msg in sample["messages"]:
        if msg["role"] == "system":
            system_text = msg["content"] if isinstance(msg["content"], str) else ""
        elif msg["role"] == "user":
            user_content = msg["content"] if isinstance(msg["content"], str) else ""

    # Fold the system prompt into the leading user text (see module docstring).
    if system_text:
        user_content = f"{system_text}\n\n{user_content}"

    audios = sample.get("audios", [])
    user_parts = _split_user_content_to_parts(user_content, audios, audio_base_dir)

    conversation: list[dict[str, Any]] = [{"role": "user", "content": user_parts}]

    if include_solution:
        solution = sample.get("solution", "")
        conversation.append(
            {"role": "assistant", "content": [{"type": "text", "text": solution}]}
        )

    return conversation


def load_grpo_jsonl(path: str) -> list[dict[str, Any]]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
