"""Run offline batch inference with vLLM on full-state DST data.

Supports two inference modes:
  - oracle:    Use ground truth dialogue history from the test data (independent per sample)
  - predicted: Use model's own predicted transcripts to build history for subsequent turns

Unlike incremental DST, this outputs the complete dialogue state as JSON
at each turn. In predicted mode, only the transcript cascades (not the state).

Supports two input formats:
  1. GRPO format: content is plain string with <audio> placeholder, audio paths in "audios" array
  2. SFT multimodal format: content is list of parts [{type: "text"}, {type: "audio", path: ...}]

Writes predictions JSONL compatible with eval_fullstate.py.

Usage:
  # Oracle mode (default):
  python scripts/infer/infer_fullstate.py \
      --model Qwen/Qwen2.5-Omni-7B \
      --input data/fullstate_test.jsonl \
      --output output/fullstate_predictions.jsonl

  # Predicted history mode:
  python scripts/infer/infer_fullstate.py \
      --mode predicted \
      --model Qwen/Qwen2.5-Omni-7B \
      --input data/fullstate_test.jsonl \
      --output output/fullstate_predictions.jsonl
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_transcript(text: str) -> Optional[str]:
    m = re.search(r"<transcript>(.*?)</transcript>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_answer(text: str) -> Optional[str]:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Data loading and format helpers
# ---------------------------------------------------------------------------


def load_data(path: str) -> list[dict[str, Any]]:
    """Load JSONL data."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _resolve_audio_path(path: str, audio_base_dir: str | None) -> str:
    """Resolve an audio path to an absolute path."""
    if audio_base_dir and not os.path.isabs(path):
        path = os.path.join(audio_base_dir, path)
    return str(Path(path).resolve())


def _is_multimodal_content(content: Any) -> bool:
    """Check if content is in multimodal list format (vs plain string)."""
    return isinstance(content, list)


def _convert_content_parts(
    parts: list[dict[str, Any]], audio_base_dir: str | None
) -> list[dict[str, Any]]:
    """Convert multimodal content parts to vLLM format."""
    converted = []
    for i, part in enumerate(parts):
        if part["type"] == "text":
            text = part["text"]
            next_is_audio = (
                i + 1 < len(parts) and parts[i + 1]["type"] == "audio"
            )
            if next_is_audio and not text.endswith("\n"):
                text = text + "\n"
            converted.append({"type": "text", "text": text})
        elif part["type"] == "audio":
            abs_path = _resolve_audio_path(part["path"], audio_base_dir)
            converted.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"file://{abs_path}"},
                }
            )
        else:
            converted.append(part)
    return converted


def _convert_grpo_user_content(
    content: str, audios: list[str], audio_base_dir: str | None
) -> tuple[list[dict[str, Any]], int]:
    """Convert GRPO-format plain string content with <audio> placeholders."""
    parts = content.split("<audio>")
    content_parts: list[dict[str, Any]] = []
    audio_idx = 0

    for i, part in enumerate(parts):
        if part:
            content_parts.append({"type": "text", "text": part})
        if i < len(parts) - 1 and audio_idx < len(audios):
            abs_path = _resolve_audio_path(audios[audio_idx], audio_base_dir)
            content_parts.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"file://{abs_path}"},
                }
            )
            audio_idx += 1

    return content_parts, audio_idx


def convert_messages(
    sample: dict[str, Any], audio_base_dir: str | None
) -> list[dict[str, Any]]:
    """Convert sample messages to vLLM multimodal chat format.

    Skips assistant messages (ground truth).
    """
    audios = sample.get("audios", [])
    audio_idx = 0
    messages = []

    for msg in sample["messages"]:
        role = msg["role"]
        content = msg["content"]

        if role == "assistant":
            continue

        if _is_multimodal_content(content):
            converted = _convert_content_parts(content, audio_base_dir)
            messages.append({"role": role, "content": converted})
        elif role == "user" and "<audio>" in content:
            content_parts, consumed = _convert_grpo_user_content(
                content, audios[audio_idx:], audio_base_dir
            )
            audio_idx += consumed
            messages.append({"role": role, "content": content_parts})
        else:
            messages.append({"role": role, "content": content})

    return messages


def extract_solution(sample: dict[str, Any]) -> str:
    """Extract the ground truth solution text from a sample."""
    if "solution" in sample:
        return sample["solution"]

    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            content = msg["content"]
            if isinstance(content, list):
                return "".join(
                    p["text"] for p in content if p["type"] == "text"
                )
            return content

    return ""


def extract_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    """Extract evaluation metadata from a sample."""
    def to_json_str(val: Any) -> str:
        if isinstance(val, dict):
            return json.dumps(val, ensure_ascii=False)
        if isinstance(val, str):
            return val
        return "{}"

    turn_idx = sample.get("turn_idx", 0)
    if turn_idx == 0 and "id" in sample:
        parts = sample["id"].rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            turn_idx = int(parts[1])

    return {
        "belief_state": to_json_str(sample.get("belief_state", "{}")),
        "dialogue_id": sample.get("dialogue_id", "unknown"),
        "turn_idx": turn_idx,
    }


def _get_turn_idx(sample: dict[str, Any]) -> int:
    """Extract turn index for ordering within a dialogue."""
    turn_idx = sample.get("turn_idx", -1)
    if turn_idx < 0 and "id" in sample:
        parts = sample["id"].rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            turn_idx = int(parts[1])
    return turn_idx


# ---------------------------------------------------------------------------
# Predicted mode helpers
# ---------------------------------------------------------------------------


def _extract_audio_parts(
    sample: dict[str, Any], audio_base_dir: str | None
) -> list[dict[str, Any]]:
    """Extract audio content parts from a sample's user message."""
    for msg in sample["messages"]:
        if msg["role"] != "user":
            continue
        content = msg["content"]
        if _is_multimodal_content(content):
            parts = []
            for part in content:
                if part.get("type") == "audio":
                    abs_path = _resolve_audio_path(part["path"], audio_base_dir)
                    parts.append(
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"file://{abs_path}"},
                        }
                    )
            return parts
        elif "<audio>" in content:
            audios = sample.get("audios", [])
            return [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"file://{_resolve_audio_path(a, audio_base_dir)}"
                    },
                }
                for a in audios
            ]
    return []


def _build_user_text(history_lines: list[str]) -> str:
    """Build user message text from predicted history (full history).

    Matches full-state training format:
      [Dialogue History]
      System: ...
      User: ...

      [New Audio]
    """
    parts = []
    if history_lines:
        parts.append("[Dialogue History]")
        parts.extend(history_lines)
        parts.append("")  # blank line

    parts.append("[New Audio]")
    return "\n".join(parts)


def _get_system_content(
    sample: dict[str, Any], audio_base_dir: str | None
) -> str | list[dict[str, Any]]:
    """Extract the system message content from a sample."""
    for msg in sample["messages"]:
        if msg["role"] == "system":
            content = msg["content"]
            if _is_multimodal_content(content):
                return _convert_content_parts(content, audio_base_dir)
            return content
    return ""


def _build_predicted_messages(
    sample: dict[str, Any],
    history_lines: list[str],
    audio_base_dir: str | None,
) -> list[dict[str, Any]]:
    """Build messages with predicted history for a single sample.

    System message is kept as-is. User message text is rebuilt from
    predicted history, with original audio parts preserved.
    """
    messages = []

    # System message
    sys_content = _get_system_content(sample, audio_base_dir)
    messages.append({"role": "system", "content": sys_content})

    # User message with predicted history
    user_text = _build_user_text(history_lines)
    user_text += "\n"  # trailing \n before audio
    audio_parts = _extract_audio_parts(sample, audio_base_dir)
    content_parts: list[dict[str, Any]] = [
        {"type": "text", "text": user_text}
    ]
    content_parts.extend(audio_parts)
    messages.append({"role": "user", "content": content_parts})

    return messages


# ---------------------------------------------------------------------------
# Oracle mode: batch inference with ground truth history
# ---------------------------------------------------------------------------


def run_oracle(
    llm: LLM,
    data: list[dict[str, Any]],
    sampling_params: SamplingParams,
    audio_base_dir: str | None,
    lora_request: Optional[LoRARequest],
) -> list[dict[str, Any]]:
    """Run inference using ground truth dialogue history."""
    print("[INFO] Mode: oracle (ground truth history)")
    print("[INFO] Building multimodal messages...")

    all_messages = []
    for sample in data:
        messages = convert_messages(sample, audio_base_dir)
        all_messages.append(messages)

    print(f"[INFO] Running inference on {len(all_messages)} samples...")
    chat_kwargs: dict[str, Any] = dict(
        messages=all_messages,
        sampling_params=sampling_params,
    )
    if lora_request:
        chat_kwargs["lora_request"] = lora_request
    outputs = llm.chat(**chat_kwargs)

    results = []
    for sample, output in zip(data, outputs):
        prediction = output.outputs[0].text
        solution = extract_solution(sample)
        meta = extract_metadata(sample)
        results.append({"prediction": prediction, "solution": solution, **meta})

    return results


# ---------------------------------------------------------------------------
# Predicted mode: cascading inference with model-predicted history
# ---------------------------------------------------------------------------


def run_predicted(
    llm: LLM,
    data: list[dict[str, Any]],
    sampling_params: SamplingParams,
    audio_base_dir: str | None,
    lora_request: Optional[LoRARequest],
) -> list[dict[str, Any]]:
    """Run inference using model-predicted history (cascading evaluation).

    Processes each dialogue turn by turn. After each round, the model's
    predicted transcript is used to build the text context for the next turn.
    Only transcript cascades; each turn outputs the full state independently.
    """
    print("[INFO] Mode: predicted (model-predicted history)")

    # Group by dialogue_id and sort by turn order
    dialogues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in data:
        did = sample.get("dialogue_id", "unknown")
        dialogues[did].append(sample)

    for did in dialogues:
        dialogues[did].sort(key=_get_turn_idx)

    dialogue_ids = sorted(dialogues.keys())
    max_turns = max(len(turns) for turns in dialogues.values())
    print(
        f"[INFO] {len(dialogue_ids)} dialogues, "
        f"max {max_turns} turns per dialogue"
    )

    # Per-dialogue predicted history (accumulated utterance lines)
    pred_history: dict[str, list[str]] = {did: [] for did in dialogue_ids}

    # Collect all results keyed by sample id
    results_map: dict[str, dict[str, Any]] = {}

    for round_idx in range(max_turns):
        round_dids = [
            did for did in dialogue_ids if round_idx < len(dialogues[did])
        ]
        round_samples = [dialogues[did][round_idx] for did in round_dids]

        # Build messages with predicted history
        round_messages = []
        for sample, did in zip(round_samples, round_dids):
            messages = _build_predicted_messages(
                sample, pred_history[did], audio_base_dir
            )
            round_messages.append(messages)

        # Batch inference for this round
        print(
            f"[INFO] Round {round_idx + 1}/{max_turns}: "
            f"{len(round_messages)} samples"
        )
        chat_kwargs: dict[str, Any] = dict(
            messages=round_messages,
            sampling_params=sampling_params,
        )
        if lora_request:
            chat_kwargs["lora_request"] = lora_request
        outputs = llm.chat(**chat_kwargs)

        # Process outputs and update predicted history
        for sample, did, output in zip(round_samples, round_dids, outputs):
            prediction = output.outputs[0].text

            # Update dialogue history with predicted transcript
            transcript = _extract_transcript(prediction)
            if transcript:
                for line in transcript.strip().split("\n"):
                    line = line.strip()
                    if line:
                        pred_history[did].append(line)

            # Store result
            solution = extract_solution(sample)
            meta = extract_metadata(sample)
            sample_id = sample.get("id", f"{did}_{round_idx}")
            results_map[sample_id] = {
                "prediction": prediction,
                "solution": solution,
                **meta,
            }

    # Return results in original data order
    results = []
    for sample in data:
        sample_id = sample.get("id", "unknown")
        if sample_id in results_map:
            results.append(results_map[sample_id])
        else:
            results.append(
                {
                    "prediction": "",
                    "solution": extract_solution(sample),
                    **extract_metadata(sample),
                }
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM inference on full-state DST data"
    )
    parser.add_argument(
        "--mode",
        choices=["oracle", "predicted"],
        default="oracle",
        help="oracle: use ground truth history; predicted: use model predictions",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-Omni-7B", help="Model name or path"
    )
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output predictions JSONL")
    parser.add_argument(
        "--audio-base-dir",
        default=None,
        help="Base directory prepended to relative audio paths",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9
    )
    parser.add_argument("--max-lora-rank", type=int, default=64)
    args = parser.parse_args()

    # --- Load data ---
    print(f"[INFO] Loading data from {args.input}")
    data = load_data(args.input)
    print(f"[INFO] Loaded {len(data)} samples")

    # --- Initialize vLLM ---
    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        limit_mm_per_prompt={"audio": 5},
        allowed_local_media_path="/",
    )
    if args.adapter:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank

    print(f"[INFO] Initializing vLLM with model: {args.model}")
    if args.adapter:
        print(f"[INFO] LoRA adapter: {args.adapter}")
    llm = LLM(**llm_kwargs)

    # --- Inference ---
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    lora_request = None
    if args.adapter:
        lora_request = LoRARequest("adapter", 1, args.adapter)

    if args.mode == "oracle":
        results = run_oracle(
            llm, data, sampling_params, args.audio_base_dir, lora_request
        )
    else:
        results = run_predicted(
            llm, data, sampling_params, args.audio_base_dir, lora_request,
        )

    # --- Write results ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"[INFO] Predictions saved to {args.output}")
    print(f"[INFO] Total: {len(results)} samples")


if __name__ == "__main__":
    main()
