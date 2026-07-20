"""Convert SpokenWOZ data to GRPO-format JSONL for incremental DST training.

Expected SpokenWOZ raw format (e.g. train.json):
  {
    "DIALOGUE_ID": {
      "log": [
        {
          "tag":  "user",
          "text": "yes , i'm looking for a restaurant .",
          "metadata": {},
          "words": [{"Word": "yes", "BeginTime": 6550, "EndTime": 6857, ...}, ...]
        },
        {
          "tag":  "system",
          "text": "okay , any requirement .",
          "metadata": {
            "restaurant": {"book": {"booked": []}, "semi": {"area": "west", ...}},
            ...
          },
          "words": [...]
        },
        ...
      ]
    }
  }

Turn ordering: user-first (log[0].tag == "user").
Belief states (MultiWOZ 2.1 nested format) are stored on system turns.

Each output sample corresponds to one (system, user) pair:
  - system turn:  log[2k+1]   (k = 0, 1, 2, ...)
  - user turn:    log[2k+2]
  - prev_state:   flatten(log[2k+1].metadata)   — state before this exchange
  - curr_state:   flatten(log[2k+3].metadata)   — state after user's reply

Input to the model:
  - Text: dialogue history including the current system turn (log[0..2k+1])
  - Audio: current user turn only

Audio files must be pre-extracted with split_audio.py. Each sample references:
  audios: ["{dialogue_id}_{sys_idx}_{user_idx}.wav"]  (named with both indices for identification)

At inference, set --audio-base-dir to the audio output directory.

Each sample also carries "sys_text" (this turn's system text) and
"opening_user_text" (the dialogue's log[0] user text) verbatim, so that
infer.py's predicted (cascading) mode can inject them as ground truth
directly instead of depending on the model to reproduce them.

Usage:
  # 1. Split audio first (see split_audio.py)
  # 2. Then prepare JSONL. --audio-base-dir bakes in absolute audio paths,
  #    since training scripts (unlike infer.py) have no --audio-base-dir of
  #    their own to resolve bare filenames at load time:
  python scripts/train/prepare_data.py \\
      --data      data/raw/train.json \\
      --output    data/train.jsonl \\
      --audio-base-dir data/audio/train

  python scripts/train/prepare_data.py \\
      --data      data/raw/test.json \\
      --output    data/test.jsonl \\
      --audio-base-dir data/audio/test
"""

import argparse
import json
from pathlib import Path

from dst_common import (
    build_solution,
    build_user_message,
    flatten_multiwoz_metadata,
    load_system_prompt,
    write_jsonl,
)


SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "incremental.txt"


def process_dialogue(
    dialogue_id: str,
    log: list[dict],
    system_prompt: str,
    audio_base_dir: str | None = None,
) -> list[dict]:
    """Convert one dialogue into GRPO samples.

    Pairs: (log[2k+1]=system, log[2k+2]=user) for k = 0, 1, ...
    log[0] (first user turn) is always included as the first history line.
    """
    if len(log) < 3:
        return []

    samples: list[dict] = []

    # log[0] is always a user turn; it goes directly into history
    opening_user_text = log[0]["text"].strip()
    history_lines: list[str] = [f"User: {opening_user_text}"]

    k = 0
    while True:
        sys_idx = 2 * k + 1
        user_idx = 2 * k + 2
        next_sys_idx = 2 * k + 3

        if user_idx >= len(log):
            break  # no more complete (system, user) pairs

        sys_turn = log[sys_idx]
        user_turn = log[user_idx]

        if sys_turn.get("tag") != "system" or user_turn.get("tag") != "user":
            # Unexpected ordering; skip this pair
            k += 1
            continue

        sys_text = sys_turn["text"].strip()
        user_text = user_turn["text"].strip()

        prev_state = flatten_multiwoz_metadata(sys_turn.get("metadata", {}))
        if next_sys_idx < len(log):
            curr_state = flatten_multiwoz_metadata(log[next_sys_idx].get("metadata", {}))
        else:
            curr_state = prev_state  # last pair: no further annotation

        # History includes current system turn (text); audio is user turn only
        history_with_sys = list(history_lines) + [f"System: {sys_text}"]

        audio_filename = f"{dialogue_id}_{sys_idx}_{user_idx}.wav"
        audio_path = str(Path(audio_base_dir) / audio_filename) if audio_base_dir else audio_filename

        samples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_message(history_with_sys, prev_state)},
            ],
            "audios": [audio_path],
            "solution": build_solution(user_text, prev_state, curr_state),
            "belief_state": json.dumps(curr_state, ensure_ascii=False),
            "prev_belief_state": json.dumps(prev_state, ensure_ascii=False),
            "dialogue_id": dialogue_id,
            "turn_idx": k,
            "sys_text": sys_text,
            "opening_user_text": opening_user_text,
        })

        history_lines.append(f"System: {sys_text}")
        history_lines.append(f"User: {user_text}")
        k += 1

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="SpokenWOZ JSON file")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--system-prompt", default=None, help="Override system prompt file")
    parser.add_argument(
        "--audio-base-dir",
        default=None,
        help=(
            "Directory the split_audio.py output for this split lives in (e.g. data/audio/train). "
            "If given, audio paths are written as absolute paths so training scripts (which have no "
            "audio-dir mechanism of their own) can resolve them regardless of working directory. "
            "If omitted, bare filenames are written (matching split_audio.py's output naming)."
        ),
    )
    args = parser.parse_args()

    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH, args.system_prompt)
    audio_base_dir = str(Path(args.audio_base_dir).resolve()) if args.audio_base_dir else None

    with open(args.data, encoding="utf-8") as f:
        data: dict = json.load(f)

    samples: list[dict] = []
    skipped = 0
    for dialogue_id, dialogue in data.items():
        log = dialogue.get("log", [])
        result = process_dialogue(dialogue_id, log, system_prompt, audio_base_dir)
        if not result:
            skipped += 1
        samples.extend(result)

    write_jsonl(samples, args.output)

    print(f"Wrote {len(samples)} samples from {len(data) - skipped} dialogues → {args.output}")
    if skipped:
        print(f"Skipped {skipped} dialogues (too short)")


if __name__ == "__main__":
    main()
