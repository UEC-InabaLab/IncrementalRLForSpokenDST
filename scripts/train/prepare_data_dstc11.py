"""Convert DSTC-11 Track "Speech Aware Task-Oriented Dialog Modeling" data to
GRPO-format JSONL for incremental DST training.

Reference: Soltau et al., "DSTC-11: Speech Aware Task-Oriented Dialog
Modeling Track", DSTC11 workshop paper (aclanthology.org/2023.dstc-1.25).

The track re-releases MultiWOZ 2.1 dialogues with spoken user turns in three
variants, alongside the original text and belief-state annotations:
  - tts_verbatim:       user turns synthesized (TTS) from the original text
  - human_verbatim:     user turns read aloud verbatim by a human speaker
  - human_paraphrased:  user turns spoken freely by a human (paraphrased)

NOTE ON RAW FORMAT: as of writing this script, the raw DSTC-11 track data
had not yet been downloaded/inspected in this repo. The layout below mirrors
the SpokenWOZ / MultiWOZ 2.1 "log" structure (same belief-state schema, same
alternating system/user turns), which the track's own description says it
builds on. Field names are centralized in the CONFIG section below — once
the actual files are available, adjust only those constants (and, if the
per-turn ASR/human-speech text lives under a different key, `USER_TEXT_KEYS`)
rather than the parsing logic.

Assumed raw format (one file per split, per variant):
  {
    "DIALOGUE_ID": {
      "log": [
        {
          "tag": "user",
          "text": "yes , i'm looking for a restaurant .",   # original MultiWOZ text (fallback)
          "asr_text": {                                       # per-variant spoken text
            "tts_verbatim": "yes i'm looking for a restaurant",
            "human_verbatim": "yeah i'm looking for a restaurant",
            "human_paraphrased": "i need a place to eat"
          },
          "metadata": {}
        },
        {
          "tag": "system",
          "text": "okay , any requirement .",
          "metadata": {"restaurant": {"book": {"booked": []}, "semi": {"area": "west", ...}}, ...}
        },
        ...
      ]
    }
  }

Belief states use the same MultiWOZ 2.1 nested format as SpokenWOZ, so this
script reuses `flatten_multiwoz_metadata` from dst_common.py unchanged.

Audio (if provided as one WAV per dialogue with word timing) can be split
per-sample with split_audio_dstc11.py, following the same convention as
SpokenWOZ: "{dialogue_id}_{sys_idx}_{user_idx}.wav".

Usage:
  python scripts/train/prepare_data_dstc11.py \\
      --data data/raw_dstc11/train.json \\
      --variant human_verbatim \\
      --output data/dstc11/train.jsonl
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

# --- CONFIG: adjust these to match the actual raw files once inspected ----
VARIANTS = ("tts_verbatim", "human_verbatim", "human_paraphrased")
USER_TEXT_KEY = "asr_text"  # dict of {variant: text} on user turns; falls back to "text"
# ---------------------------------------------------------------------------


def user_turn_text(user_turn: dict, variant: str) -> str:
    per_variant = user_turn.get(USER_TEXT_KEY, {})
    if isinstance(per_variant, dict) and variant in per_variant:
        return per_variant[variant].strip()
    return user_turn["text"].strip()


def process_dialogue(
    dialogue_id: str,
    log: list[dict],
    system_prompt: str,
    variant: str,
) -> list[dict]:
    """Same (system, user) pairing as SpokenWOZ; user text is variant-specific."""
    if len(log) < 3:
        return []

    samples: list[dict] = []
    history_lines: list[str] = [f"User: {user_turn_text(log[0], variant)}"]

    k = 0
    while True:
        sys_idx = 2 * k + 1
        user_idx = 2 * k + 2
        next_sys_idx = 2 * k + 3

        if user_idx >= len(log):
            break

        sys_turn = log[sys_idx]
        user_turn = log[user_idx]

        if sys_turn.get("tag") != "system" or user_turn.get("tag") != "user":
            k += 1
            continue

        sys_text = sys_turn["text"].strip()
        user_text = user_turn_text(user_turn, variant)

        prev_state = flatten_multiwoz_metadata(sys_turn.get("metadata", {}))
        if next_sys_idx < len(log):
            curr_state = flatten_multiwoz_metadata(log[next_sys_idx].get("metadata", {}))
        else:
            curr_state = prev_state

        history_with_sys = list(history_lines) + [f"System: {sys_text}"]

        samples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_message(history_with_sys, prev_state)},
            ],
            "audios": [f"{dialogue_id}_{sys_idx}_{user_idx}.wav"],
            "solution": build_solution(sys_text, user_text, prev_state, curr_state),
            "belief_state": json.dumps(curr_state, ensure_ascii=False),
            "prev_belief_state": json.dumps(prev_state, ensure_ascii=False),
            "dialogue_id": dialogue_id,
            "turn_idx": k,
            "variant": variant,
        })

        history_lines.append(f"System: {sys_text}")
        history_lines.append(f"User: {user_text}")
        k += 1

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="DSTC-11 spoken-MultiWOZ JSON file")
    parser.add_argument("--variant", required=True, choices=VARIANTS, help="Which spoken variant to use for user turns")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--system-prompt", default=None, help="Override system prompt file")
    args = parser.parse_args()

    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH, args.system_prompt)

    with open(args.data, encoding="utf-8") as f:
        data: dict = json.load(f)

    samples: list[dict] = []
    skipped = 0
    for dialogue_id, dialogue in data.items():
        log = dialogue.get("log", [])
        result = process_dialogue(dialogue_id, log, system_prompt, args.variant)
        if not result:
            skipped += 1
        samples.extend(result)

    write_jsonl(samples, args.output)

    print(f"Wrote {len(samples)} samples ({args.variant}) from {len(data) - skipped} dialogues → {args.output}")
    if skipped:
        print(f"Skipped {skipped} dialogues (too short)")


if __name__ == "__main__":
    main()
