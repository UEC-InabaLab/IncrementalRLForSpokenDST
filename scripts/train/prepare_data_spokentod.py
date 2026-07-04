"""Convert SpokenTOD data to GRPO-format JSONL for incremental DST training.

Reference: "SpokenUS: A Spoken User Simulator for Task-Oriented Dialogue"
(arXiv:2603.16783), which introduces the SpokenTOD corpus: ~52.4k dialogues /
~1034h of simulated spoken user turns across multiple domains, with per-turn
tags for cross-turn slot mentions, barge-in, disfluency, and emotional
prosody.

SPECULATIVE RAW FORMAT — the paper does not publish a download link or an
explicit schema, and the raw files have not been obtained/inspected in this
repo yet. The layout assumed below is a best-effort guess based on the paper
description (flat per-turn dialogue state, since SpokenTOD is simulator-
generated rather than derived from MultiWOZ's nested ontology). All
field names are centralized in the CONFIG section — update them (and, if the
actual nesting differs, `flatten_state`) once real data is available; the
diff/JSONL-building logic in dst_common.py does not need to change.

Assumed raw format:
  {
    "DIALOGUE_ID": {
      "turns": [
        {
          "speaker": "USER",             # or "SYSTEM"
          "text": "i'd like something cheap",
          "state": {"restaurant": {"pricerange": "cheap"}},   # cumulative, flat
          "behaviors": {                  # optional, not used for training targets yet
            "cross_turn_slot": false,
            "barge_in": false,
            "disfluency": false,
            "emotional_prosody": "neutral"
          }
        },
        ...
      ]
    }
  }

Turn ordering assumed user-first (turns[0] is a user turn), matching
SpokenWOZ / DSTC-11. Each output sample corresponds to one (system, user)
pair, identical in shape to the SpokenWOZ pipeline, so the same training /
inference / eval scripts work unchanged.

Audio is assumed to already be provided per-user-turn (simulator output),
so no split_audio step is included here — verify this once raw data lands;
add a split_audio_spokentod.py if audio instead ships per-dialogue.

Usage:
  python scripts/train/prepare_data_spokentod.py \\
      --data data/raw_spokentod/train.json \\
      --output data/spokentod/train.jsonl
"""

import argparse
import json
from pathlib import Path

from dst_common import (
    build_solution,
    build_user_message,
    load_system_prompt,
    write_jsonl,
)


SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "incremental.txt"

# --- CONFIG: adjust to match the actual raw files once inspected ----------
USER_SPEAKER_TAG = "USER"
SYSTEM_SPEAKER_TAG = "SYSTEM"
AUDIO_FILE_KEY = "audio"  # expected key on user turns pointing to a pre-split WAV filename
# ---------------------------------------------------------------------------


def flatten_state(state: dict) -> dict:
    """SpokenTOD state is assumed already flat ({domain: {slot: value}});
    just drop empty/falsy slot values for consistency with the other adapters."""
    flat: dict = {}
    for domain, slots in (state or {}).items():
        if not isinstance(slots, dict):
            continue
        kept = {slot: value for slot, value in slots.items() if value}
        if kept:
            flat[domain] = kept
    return flat


def process_dialogue(dialogue_id: str, turns: list[dict], system_prompt: str) -> list[dict]:
    if len(turns) < 3:
        return []

    samples: list[dict] = []
    history_lines: list[str] = [f"User: {turns[0]['text'].strip()}"]

    k = 0
    while True:
        sys_idx = 2 * k + 1
        user_idx = 2 * k + 2
        next_sys_idx = 2 * k + 3

        if user_idx >= len(turns):
            break

        sys_turn = turns[sys_idx]
        user_turn = turns[user_idx]

        if sys_turn.get("speaker") != SYSTEM_SPEAKER_TAG or user_turn.get("speaker") != USER_SPEAKER_TAG:
            k += 1
            continue

        sys_text = sys_turn["text"].strip()
        user_text = user_turn["text"].strip()

        prev_state = flatten_state(sys_turn.get("state", {}))
        if next_sys_idx < len(turns):
            curr_state = flatten_state(turns[next_sys_idx].get("state", {}))
        else:
            curr_state = prev_state

        history_with_sys = list(history_lines) + [f"System: {sys_text}"]

        audio_file = user_turn.get(AUDIO_FILE_KEY, f"{dialogue_id}_{sys_idx}_{user_idx}.wav")

        samples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_message(history_with_sys, prev_state)},
            ],
            "audios": [audio_file],
            "solution": build_solution(sys_text, user_text, prev_state, curr_state),
            "belief_state": json.dumps(curr_state, ensure_ascii=False),
            "prev_belief_state": json.dumps(prev_state, ensure_ascii=False),
            "dialogue_id": dialogue_id,
            "turn_idx": k,
        })

        history_lines.append(f"System: {sys_text}")
        history_lines.append(f"User: {user_text}")
        k += 1

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="SpokenTOD JSON file")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--system-prompt", default=None, help="Override system prompt file")
    args = parser.parse_args()

    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH, args.system_prompt)

    with open(args.data, encoding="utf-8") as f:
        data: dict = json.load(f)

    samples: list[dict] = []
    skipped = 0
    for dialogue_id, dialogue in data.items():
        turns = dialogue.get("turns", [])
        result = process_dialogue(dialogue_id, turns, system_prompt)
        if not result:
            skipped += 1
        samples.extend(result)

    write_jsonl(samples, args.output)

    print(f"Wrote {len(samples)} samples from {len(data) - skipped} dialogues → {args.output}")
    if skipped:
        print(f"Skipped {skipped} dialogues (too short)")


if __name__ == "__main__":
    main()
