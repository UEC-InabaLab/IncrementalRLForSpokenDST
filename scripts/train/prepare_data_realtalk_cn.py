"""Convert RealTalk-CN data to GRPO-format JSONL for incremental DST training.

Reference: RealTalk-CN, a Chinese spoken task-oriented dialogue benchmark
(arXiv:2508.10015): ~5.4k dialogues / ~150h of real human speech across 58
domains, with intent (55 types) and slot (115 types) annotations plus
disfluency labels (filler particles, repetition, self-correction,
hesitation).

SPECULATIVE RAW FORMAT — the paper states data/code "will be made
available" but does not give a schema or URL, and the raw files have not
been obtained/inspected in this repo yet. The layout assumed below mirrors
prepare_data_spokentod.py's flat per-turn state (intent/slot rather than
MultiWOZ's nested ontology). Field names are centralized in the CONFIG
section — update them once real data is available; the diff/JSONL-building
logic in dst_common.py does not need to change.

Assumed raw format:
  {
    "DIALOGUE_ID": {
      "turns": [
        {
          "speaker": "user",              # or "system"
          "text": "我想订一张明天去北京的机票",
          "intent": "book_flight",
          "state": {"flight": {"date": "明天", "destination": "北京"}},  # cumulative, flat
          "disfluency": ["filler"]          # optional, not used for training targets yet
        },
        ...
      ]
    }
  }

Turn ordering assumed user-first, matching the other adapters. Each output
sample corresponds to one (system, user) pair, identical in shape to the
SpokenWOZ pipeline, so the same training / inference / eval scripts work
unchanged (metrics such as WER apply per-character rather than per-word for
Chinese text — eval.py's `compute_transcript_wer` currently splits on
whitespace, which is wrong for Chinese; switch to a CER-style character
split there before reporting metrics on this dataset).

Audio is assumed to already be provided per-user-turn (RealTalk-CN was
recorded from real conversations), so no split_audio step is included here
— verify once raw data lands.

Usage:
  python scripts/train/prepare_data_realtalk_cn.py \\
      --data data/raw_realtalk_cn/train.json \\
      --output data/realtalk_cn/train.jsonl
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
USER_SPEAKER_TAG = "user"
SYSTEM_SPEAKER_TAG = "system"
AUDIO_FILE_KEY = "audio"
# ---------------------------------------------------------------------------


def flatten_state(state: dict) -> dict:
    """RealTalk-CN state is assumed already flat ({domain: {slot: value}});
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
    parser.add_argument("--data", required=True, help="RealTalk-CN JSON file")
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
