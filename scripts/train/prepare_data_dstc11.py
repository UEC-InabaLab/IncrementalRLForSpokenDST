"""Convert DSTC-11 Speech Aware Track data to GRPO-format JSONL for
incremental DST training.

Reference: Soltau et al., "DSTC-11: Speech Aware Task-Oriented Dialog
Modeling Track" (aclanthology.org/2023.dstc-1.25). Raw data index:
https://storage.googleapis.com/gresearch/dstc11/dstc11_20221102a.html

Confirmed (from the raw-data index page) file layout per split:
  - train:  train.tts-verbatim.2022-07-27.{txt,zip}  (TTS only — no human speech for train)
  - dev:    dev-dstc11.tts-verbatim.2022-07-27.zip, dev-dstc11.human-verbatim.2022-09-29.zip,
            dev-dstc11.human-paraphrased.2022-11-02.zip, dev-dstc11.2022-1102.gold.json
  - test:   test-dstc11-tts-verbatim.2022-09-21.zip, test-dstc11.human-verbatim.2022-09-29.zip,
            test-dstc11.human-paraphrased.2022-10-17.zip, test-dstc11.2022-1102.gold.json

The gold JSON gives, per dialogue, an ordered list of system-turn records:
  {
    "DIALOGUE_ID": [
      {"response": "okay , any requirement .", "state": {"restaurant": {"food": "eatable"}}, "active_domains": ["restaurant"]},
      ...
    ]
  }
"response" is the system utterance text; "state" is the belief state
(already flat {domain: {slot: value}}, unlike SpokenWOZ's nested MultiWOZ
metadata) accumulated through the *preceding* user turn — i.e. entry k
plays the same role as SpokenWOZ's system-turn metadata at index 2k+1.

User-turn text/audio is not in the gold JSON; it lives in the per-variant
HDF5 files (see split_audio_dstc11.py), keyed by group name
f"{dialogue_id}_{turn_idx}", with the ASR hypothesis available as the
"hyp" attribute on each group. NOT YET VERIFIED against the actual
downloaded files — adjust the CONFIG constants below (and in
split_audio_dstc11.py) once inspected.

Turn ordering mirrors SpokenWOZ: user turn k pairs with gold[k] (the
system turn produced in response to it) as "prev_state", and gold[k+1] as
"curr_state" after the user's next reply. gold[0]["response"] is treated
as history's opening system line is NOT available for turn -1 (there is
no user turn before the dialogue's first system response in this scheme,
since MultiWOZ dialogues open with a user turn) — the very first user
utterance (k=0) is instead read from the HDF5 "hyp" attribute alone and
seeded as the first history line, matching SpokenWOZ's log[0].

Usage:
  python scripts/train/prepare_data_dstc11.py \\
      --gold data/raw_dstc11/train.tts-verbatim.2022-07-27.txt \\
      --h5-dir data/raw_dstc11/train.tts-verbatim.2022-07-27 \\
      --variant tts_verbatim \\
      --output data/dstc11/train.jsonl

  python scripts/train/prepare_data_dstc11.py \\
      --gold data/raw_dstc11/dev-dstc11.2022-1102.gold.json \\
      --h5-dir data/raw_dstc11/dev-dstc11.human-verbatim.2022-09-29 \\
      --variant human_verbatim \\
      --output data/dstc11/val.jsonl
"""

import argparse
import json
import re
from pathlib import Path

import h5py

from dst_common import (
    build_solution,
    build_user_message,
    load_system_prompt,
    write_jsonl,
)


SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "incremental.txt"

# --- CONFIG: keep in sync with split_audio_dstc11.py -----------------------
GROUP_KEY_PATTERN = re.compile(r"^(?P<dialogue_id>[A-Za-z0-9]+)_(?P<turn_idx>\d+)$")
HYP_ATTR_KEY = "hyp"
VARIANTS = ("tts_verbatim", "human_verbatim", "human_paraphrased")
# ---------------------------------------------------------------------------


def load_user_hyp_index(h5_dir: Path) -> dict[str, dict[int, str]]:
    """Scan every .h5 file in h5_dir and build {dialogue_id: {turn_idx: hyp_text}}."""
    index: dict[str, dict[int, str]] = {}
    for h5_path in sorted(h5_dir.glob("*.h5")):
        with h5py.File(h5_path, "r") as f:
            for group_key in f.keys():
                m = GROUP_KEY_PATTERN.match(group_key)
                if not m:
                    continue
                dialogue_id = m.group("dialogue_id")
                turn_idx = int(m.group("turn_idx"))
                hyp = f[group_key].attrs.get(HYP_ATTR_KEY)
                if hyp is None:
                    continue
                index.setdefault(dialogue_id, {})[turn_idx] = str(hyp).strip()
    return index


def clean_state(state: dict) -> dict:
    """Drop empty/falsy slot values; gold state is already flat."""
    flat: dict = {}
    for domain, slots in (state or {}).items():
        if not isinstance(slots, dict):
            continue
        kept = {slot: value for slot, value in slots.items() if value}
        if kept:
            flat[domain] = kept
    return flat


def process_dialogue(
    dialogue_id: str,
    gold_turns: list[dict],
    user_hyps: dict[int, str],
    system_prompt: str,
    variant: str,
) -> list[dict]:
    if len(gold_turns) < 2 or 0 not in user_hyps:
        return []

    samples: list[dict] = []
    history_lines: list[str] = [f"User: {user_hyps[0]}"]

    for k, entry in enumerate(gold_turns):
        if (k + 1) not in user_hyps:
            break  # no corresponding user reply recorded for this system turn

        sys_text = entry["response"].strip()
        user_text = user_hyps[k + 1]

        prev_state = clean_state(entry.get("state", {}))
        curr_state = (
            clean_state(gold_turns[k + 1].get("state", {}))
            if k + 1 < len(gold_turns)
            else prev_state
        )

        history_with_sys = list(history_lines) + [f"System: {sys_text}"]

        samples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_message(history_with_sys, prev_state)},
            ],
            "audios": [f"{dialogue_id}_{k + 1}.wav"],
            "solution": build_solution(sys_text, user_text, prev_state, curr_state),
            "belief_state": json.dumps(curr_state, ensure_ascii=False),
            "prev_belief_state": json.dumps(prev_state, ensure_ascii=False),
            "dialogue_id": dialogue_id,
            "turn_idx": k,
            "variant": variant,
        })

        history_lines.append(f"System: {sys_text}")
        history_lines.append(f"User: {user_text}")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="DSTC-11 gold state JSON file")
    parser.add_argument("--h5-dir", required=True, help="Directory of per-turn HDF5 files for this split+variant")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--system-prompt", default=None, help="Override system prompt file")
    args = parser.parse_args()

    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH, args.system_prompt)

    with open(args.gold, encoding="utf-8") as f:
        gold: dict = json.load(f)

    user_hyp_index = load_user_hyp_index(Path(args.h5_dir))

    samples: list[dict] = []
    skipped = 0
    for dialogue_id, gold_turns in gold.items():
        result = process_dialogue(
            dialogue_id, gold_turns, user_hyp_index.get(dialogue_id, {}), system_prompt, args.variant
        )
        if not result:
            skipped += 1
        samples.extend(result)

    write_jsonl(samples, args.output)

    print(f"Wrote {len(samples)} samples ({args.variant}) from {len(gold) - skipped} dialogues → {args.output}")
    if skipped:
        print(f"Skipped {skipped} dialogues (no matching audio/hyp found)")


if __name__ == "__main__":
    main()
