"""Convert DSTC-11 Speech Aware Track data to GRPO-format JSONL for
incremental DST training.

Reference: Soltau et al., "DSTC-11: Speech Aware Task-Oriented Dialog
Modeling Track" (aclanthology.org/2023.dstc-1.25). Raw data + format spec
confirmed directly from the challenge's own index page:
https://storage.googleapis.com/gresearch/dstc11/dstc11_20221102a.html

Three files are combined here, all confirmed against the real downloads:

1. Gold state file (e.g. dev-dstc11.2022-1102.gold.json):
     {"pmul1635": [{"hotel": {"area": "east", "stars": "4"}}, ...], ...}
   A list of already-flat {domain: {slot: value}} states, one entry per
   *user* turn, in order — entry i is the belief state as of user turn_id
   2i+1 (see turn_id below). Dialogue keys are lowercase, no ".json" suffix.

2. Mapping file (e.g. dev-dstc11.2022-07-27.txt), one line per turn:
     line_nr: 1 dialog_id: pmul1635.json turn_id: 2 text: agent: i can help you with that. state:
   turn_id is a 1-indexed running counter over the whole dialogue
   (turn_id=1 is the dialogue-opening *user* turn, turn_id=2 the first
   *agent*/system reply, etc.) — this is the only source for system-turn
   text, since system turns have no audio and the gold file has no text.
   The inline "state:" field here is redundant with the gold JSON (which
   is already parsed/nested) and is not used.

3. Per-dialogue HDF5 files (see split_audio_dstc11.py) — the only source
   for user-turn text, via each turn's ASR hypothesis ("hyp" attribute).
   The ASR hyp is used (rather than the mapping file's original written
   text) because for the human_paraphrased variant the actually-spoken
   words differ from the written MultiWOZ text.

Turn indexing mirrors SpokenWOZ/dst_common: for k = 0, 1, 2, ...
  - prev_state = gold[k]                              (state before this exchange)
  - curr_state = gold[k+1] (or gold[k] if last)        (state after the reply)
  - sys_text   = mapping text at turn_id = 2k+2
  - user_text  = HDF5 "hyp" at turn_id  = 2k+3
  - audio file = "{dialogue_id}_{2k+3}.wav" (see split_audio_dstc11.py)
The dialogue's opening user turn (turn_id=1) seeds the first history line,
using its own "hyp" if available.

Each sample also carries "sys_text" (this turn's system text) and
"opening_user_text" (the dialogue's turn_id=1 user text) verbatim, so that
infer.py's predicted (cascading) mode can inject them as ground truth
directly instead of depending on the model to reproduce them.

Usage:
  python scripts/train/prepare_data_dstc11.py \\
      --gold data/raw_dstc11/dev-dstc11.2022-1102.gold.json \\
      --mapping data/raw_dstc11/dev-dstc11.2022-07-27.txt \\
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

VARIANTS = ("tts_verbatim", "human_verbatim", "human_paraphrased")

# --- CONFIG: keep in sync with split_audio_dstc11.py -----------------------
MAPPING_LINE_PATTERN = re.compile(
    r"^line_nr:\s*\d+\s+dialog_id:\s*(?P<dialog_id>\S+)\s+turn_id:\s*(?P<turn_id>\d+)\s+text:\s*(?P<rest>.*)$"
)
H5_GROUP_TURN_ID_PATTERN = re.compile(r"turn_id:\s*(?P<turn_id>\d+)")
HYP_ATTR_KEY = "hyp"
# ---------------------------------------------------------------------------


def normalize_dialogue_id(dialog_id: str) -> str:
    return dialog_id[:-len(".json")] if dialog_id.endswith(".json") else dialog_id


def load_mapping(path: Path) -> dict[str, dict[int, dict[str, str]]]:
    """Parse the mapping .txt file into {dialogue_id: {turn_id: {"speaker", "text"}}}."""
    turns_by_dialogue: dict[str, dict[int, dict[str, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = MAPPING_LINE_PATTERN.match(line.rstrip("\n"))
            if not m:
                continue
            dialogue_id = normalize_dialogue_id(m.group("dialog_id"))
            turn_id = int(m.group("turn_id"))
            text_part = m.group("rest").rpartition(" state:")[0]
            speaker, _, utterance = text_part.partition(": ")
            turns_by_dialogue.setdefault(dialogue_id, {})[turn_id] = {
                "speaker": speaker.strip(),
                "text": utterance.strip(),
            }
    return turns_by_dialogue


def load_user_hyp_index(h5_dir: Path, ext: str = "hd5") -> dict[str, dict[int, str]]:
    """Scan every per-dialogue HDF5 file and build {dialogue_id: {turn_id: hyp_text}}."""
    index: dict[str, dict[int, str]] = {}
    for h5_path in sorted(h5_dir.glob(f"*.{ext}")):
        dialogue_id = h5_path.stem.lower()
        with h5py.File(h5_path, "r") as f:
            for group_key in f.keys():
                m = H5_GROUP_TURN_ID_PATTERN.search(group_key)
                if not m:
                    continue
                hyp = f[group_key].attrs.get(HYP_ATTR_KEY)
                if hyp is None:
                    continue
                index.setdefault(dialogue_id, {})[int(m.group("turn_id"))] = str(hyp).strip()
    return index


def process_dialogue(
    dialogue_id: str,
    gold_states: list[dict],
    turns_by_id: dict[int, dict[str, str]],
    user_hyps: dict[int, str],
    system_prompt: str,
    variant: str,
) -> list[dict]:
    if not gold_states or 1 not in user_hyps:
        return []

    samples: list[dict] = []
    opening_user_text = user_hyps[1]
    history_lines: list[str] = [f"User: {opening_user_text}"]

    for k, prev_state in enumerate(gold_states):
        sys_turn_id = 2 * k + 2
        user_turn_id = 2 * k + 3

        if sys_turn_id not in turns_by_id or user_turn_id not in user_hyps:
            break  # no system text or no recorded user audio for this step

        sys_text = turns_by_id[sys_turn_id]["text"]
        user_text = user_hyps[user_turn_id]
        curr_state = gold_states[k + 1] if k + 1 < len(gold_states) else prev_state

        history_with_sys = list(history_lines) + [f"System: {sys_text}"]

        samples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_message(history_with_sys, prev_state)},
            ],
            "audios": [f"{dialogue_id}_{user_turn_id}.wav"],
            "solution": build_solution(user_text, prev_state, curr_state),
            "belief_state": json.dumps(curr_state, ensure_ascii=False),
            "prev_belief_state": json.dumps(prev_state, ensure_ascii=False),
            "dialogue_id": dialogue_id,
            "turn_idx": k,
            "variant": variant,
            "sys_text": sys_text,
            "opening_user_text": opening_user_text,
        })

        history_lines.append(f"System: {sys_text}")
        history_lines.append(f"User: {user_text}")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="DSTC-11 gold state JSON file")
    parser.add_argument("--mapping", required=True, help="DSTC-11 mapping .txt file (system/user turn text)")
    parser.add_argument("--h5-dir", required=True, help="Directory of per-dialogue .hd5 files for this split+variant")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--system-prompt", default=None, help="Override system prompt file")
    args = parser.parse_args()

    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH, args.system_prompt)

    with open(args.gold, encoding="utf-8") as f:
        gold: dict = json.load(f)

    turns_by_dialogue = load_mapping(Path(args.mapping))
    user_hyp_index = load_user_hyp_index(Path(args.h5_dir))

    samples: list[dict] = []
    skipped = 0
    for dialogue_id, gold_states in gold.items():
        dialogue_id = dialogue_id.lower()
        result = process_dialogue(
            dialogue_id,
            gold_states,
            turns_by_dialogue.get(dialogue_id, {}),
            user_hyp_index.get(dialogue_id, {}),
            system_prompt,
            args.variant,
        )
        if not result:
            skipped += 1
        samples.extend(result)

    write_jsonl(samples, args.output)

    print(f"Wrote {len(samples)} samples ({args.variant}) from {len(gold) - skipped} dialogues → {args.output}")
    if skipped:
        print(f"Skipped {skipped} dialogues (no matching mapping text / audio)")


if __name__ == "__main__":
    main()
