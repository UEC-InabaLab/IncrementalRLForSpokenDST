"""Extract per-sample user-turn audio for the DSTC-11 spoken-MultiWOZ track.

Mirrors split_audio.py (SpokenWOZ). NOTE ON RAW FORMAT: assumes one WAV per
dialogue per variant plus word-level timing on each user turn, matching the
SpokenWOZ convention this track is built on. Not yet verified against the
actual downloaded files — adjust `AUDIO_SUBDIR_BY_VARIANT` / word-timing key
names below once confirmed.

Expected input structure (one audio subdir per variant):
  <audio-dir>/<variant>/
    MUL0001.wav
    MUL0002.wav
    ...

Output structure:
  <output-dir>/
    MUL0001_1_2.wav
    MUL0001_3_4.wav
    ...

Usage:
  python scripts/train/split_audio_dstc11.py \\
      --data      data/raw_dstc11/train.json \\
      --audio-dir data/raw_dstc11/audio \\
      --variant   human_verbatim \\
      --output-dir data/audio_dstc11/human_verbatim/train
"""

import argparse
import json
from pathlib import Path

import soundfile as sf


PADDING_MS = 100

VARIANTS = ("tts_verbatim", "human_verbatim", "human_paraphrased")
USER_TEXT_KEY = "asr_text"


def extract_samples(
    dialogue_id: str,
    log: list[dict],
    audio_path: Path,
    output_dir: Path,
    variant: str,
) -> int:
    audio, sr = sf.read(str(audio_path), always_2d=False)
    total_samples = len(audio) if audio.ndim == 1 else audio.shape[0]

    saved = 0
    k = 0
    while True:
        sys_idx = 2 * k + 1
        user_idx = 2 * k + 2

        if user_idx >= len(log):
            break

        sys_turn = log[sys_idx]
        user_turn = log[user_idx]

        if sys_turn.get("tag") != "system" or user_turn.get("tag") != "user":
            k += 1
            continue

        per_variant_words = user_turn.get(USER_TEXT_KEY, {})
        words = (
            per_variant_words.get(variant, {}).get("words", [])
            if isinstance(per_variant_words, dict)
            else []
        ) or user_turn.get("words", [])

        if not words:
            print(f"  [WARN] {dialogue_id} user turn {user_idx}: no word timing, skipping")
            k += 1
            continue

        begin_ms = max(0, words[0]["BeginTime"] - PADDING_MS)
        end_ms = words[-1]["EndTime"] + PADDING_MS

        begin_sample = int(begin_ms * sr / 1000)
        end_sample = min(total_samples, int(end_ms * sr / 1000))

        if begin_sample >= end_sample:
            print(f"  [WARN] {dialogue_id} user turn {user_idx}: empty segment, skipping")
            k += 1
            continue

        segment = audio[begin_sample:end_sample]
        out_path = output_dir / f"{dialogue_id}_{sys_idx}_{user_idx}.wav"
        sf.write(str(out_path), segment, sr)
        saved += 1
        k += 1

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="DSTC-11 spoken-MultiWOZ JSON file")
    parser.add_argument("--audio-dir", required=True, help="Directory with per-dialogue WAV files")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--output-dir", required=True, help="Output directory for per-sample WAV files")
    parser.add_argument("--ext", default="wav")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir) / args.variant
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data, encoding="utf-8") as f:
        data: dict = json.load(f)

    total_saved = 0
    skipped_dialogues = 0

    for dialogue_id, dialogue in data.items():
        log = dialogue.get("log", [])
        audio_path = audio_dir / f"{dialogue_id}.{args.ext}"

        if not audio_path.exists():
            print(f"[WARN] Audio not found: {audio_path}, skipping dialogue")
            skipped_dialogues += 1
            continue

        saved = extract_samples(dialogue_id, log, audio_path, output_dir, args.variant)
        total_saved += saved

    print(f"\nDone. {total_saved} sample files written to {output_dir}/")
    if skipped_dialogues:
        print(f"Skipped {skipped_dialogues} dialogues (audio file not found)")


if __name__ == "__main__":
    main()
