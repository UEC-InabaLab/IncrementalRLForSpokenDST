"""Split per-dialogue audio files into per-turn WAV segments.

SpokenWOZ provides one WAV file per dialogue. Each turn in the JSON log has
word-level timing (BeginTime / EndTime in milliseconds). This script extracts
each turn's audio segment and saves it as an individual file.

Expected input structure:
  <audio-dir>/
    MUL0001.wav
    MUL0002.wav
    ...

Output structure:
  <output-dir>/
    MUL0001/
      turn_0.wav   # log[0]
      turn_1.wav   # log[1]
      ...
    MUL0002/
      ...

Usage:
  python scripts/train/split_audio.py \\
      --data     data/raw/train_v1.0.json \\
      --audio-dir data/raw/audio \\
      --output-dir data/audio/train

  python scripts/train/split_audio.py \\
      --data     data/raw/test_v1.0.json \\
      --audio-dir data/raw/audio \\
      --output-dir data/audio/test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


PADDING_MS = 100  # silence padding added before/after each turn


def split_dialogue(
    dialogue_id: str,
    log: list[dict],
    audio_path: Path,
    output_dir: Path,
) -> int:
    """Extract per-turn audio segments for one dialogue.

    Returns the number of turns successfully saved.
    """
    audio, sr = sf.read(str(audio_path), always_2d=False)
    total_samples = len(audio) if audio.ndim == 1 else audio.shape[0]

    out_dialogue_dir = output_dir / dialogue_id
    out_dialogue_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for turn_idx, turn in enumerate(log):
        words = turn.get("words", [])
        if not words:
            print(f"  [WARN] {dialogue_id}/turn_{turn_idx}: no word timing, skipping")
            continue

        begin_ms = max(0, words[0]["BeginTime"] - PADDING_MS)
        end_ms = words[-1]["EndTime"] + PADDING_MS

        begin_sample = int(begin_ms * sr / 1000)
        end_sample = min(total_samples, int(end_ms * sr / 1000))

        if begin_sample >= end_sample:
            print(f"  [WARN] {dialogue_id}/turn_{turn_idx}: empty segment, skipping")
            continue

        segment = audio[begin_sample:end_sample]
        out_path = out_dialogue_dir / f"turn_{turn_idx}.wav"
        sf.write(str(out_path), segment, sr)
        saved += 1

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True, help="SpokenWOZ JSON file")
    parser.add_argument("--audio-dir",   required=True, help="Directory with per-dialogue WAV files")
    parser.add_argument("--output-dir",  required=True, help="Output directory for per-turn WAV files")
    parser.add_argument("--ext",         default="wav",  help="Audio file extension (default: wav)")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data, encoding="utf-8") as f:
        data: dict = json.load(f)

    total_turns = 0
    skipped_dialogues = 0

    for dialogue_id, dialogue in data.items():
        log = dialogue.get("log", [])
        audio_path = audio_dir / f"{dialogue_id}.{args.ext}"

        if not audio_path.exists():
            print(f"[WARN] Audio not found: {audio_path}, skipping dialogue")
            skipped_dialogues += 1
            continue

        saved = split_dialogue(dialogue_id, log, audio_path, output_dir)
        total_turns += saved

    print(f"\nDone. {total_turns} turn files written to {output_dir}/")
    if skipped_dialogues:
        print(f"Skipped {skipped_dialogues} dialogues (audio file not found)")


if __name__ == "__main__":
    main()
