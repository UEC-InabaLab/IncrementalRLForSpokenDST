"""Extract per-turn user audio from DSTC-11 Speech Aware Track HDF5 files.

Reference: Soltau et al., "DSTC-11: Speech Aware Task-Oriented Dialog
Modeling Track" (aclanthology.org/2023.dstc-1.25). Raw data index:
https://storage.googleapis.com/gresearch/dstc11/dstc11_20221102a.html

Unlike SpokenWOZ (one WAV per dialogue, sliced by word timing), DSTC-11
ships audio pre-split per user turn inside HDF5 files, one archive per
"variant":
  - tts_verbatim:       train + dev + test (4 TTS speaker dirs: tpa/tpb/tpc/tpd)
  - human_verbatim:     dev + test only (no human speech for train)
  - human_paraphrased:  test only

Each user-turn group in the HDF5 file is documented (per the challenge
description) to contain:
  - "audio": raw PCM waveform
  - "feat":  512-dim speech-encoder features (unused here; we re-derive
             audio directly for the Qwen2.5-Omni audio tower instead)
  - attrs:   "hyp" (ASR hypothesis text), "align" (word alignment)

NOT YET VERIFIED AGAINST THE ACTUAL DOWNLOADED FILES: the exact HDF5 group
key naming (assumed f"{dialogue_id}_{turn_idx}" below) and the sample rate
(assumed 16 kHz, matching the TTS/ASR pipeline described in the paper).
Adjust the CONFIG constants once the real archives are unzipped and
inspected with `h5py.File(path).visit(print)`.

Output structure (matches the SpokenWOZ per-sample audio convention, so the
generic train/infer scripts don't need changes):
  <output-dir>/
    MUL0001_0.wav   # user turn k=0
    MUL0001_1.wav   # user turn k=1
    ...

Usage:
  python scripts/train/split_audio_dstc11.py \\
      --h5-dir data/raw_dstc11/train.tts-verbatim.2022-07-27 \\
      --output-dir data/audio_dstc11/tts_verbatim/train
"""

import argparse
import re
from pathlib import Path

import h5py
import soundfile as sf


# --- CONFIG: adjust once the real HDF5 files are inspected -----------------
GROUP_KEY_PATTERN = re.compile(r"^(?P<dialogue_id>[A-Za-z0-9]+)_(?P<turn_idx>\d+)$")
AUDIO_DATASET_KEY = "audio"
SAMPLE_RATE_ATTR = "sample_rate"
DEFAULT_SAMPLE_RATE = 16000
# ---------------------------------------------------------------------------


def extract_from_h5(h5_path: Path, output_dir: Path) -> int:
    saved = 0
    with h5py.File(h5_path, "r") as f:
        for group_key in f.keys():
            m = GROUP_KEY_PATTERN.match(group_key)
            if not m:
                print(f"  [WARN] {h5_path.name}: unrecognized group key '{group_key}', skipping")
                continue

            group = f[group_key]
            if AUDIO_DATASET_KEY not in group:
                print(f"  [WARN] {h5_path.name}/{group_key}: no '{AUDIO_DATASET_KEY}' dataset, skipping")
                continue

            audio = group[AUDIO_DATASET_KEY][()]
            sr = int(group.attrs.get(SAMPLE_RATE_ATTR, DEFAULT_SAMPLE_RATE))

            dialogue_id = m.group("dialogue_id")
            turn_idx = m.group("turn_idx")
            out_path = output_dir / f"{dialogue_id}_{turn_idx}.wav"
            sf.write(str(out_path), audio, sr)
            saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", required=True, help="Directory containing DSTC-11 .h5 files (one per TTS speaker, or per split)")
    parser.add_argument("--output-dir", required=True, help="Output directory for per-turn WAV files")
    parser.add_argument("--ext", default="h5")
    args = parser.parse_args()

    h5_dir = Path(args.h5_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(h5_dir.glob(f"*.{args.ext}"))
    if not h5_files:
        print(f"[WARN] No .{args.ext} files found in {h5_dir}")

    total_saved = 0
    for h5_path in h5_files:
        saved = extract_from_h5(h5_path, output_dir)
        print(f"  {h5_path.name}: {saved} turns extracted")
        total_saved += saved

    print(f"\nDone. {total_saved} sample files written to {output_dir}/")


if __name__ == "__main__":
    main()
