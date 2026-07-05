"""Extract per-turn user audio from DSTC-11 Speech Aware Track HDF5 files.

Reference: Soltau et al., "DSTC-11: Speech Aware Task-Oriented Dialog
Modeling Track" (aclanthology.org/2023.dstc-1.25). Raw data + format spec
confirmed directly from the challenge's own index page:
https://storage.googleapis.com/gresearch/dstc11/dstc11_20221102a.html

Confirmed HDF5 layout (verified against the index page's own h5py example,
under "Data Format (h5p)"): each `{split}.{variant}.zip` contains one
subdirectory per TTS speaker (tts_verbatim: tpa/tpb/tpc/tpd; human variants
have a single directory of crowd-worker recordings), and each subdirectory
contains one HDF5 file per dialogue, e.g. `tpa/mul0016.hd5`. Inside that
file there is one group per **user** turn (system turns have no audio),
keyed by a string like:
  "tpe_line_nr: 4519 dialog_id: mul0016.json turn_id: 1"
Each group contains:
  - "audio": raw PCM, int16, 16 kHz (confirmed by the index page's example,
    which writes it out via `scipy.io.wavfile.write(path, 16000, audio_pcm)`)
  - "feat":  (T, 512) float32 speech-encoder features (unused here)
  - attrs "hyp" (ASR hypothesis text) and "align" (word-level alignment,
    unused here)

turn_id is a 1-indexed running counter over the whole dialogue (user and
system turns alternating, turn_id=1 is the dialogue-opening user turn), so
it lines up directly with prepare_data_dstc11.py's turn indexing.

Output structure (one WAV per user turn, matching the SpokenWOZ per-sample
audio convention so the generic train/infer scripts don't need changes):
  <output-dir>/
    mul0016_1.wav   # user turn_id=1
    mul0016_3.wav   # user turn_id=3
    ...

Usage (point --h5-dir at one already-unzipped speaker directory):
  python scripts/train/split_audio_dstc11.py \\
      --h5-dir data/raw_dstc11/train.tts-verbatim.2022-07-27/tpa \\
      --output-dir data/audio_dstc11/tts_verbatim_tpa/train
"""

import argparse
import re
from pathlib import Path

import h5py
import soundfile as sf


SAMPLE_RATE = 16000  # fixed by the challenge; not stored per-file

# --- CONFIG: keep in sync with prepare_data_dstc11.py ----------------------
GROUP_KEY_PATTERN = re.compile(r"turn_id:\s*(?P<turn_id>\d+)")
AUDIO_DATASET_KEY = "audio"
# ---------------------------------------------------------------------------


def extract_from_h5(h5_path: Path, output_dir: Path) -> int:
    dialogue_id = h5_path.stem.lower()
    saved = 0
    with h5py.File(h5_path, "r") as f:
        for group_key in f.keys():
            m = GROUP_KEY_PATTERN.search(group_key)
            if not m:
                print(f"  [WARN] {h5_path.name}: unrecognized group key '{group_key}', skipping")
                continue
            if AUDIO_DATASET_KEY not in f[group_key]:
                print(f"  [WARN] {h5_path.name}/{group_key}: no '{AUDIO_DATASET_KEY}' dataset, skipping")
                continue

            audio = f[group_key][AUDIO_DATASET_KEY][()]
            turn_id = m.group("turn_id")
            out_path = output_dir / f"{dialogue_id}_{turn_id}.wav"
            sf.write(str(out_path), audio, SAMPLE_RATE)
            saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", required=True, help="Directory of per-dialogue .hd5 files (one TTS speaker or the human-recording set)")
    parser.add_argument("--output-dir", required=True, help="Output directory for per-turn WAV files")
    parser.add_argument("--ext", default="hd5")
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
        total_saved += saved

    print(f"\nDone. {total_saved} sample files written to {output_dir}/ from {len(h5_files)} dialogues")


if __name__ == "__main__":
    main()
