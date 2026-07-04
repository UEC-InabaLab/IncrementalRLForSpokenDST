# Incremental End-to-End Spoken Dialogue State Tracking with a Multimodal LLM and Reinforcement Learning

Code for our Interspeech 2026 paper.

> **Incremental End-to-End Spoken Dialogue State Tracking with a Multimodal LLM and Reinforcement Learning**
> Tomoya Higuchi
> *Interspeech 2026*

## Overview

We train [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) to perform **incremental Dialogue State Tracking (DST)** directly from audio — without a separate ASR step.

Instead of outputting the full belief state at every turn, the model outputs **diff operations** (set / update / delete) relative to the previous state:

```
<transcript>
System: do you have a price preference .
User: i'd like something cheap .
</transcript>
<answer>set(restaurant.pricerange=cheap)</answer>
```

Training uses a two-stage pipeline:
1. **SFT** — supervised fine-tuning with QLoRA (rank 64, α 128, 4-bit NF4)
2. **GRPO** — reinforcement learning with a composite reward (transcript WER + diff F1 + exact match + format)

Evaluated on [SpokenWOZ](https://github.com/ZekangLi/SpokenWOZ).

## Repository structure

```
.
├── prompts/
│   ├── incremental.txt       # system prompt for incremental DST
│   └── fullstate.txt         # system prompt for full-state DST (baseline)
├── scripts/
│   ├── train/
│   │   ├── train_sft.sh                          # Stage 1: SFT
│   │   ├── train_grpo.sh                         # Stage 2: GRPO (main)
│   │   ├── train_grpo_ablation_no_transcript.sh  # ablation: no WER reward
│   │   ├── train_sft_fullstate.sh                # full-state baseline SFT
│   │   ├── train_grpo_fullstate.sh               # full-state baseline GRPO
│   │   ├── dst_common.py             # shared diff-op / JSONL-building helpers (all datasets)
│   │   ├── split_audio.py            # extract per-sample user-turn audio from SpokenWOZ WAVs
│   │   ├── prepare_data.py           # convert SpokenWOZ raw data → GRPO JSONL
│   │   ├── split_audio_dstc11.py     # extract per-sample user-turn audio from DSTC-11 WAVs
│   │   ├── prepare_data_dstc11.py    # convert DSTC-11 spoken-MultiWOZ raw data → GRPO JSONL
│   │   ├── prepare_data_spokentod.py    # convert SpokenTOD raw data → GRPO JSONL (speculative, see Data section)
│   │   ├── prepare_data_realtalk_cn.py  # convert RealTalk-CN raw data → GRPO JSONL (speculative, see Data section)
│   │   ├── convert_to_sft.py         # convert GRPO-format data → SFT format
│   │   ├── prepare_fullstate_data.py # convert incremental data → full-state format
│   │   └── sample_val.py             # sample a small validation subset
│   ├── infer/
│   │   ├── infer.py / infer_fullstate.py         # vLLM batch inference
│   │   ├── infer_oracle.sh / infer_predicted.sh  # oracle / cascading inference
│   │   └── infer_fullstate_oracle.sh / infer_fullstate_predicted.sh
│   └── eval/
│       ├── eval.py            # compute WER / JGA / Slot F1 (incremental)
│       ├── eval_fullstate.py  # compute WER / JGA / Slot F1 (full-state)
│       └── plot_jga_by_turn.py
├── src/
│   └── reward.py   # ms-swift ORM plugin: GRPO reward functions
├── tests/
└── pyproject.toml
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

GPU-specific packages (`flash-attn`, `vllm`, etc.) are listed in `pyproject.toml` and resolved by `uv sync` for the target environment.

## Data

Download SpokenWOZ from [spokenwoz.github.io](https://spokenwoz.github.io/) and place the raw files under `data/raw/`:

```
data/raw/
  train.json
  val.json
  test.json
  audio/          # WAV files referenced by each JSON
```

Each model input is the text dialogue history plus the audio of the latest user
turn. SpokenWOZ ships one WAV per dialogue, so first extract per-sample
user-turn audio (named `{dialogue_id}_{sys_idx}_{user_idx}.wav`) using the
word-level timing in the JSON:

```bash
python scripts/train/split_audio.py --data data/raw/train.json --audio-dir data/raw/audio --output-dir data/audio/train
python scripts/train/split_audio.py --data data/raw/val.json   --audio-dir data/raw/audio --output-dir data/audio/val
python scripts/train/split_audio.py --data data/raw/test.json  --audio-dir data/raw/audio --output-dir data/audio/test
```

Then convert to GRPO JSONL format. Each sample carries the diff-operation target
(set / update / delete) derived from the belief state on the *next* system turn,
plus the previous state from the current system turn:

```bash
python scripts/train/prepare_data.py --data data/raw/train.json --output data/train.jsonl
python scripts/train/prepare_data.py --data data/raw/val.json   --output data/val.jsonl
python scripts/train/prepare_data.py --data data/raw/test.json  --output data/test.jsonl
```

Point training/inference at the split audio with `--audio-base-dir data/audio/<split>`.

Full-state baseline data is auto-generated from the incremental data by the training scripts.

### Additional datasets

All data-prep scripts produce the same GRPO JSONL schema (`messages` /
`audios` / `solution` / `belief_state` / `prev_belief_state`), so
`train_sft.sh`, `train_grpo.sh`, `infer_oracle.sh` / `infer_predicted.sh`,
and `eval.py` work unchanged for any dataset below — just point their data
env vars (`GRPO_TRAIN_DATA` / `TRAIN_DATA` / `VAL_DATA` / `AUDIO_BASE_DIR` /
`OUTPUT_DIR` / `WANDB_PROJECT`) at the dataset's directory. Diff-op and
JSONL-building logic shared across datasets lives in
`scripts/train/dst_common.py`.

| Dataset | Status | Prep scripts |
|---|---|---|
| SpokenWOZ | in use | `split_audio.py`, `prepare_data.py` |
| [DSTC-11 Spoken MultiWOZ](https://aclanthology.org/2023.dstc-1.25/) | schema assumed from MultiWOZ, not yet verified against raw files | `split_audio_dstc11.py`, `prepare_data_dstc11.py` |
| SpokenTOD ([arXiv:2603.16783](https://arxiv.org/html/2603.16783)) | speculative — no public download link/schema in the paper yet | `prepare_data_spokentod.py` |
| RealTalk-CN ([arXiv:2508.10015](https://arxiv.org/html/2508.10015v1)) | speculative — paper says data "will be made available", not yet public | `prepare_data_realtalk_cn.py` |

#### DSTC-11 Spoken MultiWOZ

Re-releases MultiWOZ 2.1 dialogues with three spoken user-turn variants
(`tts_verbatim`, `human_verbatim`, `human_paraphrased`) over the same
belief-state schema as SpokenWOZ. The raw-file layout in the script header
is a best-effort guess (word-level timing per dialogue WAV, mirroring
SpokenWOZ) — confirm against the actual download and adjust the `CONFIG`
constants at the top of each script if field names differ.

```bash
python scripts/train/split_audio_dstc11.py \
    --data data/raw_dstc11/train.json --audio-dir data/raw_dstc11/audio \
    --variant human_verbatim --output-dir data/audio_dstc11/human_verbatim/train

python scripts/train/prepare_data_dstc11.py \
    --data data/raw_dstc11/train.json --variant human_verbatim \
    --output data/dstc11/train.jsonl

GRPO_TRAIN_DATA=data/dstc11/train.jsonl GRPO_VAL_DATA=data/dstc11/val.jsonl \
OUTPUT_DIR=output/sft_dstc11 WANDB_PROJECT=qwenomni-sft-dstc11 \
bash scripts/train/train_sft.sh
```

#### SpokenTOD / RealTalk-CN

Neither paper publishes a confirmed schema or download URL as of writing,
so `prepare_data_spokentod.py` / `prepare_data_realtalk_cn.py` are scaffolds
based on the papers' descriptions (flat per-turn domain/slot state rather
than MultiWOZ's nested ontology), with all assumed field names collected in
a `CONFIG` block at the top of each file. Once the raw data is obtained:

1. Inspect a sample dialogue and update the `CONFIG` constants (speaker
   tags, state nesting, audio filename key) to match.
2. If audio ships one-file-per-dialogue rather than pre-split per turn, add
   a `split_audio_<dataset>.py` analogous to `split_audio_dstc11.py`.
3. For RealTalk-CN (Chinese), `scripts/eval/eval.py`'s `compute_transcript_wer`
   splits on whitespace (word-level WER); switch to a character split for a
   correct CER on Chinese transcripts before reporting metrics.

```bash
python scripts/train/prepare_data_spokentod.py \
    --data data/raw_spokentod/train.json --output data/spokentod/train.jsonl

python scripts/train/prepare_data_realtalk_cn.py \
    --data data/raw_realtalk_cn/train.json --output data/realtalk_cn/train.jsonl
```

## Training

### Stage 1 — SFT

```bash
bash scripts/train/train_sft.sh
```

Converts GRPO-format data to SFT format automatically, then runs QLoRA fine-tuning.

### Stage 2 — GRPO

```bash
# Start from SFT checkpoint (set SFT_CHECKPOINT)
SFT_CHECKPOINT=output/sft_incremental_dst/checkpoint-xxxx \
bash scripts/train/train_grpo.sh
```

Uses the custom reward plugin at `src/reward.py` via ms-swift's `--external_plugins` interface.

Default: 8 GPUs (6 for training, 2 for vLLM inference during rollout).

### Ablation (no transcript reward)

```bash
bash scripts/train/train_grpo_ablation_no_transcript.sh
```

## Inference

```bash
# Oracle mode (ground-truth dialogue history)
ADAPTER=output/grpo_incremental_dst/checkpoint-xxxx \
bash scripts/infer/infer_oracle.sh

# Predicted mode (model's own transcript cascades across turns)
ADAPTER=output/grpo_incremental_dst/checkpoint-xxxx \
bash scripts/infer/infer_predicted.sh
```

Both scripts run inference then evaluation automatically and write results to `output/`.

## Evaluation

```bash
python scripts/eval/eval.py \
    --input output/vllm_inference_results/oracle/predictions.jsonl \
    --output output/vllm_inference_results/oracle/metrics.json
```

Metrics: transcript WER, Joint Goal Accuracy (JGA), Slot F1.