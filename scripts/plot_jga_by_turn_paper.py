"""Plot per-turn JGA for paper: two separate figures.

Figure 1 (B): Binned JGA bar chart — "JGA by Turn Range"
Figure 2 (C): Cumulative JGA line chart
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> Optional[str]:
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return m.group(1).strip() if m else None


_OP_PATTERN = re.compile(r'(set|update|delete)\(([^)]+)\)')


def parse_diff_ops(answer_text: str) -> list:
    if not answer_text or not answer_text.strip():
        return []
    ops = []
    for line in answer_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        m = _OP_PATTERN.match(line)
        if not m:
            continue
        op_type = m.group(1)
        content = m.group(2)
        if op_type == 'delete':
            parts = content.split('.', 1)
            if len(parts) == 2:
                ops.append((op_type, parts[0], parts[1], None))
        else:
            if '=' in content:
                key, value = content.split('=', 1)
                parts = key.split('.', 1)
                if len(parts) == 2:
                    ops.append((op_type, parts[0], parts[1], value))
    return ops


def apply_diff_ops(state: Dict, ops: list) -> Dict:
    new_state = {}
    for domain, slots in state.items():
        new_state[domain] = dict(slots) if isinstance(slots, dict) else slots
    for op, domain, slot, value in ops:
        if op in ('set', 'update'):
            if domain not in new_state:
                new_state[domain] = {}
            new_state[domain][slot] = value
        elif op == 'delete':
            if domain in new_state and slot in new_state[domain]:
                del new_state[domain][slot]
                if not new_state[domain]:
                    del new_state[domain]
    return new_state


def get_pred_state(record: dict) -> dict:
    prediction = record['prediction']
    pred_answer = extract_answer(prediction)
    if 'input_belief_state' in record:
        input_state_str = record.get('input_belief_state', '{}')
        try:
            input_state = json.loads(input_state_str)
            if not isinstance(input_state, dict):
                input_state = {}
        except (json.JSONDecodeError, TypeError):
            input_state = {}
        if pred_answer is not None:
            ops = parse_diff_ops(pred_answer)
            return apply_diff_ops(input_state, ops)
        else:
            return input_state
    else:
        if pred_answer is not None:
            try:
                state = json.loads(pred_answer) if pred_answer.strip() else {}
                return state if isinstance(state, dict) else {}
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}


def get_gold_state(record: dict) -> dict:
    gold_state_str = record.get('belief_state', '{}')
    try:
        gold_state = json.loads(gold_state_str)
        return gold_state if isinstance(gold_state, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def load_records(filepath: str) -> List[dict]:
    lines = Path(filepath).read_text().strip().split('\n')
    records = []
    for line in lines:
        r = json.loads(line)
        r['_pred_state'] = get_pred_state(r)
        r['_gold_state'] = get_gold_state(r)
        r['_correct'] = r['_pred_state'] == r['_gold_state']
        records.append(r)
    return records


# ---------------------------------------------------------------------------
# Binned JGA
# ---------------------------------------------------------------------------

def compute_binned_jga(records: List[dict], bin_edges: List[int]):
    bins = []
    for i in range(len(bin_edges)):
        lo = bin_edges[i]
        hi = bin_edges[i + 1] - 1 if i + 1 < len(bin_edges) else float('inf')
        bins.append((lo, hi))

    bin_correct = [0] * len(bins)
    bin_total = [0] * len(bins)

    for r in records:
        t = r['turn_idx']
        for bi, (lo, hi) in enumerate(bins):
            if lo <= t <= hi:
                bin_total[bi] += 1
                if r['_correct']:
                    bin_correct[bi] += 1
                break

    labels = []
    jga = []
    counts = []
    for bi, (lo, hi) in enumerate(bins):
        if bin_total[bi] == 0:
            continue
        if hi == float('inf'):
            labels.append(f'{lo}+')
        else:
            labels.append(f'{lo}-{hi}')
        jga.append(bin_correct[bi] / bin_total[bi])
        counts.append(bin_total[bi])

    return labels, jga, counts


# ---------------------------------------------------------------------------
# Cumulative JGA
# ---------------------------------------------------------------------------

def compute_cumulative_jga(records: List[dict], min_samples: int = 100):
    dialogues = defaultdict(list)
    for r in records:
        dialogues[r['dialogue_id']].append(r)

    first_error = {}
    max_turn_per_dial = {}
    for dial_id, turns in dialogues.items():
        turns.sort(key=lambda x: x['turn_idx'])
        max_turn_per_dial[dial_id] = turns[-1]['turn_idx']
        err_turn = float('inf')
        for r in turns:
            if not r['_correct']:
                err_turn = r['turn_idx']
                break
        first_error[dial_id] = err_turn

    all_turns = sorted(set(r['turn_idx'] for r in records))

    turns_out = []
    jga_out = []
    counts_out = []

    for t in all_turns:
        eligible = [d for d, mt in max_turn_per_dial.items() if mt >= t]
        if len(eligible) < min_samples:
            continue
        correct = sum(1 for d in eligible if first_error[d] > t)
        turns_out.append(t)
        jga_out.append(correct / len(eligible))
        counts_out.append(len(eligible))

    return turns_out, jga_out, counts_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Order: Full-State, Incremental (SFT), Incremental + GRPO
FILE_ORDER = [
    'output/predictions_full.jsonl',
    'output/predictions_sft.jsonl',
    'output/predictions_grpo.jsonl',
]

LABELS = {
    'predictions_full.jsonl': 'Full-State',
    'predictions_sft.jsonl': 'Incremental',
    'predictions_grpo.jsonl': 'Incremental + GRPO',
}

COLORS = {
    'predictions_full.jsonl': '#2196F3',
    'predictions_sft.jsonl': '#4CAF50',
    'predictions_grpo.jsonl': '#FF5722',
}

MARKERS = {
    'predictions_full.jsonl': 'o',
    'predictions_sft.jsonl': '^',
    'predictions_grpo.jsonl': 's',
}


def main():
    MIN_SAMPLES = 100
    BIN_EDGES = [1, 6, 11, 16, 21, 31]

    # Load data in specified order
    all_data = {}
    for filepath in FILE_ORDER:
        if not Path(filepath).exists():
            print(f"[WARN] {filepath} not found, skipping.")
            continue
        fname = Path(filepath).name
        all_data[fname] = load_records(filepath)

    # ==================================================================
    # Figure B: Binned JGA bar chart
    # ==================================================================
    fig_b, ax_b = plt.subplots(figsize=(8, 5))
    n_models = len(all_data)
    bar_width = 0.8 / n_models
    bin_labels_ref = None

    for i, (fname, records) in enumerate(all_data.items()):
        label = LABELS[fname]
        bin_labels, jga, counts = compute_binned_jga(records, BIN_EDGES)
        if bin_labels_ref is None:
            bin_labels_ref = bin_labels
        x = list(range(len(bin_labels)))
        offsets = [xi + (i - n_models / 2 + 0.5) * bar_width for xi in x]
        bars = ax_b.bar(offsets, jga, width=bar_width,
                        color=COLORS[fname], label=label, alpha=0.85,
                        edgecolor='white', linewidth=0.5)

        print(f"\n[Binned] {label}:")
        print(f"  {'Bin':>8} {'JGA':>8} {'N':>6}")
        for bl, j, n in zip(bin_labels, jga, counts):
            print(f"  {bl:>8} {j:>8.4f} {n:>6}")

    ax_b.set_xticks(range(len(bin_labels_ref)))
    ax_b.set_xticklabels(bin_labels_ref, fontsize=11)
    ax_b.set_xlabel('Turn Range', fontsize=13)
    ax_b.set_ylabel('JGA', fontsize=13)
    ax_b.legend(fontsize=11, loc='upper right')
    ax_b.grid(True, alpha=0.3, axis='y')
    ax_b.set_ylim(0, 1.0)
    ax_b.tick_params(axis='y', labelsize=11)
    for spine in ax_b.spines.values():
        spine.set_visible(False)

    fig_b.tight_layout()
    out_b = 'output/jga_binned.png'
    fig_b.savefig(out_b, dpi=200)
    fig_b.savefig(out_b.replace('.png', '.pdf'))
    print(f"\nFigure B saved to {out_b}")

    # ==================================================================
    # Figure C: Cumulative JGA line chart
    # ==================================================================
    fig_c, ax_c = plt.subplots(figsize=(8, 5))

    for fname, records in all_data.items():
        label = LABELS[fname]
        turns, jga, counts = compute_cumulative_jga(records, min_samples=MIN_SAMPLES)
        ax_c.plot(turns, jga,
                  marker=MARKERS[fname],
                  color=COLORS[fname],
                  label=label, linewidth=2, markersize=5, alpha=0.85)

        print(f"\n[Cumulative, N>={MIN_SAMPLES}] {label}:")
        print(f"  {'Turn':>5} {'CumJGA':>8} {'N_dial':>7}")
        for t, j, n in zip(turns, jga, counts):
            print(f"  {t:>5} {j:>8.4f} {n:>7}")

    ax_c.set_xlabel('Turn Index', fontsize=13)
    ax_c.set_ylabel('Cumulative JGA', fontsize=13)
    ax_c.legend(fontsize=11)
    ax_c.grid(True, alpha=0.3)
    ax_c.set_ylim(0, 1.05)
    ax_c.tick_params(axis='both', labelsize=11)
    for spine in ax_c.spines.values():
        spine.set_visible(False)

    fig_c.tight_layout()
    out_c = 'output/jga_cumulative.png'
    fig_c.savefig(out_c, dpi=200)
    fig_c.savefig(out_c.replace('.png', '.pdf'))
    print(f"Figure C saved to {out_c}")


if __name__ == '__main__':
    main()
