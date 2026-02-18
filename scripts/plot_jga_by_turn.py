"""Plot per-turn JGA for prediction files."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def compute_jga_by_turn(filepath: str) -> dict:
    """Returns {turn_idx: (correct, total)} for each turn."""
    lines = Path(filepath).read_text().strip().split('\n')
    turn_stats = defaultdict(lambda: [0, 0])  # [correct, total]

    for line in lines:
        record = json.loads(line)
        prediction = record['prediction']
        turn_idx = record['turn_idx']
        gold_state_str = record.get('belief_state', '{}')

        try:
            gold_state = json.loads(gold_state_str)
            if not isinstance(gold_state, dict):
                gold_state = {}
        except (json.JSONDecodeError, TypeError):
            gold_state = {}

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
                pred_state = apply_diff_ops(input_state, ops)
            else:
                pred_state = input_state
        else:
            if pred_answer is not None:
                try:
                    pred_state = json.loads(pred_answer) if pred_answer.strip() else {}
                    if not isinstance(pred_state, dict):
                        pred_state = {}
                except (json.JSONDecodeError, TypeError):
                    pred_state = {}
            else:
                pred_state = {}

        turn_stats[turn_idx][1] += 1
        if pred_state == gold_state:
            turn_stats[turn_idx][0] += 1

    return turn_stats


def main():
    files = sys.argv[1:] or [
        'output/predictions_full.jsonl',
        'output/predictions_grpo.jsonl',
        'output/predictions_sft.jsonl',
    ]

    labels = {
        'predictions_full.jsonl': 'Full-state',
        'predictions_grpo.jsonl': 'GRPO (incremental)',
        'predictions_sft.jsonl': 'SFT (incremental)',
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    markers = ['o', 's', '^', 'D']

    for i, filepath in enumerate(files):
        if not Path(filepath).exists():
            print(f"[WARN] {filepath} not found, skipping.")
            continue

        turn_stats = compute_jga_by_turn(filepath)
        turns = sorted(turn_stats.keys())
        jga_values = [turn_stats[t][0] / turn_stats[t][1] for t in turns]
        counts = [turn_stats[t][1] for t in turns]

        fname = Path(filepath).name
        label = labels.get(fname, fname)

        ax.plot(
            turns, jga_values,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=label,
            linewidth=2,
            markersize=5,
            alpha=0.85,
        )

        # Print table
        print(f"\n{label} ({fname}):")
        print(f"  {'Turn':>5} {'JGA':>8} {'Correct':>8} {'Total':>8}")
        for t, jga, cnt in zip(turns, jga_values, counts):
            correct = turn_stats[t][0]
            print(f"  {t:>5} {jga:>8.4f} {correct:>8} {cnt:>8}")

    ax.set_xlabel('Turn Index', fontsize=13)
    ax.set_ylabel('JGA', fontsize=13)
    ax.set_title('Joint Goal Accuracy by Turn', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_path = 'output/jga_by_turn.png'
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
