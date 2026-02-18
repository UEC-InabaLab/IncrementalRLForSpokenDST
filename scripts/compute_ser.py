"""Compute Slot Error Rate (SER) for prediction files.

SER = (substitutions + insertions + deletions) / total_reference_slots

For turns with no reference slots, we count insertion errors if any predicted slots exist.

Supports both full-state (JSON answer) and incremental (diff ops answer) formats.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Tuple


def extract_answer(text: str) -> Optional[str]:
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return m.group(1).strip() if m else None


_OP_PATTERN = re.compile(r'(set|update|delete)\(([^)]+)\)')


def parse_diff_ops(answer_text: str) -> list:
    """Parse diff operations, returning list of (op, domain, slot, value)."""
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
    """Apply diff operations to a state dict, returning a new state."""
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


def state_to_slot_set(state: Dict) -> Set[Tuple[str, str, str]]:
    """Flatten state dict to set of (domain, slot, value) tuples."""
    out = set()
    for domain, slots in state.items():
        if not isinstance(slots, dict):
            continue
        for slot, value in slots.items():
            out.add((str(domain), str(slot), str(value)))
    return out


def compute_ser(pred_slots: Set[Tuple], gold_slots: Set[Tuple]):
    """Compute SER components: substitutions, insertions, deletions.

    Returns (substitutions, insertions, deletions, total_ref_slots).
    """
    # Group by (domain, slot) for comparison
    pred_by_key = {}
    for domain, slot, value in pred_slots:
        pred_by_key[(domain, slot)] = value

    gold_by_key = {}
    for domain, slot, value in gold_slots:
        gold_by_key[(domain, slot)] = value

    all_keys = set(pred_by_key.keys()) | set(gold_by_key.keys())

    substitutions = 0
    insertions = 0
    deletions = 0

    for key in all_keys:
        in_pred = key in pred_by_key
        in_gold = key in gold_by_key
        if in_pred and in_gold:
            if pred_by_key[key] != gold_by_key[key]:
                substitutions += 1
        elif in_pred and not in_gold:
            insertions += 1
        elif not in_pred and in_gold:
            deletions += 1

    return substitutions, insertions, deletions, len(gold_by_key)


def evaluate_file(filepath: str) -> dict:
    """Evaluate a prediction file and return SER metrics."""
    lines = Path(filepath).read_text().strip().split('\n')

    total_sub = 0
    total_ins = 0
    total_del = 0
    total_ref_slots = 0
    n_turns = 0
    n_correct_turns = 0  # JGA

    for line in lines:
        record = json.loads(line)
        prediction = record['prediction']
        gold_state_str = record.get('belief_state', '{}')

        try:
            gold_state = json.loads(gold_state_str)
            if not isinstance(gold_state, dict):
                gold_state = {}
        except (json.JSONDecodeError, TypeError):
            gold_state = {}

        pred_answer = extract_answer(prediction)

        # Determine if this is full-state or incremental format
        if 'input_belief_state' in record:
            # Incremental format: apply diff ops to input_belief_state
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
            # Full-state format: parse JSON from answer
            if pred_answer is not None:
                try:
                    pred_state = json.loads(pred_answer) if pred_answer.strip() else {}
                    if not isinstance(pred_state, dict):
                        pred_state = {}
                except (json.JSONDecodeError, TypeError):
                    pred_state = {}
            else:
                pred_state = {}

        pred_slots = state_to_slot_set(pred_state)
        gold_slots = state_to_slot_set(gold_state)

        sub, ins, dele, ref_count = compute_ser(pred_slots, gold_slots)
        total_sub += sub
        total_ins += ins
        total_del += dele
        total_ref_slots += ref_count
        n_turns += 1
        if pred_state == gold_state:
            n_correct_turns += 1

    total_errors = total_sub + total_ins + total_del
    # SER denominator: max(total_ref_slots, 1) to avoid division by zero
    ser = total_errors / max(total_ref_slots, 1)
    jga = n_correct_turns / n_turns if n_turns > 0 else 0.0

    return {
        'file': filepath,
        'n_turns': n_turns,
        'total_ref_slots': total_ref_slots,
        'substitutions': total_sub,
        'insertions': total_ins,
        'deletions': total_del,
        'total_errors': total_errors,
        'SER': ser,
        'JGA': jga,
    }


if __name__ == '__main__':
    files = sys.argv[1:] or [
        'output/predictions_full.jsonl',
        'output/predictions_grpo.jsonl',
        'output/predictions_sft.jsonl',
    ]

    results = []
    for f in files:
        if not Path(f).exists():
            print(f"[WARN] {f} not found, skipping.")
            continue
        result = evaluate_file(f)
        results.append(result)

    # Print results
    print(f"\n{'='*80}")
    print(f"{'File':<40} {'Turns':>6} {'RefSlots':>9} {'Sub':>5} {'Ins':>5} {'Del':>5} {'Errors':>7} {'SER':>8} {'JGA':>8}")
    print(f"{'-'*80}")
    for r in results:
        fname = Path(r['file']).name
        print(
            f"{fname:<40} {r['n_turns']:>6} {r['total_ref_slots']:>9} "
            f"{r['substitutions']:>5} {r['insertions']:>5} {r['deletions']:>5} "
            f"{r['total_errors']:>7} {r['SER']:>7.4f} {r['JGA']:>7.4f}"
        )
    print(f"{'='*80}")
