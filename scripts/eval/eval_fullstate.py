"""Evaluate full-state DST model predictions.

Reads prediction JSONL (one prediction per line) and computes:
    - Transcript WER
    - Joint Goal Accuracy (JGA)
    - Slot F1

Each prediction record must contain:
  - prediction: model output with <transcript> and <answer> tags
  - solution:   gold output with <transcript> and <answer> tags
  - belief_state: gold belief state (JSON string or dict)

The <answer> tag contains the complete belief state as JSON.
JGA and Slot F1 are computed by comparing the predicted JSON state
directly against the gold belief_state.

Usage:
  python scripts/eval/eval_fullstate.py --input output/fullstate_predictions.jsonl
"""

import argparse
import json
import re
from typing import Dict, List, Optional, Set, Tuple


def extract_transcript(text: str) -> Optional[str]:
    """Extract content between <transcript>...</transcript> tags."""
    m = re.search(r'<transcript>(.*?)</transcript>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_answer(text: str) -> Optional[str]:
    """Extract content between <answer>...</answer> tags."""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def parse_state_json(answer_text: str) -> Dict:
    """Parse a JSON dialogue state from answer text."""
    if not answer_text or not answer_text.strip():
        return {}
    try:
        obj = json.loads(answer_text.strip())
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _word_edit_distance(hyp: List[str], ref: List[str]) -> int:
    """Compute word-level Levenshtein edit distance via DP."""
    n, m = len(hyp), len(ref)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            if hyp[i - 1] == ref[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def compute_transcript_wer(pred_transcript: str, gold_transcript: str) -> float:
    """Compute word error rate (WER)."""
    pred_words = pred_transcript.lower().split()
    gold_words = gold_transcript.lower().split()

    if not pred_words and not gold_words:
        return 0.0
    if not gold_words:
        return 1.0

    edit_dist = _word_edit_distance(pred_words, gold_words)
    return edit_dist / len(gold_words)


def _safe_json_loads(state_val) -> Dict:
    """Safely parse a state value (string or dict) into a dict."""
    if isinstance(state_val, dict):
        return state_val
    try:
        obj = json.loads(state_val) if state_val else {}
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _state_to_slot_set(state: Dict) -> Set[Tuple[str, str, str]]:
    """Flatten belief state dict to a set of (domain, slot, value) tuples."""
    out: Set[Tuple[str, str, str]] = set()
    for domain, slots in state.items():
        if not isinstance(slots, dict):
            continue
        for slot, value in slots.items():
            out.add((str(domain), str(slot), str(value)))
    return out


def compute_slot_f1(pred_state: Dict, gold_state: Dict) -> float:
    """Compute slot-level F1 for one turn using flattened state entries."""
    pred_slots = _state_to_slot_set(pred_state)
    gold_slots = _state_to_slot_set(gold_state)

    if not pred_slots and not gold_slots:
        return 1.0
    if not pred_slots or not gold_slots:
        return 0.0

    correct = len(pred_slots & gold_slots)
    precision = correct / len(pred_slots)
    recall = correct / len(gold_slots)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(predictions: List[Dict]) -> Dict:
    """Compute WER, JGA, and slot F1 for full-state DST predictions."""
    transcript_wers = []
    slot_f1s = []
    jga_correct = 0
    jga_total = 0

    for pred in predictions:
        prediction = pred['prediction']
        solution = pred.get('solution', '')
        belief_state_raw = pred.get('belief_state', '{}')

        # --- Transcript WER ---
        pred_transcript = extract_transcript(prediction)
        gold_transcript = extract_transcript(solution)
        if pred_transcript is not None and gold_transcript is not None:
            transcript_wers.append(
                compute_transcript_wer(pred_transcript, gold_transcript)
            )

        # --- JGA & Slot F1 ---
        pred_answer = extract_answer(prediction)
        pred_state = parse_state_json(pred_answer) if pred_answer is not None else {}

        gold_state = _safe_json_loads(belief_state_raw)

        jga_total += 1
        if pred_state == gold_state:
            jga_correct += 1
        slot_f1s.append(compute_slot_f1(pred_state, gold_state))

    n = len(predictions)
    results = {
        'num_samples': n,
        'transcript_wer': sum(transcript_wers) / len(transcript_wers) if transcript_wers else 0.0,
        'jga': jga_correct / jga_total if jga_total else 0.0,
        'slot_f1': sum(slot_f1s) / len(slot_f1s) if slot_f1s else 0.0,
        'jga_total': jga_total,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate full-state DST predictions')
    parser.add_argument('--input', required=True, help='Predictions JSONL file')
    parser.add_argument('--output', default=None, help='Output metrics JSON file')
    args = parser.parse_args()

    predictions = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    results = evaluate(predictions)

    print('=' * 60)
    print('Full-State DST Evaluation Results')
    print('=' * 60)
    print(f"  Samples:          {results['num_samples']}")
    print(f"  Transcript WER:   {results['transcript_wer']:.4f}")
    print(f"  JGA:              {results['jga']:.4f} ({results['jga_total']} turns)")
    print(f"  Slot F1:          {results['slot_f1']:.4f}")
    print('=' * 60)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f'Metrics saved to {args.output}')


if __name__ == '__main__':
    main()
