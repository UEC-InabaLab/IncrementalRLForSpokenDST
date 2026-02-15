"""
Incremental DST Reward Functions for ms-swift GRPO training.

Registers custom ORM reward functions:
  - dst_incremental: Combined reward (transcript F1 + diff F1 + exact match + format)
  - dst_format: Format-only reward

Usage:
  swift rlhf --rlhf_type grpo \
      --external_plugins src/swift_plugin/dapo_reward.py \
      --reward_funcs dst_incremental dst_format \
      --reward_weights 0.9 0.1
"""

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

try:
    from swift.plugin.orm import ORM, orms
except ImportError:
    # Fallback for testing without ms-swift installed
    class ORM:
        pass
    orms = {}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_transcript(text: str) -> Optional[str]:
    """Extract content between <transcript>...</transcript> tags."""
    m = re.search(r'<transcript>(.*?)</transcript>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def extract_answer(text: str) -> Optional[str]:
    """Extract content between <answer>...</answer> tags."""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return m.group(1).strip() if m else None


_OP_PATTERN = re.compile(
    r'(set|update|delete)\(([^)]+)\)'
)


def parse_diff_ops(answer_text: str) -> Set[Tuple[str, str, str, Optional[str]]]:
    """Parse diff operations from answer text.

    Returns a set of (op, domain, slot, value) tuples.
    For delete ops, value is None.
    """
    if not answer_text or not answer_text.strip():
        return set()

    ops: Set[Tuple[str, str, str, Optional[str]]] = set()
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
            # delete(domain.slot)
            parts = content.split('.', 1)
            if len(parts) == 2:
                ops.add((op_type, parts[0], parts[1], None))
        else:
            # set(domain.slot=value) or update(domain.slot=value)
            if '=' in content:
                key, value = content.split('=', 1)
                parts = key.split('.', 1)
                if len(parts) == 2:
                    ops.add((op_type, parts[0], parts[1], value))
    return ops


def compute_gold_diff_ops(
    prev_state: Dict, curr_state: Dict
) -> Set[Tuple[str, str, str, Optional[str]]]:
    """Compute gold diff operations from previous and current belief states.

    This is used as a fallback when solution's <answer> is used for gold ops.
    """
    ops: Set[Tuple[str, str, str, Optional[str]]] = set()

    all_domains = set(list(prev_state.keys()) + list(curr_state.keys()))

    for domain in all_domains:
        prev_slots = prev_state.get(domain, {})
        curr_slots = curr_state.get(domain, {})

        # New or updated slots
        for slot, value in curr_slots.items():
            if slot not in prev_slots:
                ops.add(('set', domain, slot, value))
            elif prev_slots[slot] != value:
                ops.add(('update', domain, slot, value))

        # Deleted slots
        for slot in prev_slots:
            if slot not in curr_slots:
                ops.add(('delete', domain, slot, None))

    return ops


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

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


def compute_transcript_wer_reward(pred_transcript: str, gold_transcript: str) -> float:
    """Compute WER-based reward: max(0, 1 - WER).

    WER = (substitutions + insertions + deletions) / len(reference).
    Reward is clipped to [0, 1].
    """
    pred_words = pred_transcript.lower().split()
    gold_words = gold_transcript.lower().split()

    if not pred_words and not gold_words:
        return 1.0
    if not gold_words:
        return 0.0

    edit_dist = _word_edit_distance(pred_words, gold_words)
    wer = edit_dist / len(gold_words)
    return max(0.0, 1.0 - wer)


def compute_diff_f1(
    pred_ops: Set[Tuple], gold_ops: Set[Tuple]
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for diff operations."""
    if not pred_ops and not gold_ops:
        return 1.0, 1.0, 1.0
    if not pred_ops:
        return 0.0, 0.0, 0.0
    if not gold_ops:
        return 0.0, 0.0, 0.0

    correct = len(pred_ops & gold_ops)
    precision = correct / len(pred_ops) if pred_ops else 0.0
    recall = correct / len(gold_ops) if gold_ops else 0.0

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_format_score(response: str) -> float:
    """Evaluate format correctness of the response."""
    score = 0.0

    has_transcript = (
        '<transcript>' in response and '</transcript>' in response
    )
    has_answer = '<answer>' in response and '</answer>' in response

    if has_transcript:
        score += 0.3

    if has_answer:
        score += 0.3
        answer_content = extract_answer(response)
        if answer_content is not None:
            # Empty answer is valid (no changes)
            if not answer_content.strip():
                score += 0.4
            else:
                # Check all lines match op format
                lines = [line.strip() for line in answer_content.strip().split('\n') if line.strip()]
                all_valid = all(_OP_PATTERN.match(line) for line in lines)
                if all_valid:
                    score += 0.4

    return score


# ---------------------------------------------------------------------------
# ORM Reward Classes
# ---------------------------------------------------------------------------

class DSTRewardIncremental(ORM):
    """Combined incremental DST reward function.

    Reward = w_transcript * transcript_wer_reward
           + w_diff_f1   * diff_f1
           + w_exact     * exact_match
           + w_format    * format_score

    Default weights: transcript=0.3, diff_f1=0.5, exact_match=0.1, format=0.1
    """

    _call_count = 0
    LOG_INTERVAL = int(os.environ.get('DST_LOG_INTERVAL', '1'))

    def __call__(self, completions: List[str], solution: List[str] = None,
                 belief_state: List[str] = None,
                 prev_belief_state: List[str] = None,
                 **kwargs) -> List[float]:
        rewards = []
        breakdowns = []
        for i, completion in enumerate(completions):
            reward, breakdown = self._score_single(
                completion,
                solution[i] if solution else None,
                belief_state[i] if belief_state else None,
                prev_belief_state[i] if prev_belief_state else None,
            )
            rewards.append(reward)
            breakdowns.append(breakdown)

        DSTRewardIncremental._call_count += 1
        is_rank0 = int(os.environ.get('LOCAL_RANK', '0')) == 0
        if is_rank0 and DSTRewardIncremental._call_count % self.LOG_INTERVAL == 0:
            self._log_samples(completions, solution, rewards, breakdowns)

        return rewards

    def _log_samples(
        self,
        completions: List[str],
        solution: List[str],
        rewards: List[float],
        breakdowns: List[dict],
    ) -> None:
        """Log the first completion alongside the gold solution and reward breakdown."""
        sep = '=' * 60
        print(f'\n{sep}', flush=True)
        print(
            f'[DST Reward] call #{DSTRewardIncremental._call_count}  '
            f'num_completions={len(completions)}', flush=True)
        print(sep, flush=True)

        # --- Gold (from the first solution) ---
        gold = solution[0] if solution else '(none)'
        gold_answer = extract_answer(gold) if gold else None
        gold_transcript = extract_transcript(gold) if gold else None
        print(f'[Gold transcript] {gold_transcript}', flush=True)
        print(f'[Gold answer]     {gold_answer}', flush=True)
        print('-' * 60, flush=True)

        # --- Show first 3 completions ---
        n_show = min(3, len(completions))
        for i in range(n_show):
            bd = breakdowns[i]
            pred_transcript = extract_transcript(completions[i])
            pred_answer = extract_answer(completions[i])
            print(
                f'[Completion {i}] reward={rewards[i]:.4f}  '
                f'transcript_wer_reward={bd["transcript_wer_reward"]:.3f}  '
                f'diff_f1={bd["diff_f1"]:.3f}  '
                f'exact={bd["exact_match"]:.0f}  '
                f'format={bd["format"]:.3f}', flush=True)
            print(f'  [Pred transcript] {pred_transcript}', flush=True)
            print(f'  [Pred answer]     {pred_answer}', flush=True)

        # --- Summary of all rewards ---
        avg_r = sum(rewards) / len(rewards) if rewards else 0.0
        max_r = max(rewards) if rewards else 0.0
        min_r = min(rewards) if rewards else 0.0
        print('-' * 60, flush=True)
        print(
            f'[Rewards] avg={avg_r:.4f}  min={min_r:.4f}  max={max_r:.4f}  '
            f'all={[round(r, 4) for r in rewards]}', flush=True)
        print(sep, flush=True)

    def _score_single(
        self,
        completion: str,
        solution: Optional[str],
        belief_state_str: Optional[str],
        prev_belief_state_str: Optional[str],
    ) -> Tuple[float, dict]:
        weights = {
            'transcript': 0.3,
            'diff_f1': 0.5,
            'exact_match': 0.1,
            'format': 0.1,
        }

        # --- Format score ---
        format_score = compute_format_score(completion)

        # --- Transcript score ---
        transcript_score = 0.0
        pred_transcript = extract_transcript(completion)
        gold_transcript = extract_transcript(solution) if solution else None

        if pred_transcript is not None and gold_transcript is not None:
            transcript_score = compute_transcript_wer_reward(pred_transcript, gold_transcript)

        # --- Diff F1 and exact match ---
        diff_f1 = 0.0
        exact_match = 0.0

        pred_answer = extract_answer(completion)
        pred_ops = parse_diff_ops(pred_answer) if pred_answer is not None else set()

        # Get gold ops from solution's <answer> tag
        gold_answer = extract_answer(solution) if solution else None
        if gold_answer is not None:
            gold_ops = parse_diff_ops(gold_answer)
        elif belief_state_str and prev_belief_state_str:
            # Fallback: compute from belief states
            try:
                curr_state = json.loads(belief_state_str)
                prev_state = json.loads(prev_belief_state_str)
                gold_ops = compute_gold_diff_ops(prev_state, curr_state)
            except (json.JSONDecodeError, TypeError):
                gold_ops = set()
        else:
            gold_ops = set()

        if pred_answer is not None:
            _, _, diff_f1 = compute_diff_f1(pred_ops, gold_ops)
            exact_match = 1.0 if pred_ops == gold_ops else 0.0

        # --- Total reward ---
        total = (
            weights['transcript'] * transcript_score
            + weights['diff_f1'] * diff_f1
            + weights['exact_match'] * exact_match
            + weights['format'] * format_score
        )

        breakdown = {
            'transcript_wer_reward': transcript_score,
            'diff_f1': diff_f1,
            'exact_match': exact_match,
            'format': format_score,
            'total': min(max(total, 0.0), 1.0),
        }
        return min(max(total, 0.0), 1.0), breakdown


class DSTFormatReward(ORM):
    """Format-only reward for incremental DST output."""

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        return [compute_format_score(c) for c in completions]


# ---------------------------------------------------------------------------
# Register with ms-swift
# ---------------------------------------------------------------------------

orms['dst_incremental'] = DSTRewardIncremental
orms['dst_format'] = DSTFormatReward
