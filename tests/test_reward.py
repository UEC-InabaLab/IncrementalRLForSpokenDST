"""Unit tests for the DST reward functions (incremental and full-state)."""

import sys
sys.path.insert(0, '.')

from src.reward import (
    DSTFormatReward,
    DSTRewardFullState,
    DSTRewardIncremental,
    DSTRewardIncrementalNoTranscript,
    compute_diff_f1,
    compute_format_score,
    compute_fullstate_format_score,
    compute_gold_diff_ops,
    compute_state_slot_f1,
    compute_transcript_wer_reward,
    extract_answer,
    extract_transcript,
    parse_diff_ops,
    parse_state_json,
)


# ---------------------------------------------------------------------------
# Test parsing helpers
# ---------------------------------------------------------------------------

def test_extract_transcript():
    text = "<transcript>\nSystem: hello.\nUser: hi.\n</transcript>\n<answer></answer>"
    assert extract_transcript(text) == "System: hello.\nUser: hi."


def test_extract_answer():
    text = "<transcript>foo</transcript>\n<answer>set(restaurant.area=centre)</answer>"
    assert extract_answer(text) == "set(restaurant.area=centre)"


def test_extract_answer_empty():
    text = "<transcript>foo</transcript>\n<answer></answer>"
    assert extract_answer(text) == ""


def test_parse_diff_ops_single():
    ops = parse_diff_ops("set(restaurant.area=centre)")
    assert ops == {('set', 'restaurant', 'area', 'centre')}


def test_parse_diff_ops_multiple():
    text = "set(restaurant.area=centre)\nupdate(hotel.stars=5)\ndelete(taxi.destination)"
    ops = parse_diff_ops(text)
    assert ops == {
        ('set', 'restaurant', 'area', 'centre'),
        ('update', 'hotel', 'stars', '5'),
        ('delete', 'taxi', 'destination', None),
    }


def test_parse_diff_ops_empty():
    assert parse_diff_ops("") == set()
    assert parse_diff_ops("  ") == set()


def test_compute_gold_diff_ops():
    prev = {}
    curr = {"restaurant": {"area": "centre"}}
    ops = compute_gold_diff_ops(prev, curr)
    assert ops == {('set', 'restaurant', 'area', 'centre')}


def test_compute_gold_diff_ops_update():
    prev = {"restaurant": {"area": "centre"}}
    curr = {"restaurant": {"area": "north"}}
    ops = compute_gold_diff_ops(prev, curr)
    assert ops == {('update', 'restaurant', 'area', 'north')}


def test_compute_gold_diff_ops_delete():
    prev = {"restaurant": {"area": "centre", "food": "indian"}}
    curr = {"restaurant": {"area": "centre"}}
    ops = compute_gold_diff_ops(prev, curr)
    assert ops == {('delete', 'restaurant', 'food', None)}


def test_compute_gold_diff_ops_no_change():
    state = {"restaurant": {"area": "centre"}}
    ops = compute_gold_diff_ops(state, state)
    assert ops == set()


# ---------------------------------------------------------------------------
# Test score computations
# ---------------------------------------------------------------------------

def test_transcript_wer_reward_perfect():
    assert compute_transcript_wer_reward("hello world", "hello world") == 1.0


def test_transcript_wer_reward_partial():
    reward = compute_transcript_wer_reward("hello world foo", "hello world bar")
    # WER = 1/3 (1 substitution out of 3 words), reward = 1 - 1/3 = 2/3
    assert abs(reward - 2/3) < 1e-6


def test_transcript_wer_reward_empty():
    assert compute_transcript_wer_reward("", "") == 1.0
    assert compute_transcript_wer_reward("hello", "") == 0.0
    assert compute_transcript_wer_reward("", "hello") == 0.0


def test_diff_f1_perfect():
    ops = {('set', 'restaurant', 'area', 'centre')}
    p, r, f1 = compute_diff_f1(ops, ops)
    assert f1 == 1.0


def test_diff_f1_partial():
    pred = {('set', 'restaurant', 'area', 'centre'), ('set', 'restaurant', 'food', 'chinese')}
    gold = {('set', 'restaurant', 'area', 'centre')}
    p, r, f1 = compute_diff_f1(pred, gold)
    assert p == 0.5
    assert r == 1.0
    assert abs(f1 - 2/3) < 1e-6


def test_diff_f1_both_empty():
    p, r, f1 = compute_diff_f1(set(), set())
    assert f1 == 1.0


def test_format_score_perfect():
    text = "<transcript>foo</transcript>\n<answer>set(restaurant.area=centre)</answer>"
    assert compute_format_score(text) == 1.0


def test_format_score_empty_answer():
    text = "<transcript>foo</transcript>\n<answer></answer>"
    assert compute_format_score(text) == 1.0


def test_format_score_no_tags():
    assert compute_format_score("random text") == 0.0


def test_format_score_only_transcript():
    text = "<transcript>foo</transcript>"
    assert abs(compute_format_score(text) - 0.3) < 1e-6


# ---------------------------------------------------------------------------
# Test ORM classes
# ---------------------------------------------------------------------------

def test_dst_reward_incremental_perfect():
    reward_fn = DSTRewardIncremental()
    completion = "<transcript>\nSystem: hello.\nUser: hi.\n</transcript>\n<answer>set(restaurant.area=centre)</answer>"
    solution = "<transcript>\nSystem: hello.\nUser: hi.\n</transcript>\n<answer>set(restaurant.area=centre)</answer>"
    rewards = reward_fn(
        completions=[completion],
        solution=[solution],
        belief_state=['{"restaurant":{"area":"centre"}}'],
        prev_belief_state=['{}'],
    )
    assert len(rewards) == 1
    assert rewards[0] == 1.0


def test_dst_reward_incremental_zero():
    reward_fn = DSTRewardIncremental()
    rewards = reward_fn(
        completions=["garbage output"],
        solution=["<transcript>hello</transcript>\n<answer></answer>"],
        belief_state=['{}'],
        prev_belief_state=['{}'],
    )
    assert rewards[0] == 0.0


def test_dst_reward_incremental_no_transcript_perfect():
    reward_fn = DSTRewardIncrementalNoTranscript()
    completion = "<transcript>\nSystem: hello.\nUser: hi.\n</transcript>\n<answer>set(restaurant.area=centre)</answer>"
    solution = "<transcript>\nSystem: hello.\nUser: hi.\n</transcript>\n<answer>set(restaurant.area=centre)</answer>"
    rewards = reward_fn(
        completions=[completion],
        solution=[solution],
        belief_state=['{"restaurant":{"area":"centre"}}'],
        prev_belief_state=['{}'],
    )
    assert len(rewards) == 1
    assert rewards[0] == 1.0


def test_dst_reward_incremental_no_transcript_zero():
    reward_fn = DSTRewardIncrementalNoTranscript()
    rewards = reward_fn(
        completions=["garbage output"],
        solution=["<transcript>hello</transcript>\n<answer></answer>"],
        belief_state=['{}'],
        prev_belief_state=['{}'],
    )
    assert rewards[0] == 0.0


def test_dst_reward_incremental_no_transcript_ignores_transcript():
    """Verify that transcript quality does NOT affect the reward."""
    reward_fn = DSTRewardIncrementalNoTranscript()
    # Perfect diff ops but wrong transcript
    completion = "<transcript>\nwrong transcript\n</transcript>\n<answer>set(restaurant.area=centre)</answer>"
    solution = "<transcript>\nSystem: hello.\nUser: hi.\n</transcript>\n<answer>set(restaurant.area=centre)</answer>"
    rewards = reward_fn(
        completions=[completion],
        solution=[solution],
        belief_state=['{"restaurant":{"area":"centre"}}'],
        prev_belief_state=['{}'],
    )
    # Should still get 1.0 since transcript is not part of the reward
    assert rewards[0] == 1.0


def test_dst_format_reward():
    reward_fn = DSTFormatReward()
    rewards = reward_fn(
        completions=[
            "<transcript>foo</transcript>\n<answer>set(a.b=c)</answer>",
            "no tags at all",
        ]
    )
    assert rewards[0] == 1.0
    assert rewards[1] == 0.0


# ---------------------------------------------------------------------------
# Test full-state DST helpers
# ---------------------------------------------------------------------------

def test_parse_state_json_valid():
    assert parse_state_json('{"restaurant":{"area":"centre"}}') == {"restaurant": {"area": "centre"}}


def test_parse_state_json_empty():
    assert parse_state_json("") == {}
    assert parse_state_json("{}") == {}


def test_parse_state_json_invalid():
    assert parse_state_json("not json") == {}
    assert parse_state_json("[1,2,3]") == {}


def test_compute_state_slot_f1_perfect():
    state = {"restaurant": {"area": "centre", "food": "chinese"}}
    assert compute_state_slot_f1(state, state) == 1.0


def test_compute_state_slot_f1_partial():
    pred = {"restaurant": {"area": "centre", "food": "chinese"}}
    gold = {"restaurant": {"area": "centre"}}
    f1 = compute_state_slot_f1(pred, gold)
    # P=1/2, R=1/1, F1=2/3
    assert abs(f1 - 2/3) < 1e-6


def test_compute_state_slot_f1_both_empty():
    assert compute_state_slot_f1({}, {}) == 1.0


def test_compute_state_slot_f1_one_empty():
    assert compute_state_slot_f1({}, {"restaurant": {"area": "centre"}}) == 0.0
    assert compute_state_slot_f1({"restaurant": {"area": "centre"}}, {}) == 0.0


def test_fullstate_format_score_perfect():
    text = '<transcript>foo</transcript>\n<answer>{"restaurant":{"area":"centre"}}</answer>'
    assert compute_fullstate_format_score(text) == 1.0


def test_fullstate_format_score_empty_answer():
    text = "<transcript>foo</transcript>\n<answer></answer>"
    assert compute_fullstate_format_score(text) == 1.0


def test_fullstate_format_score_empty_json():
    text = "<transcript>foo</transcript>\n<answer>{}</answer>"
    assert compute_fullstate_format_score(text) == 1.0


def test_fullstate_format_score_invalid_json():
    text = "<transcript>foo</transcript>\n<answer>not json</answer>"
    assert abs(compute_fullstate_format_score(text) - 0.6) < 1e-6


def test_fullstate_format_score_no_tags():
    assert compute_fullstate_format_score("random text") == 0.0


# ---------------------------------------------------------------------------
# Test full-state ORM class
# ---------------------------------------------------------------------------

def test_dst_reward_fullstate_perfect():
    reward_fn = DSTRewardFullState()
    completion = '<transcript>\nSystem: hello.\nUser: hi.\n</transcript>\n<answer>{"restaurant":{"area":"centre"}}</answer>'
    solution = '<transcript>\nSystem: hello.\nUser: hi.\n</transcript>\n<answer>{"restaurant":{"area":"centre"}}</answer>'
    rewards = reward_fn(
        completions=[completion],
        solution=[solution],
        belief_state=['{"restaurant":{"area":"centre"}}'],
    )
    assert len(rewards) == 1
    assert abs(rewards[0] - 1.0) < 1e-6


def test_dst_reward_fullstate_zero():
    reward_fn = DSTRewardFullState()
    rewards = reward_fn(
        completions=["garbage output"],
        solution=['<transcript>hello</transcript>\n<answer>{}</answer>'],
        belief_state=['{}'],
    )
    assert rewards[0] == 0.0


def test_dst_reward_fullstate_empty_state():
    reward_fn = DSTRewardFullState()
    completion = "<transcript>\nUser: hello.\n</transcript>\n<answer>{}</answer>"
    solution = "<transcript>\nUser: hello.\n</transcript>\n<answer>{}</answer>"
    rewards = reward_fn(
        completions=[completion],
        solution=[solution],
        belief_state=['{}'],
    )
    assert abs(rewards[0] - 1.0) < 1e-6


if __name__ == '__main__':
    test_extract_transcript()
    test_extract_answer()
    test_extract_answer_empty()
    test_parse_diff_ops_single()
    test_parse_diff_ops_multiple()
    test_parse_diff_ops_empty()
    test_compute_gold_diff_ops()
    test_compute_gold_diff_ops_update()
    test_compute_gold_diff_ops_delete()
    test_compute_gold_diff_ops_no_change()
    test_transcript_wer_reward_perfect()
    test_transcript_wer_reward_partial()
    test_transcript_wer_reward_empty()
    test_diff_f1_perfect()
    test_diff_f1_partial()
    test_diff_f1_both_empty()
    test_format_score_perfect()
    test_format_score_empty_answer()
    test_format_score_no_tags()
    test_format_score_only_transcript()
    test_dst_reward_incremental_perfect()
    test_dst_reward_incremental_zero()
    test_dst_reward_incremental_no_transcript_perfect()
    test_dst_reward_incremental_no_transcript_zero()
    test_dst_reward_incremental_no_transcript_ignores_transcript()
    test_dst_format_reward()
    test_parse_state_json_valid()
    test_parse_state_json_empty()
    test_parse_state_json_invalid()
    test_compute_state_slot_f1_perfect()
    test_compute_state_slot_f1_partial()
    test_compute_state_slot_f1_both_empty()
    test_compute_state_slot_f1_one_empty()
    test_fullstate_format_score_perfect()
    test_fullstate_format_score_empty_answer()
    test_fullstate_format_score_empty_json()
    test_fullstate_format_score_invalid_json()
    test_fullstate_format_score_no_tags()
    test_dst_reward_fullstate_perfect()
    test_dst_reward_fullstate_zero()
    test_dst_reward_fullstate_empty_state()
    print("All tests passed!")
