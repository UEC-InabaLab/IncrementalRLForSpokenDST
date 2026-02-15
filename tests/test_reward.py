"""Unit tests for the incremental DST reward functions."""

import sys
sys.path.insert(0, '.')

from src.swift_plugin.dapo_reward import (
    DSTFormatReward,
    DSTRewardIncremental,
    compute_diff_f1,
    compute_format_score,
    compute_gold_diff_ops,
    compute_transcript_f1,
    extract_answer,
    extract_transcript,
    parse_diff_ops,
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

def test_transcript_f1_perfect():
    assert compute_transcript_f1("hello world", "hello world") == 1.0


def test_transcript_f1_partial():
    f1 = compute_transcript_f1("hello world foo", "hello world bar")
    # Common: hello, world (2); pred=3, gold=3
    # P=2/3, R=2/3, F1=2/3
    assert abs(f1 - 2/3) < 1e-6


def test_transcript_f1_empty():
    assert compute_transcript_f1("", "") == 1.0
    assert compute_transcript_f1("hello", "") == 0.0
    assert compute_transcript_f1("", "hello") == 0.0


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
    test_transcript_f1_perfect()
    test_transcript_f1_partial()
    test_transcript_f1_empty()
    test_diff_f1_perfect()
    test_diff_f1_partial()
    test_diff_f1_both_empty()
    test_format_score_perfect()
    test_format_score_empty_answer()
    test_format_score_no_tags()
    test_format_score_only_transcript()
    test_dst_reward_incremental_perfect()
    test_dst_reward_incremental_zero()
    test_dst_format_reward()
    print("All tests passed!")
