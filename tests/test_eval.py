"""Unit tests for eval.py and eval_fullstate.py pure-Python functions."""

import sys
sys.path.insert(0, '.')

from scripts.eval.eval import (
    apply_diff_ops,
    compute_slot_f1,
    compute_transcript_wer,
    evaluate,
    extract_answer,
    extract_transcript,
    parse_diff_ops,
)
from scripts.eval.eval_fullstate import (
    compute_slot_f1 as fs_compute_slot_f1,
    compute_transcript_wer as fs_compute_transcript_wer,
    evaluate as fs_evaluate,
    extract_answer as fs_extract_answer,
    extract_transcript as fs_extract_transcript,
    parse_state_json,
)


# ---------------------------------------------------------------------------
# eval.py
# ---------------------------------------------------------------------------

def test_eval_extract_transcript():
    assert extract_transcript("<transcript>hi</transcript>") == "hi"
    assert extract_transcript("no tags") is None


def test_eval_extract_answer():
    assert extract_answer("<answer>set(a.b=c)</answer>") == "set(a.b=c)"
    assert extract_answer("<answer></answer>") == ""


def test_eval_parse_diff_ops():
    ops = parse_diff_ops("set(restaurant.area=centre)\ndelete(hotel.parking)")
    assert ('set', 'restaurant', 'area', 'centre') in ops
    assert ('delete', 'hotel', 'parking', None) in ops


def test_eval_parse_diff_ops_empty():
    assert parse_diff_ops("") == set()


def test_eval_apply_diff_ops_set():
    state = apply_diff_ops({}, {('set', 'restaurant', 'area', 'centre')})
    assert state == {"restaurant": {"area": "centre"}}


def test_eval_apply_diff_ops_update():
    state = apply_diff_ops(
        {"restaurant": {"area": "centre"}},
        {('update', 'restaurant', 'area', 'north')},
    )
    assert state == {"restaurant": {"area": "north"}}


def test_eval_apply_diff_ops_delete():
    state = apply_diff_ops(
        {"restaurant": {"area": "centre", "food": "indian"}},
        {('delete', 'restaurant', 'food', None)},
    )
    assert state == {"restaurant": {"area": "centre"}}


def test_eval_apply_diff_ops_delete_empty_domain():
    state = apply_diff_ops(
        {"restaurant": {"area": "centre"}},
        {('delete', 'restaurant', 'area', None)},
    )
    assert state == {}


def test_eval_compute_transcript_wer_perfect():
    assert compute_transcript_wer("hello world", "hello world") == 0.0


def test_eval_compute_transcript_wer_one_sub():
    wer = compute_transcript_wer("hello world foo", "hello world bar")
    assert abs(wer - 1 / 3) < 1e-6


def test_eval_compute_slot_f1_perfect():
    state = {"restaurant": {"area": "centre"}}
    assert compute_slot_f1(state, state) == 1.0


def test_eval_compute_slot_f1_partial():
    pred = {"restaurant": {"area": "centre", "food": "chinese"}}
    gold = {"restaurant": {"area": "centre"}}
    f1 = compute_slot_f1(pred, gold)
    assert abs(f1 - 2 / 3) < 1e-6


def test_eval_compute_slot_f1_both_empty():
    assert compute_slot_f1({}, {}) == 1.0


def test_eval_evaluate_perfect():
    rec = {
        "prediction": "<transcript>hi</transcript><answer>set(restaurant.area=centre)</answer>",
        "solution": "<transcript>hi</transcript><answer>set(restaurant.area=centre)</answer>",
        "belief_state": '{"restaurant":{"area":"centre"}}',
        "input_belief_state": "{}",
    }
    results = evaluate([rec])
    assert results["jga"] == 1.0
    assert results["transcript_wer"] == 0.0


def test_eval_evaluate_wrong_state():
    rec = {
        "prediction": "<transcript>hi</transcript><answer>set(restaurant.area=north)</answer>",
        "solution": "<transcript>hi</transcript><answer>set(restaurant.area=centre)</answer>",
        "belief_state": '{"restaurant":{"area":"centre"}}',
        "input_belief_state": "{}",
    }
    results = evaluate([rec])
    assert results["jga"] == 0.0


# ---------------------------------------------------------------------------
# eval_fullstate.py
# ---------------------------------------------------------------------------

def test_fs_extract_transcript():
    assert fs_extract_transcript("<transcript>hi</transcript>") == "hi"


def test_fs_extract_answer():
    assert fs_extract_answer('<answer>{"a":"b"}</answer>') == '{"a":"b"}'


def test_fs_parse_state_json_valid():
    state = parse_state_json('{"restaurant":{"area":"centre"}}')
    assert state == {"restaurant": {"area": "centre"}}


def test_fs_parse_state_json_empty():
    assert parse_state_json("") == {}
    assert parse_state_json("{}") == {}


def test_fs_parse_state_json_invalid():
    assert parse_state_json("not json") == {}


def test_fs_compute_transcript_wer_perfect():
    assert fs_compute_transcript_wer("hello world", "hello world") == 0.0


def test_fs_compute_slot_f1_perfect():
    state = {"restaurant": {"area": "centre"}}
    assert fs_compute_slot_f1(state, state) == 1.0


def test_fs_evaluate_perfect():
    rec = {
        "prediction": '<transcript>hi</transcript><answer>{"restaurant":{"area":"centre"}}</answer>',
        "solution": '<transcript>hi</transcript><answer>{"restaurant":{"area":"centre"}}</answer>',
        "belief_state": '{"restaurant":{"area":"centre"}}',
    }
    results = fs_evaluate([rec])
    assert results["jga"] == 1.0
    assert results["transcript_wer"] == 0.0


def test_fs_evaluate_wrong_state():
    rec = {
        "prediction": '<transcript>hi</transcript><answer>{"restaurant":{"area":"north"}}</answer>',
        "solution": '<transcript>hi</transcript><answer>{"restaurant":{"area":"centre"}}</answer>',
        "belief_state": '{"restaurant":{"area":"centre"}}',
    }
    results = fs_evaluate([rec])
    assert results["jga"] == 0.0
