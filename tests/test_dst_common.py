"""Unit tests for the shared diff-op / JSONL-building helpers used by all
per-dataset prepare_data*.py scripts."""

import sys
sys.path.insert(0, '.')

from scripts.train.dst_common import (
    build_solution,
    build_user_message,
    compute_diff_ops,
    flatten_multiwoz_metadata,
)


def test_flatten_multiwoz_metadata_semi_book():
    metadata = {
        "restaurant": {
            "semi": {"pricerange": "cheap", "area": ""},
            "book": {"booked": [], "time": "19:00"},
        },
    }
    assert flatten_multiwoz_metadata(metadata) == {
        "restaurant": {"pricerange": "cheap", "time": "19:00"},
    }


def test_flatten_multiwoz_metadata_filters_not_mentioned():
    metadata = {"hotel": {"semi": {"area": "not mentioned"}, "book": {}}}
    assert flatten_multiwoz_metadata(metadata) == {}


def test_flatten_multiwoz_metadata_flat_domain():
    metadata = {"police": {"address": "Parkside", "name": ""}}
    assert flatten_multiwoz_metadata(metadata) == {"police": {"address": "Parkside"}}


def test_compute_diff_ops_set_update_delete():
    prev = {"hotel": {"area": "west"}}
    curr = {"hotel": {"area": "east"}, "restaurant": {"pricerange": "cheap"}}
    ops = compute_diff_ops(prev, curr)
    assert ops == [
        "update(hotel.area=east)",
        "set(restaurant.pricerange=cheap)",
    ]


def test_compute_diff_ops_delete():
    prev = {"hotel": {"area": "west"}}
    curr = {}
    assert compute_diff_ops(prev, curr) == ["delete(hotel.area)"]


def test_compute_diff_ops_no_change():
    state = {"hotel": {"area": "west"}}
    assert compute_diff_ops(state, state) == []


def test_build_user_message_includes_history_and_state():
    msg = build_user_message(["User: hi", "System: hello"], {"hotel": {"area": "west"}})
    assert "[Dialogue History]" in msg
    assert "User: hi" in msg
    assert '{"hotel": {"area": "west"}}' in msg
    assert msg.endswith("[New Audio]\n<audio>")


def test_build_user_message_omits_empty_history():
    msg = build_user_message([], {})
    assert "[Dialogue History]" not in msg


def test_build_solution_contains_transcript_and_diff():
    solution = build_solution(
        "cheap please",
        {},
        {"restaurant": {"pricerange": "cheap"}},
    )
    assert "<transcript>\nUser: cheap please\n</transcript>" in solution
    assert "<answer>set(restaurant.pricerange=cheap)</answer>" in solution


def test_build_solution_transcript_excludes_system_text():
    """System text is already given as plain-text input, so the gold
    transcript covers only the user turn — not something to transcribe."""
    solution = build_solution("cheap please", {}, {})
    assert "System:" not in solution
