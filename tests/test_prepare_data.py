"""Unit tests for prepare_data.py's SpokenWOZ conversion logic."""

import sys
sys.path.insert(0, '.')

from scripts.train.prepare_data import (
    build_solution,
    flatten_belief_state,
    process_dialogue,
)


def test_flatten_belief_state_semi_book():
    metadata = {
        "restaurant": {
            "semi": {"pricerange": "cheap", "area": ""},
            "book": {"booked": [], "time": "19:00"},
        },
    }
    assert flatten_belief_state(metadata) == {
        "restaurant": {"pricerange": "cheap", "time": "19:00"},
    }


def test_build_solution_transcript_excludes_system_text():
    """System text is already given as plain-text input, so the gold
    transcript covers only the user turn — not something to transcribe."""
    solution = build_solution("cheap please", {}, {"restaurant": {"pricerange": "cheap"}})
    assert "<transcript>\nUser: cheap please\n</transcript>" in solution
    assert "System:" not in solution
    assert "<answer>set(restaurant.pricerange=cheap)</answer>" in solution


def test_process_dialogue_carries_sys_text_and_opening_user_text():
    log = [
        {"tag": "user", "text": "hi there", "metadata": {}},
        {"tag": "system", "text": "how can i help", "metadata": {}},
        {"tag": "user", "text": "i want a cheap restaurant", "metadata": {}},
        {
            "tag": "system",
            "text": "ok noted",
            "metadata": {"restaurant": {"semi": {"pricerange": "cheap"}, "book": {"booked": []}}},
        },
    ]

    samples = process_dialogue("D1", log, "SYS_PROMPT")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["sys_text"] == "how can i help"
    assert sample["opening_user_text"] == "hi there"
    assert sample["audios"] == ["D1_1_2.wav"]
    assert "set(restaurant.pricerange=cheap)" in sample["solution"]
    assert "System:" not in sample["solution"]
