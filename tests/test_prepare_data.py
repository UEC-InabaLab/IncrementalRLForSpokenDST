"""Unit tests for prepare_data.py's SpokenWOZ conversion logic.

Shared diff-op / transcript-building behavior (build_solution,
flatten_multiwoz_metadata) is exercised directly in test_dst_common.py;
this file covers process_dialogue's SpokenWOZ-specific turn pairing.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/train')

from prepare_data import process_dialogue


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


def test_process_dialogue_too_short_returns_empty():
    log = [{"tag": "user", "text": "hi", "metadata": {}}]
    assert process_dialogue("D1", log, "SYS_PROMPT") == []
