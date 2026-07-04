"""Unit tests for the DSTC-11 gold-JSON + HDF5-hyp pairing logic."""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/train')

from prepare_data_dstc11 import clean_state, process_dialogue


def test_clean_state_drops_empty_values():
    assert clean_state({"restaurant": {"area": "west", "food": ""}}) == {
        "restaurant": {"area": "west"},
    }


def test_clean_state_empty_input():
    assert clean_state({}) == {}
    assert clean_state(None) == {}


def test_process_dialogue_builds_one_sample_per_turn():
    gold_turns = [
        {"response": "how can i help", "state": {}, "active_domains": []},
        {
            "response": "ok noted",
            "state": {"restaurant": {"pricerange": "cheap"}},
            "active_domains": ["restaurant"],
        },
    ]
    user_hyps = {0: "hiya", 1: "gimme a cheap place to eat"}

    samples = process_dialogue("D1", gold_turns, user_hyps, "SYS_PROMPT", "tts_verbatim")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["audios"] == ["D1_1.wav"]
    assert sample["dialogue_id"] == "D1"
    assert sample["variant"] == "tts_verbatim"
    assert sample["prev_belief_state"] == "{}"
    assert sample["belief_state"] == '{"restaurant": {"pricerange": "cheap"}}'
    assert "set(restaurant.pricerange=cheap)" in sample["solution"]
    assert "User: hiya" in sample["messages"][1]["content"]


def test_process_dialogue_missing_first_user_hyp_skips():
    gold_turns = [{"response": "hi", "state": {}, "active_domains": []}]
    assert process_dialogue("D1", gold_turns, {}, "SYS_PROMPT", "tts_verbatim") == []


def test_process_dialogue_stops_when_reply_missing():
    gold_turns = [
        {"response": "how can i help", "state": {}, "active_domains": []},
        {"response": "ok noted", "state": {"restaurant": {"pricerange": "cheap"}}, "active_domains": []},
    ]
    user_hyps = {0: "hiya"}  # no hyp for turn 1: no sample should be produced

    assert process_dialogue("D1", gold_turns, user_hyps, "SYS_PROMPT", "tts_verbatim") == []
