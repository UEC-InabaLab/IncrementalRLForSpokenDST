"""Unit tests for the DSTC-11 gold-JSON + mapping-file + HDF5-hyp pairing
logic, exercised against the confirmed real file formats (see
prepare_data_dstc11.py's docstring for the source)."""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/train')

from prepare_data_dstc11 import load_mapping, normalize_dialogue_id, process_dialogue


def test_normalize_dialogue_id_strips_json_suffix():
    assert normalize_dialogue_id("mul0016.json") == "mul0016"
    assert normalize_dialogue_id("mul0016") == "mul0016"


def test_load_mapping_parses_speaker_and_text(tmp_path):
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text(
        "line_nr: 0 dialog_id: pmul1635.json turn_id: 1 text: user: i need a hotel. state: hotel-area=east\n"
        "line_nr: 1 dialog_id: pmul1635.json turn_id: 2 text: agent: sure, what price range? state:\n",
        encoding="utf-8",
    )
    turns = load_mapping(mapping_file)
    assert turns["pmul1635"][1] == {"speaker": "user", "text": "i need a hotel."}
    assert turns["pmul1635"][2] == {"speaker": "agent", "text": "sure, what price range?"}


def test_process_dialogue_builds_one_sample_per_turn():
    gold_states = [
        {"hotel": {"area": "east", "stars": "4"}},
        {"hotel": {"area": "east", "stars": "4", "parking": "yes"}},
    ]
    turns_by_id = {
        2: {"speaker": "agent", "text": "what price range?"},
    }
    user_hyps = {1: "i need a hotel in the east", 3: "gimme free parking too"}

    samples = process_dialogue("pmul1635", gold_states, turns_by_id, user_hyps, "SYS_PROMPT", "tts_verbatim")

    assert len(samples) == 1
    sample = samples[0]
    assert sample["audios"] == ["pmul1635_3.wav"]
    assert sample["dialogue_id"] == "pmul1635"
    assert sample["variant"] == "tts_verbatim"
    assert sample["prev_belief_state"] == '{"hotel": {"area": "east", "stars": "4"}}'
    assert sample["belief_state"] == '{"hotel": {"area": "east", "stars": "4", "parking": "yes"}}'
    assert "set(hotel.parking=yes)" in sample["solution"]
    assert "User: i need a hotel in the east" in sample["messages"][1]["content"]


def test_process_dialogue_missing_opening_user_hyp_skips():
    gold_states = [{"hotel": {"area": "east"}}]
    assert process_dialogue("D1", gold_states, {}, {}, "SYS_PROMPT", "tts_verbatim") == []


def test_process_dialogue_stops_when_system_text_missing():
    gold_states = [{"hotel": {"area": "east"}}, {"hotel": {"area": "east", "stars": "4"}}]
    user_hyps = {1: "hi"}  # no turn 3 hyp and no turn 2 mapping text

    assert process_dialogue("D1", gold_states, {}, user_hyps, "SYS_PROMPT", "tts_verbatim") == []
