"""Unit tests for the pure-Python GRPO-format -> per-model conversation
adapters (Audio Flamingo 3 and Kimi-Audio).

Only the format-conversion logic is tested here; the training/inference scripts
that wrap these need a GPU + model weights and are verified separately.
"""

import sys

sys.path.insert(0, "scripts/train/audio_flamingo3")
sys.path.insert(0, "scripts/train/kimi_audio")

from af3_data import grpo_to_af3_conversation  # noqa: E402
from convert_to_kimia_format import grpo_to_kimia_conversation  # noqa: E402


def _sample():
    return {
        "messages": [
            {"role": "system", "content": "You are a DST model."},
            {
                "role": "user",
                "content": (
                    "[Dialogue History]\nUser: hi\nSystem: how can i help\n\n"
                    "[Previous State]\n{}\n\n[New Audio]\n<audio>"
                ),
            },
        ],
        "audios": ["mul0012_3.wav"],
        "solution": "<transcript>\nUser: i want a cheap restaurant\n</transcript>\n<answer>set(restaurant.pricerange=cheap)</answer>",
        "belief_state": '{"restaurant": {"pricerange": "cheap"}}',
        "prev_belief_state": "{}",
        "dialogue_id": "mul0012",
        "turn_idx": 0,
        "sys_text": "how can i help",
        "opening_user_text": "hi",
    }


# ---------------------------------------------------------------------------
# Audio Flamingo 3
# ---------------------------------------------------------------------------

def test_af3_inference_conversation_has_text_then_audio():
    conv = grpo_to_af3_conversation(_sample(), audio_base_dir=None, include_solution=False)
    assert len(conv) == 1
    assert conv[0]["role"] == "user"
    parts = conv[0]["content"]
    # system prompt folded into leading text, then the audio part
    assert parts[0]["type"] == "text"
    assert "You are a DST model." in parts[0]["text"]
    assert "[Dialogue History]" in parts[0]["text"]
    assert parts[-1]["type"] == "audio"
    assert parts[-1]["path"].endswith("mul0012_3.wav")
    # the <audio> placeholder must not leak into any text part
    assert all("<audio>" not in p.get("text", "") for p in parts)


def test_af3_training_conversation_appends_assistant_solution():
    conv = grpo_to_af3_conversation(_sample(), audio_base_dir=None, include_solution=True)
    assert conv[-1]["role"] == "assistant"
    assert conv[-1]["content"][0]["text"].startswith("<transcript>")


def test_af3_audio_base_dir_is_applied():
    conv = grpo_to_af3_conversation(_sample(), audio_base_dir="/data/af3", include_solution=False)
    audio_part = conv[0]["content"][-1]
    assert audio_part["type"] == "audio"
    assert "/data/af3/" in audio_part["path"] or audio_part["path"].endswith("mul0012_3.wav")


def test_af3_multiple_audios_interleave():
    sample = _sample()
    sample["audios"] = ["a.wav", "b.wav"]
    sample["messages"][1]["content"] = "before <audio> middle <audio> after"
    conv = grpo_to_af3_conversation(sample, audio_base_dir=None, include_solution=False)
    types = [p["type"] for p in conv[0]["content"]]
    assert types == ["text", "audio", "text", "audio", "text"]


# ---------------------------------------------------------------------------
# Kimi-Audio
# ---------------------------------------------------------------------------

def test_kimia_conversation_splits_text_and_audio_turns():
    conv = grpo_to_kimia_conversation(_sample(), audio_base_dir=None, include_solution=True)
    # text user turn, audio user turn, assistant text turn
    assert conv[0] == {
        "role": "user",
        "message_type": "text",
        "content": conv[0]["content"],
    }
    assert "You are a DST model." in conv[0]["content"]
    assert "<audio>" not in conv[0]["content"]
    assert conv[1]["role"] == "user"
    assert conv[1]["message_type"] == "audio"
    assert conv[1]["content"].endswith("mul0012_3.wav")
    assert conv[2]["role"] == "assistant"
    assert conv[2]["message_type"] == "text"
    assert conv[2]["content"].startswith("<transcript>")


def test_kimia_inference_omits_assistant_turn():
    conv = grpo_to_kimia_conversation(_sample(), audio_base_dir=None, include_solution=False)
    assert all(turn["role"] != "assistant" for turn in conv)


def test_kimia_multiple_audios_each_become_a_turn():
    sample = _sample()
    sample["audios"] = ["a.wav", "b.wav"]
    conv = grpo_to_kimia_conversation(sample, audio_base_dir=None, include_solution=False)
    audio_turns = [t for t in conv if t["message_type"] == "audio"]
    assert len(audio_turns) == 2
