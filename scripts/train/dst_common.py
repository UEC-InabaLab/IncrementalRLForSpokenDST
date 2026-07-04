"""Shared helpers for converting spoken-DST datasets to GRPO-format JSONL.

Dataset-specific scripts (prepare_data*.py) parse their own raw format into
a per-dialogue list of turns with a flattened belief state ({domain: {slot:
value}}), then call the functions below to build the diff-operation targets
and GRPO-format sample records shared across all datasets.
"""

import json
from pathlib import Path


def flatten_multiwoz_metadata(metadata: dict) -> dict:
    """Flatten MultiWOZ 2.1-style nested metadata to {domain: {slot: value}}.

    Input:  {"hotel": {"semi": {"pricerange": "cheap", "area": ""}, "book": {...}}}
    Output: {"hotel": {"pricerange": "cheap"}}

    Filters out empty strings, "not mentioned", and the "booked" list field.
    Shared by any dataset built on the MultiWOZ ontology (SpokenWOZ, DSTC-11
    Spoken MultiWOZ).
    """
    flat: dict = {}
    for domain, domain_data in metadata.items():
        if not isinstance(domain_data, dict):
            continue
        slots: dict = {}
        if "semi" in domain_data or "book" in domain_data:
            for section in ("semi", "book"):
                for slot, value in domain_data.get(section, {}).items():
                    if slot == "booked" or not value or value == "not mentioned":
                        continue
                    slots[slot] = value
        else:
            for slot, value in domain_data.items():
                if not value or value == "not mentioned":
                    continue
                slots[slot] = value
        if slots:
            flat[domain] = slots
    return flat


def compute_diff_ops(prev: dict, curr: dict) -> list[str]:
    """Return ordered list of diff operations (set/update/delete) from prev to curr."""
    ops: list[str] = []
    for domain in sorted(set(list(prev.keys()) + list(curr.keys()))):
        prev_slots = prev.get(domain, {})
        curr_slots = curr.get(domain, {})
        for slot in sorted(curr_slots.keys()):
            value = curr_slots[slot]
            if slot not in prev_slots:
                ops.append(f"set({domain}.{slot}={value})")
            elif prev_slots[slot] != value:
                ops.append(f"update({domain}.{slot}={value})")
        for slot in sorted(prev_slots.keys()):
            if slot not in curr_slots:
                ops.append(f"delete({domain}.{slot})")
    return ops


def build_user_message(history_lines: list[str], prev_state: dict) -> str:
    """Build GRPO-format user message.

    Format:
        [Dialogue History]        (omitted when empty)
        User: ...
        System: ...
        ...
        System: ...   <- includes current system turn

        [Previous State]
        {"domain": {"slot": "value"}}

        [New Audio]
        <audio>
    """
    parts: list[str] = []
    if history_lines:
        parts.append("[Dialogue History]")
        parts.extend(history_lines)
        parts.append("")
    parts.append("[Previous State]")
    parts.append(json.dumps(prev_state, ensure_ascii=False))
    parts.append("")
    parts.append("[New Audio]")
    parts.append("<audio>")
    return "\n".join(parts)


def build_solution(sys_text: str, user_text: str, prev: dict, curr: dict) -> str:
    """Build gold solution string."""
    ops = compute_diff_ops(prev, curr)
    answer = "\n".join(ops)
    transcript = f"System: {sys_text}\nUser: {user_text}"
    return f"<transcript>\n{transcript}\n</transcript>\n<answer>{answer}</answer>"


def load_system_prompt(default_path: Path, override: str | None = None) -> str:
    path = Path(override) if override else default_path
    return path.read_text(encoding="utf-8").strip()


def write_jsonl(samples: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
