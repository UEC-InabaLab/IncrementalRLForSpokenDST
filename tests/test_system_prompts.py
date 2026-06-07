"""Tests verifying that system prompt files and data files exist at their expected paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_incremental_system_prompt_exists():
    p = ROOT / "prompts" / "incremental.txt"
    assert p.exists(), f"Missing: {p}"


def test_fullstate_system_prompt_exists():
    p = ROOT / "prompts" / "fullstate.txt"
    assert p.exists(), f"Missing: {p}"


def test_incremental_system_prompt_has_content():
    p = ROOT / "prompts" / "incremental.txt"
    content = p.read_text()
    assert "incremental" in content.lower() or "dst" in content.lower()
    assert "<transcript>" in content
    assert "<answer>" in content


def test_fullstate_system_prompt_has_content():
    p = ROOT / "prompts" / "fullstate.txt"
    content = p.read_text()
    assert "<transcript>" in content
    assert "<answer>" in content


def test_old_incremental_system_prompt_removed():
    p = ROOT / "prompts" / "incremental_baseline_sft.txt"
    assert not p.exists(), f"Old file still exists: {p}"


def test_old_fullstate_system_prompt_removed():
    p = ROOT / "prompts" / "fullstate_baseline.txt"
    assert not p.exists(), f"Old file still exists: {p}"


def test_reward_module_importable():
    """reward.py が正しいパスに存在することを確認。"""
    p = ROOT / "src" / "reward.py"
    assert p.exists(), f"Missing: {p}"


def test_old_dapo_reward_removed():
    p = ROOT / "src" / "dapo_reward.py"
    assert not p.exists(), f"Old file still exists: {p}"


def test_infer_script_exists():
    p = ROOT / "scripts" / "infer" / "infer.py"
    assert p.exists(), f"Missing: {p}"


def test_old_infer_vllm_removed():
    p = ROOT / "scripts" / "infer_vllm.py"
    assert not p.exists(), f"Old file still exists: {p}"


def test_infer_fullstate_script_exists():
    p = ROOT / "scripts" / "infer" / "infer_fullstate.py"
    assert p.exists(), f"Missing: {p}"


def test_old_infer_fullstate_vllm_removed():
    p = ROOT / "scripts" / "infer_fullstate_vllm.py"
    assert not p.exists(), f"Old file still exists: {p}"


def test_plot_script_exists():
    p = ROOT / "scripts" / "eval" / "plot_jga_by_turn.py"
    assert p.exists(), f"Missing: {p}"


def test_old_plot_paper_removed():
    p = ROOT / "scripts" / "plot_jga_by_turn_paper.py"
    assert not p.exists(), f"Old file still exists: {p}"


def test_test_data_exists():
    p = ROOT / "data" / "test.jsonl"
    assert p.exists(), f"Missing: {p}"


def test_old_test_data_removed():
    p = ROOT / "data" / "incremental_baseline_sft_test.jsonl"
    assert not p.exists(), f"Old file still exists: {p}"
