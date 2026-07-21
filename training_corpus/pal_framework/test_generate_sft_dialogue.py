from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import generate_sft_dialogue as g

PERSONA = {
    "demographics": {"age": 45, "gender": "female", "location": "Hanoi"},
    "healthcare_behavior": {"health_literacy": "low", "preference": "traditional medicine"},
}
PERSONA_STRING = (
    "This patient is a 45-year-old female from Hanoi with low health literacy who prefers traditional medicine."
)
DIALOGUE = "Doctor: How have you been feeling?\nPatient: Better, thank you."
RESPONSE = "I take the herbs you suggested and feel calmer."


def test_build_sft_messages_basic():
    messages = g.build_sft_messages(PERSONA_STRING, DIALOGUE, RESPONSE)
    assert g.is_chatml_compliant(messages)
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == RESPONSE
    user = messages[1]["content"]
    assert "Given this persona:" in user
    assert PERSONA_STRING in user
    assert "Dialogue history:" in user
    assert DIALOGUE in user
    assert "Generate the next response." in user


def test_build_sft_messages_rejects_empty_persona():
    try:
        g.build_sft_messages("", DIALOGUE, RESPONSE)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_build_sft_messages_rejects_empty_response():
    try:
        g.build_sft_messages(PERSONA_STRING, DIALOGUE, "")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_no_json_leakage():
    example = g.build_sft_example(PERSONA, DIALOGUE, RESPONSE)
    user = example.messages[1]["content"]
    for ch in ("{", "}", "'", '"'):
        assert ch not in user


def test_format_dialogue_history_string():
    assert g.format_dialogue_history("hello") == "hello"
    assert g.format_dialogue_history("  spaced  ") == "spaced"


def test_format_dialogue_history_messages():
    turns = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = g.format_dialogue_history(turns)
    assert "User: hi" in out
    assert "Assistant: hello" in out


def test_format_dialogue_history_none():
    assert g.format_dialogue_history(None) == ""


def test_format_dialogue_history_invalid_type():
    invalid: Any = 123
    try:
        g.format_dialogue_history(invalid)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_is_chatml_compliant_rejects_garbage():
    assert not g.is_chatml_compliant([])
    assert not g.is_chatml_compliant([{"role": "bot", "content": "x"}])
    assert not g.is_chatml_compliant([{"role": "user", "content": 1}])


def test_estimate_tokens_positive():
    messages = g.build_sft_messages(PERSONA_STRING, DIALOGUE, RESPONSE)
    assert g.estimate_tokens(messages) > 0


def test_validate_token_bounds_ok():
    messages = g.build_sft_messages(PERSONA_STRING, DIALOGUE, RESPONSE)
    assert g.validate_token_bounds(messages, max_tokens=10_000)


def test_validate_token_bounds_exceeds():
    messages = g.build_sft_messages(PERSONA_STRING, DIALOGUE, RESPONSE)
    assert not g.validate_token_bounds(messages, max_tokens=1)


def test_build_sft_example_metadata():
    example = g.build_sft_example(PERSONA, DIALOGUE, RESPONSE)
    assert example.metadata["persona_string"] == PERSONA_STRING
    assert example.metadata["n_dialogue_turns"] == 1
    assert example.metadata["estimated_tokens"] > 0


def test_build_sft_example_oversize_raises():
    big_response = "x" * 10_000
    try:
        g.build_sft_example(PERSONA, DIALOGUE, big_response, max_tokens=100)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "token bound" in str(exc)


def test_generate_dataset_roundtrip(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    records = [
        {"persona": PERSONA, "dialogue": DIALOGUE, "response": RESPONSE},
        {"persona": PERSONA, "dialogue": None, "response": "I feel fine today."},
    ]
    inp.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    n = g.generate_dataset(inp, out, min_records=1)
    assert n == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert g.is_chatml_compliant(rec["messages"])
    assert rec["metadata"]["persona_string"] == PERSONA_STRING
    rec2 = json.loads(lines[1])
    assert g.is_chatml_compliant(rec2["messages"])
    assert rec2["metadata"]["n_dialogue_turns"] == 0


def test_generate_dataset_reject_oversize(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    big_response = "x" * 10_000
    records = [
        {"persona": PERSONA, "dialogue": DIALOGUE, "response": "ok"},
        {"persona": PERSONA, "dialogue": DIALOGUE, "response": big_response},
    ]
    inp.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    n = g.generate_dataset(inp, out, max_tokens=500, reject_oversize=True, min_records=1)
    assert n == 1


def test_generate_dataset_oversize_raises_without_flag(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    big_response = "x" * 10_000
    records = [
        {"persona": PERSONA, "dialogue": DIALOGUE, "response": big_response},
    ]
    inp.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    try:
        g.generate_dataset(inp, out, max_tokens=100, min_records=1)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "token bound" in str(exc)


def test_generate_dataset_below_minimum_raises(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    records = [{"persona": PERSONA, "dialogue": DIALOGUE, "response": RESPONSE}]
    inp.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    try:
        g.generate_dataset(inp, out, min_records=5000)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "5000" in str(exc)


def test_generate_dataset_skips_blank_lines(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    record = {"persona": PERSONA, "dialogue": DIALOGUE, "response": RESPONSE}
    inp.write_text(
        json.dumps(record) + "\n\n   \n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )
    n = g.generate_dataset(inp, out, min_records=1)
    assert n == 2


def test_generate_dataset_preserves_unicode(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    persona = {"demographics": {"age": 60, "gender": "female", "location": "Đà Nẵng"}}
    record = {"persona": persona, "dialogue": "Bác sĩ: Khỏe không?", "response": "Khỏe ạ."}
    inp.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    n = g.generate_dataset(inp, out, min_records=1)
    assert n == 1
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "Đà Nẵng" in rec["messages"][1]["content"]
