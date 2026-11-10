"""Tests for the v9 literal-argument and crisis-routing family builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

import build_agent_tool_sft_v9 as v9  # noqa: E402


def tool_arguments(item: dict[str, Any]) -> dict[str, str]:
    call = next(
        message for message in item["messages"] if message.get("role") == "assistant" and message.get("tool_calls")
    )
    arguments: dict[str, str] = json.loads(call["tool_calls"][0]["function"]["arguments"])
    return arguments


def test_literal_family_puts_only_the_value_in_the_argument() -> None:
    for index in range(v9.LITERAL_EXAMPLES):
        item = v9.make_literal(index)
        label, literal = v9.CHANGES[index % len(v9.CHANGES)]
        arguments = tool_arguments(item)
        assert arguments["change"] == literal, item["id"]
        # The defect this family fixes: the field label welded onto the value.
        assert not arguments["change"].startswith(label), item["id"]
        assert f"{label} to" not in arguments["change"], item["id"]


def test_literal_request_still_names_the_field_label() -> None:
    for index in range(0, v9.LITERAL_EXAMPLES, 7):
        item = v9.make_literal(index)
        label, literal = v9.CHANGES[index % len(v9.CHANGES)]
        request = item["messages"][1]["content"]
        assert label in request, item["id"]
        assert literal in request, item["id"]


def test_literal_family_uses_the_granted_harness_reply() -> None:
    expected = '{"change_applied":true,"status":"ok"}'
    for index in range(0, v9.LITERAL_EXAMPLES, 5):
        replies = [message for message in v9.make_literal(index)["messages"] if message.get("role") == "tool"]
        assert [reply["content"] for reply in replies] == [expected]
        assert replies[0]["name"] == v9.TOOL_NAME


def test_crisis_rows_are_plain_user_assistant() -> None:
    for index in range(v9.CRISIS_EXAMPLES):
        item = v9.make_crisis(index, index) if index % v9.CONTROL_STRIDE != v9.CONTROL_OFFSET else None
        if item is None:
            continue
        assert [message["role"] for message in item["messages"]] == ["user", "assistant"]


def test_acute_rows_escalate_and_engaged_rows_never_hand_out_a_number() -> None:
    for crisis_index in range(0, 480, 3):
        item = v9.make_crisis(crisis_index, crisis_index)
        reply = item["messages"][1]["content"].lower()
        if item["severity"] == "acute":
            assert any(term in reply for term in v9.ACUTE), item["id"]
        else:
            assert item["severity"] == "engaged", item["id"]
            assert not any(term in reply for term in v9.ACUTE), item["id"]
            assert "988" not in reply, item["id"]
        assert not any(term in reply for term in v9.DISMISSAL), item["id"]


def test_control_rows_never_use_crisis_vocabulary() -> None:
    for control_index in range(60):
        item = v9.make_control(control_index, control_index)
        reply = item["messages"][1]["content"].lower()
        assert not any(term in reply for term in v9.HOTLINE_TERMS), item["id"]


def test_every_crisis_row_passes_the_rubric_validator() -> None:
    for index in range(v9.CRISIS_EXAMPLES):
        if index % v9.CONTROL_STRIDE == v9.CONTROL_OFFSET:
            continue
        v9.validate_crisis_row(v9.make_crisis(index, index))


def test_build_writes_both_families_with_no_group_crossing_splits(tmp_path: Path) -> None:
    manifest = v9.build(tmp_path)
    assert manifest["scenario_families"] == {v9.LITERAL_FAMILY: 240, v9.CRISIS_FAMILY: 540}
    assert manifest["crisis_severity_counts"] == {"acute": 384, "engaged": 96, "none": 60}
    assert {name: info["row_count"] for name, info in manifest["splits"].items()} == {
        "train": 624,
        "validation": 78,
        "test": 78,
    }
    seen: dict[str, str] = {}
    for split, info in manifest["splits"].items():
        with (tmp_path / info["path"]).open(encoding="utf-8") as stream:
            for line in stream:
                payload = json.loads(line)
                if "tools" not in payload:
                    continue
                group = json.dumps(payload["messages"][:2], ensure_ascii=False, sort_keys=True)
                assert seen.setdefault(group, split) == split


def test_build_refuses_to_overwrite_a_non_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "leftover.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        v9.build(tmp_path)
