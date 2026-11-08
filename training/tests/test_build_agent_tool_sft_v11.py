"""Tests for the v11 verbatim-memory-value family builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

import build_agent_tool_sft_v11 as v11  # noqa: E402


def rows() -> dict[str, list[dict[str, Any]]]:
    return v11.collect_rows()


def all_rows() -> list[dict[str, Any]]:
    return [item for split_rows in rows().values() for item in split_rows]


def tool_arguments(item: dict[str, Any]) -> dict[str, str]:
    call = next(
        message for message in item["messages"] if message.get("role") == "assistant" and message.get("tool_calls")
    )
    return json.loads(call["tool_calls"][0]["function"]["arguments"])


def test_every_value_is_a_multi_word_phrase() -> None:
    for value in v11.VALUES:
        assert len(value.split()) >= v11.MIN_VALUE_WORDS, value


def test_values_are_not_the_evaluation_fixtures() -> None:
    assert not set(v11.VALUES) & v11.FORBIDDEN_VALUES
    assert not {v11.v4.snake(label) for label in v11.KEYS} & v11.FORBIDDEN_KEYS


def test_argument_carries_the_value_word_for_word() -> None:
    for item in all_rows():
        arguments = tool_arguments(item)
        assert arguments["value"] in v11.VALUES, item["id"]
        assert arguments["value"] in item["messages"][1]["content"], item["id"]


def test_argument_keeps_the_leading_modifier() -> None:
    """The v15 failure dropped the first word of a three word value."""
    for item in all_rows():
        value = tool_arguments(item)["value"]
        first_word = value.split()[0]
        assert value.startswith(first_word), item["id"]
        assert len(value.split()) >= v11.MIN_VALUE_WORDS, item["id"]


def test_argument_strips_the_indefinite_article() -> None:
    for item in all_rows():
        arguments = tool_arguments(item)
        assert not arguments["value"].startswith(("a ", "an ")), item["id"]
        assert arguments["key"] in {v11.v4.snake(label) for label in v11.KEYS}, item["id"]


def test_argument_does_not_weld_the_key_label_onto_the_value() -> None:
    for item in all_rows():
        arguments = tool_arguments(item)
        label = next(label for label in v11.KEYS if v11.v4.snake(label) == arguments["key"])
        assert label not in arguments["value"], item["id"]


def test_tool_reply_is_the_harness_memory_payload() -> None:
    for item in all_rows():
        arguments = tool_arguments(item)
        expected = json.dumps(
            {"status": "ok", "key": arguments["key"], "value": arguments["value"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        replies = [message["content"] for message in item["messages"] if message.get("role") == "tool"]
        assert replies == [expected], item["id"]


def test_rows_end_with_an_assistant_confirmation() -> None:
    for item in all_rows():
        assert item["messages"][-1]["role"] == "assistant"
        assert not item["messages"][-1].get("tool_calls"), item["id"]


def test_family_covers_every_key_and_value_combination() -> None:
    collected = all_rows()
    assert len(collected) == v11.EXAMPLES
    keys = {tool_arguments(item)["key"] for item in collected}
    values = {tool_arguments(item)["value"] for item in collected}
    assert keys == {v11.v4.snake(label) for label in v11.KEYS}
    assert values == set(v11.VALUES)


def test_splits_are_disjoint_and_every_split_carries_values() -> None:
    splits = rows()
    assert sum(len(split_rows) for split_rows in splits.values()) == v11.EXAMPLES
    seen: set[str] = set()
    for split_rows in splits.values():
        ids = {item["id"] for item in split_rows}
        assert not ids & seen
        seen |= ids
        assert {tool_arguments(item)["value"] for item in split_rows}
    assert len(splits["train"]) == len(v11.SPLITS["train"])


def test_every_ask_mentions_the_value_it_stores() -> None:
    for item in all_rows():
        assert tool_arguments(item)["value"] in item["messages"][1]["content"], item["id"]


def test_audit_rejects_a_stored_value_missing_its_modifier() -> None:
    item = v11.make_memory(0)
    value = tool_arguments(item)["value"]
    truncated = value.split(maxsplit=1)[1]
    call = next(message for message in item["messages"] if message.get("tool_calls"))
    call["tool_calls"][0]["function"]["arguments"] = json.dumps({"key": "summary_layout", "value": truncated})
    with pytest.raises(ValueError, match="not a pool value"):
        v11.audit_verbatim_value(item)


def test_audit_rejects_an_evaluation_value() -> None:
    item = v11.make_memory(0)
    forbidden = next(iter(v11.FORBIDDEN_VALUES))
    call = next(message for message in item["messages"] if message.get("tool_calls"))
    call["tool_calls"][0]["function"]["arguments"] = json.dumps({"key": "summary_layout", "value": forbidden})
    with pytest.raises(ValueError, match="evaluation memory value"):
        v11.audit_verbatim_value(item)


def test_audit_rejects_an_evaluation_key() -> None:
    item = v11.make_memory(0)
    call = next(message for message in item["messages"] if message.get("tool_calls"))
    call["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"key": next(iter(v11.FORBIDDEN_KEYS)), "value": "single page summary"}
    )
    with pytest.raises(ValueError, match="evaluation memory key"):
        v11.audit_verbatim_value(item)


def test_audit_rejects_a_value_that_kept_the_article() -> None:
    item = v11.make_memory(0)
    call = next(message for message in item["messages"] if message.get("tool_calls"))
    call["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"key": "summary_layout", "value": "a single page summary"}
    )
    with pytest.raises(ValueError, match="not a pool value"):
        v11.audit_verbatim_value(item)


def test_build_refuses_to_overwrite_an_existing_release(tmp_path: Path) -> None:
    target = tmp_path / "release"
    target.mkdir()
    (target / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        v11.build(target)


def test_build_writes_every_split(tmp_path: Path) -> None:
    manifest = v11.build(tmp_path / "release")
    assert manifest["example_count"] == v11.EXAMPLES
    for name, info in manifest["splits"].items():
        assert info["row_count"] == len(v11.SPLITS[name])
        path = tmp_path / "release" / name / "trajectories.jsonl"
        assert path.is_file()
        assert len(path.read_text(encoding="utf-8").splitlines()) == info["row_count"]
