"""Tests for the v16 release: v15's pipeline plus the v11 and v12 addenda."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import training.build_agent_tool_sft_v11 as v11
import training.build_agent_tool_sft_v12 as v12
import training.build_v16_release as v16


def arguments_of(row: dict[str, Any]) -> dict[str, Any]:
    """The tool-call arguments of the supervised assistant turn."""
    messages = row["messages"]
    call = messages[2]["tool_calls"][0]["function"]
    assert call["name"] == v11.TOOL_NAME
    parsed: dict[str, Any] = json.loads(call["arguments"])
    return parsed


def test_the_addendum_is_the_v11_train_split() -> None:
    total = sum(len(rows) for rows in v11.collect_rows().values())
    assert len(v16.v11_candidate_rows()) == v16.EXPECTED_V11_TRAIN_ROWS
    assert total > v16.EXPECTED_V11_TRAIN_ROWS


def test_v11_candidate_rows_are_agent_tool_rows() -> None:
    for candidate, payload in v16.v11_candidate_rows():
        assert v16.v14.is_agent_tool(candidate)
        assert candidate["tools"] == v11.TOOLS
        assert payload == v16.v14.project_agent(candidate["messages"])
        assert payload[-1]["role"] == "assistant"


def test_the_tool_argument_carries_the_bare_value() -> None:
    for candidate, _ in v16.v11_candidate_rows():
        arguments = arguments_of(candidate)
        assert set(arguments) == {"key", "value"}
        assert arguments["value"] in v11.VALUES
        assert not arguments["value"].startswith(("a ", "an "))
        assert len(arguments["value"].split()) >= v11.MIN_VALUE_WORDS
        assert arguments["value"] == arguments["value"].strip()


def test_the_key_is_the_snake_case_label_the_ask_names() -> None:
    for candidate, _ in v16.v11_candidate_rows():
        arguments = arguments_of(candidate)
        assert arguments["key"] in {v11.v4.snake(label) for label in v11.KEYS}
        ask = candidate["messages"][1]["content"]
        assert arguments["value"] in ask
        assert arguments["key"] in ask


def test_the_pool_never_reuses_the_eval_fixtures() -> None:
    for candidate, _ in v16.v11_candidate_rows():
        arguments = arguments_of(candidate)
        assert arguments["key"] not in v11.FORBIDDEN_KEYS
        assert arguments["value"] not in v11.FORBIDDEN_VALUES


def test_the_v12_addendum_is_the_v12_train_split() -> None:
    rows = v12.collect_rows()
    assert len(v16.v12_candidate_rows()) == v16.EXPECTED_V12_TRAIN_ROWS
    assert sum(len(split) for split in rows.values()) == v16.EXPECTED_V12_TRAIN_ROWS
    assert set(rows) == {"train"}


def test_v12_candidate_rows_are_chat_rows() -> None:
    for candidate, payload in v16.v12_candidate_rows():
        assert not v16.v14.is_agent_tool(candidate)
        assert "tools" not in candidate
        assert candidate["source"] == v16.V12_SOURCE_LABEL
        assert payload[-1]["role"] == "assistant"
        assert payload == v16.v14.project_chat(candidate["messages"], Counter())


def test_v12_rows_project_without_redaction() -> None:
    """A redacted v12 row would reach train mangled, so the addendum fails instead."""
    for _candidate, payload in v16.v12_candidate_rows():
        for message in payload:
            content = message.get("content")
            if isinstance(content, str):
                assert "REDACTED" not in content


def test_the_two_addenda_are_labelled_apart() -> None:
    assert v16.V11_SOURCE_LABEL != v16.V12_SOURCE_LABEL
    assert v16.V12_SOURCE_LABEL not in {
        v16.v14.V9_SOURCE_LABEL,
        v16.v15.V10_SOURCE_LABEL,
        v16.V11_SOURCE_LABEL,
    }


def test_the_release_identity_names_v16() -> None:
    assert v16.RELEASE_ID.endswith("-v16")
    assert v16.RELEASE_ID in str(v16.OUTPUT)
    assert v16.RELEASE_ID in str(v16.MANIFEST)
    assert v16.SHUFFLE_SEED not in {v16.v15.SHUFFLE_SEED, v16.v14.SHUFFLE_SEED}


def test_limitations_extend_v15_without_dropping_its_entries() -> None:
    assert v16.LIMITATIONS[: len(v16.v15.LIMITATIONS)] == v16.v15.LIMITATIONS
    assert len(v16.LIMITATIONS) > len(v16.v15.LIMITATIONS)
    added = " ".join(v16.LIMITATIONS[len(v16.v15.LIMITATIONS) :])
    assert "v11" in added
    assert "v12" in added


def sample_manifest() -> dict[str, Any]:
    """The v15-shaped manifest the v16 stamping has to correct."""
    return {
        "release_id": v16.v15.RELEASE_ID,
        "created_at_utc": v16.v15.CREATED_AT_UTC,
        "scope": {
            "appended_rows": 1_104,
            "appended_rows_by_source": {
                v16.v14.V9_SOURCE_LABEL: 624,
                v16.v15.V10_SOURCE_LABEL: 480,
            },
        },
        "payload_contract": {
            "shuffle": {"train": True, "validation": False, "test": False, "seed": v16.v15.SHUFFLE_SEED},
            "agent_tool_note": "2,807 v13 agent_tool rows plus 624 v9 rows plus 480 v10 rows",
        },
        "limitations": list(v16.v15.LIMITATIONS),
        "owner_direction": "User-directed v15 after the v14 paired eval.",
    }


def test_stamping_rewrites_every_addenda_dependent_field() -> None:
    stamped = v16.stamp_release_identity(sample_manifest(), v9_rows=624, v10_rows=480, v11_rows=240, v12_rows=672)
    assert stamped["release_id"] == v16.RELEASE_ID
    assert stamped["created_at_utc"] == v16.CREATED_AT_UTC
    assert stamped["scope"]["appended_rows_by_source"] == {
        v16.v14.V9_SOURCE_LABEL: 624,
        v16.v15.V10_SOURCE_LABEL: 480,
        v16.V11_SOURCE_LABEL: 240,
        v16.V12_SOURCE_LABEL: 672,
    }
    assert stamped["scope"]["appended_rows"] == sum(stamped["scope"]["appended_rows_by_source"].values())
    assert stamped["payload_contract"]["shuffle"]["seed"] == v16.SHUFFLE_SEED
    assert "240 v11 rows" in stamped["payload_contract"]["agent_tool_note"]
    # v12 is chat-shaped, so the agent-tool note must not claim its rows.
    assert "672 v12 rows" not in stamped["payload_contract"]["agent_tool_note"]
    assert stamped["limitations"] == list(v16.LIMITATIONS)
    assert stamped["owner_direction"] != sample_manifest()["owner_direction"]


def test_stamping_keeps_the_counts_it_is_given() -> None:
    counts = {
        v16.v14.V9_SOURCE_LABEL: 1,
        v16.v15.V10_SOURCE_LABEL: 2,
        v16.V11_SOURCE_LABEL: 3,
        v16.V12_SOURCE_LABEL: 4,
    }
    stamped = v16.stamp_release_identity(sample_manifest(), v9_rows=1, v10_rows=2, v11_rows=3, v12_rows=4)
    assert stamped["scope"]["appended_rows_by_source"] == counts
    assert stamped["scope"]["appended_rows"] == sum(counts.values())


def test_stamping_replaces_the_limitations_rather_than_extending_them() -> None:
    manifest = sample_manifest()
    manifest["limitations"] = ["stale claim from a previous release"]
    stamped = v16.stamp_release_identity(manifest, v9_rows=624, v10_rows=480, v11_rows=240, v12_rows=672)
    assert stamped["limitations"] == list(v16.LIMITATIONS)
    assert "stale claim from a previous release" not in stamped["limitations"]
