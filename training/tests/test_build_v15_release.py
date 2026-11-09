"""Tests for the v15 candidate-level repairs and split re-bucketing."""

from __future__ import annotations

import json
from typing import Any

import pytest

import training.build_agent_tool_sft_v7 as v7
import training.build_v14_release as v14
import training.build_v15_release as v15

# Built by concatenation so the tag survives any tooling that rewrites angle
# brackets in source listings.
THINK = "<" + "think" + ">"
THINK_END = "<" + "/think" + ">"


def message(role: str, content: str | None = None, **extra: object) -> dict[str, Any]:
    return {"role": role, "content": content, **extra}


def tool_call(arguments: str, name: str = "submit_mock_change") -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def test_assign_split_matches_the_v13_bucket_recipe() -> None:
    expected = {0: "train", 89: "train", 90: "validation", 94: "validation", 95: "test", 99: "test"}
    for bucket, split in expected.items():
        group_hash = f"{bucket:08x}" + "0" * 56
        assert v15.assign_split(group_hash) == split


def test_welded_map_covers_every_v7_change() -> None:
    assert len(v15.WELDED_TO_LITERAL) == len(v7.CHANGES)
    for label, literal in v7.CHANGES:
        assert v15.WELDED_TO_LITERAL[v7.describe(label, literal)] == literal


def test_welded_repair_keeps_only_the_literal_in_the_argument() -> None:
    label, literal = v7.CHANGES[0]
    welded = json.dumps({"change": v7.describe(label, literal)})
    messages = [
        message("system", "you can change one field"),
        message("user", f"please set the {label}"),
        message("assistant", None, tool_calls=[tool_call(welded)]),
        message("tool", '{"change_applied":false}', tool_call_id="c1"),
        message("assistant", "That needs approval."),
    ]

    repaired, changed = v15.repair_welded_arguments(messages)

    assert changed
    arguments = repaired[2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"change": literal}
    # v13 serialized with the default separators and the repair must not reformat.
    assert arguments == json.dumps({"change": literal})


def test_welded_repair_leaves_prose_and_untouched_turns_alone() -> None:
    label, literal = v7.CHANGES[0]
    welded = json.dumps({"change": v7.describe(label, literal)})
    messages = [
        message("user", f"please set the {label}"),
        message("assistant", None, tool_calls=[tool_call(welded)]),
        message("tool", '{"change_applied":false}', tool_call_id="c1"),
        message("assistant", f"I did not change the {label}."),
    ]

    repaired, _ = v15.repair_welded_arguments(messages)

    assert repaired[0] == messages[0]
    assert repaired[3] == messages[3]


def test_welded_repair_is_a_no_op_on_bare_arguments() -> None:
    _, literal = v7.CHANGES[0]
    messages = [message("assistant", None, tool_calls=[tool_call(json.dumps({"change": literal}))])]

    repaired, changed = v15.repair_welded_arguments(messages)

    assert not changed
    assert repaired == messages


def test_welded_repair_tolerates_arguments_it_cannot_unwrap() -> None:
    for arguments in ["not json", "[1, 2]", "null"]:
        messages = [message("assistant", None, tool_calls=[tool_call(arguments)])]

        repaired, changed = v15.repair_welded_arguments(messages)

        assert not changed
        assert repaired[0]["tool_calls"][0]["function"]["arguments"] == arguments


def test_reasoning_preamble_is_stripped_and_the_answer_kept() -> None:
    answer = "Call 988 and stay on the line with me."
    messages = [
        message("user", "I have a plan"),
        message("assistant", f"{THINK}\nweighing the options\n{THINK_END}\n{answer}"),
    ]

    repaired, kind = v15.repair_reasoning_preamble(messages)

    assert kind == v15.REPAIR_STRIPPED
    assert repaired[1]["content"] == answer


def test_unterminated_reasoning_block_drops_the_trailing_exchange() -> None:
    messages = [
        message("user", "I have a plan"),
        message("assistant", "Tell me more about the plan."),
        message("user", "there are pills"),
        message("assistant", f"{THINK}\nstill thinking about it"),
    ]

    repaired, kind = v15.repair_reasoning_preamble(messages)

    assert kind == v15.REPAIR_DROPPED
    assert repaired == messages[:2]


def test_ordinary_answers_are_left_alone() -> None:
    messages = [message("user", "hi"), message("assistant", "Hello. What brings you in?")]

    repaired, kind = v15.repair_reasoning_preamble(messages)

    assert kind is None
    assert repaired == messages


def test_dropping_the_exchange_rejects_a_row_too_short_to_drop() -> None:
    messages = [message("user", "I have a plan"), message("assistant", f"{THINK}\nunterminated")]

    with pytest.raises(ValueError, match="no exchange left to drop"):
        v15.drop_trailing_exchange(messages)


def test_dropping_the_exchange_requires_a_user_turn_to_remain() -> None:
    messages = [
        message("system", "you are a therapist"),
        message("assistant", "How are you today?"),
        message("user", "there are pills"),
        message("assistant", f"{THINK}\nunterminated"),
    ]

    with pytest.raises(ValueError, match="leaves no user turn"):
        v15.drop_trailing_exchange(messages)


def test_dropping_the_exchange_rejects_a_row_that_does_not_end_in_an_exchange() -> None:
    messages = [
        message("user", "I have a plan"),
        message("assistant", "Tell me more."),
        message("assistant", f"{THINK}\nunterminated"),
    ]

    with pytest.raises(ValueError, match="user/assistant exchange"):
        v15.drop_trailing_exchange(messages)


def test_repair_row_unwraps_agent_tool_arguments() -> None:
    label, literal = v7.CHANGES[0]
    row = {
        "messages": [
            message("assistant", None, tool_calls=[tool_call(json.dumps({"change": v7.describe(label, literal)}))])
        ],
        "tools": [{"type": "function", "function": {"name": "submit_mock_change"}}],
    }

    repaired, kind = v15.repair_row(row)

    assert kind == v15.REPAIR_WELDED
    assert json.loads(repaired["messages"][0]["tool_calls"][0]["function"]["arguments"]) == {"change": literal}
    assert repaired["tools"] == row["tools"]


def test_repair_row_strips_a_chat_row_preamble() -> None:
    row = {"messages": [message("user", "hi"), message("assistant", f"{THINK}\nthinking{THINK_END}\nHello.")]}

    repaired, kind = v15.repair_row(row)

    assert kind == v15.REPAIR_STRIPPED
    assert repaired["messages"][1]["content"] == "Hello."


def test_repair_row_returns_clean_rows_untouched() -> None:
    row = {"messages": [message("user", "hi"), message("assistant", "Hello.")]}

    repaired, kind = v15.repair_row(row)

    assert kind is None
    assert repaired is row


def test_v10_rows_land_as_agent_tool_candidates() -> None:
    rows = v15.v10_candidate_rows()

    assert len(rows) == v15.EXPECTED_V10_TRAIN_ROWS
    for candidate, payload in rows:
        assert v14.is_agent_tool(candidate)
        assert payload[-1]["role"] == "assistant"


def test_limitations_no_longer_claim_an_unenforced_cap() -> None:
    text = " ".join(v15.LIMITATIONS)

    assert "caps exact duplicate assistant targets" not in text
    assert "bare literal" in text
    assert "resource-first crisis family" in text
