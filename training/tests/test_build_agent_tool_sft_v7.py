"""The v7 approval family must not teach a welded tool argument.

The family exists to teach the approval-refusal path. It reused one string for
both the request prose and the tool argument, so every row called
submit_mock_change with the field label glued to the value
({"change": 'retention window to "90 days"'} instead of {"change": "90 days"}).
Half of the literal-argument supervision in the v14 train payload came from
these rows, which is why adding 192 correct rows in v9 did not move the eval.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from training import build_agent_tool_sft_v7 as v7

# The strings the family emitted before the argument was separated from the prose.
LEGACY_DESCRIPTIONS = [
    'announcement to "Meeting moved"',
    'team calendar entry to "Standup at 9"',
    'status label to "In review"',
    'on-call rotation to "Week 34"',
    'project name to "Northwind Refresh"',
    'release notes header to "v3 draft"',
    'document owner to "Priya Raman"',
    'office hours slot to "Tuesdays 2 PM"',
    'sprint goal to "Reduce backlog"',
    'wiki page title to "Accessibility checklist"',
    'notification channel to "ops-alerts"',
    'retention window to "90 days"',
]


def _arguments(row: dict[str, Any]) -> dict[str, Any]:
    call = next(message for message in row["messages"] if message.get("tool_calls"))
    parsed: dict[str, Any] = json.loads(call["tool_calls"][0]["function"]["arguments"])
    return parsed


def test_change_pool_is_label_literal_pairs() -> None:
    for label, literal in v7.CHANGES:
        assert label
        assert literal
        # the label must not have leaked into the value side
        assert literal not in label


def test_describe_reproduces_the_legacy_request_wording() -> None:
    assert [v7.describe(label, literal) for label, literal in v7.CHANGES] == LEGACY_DESCRIPTIONS


def test_tool_argument_carries_only_the_literal() -> None:
    for index in range(v7.EXAMPLES_PER_FAMILY):
        literal = v7.CHANGES[(index // len(v7.ASKS)) % len(v7.CHANGES)][1]
        assert _arguments(v7.make_approval(index)) == {"change": literal}


def test_request_and_refusal_prose_still_name_the_field() -> None:
    row = v7.make_approval(0)
    label, literal = v7.CHANGES[0]
    request = row["messages"][1]["content"]
    prose = row["messages"][-1]["content"]
    assert v7.describe(label, literal) in request
    assert label in prose


def test_every_row_calls_the_tool_exactly_once() -> None:
    for index in [0, 11, 12, 137, v7.EXAMPLES_PER_FAMILY - 1]:
        calls = [m for m in v7.make_approval(index)["messages"] if m.get("tool_calls")]
        assert len(calls) == 1
        assert calls[0]["tool_calls"][0]["function"]["name"] == v7.TOOL_NAME
