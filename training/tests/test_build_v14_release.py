"""Tests for the v14 release builder's projection and purge rules."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

AI_DIR = Path(__file__).resolve().parents[2]
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

import training.build_v14_release as v14  # noqa: E402


def message(role: str, content: str | None = None, **extra: object) -> dict:
    return {"role": role, "content": content, **extra}


def test_line_sha_excludes_the_trailing_newline():
    body = b'{"a":1}'
    assert v14.line_sha(body + b"\n") == hashlib.sha256(body).hexdigest()
    assert v14.line_sha(body) == hashlib.sha256(body).hexdigest()


def test_agent_tool_discriminator_needs_tools_and_no_source():
    assert v14.is_agent_tool({"messages": [], "tools": []})
    assert not v14.is_agent_tool({"messages": [], "tools": [], "source": "crisis_benchmark"})
    assert not v14.is_agent_tool({"messages": []})


def test_crisis_placeholder_matches_only_the_fabricated_target():
    row = {
        "source": "crisis_benchmark",
        "messages": [message("user", "I have a plan"), message("assistant", v14.CRISIS_PLACEHOLDER)],
    }
    assert v14.is_crisis_placeholder(row)


def test_crisis_placeholder_ignores_a_real_target():
    row = {
        "source": "crisis_benchmark",
        "messages": [message("user", "I have a plan"), message("assistant", "Call 911 now.")],
    }
    assert not v14.is_crisis_placeholder(row)


def test_crisis_placeholder_ignores_other_sources():
    row = {
        "source": "empath",
        "messages": [message("user", "hi"), message("assistant", v14.CRISIS_PLACEHOLDER)],
    }
    assert not v14.is_crisis_placeholder(row)


def test_crisis_placeholder_ignores_the_string_in_a_user_turn():
    row = {
        "source": "crisis_benchmark",
        "messages": [message("user", v14.CRISIS_PLACEHOLDER), message("assistant", "Call 911 now.")],
    }
    assert not v14.is_crisis_placeholder(row)


def test_project_chat_drops_system_and_weights_the_turns():
    counts: Counter[str] = Counter()
    projected = v14.project_chat(
        [message("system", "you are helpful"), message("user", "hello"), message("assistant", "hi")],
        counts,
    )
    assert projected == [
        {"role": "user", "content": "hello", "weight": 0},
        {"role": "assistant", "content": "hi", "weight": 1},
    ]


def test_project_chat_truncates_after_the_last_assistant():
    counts: Counter[str] = Counter()
    projected = v14.project_chat(
        [message("user", "one"), message("assistant", "two"), message("user", "three")],
        counts,
    )
    assert [item["role"] for item in projected] == ["user", "assistant"]


def test_project_chat_requires_an_assistant_and_a_user():
    counts: Counter[str] = Counter()
    assert v14.project_chat([message("user", "hello")], counts) is None
    assert v14.project_chat([message("assistant", "hi")], counts) is None


def test_project_chat_skips_blank_and_non_text_turns():
    counts: Counter[str] = Counter()
    projected = v14.project_chat(
        [message("user", "   "), message("user", None), {"role": "user"}, message("assistant", "hi")],
        counts,
    )
    assert projected is None


def test_project_chat_redacts_pii_and_counts_it():
    counts: Counter[str] = Counter()
    projected = v14.project_chat(
        [message("user", "write to jane@example.com"), message("assistant", "ok")],
        counts,
    )
    assert projected is not None
    assert "jane@example.com" not in projected[0]["content"]
    assert "[EMAIL_REDACTED]" in projected[0]["content"]
    assert counts["email"] == 1


def test_luhn_gate_separates_cards_from_digit_runs():
    assert v14.luhn_valid("4242 4242 4242 4242")
    assert not v14.luhn_valid("4242 4242 4242 4241")
    assert not v14.luhn_valid("424242")


def test_redact_replaces_only_luhn_valid_card_candidates():
    counts: Counter[str] = Counter()
    assert v14.redact("card 4242 4242 4242 4242", counts) == "card [PAYMENT_CARD_REDACTED]"
    assert counts["payment_card"] == 1
    assert v14.redact("id 4242 4242 4242 4241", Counter()) == "id 4242 4242 4242 4241"


def test_project_agent_pins_the_v13_key_order():
    projected = v14.project_agent(
        [
            message("system", "tools live here"),
            message("user", "move the meeting"),
            message(
                "assistant",
                None,
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "submit", "arguments": "{}"}}],
            ),
            message("tool", '{"status":"ok"}', tool_call_id="call_1", name="submit"),
            message("assistant", "Done."),
        ]
    )
    assert [list(item) for item in projected] == [
        ["role", "content", "weight"],
        ["role", "content", "weight"],
        ["role", "content", "tool_calls", "weight"],
        ["role", "tool_call_id", "name", "content", "weight"],
        ["role", "content", "weight"],
    ]
    assert [item["weight"] for item in projected] == [0, 0, 1, 0, 1]


def test_project_agent_omits_the_name_key_when_absent():
    projected = v14.project_agent([message("tool", "{}", tool_call_id="call_1")])
    assert list(projected[0]) == ["role", "tool_call_id", "content", "weight"]


def test_project_agent_rejects_a_tool_turn_without_a_call_id():
    with pytest.raises(ValueError, match="tool_call_id"):
        v14.project_agent([message("tool", "{}")])


def test_v9_candidate_rows_are_agent_tool_shaped():
    rows = v14.v9_candidate_rows()
    assert len(rows) == v14.EXPECTED_V9_TRAIN_ROWS
    for candidate, payload in rows:
        assert v14.is_agent_tool(candidate)
        assert payload, "v9 rows must project to a non-empty payload"
        assert payload[-1]["role"] == "assistant"
        assert list(payload[0])[-1] == "weight"


def test_canonical_hash_ignores_key_order():
    assert v14.canonical_hash([{"role": "user", "weight": 0}]) == v14.canonical_hash([{"weight": 0, "role": "user"}])


def test_write_jsonl_is_compact_and_round_trips(tmp_path):
    path = tmp_path / "rows.jsonl"
    v14.write_jsonl(path, [{"messages": [{"role": "user", "content": "héllo"}]}])
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert json.loads(raw)["messages"][0]["content"] == "héllo"
