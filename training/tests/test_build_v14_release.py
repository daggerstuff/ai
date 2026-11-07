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


def message(role: str, content: str | None = None, **extra: object) -> dict[str, object]:
    return {"role": role, "content": content, **extra}


def test_line_sha_excludes_the_trailing_newline() -> None:
    body = b'{"a":1}'
    assert v14.line_sha(body + b"\n") == hashlib.sha256(body).hexdigest()
    assert v14.line_sha(body) == hashlib.sha256(body).hexdigest()


def test_agent_tool_discriminator_needs_tools_and_no_source() -> None:
    assert v14.is_agent_tool({"messages": [], "tools": []})
    assert not v14.is_agent_tool({"messages": [], "tools": [], "source": "crisis_benchmark"})
    assert not v14.is_agent_tool({"messages": []})


def test_crisis_placeholder_matches_only_the_fabricated_target() -> None:
    row = {
        "source": "crisis_benchmark",
        "messages": [message("user", "I have a plan"), message("assistant", v14.CRISIS_PLACEHOLDER)],
    }
    assert v14.is_crisis_placeholder(row)


def test_crisis_placeholder_ignores_a_real_target() -> None:
    row = {
        "source": "crisis_benchmark",
        "messages": [message("user", "I have a plan"), message("assistant", "Call 911 now.")],
    }
    assert not v14.is_crisis_placeholder(row)


def test_crisis_placeholder_ignores_other_sources() -> None:
    row = {
        "source": "empath",
        "messages": [message("user", "hi"), message("assistant", v14.CRISIS_PLACEHOLDER)],
    }
    assert not v14.is_crisis_placeholder(row)


def test_crisis_placeholder_ignores_the_string_in_a_user_turn() -> None:
    row = {
        "source": "crisis_benchmark",
        "messages": [message("user", v14.CRISIS_PLACEHOLDER), message("assistant", "Call 911 now.")],
    }
    assert not v14.is_crisis_placeholder(row)


def test_project_chat_drops_system_and_weights_the_turns() -> None:
    counts: Counter[str] = Counter()
    projected = v14.project_chat(
        [message("system", "you are helpful"), message("user", "hello"), message("assistant", "hi")],
        counts,
    )
    assert projected == [
        {"role": "user", "content": "hello", "weight": 0},
        {"role": "assistant", "content": "hi", "weight": 1},
    ]


def test_project_chat_truncates_after_the_last_assistant() -> None:
    counts: Counter[str] = Counter()
    projected = v14.project_chat(
        [message("user", "one"), message("assistant", "two"), message("user", "three")],
        counts,
    )
    assert projected is not None
    assert [item["role"] for item in projected] == ["user", "assistant"]


def test_project_chat_requires_an_assistant_and_a_user() -> None:
    counts: Counter[str] = Counter()
    assert v14.project_chat([message("user", "hello")], counts) is None
    assert v14.project_chat([message("assistant", "hi")], counts) is None


def test_project_chat_skips_blank_and_non_text_turns() -> None:
    counts: Counter[str] = Counter()
    projected = v14.project_chat(
        [message("user", "   "), message("user", None), {"role": "user"}, message("assistant", "hi")],
        counts,
    )
    assert projected is None


def test_project_chat_redacts_pii_and_counts_it() -> None:
    counts: Counter[str] = Counter()
    projected = v14.project_chat(
        [message("user", "write to jane@example.com"), message("assistant", "ok")],
        counts,
    )
    assert projected is not None
    assert "jane@example.com" not in projected[0]["content"]
    assert "[EMAIL_REDACTED]" in projected[0]["content"]
    assert counts["email"] == 1


def test_luhn_gate_separates_cards_from_digit_runs() -> None:
    assert v14.luhn_valid("4242 4242 4242 4242")
    assert not v14.luhn_valid("4242 4242 4242 4241")
    assert not v14.luhn_valid("424242")


def test_redact_replaces_only_luhn_valid_card_candidates() -> None:
    counts: Counter[str] = Counter()
    assert v14.redact("card 4242 4242 4242 4242", counts) == "card [PAYMENT_CARD_REDACTED]"
    assert counts["payment_card"] == 1
    assert v14.redact("id 4242 4242 4242 4241", Counter()) == "id 4242 4242 4242 4241"


def test_project_agent_pins_the_v13_key_order() -> None:
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


def test_project_agent_omits_the_name_key_when_absent() -> None:
    projected = v14.project_agent([message("tool", "{}", tool_call_id="call_1")])
    assert list(projected[0]) == ["role", "tool_call_id", "content", "weight"]


def test_project_agent_rejects_a_tool_turn_without_a_call_id() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        v14.project_agent([message("tool", "{}")])


def test_v9_candidate_rows_are_agent_tool_shaped() -> None:
    rows = v14.v9_candidate_rows()
    assert len(rows) == v14.EXPECTED_V9_TRAIN_ROWS
    for candidate, payload in rows:
        assert v14.is_agent_tool(candidate)
        assert payload, "v9 rows must project to a non-empty payload"
        assert payload[-1]["role"] == "assistant"
        assert list(payload[0])[-1] == "weight"


def test_canonical_hash_ignores_key_order() -> None:
    assert v14.canonical_hash([{"role": "user", "weight": 0}]) == v14.canonical_hash([{"weight": 0, "role": "user"}])


def test_write_jsonl_is_compact_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    v14.write_jsonl(path, [{"messages": [{"role": "user", "content": "héllo"}]}])
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert json.loads(raw)["messages"][0]["content"] == "héllo"


def payload(targets: list[str], tools: list[object] | None = None) -> str:
    body: dict[str, object] = {
        "messages": [{"role": "assistant", "content": target, "weight": 1} for target in targets]
    }
    if tools is not None:
        body["tools"] = tools
    return json.dumps(body)


CAP = v14.DUPLICATE_TARGET_CAP
OVER_CAP = CAP + 1


def test_duplicate_profile_counts_the_final_target_per_row() -> None:
    lines = [payload(["first", "shared"]), payload(["shared"]), payload(["shared", "other"])]
    profile = v14.duplicate_target_profile(lines)
    assert profile["rows"] == len(lines)
    assert profile["assistant_turns"] == sum(len(json.loads(line)["messages"]) for line in lines)
    finals = {json.loads(line)["messages"][-1]["content"] for line in lines}
    assert profile["final_target"]["distinct"] == len(finals)
    assert profile["final_target"]["max_rows_per_target"] == len(lines) - 1


def test_duplicate_profile_ignores_mid_session_repeats_for_the_cap() -> None:
    """A line repeated mid-session never counts toward the final-target cap."""
    lines = [payload(["checklist item", f"reply {index}"]) for index in range(OVER_CAP)]
    profile = v14.duplicate_target_profile(lines)
    assert profile["final_target"]["targets_over_cap"] == 0
    assert profile["all_targets"]["targets_over_cap"] == 1
    assert profile["all_targets"]["rows_containing_over_cap_target"] == len(lines)


def test_duplicate_profile_flags_over_cap_final_targets() -> None:
    lines = [payload(["same"]) for _ in range(OVER_CAP)]
    profile = v14.duplicate_target_profile(lines)
    assert profile["final_target"]["targets_over_cap"] == 1
    assert profile["final_target"]["rows_in_over_cap_target"] == len(lines)
    assert profile["final_target"]["max_rows_per_target"] == len(lines)


def test_duplicate_profile_separates_chat_from_agent_rows() -> None:
    chat = [payload(["same"]) for _ in range(OVER_CAP)]
    agent = [payload(["same"], tools=[]) for _ in range(OVER_CAP)]
    profile = v14.duplicate_target_profile(chat + agent)
    final = profile["final_target"]
    assert final["chat_rows"] == len(chat)
    assert final["chat_targets_over_cap"] == 1
    assert final["agent_rows"] == len(agent)
    assert final["agent_targets_over_cap"] == 1
    assert final["max_rows_per_target"] == len(chat) + len(agent)


def test_train_duplicate_policy_accepts_a_payload_at_the_cap() -> None:
    v14.verify_train_duplicate_policy(v14.duplicate_target_profile([payload(["same"]) for _ in range(CAP)]))


def test_train_duplicate_policy_rejects_an_over_cap_chat_target() -> None:
    profile = v14.duplicate_target_profile([payload(["same"]) for _ in range(OVER_CAP)])
    with pytest.raises(ValueError, match=f"{CAP}-row cap"):
        v14.verify_train_duplicate_policy(profile)


def test_train_duplicate_policy_exempts_agent_tool_rows() -> None:
    """Agent rows were appended after the cap ran, so they report but do not fail."""
    v14.verify_train_duplicate_policy(
        v14.duplicate_target_profile([payload(["same"], tools=[]) for _ in range(OVER_CAP)])
    )


def test_limitations_no_longer_claim_a_corpus_wide_target_cap() -> None:
    """The old wording claimed a cap on every assistant target; only the final one is capped."""
    text = " ".join(v14.LIMITATIONS)
    assert "caps exact duplicate assistant targets at 50 per target" not in text
    assert "final assistant target" in text


def test_limitations_disclose_the_uncapped_held_out_partitions() -> None:
    assert "Validation and test were never duplicate-filtered" in " ".join(v14.LIMITATIONS)


def test_limitations_disclose_the_agent_tool_exemption() -> None:
    assert "never subject to it" in " ".join(v14.LIMITATIONS)


def test_release_summary_reports_the_measured_duplicate_profile() -> None:
    empty = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    release = v14._Release(empty, Path("/tmp/v14-manifest-probe"), dry_run=True)
    release.payload_lines["train"] = [payload(["shared"]), payload(["shared"])]
    release.payload_lines["validation"] = [payload(["label"])]
    release.payload_lines["test"] = [payload(["label"])]
    release.audit_duplicate_targets()
    summary = release.summary()
    assert summary["duplicate_max_rows_per_final_target"] == len(release.payload_lines["train"])
    assert summary["duplicate_final_targets_over_cap"] == 0
    validation = release.duplicate_profiles["validation"]["final_target"]
    assert validation["max_rows_per_target"] == len(release.payload_lines["validation"])
