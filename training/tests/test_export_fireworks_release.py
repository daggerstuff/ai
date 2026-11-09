"""Tests for the Fireworks dataset export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import export_fireworks_release as fw

ROUTING_REPLY = "Call 988 now. Then we can talk about the rest."
PLAIN_REPLY = "That sounds like a hard week at work."
ACUTE_PROMPT = "I took the pills an hour ago and I feel dizzy."


def row(messages: list[dict[str, str]], **extra: object) -> dict[str, object]:
    return {"messages": messages, **extra}


def chat(client: str, assistant: str) -> dict[str, object]:
    return row(
        [
            {"role": "user", "content": client},
            {"role": "assistant", "content": assistant},
        ]
    )


def write_train(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def test_weighted_messages_marks_assistant_turns_for_training() -> None:
    payload = fw._weighted_messages(
        row(
            [
                {"role": "system", "content": "be a therapist"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
    )
    assert [message.get("role") for message in payload] == ["system", "user", "assistant"]
    assert payload[0].get("weight") is None
    assert payload[1]["weight"] == 0
    assert payload[2]["weight"] == 1


def test_weighted_messages_rejects_unexpected_roles() -> None:
    with pytest.raises(ValueError, match="unexpected role 'tool'"):
        fw._weighted_messages(row([{"role": "tool", "content": "x"}]))


def test_classify_finds_routing_replies() -> None:
    assert fw.classify(chat("I keep thinking about it", ROUTING_REPLY)) == ["routing"]


def test_classify_finds_acute_client_turns() -> None:
    assert fw.classify(chat(ACUTE_PROMPT, PLAIN_REPLY)) == ["acute_client"]


def test_classify_ignores_ordinary_rows() -> None:
    assert fw.classify(chat("work was long", PLAIN_REPLY)) == []


def test_classify_ignores_tool_rows() -> None:
    assert fw.classify({**chat("I keep thinking about it", ROUTING_REPLY), "tools": []}) == []


def test_select_rows_excludes_tool_rows_by_default(tmp_path: Path) -> None:
    train = write_train(tmp_path / "train.jsonl", [{**chat("x", ROUTING_REPLY), "tools": []}])
    kept, counts = fw.select_rows(train, fw.SelectionOptions())
    assert kept == []
    assert counts["tool_rows_excluded"] == 1


def test_select_rows_keeps_every_acute_row_without_a_cap(tmp_path: Path) -> None:
    rows = [chat(f"{ACUTE_PROMPT} ({index})", PLAIN_REPLY) for index in range(5)]
    train = write_train(tmp_path / "train.jsonl", rows)
    kept, counts = fw.select_rows(train, fw.SelectionOptions(routing_cap=1))
    assert len(kept) == len(rows)
    assert counts["acute_client_rows"] == len(rows)


def test_select_rows_caps_routing_rows_to_the_lowest_hashes(tmp_path: Path) -> None:
    rows = [chat(f"disclosure {index}", ROUTING_REPLY) for index in range(10)]
    train = write_train(tmp_path / "train.jsonl", rows)
    options = fw.SelectionOptions(routing_cap=3)
    kept, counts = fw.select_rows(train, options)
    assert counts["routing_rows"] == len(rows)
    assert counts["routing_kept"] == options.routing_cap
    expected = sorted(
        range(1, len(rows) + 1),
        key=lambda number: fw._sample_key(
            json.dumps(
                {"messages": fw._weighted_messages(rows[number - 1])},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            options.seed,
        ),
    )[: options.routing_cap]
    assert sorted(selection.row_number for selection in kept) == sorted(expected)


def test_select_rows_ignores_file_order_for_the_sample(tmp_path: Path) -> None:
    rows = [chat(f"day {index}", PLAIN_REPLY) for index in range(10)]
    options = fw.SelectionOptions(chat_sample=3)
    forward = write_train(tmp_path / "forward.jsonl", rows)
    backward = write_train(tmp_path / "backward.jsonl", list(reversed(rows)))
    kept_forward, _ = fw.select_rows(forward, options)
    kept_backward, _ = fw.select_rows(backward, options)
    assert sorted(selection.line for selection in kept_forward) == sorted(selection.line for selection in kept_backward)
    assert len(kept_forward) == options.chat_sample


def test_select_rows_drops_rows_longer_than_the_context(tmp_path: Path) -> None:
    long_row = chat("x" * 500, ROUTING_REPLY)
    train = write_train(tmp_path / "train.jsonl", [long_row])
    kept, counts = fw.select_rows(train, fw.SelectionOptions(max_chars=100))
    assert kept == []
    assert counts["too_long_excluded"] == 1


def test_select_rows_keeps_filler_rows_under_the_sample_size(tmp_path: Path) -> None:
    long_filler = chat("x" * 3000, PLAIN_REPLY)
    train = write_train(tmp_path / "train.jsonl", [long_filler])
    kept, counts = fw.select_rows(train, fw.SelectionOptions(sample_max_chars=100))
    assert kept == []
    assert counts["sample"] == 0
