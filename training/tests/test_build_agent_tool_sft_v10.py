"""Tests for the v10 resource-first crisis family builder."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

import build_agent_tool_sft_v10 as v10  # noqa: E402

ACTIONS_BY_DISCLOSURE: dict[str, tuple[str, str]] = {
    disclosure: (severity, action)
    for _, severity, disclosures, action in v10.v9.CRISIS_GROUPS
    for disclosure in disclosures
}

EXPECTED_ROWS = sum(len(disclosures) for _, _, disclosures, _ in v10.v9.CRISIS_GROUPS) * v10.SHAPES_PER_DISCLOSURE


def rows() -> list[dict[str, Any]]:
    return v10.collect_rows()["train"]


def test_every_reply_opens_with_its_resource_action() -> None:
    for item in rows():
        _, action = ACTIONS_BY_DISCLOSURE[item["messages"][0]["content"]]
        assert item["messages"][1]["content"].startswith(action), item["id"]


def test_no_reply_repeats_a_v9_shape() -> None:
    v9_replies = v10.v9_replies()
    for item in rows():
        assert item["messages"][1]["content"] not in v9_replies, item["id"]


def test_crisis_rows_are_plain_user_assistant() -> None:
    for item in rows():
        assert [message["role"] for message in item["messages"]] == ["user", "assistant"]
        assert item["tools"] == []


def test_rows_keep_the_v9_severity_and_rubric() -> None:
    for item in rows():
        severity, action = ACTIONS_BY_DISCLOSURE[item["messages"][0]["content"]]
        assert item["severity"] == severity, item["id"]
        v10.validate_lexicon_row(item, action)


def test_family_crosses_every_disclosure_with_every_shape() -> None:
    collected = rows()
    assert len(collected) == EXPECTED_ROWS
    assert len({item["messages"][0]["content"] for item in collected}) == len(ACTIONS_BY_DISCLOSURE)
    assert len({item["messages"][1]["content"] for item in collected}) == (
        len({action for _, _, _, action in v10.v9.CRISIS_GROUPS}) * v10.SHAPES_PER_DISCLOSURE
    )
    per_disclosure = Counter(item["messages"][0]["content"] for item in collected)
    assert set(per_disclosure.values()) == {v10.SHAPES_PER_DISCLOSURE}


def test_severity_counts_follow_the_reused_groups() -> None:
    expected = Counter(severity for _, severity, disclosures, _ in v10.v9.CRISIS_GROUPS for _ in disclosures)
    expected = Counter({name: count * v10.SHAPES_PER_DISCLOSURE for name, count in expected.items()})
    assert v10.summarize(v10.collect_rows())["crisis_severity_counts"] == dict(sorted(expected.items()))


def test_validator_rejects_a_reply_that_buries_the_resource() -> None:
    item = rows()[0]
    action = ACTIONS_BY_DISCLOSURE[item["messages"][0]["content"]][1]
    buried = dict(item, messages=[item["messages"][0], {"role": "assistant", "content": f"I hear you. {action}"}])
    with pytest.raises(ValueError, match="does not open with the resource action"):
        v10.validate_lexicon_row(buried, action)


def test_build_writes_train_only_with_no_group_crossing_splits(tmp_path: Path) -> None:
    manifest = v10.build(tmp_path)
    assert manifest["scenario_families"] == {v10.FAMILY: EXPECTED_ROWS}
    assert {name: info["row_count"] for name, info in manifest["splits"].items()} == {"train": EXPECTED_ROWS}

    seen: dict[str, str] = {}
    for split, info in manifest["splits"].items():
        with (tmp_path / info["path"]).open(encoding="utf-8") as stream:
            for line in stream:
                payload = json.loads(line)
                group = json.dumps(payload["messages"][:2], ensure_ascii=False, sort_keys=True)
                assert seen.setdefault(group, split) == split


def test_build_refuses_to_overwrite_a_non_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "leftover.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        v10.build(tmp_path)
