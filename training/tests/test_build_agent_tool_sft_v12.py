"""Tests for the v12 crisis-surface family builder."""

from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

import build_agent_tool_sft_v12 as v12  # noqa: E402

CRISIS_ROWS = len(v12.SEVERITIES) * v12.PARAPHRASES_PER_CATEGORY * v12.SHAPES_PER_DISCLOSURE
CONTROL_ROWS = len(v12.CONTROL_TOPICS) * len(v12.CONTROL_REPLIES)
EXPECTED_ROWS = CRISIS_ROWS + CONTROL_ROWS


def rows() -> list[dict[str, Any]]:
    return v12.collect_rows()["train"]


def crisis_rows() -> list[dict[str, Any]]:
    return [item for item in rows() if item["category"] != "control"]


def control_rows() -> list[dict[str, Any]]:
    return [item for item in rows() if item["category"] == "control"]


def action_of(item: dict[str, Any]) -> str:
    reply = item["messages"][1]["content"]
    matches = [action for action in v12.ACTIONS[item["category"]] if reply.startswith(action)]
    assert len(matches) == 1, item["id"]
    return matches[0]


def test_every_crisis_reply_opens_with_one_of_its_category_actions() -> None:
    for item in crisis_rows():
        action_of(item)


def test_no_reply_repeats_a_v9_or_v10_reply() -> None:
    prior = v12.prior_replies()
    for item in rows():
        assert item["messages"][1]["content"] not in prior, item["id"]


def test_every_action_variant_is_used_and_differs_from_the_v9_action() -> None:
    used = Counter((item["category"], action_of(item)) for item in crisis_rows())
    for category, _, _, v9_action in v12.v9.CRISIS_GROUPS:
        actions = v12.ACTIONS[category]
        assert len(actions) == len(set(actions))
        assert v9_action not in actions
        assert {action for name, action in used if name == category} == set(actions)


def test_no_action_is_welded_to_a_single_frame() -> None:
    for category in v12.SEVERITIES:
        pairs = {
            (
                v12.ACTIONS[category].index(action_of(item)),
                v12.SURFACE_SHAPES.index(
                    item["messages"][1]["content"].replace(action_of(item), "{action}", 1),
                ),
            )
            for item in crisis_rows()
            if item["category"] == category
        }
        assert pairs == set(product(range(len(v12.ACTIONS[category])), range(v12.SHAPES_PER_DISCLOSURE)))


def test_crisis_rows_are_plain_user_assistant_with_no_tools() -> None:
    for item in crisis_rows():
        assert [message["role"] for message in item["messages"]] == ["user", "assistant"]
        assert item["tools"] == []


def test_rows_keep_the_v9_severity_and_rubric() -> None:
    for item in crisis_rows():
        assert item["severity"] == v12.SEVERITIES[item["category"]]
        v12.v9.validate_crisis_row(item)


def test_paraphrases_never_reuse_a_v9_disclosure_or_control_topic() -> None:
    prior = v12.v9_disclosures() | set(v12.v9.CONTROL_TOPICS)
    for item in rows():
        assert item["messages"][0]["content"] not in prior, item["id"]


def test_paraphrases_never_reuse_evaluation_text() -> None:
    texts = v12.v9.evaluation_texts()
    assert texts
    for item in rows():
        assert all(message["content"] not in texts for message in item["messages"]), item["id"]


def test_every_category_gets_the_same_surface_breadth() -> None:
    per_disclosure = Counter(item["messages"][0]["content"] for item in crisis_rows())
    per_category = Counter(item["category"] for item in crisis_rows())
    assert len(per_disclosure) == len(v12.SEVERITIES) * v12.PARAPHRASES_PER_CATEGORY
    assert set(per_disclosure.values()) == {v12.SHAPES_PER_DISCLOSURE}
    assert set(per_category.values()) == {v12.PARAPHRASES_PER_CATEGORY * v12.SHAPES_PER_DISCLOSURE}


def test_controls_stay_off_the_crisis_vocabulary() -> None:
    assert len(control_rows()) == CONTROL_ROWS
    for item in control_rows():
        assert item["severity"] == "none"
        v12.v9.validate_crisis_row(item)


def test_summary_reports_the_expected_shape() -> None:
    summary = v12.summarize(v12.collect_rows())
    assert summary["example_count"] == EXPECTED_ROWS
    assert summary["crisis_severity_counts"] == {
        "acute": 8 * v12.PARAPHRASES_PER_CATEGORY * v12.SHAPES_PER_DISCLOSURE,
        "hotline": 2 * v12.PARAPHRASES_PER_CATEGORY * v12.SHAPES_PER_DISCLOSURE,
        "none": CONTROL_ROWS,
    }
    assert summary["distinct_disclosures"] == (
        len(v12.SEVERITIES) * v12.PARAPHRASES_PER_CATEGORY + len(v12.CONTROL_TOPICS)
    )
    assert summary["distinct_replies"] > len(v12.SEVERITIES) * len(v12.ACTIONS["suicide_plan"])


def test_validator_rejects_a_reply_that_buries_the_resource() -> None:
    item = crisis_rows()[0]
    action = action_of(item)
    buried = dict(
        item,
        messages=[item["messages"][0], {"role": "assistant", "content": f"I hear you. {action}"}],
    )
    with pytest.raises(ValueError, match="does not open with one of the category actions"):
        v12.validate_surface_row(buried, v12.ACTIONS[item["category"]])


def test_no_v12_action_is_reachable_from_a_prior_reply() -> None:
    prior = v12.prior_replies()
    for category, actions in v12.ACTIONS.items():
        for action in actions:
            assert not any(reply.startswith(action) for reply in prior), (category, action)


def test_build_writes_train_only(tmp_path: Path) -> None:
    manifest = v12.build(tmp_path)
    assert manifest["scenario_families"] == {v12.FAMILY: EXPECTED_ROWS}
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
        v12.build(tmp_path)
