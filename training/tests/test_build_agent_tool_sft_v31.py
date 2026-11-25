"""Tests for v31: the control family speaks to the topic, not to a template.

The v31 SFT arm failed its held-out controls: every control reply in the
graded run was 43-55 characters, opening with molds like "Start here: Pick
the version you can start this week." The cause was structural: v30 (and
v29, v26, v23, v22 before it) built 600 control rows as 100 everyday topics
times the same handful of reply molds, with the embedded step cycled
independently of the topic. The v31 training file measured 600 control rows
carrying exactly 36 distinct replies, 15.2 words on average.

v31 replaces the molds with 300 hand-written, topic-conditioned replies:
three per topic, every one naming the situation and a concrete next step,
two to four sentences, at least 20 words. The crisis family is untouched —
byte-identical to v30 — so these tests pin both halves of that claim.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

import training.build_agent_tool_sft_v12 as v12
import training.build_agent_tool_sft_v22 as v22
import training.build_agent_tool_sft_v30 as v30
import training.build_agent_tool_sft_v31 as v31

EXPECTED_TOTAL_ROWS = 6204
EXPECTED_CRISIS_ROWS = 5904
EXPECTED_CONTROLS = 300
EXPECTED_DISCLOSURES = 838
EXPECTED_DISTINCT_REPLIES = 1336
EXPECTED_REFLECTIVE_ROWS = 2214
REPLIES_PER_TOPIC = 3
MIN_CONTROL_WORDS = 20


@cache
def rows() -> tuple[dict[str, Any], ...]:
    return tuple(v31.collect_rows()["train"])


@cache
def summary() -> dict[str, Any]:
    return v31.summarize(v31.collect_rows())


@cache
def crisis_rows() -> tuple[dict[str, Any], ...]:
    return tuple(item for item in rows() if item["category"] != "control")


@cache
def control_rows() -> tuple[dict[str, Any], ...]:
    return tuple(item for item in rows() if item["category"] == "control")


def _reply(item: dict[str, Any]) -> str:
    return item["messages"][1]["content"]


def test_crisis_rows_are_byte_identical_to_v30() -> None:
    """The fix must not move a single crisis row."""
    theirs = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in v30.collect_rows()["train"] if item["category"] != "control"]
    ours = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in crisis_rows()]
    # v31 stamps its own ids, so compare everything but the id.
    def strip_id(blob: str) -> str:
        item = json.loads(blob)
        item["id"] = item["group_id"] = ""
        return json.dumps(item, sort_keys=True, ensure_ascii=False)

    assert [strip_id(blob) for blob in ours] == [strip_id(blob) for blob in theirs]


def test_row_counts_and_summary() -> None:
    assert len(rows()) == EXPECTED_TOTAL_ROWS
    assert len(crisis_rows()) == EXPECTED_CRISIS_ROWS
    assert len(control_rows()) == EXPECTED_CONTROLS
    assert summary()["control_rows"] == EXPECTED_CONTROLS
    assert summary()["distinct_disclosures"] == EXPECTED_DISCLOSURES
    assert summary()["distinct_replies"] == EXPECTED_DISTINCT_REPLIES
    assert summary()["escalation_first_crisis_rows"] == EXPECTED_CRISIS_ROWS
    assert summary()["reflection_first_crisis_rows"] == 0
    assert summary()["reflective_rows"] == EXPECTED_REFLECTIVE_ROWS


def test_control_replies_are_all_unique_and_topic_conditioned() -> None:
    replies = [_reply(item) for item in control_rows()]
    assert len(set(replies)) == EXPECTED_CONTROLS
    # Every topic carries exactly three replies, and no two topics share one.
    for topic_index in range(len(v31.CONTROL_TOPICS)):
        block = v31.CONTROL_REPLIES_BY_TOPIC[topic_index]
        assert len(block) == REPLIES_PER_TOPIC
        assert len(set(block)) == REPLIES_PER_TOPIC
    assert len({reply for block in v31.CONTROL_REPLIES_BY_TOPIC for reply in block}) == EXPECTED_CONTROLS


def test_control_topics_match_the_v22_surface() -> None:
    assert list(v31.CONTROL_TOPICS) == list(v22.CONTROL_TOPICS)
    for item in control_rows():
        control_index = int(item["id"].rsplit("-", 1)[1]) - EXPECTED_CRISIS_ROWS
        topic, _slot = divmod(control_index, REPLIES_PER_TOPIC)
        assert item["messages"][0]["content"] == v31.CONTROL_TOPICS[topic]


def test_control_replies_carry_the_floor_of_detail() -> None:
    words = [len(_reply(item).split()) for item in control_rows()]
    assert min(words) >= MIN_CONTROL_WORDS
    assert summary()["control_reply_words_min"] >= MIN_CONTROL_WORDS


def test_control_replies_avoid_the_six_v22_molds() -> None:
    """No reply may be one of the templates that caused the collapse."""
    mold_texts = set()
    for step in v12.CONTROL_STEPS:
        for mold in v22.CONTROL_REPLIES:
            mold_texts.add(mold.format(step=step))
    ours = {_reply(item) for item in control_rows()}
    assert not (ours & mold_texts)


def test_control_replies_never_escalate() -> None:
    for item in control_rows():
        lowered = _reply(item).lower()
        for term in v31.ESCALATION_TERMS_LOWER:
            assert not v31._mentions(term, _reply(item)), (item["id"], term)
        assert not any(term.lower() in lowered for term in v31.HOTLINE_TERMS)


def test_control_replies_neither_ask_nor_use_filler() -> None:
    for item in control_rows():
        reply = _reply(item)
        lowered = reply.lower()
        assert "?" not in reply, item["id"]
        assert not [f for f in v31.FORBIDDEN_FILLER if f in lowered], item["id"]
        assert not [d for d in v31.DISMISSAL if d in lowered], item["id"]


def test_validator_rejects_a_question() -> None:
    item = {"id": "t", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "What do you think you should do about the noise complaint you filed last week now?"}]}
    with pytest.raises(ValueError, match="asks a question"):
        v31.validate_control_reply(item)


def test_validator_rejects_filler() -> None:
    item = {"id": "t", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "I hear you on the noise complaint. Put the dates in writing, name the flats, and hand it to the managing agent with the log attached today."}]}
    with pytest.raises(ValueError, match="filler"):
        v31.validate_control_reply(item)


def test_validator_rejects_escalation_vocabulary() -> None:
    item = {"id": "t", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "Call the council and tell them about the drain smell. If it is a gas leak you should ring them; if it is a genuine emergency they will send someone out today."}]}
    with pytest.raises(ValueError, match="escalates"):
        v31.validate_control_reply(item)


def test_validator_rejects_a_thin_reply() -> None:
    item = {"id": "t", "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "Write it down."}]}
    with pytest.raises(ValueError, match="too thin"):
        v31.validate_control_reply(item)


def test_ids_are_unique_and_stamped_with_this_release() -> None:
    ids = [item["id"] for item in rows()]
    assert len(set(ids)) == len(ids)
    assert all("-v31-" in example_id for example_id in ids)


def test_build_writes_the_family(tmp_path: Path) -> None:
    result = v31.build(tmp_path / "v31")
    path = Path(result["train_path"])
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == EXPECTED_TOTAL_ROWS
    assert result["control_rows"] == EXPECTED_CONTROLS


def test_build_refuses_to_overwrite_a_populated_directory(tmp_path: Path) -> None:
    target = tmp_path / "v31"
    target.mkdir()
    (target / "existing.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        v31.build(target)
