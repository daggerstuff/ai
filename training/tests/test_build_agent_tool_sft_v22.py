"""Tests for the v22 crisis family: volume and variety instead of phrase surgery.

v21 tried to fix the register by deleting the phrases that produced the failures
and lost, because a phrase is a surface form: the model substituted a sibling the
corpus already taught. v22 leaves the corpus alone and grows the family instead —
more deadline frames, more hand-written actions per category, more disclosures.
What matters here is that the growth is real variety and not the same sentence
repeated eight times, because eight copies of one action would teach exactly the
frozen register v21 was criticised for.
"""

from __future__ import annotations

from collections import Counter

import training.build_agent_tool_sft_v9 as v9
import training.build_agent_tool_sft_v20 as v20
import training.build_agent_tool_sft_v21 as v21
import training.build_agent_tool_sft_v22 as v22
from ai.qa.validation.therapy_bench import generate_crisis_capability_v1 as rubric

MESSAGES_PER_ROW = 2
EXPECTED_SHAPES = 8
EXPECTED_ACTIONS_PER_CATEGORY = 8
EXPECTED_TOTAL_ROWS = 6184
EXPECTED_CRISIS_ROWS = 5584
EXPECTED_CONTROLS = 600
V21_ROWS = 2352
# A family this size should produce hundreds of distinct replies; 100 is the
# floor below which the action table has collapsed onto a handful of sentences.
MIN_DISTINCT_REPLIES = 100


def rows() -> list[dict]:
    return v22.collect_rows()["train"]


def crisis_rows(items: list[dict]) -> list[dict]:
    return [item for item in items if item["category"] != "control"]


def test_the_family_keeps_the_shared_identity() -> None:
    assert v22.FAMILY == v21.FAMILY == v20.FAMILY
    assert v22.BASE_MODEL == v21.BASE_MODEL
    assert v22.RELEASE_ID.endswith("-v22")


def test_the_shape_table_doubles_without_duplicating() -> None:
    assert len(v22.DIRECT_SHAPES) == EXPECTED_SHAPES
    assert len(set(v22.DIRECT_SHAPES)) == EXPECTED_SHAPES
    assert v22.DIRECT_SHAPES[: len(v21.DIRECT_SHAPES)] == v21.DIRECT_SHAPES
    assert all("{action}" in shape for shape in v22.DIRECT_SHAPES)


def test_every_category_carries_the_full_action_table() -> None:
    assert v22.ACTIONS.keys() == v21.ACTIONS.keys()
    for category, actions in v22.ACTIONS.items():
        assert len(actions) == EXPECTED_ACTIONS_PER_CATEGORY, category
        assert len(set(actions)) == EXPECTED_ACTIONS_PER_CATEGORY, category
        assert len(set(actions)) > len(v21.ACTIONS[category]), category


def test_the_new_actions_are_imperative_and_anchor_bearing() -> None:
    """Every hand-written action must pass the same guard as the emitted reply."""
    for category, actions in v22.ACTIONS.items():
        anchors = v22.ANCHORS[category]
        for action in actions:
            assert "?" not in action, action
            folded = action.casefold()
            for filler in v22.FORBIDDEN_FILLER:
                assert filler not in folded, (action, filler)
            for phrase in rubric.DISMISSAL:
                assert phrase not in folded, (action, phrase)
            assert any(anchor in folded for anchor in anchors), (category, action)


def test_the_extra_actions_are_new_text_not_repeats() -> None:
    for category, actions in v22.EXTRA_ACTIONS.items():
        for action in actions:
            assert action not in v21.ACTIONS[category], (category, action)
            assert action in v22.ACTIONS[category], (category, action)


def test_the_surface_grows_past_v21() -> None:
    surface = v22.paraphrases()
    assert surface.keys() == v21.paraphrases().keys()
    for category, items in surface.items():
        assert len(set(items)) > len(set(v21.paraphrases()[category])), category


def test_the_builder_emits_the_planned_volume() -> None:
    items = rows()
    assert len(items) == EXPECTED_TOTAL_ROWS
    assert len(crisis_rows(items)) == EXPECTED_CRISIS_ROWS
    assert len(items) - len(crisis_rows(items)) == EXPECTED_CONTROLS
    assert EXPECTED_CRISIS_ROWS > V21_ROWS


def test_every_row_is_a_two_turn_chat_row_with_a_v22_id() -> None:
    for item in rows():
        assert len(item["messages"]) == MESSAGES_PER_ROW
        assert item["messages"][0]["role"] == "user"
        assert item["messages"][1]["role"] == "assistant"
        assert item["tools"] == []
        assert item["id"].startswith("crisis_direct-v22-")


def test_no_crisis_reply_asks_a_question_or_coddles() -> None:
    for item in crisis_rows(rows()):
        reply = item["messages"][1]["content"]
        folded = reply.casefold()
        assert "?" not in reply, item["id"]
        for filler in v22.FORBIDDEN_FILLER:
            assert filler not in folded, (item["id"], filler)
        for phrase in rubric.CODDLING:
            assert phrase not in folded, (item["id"], phrase)
        for phrase in rubric.DISMISSAL:
            assert phrase not in folded, (item["id"], phrase)


def test_no_control_reply_recites_a_hotline() -> None:
    for item in rows():
        if item["category"] != "control":
            continue
        folded = item["messages"][1]["content"].casefold()
        assert item["severity"] == "none"
        for term in rubric.HOTLINE_TERMS:
            assert term not in folded, (item["id"], term)


def test_every_acute_reply_escalates() -> None:
    acute = [item for item in crisis_rows(rows()) if item["severity"] == "acute"]
    assert acute
    for item in acute:
        folded = item["messages"][1]["content"].casefold()
        assert any(term in folded for term in v9.ACUTE), item["id"]


def test_the_reply_surface_is_far_wider_than_v20s() -> None:
    """Eight actions per category is the whole point: the reply set must widen."""
    replies = {item["messages"][1]["content"] for item in crisis_rows(rows())}
    assert len(replies) > len(v22.prior_replies())
    assert len(replies) > 3 * len(v20.ACTIONS) * len(v20.DIRECT_SHAPES)


def test_the_severity_split_still_has_a_quiet_majority() -> None:
    severities = Counter(item["severity"] for item in rows())
    assert severities["none"] == EXPECTED_CONTROLS
    assert severities["acute"] > severities["engaged"]


def test_summarize_reports_the_built_surface() -> None:
    summary = v22.summarize(v22.collect_rows())
    assert summary["example_count"] == EXPECTED_TOTAL_ROWS
    assert summary["crisis_category_counts"]["control"] == EXPECTED_CONTROLS
    assert summary["distinct_disclosures"] > len(set(v21.paraphrases()["self_harm_active"]))
    assert summary["distinct_replies"] > MIN_DISTINCT_REPLIES


def test_the_module_records_why_it_replaces_phrase_surgery_with_volume() -> None:
    docstring = v22.__doc__ or ""
    assert "73/88" in docstring
    assert "84/88" in docstring
    assert "not going to jump" in docstring
