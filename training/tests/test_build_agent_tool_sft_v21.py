"""Tests for the v21 crisis family: v20's register, a wider surface, more volume.

v21 keeps v20's action table and anchor guard and grows the two things that
still decide the paired crisis suite: how many ways a disclosure can be phrased,
and how many rows the family contributes against the corpus. Two extra action
variants are added for each category that still appears in a loss, so every
reply must still open with exactly one action and name its category anchor.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

import training.build_agent_tool_sft_v13 as v13
import training.build_agent_tool_sft_v20 as v20
import training.build_agent_tool_sft_v21 as v21

SHAPES_PER_DISCLOSURE = 4
MIN_CRISIS_ROWS = 2_100
CONTROL_ROWS = len(v21.CONTROL_TOPICS) * len(v21.CONTROL_REPLIES)
GROWN_CATEGORIES = {"self_harm_active", "psychosis_risk_other", "suicide_plan"}


def surface() -> dict[str, tuple[str, ...]]:
    return v21.paraphrases()


def crisis_rows() -> list[dict[str, Any]]:
    return [item for item in v21.collect_rows()["train"] if item["category"] != "control"]


def control_rows() -> list[dict[str, Any]]:
    return [item for item in v21.collect_rows()["train"] if item["category"] == "control"]


def test_the_family_has_the_expected_shape() -> None:
    crisis = sum(len(items) for items in surface().values()) * SHAPES_PER_DISCLOSURE
    assert len(crisis_rows()) == crisis
    assert len(control_rows()) == CONTROL_ROWS
    assert len(crisis_rows()) + len(control_rows()) == crisis + CONTROL_ROWS


def test_the_family_is_larger_than_v20() -> None:
    """v20's 1,760 rows took the crisis suite to 11/11; v21 keeps the ratio moving."""
    v20_crisis = len([item for item in v20.collect_rows()["train"] if item["category"] != "control"])
    assert len(crisis_rows()) >= MIN_CRISIS_ROWS
    assert len(crisis_rows()) > v20_crisis


def test_every_crisis_reply_opens_with_exactly_one_action() -> None:
    for item in crisis_rows():
        reply = item["messages"][1]["content"]
        matches = [action for action in v21.ACTIONS[item["category"]] if reply.startswith(action)]
        assert len(matches) == 1, item["id"]


def test_every_crisis_reply_names_its_category_anchor() -> None:
    for item in crisis_rows():
        lowered = item["messages"][1]["content"].lower()
        anchors = v21.ANCHORS[item["category"]]
        assert any(anchor in lowered for anchor in anchors), (item["id"], anchors)


def test_the_extra_actions_satisfy_the_same_contract() -> None:
    """A new variant must be as terse and as concrete as the v13 ones it joins."""
    for category, variants in v21.EXTRA_ACTIONS.items():
        for action in variants:
            lowered = action.lower()
            assert "?" not in action, category
            assert not [phrase for phrase in v21.FORBIDDEN_FILLER if phrase in lowered], (category, action)
            assert any(anchor in lowered for anchor in v21.ANCHORS[category]), (category, action)


def test_no_crisis_reply_softens_the_action_or_asks_a_question() -> None:
    for item in crisis_rows():
        reply = item["messages"][1]["content"]
        lowered = reply.lower()
        assert "?" not in reply, item["id"]
        assert not [phrase for phrase in v21.FORBIDDEN_FILLER if phrase in lowered], item["id"]


def test_the_forbidden_list_covers_the_v20_loss_openers() -> None:
    """Every v20 crisis loss opened with one of these, and the list must catch them."""
    openers = (
        "i'm not going to pretend that's not serious.",
        "i'm taking this seriously.",
        "let's slow down.",
        "let's get you to a hospital",
    )
    for opener in openers:
        assert [phrase for phrase in v21.FORBIDDEN_FILLER if phrase in opener], opener


def test_the_surface_grows_where_the_losses_are() -> None:
    """Growth is aimed at the three categories that keep appearing in a loss."""
    grown = surface()
    inherited = v20.paraphrases()
    for category in GROWN_CATEGORIES:
        assert len(grown[category]) > len(inherited[category]), category
    assert sum(len(items) for items in grown.values()) > sum(len(items) for items in inherited.values())


def test_the_largest_categories_stay_the_failing_ones() -> None:
    counts = Counter(item["category"] for item in crisis_rows())
    largest = {category for category, _ in counts.most_common(3)}
    assert largest == GROWN_CATEGORIES
    assert counts["self_harm_active"] > counts["suicide_plan"]


def test_the_surface_keeps_every_v20_disclosure() -> None:
    """The growth is additive: v21 must not drop a disclosure v20 already taught."""
    inherited = {text for items in v20.paraphrases().values() for text in items}
    grown = {text for items in surface().values() for text in items}
    assert inherited <= grown
    assert len(grown) > len(inherited)


def test_the_action_table_keeps_every_v20_variant() -> None:
    for category, variants in v20.ACTIONS.items():
        assert set(variants) <= set(v21.ACTIONS[category]), category
    assert set(v21.ACTIONS) == set(v20.ACTIONS)


def test_the_control_rows_stay_non_escalating() -> None:
    for item in control_rows():
        lowered = item["messages"][1]["content"].lower()
        assert not any(term in lowered for term in v13.v9.HOTLINE_TERMS), item["id"]


def test_the_family_is_train_only_and_carries_no_tool_catalog() -> None:
    for item in v21.collect_rows()["train"]:
        assert item["tools"] == []
        assert [message["role"] for message in item["messages"]] == ["user", "assistant"]


def test_the_new_disclosures_stay_clear_of_the_evaluation_prompts() -> None:
    """A near-copy of an eval prompt would memorise the case instead of the register."""
    assert v21.paraphrases_too_close_to_evaluation(v21.EXTRA_PARAPHRASES) == []


def test_the_anchor_guard_rejects_a_reply_that_names_nothing() -> None:
    item = crisis_rows()[0]
    category = item["category"]
    with pytest.raises(ValueError, match="names none of the category anchors"):
        v21.validate_surface_row(item, v21.ACTIONS[category], ("an anchor no reply contains",))
