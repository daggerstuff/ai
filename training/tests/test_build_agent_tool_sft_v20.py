"""Tests for the v20 crisis family: more rows, a wider surface, and an anchor guard.

v20 keeps v13's register and action table and changes two things: the family is
large enough to compete with the corpus majority register, and every reply has
to name the concrete thing its disclosure named. Both are asserted here.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

import training.build_agent_tool_sft_v13 as v13
import training.build_agent_tool_sft_v20 as v20

SHAPES_PER_DISCLOSURE = 4
MIN_CRISIS_ROWS = 1_600
CONTROL_ROWS = len(v20.CONTROL_TOPICS) * len(v20.CONTROL_REPLIES)


def surface() -> dict[str, tuple[str, ...]]:
    return v20.paraphrases()


def crisis_rows() -> list[dict[str, Any]]:
    return [item for item in v20.collect_rows()["train"] if item["category"] != "control"]


def control_rows() -> list[dict[str, Any]]:
    return [item for item in v20.collect_rows()["train"] if item["category"] == "control"]


def test_the_family_has_the_expected_shape() -> None:
    crisis = sum(len(items) for items in surface().values()) * SHAPES_PER_DISCLOSURE
    assert len(crisis_rows()) == crisis
    assert len(control_rows()) == CONTROL_ROWS
    assert len(crisis_rows()) + len(control_rows()) == crisis + CONTROL_ROWS


def test_the_family_is_large_enough_to_compete_with_the_corpus() -> None:
    """v13's 636 crisis rows were 0.77% of train, which is why v19 still regressed."""
    v13_rows = len([item for item in v13.collect_rows()["train"] if item["category"] != "control"])
    assert len(crisis_rows()) >= MIN_CRISIS_ROWS
    assert len(crisis_rows()) > 2 * v13_rows


def test_every_crisis_reply_opens_with_one_of_its_category_actions() -> None:
    for item in crisis_rows():
        reply = item["messages"][1]["content"]
        matches = [action for action in v20.ACTIONS[item["category"]] if reply.startswith(action)]
        assert len(matches) == 1, item["id"]


def test_every_crisis_reply_names_its_category_anchor() -> None:
    """A category-level sentence that answers nothing must not be able to ship."""
    for item in crisis_rows():
        lowered = item["messages"][1]["content"].lower()
        anchors = v20.ANCHORS[item["category"]]
        assert any(anchor in lowered for anchor in anchors), (item["id"], anchors)


def test_the_anchor_guard_rejects_a_reply_that_names_nothing() -> None:
    """The guard is what stops a category-level sentence from shipping again."""
    item = crisis_rows()[0]
    category = item["category"]
    with pytest.raises(ValueError, match="names none of the category anchors"):
        v20.validate_surface_row(item, v20.ACTIONS[category], ("an anchor no reply contains",))


def test_no_crisis_reply_softens_the_action_or_asks_a_question() -> None:
    for item in crisis_rows():
        reply = item["messages"][1]["content"]
        lowered = reply.lower()
        assert "?" not in reply, item["id"]
        assert not [phrase for phrase in v20.FORBIDDEN_FILLER if phrase in lowered], item["id"]


def test_the_forbidden_list_covers_the_v19_loss_openers() -> None:
    """Both v19 crisis losses opened exactly like this, and the list must catch them."""
    for opener in ("that's a lot to carry alone.", "let's get you to a hospital", "i'll stay with you"):
        assert [phrase for phrase in v20.FORBIDDEN_FILLER if phrase in opener], opener


def test_the_surface_grows_in_every_category() -> None:
    grown = surface()
    inherited = v13.paraphrases()
    for category in v20.SEVERITIES:
        assert len(grown[category]) > len(inherited[category]), category


def test_the_surface_grows_where_v19_failed() -> None:
    """The two categories v19 lost are the two with the most rows."""
    counts = Counter(item["category"] for item in crisis_rows())
    largest = [category for category, _ in counts.most_common(2)]
    assert set(largest) == {"self_harm_active", "psychosis_risk_other"}
    assert counts["self_harm_active"] > counts["suicide_plan"]


def test_the_surface_keeps_every_v13_disclosure() -> None:
    """The growth is additive: v20 must not drop a disclosure v13 already taught."""
    inherited = {text for items in v13.paraphrases().values() for text in items}
    grown = {text for items in surface().values() for text in items}
    assert inherited <= grown
    assert len(grown) > len(inherited)


def test_the_control_rows_stay_non_escalating() -> None:
    for item in control_rows():
        lowered = item["messages"][1]["content"].lower()
        assert not any(term in lowered for term in v13.v9.HOTLINE_TERMS), item["id"]


def test_the_family_is_train_only_and_carries_no_tool_catalog() -> None:
    for item in v20.collect_rows()["train"]:
        assert item["tools"] == []
        assert [message["role"] for message in item["messages"]] == ["user", "assistant"]
