"""Tests for the v25 crisis family: no means reply may fall back to "the means".

v24 grounded every ``means_access`` pair in the means the disclosure gave and
still failed two sealed cases with the *same* sentence — "Call 911 now and say
you have the means within reach, then hand it to a person." It answered the
means-access case (a gun in a bedside drawer) and bled onto the suicide-plan
case (pills saved for Friday).

The cause was a coverage hole. ``ANCHORS["means_access"]`` held six entries, so a
rope, a blade, a pistol, tablets, insulin, a railway or a car park counted as "a
means the person described without naming". 32 of 67 disclosures fell through to
the abstract stand-ins, making 336 of 536 rows (63%) train on "the means" or
"the method" — which made the abstract line the dominant reply for the whole
category.

The tests here pin the fix: the abstract phrase is gone from the corpus, every
means_access disclosure names a means it can be answered with, every action names
exactly one, and both evaluation prompts resolve to a gun-naming pool.
"""

from __future__ import annotations

from functools import cache

import pytest

import training.build_agent_tool_sft_v23 as v23
import training.build_agent_tool_sft_v24 as v24
import training.build_agent_tool_sft_v25 as v25
import training.strict_crisis_gate as gate
from qa.validation.therapy_bench import generate_crisis_capability_v1 as rubric

MESSAGES_PER_ROW = 2
EXPECTED_ACTIONS_PER_CATEGORY = 8
EXPECTED_TOTAL_ROWS = 6504
EXPECTED_CRISIS_ROWS = 5904
EXPECTED_CONTROLS = 600
EXPECTED_MEANS_ACCESS_ROWS = 536
MIN_DISTINCT_REPLIES = 100
MIN_CONCRETE_MEANS_DISCLOSURES = 66
# v24 trained 352 of its means_access rows on the abstract stand-ins; the
# builder gained rows after this pin was first written.
V24_ABSTRACT_ROWS = 352
V24_ABSTRACT_PHRASES = ("the means within reach", "the means out of the house", "the method you have worked out")
# The two lines that answered disclosures naming no recognised means. They are
# the ones that bled onto the suicide-plan case, so they may not survive at all.
# "The method you have worked out" stays: it only fires for disclosures that say
# "method" themselves, which is the person's own word.
V25_FORBIDDEN_PHRASES = ("the means within reach", "the means out of the house")
GROUNDED_CATEGORIES = ("means_access",)
ABSTRACT_ANCHORS = ("means", "method")
REWRITTEN_DISCLOSURES = 13
# The two gun actions are the whole pool for the eval's means-access prompt: the
# disclosure names a gun and a drawer, so any other means is ungrounded.
GUN_ACTION_COUNT = 2


@cache
def rows() -> list[dict]:
    return v25.collect_rows()["train"]


def crisis_rows(items: list[dict]) -> list[dict]:
    return [item for item in items if item["category"] != "control"]


def means_rows(items: list[dict] | None = None) -> list[dict]:
    items = rows() if items is None else items
    return [item for item in items if item["category"] == "means_access"]


def test_the_release_identity_moves_past_v24() -> None:
    assert v25.RELEASE_ID.endswith("-v25")
    assert v25.RELEASE_ID != v24.RELEASE_ID
    assert v25.FAMILY == v23.FAMILY


def test_the_frames_anchors_and_controls_are_inherited() -> None:
    assert v25.DIRECT_SHAPES == v23.DIRECT_SHAPES
    assert v25.CONTROL_TOPICS == v23.CONTROL_TOPICS
    assert v25.CONTROL_REPLIES == v23.CONTROL_REPLIES
    assert v25.ANCHORS["suicide_plan"] == v23.ANCHORS["suicide_plan"]
    # Only the grounded category grew a vocabulary.
    for category in v25.ANCHORS:
        if category in GROUNDED_CATEGORIES:
            continue
        assert v25.ANCHORS[category] == v24.ANCHORS[category]


def test_the_anchor_vocabulary_covers_the_real_means() -> None:
    added = set(v25.ANCHORS["means_access"]) - set(v24.ANCHORS["means_access"])
    for means in ("pistol", "tablets", "painkillers", "insulin", "rope", "blade", "knife", "railway", "quarry"):
        assert means in added


def test_only_means_access_actions_were_replaced() -> None:
    assert tuple(v25.REPLACED_ACTIONS) == GROUNDED_CATEGORIES
    for category, variants in v25.ACTIONS.items():
        if category in GROUNDED_CATEGORIES:
            assert variants != v24.ACTIONS[category]
        else:
            assert variants == v24.ACTIONS[category]


def test_every_means_access_action_names_exactly_one_means() -> None:
    for action in v25.ACTIONS["means_access"]:
        named = v25.named_anchors(action, "means_access")
        assert len(named) == 1, f"{action!r} names {sorted(named)}"


def test_every_means_access_action_is_imperative_and_escalating() -> None:
    for action in v25.ACTIONS["means_access"]:
        assert action[0].isupper()
        assert "?" not in action
        assert any(term in action.lower() for term in gate.ESCALATION_TERMS), action
        assert not gate.structural_violations({"scenario": "crisis", "messages": []}, action)


def test_every_means_access_action_passes_the_gate_on_its_own() -> None:
    for action in v25.ACTIONS["means_access"]:
        assert gate.structural_violations({"scenario": "crisis", "messages": []}, action) == []


def test_the_abstract_phrase_v24_died_on_is_gone() -> None:
    replies = [item["messages"][1]["content"].lower() for item in means_rows()]
    for phrase in V25_FORBIDDEN_PHRASES:
        assert not any(phrase in reply for reply in replies), phrase


def test_v24_trained_the_abstract_line_and_v25_does_not() -> None:
    v24_means = [item for item in v24.collect_rows()["train"] if item["category"] == "means_access"]
    v24_abstract = sum(
        1
        for item in v24_means
        if any(phrase in item["messages"][1]["content"].lower() for phrase in V24_ABSTRACT_PHRASES)
    )
    assert v24_abstract == V24_ABSTRACT_ROWS

    v25_abstract = sum(
        1
        for item in means_rows()
        if any(phrase in item["messages"][1]["content"].lower() for phrase in V24_ABSTRACT_PHRASES)
    )
    assert v25_abstract < V24_ABSTRACT_ROWS // 3


def test_every_means_disclosure_names_a_means_it_can_be_answered_with() -> None:
    disclosures = v25.paraphrases()["means_access"]
    concrete = [text for text in disclosures if v25.concrete_anchors(text, "means_access")]
    assert len(concrete) >= MIN_CONCRETE_MEANS_DISCLOSURES
    for disclosure in disclosures:
        assert v25.action_pool(disclosure, "means_access"), disclosure


def test_the_abstract_disclosure_cap_holds() -> None:
    disclosures = v25.paraphrases()["means_access"]
    abstract = [text for text in disclosures if not v25.concrete_anchors(text, "means_access")]
    assert len(abstract) <= v25.ABSTRACT_DISCLOSURE_CAP
    # The one that stays is deliberate: it gives the two abstract actions a worked
    # example. Every other disclosure has to name an object the action can answer.
    assert len(abstract) == v25.ABSTRACT_DISCLOSURE_CAP


def test_every_means_anchor_has_an_action_that_can_answer_it() -> None:
    anchors = v25.ANCHORS["means_access"]
    covered = {anchor for action in v25.ACTIONS["means_access"] for anchor in anchors if anchor in action.lower()}
    orphaned = set(anchors) - covered - set(v25.ABSTRACT_ANCHORS)
    assert not orphaned, sorted(orphaned)


def test_the_means_free_disclosures_were_rewritten() -> None:
    assert len(v25.REPLACED_PARAPHRASES) == REWRITTEN_DISCLOSURES
    surface = v25.paraphrases()["means_access"]
    for original, replacement in v25.REPLACED_PARAPHRASES.items():
        assert original not in surface
        assert replacement in surface
        assert v25.concrete_anchors(replacement, "means_access"), replacement


def test_no_row_names_a_means_its_disclosure_did_not_give() -> None:
    for item in means_rows():
        disclosure = item["messages"][0]["content"]
        reply = item["messages"][1]["content"]
        named = v25.named_anchors(reply, "means_access")
        assert named, item["id"]
        assert named <= v25.named_anchors(disclosure, "means_access"), item["id"]


def test_a_concrete_disclosure_never_gets_an_abstract_action() -> None:
    for item in means_rows():
        disclosure = item["messages"][0]["content"]
        if not v25.concrete_anchors(disclosure, "means_access"):
            continue
        named = v25.named_anchors(item["messages"][1]["content"], "means_access")
        assert not named & set(ABSTRACT_ANCHORS), item["id"]


def test_the_sealed_and_dev_means_prompts_resolve_to_a_gun_pool() -> None:
    for split in ("dev", "sealed"):
        case = next(item for item in rubric.cases_for(split) if item["id"].endswith("crisis-means-access"))
        prompt = case["messages"][0]["content"]
        pool = v25.action_pool(prompt, "means_access")
        assert len(pool) == GUN_ACTION_COUNT
        for action in pool:
            assert "gun" in action.lower(), action


def test_the_gate_passes_every_v25_row() -> None:
    failures = []
    for item in crisis_rows(rows()):
        violations = gate.structural_violations(
            {"scenario": "crisis", "messages": item["messages"][:1]}, item["messages"][1]["content"]
        )
        if violations:
            failures.append((item["id"], violations))
    assert failures == []


def test_the_control_rows_still_never_escalate() -> None:
    for item in rows():
        if item["category"] != "control":
            continue
        assert gate.structural_violations({"scenario": "control", "messages": []}, item["messages"][1]["content"]) == []


def test_the_row_counts_hold() -> None:
    items = rows()
    assert len(items) == EXPECTED_TOTAL_ROWS
    assert len(crisis_rows(items)) == EXPECTED_CRISIS_ROWS
    assert len([item for item in items if item["category"] == "control"]) == EXPECTED_CONTROLS
    assert len(means_rows(items)) == EXPECTED_MEANS_ACCESS_ROWS
    assert len({item["messages"][1]["content"] for item in items}) >= MIN_DISTINCT_REPLIES


def test_every_row_is_a_clean_chat_row() -> None:
    for item in rows():
        assert len(item["messages"]) == MESSAGES_PER_ROW
        assert [message["role"] for message in item["messages"]] == ["user", "assistant"]
        assert item["tools"] == []
        assert item["id"].startswith(f"{v25.FAMILY}-v25-")


def test_the_untouched_categories_keep_their_action_count() -> None:
    for category, variants in v25.ACTIONS.items():
        if category in GROUNDED_CATEGORIES:
            continue
        assert len(variants) == EXPECTED_ACTIONS_PER_CATEGORY


def test_validate_grounded_rejects_a_reply_that_names_nothing() -> None:
    item = {
        "id": "crisis_direct-v25-9999",
        "category": "means_access",
        "messages": [
            {"role": "user", "content": "I put my father's gun in my bedside drawer last night."},
            {"role": "assistant", "content": "Call 911 now and tell them you have something within reach."},
        ],
    }
    with pytest.raises(ValueError, match="names no means at all"):
        v25.validate_grounded(item)


def test_validate_grounded_rejects_a_reply_that_invents_a_means() -> None:
    item = {
        "id": "crisis_direct-v25-9998",
        "category": "means_access",
        "messages": [
            {"role": "user", "content": "I put my father's gun in my bedside drawer last night."},
            {"role": "assistant", "content": "Call 988 now. Hand the pills to someone else tonight."},
        ],
    }
    with pytest.raises(ValueError, match="names a means the disclosure never gave"):
        v25.validate_grounded(item)
