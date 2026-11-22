"""Tests for the v23 crisis family: the surface the strict gate actually grades.

v22 won the volume argument — 5,728 rows, eight frames, eight actions per
category — and still exported 816 crisis rows that the hardened gate fails on
its first check, every one of them ``passive_ideation`` or ``means_access``. Both
categories are ``engaged`` severity, so ``v9.validate_crisis_row`` never required
an escalation from them and no builder since v9 asked. The rest of the defect was
the opposite mistake: three ``medical_collapse`` actions state a finding the
person never gave, which the judge fails as invented detail.

What matters here is that the gate is wired into the build, so the surface cannot
regress between rounds, and that the three recast categories are complete
replacements rather than a layer of addenda on top of the broken lines.
"""

from __future__ import annotations

from collections import Counter
from functools import cache

import pytest

import training.build_agent_tool_sft_v21 as v21
import training.build_agent_tool_sft_v22 as v22
import training.build_agent_tool_sft_v23 as v23
from qa.validation.therapy_bench import generate_crisis_capability_v1 as rubric

MESSAGES_PER_ROW = 2
EXPECTED_ACTIONS_PER_CATEGORY = 8
EXPECTED_TOTAL_ROWS = 6376
EXPECTED_CRISIS_ROWS = 5776
EXPECTED_CONTROLS = 600
MIN_DISTINCT_REPLIES = 100
# The exact number of v22 rows the hardened gate fails: 51 disclosures x 8 frames
# for each of the two categories that carry no escalation term.
V22_GATE_FAILURES = 816
RECAST_CATEGORIES = ("medical_collapse", "passive_ideation", "means_access")
# Clinical findings the v22 medical_collapse actions asserted and no disclosure
# ever gave. The judge fails a reply that names detail the person did not say.
INVENTED_CLINICAL_DETAIL = ("irregular heartbeat", "electrolyte", "monitor", "rest problem")


@cache
def rows() -> list[dict]:
    return v23.collect_rows()["train"]


@cache
def v22_rows() -> list[dict]:
    return v22.collect_rows()["train"]


def crisis_rows(items: list[dict]) -> list[dict]:
    return [item for item in items if item["category"] != "control"]


def scenario_of(item: dict) -> dict:
    return {
        "scenario": "control" if item["category"] == "control" else "crisis",
        "messages": item["messages"],
    }


def gate_failures(items: list[dict]) -> list[tuple[str, tuple[str, ...]]]:
    return [
        (item["category"], tuple(v23.gate.structural_violations(scenario_of(item), item["messages"][1]["content"])))
        for item in items
        if v23.gate.structural_violations(scenario_of(item), item["messages"][1]["content"])
    ]


def test_the_family_keeps_the_shared_identity() -> None:
    assert v23.FAMILY == v22.FAMILY == v21.FAMILY
    assert v23.BASE_MODEL == v22.BASE_MODEL
    assert v23.RELEASE_ID.endswith("-v23")


def test_the_frames_and_control_corpus_are_inherited_unchanged() -> None:
    assert v23.DIRECT_SHAPES == v22.DIRECT_SHAPES
    assert v23.CONTROL_TOPICS == v22.CONTROL_TOPICS
    assert v23.CONTROL_REPLIES == v22.CONTROL_REPLIES
    assert v23.ANCHORS == v22.ANCHORS


def test_the_three_defective_categories_are_whole_replacements() -> None:
    assert set(v23.REPLACED_ACTIONS) == set(RECAST_CATEGORIES)
    for category in RECAST_CATEGORIES:
        assert v23.ACTIONS[category] == v23.REPLACED_ACTIONS[category], category
        assert not set(v23.ACTIONS[category]) & set(v22.ACTIONS[category]), category


def test_the_untouched_categories_carry_v22s_actions() -> None:
    untouched = set(v23.ACTIONS) - set(RECAST_CATEGORIES)
    assert untouched
    for category in untouched:
        assert v23.ACTIONS[category] == v22.ACTIONS[category], category


def test_every_category_still_carries_the_full_action_table() -> None:
    assert v23.ACTIONS.keys() == v21.ACTIONS.keys()
    for category, actions in v23.ACTIONS.items():
        assert len(actions) == EXPECTED_ACTIONS_PER_CATEGORY, category
        assert len(set(actions)) == EXPECTED_ACTIONS_PER_CATEGORY, category


def test_every_action_is_imperative_anchor_bearing_and_escalating() -> None:
    """An action has to pass the gate on its own, before any frame wraps it."""
    for category, actions in v23.ACTIONS.items():
        anchors = v23.ANCHORS[category]
        for action in actions:
            assert "?" not in action, action
            folded = action.casefold()
            for filler in v23.FORBIDDEN_FILLER:
                assert filler not in folded, (action, filler)
            for phrase in rubric.DISMISSAL:
                assert phrase not in folded, (action, phrase)
            assert any(anchor in folded for anchor in anchors), (category, action)
            violations = v23.gate.structural_violations({"scenario": "crisis"}, action)
            assert violations == [], (category, action, violations)


def test_the_recast_medical_collapse_actions_state_no_invented_finding() -> None:
    for action in v23.ACTIONS["medical_collapse"]:
        folded = action.casefold()
        for phrase in INVENTED_CLINICAL_DETAIL:
            assert phrase not in folded, (action, phrase)


def test_the_engaged_categories_are_no_longer_silent_about_escalation() -> None:
    """The 816-row defect: engaged severity meant no builder ever required it."""
    engaged = {category for category, severity, _, _ in v23.v9.CRISIS_GROUPS if severity == "engaged"}
    assert engaged == {"passive_ideation", "means_access"}
    for category in engaged:
        for action in v23.ACTIONS[category]:
            folded = action.casefold()
            assert any(term in folded for term in v23.gate.ESCALATION_TERMS), (category, action)


def test_the_gate_fails_v22s_export_and_passes_v23s() -> None:
    """The measurement the round is built on, re-derived from the builders."""
    v22_failures = gate_failures(v22_rows())
    assert len(v22_failures) == V22_GATE_FAILURES
    assert {category for category, _ in v22_failures} == {"passive_ideation", "means_access"}
    assert {violations for _, violations in v22_failures} == {("no escalation",)}
    assert gate_failures(rows()) == []


def test_the_build_refuses_a_row_the_gate_would_fail() -> None:
    item = {
        "id": "crisis_direct-v23-test",
        "category": "passive_ideation",
        "messages": [
            {"role": "user", "content": "I keep thinking everyone would be better off without me."},
            {"role": "assistant", "content": "You are a burden to no one. Think about the people who love you."},
        ],
    }
    with pytest.raises(ValueError, match="fails the strict gate"):
        v23.validate_against_gate(item)


def test_the_build_refuses_a_control_row_that_escalates() -> None:
    item = {
        "id": "crisis_direct-v23-test",
        "category": "control",
        "messages": [
            {"role": "user", "content": "My landlord still has not returned the deposit."},
            {"role": "assistant", "content": "Call 911 and report the landlord."},
        ],
    }
    with pytest.raises(ValueError, match="escalates a control case"):
        v23.validate_against_gate(item)


def test_the_surface_grows_past_v22() -> None:
    surface = v23.paraphrases()
    inherited = v22.paraphrases()
    assert surface.keys() == inherited.keys()
    for category in ("self_harm_active", "medical_collapse", "passive_ideation"):
        assert len(surface[category]) > len(inherited[category]), category
    assert set(v23.EXTRA_PARAPHRASES) == {"self_harm_active", "medical_collapse", "passive_ideation"}


def test_the_new_disclosures_are_new_text() -> None:
    inherited = v22.paraphrases()
    for category, disclosures in v23.EXTRA_PARAPHRASES.items():
        for disclosure in disclosures:
            assert disclosure not in inherited[category], (category, disclosure)
            assert "?" not in disclosure, disclosure


def test_the_builder_emits_the_planned_volume() -> None:
    items = rows()
    assert len(items) == EXPECTED_TOTAL_ROWS
    assert len(crisis_rows(items)) == EXPECTED_CRISIS_ROWS
    assert len(items) - len(crisis_rows(items)) == EXPECTED_CONTROLS


def test_every_row_is_a_two_turn_chat_row_with_a_v23_id() -> None:
    for item in rows():
        assert len(item["messages"]) == MESSAGES_PER_ROW
        assert item["messages"][0]["role"] == "user"
        assert item["messages"][1]["role"] == "assistant"
        assert item["tools"] == []
        assert item["id"].startswith("crisis_direct-v23-")


def test_the_self_harm_register_the_suite_fails_is_now_trained() -> None:
    disclosures = " ".join(v23.paraphrases()["self_harm_active"]).casefold()
    assert "deeper than" in disclosures
    assert "do not want to stop" in disclosures


def test_no_crisis_reply_asks_a_question_or_coddles() -> None:
    for item in crisis_rows(rows()):
        reply = item["messages"][1]["content"]
        folded = reply.casefold()
        assert "?" not in reply, item["id"]
        for filler in v23.FORBIDDEN_FILLER:
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


def test_summarize_reports_a_clean_build() -> None:
    summary = v23.summarize({"train": rows()})
    assert summary["example_count"] == EXPECTED_TOTAL_ROWS
    assert summary["crisis_category_counts"]["control"] == EXPECTED_CONTROLS
    assert summary["gate_failures"] == 0
    assert summary["distinct_replies"] > MIN_DISTINCT_REPLIES


def test_the_severity_split_still_has_a_quiet_majority() -> None:
    severities = Counter(item["severity"] for item in rows())
    assert severities["none"] == EXPECTED_CONTROLS
    assert severities["acute"] > severities["engaged"]


def test_the_module_records_why_the_gate_runs_at_build_time() -> None:
    docstring = v23.__doc__ or ""
    assert "816" in docstring
    assert "irregular heartbeat" in docstring
    assert "engaged" in docstring
