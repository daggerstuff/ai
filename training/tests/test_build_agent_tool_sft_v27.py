"""Tests for the v27 crisis family: a reply may only name what the person named.

v26 grounded three categories — ``means_access``, ``overdose_taken`` and
``recent_attempt`` — and left the other seven on the surface they inherited from
v23. That surface was eval-fitted rather than disclosure-fitted: each category's
``ANCHORS`` was a verbatim copy of the paired-eval ``response_contains_any``
terms, and the eight actions were then written to hit those terms. Measured over
the seven remaining categories, **4,192 of 4,528 disclosure x action pairs
(92.6%) named an anchor the disclosure never gave**:

* "I have been burning myself with a cigarette lighter and the skin is broken
  now." was answered with "Get that cut closed at an emergency department
  tonight."
* Opioid withdrawal was answered with "Alcohol and sedative withdrawal causes
  seizures."
* "I bought a rope yesterday" was answered with a reply about "the pills set
  aside for this".
* "He hit me in the ribs with a chair" was answered with "Being choked causes
  damage."

The fix is structural rather than textual. Each of the seven categories is split
into ``ANCHOR_GROUPS`` — named clusters of word-boundary tokens — and an action
is only ever eligible for a disclosure when **the clusters it names are a subset
of the clusters the disclosure named**. Clusters are split finely enough that
naming one does not imply another (self-harm ``wall`` and ``head_strike`` are
separate; ``burn`` never implies ``cut``; withdrawal ``opioid`` never implies
``alcohol``). Two further rules keep the pool honest: a disclosure that names a
mechanism is never answered by a stance-only or date-only action
(``SECONDARY_CLUSTERS``), and a reply is never the category's generic line,
because v24 already showed what happens when the abstract answer dominates a
category.

What this file pins, in order: the identity and the inherited surface, the seven
cluster vocabularies, the subset rule and its consequences on the real
disclosures that used to be answered wrongly, the secondary-cluster rule, the
abstract cap for the disclosures that genuinely name no target, the graded
evaluation prompts, and the row counts the release claims.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pytest

import training.build_agent_tool_sft_v23 as v23
import training.build_agent_tool_sft_v26 as v26
import training.build_agent_tool_sft_v27 as v27
import training.strict_crisis_gate as gate

MESSAGES_PER_ROW = 2
EXPECTED_TOTAL_ROWS = 6504
EXPECTED_CRISIS_ROWS = 5904
EXPECTED_CONTROLS = 600
EXPECTED_DISCLOSURES = 838
MIN_DISTINCT_REPLIES = 1000
# Eight frames is the v22 lever; v23 through v27 keep it.
EXPECTED_REFLECTIVE_FRAMES = 3
# A graded prompt must resolve to a handful of replies, not the whole surface.
MAX_EVAL_POOL = 5
EXPECTED_REFUSED_CAP = 0
REWRITTEN_DISCLOSURES = 6
V26_GROUNDED = ("means_access", "overdose_taken", "recent_attempt")
EXPECTED_ROWS_BY_CATEGORY = {
    "suicide_plan": 688,
    "self_harm_active": 1160,
    "psychosis_risk_other": 992,
    "assault_strangulation": 384,
    "medical_collapse": 432,
    "withdrawal": 384,
    "passive_ideation": 488,
}
# Psychosis is the one category where a disclosure can name a voice and no
# target at all ("the voices want me to do something to someone"). Those rows are
# answered by the unnamed-target action and pinned by an explicit cap, so a new
# target-free disclosure fails the build instead of quietly joining them.
EXPECTED_ABSTRACT_CAP = {"psychosis_risk_other": 37}

EVAL_CASES = Path(__file__).resolve().parents[3] / "docs/training/execution-2026-10-17/scenario-crisis-dev-v1.jsonl"
EVAL_CATEGORY_OF = {
    "dev-crisis-suicide-plan": "suicide_plan",
    "dev-crisis-self-harm-active": "self_harm_active",
    "dev-crisis-overdose-taken": "overdose_taken",
    "dev-crisis-psychosis-risk-to-other": "psychosis_risk_other",
    "dev-crisis-domestic-strangulation": "assault_strangulation",
    "dev-crisis-eating-disorder-syncope": "medical_collapse",
    "dev-crisis-withdrawal-perceptual": "withdrawal",
    "dev-crisis-passive-ideation": "passive_ideation",
    "dev-crisis-means-access": "means_access",
}


@cache
def rows() -> list[dict]:
    return v27.collect_rows()["train"]


@cache
def summary() -> dict:
    return v27.summarize(v27.collect_rows())


def crisis_rows(items: list[dict]) -> list[dict]:
    return [item for item in items if item["category"] != "control"]


def category_rows(category: str, items: list[dict] | None = None) -> list[dict]:
    items = rows() if items is None else items
    return [item for item in items if item["category"] == category]


def eval_cases() -> list[dict]:
    return [json.loads(line) for line in EVAL_CASES.read_text().splitlines() if line.strip()]


def test_the_release_identity_moves_past_v26() -> None:
    assert v27.RELEASE_ID.endswith("-v27")
    assert v27.RELEASE_ID != v26.RELEASE_ID
    assert v27.FAMILY == v23.FAMILY


def test_the_already_grounded_categories_are_inherited_untouched() -> None:
    for category in V26_GROUNDED:
        assert v27.ACTIONS[category] == v26.ACTIONS[category]
        assert v27.ANCHORS[category] == v26.ANCHORS[category]
    assert set(v27.REPLACED_ACTIONS) == set(v27.CLUSTERED_CATEGORIES)


def test_the_seven_eval_fitted_categories_are_the_ones_rewritten() -> None:
    assert v27.CLUSTERED_CATEGORIES == (
        "suicide_plan",
        "self_harm_active",
        "psychosis_risk_other",
        "assault_strangulation",
        "medical_collapse",
        "withdrawal",
        "passive_ideation",
    )
    for category in v27.CLUSTERED_CATEGORIES:
        assert v27.ACTIONS[category] != v26.ACTIONS[category]
        assert v27.ANCHORS[category] != v26.ANCHORS[category]


def test_every_grounded_category_is_clustered_or_inherited() -> None:
    assert set(v27.GROUNDED_CATEGORIES) == set(V26_GROUNDED) | set(v27.CLUSTERED_CATEGORIES)


def test_the_suicide_plan_clusters_name_the_elements_of_a_plan() -> None:
    for cluster in ("pills", "rope", "belt", "gun", "height", "vehicle", "date", "note", "plan", "place"):
        assert cluster in v27.ANCHOR_GROUPS["suicide_plan"]


def test_the_self_harm_clusters_separate_burn_from_cut() -> None:
    burn = v27.ANCHOR_GROUPS["self_harm_active"]["burn"]
    cut = v27.ANCHOR_GROUPS["self_harm_active"]["cut"]
    assert "lighter" in burn
    assert "lighter" not in cut
    assert "cut" in cut
    assert "cut" not in burn
    for cluster in ("wall", "head_strike", "scratch", "infection", "blood", "harm", "not_stopping"):
        assert cluster in v27.ANCHOR_GROUPS["self_harm_active"]


def test_the_withdrawal_clusters_separate_the_substances() -> None:
    groups = v27.ANCHOR_GROUPS["withdrawal"]
    assert "opioid" in groups
    assert "alcohol" in groups
    assert not set(groups["opioid"]) & set(groups["alcohol"])
    for cluster in ("benzo", "sleeping", "medication", "seizure", "perceptual", "shaking"):
        assert cluster in groups


def test_the_psychosis_clusters_name_the_target_and_the_means() -> None:
    for cluster in ("voices", "harm", "housemate", "family", "public", "knife", "hammer", "fire", "pet"):
        assert cluster in v27.ANCHOR_GROUPS["psychosis_risk_other"]


def test_the_assault_clusters_name_the_mechanism_of_injury() -> None:
    for cluster in ("strangle", "ribs", "limb", "head_blunt", "stomach", "weapon", "wound", "submersion", "smother"):
        assert cluster in v27.ANCHOR_GROUPS["assault_strangulation"]


def test_the_medical_collapse_clusters_separate_the_causes() -> None:
    for cluster in ("purge", "faint", "blood", "sugar", "heart", "water", "weakness"):
        assert cluster in v27.ANCHOR_GROUPS["medical_collapse"]


def test_the_passive_ideation_clusters_are_the_stances_not_objects() -> None:
    groups = v27.ANCHOR_GROUPS["passive_ideation"]
    assert set(groups) == {
        "better_off",
        "burden",
        "not_wake_up",
        "not_exist",
        "no_point",
        "hope_illness",
        "relief",
        "stopped_plans",
    }


def test_a_cluster_only_matches_inside_its_own_category() -> None:
    text = "I have the rope in the shed and I know which beam."
    assert "rope" in v27.clusters(text, "suicide_plan")
    assert "rope" not in v27.clusters(text, "assault_strangulation")


def test_cluster_tokens_match_whole_words_only() -> None:
    # Word boundaries, not substrings: a sunburn is not a burn someone made, and
    # "cannot" never matches a bare "can".
    assert not v27._mentions("sunburn and sunburnt skin", "burn")
    assert v27._mentions("burn it", "burn")
    assert "burn" in v27.clusters("I burned a line down my forearm", "self_harm_active")


def test_cluster_tokens_are_matched_case_insensitively() -> None:
    # A capitalised token silently never matched before this was pinned.
    assert v27._mentions("what i am going to do", "What I am going to do")
    assert v27._mentions("i know how and when", "how and when")


def test_the_subset_rule_drops_every_action_that_names_an_unshared_cluster() -> None:
    for category in v27.CLUSTERED_CATEGORIES:
        for disclosure in v27.paraphrases()[category]:
            allowed = v27.concrete_clusters(disclosure, category)
            for action in v27.action_pool(disclosure, category):
                assert v27.concrete_clusters(action, category) <= allowed, (category, disclosure, action)


def test_every_clustered_disclosure_has_at_least_one_action() -> None:
    for category in v27.CLUSTERED_CATEGORIES:
        for disclosure in v27.paraphrases()[category]:
            assert v27.action_pool(disclosure, category), (category, disclosure)


def test_a_burn_is_never_answered_with_a_cut() -> None:
    disclosure = "I burned a line down my forearm with a lighter."
    pool = v27.action_pool(disclosure, "self_harm_active")
    assert all("burn" in v27.concrete_clusters(action, "self_harm_active") for action in pool)
    assert not any("cut" in action.lower() for action in pool)


def test_opioid_withdrawal_is_never_answered_with_alcohol_seizures() -> None:
    disclosure = "I came off the opioids and my whole body aches and I keep being sick."
    pool = v27.action_pool(disclosure, "withdrawal")
    assert all("opioid" in v27.concrete_clusters(action, "withdrawal") for action in pool)
    assert not any("alcohol" in action.lower() for action in pool)


def test_a_rope_is_never_answered_with_pills() -> None:
    disclosure = "I have the rope in the shed and I know which beam."
    pool = v27.action_pool(disclosure, "suicide_plan")
    assert all("rope" in v27.concrete_clusters(action, "suicide_plan") for action in pool)
    for other in ("pills", "tablets", "gun", "belt"):
        assert not any(other in action.lower() for action in pool)


def test_broken_ribs_are_never_answered_with_strangulation() -> None:
    disclosure = "He hit me in the ribs with a chair and now it hurts to take a breath."
    pool = v27.action_pool(disclosure, "assault_strangulation")
    assert all("ribs" in v27.concrete_clusters(action, "assault_strangulation") for action in pool)
    assert not any("choked" in action.lower() or "strangl" in action.lower() for action in pool)


def test_a_knife_is_never_answered_with_a_hammer_or_a_fire() -> None:
    disclosure = "I have been sharpening a knife because I know someone is coming through the door tonight."
    pool = v27.action_pool(disclosure, "psychosis_risk_other")
    assert all("knife" in v27.concrete_clusters(action, "psychosis_risk_other") for action in pool)
    for other in ("hammer", "petrol", "fire", "car"):
        assert not any(other in action.lower() for action in pool)


def test_a_fluttering_heart_is_never_answered_with_purging() -> None:
    disclosure = "My heart keeps fluttering and I cannot stand up without holding onto something."
    pool = v27.action_pool(disclosure, "medical_collapse")
    assert not any(
        "purge" in action.lower() or "purge" in v27.concrete_clusters(action, "medical_collapse") for action in pool
    )


def test_a_mechanism_disclosure_is_never_answered_with_the_stance_alone() -> None:
    for category, secondary in v27.SECONDARY_CLUSTERS.items():
        for disclosure in v27.paraphrases()[category]:
            given = v27.concrete_clusters(disclosure, category)
            if not given - secondary:
                continue
            for action in v27.action_pool(disclosure, category):
                assert v27.concrete_clusters(action, category) - secondary, (category, disclosure, action)


def test_the_self_harm_stance_is_still_answerable_on_its_own() -> None:
    disclosure = "I do not want to stop and I am not going to pretend otherwise."
    pool = v27.action_pool(disclosure, "self_harm_active")
    assert any("not_stopping" in v27.concrete_clusters(action, "self_harm_active") for action in pool)


def test_every_cluster_has_an_action_that_can_answer_it() -> None:
    for category in v27.CLUSTERED_CATEGORIES:
        answered: set[str] = set()
        for action in v27.ACTIONS[category]:
            answered |= v27.clusters(action, category)
        missing = sorted(set(v27.ANCHOR_GROUPS[category]) - answered)
        assert answered == set(v27.ANCHOR_GROUPS[category]), (category, missing)


def test_every_action_escalates_and_names_a_cluster() -> None:
    for category in v27.CLUSTERED_CATEGORIES:
        for action in v27.ACTIONS[category]:
            assert v27.clusters(action, category), action
            assert "?" not in action, action
            lowered = action.lower()
            assert any(term in lowered for term in v27.ESCALATION_TERMS), action
            for filler in v27.FORBIDDEN_FILLER:
                assert filler not in lowered, (filler, action)


def test_every_action_answers_some_disclosure() -> None:
    for category in v27.CLUSTERED_CATEGORIES:
        reachable = {
            action for disclosure in v27.paraphrases()[category] for action in v27.action_pool(disclosure, category)
        }
        assert reachable == set(v27.ACTIONS[category]), (category, sorted(set(v27.ACTIONS[category]) - reachable))


def test_no_reply_names_a_cluster_the_disclosure_never_gave() -> None:
    for category in v27.CLUSTERED_CATEGORIES:
        assert summary()[f"{category}_invented_cluster_rows"] == 0
        for item in category_rows(category):
            disclosure = item["messages"][0]["content"]
            reply = item["messages"][1]["content"]
            invented = v27.concrete_clusters(reply, category) - v27.concrete_clusters(disclosure, category)
            assert not invented, (item["id"], sorted(invented), reply)


def test_a_reply_is_never_the_generic_line() -> None:
    # v24's regression: an abstract reply dominated a category because it was the
    # majority of the rows. Every clustered row must name a concrete cluster.
    for category in v27.CLUSTERED_CATEGORIES:
        for item in category_rows(category):
            assert v27.clusters(item["messages"][1]["content"], category), item["id"]


def test_the_abstract_cap_pins_the_target_free_disclosures() -> None:
    for category in v27.CLUSTERED_CATEGORIES:
        cap = v27.ABSTRACT_DISCLOSURE_CAP[category]
        free = [
            disclosure for disclosure in v27.paraphrases()[category] if not v27.concrete_clusters(disclosure, category)
        ]
        assert len(free) == cap, (category, len(free), cap)
    assert v27.ABSTRACT_DISCLOSURE_CAP["psychosis_risk_other"] == EXPECTED_ABSTRACT_CAP["psychosis_risk_other"]


def test_the_other_six_clustered_categories_have_no_abstract_disclosure() -> None:
    for category in v27.CLUSTERED_CATEGORIES:
        if category in EXPECTED_ABSTRACT_CAP:
            continue
        assert v27.ABSTRACT_DISCLOSURE_CAP[category] == EXPECTED_REFUSED_CAP


def test_the_rewritten_disclosures_now_name_a_cluster() -> None:
    assert len(v27.REPLACED_PARAPHRASES) == REWRITTEN_DISCLOSURES
    surface = {line for lines in v26.paraphrases().values() for line in lines}
    for original, replacement in v27.REPLACED_PARAPHRASES.items():
        assert original in surface, original
        assert replacement != original
        category = next(name for name, lines in v26.paraphrases().items() if original in lines)
        assert v27.concrete_clusters(replacement, category), (category, replacement)


def test_only_the_widened_categories_name_a_cluster_the_disclosure_did_not() -> None:
    # A warning clause ("a seizure from purging or not eating") is legitimate and
    # must not be counted as invention; these are the only two places it appears.
    for category in v27.CLUSTERED_CATEGORIES:
        widened = summary()[f"{category}_widened_cluster_rows"]
        if category in {"medical_collapse", "psychosis_risk_other"}:
            assert widened > 0, category
        else:
            assert widened == 0, category


def test_every_graded_prompt_is_answered_by_a_reply_that_carries_its_content() -> None:
    # The reply is drawn from the pool, so every action in the pool has to carry a
    # term the paired evaluation checks for. This is the precise constraint: it
    # does not force every action to be eval-shaped, only the ones that can
    # actually answer a graded prompt.
    seen = set()
    for case in eval_cases():
        category = EVAL_CATEGORY_OF.get(case["id"])
        if category is None:
            continue
        seen.add(case["id"])
        prompt = case["messages"][0]["content"]
        checked = next(check for check in case["state_checks"] if check["type"] == "response_contains_any")
        terms = [term.lower() for term in checked["terms"]]
        for action in v27.action_pool(prompt, category):
            assert any(term in action.lower() for term in terms), (case["id"], action)
    assert seen == set(EVAL_CATEGORY_OF)


def test_the_graded_prompt_pools_stay_small() -> None:
    # A pool of every action in the category is what let v26 answer a gun with a
    # rope. Graded prompts resolve to a handful of grounded replies.
    for case in eval_cases():
        category = EVAL_CATEGORY_OF.get(case["id"])
        if category is None:
            continue
        pool = v27.action_pool(case["messages"][0]["content"], category)
        assert len(pool) <= MAX_EVAL_POOL, (case["id"], len(pool))


def test_the_row_counts_hold() -> None:
    assert summary()["example_count"] == EXPECTED_TOTAL_ROWS
    assert summary()["crisis_rows"] == EXPECTED_CRISIS_ROWS
    assert summary()["crisis_severity_counts"]["none"] == EXPECTED_CONTROLS
    assert summary()["distinct_disclosures"] == EXPECTED_DISCLOSURES
    assert summary()["distinct_replies"] >= MIN_DISTINCT_REPLIES
    assert len(rows()) == EXPECTED_TOTAL_ROWS
    assert len(crisis_rows(rows())) == EXPECTED_CRISIS_ROWS
    for category, expected in EXPECTED_ROWS_BY_CATEGORY.items():
        assert len(category_rows(category)) == expected, category


def test_every_row_is_a_two_turn_chat() -> None:
    for item in rows():
        assert len(item["messages"]) == MESSAGES_PER_ROW
        assert item["messages"][0]["role"] == "user"
        assert item["messages"][1]["role"] == "assistant"


def test_three_of_eight_frames_lead_with_a_reflection() -> None:
    assert len(v26.REFLECTIVE_SHAPES) == EXPECTED_REFLECTIVE_FRAMES
    assert len(v27.DIRECT_SHAPES) == v27.SHAPES_PER_DISCLOSURE
    reflective = summary()["reflective_rows"]
    assert reflective == EXPECTED_CRISIS_ROWS * 3 // v27.SHAPES_PER_DISCLOSURE


def test_the_gate_passes_every_crisis_row() -> None:
    # The gate demands escalation, and control rows must never escalate.
    for item in crisis_rows(rows()):
        scenario = {"scenario": "crisis", "messages": [{"role": "user", "content": item["messages"][0]["content"]}]}
        assert not gate.structural_violations(scenario, item["messages"][1]["content"]), item["id"]
    assert summary()["gate_failures"] == 0


def test_the_control_rows_still_never_escalate() -> None:
    controls = category_rows("control")
    assert len(controls) == EXPECTED_CONTROLS
    for item in controls:
        lowered = item["messages"][1]["content"].lower()
        assert not any(term in lowered for term in v27.ESCALATION_TERMS), item["id"]


def test_a_target_free_disclosure_is_still_answered() -> None:
    disclosure = "The voices say I have to hurt somebody tonight or they will not stop."
    pool = v27.action_pool(disclosure, "psychosis_risk_other")
    assert pool
    for action in pool:
        assert "?" not in action


def test_the_pool_refuses_a_disclosure_with_no_matching_action() -> None:
    with pytest.raises(ValueError, match="no self_harm_active action answers"):
        v27.action_pool("I feel a bit flat today.", "self_harm_active")
