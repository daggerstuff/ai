"""Tests for the v29 crisis family: every grounded reply opens on its own anchor.

v28 grounded the last seven categories structurally: each category is split into
``ANCHOR_GROUPS`` — named clusters of word-boundary tokens — and an action is
only eligible for a disclosure when the clusters it names are a subset of the
clusters the disclosure named. That work is inherited here and pinned in full,
because v29 changes nothing about it.

v29 fixes one defect the v28 paired evaluation exposed and one it masked. The
graded firearm-access prompt was answered with an eating-disorder line ("A
seizure from purging or not eating needs a doctor tonight"), because 26 action
surfaces opened on a generic clause — a 911 call, "get to an emergency
department" — and only named the concrete cause in the second sentence. With no
anchor in the first clause there was nothing tying the continuation to the
disclosure, so the model could carry a neighbouring category's action into the
reply. Every one of those 26 surfaces is rewritten to open on the thing the
person actually named, and ``collect_rows`` refuses to build if any clustered
action still opens blind.

The rewrite is a reorder, not a new surface: it changes the cluster membership
of zero actions and leaves every action count and every disclosure untouched.
That is what the tests below pin, alongside the inherited v28 work — the seven
cluster vocabularies, the subset rule, the secondary-cluster rule, the abstract
cap, contraction normalisation, the graded evaluation prompts, and the row
counts the release claims.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

import training.build_agent_tool_sft_v23 as v23
import training.build_agent_tool_sft_v26 as v26
import training.build_agent_tool_sft_v28 as v28
import training.build_agent_tool_sft_v29 as v29
import training.strict_crisis_gate as gate
from qa.validation.therapy_bench import generate_crisis_capability_v2 as rubric_v2

MESSAGES_PER_ROW = 2
EXPECTED_TOTAL_ROWS = 6504
EXPECTED_CRISIS_ROWS = 5904
EXPECTED_CONTROLS = 600
EXPECTED_DISCLOSURES = 838
MIN_DISTINCT_REPLIES = 1000
# Eight frames is the v22 lever; v23 through v29 keep it.
EXPECTED_REFLECTIVE_FRAMES = 3
# A graded prompt must resolve to a handful of replies, not the whole surface.
MAX_EVAL_POOL = 5
EXPECTED_REFUSED_CAP = 0
REWRITTEN_DISCLOSURES = 7
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
# The v2 suite keeps the nine v1 dev prompts and replaces the sealed split with
# nine held-out prompts for the same categories, so the pools are checked against
# prompts the surfaces were never written for.
SUITE_ROOT = Path(__file__).resolve().parents[3] / "docs/training/execution-2026-10-17"
EVAL_CATEGORY_BY_VARIANT = {case_id.split("-", 1)[1]: category for case_id, category in EVAL_CATEGORY_OF.items()}


@cache
def rows() -> list[dict[str, Any]]:
    return v29.collect_rows()["train"]


@cache
def summary() -> dict[str, Any]:
    return v29.summarize(v29.collect_rows())


def crisis_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item["category"] != "control"]


def category_rows(category: str, items: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    items = rows() if items is None else items
    return [item for item in items if item["category"] == category]


def eval_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in EVAL_CASES.read_text().splitlines() if line.strip()]


def v2_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for split in ("dev", "sealed"):
        path = SUITE_ROOT / f"scenario-crisis-{split}-v2.jsonl"
        cases.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return cases


def test_the_release_identity_moves_past_v28() -> None:
    assert v29.RELEASE_ID.endswith("-v29")
    assert v29.RELEASE_ID != v26.RELEASE_ID
    assert v29.FAMILY == v23.FAMILY


def test_the_already_grounded_categories_are_inherited_untouched() -> None:
    for category in V26_GROUNDED:
        assert v29.ACTIONS[category] == v26.ACTIONS[category]
        assert v29.ANCHORS[category] == v26.ANCHORS[category]
    assert set(v29.REPLACED_ACTIONS) == set(v29.CLUSTERED_CATEGORIES)


def test_the_seven_eval_fitted_categories_are_the_ones_rewritten() -> None:
    assert v29.CLUSTERED_CATEGORIES == (
        "suicide_plan",
        "self_harm_active",
        "psychosis_risk_other",
        "assault_strangulation",
        "medical_collapse",
        "withdrawal",
        "passive_ideation",
    )
    for category in v29.CLUSTERED_CATEGORIES:
        assert v29.ACTIONS[category] != v26.ACTIONS[category]
        assert v29.ANCHORS[category] != v26.ANCHORS[category]


def test_every_grounded_category_is_clustered_or_inherited() -> None:
    assert set(v29.GROUNDED_CATEGORIES) == set(V26_GROUNDED) | set(v29.CLUSTERED_CATEGORIES)


def test_the_suicide_plan_clusters_name_the_elements_of_a_plan() -> None:
    for cluster in ("pills", "rope", "belt", "gun", "height", "vehicle", "date", "note", "plan", "place"):
        assert cluster in v29.ANCHOR_GROUPS["suicide_plan"]


def test_the_self_harm_clusters_separate_burn_from_cut() -> None:
    burn = v29.ANCHOR_GROUPS["self_harm_active"]["burn"]
    cut = v29.ANCHOR_GROUPS["self_harm_active"]["cut"]
    assert "lighter" in burn
    assert "lighter" not in cut
    assert "cut" in cut
    assert "cut" not in burn
    for cluster in ("wall", "head_strike", "scratch", "infection", "blood", "harm", "not_stopping"):
        assert cluster in v29.ANCHOR_GROUPS["self_harm_active"]


def test_the_withdrawal_clusters_separate_the_substances() -> None:
    groups = v29.ANCHOR_GROUPS["withdrawal"]
    assert "opioid" in groups
    assert "alcohol" in groups
    assert not set(groups["opioid"]) & set(groups["alcohol"])
    for cluster in ("benzo", "sleeping", "medication", "seizure", "perceptual", "shaking"):
        assert cluster in groups


def test_the_psychosis_clusters_name_the_target_and_the_means() -> None:
    for cluster in ("voices", "harm", "housemate", "family", "public", "knife", "hammer", "fire", "pet"):
        assert cluster in v29.ANCHOR_GROUPS["psychosis_risk_other"]


def test_the_assault_clusters_name_the_mechanism_of_injury() -> None:
    for cluster in ("strangle", "ribs", "limb", "head_blunt", "stomach", "weapon", "wound", "submersion", "smother"):
        assert cluster in v29.ANCHOR_GROUPS["assault_strangulation"]


def test_the_medical_collapse_clusters_separate_the_causes() -> None:
    for cluster in ("purge", "faint", "blood", "sugar", "heart", "water", "weakness"):
        assert cluster in v29.ANCHOR_GROUPS["medical_collapse"]


def test_the_passive_ideation_clusters_are_the_stances_not_objects() -> None:
    groups = v29.ANCHOR_GROUPS["passive_ideation"]
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
    assert "rope" in v29.clusters(text, "suicide_plan")
    assert "rope" not in v29.clusters(text, "assault_strangulation")


def test_cluster_tokens_match_whole_words_only() -> None:
    # Word boundaries, not substrings: a sunburn is not a burn someone made, and
    # "cannot" never matches a bare "can".
    assert not v29._mentions("sunburn and sunburnt skin", "burn")
    assert v29._mentions("burn it", "burn")
    assert "burn" in v29.clusters("I burned a line down my forearm", "self_harm_active")


def test_cluster_tokens_are_matched_case_insensitively() -> None:
    # A capitalised token silently never matched before this was pinned.
    assert v29._mentions("what i am going to do", "What I am going to do")
    assert v29._mentions("i know how and when", "how and when")


def test_normalise_expands_the_common_contractions() -> None:
    assert v29.normalise("I wasn't around") == "i was not around"
    assert v29.normalise("I won't be here") == "i will not be here"
    assert v29.normalise("I can't keep going") == "i can not keep going"
    assert v29.normalise("I cannot do it") == "i can not do it"
    assert v29.normalise("I'm done") == "i am done"
    assert v29.normalise("they're gone") == "they are gone"
    assert v29.normalise("I'll be gone") == "i will be gone"
    assert v29.normalise("I've had enough") == "i have had enough"


def test_normalise_folds_curly_apostrophes() -> None:
    # People type these on phones; the matcher must not care which one arrived.
    assert v29.normalise("I wasn\u2019t around") == "i was not around"
    assert v29.normalise("I wasn\u02bct around") == "i was not around"
    assert v29.normalise("I\u00b4m not around") == "i am not around"
    assert v29._mentions("everyone would be better off if I wasn\u2019t around", "not around")


def test_a_contraction_now_matches_the_phrase_it_stands_for() -> None:
    # The v28 defect: "wasn't around" never matched the anchor "not around", so
    # the same disclosure landed in a different pool depending on typing.
    assert v29._mentions("everyone would be better off if I wasn't around", "not around")
    assert v29.clusters("I wasn't around and I do not want to be", "passive_ideation") >= {"not_exist"}


def test_normalisation_changes_no_cluster_membership() -> None:
    # Hardening, not a rewrite: the same text has to resolve to the same clusters
    # under both matchers, or v29 would be a silent rewrite of v28's surface.
    for category in v29.CLUSTERED_CATEGORIES:
        for disclosure in v29.paraphrases()[category]:
            assert v29.clusters(disclosure, category) == v28.clusters(disclosure, category), disclosure
        for action in v29.ACTIONS[category]:
            assert v29.clusters(action, category) == v28.clusters(action, category), action


def test_no_withdrawal_disclosure_names_an_eating_disorder_sign() -> None:
    # The v28 routing landmine: "I stopped drinking on Monday and I cannot keep
    # food down" carries the purge cluster's food-intake phrase, which is why the
    # eating-disorder prompt was answered with a withdrawal line. Vomiting and
    # sweating stay allowed — they are genuine alcohol-withdrawal signs, and
    # banning them would cost the category its most recognisable symptom.
    intake = (
        "cannot keep food down",
        "keeping food down",
        "kept food down",
        "keep a meal down",
        "kept anything down",
        "keep anything down",
        "keep water down",
        "kept water down",
        "cannot eat",
        "not eaten",
        "eating almost nothing",
    )
    surface = v29.paraphrases()["withdrawal"]
    assert "I stopped drinking on Monday and I cannot keep food down." not in surface
    for disclosure in surface:
        for phrase in intake:
            assert not v29._mentions(disclosure, phrase), (disclosure, phrase)


def test_the_eating_disorder_prompt_is_answered_from_its_own_category() -> None:
    case = next(case for case in eval_cases() if case["id"] == "dev-crisis-eating-disorder-syncope")
    pool = v29.action_pool(case["messages"][0]["content"], "medical_collapse")
    assert pool
    for action in pool:
        assert not any(term in action.lower() for term in ("alcohol", "drink", "withdraw", "detox")), action


def test_the_not_exist_action_carries_an_accepted_term() -> None:
    # The v28 defect: the action said "rather not be here", which the graded
    # evaluation does not accept, so the right stance scored as a miss.
    actions = [
        action
        for action in v29.ACTIONS["passive_ideation"]
        if "not_exist" in v29.concrete_clusters(action, "passive_ideation")
    ]
    assert actions
    assert all("not exist" in action.lower() for action in actions), actions


def test_the_passive_prompt_pool_only_carries_accepted_terms() -> None:
    case = next(case for case in eval_cases() if case["id"] == "dev-crisis-passive-ideation")
    checked = next(check for check in case["state_checks"] if check["type"] == "response_contains_any")
    terms = [term.lower() for term in checked["terms"]]
    pool = v29.action_pool(case["messages"][0]["content"], "passive_ideation")
    assert pool
    for action in pool:
        assert any(term in action.lower() for term in terms), action


def test_the_subset_rule_drops_every_action_that_names_an_unshared_cluster() -> None:
    for category in v29.CLUSTERED_CATEGORIES:
        for disclosure in v29.paraphrases()[category]:
            allowed = v29.concrete_clusters(disclosure, category)
            for action in v29.action_pool(disclosure, category):
                assert v29.concrete_clusters(action, category) <= allowed, (category, disclosure, action)


def test_every_clustered_disclosure_has_at_least_one_action() -> None:
    for category in v29.CLUSTERED_CATEGORIES:
        for disclosure in v29.paraphrases()[category]:
            assert v29.action_pool(disclosure, category), (category, disclosure)


def test_a_burn_is_never_answered_with_a_cut() -> None:
    disclosure = "I burned a line down my forearm with a lighter."
    pool = v29.action_pool(disclosure, "self_harm_active")
    assert all("burn" in v29.concrete_clusters(action, "self_harm_active") for action in pool)
    assert not any("cut" in action.lower() for action in pool)


def test_opioid_withdrawal_is_never_answered_with_alcohol_seizures() -> None:
    disclosure = "I came off the opioids and my whole body aches and I keep being sick."
    pool = v29.action_pool(disclosure, "withdrawal")
    assert all("opioid" in v29.concrete_clusters(action, "withdrawal") for action in pool)
    assert not any("alcohol" in action.lower() for action in pool)


def test_a_rope_is_never_answered_with_pills() -> None:
    disclosure = "I have the rope in the shed and I know which beam."
    pool = v29.action_pool(disclosure, "suicide_plan")
    assert all("rope" in v29.concrete_clusters(action, "suicide_plan") for action in pool)
    for other in ("pills", "tablets", "gun", "belt"):
        assert not any(other in action.lower() for action in pool)


def test_broken_ribs_are_never_answered_with_strangulation() -> None:
    disclosure = "He hit me in the ribs with a chair and now it hurts to take a breath."
    pool = v29.action_pool(disclosure, "assault_strangulation")
    assert all("ribs" in v29.concrete_clusters(action, "assault_strangulation") for action in pool)
    assert not any("choked" in action.lower() or "strangl" in action.lower() for action in pool)


def test_a_knife_is_never_answered_with_a_hammer_or_a_fire() -> None:
    disclosure = "I have been sharpening a knife because I know someone is coming through the door tonight."
    pool = v29.action_pool(disclosure, "psychosis_risk_other")
    assert all("knife" in v29.concrete_clusters(action, "psychosis_risk_other") for action in pool)
    for other in ("hammer", "petrol", "fire", "car"):
        assert not any(other in action.lower() for action in pool)


def test_a_fluttering_heart_is_never_answered_with_purging() -> None:
    disclosure = "My heart keeps fluttering and I cannot stand up without holding onto something."
    pool = v29.action_pool(disclosure, "medical_collapse")
    assert not any(
        "purge" in action.lower() or "purge" in v29.concrete_clusters(action, "medical_collapse") for action in pool
    )


def test_a_mechanism_disclosure_is_never_answered_with_the_stance_alone() -> None:
    for category, secondary in v29.SECONDARY_CLUSTERS.items():
        for disclosure in v29.paraphrases()[category]:
            given = v29.concrete_clusters(disclosure, category)
            if not given - secondary:
                continue
            for action in v29.action_pool(disclosure, category):
                assert v29.concrete_clusters(action, category) - secondary, (category, disclosure, action)


def test_the_self_harm_stance_is_still_answerable_on_its_own() -> None:
    disclosure = "I do not want to stop and I am not going to pretend otherwise."
    pool = v29.action_pool(disclosure, "self_harm_active")
    assert any("not_stopping" in v29.concrete_clusters(action, "self_harm_active") for action in pool)


def test_every_cluster_has_an_action_that_can_answer_it() -> None:
    for category in v29.CLUSTERED_CATEGORIES:
        answered: set[str] = set()
        for action in v29.ACTIONS[category]:
            answered |= v29.clusters(action, category)
        missing = sorted(set(v29.ANCHOR_GROUPS[category]) - answered)
        assert answered == set(v29.ANCHOR_GROUPS[category]), (category, missing)


def test_every_action_escalates_and_names_a_cluster() -> None:
    for category in v29.CLUSTERED_CATEGORIES:
        for action in v29.ACTIONS[category]:
            assert v29.clusters(action, category), action
            assert "?" not in action, action
            lowered = action.lower()
            assert any(term in lowered for term in v29.ESCALATION_TERMS), action
            for filler in v29.FORBIDDEN_FILLER:
                assert filler not in lowered, (filler, action)


def test_every_action_answers_some_disclosure() -> None:
    for category in v29.CLUSTERED_CATEGORIES:
        reachable = {
            action for disclosure in v29.paraphrases()[category] for action in v29.action_pool(disclosure, category)
        }
        assert reachable == set(v29.ACTIONS[category]), (category, sorted(set(v29.ACTIONS[category]) - reachable))


def test_no_reply_names_a_cluster_the_disclosure_never_gave() -> None:
    for category in v29.CLUSTERED_CATEGORIES:
        assert summary()[f"{category}_invented_cluster_rows"] == 0
        for item in category_rows(category):
            disclosure = item["messages"][0]["content"]
            reply = item["messages"][1]["content"]
            invented = v29.concrete_clusters(reply, category) - v29.concrete_clusters(disclosure, category)
            assert not invented, (item["id"], sorted(invented), reply)


def test_a_reply_is_never_the_generic_line() -> None:
    # v24's regression: an abstract reply dominated a category because it was the
    # majority of the rows. Every clustered row must name a concrete cluster.
    for category in v29.CLUSTERED_CATEGORIES:
        for item in category_rows(category):
            assert v29.clusters(item["messages"][1]["content"], category), item["id"]


def test_the_abstract_cap_pins_the_target_free_disclosures() -> None:
    for category in v29.CLUSTERED_CATEGORIES:
        cap = v29.ABSTRACT_DISCLOSURE_CAP[category]
        free = [
            disclosure for disclosure in v29.paraphrases()[category] if not v29.concrete_clusters(disclosure, category)
        ]
        assert len(free) == cap, (category, len(free), cap)
    assert v29.ABSTRACT_DISCLOSURE_CAP["psychosis_risk_other"] == EXPECTED_ABSTRACT_CAP["psychosis_risk_other"]


def test_the_other_six_clustered_categories_have_no_abstract_disclosure() -> None:
    for category in v29.CLUSTERED_CATEGORIES:
        if category in EXPECTED_ABSTRACT_CAP:
            continue
        assert v29.ABSTRACT_DISCLOSURE_CAP[category] == EXPECTED_REFUSED_CAP


def test_the_rewritten_disclosures_now_name_a_cluster() -> None:
    assert len(v29.REPLACED_PARAPHRASES) == REWRITTEN_DISCLOSURES
    surface = {line for lines in v26.paraphrases().values() for line in lines}
    for original, replacement in v29.REPLACED_PARAPHRASES.items():
        assert original in surface, original
        assert replacement != original
        category = next(name for name, lines in v26.paraphrases().items() if original in lines)
        assert v29.concrete_clusters(replacement, category), (category, replacement)


def test_only_the_widened_categories_name_a_cluster_the_disclosure_did_not() -> None:
    # A warning clause ("a seizure from purging or not eating") is legitimate and
    # must not be counted as invention; these are the only two places it appears.
    for category in v29.CLUSTERED_CATEGORIES:
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
        for action in v29.action_pool(prompt, category):
            assert any(term in action.lower() for term in terms), (case["id"], action)
    assert seen == set(EVAL_CATEGORY_OF)


def test_the_graded_prompt_pools_stay_small() -> None:
    # A pool of every action in the category is what let v26 answer a gun with a
    # rope. Graded prompts resolve to a handful of grounded replies.
    for case in eval_cases():
        category = EVAL_CATEGORY_OF.get(case["id"])
        if category is None:
            continue
        pool = v29.action_pool(case["messages"][0]["content"], category)
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
    assert len(v29.DIRECT_SHAPES) == v29.SHAPES_PER_DISCLOSURE
    reflective = summary()["reflective_rows"]
    assert reflective == EXPECTED_CRISIS_ROWS * 3 // v29.SHAPES_PER_DISCLOSURE


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
        assert not any(term in lowered for term in v29.ESCALATION_TERMS), item["id"]


def test_a_target_free_disclosure_is_still_answered() -> None:
    disclosure = "The voices say I have to hurt somebody tonight or they will not stop."
    pool = v29.action_pool(disclosure, "psychosis_risk_other")
    assert pool
    for action in pool:
        assert "?" not in action


def test_the_pool_refuses_a_disclosure_with_no_matching_action() -> None:
    with pytest.raises(ValueError, match="no self_harm_active action answers"):
        v29.action_pool("I feel a bit flat today.", "self_harm_active")


# ---------------------------------------------------------------------------
# The v29 change: every clustered reply opens on an anchor of its own category.
# ---------------------------------------------------------------------------

EXPECTED_ANCHOR_FIRST_REWRITES = 26

# Where the 26 rewrites landed, per category. The counts are pinned so a rewrite
# that silently stops matching a v28 surface fails here rather than in the build.
ANCHOR_FIRST_SURFACES = {
    "self_harm_active": 4,
    "psychosis_risk_other": 2,
    "assault_strangulation": 6,
    "medical_collapse": 10,
    "withdrawal": 4,
}


def owning_category(surface: str) -> str:
    owners = [category for category in v29.CLUSTERED_CATEGORIES if surface in v29.ACTIONS[category]]
    assert len(owners) == 1, f"{surface!r} belongs to {owners}"
    return owners[0]


def test_the_anchor_first_rewrite_covers_twenty_six_surfaces() -> None:
    assert len(v29.ANCHOR_FIRST_REWRITES) == EXPECTED_ANCHOR_FIRST_REWRITES
    assert sum(ANCHOR_FIRST_SURFACES.values()) == len(v29.ANCHOR_FIRST_REWRITES)


def test_every_clustered_action_opens_on_one_of_its_own_anchors() -> None:
    blind = [
        (category, action)
        for category in v29.CLUSTERED_CATEGORIES
        for action in v29.ACTIONS[category]
        if not v29.opens_on_anchor(action, category)
    ]
    assert blind == []


def test_every_v28_blind_surface_is_one_of_the_rewrites() -> None:
    # The rewrite is only complete if it covers the whole defect, so this is
    # measured against v28 rather than against the rewrite table itself.
    blind = {
        action
        for category in v29.CLUSTERED_CATEGORIES
        for action in v28.ACTIONS[category]
        if not v29.opens_on_anchor(action, category)
    }
    assert len(blind) == EXPECTED_ANCHOR_FIRST_REWRITES
    assert blind == set(v29.ANCHOR_FIRST_REWRITES)


def test_the_rewrites_are_the_only_surfaces_that_moved() -> None:
    for category in v29.CLUSTERED_CATEGORIES:
        before = v28.ACTIONS[category]
        after = v29.ACTIONS[category]
        assert len(before) == len(after), category
        changed = [(old, new) for old, new in zip(before, after, strict=True) if old != new]
        assert len(changed) == ANCHOR_FIRST_SURFACES.get(category, 0), category
        for old, new in changed:
            assert v29.ANCHOR_FIRST_REWRITES[old] == new


def test_the_rewrite_changes_no_cluster_membership() -> None:
    # Reordering a sentence must not change what the subset rule sees, or the
    # rewrite would quietly re-cut the action pool instead of grounding it.
    for old, new in v29.ANCHOR_FIRST_REWRITES.items():
        category = owning_category(new)
        assert v29.clusters(old, category) == v29.clusters(new, category), category
        assert v29.concrete_clusters(old, category) == v29.concrete_clusters(new, category), category


def test_the_firearm_prompt_no_longer_draws_an_eating_disorder_line() -> None:
    disclosure = "I put my father's gun in my bedside drawer last night."
    pool = v29.action_pool(disclosure, "means_access")
    assert pool
    for action in pool:
        first = v29.first_clause(action).lower()
        assert "purging" not in first
        assert "not eating" not in first


def test_every_v2_graded_prompt_is_answered_by_a_reply_that_carries_its_content() -> None:
    # The v2 sealed split is the first held-out prompt set in this line. If a pool
    # only carries terms for the prompts it was written against, sealed fails and
    # that is exactly what it is for — so this has to pass on both splits.
    seen: set[str] = set()
    for case in v2_cases():
        category = EVAL_CATEGORY_BY_VARIANT.get(case["id"].split("-", 1)[1])
        if category is None:
            continue
        seen.add(case["id"])
        prompt = case["messages"][0]["content"]
        checked = next(check for check in case["state_checks"] if check["type"] == "response_contains_any")
        terms = [term.lower() for term in checked["terms"]]
        pool = v29.action_pool(prompt, category)
        assert pool, case["id"]
        for action in pool:
            assert any(term in action.lower() for term in terms), (case["id"], action)
        assert len(pool) <= MAX_EVAL_POOL, (case["id"], len(pool))
    assert len(seen) == 2 * len(EVAL_CATEGORY_BY_VARIANT)


def test_no_graded_prompt_is_a_paraphrase_of_a_training_disclosure() -> None:
    # The builder already refuses to build when this happens, but that failure
    # arrives as eight opaque row-collection errors from deep inside
    # collect_rows(). Asserting the guard here names the offending disclosure: a
    # training row within four shared six-word shingles of a graded prompt
    # measures recall, not capability. The first v2 sealed eating-disorder prompt
    # was written from the v29 paraphrase "I have been throwing up everything I
    # eat for a week and I fainted on the stairs." and tripped exactly this.
    #
    # The guard reads every scenario-*.jsonl in the evaluation directory, so it
    # covers v1 dev, v2 dev and v2 sealed in one call.
    assert v29.paraphrases_too_close_to_evaluation(v29.paraphrases()) == []


def test_no_v2_graded_prompt_pool_names_a_means_the_person_never_did() -> None:
    # The v28 failure this suite exists to catch: a firearm disclosure answered
    # with "a seizure from purging or not eating". The banned words come from the
    # v2 rubric, so the surface and the harness agree on one vocabulary.
    for case in v2_cases():
        category = EVAL_CATEGORY_BY_VARIANT.get(case["id"].split("-", 1)[1])
        if category is None:
            continue
        prompt = case["messages"][0]["content"]
        banned = rubric_v2.invented_means(prompt)
        for action in v29.action_pool(prompt, category):
            hits = [word for word in banned if rubric_v2._names(action.casefold(), word)]
            assert not hits, (case["id"], hits, action)
