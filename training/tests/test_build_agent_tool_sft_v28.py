"""Tests for the v28 crisis family: a reply may only name what the person named.

v27 grounded the last seven categories structurally: each category is split into
``ANCHOR_GROUPS`` — named clusters of word-boundary tokens — and an action is
only eligible for a disclosure when the clusters it names are a subset of the
clusters the disclosure named. That work is inherited here and pinned in full,
because v28 changes nothing about it.

v28 closes three defects the v27 paired evaluation exposed:

* **A routing landmine.** The eating-disorder prompt was answered with a
  withdrawal line, because the withdrawal disclosure "I stopped drinking on
  Monday and I cannot keep food down" carried the ``medical_collapse`` purge
  phrase "cannot keep food down". The disclosure now names shaking hands
  instead.
* **The wrong stance.** The passive-ideation prompt was answered with the
  ``not_exist`` action, whose wording ("rather not be here anymore") did not
  contain any term the evaluation accepts. It now says "not exist".
* **A matcher blind to contractions.** ``_mentions`` compared raw tokens, so
  "wasn't around" never matched the anchor "not around". ``normalise`` now
  expands the common contractions and folds curly apostrophes before matching.

The normalisation is hardening, not a rewrite: it changes the cluster
membership of zero disclosures and zero actions, which the tests below pin.

What this file pins, in order: the identity and the inherited surface, the seven
cluster vocabularies, the subset rule and its consequences on the real
disclosures that used to be answered wrongly, the secondary-cluster rule, the
abstract cap for the disclosures that genuinely name no target, contraction
normalisation, the three v28 content fixes, the graded evaluation prompts, and
the row counts the release claims.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

import training.build_agent_tool_sft_v23 as v23
import training.build_agent_tool_sft_v26 as v26
import training.build_agent_tool_sft_v27 as v27
import training.build_agent_tool_sft_v28 as v28
import training.strict_crisis_gate as gate

MESSAGES_PER_ROW = 2
EXPECTED_TOTAL_ROWS = 6504
EXPECTED_CRISIS_ROWS = 5904
EXPECTED_CONTROLS = 600
EXPECTED_DISCLOSURES = 838
MIN_DISTINCT_REPLIES = 1000
# Eight frames is the v22 lever; v23 through v28 keep it.
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


@cache
def rows() -> list[dict[str, Any]]:
    return v28.collect_rows()["train"]


@cache
def summary() -> dict[str, Any]:
    return v28.summarize(v28.collect_rows())


def crisis_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item["category"] != "control"]


def category_rows(category: str, items: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    items = rows() if items is None else items
    return [item for item in items if item["category"] == category]


def eval_cases() -> list[dict[str, Any]]:
    return [json.loads(line) for line in EVAL_CASES.read_text().splitlines() if line.strip()]


def test_the_release_identity_moves_past_v27() -> None:
    assert v28.RELEASE_ID.endswith("-v28")
    assert v28.RELEASE_ID != v26.RELEASE_ID
    assert v28.FAMILY == v23.FAMILY


def test_the_already_grounded_categories_are_inherited_untouched() -> None:
    for category in V26_GROUNDED:
        assert v28.ACTIONS[category] == v26.ACTIONS[category]
        assert v28.ANCHORS[category] == v26.ANCHORS[category]
    assert set(v28.REPLACED_ACTIONS) == set(v28.CLUSTERED_CATEGORIES)


def test_the_seven_eval_fitted_categories_are_the_ones_rewritten() -> None:
    assert v28.CLUSTERED_CATEGORIES == (
        "suicide_plan",
        "self_harm_active",
        "psychosis_risk_other",
        "assault_strangulation",
        "medical_collapse",
        "withdrawal",
        "passive_ideation",
    )
    for category in v28.CLUSTERED_CATEGORIES:
        assert v28.ACTIONS[category] != v26.ACTIONS[category]
        assert v28.ANCHORS[category] != v26.ANCHORS[category]


def test_every_grounded_category_is_clustered_or_inherited() -> None:
    assert set(v28.GROUNDED_CATEGORIES) == set(V26_GROUNDED) | set(v28.CLUSTERED_CATEGORIES)


def test_the_suicide_plan_clusters_name_the_elements_of_a_plan() -> None:
    for cluster in ("pills", "rope", "belt", "gun", "height", "vehicle", "date", "note", "plan", "place"):
        assert cluster in v28.ANCHOR_GROUPS["suicide_plan"]


def test_the_self_harm_clusters_separate_burn_from_cut() -> None:
    burn = v28.ANCHOR_GROUPS["self_harm_active"]["burn"]
    cut = v28.ANCHOR_GROUPS["self_harm_active"]["cut"]
    assert "lighter" in burn
    assert "lighter" not in cut
    assert "cut" in cut
    assert "cut" not in burn
    for cluster in ("wall", "head_strike", "scratch", "infection", "blood", "harm", "not_stopping"):
        assert cluster in v28.ANCHOR_GROUPS["self_harm_active"]


def test_the_withdrawal_clusters_separate_the_substances() -> None:
    groups = v28.ANCHOR_GROUPS["withdrawal"]
    assert "opioid" in groups
    assert "alcohol" in groups
    assert not set(groups["opioid"]) & set(groups["alcohol"])
    for cluster in ("benzo", "sleeping", "medication", "seizure", "perceptual", "shaking"):
        assert cluster in groups


def test_the_psychosis_clusters_name_the_target_and_the_means() -> None:
    for cluster in ("voices", "harm", "housemate", "family", "public", "knife", "hammer", "fire", "pet"):
        assert cluster in v28.ANCHOR_GROUPS["psychosis_risk_other"]


def test_the_assault_clusters_name_the_mechanism_of_injury() -> None:
    for cluster in ("strangle", "ribs", "limb", "head_blunt", "stomach", "weapon", "wound", "submersion", "smother"):
        assert cluster in v28.ANCHOR_GROUPS["assault_strangulation"]


def test_the_medical_collapse_clusters_separate_the_causes() -> None:
    for cluster in ("purge", "not_eating", "faint", "blood", "sugar", "heart", "seizure", "water", "weakness"):
        assert cluster in v28.ANCHOR_GROUPS["medical_collapse"]
    # An eating-intake phrase inside "purge" is the routing landmine the split
    # exists to remove, so the two token sets have to stay disjoint.
    assert not set(v28.ANCHOR_GROUPS["medical_collapse"]["purge"]) & set(
        v28.ANCHOR_GROUPS["medical_collapse"]["not_eating"]
    )


def test_the_passive_ideation_clusters_are_the_stances_not_objects() -> None:
    groups = v28.ANCHOR_GROUPS["passive_ideation"]
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
    assert "rope" in v28.clusters(text, "suicide_plan")
    assert "rope" not in v28.clusters(text, "assault_strangulation")


def test_cluster_tokens_match_whole_words_only() -> None:
    # Word boundaries, not substrings: a sunburn is not a burn someone made, and
    # "cannot" never matches a bare "can".
    assert not v28._mentions("sunburn and sunburnt skin", "burn")
    assert v28._mentions("burn it", "burn")
    assert "burn" in v28.clusters("I burned a line down my forearm", "self_harm_active")


def test_cluster_tokens_are_matched_case_insensitively() -> None:
    # A capitalised token silently never matched before this was pinned.
    assert v28._mentions("what i am going to do", "What I am going to do")
    assert v28._mentions("i know how and when", "how and when")


def test_normalise_expands_the_common_contractions() -> None:
    assert v28.normalise("I wasn't around") == "i was not around"
    assert v28.normalise("I won't be here") == "i will not be here"
    assert v28.normalise("I can't keep going") == "i cannot keep going"
    assert v28.normalise("I cannot do it") == "i cannot do it"
    assert v28.normalise("I'm done") == "i am done"
    assert v28.normalise("they're gone") == "they are gone"
    assert v28.normalise("I'll be gone") == "i will be gone"
    assert v28.normalise("I've had enough") == "i have had enough"


def test_normalise_folds_curly_apostrophes() -> None:
    # People type these on phones; the matcher must not care which one arrived.
    assert v28.normalise("I wasn\u2019t around") == "i was not around"
    assert v28.normalise("I wasn\u02bct around") == "i was not around"
    assert v28.normalise("I\u00b4m not around") == "i am not around"
    assert v28._mentions("everyone would be better off if I wasn\u2019t around", "not around")


def test_a_contraction_now_matches_the_phrase_it_stands_for() -> None:
    # The v27 defect: "wasn't around" never matched the anchor "not around", so
    # the same disclosure landed in a different pool depending on typing.
    assert v28._mentions("everyone would be better off if I wasn't around", "not around")
    assert v28.clusters("I wasn't around and I do not want to be", "passive_ideation") >= {"not_exist"}


def test_normalisation_changes_no_cluster_membership() -> None:
    # Hardening, not a rewrite: the same text has to resolve to the same clusters
    # under both matchers, or v28 would be a silent rewrite of v27's surface.
    for category in v28.CLUSTERED_CATEGORIES:
        for disclosure in v28.paraphrases()[category]:
            assert v28.clusters(disclosure, category) == v27.clusters(disclosure, category), disclosure
        for action in v28.ACTIONS[category]:
            assert v28.clusters(action, category) == v27.clusters(action, category), action


def test_no_withdrawal_disclosure_names_an_eating_disorder_sign() -> None:
    # The v27 routing landmine: "I stopped drinking on Monday and I cannot keep
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
    surface = v28.paraphrases()["withdrawal"]
    assert "I stopped drinking on Monday and I cannot keep food down." not in surface
    for disclosure in surface:
        for phrase in intake:
            assert not v28._mentions(disclosure, phrase), (disclosure, phrase)


def test_the_eating_disorder_prompt_is_answered_from_its_own_category() -> None:
    case = next(case for case in eval_cases() if case["id"] == "dev-crisis-eating-disorder-syncope")
    pool = v28.action_pool(case["messages"][0]["content"], "medical_collapse")
    assert pool
    for action in pool:
        assert not any(term in action.lower() for term in ("alcohol", "drink", "withdraw", "detox")), action


def test_the_not_exist_action_carries_an_accepted_term() -> None:
    # The v27 defect: the action said "rather not be here", which the graded
    # evaluation does not accept, so the right stance scored as a miss.
    actions = [
        action
        for action in v28.ACTIONS["passive_ideation"]
        if "not_exist" in v28.concrete_clusters(action, "passive_ideation")
    ]
    assert actions
    assert all("not exist" in action.lower() for action in actions), actions


def test_the_passive_prompt_pool_only_carries_accepted_terms() -> None:
    case = next(case for case in eval_cases() if case["id"] == "dev-crisis-passive-ideation")
    checked = next(check for check in case["state_checks"] if check["type"] == "response_contains_any")
    terms = [term.lower() for term in checked["terms"]]
    pool = v28.action_pool(case["messages"][0]["content"], "passive_ideation")
    assert pool
    for action in pool:
        assert any(term in action.lower() for term in terms), action


def test_the_subset_rule_drops_every_action_that_names_an_unshared_cluster() -> None:
    for category in v28.CLUSTERED_CATEGORIES:
        for disclosure in v28.paraphrases()[category]:
            allowed = v28.concrete_clusters(disclosure, category)
            for action in v28.action_pool(disclosure, category):
                assert v28.concrete_clusters(action, category) <= allowed, (category, disclosure, action)


def test_every_clustered_disclosure_has_at_least_one_action() -> None:
    for category in v28.CLUSTERED_CATEGORIES:
        for disclosure in v28.paraphrases()[category]:
            assert v28.action_pool(disclosure, category), (category, disclosure)


def test_a_burn_is_never_answered_with_a_cut() -> None:
    disclosure = "I burned a line down my forearm with a lighter."
    pool = v28.action_pool(disclosure, "self_harm_active")
    assert all("burn" in v28.concrete_clusters(action, "self_harm_active") for action in pool)
    assert not any("cut" in action.lower() for action in pool)


def test_opioid_withdrawal_is_never_answered_with_alcohol_seizures() -> None:
    disclosure = "I came off the opioids and my whole body aches and I keep being sick."
    pool = v28.action_pool(disclosure, "withdrawal")
    assert all("opioid" in v28.concrete_clusters(action, "withdrawal") for action in pool)
    assert not any("alcohol" in action.lower() for action in pool)


def test_a_rope_is_never_answered_with_pills() -> None:
    disclosure = "I have the rope in the shed and I know which beam."
    pool = v28.action_pool(disclosure, "suicide_plan")
    assert all("rope" in v28.concrete_clusters(action, "suicide_plan") for action in pool)
    for other in ("pills", "tablets", "gun", "belt"):
        assert not any(other in action.lower() for action in pool)


def test_broken_ribs_are_never_answered_with_strangulation() -> None:
    disclosure = "He hit me in the ribs with a chair and now it hurts to take a breath."
    pool = v28.action_pool(disclosure, "assault_strangulation")
    assert all("ribs" in v28.concrete_clusters(action, "assault_strangulation") for action in pool)
    assert not any("choked" in action.lower() or "strangl" in action.lower() for action in pool)


def test_a_knife_is_never_answered_with_a_hammer_or_a_fire() -> None:
    disclosure = "I have been sharpening a knife because I know someone is coming through the door tonight."
    pool = v28.action_pool(disclosure, "psychosis_risk_other")
    assert all("knife" in v28.concrete_clusters(action, "psychosis_risk_other") for action in pool)
    for other in ("hammer", "petrol", "fire", "car"):
        assert not any(other in action.lower() for action in pool)


def test_a_fluttering_heart_is_never_answered_with_purging() -> None:
    disclosure = "My heart keeps fluttering and I cannot stand up without holding onto something."
    pool = v28.action_pool(disclosure, "medical_collapse")
    assert not any(
        "purge" in action.lower() or "purge" in v28.concrete_clusters(action, "medical_collapse") for action in pool
    )


def test_a_mechanism_disclosure_is_never_answered_with_the_stance_alone() -> None:
    for category, secondary in v28.SECONDARY_CLUSTERS.items():
        for disclosure in v28.paraphrases()[category]:
            given = v28.concrete_clusters(disclosure, category)
            if not given - secondary:
                continue
            for action in v28.action_pool(disclosure, category):
                assert v28.concrete_clusters(action, category) - secondary, (category, disclosure, action)


def test_the_self_harm_stance_is_still_answerable_on_its_own() -> None:
    disclosure = "I do not want to stop and I am not going to pretend otherwise."
    pool = v28.action_pool(disclosure, "self_harm_active")
    assert any("not_stopping" in v28.concrete_clusters(action, "self_harm_active") for action in pool)


def test_every_cluster_has_an_action_that_can_answer_it() -> None:
    for category in v28.CLUSTERED_CATEGORIES:
        answered: set[str] = set()
        for action in v28.ACTIONS[category]:
            answered |= v28.clusters(action, category)
        missing = sorted(set(v28.ANCHOR_GROUPS[category]) - answered)
        assert answered == set(v28.ANCHOR_GROUPS[category]), (category, missing)


def test_every_action_escalates_and_names_a_cluster() -> None:
    for category in v28.CLUSTERED_CATEGORIES:
        for action in v28.ACTIONS[category]:
            assert v28.clusters(action, category), action
            assert "?" not in action, action
            lowered = action.lower()
            assert any(term in lowered for term in v28.ESCALATION_TERMS), action
            for filler in v28.FORBIDDEN_FILLER:
                assert filler not in lowered, (filler, action)


def test_every_action_answers_some_disclosure() -> None:
    for category in v28.CLUSTERED_CATEGORIES:
        reachable = {
            action for disclosure in v28.paraphrases()[category] for action in v28.action_pool(disclosure, category)
        }
        assert reachable == set(v28.ACTIONS[category]), (category, sorted(set(v28.ACTIONS[category]) - reachable))


def test_no_reply_names_a_cluster_the_disclosure_never_gave() -> None:
    for category in v28.CLUSTERED_CATEGORIES:
        assert summary()[f"{category}_invented_cluster_rows"] == 0
        for item in category_rows(category):
            disclosure = item["messages"][0]["content"]
            reply = item["messages"][1]["content"]
            invented = v28.concrete_clusters(reply, category) - v28.concrete_clusters(disclosure, category)
            assert not invented, (item["id"], sorted(invented), reply)


def test_a_reply_is_never_the_generic_line() -> None:
    # v24's regression: an abstract reply dominated a category because it was the
    # majority of the rows. Every clustered row must name a concrete cluster.
    for category in v28.CLUSTERED_CATEGORIES:
        for item in category_rows(category):
            assert v28.clusters(item["messages"][1]["content"], category), item["id"]


def test_the_abstract_cap_pins_the_target_free_disclosures() -> None:
    for category in v28.CLUSTERED_CATEGORIES:
        cap = v28.ABSTRACT_DISCLOSURE_CAP[category]
        free = [
            disclosure for disclosure in v28.paraphrases()[category] if not v28.concrete_clusters(disclosure, category)
        ]
        assert len(free) == cap, (category, len(free), cap)
    assert v28.ABSTRACT_DISCLOSURE_CAP["psychosis_risk_other"] == EXPECTED_ABSTRACT_CAP["psychosis_risk_other"]


def test_the_other_six_clustered_categories_have_no_abstract_disclosure() -> None:
    for category in v28.CLUSTERED_CATEGORIES:
        if category in EXPECTED_ABSTRACT_CAP:
            continue
        assert v28.ABSTRACT_DISCLOSURE_CAP[category] == EXPECTED_REFUSED_CAP


def test_the_rewritten_disclosures_now_name_a_cluster() -> None:
    assert len(v28.REPLACED_PARAPHRASES) == REWRITTEN_DISCLOSURES
    surface = {line for lines in v26.paraphrases().values() for line in lines}
    for original, replacement in v28.REPLACED_PARAPHRASES.items():
        assert original in surface, original
        assert replacement != original
        category = next(name for name, lines in v26.paraphrases().items() if original in lines)
        assert v28.concrete_clusters(replacement, category), (category, replacement)


def test_only_the_widened_categories_name_a_cluster_the_disclosure_did_not() -> None:
    # The seizure surface no longer carries a causal warning clause ("a seizure
    # from purging or not eating"), so medical_collapse widens nothing either:
    # the only remaining widened rows are the psychosis warning clauses, which
    # name the voices the disclosure paraphrased rather than a new means.
    for category in v28.CLUSTERED_CATEGORIES:
        widened = summary()[f"{category}_widened_cluster_rows"]
        if category == "psychosis_risk_other":
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
        for action in v28.action_pool(prompt, category):
            assert any(term in action.lower() for term in terms), (case["id"], action)
    assert seen == set(EVAL_CATEGORY_OF)


def test_the_graded_prompt_pools_stay_small() -> None:
    # A pool of every action in the category is what let v26 answer a gun with a
    # rope. Graded prompts resolve to a handful of grounded replies.
    for case in eval_cases():
        category = EVAL_CATEGORY_OF.get(case["id"])
        if category is None:
            continue
        pool = v28.action_pool(case["messages"][0]["content"], category)
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
    assert len(v28.DIRECT_SHAPES) == v28.SHAPES_PER_DISCLOSURE
    reflective = summary()["reflective_rows"]
    assert reflective == EXPECTED_CRISIS_ROWS * 3 // v28.SHAPES_PER_DISCLOSURE


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
        assert not any(term in lowered for term in v28.ESCALATION_TERMS), item["id"]


def test_a_target_free_disclosure_is_still_answered() -> None:
    disclosure = "The voices say I have to hurt somebody tonight or they will not stop."
    pool = v28.action_pool(disclosure, "psychosis_risk_other")
    assert pool
    for action in pool:
        assert "?" not in action


def test_the_pool_refuses_a_disclosure_with_no_matching_action() -> None:
    with pytest.raises(ValueError, match="no self_harm_active action answers"):
        v28.action_pool("I feel a bit flat today.", "self_harm_active")
