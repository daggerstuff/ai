"""Tests for the v27 release: v26's pipeline with cluster-grounded crisis rows.

v26 grounded three of the ten crisis categories on the object the disclosure
named. The other seven still rotated eight actions across every disclosure, so
4,192 of 4,528 disclosure x action pairs (92.6%) named an anchor the person never
gave — a cigarette-lighter burn answered with "get that cut closed", opioid
withdrawal answered with an alcohol/sedative seizure line, "I bought a rope
yesterday" answered with "the pills set aside for this". The anchors were the
paired cases' ``response_contains_any`` terms, so the surface had been fitted to
the eval.

The tests here pin the release side: the pipeline must be v26's unchanged except
for the addendum, the family must be the v27 builder's rows verbatim, no shipped
row may name a concrete cluster its disclosure did not, the abstract caps must
hold, and the manifest must record the all-ten-category coverage, the
concrete-only invention metric, the widened surfaces and the closed lists that
remain.
"""

from __future__ import annotations

from typing import Any

import training.build_agent_tool_sft_v23 as v23
import training.build_agent_tool_sft_v25 as v25
import training.build_agent_tool_sft_v26 as v26
import training.build_agent_tool_sft_v27 as v27
import training.build_v14_release as v14
import training.build_v15_release as v15
import training.build_v16_release as v16
import training.build_v17_release as v17
import training.build_v19_release as v19
import training.build_v20_release as v20
import training.build_v21_release as v21
import training.build_v22_release as v22
import training.build_v23_release as v23_release
import training.build_v24_release as v24_release
import training.build_v25_release as v25_release
import training.build_v26_release as prev
import training.build_v27_release as release

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V27_ROWS = 6504
MESSAGES_PER_ROW = 2
# Eight frames is the v22 lever; v23 through v27 all keep it and change the rows.
EXPECTED_SHAPES = 8
EXPECTED_REFLECTIVE_FRAMES = 3
EXPECTED_REWRITES = 6
EXPECTED_ABSTRACT_PSYCHOSIS = 37
# v26's surfaces for the three categories it already grounded.
INHERITED_SURFACES = {"means_access": 24, "overdose_taken": 17, "recent_attempt": 18}
# The seven categories v27 rewrites, with the per-cluster surface each one got.
CLUSTERED_SURFACES = {
    "suicide_plan": 13,
    "self_harm_active": 15,
    "psychosis_risk_other": 13,
    "assault_strangulation": 13,
    "medical_collapse": 11,
    "withdrawal": 8,
    "passive_ideation": 8,
}


def test_the_release_identity_names_v27() -> None:
    assert release.RELEASE_ID.endswith("-v27")
    assert release.RELEASE_ID in str(release.OUTPUT)
    assert release.RELEASE_ID in str(release.MANIFEST)
    assert release.SHUFFLE_SEED not in {
        prev.SHUFFLE_SEED,
        v25_release.SHUFFLE_SEED,
        v24_release.SHUFFLE_SEED,
        v23_release.SHUFFLE_SEED,
        v22.SHUFFLE_SEED,
        v21.SHUFFLE_SEED,
        v20.SHUFFLE_SEED,
        v19.SHUFFLE_SEED,
        v17.SHUFFLE_SEED,
        v16.SHUFFLE_SEED,
    }


def test_limitations_drop_the_stale_v26_family_claims() -> None:
    """v26 described its own family's size, ratio and placement; v27 replaces it."""
    inherited = prev._limitations()
    for stale in release._STALE_V26_LIMITATIONS:
        assert any(text.startswith(stale) for text in inherited), stale
        assert not any(text.startswith(stale) for text in release._limitations()), stale
    assert len(release._limitations()) == (
        len(inherited) - len(release._STALE_V26_LIMITATIONS) + len(release.LIMITATIONS_V27)
    )


def test_limitations_keep_every_other_v26_entry() -> None:
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(release._STALE_V26_LIMITATIONS))
    assert release._limitations()[: len(kept)] == kept


def test_limitations_record_the_size_the_ratio_and_the_kept_revert() -> None:
    added = " ".join(release.LIMITATIONS_V27)
    assert "6,504 rows" in added
    assert "7.9% of train" in added
    assert "838" in added
    assert "9.8 escalations per ordinary reply" in added
    assert "not reviewed by a clinician" in added
    assert "carries no corpus repair" in added


def test_limitations_say_grounding_now_covers_all_ten_categories() -> None:
    added = " ".join(release.LIMITATIONS_V27)
    assert "Grounding now covers all ten categories" in added
    assert "grounding by cluster, not by sentence" in added
    assert "24 for means_access, 17 for overdose_taken, 18 for recent_attempt" in added
    assert "13 for suicide_plan, 15 for self_harm_active" in added
    assert "11 for medical_collapse" in added
    assert "8 for withdrawal and 8 for passive_ideation" in added


def test_limitations_record_the_concrete_only_invention_metric() -> None:
    """A warning clause is legitimate; claiming a gun is not. The metric says so."""
    added = " ".join(release.LIMITATIONS_V27)
    assert "counts concrete clusters only" in added
    assert "51 medical_collapse rows" in added
    assert "279 psychosis rows" in added
    assert "rather than hidden" in added


def test_limitations_record_the_remaining_closed_lists_and_caps() -> None:
    added = " ".join(release.LIMITATIONS_V27)
    assert "closed lists" in added
    assert "lithium blister" in added
    assert "capped and the cap is enforced at build time" in added
    assert f"{EXPECTED_ABSTRACT_PSYCHOSIS} abstract psychosis disclosures" in added
    assert "a new cluster-free disclosure fails the build loudly" in added


def test_limitations_record_the_frame_invariance_as_the_mechanism() -> None:
    added = " ".join(release.LIMITATIONS_V27)
    assert "three reflective-lead frames" in added
    assert "escalation competes with that register from a trained position" in added


def test_the_pipeline_is_v26s_with_only_the_family_swapped() -> None:
    assert issubclass(release._Release, prev._Release)
    assert release._Release.carry is prev._Release.carry
    assert release._Release.append_addenda is not prev._Release.append_addenda
    assert release._Release.verify_totals is not prev._Release.verify_totals


def test_the_addendum_is_v27_and_the_older_families_are_gone() -> None:
    assert release.V27_SOURCE_LABEL == "crisis_direct_v27"
    assert release.V27_SOURCE_LABEL != prev.V26_SOURCE_LABEL
    assert release.V27_SOURCE_LABEL != v25_release.V25_SOURCE_LABEL
    assert release.V27_SOURCE_LABEL != v24_release.V24_SOURCE_LABEL
    assert release.V27_SOURCE_LABEL != v23_release.V23_SOURCE_LABEL
    assert release.EXPECTED_V27_TRAIN_ROWS == V27_ROWS
    assert release.EXPECTED_V27_TRAIN_ROWS == prev.EXPECTED_V26_TRAIN_ROWS


def test_v27_candidate_rows_are_clean_chat_rows() -> None:
    rows = release.v27_candidate_rows()
    assert len(rows) == V27_ROWS
    for candidate, payload in rows:
        assert set(candidate) == {"messages", "source"}
        assert candidate["source"] == release.V27_SOURCE_LABEL
        assert not v14.is_agent_tool(candidate)
        assert len(candidate["messages"]) == MESSAGES_PER_ROW
        assert candidate["messages"][0]["role"] == "user"
        assert candidate["messages"][1]["role"] == "assistant"
        assert payload, "a v27 row must project to an assistant target"


def test_the_family_is_the_v27_builder_rows_verbatim() -> None:
    """The release must ship what the builder validated, not a reshaped copy."""
    built = v27.collect_rows()["train"]
    candidates = release.v27_candidate_rows()
    assert [item["messages"] for item in built] == [candidate["messages"] for candidate, _ in candidates]


def test_the_family_is_built_by_the_v27_builder() -> None:
    assert release.v27 is v27
    assert prev.v26 is v26
    assert v27.SHAPES_PER_DISCLOSURE == EXPECTED_SHAPES
    assert len(v26.REFLECTIVE_SHAPES) == EXPECTED_REFLECTIVE_FRAMES
    assert len(v27.ACTIONS) == len(v26.ACTIONS)
    assert (*v26.GROUNDED_CATEGORIES, *v27.CLUSTERED_CATEGORIES) == v27.GROUNDED_CATEGORIES


def test_the_three_v26_categories_are_inherited_unchanged() -> None:
    """v27 must not disturb the surfaces v26's eval already proved out."""
    for category, expected in INHERITED_SURFACES.items():
        assert v27.ACTIONS[category] == v26.ACTIONS[category], category
        assert len(v27.ACTIONS[category]) == expected, category
        assert category not in v27.CLUSTERED_CATEGORIES


def test_the_seven_clustered_categories_carry_their_own_surfaces() -> None:
    assert tuple(CLUSTERED_SURFACES) == v27.CLUSTERED_CATEGORIES
    for category, expected in CLUSTERED_SURFACES.items():
        assert len(v27.ACTIONS[category]) == expected, category
        assert v27.ANCHOR_GROUPS[category], category
        # Every one was rewritten per cluster, and the seven grew from v26's flat
        # eight-action rotation except withdrawal and passive_ideation, which had
        # a cluster per action already and were rewritten in place.
        assert v27.ACTIONS[category] != v26.ACTIONS[category], category
        assert len(v27.ACTIONS[category]) >= len(v26.ACTIONS[category]), category


def test_every_grounded_row_is_answered_by_its_own_categorys_actions() -> None:
    """The v26 failure: an overdose disclosure answered by a recent_attempt line."""
    built = v27.collect_rows()["train"]
    for category in v27.GROUNDED_CATEGORIES:
        actions = v27.ACTIONS[category]
        items = [item for item in built if item["category"] == category]
        assert items, category
        for item in items:
            reply = item["messages"][1]["content"]
            assert any(action in reply for action in actions), (category, reply)


def test_no_shipped_row_invents_a_concrete_cluster() -> None:
    """The whole point of v27: no reply names a mechanism the person never gave."""
    built = v27.collect_rows()["train"]
    for category in v27.CLUSTERED_CATEGORIES:
        items = [item for item in built if item["category"] == category]
        assert items, category
        for item in items:
            disclosure = item["messages"][0]["content"]
            reply = item["messages"][1]["content"]
            invented = v27.concrete_clusters(reply, category) - v27.concrete_clusters(disclosure, category)
            assert not invented, (category, sorted(invented), disclosure, reply)


def test_the_abstract_caps_hold_on_the_shipped_surface() -> None:
    """A cluster-free disclosure fails the build unless the cap admits it."""
    built = v27.collect_rows()["train"]
    for category in v27.CLUSTERED_CATEGORIES:
        disclosures = {item["messages"][0]["content"] for item in built if item["category"] == category}
        abstract = {text for text in disclosures if not v27.concrete_clusters(text, category)}
        assert len(abstract) == v27.ABSTRACT_DISCLOSURE_CAP[category], category
    assert v27.ABSTRACT_DISCLOSURE_CAP["psychosis_risk_other"] == EXPECTED_ABSTRACT_PSYCHOSIS


def test_the_shipped_family_escalates_inside_the_reflective_frames() -> None:
    replies = [candidate["messages"][1]["content"] for candidate, _ in release.v27_candidate_rows()]
    lead = v26.REFLECTIVE_SHAPES[0].split("{action}")[0].strip()
    inside = [reply for reply in replies if reply.startswith(lead)]
    assert inside, "the reflective frames must carry rows"
    for reply in inside:
        assert any(term in reply.lower() for term in v27.ESCALATION_TERMS), reply


def test_the_release_counts_the_rewrites_it_ships() -> None:
    assert len(v27.REPLACED_PARAPHRASES) == EXPECTED_REWRITES
    assert len(v27.REPLACED_PARAPHRASES) < len(v26.REPLACED_PARAPHRASES)


def sample_manifest() -> dict[str, Any]:
    return {
        "scope": {"appended_rows": 0, "appended_rows_by_source": {}},
        "payload_contract": {"shuffle": {"train": False, "validation": False, "test": False, "seed": 0}},
        "limitations": list(prev._limitations()),
        "owner_direction": "User-directed v26 after the v25 eval.",
    }


def test_stamping_rewrites_every_release_dependent_field() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v27_rows=V27_ROWS
    )
    assert stamped["release_id"] == release.RELEASE_ID
    assert stamped["created_at_utc"] == release.CREATED_AT_UTC
    assert stamped["payload_contract"]["shuffle"] == {
        "train": True,
        "validation": False,
        "test": False,
        "seed": release.SHUFFLE_SEED,
    }
    assert stamped["limitations"] == list(release._limitations())
    assert stamped["owner_direction"] != sample_manifest()["owner_direction"]


def test_stamping_replaces_the_addenda_counts_it_is_given() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v27_rows=V27_ROWS
    )
    assert stamped["scope"]["appended_rows_by_source"] == {
        v14.V9_SOURCE_LABEL: V9_ROWS,
        v15.V10_SOURCE_LABEL: V10_ROWS,
        v16.V11_SOURCE_LABEL: V11_ROWS,
        release.V27_SOURCE_LABEL: V27_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V27_ROWS
    assert prev.V26_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]
    assert v25_release.V25_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]


def test_the_owner_direction_records_the_measured_cause_and_the_fix() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v27_rows=V27_ROWS
    )
    direction = stamped["owner_direction"]
    assert "4,192 of 4,528 disclosure x action pairs (92.6%)" in direction
    assert "copied from the paired eval cases" in direction
    assert "11 to 13 actions" in direction
    assert "SECONDARY_CLUSTERS" in direction
    assert "rewrites the 6 disclosures" in direction
    assert "Concrete invention across the seven is 0 of 4,528 pairs" in direction
    assert "7.9% of train" in direction


def release_fixture(tmp_path) -> release._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return release._Release(indexes, tmp_path, dry_run=True)


def test_summary_reports_the_v27_release(tmp_path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.v27_rows = release.v27_candidate_rows()

    summary = fixture.summary()

    assert summary["release_id"] == release.RELEASE_ID
    assert summary["appended_v27_rows"] == V27_ROWS
    assert "appended_v26_rows" not in summary


def test_the_release_is_not_the_v26_builder() -> None:
    """Guard against a copy-paste that leaves the old family wired in."""
    assert release.v27 is not v23
    assert v27.FAMILY == v26.FAMILY
    assert v27.FAMILY == v25.FAMILY
    assert v27.RELEASE_ID != v26.RELEASE_ID
