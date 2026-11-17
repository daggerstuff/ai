"""Tests for the v28 release: v27's pipeline with the three measured defects closed.

v27 grounded all ten crisis categories on the clusters the disclosure named and
scored 10/12 under the harness on both splits. The two graded failures and the
matcher gap behind one of them were measured:

* ``crisis-eating-disorder-syncope`` was answered with a withdrawal line, because
  the withdrawal disclosure "I stopped drinking on Monday and I cannot keep food
  down." carries ``medical_collapse``'s purge phrase "cannot keep food down".
* ``crisis-passive-ideation`` was answered with the ``not_exist`` action, whose
  wording was "…say you would rather not be here anymore." The graded case
  accepts "not around" as a substring and "not be here" is not it, so the action
  meant to satisfy the case never did.
* ``_mentions`` compared raw text, so "wasn't around" never matched "not around".

The tests here pin the release side: the pipeline must be v27's unchanged except
for the addendum, the family must be the v28 builder's rows verbatim, every
surface must be v27's except the one action v28 rewrites, the corpus landmine
must be gone from the shipped rows, and the manifest must record the three
defects and the inherited grounding, caps and closed lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import training.build_agent_tool_sft_v23 as v23
import training.build_agent_tool_sft_v25 as v25
import training.build_agent_tool_sft_v26 as v26
import training.build_agent_tool_sft_v27 as v27
import training.build_agent_tool_sft_v28 as v28
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
import training.build_v26_release as v26_release
import training.build_v27_release as prev
import training.build_v28_release as release

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V28_ROWS = 6504
MESSAGES_PER_ROW = 2
# Eight frames is the v22 lever; v23 through v28 all keep it and change the rows.
EXPECTED_SHAPES = 8
EXPECTED_REFLECTIVE_FRAMES = 3
EXPECTED_REWRITES = 7
EXPECTED_ABSTRACT_PSYCHOSIS = 37
# v27's surfaces for the seven categories it grounded by cluster; v28 inherits
# them and rewrites exactly one action inside passive_ideation.
CLUSTERED_SURFACES = {
    "suicide_plan": 13,
    "self_harm_active": 15,
    "psychosis_risk_other": 13,
    "assault_strangulation": 13,
    "medical_collapse": 11,
    "withdrawal": 8,
    "passive_ideation": 8,
}
# The three categories v26 grounded and v27 held unchanged.
INHERITED_SURFACES = {"means_access": 24, "overdose_taken": 17, "recent_attempt": 18}
# The one v27 disclosure that was a routing landmine, and the wording that made it.
LANDMINE_DISCLOSURE = "I stopped drinking on Monday and I cannot keep food down."
LANDMINE_FRAGMENT = "keep food down"


def test_the_release_identity_names_v28() -> None:
    assert release.RELEASE_ID.endswith("-v28")
    assert release.RELEASE_ID in str(release.OUTPUT)
    assert release.RELEASE_ID in str(release.MANIFEST)
    assert release.SHUFFLE_SEED not in {
        prev.SHUFFLE_SEED,
        v26_release.SHUFFLE_SEED,
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


def test_limitations_drop_the_stale_v27_family_claims() -> None:
    """v27 described its own family's size, ratio and placement; v28 replaces it."""
    inherited = prev._limitations()
    for stale in release._STALE_V27_LIMITATIONS:
        assert any(text.startswith(stale) for text in inherited), stale
        assert not any(text.startswith(stale) for text in release._limitations()), stale
    assert len(release._limitations()) == (
        len(inherited) - len(release._STALE_V27_LIMITATIONS) + len(release.LIMITATIONS_V28)
    )


def test_limitations_keep_every_other_v27_entry() -> None:
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(release._STALE_V27_LIMITATIONS))
    assert release._limitations()[: len(kept)] == kept


def test_limitations_record_the_size_the_ratio_and_the_kept_revert() -> None:
    added = " ".join(release.LIMITATIONS_V28)
    assert "6,504 rows" in added
    assert "7.9% of train" in added
    assert "838" in added
    assert "9.8 escalations per ordinary reply" in added
    assert "not reviewed by a clinician" in added
    assert "carries no corpus repair" in added


def test_limitations_record_the_three_defects_v28_closes() -> None:
    added = " ".join(release.LIMITATIONS_V28)
    assert "v28 changes three things and no row count" in added
    assert "one disclosure that was a routing landmine" in added
    assert "one action whose wording did not carry the term the graded case accepts" in added
    assert "the matcher that reads contractions" in added


def test_limitations_keep_the_grounding_statement_and_the_surfaces() -> None:
    added = " ".join(release.LIMITATIONS_V28)
    assert "grounding by cluster, not by sentence" in added
    assert "24 for means_access, 17 for overdose_taken, 18 for recent_attempt" in added
    assert "13 for suicide_plan, 15 for self_harm_active" in added
    assert "11 for medical_collapse" in added
    assert "8 for withdrawal and 8 for passive_ideation" in added


def test_limitations_record_the_concrete_only_invention_metric() -> None:
    """A warning clause is legitimate; claiming a gun is not. The metric says so."""
    added = " ".join(release.LIMITATIONS_V28)
    assert "counts concrete clusters only" in added
    assert "51 medical_collapse rows" in added
    assert "279 psychosis rows" in added
    assert "rather than hidden" in added


def test_limitations_record_the_remaining_closed_lists_and_caps() -> None:
    added = " ".join(release.LIMITATIONS_V28)
    assert "closed lists" in added
    assert "lithium blister" in added
    assert "capped and the cap is enforced at build time" in added
    assert f"{EXPECTED_ABSTRACT_PSYCHOSIS} abstract psychosis disclosures" in added
    assert "a new cluster-free disclosure fails the build loudly" in added


def test_limitations_record_the_frame_invariance_as_the_mechanism() -> None:
    added = " ".join(release.LIMITATIONS_V28)
    assert "three reflective-lead frames" in added
    assert "escalation competes with that register from a trained position" in added


def test_the_pipeline_is_v27s_with_only_the_family_swapped() -> None:
    assert issubclass(release._Release, prev._Release)
    assert release._Release.carry is prev._Release.carry
    assert release._Release.append_addenda is not prev._Release.append_addenda
    assert release._Release.verify_totals is not prev._Release.verify_totals


def test_the_addendum_is_v28_and_the_older_families_are_gone() -> None:
    assert release.V28_SOURCE_LABEL == "crisis_direct_v28"
    assert release.V28_SOURCE_LABEL != prev.V27_SOURCE_LABEL
    assert release.V28_SOURCE_LABEL != v26_release.V26_SOURCE_LABEL
    assert release.V28_SOURCE_LABEL != v25_release.V25_SOURCE_LABEL
    assert release.V28_SOURCE_LABEL != v24_release.V24_SOURCE_LABEL
    assert release.V28_SOURCE_LABEL != v23_release.V23_SOURCE_LABEL
    assert release.EXPECTED_V28_TRAIN_ROWS == V28_ROWS
    assert release.EXPECTED_V28_TRAIN_ROWS == prev.EXPECTED_V27_TRAIN_ROWS


def test_v28_candidate_rows_are_clean_chat_rows() -> None:
    rows = release.v28_candidate_rows()
    assert len(rows) == V28_ROWS
    for candidate, payload in rows:
        assert set(candidate) == {"messages", "source"}
        assert candidate["source"] == release.V28_SOURCE_LABEL
        assert not v14.is_agent_tool(candidate)
        assert len(candidate["messages"]) == MESSAGES_PER_ROW
        assert candidate["messages"][0]["role"] == "user"
        assert candidate["messages"][1]["role"] == "assistant"
        assert payload, "a v28 row must project to an assistant target"


def test_the_family_is_the_v28_builder_rows_verbatim() -> None:
    """The release must ship what the builder validated, not a reshaped copy."""
    built = v28.collect_rows()["train"]
    candidates = release.v28_candidate_rows()
    assert [item["messages"] for item in built] == [candidate["messages"] for candidate, _ in candidates]


def test_the_family_is_built_by_the_v28_builder() -> None:
    assert release.v28 is v28
    assert prev.v27 is v27
    assert v28.SHAPES_PER_DISCLOSURE == EXPECTED_SHAPES
    assert len(v28.DIRECT_SHAPES) == EXPECTED_SHAPES
    assert len(v26.REFLECTIVE_SHAPES) == EXPECTED_REFLECTIVE_FRAMES
    assert v28.DIRECT_SHAPES == v27.DIRECT_SHAPES
    assert v28.GROUNDED_CATEGORIES == v27.GROUNDED_CATEGORIES
    assert v28.CLUSTERED_CATEGORIES == v27.CLUSTERED_CATEGORIES
    assert v28.ABSTRACT_DISCLOSURE_CAP == v27.ABSTRACT_DISCLOSURE_CAP


def test_every_surface_is_v27s_except_the_one_action_v28_rewrites() -> None:
    """v28 closes three defects; it must not reshape a surface it did not have to."""
    changed = {category for category in v28.ACTIONS if v28.ACTIONS[category] != v27.ACTIONS[category]}
    assert changed == {"passive_ideation"}
    for category, expected in {**CLUSTERED_SURFACES, **INHERITED_SURFACES}.items():
        assert len(v28.ACTIONS[category]) == expected, category
    differing = [
        (before, after)
        for before, after in zip(v27.ACTIONS["passive_ideation"], v28.ACTIONS["passive_ideation"])
        if before != after
    ]
    assert len(differing) == 1
    before, after = differing[0]
    assert "not be here" in before
    assert "not be here" not in after
    assert "not exist" in after


def test_every_grounded_row_is_answered_by_its_own_categorys_actions() -> None:
    built = v28.collect_rows()["train"]
    for category in v28.GROUNDED_CATEGORIES:
        actions = v28.ACTIONS[category]
        items = [item for item in built if item["category"] == category]
        assert items, category
        for item in items:
            reply = item["messages"][1]["content"]
            assert any(action in reply for action in actions), (category, reply)


def test_no_shipped_row_invents_a_concrete_cluster() -> None:
    """v27's invariant, which v28 must not weaken while closing its own defects."""
    built = v28.collect_rows()["train"]
    for category in v28.CLUSTERED_CATEGORIES:
        items = [item for item in built if item["category"] == category]
        assert items, category
        for item in items:
            disclosure = item["messages"][0]["content"]
            reply = item["messages"][1]["content"]
            invented = v28.concrete_clusters(reply, category) - v28.concrete_clusters(disclosure, category)
            assert not invented, (category, sorted(invented), disclosure, reply)


def test_the_abstract_caps_hold_on_the_shipped_surface() -> None:
    built = v28.collect_rows()["train"]
    for category in v28.CLUSTERED_CATEGORIES:
        disclosures = {item["messages"][0]["content"] for item in built if item["category"] == category}
        abstract = {text for text in disclosures if not v28.concrete_clusters(text, category)}
        assert len(abstract) == v28.ABSTRACT_DISCLOSURE_CAP[category], category
    assert v28.ABSTRACT_DISCLOSURE_CAP["psychosis_risk_other"] == EXPECTED_ABSTRACT_PSYCHOSIS


def test_the_corpus_landmine_is_gone_from_the_shipped_rows() -> None:
    """The withdrawal disclosure must not carry a purge phrase any more."""
    disclosures = {candidate["messages"][0]["content"] for candidate, _ in release.v28_candidate_rows()}
    assert LANDMINE_DISCLOSURE not in disclosures
    withdrawal = [item for item in v28.collect_rows()["train"] if item["category"] == "withdrawal"]
    for item in withdrawal:
        assert LANDMINE_FRAGMENT not in item["messages"][0]["content"], item["messages"][0]["content"]


def test_the_shipped_family_escalates_inside_the_reflective_frames() -> None:
    replies = [candidate["messages"][1]["content"] for candidate, _ in release.v28_candidate_rows()]
    lead = v26.REFLECTIVE_SHAPES[0].split("{action}")[0].strip()
    inside = [reply for reply in replies if reply.startswith(lead)]
    assert inside, "the reflective frames must carry rows"
    for reply in inside:
        assert any(term in reply.lower() for term in v28.ESCALATION_TERMS), reply


def test_the_release_counts_the_rewrites_it_ships() -> None:
    assert len(v28.REPLACED_PARAPHRASES) == EXPECTED_REWRITES
    assert len(v28.REPLACED_PARAPHRASES) > len(v27.REPLACED_PARAPHRASES)


def sample_manifest() -> dict[str, Any]:
    return {
        "scope": {"appended_rows": 0, "appended_rows_by_source": {}},
        "payload_contract": {"shuffle": {"train": False, "validation": False, "test": False, "seed": 0}},
        "limitations": list(prev._limitations()),
        "owner_direction": "User-directed v27 after the v26 eval.",
    }


def test_stamping_rewrites_every_release_dependent_field() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v28_rows=V28_ROWS
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
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v28_rows=V28_ROWS
    )
    assert stamped["scope"]["appended_rows_by_source"] == {
        v14.V9_SOURCE_LABEL: V9_ROWS,
        v15.V10_SOURCE_LABEL: V10_ROWS,
        v16.V11_SOURCE_LABEL: V11_ROWS,
        release.V28_SOURCE_LABEL: V28_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V28_ROWS
    assert prev.V27_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]
    assert v26_release.V26_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]


def test_the_owner_direction_records_the_measured_cause_and_the_fix() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v28_rows=V28_ROWS
    )
    direction = stamped["owner_direction"]
    assert "byte-identical replies on dev and sealed" in direction
    assert "one of medical_collapse's purge tokens" in direction
    assert "not be here" in direction
    assert "is not it" in direction
    assert "_mentions compared raw text" in direction
    assert "moves zero disclosures and zero actions" in direction
    assert "7.9% of train" in direction


def release_fixture(tmp_path: Path) -> release._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return release._Release(indexes, tmp_path, dry_run=True)


def test_summary_reports_the_v28_release(tmp_path: Path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.v28_rows = release.v28_candidate_rows()

    summary = fixture.summary()

    assert summary["release_id"] == release.RELEASE_ID
    assert summary["appended_v28_rows"] == V28_ROWS
    assert "appended_v27_rows" not in summary


def test_the_release_is_not_the_v27_builder() -> None:
    """Guard against a copy-paste that leaves the old family wired in."""
    assert release.v28 is not v23
    assert v28.FAMILY == v27.FAMILY
    assert v28.FAMILY == v25.FAMILY
    assert v28.RELEASE_ID != v27.RELEASE_ID
