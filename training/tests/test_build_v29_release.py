"""Tests for the v29 release: the whole crisis family reaches training.

v24 through v28 each scored 10/12 under the harness on both splits, and the two
failing cases moved between releases while the row logic did not. The measured
cause was selection, not grounding: ``select_rows`` bucketed the appended crisis
rows into a routing pool capped at 2,000 or a hash-sampled filler of 4,000, so
5,904 crisis rows became 1,484 selected rows — 25% — keyed on a hash of the
serialized line, which is why a different 75% was dropped on every rebuild.

The tests here pin the release side: the pipeline must be v28's unchanged except
for the addendum, the family must be the v29 builder's rows verbatim, the surface
must be v28's except the 26 actions v29 rewrites to open on their own anchor, and
the manifest must record the selection defect and the anchor-first fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import training.build_agent_tool_sft_v23 as v23
import training.build_agent_tool_sft_v25 as v25
import training.build_agent_tool_sft_v26 as v26
import training.build_agent_tool_sft_v28 as v28
import training.build_agent_tool_sft_v29 as v29
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
import training.build_v27_release as v27_release
import training.build_v28_release as prev
import training.build_v29_release as release

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V29_ROWS = 6504
MESSAGES_PER_ROW = 2
# Eight frames is the v22 lever; v23 through v29 all keep it and change the rows.
EXPECTED_SHAPES = 8
EXPECTED_REFLECTIVE_FRAMES = 3
EXPECTED_REWRITES = 7
EXPECTED_ABSTRACT_PSYCHOSIS = 37
# The 26 surfaces v29 rewrites so the first clause names an anchor the disclosure
# gave. Action counts are untouched: the rewrite reorders, it does not add.
ANCHOR_FIRST_SURFACES = {
    "self_harm_active": 4,
    "psychosis_risk_other": 2,
    "assault_strangulation": 6,
    "medical_collapse": 10,
    "withdrawal": 4,
}
# The surfaces v28 and v29 share unchanged.
UNTOUCHED_SURFACES = {
    "suicide_plan": 13,
    "overdose_taken": 17,
    "recent_attempt": 18,
    "passive_ideation": 8,
    "means_access": 24,
}
# The v28 disclosure that was a routing landmine, and the wording that made it.
LANDMINE_DISCLOSURE = "I stopped drinking on Monday and I cannot keep food down."
LANDMINE_FRAGMENT = "keep food down"


def test_the_release_identity_names_v29() -> None:
    assert release.RELEASE_ID.endswith("-v29")
    assert release.RELEASE_ID in str(release.OUTPUT)
    assert release.RELEASE_ID in str(release.MANIFEST)
    assert release.SHUFFLE_SEED not in {
        prev.SHUFFLE_SEED,
        v27_release.SHUFFLE_SEED,
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


def test_limitations_drop_the_stale_v28_family_claims() -> None:
    """v28 described its own family's size, ratio and placement; v29 replaces it."""
    inherited = prev._limitations()
    for stale in release._STALE_V28_LIMITATIONS:
        assert any(text.startswith(stale) for text in inherited), stale
        assert not any(text.startswith(stale) for text in release._limitations()), stale
    assert len(release._limitations()) == (
        len(inherited) - len(release._STALE_V28_LIMITATIONS) + len(release.LIMITATIONS_V29)
    )


def test_limitations_keep_every_other_v28_entry() -> None:
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(release._STALE_V28_LIMITATIONS))
    assert release._limitations()[: len(kept)] == kept


def test_limitations_record_the_size_the_ratio_and_the_kept_revert() -> None:
    added = " ".join(release.LIMITATIONS_V29)
    assert "6,504 rows" in added
    assert "7.9% of train" in added
    assert "838" in added
    assert "9.8 escalations per ordinary reply" in added
    assert "not reviewed by a clinician" in added
    assert "carries no corpus repair" in added


def test_limitations_record_the_selection_defect_v29_closes() -> None:
    added = " ".join(release.LIMITATIONS_V29)
    assert "25% of the crisis rows survived the export" in added
    assert "capped routing pool or a hash-sampled filler" in added
    assert "moved on every rebuild" in added
    assert "All 6,504 rows now select" in added
    assert "48%" in added


def test_limitations_record_that_the_ratio_exposure_went_up() -> None:
    """Training on 4.4x more crisis rows multiplies whatever the ratio leaks."""
    added = " ".join(release.LIMITATIONS_V29)
    assert "training on 4.4x more crisis rows than v28 did" in added
    assert "the ratio is the thing to lower next" in added


def test_limitations_record_the_anchor_first_fix() -> None:
    added = " ".join(release.LIMITATIONS_V29)
    assert "opens on one of the clusters its own disclosure named" in added
    assert "rewrites the 26 surfaces that opened on a generic frame" in added
    assert "how long you have been off food" in added


def test_limitations_keep_the_grounding_statement_and_the_surfaces() -> None:
    added = " ".join(release.LIMITATIONS_V29)
    assert "grounding by cluster, not by sentence" in added
    assert "24 for means_access, 17 for overdose_taken, 18 for recent_attempt" in added
    assert "13 for suicide_plan, 15 for self_harm_active" in added
    assert "11 for medical_collapse" in added
    assert "8 for withdrawal and 8 for passive_ideation" in added


def test_limitations_record_the_concrete_only_invention_metric() -> None:
    """A warning clause is legitimate; claiming a gun is not. The metric says so."""
    added = " ".join(release.LIMITATIONS_V29)
    assert "counts concrete clusters only" in added
    assert "51 medical_collapse rows" in added
    assert "279 psychosis rows" in added
    assert "rather than hidden" in added


def test_limitations_record_the_remaining_closed_lists_and_caps() -> None:
    added = " ".join(release.LIMITATIONS_V29)
    assert "closed lists" in added
    assert "lithium blister" in added
    assert "capped and the cap is enforced at build time" in added
    assert f"{EXPECTED_ABSTRACT_PSYCHOSIS} abstract psychosis disclosures" in added
    assert "a new cluster-free disclosure fails the build loudly" in added


def test_limitations_record_the_frame_invariance_as_the_mechanism() -> None:
    added = " ".join(release.LIMITATIONS_V29)
    assert "three reflective-lead frames" in added
    assert "escalation competes with that register from a trained position" in added


def test_limitations_warn_that_full_coverage_makes_memorisation_reachable() -> None:
    added = " ".join(release.LIMITATIONS_V29)
    assert "Training on the whole family makes memorisation more reachable" in added


def test_the_pipeline_is_v28s_with_only_the_family_swapped() -> None:
    assert issubclass(release._Release, prev._Release)
    assert release._Release.carry is prev._Release.carry
    assert release._Release.append_addenda is not prev._Release.append_addenda
    assert release._Release.verify_totals is not prev._Release.verify_totals


def test_the_addendum_is_v29_and_the_older_families_are_gone() -> None:
    assert release.V29_SOURCE_LABEL == "crisis_direct_v29"
    assert release.V29_SOURCE_LABEL != prev.V28_SOURCE_LABEL
    assert release.V29_SOURCE_LABEL != v27_release.V27_SOURCE_LABEL
    assert release.V29_SOURCE_LABEL != v26_release.V26_SOURCE_LABEL
    assert release.V29_SOURCE_LABEL != v25_release.V25_SOURCE_LABEL
    assert release.V29_SOURCE_LABEL != v24_release.V24_SOURCE_LABEL
    assert release.V29_SOURCE_LABEL != v23_release.V23_SOURCE_LABEL
    assert release.EXPECTED_V29_TRAIN_ROWS == V29_ROWS
    assert release.EXPECTED_V29_TRAIN_ROWS == prev.EXPECTED_V28_TRAIN_ROWS


def test_v29_candidate_rows_are_clean_chat_rows() -> None:
    rows = release.v29_candidate_rows()
    assert len(rows) == V29_ROWS
    for candidate, payload in rows:
        assert set(candidate) == {"messages", "source"}
        assert candidate["source"] == release.V29_SOURCE_LABEL
        assert not v14.is_agent_tool(candidate)
        assert len(candidate["messages"]) == MESSAGES_PER_ROW
        assert candidate["messages"][0]["role"] == "user"
        assert candidate["messages"][1]["role"] == "assistant"
        assert payload, "a v29 row must project to an assistant target"


def test_the_family_is_the_v29_builder_rows_verbatim() -> None:
    """The release must ship what the builder validated, not a reshaped copy."""
    built = v29.collect_rows()["train"]
    candidates = release.v29_candidate_rows()
    assert [item["messages"] for item in built] == [candidate["messages"] for candidate, _ in candidates]


def test_the_family_is_built_by_the_v29_builder() -> None:
    assert release.v29 is v29
    assert prev.v28 is v28
    assert v29.SHAPES_PER_DISCLOSURE == EXPECTED_SHAPES
    assert len(v29.DIRECT_SHAPES) == EXPECTED_SHAPES
    assert len(v26.REFLECTIVE_SHAPES) == EXPECTED_REFLECTIVE_FRAMES
    assert v29.DIRECT_SHAPES == v28.DIRECT_SHAPES
    assert v29.GROUNDED_CATEGORIES == v28.GROUNDED_CATEGORIES
    assert v29.CLUSTERED_CATEGORIES == v28.CLUSTERED_CATEGORIES
    assert v29.ABSTRACT_DISCLOSURE_CAP == v28.ABSTRACT_DISCLOSURE_CAP


def test_every_surface_is_v28s_except_the_anchor_first_rewrites() -> None:
    """v29 fixes routing; it must not reshape a surface it did not have to."""
    changed = {category for category in v29.ACTIONS if v29.ACTIONS[category] != v28.ACTIONS[category]}
    assert changed == set(ANCHOR_FIRST_SURFACES)
    for category in v29.ACTIONS:
        assert len(v29.ACTIONS[category]) == len(v28.ACTIONS[category]), category
    for category, expected in UNTOUCHED_SURFACES.items():
        assert len(v29.ACTIONS[category]) == expected, category
    rewritten = {
        category: sum(
            1 for before, after in zip(v28.ACTIONS[category], v29.ACTIONS[category], strict=True) if before != after
        )
        for category in ANCHOR_FIRST_SURFACES
    }
    assert rewritten == ANCHOR_FIRST_SURFACES
    assert sum(rewritten.values()) == len(v29.ANCHOR_FIRST_REWRITES)


def test_every_grounded_action_opens_on_one_of_its_own_anchors() -> None:
    """The v28 losses shared one shape: a generic first clause. v29 bans it."""
    for category in v29.CLUSTERED_CATEGORIES:
        for action in v29.ACTIONS[category]:
            assert v29.opens_on_anchor(action, category), (category, action)


def test_the_anchor_first_rewrite_only_reorders_the_same_clusters() -> None:
    """A rewrite that changed cluster membership would be a surface change."""
    for before, after in v29.ANCHOR_FIRST_REWRITES.items():
        owners = [name for name, lines in v28.ACTIONS.items() if before in lines]
        assert len(owners) == 1, before
        category = owners[0]
        assert v29.clusters(before, category) == v29.clusters(after, category), before


def test_every_grounded_row_is_answered_by_its_own_categorys_actions() -> None:
    built = v29.collect_rows()["train"]
    for category in v29.GROUNDED_CATEGORIES:
        actions = v29.ACTIONS[category]
        items = [item for item in built if item["category"] == category]
        assert items, category
        for item in items:
            reply = item["messages"][1]["content"]
            assert any(action in reply for action in actions), (category, reply)


def test_no_shipped_row_invents_a_concrete_cluster() -> None:
    """v27's invariant, which v29 must not weaken while reordering the surface."""
    built = v29.collect_rows()["train"]
    for category in v29.CLUSTERED_CATEGORIES:
        items = [item for item in built if item["category"] == category]
        assert items, category
        for item in items:
            disclosure = item["messages"][0]["content"]
            reply = item["messages"][1]["content"]
            invented = v29.concrete_clusters(reply, category) - v29.concrete_clusters(disclosure, category)
            assert not invented, (category, sorted(invented), disclosure, reply)


def test_the_abstract_caps_hold_on_the_shipped_surface() -> None:
    built = v29.collect_rows()["train"]
    for category in v29.CLUSTERED_CATEGORIES:
        disclosures = {item["messages"][0]["content"] for item in built if item["category"] == category}
        abstract = {text for text in disclosures if not v29.concrete_clusters(text, category)}
        assert len(abstract) == v29.ABSTRACT_DISCLOSURE_CAP[category], category
    assert v29.ABSTRACT_DISCLOSURE_CAP["psychosis_risk_other"] == EXPECTED_ABSTRACT_PSYCHOSIS


def test_the_corpus_landmine_stays_gone_from_the_shipped_rows() -> None:
    """The withdrawal disclosure must not carry a purge phrase any more."""
    disclosures = {candidate["messages"][0]["content"] for candidate, _ in release.v29_candidate_rows()}
    assert LANDMINE_DISCLOSURE not in disclosures
    withdrawal = [item for item in v29.collect_rows()["train"] if item["category"] == "withdrawal"]
    for item in withdrawal:
        assert LANDMINE_FRAGMENT not in item["messages"][0]["content"], item["messages"][0]["content"]


def test_the_shipped_family_escalates_inside_the_reflective_frames() -> None:
    replies = [candidate["messages"][1]["content"] for candidate, _ in release.v29_candidate_rows()]
    lead = v26.REFLECTIVE_SHAPES[0].split("{action}")[0].strip()
    inside = [reply for reply in replies if reply.startswith(lead)]
    assert inside, "the reflective frames must carry rows"
    for reply in inside:
        assert any(term in reply.lower() for term in v29.ESCALATION_TERMS), reply


def test_the_release_inherits_the_paraphrase_rewrites_it_ships() -> None:
    assert len(v29.REPLACED_PARAPHRASES) == EXPECTED_REWRITES
    assert v29.REPLACED_PARAPHRASES == v28.REPLACED_PARAPHRASES


def sample_manifest() -> dict[str, Any]:
    return {
        "scope": {"appended_rows": 0, "appended_rows_by_source": {}},
        "payload_contract": {"shuffle": {"train": False, "validation": False, "test": False, "seed": 0}},
        "limitations": list(prev._limitations()),
        "owner_direction": "User-directed v28 after the v27 eval.",
    }


def test_stamping_rewrites_every_release_dependent_field() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v29_rows=V29_ROWS
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
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v29_rows=V29_ROWS
    )
    assert stamped["scope"]["appended_rows_by_source"] == {
        v14.V9_SOURCE_LABEL: V9_ROWS,
        v15.V10_SOURCE_LABEL: V10_ROWS,
        v16.V11_SOURCE_LABEL: V11_ROWS,
        release.V29_SOURCE_LABEL: V29_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V29_ROWS
    assert prev.V28_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]
    assert v27_release.V27_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]


def test_the_owner_direction_records_the_measured_cause_and_the_fix() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v29_rows=V29_ROWS
    )
    direction = stamped["owner_direction"]
    assert "Don't limit to what we have/see right here" in direction
    assert "5,904 crisis rows in the train partition became 1,484 selected rows" in direction
    assert "keyed on a hash of the serialized line" in direction
    assert "SelectionOptions.keep_replies" in direction
    assert "1,072 distinct assistant replies" in direction
    assert "7,104 to 13,608 rows" in direction
    assert "opens on its own anchor" in direction


def release_fixture(tmp_path: Path) -> release._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return release._Release(indexes, tmp_path, dry_run=True)


def test_summary_reports_the_v29_release(tmp_path: Path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.v29_rows = release.v29_candidate_rows()

    summary = fixture.summary()

    assert summary["release_id"] == release.RELEASE_ID
    assert summary["appended_v29_rows"] == V29_ROWS
    assert "appended_v28_rows" not in summary


def test_the_release_is_not_the_v28_builder() -> None:
    """Guard against a copy-paste that leaves the old family wired in."""
    assert release.v29 is not v23
    assert v29.FAMILY == v28.FAMILY
    assert v29.FAMILY == v25.FAMILY
    assert v29.RELEASE_ID != v28.RELEASE_ID
