"""Tests for the v26 release: v25's pipeline with the grounded, frame-invariant family.

v25 removed the abstract means stand-ins and still lost the overdose case and the
means-access case with byte-identical replies on dev and sealed: the overdose
reply was a generic ``recent_attempt`` line because no action named the
substance, and the means-access reply grounded the gun and then never escalated
because every frame was action-first. The tests here pin the release side: the
pipeline must be v25's unchanged except for the addendum, the family must be the
v26 builder's rows verbatim, and the manifest must record the two grounded
vocabularies, the reflective frames, the ratio it raised, and the closed lists
that remain.
"""

from __future__ import annotations

from typing import Any

import training.build_agent_tool_sft_v23 as v23
import training.build_agent_tool_sft_v24 as v24
import training.build_agent_tool_sft_v25 as v25
import training.build_agent_tool_sft_v26 as v26
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
import training.build_v25_release as prev
import training.build_v26_release as release

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V26_ROWS = 6504
MESSAGES_PER_ROW = 2
# Eight frames is the v22 lever; v23 through v26 all keep it and change the rows.
EXPECTED_SHAPES = 8
EXPECTED_REFLECTIVE_FRAMES = 3
EXPECTED_MEANS_ACCESS_ACTIONS = 24
EXPECTED_OVERDOSE_ACTIONS = 17
EXPECTED_RECENT_ATTEMPT_ACTIONS = 18
EXPECTED_REWRITES = 52


def test_the_release_identity_names_v26() -> None:
    assert release.RELEASE_ID.endswith("-v26")
    assert release.RELEASE_ID in str(release.OUTPUT)
    assert release.RELEASE_ID in str(release.MANIFEST)
    assert release.SHUFFLE_SEED not in {
        prev.SHUFFLE_SEED,
        v24_release.SHUFFLE_SEED,
        v23_release.SHUFFLE_SEED,
        v22.SHUFFLE_SEED,
        v21.SHUFFLE_SEED,
        v20.SHUFFLE_SEED,
        v19.SHUFFLE_SEED,
        v17.SHUFFLE_SEED,
        v16.SHUFFLE_SEED,
    }


def test_limitations_drop_the_stale_v25_family_claims() -> None:
    """v25 described its own family's size, ratio and placement; v26 replaces it."""
    inherited = prev._limitations()
    for stale in release._STALE_V25_LIMITATIONS:
        assert any(text.startswith(stale) for text in inherited), stale
        assert not any(text.startswith(stale) for text in release._limitations()), stale
    assert len(release._limitations()) == (
        len(inherited) - len(release._STALE_V25_LIMITATIONS) + len(release.LIMITATIONS_V26)
    )


def test_limitations_keep_every_other_v25_entry() -> None:
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(release._STALE_V25_LIMITATIONS))
    assert release._limitations()[: len(kept)] == kept


def test_limitations_record_the_size_the_ratio_and_the_kept_revert() -> None:
    added = " ".join(release.LIMITATIONS_V26)
    assert "6,504 rows" in added
    assert "7.9% of train" in added
    assert "838" in added
    assert "9.8 escalations per ordinary reply" in added
    assert "not reviewed by a clinician" in added
    assert "carries no corpus repair" in added


def test_limitations_say_grounding_covers_three_categories() -> None:
    added = " ".join(release.LIMITATIONS_V26)
    assert "Grounding covers three of the ten categories" in added
    assert "means_access, overdose_taken and recent_attempt" in added
    assert "still rotate their eight actions" in added
    assert "24 actions for means_access, 17 for overdose_taken and 18 for recent_attempt" in added


def test_limitations_record_the_remaining_closed_lists() -> None:
    """The vocabularies are closed, so an unlisted substance has no action."""
    added = " ".join(release.LIMITATIONS_V26)
    assert "closed lists" in added
    assert "lithium blister" in added
    assert "All 54 overdose disclosures and all 51 recent_attempt disclosures name an object" in added
    assert "66 of 67 means-access disclosures" in added


def test_limitations_record_the_frame_change_as_the_mechanism() -> None:
    added = " ".join(release.LIMITATIONS_V26)
    assert "three reflective-lead frames" in added
    assert "frame-invariant" in added


def test_the_pipeline_is_v25s_with_only_the_family_swapped() -> None:
    assert issubclass(release._Release, prev._Release)
    assert release._Release.carry is prev._Release.carry
    assert release._Release.append_addenda is not prev._Release.append_addenda
    assert release._Release.verify_totals is not prev._Release.verify_totals


def test_the_addendum_is_v26_and_the_older_families_are_gone() -> None:
    assert release.V26_SOURCE_LABEL == "crisis_direct_v26"
    assert release.V26_SOURCE_LABEL != prev.V25_SOURCE_LABEL
    assert release.V26_SOURCE_LABEL != v24_release.V24_SOURCE_LABEL
    assert release.V26_SOURCE_LABEL != v23_release.V23_SOURCE_LABEL
    assert release.EXPECTED_V26_TRAIN_ROWS == V26_ROWS
    assert release.EXPECTED_V26_TRAIN_ROWS == prev.EXPECTED_V25_TRAIN_ROWS


def test_v26_candidate_rows_are_clean_chat_rows() -> None:
    rows = release.v26_candidate_rows()
    assert len(rows) == V26_ROWS
    for candidate, payload in rows:
        assert set(candidate) == {"messages", "source"}
        assert candidate["source"] == release.V26_SOURCE_LABEL
        assert not v14.is_agent_tool(candidate)
        assert len(candidate["messages"]) == MESSAGES_PER_ROW
        assert candidate["messages"][0]["role"] == "user"
        assert candidate["messages"][1]["role"] == "assistant"
        assert payload, "a v26 row must project to an assistant target"


def test_the_family_is_the_v26_builder_rows_verbatim() -> None:
    """The release must ship what the builder validated, not a reshaped copy."""
    built = v26.collect_rows()["train"]
    candidates = release.v26_candidate_rows()
    assert [item["messages"] for item in built] == [candidate["messages"] for candidate, _ in candidates]


def test_the_family_is_built_by_the_v26_builder() -> None:
    assert release.v26 is v26
    assert prev.v25 is v25
    assert v26.SHAPES_PER_DISCLOSURE == EXPECTED_SHAPES
    assert len(v26.REFLECTIVE_SHAPES) == EXPECTED_REFLECTIVE_FRAMES
    assert len(v26.ACTIONS) == len(v25.ACTIONS)
    assert len(v26.ACTIONS["means_access"]) == EXPECTED_MEANS_ACCESS_ACTIONS
    assert len(v26.ACTIONS["overdose_taken"]) == EXPECTED_OVERDOSE_ACTIONS
    assert len(v26.ACTIONS["recent_attempt"]) == EXPECTED_RECENT_ATTEMPT_ACTIONS
    assert len(v25.ACTIONS["overdose_taken"]) < EXPECTED_OVERDOSE_ACTIONS


def test_every_grounded_row_is_answered_by_its_own_categorys_actions() -> None:
    """The v25 failure: an overdose disclosure answered by a recent_attempt line."""
    built = v26.collect_rows()["train"]
    for category in v26.GROUNDED_CATEGORIES:
        actions = v26.ACTIONS[category]
        items = [item for item in built if item["category"] == category]
        assert items, category
        for item in items:
            reply = item["messages"][1]["content"]
            assert any(action in reply for action in actions), (category, reply)


def test_the_shipped_family_escalates_inside_the_reflective_frames() -> None:
    replies = [candidate["messages"][1]["content"] for candidate, _ in release.v26_candidate_rows()]
    lead = v26.REFLECTIVE_SHAPES[0].split("{action}")[0].strip()
    inside = [reply for reply in replies if reply.startswith(lead)]
    assert inside, "the reflective frames must carry rows"
    for reply in inside:
        assert any(term in reply.lower() for term in v26.ESCALATION_TERMS), reply


def test_the_release_counts_the_rewrites_it_ships() -> None:
    assert len(v26.REPLACED_PARAPHRASES) == EXPECTED_REWRITES
    assert len(v26.REPLACED_PARAPHRASES) > len(v25.REPLACED_PARAPHRASES)


def sample_manifest() -> dict[str, Any]:
    return {
        "scope": {"appended_rows": 0, "appended_rows_by_source": {}},
        "payload_contract": {"shuffle": {"train": False, "validation": False, "test": False, "seed": 0}},
        "limitations": list(prev._limitations()),
        "owner_direction": "User-directed v25 after the v24 eval.",
    }


def test_stamping_rewrites_every_release_dependent_field() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v26_rows=V26_ROWS
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
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v26_rows=V26_ROWS
    )
    assert stamped["scope"]["appended_rows_by_source"] == {
        v14.V9_SOURCE_LABEL: V9_ROWS,
        v15.V10_SOURCE_LABEL: V10_ROWS,
        v16.V11_SOURCE_LABEL: V11_ROWS,
        release.V26_SOURCE_LABEL: V26_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V26_ROWS
    assert prev.V25_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]
    assert v24_release.V24_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]


def test_the_owner_direction_records_the_measured_cause_and_the_fix() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v26_rows=V26_ROWS
    )
    direction = stamped["owner_direction"]
    assert "31 of its 54 disclosures named only" in direction
    assert "17 actions" in direction
    assert "18 actions" in direction
    assert "rewrites the 52 disclosures" in direction
    assert "frame-invariant" in direction
    assert "7.9% of train" in direction


def release_fixture(tmp_path) -> release._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return release._Release(indexes, tmp_path, dry_run=True)


def test_summary_reports_the_v26_release(tmp_path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.v26_rows = release.v26_candidate_rows()

    summary = fixture.summary()

    assert summary["release_id"] == release.RELEASE_ID
    assert summary["appended_v26_rows"] == V26_ROWS
    assert "appended_v25_rows" not in summary


def test_the_release_is_not_the_v24_builder() -> None:
    """Guard against a copy-paste that leaves the old family wired in."""
    assert release.v26 is not v23
    assert v26.FAMILY == v25.FAMILY
    assert v26.RELEASE_ID != v25.RELEASE_ID
    assert v26.FAMILY == v24.FAMILY
