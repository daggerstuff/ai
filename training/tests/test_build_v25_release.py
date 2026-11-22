"""Tests for the v25 release: v24's pipeline with the 23-action grounded family.

v24 grounded every means-access pair and still lost two cases to one sentence —
"Call 911 now and say you have the means within reach" — because 32 of 67
disclosures named no recognised means and 63% of the category trained on the
abstract stand-in. The tests here pin the release side: the pipeline must be
v24's unchanged except for the addendum, the family must be the v25 builder's rows
verbatim, and the manifest must record the corrected surface, the deleted
sentence, the ratio it raised, and the fact that grounding covers one category
only.
"""

from __future__ import annotations

from typing import Any

import training.build_agent_tool_sft_v23 as v23
import training.build_agent_tool_sft_v24 as v24
import training.build_agent_tool_sft_v25 as v25
import training.build_v14_release as v14
import training.build_v15_release as v15
import training.build_v16_release as v16
import training.build_v17_release as v17
import training.build_v19_release as v19
import training.build_v20_release as v20
import training.build_v21_release as v21
import training.build_v22_release as v22
import training.build_v23_release as v23_release
import training.build_v24_release as prev
import training.build_v25_release as release

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V25_ROWS = 6504
MESSAGES_PER_ROW = 2
# Eight frames is the v22 lever; v23, v24 and v25 all keep it and change the rows.
EXPECTED_SHAPES = 8
EXPECTED_MEANS_ACCESS_ACTIONS = 24


def test_the_release_identity_names_v25() -> None:
    assert release.RELEASE_ID.endswith("-v25")
    assert release.RELEASE_ID in str(release.OUTPUT)
    assert release.RELEASE_ID in str(release.MANIFEST)
    assert release.SHUFFLE_SEED not in {
        prev.SHUFFLE_SEED,
        v23_release.SHUFFLE_SEED,
        v22.SHUFFLE_SEED,
        v21.SHUFFLE_SEED,
        v20.SHUFFLE_SEED,
        v19.SHUFFLE_SEED,
        v17.SHUFFLE_SEED,
        v16.SHUFFLE_SEED,
    }


def test_limitations_drop_the_stale_v24_family_claims() -> None:
    """v24 described its own family's size, ratio and placement; v25 replaces it."""
    inherited = prev._limitations()
    for stale in release._STALE_V24_LIMITATIONS:
        assert any(text.startswith(stale) for text in inherited), stale
        assert not any(text.startswith(stale) for text in release._limitations()), stale
    assert len(release._limitations()) == (
        len(inherited) - len(release._STALE_V24_LIMITATIONS) + len(release.LIMITATIONS_V25)
    )


def test_limitations_keep_every_other_v24_entry() -> None:
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(release._STALE_V24_LIMITATIONS))
    assert release._limitations()[: len(kept)] == kept


def test_limitations_record_the_size_the_ratio_and_the_kept_revert() -> None:
    added = " ".join(release.LIMITATIONS_V25)
    assert "6,504 rows" in added
    assert "7.9% of train" in added
    assert "838" in added
    assert "9.8 escalations per ordinary reply" in added
    assert "not reviewed by a clinician" in added
    assert "carries no corpus repair" in added


def test_limitations_say_grounding_covers_one_category_only() -> None:
    added = " ".join(release.LIMITATIONS_V25)
    assert "Grounding covers means_access only" in added
    assert "still rotate their eight actions" in added
    assert "24 actions for means_access" in added


def test_limitations_record_the_closed_anchor_list() -> None:
    """The 17-anchor vocabulary is the release's remaining coverage risk."""
    added = " ".join(release.LIMITATIONS_V25)
    assert "closed list" in added
    assert "barbell" in added
    assert "66 of those 67 disclosures name an object" in added


def test_the_pipeline_is_v24s_with_only_the_family_swapped() -> None:
    assert issubclass(release._Release, prev._Release)
    assert release._Release.carry is prev._Release.carry
    assert release._Release.append_addenda is not prev._Release.append_addenda
    assert release._Release.verify_totals is not prev._Release.verify_totals


def test_the_addendum_is_v25_and_the_older_families_are_gone() -> None:
    assert release.V25_SOURCE_LABEL == "crisis_direct_v25"
    assert release.V25_SOURCE_LABEL != prev.V24_SOURCE_LABEL
    assert release.V25_SOURCE_LABEL != v23_release.V23_SOURCE_LABEL
    assert release.V25_SOURCE_LABEL != v22.V22_SOURCE_LABEL
    assert release.EXPECTED_V25_TRAIN_ROWS == V25_ROWS
    assert release.EXPECTED_V25_TRAIN_ROWS == prev.EXPECTED_V24_TRAIN_ROWS


def test_v25_candidate_rows_are_clean_chat_rows() -> None:
    rows = release.v25_candidate_rows()
    assert len(rows) == V25_ROWS
    for candidate, payload in rows:
        assert set(candidate) == {"messages", "source"}
        assert candidate["source"] == release.V25_SOURCE_LABEL
        assert not v14.is_agent_tool(candidate)
        assert len(candidate["messages"]) == MESSAGES_PER_ROW
        assert candidate["messages"][0]["role"] == "user"
        assert candidate["messages"][1]["role"] == "assistant"
        assert payload, "a v25 row must project to an assistant target"


def test_the_family_is_the_v25_builder_rows_verbatim() -> None:
    """The release must ship what the builder validated, not a reshaped copy."""
    built = v25.collect_rows()["train"]
    candidates = release.v25_candidate_rows()
    assert [item["messages"] for item in built] == [candidate["messages"] for candidate, _ in candidates]


def test_the_family_is_built_by_the_v25_builder() -> None:
    assert release.v25 is v25
    assert prev.v24 is v24
    assert v25.SHAPES_PER_DISCLOSURE == EXPECTED_SHAPES
    assert len(v25.ACTIONS) == len(v24.ACTIONS)
    assert len(v25.ACTIONS["means_access"]) == EXPECTED_MEANS_ACCESS_ACTIONS
    assert len(v24.ACTIONS["means_access"]) < EXPECTED_MEANS_ACCESS_ACTIONS


def test_the_shipped_family_carries_no_abstract_stand_in() -> None:
    """The whole point of the release: the sentence that answered everything is gone."""
    replies = [candidate["messages"][1]["content"] for candidate, _ in release.v25_candidate_rows()]
    for phrase in v25.ABSTRACT_STAND_INS:
        assert not any(phrase in reply.lower() for reply in replies), phrase


def sample_manifest() -> dict[str, Any]:
    return {
        "scope": {"appended_rows": 0, "appended_rows_by_source": {}},
        "payload_contract": {"shuffle": {"train": False, "validation": False, "test": False, "seed": 0}},
        "limitations": list(prev._limitations()),
        "owner_direction": "User-directed v24 after the v23 eval.",
    }


def test_stamping_rewrites_every_release_dependent_field() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v25_rows=V25_ROWS
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
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v25_rows=V25_ROWS
    )
    assert stamped["scope"]["appended_rows_by_source"] == {
        v14.V9_SOURCE_LABEL: V9_ROWS,
        v15.V10_SOURCE_LABEL: V10_ROWS,
        v16.V11_SOURCE_LABEL: V11_ROWS,
        release.V25_SOURCE_LABEL: V25_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V25_ROWS
    assert prev.V24_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]
    assert v23_release.V23_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]


def test_the_owner_direction_records_the_measured_cause_and_the_fix() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v25_rows=V25_ROWS
    )
    direction = stamped["owner_direction"]
    assert "32 of 67 disclosures" in direction
    assert "336 of 536 rows (63%)" in direction
    assert "17 anchors" in direction
    assert "0 abstract rows, down from 336" in direction
    assert "7.9% of train" in direction


def release_fixture(tmp_path) -> release._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return release._Release(indexes, tmp_path, dry_run=True)


def test_summary_reports_the_v25_release(tmp_path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.v25_rows = release.v25_candidate_rows()

    summary = fixture.summary()

    assert summary["release_id"] == release.RELEASE_ID
    assert summary["appended_v25_rows"] == V25_ROWS
    assert "appended_v24_rows" not in summary


def test_the_release_is_not_the_v23_builder() -> None:
    """Guard against a copy-paste that leaves the old family wired in."""
    assert release.v25 is not v23
    assert v25.FAMILY == v23.FAMILY
    assert v25.RELEASE_ID != v24.RELEASE_ID
