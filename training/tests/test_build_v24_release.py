"""Tests for the v24 release: v23's pipeline with the grounded family.

v23 closed the escalation hole and still failed the means-access case once the
judge read it, because the category had no time-cued disclosure, every action
named two means at once, and actions were rotated instead of grounded. The tests
here pin the release side: the pipeline must be v23's unchanged except for the
addendum, the family must be the v24 builder's rows verbatim, and the manifest
must record the corrected size, the ratio it raised, the kept revert, and the
fact that grounding covers one category only.
"""

from __future__ import annotations

from typing import Any

import training.build_agent_tool_sft_v23 as v23
import training.build_agent_tool_sft_v24 as v24
import training.build_v14_release as v14
import training.build_v15_release as v15
import training.build_v16_release as v16
import training.build_v17_release as v17
import training.build_v18_release as v18
import training.build_v19_release as v19
import training.build_v20_release as v20
import training.build_v21_release as v21
import training.build_v22_release as v22
import training.build_v23_release as prev
import training.build_v24_release as release

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V24_ROWS = 6504
MESSAGES_PER_ROW = 2
# Eight frames is the v22 lever; v23 and v24 both keep it and change the rows.
EXPECTED_SHAPES = 8


def test_the_release_identity_names_v24() -> None:
    assert release.RELEASE_ID.endswith("-v24")
    assert release.RELEASE_ID in str(release.OUTPUT)
    assert release.RELEASE_ID in str(release.MANIFEST)
    assert release.SHUFFLE_SEED not in {
        prev.SHUFFLE_SEED,
        v22.SHUFFLE_SEED,
        v21.SHUFFLE_SEED,
        v20.SHUFFLE_SEED,
        v19.SHUFFLE_SEED,
        v18.SHUFFLE_SEED,
        v17.SHUFFLE_SEED,
        v16.SHUFFLE_SEED,
    }


def test_limitations_drop_the_stale_v23_family_claims() -> None:
    """v23 described its own family's size, ratio and placement; v24 replaces it."""
    inherited = prev._limitations()
    for stale in release._STALE_V23_LIMITATIONS:
        assert any(text.startswith(stale) for text in inherited), stale
        assert not any(text.startswith(stale) for text in release._limitations()), stale
    assert len(release._limitations()) == (
        len(inherited) - len(release._STALE_V23_LIMITATIONS) + len(release.LIMITATIONS_V24)
    )


def test_limitations_keep_every_other_v23_entry() -> None:
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(release._STALE_V23_LIMITATIONS))
    assert release._limitations()[: len(kept)] == kept


def test_limitations_record_the_size_the_ratio_and_the_kept_revert() -> None:
    added = " ".join(release.LIMITATIONS_V24)
    assert "6,504 rows" in added
    assert "7.9% of train" in added
    assert "838" in added
    assert "9.8 escalations per ordinary reply" in added
    assert "not reviewed by a clinician" in added
    assert "carries no corpus repair" in added


def test_limitations_say_grounding_covers_means_and_inherits_v23() -> None:
    added = " ".join(release.LIMITATIONS_V24)
    assert "Grounding covers means_access with a one-anchor rewrite" in added
    assert "inherit v23's per-disclosure pools" in added


def test_the_pipeline_is_v23s_with_only_the_family_swapped() -> None:
    assert issubclass(release._Release, prev._Release)
    assert release._Release.carry is prev._Release.carry
    assert release._Release.append_addenda is not prev._Release.append_addenda
    assert release._Release.verify_totals is not prev._Release.verify_totals


def test_the_addendum_is_v24_and_the_older_families_are_gone() -> None:
    assert release.V24_SOURCE_LABEL == "crisis_direct_v24"
    assert release.V24_SOURCE_LABEL != prev.V23_SOURCE_LABEL
    assert release.V24_SOURCE_LABEL != v22.V22_SOURCE_LABEL
    assert release.V24_SOURCE_LABEL != v21.V21_SOURCE_LABEL
    assert release.EXPECTED_V24_TRAIN_ROWS == V24_ROWS
    assert release.EXPECTED_V24_TRAIN_ROWS > prev.EXPECTED_V23_TRAIN_ROWS


def test_v24_candidate_rows_are_clean_chat_rows() -> None:
    rows = release.v24_candidate_rows()
    assert len(rows) == V24_ROWS
    for candidate, payload in rows:
        assert set(candidate) == {"messages", "source"}
        assert candidate["source"] == release.V24_SOURCE_LABEL
        assert not v14.is_agent_tool(candidate)
        assert len(candidate["messages"]) == MESSAGES_PER_ROW
        assert candidate["messages"][0]["role"] == "user"
        assert candidate["messages"][1]["role"] == "assistant"
        assert payload, "a v24 row must project to an assistant target"


def test_the_family_is_the_v24_builder_rows_verbatim() -> None:
    """The release must ship what the builder validated, not a reshaped copy."""
    built = v24.collect_rows()["train"]
    candidates = release.v24_candidate_rows()
    assert [item["messages"] for item in built] == [candidate["messages"] for candidate, _ in candidates]


def test_the_family_is_built_by_the_v24_builder() -> None:
    assert release.v24 is v24
    assert prev.v23 is v23
    assert v24.SHAPES_PER_DISCLOSURE == EXPECTED_SHAPES
    assert len(v24.ACTIONS) == len(v23.ACTIONS)


def sample_manifest() -> dict[str, Any]:
    return {
        "scope": {"appended_rows": 0, "appended_rows_by_source": {}},
        "payload_contract": {"shuffle": {"train": False, "validation": False, "test": False, "seed": 0}},
        "limitations": list(prev._limitations()),
        "owner_direction": "User-directed v23 after auditing v22 against the strict crisis gate.",
    }


def test_stamping_rewrites_every_release_dependent_field() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v24_rows=V24_ROWS
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
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v24_rows=V24_ROWS
    )
    assert stamped["scope"]["appended_rows_by_source"] == {
        v14.V9_SOURCE_LABEL: V9_ROWS,
        v15.V10_SOURCE_LABEL: V10_ROWS,
        v16.V11_SOURCE_LABEL: V11_ROWS,
        release.V24_SOURCE_LABEL: V24_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V24_ROWS
    assert prev.V23_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]
    assert v22.V22_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]


def test_the_owner_direction_records_the_measured_cause_and_the_fix() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v24_rows=V24_ROWS
    )
    direction = stamped["owner_direction"]
    assert "366 of 408 pairs" in direction
    assert "recent_attempt" in direction
    assert "16 completed-past time-cued disclosures" in direction
    assert "7.9% of train" in direction


def release_fixture(tmp_path) -> release._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return release._Release(indexes, tmp_path, dry_run=True)


def test_summary_reports_the_v24_release(tmp_path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.v24_rows = release.v24_candidate_rows()

    summary = fixture.summary()

    assert summary["release_id"] == release.RELEASE_ID
    assert summary["appended_v24_rows"] == V24_ROWS
    assert "appended_v23_rows" not in summary
