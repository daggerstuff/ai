"""Tests for the v23 release: v22's pipeline with the escalation-bearing family.

v22's whole lever was volume, and it still shipped 816 rows that never escalated
because two categories are engaged rather than acute. The tests here pin the
release side of the fix: the pipeline must be v22's unchanged except for the
addendum, the family must be the v23 builder's rows verbatim, and the manifest
must record the corrected size, the ratio it raised and the v21 revert it kept.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import training.build_agent_tool_sft_v22 as v22
import training.build_agent_tool_sft_v23 as v23
import training.build_v14_release as v14
import training.build_v15_release as v15
import training.build_v16_release as v16
import training.build_v17_release as v17
import training.build_v19_release as v19
import training.build_v20_release as v20
import training.build_v21_release as v21
import training.build_v22_release as prev
import training.build_v23_release as release

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V23_ROWS = 6376
MESSAGES_PER_ROW = 2
# Eight frames is the v22 lever; v23 keeps it and replaces the actions instead.
EXPECTED_SHAPES = 8


def test_the_release_identity_names_v23() -> None:
    assert release.RELEASE_ID.endswith("-v23")
    assert release.RELEASE_ID in str(release.OUTPUT)
    assert release.RELEASE_ID in str(release.MANIFEST)
    assert release.SHUFFLE_SEED not in {
        prev.SHUFFLE_SEED,
        v21.SHUFFLE_SEED,
        v20.SHUFFLE_SEED,
        v19.SHUFFLE_SEED,
        v17.SHUFFLE_SEED,
        v16.SHUFFLE_SEED,
    }


def test_limitations_drop_the_stale_v22_family_claims() -> None:
    """v22 described its own family's size, ratio and placement; v23 replaces it."""
    inherited = prev._limitations()
    for stale in release._STALE_V22_LIMITATIONS:
        assert any(text.startswith(stale) for text in inherited), stale
        assert not any(text.startswith(stale) for text in release._limitations()), stale
    assert len(release._limitations()) == (
        len(inherited) - len(release._STALE_V22_LIMITATIONS) + len(release.LIMITATIONS_V23)
    )


def test_limitations_keep_every_other_v22_entry() -> None:
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(release._STALE_V22_LIMITATIONS))
    assert release._limitations()[: len(kept)] == kept


def test_limitations_record_the_size_the_ratio_and_the_kept_revert() -> None:
    added = " ".join(release.LIMITATIONS_V23)
    assert "6,376 rows" in added
    assert "7.2% of train" in added
    assert "822" in added
    assert "9.6 escalations per ordinary reply" in added
    assert "not reviewed by a clinician" in added
    assert "carries no corpus repair" in added


def test_the_pipeline_is_v22s_with_only_the_family_swapped() -> None:
    assert issubclass(release._Release, prev._Release)
    assert release._Release.carry is prev._Release.carry
    assert release._Release.append_addenda is not prev._Release.append_addenda
    assert release._Release.verify_totals is not prev._Release.verify_totals


def test_the_addendum_is_v23_and_the_older_families_are_gone() -> None:
    assert release.V23_SOURCE_LABEL == "crisis_direct_v23"
    assert release.V23_SOURCE_LABEL != prev.V22_SOURCE_LABEL
    assert release.V23_SOURCE_LABEL != v21.V21_SOURCE_LABEL
    assert release.V23_SOURCE_LABEL != v20.V20_SOURCE_LABEL
    assert release.EXPECTED_V23_TRAIN_ROWS == V23_ROWS
    assert release.EXPECTED_V23_TRAIN_ROWS > prev.EXPECTED_V22_TRAIN_ROWS


def test_v23_candidate_rows_are_clean_chat_rows() -> None:
    rows = release.v23_candidate_rows()
    assert len(rows) == V23_ROWS
    for candidate, payload in rows:
        assert set(candidate) == {"messages", "source"}
        assert candidate["source"] == release.V23_SOURCE_LABEL
        assert not v14.is_agent_tool(candidate)
        assert len(candidate["messages"]) == MESSAGES_PER_ROW
        assert candidate["messages"][0]["role"] == "user"
        assert candidate["messages"][1]["role"] == "assistant"
        # The candidate carries only messages and source, so the identifying
        # value in the failure output is the disclosure text, not an id key the
        # candidate does not carry (a KeyError there would mask the mismatch).
        assert payload == v14.project_chat(candidate["messages"], Counter()), candidate["messages"][0]["content"]


def test_the_family_is_the_v23_builder_rows_verbatim() -> None:
    """The release must ship what the builder validated, not a reshaped copy."""
    built = v23.collect_rows()["train"]
    candidates = release.v23_candidate_rows()
    assert [item["messages"] for item in built] == [candidate["messages"] for candidate, _ in candidates]


def test_the_family_is_built_by_the_v23_builder() -> None:
    assert release.v23 is v23
    assert prev.v22 is v22
    assert v23.SHAPES_PER_DISCLOSURE == EXPECTED_SHAPES
    assert len(v23.ACTIONS) == len(v22.ACTIONS)


def sample_manifest() -> dict[str, Any]:
    return {
        "scope": {"appended_rows": 0, "appended_rows_by_source": {}},
        "payload_contract": {"shuffle": {"train": False, "validation": False, "test": False, "seed": 0}},
        "limitations": list(prev._limitations()),
        "owner_direction": "User-directed v22 after the v21 paired eval.",
    }


def test_stamping_rewrites_every_release_dependent_field() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v23_rows=V23_ROWS
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
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v23_rows=V23_ROWS
    )
    assert stamped["scope"]["appended_rows_by_source"] == {
        v14.V9_SOURCE_LABEL: V9_ROWS,
        v15.V10_SOURCE_LABEL: V10_ROWS,
        v16.V11_SOURCE_LABEL: V11_ROWS,
        release.V23_SOURCE_LABEL: V23_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V23_ROWS
    assert prev.V22_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]
    assert v21.V21_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]


def test_the_owner_direction_records_the_816_rows_and_the_fix() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v23_rows=V23_ROWS
    )
    direction = stamped["owner_direction"]
    assert "816" in direction
    assert "passive_ideation" in direction
    assert "means_access" in direction
    assert "irregular heartbeat" in direction
    assert "7.2% of train" in direction


def release_fixture(tmp_path) -> release._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return release._Release(indexes, tmp_path, dry_run=True)


def test_summary_reports_the_v23_release(tmp_path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.v23_rows = release.v23_candidate_rows()

    summary = fixture.summary()

    assert summary["release_id"] == release.RELEASE_ID
    assert summary["appended_v23_rows"] == V23_ROWS
    assert "appended_v22_rows" not in summary
