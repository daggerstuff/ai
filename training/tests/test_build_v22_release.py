"""Tests for the v22 release: v20's pipeline with a much larger crisis family.

v22 carries no corpus repair — that is the whole difference from v21, which
stripped two phrases out of train and lost. The tests here pin that down: the
pipeline must be v20's unchanged except for the addendum, the strip helpers must
not exist, and the release must never mutate a carried row.
"""

from __future__ import annotations

from typing import Any

import training.build_agent_tool_sft_v20 as v20
import training.build_agent_tool_sft_v22 as v22
import training.build_v14_release as v14
import training.build_v15_release as v15
import training.build_v16_release as v16
import training.build_v17_release as v17
import training.build_v19_release as v19
import training.build_v20_release as prev
import training.build_v21_release as v21
import training.build_v22_release as release

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V20_ROWS = 1760
V22_ROWS = 5728
MESSAGES_PER_ROW = 2
# Eight frames is the v22 lever; v20 and v21 both used four.
EXPECTED_SHAPES = 8


def test_the_release_identity_names_v22() -> None:
    assert release.RELEASE_ID.endswith("-v22")
    assert release.RELEASE_ID in str(release.OUTPUT)
    assert release.RELEASE_ID in str(release.MANIFEST)
    assert release.SHUFFLE_SEED not in {
        v21.SHUFFLE_SEED,
        prev.SHUFFLE_SEED,
        v19.SHUFFLE_SEED,
        v17.SHUFFLE_SEED,
        v16.SHUFFLE_SEED,
    }


def test_limitations_drop_the_stale_v20_family_claims() -> None:
    """v20 described its own family's size, register and placement; v22 replaces it."""
    inherited = prev._limitations()
    for stale in release._STALE_V20_LIMITATIONS:
        assert any(text.startswith(stale) for text in inherited), stale
        assert not any(text.startswith(stale) for text in release._limitations()), stale
    assert len(release._limitations()) == (
        len(inherited) - len(release._STALE_V20_LIMITATIONS) + len(release.LIMITATIONS_V22)
    )


def test_limitations_keep_every_other_v20_entry() -> None:
    inherited = prev._limitations()
    kept = tuple(text for text in inherited if not text.startswith(release._STALE_V20_LIMITATIONS))
    assert release._limitations()[: len(kept)] == kept


def test_limitations_record_the_size_the_register_and_the_reverted_strip() -> None:
    added = " ".join(release.LIMITATIONS_V22)
    assert "5,728 rows" in added
    assert "722" in added
    assert "6.5% of train" in added
    assert "not reviewed by a clinician" in added
    assert "carries no corpus repair" in added


def test_the_pipeline_is_v20s_with_only_the_family_swapped() -> None:
    assert issubclass(release._Release, prev._Release)
    assert release._Release.carry is prev._Release.carry
    assert release._Release.append_addenda is not prev._Release.append_addenda
    assert release._Release.verify_totals is not prev._Release.verify_totals


def test_the_register_strip_is_gone() -> None:
    """v21's strip lost to v20; v22 must not resurrect it by inheritance."""
    assert release._Release.record_retained is prev._Release.record_retained
    for name in ("strip_train_register", "strip_register_sentences", "split_ledger", "STRIP_PHRASES"):
        assert not hasattr(release, name), name


def test_the_addendum_is_v22_and_the_older_families_are_gone() -> None:
    assert release.V22_SOURCE_LABEL == "crisis_direct_v22"
    assert release.V22_SOURCE_LABEL != prev.V20_SOURCE_LABEL
    assert release.V22_SOURCE_LABEL != v21.V21_SOURCE_LABEL
    assert release.EXPECTED_V22_TRAIN_ROWS == V22_ROWS
    assert release.EXPECTED_V22_TRAIN_ROWS > v21.EXPECTED_V21_TRAIN_ROWS


def test_v22_candidate_rows_are_clean_chat_rows() -> None:
    rows = release.v22_candidate_rows()
    assert len(rows) == V22_ROWS
    for candidate, payload in rows:
        assert candidate["source"] == release.V22_SOURCE_LABEL
        assert not v14.is_agent_tool(candidate)
        assert len(candidate["messages"]) == MESSAGES_PER_ROW
        assert candidate["messages"][0]["role"] == "user"
        assert candidate["messages"][1]["role"] == "assistant"
        assert payload, "a v22 row must project to an assistant target"


def test_the_family_is_built_by_the_v22_builder() -> None:
    assert release.v22 is v22
    assert v22.SHAPES_PER_DISCLOSURE == EXPECTED_SHAPES
    assert len(v22.ACTIONS) == len(v20.ACTIONS)


def sample_manifest() -> dict[str, Any]:
    return {
        "scope": {"appended_rows": 0, "appended_rows_by_source": {}},
        "payload_contract": {"shuffle": {"train": False, "validation": False, "test": False, "seed": 0}},
        "limitations": list(prev._limitations()),
        "owner_direction": "User-directed v20 after the v19 paired eval.",
    }


def test_stamping_rewrites_every_release_dependent_field() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v22_rows=V22_ROWS
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
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v22_rows=V22_ROWS
    )
    assert stamped["scope"]["appended_rows_by_source"] == {
        v14.V9_SOURCE_LABEL: V9_ROWS,
        v15.V10_SOURCE_LABEL: V10_ROWS,
        v16.V11_SOURCE_LABEL: V11_ROWS,
        release.V22_SOURCE_LABEL: V22_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V22_ROWS
    assert prev.V20_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]
    assert v21.V21_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]


def test_the_owner_direction_records_the_v21_loss_and_the_new_lever() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v22_rows=V22_ROWS
    )
    direction = stamped["owner_direction"]
    assert "84/88" in direction
    assert "73/88" in direction
    assert "not going to jump" in direction
    assert "6.5% of train" in direction


def release_fixture(tmp_path) -> release._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return release._Release(indexes, tmp_path, dry_run=True)


def test_summary_reports_the_v22_release(tmp_path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.v22_rows = release.v22_candidate_rows()

    summary = fixture.summary()

    assert summary["release_id"] == release.RELEASE_ID
    assert summary["appended_v22_rows"] == V22_ROWS
    assert "appended_v20_rows" not in summary
