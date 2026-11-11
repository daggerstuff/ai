"""Tests for the v20 release: v17's pipeline with a crisis family big enough to count.

v19 fixed the register and still lost two crisis cases to the corpus majority
register. v20 changes the *amount* of crisis signal and the guard that keeps it
concrete, while leaving the corpus untouched. Both properties are asserted here
rather than discovered in a build log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import training.build_agent_tool_sft_v13 as v13
import training.build_agent_tool_sft_v20 as v20
import training.build_v14_release as v14
import training.build_v15_release as v15
import training.build_v16_release as v16
import training.build_v17_release as v17
import training.build_v18_release as v18
import training.build_v19_release as v19
import training.build_v20_release as release

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V20_ROWS = 1760
MESSAGES_PER_ROW = 2


def test_the_release_identity_names_v20() -> None:
    assert release.RELEASE_ID.endswith("-v20")
    assert release.RELEASE_ID in str(release.OUTPUT)
    assert release.RELEASE_ID in str(release.MANIFEST)
    assert release.SHUFFLE_SEED not in {
        v19.SHUFFLE_SEED,
        v18.SHUFFLE_SEED,
        v17.SHUFFLE_SEED,
        v16.SHUFFLE_SEED,
    }


def test_limitations_drop_the_stale_v13_family_claims() -> None:
    """v19 still described the v13 family's size and placement; v20 replaces it."""
    inherited = v19._limitations()
    for stale in release._STALE_V13_LIMITATIONS:
        assert any(text.startswith(stale) for text in inherited), stale
        assert not any(text.startswith(stale) for text in release._limitations()), stale
    assert len(release._limitations()) == (
        len(inherited) - len(release._STALE_V13_LIMITATIONS) + len(release.LIMITATIONS_V20)
    )


def test_limitations_keep_every_other_v19_entry() -> None:
    inherited = v19._limitations()
    kept = tuple(text for text in inherited if not text.startswith(release._STALE_V13_LIMITATIONS))
    assert release._limitations()[: len(kept)] == kept


def test_limitations_record_the_size_and_the_unreviewed_register() -> None:
    added = " ".join(release.LIMITATIONS_V20)
    assert "1,760 rows" in added
    assert "428" in added
    assert "2.1% of train" in added
    assert "not reviewed by a clinician" in added


def test_the_pipeline_is_v17s_with_only_the_addendum_swapped() -> None:
    assert issubclass(release._Release, v17._Release)
    assert not issubclass(release._Release, v19._Release)
    assert release._Release.carry is v17._Release.carry
    # The family replacement is the one deliberate difference from v19.
    assert release._Release.append_addenda is not v17._Release.append_addenda
    assert release._Release.verify_totals is not v17._Release.verify_totals


def test_the_addendum_is_v20_and_v13_is_gone() -> None:
    assert release.V20_SOURCE_LABEL == "crisis_direct_v20"
    assert release.V20_SOURCE_LABEL != v17.V13_SOURCE_LABEL
    assert release.EXPECTED_V20_TRAIN_ROWS == V20_ROWS
    assert release.EXPECTED_V20_TRAIN_ROWS > v17.EXPECTED_V13_TRAIN_ROWS


def test_v20_candidate_rows_are_clean_chat_rows() -> None:
    rows = release.v20_candidate_rows()
    assert len(rows) == V20_ROWS
    for candidate, payload in rows:
        assert candidate["source"] == release.V20_SOURCE_LABEL
        assert not v14.is_agent_tool(candidate)
        assert len(candidate["messages"]) == MESSAGES_PER_ROW
        assert candidate["messages"][0]["role"] == "user"
        assert candidate["messages"][1]["role"] == "assistant"
        assert payload, "a v20 row must project to an assistant target"


def test_v20_reuses_the_v13_action_table() -> None:
    """The register v19 shipped is the one v20 keeps; only volume and surface grow."""
    assert v20.ACTIONS == v13.ACTIONS
    surface = v20.paraphrases()
    assert sum(len(items) for items in surface.values()) > sum(len(items) for items in v13.paraphrases().values())


def sample_manifest() -> dict[str, Any]:
    """The v19-shaped manifest the v20 stamping has to correct."""
    return {
        "release_id": v19.RELEASE_ID,
        "created_at_utc": v19.CREATED_AT_UTC,
        "scope": {
            "appended_rows": V9_ROWS + V10_ROWS + V11_ROWS + v17.EXPECTED_V13_TRAIN_ROWS,
            "appended_rows_by_source": {
                v14.V9_SOURCE_LABEL: V9_ROWS,
                v15.V10_SOURCE_LABEL: V10_ROWS,
                v16.V11_SOURCE_LABEL: V11_ROWS,
                v17.V13_SOURCE_LABEL: v17.EXPECTED_V13_TRAIN_ROWS,
            },
        },
        "payload_contract": {
            "shuffle": {"train": True, "validation": False, "test": False, "seed": v19.SHUFFLE_SEED},
            "agent_tool_note": f"2,807 v13 agent_tool rows plus {V9_ROWS} v9 rows",
        },
        "limitations": list(v19._limitations()),
        "owner_direction": "User-directed v19 after the v18 crisis re-run.",
    }


def test_stamping_rewrites_every_release_dependent_field() -> None:
    stamped = release.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v20_rows=V20_ROWS
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
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v20_rows=V20_ROWS
    )
    assert stamped["scope"]["appended_rows_by_source"] == {
        v14.V9_SOURCE_LABEL: V9_ROWS,
        v15.V10_SOURCE_LABEL: V10_ROWS,
        v16.V11_SOURCE_LABEL: V11_ROWS,
        release.V20_SOURCE_LABEL: V20_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V20_ROWS
    assert v17.V13_SOURCE_LABEL not in stamped["scope"]["appended_rows_by_source"]


def test_stamping_replaces_the_limitations_rather_than_extending_them() -> None:
    manifest = sample_manifest()
    manifest["limitations"] = ["stale claim from a previous release"]
    stamped = release.stamp_release_identity(
        manifest, v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v20_rows=V20_ROWS
    )
    assert stamped["limitations"] == list(release._limitations())
    assert "stale claim from a previous release" not in stamped["limitations"]


def release_fixture(tmp_path: Path) -> release._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return release._Release(indexes, tmp_path, dry_run=True)


def test_summary_reports_the_v20_release(tmp_path: Path) -> None:
    fixture = release_fixture(tmp_path)
    fixture.v20_rows = release.v20_candidate_rows()

    summary = fixture.summary()

    assert summary["release_id"] == release.RELEASE_ID
    assert summary["appended_v20_rows"] == V20_ROWS
    assert "appended_v13_rows" not in summary
