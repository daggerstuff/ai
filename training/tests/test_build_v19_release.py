"""Tests for the v19 release: v17's pipeline with the corrected crisis families.

v19 changes the text of the synthetic crisis rows, not how many of them exist,
and it deliberately drops the v18 rewrite overlay. Both properties are asserted
here rather than discovered in a build log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import training.build_v14_release as v14
import training.build_v15_release as v15
import training.build_v16_release as v16
import training.build_v17_release as v17
import training.build_v18_release as v18
import training.build_v19_release as v19

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V13_ROWS = 708


def test_the_release_identity_names_v19() -> None:
    assert v19.RELEASE_ID.endswith("-v19")
    assert v19.RELEASE_ID in str(v19.OUTPUT)
    assert v19.RELEASE_ID in str(v19.MANIFEST)
    assert v19.SHUFFLE_SEED not in {
        v18.SHUFFLE_SEED,
        v17.SHUFFLE_SEED,
        v16.SHUFFLE_SEED,
    }


def test_limitations_drop_the_stale_v17_routing_claim() -> None:
    """v17 described a family that routes everything; v19 splits that family in two."""
    inherited = v17._limitations()
    assert any(text.startswith(v19._STALE_ROUTING_LIMITATION) for text in inherited)
    limitations = v19._limitations()
    assert not any(text.startswith(v19._STALE_ROUTING_LIMITATION) for text in limitations)
    assert len(limitations) == len(inherited) - 1 + len(v19.LIMITATIONS_V19)


def test_limitations_keep_every_other_v17_entry() -> None:
    inherited = v17._limitations()
    kept = tuple(text for text in inherited if not text.startswith(v19._STALE_ROUTING_LIMITATION))
    assert v19._limitations()[: len(kept)] == kept


def test_limitations_record_the_reversal_and_the_unrepaired_corpus() -> None:
    added = " ".join(v19.LIMITATIONS_V19)
    assert "passive ideation" in added
    assert "means access" in added
    assert "double opening" in added
    assert "not carried forward" in added


def test_the_pipeline_is_v17s() -> None:
    assert issubclass(v19._Release, v17._Release)
    assert not issubclass(v19._Release, v18._Release)
    assert v19._Release.append_addenda is v17._Release.append_addenda
    assert v19._Release.verify_totals is v17._Release.verify_totals


def test_the_release_carries_no_rewrite_overlay() -> None:
    """The overlay is the artefact v19 exists to undo, so none of it may ride along."""
    assert v19._Release.carry is v17._Release.carry
    assert v18._Release.carry is not v17._Release.carry
    assert not hasattr(v19, "REWRITE_PATH")
    assert not hasattr(v19, "HEADER_PATH")
    assert not hasattr(v19, "build_overlay")


def sample_manifest() -> dict[str, Any]:
    """The v17-shaped manifest the v19 stamping has to correct."""
    return {
        "release_id": v17.RELEASE_ID,
        "created_at_utc": v17.CREATED_AT_UTC,
        "scope": {
            "appended_rows": V9_ROWS + V10_ROWS + V11_ROWS + V13_ROWS,
            "appended_rows_by_source": {
                v14.V9_SOURCE_LABEL: V9_ROWS,
                v15.V10_SOURCE_LABEL: V10_ROWS,
                v16.V11_SOURCE_LABEL: V11_ROWS,
                v17.V13_SOURCE_LABEL: V13_ROWS,
            },
        },
        "payload_contract": {
            "shuffle": {"train": True, "validation": False, "test": False, "seed": v17.SHUFFLE_SEED},
            "agent_tool_note": f"2,807 v13 agent_tool rows plus {V9_ROWS} v9 rows",
        },
        "limitations": list(v17._limitations()),
        "owner_direction": "User-directed v17 after the v16 paired eval.",
    }


def test_stamping_rewrites_every_release_dependent_field() -> None:
    stamped = v19.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v13_rows=V13_ROWS
    )
    assert stamped["release_id"] == v19.RELEASE_ID
    assert stamped["created_at_utc"] == v19.CREATED_AT_UTC
    assert stamped["payload_contract"]["shuffle"] == {
        "train": True,
        "validation": False,
        "test": False,
        "seed": v19.SHUFFLE_SEED,
    }
    assert stamped["limitations"] == list(v19._limitations())
    assert stamped["owner_direction"] != sample_manifest()["owner_direction"]


def test_stamping_keeps_the_addenda_counts_it_is_given() -> None:
    stamped = v19.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v13_rows=V13_ROWS
    )
    assert stamped["scope"]["appended_rows_by_source"] == {
        v14.V9_SOURCE_LABEL: V9_ROWS,
        v15.V10_SOURCE_LABEL: V10_ROWS,
        v16.V11_SOURCE_LABEL: V11_ROWS,
        v17.V13_SOURCE_LABEL: V13_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == V9_ROWS + V10_ROWS + V11_ROWS + V13_ROWS


def test_stamping_replaces_the_limitations_rather_than_extending_them() -> None:
    manifest = sample_manifest()
    manifest["limitations"] = ["stale claim from a previous release"]
    stamped = v19.stamp_release_identity(
        manifest, v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v13_rows=V13_ROWS
    )
    assert stamped["limitations"] == list(v19._limitations())
    assert "stale claim from a previous release" not in stamped["limitations"]


def release_fixture(tmp_path: Path) -> v19._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return v19._Release(indexes, tmp_path, dry_run=True)


def test_summary_reports_the_v19_release(tmp_path: Path) -> None:
    release = release_fixture(tmp_path)
    release.v13_rows = v17.v13_candidate_rows()

    summary = release.summary()

    assert summary["release_id"] == v19.RELEASE_ID
    assert summary["appended_v13_rows"] == V13_ROWS
