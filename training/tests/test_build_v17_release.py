"""Tests for the v17 release: v16's pipeline with the v13 family in place of v12."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import training.build_agent_tool_sft_v13 as v13
import training.build_v17_release as v17

V9_ROWS = 624
V10_ROWS = 480
V11_ROWS = 240
V13_ROWS = 708


def test_the_v13_addendum_is_the_v13_train_split() -> None:
    rows = v13.collect_rows()
    assert len(v17.v13_candidate_rows()) == v17.EXPECTED_V13_TRAIN_ROWS
    assert sum(len(split) for split in rows.values()) == v17.EXPECTED_V13_TRAIN_ROWS
    assert set(rows) == {"train"}


def test_v13_candidate_rows_are_chat_rows() -> None:
    for candidate, payload in v17.v13_candidate_rows():
        assert not v17.v14.is_agent_tool(candidate)
        assert "tools" not in candidate
        assert candidate["source"] == v17.V13_SOURCE_LABEL
        assert payload[-1]["role"] == "assistant"
        assert payload == v17.v14.project_chat(candidate["messages"], Counter())


def test_v13_rows_project_without_redaction() -> None:
    """A redacted v13 row would reach train mangled, so the addendum fails instead."""
    for _candidate, payload in v17.v13_candidate_rows():
        for message in payload:
            content = message.get("content")
            if isinstance(content, str):
                assert "REDACTED" not in content


def test_the_addendum_labels_are_distinct() -> None:
    assert v17.V13_SOURCE_LABEL not in {
        v17.v14.V9_SOURCE_LABEL,
        v17.v15.V10_SOURCE_LABEL,
        v17.v16.V11_SOURCE_LABEL,
        v17.v16.V12_SOURCE_LABEL,
    }


def test_the_release_identity_names_v17() -> None:
    assert v17.RELEASE_ID.endswith("-v17")
    assert v17.RELEASE_ID in str(v17.OUTPUT)
    assert v17.RELEASE_ID in str(v17.MANIFEST)
    assert v17.SHUFFLE_SEED not in {
        v17.v16.SHUFFLE_SEED,
        v17.v15.SHUFFLE_SEED,
        v17.v14.SHUFFLE_SEED,
    }


def test_limitations_drop_the_stale_v12_claim() -> None:
    """v17 replaces the v12 family, so v16's sentence about it must go."""
    inherited = v17.v16._limitations()
    assert any(text.startswith(v17._STALE_V12_LIMITATION) for text in inherited)
    limitations = v17._limitations()
    assert not any(text.startswith(v17._STALE_V12_LIMITATION) for text in limitations)
    assert len(limitations) == len(inherited) - 1 + len(v17.LIMITATIONS_V17)


def test_limitations_keep_every_other_v16_entry() -> None:
    inherited = v17.v16._limitations()
    kept = tuple(text for text in inherited if not text.startswith(v17._STALE_V12_LIMITATION))
    assert v17._limitations()[: len(kept)] == kept
    added = " ".join(v17.LIMITATIONS_V17)
    assert "v13" in added
    assert "direct" in added
    assert "clinician" in added


def sample_manifest() -> dict[str, Any]:
    """The v16-shaped manifest the v17 stamping has to correct."""
    return {
        "release_id": v17.v16.RELEASE_ID,
        "created_at_utc": v17.v16.CREATED_AT_UTC,
        "scope": {
            "appended_rows": 1_776,
            "appended_rows_by_source": {
                v17.v14.V9_SOURCE_LABEL: V9_ROWS,
                v17.v15.V10_SOURCE_LABEL: V10_ROWS,
                v17.v16.V11_SOURCE_LABEL: V11_ROWS,
                v17.v16.V12_SOURCE_LABEL: 672,
            },
        },
        "payload_contract": {
            "shuffle": {"train": True, "validation": False, "test": False, "seed": v17.v16.SHUFFLE_SEED},
            "agent_tool_note": "2,807 v13 agent_tool rows plus 624 v9 rows plus 480 v10 rows",
        },
        "limitations": list(v17.v16._limitations()),
        "owner_direction": "User-directed v16 after the v15 paired eval.",
    }


def test_stamping_rewrites_every_addenda_dependent_field() -> None:
    stamped = v17.stamp_release_identity(
        sample_manifest(), v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v13_rows=V13_ROWS
    )
    assert stamped["release_id"] == v17.RELEASE_ID
    assert stamped["created_at_utc"] == v17.CREATED_AT_UTC
    assert stamped["scope"]["appended_rows_by_source"] == {
        v17.v14.V9_SOURCE_LABEL: V9_ROWS,
        v17.v15.V10_SOURCE_LABEL: V10_ROWS,
        v17.v16.V11_SOURCE_LABEL: V11_ROWS,
        v17.V13_SOURCE_LABEL: V13_ROWS,
    }
    assert stamped["scope"]["appended_rows"] == sum(stamped["scope"]["appended_rows_by_source"].values())
    assert stamped["payload_contract"]["shuffle"]["seed"] == v17.SHUFFLE_SEED
    assert f"{V11_ROWS} v11 rows" in stamped["payload_contract"]["agent_tool_note"]
    # v13 is chat-shaped, so the agent-tool note must not claim its rows.
    assert f"{V13_ROWS} v13 rows" not in stamped["payload_contract"]["agent_tool_note"]
    assert stamped["limitations"] == list(v17._limitations())
    assert stamped["owner_direction"] != sample_manifest()["owner_direction"]


def test_stamping_keeps_the_counts_it_is_given() -> None:
    counts = {
        v17.v14.V9_SOURCE_LABEL: 1,
        v17.v15.V10_SOURCE_LABEL: 2,
        v17.v16.V11_SOURCE_LABEL: 3,
        v17.V13_SOURCE_LABEL: 4,
    }
    stamped = v17.stamp_release_identity(sample_manifest(), v9_rows=1, v10_rows=2, v11_rows=3, v13_rows=4)
    assert stamped["scope"]["appended_rows_by_source"] == counts
    assert stamped["scope"]["appended_rows"] == sum(counts.values())


def test_stamping_replaces_the_limitations_rather_than_extending_them() -> None:
    manifest = sample_manifest()
    manifest["limitations"] = ["stale claim from a previous release"]
    stamped = v17.stamp_release_identity(
        manifest, v9_rows=V9_ROWS, v10_rows=V10_ROWS, v11_rows=V11_ROWS, v13_rows=V13_ROWS
    )
    assert stamped["limitations"] == list(v17._limitations())
    assert "stale claim from a previous release" not in stamped["limitations"]


def release_fixture(tmp_path: Path) -> v17._Release:
    """A dry-run release with no candidate to carry: only the addenda are exercised."""
    indexes = v17.v14._Indexes(references={}, internal_source={}, heldout_source={}, repairs={})
    return v17._Release(indexes, tmp_path, dry_run=True)


def test_append_addenda_appends_v13_and_never_v12(monkeypatch: Any, tmp_path: Path) -> None:
    release = release_fixture(tmp_path)
    # The v9 and v10 addenda read the built corpus and have their own suites; stub
    # them so this test measures only the family the swap changes.
    monkeypatch.setattr(v17.v15._Release, "append_addenda", lambda _self: None)
    monkeypatch.setattr(v17.v16, "v11_candidate_rows", list)

    release.append_addenda()

    assert release.source_counts == Counter({v17.V13_SOURCE_LABEL: V13_ROWS})
    assert {record["source_label"] for record in release.internal_records} == {v17.V13_SOURCE_LABEL}
    assert {record["split"] for record in release.internal_records} == {"train"}
    assert len(release.payload_lines["train"]) == V13_ROWS
    assert len(release.group_sets["train"]) == V13_ROWS


def test_the_swap_leaves_admission_untouched() -> None:
    """Only the addenda change: carry, run and the held-out cap stay v16's."""
    assert issubclass(v17._Release, v17.v16._Release)
    assert v17._Release.carry is v17.v16._Release.carry
    assert v17._Release.run is v17.v16._Release.run
    assert v17._Release.append_chat_row is v17.v16._Release.append_chat_row


def test_summary_reports_the_v13_family_and_drops_the_v12_counter(tmp_path: Path) -> None:
    release = release_fixture(tmp_path)
    release.v13_rows = v17.v13_candidate_rows()

    summary = release.summary()

    assert summary["appended_v13_rows"] == V13_ROWS
    assert "appended_v12_rows" not in summary
