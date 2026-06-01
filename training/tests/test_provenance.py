"""Tests for training data provenance helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from training.provenance import (
    ProvenanceOptions,
    attach_provenance,
    build_provenance,
    validate_license,
)


def test_build_provenance_validates_required_fields() -> None:
    provenance = build_provenance(
        "file:///tmp/source.jsonl",
        "youtube",
        options=ProvenanceOptions(
            license_id="NOASSERTION",
            transformations=("normalize", "deduplicate"),
        ),
    )

    assert provenance["source_url"] == "file:///tmp/source.jsonl"
    assert provenance["source_type"] == "youtube"
    assert provenance["license"] == "NOASSERTION"
    assert provenance["transformations"] == ["normalize", "deduplicate"]
    assert provenance["acquired_at"].endswith("+00:00")


def test_license_validation_rejects_unknown_identifier() -> None:
    with pytest.raises(ValueError, match="Unsupported license"):
        validate_license("made-up-license")


def test_attach_provenance_does_not_mutate_input() -> None:
    record = {"instruction": "hello", "output": "world"}
    provenance = build_provenance("synthetic://unit", "synthetic_sdg")

    enriched = attach_provenance(record, provenance)

    assert "provenance" not in record
    assert enriched["provenance"]["source_url"] == "synthetic://unit"


def test_query_provenance_script_filters_jsonl(tmp_path: Path) -> None:
    data_path = tmp_path / "records.jsonl"
    kept = {
        "instruction": "a",
        "provenance": build_provenance("synthetic://kept", "synthetic_sdg"),
    }
    skipped = {
        "instruction": "b",
        "provenance": build_provenance("file:///yt", "youtube"),
    }
    data_path.write_text(
        json.dumps(kept, sort_keys=True) + "\n" + json.dumps(skipped, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["../scripts/devops/query-provenance.py", str(data_path), "--source-type", "synthetic_sdg"],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(lines) == 1
    assert lines[0]["provenance"]["source_url"] == "synthetic://kept"


def test_backfill_provenance_script_enriches_legacy_jsonl(tmp_path: Path) -> None:
    data_path = tmp_path / "legacy.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "instruction": "summarize",
                "output": "summary",
                "language": "en",
                "source_channel": "unit-channel",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "../scripts/devops/backfill-training-provenance.py",
            str(data_path),
            "--source-type",
            "youtube",
            "--acquired-at",
            "2026-05-06T22:31:18.221207+00:00",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stats = json.loads(result.stdout)
    assert stats["records_changed"] == 1
    [record] = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    provenance = record["provenance"]
    assert provenance["source_type"] == "youtube"
    assert provenance["license"] == "NOASSERTION"
    assert provenance["transformations"] == ["legacy_jsonl_backfill"]
    assert provenance["metadata"]["channel"] == "unit-channel"
