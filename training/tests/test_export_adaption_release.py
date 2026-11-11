"""Contract tests for the Adaption AI export.

The upload format is ``prompt``/``completion`` pairs, so every multi-turn release
row has to become one datum per assistant turn. These tests pin the two things
that would silently corrupt a paid run: the prompt must contain every turn before
the completion (ledger intact, no weights), and an assistant-first arc turn must
be dropped rather than shipped as a completion with an empty prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from training.export_adaption_release import (
    DATUM_FLOOR,
    _datum_chars,
    export,
    main,
    unroll,
)


def _message(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


def test_unroll_emits_one_datum_per_assistant_turn_with_full_prompt() -> None:
    messages = [
        _message("user", "first"),
        _message("assistant", "reply one"),
        _message("user", "second"),
        _message("assistant", "reply two"),
    ]

    datums, skipped = unroll(messages)

    assert skipped == 0
    assert len(datums) == 2
    assert datums[0]["prompt"] == [_message("user", "first")]
    assert datums[0]["completion"] == [_message("assistant", "reply one")]
    assert datums[1]["prompt"] == [
        _message("user", "first"),
        _message("assistant", "reply one"),
        _message("user", "second"),
    ]
    assert datums[1]["completion"] == [_message("assistant", "reply two")]


def test_unroll_drops_the_weight_field_the_release_carries() -> None:
    messages = [
        {"role": "user", "content": "hi", "weight": 0},
        {"role": "assistant", "content": "hello", "weight": 1},
    ]

    datums, _ = unroll(messages)

    assert all("weight" not in message for message in datums[0]["prompt"])
    assert all("weight" not in message for message in datums[0]["completion"])


def test_unroll_skips_an_assistant_first_arc_turn_instead_of_shipping_it() -> None:
    messages = [
        _message("assistant", "ledger opener"),
        _message("user", "client turn"),
        _message("assistant", "therapist reply"),
    ]

    datums, skipped = unroll(messages)

    assert skipped == 1
    assert len(datums) == 1
    assert datums[0]["completion"] == [_message("assistant", "therapist reply")]
    assert datums[0]["prompt"] == [
        _message("assistant", "ledger opener"),
        _message("user", "client turn"),
    ]


def test_unroll_of_a_bare_assistant_row_yields_nothing() -> None:
    datums, skipped = unroll([_message("assistant", "classification")])

    assert datums == []
    assert skipped == 1


def test_datum_chars_counts_prompt_and_completion() -> None:
    datum = {
        "prompt": [_message("user", "abcd")],
        "completion": [_message("assistant", "ef")],
    }

    assert _datum_chars(datum) == 6


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    train_path = tmp_path / "train.jsonl"
    rows = [
        {"messages": [_message("user", "I feel low."), _message("assistant", "Call 988 tonight.")]},
        {"messages": [_message("user", "Hello."), _message("assistant", "Hi there.")]},
        {
            "messages": [
                _message("assistant", "ledger"),
                _message("user", "I cut again."),
                _message("assistant", "Call 911 now."),
            ]
        },
        {
            "messages": [_message("system", "tools"), _message("user", "save it")],
            "tools": [{"name": "memory_write"}],
        },
    ]
    train_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "release_id": "test-release-v99",
                "partitions": {
                    "train": {
                        "path": str(train_path),
                        "sha256": "0" * 64,
                        "row_count": len(rows),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, train_path


def test_export_dry_run_reports_without_writing(tmp_path: Path) -> None:
    manifest_path, _ = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"

    summary = export(manifest_path, out_dir, dry_run=True)

    assert summary["release_id"] == "test-release-v99"
    assert summary["rows"] == 3
    assert summary["datums"] == 3
    assert summary["datums_skipped_no_prompt"] == 1
    assert summary["selection"]["tool_rows_excluded"] == 1
    assert summary["meets_adaption_floor"] is False
    assert summary["chars"] > 0
    assert summary["estimated_tokens"] == round(summary["chars"] / 4)
    assert not out_dir.exists()


def test_export_writes_prompt_completion_jsonl_and_identity(tmp_path: Path) -> None:
    manifest_path, _ = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"

    summary = export(manifest_path, out_dir)

    dataset_path = Path(summary["dataset_path"])
    lines = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == summary["datums"] == 3
    assert all(set(datum) == {"prompt", "completion"} for datum in lines)
    assert all(datum["completion"][0]["role"] == "assistant" for datum in lines)
    assert all(datum["prompt"][-1]["role"] == "user" for datum in lines)
    assert summary["dataset_bytes"] == dataset_path.stat().st_size
    assert len(summary["dataset_sha256"]) == 64

    identity = json.loads((out_dir / "adaption-dataset.json").read_text(encoding="utf-8"))
    assert identity["dataset_sha256"] == summary["dataset_sha256"]
    assert identity["format"] == "adaption_prompt_completion_unrolled_v1"


def test_export_fails_when_the_train_partition_is_missing(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "release_id": "test-release-v99",
                "partitions": {
                    "train": {"path": str(tmp_path / "nope.jsonl"), "sha256": "0" * 64, "row_count": 0}
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        export(manifest_path, tmp_path / "out", dry_run=True)


def test_main_warns_below_the_adaption_datum_floor(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, _ = _write_fixture(tmp_path)

    code = main(["--manifest", str(manifest_path), "--out-dir", str(tmp_path / "out")])

    assert code == 0
    assert f"below the {DATUM_FLOOR} Adaption floor" in capsys.readouterr().out
