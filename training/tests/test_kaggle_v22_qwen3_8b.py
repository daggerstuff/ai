"""The Kaggle kernel runs somewhere we cannot debug, so pin its data path here first."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from training.kaggle_v22_qwen3_8b import train as kernel

EXPORT_DIR = Path(__file__).resolve().parents[1] / "output" / "cpu-v22-qwen06"


def write_export(tmp_path: Path, rows: list[dict[str, Any]], *, tokenizer: str = "Qwen/Qwen3-0.6B") -> Path:
    dataset_dir = tmp_path / "export"
    dataset_dir.mkdir()
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (dataset_dir / kernel.DATASET_FILE).write_bytes(payload)
    (dataset_dir / kernel.RECORD_FILE).write_text(
        json.dumps(
            {
                "schema": "tinker-tokenized-datums-v1",
                "tokenizer": tokenizer,
                "datums": len(rows),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_release_id": "wandb-full-corpus-current-master-20261110-v22",
            }
        )
    )
    return dataset_dir


def datum(length: int, *, trained: int = 3) -> dict[str, Any]:
    """The export trains the assistant tail and masks the prompt, so mirror that."""
    return {
        "input_ids": list(range(length)),
        "target_ids": list(range(1, length + 1)),
        "weights": [0] * (length - trained) + [1] * trained,
    }


def test_verify_export_refuses_a_tampered_file(tmp_path: Path) -> None:
    dataset_dir = write_export(tmp_path, [datum(6), datum(6)])
    verified = kernel.verify_export(dataset_dir)
    assert len(verified["rows"]) == 2

    (dataset_dir / kernel.DATASET_FILE).write_text(json.dumps(datum(6)) + "\n")
    with pytest.raises(ValueError, match="does not match the recorded sha256"):
        kernel.verify_export(dataset_dir)


def test_verify_export_refuses_a_foreign_tokenizer(tmp_path: Path) -> None:
    dataset_dir = write_export(tmp_path, [datum(6)], tokenizer="Qwen/Qwen2.5-7B")
    with pytest.raises(ValueError, match="was not the tokenizer behind this export"):
        kernel.verify_export(dataset_dir)


def test_collator_masks_the_prompt_and_pads_with_ignore_index() -> None:
    rows = [datum(5, trained=2), datum(3, trained=1)]
    batch = kernel.build_collator(pad_id=0)(rows)

    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert batch["labels"].tolist() == [
        [kernel.IGNORE_INDEX, kernel.IGNORE_INDEX, kernel.IGNORE_INDEX, 4, 5],
        [kernel.IGNORE_INDEX, kernel.IGNORE_INDEX, 3, kernel.IGNORE_INDEX, kernel.IGNORE_INDEX],
    ]


def test_lr_schedule_peaks_at_warmup_and_lands_on_the_floor() -> None:
    schedule = kernel.lr_lambda(1000)
    assert schedule(0) < schedule(int(1000 * kernel.WARMUP_FRACTION) - 1) <= 1.0
    assert schedule(int(1000 * kernel.WARMUP_FRACTION) + 1) == pytest.approx(1.0)
    assert schedule(int(1000 * kernel.HOLD_FRACTION) - 1) == pytest.approx(1.0)
    assert schedule(1000) == pytest.approx(kernel.MIN_LR / kernel.LR)


def test_the_real_export_still_matches_its_record() -> None:
    if not (EXPORT_DIR / kernel.RECORD_FILE).exists():
        pytest.skip("the v22 export is not on this machine")

    verified = kernel.verify_export(EXPORT_DIR)
    record = verified["record"]
    assert len(verified["rows"]) == record["datums"]
    assert record["tokenizer"] in kernel.EXPORT_TOKENIZERS
    assert sum(sum(row["weights"]) for row in verified["rows"]) == record["counts"]["trained_tokens"]
