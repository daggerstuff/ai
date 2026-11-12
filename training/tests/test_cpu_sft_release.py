"""The CPU launcher reads the shared export, so its plan, batching and loss are cheap to pin down."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest

from training import cpu_sft_release as cpu


def write_export(tmp_path: Path, rows: list[dict[str, Any]], *, base_model: str = "Qwen/Qwen3-0.6B") -> Path:
    dataset_dir = tmp_path / "export"
    dataset_dir.mkdir()
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (dataset_dir / cpu.DATASET_FILE).write_bytes(payload)
    (dataset_dir / cpu.RECORD_FILE).write_text(
        json.dumps(
            {
                "schema": "tinker-tokenized-datums-v1",
                "base_model": base_model,
                "datums": len(rows),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_release_id": "wandb-full-corpus-current-master-20261110-v22",
            }
        )
    )
    return dataset_dir


def datum(length: int, *, trained: int = 4) -> dict[str, Any]:
    return {
        "input_ids": list(range(length)),
        "target_ids": list(range(1, length + 1)),
        "weights": [1] * trained + [0] * (length - trained),
    }


def test_plan_counts_steps_and_refuses_a_digest_mismatch(tmp_path: Path) -> None:
    dataset_dir = write_export(tmp_path, [datum(6), datum(6), datum(6)])
    plan, rows, _record = cpu.plan_run(dataset_dir, recipe=cpu.Recipe(batch_size=2))
    assert plan.datums == len(rows) == 3
    assert plan.steps == 2
    assert plan.trained_tokens == 12

    (dataset_dir / cpu.DATASET_FILE).write_text(json.dumps(datum(6)) + "\n")
    with pytest.raises(ValueError, match="does not match the recorded"):
        cpu.plan_run(dataset_dir, recipe=cpu.Recipe())


def test_plan_refuses_a_base_model_the_export_was_not_tokenized_with(tmp_path: Path) -> None:
    dataset_dir = write_export(tmp_path, [datum(6)])
    with pytest.raises(ValueError, match="was not the tokenizer behind this export"):
        cpu.plan_run(dataset_dir, base_model="Qwen/Qwen3-8B", recipe=cpu.Recipe())


def test_batches_cover_every_datum_once_and_group_similar_lengths() -> None:
    rows = [datum(4), datum(9), datum(5), datum(40), datum(6)]
    batches = cpu.make_batches(rows, 2, random.Random(0))
    assert sorted(len(batch) for batch in batches) == [1, 2, 2]
    assert sum(len(batch) for batch in batches) == len(rows)
    seen = sorted(row["input_ids"][0] for batch in batches for row in batch)
    assert seen == sorted(row["input_ids"][0] for row in rows)


def test_lr_trapezoid_walks_warmup_hold_and_decay() -> None:
    peak, floor, start = 1e-4, 1e-6, 2e-5
    assert cpu.lr_at(0.0, peak=peak, floor=floor, start=start) == pytest.approx(start)
    assert cpu.lr_at(cpu.WARMUP_FRACTION, peak=peak, floor=floor, start=start) == pytest.approx(peak)
    assert cpu.lr_at(cpu.HOLD_FRACTION, peak=peak, floor=floor, start=start) == pytest.approx(peak)
    assert cpu.lr_at(1.0, peak=peak, floor=floor, start=start) == pytest.approx(floor)


def test_masked_loss_ignores_tokens_with_zero_weight() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros(1, 3, 5)
    logits[0, 0, 0] = 3.0
    logits[0, 1, 1] = -3.0
    targets = torch.tensor([[0, 1, 2]])

    first_token_only = torch.nn.functional.cross_entropy(logits[0, 0:1], targets[0, 0:1]).item()
    everything = cpu.masked_cross_entropy(logits, targets, torch.ones(1, 3), torch).item()
    prompt_only = cpu.masked_cross_entropy(logits, targets, torch.tensor([[1.0, 0.0, 0.0]]), torch).item()

    assert prompt_only == pytest.approx(first_token_only)
    assert prompt_only < everything
    assert cpu.masked_cross_entropy(logits, targets, torch.zeros(1, 3), torch).item() == pytest.approx(0.0)
