"""Integration tests for ``curate_pipeline.run_pipeline`` Stage 1 wiring (PIX-4342).

Exercises the live ``run_pipeline`` loop on a tiny synthetic JSONL input to
verify Stage 1 filters (language, PII, near-dup) are invoked in the right
order and that PII-redacted transformed text propagates into the ChatML
output shard.
"""

from __future__ import annotations

import json
from pathlib import Path

from training.curate_pipeline import run_pipeline


def _write_input(tmp_path: Path, records: list[dict]) -> Path:
    input_path = tmp_path / "input.jsonl"
    with open(input_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return input_path


def _read_shard(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def test_run_pipeline_filters_non_english(tmp_path: Path):
    records = [
        # Valid English — should be kept.
        {
            "source": "test",
            "task_type": "sft",
            "messages": [
                {"role": "user", "content": "I feel sad today and need someone to talk to."},
                {"role": "assistant", "content": "I am here for you. Tell me more about what you feel."},
            ],
        },
        # Heavy CJK — should be dropped by language filter.
        {
            "source": "test",
            "task_type": "sft",
            "messages": [
                {"role": "user", "content": "你好你好你好你好你好你好你好你好你好你好"},
            ],
        },
    ]
    input_path = _write_input(tmp_path, records)
    output_dir = tmp_path / "out"
    stats = run_pipeline(str(input_path), str(output_dir), dry_run=True,
                        dedup_store_path=str(tmp_path / "dedup.db"))
    assert stats.total_read == 2
    # One record excluded (non-English), one included.
    assert stats.total_excluded >= 1
    assert stats.total_included >= 1


def test_run_pipeline_dedup_skips_exact_repeat(tmp_path: Path):
    duplicate = {
        "source": "test",
        "task_type": "sft",
        "messages": [
            {"role": "user", "content": "Identical message content here for dedup testing purposes."},
            {"role": "assistant", "content": "Acknowledged, let us proceed with the session today."},
        ],
    }
    records = [duplicate, duplicate]
    input_path = _write_input(tmp_path, records)
    output_dir = tmp_path / "out"
    stats = run_pipeline(str(input_path), str(output_dir), dry_run=True,
                        dedup_store_path=str(tmp_path / "dedup.db"))
    assert stats.total_read == 2
    assert stats.total_deduped >= 1


def test_run_pipeline_pii_redaction_writes_transformed(tmp_path: Path):
    record = {
        "source": "test",
        "task_type": "sft",
        "messages": [
            {"role": "user", "content": "Please email me at patient@example.com about my depression."},
            {"role": "assistant", "content": "I will follow up with you there."},
        ],
    }
    input_path = _write_input(tmp_path, [record])
    output_dir = tmp_path / "out"
    run_pipeline(str(input_path), str(output_dir), dry_run=False,
                 dedup_store_path=str(tmp_path / "dedup.db"))
    chatml_train = output_dir / "sft_chatml" / "train.jsonl"
    chatml_val = output_dir / "sft_chatml" / "val.jsonl"
    chatml_test = output_dir / "sft_chatml" / "test.jsonl"
    kept = _read_shard(chatml_train) + _read_shard(chatml_val) + _read_shard(chatml_test)
    assert len(kept) >= 1
    written_blob = json.dumps(kept)
    assert "patient@example.com" not in written_blob
    assert "[REDACTED]" in written_blob
