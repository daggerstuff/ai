from __future__ import annotations

import json

from ai.core.pipelines.processing.data_normalizer import DataNormalizer
from ai.core.pipelines.processing.normalization_pipeline import (
    DedupStrategy,
    NormalizationPipeline,
)


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_data_normalizer_canonicalizes_record_shape() -> None:
    normalizer = DataNormalizer(enforce_license=True, enforce_phi_scan=True)

    record = normalizer.normalize_record(
        {
            "ID": "sample-1",
            "Source": "synthetic",
            "Content-Type": "conversation",
            "Text": "  Hello\u00a0there.  ",
            "License": "cc-by-4.0",
            "PHI Scan Passed": True,
            "Metadata": {"Topic Tags": ["synthetic"], "QualityScore": 0.9},
        }
    )

    validation = normalizer.validate_record(record)

    assert validation.valid
    assert record["id"] == "sample-1"
    assert record["content_type"] == "conversation"
    assert record["messages"] == [{"role": "user", "content": "Hello there."}]
    assert record["metadata"] == {"topic_tags": ["synthetic"], "quality_score": 0.9}


def test_normalization_pipeline_writes_duplicate_evidence(tmp_path) -> None:
    input_record_count = 2
    input_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "normalized.jsonl"
    reject_path = tmp_path / "rejections.jsonl"
    duplicate_text = "This is a synthetic support dialogue."
    _write_jsonl(
        input_path,
        [
            {
                "id": "low-priority",
                "source": "synthetic",
                "content_type": "conversation",
                "text": duplicate_text,
                "metadata": {"stage": "supplementary"},
            },
            {
                "id": "high-priority",
                "source": "synthetic",
                "content_type": "conversation",
                "text": duplicate_text,
                "metadata": {"stage": "stage1_foundation"},
            },
        ],
    )

    result = NormalizationPipeline(dedup_strategy=DedupStrategy.STAGE_AWARE).run(
        [input_path],
        output_path=output_path,
        reject_path=reject_path,
    )

    output_records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rejection_summary = json.loads(reject_path.read_text(encoding="utf-8"))

    assert result.total_records == input_record_count
    assert result.final_records == 1
    assert result.duplicates_removed == 1
    assert result.duplicate_evidence[0].retained_id == "high-priority"
    assert result.duplicate_evidence[0].duplicate_id == "low-priority"
    assert output_records[0]["conversation_id"] == "high-priority"
    assert rejection_summary["duplicate_evidence"][0]["duplicate_id"] == "low-priority"
