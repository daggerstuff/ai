"""Tests for stage-aware ingestion deduplication (PIX-4192)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ai.pipelines.ingestion_deduplication import (
    STAGE_PRIORITY,
    compute_primary_hash,
    compute_secondary_hash,
    deduplicate_records,
    get_stage_priority,
    process_jsonl_file,
)


class TestPrimaryHash:
    """Test primary hash computation: sha256(lowercase(concat(messages.role + messages.content)))."""

    def test_primary_hash_basic(self):
        """Primary hash matches spec for simple message."""
        record = {"messages": [{"role": "user", "content": "Hello"}]}
        expected = hashlib.sha256("userhello".encode("utf-8")).hexdigest()
        assert compute_primary_hash(record) == expected

    def test_primary_hash_multiple_messages(self):
        """Primary hash concatenates all messages."""
        record = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }
        expected = hashlib.sha256("userhiassistanthello".encode("utf-8")).hexdigest()
        assert compute_primary_hash(record) == expected

    def test_primary_hash_lowercase(self):
        """Primary hash normalizes to lowercase."""
        record1 = {"messages": [{"role": "User", "content": "HELLO"}]}
        record2 = {"messages": [{"role": "user", "content": "hello"}]}
        assert compute_primary_hash(record1) == compute_primary_hash(record2)

    def test_primary_hash_empty_messages(self):
        """Primary hash handles empty messages."""
        record = {"messages": []}
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_primary_hash(record) == expected

    def test_primary_hash_missing_messages(self):
        """Primary hash handles missing messages field."""
        record = {}
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_primary_hash(record) == expected


class TestSecondaryHash:
    """Test secondary hash computation: sha1(conversation_id + stage + source + crisis_intensity)."""

    def test_secondary_hash_basic(self):
        """Secondary hash matches spec."""
        record = {
            "metadata": {
                "conversation_id": "conv-123",
                "stage": "stage1_foundation",
                "source": "therapy_data",
                "crisis_intensity": "low",
            }
        }
        expected = hashlib.sha1("conv-123stage1_foundationtherapy_datalow".encode("utf-8")).hexdigest()
        assert compute_secondary_hash(record) == expected

    def test_secondary_hash_missing_fields(self):
        """Secondary hash handles missing metadata fields."""
        record = {"metadata": {"conversation_id": "conv-456"}}
        expected = hashlib.sha1("conv-456".encode("utf-8")).hexdigest()
        assert compute_secondary_hash(record) == expected

    def test_secondary_hash_empty_metadata(self):
        """Secondary hash handles empty metadata."""
        record = {"metadata": {}}
        expected = hashlib.sha1("".encode("utf-8")).hexdigest()
        assert compute_secondary_hash(record) == expected


class TestStagePriority:
    """Test stage priority hierarchy."""

    def test_stage_priority_order(self):
        """Stage priority follows spec: stage4 > stage3 > stage2 > stage1 > supplementary."""
        assert STAGE_PRIORITY["stage4_voice_persona"] > STAGE_PRIORITY["stage3_edge_stress_test"]
        assert STAGE_PRIORITY["stage3_edge_stress_test"] > STAGE_PRIORITY["stage2_therapeutic_expertise"]
        assert STAGE_PRIORITY["stage2_therapeutic_expertise"] > STAGE_PRIORITY["stage1_foundation"]
        assert STAGE_PRIORITY["stage1_foundation"] > STAGE_PRIORITY["supplementary"]

    def test_get_stage_priority(self):
        """get_stage_priority returns correct numeric values."""
        assert get_stage_priority({"metadata": {"stage": "stage4_voice_persona"}}) == 5
        assert get_stage_priority({"metadata": {"stage": "stage3_edge_stress_test"}}) == 4
        assert get_stage_priority({"metadata": {"stage": "stage2_therapeutic_expertise"}}) == 3
        assert get_stage_priority({"metadata": {"stage": "stage1_foundation"}}) == 2
        assert get_stage_priority({"metadata": {"stage": "supplementary"}}) == 1

    def test_get_stage_priority_unknown(self):
        """Unknown stage defaults to supplementary priority (1)."""
        assert get_stage_priority({"metadata": {"stage": "unknown_stage"}}) == 1
        assert get_stage_priority({}) == 1


class TestDeduplication:
    """Test stage-aware deduplication."""

    def test_exact_duplicate_same_stage(self):
        """Exact duplicates within same stage are removed."""
        records = [
            {"messages": [{"role": "user", "content": "test"}], "metadata": {"stage": "stage1_foundation"}},
            {"messages": [{"role": "user", "content": "test"}], "metadata": {"stage": "stage1_foundation"}},
        ]
        deduped, stats = deduplicate_records(records)
        assert len(deduped) == 1
        assert stats.duplicates_removed == 1
        assert stats.stage_conflicts_resolved == 0

    def test_stage_conflict_higher_priority_wins(self):
        """Higher priority stage survives collision."""
        records = [
            {"messages": [{"role": "user", "content": "test"}], "metadata": {"stage": "stage1_foundation"}},
            {"messages": [{"role": "user", "content": "test"}], "metadata": {"stage": "stage4_voice_persona"}},
        ]
        deduped, stats = deduplicate_records(records)
        assert len(deduped) == 1
        assert deduped[0]["metadata"]["stage"] == "stage4_voice_persona"
        assert stats.stage_conflicts_resolved == 1

    def test_stage_conflict_all_priorities(self):
        """Highest priority stage wins across all stages."""
        records = [
            {"messages": [{"role": "user", "content": "same"}], "metadata": {"stage": "supplementary"}},
            {"messages": [{"role": "user", "content": "same"}], "metadata": {"stage": "stage1_foundation"}},
            {"messages": [{"role": "user", "content": "same"}], "metadata": {"stage": "stage2_therapeutic_expertise"}},
            {"messages": [{"role": "user", "content": "same"}], "metadata": {"stage": "stage3_edge_stress_test"}},
            {"messages": [{"role": "user", "content": "same"}], "metadata": {"stage": "stage4_voice_persona"}},
        ]
        deduped, stats = deduplicate_records(records)
        assert len(deduped) == 1
        assert deduped[0]["metadata"]["stage"] == "stage4_voice_persona"
        assert stats.stage_conflicts_resolved == 4

    def test_unique_records_preserved(self):
        """Unique records are all preserved."""
        records = [
            {"messages": [{"role": "user", "content": "a"}], "metadata": {"stage": "stage1_foundation"}},
            {"messages": [{"role": "user", "content": "b"}], "metadata": {"stage": "stage1_foundation"}},
            {"messages": [{"role": "user", "content": "c"}], "metadata": {"stage": "stage2_therapeutic_expertise"}},
        ]
        deduped, stats = deduplicate_records(records)
        assert len(deduped) == 3
        assert stats.duplicates_removed == 0

    def test_dedup_loss_under_one_percent(self):
        """Deduplication loss is <1% for typical dataset."""
        records = []
        for i in range(1000):
            records.append(
                {"messages": [{"role": "user", "content": f"unique_{i}"}], "metadata": {"stage": "stage1_foundation"}}
            )
        for i in range(5):
            records.append(
                {
                    "messages": [{"role": "user", "content": f"unique_{i}"}],
                    "metadata": {"stage": "stage2_therapeutic_expertise"},
                }
            )
        deduped, stats = deduplicate_records(records)
        loss_rate = (stats.total_records - stats.unique_records) / stats.total_records
        assert loss_rate < 0.01, f"Dedup loss {loss_rate:.2%} exceeds 1% target"

    def test_secondary_hash_dedup(self):
        """Secondary hash deduplication uses metadata."""
        records = [
            {
                "messages": [{"role": "user", "content": "a"}],
                "metadata": {"conversation_id": "conv-1", "stage": "stage1_foundation"},
            },
            {
                "messages": [{"role": "user", "content": "b"}],
                "metadata": {"conversation_id": "conv-1", "stage": "stage1_foundation"},
            },
            {
                "messages": [{"role": "user", "content": "c"}],
                "metadata": {"conversation_id": "conv-2", "stage": "stage1_foundation"},
            },
        ]
        deduped, stats = deduplicate_records(records, use_secondary_hash=True)
        assert len(deduped) == 2
        assert stats.duplicates_removed == 1


class TestProcessJsonlFile:
    """Test JSONL file processing."""

    def test_process_jsonl_file(self, tmp_path):
        """Process JSONL file end-to-end."""
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"

        records = [
            {"messages": [{"role": "user", "content": "test"}], "metadata": {"stage": "stage1_foundation"}},
            {"messages": [{"role": "user", "content": "test"}], "metadata": {"stage": "stage4_voice_persona"}},
            {
                "messages": [{"role": "user", "content": "unique"}],
                "metadata": {"stage": "stage2_therapeutic_expertise"},
            },
        ]
        with open(input_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        stats = process_jsonl_file(input_path, output_path)

        assert stats.total_records == 3
        assert stats.unique_records == 2
        assert stats.stage_conflicts_resolved == 1

        with open(output_path) as f:
            output_records = [json.loads(line) for line in f]
        assert len(output_records) == 2
        stages = {r["metadata"]["stage"] for r in output_records}
        assert "stage4_voice_persona" in stages
        assert "stage2_therapeutic_expertise" in stages
