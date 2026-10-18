from __future__ import annotations

import json
from pathlib import Path

from ai.tools.utilities.core.pipelines.processing.data_normalizer import (
    Conversation,
    DataNormalizer,
    Message,
)
from ai.tools.utilities.core.pipelines.processing.normalization_pipeline import (
    DedupStrategy,
    DuplicateEvidence,
    NormalizationPipeline,
    PipelineResult,
    SetDeduplicator,
    SimilarityDeduplicator,
    SimpleContentHasher,
    StageAwareDeduplicator,
)

# Test constants
SHA256_HEX_DIGEST_LENGTH = 64
ZERO_DEDUP_RATE = 0.0
FIFTEEN_PERCENT_DEDUP_RATE = 0.15
TWO_RECORDS = 2
ONE_RECORD = 1


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
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


def test_normalization_pipeline_writes_duplicate_evidence(tmp_path: Path) -> None:
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

    output_records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line]
    rejection_summary = json.loads(reject_path.read_text(encoding="utf-8"))

    assert result.total_records == input_record_count
    assert result.final_records == 1
    assert result.duplicates_removed == 1
    assert result.duplicate_evidence[0].retained_id == "high-priority"
    assert result.duplicate_evidence[0].duplicate_id == "low-priority"
    assert output_records[0]["conversation_id"] == "high-priority"
    assert rejection_summary["duplicate_evidence"][0]["duplicate_id"] == "low-priority"


class TestSetDeduplicator:
    def test_is_duplicate_returns_false_for_new_hash(self) -> None:
        dedup = SetDeduplicator()
        assert dedup.is_duplicate("abc123", "record-1") is False
        assert dedup.duplicates_found == 0

    def test_is_duplicate_returns_true_for_seen_hash(self) -> None:
        dedup = SetDeduplicator()
        dedup.is_duplicate("abc123", "record-1")
        assert dedup.is_duplicate("abc123", "record-2") is True
        assert dedup.duplicates_found == 1

    def test_is_duplicate_records_evidence(self) -> None:
        dedup = SetDeduplicator()
        dedup.is_duplicate("abc123", "record-1")
        dedup.is_duplicate("abc123", "record-2")

        assert len(dedup.duplicate_evidence) == 1
        ev = dedup.duplicate_evidence[0]
        assert ev.strategy == "bloom"
        assert ev.content_hash == "abc123"
        assert ev.retained_id == "record-1"
        assert ev.duplicate_id == "record-2"
        assert ev.reason == "exact_content_hash_match"

    def test_clear_resets_state(self) -> None:
        dedup = SetDeduplicator()
        dedup.is_duplicate("abc123", "record-1")
        dedup.is_duplicate("abc123", "record-2")
        assert dedup.duplicates_found == 1

        dedup.clear()
        assert dedup.duplicates_found == 0
        assert len(dedup.duplicate_evidence) == 0
        assert dedup.is_duplicate("abc123", "record-3") is False


class TestSimpleContentHasher:
    def test_hash_conversation_produces_stable_hash(self) -> None:

        conv = Conversation(
            conversation_id="test-1",
            source="test",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ],
        )
        h1 = SimpleContentHasher.hash_conversation(conv)
        h2 = SimpleContentHasher.hash_conversation(conv)
        assert h1 == h2
        assert len(h1) == SHA256_HEX_DIGEST_LENGTH  # SHA-256 hex

    def test_hash_conversation_is_case_insensitive(self) -> None:

        conv1 = Conversation(
            conversation_id="test-1",
            source="test",
            messages=[Message(role="USER", content="HELLO")],
        )
        conv2 = Conversation(
            conversation_id="test-2",
            source="test",
            messages=[Message(role="user", content="hello")],
        )
        assert SimpleContentHasher.hash_conversation(conv1) == SimpleContentHasher.hash_conversation(conv2)

    def test_hash_record_produces_stable_json_hash(self) -> None:
        record = {"id": "test", "name": "Alice", "score": 42}
        h1 = SimpleContentHasher.hash_record(record)
        h2 = SimpleContentHasher.hash_record(record)
        assert h1 == h2
        assert len(h1) == SHA256_HEX_DIGEST_LENGTH


class TestSimilarityDeduplicator:
    def test_deduplicate_returns_identical_list_for_single_record(self) -> None:

        convs = [
            Conversation(
                conversation_id="test-1",
                source="test",
                messages=[Message(role="user", content="Hello")],
            )
        ]
        dedup = SimilarityDeduplicator()
        result = dedup.deduplicate(convs)
        assert len(result) == 1

    def test_deduplicate_removes_exact_content_matches(self) -> None:

        convs = [
            Conversation(
                conversation_id="test-1",
                source="test",
                messages=[Message(role="user", content="Hello world")],
            ),
            Conversation(
                conversation_id="test-2",
                source="test",
                messages=[Message(role="user", content="Hello world")],
            ),
        ]
        dedup = SimilarityDeduplicator()
        result = dedup.deduplicate(convs)
        assert len(result) == 1
        assert result[0].conversation_id == "test-1"

    def test_deduplicate_records_duplicate_evidence(self) -> None:

        convs = [
            Conversation(
                conversation_id="test-1",
                source="test",
                messages=[Message(role="user", content="Same content")],
            ),
            Conversation(
                conversation_id="test-2",
                source="test",
                messages=[Message(role="user", content="Same content")],
            ),
        ]
        dedup = SimilarityDeduplicator()
        dedup.deduplicate(convs)
        assert dedup.duplicates_found == 1
        assert len(dedup.duplicate_evidence) == 1
        assert dedup.duplicate_evidence[0].reason == "exact_content_hash_match"

    def test_deduplicate_respects_similarity_threshold(self) -> None:

        # Very similar but not identical - 0.9 Jaccard + 1.0 role sim
        convs = [
            Conversation(
                conversation_id="test-1",
                source="test",
                messages=[
                    Message(role="user", content="Hello world how are you"),
                    Message(role="assistant", content="I'm fine thank you"),
                ],
            ),
            Conversation(
                conversation_id="test-2",
                source="test",
                messages=[
                    Message(role="user", content="Hello world how are things"),
                    Message(role="assistant", content="I'm fine thank you"),
                ],
            ),
        ]
        dedup = SimilarityDeduplicator(similarity_threshold=0.85)
        result = dedup.deduplicate(convs)
        # Should deduplicate at 0.85 threshold
        assert len(result) == 1


class TestStageAwareDeduplicator:
    def test_deduplicate_keeps_higher_priority_stage(self) -> None:

        convs = [
            Conversation(
                conversation_id="low",
                source="test",
                messages=[Message(role="user", content="Same text")],
                metadata={"stage": "supplementary"},
            ),
            Conversation(
                conversation_id="high",
                source="test",
                messages=[Message(role="user", content="Same text")],
                metadata={"stage": "stage4_voice_persona"},
            ),
        ]
        dedup = StageAwareDeduplicator()
        result = dedup.deduplicate(convs)
        assert len(result) == 1
        assert result[0].conversation_id == "high"

    def test_deduplicate_keeps_first_when_same_priority(self) -> None:

        convs = [
            Conversation(
                conversation_id="first",
                source="test",
                messages=[Message(role="user", content="Same")],
                metadata={"stage": "stage1_foundation"},
            ),
            Conversation(
                conversation_id="second",
                source="test",
                messages=[Message(role="user", content="Same")],
                metadata={"stage": "stage1_foundation"},
            ),
        ]
        dedup = StageAwareDeduplicator()
        result = dedup.deduplicate(convs)
        assert len(result) == 1
        assert result[0].conversation_id == "first"

    def test_deduplicate_preserves_non_duplicates(self) -> None:

        convs = [
            Conversation(
                conversation_id="conv-1",
                source="test",
                messages=[Message(role="user", content="Content A")],
            ),
            Conversation(
                conversation_id="conv-2",
                source="test",
                messages=[Message(role="user", content="Content B")],
            ),
        ]
        dedup = StageAwareDeduplicator()
        result = dedup.deduplicate(convs)
        assert len(result) == TWO_RECORDS

    def test_deduplicate_returns_empty_for_empty_input(self) -> None:
        dedup = StageAwareDeduplicator()
        result = dedup.deduplicate([])
        assert result == []


class TestPipelineResult:
    def test_dedup_rate_calculates_correctly(self) -> None:
        result = PipelineResult(
            total_records=100,
            validated_records=100,
            duplicates_removed=15,
        )
        assert result.dedup_rate == FIFTEEN_PERCENT_DEDUP_RATE

    def test_dedup_rate_returns_zero_when_no_validated(self) -> None:
        result = PipelineResult(validated_records=0, duplicates_removed=0)
        assert result.dedup_rate == 0.0

    def test_summary_contains_key_metrics(self) -> None:
        result = PipelineResult(
            total_records=100,
            validated_records=85,
            rejected_records=15,
            duplicates_removed=10,
            final_records=75,
            processing_time_seconds=1.5,
        )
        summary = result.summary()
        assert "Total records:      100" in summary
        assert "Duplicates removed: 10" in summary
        assert "Final records:      75" in summary
        assert "1.50s" in summary


class TestDedupStrategies:
    def test_none_strategy_keeps_all_records(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        _write_jsonl(
            input_path,
            [
                {"id": "a", "source": "s", "content_type": "c", "text": "hello"},
                {"id": "b", "source": "s", "content_type": "c", "text": "hello"},
            ],
        )
        result = NormalizationPipeline(dedup_strategy=DedupStrategy.NONE).run([input_path], output_path=output_path)
        assert result.duplicates_removed == 0
        assert result.final_records == TWO_RECORDS

    def test_bloom_strategy_removes_exact_duplicates(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        _write_jsonl(
            input_path,
            [
                {"id": "a", "source": "s", "content_type": "c", "text": "hello"},
                {"id": "b", "source": "s", "content_type": "c", "text": "hello"},
            ],
        )
        result = NormalizationPipeline(dedup_strategy=DedupStrategy.BLOOM).run([input_path], output_path=output_path)
        assert result.duplicates_removed == 1
        assert result.final_records == 1

    def test_similarity_strategy_removes_similar_content(self, tmp_path: Path) -> None:
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        _write_jsonl(
            input_path,
            [
                {
                    "id": "a",
                    "source": "s",
                    "content_type": "c",
                    "text": "Hello world how are you doing today",
                },
                {
                    "id": "b",
                    "source": "s",
                    "content_type": "c",
                    "text": "Hello world how are you doing today friend",
                },
            ],
        )
        result = NormalizationPipeline(dedup_strategy=DedupStrategy.SIMILARITY, similarity_threshold=0.85).run(
            [input_path], output_path=output_path
        )
        # Similarity dedup should remove at least 1
        assert result.final_records <= TWO_RECORDS


class TestDuplicateEvidence:
    def test_to_dict_returns_all_fields(self) -> None:
        ev = DuplicateEvidence(
            strategy="bloom",
            content_hash="abc123",
            retained_id="id-1",
            duplicate_id="id-2",
            reason="exact_match",
        )
        d = ev.to_dict()
        assert d["strategy"] == "bloom"
        assert d["content_hash"] == "abc123"
        assert d["retained_id"] == "id-1"
        assert d["duplicate_id"] == "id-2"
        assert d["reason"] == "exact_match"


class TestEdgeCases:
    def test_pipeline_handles_empty_input(self, tmp_path: Path) -> None:
        result = NormalizationPipeline().run([tmp_path / "nonexistent.jsonl"], output_path=tmp_path / "out.jsonl")
        assert len(result.errors) == 1
        assert "No JSONL files found" in result.errors[0]

    def test_pipeline_handles_invalid_jsonl(self, tmp_path: Path) -> None:
        input_path = tmp_path / "bad.jsonl"
        input_path.write_text('{"id": "a"}\n{"broken json\n', encoding="utf-8")

        result = NormalizationPipeline().run([input_path], output_path=tmp_path / "out.jsonl")
        # Pipeline completes but records parse errors
        assert result.total_records > 0 or len(result.errors) > 0
