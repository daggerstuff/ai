"""Tests for the PIX-4387 data lineage tracking implementation.

Covers DataLineageTracker (record/query/upstream/downstream traversal,
coverage stats) and the QualityAuditSystem integration: lineage stamping
from the conversations table, measured governance audit records, and the
project-root-relative default paths.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai.qa.reports.quality_audit_system import (
    _PROJECT_ROOT,
    DataLineageTracker,
    LineageRecord,
    QualityAuditSystem,
)

AUDITED_CONVERSATION_COUNT = 4
MENTAL_HEALTH_ROWS = 2
DATASET_NODE_COUNT = 2
TOTAL_LINEAGE_NODES = AUDITED_CONVERSATION_COUNT + DATASET_NODE_COUNT
HALF_COVERAGE = 0.5
FULL_COVERAGE = 1.0
CONVERSATION_STEPS = ["chatml_convert", "stamp_lineage", "quality_filter"]
DOWNSTREAM_AUDITOR = "quality_audit_system"


@pytest.fixture
def tracker(tmp_path: Path) -> DataLineageTracker:
    return DataLineageTracker(db_path=str(tmp_path / "lineage.db"))


def _record(
    artifact_id: str,
    *,
    source: str = "test_source",
    parents: list[str] | None = None,
    consumers: list[str] | None = None,
    steps: list[str] | None = None,
) -> LineageRecord:
    return LineageRecord(
        artifact_id=artifact_id,
        artifact_type="conversation",
        source=source,
        transformation_steps=steps or [],
        downstream_consumers=consumers or [],
        parent_artifact_ids=parents or [],
    )


class TestRecordAndRetrieve:
    def test_record_returns_record_with_stamped_timestamp(self, tracker: DataLineageTracker) -> None:
        before = datetime.now(UTC)
        record = tracker.record(_record("conv-1", steps=["chatml_convert"]))
        after = datetime.now(UTC)
        assert record.created_at is not None
        assert before <= record.created_at <= after

    def test_get_lineage_for_artifact_roundtrip(self, tracker: DataLineageTracker) -> None:
        tracker.record(
            _record(
                "conv-1",
                source="mental_health",
                parents=["dataset:mental_health"],
                consumers=[DOWNSTREAM_AUDITOR],
                steps=CONVERSATION_STEPS,
            )
        )
        lineage = tracker.get_lineage_for_artifact("conv-1")
        assert lineage is not None
        assert lineage.artifact_id == "conv-1"
        assert lineage.source == "mental_health"
        assert lineage.parent_artifact_ids == ["dataset:mental_health"]
        assert lineage.downstream_consumers == [DOWNSTREAM_AUDITOR]
        assert lineage.transformation_steps == CONVERSATION_STEPS

    def test_get_lineage_for_missing_artifact_returns_none(self, tracker: DataLineageTracker) -> None:
        assert tracker.get_lineage_for_artifact("nope") is None

    def test_record_upserts_on_same_artifact_id(self, tracker: DataLineageTracker) -> None:
        tracker.record(_record("conv-1", source="a"))
        tracker.record(_record("conv-1", source="b"))
        lineage = tracker.get_lineage_for_artifact("conv-1")
        assert lineage is not None
        assert lineage.source == "b"


class TestTraversal:
    @pytest.fixture
    def chained_tracker(self, tmp_path: Path) -> DataLineageTracker:
        """raw_upload -> dataset:raw -> conv-1 -> conv-2 (derived)."""
        tracker = DataLineageTracker(db_path=str(tmp_path / "lineage.db"))
        tracker.record(
            LineageRecord(
                artifact_id="raw_upload",
                artifact_type="raw_file",
                source="gdrive",
            )
        )
        tracker.record(
            LineageRecord(
                artifact_id="dataset:raw",
                artifact_type="dataset",
                source="gdrive",
                parent_artifact_ids=["raw_upload"],
                transformation_steps=["ingestion"],
            )
        )
        tracker.record(_record("conv-1", parents=["dataset:raw"], steps=["chatml_convert"]))
        tracker.record(
            LineageRecord(
                artifact_id="conv-2",
                artifact_type="conversation",
                source="test_source",
                parent_artifact_ids=["conv-1"],
                transformation_steps=["dedup"],
            )
        )
        return tracker

    def test_get_upstream_traverses_to_source(self, chained_tracker: DataLineageTracker) -> None:
        upstream_ids = [r.artifact_id for r in chained_tracker.get_upstream("conv-1")]
        assert upstream_ids == ["dataset:raw", "raw_upload"]

    def test_get_upstream_unknown_artifact_is_empty(self, chained_tracker: DataLineageTracker) -> None:
        assert chained_tracker.get_upstream("nope") == []

    def test_get_downstream_traverses_to_consumers(self, chained_tracker: DataLineageTracker) -> None:
        downstream_ids = [r.artifact_id for r in chained_tracker.get_downstream("dataset:raw")]
        assert downstream_ids == ["conv-1", "conv-2"]

    def test_traversal_terminates_on_cycle(self, chained_tracker: DataLineageTracker) -> None:
        # A↔B cycle must not hang the traversal
        chained_tracker.record(_record("a", parents=["b"]))
        chained_tracker.record(_record("b", parents=["a"]))
        assert chained_tracker.get_upstream("a") is not None
        assert chained_tracker.get_downstream("a") is not None


class TestCoverageStats:
    def test_empty_tracker(self, tracker: DataLineageTracker) -> None:
        stats = tracker.lineage_coverage_stats()
        assert stats["tracked_artifacts"] == 0
        assert stats["population"] == 0
        assert stats["coverage"] == 0.0

    def test_population_argument_overrides_registry_count(self, tracker: DataLineageTracker) -> None:
        tracker.record(_record("conv-1"))
        tracker.record(_record("conv-2"))
        stats = tracker.lineage_coverage_stats(
            tracked_artifact_ids={f"conv-{i}" for i in range(1, AUDITED_CONVERSATION_COUNT + 1)}
        )
        assert stats["tracked_artifacts"] == AUDITED_CONVERSATION_COUNT - MENTAL_HEALTH_ROWS
        assert stats["population"] == AUDITED_CONVERSATION_COUNT
        assert stats["coverage"] == HALF_COVERAGE
        assert stats["with_source"] == MENTAL_HEALTH_ROWS
        assert stats["with_transformations"] == 0
        assert stats["with_consumers"] == 0
        assert stats["distinct_sources"] == ["test_source"]

    def test_full_coverage_when_population_matches(self, tracker: DataLineageTracker) -> None:
        tracker.record(_record("conv-1", consumers=[DOWNSTREAM_AUDITOR], steps=["s1"]))
        stats = tracker.lineage_coverage_stats(tracked_artifact_ids={"conv-1"})
        assert stats["coverage"] == FULL_COVERAGE
        assert stats["with_transformations"] == 1
        assert stats["with_consumers"] == 1


class TestQualityAuditSystemIntegration:
    @pytest.fixture
    def audit_db(self, tmp_path: Path) -> Path:
        db_path = tmp_path / "conversations.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                dataset_source TEXT,
                tier TEXT,
                conversations_json TEXT,
                character_count INTEGER,
                word_count INTEGER,
                turn_count INTEGER,
                created_at TIMESTAMP,
                processed_at TIMESTAMP,
                processing_status TEXT,
                language TEXT
            )
            """
        )
        rows = []
        for i in range(1, AUDITED_CONVERSATION_COUNT + 1):
            source = "mental_health" if i <= MENTAL_HEALTH_ROWS else "crisis_intervention"
            rows.append(
                (
                    f"conv-{i}",
                    source,
                    "priority_1",
                    '[{"human": "Hi", "assistant": "Hello"}]',
                    20,
                    4,
                    2,
                    "2026-01-01 10:00:00",
                    "2026-01-01 10:01:00",
                    "processed",
                    "en",
                )
            )
        conn.executemany(
            "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        conn.commit()
        conn.close()
        return db_path

    @pytest.fixture
    def audit_system(self, audit_db: Path) -> QualityAuditSystem:
        return QualityAuditSystem(db_path=str(audit_db))

    def test_default_paths_are_project_root_relative(self) -> None:
        assert Path(__file__).resolve().parents[3] == _PROJECT_ROOT
        assert (_PROJECT_ROOT / "ai").is_dir()

    def test_audit_data_stamps_lineage_for_every_conversation(
        self, audit_system: QualityAuditSystem
    ) -> None:
        quality_data = audit_system._get_quality_audit_data()
        assert quality_data["total_conversations"] == AUDITED_CONVERSATION_COUNT
        stats = quality_data["lineage_stats"]
        assert stats["population"] == AUDITED_CONVERSATION_COUNT
        assert stats["tracked_artifacts"] >= AUDITED_CONVERSATION_COUNT
        assert set(stats["distinct_sources"]) == {"mental_health", "crisis_intervention"}
        for i in range(1, AUDITED_CONVERSATION_COUNT + 1):
            lineage = audit_system.lineage_tracker.get_lineage_for_artifact(f"conversation:conv-{i}")
            assert lineage is not None
            assert lineage.transformation_steps == CONVERSATION_STEPS
            assert DOWNSTREAM_AUDITOR in lineage.downstream_consumers
            assert f"dataset:{lineage.source}" in lineage.parent_artifact_ids

    def test_dataset_nodes_registered_with_ingestion_step(
        self, audit_system: QualityAuditSystem
    ) -> None:
        audit_system._get_quality_audit_data()
        dataset = audit_system.lineage_tracker.get_lineage_for_artifact("dataset:mental_health")
        assert dataset is not None
        assert dataset.artifact_type == "dataset"
        assert dataset.transformation_steps == ["ingestion"]

    def test_get_lineage_for_artifact_returns_conversation(
        self, audit_system: QualityAuditSystem
    ) -> None:
        audit_system._get_quality_audit_data()
        lineage = audit_system.lineage_tracker.get_lineage_for_artifact("conversation:conv-1")
        assert lineage is not None
        upstream = audit_system.lineage_tracker.get_upstream("conversation:conv-1")
        assert [r.artifact_id for r in upstream] == [f"dataset:{lineage.source}"]

    def test_governance_audit_reflects_measured_coverage(
        self, audit_system: QualityAuditSystem
    ) -> None:
        audit_system._get_quality_audit_data()
        governance = audit_system._audit_data_governance()
        lineage_records = [r for r in governance if r.component == "data_governance.lineage"]
        assert len(lineage_records) == 1
        record = lineage_records[0]
        # Full coverage over the audited population — must pass, not warn
        assert record.status == "pass"
        assert record.evidence["lineage_coverage"] == FULL_COVERAGE
        assert record.evidence["tracked_artifacts"] >= TOTAL_LINEAGE_NODES

    def test_governance_audit_fails_with_empty_tracker(
        self, tmp_path: Path, audit_db: Path
    ) -> None:
        # Fresh tracker with nothing registered — coverage 0.0 → fail path
        system = QualityAuditSystem(db_path=str(audit_db))
        empty_tracker = DataLineageTracker(db_path=str(tmp_path / "empty.db"))
        system.lineage_tracker = empty_tracker
        governance = system._audit_data_governance()
        lineage_records = [r for r in governance if r.component == "data_governance.lineage"]
        assert lineage_records[0].status == "fail"
        assert lineage_records[0].evidence["lineage_coverage"] == 0.0
        assert lineage_records[0].risk_level == "high"
