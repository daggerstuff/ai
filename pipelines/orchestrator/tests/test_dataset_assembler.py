from __future__ import annotations

import pytest

from ai.pipelines.orchestrator.orchestration.dataset_assembler import DatasetAssembler


def test_dataset_assembler_rejects_oversized_prebalance_materialization():
    assembler = DatasetAssembler(
        enable_bias_detection=False,
        enable_quality_validation=False,
        max_prebalance_records=1,
        balance_dataset=lambda records: (records, {"stage1_foundation": records}),
        run_bias_detection=lambda records: records,
        run_quality_validation=lambda records: records,
        validate_final_stage_balance=lambda records: None,
        finalize_stats=lambda records: None,
        save_dataset=lambda records: "ignored.json",
        write_stage_outputs=lambda segments: None,
        write_split_outputs=lambda records: None,
        run_artifact_service=_RunArtifactServiceStub(),
        checklist_tracker_sync_service=_ChecklistTrackerSyncServiceStub(),
    )

    with pytest.raises(ValueError, match="Pre-balance training corpus exceeds"):
        assembler.assemble(iter([{"text": "a"}, {"text": "b"}]))


class _RunArtifactServiceStub:
    def generate_report(self):
        return {}

    def build_stage_health_report(self, report):
        return {}

    def write_stage_health_report(self, stage_health_report):
        return None

    def build_mtgc_closure_pack(self, report, stage_health_report):
        return {}

    def write_mtgc_closure_pack(self, closure_pack):
        return None


class _ChecklistTrackerSyncServiceStub:
    def sync_run_checklist(self, report):
        return None
