from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai.pipelines.orchestrator.orchestration.asana_progress_service import (
    AsanaProgressSyncService,
)


@dataclass
class _ConfigStub:
    enable_asana_sync: bool = True
    asana_project_gid: str | None = "12345"
    asana_section_gid: str | None = "23456"
    asana_parent_task_gid: str | None = None
    asana_task_gid_output_path: str = ""
    asana_task_key_mapping_output_path: str = ""
    asana_task_transition_output_path: str = ""


@dataclass
class _StatsStub:
    warnings: list[str] = field(default_factory=list)


class _AsanaClientStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]] = []

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
    ) -> Any:
        self.calls.append((method, path, payload, query_params))
        if method == "POST" and path == "/tasks":
            return {"gid": "90001"}
        if method == "GET" and path == "/projects/12345/tasks":
            return [
                {"gid": "111", "name": "MTGC-09 Authenticate Asana"},
                {"gid": "222", "name": "MTGC-10 Resolve mapping"},
                {"gid": "333", "name": "MTGC-12 Fresh ops"},
                {"gid": "444", "name": "MTGC-01 Drift"},
                {"gid": "555", "name": "MTGC-08 Drift followup"},
                {"gid": "666", "name": "MTGC-06 Splits"},
            ]
        return {"gid": "subtask-gid"}

    @staticmethod
    def has_auth_context() -> bool:
        return True


def test_asana_progress_service_creates_task_and_transition_artifacts(tmp_path: Path):
    config = _ConfigStub(
        asana_task_gid_output_path=str(tmp_path / "run_task_gid.txt"),
        asana_task_key_mapping_output_path=str(tmp_path / "task_key_map.json"),
        asana_task_transition_output_path=str(tmp_path / "transition_results.json"),
    )
    stats = _StatsStub()
    client = _AsanaClientStub()
    service = AsanaProgressSyncService(config=config, stats=stats, asana_client=client)

    checklist = {
        "generated_at": "2026-03-27T16:00:00+00:00",
        "total_samples": 12,
        "stage_drift_within_tolerance": True,
        "stage_drift_failures": [],
        "split_counts": {"aggregate": {"train": 8, "val": 2, "test": 2}},
        "ops_freshness": {"all_fresh": True},
        "report": {
            "stage_balance": {
                "stage1_foundation": {"final_actual": 7, "drift_vs_target": 0.01},
                "stage4_voice_persona": {"final_actual": 5, "drift_vs_target": 0.0},
            }
        },
    }

    service.sync_checklist_task(checklist, tmp_path / "training_run_checklist.json")

    assert Path(config.asana_task_gid_output_path).read_text(encoding="utf-8") == "90001"
    task_key_map = json.loads(
        Path(config.asana_task_key_mapping_output_path).read_text(encoding="utf-8")
    )
    assert task_key_map["MTGC-10"] == "222"

    transition_results = json.loads(
        Path(config.asana_task_transition_output_path).read_text(encoding="utf-8")
    )
    assert transition_results["updates"]["MTGC-09"]["updated"] is True
    assert transition_results["updates"]["MTGC-10"]["updated"] is True
    assert transition_results["updates"]["MTGC-12"]["updated"] is True
    assert stats.warnings == []


def test_asana_progress_service_warns_on_invalid_project_gid(tmp_path: Path):
    config = _ConfigStub(
        asana_project_gid="invalid",
        asana_task_gid_output_path=str(tmp_path / "run_task_gid.txt"),
        asana_task_key_mapping_output_path=str(tmp_path / "task_key_map.json"),
        asana_task_transition_output_path=str(tmp_path / "transition_results.json"),
    )
    stats = _StatsStub()
    client = _AsanaClientStub()
    service = AsanaProgressSyncService(config=config, stats=stats, asana_client=client)

    service.sync_checklist_task({"generated_at": "2026-03-27T16:00:00+00:00"}, tmp_path / "checklist.json")

    assert any("ASANA_PROJECT_GID" in warning for warning in stats.warnings)
    assert not Path(config.asana_task_gid_output_path).exists()
