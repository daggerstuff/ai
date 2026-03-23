from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ai.pipelines.orchestrator.orchestration.tracker_sync import (
    TrackerSyncCoordinator,
    TrackerSyncEvent,
    map_tracker_status,
)


def _completed_process(stdout: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


def test_sync_checklist_creates_beads_issue_and_pushes_other_trackers(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    asana_calls: list[tuple[dict[str, object], Path]] = []

    def fake_runner(command: list[str]) -> SimpleNamespace:
        commands.append(command)
        if command[:2] == ["bd", "create"]:
            return _completed_process("bd-123\n")
        return _completed_process()

    def fake_asana_sync(payload: dict[str, object], checklist_path: Path) -> None:
        asana_calls.append((payload, checklist_path))

    coordinator = TrackerSyncCoordinator(
        state_path=tmp_path / "tracker-sync-state.json",
        run_command=fake_runner,
        asana_sync=fake_asana_sync,
    )

    event = TrackerSyncEvent(
        source="pipeline",
        source_id="2026-03-23T10:11:12Z",
        title="Training checklist",
        body="Checklist payload",
        status="done",
        checklist_path=tmp_path / "training_run_checklist.json",
        payload={"generated_at": "2026-03-23T10:11:12Z"},
    )

    summary = coordinator.sync(event)

    assert summary.success is True
    assert summary.beads_issue_id == "bd-123"
    assert asana_calls == [
        (
            {"generated_at": "2026-03-23T10:11:12Z"},
            tmp_path / "training_run_checklist.json",
        )
    ]
    assert any(command[:2] == ["bd", "create"] for command in commands)
    assert any(command[:3] == ["bd", "jira", "sync"] for command in commands)
    assert any(command[:3] == ["bd", "linear", "sync"] for command in commands)

    state = json.loads((tmp_path / "tracker-sync-state.json").read_text())
    assert state["records"][summary.sync_key]["beads_issue_id"] == "bd-123"
    assert state["records"][summary.sync_key]["source"] == "pipeline"


def test_sync_checklist_updates_existing_beads_issue(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_runner(command: list[str]) -> SimpleNamespace:
        commands.append(command)
        return _completed_process()

    state_path = tmp_path / "tracker-sync-state.json"
    state_path.write_text(
        json.dumps(
            {
                "records": {
                    "pipeline:2026-03-23T10:11:12Z": {
                        "source": "pipeline",
                        "source_id": "2026-03-23T10:11:12Z",
                        "title": "Training checklist",
                        "body": "Checklist payload",
                        "status": "open",
                        "beads_issue_id": "bd-999",
                        "checklist_path": str(tmp_path / "training_run_checklist.json"),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    coordinator = TrackerSyncCoordinator(
        state_path=state_path,
        run_command=fake_runner,
        asana_sync=None,
    )

    event = TrackerSyncEvent(
        source="pipeline",
        source_id="2026-03-23T10:11:12Z",
        title="Training checklist",
        body="Checklist payload",
        status="closed",
        checklist_path=tmp_path / "training_run_checklist.json",
        payload={"generated_at": "2026-03-23T10:11:12Z"},
    )

    summary = coordinator.sync(event)

    assert summary.success is True
    assert summary.beads_issue_id == "bd-999"
    assert any(command[:2] == ["bd", "update"] for command in commands)
    assert not any(command[:2] == ["bd", "create"] for command in commands)
    update_command = next(
        command for command in commands if command[:2] == ["bd", "update"]
    )
    assert "--status" in update_command
    assert update_command[update_command.index("--status") + 1] == "closed"


def test_map_tracker_status_handles_known_states() -> None:
    assert map_tracker_status("open") == "open"
    assert map_tracker_status("in_progress") == "in_progress"
    assert map_tracker_status("done") == "closed"
    assert map_tracker_status("unknown") == "open"
