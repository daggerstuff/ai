from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol


class CommandResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


RunCommand = Callable[[list[str]], CommandResult]
AsanaSync = Callable[[dict[str, Any], Path], None]


def map_tracker_status(status: str) -> str:
    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {
        "done",
        "closed",
        "complete",
        "completed",
        "resolved",
        "finished",
    }:
        return "closed"
    if normalized in {"in_progress", "doing", "active", "started", "inflight"}:
        return "in_progress"
    if normalized in {"blocked", "blocked_on", "waiting"}:
        return "blocked"
    return "open"


@dataclass(frozen=True)
class TrackerSyncEvent:
    source: str
    source_id: str
    title: str
    body: str
    status: str
    checklist_path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class TrackerSyncSummary:
    sync_key: str
    source: str
    source_id: str
    success: bool
    beads_issue_id: str | None = None
    jira_synced: bool = False
    linear_synced: bool = False
    asana_synced: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class TrackerSyncCoordinator:
    def __init__(
        self,
        *,
        state_path: Path,
        run_command: RunCommand | None = None,
        asana_sync: AsanaSync | None = None,
        enable_beads_sync: bool = True,
        enable_jira_sync: bool = True,
        enable_linear_sync: bool = True,
    ) -> None:
        self.state_path = state_path
        self.run_command = run_command or self._default_run_command
        self.asana_sync = asana_sync
        self.enable_beads_sync = enable_beads_sync
        self.enable_jira_sync = enable_jira_sync
        self.enable_linear_sync = enable_linear_sync

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> TrackerSyncCoordinator:
        config = pipeline.config
        return cls(
            state_path=Path(config.tracker_sync_state_output_path),
            asana_sync=pipeline._sync_to_asana,
            enable_beads_sync=getattr(config, "enable_beads_sync", True),
            enable_jira_sync=getattr(config, "enable_jira_sync", True),
            enable_linear_sync=getattr(config, "enable_linear_sync", True),
        )

    def sync(self, event: TrackerSyncEvent) -> TrackerSyncSummary:
        sync_key = self._sync_key(event)
        state = self._load_state()
        records = state.setdefault("records", {})
        record = dict(records.get(sync_key, {}))
        record.update(
            {
                "source": event.source,
                "source_id": event.source_id,
                "title": event.title,
                "body": event.body,
                "status": event.status,
                "checklist_path": str(event.checklist_path),
                "payload": event.payload,
                "updated_at": self._utc_now(),
            }
        )

        warnings: list[str] = []
        errors: list[str] = []
        beads_issue_id = record.get("beads_issue_id")

        if self.enable_beads_sync:
            beads_issue_id, bead_warnings, bead_errors = self._sync_beads_issue(
                sync_key=sync_key,
                record=record,
            )
            warnings.extend(bead_warnings)
            errors.extend(bead_errors)
            if beads_issue_id:
                record["beads_issue_id"] = beads_issue_id

        jira_synced = False
        if self.enable_jira_sync:
            jira_result, jira_warnings, jira_errors = self._run_tracker_command(
                ["bd", "jira", "sync"],
                "Jira",
            )
            jira_synced = jira_result is not None
            warnings.extend(jira_warnings)
            errors.extend(jira_errors)

        linear_synced = False
        if self.enable_linear_sync:
            linear_result, linear_warnings, linear_errors = self._run_tracker_command(
                ["bd", "linear", "sync"],
                "Linear",
            )
            linear_synced = linear_result is not None
            warnings.extend(linear_warnings)
            errors.extend(linear_errors)

        asana_synced = False
        if self.asana_sync is not None:
            try:
                self.asana_sync(event.payload, event.checklist_path)
                asana_synced = True
            except (
                Exception
            ) as exc:  # pragma: no cover - safety net for runtime integration
                errors.append(f"Asana sync failed: {exc}")

        record["last_synced_at"] = self._utc_now()
        record["jira_synced"] = jira_synced
        record["linear_synced"] = linear_synced
        record["asana_synced"] = asana_synced
        if beads_issue_id:
            record["beads_issue_id"] = beads_issue_id
        records[sync_key] = record
        state["last_synced_at"] = record["last_synced_at"]
        self._write_state(state)

        success = not errors
        return TrackerSyncSummary(
            sync_key=sync_key,
            source=event.source,
            source_id=event.source_id,
            success=success,
            beads_issue_id=beads_issue_id,
            jira_synced=jira_synced,
            linear_synced=linear_synced,
            asana_synced=asana_synced,
            warnings=warnings,
            errors=errors,
        )

    def _sync_beads_issue(
        self,
        *,
        sync_key: str,
        record: dict[str, Any],
    ) -> tuple[str | None, list[str], list[str]]:
        warnings: list[str] = []
        errors: list[str] = []
        beads_issue_id = record.get("beads_issue_id")
        status = map_tracker_status(str(record.get("status", "open")))

        if beads_issue_id:
            command = [
                "bd",
                "update",
                str(beads_issue_id),
                "--title",
                str(record.get("title", "Tracker sync")),
                "--description",
                str(record.get("body", "")),
                "--status",
                status,
                "--external-ref",
                sync_key,
                "--notes",
                self._build_notes(record),
            ]
            result, update_warnings, update_errors = self._run_tracker_command(
                command,
                "beads update",
            )
            warnings.extend(update_warnings)
            errors.extend(update_errors)
            if result is None:
                return beads_issue_id, warnings, errors
            return beads_issue_id, warnings, errors

        command = [
            "bd",
            "create",
            str(record.get("title", "Tracker sync")),
            "--description",
            str(record.get("body", "")),
            "--status",
            status,
            "--external-ref",
            sync_key,
            "--notes",
            self._build_notes(record),
            "--silent",
        ]
        result, create_warnings, create_errors = self._run_tracker_command(
            command,
            "beads create",
        )
        warnings.extend(create_warnings)
        errors.extend(create_errors)
        if result is None:
            return None, warnings, errors

        created_id = self._parse_issue_id(getattr(result, "stdout", ""))
        if not created_id:
            errors.append("beads create did not return an issue id")
        return created_id, warnings, errors

    def _run_tracker_command(
        self,
        command: list[str],
        label: str,
    ) -> tuple[CommandResult | None, list[str], list[str]]:
        warnings: list[str] = []
        errors: list[str] = []
        try:
            result = self.run_command(command)
        except Exception as exc:
            errors.append(f"{label} command failed: {exc}")
            return None, warnings, errors

        if getattr(result, "returncode", 1) != 0:
            stderr = str(getattr(result, "stderr", "")).strip()
            errors.append(
                f"{label} returned non-zero exit status: {stderr or result.returncode}"
            )
            return None, warnings, errors

        return result, warnings, errors

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"records": {}}

        try:
            with open(self.state_path, encoding="utf-8") as handle:
                state = json.load(handle)
        except Exception:
            return {"records": {}}

        if not isinstance(state, dict):
            return {"records": {}}
        if not isinstance(state.get("records"), dict):
            state["records"] = {}
        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

    @staticmethod
    def _parse_issue_id(output: object) -> str | None:
        if output is None:
            return None
        text = str(output).strip()
        if not text:
            return None
        return text.splitlines()[0].strip()

    @staticmethod
    def _build_notes(record: dict[str, Any]) -> str:
        notes = [
            f"Source: {record.get('source', '')}",
            f"Source ID: {record.get('source_id', '')}",
            f"Status: {record.get('status', '')}",
            f"Checklist path: {record.get('checklist_path', '')}",
        ]
        payload = record.get("payload")
        if isinstance(payload, dict) and payload:
            notes.append("Payload: " + json.dumps(payload, sort_keys=True))
        return "\n".join(notes)

    @staticmethod
    def _sync_key(event: TrackerSyncEvent) -> str:
        return f"{event.source}:{event.source_id}"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _default_run_command(command: list[str]) -> CommandResult:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
