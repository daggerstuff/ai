#!/usr/bin/env python3
"""Tests for Linear backlog dispatch behavior."""

import time

from ai.qa.reports.linear_backlog_action_builder import build_linear_backlog_payload
from ai.qa.reports.linear_backlog_dispatcher import LinearBacklogDispatcher
from ai.qa.reports.performance_gap_backlog_converter import (
    BacklogChange,
    BacklogConversionResult,
    RulePriority,
)


def test_dispatch_queues_without_linear_credentials(monkeypatch, tmp_path):
    payload = build_linear_backlog_payload(
        BacklogConversionResult(
            generated_at="2026-05-13T00:00:00+00:00",
            metric_count=1,
            generated_changes=1,
            changes=[
                BacklogChange(
                    change_id="rule:one",
                    priority=RulePriority.HIGH,
                    area="acquisition",
                    title="Prioritize clinical sources",
                    summary="Action summary",
                    trigger="clinical_reasoning_accuracy=70.0 (< 85.0)",
                    actions=["Queue source review"],
                    suggested_sop="Run pilot",
                    expected_impact="Increase score",
                    evidence={"metric": "clinical_reasoning_accuracy"},
                )
            ],
        )
    )

    queue_path = tmp_path / "queue.jsonl"
    monkeypatch.delenv("LINEAR_API_KEY_TEST", raising=False)
    dispatcher = LinearBacklogDispatcher(
        queue_path=str(queue_path),
        linear_api_key_env="LINEAR_API_KEY_TEST",
    )
    result = dispatcher.dispatch_backlog_actions(payload)

    assert result["mode"] == "queue_only"
    assert result["queued"] == 1
    assert result["failed"] == 0
    assert queue_path.exists()
    lines = queue_path.read_text().splitlines()
    assert len(lines) == 1


def test_dispatch_creates_items_when_api_called(monkeypatch, tmp_path):
    payload = build_linear_backlog_payload(
        BacklogConversionResult(
            generated_at="2026-05-13T00:00:00+00:00",
            metric_count=1,
            generated_changes=1,
            changes=[
                BacklogChange(
                    change_id="rule:two",
                    priority=RulePriority.CRITICAL,
                    area="review_focus",
                    title="Escalate safety review",
                    summary="Safety rule summary",
                    trigger="safety_score=88.0 (< 90.0)",
                    actions=["Add second reviewer"],
                    suggested_sop="Create hotfix lane",
                    expected_impact="Reduce incidents",
                    evidence={"metric": "safety_score"},
                )
            ],
        )
    )

    monkeypatch.setenv("LINEAR_API_KEY_TEST", "token")
    queue_path = tmp_path / "queue.jsonl"
    dispatcher = LinearBacklogDispatcher(
        queue_path=str(queue_path),
        linear_api_key_env="LINEAR_API_KEY_TEST",
    )

    class DummyResponse:
        def __init__(self):
            self._body = b'{"data": {"issueCreate": {"success": true, "issue": {"id": "ISSUE-1", "title": "Escalate safety review"}}}}'

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    def fake_urlopen(request_obj, timeout=12):
        assert b"issueCreate" in request_obj.data
        return DummyResponse()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = dispatcher.dispatch_backlog_actions(payload)
    assert result["mode"] == "create"
    assert result["created"] == 1
    assert result["failed"] == 0
    assert result["queued"] == 0


def test_dispatch_updates_existing_issue_when_cached_state_present(monkeypatch, tmp_path):
    payload = build_linear_backlog_payload(
        BacklogConversionResult(
            generated_at="2026-05-13T00:00:00+00:00",
            metric_count=1,
            generated_changes=1,
            changes=[
                BacklogChange(
                    change_id="rule:update-1",
                    priority=RulePriority.HIGH,
                    area="acquisition",
                    title="Retune curation priorities",
                    summary="Retune",
                    trigger="validation_gap=45.0 (> 30.0)",
                    actions=["Update existing action"],
                    suggested_sop="Retune pilot",
                    expected_impact="Reduce backlog lag",
                    evidence={"metric": "validation_gap"},
                )
            ],
        )
    )

    monkeypatch.setenv("LINEAR_API_KEY_TEST", "token")
    queue_path = tmp_path / "queue.jsonl"
    state_path = tmp_path / "state.json"
    state_path.write_text('{"rule:update-1": "ISSUE-OLD"}', encoding="utf-8")

    dispatcher = LinearBacklogDispatcher(
        queue_path=str(queue_path),
        linear_api_key_env="LINEAR_API_KEY_TEST",
        issue_state_path=str(state_path),
    )

    class DummyResponse:
        def __init__(self):
            self._body = (
                b'{ "data": { "issueUpdate": { "success": true, '
                b'"issue": { "id": "ISSUE-NEW", "title": "Retune curation priorities" }}}}'
            )

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    seen_payloads = []

    def fake_urlopen(request_obj, timeout=12):
        payload_text = request_obj.data.decode("utf-8")
        seen_payloads.append(payload_text)
        parsed = __import__("json").loads(payload_text)
        assert "issueUpdate" in parsed["query"]
        assert parsed["variables"]["input"]["id"] == "ISSUE-OLD"
        return DummyResponse()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = dispatcher.dispatch_backlog_actions(payload)
    assert result["mode"] == "create"
    assert result["updated"] == 1
    assert result["created"] == 0
    assert result["failed"] == 0
    assert result["queued"] == 0
    if queue_path.exists():
        assert queue_path.read_text() == ""
    state = __import__("json").loads(state_path.read_text(encoding="utf-8"))
    assert state["rule:update-1"] == "ISSUE-NEW"
    assert len(seen_payloads) == 1


def test_dispatch_includes_configured_linear_ids(monkeypatch, tmp_path):
    payload = build_linear_backlog_payload(
        BacklogConversionResult(
            generated_at="2026-05-13T00:00:00+00:00",
            metric_count=1,
            generated_changes=1,
            changes=[
                BacklogChange(
                    change_id="rule:ids-1",
                    priority=RulePriority.LOW,
                    area="review_focus",
                    title="Apply identity check",
                    summary="Identity checks",
                    trigger="safety_score=85.0 (< 90.0)",
                    actions=["Apply parent linkage"],
                    suggested_sop="Parent lane check",
                    expected_impact="Improve traceability",
                    evidence={"metric": "safety_score"},
                )
            ],
        )
    )

    monkeypatch.setenv("LINEAR_API_KEY_TEST", "token")
    monkeypatch.setenv("LINEAR_TEAM_ID", "TEAM-123")
    monkeypatch.setenv("LINEAR_PROJECT_ID", "PRJ-456")
    monkeypatch.setenv("LINEAR_PARENT_ISSUE_ID", "ISS-789")

    queue_path = tmp_path / "queue.jsonl"
    dispatcher = LinearBacklogDispatcher(
        queue_path=str(queue_path),
        linear_api_key_env="LINEAR_API_KEY_TEST",
        linear_team_id_env="LINEAR_TEAM_ID",
        linear_project_id_env="LINEAR_PROJECT_ID",
        linear_parent_issue_id_env="LINEAR_PARENT_ISSUE_ID",
    )

    class DummyResponse:
        def __init__(self):
            self._body = (
                b'{ "data": { "issueCreate": { "success": true, '
                b'"issue": { "id": "ISSUE-42", "title": "Apply identity check" }}}}'
            )

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    captured = {}

    def fake_urlopen(request_obj, timeout=12):
        parsed = __import__("json").loads(request_obj.data.decode("utf-8"))
        variables = parsed["variables"]["input"]
        captured.update(
            team_id=variables.get("teamId"),
            project_id=variables.get("projectId"),
            parent_id=variables.get("parentId"),
            query=parsed["query"],
        )
        return DummyResponse()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = dispatcher.dispatch_backlog_actions(payload)
    assert result["created"] == 1
    assert result["mode"] == "create"
    assert captured["team_id"] == "TEAM-123"
    assert captured["project_id"] == "PRJ-456"
    assert captured["parent_id"] == "ISS-789"


def test_dispatch_retries_transient_url_errors(monkeypatch, tmp_path):
    payload = build_linear_backlog_payload(
        BacklogConversionResult(
            generated_at="2026-05-13T00:00:00+00:00",
            metric_count=1,
            generated_changes=1,
            changes=[
                BacklogChange(
                    change_id="rule:retry",
                    priority=RulePriority.HIGH,
                    area="acquisition",
                    title="Retry transient failure",
                    summary="Retry test",
                    trigger="clinical_reasoning_accuracy=70.0 (< 85.0)",
                    actions=["Retry"],
                    suggested_sop="Retry",
                    expected_impact="Retry",
                    evidence={"metric": "clinical_reasoning_accuracy"},
                )
            ],
        )
    )

    monkeypatch.setenv("LINEAR_API_KEY_TEST", "token")
    queue_path = tmp_path / "queue.jsonl"
    dispatcher = LinearBacklogDispatcher(
        queue_path=str(queue_path),
        linear_api_key_env="LINEAR_API_KEY_TEST",
        max_retries=3,
    )

    class DummyResponse:
        def __init__(self):
            self._body = (
                b'{"data": {"issueCreate": {"success": true, '
                b'"issue": {"id": "ISSUE-RETRY", "title": "Retry transient failure"}}}}'
            )

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    attempts = {"count": 0}

    def fake_urlopen(request_obj, timeout=12):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.URLError("temporary network failure")
        return DummyResponse()

    import urllib.error
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    result = dispatcher.dispatch_backlog_actions(payload)
    assert attempts["count"] == 2
    assert result["created"] == 1
    assert result["failed"] == 0


def test_dispatch_falls_back_to_create_when_update_id_is_stale(monkeypatch, tmp_path):
    payload = build_linear_backlog_payload(
        BacklogConversionResult(
            generated_at="2026-05-13T00:00:00+00:00",
            metric_count=1,
            generated_changes=1,
            changes=[
                BacklogChange(
                    change_id="rule:stale-1",
                    priority=RulePriority.MEDIUM,
                    area="pipeline_allocation",
                    title="Retire stale assignment",
                    summary="Retire",
                    trigger="clinical_reasoning_accuracy=50.0 (< 85.0)",
                    actions=["Fallback create"],
                    suggested_sop="Retry create",
                    expected_impact="Recover from stale state",
                    evidence={"metric": "clinical_reasoning_accuracy"},
                )
            ],
        )
    )

    monkeypatch.setenv("LINEAR_API_KEY_TEST", "token")
    queue_path = tmp_path / "queue.jsonl"
    state_path = tmp_path / "state.json"
    state_path.write_text('{"rule:stale-1": "MISSING-ISSUE"}', encoding="utf-8")

    dispatcher = LinearBacklogDispatcher(
        queue_path=str(queue_path),
        linear_api_key_env="LINEAR_API_KEY_TEST",
        issue_state_path=str(state_path),
    )

    phases = []

    class DummyResponse:
        def __init__(self, text: str):
            self._body = text.encode("utf-8")

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    def fake_urlopen(request_obj, timeout=12):
        parsed = __import__("json").loads(request_obj.data.decode("utf-8"))
        phases.append(parsed["query"].strip())
        if "issueUpdate" in parsed["query"]:
            return DummyResponse(
                '{"data": {"issueUpdate": {"success": false, "issue": null}}, "errors": ["issue not found"]}'
            )
        if "issueCreate" in parsed["query"]:
            return DummyResponse(
                '{ "data": { "issueCreate": { "success": true, "issue": '
                '{ "id": "ISSUE-RECOVERED", "title": "Retire stale assignment" }}}}'
            )
        raise AssertionError("unexpected mutation")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = dispatcher.dispatch_backlog_actions(payload)
    assert result["mode"] == "create"
    assert result["updated"] == 0
    assert result["created"] == 1
    assert result["failed"] == 0
    assert result["queued"] == 0
    assert len(phases) == 2
    assert phases[0].startswith("mutation($input: IssueUpdateInput!")
    assert phases[1].startswith("mutation($input: IssueCreateInput!")
    state = __import__("json").loads(state_path.read_text(encoding="utf-8"))
    assert state["rule:stale-1"] == "ISSUE-RECOVERED"
