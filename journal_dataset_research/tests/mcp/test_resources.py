import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from ai.journal_dataset_research.mcp.protocol import MCPError, JSONRPCErrorCode
from ai.journal_dataset_research.mcp.resources.progress import (
    ProgressMetricsResource,
    ProgressHistoryResource,
)
from ai.journal_dataset_research.mcp.resources.sessions import SessionStateResource
from ai.journal_dataset_research.mcp.resources.metrics import SessionMetricsResource
from ai.journal_dataset_research.models.dataset_models import (
    ResearchProgress,
    ResearchLog,
)
from ai.journal_dataset_research.orchestrator.types import (
    SessionState,
    ProgressSnapshot,
)


class _FakeOrchestrator:
    """Simple orchestrator stub for resource tests."""

    def __init__(
        self,
        session_id: str,
        progress: ResearchProgress,
        history: List[ProgressSnapshot],
        state: SessionState,
        activity_log: List[ResearchLog],
        error_log: List[Dict[str, Any]],
        report: Dict[str, Any],
    ) -> None:
        self._session_id = session_id
        self._progress = progress
        self._history = history
        self._state = state
        self._activity_log = activity_log
        self._error_log = error_log
        self._report = report

    def _ensure_session(self, session_id: str) -> None:
        if session_id != self._session_id:
            raise KeyError(session_id)

    def get_progress(self, session_id: str) -> ResearchProgress:
        self._ensure_session(session_id)
        return self._progress

    def get_progress_history(self, session_id: str) -> List[ProgressSnapshot]:
        self._ensure_session(session_id)
        return self._history

    def get_session_state(self, session_id: str) -> SessionState:
        self._ensure_session(session_id)
        return self._state

    def get_activity_log(self, session_id: str) -> List[ResearchLog]:
        self._ensure_session(session_id)
        return self._activity_log

    def get_error_log(self, session_id: str) -> List[Dict[str, Any]]:
        self._ensure_session(session_id)
        return self._error_log

    def generate_progress_report(self, session_id: str) -> Dict[str, Any]:
        self._ensure_session(session_id)
        return self._report


class _FakeService:
    """Service shim exposing orchestrator attribute."""

    def __init__(self, orchestrator: _FakeOrchestrator) -> None:
        self.orchestrator = orchestrator


def _build_orchestrator(
    session_id: str,
    session_state: SessionState,
    progress_report: Optional[Dict[str, Any]] = None,
) -> _FakeOrchestrator:
    progress = ResearchProgress(
        sources_identified=3,
        datasets_evaluated=2,
        access_established=1,
        datasets_acquired=1,
        integration_plans_created=1,
        last_updated=datetime(2025, 1, 1, 12, 0, 0),
    )
    history = [
        ProgressSnapshot(
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            progress=progress,
            metrics={"sources_identified": 3},
        )
    ]
    activity_log = [
        ResearchLog(
            timestamp=datetime(2025, 1, 1, 10, 0, 0),
            activity_type="session_start",
            description="Session created",
        )
    ]
    error_log = [{"timestamp": "2025-01-01T10:10:00Z", "message": "none"}]
    report = progress_report or {"summary": "steady progress"}

    orchestrator = _FakeOrchestrator(
        session_id=session_id,
        progress=progress,
        history=history,
        state=session_state,
        activity_log=activity_log,
        error_log=error_log,
        report=report,
    )
    return orchestrator


@pytest.mark.asyncio
async def test_progress_metrics_resource_returns_serialized_progress(
    sample_dataset_source,
    sample_evaluation,
    sample_acquired_dataset,
    sample_integration_plan,
) -> None:
    session_state = SessionState(
        sources=[sample_dataset_source],
        evaluations=[sample_evaluation],
        acquired_datasets=[sample_acquired_dataset],
        integration_plans=[sample_integration_plan],
    )
    orchestrator = _build_orchestrator("session-1", session_state)
    resource = ProgressMetricsResource(_FakeService(orchestrator))

    result = await resource.read({"session_id": "session-1"})

    assert "contents" in result
    payload = json.loads(result["contents"][0]["text"])
    assert payload["sources_identified"] == 3
    assert payload["integration_plans_created"] == 1


@pytest.mark.asyncio
async def test_progress_metrics_resource_missing_params_raises() -> None:
    resource = ProgressMetricsResource(_FakeService(_build_orchestrator("session-1", SessionState())))

    with pytest.raises(MCPError) as exc:
        await resource.read({})

    assert exc.value.code == JSONRPCErrorCode.INVALID_PARAMS


@pytest.mark.asyncio
async def test_progress_history_resource_lists_snapshots(
    sample_dataset_source,
    sample_evaluation,
) -> None:
    session_state = SessionState(
        sources=[sample_dataset_source],
        evaluations=[sample_evaluation],
    )
    orchestrator = _build_orchestrator("session-2", session_state)
    resource = ProgressHistoryResource(_FakeService(orchestrator))

    result = await resource.read({"session_id": "session-2"})
    payload = json.loads(result["contents"][0]["text"])

    assert isinstance(payload, list)
    assert payload[0]["metrics"]["sources_identified"] == 3


@pytest.mark.asyncio
async def test_session_state_resource_returns_state(
    sample_dataset_source,
    sample_evaluation,
    sample_access_request,
    sample_acquired_dataset,
    sample_integration_plan,
) -> None:
    session_state = SessionState(
        sources=[sample_dataset_source],
        evaluations=[sample_evaluation],
        access_requests=[sample_access_request],
        acquired_datasets=[sample_acquired_dataset],
        integration_plans=[sample_integration_plan],
    )
    orchestrator = _build_orchestrator("session-3", session_state)
    resource = SessionStateResource(_FakeService(orchestrator))

    result = await resource.read({"session_id": "session-3"})
    payload = json.loads(result["contents"][0]["text"])

    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["source_id"] == sample_dataset_source.source_id
    assert len(payload["evaluations"]) == 1


@pytest.mark.asyncio
async def test_session_metrics_resource_activity_only(
    sample_dataset_source,
    sample_evaluation,
) -> None:
    session_state = SessionState(
        sources=[sample_dataset_source],
        evaluations=[sample_evaluation],
    )
    orchestrator = _build_orchestrator("session-4", session_state, progress_report={"summary": "report"})
    resource = SessionMetricsResource(_FakeService(orchestrator))

    result = await resource.read({"session_id": "session-4", "metric_type": "activity"})
    payload = json.loads(result["contents"][0]["text"])

    assert "activity_log" in payload
    assert "progress_report" not in payload
    assert payload["activity_log"][0]["activity_type"] == "session_start"

