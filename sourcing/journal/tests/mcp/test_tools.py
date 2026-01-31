import pytest
from datetime import datetime
from typing import Any, Dict, List, Optional

from ai.sourcing.journal.mcp.protocol import MCPError, MCPErrorCode
from ai.sourcing.journal.mcp.tools.sessions import CreateSessionTool
from ai.sourcing.journal.mcp.tools.discovery import DiscoverSourcesTool
from ai.sourcing.journal.mcp.tools.evaluation import EvaluateSourcesTool
from ai.sourcing.journal.mcp.tools.acquisition import AcquireDatasetsTool
from ai.sourcing.journal.mcp.tools.integration import CreateIntegrationPlansTool
from ai.sourcing.journal.mcp.tools.reports import GenerateReportTool
from ai.sourcing.journal.mcp.tools.registry import ToolRegistry
from ai.sourcing.journal.mcp.tools.executor import ToolExecutor
from ai.sourcing.journal.mcp.tools.base import MCPTool
from ai.sourcing.journal.models.dataset_models import (
    ResearchSession,
    DatasetSource,
    DatasetEvaluation,
    AcquiredDataset,
    IntegrationPlan,
)


class _SessionServiceStub:
    """Small stub for session creation tests."""

    def __init__(self, session: ResearchSession) -> None:
        self._session = session
        self.called_with: Dict[str, Any] = {}

    def create_session(
        self,
        target_sources,
        search_keywords,
        weekly_targets=None,
        session_id: Optional[str] = None,
    ) -> ResearchSession:
        self.called_with = {
            "target_sources": target_sources,
            "search_keywords": search_keywords,
            "weekly_targets": weekly_targets,
            "session_id": session_id,
        }
        return self._session


class _DiscoveryServiceStub:
    """Stub for discovery tool tests."""

    def __init__(self, sources: List[DatasetSource]) -> None:
        self.sources = sources
        self.request_log: Dict[str, Any] = {}

    def initiate_discovery(
        self,
        session_id: str,
        keywords: List[str],
        sources: List[str],
    ) -> Dict[str, Any]:
        self.request_log = {
            "session_id": session_id,
            "keywords": keywords,
            "sources": sources,
        }
        return {"session_id": session_id, "total_sources": len(self.sources)}

    def get_sources(self, session_id: str) -> List[DatasetSource]:
        self.request_log["get_sources_for"] = session_id
        return self.sources


class _EvaluationServiceStub:
    """Stub for evaluation tool tests."""

    def __init__(self, evaluations: List[Dict[str, Any]]) -> None:
        self.evaluations = evaluations

    def initiate_evaluation(
        self,
        session_id: str,
        source_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {"session_id": session_id, "evaluations": self.evaluations}


class _AcquisitionServiceStub:
    """Stub for acquisition tool tests."""

    def __init__(self, acquisitions: List[AcquiredDataset]) -> None:
        self.acquisitions = acquisitions

    def initiate_acquisition(
        self,
        session_id: str,
        source_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {"session_id": session_id, "acquired": source_ids or []}

    def get_acquisitions(self, session_id: str) -> List[AcquiredDataset]:
        return self.acquisitions


class _IntegrationServiceStub:
    """Stub for integration planning tests."""

    def __init__(self, plans: List[IntegrationPlan]) -> None:
        self.plans = plans

    def initiate_integration(
        self,
        session_id: str,
        source_ids: Optional[List[str]] = None,
        target_format: str = "chatml",
    ) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "plans": [{"source_id": plan.source_id, "target_format": target_format} for plan in self.plans],
        }

    def get_integration_plans(self, session_id: str) -> List[IntegrationPlan]:
        return self.plans


class _ReportServiceStub:
    """Stub for report generation tests."""

    def __init__(self) -> None:
        self.last_params: Dict[str, Any] = {}

    def generate_report(
        self,
        session_id: str,
        report_type: str,
        format: str,
        date_range: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        self.last_params = {
            "session_id": session_id,
            "report_type": report_type,
            "format": format,
            "date_range": date_range,
        }
        return {
            "report_id": f"report_{session_id}",
            "session_id": session_id,
            "report_type": report_type,
            "format": format,
            "generated_date": datetime(2025, 1, 1, 0, 0, 0),
            "content": {"summary": "ok"},
            "file_path": f"/tmp/{session_id}.json",
        }


@pytest.mark.asyncio
async def test_create_session_tool_success(sample_research_session: ResearchSession) -> None:
    service = _SessionServiceStub(sample_research_session)
    tool = CreateSessionTool(service)

    params = {
        "target_sources": ["pubmed", "zenodo"],
        "search_keywords": {"therapy": ["cbt"]},
        "weekly_targets": {"sources_identified": 5},
    }

    result = await tool.execute(params)

    assert result["session_id"] == sample_research_session.session_id
    assert service.called_with["target_sources"] == ["pubmed", "zenodo"]
    assert service.called_with["weekly_targets"]["sources_identified"] == 5


@pytest.mark.asyncio
async def test_create_session_tool_validation_error(sample_research_session: ResearchSession) -> None:
    service = _SessionServiceStub(sample_research_session)
    tool = CreateSessionTool(service)

    with pytest.raises(MCPError) as exc:
        await tool.execute({"target_sources": [], "search_keywords": {}})

    assert exc.value.code == MCPErrorCode.TOOL_VALIDATION_ERROR


@pytest.mark.asyncio
async def test_discover_sources_tool_returns_sources(
    sample_research_session: ResearchSession,
    sample_dataset_source: DatasetSource,
) -> None:
    service = _DiscoveryServiceStub([sample_dataset_source])
    tool = DiscoverSourcesTool(service)

    params = {
        "session_id": sample_research_session.session_id,
        "keywords": ["therapy", "mental health"],
        "sources": ["pubmed", "dryad"],
    }

    result = await tool.execute(params)

    assert result["total_sources"] == 1
    assert result["sources"][0]["source_id"] == sample_dataset_source.source_id
    assert service.request_log["session_id"] == sample_research_session.session_id


@pytest.mark.asyncio
async def test_discover_sources_tool_requires_keywords(
    sample_research_session: ResearchSession,
) -> None:
    service = _DiscoveryServiceStub([])
    tool = DiscoverSourcesTool(service)

    with pytest.raises(MCPError) as exc:
        await tool.execute(
            {
                "session_id": sample_research_session.session_id,
                "keywords": [],
                "sources": ["pubmed"],
            }
        )

    assert exc.value.code == MCPErrorCode.TOOL_VALIDATION_ERROR


@pytest.mark.asyncio
async def test_evaluate_sources_tool_success(
    sample_research_session: ResearchSession,
    sample_evaluation: DatasetEvaluation,
) -> None:
    evaluation_dict = {
        "evaluation_id": f"eval_{sample_evaluation.source_id}",
        "source_id": sample_evaluation.source_id,
        "overall_score": sample_evaluation.overall_score,
    }
    service = _EvaluationServiceStub([evaluation_dict])
    tool = EvaluateSourcesTool(service)

    result = await tool.execute({"session_id": sample_research_session.session_id})

    assert result["total_evaluated"] == 1
    assert result["evaluations"][0]["source_id"] == sample_evaluation.source_id


@pytest.mark.asyncio
async def test_acquire_datasets_tool_success(
    sample_research_session: ResearchSession,
    sample_acquired_dataset: AcquiredDataset,
) -> None:
    service = _AcquisitionServiceStub([sample_acquired_dataset])
    tool = AcquireDatasetsTool(service)

    result = await tool.execute({"session_id": sample_research_session.session_id})

    assert result["total_acquired"] == 1
    assert result["acquisitions"][0]["source_id"] == sample_acquired_dataset.source_id
    assert result["acquisitions"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_create_integration_plans_tool_success(
    sample_research_session: ResearchSession,
    sample_integration_plan: IntegrationPlan,
) -> None:
    service = _IntegrationServiceStub([sample_integration_plan])
    tool = CreateIntegrationPlansTool(service)

    result = await tool.execute({"session_id": sample_research_session.session_id, "target_format": "chatml"})

    assert result["total_plans"] == 1
    assert result["integration_plans"][0]["source_id"] == sample_integration_plan.source_id
    assert result["integration_plans"][0]["dataset_format"] == sample_integration_plan.dataset_format


@pytest.mark.asyncio
async def test_generate_report_tool_formats_datetime(
    sample_research_session: ResearchSession,
) -> None:
    service = _ReportServiceStub()
    tool = GenerateReportTool(service)

    result = await tool.execute(
        {
            "session_id": sample_research_session.session_id,
            "report_type": "summary_report",
            "format": "json",
        }
    )

    assert result["generated_date"].endswith("00:00")
    assert result["report_type"] == "summary_report"
    assert service.last_params["report_type"] == "summary_report"


class _EchoTool(MCPTool):
    """Simple tool for ToolExecutor tests."""

    def __init__(self) -> None:
        super().__init__(
            name="echo",
            description="Echo tool for testing",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "minLength": 1},
                },
                "required": ["message"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"echo": params["message"]}


@pytest.mark.asyncio
async def test_tool_executor_runs_registered_tool() -> None:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    executor = ToolExecutor(registry)

    result = await executor.execute_tool("echo", {"message": "hello"})

    assert result == {"echo": "hello"}


@pytest.mark.asyncio
async def test_tool_executor_validation_error() -> None:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    executor = ToolExecutor(registry)

    with pytest.raises(MCPError) as exc:
        await executor.execute_tool("echo", {"message": ""})

    assert exc.value.code == MCPErrorCode.TOOL_VALIDATION_ERROR


@pytest.mark.asyncio
async def test_tool_executor_missing_tool_raises() -> None:
    executor = ToolExecutor(ToolRegistry())

    with pytest.raises(MCPError) as exc:
        await executor.execute_tool("missing_tool")

    assert exc.value.code == MCPErrorCode.TOOL_EXECUTION_ERROR

