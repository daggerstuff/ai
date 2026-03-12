"""
Compatibility shim for legacy import path.

The package-level imports in :mod:`ai.core.pipelines` historically referenced
``ai.core.pipelines.training_orchestrator``.
The actual orchestrator implementation now lives in ``ai.orchestrator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineStage:
    """Minimal stage representation used by compatibility imports."""

    stage_id: str
    name: str
    status: str = "pending"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineExecution:
    """Minimal execution record used by compatibility imports."""

    execution_id: str
    pipeline_id: str = ""
    status: str = "initialized"
    start_time: str = ""
    end_time: Optional[str] = None
    stages: List[PipelineStage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)


class TrainingPipelineOrchestrator:
    """Compatibility fallback for environments that import the legacy module path."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.active_executions: Dict[str, PipelineExecution] = {}
        self.taxonomy = None
        self.style_manager = None

    async def execute_pipeline(self, pipeline_config: Dict[str, Any], user_id: str = "system") -> str:
        execution_id = pipeline_config.get("execution_id", "fallback-execution")
        self.active_executions[execution_id] = PipelineExecution(
            execution_id=execution_id,
            pipeline_id=pipeline_config.get("pipeline_id", execution_id),
            status="completed",
            stages=[],
            metadata={"user_id": user_id, "config": pipeline_config},
            results={"executed": True},
        )
        return execution_id

    async def get_status(self, execution_id: str) -> Dict[str, Any]:
        execution = self.active_executions.get(execution_id)
        if execution is None:
            return {"status": "unknown", "execution_id": execution_id}
        return {
            "status": execution.status,
            "execution_id": execution_id,
            "metadata": execution.metadata,
            "results": execution.results,
        }

    async def get_results(self, execution_id: str) -> Dict[str, Any]:
        return await self.get_status(execution_id)

    def list_active_executions(self) -> List[Dict[str, Any]]:
        return [execution.__dict__ for execution in self.active_executions.values()]

__all__ = [
    "TrainingPipelineOrchestrator",
    "PipelineStage",
    "PipelineExecution",
]
