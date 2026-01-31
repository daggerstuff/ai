"""
Simple Pipeline Orchestrator

Simplified pipeline orchestration system for dataset processing workflows.
"""

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineTask:
    """Pipeline task definition."""

    task_id: str = ""
    name: str = ""
    function: Callable | None = None
    dependencies: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


@dataclass
class PipelineExecution:
    """Pipeline execution state."""

    pipeline_id: str = ""
    status: PipelineStatus = PipelineStatus.PENDING
    tasks: list[PipelineTask] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    context: dict[str, Any] = field(default_factory=dict)


class SimplePipelineOrchestrator:
    """Simple pipeline orchestration system."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.executions: dict[str, PipelineExecution] = {}
        self.logger = get_logger(__name__)

        logger.info("SimplePipelineOrchestrator initialized")

    def create_pipeline(
        self, pipeline_id: str, tasks: list[PipelineTask]
    ) -> PipelineExecution:
        """Create a new pipeline."""
        execution = PipelineExecution(
            pipeline_id=pipeline_id, tasks=tasks, status=PipelineStatus.PENDING
        )

        self.executions[pipeline_id] = execution
        logger.info(f"Pipeline created: {pipeline_id} with {len(tasks)} tasks")

        return execution

    async def execute_pipeline(self, pipeline_id: str) -> PipelineExecution:
        """Execute a pipeline."""
        if pipeline_id not in self.executions:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        execution = self.executions[pipeline_id]
        execution.status = PipelineStatus.RUNNING
        execution.start_time = datetime.now()

        logger.info(f"Starting pipeline execution: {pipeline_id}")

        try:
            # Execute tasks in dependency order
            await self._execute_tasks(execution)

            execution.status = PipelineStatus.COMPLETED
            execution.end_time = datetime.now()

            logger.info(f"Pipeline completed: {pipeline_id}")

        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.end_time = datetime.now()
            logger.error(f"Pipeline failed: {pipeline_id} - {e}")
            raise

        return execution

    async def _execute_tasks(self, execution: PipelineExecution):
        """Execute tasks in dependency order."""
        completed_tasks = set()

        while len(completed_tasks) < len(execution.tasks):
            # Find tasks ready to execute
            ready_tasks = []
            for task in execution.tasks:
                if task.status == TaskStatus.PENDING and all(
                    dep in completed_tasks for dep in task.dependencies
                ):
                    ready_tasks.append(task)

            if not ready_tasks:
                # Check for circular dependencies or all tasks failed
                pending_tasks = [
                    t for t in execution.tasks if t.status == TaskStatus.PENDING
                ]
                if pending_tasks:
                    raise RuntimeError(
                        "Circular dependency or all remaining tasks have unmet dependencies"
                    )
                break

            # Execute ready tasks in parallel
            await self._execute_task_batch(ready_tasks, execution.context)

            # Update completed tasks
            for task in ready_tasks:
                if task.status in [
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.SKIPPED,
                ]:
                    completed_tasks.add(task.task_id)

    async def _execute_task_batch(
        self, tasks: list[PipelineTask], context: dict[str, Any]
    ):
        """Execute a batch of tasks in parallel."""
        futures = []

        for task in tasks:
            future = asyncio.create_task(self._execute_single_task(task, context))
            futures.append(future)

        # Wait for all tasks to complete
        await asyncio.gather(*futures, return_exceptions=True)

    async def _execute_single_task(self, task: PipelineTask, context: dict[str, Any]):
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()

        logger.info(f"Executing task: {task.task_id}")

        try:
            if task.function:
                # Prepare parameters
                params = {**task.parameters, **context}

                # Execute function
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, task.function, **params
                )

                task.result = result
                task.status = TaskStatus.COMPLETED

                # Update context with result
                context[f"task_{task.task_id}_result"] = result

            else:
                # No function to execute, mark as skipped
                task.status = TaskStatus.SKIPPED

            task.end_time = datetime.now()
            logger.info(f"Task completed: {task.task_id}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.end_time = datetime.now()
            logger.error(f"Task failed: {task.task_id} - {e}")
            raise

    def get_pipeline_status(self, pipeline_id: str) -> dict[str, Any] | None:
        """Get pipeline execution status."""
        if pipeline_id not in self.executions:
            return None

        execution = self.executions[pipeline_id]

        task_summary = {}
        for status in TaskStatus:
            task_summary[status.value] = len(
                [t for t in execution.tasks if t.status == status]
            )

        duration = None
        if execution.start_time:
            end_time = execution.end_time or datetime.now()
            duration = (end_time - execution.start_time).total_seconds()

        return {
            "pipeline_id": pipeline_id,
            "status": execution.status.value,
            "start_time": (
                execution.start_time.isoformat() if execution.start_time else None
            ),
            "end_time": execution.end_time.isoformat() if execution.end_time else None,
            "duration_seconds": duration,
            "task_summary": task_summary,
            "total_tasks": len(execution.tasks),
        }

    def list_pipelines(self) -> list[dict[str, Any]]:
        """List all pipelines."""
        return [
            {
                "pipeline_id": pipeline_id,
                "status": execution.status.value,
                "task_count": len(execution.tasks),
                "created_time": (
                    execution.start_time.isoformat() if execution.start_time else None
                ),
            }
            for pipeline_id, execution in self.executions.items()
        ]

    def create_sample_pipeline(self) -> str:
        """Create a sample dataset processing pipeline."""

        def download_task(**kwargs):
            """Sample download task."""
            time.sleep(1)  # Simulate work
            return {"files": ["dataset1.json", "dataset2.json"]}

        def validate_task(**kwargs):
            """Sample validation task."""
            time.sleep(0.5)  # Simulate work
            return {"validation_passed": True}

        def process_task(**kwargs):
            """Sample processing task."""
            time.sleep(2)  # Simulate work
            return {"processed_items": 1000}

        tasks = [
            PipelineTask(
                task_id="download",
                name="Download Dataset",
                function=download_task,
                dependencies=[],
            ),
            PipelineTask(
                task_id="validate",
                name="Validate Dataset",
                function=validate_task,
                dependencies=["download"],
            ),
            PipelineTask(
                task_id="process",
                name="Process Dataset",
                function=process_task,
                dependencies=["validate"],
            ),
        ]

        pipeline_id = f"sample_pipeline_{int(time.time())}"
        self.create_pipeline(pipeline_id, tasks)

        return pipeline_id

    def shutdown(self):
        """Shutdown the orchestrator."""
        self.executor.shutdown(wait=True)
        logger.info("SimplePipelineOrchestrator shutdown complete")


# Example usage
if __name__ == "__main__":

    async def main():
        orchestrator = SimplePipelineOrchestrator()

        # Create and execute sample pipeline
        pipeline_id = orchestrator.create_sample_pipeline()

        await orchestrator.execute_pipeline(pipeline_id)

        # Show final status
        orchestrator.get_pipeline_status(pipeline_id)

        orchestrator.shutdown()

    # Run example
    asyncio.run(main())
