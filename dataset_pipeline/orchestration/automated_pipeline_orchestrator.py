"""
Automated Pipeline Orchestration System

Advanced orchestration system for dataset processing pipelines with
dependency management, parallel execution, error recovery, and monitoring.
"""

import asyncio
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class PipelineStatus(Enum):
    """Pipeline execution status."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class TaskResult:
    """Task execution result."""

    task_id: str
    status: TaskStatus
    output: Any = None
    error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration: float = 0.0
    retry_count: int = 0


@dataclass
class PipelineTask:
    """Pipeline task definition."""

    id: str
    name: str
    function: Callable
    dependencies: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 3
    timeout_seconds: int | None = None
    condition: Callable | None = None  # Conditional execution
    on_success: Callable | None = None
    on_failure: Callable | None = None
    parallel_group: str | None = None


@dataclass
class PipelineDefinition:
    """Complete pipeline definition."""

    id: str
    name: str
    description: str
    tasks: list[PipelineTask]
    global_timeout: int | None = None
    max_parallel_tasks: int = 5
    failure_strategy: str = "stop"  # "stop", "continue", "retry"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineExecution:
    """Pipeline execution state."""

    pipeline_id: str
    execution_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: datetime | None = None
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    current_tasks: set[str] = field(default_factory=set)
    completed_tasks: set[str] = field(default_factory=set)
    failed_tasks: set[str] = field(default_factory=set)
    context: dict[str, Any] = field(default_factory=dict)


class TaskExecutor:
    """Individual task executor with retry and timeout logic."""

    def __init__(self, max_workers: int = 10):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = get_logger(__name__)

    async def execute_task(
        self, task: PipelineTask, context: dict[str, Any]
    ) -> TaskResult:
        """Execute a single task with retry and timeout logic."""
        task_result = TaskResult(
            task_id=task.id, status=TaskStatus.PENDING, start_time=datetime.now()
        )

        # Check condition if specified
        if task.condition and not task.condition(context):
            task_result.status = TaskStatus.SKIPPED
            task_result.end_time = datetime.now()
            self.logger.info(f"Task {task.id} skipped due to condition")
            return task_result

        # Execute with retries
        for attempt in range(task.retry_count + 1):
            try:
                task_result.status = TaskStatus.RUNNING
                task_result.retry_count = attempt

                self.logger.info(f"Executing task {task.id} (attempt {attempt + 1})")

                # Prepare parameters
                params = {**task.parameters, **context}

                # Execute with timeout
                if task.timeout_seconds:
                    future = self.executor.submit(task.function, **params)
                    try:
                        output = future.result(timeout=task.timeout_seconds)
                    except TimeoutError:
                        future.cancel()
                        raise TimeoutError(
                            f"Task {task.id} timed out after {task.timeout_seconds} seconds"
                        )
                else:
                    # Run in thread pool
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        self.executor, lambda: task.function(**params)
                    )

                # Success
                task_result.status = TaskStatus.COMPLETED
                task_result.output = output
                task_result.end_time = datetime.now()
                task_result.duration = (
                    task_result.end_time - task_result.start_time
                ).total_seconds()

                self.logger.info(f"Task {task.id} completed successfully")

                # Call success callback
                if task.on_success:
                    try:
                        task.on_success(task_result, context)
                    except Exception as e:
                        self.logger.error(
                            f"Success callback failed for task {task.id}: {e}"
                        )

                return task_result

            except Exception as e:
                error_msg = f"Task {task.id} failed (attempt {attempt + 1}): {e!s}"
                self.logger.error(error_msg)

                if attempt < task.retry_count:
                    # Wait before retry (exponential backoff)
                    wait_time = 2**attempt
                    self.logger.info(f"Retrying task {task.id} in {wait_time} seconds")
                    await asyncio.sleep(wait_time)
                else:
                    # Final failure
                    task_result.status = TaskStatus.FAILED
                    task_result.error = error_msg
                    task_result.end_time = datetime.now()
                    task_result.duration = (
                        task_result.end_time - task_result.start_time
                    ).total_seconds()

                    # Call failure callback
                    if task.on_failure:
                        try:
                            task.on_failure(task_result, context)
                        except Exception as callback_error:
                            self.logger.error(
                                f"Failure callback failed for task {task.id}: {callback_error}"
                            )

                    return task_result

        return task_result

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class DependencyResolver:
    """Resolves task dependencies and determines execution order."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def resolve_dependencies(self, tasks: list[PipelineTask]) -> list[list[str]]:
        """Resolve dependencies and return execution levels."""
        task_map = {task.id: task for task in tasks}

        # Build dependency graph
        dependencies = {}
        dependents = {}

        for task in tasks:
            dependencies[task.id] = set(task.dependencies)
            dependents[task.id] = set()

        # Build reverse dependencies
        for task_id, deps in dependencies.items():
            for dep in deps:
                if dep in dependents:
                    dependents[dep].add(task_id)

        # Topological sort to find execution levels
        levels = []
        remaining_tasks = set(task_map.keys())

        while remaining_tasks:
            # Find tasks with no remaining dependencies
            ready_tasks = []
            for task_id in remaining_tasks:
                if not dependencies[task_id]:
                    ready_tasks.append(task_id)

            if not ready_tasks:
                # Circular dependency detected
                raise ValueError(
                    f"Circular dependency detected in tasks: {remaining_tasks}"
                )

            # Group by parallel groups
            level_groups = {}
            for task_id in ready_tasks:
                task = task_map[task_id]
                group = task.parallel_group or "default"
                if group not in level_groups:
                    level_groups[group] = []
                level_groups[group].append(task_id)

            # Add level (flatten groups for now, could be enhanced for true parallel groups)
            level = []
            for group_tasks in level_groups.values():
                level.extend(group_tasks)
            levels.append(level)

            # Remove completed tasks and update dependencies
            for task_id in ready_tasks:
                remaining_tasks.remove(task_id)
                for dependent in dependents[task_id]:
                    dependencies[dependent].discard(task_id)

        return levels


class PipelineMonitor:
    """Monitors pipeline execution and provides metrics."""

    def __init__(self):
        self.executions: dict[str, PipelineExecution] = {}
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)

    def start_execution(self, pipeline: PipelineDefinition) -> str:
        """Start monitoring a new pipeline execution."""
        execution_id = str(uuid.uuid4())

        with self.lock:
            execution = PipelineExecution(
                pipeline_id=pipeline.id,
                execution_id=execution_id,
                status=PipelineStatus.RUNNING,
                start_time=datetime.now(),
            )
            self.executions[execution_id] = execution

        self.logger.info(f"Started monitoring pipeline execution: {execution_id}")
        return execution_id

    def update_task_result(self, execution_id: str, task_result: TaskResult):
        """Update task result in execution."""
        with self.lock:
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                execution.task_results[task_result.task_id] = task_result

                if task_result.status == TaskStatus.COMPLETED:
                    execution.completed_tasks.add(task_result.task_id)
                elif task_result.status == TaskStatus.FAILED:
                    execution.failed_tasks.add(task_result.task_id)

                execution.current_tasks.discard(task_result.task_id)

    def complete_execution(self, execution_id: str, status: PipelineStatus):
        """Mark execution as completed."""
        with self.lock:
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                execution.status = status
                execution.end_time = datetime.now()

                self.logger.info(
                    f"Pipeline execution completed: {execution_id} ({status.value})"
                )

    def get_execution_status(self, execution_id: str) -> PipelineExecution | None:
        """Get execution status."""
        with self.lock:
            return self.executions.get(execution_id)

    def get_execution_metrics(self, execution_id: str) -> dict[str, Any]:
        """Get detailed execution metrics."""
        with self.lock:
            execution = self.executions.get(execution_id)
            if not execution:
                return {}

            total_tasks = len(execution.task_results)
            completed_tasks = len(execution.completed_tasks)
            failed_tasks = len(execution.failed_tasks)
            running_tasks = len(execution.current_tasks)

            total_duration = 0
            if execution.end_time:
                total_duration = (
                    execution.end_time - execution.start_time
                ).total_seconds()
            else:
                total_duration = (datetime.now() - execution.start_time).total_seconds()

            return {
                "execution_id": execution_id,
                "pipeline_id": execution.pipeline_id,
                "status": execution.status.value,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "running_tasks": running_tasks,
                "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
                "total_duration": total_duration,
                "start_time": execution.start_time.isoformat(),
                "end_time": (
                    execution.end_time.isoformat() if execution.end_time else None
                ),
            }


class AutomatedPipelineOrchestrator:
    """Main pipeline orchestration system."""

    def __init__(
        self, max_concurrent_pipelines: int = 5, max_workers_per_pipeline: int = 10
    ):
        self.max_concurrent_pipelines = max_concurrent_pipelines
        self.max_workers_per_pipeline = max_workers_per_pipeline

        # Components
        self.task_executor = TaskExecutor(max_workers_per_pipeline)
        self.dependency_resolver = DependencyResolver()
        self.monitor = PipelineMonitor()

        # Pipeline storage
        self.pipelines: dict[str, PipelineDefinition] = {}
        self.active_executions: set[str] = set()

        # Threading
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_pipelines)

        self.logger = get_logger(__name__)
        logger.info("AutomatedPipelineOrchestrator initialized")

    def register_pipeline(self, pipeline: PipelineDefinition):
        """Register a pipeline definition."""
        self.pipelines[pipeline.id] = pipeline
        self.logger.info(f"Registered pipeline: {pipeline.id}")

    async def execute_pipeline(
        self, pipeline_id: str, context: dict[str, Any] | None = None
    ) -> str:
        """Execute a pipeline asynchronously."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        pipeline = self.pipelines[pipeline_id]
        context = context or {}

        # Start monitoring
        execution_id = self.monitor.start_execution(pipeline)

        async with self.execution_semaphore:
            self.active_executions.add(execution_id)

            try:
                await self._execute_pipeline_internal(pipeline, execution_id, context)
                self.monitor.complete_execution(execution_id, PipelineStatus.COMPLETED)

            except Exception as e:
                self.logger.error(f"Pipeline execution failed: {e}")
                self.monitor.complete_execution(execution_id, PipelineStatus.FAILED)
                raise

            finally:
                self.active_executions.discard(execution_id)

        return execution_id

    async def _execute_pipeline_internal(
        self, pipeline: PipelineDefinition, execution_id: str, context: dict[str, Any]
    ):
        """Internal pipeline execution logic."""
        # Resolve dependencies
        execution_levels = self.dependency_resolver.resolve_dependencies(pipeline.tasks)
        task_map = {task.id: task for task in pipeline.tasks}

        self.logger.info(
            f"Executing pipeline {pipeline.id} with {len(execution_levels)} levels"
        )

        # Execute level by level
        for level_index, level_tasks in enumerate(execution_levels):
            self.logger.info(
                f"Executing level {level_index + 1} with {len(level_tasks)} tasks"
            )

            # Execute tasks in parallel within the level
            level_futures = []

            for task_id in level_tasks:
                task = task_map[task_id]

                # Update monitoring
                execution = self.monitor.get_execution_status(execution_id)
                execution.current_tasks.add(task_id)

                # Execute task
                future = asyncio.create_task(
                    self.task_executor.execute_task(task, context)
                )
                level_futures.append((task_id, future))

            # Wait for all tasks in level to complete
            for task_id, future in level_futures:
                try:
                    task_result = await future

                    # Update monitoring
                    self.monitor.update_task_result(execution_id, task_result)

                    # Update context with task output
                    if (
                        task_result.status == TaskStatus.COMPLETED
                        and task_result.output is not None
                    ):
                        context[f"task_{task_id}_output"] = task_result.output

                    # Handle failure based on strategy
                    if task_result.status == TaskStatus.FAILED:
                        if pipeline.failure_strategy == "stop":
                            raise RuntimeError(
                                f"Task {task_id} failed: {task_result.error}"
                            )
                        if pipeline.failure_strategy == "continue":
                            self.logger.warning(
                                f"Task {task_id} failed but continuing: {task_result.error}"
                            )

                except Exception as e:
                    self.logger.error(f"Task execution error: {e}")
                    if pipeline.failure_strategy == "stop":
                        raise

    def get_pipeline_status(self, execution_id: str) -> dict[str, Any] | None:
        """Get pipeline execution status."""
        return self.monitor.get_execution_metrics(execution_id)

    def list_pipelines(self) -> list[dict[str, Any]]:
        """List all registered pipelines."""
        return [
            {
                "id": pipeline.id,
                "name": pipeline.name,
                "description": pipeline.description,
                "task_count": len(pipeline.tasks),
                "max_parallel_tasks": pipeline.max_parallel_tasks,
            }
            for pipeline in self.pipelines.values()
        ]

    def create_dataset_processing_pipeline(self) -> PipelineDefinition:
        """Create a standard dataset processing pipeline."""

        def download_task(**kwargs):
            """Mock download task."""
            time.sleep(1)  # Simulate download
            return {"downloaded_files": ["file1.json", "file2.json"]}

        def validate_task(**kwargs):
            """Mock validation task."""
            time.sleep(0.5)  # Simulate validation
            return {"validation_passed": True}

        def process_task(**kwargs):
            """Mock processing task."""
            time.sleep(2)  # Simulate processing
            return {"processed_items": 1000}

        def quality_check_task(**kwargs):
            """Mock quality check task."""
            time.sleep(0.5)  # Simulate quality check
            return {"quality_score": 0.85}

        def export_task(**kwargs):
            """Mock export task."""
            time.sleep(1)  # Simulate export
            return {"export_path": "/output/processed_dataset.json"}

        tasks = [
            PipelineTask(
                id="download",
                name="Download Dataset",
                function=download_task,
                dependencies=[],
                retry_count=3,
                timeout_seconds=300,
            ),
            PipelineTask(
                id="validate",
                name="Validate Dataset",
                function=validate_task,
                dependencies=["download"],
                retry_count=2,
            ),
            PipelineTask(
                id="process",
                name="Process Dataset",
                function=process_task,
                dependencies=["validate"],
                retry_count=2,
                timeout_seconds=600,
            ),
            PipelineTask(
                id="quality_check",
                name="Quality Check",
                function=quality_check_task,
                dependencies=["process"],
                retry_count=1,
            ),
            PipelineTask(
                id="export",
                name="Export Results",
                function=export_task,
                dependencies=["quality_check"],
                retry_count=2,
            ),
        ]

        return PipelineDefinition(
            id="dataset_processing",
            name="Dataset Processing Pipeline",
            description="Standard pipeline for dataset download, validation, processing, and export",
            tasks=tasks,
            max_parallel_tasks=3,
            failure_strategy="stop",
        )

    def shutdown(self):
        """Shutdown the orchestrator."""
        self.task_executor.shutdown()
        self.logger.info("Pipeline orchestrator shutdown complete")


# Example usage
if __name__ == "__main__":

    async def main():
        # Initialize orchestrator
        orchestrator = AutomatedPipelineOrchestrator()

        # Create and register a pipeline
        pipeline = orchestrator.create_dataset_processing_pipeline()
        orchestrator.register_pipeline(pipeline)

        # Execute pipeline
        execution_id = await orchestrator.execute_pipeline("dataset_processing")

        # Monitor execution
        while True:
            status = orchestrator.get_pipeline_status(execution_id)
            if status and status["status"] in ["completed", "failed"]:
                break

            await asyncio.sleep(1)

        # Final status
        orchestrator.get_pipeline_status(execution_id)

        orchestrator.shutdown()

    # Run example
    asyncio.run(main())
