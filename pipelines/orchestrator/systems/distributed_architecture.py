#!/usr/bin/env python3
"""
Distributed Processing Architecture for Task 6.1
Enterprise-grade distributed processing system for large-scale conversation analysis.
"""

import asyncio
import concurrent.futures
import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for distributed architecture."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"
    STREAMING = "streaming"


class NodeStatus(Enum):
    """Status of processing nodes."""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    OFFLINE = "offline"


class DataTier(Enum):
    """Data processing tiers for distributed architecture."""
    RAW = "raw"
    PROCESSED = "processed"
    VALIDATED = "validated"
    PRODUCTION = "production"


@dataclass
class ProcessingNode:
    """Individual processing node in distributed system."""
    node_id: str
    node_type: str
    capacity: int
    current_load: int = 0
    status: NodeStatus = NodeStatus.IDLE
    last_heartbeat: datetime = field(default_factory=datetime.now)
    processed_count: int = 0
    error_count: int = 0
    performance_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ProcessingTask:
    """Task to be processed in distributed system."""
    task_id: str
    task_type: str
    data: Any
    priority: int = 1
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    assigned_node: str | None = None
    status: str = "pending"
    result: Any | None = None
    error: str | None = None


@dataclass
class ProcessingResult:
    """Result from distributed processing."""
    task_id: str
    node_id: str
    result: Any
    processing_time: float
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DistributedArchitecture:
    """
    Enterprise-grade distributed processing architecture.
    """

    def __init__(self, max_workers: int = 8):
        """Initialize distributed architecture."""
        self.max_workers = max_workers
        self.nodes: dict[str, ProcessingNode] = {}
        self.task_queue: list[ProcessingTask] = []
        self.completed_tasks: dict[str, ProcessingResult] = {}
        self.processing_functions: dict[str, Callable] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False

        # Initialize local processing nodes
        self._initialize_nodes()

        logger.info(f"DistributedArchitecture initialized with {max_workers} workers")

    def _initialize_nodes(self):
        """Initialize processing nodes."""
        for i in range(self.max_workers):
            node = ProcessingNode(
                node_id=f"node_{i:03d}",
                node_type="worker",
                capacity=10,
                performance_metrics={
                    "avg_processing_time": 0.0,
                    "success_rate": 1.0,
                    "throughput": 0.0
                }
            )
            self.nodes[node.node_id] = node

    def register_processing_function(self, task_type: str, func: Callable):
        """Register a processing function for a task type."""
        self.processing_functions[task_type] = func
        logger.info(f"Registered processing function for task type: {task_type}")

    def submit_task(self, task_type: str, data: Any, priority: int = 1) -> str:
        """Submit a task for distributed processing."""
        task_id = self._generate_task_id(task_type, data)

        task = ProcessingTask(
            task_id=task_id,
            task_type=task_type,
            data=data,
            priority=priority
        )

        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)

        logger.info(f"Task submitted: {task_id} (type: {task_type}, priority: {priority})")
        return task_id

    def submit_batch(self, tasks: list[dict[str, Any]]) -> list[str]:
        """Submit multiple tasks as a batch."""
        task_ids = []

        for task_data in tasks:
            task_id = self.submit_task(
                task_type=task_data["task_type"],
                data=task_data["data"],
                priority=task_data.get("priority", 1)
            )
            task_ids.append(task_id)

        logger.info(f"Batch submitted: {len(task_ids)} tasks")
        return task_ids

    async def process_tasks_async(self, mode: ProcessingMode = ProcessingMode.PARALLEL):
        """Process tasks asynchronously."""
        self.is_running = True
        logger.info(f"Starting async processing in {mode.value} mode")

        try:
            if mode == ProcessingMode.SEQUENTIAL:
                await self._process_sequential()
            elif mode == ProcessingMode.PARALLEL:
                await self._process_parallel()
            elif mode == ProcessingMode.STREAMING:
                await self._process_streaming()
            else:
                await self._process_distributed()
        except Exception as e:
            logger.error(f"Error in async processing: {e}")
        finally:
            self.is_running = False

    def process_tasks_sync(self, mode: ProcessingMode = ProcessingMode.PARALLEL) -> dict[str, ProcessingResult]:
        """Process tasks synchronously."""
        logger.info(f"Starting sync processing in {mode.value} mode")

        if mode == ProcessingMode.SEQUENTIAL:
            return self._process_sequential_sync()
        return self._process_parallel_sync()

    async def _process_sequential(self):
        """Process tasks sequentially."""
        while self.task_queue and self.is_running:
            task = self.task_queue.pop(0)
            result = await self._process_single_task_async(task)
            self.completed_tasks[task.task_id] = result
            await asyncio.sleep(0.01)  # Prevent blocking

    async def _process_parallel(self):
        """Process tasks in parallel."""
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(task):
            async with semaphore:
                return await self._process_single_task_async(task)

        while self.task_queue and self.is_running:
            # Process up to max_workers tasks concurrently
            current_batch = self.task_queue[:self.max_workers]
            self.task_queue = self.task_queue[self.max_workers:]

            if current_batch:
                tasks = [process_with_semaphore(task) for task in current_batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for task, result in zip(current_batch, results, strict=False):
                    if isinstance(result, Exception):
                        result = ProcessingResult(
                            task_id=task.task_id,
                            node_id="error",
                            result=None,
                            processing_time=0.0,
                            success=False,
                            error=str(result)
                        )
                    self.completed_tasks[task.task_id] = result

            await asyncio.sleep(0.1)

    async def _process_streaming(self):
        """Process tasks in streaming mode."""
        async def task_generator():
            while self.task_queue and self.is_running:
                if self.task_queue:
                    yield self.task_queue.pop(0)
                await asyncio.sleep(0.01)

        async for task in task_generator():
            result = await self._process_single_task_async(task)
            self.completed_tasks[task.task_id] = result

            # Yield control to allow other operations
            await asyncio.sleep(0.001)

    async def _process_distributed(self):
        """Process tasks in distributed mode across nodes."""
        while self.task_queue and self.is_running:
            available_nodes = [
                node for node in self.nodes.values()
                if node.status == NodeStatus.IDLE and node.current_load < node.capacity
            ]

            if not available_nodes:
                await asyncio.sleep(0.1)
                continue

            # Assign tasks to available nodes
            tasks_to_process = []
            for node in available_nodes:
                if self.task_queue:
                    task = self.task_queue.pop(0)
                    task.assigned_node = node.node_id
                    node.current_load += 1
                    node.status = NodeStatus.PROCESSING
                    tasks_to_process.append(task)

            if tasks_to_process:
                # Process tasks concurrently
                coroutines = [self._process_task_on_node(task) for task in tasks_to_process]
                await asyncio.gather(*coroutines, return_exceptions=True)

            await asyncio.sleep(0.01)

    async def _process_task_on_node(self, task: ProcessingTask):
        """Process a task on a specific node."""
        node = self.nodes[task.assigned_node]
        start_time = time.time()

        try:
            result = await self._process_single_task_async(task)
            processing_time = time.time() - start_time

            # Update node metrics
            node.processed_count += 1
            node.performance_metrics["avg_processing_time"] = (
                (node.performance_metrics["avg_processing_time"] * (node.processed_count - 1) + processing_time)
                / node.processed_count
            )
            node.performance_metrics["success_rate"] = (
                node.processed_count - node.error_count
            ) / node.processed_count

            self.completed_tasks[task.task_id] = result

        except Exception as e:
            node.error_count += 1
            node.performance_metrics["success_rate"] = (
                node.processed_count - node.error_count
            ) / max(node.processed_count, 1)

            result = ProcessingResult(
                task_id=task.task_id,
                node_id=node.node_id,
                result=None,
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
            self.completed_tasks[task.task_id] = result

        finally:
            node.current_load -= 1
            node.status = NodeStatus.IDLE
            node.last_heartbeat = datetime.now()

    def _process_sequential_sync(self) -> dict[str, ProcessingResult]:
        """Process tasks sequentially in sync mode."""
        results = {}

        while self.task_queue:
            task = self.task_queue.pop(0)
            result = self._process_single_task_sync(task)
            results[task.task_id] = result
            self.completed_tasks[task.task_id] = result

        return results

    def _process_parallel_sync(self) -> dict[str, ProcessingResult]:
        """Process tasks in parallel sync mode."""
        results = {}

        while self.task_queue:
            # Process batch of tasks
            current_batch = self.task_queue[:self.max_workers]
            self.task_queue = self.task_queue[self.max_workers:]

            if current_batch:
                futures = []
                for task in current_batch:
                    future = self.executor.submit(self._process_single_task_sync, task)
                    futures.append((task.task_id, future))

                # Collect results
                for task_id, future in futures:
                    try:
                        result = future.result(timeout=30)  # 30 second timeout
                        results[task_id] = result
                        self.completed_tasks[task_id] = result
                    except Exception as e:
                        error_result = ProcessingResult(
                            task_id=task_id,
                            node_id="error",
                            result=None,
                            processing_time=0.0,
                            success=False,
                            error=str(e)
                        )
                        results[task_id] = error_result
                        self.completed_tasks[task_id] = error_result

        return results

    async def _process_single_task_async(self, task: ProcessingTask) -> ProcessingResult:
        """Process a single task asynchronously."""
        start_time = time.time()

        try:
            if task.task_type not in self.processing_functions:
                raise ValueError(f"No processing function registered for task type: {task.task_type}")

            func = self.processing_functions[task.task_type]

            # Run the function in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, func, task.data)

            processing_time = time.time() - start_time

            return ProcessingResult(
                task_id=task.task_id,
                node_id=task.assigned_node or "default",
                result=result,
                processing_time=processing_time,
                success=True,
                metadata={"task_type": task.task_type}
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing task {task.task_id}: {e}")

            return ProcessingResult(
                task_id=task.task_id,
                node_id=task.assigned_node or "error",
                result=None,
                processing_time=processing_time,
                success=False,
                error=str(e)
            )

    def _process_single_task_sync(self, task: ProcessingTask) -> ProcessingResult:
        """Process a single task synchronously."""
        start_time = time.time()

        try:
            if task.task_type not in self.processing_functions:
                raise ValueError(f"No processing function registered for task type: {task.task_type}")

            func = self.processing_functions[task.task_type]
            result = func(task.data)

            processing_time = time.time() - start_time

            return ProcessingResult(
                task_id=task.task_id,
                node_id="sync_worker",
                result=result,
                processing_time=processing_time,
                success=True,
                metadata={"task_type": task.task_type}
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing task {task.task_id}: {e}")

            return ProcessingResult(
                task_id=task.task_id,
                node_id="error",
                result=None,
                processing_time=processing_time,
                success=False,
                error=str(e)
            )

    def _generate_task_id(self, task_type: str, data: Any) -> str:
        """Generate unique task ID."""
        content = f"{task_type}_{data!s}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def get_status(self) -> dict[str, Any]:
        """Get current system status."""
        total_processed = sum(node.processed_count for node in self.nodes.values())
        total_errors = sum(node.error_count for node in self.nodes.values())

        return {
            "is_running": self.is_running,
            "nodes": len(self.nodes),
            "active_nodes": len([n for n in self.nodes.values() if n.status != NodeStatus.OFFLINE]),
            "queue_size": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "total_processed": total_processed,
            "total_errors": total_errors,
            "success_rate": (total_processed - total_errors) / max(total_processed, 1),
            "node_status": {
                node_id: {
                    "status": node.status.value,
                    "load": f"{node.current_load}/{node.capacity}",
                    "processed": node.processed_count,
                    "errors": node.error_count,
                    "avg_time": node.performance_metrics.get("avg_processing_time", 0.0)
                }
                for node_id, node in self.nodes.items()
            }
        }

    def get_results(self, task_ids: list[str] | None = None) -> dict[str, ProcessingResult]:
        """Get processing results."""
        if task_ids is None:
            return self.completed_tasks.copy()

        return {
            task_id: self.completed_tasks[task_id]
            for task_id in task_ids
            if task_id in self.completed_tasks
        }

    def clear_completed_tasks(self):
        """Clear completed tasks to free memory."""
        self.completed_tasks.clear()
        logger.info("Cleared completed tasks")

    def shutdown(self):
        """Shutdown the distributed architecture."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("DistributedArchitecture shutdown complete")


# Example processing functions
def example_conversation_processor(data: dict[str, Any]) -> dict[str, Any]:
    """Example conversation processing function."""
    conversation = data.get("conversation", {})

    # Simulate processing
    time.sleep(0.1)

    return {
        "conversation_id": conversation.get("id", "unknown"),
        "word_count": len(conversation.get("content", "").split()),
        "processed_at": datetime.now().isoformat(),
        "quality_score": 0.85  # Mock quality score
    }


def example_quality_validator(data: dict[str, Any]) -> dict[str, Any]:
    """Example quality validation function."""
    conversation = data.get("conversation", {})

    # Simulate validation
    time.sleep(0.05)

    return {
        "conversation_id": conversation.get("id", "unknown"),
        "is_valid": True,
        "quality_score": 0.92,
        "validation_notes": ["Good therapeutic content", "Appropriate length"]
    }


def main():
    """Test the distributed architecture."""
    # Initialize architecture
    arch = DistributedArchitecture(max_workers=4)

    # Register processing functions
    arch.register_processing_function("conversation_processing", example_conversation_processor)
    arch.register_processing_function("quality_validation", example_quality_validator)

    # Submit test tasks
    test_conversations = [
        {"id": f"conv_{i}", "content": f"This is test conversation {i} with some therapeutic content."}
        for i in range(10)
    ]

    task_ids = []
    for i, conv in enumerate(test_conversations):
        task_id = arch.submit_task(
            task_type="conversation_processing" if i % 2 == 0 else "quality_validation",
            data={"conversation": conv},
            priority=1 if i < 5 else 2
        )
        task_ids.append(task_id)


    # Process tasks synchronously
    results = arch.process_tasks_sync(ProcessingMode.PARALLEL)

    # Display results

    successful = sum(1 for r in results.values() if r.success)
    len(results) - successful


    if successful > 0:
        sum(r.processing_time for r in results.values() if r.success) / successful

    # Show system status
    arch.get_status()

    # Shutdown
    arch.shutdown()


if __name__ == "__main__":
    main()
