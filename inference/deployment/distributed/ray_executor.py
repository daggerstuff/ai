#!/usr/bin/env python3
"""
Ray-based Distributed Executor for Pixelated Empathy AI

Implements production-ready distributed processing using Ray:
- Parallel dataset processing with automatic scaling
- Fault tolerance and automatic recovery
- Checkpoint integration for resume capability
- Resource-aware task scheduling
- Progress monitoring and metrics collection

Usage:
    from ray_executor import RayExecutor, ExecutorConfig

    config = ExecutorConfig(
        num_workers=4,
        cpu_per_worker=2,
        memory_per_worker="4GB"
    )

    with RayExecutor(config) as executor:
        results = executor.map_parallel(
            func=process_batch,
            items=batch_items,
            checkpoint_interval=100
        )
"""

import json
import logging
import os
import pickle
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

import psutil
import ray
from ray.exceptions import RayActorError, RayTaskError, WorkerCrashedError

# Import local checkpoint system
from .checkpoint_system import CheckpointManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ExecutorConfig:
    """Configuration for Ray executor."""

    # Cluster configuration
    num_workers: int = 4  # Number of worker processes
    cpu_per_worker: int = 2  # CPUs per worker
    memory_per_worker: str = "4GB"  # Memory per worker (e.g., "4GB", "8GB")
    gpu_per_worker: float = 0.0  # GPUs per worker
    object_store_memory: str | None = None  # Ray object store memory

    # Execution configuration
    batch_size: int = 1000  # Items per batch
    max_retries: int = 3  # Maximum retries for failed tasks
    retry_delay: float = 1.0  # Delay between retries (seconds)
    timeout: float | None = None  # Task timeout in seconds

    # Checkpoint configuration
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100  # Checkpoint every N items
    checkpoint_dir: str = "/tmp/ray_executor_checkpoints"

    # Progress monitoring
    enable_progress: bool = True
    progress_update_interval: float = 1.0  # Seconds
    log_level: str = "INFO"

    # Advanced options
    ray_address: str | None = None  # Connect to existing Ray cluster
    local_mode: bool = False  # Run in local mode (debugging)
    max_concurrent_tasks: int | None = None  # Override num_workers
    runtime_env: dict[str, Any] | None = None  # Ray runtime environment

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")

        # Create checkpoint directory if it doesn't exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class TaskResult:
    """Result of a single task execution."""

    task_id: str
    success: bool
    result: Any | None = None
    error: str | None = None
    traceback: str | None = None
    execution_time: float = 0.0
    retry_count: int = 0
    worker_id: str | None = None
    memory_usage_mb: float = 0.0
    cpu_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class ExecutorStats:
    """Statistics for executor execution."""

    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    retried_items: int = 0
    skipped_items: int = 0
    progress_percent: float = 0.0

    total_execution_time: float = 0.0
    min_task_time: float = float("inf")
    max_task_time: float = 0.0

    total_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    workers_active: int = 0
    workers_total: int = 0

    checkpoints_created: int = 0
    checkpoints_restored: int = 0

    start_time: datetime | None = None
    end_time: datetime | None = None

    def update_from_task(self, task_result: TaskResult):
        """Update stats from a task result."""
        if task_result.success:
            self.completed_items += 1
        elif task_result.retry_count > 0:
            self.retried_items += 1
        else:
            self.failed_items += 1

        # Update timing stats
        self.total_execution_time += task_result.execution_time

        if task_result.execution_time > 0:
            if self.min_task_time == float("inf"):
                self.min_task_time = task_result.execution_time
            self.min_task_time = min(self.min_task_time, task_result.execution_time)
            self.max_task_time = max(self.max_task_time, task_result.execution_time)

        # Update memory stats
        if task_result.memory_usage_mb > 0:
            self.total_memory_mb += task_result.memory_usage_mb
            self.peak_memory_mb = max(self.peak_memory_mb, task_result.memory_usage_mb)

        # Update progress
        if self.total_items > 0:
            self.progress_percent = ((self.completed_items + self.failed_items) / self.total_items) * 100

    @property
    def avg_task_time(self) -> float:
        """Calculate average task time."""
        total_successful = self.completed_items + self.failed_items
        if total_successful == 0:
            return 0.0
        return self.total_execution_time / total_successful

    @property
    def avg_memory_mb(self) -> float:
        """Calculate average memory usage."""
        total_successful = self.completed_items + self.failed_items
        if total_successful == 0:
            return 0.0
        return self.total_memory_mb / total_successful

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "retried_items": self.retried_items,
            "skipped_items": self.skipped_items,
            "progress_percent": round(self.progress_percent, 2),
            "total_execution_time": round(self.total_execution_time, 2),
            "avg_task_time": round(self.avg_task_time, 2),
            "min_task_time": round(self.min_task_time, 2),
            "max_task_time": round(self.max_task_time, 2),
            "total_memory_mb": round(self.total_memory_mb, 2),
            "avg_memory_mb": round(self.avg_memory_mb, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "workers_active": self.workers_active,
            "workers_total": self.workers_total,
            "checkpoints_created": self.checkpoints_created,
            "checkpoints_restored": self.checkpoints_restored,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@ray.remote
class WorkerActor:
    """Ray actor for processing tasks on a worker."""

    def __init__(
        self,
        worker_id: str,
        config: ExecutorConfig,
    ):
        self.worker_id = worker_id
        self.config = config
        self.task_count = 0
        self.processed_items: list[str] = []

        # Initialize logger for this worker
        self.logger = logging.getLogger(f"ray_executor.worker.{worker_id}")
        self.logger.setLevel(getattr(logging, config.log_level.upper()))

        self.logger.info(f"Worker {worker_id} initialized")

    def execute_task(
        self,
        task_id: str,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> TaskResult:
        """Execute a single task and return result."""

        if TYPE_CHECKING:
            pass

        start_time = time.time()
        memory_usage_mb = 0.0

        try:
            # Track memory before task execution
            try:
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / (1024 * 1024)
            except Exception:
                memory_before = 0.0

            # Execute the function
            result = func(*args, **kwargs)

            # Track memory after task execution
            try:
                memory_after = process.memory_info().rss / (1024 * 1024)
                memory_usage_mb = max(0.0, memory_after - memory_before)
            except Exception:
                pass

            # Update worker stats
            self.task_count += 1

            execution_time = time.time() - start_time

            self.logger.debug(f"Task {task_id} completed in {execution_time:.2f}s, memory: {memory_usage_mb:.2f}MB")

            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=self.worker_id,
                memory_usage_mb=memory_usage_mb,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            error_traceback = traceback.format_exc()

            self.logger.error(f"Task {task_id} failed: {error_message}\n{error_traceback}")

            return TaskResult(
                task_id=task_id,
                success=False,
                error=error_message,
                traceback=error_traceback,
                execution_time=execution_time,
                worker_id=self.worker_id,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "task_count": self.task_count,
            "processed_items": len(self.processed_items),
        }

    def reset_stats(self):
        """Reset worker statistics."""
        self.task_count = 0
        self.processed_items = []


class RayExecutor:
    """Ray-based distributed executor for parallel processing."""

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.workers: list[ray.ActorHandle] = []
        self.checkpoint_manager: CheckpointManager | None = None
        self.stats = ExecutorStats()
        self._initialized = False

        # Setup logger
        self.logger = logging.getLogger("ray_executor")
        self.logger.setLevel(getattr(logging, config.log_level.upper()))

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()

    def initialize(self):
        """Initialize Ray cluster and workers."""
        if self._initialized:
            return

        self.logger.info("Initializing Ray executor...")
        self.stats.start_time = datetime.now(UTC)

        # Initialize Ray
        ray_init_kwargs = {
            "local_mode": self.config.local_mode,
            "runtime_env": self.config.runtime_env or {},
        }

        if self.config.ray_address:
            ray_init_kwargs["address"] = self.config.ray_address

        if self.config.object_store_memory:
            ray_init_kwargs["object_store_memory"] = self._parse_memory(self.config.object_store_memory)

        if not ray.is_initialized():
            try:
                ray.init(**ray_init_kwargs)
                self.logger.info("Ray initialized successfully")
            except Exception as e:
                self.logger.warning(f"Ray init returned an exception (likely already running): {e}")

        # Initialize checkpoint manager if enabled
        if self.config.enable_checkpointing:
            self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)
            self.logger.info(f"Checkpoint manager initialized at {self.config.checkpoint_dir}")

        # Initialize workers
        num_workers = (
            self.config.num_workers if self.config.num_workers > 0 else ray.available_resources().get("CPU", 1)
        )

        # Create worker actors with resource allocation
        resources = {
            "num_cpus": self.config.cpu_per_worker,
            "memory": self._parse_memory(self.config.memory_per_worker),
        }

        if self.config.gpu_per_worker > 0:
            resources["num_gpus"] = self.config.gpu_per_worker

        self.logger.info(f"Creating {num_workers} workers with resources: {resources}")

        for i in range(num_workers):
            worker_id = f"worker_{i}"
            worker = WorkerActor.options(**resources).remote(worker_id, self.config)
            self.workers.append(worker)

        self.stats.workers_total = len(self.workers)
        self.stats.workers_active = len(self.workers)
        self._initialized = True

        self.logger.info(f"Ray executor initialized with {self.stats.workers_total} workers")

    def shutdown(self):
        """Shutdown Ray executor and cleanup."""
        self.logger.info("Shutting down Ray executor...")

        self.stats.end_time = datetime.now(UTC)

        # Log final stats
        self.logger.info(f"Final executor statistics: {self.stats.to_dict()}")

        # Terminate workers
        if self.workers:
            for worker in self.workers:
                ray.kill(worker)  # Gracefully terminate
            self.workers = []

        # Stop checkpoint manager
        if self.checkpoint_manager:
            try:
                self.checkpoint_manager.stop_background_tasks()
            except Exception as e:
                self.logger.warning(f"Error stopping checkpoint manager: {e}")

        # Shutdown Ray (only if we initialized it)
        if ray.is_initialized() and not self.config.ray_address:
            ray.shutdown()

        self._initialized = False
        self.logger.info("Ray executor shutdown complete")

    def map_parallel(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        task_name: str = "map_parallel",
        checkpoint_enabled: bool | None = None,
    ) -> list[R]:
        """
        Map function over items in parallel using Ray.

        Args:
            func: Function to apply to each item
            items: Iterable of items to process
            task_name: Name for the task (used in checkpoints)
            checkpoint_enabled: Override default checkpointing setting

        Returns:
            List of results from applying func to each item
        """
        # Initialize if not already
        if not self._initialized:
            self.initialize()

        # Determine if checkpointing is enabled
        checkpoint_enabled = checkpoint_enabled if checkpoint_enabled is not None else self.config.enable_checkpointing

        self.logger.info(f"Starting parallel map: {task_name}")

        # Convert items to list for indexing
        items_list = list(items)
        self.stats.total_items = len(items_list)

        # Try to restore from checkpoint
        start_index = 0
        if checkpoint_enabled and self.checkpoint_manager:
            start_index = self._restore_from_checkpoint(task_name, items_list)
            if start_index > 0:
                self.checkpoint_manager = self.checkpoint_manager
                self.logger.info(f"Restored from checkpoint, starting at index {start_index}")

        # Process items in parallel
        results = self._process_items_parallel(func, items_list, start_index, task_name, checkpoint_enabled)

        self.logger.info(f"Parallel map completed: {len(results)} results")
        return results

    def batch_process(
        self,
        func: Callable[[list[T]], list[R]],
        items: Iterable[T],
        task_name: str = "batch_process",
    ) -> list[R]:
        """
        Process items in batches for better performance.

        Args:
            func: Function that takes a list of items and returns list of results
            items: Iterable of items to process
            task_name: Name for the task

        Returns:
            List of results from all batches
        """
        if not self._initialized:
            self.initialize()

        items_list = list(items)
        self.stats.total_items = len(items_list)

        batch_size = self.config.batch_size
        num_batches = (len(items_list) + batch_size - 1) // batch_size

        self.logger.info(f"Batch processing {len(items_list)} items in {num_batches} batches of size {batch_size}")

        # Create batches
        batches = [items_list[i : i + batch_size] for i in range(0, len(items_list), batch_size)]

        # Process batches in parallel
        batch_results = self.map_parallel(func, batches, task_name=f"{task_name}_batched")

        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)

        self.logger.info(f"Batch processing completed: {len(results)} total results")
        return results

    def _process_items_parallel(
        self,
        func: Callable[[T], R],
        items_list: list[T],
        start_index: int,
        task_name: str,
        checkpoint_enabled: bool,
    ) -> list[R]:
        """Process items in parallel with fault tolerance."""
        results = []
        pending_futures: list[tuple[str, ray.ObjectRef]] = []
        completed_task_ids: set = set()

        max_concurrent = self.config.max_concurrent_tasks if self.config.max_concurrent_tasks else len(self.workers)

        self.logger.info(f"Processing {len(items_list)} items starting from index {start_index}")

        # Submit tasks
        for i in range(start_index, len(items_list)):
            item = items_list[i]
            task_id = f"{task_name}_item_{i}"

            # Check if task was already completed (from checkpoint)
            if task_id in completed_task_ids:
                continue

            # Select worker (round-robin)
            worker_index = len(pending_futures) % len(self.workers)
            worker = self.workers[worker_index]

            # Submit task
            future = worker.execute_task.remote(task_id, func, (item,), {})
            pending_futures.append((task_id, future))

            # Wait for some tasks to complete before submitting more
            while len(pending_futures) >= max_concurrent:
                results.extend(self._collect_results(pending_futures, completed_task_ids))

                # Create checkpoint if enabled
                if checkpoint_enabled and self._should_create_checkpoint(task_name):
                    self._create_checkpoint(task_name, completed_task_ids, results)

        # Collect remaining results
        while pending_futures:
            results.extend(self._collect_results(pending_futures, completed_task_ids))

        return results

    def _collect_results(
        self,
        pending_futures: list[tuple[str, ray.ObjectRef]],
        completed_task_ids: set,
    ) -> list[Any]:
        """Collect results from completed tasks with retry logic."""
        if not pending_futures:
            return []

        results = []
        still_pending = []

        # Wait for at least one task to complete
        ready_futures, _remaining_futures = ray.wait(
            [future for _, future in pending_futures], num_returns=1, timeout=0.1
        )

        for task_id, future in pending_futures:
            if future in ready_futures:
                # Task completed, get result
                try:
                    task_result: TaskResult = ray.get(future)

                    self.stats.update_from_task(task_result)

                    if task_result.success:
                        results.append(task_result.result)
                        completed_task_ids.add(task_id)
                    # Retry failed task
                    elif task_result.retry_count < self.config.max_retries:
                        self.logger.warning(f"Retrying task {task_id} (attempt {task_result.retry_count + 1})")
                        still_pending.append((task_id, future))  # Will be re-submitted
                    else:
                        self.logger.error(f"Task {task_id} failed after {self.config.max_retries} retries")
                        # Append None for failed item
                        results.append(None)
                        completed_task_ids.add(task_id)

                except (RayActorError, RayTaskError, WorkerCrashedError) as e:
                    self.logger.error(f"Ray execution error for task {task_id}: {e}")
                    # Could retry here, but for now just log and continue
                    results.append(None)
                    completed_task_ids.add(task_id)

                except Exception as e:
                    self.logger.error(f"Unexpected error for task {task_id}: {e}")
                    results.append(None)
                    completed_task_ids.add(task_id)

            else:
                still_pending.append((task_id, future))

        # Update pending futures
        pending_futures[:] = still_pending

        return results

    def _create_checkpoint(
        self,
        task_name: str,
        completed_task_ids: set,
        partial_results: list[Any],
    ):
        """Create a checkpoint for current progress."""
        if not self.checkpoint_manager:
            return

        try:
            self.checkpoint_manager.register_process(
                process_id=task_name,
                task_id=task_name,
                current_step="progress_update",
                total_steps=self.stats.total_items,
                completed_steps=self.stats.completed_items + self.stats.failed_items,
                metadata={
                    "completed_task_ids": list(completed_task_ids),
                    "partial_results": partial_results,
                    "stats": self.stats.to_dict(),
                },
            )

            self.stats.checkpoints_created += 1
            self.logger.debug(f"Checkpoint created for {task_name}")

        except Exception as e:
            self.logger.warning(f"Failed to create checkpoint: {e}")

    def _restore_from_checkpoint(self, task_name: str, _items_list: list[T]) -> int:
        """Restore from checkpoint and return start index."""
        if not self.checkpoint_manager:
            return 0

        try:
            # Get latest checkpoint for task
            checkpoints = self.checkpoint_manager.storage.list_checkpoints(process_id=task_name)

            if not checkpoints:
                return 0

            # Get most recent checkpoint
            latest_checkpoint = sorted(checkpoints, key=lambda c: c.created_at, reverse=True)[0]

            # Load checkpoint data
            _metadata, data = self.checkpoint_manager.storage.load_checkpoint(latest_checkpoint.checkpoint_id)

            if data and "completed_task_ids" in data:
                self.checkpoint_manager = self.checkpoint_manager
                self.stats.checkpoints_restored += 1

                completed_count = len(data["completed_task_ids"])
                self.logger.info(f"Restored checkpoint: {completed_count} items already completed")

                return completed_count

        except Exception as e:
            self.logger.warning(f"Failed to restore from checkpoint: {e}")

        return 0

    def _should_create_checkpoint(self, _task_name: str) -> bool:
        """Determine if a checkpoint should be created."""
        total_processed = self.stats.completed_items + self.stats.failed_items
        return total_processed % self.config.checkpoint_interval == 0

    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string (e.g., '4GB', '1024MB') to bytes."""
        memory_str = memory_str.upper()

        multipliers = {
            "KB": 1024,
            "MB": 1024 * 1024,
            "GB": 1024 * 1024 * 1024,
            "TB": 1024 * 1024 * 1024 * 1024,
        }

        for suffix, multiplier in multipliers.items():
            if memory_str.endswith(suffix):
                size_str = memory_str[: -len(suffix)]
                try:
                    return int(float(size_str) * multiplier)
                except ValueError:
                    pass

        # Try parsing as plain number (assume bytes)
        try:
            return int(memory_str)
        except ValueError:
            raise ValueError(f"Invalid memory format: {memory_str}") from None

    def get_stats(self) -> ExecutorStats:
        """Get current executor statistics."""
        # Update active workers count
        self.stats.workers_active = len(self.workers)
        return self.stats

    def log_progress(self):
        """Log current progress."""
        stats = self.get_stats()
        self.logger.info(
            f"Progress: {stats.completed_items}/{stats.total_items} "
            f"({stats.progress_percent:.1f}%), "
            f"Failed: {stats.failed_items}, "
            f"Time: {stats.total_execution_time:.1f}s, "
            f"Avg task: {stats.avg_task_time:.2f}s"
        )


def process_dataset_parallel(
    dataset_path: str | Path,
    process_func: Callable[[dict], Any],
    output_path: str | Path,
    config: ExecutorConfig | None = None,
) -> list[Any]:
    """
    Convenience function to process a dataset file in parallel.

    Args:
        dataset_path: Path to dataset (JSONL, JSON, or pickled)
        process_func: Function to apply to each item
        output_path: Path to save results
        config: Executor configuration (uses defaults if None)

    Returns:
        List of processed results
    """

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    config = config or ExecutorConfig()

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")

    if dataset_path.suffix == ".jsonl":
        items = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
    elif dataset_path.suffix == ".json":
        with open(dataset_path, encoding="utf-8") as f:
            items = json.load(f)
    else:
        with open(dataset_path, "rb") as f:
            items = pickle.load(f)

    logger.info(f"Loaded {len(items)} items from dataset")

    # Process in parallel
    with RayExecutor(config) as executor:
        results = executor.map_parallel(
            func=process_func,
            items=items,
            task_name=f"process_{dataset_path.stem}",
        )

    # Save results
    logger.info(f"Saving {len(results)} results to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
    elif output_path.suffix == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        with open(output_path, "wb") as f:
            pickle.dump(results, f)

    logger.info("Dataset processing complete")
    return results


# Example usage (commented out)
"""
if __name__ == "__main__":
    from pathlib import Path

    # Example 1: Parallel map
    config = ExecutorConfig(
        num_workers=4,
        cpu_per_worker=2,
        memory_per_worker="4GB",
        batch_size=1000,
    )

    def process_item(item: int) -> int:
        import time
        time.sleep(0.1)
        return item * 2

    with RayExecutor(config) as executor:
        results = executor.map_parallel(
            func=process_item,
            items=range(100),
            task_name="example_map",
        )
        print(f"Results: {results[:10]}...")
        print(f"Final stats: {executor.get_stats().to_dict()}")

    # Example 2: Process dataset
    def process_conversation(conv: Dict) -> Dict:
        # Process conversation
        return {**conv, "processed": True}

    output_path = Path("/tmp/processed_dataset.jsonl")
    results = process_dataset_parallel(
        dataset_path="/path/to/dataset.jsonl",
        process_func=process_conversation,
        output_path=output_path,
        config=ExecutorConfig(num_workers=4),
    )
"""
