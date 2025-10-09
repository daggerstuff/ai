#!/usr/bin/env python3
"""
Task 6.1: Distributed Processing Architecture for 6-Tier Data Ecosystem

This module implements a distributed processing architecture for handling the complete
mental health data ecosystem across 6 tiers: Priority → Professional → CoT → Reddit → Research → Knowledge Base.

Strategic Goal: Process 2.59M+ conversations with intelligent load balancing,
fault tolerance, and hierarchical quality management.
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class DataTier(Enum):
    """6-tier data ecosystem hierarchy."""
    TIER_1_PRIORITY = "priority"          # Gold standard (40% weight)
    TIER_2_PROFESSIONAL = "professional"  # Clinical grade (25% weight)
    TIER_3_COT = "cot_reasoning"          # Advanced reasoning (20% weight)
    TIER_4_REDDIT = "reddit"              # Real-world data (10% weight)
    TIER_5_RESEARCH = "research"          # Academic datasets (4% weight)
    TIER_6_KNOWLEDGE = "knowledge_base"   # Reference materials (1% weight)


@dataclass
class ProcessingNode:
    """Represents a processing node in the distributed system."""
    node_id: str
    tier: DataTier
    capacity: int
    current_load: int = 0
    status: str = "idle"  # idle, processing, overloaded, error
    last_heartbeat: datetime = None
    processed_count: int = 0
    error_count: int = 0

    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now()


@dataclass
class ProcessingTask:
    """Represents a processing task in the distributed system."""
    task_id: str
    tier: DataTier
    dataset_name: str
    data_path: str
    priority: int  # 1-6 based on tier
    estimated_size: int
    assigned_node: str | None = None
    status: str = "pending"  # pending, assigned, processing, completed, failed
    created_at: datetime = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class EcosystemMetrics:
    """System-wide metrics for the distributed ecosystem."""
    total_conversations: int = 0
    processed_conversations: int = 0
    processing_rate: float = 0.0
    active_nodes: int = 0
    tier_distribution: dict[str, int] = None
    quality_scores: dict[str, float] = None
    error_rate: float = 0.0
    system_health: str = "healthy"

    def __post_init__(self):
        if self.tier_distribution is None:
            self.tier_distribution = {tier.value: 0 for tier in DataTier}
        if self.quality_scores is None:
            self.quality_scores = {tier.value: 0.0 for tier in DataTier}


class DistributedProcessingCoordinator:
    """Coordinates distributed processing across the 6-tier ecosystem."""

    def __init__(self, config_path: str = "ecosystem_config.json"):
        self.config = self._load_config(config_path)

        # Node management
        self.processing_nodes: dict[str, ProcessingNode] = {}
        self.task_queue = deque()
        self.completed_tasks = deque(maxlen=10000)

        # Tier configuration
        self.tier_weights = {
            DataTier.TIER_1_PRIORITY: 0.40,
            DataTier.TIER_2_PROFESSIONAL: 0.25,
            DataTier.TIER_3_COT: 0.20,
            DataTier.TIER_4_REDDIT: 0.10,
            DataTier.TIER_5_RESEARCH: 0.04,
            DataTier.TIER_6_KNOWLEDGE: 0.01
        }

        # Processing executors
        self.thread_executor = ThreadPoolExecutor(max_workers=self.config.get("max_threads", 8))
        self.process_executor = ProcessPoolExecutor(max_workers=self.config.get("max_processes", 4))

        # Metrics and monitoring
        self.metrics = EcosystemMetrics()
        self.processing_history = deque(maxlen=1000)

        # System state
        self.is_running = False
        self.coordinator_lock = threading.Lock()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load distributed processing configuration."""
        default_config = {
            "max_threads": 8,
            "max_processes": 4,
            "node_capacity": 1000,
            "heartbeat_interval": 30,
            "task_timeout": 3600,
            "quality_threshold": 0.6,
            "tier_priorities": {
                "priority": 1,
                "professional": 2,
                "cot_reasoning": 3,
                "reddit": 4,
                "research": 5,
                "knowledge_base": 6
            },
            "load_balancing": {
                "algorithm": "weighted_round_robin",
                "enable_auto_scaling": True,
                "max_load_factor": 0.8
            }
        }

        try:
            if Path(config_path).exists():
                with open(config_path) as f:
                    config = json.load(f)
                    default_config.update(config)
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")

        return default_config

    def register_processing_node(self, node_id: str, tier: DataTier, capacity: int | None = None) -> bool:
        """Register a new processing node."""
        if capacity is None:
            capacity = self.config.get("node_capacity", 1000)

        node = ProcessingNode(
            node_id=node_id,
            tier=tier,
            capacity=capacity
        )

        with self.coordinator_lock:
            self.processing_nodes[node_id] = node
            self.metrics.active_nodes = len(self.processing_nodes)

        self.logger.info(f"Registered processing node: {node_id} (tier: {tier.value}, capacity: {capacity})")
        return True

    def submit_processing_task(self, dataset_name: str, data_path: str, tier: DataTier,
                             estimated_size: int | None = None) -> str:
        """Submit a processing task to the distributed system."""
        task_id = f"task_{tier.value}_{int(time.time() * 1000)}"

        if estimated_size is None:
            estimated_size = self._estimate_dataset_size(data_path)

        task = ProcessingTask(
            task_id=task_id,
            tier=tier,
            dataset_name=dataset_name,
            data_path=data_path,
            priority=self.config["tier_priorities"][tier.value],
            estimated_size=estimated_size
        )

        # Add to priority queue (lower priority number = higher priority)
        self.task_queue.append(task)
        self._sort_task_queue()

        self.logger.info(f"Submitted task: {task_id} (dataset: {dataset_name}, tier: {tier.value})")
        return task_id

    def _sort_task_queue(self):
        """Sort task queue by priority and tier weights."""
        self.task_queue = deque(sorted(
            self.task_queue,
            key=lambda t: (t.priority, -self.tier_weights[t.tier], t.created_at)
        ))

    def _estimate_dataset_size(self, data_path: str) -> int:
        """Estimate dataset size for load balancing."""
        try:
            path = Path(data_path)
            if path.exists():
                if path.is_file():
                    # Estimate conversations based on file size
                    file_size_mb = path.stat().st_size / (1024 * 1024)
                    return int(file_size_mb * 100)  # Rough estimate: 100 conversations per MB
                if path.is_dir():
                    # Count files in directory
                    return len(list(path.glob("**/*")))
        except Exception as e:
            self.logger.warning(f"Could not estimate size for {data_path}: {e}")

        return 1000  # Default estimate

    def assign_task_to_node(self, task: ProcessingTask) -> str | None:
        """Assign a task to the best available node using load balancing."""
        best_node = None
        best_score = float("inf")

        # Find nodes that can handle this tier
        eligible_nodes = [
            node for node in self.processing_nodes.values()
            if (node.tier == task.tier and
                node.status in ["idle", "processing"] and
                node.current_load < node.capacity * self.config["load_balancing"]["max_load_factor"])
        ]

        if not eligible_nodes:
            return None

        # Load balancing algorithm
        algorithm = self.config["load_balancing"]["algorithm"]

        if algorithm == "weighted_round_robin":
            # Consider load, capacity, and recent performance
            for node in eligible_nodes:
                load_factor = node.current_load / node.capacity
                error_rate = node.error_count / max(node.processed_count, 1)

                # Lower score is better
                score = load_factor * 0.6 + error_rate * 0.4

                if score < best_score:
                    best_score = score
                    best_node = node

        elif algorithm == "least_loaded":
            best_node = min(eligible_nodes, key=lambda n: n.current_load)

        elif algorithm == "round_robin":
            # Simple round-robin based on processed count
            best_node = min(eligible_nodes, key=lambda n: n.processed_count)

        if best_node:
            # Assign task to node
            task.assigned_node = best_node.node_id
            task.status = "assigned"
            best_node.current_load += task.estimated_size

            return best_node.node_id

        return None

    async def start_coordinator(self):
        """Start the distributed processing coordinator."""
        self.is_running = True
        self.logger.info("Starting distributed processing coordinator...")

        # Start coordinator tasks
        tasks = [
            asyncio.create_task(self._task_scheduler()),
            asyncio.create_task(self._node_monitor()),
            asyncio.create_task(self._metrics_updater()),
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Coordinator error: {e}")
        finally:
            await self.stop_coordinator()

    async def stop_coordinator(self):
        """Stop the distributed processing coordinator."""
        self.is_running = False
        self.logger.info("Stopping distributed processing coordinator...")

        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

        self.logger.info("Coordinator stopped")

    async def _task_scheduler(self):
        """Schedule tasks to available nodes."""
        while self.is_running:
            try:
                if self.task_queue:
                    task = self.task_queue.popleft()

                    # Try to assign task to a node
                    assigned_node = self.assign_task_to_node(task)

                    if assigned_node:
                        # Start task processing
                        await self._start_task_processing(task)
                    else:
                        # No available nodes, put task back in queue
                        self.task_queue.appendleft(task)
                        await asyncio.sleep(5)  # Wait before retrying

                await asyncio.sleep(1)  # Check for new tasks every second

            except Exception as e:
                self.logger.error(f"Task scheduler error: {e}")

    async def _start_task_processing(self, task: ProcessingTask):
        """Start processing a task on the assigned node."""
        try:
            task.status = "processing"
            task.started_at = datetime.now()

            # Submit task to thread pool for processing
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.thread_executor,
                self._process_task,
                task
            )

        except Exception as e:
            self.logger.error(f"Error starting task {task.task_id}: {e}")
            task.status = "failed"
            self._update_node_after_task(task, success=False)

    def _process_task(self, task: ProcessingTask):
        """Process a task (runs in thread pool)."""
        try:
            self.logger.info(f"Processing task: {task.task_id} on node: {task.assigned_node}")

            # Simulate processing (in real implementation, this would call actual processing)
            processing_time = min(task.estimated_size / 100, 60)  # Max 60 seconds
            time.sleep(processing_time)

            # Mark task as completed
            task.status = "completed"
            task.completed_at = datetime.now()

            # Update metrics
            self.metrics.processed_conversations += task.estimated_size
            self.metrics.tier_distribution[task.tier.value] += task.estimated_size

            # Add to completed tasks
            self.completed_tasks.append(task)

            # Update node status
            self._update_node_after_task(task, success=True)

            self.logger.info(f"Completed task: {task.task_id}")

        except Exception as e:
            self.logger.error(f"Error processing task {task.task_id}: {e}")
            task.status = "failed"
            self._update_node_after_task(task, success=False)

    def _update_node_after_task(self, task: ProcessingTask, success: bool):
        """Update node status after task completion."""
        if task.assigned_node and task.assigned_node in self.processing_nodes:
            node = self.processing_nodes[task.assigned_node]

            # Update load
            node.current_load = max(0, node.current_load - task.estimated_size)

            # Update counters
            if success:
                node.processed_count += 1
            else:
                node.error_count += 1

            # Update status
            if node.current_load == 0:
                node.status = "idle"
            elif node.current_load < node.capacity * 0.8:
                node.status = "processing"
            else:
                node.status = "overloaded"

    async def _node_monitor(self):
        """Monitor node health and performance."""
        heartbeat_interval = self.config.get("heartbeat_interval", 30)

        while self.is_running:
            try:
                current_time = datetime.now()

                for node_id, node in self.processing_nodes.items():
                    # Check for stale nodes
                    time_since_heartbeat = (current_time - node.last_heartbeat).total_seconds()

                    if time_since_heartbeat > heartbeat_interval * 2:
                        node.status = "error"
                        self.logger.warning(f"Node {node_id} appears to be unresponsive")

                    # Update heartbeat
                    node.last_heartbeat = current_time

                await asyncio.sleep(heartbeat_interval)

            except Exception as e:
                self.logger.error(f"Node monitor error: {e}")

    async def _metrics_updater(self):
        """Update system-wide metrics."""
        while self.is_running:
            try:
                # Calculate processing rate
                if self.processing_history:
                    recent_completions = len([
                        task for task in self.completed_tasks
                        if task.completed_at and
                        (datetime.now() - task.completed_at).total_seconds() < 60
                    ])
                    self.metrics.processing_rate = recent_completions

                # Update active nodes count
                self.metrics.active_nodes = len([
                    node for node in self.processing_nodes.values()
                    if node.status != "error"
                ])

                # Calculate error rate
                total_tasks = len(self.completed_tasks)
                failed_tasks = len([task for task in self.completed_tasks if task.status == "failed"])
                self.metrics.error_rate = failed_tasks / max(total_tasks, 1)

                # Determine system health
                if self.metrics.error_rate > 0.1:
                    self.metrics.system_health = "critical"
                elif self.metrics.error_rate > 0.05 or self.metrics.active_nodes < 2:
                    self.metrics.system_health = "warning"
                else:
                    self.metrics.system_health = "healthy"

                await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                self.logger.error(f"Metrics updater error: {e}")

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "metrics": asdict(self.metrics),
            "nodes": {
                node_id: asdict(node)
                for node_id, node in self.processing_nodes.items()
            },
            "queue_status": {
                "pending_tasks": len(self.task_queue),
                "completed_tasks": len(self.completed_tasks),
                "processing_tasks": len([
                    task for task in self.completed_tasks
                    if task.status == "processing"
                ])
            },
            "tier_progress": {
                tier.value: {
                    "processed": self.metrics.tier_distribution.get(tier.value, 0),
                    "weight": self.tier_weights[tier],
                    "target_percentage": self.tier_weights[tier] * 100
                }
                for tier in DataTier
            }
        }


# Example usage and testing
async def main():
    """Example usage of the distributed processing coordinator."""

    # Create coordinator
    coordinator = DistributedProcessingCoordinator()

    # Register processing nodes for each tier
    coordinator.register_processing_node("priority_node_1", DataTier.TIER_1_PRIORITY, 2000)
    coordinator.register_processing_node("priority_node_2", DataTier.TIER_1_PRIORITY, 2000)
    coordinator.register_processing_node("professional_node_1", DataTier.TIER_2_PROFESSIONAL, 1500)
    coordinator.register_processing_node("cot_node_1", DataTier.TIER_3_COT, 1000)
    coordinator.register_processing_node("reddit_node_1", DataTier.TIER_4_REDDIT, 3000)
    coordinator.register_processing_node("research_node_1", DataTier.TIER_5_RESEARCH, 500)
    coordinator.register_processing_node("knowledge_node_1", DataTier.TIER_6_KNOWLEDGE, 200)

    # Submit sample tasks
    coordinator.submit_processing_task("priority_1_dataset", "data/priority/priority_1.jsonl",
                                     DataTier.TIER_1_PRIORITY, 10000)
    coordinator.submit_processing_task("psych8k_dataset", "data/professional/psych8k.json",
                                     DataTier.TIER_2_PROFESSIONAL, 8000)
    coordinator.submit_processing_task("cot_clinical_dataset", "data/cot/clinical_diagnosis.json",
                                     DataTier.TIER_3_COT, 5000)

    # Start coordinator (run for 30 seconds for demo)
    try:
        coordinator_task = asyncio.create_task(coordinator.start_coordinator())
        await asyncio.sleep(30)
        coordinator_task.cancel()
    except asyncio.CancelledError:
        pass

    # Get final status
    status = coordinator.get_system_status()

    for _tier, _progress in status["tier_progress"].items():
        pass


if __name__ == "__main__":
    asyncio.run(main())
