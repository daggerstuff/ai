"""
Task Orchestration System for MCP Server.

This module implements task delegation, assignment, and tracking across agents,
with built-in error handling and real-time progress updates.
"""

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# Internal imports
from ai.inference.api.mcp_server.integration.mongodb_client import MCPMongoDBClient
from ai.inference.api.mcp_server.integration.redis_client import MCPRedisClient

from .agent_manager import Agent, AgentDiscoveryCriteria, AgentManager, AgentStatus

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Enumeration of task lifecycle statuses."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskCreationData:
    """Data required for task creation."""

    pipeline_id: str
    stage: str
    parameters: dict[str, Any]
    required_capabilities: list[str]
    priority: int = 1


@dataclass
class Task:
    """Task instance model."""

    id: str
    pipeline_id: str
    stage: str
    parameters: dict[str, Any]
    required_capabilities: list[str]
    priority: int
    status: TaskStatus
    created_at: datetime
    assigned_at: datetime | None = None
    completed_at: datetime | None = None
    agent_id: str | None = None
    progress: float = 0.0
    error: str | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary for JSON serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        if self.assigned_at:
            data["assigned_at"] = self.assigned_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data


class TaskQueue:
    """Handles task prioritizing and queuing logic."""

    def __init__(self, mongodb_client: MCPMongoDBClient):
        self.mongodb = mongodb_client
        self.collection = "tasks"

    async def enqueue(self, task: Task) -> None:
        """Add task to persistent queue."""
        await self.mongodb.insert_one(self.collection, task.to_dict())
        logger.debug(f"Task {task.id} enqueued")

    async def get_pending_tasks(self) -> list[Task]:
        """Retrieve all pending tasks ordered by priority."""
        cursor = await self.mongodb.find_many(
            self.collection, {"status": TaskStatus.PENDING.value}, sort=[("priority", -1), ("created_at", 1)]
        )
        return [self._from_dict(d) for d in cursor]

    async def update_task(self, task: Task) -> None:
        """Update task in persistence."""
        await self.mongodb.update_one(self.collection, {"id": task.id}, {"$set": task.to_dict()})
        logger.debug(f"Task {task.id} updated in queue")

    def _from_dict(self, data: dict[str, Any]) -> Task:
        """Create Task instance from dictionary."""
        return Task(
            id=data.get("id", str(uuid.uuid4())),
            pipeline_id=data.get("pipeline_id", ""),
            stage=data.get("stage", ""),
            parameters=data.get("parameters", {}),
            required_capabilities=data.get("required_capabilities", []),
            priority=data.get("priority", 1),
            status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now(UTC).isoformat())),
            assigned_at=(datetime.fromisoformat(data["assigned_at"]) if data.get("assigned_at") else None),
            completed_at=(datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None),
            agent_id=data.get("agent_id"),
            progress=data.get("progress", 0.0),
            error=data.get("error"),
            result=data.get("result"),
        )


class TaskAssigner:
    """Handles logic for matching tasks to appropriate agents."""

    async def assign_agent(self, task: Task, agents: list[Agent]) -> Agent | None:
        """Select best agent for task based on requirements and load."""
        # Baseline assignment logic: find first active agent with required capabilities
        for agent in agents:
            if agent.status == AgentStatus.ACTIVE:
                # All required capabilities must be present
                if all(cap in agent.capabilities for cap in task.required_capabilities):
                    logger.info(f"Assigning task {task.id} to agent {agent.id}")
                    return agent
        return None


class ProgressTracker:
    """Manages real-time task progress monitoring."""

    def __init__(self, redis_client: MCPRedisClient):
        self.redis = redis_client

    async def notify_progress(self, task_id: str, progress: float, status: TaskStatus) -> None:
        """Propagate progress updates to system observers."""
        await self.redis.publish(
            "task.progress",
            {
                "task_id": task_id,
                "progress": progress,
                "status": status.value,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        logger.debug(f"Progress update for task {task_id}: {progress}%")


class TaskOrchestrator:
    """Orchestrate task allocation and tracking."""

    def __init__(
        self,
        agent_manager: AgentManager,
        redis_client: MCPRedisClient,
        mongodb_client: MCPMongoDBClient,
    ):
        self.agent_manager = agent_manager
        self.redis = redis_client
        self.mongodb = mongodb_client
        self.task_queue = TaskQueue(mongodb_client)
        self.task_assigner = TaskAssigner()
        self.progress_tracker = ProgressTracker(redis_client)

    async def create_task(self, task_data: TaskCreationData) -> Task:
        """Create and enqueue task."""
        task = Task(
            id=str(uuid.uuid4()),
            pipeline_id=task_data.pipeline_id,
            stage=task_data.stage,
            parameters=task_data.parameters,
            required_capabilities=task_data.required_capabilities,
            priority=task_data.priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        await self.task_queue.enqueue(task)

        # Trigger assignment asynchronously
        asyncio.create_task(self.assign_task(task.id))

        return task

    async def assign_task(self, task_id: str) -> bool:
        """Assign task to appropriate agent."""
        task_data = await self.mongodb.find_one("tasks", {"id": task_id})
        if not task_data:
            return False

        task = self.task_queue._from_dict(task_data)
        if task.status != TaskStatus.PENDING:
            return False

        # Discover suitable agents
        criteria = AgentDiscoveryCriteria(capabilities=task.required_capabilities, status=AgentStatus.ACTIVE)
        agents = await self.agent_manager.discover_agents(criteria)

        # Assign agent
        agent = await self.task_assigner.assign_agent(task, agents)
        if not agent:
            logger.info(f"No suitable agent found for task {task_id}, remaining in pending state")
            return False

        # Update task status
        task.status = TaskStatus.ASSIGNED
        task.agent_id = agent.id
        task.assigned_at = datetime.now(UTC)

        await self.task_queue.update_task(task)

        # Update agent status to BUSY
        await self.agent_manager.update_agent_status(agent.id, AgentStatus.BUSY)

        # Notify assigned agent via Redis
        await self.redis.publish(f"agent.{agent.id}.tasks", {"action": "new_task", "task": task.to_dict()})

        return True

    async def update_task_progress(
        self,
        task_id: str,
        progress: float,
        status: TaskStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        """Update progress and publish events."""
        task_data = await self.mongodb.find_one("tasks", {"id": task_id})
        if not task_data:
            return False

        task = self.task_queue._from_dict(task_data)

        # Update task fields
        task.progress = progress
        task.status = status
        if result:
            task.result = result
        if error:
            task.error = error

        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.completed_at = datetime.now(UTC)

            # Free up agent if one was assigned
            if task.agent_id:
                active_status = AgentStatus.ACTIVE
                await self.agent_manager.update_agent_status(task.agent_id, active_status)

        await self.task_queue.update_task(task)

        # Notify progress
        await self.progress_tracker.notify_progress(task_id, progress, status)

        return True
