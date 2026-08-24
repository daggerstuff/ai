"""
Agent Management System for MCP Server.

This module implements the core agent management logic, including registration,
discovery, and lifecycle management, with sub-50ms performance targets.
"""

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# Internal imports
from ai.inference.api.mcp_server.integration.mongodb_client import MCPMongoDBClient
from ai.inference.api.mcp_server.integration.redis_client import MCPRedisClient

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Enumeration of agent lifecycle statuses."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class AgentRegistrationData:
    """Data required for agent registration."""

    name: str
    type: str
    capabilities: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDiscoveryCriteria:
    """Criteria for discovering agents based on capabilities."""

    capabilities: list[str] | None = None
    agent_type: str | None = None
    status: AgentStatus | None = None

    @classmethod
    def from_request(cls, args: dict[str, str]) -> "AgentDiscoveryCriteria":
        """Create criteria from request arguments."""
        capabilities = args.get("capabilities", "").split(",") if args.get("capabilities") else None
        agent_type = args.get("type")
        status_val = args.get("status")
        status = AgentStatus(status_val) if status_val else None
        return cls(capabilities=capabilities, agent_type=agent_type, status=status)


@dataclass
class Agent:
    """Agent instance model."""

    id: str
    name: str
    type: str
    capabilities: list[str]
    status: AgentStatus
    registered_at: datetime
    last_seen: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert agent to dictionary for JSON serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        data["registered_at"] = self.registered_at.isoformat()
        data["last_seen"] = self.last_seen.isoformat()
        return data


class AgentRegistry:
    """In-memory and persistent registry for agents."""

    def __init__(self, mongodb_client: MCPMongoDBClient):
        self.mongodb = mongodb_client
        self.collection = "agents"
        self._local_cache: dict[str, Agent] = {}

    async def add_agent(self, agent: Agent) -> None:
        """Add agent to registry and persistence."""
        self._local_cache[agent.id] = agent
        await self.mongodb.insert_one(self.collection, agent.to_dict())
        logger.debug(f"Agent {agent.id} added to registry")

    async def get_agent(self, agent_id: str) -> Agent | None:
        """Retrieve agent from registry."""
        if agent_id in self._local_cache:
            return self._local_cache[agent_id]

        # Fallback to DB
        data = await self.mongodb.find_one(self.collection, {"id": agent_id})
        if data:
            agent = self._from_dict(data)
            self._local_cache[agent.id] = agent
            return agent
        return None

    async def update_agent(self, agent: Agent) -> None:
        """Update agent in registry and persistence."""
        self._local_cache[agent.id] = agent
        await self.mongodb.update_one(self.collection, {"id": agent.id}, {"$set": agent.to_dict()})
        logger.debug(f"Agent {agent.id} updated in registry")

    async def find_agents(self, criteria: AgentDiscoveryCriteria) -> list[Agent]:
        """Find agents matching criteria."""
        query = {}
        if criteria.agent_type:
            query["type"] = criteria.agent_type
        if criteria.status:
            query["status"] = criteria.status.value
        if criteria.capabilities:
            query["capabilities"] = {"$all": criteria.capabilities}

        cursor = await self.mongodb.find_many(self.collection, query)
        return [self._from_dict(d) for d in cursor]

    def _from_dict(self, data: dict[str, Any]) -> Agent:
        """Create Agent instance from dictionary."""
        return Agent(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "unknown"),
            type=data.get("type", "generic"),
            capabilities=data.get("capabilities", []),
            status=AgentStatus(data.get("status", AgentStatus.ACTIVE.value)),
            registered_at=datetime.fromisoformat(data.get("registered_at", datetime.now(UTC).isoformat())),
            last_seen=datetime.fromisoformat(data.get("last_seen", datetime.now(UTC).isoformat())),
            metadata=data.get("metadata", {}),
        )


class AgentHealthChecker:
    """Handles health check logic for agents."""

    async def check_health(self, agent: Agent) -> dict[str, Any]:
        """Perform health check on agent."""
        # Baseline health check logic
        is_alive = (datetime.now(UTC) - agent.last_seen).total_seconds() < 300
        health_status = "healthy" if is_alive and agent.status == AgentStatus.ACTIVE else "degraded"
        return {
            "agent_id": agent.id,
            "status": agent.status.value,
            "overall_health": health_status,
            "last_health_check": datetime.now(UTC).isoformat(),
            "last_seen": agent.last_seen.isoformat(),
        }


class CapabilityValidator:
    """Validates agent capabilities against system standards."""

    def __init__(self):
        self.allowed_capabilities: set[str] = {
            "ingestion",
            "standardization",
            "validation",
            "processing",
            "quality_assessment",
            "export",
            "bias_detection",
            "therapeutic_analysis",
        }

    async def validate(self, capabilities: list[str]) -> list[str]:
        """Validate and return normalized capabilities."""
        validated = [cap.lower() for cap in capabilities if cap.lower() in self.allowed_capabilities]
        if not validated and capabilities:
            logger.warning(f"None of the provided capabilities matched system standards: {capabilities}")
        return validated


class AgentManager:
    """Manage agent registration, discovery, and lifecycle."""

    def __init__(self, redis_client: MCPRedisClient, mongodb_client: MCPMongoDBClient):
        self.redis = redis_client
        self.mongodb = mongodb_client
        self.agent_registry = AgentRegistry(mongodb_client)
        self.health_checker = AgentHealthChecker()
        self.capability_validator = CapabilityValidator()

    async def register_agent(self, agent_data: AgentRegistrationData) -> Agent:
        """Register new agent with capability validation."""
        logger.info(f"Registering new agent: {agent_data.name}")

        # Validate agent capabilities
        validated_capabilities = await self.capability_validator.validate(agent_data.capabilities)

        # Create agent instance
        agent = Agent(
            id=str(uuid.uuid4()),
            name=agent_data.name,
            type=agent_data.type,
            capabilities=validated_capabilities,
            status=AgentStatus.ACTIVE,
            registered_at=datetime.now(UTC),
            metadata=agent_data.metadata,
        )

        # Store in registry
        await self.agent_registry.add_agent(agent)

        # Publish registration event
        await self.redis.publish("agent.registered", agent.to_dict())

        return agent

    async def discover_agents(self, criteria: AgentDiscoveryCriteria) -> list[Agent]:
        """Discover agents based on capability criteria."""
        return await self.agent_registry.find_agents(criteria)

    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status and health metrics."""
        agent = await self.agent_registry.get_agent(agent_id)
        if not agent:
            logger.warning(f"Attempted to update status for non-existent agent: {agent_id}")
            return False

        agent.status = status
        agent.last_seen = datetime.now(UTC)

        await self.agent_registry.update_agent(agent)

        # Publish status update
        await self.redis.publish(
            "agent.status_updated",
            {
                "agent_id": agent_id,
                "status": status.value,
                "timestamp": agent.last_seen.isoformat(),
            },
        )

        return True

    async def check_agent_health(self, agent_id: str) -> dict[str, Any] | None:
        """Check agent health status."""
        agent = await self.agent_registry.get_agent(agent_id)
        if not agent:
            return None

        return await self.health_checker.check_health(agent)
