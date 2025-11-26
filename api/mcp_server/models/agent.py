"""
Agent models for MCP (Management Control Panel) Server.

This module provides comprehensive data models for agent management,
including registration, capabilities, health monitoring, and lifecycle management.
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .base import BaseModel, HealthStatus, PerformanceMetrics, ValidationResult


class AgentStatus(Enum):
    """Agent lifecycle status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DECOMMISSIONED = "decommissioned"


class AgentType(Enum):
    """Types of agents supported by the MCP server."""
    AI_MODEL = "ai_model"
    DATA_PROCESSOR = "data_processor"
    VALIDATION_AGENT = "validation_agent"
    EXTERNAL_API = "external_api"
    CUSTOM_AGENT = "custom_agent"
    EMBEDDING_AGENT = "embedding_agent"


class CapabilityType(Enum):
    """Types of agent capabilities."""
    PIPELINE_PROCESSING = "pipeline_processing"
    BIAS_DETECTION = "bias_detection"
    VALIDATION = "validation"
    DATA_ANALYSIS = "data_analysis"
    EXPORT_GENERATION = "export_generation"
    CUSTOM_PROCESSING = "custom_processing"
    TEXT_EMBEDDING = "text_embedding"
    BATCH_EMBEDDING = "batch_embedding"
    SIMILARITY_SEARCH = "similarity_search"
    KNOWLEDGE_INDEXING = "knowledge_indexing"


@dataclass
class AgentCapability:
    """Agent capability definition."""

    type: CapabilityType
    name: str
    description: str
    version: str
    supported_formats: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 1
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ValidationResult:
        """Validate capability configuration."""
        result = ValidationResult(is_valid=True)

        if not self.name or len(self.name.strip()) == 0:
            result.add_error("INVALID_NAME", "Capability name cannot be empty")

        if not self.type:
            result.add_error("INVALID_TYPE", "Capability type is required")

        if self.max_concurrent_tasks < 1:
            result.add_error("INVALID_CONCURRENCY", "Max concurrent tasks must be at least 1")

        return result


@dataclass
class AgentRegistrationData:
    """Data required for agent registration."""

    name: str
    type: AgentType
    capabilities: List[AgentCapability]
    endpoint_url: Optional[str] = None
    authentication_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ValidationResult:
        """Validate registration data."""
        result = ValidationResult(is_valid=True)

        if not self.name or len(self.name.strip()) == 0:
            result.add_error("INVALID_NAME", "Agent name cannot be empty")

        if not self.type:
            result.add_error("INVALID_TYPE", "Agent type is required")

        if not self.capabilities or len(self.capabilities) == 0:
            result.add_error("NO_CAPABILITIES", "Agent must have at least one capability")

        # Validate each capability
        for i, capability in enumerate(self.capabilities):
            capability_result = capability.validate()
            if not capability_result.is_valid:
                for error in capability_result.errors:
                    result.add_error(
                        f"CAPABILITY_{i}_{error.code}",
                        f"Capability {i}: {error.message}"
                    )

        # Validate endpoint URL if provided
        if self.endpoint_url and not self._is_valid_url(self.endpoint_url):
            result.add_error("INVALID_ENDPOINT", "Invalid endpoint URL format")

        return result

    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation."""
        import re
        url_pattern = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(url))


@dataclass
class Agent(BaseModel):
    """Agent model with comprehensive lifecycle management."""

    name: str
    type: AgentType
    capabilities: List[AgentCapability]
    status: AgentStatus = AgentStatus.ACTIVE
    health_status: HealthStatus = field(default_factory=lambda: HealthStatus(
        status="healthy",
        last_check=datetime.utcnow()
    ))
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    last_seen: Optional[datetime] = None
    registered_by: Optional[str] = None  # User ID who registered the agent
    endpoint_url: Optional[str] = None
    authentication_token: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization processing."""
        super().__post_init__()
        if not self.last_seen:
            self.last_seen = self.created_at

    def update_health(self, health_status: HealthStatus) -> None:
        """Update agent health status."""
        self.health_status = health_status
        self.update_timestamp()

    def update_last_seen(self) -> None:
        """Update last seen timestamp."""
        self.last_seen = datetime.utcnow()
        self.update_timestamp()

    def set_status(self, status: AgentStatus) -> None:
        """Set agent status."""
        self.status = status
        self.update_timestamp()

    def add_capability(self, capability: AgentCapability) -> None:
        """Add a new capability to the agent."""
        self.capabilities.append(capability)
        self.update_timestamp()

    def remove_capability(self, capability_type: CapabilityType) -> bool:
        """Remove a capability by type."""
        original_count = len(self.capabilities)
        self.capabilities = [cap for cap in self.capabilities if cap.type != capability_type]
        removed = len(self.capabilities) < original_count
        if removed:
            self.update_timestamp()
        return removed

    def has_capability(self, capability_type: CapabilityType) -> bool:
        """Check if agent has a specific capability."""
        return any(cap.type == capability_type for cap in self.capabilities)

    def get_capability(self, capability_type: CapabilityType) -> Optional[AgentCapability]:
        """Get a specific capability by type."""
        for cap in self.capabilities:
            if cap.type == capability_type:
                return cap
        return None

    def get_total_concurrent_capacity(self) -> int:
        """Get total concurrent task capacity."""
        return sum(cap.max_concurrent_tasks for cap in self.capabilities)

    def is_available(self) -> bool:
        """Check if agent is available for task assignment."""
        return (self.status == AgentStatus.ACTIVE and
                self.health_status.status == "healthy" and
                self.health_status.availability_percentage > 80.0)

    def validate(self) -> ValidationResult:
        """Validate agent configuration."""
        result = ValidationResult(is_valid=True)

        if not self.name or len(self.name.strip()) == 0:
            result.add_error("INVALID_NAME", "Agent name cannot be empty")

        if not self.type:
            result.add_error("INVALID_TYPE", "Agent type is required")

        if not self.capabilities or len(self.capabilities) == 0:
            result.add_error("NO_CAPABILITIES", "Agent must have at least one capability")

        # Validate each capability
        for i, capability in enumerate(self.capabilities):
            capability_result = capability.validate()
            if not capability_result.is_valid:
                for error in capability_result.errors:
                    result.add_error(
                        f"CAPABILITY_{i}_{error.code}",
                        f"Capability {i}: {error.message}"
                    )

        # Validate health status
        if self.health_status.availability_percentage < 0 or self.health_status.availability_percentage > 100:
            result.add_error("INVALID_AVAILABILITY", "Availability percentage must be between 0 and 100")

        return result

    def to_summary_dict(self) -> Dict[str, Any]:
        """Get agent summary for listings."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'status': self.status.value,
            'capabilities': [cap.type.value for cap in self.capabilities],
            'health_status': {
                'status': self.health_status.status,
                'availability_percentage': self.health_status.availability_percentage,
                'last_check': self.health_status.last_check.isoformat()
            },
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags
        }


@dataclass
class AgentDiscoveryCriteria:
    """Criteria for discovering agents."""

    capability_types: List[CapabilityType] = field(default_factory=list)
    agent_types: List[AgentType] = field(default_factory=list)
    status: Optional[AgentStatus] = None
    min_availability_percentage: float = 80.0
    tags: List[str] = field(default_factory=list)
    exclude_agent_ids: List[str] = field(default_factory=list)
    max_response_time_ms: Optional[int] = None

    def matches_agent(self, agent: Agent) -> bool:
        """Check if an agent matches the discovery criteria."""
        # Check status
        if self.status and agent.status != self.status:
            return False

        # Check availability
        if agent.health_status.availability_percentage < self.min_availability_percentage:
            return False

        # Check capability types
        if self.capability_types:
            agent_capability_types = {cap.type for cap in agent.capabilities}
            if not any(cap_type in agent_capability_types for cap_type in self.capability_types):
                return False

        # Check agent types
        if self.agent_types and agent.type not in self.agent_types:
            return False

        # Check tags
        if self.tags:
            if not any(tag in agent.tags for tag in self.tags):
                return False

        # Check excluded agent IDs
        if agent.id in self.exclude_agent_ids:
            return False

        # Check response time
        if (self.max_response_time_ms and
            agent.health_status.response_time_ms and
            agent.health_status.response_time_ms > self.max_response_time_ms):
            return False

        return True


@dataclass
class AgentHealthData:
    """Agent health monitoring data."""

    agent_id: str
    timestamp: datetime
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    disk_usage_mb: Optional[float] = None
    network_io_mb: Optional[float] = None
    active_tasks: int = 0
    queue_size: int = 0
    error_rate: float = 0.0
    response_time_ms: Optional[float] = None

    def to_health_status(self) -> HealthStatus:
        """Convert health data to HealthStatus."""
        # Determine overall status based on metrics
        if self.error_rate > 0.1:  # 10% error rate threshold
            status = "unhealthy"
        elif self.error_rate > 0.05:  # 5% error rate threshold
            status = "degraded"
        else:
            status = "healthy"

        return HealthStatus(
            status=status,
            last_check=self.timestamp,
            response_time_ms=self.response_time_ms,
            error_count=int(self.error_rate * 100),  # Approximate
            success_count=int((1 - self.error_rate) * 100)  # Approximate
        )


# Agent registry for managing agent instances
agent_registry = ModelRegistry()
