"""
Foresight + Gemini Integration Package.

Provides memory management for therapeutic AI with:
- GeminiForesightManager: Core Gemini + Foresight integration
- TherapeuticMemoryConfig: Memory ingestion controls
- AgentMemoryTools: Async tools for agent frameworks
- MultiAgentMemory: Shared memory for multi-agent workflows
"""

from .agent_memory_tools import (
    AgentContext,
    AgentMemoryTools,
    create_memory_tool_handler,
)
from .manager import GeminiForesightConfig, GeminiForesightManager
from .memory_ingestion_config import (
    CrisisDetector,
    InferenceMode,
    MemoryCategory,
    PIIFilter,
    SpeculationFilter,
    TherapeuticMemoryConfig,
)
from .multi_agent_memory import (
    AgentIdentity,
    AgentRole,
    CollaborationContext,
    MemoryScope,
    MultiAgentMemory,
    create_empathy_gym_context,
)

__all__ = [
    # Agent Tools
    "AgentContext",
    # Multi-Agent Memory
    "AgentIdentity",
    "AgentMemoryTools",
    "AgentRole",
    "CollaborationContext",
    "CrisisDetector",
    # Core Manager
    "GeminiForesightConfig",
    "GeminiForesightManager",
    "InferenceMode",
    "MemoryCategory",
    "MemoryScope",
    "MultiAgentMemory",
    "PIIFilter",
    "SpeculationFilter",
    # Memory Ingestion Config
    "TherapeuticMemoryConfig",
    "create_empathy_gym_context",
    "create_memory_tool_handler",
]
