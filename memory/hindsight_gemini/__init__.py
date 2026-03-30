"""
Hindsight + Gemini Integration Package.

Provides memory management for therapeutic AI with:
- GeminiHindsightManager: Core Gemini + Hindsight integration
- TherapeuticMemoryConfig: Memory ingestion controls
- AgentMemoryTools: Async tools for agent frameworks
- MultiAgentMemory: Shared memory for multi-agent workflows
"""

from .agent_memory_tools import (
    AgentContext,
    AgentMemoryTools,
    create_memory_tool_handler,
)
from .manager import GeminiHindsightConfig, GeminiHindsightManager
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
    # Core Manager
    "GeminiHindsightConfig",
    "GeminiHindsightManager",
    # Memory Ingestion Config
    "TherapeuticMemoryConfig",
    "InferenceMode",
    "MemoryCategory",
    "PIIFilter",
    "SpeculationFilter",
    "CrisisDetector",
    # Agent Tools
    "AgentContext",
    "AgentMemoryTools",
    "create_memory_tool_handler",
    # Multi-Agent Memory
    "AgentIdentity",
    "AgentRole",
    "CollaborationContext",
    "MemoryScope",
    "MultiAgentMemory",
    "create_empathy_gym_context",
]
