"""
Data models for MCP (Management Control Panel) Server.

This module provides comprehensive data models for agent management,
task orchestration, and pipeline integration with proper validation
and serialization capabilities.
"""

# Base models
from .base import (
    BaseModel, 
    SerializableMixin, 
    HealthStatus, 
    PerformanceMetrics, 
    ErrorInfo, 
    ValidationResult, 
    ModelRegistry,
    generate_uuid,
    current_timestamp,
    sanitize_for_json
)

# Agent models
from .agent import (
    Agent,
    AgentRegistrationData,
    AgentDiscoveryCriteria,
    AgentHealthData,
    AgentStatus,
    AgentType,
    CapabilityType,
    AgentCapability,
    agent_registry
)

# Task models
from .task import (
    Task,
    TaskCreationData,
    TaskRequirements,
    TaskAssignment,
    TaskProgress,
    ExecutionData,
    ExecutionResult,
    TaskStatus,
    TaskType,
    TaskPriority,
    AssignmentStatus,
    TaskQueueStats,
    task_registry
)

# Pipeline models
from .pipeline import (
    PipelineExecution,
    PipelineRequest,
    PipelineResult,
    PipelineConfiguration,
    StageConfiguration,
    StageDependency,
    StageExecution,
    PipelineStage,
    PipelineStatus,
    StageStatus,
    pipeline_registry
)

__all__ = [
    # Base models
    'BaseModel',
    'SerializableMixin',
    'HealthStatus',
    'PerformanceMetrics',
    'ErrorInfo',
    'ValidationResult',
    'ModelRegistry',
    'generate_uuid',
    'current_timestamp',
    'sanitize_for_json',
    
    # Agent models
    'Agent',
    'AgentRegistrationData',
    'AgentDiscoveryCriteria',
    'AgentHealthData',
    'AgentStatus',
    'AgentType',
    'CapabilityType',
    'AgentCapability',
    'agent_registry',
    
    # Task models
    'Task',
    'TaskCreationData',
    'TaskRequirements',
    'TaskAssignment',
    'TaskProgress',
    'ExecutionData',
    'ExecutionResult',
    'TaskStatus',
    'TaskType',
    'TaskPriority',
    'AssignmentStatus',
    'TaskQueueStats',
    'task_registry',
    
    # Pipeline models
    'PipelineExecution',
    'PipelineRequest',
    'PipelineResult',
    'PipelineConfiguration',
    'StageConfiguration',
    'StageDependency',
    'StageExecution',
    'PipelineStage',
    'PipelineStatus',
    'StageStatus',
    'pipeline_registry',
]