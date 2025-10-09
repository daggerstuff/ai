"""
Task models for MCP (Management Control Panel) Server.

This module provides comprehensive data models for task management,
including creation, assignment, execution, and status tracking.
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .base import BaseModel, HealthStatus, PerformanceMetrics, ValidationResult, ErrorInfo


class TaskStatus(Enum):
    """Task lifecycle status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    TIMEOUT = "timeout"


class TaskType(Enum):
    """Types of tasks supported by the MCP server."""
    PIPELINE_PROCESSING = "pipeline_processing"
    BIAS_DETECTION = "bias_detection"
    VALIDATION = "validation"
    DATA_ANALYSIS = "data_analysis"
    EXPORT_GENERATION = "export_generation"
    CUSTOM_PROCESSING = "custom_processing"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AssignmentStatus(Enum):
    """Task assignment status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class TaskRequirements:
    """Task requirements and constraints."""
    
    capabilities: List[str] = field(default_factory=list)
    min_agent_version: Optional[str] = None
    max_concurrent_agents: int = 1
    required_resources: Dict[str, Any] = field(default_factory=dict)
    execution_timeout: int = 1800  # 30 minutes default
    retry_attempts: int = 3
    retry_delay: int = 5
    priority: TaskPriority = TaskPriority.MEDIUM
    
    def validate(self) -> ValidationResult:
        """Validate task requirements."""
        result = ValidationResult(is_valid=True)
        
        if not self.capabilities:
            result.add_error("NO_CAPABILITIES", "Task must specify required capabilities")
        
        if self.max_concurrent_agents < 1:
            result.add_error("INVALID_CONCURRENCY", "Max concurrent agents must be at least 1")
        
        if self.execution_timeout < 60:
            result.add_error("INVALID_TIMEOUT", "Execution timeout must be at least 60 seconds")
        
        if self.retry_attempts < 0:
            result.add_error("INVALID_RETRIES", "Retry attempts cannot be negative")
        
        return result


@dataclass
class TaskCreationData:
    """Data required for task creation."""
    
    type: TaskType
    priority: TaskPriority
    requirements: TaskRequirements
    payload: Dict[str, Any] = field(default_factory=dict)
    pipeline_id: Optional[str] = None
    stage: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> ValidationResult:
        """Validate task creation data."""
        result = ValidationResult(is_valid=True)
        
        if not self.type:
            result.add_error("INVALID_TYPE", "Task type is required")
        
        if not self.priority:
            result.add_error("INVALID_PRIORITY", "Task priority is required")
        
        if not self.requirements:
            result.add_error("NO_REQUIREMENTS", "Task requirements are required")
        else:
            # Validate requirements
            requirements_result = self.requirements.validate()
            if not requirements_result.is_valid:
                result.merge(requirements_result)
        
        return result


@dataclass
class TaskAssignment:
    """Task assignment information."""
    
    task_id: str
    agent_id: str
    assigned_at: datetime
    status: AssignmentStatus
    accepted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def accept(self) -> None:
        """Mark assignment as accepted."""
        self.status = AssignmentStatus.ACCEPTED
        self.accepted_at = datetime.utcnow()
    
    def reject(self) -> None:
        """Mark assignment as rejected."""
        self.status = AssignmentStatus.REJECTED
        self.completed_at = datetime.utcnow()
    
    def timeout(self) -> None:
        """Mark assignment as timed out."""
        self.status = AssignmentStatus.TIMEOUT
        self.completed_at = datetime.utcnow()


@dataclass
class TaskProgress:
    """Task execution progress information."""
    
    task_id: str
    progress_percentage: float = 0.0
    current_stage: Optional[str] = None
    stages_completed: List[str] = field(default_factory=list)
    estimated_time_remaining: Optional[int] = None
    last_update: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_progress(self, percentage: float, stage: Optional[str] = None) -> None:
        """Update task progress."""
        self.progress_percentage = max(0.0, min(100.0, percentage))
        self.last_update = datetime.utcnow()
        
        if stage and stage not in self.stages_completed:
            self.stages_completed.append(stage)
            self.current_stage = stage
    
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.progress_percentage >= 100.0


@dataclass
class ExecutionData:
    """Task execution data."""
    
    input_data: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> ValidationResult:
        """Validate execution data."""
        result = ValidationResult(is_valid=True)
        
        # Basic validation - can be extended based on task type
        if not isinstance(self.input_data, dict):
            result.add_error("INVALID_INPUT_DATA", "Input data must be a dictionary")
        
        if not isinstance(self.parameters, dict):
            result.add_error("INVALID_PARAMETERS", "Parameters must be a dictionary")
        
        return result


@dataclass
class ExecutionResult:
    """Task execution result."""
    
    task_id: str
    success: bool
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[ErrorInfo] = None
    execution_time_ms: Optional[float] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Get execution result summary."""
        return {
            'task_id': self.task_id,
            'success': self.success,
            'execution_time_ms': self.execution_time_ms,
            'error_code': self.error.code if self.error else None,
            'error_message': self.error.message if self.error else None
        }


@dataclass
class Task(BaseModel):
    """Task model with comprehensive lifecycle management."""
    
    type: TaskType
    priority: TaskPriority
    requirements: TaskRequirements
    status: TaskStatus = TaskStatus.PENDING
    payload: Dict[str, Any] = field(default_factory=dict)
    pipeline_id: Optional[str] = None
    stage: Optional[int] = None
    assigned_agent_id: Optional[str] = None
    progress: TaskProgress = field(default_factory=lambda: TaskProgress(task_id=""))
    execution_result: Optional[ExecutionResult] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    parent_task_id: Optional[str] = None
    child_task_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization processing."""
        super().__post_init__()
        if not self.progress.task_id:
            self.progress.task_id = self.id
    
    def assign_to_agent(self, agent_id: str) -> TaskAssignment:
        """Assign task to an agent."""
        self.assigned_agent_id = agent_id
        self.status = TaskStatus.ASSIGNED
        self.update_timestamp()
        
        return TaskAssignment(
            task_id=self.id,
            agent_id=agent_id,
            assigned_at=datetime.utcnow(),
            status=AssignmentStatus.ASSIGNED
        )
    
    def start_execution(self) -> None:
        """Mark task as running."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.update_timestamp()
    
    def complete_execution(self, result: ExecutionResult) -> None:
        """Mark task as completed with result."""
        self.status = TaskStatus.COMPLETED
        self.execution_result = result
        self.completed_at = datetime.utcnow()
        self.progress.update_progress(100.0)
        self.update_timestamp()
    
    def fail_execution(self, error: ErrorInfo) -> None:
        """Mark task as failed with error."""
        self.status = TaskStatus.FAILED
        self.execution_result = ExecutionResult(
            task_id=self.id,
            success=False,
            error=error
        )
        self.completed_at = datetime.utcnow()
        self.update_timestamp()
    
    def retry_execution(self) -> bool:
        """Attempt to retry task execution."""
        if self.retry_count >= self.requirements.retry_attempts:
            return False
        
        self.retry_count += 1
        self.status = TaskStatus.RETRYING
        self.started_at = None
        self.completed_at = None
        self.execution_result = None
        self.progress = TaskProgress(task_id=self.id)
        self.update_timestamp()
        
        return True
    
    def cancel_execution(self) -> None:
        """Cancel task execution."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.update_timestamp()
    
    def update_progress(self, percentage: float, stage: Optional[str] = None) -> None:
        """Update task progress."""
        self.progress.update_progress(percentage, stage)
        self.update_timestamp()
    
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
    
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.RUNNING
    
    def can_be_retried(self) -> bool:
        """Check if task can be retried."""
        return (self.status == TaskStatus.FAILED and 
                self.retry_count < self.requirements.retry_attempts)
    
    def get_execution_time(self) -> Optional[float]:
        """Get task execution time in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def validate(self) -> ValidationResult:
        """Validate task configuration."""
        result = ValidationResult(is_valid=True)
        
        if not self.type:
            result.add_error("INVALID_TYPE", "Task type is required")
        
        if not self.priority:
            result.add_error("INVALID_PRIORITY", "Task priority is required")
        
        if not self.requirements:
            result.add_error("NO_REQUIREMENTS", "Task requirements are required")
        else:
            # Validate requirements
            requirements_result = self.requirements.validate()
            if not requirements_result.is_valid:
                result.merge(requirements_result)
        
        if self.retry_count < 0:
            result.add_error("INVALID_RETRY_COUNT", "Retry count cannot be negative")
        
        return result
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Get task summary for listings."""
        return {
            'id': self.id,
            'type': self.type.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'progress': self.progress.progress_percentage,
            'assigned_agent_id': self.assigned_agent_id,
            'pipeline_id': self.pipeline_id,
            'stage': self.stage,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'retry_count': self.retry_count,
            'execution_time': self.get_execution_time()
        }


@dataclass
class TaskQueueStats:
    """Task queue statistics."""
    
    total_tasks: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    average_execution_time: Optional[float] = None
    queue_depth: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_tasks': self.total_tasks,
            'pending_tasks': self.pending_tasks,
            'running_tasks': self.running_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'cancelled_tasks': self.cancelled_tasks,
            'average_execution_time': self.average_execution_time,
            'queue_depth': self.queue_depth
        }


# Task registry for managing task instances
task_registry = ModelRegistry()