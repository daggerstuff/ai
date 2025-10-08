"""
Pipeline models for MCP (Management Control Panel) Server.

This module provides comprehensive data models for 6-stage pipeline orchestration,
including stage management, dependency resolution, and execution tracking.
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .base import BaseModel, HealthStatus, PerformanceMetrics, ValidationResult, ErrorInfo
from .task import Task, TaskType, TaskPriority, TaskStatus


class PipelineStage(Enum):
    """6-stage pipeline stages."""
    INGESTION = "ingestion"                    # Stage 1: Data ingestion
    STANDARDIZATION = "standardization"        # Stage 2: Data standardization
    VALIDATION = "validation"                  # Stage 3: Data validation (with bias detection)
    PROCESSING = "processing"                  # Stage 4: Data processing
    QUALITY_ASSESSMENT = "quality_assessment"  # Stage 5: Quality assessment
    EXPORT = "export"                          # Stage 6: Export generation


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


class StageStatus(Enum):
    """Individual stage execution status."""
    PENDING = "pending"
    READY = "ready"           # Dependencies satisfied, ready to run
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageDependency:
    """Dependency between pipeline stages."""
    
    from_stage: PipelineStage
    to_stage: PipelineStage
    condition: Optional[str] = None  # Optional condition for dependency
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> ValidationResult:
        """Validate dependency configuration."""
        result = ValidationResult(is_valid=True)
        
        if self.from_stage == self.to_stage:
            result.add_error("SELF_DEPENDENCY", "Stage cannot depend on itself")
        
        return result


@dataclass
class StageConfiguration:
    """Configuration for a pipeline stage."""
    
    stage: PipelineStage
    name: str
    description: str
    required: bool = True
    timeout: int = 1800  # 30 minutes default
    retry_attempts: int = 3
    retry_delay: int = 5
    agent_capabilities: List[str] = field(default_factory=list)
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    output_parameters: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> ValidationResult:
        """Validate stage configuration."""
        result = ValidationResult(is_valid=True)
        
        if not self.name or len(self.name.strip()) == 0:
            result.add_error("INVALID_NAME", "Stage name cannot be empty")
        
        if not self.stage:
            result.add_error("INVALID_STAGE", "Stage type is required")
        
        if self.timeout < 60:
            result.add_error("INVALID_TIMEOUT", "Stage timeout must be at least 60 seconds")
        
        if self.retry_attempts < 0:
            result.add_error("INVALID_RETRIES", "Retry attempts cannot be negative")
        
        return result


@dataclass
class PipelineConfiguration:
    """Configuration for the 6-stage pipeline."""
    
    name: str
    description: str
    stages: List[StageConfiguration]
    dependencies: List[StageDependency]
    global_timeout: int = 7200  # 2 hours default
    max_concurrent_stages: int = 2
    enable_bias_detection: bool = True
    enable_quality_gates: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> ValidationResult:
        """Validate pipeline configuration."""
        result = ValidationResult(is_valid=True)
        
        if not self.name or len(self.name.strip()) == 0:
            result.add_error("INVALID_NAME", "Pipeline name cannot be empty")
        
        if not self.stages or len(self.stages) == 0:
            result.add_error("NO_STAGES", "Pipeline must have at least one stage")
        
        if not self.dependencies or len(self.dependencies) == 0:
            result.add_error("NO_DEPENDENCIES", "Pipeline must have stage dependencies")
        
        # Validate each stage
        for i, stage in enumerate(self.stages):
            stage_result = stage.validate()
            if not stage_result.is_valid:
                for error in stage_result.errors:
                    result.add_error(
                        f"STAGE_{i}_{error.code}",
                        f"Stage {i}: {error.message}"
                    )
        
        # Validate each dependency
        for i, dependency in enumerate(self.dependencies):
            dependency_result = dependency.validate()
            if not dependency_result.is_valid:
                for error in dependency_result.errors:
                    result.add_error(
                        f"DEPENDENCY_{i}_{error.code}",
                        f"Dependency {i}: {error.message}"
                    )
        
        # Validate stage order and dependencies
        stage_types = {stage.stage for stage in self.stages}
        for dependency in self.dependencies:
            if dependency.from_stage not in stage_types:
                result.add_error(
                    "INVALID_DEPENDENCY_FROM",
                    f"Dependency from non-existent stage: {dependency.from_stage.value}"
                )
            if dependency.to_stage not in stage_types:
                result.add_error(
                    "INVALID_DEPENDENCY_TO",
                    f"Dependency to non-existent stage: {dependency.to_stage.value}"
                )
        
        return result
    
    def get_stage_config(self, stage: PipelineStage) -> Optional[StageConfiguration]:
        """Get configuration for a specific stage."""
        for stage_config in self.stages:
            if stage_config.stage == stage:
                return stage_config
        return None
    
    def get_stage_dependencies(self, stage: PipelineStage) -> List[StageDependency]:
        """Get dependencies for a specific stage."""
        return [dep for dep in self.dependencies if dep.to_stage == stage]
    
    def get_dependent_stages(self, stage: PipelineStage) -> List[PipelineStage]:
        """Get stages that depend on the given stage."""
        return [dep.to_stage for dep in self.dependencies if dep.from_stage == stage]


@dataclass
class StageExecution:
    """Execution information for a single stage."""
    
    stage: PipelineStage
    status: StageStatus = StageStatus.PENDING
    task_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None
    error: Optional[ErrorInfo] = None
    output_data: Dict[str, Any] = field(default_factory=dict)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start_execution(self, task_id: str) -> None:
        """Mark stage as running."""
        self.status = StageStatus.RUNNING
        self.task_id = task_id
        self.started_at = datetime.utcnow()
    
    def complete_execution(self, output_data: Dict[str, Any] = None, 
                          validation_results: List[Dict[str, Any]] = None) -> None:
        """Mark stage as completed."""
        self.status = StageStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        
        if output_data:
            self.output_data.update(output_data)
        
        if validation_results:
            self.validation_results = validation_results
        
        if self.started_at:
            self.execution_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000
    
    def fail_execution(self, error: ErrorInfo) -> None:
        """Mark stage as failed."""
        self.status = StageStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error = error
        
        if self.started_at:
            self.execution_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000
    
    def skip_execution(self, reason: str = None) -> None:
        """Mark stage as skipped."""
        self.status = StageStatus.SKIPPED
        if reason:
            self.metadata['skip_reason'] = reason
    
    def is_completed(self) -> bool:
        """Check if stage execution is completed."""
        return self.status in [StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.SKIPPED]
    
    def is_ready(self) -> bool:
        """Check if stage is ready to run."""
        return self.status == StageStatus.READY
    
    def is_running(self) -> bool:
        """Check if stage is currently running."""
        return self.status == StageStatus.RUNNING


@dataclass
class PipelineExecution(BaseModel):
    """Pipeline execution with 6-stage orchestration."""
    
    configuration: PipelineConfiguration
    status: PipelineStatus = PipelineStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    stage_executions: Dict[PipelineStage, StageExecution] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_execution_time_ms: Optional[float] = None
    error: Optional[ErrorInfo] = None
    current_stage: Optional[PipelineStage] = None
    completed_stages: List[PipelineStage] = field(default_factory=list)
    failed_stages: List[PipelineStage] = field(default_factory=list)
    skipped_stages: List[PipelineStage] = field(default_factory=list)
    progress_percentage: float = 0.0
    
    def __post_init__(self):
        """Post-initialization processing."""
        super().__post_init__()
        
        # Initialize stage executions
        if not self.stage_executions:
            for stage_config in self.configuration.stages:
                self.stage_executions[stage_config.stage] = StageExecution(
                    stage=stage_config.stage
                )
    
    def start_execution(self) -> None:
        """Start pipeline execution."""
        self.status = PipelineStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.update_timestamp()
    
    def complete_execution(self, output_data: Dict[str, Any] = None) -> None:
        """Complete pipeline execution."""
        self.status = PipelineStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        
        if output_data:
            self.output_data.update(output_data)
        
        if self.started_at:
            self.total_execution_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        
        self.progress_percentage = 100.0
        self.update_timestamp()
    
    def fail_execution(self, error: ErrorInfo) -> None:
        """Fail pipeline execution."""
        self.status = PipelineStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error = error
        
        if self.started_at:
            self.total_execution_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        
        self.update_timestamp()
    
    def cancel_execution(self) -> None:
        """Cancel pipeline execution."""
        self.status = PipelineStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.update_timestamp()
    
    def update_stage_status(self, stage: PipelineStage, status: StageStatus) -> None:
        """Update status of a specific stage."""
        if stage in self.stage_executions:
            self.stage_executions[stage].status = status
            self.update_timestamp()
    
    def get_stage_execution(self, stage: PipelineStage) -> Optional[StageExecution]:
        """Get execution information for a specific stage."""
        return self.stage_executions.get(stage)
    
    def get_ready_stages(self) -> List[PipelineStage]:
        """Get stages that are ready to execute."""
        ready_stages = []
        
        for stage_config in self.configuration.stages:
            stage = stage_config.stage
            stage_execution = self.stage_executions[stage]
            
            if stage_execution.status == StageStatus.PENDING:
                # Check if all dependencies are satisfied
                dependencies = self.configuration.get_stage_dependencies(stage)
                
                all_dependencies_completed = True
                for dependency in dependencies:
                    dep_execution = self.stage_executions.get(dependency.from_stage)
                    if not dep_execution or not dep_execution.is_completed():
                        all_dependencies_completed = False
                        break
                
                if all_dependencies_completed:
                    stage_execution.status = StageStatus.READY
                    ready_stages.append(stage)
        
        return ready_stages
    
    def get_running_stages(self) -> List[PipelineStage]:
        """Get stages that are currently running."""
        return [
            stage for stage, execution in self.stage_executions.items()
            if execution.is_running()
        ]
    
    def get_completed_stages(self) -> List[PipelineStage]:
        """Get stages that have been completed."""
        return [
            stage for stage, execution in self.stage_executions.items()
            if execution.status == StageStatus.COMPLETED
        ]
    
    def get_failed_stages(self) -> List[PipelineStage]:
        """Get stages that have failed."""
        return [
            stage for stage, execution in self.stage_executions.items()
            if execution.status == StageStatus.FAILED
        ]
    
    def calculate_progress(self) -> float:
        """Calculate overall pipeline progress percentage."""
        total_stages = len(self.configuration.stages)
        if total_stages == 0:
            return 0.0
        
        completed_stages = len(self.get_completed_stages())
        failed_stages = len(self.get_failed_stages())
        skipped_stages = len([
            stage for stage, execution in self.stage_executions.items()
            if execution.status == StageStatus.SKIPPED
        ])
        
        completed_count = completed_stages + failed_stages + skipped_stages
        return (completed_count / total_stages) * 100.0
    
    def is_stage_ready(self, stage: PipelineStage) -> bool:
        """Check if a stage is ready to execute."""
        stage_execution = self.stage_executions.get(stage)
        return stage_execution and stage_execution.is_ready()
    
    def can_stage_start(self, stage: PipelineStage) -> bool:
        """Check if a stage can start execution."""
        stage_execution = self.stage_executions.get(stage)
        if not stage_execution or stage_execution.status != StageStatus.READY:
            return False
        
        # Check if maximum concurrent stages limit is reached
        running_stages = self.get_running_stages()
        if len(running_stages) >= self.configuration.max_concurrent_stages:
            return False
        
        return True
    
    def get_next_stage(self) -> Optional[PipelineStage]:
        """Get the next stage that should be executed."""
        ready_stages = self.get_ready_stages()
        
        # Prioritize stages based on the 6-stage order
        stage_priority = [
            PipelineStage.INGESTION,
            PipelineStage.STANDARDIZATION,
            PipelineStage.VALIDATION,
            PipelineStage.PROCESSING,
            PipelineStage.QUALITY_ASSESSMENT,
            PipelineStage.EXPORT
        ]
        
        for stage in stage_priority:
            if stage in ready_stages and self.can_stage_start(stage):
                return stage
        
        return None
    
    def validate_execution(self) -> ValidationResult:
        """Validate pipeline execution state."""
        result = ValidationResult(is_valid=True)
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            result.add_error("CIRCULAR_DEPENDENCIES", "Pipeline has circular dependencies")
        
        # Check if all required stages are present
        required_stages = {stage for stage in PipelineStage}
        configured_stages = {stage_config.stage for stage_config in self.configuration.stages}
        
        missing_required_stages = required_stages - configured_stages
        if missing_required_stages:
            result.add_error(
                "MISSING_REQUIRED_STAGES",
                f"Missing required stages: {[s.value for s in missing_required_stages]}"
            )
        
        return result
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in the pipeline."""
        # Simple cycle detection using DFS
        visited = set()
        recursion_stack = set()
        
        def has_cycle(stage: PipelineStage) -> bool:
            visited.add(stage)
            recursion_stack.add(stage)
            
            # Get dependent stages
            dependent_stages = self.configuration.get_dependent_stages(stage)
            
            for dependent_stage in dependent_stages:
                if dependent_stage not in visited:
                    if has_cycle(dependent_stage):
                        return True
                elif dependent_stage in recursion_stack:
                    return True
            
            recursion_stack.remove(stage)
            return False
        
        for stage in PipelineStage:
            if stage not in visited:
                if has_cycle(stage):
                    return True
        
        return False
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Get pipeline execution summary."""
        return {
            'id': self.id,
            'configuration_name': self.configuration.name,
            'status': self.status.value,
            'progress_percentage': self.calculate_progress(),
            'current_stage': self.current_stage.value if self.current_stage else None,
            'completed_stages': [stage.value for stage in self.get_completed_stages()],
            'failed_stages': [stage.value for stage in self.get_failed_stages()],
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_execution_time_ms': self.total_execution_time_ms,
            'total_stages': len(self.configuration.stages),
            'running_stages': len(self.get_running_stages())
        }


@dataclass
class PipelineRequest:
    """Request to execute a pipeline."""
    
    configuration: PipelineConfiguration
    input_data: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> ValidationResult:
        """Validate pipeline request."""
        result = ValidationResult(is_valid=True)
        
        if not self.configuration:
            result.add_error("NO_CONFIGURATION", "Pipeline configuration is required")
        else:
            # Validate configuration
            config_result = self.configuration.validate()
            if not config_result.is_valid:
                result.merge(config_result)
        
        return result


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    
    execution_id: str
    status: PipelineStatus
    output_data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[ErrorInfo] = None
    execution_summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Get pipeline result summary."""
        return {
            'execution_id': self.execution_id,
            'status': self.status.value,
            'error_code': self.error.code if self.error else None,
            'error_message': self.error.message if self.error else None,
            'metrics': self.metrics,
            'execution_summary': self.execution_summary
        }


# Pipeline registry for managing pipeline executions
pipeline_registry = ModelRegistry()