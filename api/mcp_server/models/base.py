"""
Base models and mixins for MCP (Management Control Panel) Server.

This module provides foundational classes and utilities for data models
used throughout the MCP server architecture.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict, field


class SerializableMixin:
    """Mixin for serializable data models."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert model to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SerializableMixin':
        """Create model instance from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SerializableMixin':
        """Create model instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class BaseModel(SerializableMixin):
    """Base model with common attributes for all MCP entities."""
    
    id: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not hasattr(self, 'id') or not self.id:
            self.id = str(uuid.uuid4())
        
        if not hasattr(self, 'created_at') or not self.created_at:
            self.created_at = datetime.utcnow()
        
        if not hasattr(self, 'updated_at') or not self.updated_at:
            self.updated_at = datetime.utcnow()
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the model."""
        if not self.metadata:
            self.metadata = {}
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key, default) if self.metadata else default


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


def current_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.utcnow()


def sanitize_for_json(obj: Any) -> Any:
    """Sanitize object for JSON serialization."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    else:
        return obj


@dataclass
class HealthStatus:
    """Health status information for agents and services."""
    
    status: str
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_count: int = 0
    success_count: int = 0
    availability_percentage: float = 100.0
    
    def record_success(self, response_time_ms: float) -> None:
        """Record a successful health check."""
        self.success_count += 1
        self.response_time_ms = response_time_ms
        self.last_check = current_timestamp()
        self._calculate_availability()
    
    def record_failure(self) -> None:
        """Record a failed health check."""
        self.error_count += 1
        self.last_check = current_timestamp()
        self._calculate_availability()
    
    def _calculate_availability(self) -> None:
        """Calculate availability percentage."""
        total_checks = self.success_count + self.error_count
        if total_checks > 0:
            self.availability_percentage = (self.success_count / total_checks) * 100.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for agents and tasks."""
    
    execution_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    network_io_mb: Optional[float] = None
    disk_io_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering out None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ErrorInfo:
    """Error information for logging and debugging."""
    
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=current_timestamp)
    stack_trace: Optional[str] = None
    
    @classmethod
    def from_exception(cls, exception: Exception, code: str = "INTERNAL_ERROR") -> 'ErrorInfo':
        """Create ErrorInfo from an exception."""
        return cls(
            code=code,
            message=str(exception),
            stack_trace=getattr(exception, '__traceback__', None)
        )


@dataclass
class ValidationResult:
    """Validation result with detailed information."""
    
    is_valid: bool
    errors: List[ErrorInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Add validation error."""
        self.errors.append(ErrorInfo(code=code, message=message, details=details))
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add validation warning."""
        self.warnings.append(message)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.metadata.update(other.metadata)
        if not other.is_valid:
            self.is_valid = False


class ModelRegistry:
    """Registry for managing model instances."""
    
    def __init__(self):
        self._models: Dict[str, BaseModel] = {}
    
    def register(self, model: BaseModel) -> None:
        """Register a model instance."""
        self._models[model.id] = model
    
    def get(self, model_id: str) -> Optional[BaseModel]:
        """Get a model by ID."""
        return self._models.get(model_id)
    
    def remove(self, model_id: str) -> bool:
        """Remove a model by ID."""
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False
    
    def list_all(self) -> List[BaseModel]:
        """List all registered models."""
        return list(self._models.values())
    
    def filter_by_type(self, model_type: type) -> List[BaseModel]:
        """Filter models by type."""
        return [model for model in self._models.values() if isinstance(model, model_type)]
    
    def count(self) -> int:
        """Get the number of registered models."""
        return len(self._models)
    
    def clear(self) -> None:
        """Clear all registered models."""
        self._models.clear()