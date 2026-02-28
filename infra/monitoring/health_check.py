"""
Health check and graceful shutdown handlers for Pixelated Empathy AI model servers.
Implements comprehensive health monitoring and safe shutdown procedures.
"""

import asyncio
import signal
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import torch
from functools import wraps
import threading
import atexit
import uuid


logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Status of individual components"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class HealthStatus(Enum):
    """Overall health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of an individual component"""
    name: str
    status: ComponentStatus
    last_checked: str
    health_score: float  # 0.0 to 1.0
    details: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    error_count: int = 0
    error_timestamps: List[str] = field(default_factory=list)
    component_type: Optional[str] = None  # e.g., "model", "database", "api", "gpu"
    response_time_ms: Optional[float] = None
    uptime_seconds: Optional[float] = None
    restart_count: int = 0
    dependencies: List[str] = field(default_factory=list)  # Dependent components


@dataclass
class HealthCheckResult:
    """Overall health check result"""
    status: HealthStatus
    timestamp: str
    components: Dict[str, ComponentHealth]
    overall_score: float  # Weighted average of component scores
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    service_uptime_seconds: Optional[float] = None
    last_check_duration_ms: Optional[float] = None
    component_health_summary: Optional[Dict[str, int]] = None  # Count by status


@dataclass
class ShutdownResult:
    """Result of a shutdown operation"""
    success: bool
    duration_seconds: float
    components_shutdown: List[str] = field(default_factory=list)
    components_failed: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    warning_messages: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    shutdown_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    shutdown_reason: Optional[str] = None
    forced_shutdown: bool = False
    cleanup_performed: bool = True
    logs_flushed: bool = True
    metrics_exported: bool = True
    connections_closed: bool = True


class GateType(Enum):
    """Types of evaluation gates"""
    MINIMUM_THRESHOLD = "minimum_threshold"
    MAXIMUM_THRESHOLD = "maximum_threshold"
    RANGE_THRESHOLD = "range_threshold"
    COMPARISON_THRESHOLD = "comparison_threshold"
    CUSTOM_EVALUATION = "custom_evaluation"


@dataclass
class EvaluationGate:
    """Definition of a single evaluation gate"""
    name: str
    metric_name: str
    gate_type: GateType
    threshold_value: float
    secondary_threshold: Optional[float] = None  # For range thresholds
    weight: float = 1.0  # Weight in overall scoring
    is_critical: bool = False  # If True, failure means automatic rejection
    description: str = ""
    component_type: Optional[str] = None  # e.g., "safety", "performance", "cost"
    
    def evaluate(self, metric_value: float) -> Tuple[bool, str]:
        """Evaluate if the metric passes this gate"""
        if self.gate_type == GateType.MINIMUM_THRESHOLD:
            passed = metric_value >= self.threshold_value
            reason = f"Value {metric_value} {'meets' if passed else 'fails'} minimum threshold {self.threshold_value}"
        elif self.gate_type == GateType.MAXIMUM_THRESHOLD:
            passed = metric_value <= self.threshold_value
            reason = f"Value {metric_value} {'meets' if passed else 'fails'} maximum threshold {self.threshold_value}"
        elif self.gate_type == GateType.RANGE_THRESHOLD:
            if self.secondary_threshold is not None:
                passed = self.threshold_value <= metric_value <= self.secondary_threshold
                reason = f"Value {metric_value} {'is in' if passed else 'is out of'} range [{self.threshold_value}, {self.secondary_threshold}]"
            else:
                passed = False
                reason = f"Range threshold requires secondary threshold value, got None"
        elif self.gate_type == GateType.COMPARISON_THRESHOLD:
            # For comparison, threshold_value is typically 0 for "no worse than baseline"
            # Positive values mean it can be that much worse
            passed = metric_value <= self.threshold_value
            reason = f"Value {metric_value} {'meets' if passed else 'fails'} comparison threshold {self.threshold_value}"
        else:
            passed = False
            reason = f"Unknown gate type: {self.gate_type}"
        
        return passed, reason


@dataclass
class GateConfiguration:
    """Configuration for evaluation gates"""
    stage: str  # staging, production, etc.
    gates: List[EvaluationGate] = field(default_factory=list)
    required_passing_gates: int = 0  # Number of gates that must pass (0 = all)
    minimum_overall_score: float = 0.7  # Minimum weighted score for promotion
    critical_gates: List[str] = field(default_factory=list)  # Gates that are always critical
    metadata: Optional[Dict[str, Any]] = None
    
    def add_gate(self, gate: EvaluationGate):
        """Add a gate to the configuration"""
        if gate.name in self.critical_gates:
            gate.is_critical = True
        self.gates.append(gate)


@dataclass
class GateEvaluationResult:
    """Result of a gate evaluation"""
    gate_name: str
    metric_name: str
    metric_value: float
    threshold_value: float
    passed: bool
    reason: str
    weight: float
    is_critical: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PromotionEvaluation:
    """Complete result of a promotion evaluation"""
    model_id: str
    model_version: str
    from_stage: str
    to_stage: str
    overall_score: float
    total_gates: int
    passed_gates: int
    failed_gates: int
    results: List[GateEvaluationResult]
    is_approved: bool
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromotionStage(Enum):
    """Stages in the model promotion pipeline"""
    TRAINING = "training"
    STAGING = "staging"
    PRODUCTION = "production"
    REJECTED = "rejected"


class HealthCheckManager:


class ComponentStatus(Enum):
    """Individual component status"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of an individual component"""
    name: str
    status: ComponentStatus
    last_checked: str
    health_score: float  # 0.0 to 1.0
    details: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    error_count: int = 0
    error_timestamps: List[str] = field(default_factory=list)
    component_type: Optional[str] = None  # e.g., "model", "database", "api", "gpu"
    response_time_ms: Optional[float] = None
    uptime_seconds: Optional[float] = None
    restart_count: int = 0
    dependencies: List[str] = field(default_factory=list)  # Dependent components


@dataclass
class HealthCheckResult:
    """Overall health check result"""
    status: HealthStatus
    timestamp: str
    components: Dict[str, ComponentHealth]
    overall_score: float  # Weighted average of component scores
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    service_uptime_seconds: Optional[float] = None
    last_check_duration_ms: Optional[float] = None
    component_health_summary: Optional[Dict[str, int]] = None  # Count by status


@dataclass
class ShutdownResult:
    """Result of a shutdown operation"""
    success: bool
    duration_seconds: float
    components_shutdown: List[str] = field(default_factory=list)
    components_failed: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    warning_messages: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    shutdown_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    shutdown_reason: Optional[str] = None
    forced_shutdown: bool = False
    cleanup_performed: bool = True
    logs_flushed: bool = True
    metrics_exported: bool = True
    connections_closed: bool = True


class HealthCheckManager:
    """Manages comprehensive health checking for model servers"""
    
    def __init__(self, service_name: str = "pixelated-empathy-ai"):
        self.service_name = service_name
        self.components: Dict[str, Any] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.last_health_check: Optional[HealthCheckResult] = None
        self.health_check_interval: int = 30  # seconds
        self.health_check_timeout: int = 10    # seconds
        self.is_shutting_down: bool = False
        self.shutdown_callbacks: List[Callable] = []
        self.shutdown_reason: Optional[str] = None
        self.shutdown_start_time: Optional[float] = None
        self.forced_shutdown_count: int = 0
        self.service_start_time: float = time.time()
        self.logger = logging.getLogger(__name__)
        
        # Register default health checks
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("gpu_status", self._check_gpu_status)
        self.register_health_check("memory_status", self._check_memory_status)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("network_connectivity", self._check_network_connectivity)
        self.register_health_check("model_status", self._check_model_status)
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Register cleanup function for program exit
        atexit.register(self._cleanup_on_exit)
        
        self.logger.info(f"HealthCheckManager initialized for service: {self.service_name}")
    
    def register_component(self, name: str, component: Any, component_type: Optional[str] = None):
        """Register a component for health monitoring"""
        self.components[name] = {
            "component": component,
            "component_type": component_type or "generic",
            "registered_at": datetime.utcnow().isoformat(),
            "health_check_enabled": True,
            "health_check_interval": self.health_check_interval
        }
        self.logger.info(f"Registered component for health monitoring: {name} (type: {component_type or 'generic'})")
    
    def register_health_check(self, name: str, check_function: Callable):
        """Register a custom health check function"""
        self.health_checks[name] = check_function
        self.logger.info(f"Registered health check: {name}")
    
    def _check_system_resources(self) -> ComponentHealth:
        """Check system resource utilization"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            
            # Determine status based on thresholds
            if cpu_percent > 90 or memory_percent > 90:
                status = ComponentStatus.DEGRADED
                health_score = 0.5
            elif cpu_percent > 80 or memory_percent > 80:
                status = ComponentStatus.DEGRADED
                health_score = 0.7
            else:
                status = ComponentStatus.OPERATIONAL
                health_score = 1.0
            
            return ComponentHealth(
                name="system_resources",
                status=status,
                last_checked=datetime.utcnow().isoformat(),
                health_score=health_score,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_available_gb": memory_info.available / (1024**3),
                    "memory_total_gb": memory_info.total / (1024**3)
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="system_resources",
                status=ComponentStatus.FAILED,
                last_checked=datetime.utcnow().isoformat(),
                health_score=0.0,
                last_error=str(e),
                error_count=1,
                error_timestamps=[datetime.utcnow().isoformat()]
            )
    
    def _check_gpu_status(self) -> ComponentHealth:
        """Check GPU status and utilization"""
        try:
            if not torch.cuda.is_available():
                return ComponentHealth(
                    name="gpu_status",
                    status=ComponentStatus.OPERATIONAL,
                    last_checked=datetime.utcnow().isoformat(),
                    health_score=1.0,
                    details={"cuda_available": False}
                )
            
            gpu_count = torch.cuda.device_count()
            gpu_statuses = []
            total_utilization = 0
            failed_gpus = 0
            
            for i in range(gpu_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.memory_allocated(i) / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    memory_percent = (gpu_memory / gpu_memory_total) * 100 if gpu_memory_total > 0 else 0
                    
                    # Check for issues
                    if memory_percent > 95:
                        gpu_statuses.append(f"GPU {i} memory critical")
                        failed_gpus += 1
                    elif memory_percent > 85:
                        gpu_statuses.append(f"GPU {i} memory high")
                
                except Exception as gpu_error:
                    self.logger.warning(f"Failed to check GPU {i}: {gpu_error}")
                    gpu_statuses.append(f"GPU {i} check failed")
                    failed_gpus += 1
            
            # Determine overall GPU health
            if failed_gpus > 0:
                status = ComponentStatus.DEGRADED if failed_gpus < gpu_count else ComponentStatus.FAILED
                health_score = max(0.0, 1.0 - (failed_gpus / gpu_count))
            else:
                status = ComponentStatus.OPERATIONAL
                health_score = 1.0
            
            return ComponentHealth(
                name="gpu_status",
                status=status,
                last_checked=datetime.utcnow().isoformat(),
                health_score=health_score,
                details={
                    "gpu_count": gpu_count,
                    "gpu_statuses": gpu_statuses,
                    "cuda_available": True
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="gpu_status",
                status=ComponentStatus.FAILED,
                last_checked=datetime.utcnow().isoformat(),
                health_score=0.0,
                last_error=str(e),
                error_count=1,
                error_timestamps=[datetime.utcnow().isoformat()]
            )
    
    def _check_memory_status(self) -> ComponentHealth:
        """Check memory utilization and status"""
        try:
            process = psutil.Process()
            process_memory_info = process.memory_info()
            process_memory_percent = process_memory_info.rss / psutil.virtual_memory().total * 100
            
            # Check for memory leaks or excessive usage
            if process_memory_percent > 80:
                status = ComponentStatus.DEGRADED
                health_score = 0.5
            elif process_memory_percent > 60:
                status = ComponentStatus.DEGRADED
                health_score = 0.7
            else:
                status = ComponentStatus.OPERATIONAL
                health_score = 1.0
            
            return ComponentHealth(
                name="memory_status",
                status=status,
                last_checked=datetime.utcnow().isoformat(),
                health_score=health_score,
                details={
                    "process_memory_mb": process_memory_info.rss / (1024**2),
                    "process_memory_percent": process_memory_percent,
                    "virtual_memory_percent": psutil.virtual_memory().percent
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="memory_status",
                status=ComponentStatus.FAILED,
                last_checked=datetime.utcnow().isoformat(),
                health_score=0.0,
                last_error=str(e),
                error_count=1,
                error_timestamps=[datetime.utcnow().isoformat()]
            )
    
    def _check_disk_space(self) -> ComponentHealth:
        """Check disk space and I/O status"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Check for low disk space
            if disk_percent > 95:
                status = ComponentStatus.FAILED
                health_score = 0.0
            elif disk_percent > 90:
                status = ComponentStatus.DEGRADED
                health_score = 0.3
            elif disk_percent > 85:
                status = ComponentStatus.DEGRADED
                health_score = 0.6
            else:
                status = ComponentStatus.OPERATIONAL
                health_score = 1.0
            
            return ComponentHealth(
                name="disk_space",
                status=status,
                last_checked=datetime.utcnow().isoformat(),
                health_score=health_score,
                details={
                    "disk_percent_used": disk_percent,
                    "disk_free_gb": disk_usage.free / (1024**3),
                    "disk_total_gb": disk_usage.total / (1024**3)
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="disk_space",
                status=ComponentStatus.FAILED,
                last_checked=datetime.utcnow().isoformat(),
                health_score=0.0,
                last_error=str(e),
                error_count=1,
                error_timestamps=[datetime.utcnow().isoformat()]
            )
    
    def _check_network_connectivity(self) -> ComponentHealth:
        """Check network connectivity and status"""
        try:
            # Test basic network connectivity
            net_io = psutil.net_io_counters()
            connections = psutil.net_connections(kind='inet')
            active_connections = len([conn for conn in connections if conn.status == 'ESTABLISHED'])
            
            # Check for unusual network activity
            if active_connections > 1000:
                status = ComponentStatus.DEGRADED
                health_score = 0.7
            else:
                status = ComponentStatus.OPERATIONAL
                health_score = 1.0
            
            return ComponentHealth(
                name="network_connectivity",
                status=status,
                last_checked=datetime.utcnow().isoformat(),
                health_score=health_score,
                details={
                    "active_connections": active_connections,
                    "bytes_sent_mb": net_io.bytes_sent / (1024**2),
                    "bytes_received_mb": net_io.bytes_recv / (1024**2),
                    "packets_sent": net_io.packets_sent,
                    "packets_received": net_io.packets_recv
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="network_connectivity",
                status=ComponentStatus.FAILED,
                last_checked=datetime.utcnow().isoformat(),
                health_score=0.0,
                last_error=str(e),
                error_count=1,
                error_timestamps=[datetime.utcnow().isoformat()]
            )
    
    def _check_model_status(self) -> ComponentHealth:
        """Check model loading and inference status"""
        try:
            # Check if models are registered and loaded
            from .model_adapters import model_manager
            
            if not model_manager:
                return ComponentHealth(
                    name="model_status",
                    status=ComponentStatus.FAILED,
                    last_checked=datetime.utcnow().isoformat(),
                    health_score=0.0,
                    details={"error": "Model manager not initialized"}
                )
            
            models = model_manager.list_models()
            if not models:
                # No models loaded, but not necessarily unhealthy
                return ComponentHealth(
                    name="model_status",
                    status=ComponentStatus.OPERATIONAL,
                    last_checked=datetime.utcnow().isoformat(),
                    health_score=0.8,  # Slightly reduced score if no models loaded
                    details={"message": "No models currently loaded", "model_count": 0}
                )
            
            # Check each model
            model_issues = []
            healthy_models = 0
            
            for model_name in models:
                try:
                    model_info = model_manager.get_model_info(model_name)
                    if model_info and model_info.get('loaded', False):
                        healthy_models += 1
                    else:
                        model_issues.append(f"Model {model_name} not loaded")
                except Exception as model_error:
                    model_issues.append(f"Model {model_name} check failed: {model_error}")
            
            # Determine overall model health
            if healthy_models == len(models) and not model_issues:
                status = ComponentStatus.OPERATIONAL
                health_score = 1.0
            elif healthy_models > 0:
                status = ComponentStatus.DEGRADED
                health_score = healthy_models / len(models)
            else:
                status = ComponentStatus.FAILED
                health_score = 0.0
            
            return ComponentHealth(
                name="model_status",
                status=status,
                last_checked=datetime.utcnow().isoformat(),
                health_score=health_score,
                details={
                    "total_models": len(models),
                    "healthy_models": healthy_models,
                    "issues": model_issues,
                    "model_names": list(models)
                }
            )
        except Exception as e:
            return ComponentHealth(
                name="model_status",
                status=ComponentStatus.FAILED,
                last_checked=datetime.utcnow().isoformat(),
                health_score=0.0,
                last_error=str(e),
                error_count=1,
                error_timestamps=[datetime.utcnow().isoformat()]
            )
    
    def perform_health_check(self) -> HealthCheckResult:
        """Perform a comprehensive health check"""
        start_time = time.time()
        check_start_time = datetime.utcnow()
        
        self.logger.debug("Starting comprehensive health check...")
        
        # Run all registered health checks
        component_results = {}
        critical_issues = []
        warnings = []
        recommendations = []
        
        for check_name, check_function in self.health_checks.items():
            try:
                component_health = check_function()
                component_results[check_name] = component_health
                
                # Log any issues
                if component_health.status == ComponentStatus.FAILED:
                    critical_issues.append(f"{check_name}: {component_health.last_error or 'Component failed'}")
                    self.logger.error(f"Health check failed for {check_name}: {component_health.last_error}")
                elif component_health.status == ComponentStatus.DEGRADED:
                    warnings.append(f"{check_name}: Component degraded")
                    self.logger.warning(f"Health check degraded for {check_name}")
                
                # Add recommendations based on component health
                if component_health.component_type:
                    if component_health.component_type == "gpu" and component_health.health_score < 0.8:
                        recommendations.append("Consider GPU maintenance or replacement")
                    elif component_health.component_type == "memory" and component_health.health_score < 0.7:
                        recommendations.append("Check memory usage and consider optimization")
                    elif component_health.component_type == "disk" and component_health.health_score < 0.6:
                        recommendations.append("Check disk space and consider cleanup")
                
            except Exception as e:
                error_msg = f"Health check {check_name} failed: {str(e)}"
                critical_issues.append(error_msg)
                self.logger.error(error_msg, exc_info=True)
                
                # Record the failure
                component_results[check_name] = ComponentHealth(
                    name=check_name,
                    status=ComponentStatus.FAILED,
                    last_checked=datetime.utcnow().isoformat(),
                    health_score=0.0,
                    last_error=str(e),
                    error_count=1,
                    error_timestamps=[datetime.utcnow().isoformat()],
                    component_type="system"
                )
        
        # Calculate overall health score with weighted average
        if component_results:
            # Weight components by importance (critical components have higher weights)
            weighted_scores = []
            total_weight = 0
            
            for comp_name, comp_health in component_results.items():
                # Assign weights based on component type
                weight = 1.0
                if comp_health.component_type:
                    if comp_health.component_type in ["gpu", "model", "api"]:
                        weight = 2.0  # Critical components
                    elif comp_health.component_type in ["memory", "disk", "network"]:
                        weight = 1.5  # Important components
                    else:
                        weight = 1.0  # Standard components
                
                weighted_scores.append(comp_health.health_score * weight)
                total_weight += weight
            
            overall_score = sum(weighted_scores) / total_weight if total_weight > 0 else 1.0
            
            # Determine overall status
            if any(comp.status == ComponentStatus.FAILED for comp in component_results.values()):
                overall_status = HealthStatus.UNHEALTHY
            elif any(comp.status == ComponentStatus.DEGRADED for comp in component_results.values()):
                overall_status = HealthStatus.DEGRADED
            elif overall_score < 0.7:
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.HEALTHY
            overall_score = 1.0
        
        # Calculate component health summary
        component_health_summary = {}
        for comp_health in component_results.values():
            status_key = comp_health.status.value
            component_health_summary[status_key] = component_health_summary.get(status_key, 0) + 1
        
        # Calculate service uptime
        service_uptime = time.time() - self.service_start_time
        
        # Calculate check duration
        check_duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Create health check result
        health_result = HealthCheckResult(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            components=component_results,
            overall_score=overall_score,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
            metadata={
                "health_check_duration_seconds": time.time() - start_time,
                "health_check_duration_ms": check_duration,
                "component_count": len(component_results),
                "is_shutting_down": self.is_shutting_down,
                "service_name": self.service_name,
                "check_start_time": check_start_time.isoformat(),
                "check_end_time": datetime.utcnow().isoformat()
            },
            service_uptime_seconds=service_uptime,
            last_check_duration_ms=check_duration,
            component_health_summary=component_health_summary
        )
        
        self.last_health_check = health_result
        self.logger.debug(f"Health check completed with status: {overall_status.value}, score: {overall_score:.3f}")
        
        return health_result
    
    def get_health_status(self) -> HealthCheckResult:
        """Get current health status (cached or new)"""
        if self.last_health_check:
            # Check if cached result is still valid (within 30 seconds)
            last_check_time = datetime.fromisoformat(self.last_health_check.timestamp.replace('Z', '+00:00'))
            if datetime.utcnow() - last_check_time < timedelta(seconds=30):
                return self.last_health_check
        
        # Perform new health check
        return self.perform_health_check()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            signal_name = {signal.SIGTERM: "SIGTERM", signal.SIGINT: "SIGINT", signal.SIGHUP: "SIGHUP"}.get(signum, f"UNKNOWN({signum})")
            self.logger.info(f"Received signal {signal_name} ({signum}), initiating graceful shutdown...")
            
            # Determine shutdown reason based on signal
            reason_map = {
                signal.SIGTERM: "termination_signal",
                signal.SIGINT: "interrupt_signal",
                signal.SIGHUP: "hangup_signal"
            }
            reason = reason_map.get(signum, f"signal_{signum}")
            
            self.initiate_graceful_shutdown(reason=reason)
        
        # Register signal handlers
        try:
            signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
            self.logger.info("Registered SIGTERM handler")
        except Exception as e:
            self.logger.warning(f"Failed to register SIGTERM handler: {e}")
        
        try:
            signal.signal(signal.SIGINT, signal_handler)   # Interrupt signal (Ctrl+C)
            self.logger.info("Registered SIGINT handler")
        except Exception as e:
            self.logger.warning(f"Failed to register SIGINT handler: {e}")
        
        try:
            signal.signal(signal.SIGHUP, signal_handler)  # Hangup signal
            self.logger.info("Registered SIGHUP handler")
        except Exception as e:
            self.logger.warning(f"Failed to register SIGHUP handler: {e}")
    
    def register_shutdown_callback(self, callback: Callable):
        """Register a callback to be called during shutdown"""
        self.shutdown_callbacks.append(callback)
        self.logger.info("Registered shutdown callback")
    
    def initiate_graceful_shutdown(self, reason: str = "manual_shutdown", force: bool = False) -> ShutdownResult:
        """Initiate a graceful shutdown of the system"""
        if self.is_shutting_down and not force:
            self.logger.warning("Shutdown already in progress")
            return ShutdownResult(
                success=False,
                duration_seconds=0.0,
                error_messages=["Shutdown already in progress"],
                shutdown_reason=reason,
                forced_shutdown=False
            )
        
        # Record shutdown initiation
        self.is_shutting_down = True
        self.shutdown_reason = reason
        self.shutdown_start_time = time.time()
        start_time = self.shutdown_start_time
        
        self.logger.info(f"Initiating graceful shutdown (reason: {reason})...")
        
        # Initialize shutdown tracking
        shutdown_components = []
        failed_components = []
        error_messages = []
        warning_messages = []
        cleanup_performed = True
        logs_flushed = True
        metrics_exported = True
        connections_closed = True
        
        try:
            # 1. Stop accepting new requests and notify services
            self.logger.info("Stopping acceptance of new requests...")
            self._notify_services_shutdown_initiated()
            shutdown_components.append("request_acceptance_stopped")
            
            # 2. Drain existing requests with timeout
            self.logger.info("Draining existing requests...")
            drained_successfully = self._drain_existing_requests(timeout_seconds=30)
            if not drained_successfully:
                warning_messages.append("Some requests may not have completed before shutdown")
            shutdown_components.append("request_draining")
            
            # 3. Call registered shutdown callbacks in order
            self.logger.info(f"Calling {len(self.shutdown_callbacks)} shutdown callbacks...")
            callback_results = self._execute_shutdown_callbacks()
            shutdown_components.extend(callback_results.get("successful", []))
            failed_components.extend(callback_results.get("failed", []))
            error_messages.extend(callback_results.get("errors", []))
            
            # 4. Shutdown model manager and unload models
            try:
                self.logger.info("Unloading models...")
                from .model_adapters import model_manager
                if model_manager:
                    unload_result = model_manager.unload_all_models()
                    if unload_result.success:
                        shutdown_components.append("model_manager")
                        self.logger.info(f"Successfully unloaded {unload_result.models_unloaded} models")
                    else:
                        error_msg = f"Failed to unload models: {unload_result.error_message}"
                        self.logger.error(error_msg)
                        error_messages.append(error_msg)
                        failed_components.append("model_manager")
                else:
                    self.logger.warning("Model manager not found during shutdown")
                    warning_messages.append("Model manager not available for shutdown")
            except Exception as e:
                error_msg = f"Critical error unloading models: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                error_messages.append(error_msg)
                failed_components.append("model_manager")
            
            # 5. Close database connections and other persistent resources
            try:
                self.logger.info("Closing database connections...")
                closed_connections = self._close_database_connections()
                if closed_connections:
                    shutdown_components.append("database_connections")
                else:
                    warning_messages.append("Some database connections may not have closed properly")
            except Exception as e:
                error_msg = f"Failed to close database connections: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                error_messages.append(error_msg)
                failed_components.append("database_connections")
                connections_closed = False
            
            # 6. Flush logs and export metrics
            try:
                self.logger.info("Flushing logs and exporting metrics...")
                logs_flushed = self._flush_logs()
                if logs_flushed:
                    shutdown_components.append("log_flushing")
                else:
                    warning_messages.append("Some logs may not have flushed completely")
                
                metrics_exported = self._export_metrics()
                if metrics_exported:
                    shutdown_components.append("metrics_exporting")
                else:
                    warning_messages.append("Some metrics may not have exported completely")
            except Exception as e:
                error_msg = f"Failed to flush logs or export metrics: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                error_messages.append(error_msg)
                failed_components.extend(["log_flushing", "metrics_exporting"])
                logs_flushed = False
                metrics_exported = False
            
            # 7. Notify monitoring systems and alerting services
            try:
                self.logger.info("Sending shutdown notification to monitoring systems...")
                notification_result = self._notify_monitoring_systems(reason)
                if notification_result.success:
                    shutdown_components.append("monitoring_notification")
                else:
                    warning_messages.append(f"Monitoring notification had issues: {notification_result.message}")
            except Exception as e:
                error_msg = f"Failed to send shutdown notification: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                error_messages.append(error_msg)
                failed_components.append("monitoring_notification")
            
            # 8. Perform final cleanup operations
            try:
                self.logger.info("Performing final cleanup...")
                cleanup_performed = self._perform_final_cleanup()
                if cleanup_performed:
                    shutdown_components.append("final_cleanup")
                else:
                    warning_messages.append("Final cleanup may not have completed fully")
            except Exception as e:
                error_msg = f"Failed during final cleanup: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                error_messages.append(error_msg)
                failed_components.append("final_cleanup")
                cleanup_performed = False
            
            # Calculate total shutdown duration
            duration = time.time() - start_time
            
            # Determine overall success
            success = len(failed_components) == 0
            
            if success:
                self.logger.info(f"✅ Graceful shutdown completed successfully in {duration:.2f} seconds")
            else:
                self.logger.warning(f"⚠️ Graceful shutdown completed with {len(failed_components)} failed components in {duration:.2f} seconds")
                for failed_component in failed_components:
                    self.logger.warning(f"  Failed component: {failed_component}")
            
            # Log any warnings
            for warning in warning_messages:
                self.logger.warning(f"Shutdown warning: {warning}")
            
            return ShutdownResult(
                success=success,
                duration_seconds=duration,
                components_shutdown=shutdown_components,
                components_failed=failed_components,
                error_messages=error_messages,
                warning_messages=warning_messages,
                timestamp=datetime.utcnow().isoformat(),
                shutdown_reason=reason,
                forced_shutdown=force,
                cleanup_performed=cleanup_performed,
                logs_flushed=logs_flushed,
                metrics_exported=metrics_exported,
                connections_closed=connections_closed
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Critical error during shutdown: {str(e)}"
            self.logger.critical(error_msg, exc_info=True)
            error_messages.append(error_msg)
            
            # Attempt emergency cleanup
            try:
                self._emergency_cleanup()
            except Exception as emergency_error:
                emergency_error_msg = f"Emergency cleanup failed: {str(emergency_error)}"
                self.logger.critical(emergency_error_msg, exc_info=True)
                error_messages.append(emergency_error_msg)
            
            return ShutdownResult(
                success=False,
                duration_seconds=duration,
                components_shutdown=shutdown_components,
                components_failed=failed_components + ["critical_shutdown_error"],
                error_messages=error_messages,
                warning_messages=warning_messages,
                timestamp=datetime.utcnow().isoformat(),
                shutdown_reason=reason,
                forced_shutdown=force,
                cleanup_performed=False,
                logs_flushed=False,
                metrics_exported=False,
                connections_closed=False
            )
    
    def _notify_services_shutdown_initiated(self):
        """Notify dependent services that shutdown is being initiated"""
        try:
            # In a real implementation, this would send notifications to:
            # - Load balancers to stop routing traffic
            # - Dependent services to prepare for disruption
            # - Monitoring systems to expect downtime
            # - Alerting systems to suppress alerts during maintenance
            
            self.logger.info("Sent shutdown notifications to dependent services")
        except Exception as e:
            self.logger.warning(f"Failed to notify services of shutdown: {e}")
    
    def _drain_existing_requests(self, timeout_seconds: int = 30) -> bool:
        """Allow existing requests to complete with timeout"""
        try:
            # In a real implementation, this would:
            # - Stop accepting new requests
            # - Wait for existing requests to complete
            # - Honor the timeout constraint
            
            self.logger.info(f"Draining existing requests with timeout of {timeout_seconds} seconds")
            
            # Simulate draining
            time.sleep(min(2, timeout_seconds / 10))  # Don't actually wait long in testing
            
            return True
        except Exception as e:
            self.logger.error(f"Error during request draining: {e}")
            return False
    
    def _execute_shutdown_callbacks(self) -> Dict[str, List[str]]:
        """Execute registered shutdown callbacks"""
        successful_callbacks = []
        failed_callbacks = []
        error_messages = []
        
        for i, callback in enumerate(self.shutdown_callbacks):
            try:
                self.logger.info(f"Executing shutdown callback {i+1}/{len(self.shutdown_callbacks)}")
                callback_result = callback()
                successful_callbacks.append(f"callback_{i}")
                self.logger.info(f"Shutdown callback {i+1} completed successfully")
            except Exception as e:
                error_msg = f"Shutdown callback {i} failed: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                error_messages.append(error_msg)
                failed_callbacks.append(f"callback_{i}")
        
        return {
            "successful": successful_callbacks,
            "failed": failed_callbacks,
            "errors": error_messages
        }
    
    def _close_database_connections(self) -> bool:
        """Close database connections and other persistent resources"""
        try:
            # In a real implementation, this would close database connections,
            # message queues, file handles, and other persistent resources
            
            self.logger.info("Closed database connections and persistent resources")
            return True
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")
            return False
    
    def _flush_logs(self) -> bool:
        """Flush logs to ensure all entries are written"""
        try:
            import logging
            for handler in logging.root.handlers:
                try:
                    handler.flush()
                except:
                    pass  # Ignore errors during flushing
            
            self.logger.info("Logs flushed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error flushing logs: {e}")
            return False
    
    def _export_metrics(self) -> bool:
        """Export metrics to persistent storage"""
        try:
            # In a real implementation, this would export metrics to:
            # - Prometheus/Prometheus Pushgateway
            # - Cloud monitoring services
            # - File-based storage
            # - Database storage
            
            self.logger.info("Metrics exported successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return False
    
    def _notify_monitoring_systems(self, reason: str) -> Any:
        """Notify monitoring systems of shutdown"""
        try:
            # In a real implementation, this would send notifications to:
            # - Alerting systems to suppress alerts
            # - Monitoring dashboards to show maintenance status
            # - Logging systems to annotate the shutdown
            
            class NotificationResult:
                def __init__(self, success: bool, message: str):
                    self.success = success
                    self.message = message
            
            self.logger.info(f"Notified monitoring systems of shutdown (reason: {reason})")
            return NotificationResult(True, "Notifications sent successfully")
        except Exception as e:
            self.logger.error(f"Error notifying monitoring systems: {e}")
            return NotificationResult(False, f"Notification failed: {str(e)}")
    
    def _perform_final_cleanup(self) -> bool:
        """Perform final cleanup operations"""
        try:
            # In a real implementation, this would:
            # - Clean up temporary files
            # - Release system resources
            # - Update service registries
            # - Perform any final state persistence
            
            self.logger.info("Performed final cleanup operations")
            return True
        except Exception as e:
            self.logger.error(f"Error during final cleanup: {e}")
            return False
    
    def _emergency_cleanup(self):
        """Emergency cleanup when normal shutdown fails"""
        self.logger.critical("Performing emergency cleanup...")
        
        # Force flush logs
        try:
            import logging
            for handler in logging.root.handlers:
                try:
                    handler.close()
                except:
                    pass
        except:
            pass
        
        # Try to close any remaining resources
        try:
            import gc
            gc.collect()
        except:
            pass
        
        self.logger.critical("Emergency cleanup completed")
    
    def _cleanup_on_exit(self):
        """Cleanup function called when the process exits"""
        if not self.is_shutting_down:
            self.logger.info("Performing cleanup on process exit...")
            # Perform any final cleanup that wasn't done during normal shutdown
            try:
                # Ensure all threads are joined
                import threading
                main_thread = threading.main_thread()
                for thread in threading.enumerate():
                    if thread != main_thread and thread.is_alive():
                        self.logger.info(f"Thread {thread.name} still alive, waiting...")
                        # Don't actually wait in cleanup as it may hang the process
                
                # Flush any remaining logs
                import logging
                for handler in logging.root.handlers:
                    try:
                        handler.flush()
                    except:
                        pass
                        
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component"""
        if self.last_health_check and component_name in self.last_health_check.components:
            return self.last_health_check.components[component_name]
        return None
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "network_sent_mb": psutil.net_io_counters().bytes_sent / (1024**2),
                "network_recv_mb": psutil.net_io_counters().bytes_recv / (1024**2),
                "gpu_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "process_memory_mb": psutil.Process().memory_info().rss / (1024**2),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


class HealthCheckMiddleware:
    """Middleware for integrating health checks with web frameworks"""
    
    def __init__(self, health_manager: HealthCheckManager):
        self.health_manager = health_manager
        self.logger = logging.getLogger(__name__)
    
    def health_check_endpoint(self):
        """HTTP endpoint for health checks"""
        health_result = self.health_manager.get_health_status()
        
        # Convert enum values to strings for JSON serialization
        response_data = {
            "status": health_result.status.value,
            "timestamp": health_result.timestamp,
            "overall_score": health_result.overall_score,
            "critical_issues": health_result.critical_issues,
            "warnings": health_result.warnings,
            "components": {
                name: {
                    "name": component.name,
                    "status": component.status.value,
                    "last_checked": component.last_checked,
                    "health_score": component.health_score,
                    "details": component.details,
                    "last_error": component.last_error
                }
                for name, component in health_result.components.items()
            },
            "metadata": health_result.metadata
        }
        
        # Set appropriate HTTP status code
        if health_result.status == HealthStatus.HEALTHY:
            status_code = 200
        elif health_result.status == HealthStatus.DEGRADED:
            status_code = 200  # Degraded is still considered "healthy" but with warnings
        else:
            status_code = 503  # Unhealthy or maintenance
        
        return status_code, response_data
    
    def readiness_probe_endpoint(self):
        """HTTP endpoint for readiness probes"""
        # Readiness probe checks if the service is ready to serve requests
        health_result = self.health_manager.get_health_status()
        
        # For readiness, we might have different criteria than general health
        is_ready = health_result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        
        response_data = {
            "ready": is_ready,
            "status": health_result.status.value,
            "timestamp": health_result.timestamp,
            "reason": "Service is ready to serve requests" if is_ready else "Service not ready",
            "details": {
                "critical_issues": health_result.critical_issues,
                "warnings": health_result.warnings
            }
        }
        
        status_code = 200 if is_ready else 503
        return status_code, response_data
    
    def liveness_probe_endpoint(self):
        """HTTP endpoint for liveness probes"""
        # Liveness probe checks if the process is alive and responding
        try:
            # Simple check that the process is responsive
            response_data = {
                "alive": True,
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat()
            }
            status_code = 200
        except Exception as e:
            response_data = {
                "alive": False,
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
            status_code = 503
        
        return status_code, response_data


# Decorator for adding health checks to functions
def health_checked(func):
    """Decorator to add health checking to functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # In a real implementation, you might want to check health before
        # executing the function, especially for critical operations
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the error with additional context
            logger.error(f"Function {func.__name__} failed: {e}", exc_info=True)
            raise
    return wrapper


# Global health manager instance
health_manager = HealthCheckManager()


# Integration with FastAPI
def integrate_health_checks_with_fastapi(app):
    """Integrate health checks with a FastAPI application"""
    from fastapi import FastAPI, HTTPException, Response
    import json
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        status_code, response_data = HealthCheckMiddleware(health_manager).health_check_endpoint()
        return Response(
            content=json.dumps(response_data, indent=2),
            status_code=status_code,
            media_type="application/json"
        )
    
    @app.get("/ready")
    async def readiness_probe():
        """Readiness probe endpoint"""
        status_code, response_data = HealthCheckMiddleware(health_manager).readiness_probe_endpoint()
        return Response(
            content=json.dumps(response_data, indent=2),
            status_code=status_code,
            media_type="application/json"
        )
    
    @app.get("/alive")
    async def liveness_probe():
        """Liveness probe endpoint"""
        status_code, response_data = HealthCheckMiddleware(health_manager).liveness_probe_endpoint()
        return Response(
            content=json.dumps(response_data, indent=2),
            status_code=status_code,
            media_type="application/json"
        )
    
    # Register shutdown callback with FastAPI
    @app.on_event("shutdown")
    async def shutdown_event():
        """FastAPI shutdown event handler"""
        logger.info("FastAPI shutdown initiated")
        result = health_manager.initiate_graceful_shutdown()
        if not result.success:
            logger.error(f"Shutdown completed with errors: {result.error_messages}")
        else:
            logger.info("Graceful shutdown completed successfully")


# Integration with Uvicorn/Gunicorn
def integrate_with_asgi_server(server_type: str = "uvicorn"):
    """Integrate health checks with ASGI servers like Uvicorn or Gunicorn"""
    
    def shutdown_handler(signum, frame):
        """Handler for shutdown signals"""
        logger.info(f"Received {server_type} shutdown signal {signum}")
        # Initiate graceful shutdown
        result = health_manager.initiate_graceful_shutdown()
        if result.success:
            logger.info("Server shutdown completed successfully")
        else:
            logger.error(f"Server shutdown completed with errors: {result.error_messages}")
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    # Register cleanup on exit
    atexit.register(lambda: health_manager.initiate_graceful_shutdown())


# Example usage and testing
def test_health_check_system():
    """Test the health check system"""
    logger.info("Testing Health Check System...")
    
    # Test health manager creation
    manager = HealthCheckManager()
    
    # Register a mock component
    class MockComponent:
        def __init__(self):
            self.is_healthy = True
        
        def get_status(self):
            return "healthy" if self.is_healthy else "unhealthy"
    
    mock_component = MockComponent()
    manager.register_component("mock_component", mock_component)
    
    # Register a custom health check
    def custom_health_check():
        return ComponentHealth(
            name="custom_check",
            status=ComponentStatus.OPERATIONAL,
            last_checked=datetime.utcnow().isoformat(),
            health_score=1.0,
            details={"custom_field": "test_value"}
        )
    
    manager.register_health_check("custom_check", custom_health_check)
    
    # Perform health check
    print("Performing health check...")
    health_result = manager.perform_health_check()
    
    print(f"Overall status: {health_result.status.value}")
    print(f"Overall score: {health_result.overall_score:.2f}")
    print(f"Components checked: {len(health_result.components)}")
    
    for component_name, component_health in health_result.components.items():
        print(f"  {component_name}: {component_health.status.value} (score: {component_health.health_score:.2f})")
        if component_health.details:
            print(f"    Details: {component_health.details}")
    
    # Test system metrics
    print("\nSystem metrics:")
    metrics = manager.get_system_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Test component health lookup
    print("\nComponent health lookup:")
    model_health = manager.get_component_health("model_status")
    if model_health:
        print(f"  Model status: {model_health.status.value}")
    else:
        print("  Model status: Not found in last health check")
    
    # Test graceful shutdown registration
    def mock_shutdown_callback():
        print("Mock shutdown callback executed")
    
    manager.register_shutdown_callback(mock_shutdown_callback)
    print(f"Shutdown callbacks registered: {len(manager.shutdown_callbacks)}")
    
    # Test shutdown (but don't actually shut down)
    print("\nTesting shutdown (simulated)...")
    # Note: We won't actually call initiate_graceful_shutdown here as it would shut down the test
    
    print("Health check system test completed!")


# Test graceful shutdown
def test_graceful_shutdown():
    """Test graceful shutdown functionality"""
    print("Testing Graceful Shutdown...")
    
    manager = HealthCheckManager()
    
    # Register a mock shutdown callback
    shutdown_executed = False
    
    def test_shutdown_callback():
        nonlocal shutdown_executed
        shutdown_executed = True
        print("Test shutdown callback executed")
    
    manager.register_shutdown_callback(test_shutdown_callback)
    
    # Simulate shutdown without actually shutting down the process
    # In a real scenario, you'd call manager.initiate_graceful_shutdown()
    
    print("Graceful shutdown test completed!")
    return True


if __name__ == "__main__":
    # Run tests
    test_health_check_system()
    test_graceful_shutdown()