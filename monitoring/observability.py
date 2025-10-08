"""
Observability system for Pixelated Empathy AI project.
Implements comprehensive logging, monitoring, and metrics collection.
"""

import json
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import torch
import numpy as np
from functools import wraps
import hashlib
import re


logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Logging levels for observability"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to collect"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    MODEL_PERFORMANCE = "model_performance"
    USER_ACTIVITY = "user_activity"
    SECURITY_EVENTS = "security_events"


@dataclass
class LogEntry:
    """Structured log entry for observability"""
    timestamp: str
    level: LogLevel
    service: str
    message: str
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    model_name: Optional[str] = None
    trace_id: Optional[str] = None
    extra_fields: Optional[Dict[str, Any]] = None


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: str
    tags: Optional[Dict[str, str]] = None
    unit: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Span:
    """Distributed tracing span for request tracking"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"
    tags: Optional[Dict[str, Any]] = None
    logs: List[LogEntry] = field(default_factory=list)


class RedactionEngine:
    """Engine for redacting sensitive information from logs"""
    
    def __init__(self):
        self.redaction_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'ssn': r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b',
            'api_key': r'(?:api[_-]?key|token)[=:]\s*[a-zA-Z0-9_-]{20,}',
            'password': r'(?:password|passwd|pwd)[=:]\s*[^\s&]+'
        }
        self.placeholder = '[REDACTED]'
    
    def redact_text(self, text: str) -> str:
        """Redact sensitive information from text"""
        if not isinstance(text, str):
            return str(text)
        
        redacted_text = text
        for pattern_name, pattern in self.redaction_patterns.items():
            redacted_text = re.sub(pattern, f'{self.placeholder}_{pattern_name.upper()}', redacted_text, flags=re.IGNORECASE)
        
        return redacted_text
    
    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from dictionary"""
        if not isinstance(data, dict):
            return data
        
        redacted_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                # Check if the key itself might contain sensitive info
                if self._is_sensitive_key(key):
                    redacted_data[key] = self.placeholder
                else:
                    redacted_data[key] = self.redact_text(value)
            elif isinstance(value, dict):
                redacted_data[key] = self.redact_dict(value)
            elif isinstance(value, list):
                redacted_data[key] = [self.redact_text(item) if isinstance(item, str) else item for item in value]
            else:
                redacted_data[key] = value
        
        return redacted_data
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key indicates sensitive information"""
        sensitive_keywords = ['password', 'secret', 'token', 'key', 'api_key', 'auth']
        key_lower = key.lower()
        return any(keyword in key_lower for keyword in sensitive_keywords)


class Logger:
    """Enhanced logger with structured logging and redaction"""
    
    def __init__(self, service_name: str = "pixelated-empathy"):
        self.service_name = service_name
        self.redaction_engine = RedactionEngine()
        self.logger = logging.getLogger(service_name)
    
    def log(self, 
            level: LogLevel, 
            message: str, 
            context: Optional[Dict[str, Any]] = None,
            user_id: Optional[str] = None,
            request_id: Optional[str] = None,
            model_name: Optional[str] = None,
            trace_id: Optional[str] = None,
            **extra_fields) -> LogEntry:
        """Log a structured message"""
        # Redact sensitive information
        redacted_message = self.redaction_engine.redact_text(message)
        redacted_context = self.redaction_engine.redact_dict(context) if context else None
        redacted_extra = self.redaction_engine.redact_dict(extra_fields) if extra_fields else None
        
        # Create log entry
        log_entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level=level,
            service=self.service_name,
            message=redacted_message,
            context=redacted_context,
            user_id=user_id,
            request_id=request_id,
            model_name=model_name,
            trace_id=trace_id,
            extra_fields=redacted_extra
        )
        
        # Actually log the message
        log_func = getattr(self.logger, level.value)
        log_message = f"[{self.service_name}] {redacted_message}"
        if redacted_context:
            log_message += f" | Context: {json.dumps(redacted_context, default=str)}"
        if redacted_extra:
            log_message += f" | Extra: {json.dumps(redacted_extra, default=str)}"
        
        log_func(log_message)
        
        return log_entry
    
    def debug(self, message: str, **kwargs) -> LogEntry:
        return self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> LogEntry:
        return self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> LogEntry:
        return self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> LogEntry:
        return self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> LogEntry:
        return self.log(LogLevel.CRITICAL, message, **kwargs)


class MetricsCollector:
    """Collector for system and application metrics"""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.histograms: Dict[str, List[float]] = {}
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     metric_type: MetricType,
                     tags: Optional[Dict[str, str]] = None,
                     unit: Optional[str] = None,
                     description: Optional[str] = None) -> Metric:
        """Record a single metric point"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.utcnow().isoformat(),
            tags=tags,
            unit=unit,
            description=description
        )
        
        self.metrics.append(metric)
        return metric
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        self.counters[name] = self.counters.get(name, 0) + value
        self.record_metric(
            name=name,
            value=self.counters[name],
            metric_type=MetricType.THROUGHPUT,
            tags=tags,
            description=f"Counter: {name}"
        )
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        self.gauges[name] = value
        self.record_metric(
            name=name,
            value=value,
            metric_type=MetricType.RESOURCE_USAGE,
            tags=tags,
            description=f"Gauge: {name}"
        )
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        if name not in self.histograms:
            self.histograms[name] = []
        self.histograms[name].append(value)
        
        # Keep only recent values to prevent memory issues
        if len(self.histograms[name]) > 10000:
            self.histograms[name] = self.histograms[name][-1000:]
        
        self.record_metric(
            name=name,
            value=value,
            metric_type=MetricType.LATENCY,
            tags=tags,
            unit="milliseconds",
            description=f"Histogram: {name}"
        )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_percent': psutil.disk_usage('/').percent,
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3),
            'network_sent_mb': psutil.net_io_counters().bytes_sent / (1024**2),
            'network_recv_mb': psutil.net_io_counters().bytes_recv / (1024**2)
        }
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            metrics['gpu_count'] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                metrics[f'gpu_{i}_memory_gb'] = torch.cuda.memory_allocated(i) / (1024**3)
                metrics[f'gpu_{i}_memory_percent'] = (torch.cuda.memory_allocated(i) / 
                                                    torch.cuda.get_device_properties(i).total_memory) * 100
                metrics[f'gpu_{i}_utilization'] = torch.cuda.utilization(i) if torch.cuda.utilization else 0
        
        return metrics


class Tracer:
    """Distributed tracing system"""
    
    def __init__(self):
        self.spans: Dict[str, Span] = {}
        self.active_spans: Dict[str, str] = {}  # trace_id -> span_id
    
    def start_span(self, 
                   operation_name: str,
                   trace_id: Optional[str] = None,
                   parent_span_id: Optional[str] = None,
                   tags: Optional[Dict[str, Any]] = None) -> Span:
        """Start a new tracing span"""
        if not trace_id:
            trace_id = self._generate_id()
        
        span_id = self._generate_id()
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            tags=tags or {}
        )
        
        self.spans[span_id] = span
        self.active_spans[trace_id] = span_id
        
        return span
    
    def end_span(self, span: Span, status: str = "success"):
        """End a tracing span"""
        span.end_time = time.time()
        span.status = status
        
        # Remove from active spans
        if span.trace_id in self.active_spans and self.active_spans[span.trace_id] == span.span_id:
            del self.active_spans[span.trace_id]
    
    def add_span_log(self, span: Span, log_entry: LogEntry):
        """Add a log entry to a span"""
        span.logs.append(log_entry)
    
    def get_span_duration(self, span: Span) -> float:
        """Get the duration of a span in milliseconds"""
        if span.end_time:
            return (span.end_time - span.start_time) * 1000
        else:
            return (time.time() - span.start_time) * 1000
    
    def _generate_id(self) -> str:
        """Generate a unique ID for tracing"""
        return hashlib.md5(f"{time.time()}{id(self)}".encode()).hexdigest()[:16]


class ObservabilityManager:
    """Main observability manager coordinating logging, metrics, and tracing"""
    
    def __init__(self, service_name: str = "pixelated-empathy"):
        self.service_name = service_name
        self.logger = Logger(service_name)
        self.metrics_collector = MetricsCollector()
        self.tracer = Tracer()
        self.request_metrics = {}
    
    def log_request(self, 
                   request_id: str,
                   user_id: Optional[str] = None,
                   model_name: Optional[str] = None,
                   input_data: Optional[Union[Dict, str]] = None,
                   output_data: Optional[Union[Dict, str]] = None,
                   processing_time_ms: Optional[float] = None,
                   status: str = "success",
                   error_message: Optional[str] = None,
                   trace_id: Optional[str] = None):
        """Log a complete request with all relevant information"""
        # Record processing time
        if processing_time_ms is not None:
            self.metrics_collector.record_histogram("request_latency_ms", processing_time_ms)
            self.metrics_collector.increment_counter("requests_processed")
        
        # Record error if applicable
        if status == "error":
            self.metrics_collector.increment_counter("request_errors")
            self.logger.error(
                f"Request failed: {error_message or 'Unknown error'}",
                user_id=user_id,
                request_id=request_id,
                model_name=model_name,
                trace_id=trace_id
            )
        else:
            self.logger.info(
                f"Request processed successfully in {processing_time_ms:.2f}ms",
                user_id=user_id,
                request_id=request_id,
                model_name=model_name,
                trace_id=trace_id
            )
    
    def log_model_inference(self,
                           model_name: str,
                           input_tokens: int,
                           output_tokens: int,
                           processing_time_ms: float,
                           success: bool = True,
                           error_type: Optional[str] = None):
        """Log model inference metrics"""
        # Record inference metrics
        self.metrics_collector.record_histogram("model_inference_latency_ms", processing_time_ms)
        self.metrics_collector.record_metric(
            "model_input_tokens",
            input_tokens,
            MetricType.MODEL_PERFORMANCE,
            tags={"model": model_name}
        )
        self.metrics_collector.record_metric(
            "model_output_tokens",
            output_tokens,
            MetricType.MODEL_PERFORMANCE,
            tags={"model": model_name}
        )
        
        # Increment counters
        self.metrics_collector.increment_counter("model_inferences")
        if success:
            self.metrics_collector.increment_counter("model_inferences_successful")
        else:
            self.metrics_collector.increment_counter("model_inferences_failed")
            if error_type:
                self.metrics_collector.increment_counter(f"model_inferences_error_{error_type}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        return self.metrics_collector.get_system_metrics()
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get aggregated service metrics"""
        # Calculate rates and averages
        requests_processed = self.metrics_collector.counters.get("requests_processed", 0)
        request_errors = self.metrics_collector.counters.get("request_errors", 0)
        error_rate = request_errors / requests_processed if requests_processed > 0 else 0.0
        
        # Get latency metrics
        latencies = self.metrics_collector.histograms.get("request_latency_ms", [])
        avg_latency = np.mean(latencies) if latencies else 0.0
        p95_latency = np.percentile(latencies, 95) if latencies else 0.0
        p99_latency = np.percentile(latencies, 99) if latencies else 0.0
        
        return {
            "requests_processed": requests_processed,
            "request_errors": request_errors,
            "error_rate": error_rate,
            "average_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "system_metrics": self.get_system_health()
        }
    
    def create_traced_request(self, 
                            operation_name: str,
                            user_id: Optional[str] = None,
                            model_name: Optional[str] = None) -> Span:
        """Create a traced request span"""
        trace_id = self.tracer._generate_id()
        span = self.tracer.start_span(
            operation_name=operation_name,
            trace_id=trace_id,
            tags={
                "user_id": user_id,
                "model_name": model_name,
                "service": self.service_name
            }
        )
        
        # Log the start
        self.logger.info(
            f"Starting traced operation: {operation_name}",
            user_id=user_id,
            model_name=model_name,
            trace_id=trace_id,
            span_id=span.span_id
        )
        
        return span
    
    def end_traced_request(self, span: Span, status: str = "success", error_message: Optional[str] = None):
        """End a traced request"""
        self.tracer.end_span(span, status)
        duration = self.tracer.get_span_duration(span)
        
        # Log completion
        if status == "success":
            self.logger.info(
                f"Completed traced operation: {span.operation_name} in {duration:.2f}ms",
                trace_id=span.trace_id,
                span_id=span.span_id,
                duration_ms=duration
            )
        else:
            self.logger.error(
                f"Failed traced operation: {span.operation_name} after {duration:.2f}ms - {error_message or 'Unknown error'}",
                trace_id=span.trace_id,
                span_id=span.span_id,
                duration_ms=duration
            )


# Decorator for automatic observability
def observable(func):
    """Decorator to automatically add observability to functions"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Create observability manager
        obs_manager = ObservabilityManager()
        
        # Create span for the function call
        span = obs_manager.create_traced_request(
            operation_name=f"{func.__module__}.{func.__name__}",
            user_id=kwargs.get('user_id')
        )
        
        start_time = time.time()
        
        try:
            # Call the original function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # End the span successfully
            obs_manager.end_traced_request(span, "success")
            
            # Log the successful execution
            duration_ms = (time.time() - start_time) * 1000
            obs_manager.logger.info(
                f"Function {func.__name__} executed successfully in {duration_ms:.2f}ms",
                trace_id=span.trace_id,
                duration_ms=duration_ms
            )
            
            return result
            
        except Exception as e:
            # End the span with error
            obs_manager.end_traced_request(span, "error", str(e))
            
            # Log the error
            obs_manager.logger.error(
                f"Function {func.__name__} failed: {str(e)}",
                trace_id=span.trace_id,
                error=str(e)
            )
            
            # Re-raise the exception
            raise
    
    # Handle both sync and async functions
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        return sync_wrapper


# Global observability manager instance
observability = ObservabilityManager()


# Middleware for FastAPI integration
def observability_middleware(app):
    """Add observability middleware to FastAPI app"""
    @app.middleware("http")
    async def add_observability(request, call_next):
        # Generate request ID
        request_id = hashlib.md5(f"{time.time()}{request.client.host}".encode()).hexdigest()[:16]
        
        # Create span for the request
        span = observability.create_traced_request(
            operation_name=f"{request.method} {request.url.path}",
            user_id=getattr(request.state, 'user_id', None) if hasattr(request.state, 'user_id') else None
        )
        
        start_time = time.time()
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Log the request
            observability.log_request(
                request_id=request_id,
                user_id=getattr(request.state, 'user_id', None) if hasattr(request.state, 'user_id') else None,
                model_name=getattr(request.state, 'model_name', None) if hasattr(request.state, 'model_name') else None,
                processing_time_ms=processing_time_ms,
                status="success",
                trace_id=span.trace_id
            )
            
            # End the span
            observability.end_traced_request(span, "success")
            
            # Add observability headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = str(processing_time_ms)
            response.headers["X-Trace-ID"] = span.trace_id
            
            return response
            
        except Exception as e:
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Log the error
            observability.log_request(
                request_id=request_id,
                user_id=getattr(request.state, 'user_id', None) if hasattr(request.state, 'user_id') else None,
                model_name=getattr(request.state, 'model_name', None) if hasattr(request.state, 'model_name') else None,
                processing_time_ms=processing_time_ms,
                status="error",
                error_message=str(e),
                trace_id=span.trace_id
            )
            
            # End the span with error
            observability.end_traced_request(span, "error", str(e))
            
            # Re-raise the exception
            raise
    
    return app


# Example usage and testing
def test_observability():
    """Test the observability system"""
    logger.info("Testing Observability System...")
    
    # Test logging
    log_entry = observability.logger.info(
        "Test log message with user context",
        user_id="user_123",
        model_name="therapy_model_v1",
        extra_field="test_value"
    )
    print(f"Logged: {log_entry.message}")
    
    # Test metrics collection
    observability.metrics_collector.record_histogram("request_latency_ms", 150.5)
    observability.metrics_collector.record_histogram("request_latency_ms", 200.2)
    observability.metrics_collector.increment_counter("requests_processed")
    observability.metrics_collector.increment_counter("requests_processed")
    
    # Test system metrics
    system_metrics = observability.get_system_health()
    print(f"System metrics collected: {len(system_metrics)} metrics")
    
    # Test service metrics
    service_metrics = observability.get_service_metrics()
    print(f"Service metrics: {service_metrics}")
    
    # Test tracing
    span = observability.create_traced_request("test_operation", "user_123", "test_model")
    time.sleep(0.1)  # Simulate work
    observability.end_traced_request(span, "success")
    
    # Test model inference logging
    observability.log_model_inference(
        model_name="therapy_model_v1",
        input_tokens=150,
        output_tokens=75,
        processing_time_ms=180.5,
        success=True
    )
    
    print("Observability system test completed!")


if __name__ == "__main__":
    test_observability()