"""Monitoring dashboard types and utilities."""
from enum import Enum


class MetricType(Enum):
    """Types of metrics that can be monitored."""

    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    QUEUE_DEPTH = "queue_depth"
    PROCESSING_TIME = "processing_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
