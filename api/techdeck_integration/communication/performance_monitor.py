"""
Performance Monitor for Pipeline Communication - Comprehensive Metrics Collection.

This module provides sophisticated performance monitoring with sub-50ms tracking,
HIPAA++ compliant metrics collection, and real-time performance analysis.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from .event_bus import EventBus, EventMessage, EventType
from ..integration.redis_client import RedisClient
from ..error_handling.custom_errors import (
    PerformanceMonitoringError, ValidationError
)
from ..utils.logger import get_request_logger


@dataclass
class PerformanceMetric:
    """Individual performance metric with comprehensive tracking."""
    metric_name: str
    execution_id: str
    stage_name: Optional[str]
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any]
    threshold_exceeded: bool = False


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    unit: str
    enabled: bool = True


@dataclass
class PerformanceSummary:
    """Performance summary for analysis."""
    execution_id: str
    stage_name: Optional[str]
    average_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    throughput_per_second: float
    error_rate: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    measurement_period_seconds: float


class PerformanceMonitor:
    """Comprehensive performance monitor for pipeline operations."""
    
    def __init__(self, redis_client: RedisClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance monitor with Redis persistence.
        
        Args:
            redis_client: Redis client for metrics persistence
            config: Optional configuration dictionary
        """
        self.redis_client = redis_client
        self.config = config or {}
        self.logger = get_request_logger()
        
        # Configuration
        self.metrics_retention_hours = self.config.get('metrics_retention_hours', 24)
        self.threshold_check_interval = self.config.get('threshold_check_interval', 60)  # seconds
        self.performance_window_size = self.config.get('performance_window_size', 1000)
        self.enable_real_time_alerts = self.config.get('enable_real_time_alerts', True)
        
        # Performance thresholds for sub-50ms requirements
        self.performance_thresholds = self._initialize_performance_thresholds()
        
        # In-memory metrics storage
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.performance_window_size))
        self.execution_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # Real-time performance tracking
        self.current_performance: Dict[str, Dict[str, Any]] = {}
        
        # Performance statistics
        self.performance_stats = {
            'total_metrics_recorded': 0,
            'threshold_violations': 0,
            'critical_violations': 0,
            'average_detection_time': 0.0,
            'sub_50ms_compliance_rate': 0.0
        }
        
        # Redis key prefixes
        self.metrics_prefix = "pipeline:performance:"
        self.thresholds_prefix = "pipeline:thresholds:"
        self.summary_prefix = "pipeline:summary:"
        
        self.logger.info("PerformanceMonitor initialized with sub-50ms tracking")
    
    def _initialize_performance_thresholds(self) -> Dict[str, PerformanceThreshold]:
        """Initialize performance thresholds for critical operations."""
        return {
            'stage_execution_time': PerformanceThreshold(
                metric_name='stage_execution_time',
                warning_threshold=50.0,  # 50ms warning
                critical_threshold=100.0,  # 100ms critical
                unit='milliseconds'
            ),
            'bias_detection_time': PerformanceThreshold(
                metric_name='bias_detection_time',
                warning_threshold=50.0,  # 50ms warning
                critical_threshold=100.0,  # 100ms critical
                unit='milliseconds'
            ),
            'event_processing_time': PerformanceThreshold(
                metric_name='event_processing_time',
                warning_threshold=10.0,  # 10ms warning
                critical_threshold=25.0,  # 25ms critical
                unit='milliseconds'
            ),
            'redis_operation_time': PerformanceThreshold(
                metric_name='redis_operation_time',
                warning_threshold=5.0,  # 5ms warning
                critical_threshold=15.0,  # 15ms critical
                unit='milliseconds'
            ),
            'state_update_time': PerformanceThreshold(
                metric_name='state_update_time',
                warning_threshold=20.0,  # 20ms warning
                critical_threshold=50.0,  # 50ms critical
                unit='milliseconds'
            ),
            'progress_update_time': PerformanceThreshold(
                metric_name='progress_update_time',
                warning_threshold=5.0,  # 5ms warning
                critical_threshold=15.0,  # 15ms critical
                unit='milliseconds'
            )
        }
    
    async def record_metric(self, metric_name: str, execution_id: str,
                          value: float, unit: str = 'ms',
                          stage_name: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetric:
        """
        Record a performance metric with comprehensive tracking.
        
        Args:
            metric_name: Name of the metric
            execution_id: Execution ID
            value: Metric value
            unit: Unit of measurement
            stage_name: Optional stage name
            metadata: Optional metadata
            
        Returns:
            Recorded performance metric
            
        Raises:
            PerformanceMonitoringError: If metric recording fails
        """
        try:
            # Validate inputs
            if not metric_name or not isinstance(metric_name, str):
                raise ValidationError("Metric name must be a non-empty string")
            
            if not execution_id or not isinstance(execution_id, str):
                raise ValidationError("Execution ID must be a non-empty string")
            
            if value < 0:
                raise ValidationError("Metric value must be non-negative")
            
            # Create performance metric
            metric = PerformanceMetric(
                metric_name=metric_name,
                execution_id=execution_id,
                stage_name=stage_name,
                value=value,
                unit=unit,
                timestamp=datetime.utcnow(),
                metadata=metadata or {},
                threshold_exceeded=False
            )
            
            # Check against thresholds
            if metric_name in self.performance_thresholds:
                threshold = self.performance_thresholds[metric_name]
                if threshold.enabled:
                    metric.threshold_exceeded = self._check_threshold(value, threshold)
            
            # Store in memory buffer
            buffer_key = f"{metric_name}:{execution_id}"
            if stage_name:
                buffer_key += f":{stage_name}"
            
            self.metrics_buffer[buffer_key].append(value)
            
            # Store in execution metrics
            exec_key = f"{execution_id}:{metric_name}"
            if stage_name:
                exec_key += f":{stage_name}"
            
            self.execution_metrics[execution_id][exec_key].append(value)
            
            # Update real-time performance tracking
            self._update_real_time_performance(metric)
            
            # Store in Redis for persistence
            await self._store_metric_in_redis(metric)
            
            # Update performance statistics
            self._update_performance_statistics(metric)
            
            # Log sub-50ms compliance
            if unit == 'milliseconds' and value <= 50.0:
                self.logger.debug(
                    f"Sub-50ms compliance achieved: {metric_name} = {value:.2f}ms "
                    f"for execution {execution_id}"
                )
            elif unit == 'milliseconds' and value > 50.0:
                self.logger.warning(
                    f"Sub-50ms compliance exceeded: {metric_name} = {value:.2f}ms "
                    f"for execution {execution_id}"
                )
            
            self.performance_stats['total_metrics_recorded'] += 1
            
            return metric
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to record performance metric: {e}")
            raise PerformanceMonitoringError(f"Metric recording failed: {str(e)}")
    
    def _check_threshold(self, value: float, threshold: PerformanceThreshold) -> bool:
        """Check if metric value exceeds threshold."""
        return value > threshold.warning_threshold
    
    def _update_real_time_performance(self, metric: PerformanceMetric) -> None:
        """Update real-time performance tracking."""
        key = f"{metric.execution_id}:{metric.metric_name}"
        if metric.stage_name:
            key += f":{metric.stage_name}"
        
        if key not in self.current_performance:
            self.current_performance[key] = {
                'last_value': 0.0,
                'average': 0.0,
                'min_value': float('inf'),
                'max_value': 0.0,
                'count': 0,
                'threshold_exceeded': False
            }
        
        perf_data = self.current_performance[key]
        perf_data['last_value'] = metric.value
        perf_data['min_value'] = min(perf_data['min_value'], metric.value)
        perf_data['max_value'] = max(perf_data['max_value'], metric.value)
        perf_data['count'] += 1
        perf_data['threshold_exceeded'] = metric.threshold_exceeded
        
        # Calculate running average
        if perf_data['count'] == 1:
            perf_data['average'] = metric.value
        else:
            perf_data['average'] = (
                (perf_data['average'] * (perf_data['count'] - 1) + metric.value) / 
                perf_data['count']
            )
    
    async def _store_metric_in_redis(self, metric: PerformanceMetric) -> None:
        """Store metric in Redis for persistence."""
        try:
            key = f"{self.metrics_prefix}{metric.execution_id}:{metric.metric_name}"
            if metric.stage_name:
                key += f":{metric.stage_name}"
            
            metric_dict = asdict(metric)
            metric_dict['timestamp'] = metric.timestamp.isoformat()
            
            # Store with TTL
            await self.redis_client.setex(
                key,
                self.metrics_retention_hours * 3600,  # Convert to seconds
                json.dumps(metric_dict, default=str)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store metric in Redis: {e}")
            # Don't raise - Redis storage failure shouldn't break monitoring
    
    def _update_performance_statistics(self, metric: PerformanceMetric) -> None:
        """Update overall performance statistics."""
        try:
            # Update sub-50ms compliance rate
            if metric.unit == 'milliseconds':
                if metric.value <= 50.0:
                    # Count as compliant
                    total_compliant = int(self.performance_stats['sub_50ms_compliance_rate'] * 
                                        (self.performance_stats['total_metrics_recorded'] - 1)) + 1
                else:
                    # Count as non-compliant
                    total_compliant = int(self.performance_stats['sub_50ms_compliance_rate'] * 
                                        (self.performance_stats['total_metrics_recorded'] - 1))
                
                self.performance_stats['sub_50ms_compliance_rate'] = (
                    total_compliant / self.performance_stats['total_metrics_recorded']
                )
            
            # Update threshold violations
            if metric.threshold_exceeded:
                self.performance_stats['threshold_violations'] += 1
                
                # Check if it's a critical violation
                if (metric.metric_name in self.performance_thresholds and 
                    metric.value > self.performance_thresholds[metric.metric_name].critical_threshold):
                    self.performance_stats['critical_violations'] += 1
            
            # Update average detection time
            if self.performance_stats['average_detection_time'] == 0.0:
                self.performance_stats['average_detection_time'] = metric.value
            else:
                self.performance_stats['average_detection_time'] = (
                    (self.performance_stats['average_detection_time'] * 
                     (self.performance_stats['total_metrics_recorded'] - 1) + metric.value) / 
                    self.performance_stats['total_metrics_recorded']
                )
                
        except Exception as e:
            self.logger.error(f"Failed to update performance statistics: {e}")
    
    async def get_execution_performance_summary(self, execution_id: str,
                                              stage_name: Optional[str] = None) -> Optional[PerformanceSummary]:
        """
        Get performance summary for an execution.
        
        Args:
            execution_id: Execution ID
            stage_name: Optional stage name to filter by
            
        Returns:
            Performance summary or None if not found
            
        Raises:
            PerformanceMonitoringError: If summary generation fails
        """
        try:
            # Get execution metrics
            execution_metrics = self.execution_metrics.get(execution_id, {})
            
            if not execution_metrics:
                return None
            
            # Filter by stage if specified
            if stage_name:
                filtered_metrics = {
                    k: v for k, v in execution_metrics.items() 
                    if f":{stage_name}" in k
                }
                if not filtered_metrics:
                    return None
            else:
                filtered_metrics = execution_metrics
            
            # Calculate statistics
            all_values = []
            for metric_values in filtered_metrics.values():
                all_values.extend(metric_values)
            
            if not all_values:
                return None
            
            all_values.sort()
            
            # Calculate percentiles
            n = len(all_values)
            p50_index = int(0.5 * n)
            p95_index = int(0.95 * n)
            p99_index = int(0.99 * n)
            
            # Calculate throughput (assuming metrics are recorded over time)
            measurement_period = 300.0  # 5 minutes default
            if all_values:
                measurement_period = min(measurement_period, len(all_values) * 1.0)  # Rough estimate
            
            # Count successes and failures (simplified)
            successful_requests = len([v for v in all_values if v <= 50.0])  # Sub-50ms considered success
            failed_requests = len(all_values) - successful_requests
            
            return PerformanceSummary(
                execution_id=execution_id,
                stage_name=stage_name,
                average_response_time_ms=sum(all_values) / len(all_values),
                min_response_time_ms=min(all_values),
                max_response_time_ms=max(all_values),
                p50_response_time_ms=all_values[p50_index],
                p95_response_time_ms=all_values[p95_index],
                p99_response_time_ms=all_values[min(p99_index, n - 1)],
                throughput_per_second=len(all_values) / measurement_period,
                error_rate=failed_requests / len(all_values) if all_values else 0.0,
                total_requests=len(all_values),
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                measurement_period_seconds=measurement_period
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance summary: {e}")
            raise PerformanceMonitoringError(f"Summary generation failed: {str(e)}")
    
    async def check_performance_thresholds(self) -> List[Dict[str, Any]]:
        """
        Check all performance thresholds and return violations.
        
        Returns:
            List of threshold violations
            
        Raises:
            PerformanceMonitoringError: If threshold checking fails
        """
        try:
            violations = []
            
            for key, perf_data in self.current_performance.items():
                if perf_data['threshold_exceeded']:
                    parts = key.split(':')
                    execution_id = parts[0]
                    metric_name = parts[1]
                    stage_name = parts[2] if len(parts) > 2 else None
                    
                    threshold = self.performance_thresholds.get(metric_name)
                    if threshold:
                        severity = 'critical' if perf_data['last_value'] > threshold.critical_threshold else 'warning'
                        
                        violations.append({
                            'execution_id': execution_id,
                            'metric_name': metric_name,
                            'stage_name': stage_name,
                            'current_value': perf_data['last_value'],
                            'threshold_value': threshold.warning_threshold,
                            'severity': severity,
                            'timestamp': datetime.utcnow().isoformat(),
                            'recommendations': self._generate_threshold_recommendations(
                                metric_name, perf_data['last_value'], threshold
                            )
                        })
            
            # Update violation count
            self.performance_stats['threshold_violations'] = len(violations)
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Failed to check performance thresholds: {e}")
            raise PerformanceMonitoringError(f"Threshold checking failed: {str(e)}")
    
    def _generate_threshold_recommendations(self, metric_name: str, current_value: float,
                                          threshold: PerformanceThreshold) -> List[str]:
        """Generate recommendations for threshold violations."""
        recommendations = []
        
        if current_value > threshold.critical_threshold:
            recommendations.append(f"Critical performance issue: {metric_name} exceeded {threshold.critical_threshold}{threshold.unit}")
            recommendations.append("Immediate investigation required")
        elif current_value > threshold.warning_threshold:
            recommendations.append(f"Performance warning: {metric_name} exceeded {threshold.warning_threshold}{threshold.unit}")
            recommendations.append("Monitor closely and investigate if trend continues")
        
        # Add specific recommendations based on metric type
        if 'redis' in metric_name.lower():
            recommendations.extend([
                "Check Redis connection pool configuration",
                "Monitor Redis server performance",
                "Consider Redis cluster scaling if needed"
            ])
        elif 'bias' in metric_name.lower():
            recommendations.extend([
                "Optimize bias detection algorithm",
                "Check model loading performance",
                "Consider bias detection caching"
            ])
        elif 'stage' in metric_name.lower():
            recommendations.extend([
                "Review stage processing logic",
                "Check for resource bottlenecks",
                "Consider stage optimization"
            ])
        
        return recommendations[:3]  # Limit to top 3
    
    async def get_real_time_performance_dashboard(self) -> Dict[str, Any]:
        """
        Get real-time performance dashboard data.
        
        Returns:
            Performance dashboard data
            
        Raises:
            PerformanceMonitoringError: If dashboard generation fails
        """
        try:
            dashboard = {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_stats': self.performance_stats.copy(),
                'current_performance': {},
                'threshold_violations': await self.check_performance_thresholds(),
                'sub_50ms_compliance': {
                    'current_rate': self.performance_stats['sub_50ms_compliance_rate'],
                    'target_rate': 0.95,  # 95% target
                    'status': 'meeting_target' if self.performance_stats['sub_50ms_compliance_rate'] >= 0.95 else 'below_target'
                },
                'performance_trends': self._calculate_performance_trends(),
                'alerts': []
            }
            
            # Add current performance data
            for key, perf_data in self.current_performance.items():
                parts = key.split(':')
                execution_id = parts[0]
                metric_name = parts[1]
                stage_name = parts[2] if len(parts) > 2 else None
                
                if execution_id not in dashboard['current_performance']:
                    dashboard['current_performance'][execution_id] = {}
                
                dashboard['current_performance'][execution_id][metric_name] = {
                    'current_value': perf_data['last_value'],
                    'average': perf_data['average'],
                    'min_value': perf_data['min_value'] if perf_data['min_value'] != float('inf') else 0.0,
                    'max_value': perf_data['max_value'],
                    'count': perf_data['count'],
                    'threshold_exceeded': perf_data['threshold_exceeded']
                }
                
                if stage_name:
                    dashboard['current_performance'][execution_id][metric_name]['stage'] = stage_name
            
            # Generate alerts
            dashboard['alerts'] = self._generate_performance_alerts(dashboard)
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance dashboard: {e}")
            raise PerformanceMonitoringError(f"Dashboard generation failed: {str(e)}")
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        try:
            trends = {
                'response_time_trend': 'stable',
                'throughput_trend': 'stable',
                'error_rate_trend': 'stable',
                'compliance_trend': 'stable'
            }
            
            # Simple trend analysis based on recent metrics
            recent_metrics = []
            for buffer in self.metrics_buffer.values():
                if buffer:
                    recent_metrics.extend(list(buffer)[-10:])  # Last 10 values
            
            if len(recent_metrics) >= 5:
                # Calculate trend (simplified)
                first_half = recent_metrics[:len(recent_metrics)//2]
                second_half = recent_metrics[len(recent_metrics)//2:]
                
                if first_half and second_half:
                    first_avg = sum(first_half) / len(first_half)
                    second_avg = sum(second_half) / len(second_half)
                    
                    if second_avg < first_avg * 0.9:
                        trends['response_time_trend'] = 'improving'
                    elif second_avg > first_avg * 1.1:
                        trends['response_time_trend'] = 'degrading'
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance trends: {e}")
            return {
                'response_time_trend': 'unknown',
                'throughput_trend': 'unknown',
                'error_rate_trend': 'unknown',
                'compliance_trend': 'unknown'
            }
    
    def _generate_performance_alerts(self, dashboard: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance alerts based on dashboard data."""
        alerts = []
        
        try:
            # Check sub-50ms compliance
            compliance_rate = dashboard['sub_50ms_compliance']['current_rate']
            if compliance_rate < 0.90:  # Below 90%
                alerts.append({
                    'type': 'compliance_warning',
                    'severity': 'high' if compliance_rate < 0.80 else 'medium',
                    'message': f"Sub-50ms compliance rate is {compliance_rate:.1%}",
                    'recommendation': "Investigate performance bottlenecks"
                })
            
            # Check threshold violations
            violations = dashboard['threshold_violations']
            if violations:
                critical_violations = [v for v in violations if v['severity'] == 'critical']
                if critical_violations:
                    alerts.append({
                        'type': 'critical_performance_issue',
                        'severity': 'critical',
                        'message': f"{len(critical_violations)} critical performance violations detected",
                        'recommendation': "Immediate investigation required"
                    })
                else:
                    alerts.append({
                        'type': 'performance_warning',
                        'severity': 'medium',
                        'message': f"{len(violations)} performance threshold violations detected",
                        'recommendation': "Monitor and investigate trends"
                    })
            
            # Check error rate
            if dashboard['overall_stats']['total_metrics_recorded'] > 0:
                error_rate = dashboard['overall_stats']['threshold_violations'] / dashboard['overall_stats']['total_metrics_recorded']
                if error_rate > 0.1:  # More than 10% violations
                    alerts.append({
                        'type': 'high_error_rate',
                        'severity': 'high',
                        'message': f"High error rate detected: {error_rate:.1%}",
                        'recommendation': "Review system configuration and resource allocation"
                    })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance alerts: {e}")
            return []
    
    async def cleanup_old_metrics(self, max_age_hours: int = 24) -> int:
        """
        Clean up old performance metrics.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of cleaned up metrics
            
        Raises:
            PerformanceMonitoringError: If cleanup fails
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            # Clean up Redis metrics
            pattern = f"{self.metrics_prefix}*"
            metric_keys = await self.redis_client.keys(pattern)
            
            for key in metric_keys:
                try:
                    metric_data = await self.redis_client.get(key)
                    if metric_data:
                        metric_dict = json.loads(metric_data)
                        metric_time = datetime.fromisoformat(metric_dict['timestamp'])
                        
                        if metric_time < cutoff_time:
                            await self.redis_client.delete(key)
                            cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to process metric key {key}: {e}")
            
            # Clean up summary data
            summary_pattern = f"{self.summary_prefix}*"
            summary_keys = await self.redis_client.keys(summary_pattern)
            
            for key in summary_keys:
                try:
                    await self.redis_client.delete(key)
                    cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete summary key {key}: {e}")
            
            self.logger.info(f"Cleaned up {cleaned_count} old performance metrics")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old metrics: {e}")
            raise PerformanceMonitoringError(f"Metrics cleanup failed: {str(e)}")
    
    def get_performance_report(self, execution_id: Optional[str] = None,
                             time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            execution_id: Optional execution ID to filter by
            time_range_hours: Time range for the report in hours
            
        Returns:
            Performance report data
            
        Raises:
            PerformanceMonitoringError: If report generation fails
        """
        try:
            report = {
                'report_timestamp': datetime.utcnow().isoformat(),
                'time_range_hours': time_range_hours,
                'execution_filter': execution_id,
                'summary': {},
                'detailed_metrics': {},
                'recommendations': []
            }
            
            # Generate summary statistics
            report['summary'] = {
                'total_metrics_recorded': self.performance_stats['total_metrics_recorded'],
                'sub_50ms_compliance_rate': self.performance_stats['sub_50ms_compliance_rate'],
                'threshold_violations': self.performance_stats['threshold_violations'],
                'critical_violations': self.performance_stats['critical_violations'],
                'average_response_time': self.performance_stats['average_detection_time']
            }
            
            # Add detailed metrics if execution ID is specified
            if execution_id and execution_id in self.execution_metrics:
                report['detailed_metrics'] = {
                    metric_key: {
                        'count': len(values),
                        'average': sum(values) / len(values) if values else 0.0,
                        'min': min(values) if values else 0.0,
                        'max': max(values) if values else 0.0,
                        'p95': self._calculate_percentile(values, 0.95) if values else 0.0
                    }
                    for metric_key, values in self.execution_metrics[execution_id].items()
                }
            
            # Generate recommendations
            report['recommendations'] = self._generate_performance_recommendations(report['summary'])
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            raise PerformanceMonitoringError(f"Report generation failed: {str(e)}")
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(percentile * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        
        return sorted_values[index]
    
    def _generate_performance_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on summary."""
        recommendations = []
        
        # Check sub-50ms compliance
        compliance_rate = summary.get('sub_50ms_compliance_rate', 0.0)
        if compliance_rate < 0.95:
            recommendations.append(
                f"Sub-50ms compliance rate is {compliance_rate:.1%} (target: 95%). "
                "Consider optimizing critical path operations."
            )
        
        # Check threshold violations
        violations = summary.get('threshold_violations', 0)
        if violations > 0:
            recommendations.append(
                f"{violations} threshold violations detected. "
                "Review system configuration and resource allocation."
            )
        
        # Check critical violations
        critical_violations = summary.get('critical_violations', 0)
        if critical_violations > 0:
            recommendations.append(
                f"{critical_violations} critical performance violations detected. "
                "Immediate investigation required."
            )
        
        # Check average response time
        avg_time = summary.get('average_response_time', 0.0)
        if avg_time > 50.0:
            recommendations.append(
                f"Average response time is {avg_time:.1f}ms (target: <50ms). "
                "Consider performance optimization strategies."
            )
        
        if not recommendations:
            recommendations.append("Performance metrics are within acceptable ranges.")
        
        return recommendations[:3]  # Limit to top 3
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of performance monitor.
        
        Returns:
            Health check results
        """
        try:
            # Check Redis connection
            redis_health = self.redis_client.ping() if hasattr(self.redis_client, 'ping') else True
            
            # Check configuration
            config_healthy = (
                self.metrics_retention_hours > 0 and
                self.threshold_check_interval > 0 and
                self.performance_window_size > 0
            )
            
            # Check if we have any metrics recorded
            has_metrics = self.performance_stats['total_metrics_recorded'] > 0
            
            status = 'healthy' if all([redis_health, config_healthy]) else 'degraded'
            
            return {
                'status': status,
                'redis_connection': redis_health,
                'configuration_healthy': config_healthy,
                'has_recorded_metrics': has_metrics,
                'total_metrics_recorded': self.performance_stats['total_metrics_recorded'],
                'threshold_violations': self.performance_stats['threshold_violations'],
                'sub_50ms_compliance_rate': self.performance_stats['sub_50ms_compliance_rate'],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"PerformanceMonitor health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }