#!/usr/bin/env python3
"""
Enterprise Monitoring and Health Check System

Provides comprehensive system monitoring with:
- Health checks for all components
- Performance metrics collection
- Resource usage monitoring
- Alerting and notifications
- Status dashboard
"""

import psutil
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import logging

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    load_average: List[float]

class EnterpriseMonitor:
    """Enterprise monitoring system."""
    
    def __init__(self, check_interval: int = 60, metrics_retention_hours: int = 24):
        self.check_interval = check_interval
        self.metrics_retention = timedelta(hours=metrics_retention_hours)
        
        # Health check registry
        self.health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=metrics_retention_hours * 60)  # 1 per minute
        self.health_history: deque = deque(maxlen=1000)  # Last 1000 health checks
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Alerting
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'error_rate': 0.05  # 5% error rate
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Register default health checks
        self._register_default_health_checks()
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def _register_default_health_checks(self):
        """Register default system health checks."""
        
        def system_resources_check() -> HealthCheckResult:
            """Check system resource usage."""
            start_time = time.time()
            
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Determine status
                if cpu_percent > 90 or memory.percent > 90 or disk.percent > 95:
                    status = 'unhealthy'
                    message = 'Critical resource usage detected'
                elif cpu_percent > 70 or memory.percent > 80 or disk.percent > 85:
                    status = 'degraded'
                    message = 'High resource usage detected'
                else:
                    status = 'healthy'
                    message = 'System resources within normal limits'
                
                metrics = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent
                }
                
                duration_ms = (time.time() - start_time) * 1000
                
                return HealthCheckResult(
                    component='system_resources',
                    status=status,
                    message=message,
                    timestamp=datetime.utcnow(),
                    metrics=metrics,
                    duration_ms=duration_ms
                )
            
            except Exception as e:
                return HealthCheckResult(
                    component='system_resources',
                    status='unhealthy',
                    message=f'Health check failed: {e}',
                    timestamp=datetime.utcnow(),
                    duration_ms=(time.time() - start_time) * 1000
                )
        
        def data_pipeline_check() -> HealthCheckResult:
            """Check data pipeline health."""
            start_time = time.time()
            
            try:
                # Check if key directories exist
                data_dir = Path("/home/vivi/pixelated/ai/data")
                logs_dir = Path("/home/vivi/pixelated/ai/logs")
                
                if not data_dir.exists():
                    return HealthCheckResult(
                        component='data_pipeline',
                        status='unhealthy',
                        message='Data directory not found',
                        timestamp=datetime.utcnow(),
                        duration_ms=(time.time() - start_time) * 1000
                    )
                
                # Check recent processing activity
                processed_files = list(data_dir.glob("processed/**/*.jsonl"))
                recent_files = [
                    f for f in processed_files 
                    if datetime.fromtimestamp(f.stat().st_mtime) > datetime.now() - timedelta(hours=24)
                ]
                
                if len(recent_files) > 0:
                    status = 'healthy'
                    message = f'Data pipeline active, {len(recent_files)} recent files'
                else:
                    status = 'degraded'
                    message = 'No recent processing activity detected'
                
                metrics = {
                    'total_processed_files': len(processed_files),
                    'recent_files_24h': len(recent_files),
                    'data_directory_size_gb': sum(f.stat().st_size for f in processed_files) / (1024**3)
                }
                
                return HealthCheckResult(
                    component='data_pipeline',
                    status=status,
                    message=message,
                    timestamp=datetime.utcnow(),
                    metrics=metrics,
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            except Exception as e:
                return HealthCheckResult(
                    component='data_pipeline',
                    status='unhealthy',
                    message=f'Pipeline check failed: {e}',
                    timestamp=datetime.utcnow(),
                    duration_ms=(time.time() - start_time) * 1000
                )
        
        def quality_validation_check() -> HealthCheckResult:
            """Check quality validation system."""
            start_time = time.time()
            
            try:
                # Try to import and initialize quality validator
                from dataset_pipeline.real_quality_validator import RealQualityValidator
                
                validator = RealQualityValidator()
                
                # Test with a simple conversation
                test_conversation = {
                    'conversations': [
                        {'human': 'Hello, how are you?'},
                        {'assistant': 'I am doing well, thank you for asking.'}
                    ]
                }
                
                quality_result = validator.validate_conversation_quality(test_conversation)
                
                if quality_result.overall_quality > 0:
                    status = 'healthy'
                    message = 'Quality validation system operational'
                else:
                    status = 'degraded'
                    message = 'Quality validation returning low scores'
                
                metrics = {
                    'test_quality_score': quality_result.overall_quality,
                    'therapeutic_accuracy': quality_result.therapeutic_accuracy,
                    'safety_score': quality_result.safety_score
                }
                
                return HealthCheckResult(
                    component='quality_validation',
                    status=status,
                    message=message,
                    timestamp=datetime.utcnow(),
                    metrics=metrics,
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            except Exception as e:
                return HealthCheckResult(
                    component='quality_validation',
                    status='unhealthy',
                    message=f'Quality validation failed: {e}',
                    timestamp=datetime.utcnow(),
                    duration_ms=(time.time() - start_time) * 1000
                )
        
        # Register the default checks
        self.register_health_check('system_resources', system_resources_check)
        self.register_health_check('data_pipeline', data_pipeline_check)
        self.register_health_check('quality_validation', quality_validation_check)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]  # Windows doesn't have load average
            
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024**3),
                disk_free_gb=disk.free / (1024**3),
                network_sent_mb=network.bytes_sent / (1024**2),
                network_recv_mb=network.bytes_recv / (1024**2),
                process_count=process_count,
                load_average=load_avg
            )
        
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return None
    
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = result
                self.health_history.append(result)
                
                # Log health check results
                if result.status == 'unhealthy':
                    self.logger.error(f"Health check FAILED: {name} - {result.message}")
                elif result.status == 'degraded':
                    self.logger.warning(f"Health check DEGRADED: {name} - {result.message}")
                else:
                    self.logger.debug(f"Health check OK: {name} - {result.message}")
            
            except Exception as e:
                self.logger.error(f"Health check error for {name}: {e}")
                results[name] = HealthCheckResult(
                    component=name,
                    status='unhealthy',
                    message=f'Check execution failed: {e}',
                    timestamp=datetime.utcnow()
                )
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        health_results = self.run_health_checks()
        metrics = self.collect_system_metrics()
        
        # Determine overall status
        statuses = [result.status for result in health_results.values()]
        if 'unhealthy' in statuses:
            overall_status = 'unhealthy'
        elif 'degraded' in statuses:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'health_checks': {name: {
                'status': result.status,
                'message': result.message,
                'metrics': result.metrics,
                'duration_ms': result.duration_ms
            } for name, result in health_results.items()},
            'system_metrics': {
                'cpu_percent': metrics.cpu_percent if metrics else 0,
                'memory_percent': metrics.memory_percent if metrics else 0,
                'disk_percent': metrics.disk_percent if metrics else 0,
                'process_count': metrics.process_count if metrics else 0
            } if metrics else {},
            'uptime_seconds': time.time() - psutil.boot_time()
        }
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"Started monitoring with {self.check_interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Stopped monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self.collect_system_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                
                # Run health checks
                self.run_health_checks()
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Clean up old data
                self._cleanup_old_data()
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
            
            time.sleep(self.check_interval)
    
    def _check_alerts(self, metrics: Optional[SystemMetrics]):
        """Check for alert conditions."""
        if not metrics:
            return
        
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_percent > self.alert_thresholds['disk_percent']:
            alerts.append(f"High disk usage: {metrics.disk_percent:.1f}%")
        
        for alert in alerts:
            self.logger.warning(f"ALERT: {alert}")
    
    def _cleanup_old_data(self):
        """Clean up old metrics and health check data."""
        cutoff_time = datetime.utcnow() - self.metrics_retention
        
        # Clean metrics history
        while (self.metrics_history and 
               self.metrics_history[0].timestamp < cutoff_time):
            self.metrics_history.popleft()
        
        # Clean health history
        while (self.health_history and 
               self.health_history[0].timestamp < cutoff_time):
            self.health_history.popleft()
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {'error': 'No metrics available for the specified period'}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_disk = sum(m.disk_percent for m in recent_metrics) / len(recent_metrics)
        
        return {
            'period_hours': hours,
            'sample_count': len(recent_metrics),
            'averages': {
                'cpu_percent': round(avg_cpu, 2),
                'memory_percent': round(avg_memory, 2),
                'disk_percent': round(avg_disk, 2)
            },
            'current': {
                'cpu_percent': recent_metrics[-1].cpu_percent,
                'memory_percent': recent_metrics[-1].memory_percent,
                'disk_percent': recent_metrics[-1].disk_percent
            } if recent_metrics else {}
        }

# Global monitor instance
_enterprise_monitor = None

def get_monitor() -> EnterpriseMonitor:
    """Get global enterprise monitor."""
    global _enterprise_monitor
    if _enterprise_monitor is None:
        _enterprise_monitor = EnterpriseMonitor()
    return _enterprise_monitor

def start_monitoring():
    """Start enterprise monitoring."""
    monitor = get_monitor()
    monitor.start_monitoring()
    return monitor

if __name__ == "__main__":
    # Test the monitoring system
    monitor = EnterpriseMonitor(check_interval=5)  # 5 second interval for testing
    
    print("🔍 Running health checks...")
    status = monitor.get_system_status()
    print(f"Overall status: {status['overall_status']}")
    
    for name, check in status['health_checks'].items():
        print(f"  {name}: {check['status']} - {check['message']}")
    
    print(f"\n📊 System metrics:")
    metrics = status['system_metrics']
    print(f"  CPU: {metrics.get('cpu_percent', 0):.1f}%")
    print(f"  Memory: {metrics.get('memory_percent', 0):.1f}%")
    print(f"  Disk: {metrics.get('disk_percent', 0):.1f}%")
    
    print("\n✅ Enterprise monitoring system ready")
