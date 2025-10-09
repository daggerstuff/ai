#!/usr/bin/env python3
"""
Monitoring-Notification Bridge for Pixelated Empathy AI
Integrates the notification system with existing monitoring infrastructure
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from notification_integrations import (
    NotificationManager, 
    NotificationPriority, 
    NotificationChannel,
    NotificationMessage
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringBridge:
    """Bridge between monitoring systems and notification channels"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.notification_manager = NotificationManager(config_path)
        self.monitoring_db_path = "/home/vivi/pixelated/ai/monitoring/monitoring.db"
        self.alert_history_db_path = "/home/vivi/pixelated/ai/monitoring/alert_history.db"
        self.last_check_time = datetime.utcnow()
        self.alert_cooldowns = {}  # Prevent alert spam
        self.setup_databases()
    
    def setup_databases(self):
        """Initialize monitoring and alert history databases"""
        # Create monitoring database if it doesn't exist
        os.makedirs(os.path.dirname(self.monitoring_db_path), exist_ok=True)
        
        with sqlite3.connect(self.monitoring_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    service_name TEXT,
                    status TEXT DEFAULT 'normal'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT UNIQUE NOT NULL,
                    metric_name TEXT NOT NULL,
                    threshold_value REAL NOT NULL,
                    comparison_operator TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    cooldown_minutes INTEGER DEFAULT 15,
                    enabled BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        
        # Create alert history database
        with sqlite3.connect(self.alert_history_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    rule_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    channels TEXT NOT NULL,
                    triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved_at DATETIME,
                    status TEXT DEFAULT 'active',
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notification_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    sent_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    error_message TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alert_history (alert_id)
                )
            """)
    
    def add_alert_rule(self, rule_name: str, metric_name: str, threshold: float, 
                      operator: str, priority: str, cooldown_minutes: int = 15):
        """Add a new alert rule"""
        with sqlite3.connect(self.monitoring_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alert_rules 
                (rule_name, metric_name, threshold_value, comparison_operator, priority, cooldown_minutes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (rule_name, metric_name, threshold, operator, priority, cooldown_minutes))
        
        logger.info(f"Added alert rule: {rule_name}")
    
    def setup_default_alert_rules(self):
        """Setup default monitoring alert rules"""
        default_rules = [
            # CPU Usage Rules
            ("high_cpu_usage", "cpu_usage_percent", 85.0, ">", "medium", 10),
            ("critical_cpu_usage", "cpu_usage_percent", 95.0, ">", "high", 5),
            
            # Memory Usage Rules
            ("high_memory_usage", "memory_usage_percent", 80.0, ">", "medium", 10),
            ("critical_memory_usage", "memory_usage_percent", 90.0, ">", "high", 5),
            
            # Disk Usage Rules
            ("high_disk_usage", "disk_usage_percent", 85.0, ">", "medium", 30),
            ("critical_disk_usage", "disk_usage_percent", 95.0, ">", "high", 15),
            
            # Processing Queue Rules
            ("large_processing_queue", "queue_size", 1000, ">", "medium", 15),
            ("critical_processing_queue", "queue_size", 5000, ">", "high", 10),
            
            # Error Rate Rules
            ("high_error_rate", "error_rate_percent", 5.0, ">", "medium", 10),
            ("critical_error_rate", "error_rate_percent", 15.0, ">", "high", 5),
            
            # Response Time Rules
            ("slow_response_time", "avg_response_time_ms", 2000, ">", "medium", 15),
            ("critical_response_time", "avg_response_time_ms", 5000, ">", "high", 10),
            
            # Database Connection Rules
            ("low_db_connections", "available_db_connections", 10, "<", "medium", 20),
            ("critical_db_connections", "available_db_connections", 5, "<", "high", 10),
            
            # Worker Process Rules
            ("few_active_workers", "active_worker_count", 2, "<", "medium", 15),
            ("no_active_workers", "active_worker_count", 1, "<", "critical", 5)
        ]
        
        for rule in default_rules:
            self.add_alert_rule(*rule)
        
        logger.info(f"Setup {len(default_rules)} default alert rules")
    
    def record_metric(self, metric_name: str, value: float, unit: str = None, 
                     service_name: str = None, status: str = "normal"):
        """Record a system metric"""
        with sqlite3.connect(self.monitoring_db_path) as conn:
            conn.execute("""
                INSERT INTO system_metrics 
                (metric_name, metric_value, metric_unit, service_name, status)
                VALUES (?, ?, ?, ?, ?)
            """, (metric_name, value, unit, service_name, status))
    
    async def check_alert_conditions(self):
        """Check all alert conditions and trigger notifications if needed"""
        with sqlite3.connect(self.monitoring_db_path) as conn:
            # Get all enabled alert rules
            rules = conn.execute("""
                SELECT rule_name, metric_name, threshold_value, comparison_operator, 
                       priority, cooldown_minutes
                FROM alert_rules 
                WHERE enabled = 1
            """).fetchall()
            
            for rule in rules:
                rule_name, metric_name, threshold, operator, priority, cooldown = rule
                
                # Check if rule is in cooldown
                if self._is_in_cooldown(rule_name, cooldown):
                    continue
                
                # Get latest metric value
                latest_metric = conn.execute("""
                    SELECT metric_value, timestamp, service_name
                    FROM system_metrics 
                    WHERE metric_name = ?
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (metric_name,)).fetchone()
                
                if not latest_metric:
                    continue
                
                value, timestamp, service_name = latest_metric
                
                # Check if alert condition is met
                if self._evaluate_condition(value, threshold, operator):
                    await self._trigger_alert(
                        rule_name, metric_name, value, threshold, operator, 
                        priority, service_name, timestamp
                    )
    
    def _is_in_cooldown(self, rule_name: str, cooldown_minutes: int) -> bool:
        """Check if alert rule is in cooldown period"""
        if rule_name not in self.alert_cooldowns:
            return False
        
        last_alert_time = self.alert_cooldowns[rule_name]
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.utcnow() - last_alert_time < cooldown_period
    
    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate alert condition"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        else:
            logger.error(f"Unknown operator: {operator}")
            return False
    
    async def _trigger_alert(self, rule_name: str, metric_name: str, value: float,
                           threshold: float, operator: str, priority: str,
                           service_name: str, timestamp: str):
        """Trigger an alert notification"""
        
        # Generate alert ID
        alert_id = f"{rule_name}_{int(time.time())}"
        
        # Create alert message
        title = f"Alert: {rule_name.replace('_', ' ').title()}"
        
        if service_name:
            message = f"Service '{service_name}' metric '{metric_name}' is {value} (threshold: {operator} {threshold})"
        else:
            message = f"System metric '{metric_name}' is {value} (threshold: {operator} {threshold})"
        
        # Convert priority string to enum
        try:
            priority_enum = NotificationPriority(priority.lower())
        except ValueError:
            priority_enum = NotificationPriority.MEDIUM
            logger.warning(f"Unknown priority '{priority}', using MEDIUM")
        
        # Create metadata
        metadata = {
            "rule_name": rule_name,
            "metric_name": metric_name,
            "current_value": value,
            "threshold_value": threshold,
            "operator": operator,
            "service_name": service_name,
            "metric_timestamp": timestamp,
            "alert_id": alert_id
        }
        
        # Send notification
        results = await self.notification_manager.send_alert(
            title=title,
            message=message,
            priority=priority_enum,
            metadata=metadata
        )
        
        # Record alert in history
        self._record_alert_history(alert_id, rule_name, title, message, priority, results, metadata)
        
        # Update cooldown
        self.alert_cooldowns[rule_name] = datetime.utcnow()
        
        logger.info(f"Alert triggered: {rule_name} (ID: {alert_id})")
    
    def _record_alert_history(self, alert_id: str, rule_name: str, title: str,
                            message: str, priority: str, results: Dict, metadata: Dict):
        """Record alert in history database"""
        
        channels_sent = list(results.keys())
        channels_str = ",".join([ch.value for ch in channels_sent])
        metadata_str = json.dumps(metadata)
        
        with sqlite3.connect(self.alert_history_db_path) as conn:
            # Insert alert record
            conn.execute("""
                INSERT INTO alert_history 
                (alert_id, rule_name, title, message, priority, channels, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (alert_id, rule_name, title, message, priority, channels_str, metadata_str))
            
            # Insert notification results
            for channel, success in results.items():
                conn.execute("""
                    INSERT INTO notification_results 
                    (alert_id, channel, success, error_message)
                    VALUES (?, ?, ?, ?)
                """, (alert_id, channel.value, success, None if success else "Failed to send"))
    
    async def process_external_alerts(self, alert_data: Dict[str, Any]):
        """Process alerts from external monitoring systems"""
        
        # Extract alert information
        title = alert_data.get('title', 'External Alert')
        message = alert_data.get('message', 'No message provided')
        priority = alert_data.get('priority', 'medium')
        source = alert_data.get('source', 'external')
        
        # Convert priority
        try:
            priority_enum = NotificationPriority(priority.lower())
        except ValueError:
            priority_enum = NotificationPriority.MEDIUM
        
        # Add source to metadata
        metadata = alert_data.get('metadata', {})
        metadata['source'] = source
        metadata['external_alert'] = True
        
        # Send notification
        results = await self.notification_manager.send_alert(
            title=f"[{source.upper()}] {title}",
            message=message,
            priority=priority_enum,
            metadata=metadata
        )
        
        logger.info(f"Processed external alert from {source}: {title}")
        return results
    
    async def start_monitoring_loop(self, check_interval: int = 60):
        """Start the main monitoring loop"""
        logger.info(f"Starting monitoring loop (check interval: {check_interval}s)")
        
        while True:
            try:
                await self.check_alert_conditions()
                await self.notification_manager.flush_groups()
                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(check_interval)
    
    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """Get recent alert history"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.alert_history_db_path) as conn:
            alerts = conn.execute("""
                SELECT alert_id, rule_name, title, message, priority, 
                       triggered_at, resolved_at, status, metadata
                FROM alert_history 
                WHERE triggered_at > ?
                ORDER BY triggered_at DESC
            """, (cutoff_time.isoformat(),)).fetchall()
            
            result = []
            for alert in alerts:
                alert_dict = {
                    'alert_id': alert[0],
                    'rule_name': alert[1],
                    'title': alert[2],
                    'message': alert[3],
                    'priority': alert[4],
                    'triggered_at': alert[5],
                    'resolved_at': alert[6],
                    'status': alert[7],
                    'metadata': json.loads(alert[8]) if alert[8] else {}
                }
                result.append(alert_dict)
            
            return result
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get current system health summary"""
        with sqlite3.connect(self.monitoring_db_path) as conn:
            # Get latest metrics for key indicators
            key_metrics = [
                'cpu_usage_percent', 'memory_usage_percent', 'disk_usage_percent',
                'queue_size', 'error_rate_percent', 'active_worker_count'
            ]
            
            health_data = {}
            for metric in key_metrics:
                latest = conn.execute("""
                    SELECT metric_value, timestamp 
                    FROM system_metrics 
                    WHERE metric_name = ?
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (metric,)).fetchone()
                
                if latest:
                    health_data[metric] = {
                        'value': latest[0],
                        'timestamp': latest[1]
                    }
            
            # Get alert counts
            recent_alerts = len(self.get_alert_history(hours=1))
            active_alerts = len([a for a in self.get_alert_history(hours=24) if a['status'] == 'active'])
            
            return {
                'metrics': health_data,
                'alerts': {
                    'recent_hour': recent_alerts,
                    'active_24h': active_alerts
                },
                'timestamp': datetime.utcnow().isoformat()
            }

# Example usage and testing
async def example_usage():
    """Example of how to use the monitoring bridge"""
    
    # Initialize bridge
    bridge = MonitoringBridge()
    
    # Setup default alert rules
    bridge.setup_default_alert_rules()
    
    # Record some example metrics
    bridge.record_metric("cpu_usage_percent", 45.2, "%", "api-server")
    bridge.record_metric("memory_usage_percent", 67.8, "%", "api-server")
    bridge.record_metric("queue_size", 234, "items", "processor")
    bridge.record_metric("error_rate_percent", 2.1, "%", "api-server")
    
    # Simulate high CPU usage to trigger alert
    bridge.record_metric("cpu_usage_percent", 87.5, "%", "api-server")
    
    # Check for alerts
    await bridge.check_alert_conditions()
    
    # Process external alert
    external_alert = {
        'title': 'Database Connection Pool Exhausted',
        'message': 'All database connections are in use. New requests are being queued.',
        'priority': 'high',
        'source': 'database-monitor',
        'metadata': {
            'pool_size': 20,
            'active_connections': 20,
            'queued_requests': 15
        }
    }
    
    await bridge.process_external_alerts(external_alert)
    
    # Get system health summary
    health = bridge.get_system_health_summary()
    print("System Health Summary:")
    print(json.dumps(health, indent=2))
    
    # Get recent alerts
    recent_alerts = bridge.get_alert_history(hours=1)
    print(f"\nRecent Alerts ({len(recent_alerts)}):")
    for alert in recent_alerts:
        print(f"  - {alert['title']} ({alert['priority']}) at {alert['triggered_at']}")

if __name__ == "__main__":
    asyncio.run(example_usage())
