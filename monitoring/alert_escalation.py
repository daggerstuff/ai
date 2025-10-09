#!/usr/bin/env python3
"""
Alert Escalation System for Pixelated Empathy AI
Implements intelligent alert escalation procedures based on severity levels
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import threading
import sqlite3
from collections import defaultdict
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"


class EscalationLevel(Enum):
    """Escalation levels"""
    LEVEL_1 = "level_1"  # Initial notification
    LEVEL_2 = "level_2"  # Team lead notification
    LEVEL_3 = "level_3"  # Manager notification
    LEVEL_4 = "level_4"  # Executive notification
    LEVEL_5 = "level_5"  # Emergency contacts


@dataclass
class Alert:
    """Alert information"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: str = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    escalation_level: EscalationLevel = EscalationLevel.LEVEL_1
    escalation_count: int = 0
    last_escalated_at: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['escalation_level'] = self.escalation_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create from dictionary"""
        data['severity'] = AlertSeverity(data['severity'])
        data['status'] = AlertStatus(data['status'])
        data['escalation_level'] = EscalationLevel(data['escalation_level'])
        return cls(**data)


@dataclass
class EscalationRule:
    """Escalation rule configuration"""
    severity: AlertSeverity
    level: EscalationLevel
    delay_minutes: int
    recipients: List[str]
    channels: List[str]  # email, slack, sms, webhook
    conditions: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}


@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    name: str
    type: str  # email, slack, sms, webhook
    config: Dict[str, Any]
    enabled: bool = True


class AlertEscalationManager:
    """Manages alert escalation procedures"""
    
    def __init__(self, db_path: str = None, config_path: str = None):
        self.db_path = db_path or "alert_escalation.db"
        self.config_path = config_path or "escalation_config.json"
        
        # Initialize database
        self._init_database()
        
        # Load configuration
        self.escalation_rules: Dict[AlertSeverity, List[EscalationRule]] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self._load_configuration()
        
        # Active alerts
        self.active_alerts: Dict[str, Alert] = {}
        self.escalation_timers: Dict[str, threading.Timer] = {}
        
        # Load active alerts from database
        self._load_active_alerts()
        
        # Start escalation monitoring
        self.monitoring_thread = threading.Thread(target=self._monitor_escalations, daemon=True)
        self.monitoring_thread.start()
    
    def _init_database(self):
        """Initialize SQLite database for alert tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    acknowledged_at TEXT,
                    resolved_at TEXT,
                    acknowledged_by TEXT,
                    resolved_by TEXT,
                    escalation_level TEXT NOT NULL,
                    escalation_count INTEGER DEFAULT 0,
                    last_escalated_at TEXT,
                    metadata TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS escalation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    escalation_level TEXT NOT NULL,
                    recipients TEXT NOT NULL,
                    channels TEXT NOT NULL,
                    escalated_at TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts (alert_id)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alert_status ON alerts(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alert_severity ON alerts(severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_escalation_history_alert ON escalation_history(alert_id)")
    
    def _load_configuration(self):
        """Load escalation configuration from file"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load escalation rules
                for severity_str, rules_data in config.get('escalation_rules', {}).items():
                    severity = AlertSeverity(severity_str)
                    rules = []
                    
                    for rule_data in rules_data:
                        rule = EscalationRule(
                            severity=severity,
                            level=EscalationLevel(rule_data['level']),
                            delay_minutes=rule_data['delay_minutes'],
                            recipients=rule_data['recipients'],
                            channels=rule_data['channels'],
                            conditions=rule_data.get('conditions', {})
                        )
                        rules.append(rule)
                    
                    self.escalation_rules[severity] = rules
                
                # Load notification channels
                for channel_data in config.get('notification_channels', []):
                    channel = NotificationChannel(
                        name=channel_data['name'],
                        type=channel_data['type'],
                        config=channel_data['config'],
                        enabled=channel_data.get('enabled', True)
                    )
                    self.notification_channels[channel.name] = channel
                
                logger.info(f"Loaded escalation configuration from {self.config_path}")
            else:
                # Create default configuration
                self._create_default_configuration()
                
        except Exception as e:
            logger.error(f"Failed to load escalation configuration: {e}")
            self._create_default_configuration()
    
    def _create_default_configuration(self):
        """Create default escalation configuration"""
        default_config = {
            "escalation_rules": {
                "critical": [
                    {
                        "level": "level_1",
                        "delay_minutes": 0,
                        "recipients": ["oncall@pixelated.ai"],
                        "channels": ["email", "slack"]
                    },
                    {
                        "level": "level_2",
                        "delay_minutes": 5,
                        "recipients": ["team-lead@pixelated.ai"],
                        "channels": ["email", "slack", "sms"]
                    },
                    {
                        "level": "level_3",
                        "delay_minutes": 15,
                        "recipients": ["manager@pixelated.ai"],
                        "channels": ["email", "sms"]
                    }
                ],
                "high": [
                    {
                        "level": "level_1",
                        "delay_minutes": 0,
                        "recipients": ["oncall@pixelated.ai"],
                        "channels": ["email", "slack"]
                    },
                    {
                        "level": "level_2",
                        "delay_minutes": 15,
                        "recipients": ["team-lead@pixelated.ai"],
                        "channels": ["email", "slack"]
                    }
                ],
                "medium": [
                    {
                        "level": "level_1",
                        "delay_minutes": 0,
                        "recipients": ["team@pixelated.ai"],
                        "channels": ["email", "slack"]
                    }
                ],
                "low": [
                    {
                        "level": "level_1",
                        "delay_minutes": 0,
                        "recipients": ["team@pixelated.ai"],
                        "channels": ["email"]
                    }
                ]
            },
            "notification_channels": [
                {
                    "name": "email",
                    "type": "email",
                    "config": {
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587,
                        "username": "",
                        "password": "",
                        "from_address": "alerts@pixelated.ai"
                    }
                },
                {
                    "name": "slack",
                    "type": "slack",
                    "config": {
                        "webhook_url": "",
                        "channel": "#alerts"
                    }
                }
            ]
        }
        
        # Save default configuration
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default escalation configuration: {self.config_path}")
        
        # Load the default configuration
        self._load_configuration()
    
    def create_alert(self, title: str, description: str, severity: AlertSeverity, 
                    source: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new alert and start escalation process"""
        alert_id = self._generate_alert_id()
        
        alert = Alert(
            alert_id=alert_id,
            title=title,
            description=description,
            severity=severity,
            source=source,
            metadata=metadata or {}
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self._store_alert_in_db(alert)
        
        # Start escalation process
        self._start_escalation(alert)
        
        logger.info(f"Created alert {alert_id}: {title} ({severity.value})")
        return alert_id
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.active_alerts:
            logger.warning(f"Alert {alert_id} not found")
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(timezone.utc).isoformat()
        alert.acknowledged_by = acknowledged_by
        
        # Cancel escalation timer
        self._cancel_escalation_timer(alert_id)
        
        # Update database
        self._store_alert_in_db(alert)
        
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert"""
        if alert_id not in self.active_alerts:
            logger.warning(f"Alert {alert_id} not found")
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc).isoformat()
        alert.resolved_by = resolved_by
        
        # Cancel escalation timer
        self._cancel_escalation_timer(alert_id)
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        # Update database
        self._store_alert_in_db(alert)
        
        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True
    
    def _start_escalation(self, alert: Alert):
        """Start the escalation process for an alert"""
        rules = self.escalation_rules.get(alert.severity, [])
        if not rules:
            logger.warning(f"No escalation rules found for severity {alert.severity.value}")
            return
        
        # Find the first rule (Level 1)
        level_1_rule = next((rule for rule in rules if rule.level == EscalationLevel.LEVEL_1), None)
        if level_1_rule:
            if level_1_rule.delay_minutes == 0:
                # Immediate notification
                self._escalate_alert(alert, level_1_rule)
            else:
                # Schedule notification
                self._schedule_escalation(alert, level_1_rule)
    
    def _escalate_alert(self, alert: Alert, rule: EscalationRule):
        """Escalate an alert according to the rule"""
        if alert.status in [AlertStatus.ACKNOWLEDGED, AlertStatus.RESOLVED]:
            logger.info(f"Alert {alert.alert_id} already handled, skipping escalation")
            return
        
        # Update alert escalation info
        alert.escalation_level = rule.level
        alert.escalation_count += 1
        alert.last_escalated_at = datetime.now(timezone.utc).isoformat()
        alert.status = AlertStatus.ESCALATED
        
        # Send notifications
        success = self._send_notifications(alert, rule)
        
        # Log escalation
        self._log_escalation(alert, rule, success)
        
        # Update database
        self._store_alert_in_db(alert)
        
        # Schedule next escalation if available
        self._schedule_next_escalation(alert)
        
        logger.info(f"Escalated alert {alert.alert_id} to {rule.level.value}")
    
    def _schedule_escalation(self, alert: Alert, rule: EscalationRule):
        """Schedule an escalation"""
        delay_seconds = rule.delay_minutes * 60
        
        timer = threading.Timer(delay_seconds, self._escalate_alert, args=[alert, rule])
        timer.start()
        
        self.escalation_timers[alert.alert_id] = timer
        
        logger.info(f"Scheduled escalation for alert {alert.alert_id} in {rule.delay_minutes} minutes")
    
    def _schedule_next_escalation(self, alert: Alert):
        """Schedule the next escalation level"""
        rules = self.escalation_rules.get(alert.severity, [])
        
        # Find next escalation level
        current_level_value = list(EscalationLevel).index(alert.escalation_level)
        next_levels = [level for level in EscalationLevel if list(EscalationLevel).index(level) > current_level_value]
        
        if not next_levels:
            logger.info(f"No more escalation levels for alert {alert.alert_id}")
            return
        
        next_level = next_levels[0]
        next_rule = next((rule for rule in rules if rule.level == next_level), None)
        
        if next_rule:
            self._schedule_escalation(alert, next_rule)
    
    def _cancel_escalation_timer(self, alert_id: str):
        """Cancel escalation timer for an alert"""
        if alert_id in self.escalation_timers:
            timer = self.escalation_timers[alert_id]
            timer.cancel()
            del self.escalation_timers[alert_id]
            logger.debug(f"Cancelled escalation timer for alert {alert_id}")
    
    def _send_notifications(self, alert: Alert, rule: EscalationRule) -> bool:
        """Send notifications through configured channels"""
        success = True
        
        for channel_name in rule.channels:
            if channel_name not in self.notification_channels:
                logger.warning(f"Notification channel {channel_name} not configured")
                continue
            
            channel = self.notification_channels[channel_name]
            if not channel.enabled:
                logger.debug(f"Notification channel {channel_name} is disabled")
                continue
            
            try:
                if channel.type == "email":
                    self._send_email_notification(alert, rule, channel)
                elif channel.type == "slack":
                    self._send_slack_notification(alert, rule, channel)
                elif channel.type == "webhook":
                    self._send_webhook_notification(alert, rule, channel)
                else:
                    logger.warning(f"Unsupported notification channel type: {channel.type}")
                    
            except Exception as e:
                logger.error(f"Failed to send notification via {channel_name}: {e}")
                success = False
        
        return success
    
    def _send_email_notification(self, alert: Alert, rule: EscalationRule, channel: NotificationChannel):
        """Send email notification"""
        config = channel.config
        
        # Create message
        msg = MimeMultipart()
        msg['From'] = config.get('from_address', 'alerts@pixelated.ai')
        msg['To'] = ', '.join(rule.recipients)
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        # Create email body
        body = f"""
Alert Details:
==============
ID: {alert.alert_id}
Title: {alert.title}
Severity: {alert.severity.value.upper()}
Source: {alert.source}
Created: {alert.created_at}
Escalation Level: {alert.escalation_level.value}

Description:
{alert.description}

Metadata:
{json.dumps(alert.metadata, indent=2)}

Please acknowledge this alert in the monitoring system.
"""
        
        msg.attach(MimeText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['username'], config['password'])
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Sent email notification for alert {alert.alert_id}")
    
    def _send_slack_notification(self, alert: Alert, rule: EscalationRule, channel: NotificationChannel):
        """Send Slack notification"""
        import requests
        
        config = channel.config
        webhook_url = config.get('webhook_url')
        
        if not webhook_url:
            raise ValueError("Slack webhook URL not configured")
        
        # Create Slack message
        color_map = {
            AlertSeverity.CRITICAL: "danger",
            AlertSeverity.HIGH: "warning",
            AlertSeverity.MEDIUM: "good",
            AlertSeverity.LOW: "#439FE0"
        }
        
        payload = {
            "channel": config.get('channel', '#alerts'),
            "username": "Alert Bot",
            "icon_emoji": ":warning:",
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "good"),
                    "title": f"{alert.severity.value.upper()}: {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {"title": "Alert ID", "value": alert.alert_id, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Escalation Level", "value": alert.escalation_level.value, "short": True},
                        {"title": "Created", "value": alert.created_at, "short": True}
                    ],
                    "footer": "Pixelated Empathy AI Monitoring",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }
        
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        
        logger.info(f"Sent Slack notification for alert {alert.alert_id}")
    
    def _send_webhook_notification(self, alert: Alert, rule: EscalationRule, channel: NotificationChannel):
        """Send webhook notification"""
        import requests
        
        config = channel.config
        webhook_url = config.get('url')
        
        if not webhook_url:
            raise ValueError("Webhook URL not configured")
        
        payload = {
            "alert": alert.to_dict(),
            "escalation_rule": {
                "level": rule.level.value,
                "recipients": rule.recipients,
                "channels": rule.channels
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        headers = config.get('headers', {})
        timeout = config.get('timeout', 30)
        
        response = requests.post(webhook_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        logger.info(f"Sent webhook notification for alert {alert.alert_id}")
    
    def _log_escalation(self, alert: Alert, rule: EscalationRule, success: bool, error_message: str = None):
        """Log escalation attempt"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO escalation_history (
                        alert_id, escalation_level, recipients, channels,
                        escalated_at, success, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    rule.level.value,
                    json.dumps(rule.recipients),
                    json.dumps(rule.channels),
                    alert.last_escalated_at,
                    success,
                    error_message
                ))
        except Exception as e:
            logger.error(f"Failed to log escalation: {e}")
    
    def _store_alert_in_db(self, alert: Alert):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO alerts (
                        alert_id, title, description, severity, source, status,
                        created_at, acknowledged_at, resolved_at, acknowledged_by,
                        resolved_by, escalation_level, escalation_count,
                        last_escalated_at, metadata, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    alert.alert_id,
                    alert.title,
                    alert.description,
                    alert.severity.value,
                    alert.source,
                    alert.status.value,
                    alert.created_at,
                    alert.acknowledged_at,
                    alert.resolved_at,
                    alert.acknowledged_by,
                    alert.resolved_by,
                    alert.escalation_level.value,
                    alert.escalation_count,
                    alert.last_escalated_at,
                    json.dumps(alert.metadata)
                ))
        except Exception as e:
            logger.error(f"Failed to store alert in database: {e}")
    
    def _load_active_alerts(self):
        """Load active alerts from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM alerts
                    WHERE status IN ('active', 'escalated', 'acknowledged')
                """)
                
                for row in cursor.fetchall():
                    alert = Alert(
                        alert_id=row[0],
                        title=row[1],
                        description=row[2],
                        severity=AlertSeverity(row[3]),
                        source=row[4],
                        status=AlertStatus(row[5]),
                        created_at=row[6],
                        acknowledged_at=row[7],
                        resolved_at=row[8],
                        acknowledged_by=row[9],
                        resolved_by=row[10],
                        escalation_level=EscalationLevel(row[11]),
                        escalation_count=row[12],
                        last_escalated_at=row[13],
                        metadata=json.loads(row[14]) if row[14] else {}
                    )
                    
                    self.active_alerts[alert.alert_id] = alert
                    
                    # Restart escalation if needed
                    if alert.status == AlertStatus.ACTIVE:
                        self._start_escalation(alert)
                
                logger.info(f"Loaded {len(self.active_alerts)} active alerts from database")
                
        except Exception as e:
            logger.error(f"Failed to load active alerts: {e}")
    
    def _monitor_escalations(self):
        """Monitor escalations in background thread"""
        while True:
            try:
                # Check for alerts that need escalation
                current_time = datetime.now(timezone.utc)
                
                for alert in list(self.active_alerts.values()):
                    if alert.status in [AlertStatus.ACKNOWLEDGED, AlertStatus.RESOLVED]:
                        continue
                    
                    # Check if alert has been active too long without escalation
                    created_time = datetime.fromisoformat(alert.created_at.replace('Z', '+00:00'))
                    time_since_created = (current_time - created_time).total_seconds() / 60
                    
                    # Find appropriate escalation rule
                    rules = self.escalation_rules.get(alert.severity, [])
                    for rule in rules:
                        if (rule.level.value > alert.escalation_level.value and 
                            time_since_created >= rule.delay_minutes):
                            self._escalate_alert(alert, rule)
                            break
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in escalation monitoring: {e}")
                time.sleep(60)
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        import uuid
        return f"alert_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, alert_id: str) -> List[Dict[str, Any]]:
        """Get escalation history for an alert"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM escalation_history
                    WHERE alert_id = ?
                    ORDER BY escalated_at
                """, (alert_id,))
                
                history = []
                for row in cursor.fetchall():
                    history.append({
                        'escalation_level': row[2],
                        'recipients': json.loads(row[3]),
                        'channels': json.loads(row[4]),
                        'escalated_at': row[5],
                        'success': bool(row[6]),
                        'error_message': row[7]
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to get alert history: {e}")
            return []
    
    def get_escalation_statistics(self) -> Dict[str, Any]:
        """Get escalation statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Alert counts by severity
                cursor = conn.execute("""
                    SELECT severity, COUNT(*) FROM alerts
                    GROUP BY severity
                """)
                severity_counts = dict(cursor.fetchall())
                
                # Alert counts by status
                cursor = conn.execute("""
                    SELECT status, COUNT(*) FROM alerts
                    GROUP BY status
                """)
                status_counts = dict(cursor.fetchall())
                
                # Average escalation count
                cursor = conn.execute("""
                    SELECT AVG(escalation_count) FROM alerts
                    WHERE escalation_count > 0
                """)
                avg_escalations = cursor.fetchone()[0] or 0
                
                # Escalation success rate
                cursor = conn.execute("""
                    SELECT 
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                    FROM escalation_history
                """)
                success_rate = cursor.fetchone()[0] or 0
                
                return {
                    'active_alerts': len(self.active_alerts),
                    'severity_distribution': severity_counts,
                    'status_distribution': status_counts,
                    'average_escalations_per_alert': avg_escalations,
                    'escalation_success_rate': success_rate
                }
                
        except Exception as e:
            logger.error(f"Failed to get escalation statistics: {e}")
            return {}


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alert Escalation Manager")
    parser.add_argument('--db-path', help="Database path for alert storage")
    parser.add_argument('--config-path', help="Configuration file path")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create alert command
    create_parser = subparsers.add_parser('create', help='Create a new alert')
    create_parser.add_argument('title', help='Alert title')
    create_parser.add_argument('description', help='Alert description')
    create_parser.add_argument('severity', choices=['critical', 'high', 'medium', 'low', 'info'])
    create_parser.add_argument('source', help='Alert source')
    
    # Acknowledge command
    ack_parser = subparsers.add_parser('acknowledge', help='Acknowledge an alert')
    ack_parser.add_argument('alert_id', help='Alert ID to acknowledge')
    ack_parser.add_argument('user', help='User acknowledging the alert')
    
    # Resolve command
    resolve_parser = subparsers.add_parser('resolve', help='Resolve an alert')
    resolve_parser.add_argument('alert_id', help='Alert ID to resolve')
    resolve_parser.add_argument('user', help='User resolving the alert')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List active alerts')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show escalation statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create escalation manager
    manager = AlertEscalationManager(args.db_path, args.config_path)
    
    if args.command == 'create':
        severity = AlertSeverity(args.severity)
        alert_id = manager.create_alert(args.title, args.description, severity, args.source)
        print(f"Created alert: {alert_id}")
    
    elif args.command == 'acknowledge':
        success = manager.acknowledge_alert(args.alert_id, args.user)
        if success:
            print(f"Alert {args.alert_id} acknowledged")
        else:
            print(f"Failed to acknowledge alert {args.alert_id}")
    
    elif args.command == 'resolve':
        success = manager.resolve_alert(args.alert_id, args.user)
        if success:
            print(f"Alert {args.alert_id} resolved")
        else:
            print(f"Failed to resolve alert {args.alert_id}")
    
    elif args.command == 'list':
        alerts = manager.get_active_alerts()
        for alert in alerts:
            print(json.dumps(alert.to_dict(), indent=2))
    
    elif args.command == 'stats':
        stats = manager.get_escalation_statistics()
        print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
