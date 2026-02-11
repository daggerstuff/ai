#!/usr/bin/env python3
"""
Alert Fatigue Prevention System for Pixelated Empathy AI
Implements intelligent alert grouping, deduplication, and fatigue prevention
"""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertState(Enum):
    """Alert states for tracking"""
    NEW = "new"
    GROUPED = "grouped"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"
    RESOLVED = "resolved"

class GroupingStrategy(Enum):
    """Alert grouping strategies"""
    SIMILARITY = "similarity"
    TIME_WINDOW = "time_window"
    SOURCE = "source"
    PATTERN = "pattern"
    CUSTOM = "custom"

@dataclass
class AlertFingerprint:
    """Unique fingerprint for alert identification"""
    source: str
    alert_type: str
    severity: str
    key_attributes: Dict[str, str]
    fingerprint_hash: str = field(init=False)
    
    def __post_init__(self):
        # Create deterministic hash from key attributes
        content = f"{self.source}:{self.alert_type}:{self.severity}"
        for key in sorted(self.key_attributes.keys()):
            content += f":{key}={self.key_attributes[key]}"
        self.fingerprint_hash = hashlib.md5(content.encode()).hexdigest()

@dataclass
class AlertGroup:
    """Group of related alerts"""
    group_id: str
    fingerprint: AlertFingerprint
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    count: int = 0
    state: AlertState = AlertState.NEW
    suppression_count: int = 0
    escalation_level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FatigueRule:
    """Rule for preventing alert fatigue"""
    rule_id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    actions: List[str]
    enabled: bool = True
    priority: int = 100
    created_at: datetime = field(default_factory=datetime.utcnow)

class AlertFatiguePreventionSystem:
    """Main alert fatigue prevention and intelligent grouping system"""
    
    def __init__(self, db_path: str = "/home/vivi/pixelated/ai/monitoring/alert_fatigue.db"):
        self.db_path = db_path
        self.active_groups: Dict[str, AlertGroup] = {}
        self.fatigue_rules: Dict[str, FatigueRule] = {}
        self.suppression_patterns: List[Dict[str, Any]] = []
        self.grouping_config = self._load_default_grouping_config()
        self.setup_database()
        self.load_fatigue_rules()
    
    def setup_database(self):
        """Initialize the alert fatigue database"""
        with sqlite3.connect(self.db_path) as conn:
            # Alert groups table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_groups (
                    group_id TEXT PRIMARY KEY,
                    fingerprint_hash TEXT NOT NULL,
                    source TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    first_seen DATETIME NOT NULL,
                    last_seen DATETIME NOT NULL,
                    count INTEGER DEFAULT 1,
                    state TEXT DEFAULT 'new',
                    suppression_count INTEGER DEFAULT 0,
                    escalation_level INTEGER DEFAULT 0,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Individual alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS grouped_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    alert_data TEXT NOT NULL,
                    received_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (group_id) REFERENCES alert_groups (group_id)
                )
            """)
            
            # Fatigue rules table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fatigue_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    conditions TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT 1,
                    priority INTEGER DEFAULT 100,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Suppression history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS suppression_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    rule_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT,
                    suppressed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (group_id) REFERENCES alert_groups (group_id),
                    FOREIGN KEY (rule_id) REFERENCES fatigue_rules (rule_id)
                )
            """)
    
    def _load_default_grouping_config(self) -> Dict[str, Any]:
        """Load default grouping configuration"""
        return {
            "time_window_minutes": 5,
            "similarity_threshold": 0.8,
            "max_group_size": 100,
            "escalation_thresholds": [10, 50, 100],
            "suppression_window_minutes": 60,
            "grouping_strategies": [
                GroupingStrategy.SIMILARITY,
                GroupingStrategy.TIME_WINDOW,
                GroupingStrategy.SOURCE
            ]
        }
    
    def load_fatigue_rules(self):
        """Load fatigue prevention rules from database"""
        with sqlite3.connect(self.db_path) as conn:
            rules = conn.execute("""
                SELECT rule_id, name, description, conditions, actions, enabled, priority
                FROM fatigue_rules WHERE enabled = 1
                ORDER BY priority ASC
            """).fetchall()
            
            for rule_data in rules:
                rule_id, name, desc, conditions_json, actions_json, enabled, priority = rule_data
                
                rule = FatigueRule(
                    rule_id=rule_id,
                    name=name,
                    description=desc,
                    conditions=json.loads(conditions_json),
                    actions=json.loads(actions_json),
                    enabled=bool(enabled),
                    priority=priority
                )
                
                self.fatigue_rules[rule_id] = rule
        
        # Add default rules if none exist
        if not self.fatigue_rules:
            self._create_default_fatigue_rules()
    
    def _create_default_fatigue_rules(self):
        """Create default fatigue prevention rules"""
        default_rules = [
            {
                "rule_id": "duplicate_suppression",
                "name": "Duplicate Alert Suppression",
                "description": "Suppress duplicate alerts within time window",
                "conditions": {
                    "duplicate_count": {">=": 3},
                    "time_window_minutes": 15
                },
                "actions": ["suppress", "group"],
                "priority": 10
            },
            {
                "rule_id": "high_frequency_suppression",
                "name": "High Frequency Alert Suppression",
                "description": "Suppress alerts with high frequency",
                "conditions": {
                    "alerts_per_minute": {">=": 5},
                    "time_window_minutes": 10
                },
                "actions": ["suppress", "escalate_summary"],
                "priority": 20
            },
            {
                "rule_id": "similar_alert_grouping",
                "name": "Similar Alert Grouping",
                "description": "Group alerts with similar content",
                "conditions": {
                    "similarity_score": {">=": 0.8},
                    "max_group_size": 50
                },
                "actions": ["group", "summarize"],
                "priority": 30
            },
            {
                "rule_id": "escalation_threshold",
                "name": "Alert Escalation Threshold",
                "description": "Escalate when alert count exceeds threshold",
                "conditions": {
                    "group_count": {">=": 10},
                    "time_window_minutes": 30
                },
                "actions": ["escalate", "notify_summary"],
                "priority": 40
            },
            {
                "rule_id": "maintenance_window_suppression",
                "name": "Maintenance Window Suppression",
                "description": "Suppress alerts during maintenance windows",
                "conditions": {
                    "maintenance_mode": True,
                    "alert_types": ["system_down", "service_unavailable"]
                },
                "actions": ["suppress", "log_only"],
                "priority": 5
            }
        ]
        
        for rule_data in default_rules:
            rule = FatigueRule(**rule_data)
            self.add_fatigue_rule(rule)
    
    def add_fatigue_rule(self, rule: FatigueRule):
        """Add a new fatigue prevention rule"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO fatigue_rules 
                (rule_id, name, description, conditions, actions, enabled, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                rule.rule_id, rule.name, rule.description,
                json.dumps(rule.conditions), json.dumps(rule.actions),
                rule.enabled, rule.priority
            ))
        
        self.fatigue_rules[rule.rule_id] = rule
        logger.info(f"Added fatigue rule: {rule.name}")
    
    def create_alert_fingerprint(self, alert_data: Dict[str, Any]) -> AlertFingerprint:
        """Create a unique fingerprint for an alert"""
        # Extract key attributes for fingerprinting
        source = alert_data.get('source', 'unknown')
        alert_type = alert_data.get('alert_type', alert_data.get('title', 'unknown'))
        severity = alert_data.get('priority', alert_data.get('severity', 'medium'))
        
        # Extract key attributes that should be considered for grouping
        key_attributes = {}
        
        # Common attributes to consider
        for attr in ['service', 'host', 'component', 'error_type', 'metric_name']:
            if attr in alert_data:
                key_attributes[attr] = str(alert_data[attr])
        
        # Extract from metadata if available
        metadata = alert_data.get('metadata', {})
        for attr in ['service_name', 'server', 'component', 'error_code']:
            if attr in metadata:
                key_attributes[attr] = str(metadata[attr])
        
        # Normalize alert message for pattern matching
        message = alert_data.get('message', '')
        normalized_message = self._normalize_message(message)
        if normalized_message:
            key_attributes['message_pattern'] = normalized_message
        
        return AlertFingerprint(
            source=source,
            alert_type=alert_type,
            severity=severity,
            key_attributes=key_attributes
        )
    
    def _normalize_message(self, message: str) -> str:
        """Normalize alert message for pattern matching"""
        if not message:
            return ""
        
        # Remove timestamps, IDs, and other variable content
        patterns_to_remove = [
            r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}',  # Timestamps
            r'\b\d+\.\d+\.\d+\.\d+\b',  # IP addresses
            r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b',  # UUIDs
            r'\b\d{10,}\b',  # Large numbers (likely IDs)
            r'\b[A-Za-z0-9]{20,}\b'  # Long alphanumeric strings (likely tokens)
        ]
        
        normalized = message.lower()
        for pattern in patterns_to_remove:
            normalized = re.sub(pattern, '<REDACTED>', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    async def process_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming alert through the fatigue prevention system"""
        
        # Create fingerprint for the alert
        fingerprint = self.create_alert_fingerprint(alert_data)
        
        # Check if this alert should be grouped with existing alerts
        group = await self._find_or_create_group(fingerprint, alert_data)
        
        # Add alert to the group
        group.alerts.append(alert_data)
        group.count += 1
        group.last_seen = datetime.utcnow()
        
        # Apply fatigue prevention rules
        actions = await self._apply_fatigue_rules(group, alert_data)
        
        # Update group state based on actions
        await self._update_group_state(group, actions)
        
        # Persist group to database
        await self._persist_group(group)
        
        return {
            "group_id": group.group_id,
            "fingerprint_hash": fingerprint.fingerprint_hash,
            "actions_taken": actions,
            "group_count": group.count,
            "state": group.state.value,
            "should_notify": "suppress" not in actions
        }
    
    async def _find_or_create_group(self, fingerprint: AlertFingerprint, 
                                  alert_data: Dict[str, Any]) -> AlertGroup:
        """Find existing group or create new one for the alert"""
        
        # Check if we have an active group with the same fingerprint
        if fingerprint.fingerprint_hash in self.active_groups:
            group = self.active_groups[fingerprint.fingerprint_hash]
            
            # Check if group is still within time window
            time_window = timedelta(minutes=self.grouping_config["time_window_minutes"])
            if datetime.utcnow() - group.last_seen <= time_window:
                return group
        
        # Check for similar groups using different strategies
        similar_group = await self._find_similar_group(fingerprint, alert_data)
        if similar_group:
            return similar_group
        
        # Create new group
        group_id = f"{fingerprint.fingerprint_hash}_{int(time.time())}"
        group = AlertGroup(
            group_id=group_id,
            fingerprint=fingerprint
        )
        
        self.active_groups[fingerprint.fingerprint_hash] = group
        return group
    
    async def _find_similar_group(self, fingerprint: AlertFingerprint, 
                                alert_data: Dict[str, Any]) -> Optional[AlertGroup]:
        """Find similar existing group using various strategies"""
        
        # Strategy 1: Source and alert type matching
        for group in self.active_groups.values():
            if (group.fingerprint.source == fingerprint.source and
                group.fingerprint.alert_type == fingerprint.alert_type and
                group.state != AlertState.RESOLVED):
                
                # Check time window
                time_window = timedelta(minutes=self.grouping_config["time_window_minutes"])
                if datetime.utcnow() - group.last_seen <= time_window:
                    
                    # Calculate similarity score
                    similarity = self._calculate_similarity(group.fingerprint, fingerprint)
                    if similarity >= self.grouping_config["similarity_threshold"]:
                        return group
        
        return None
    
    def _calculate_similarity(self, fp1: AlertFingerprint, fp2: AlertFingerprint) -> float:
        """Calculate similarity score between two fingerprints"""
        
        # Base similarity from matching fields
        similarity_score = 0.0
        total_weight = 0.0
        
        # Weight different attributes
        weights = {
            'source': 0.3,
            'alert_type': 0.3,
            'severity': 0.2,
            'key_attributes': 0.2
        }
        
        # Compare source
        if fp1.source == fp2.source:
            similarity_score += weights['source']
        total_weight += weights['source']
        
        # Compare alert type
        if fp1.alert_type == fp2.alert_type:
            similarity_score += weights['alert_type']
        total_weight += weights['alert_type']
        
        # Compare severity
        if fp1.severity == fp2.severity:
            similarity_score += weights['severity']
        total_weight += weights['severity']
        
        # Compare key attributes
        common_keys = set(fp1.key_attributes.keys()) & set(fp2.key_attributes.keys())
        if common_keys:
            matching_attrs = sum(1 for key in common_keys 
                               if fp1.key_attributes[key] == fp2.key_attributes[key])
            attr_similarity = matching_attrs / len(common_keys)
            similarity_score += weights['key_attributes'] * attr_similarity
        total_weight += weights['key_attributes']
        
        return similarity_score / total_weight if total_weight > 0 else 0.0
    
    async def _apply_fatigue_rules(self, group: AlertGroup, 
                                 alert_data: Dict[str, Any]) -> List[str]:
        """Apply fatigue prevention rules to determine actions"""
        
        actions = []
        
        # Sort rules by priority
        sorted_rules = sorted(self.fatigue_rules.values(), key=lambda r: r.priority)
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            # Check if rule conditions are met
            if await self._evaluate_rule_conditions(rule, group, alert_data):
                actions.extend(rule.actions)
                
                # Log rule application
                await self._log_rule_application(rule, group, actions)
                
                logger.info(f"Applied fatigue rule '{rule.name}' to group {group.group_id}")
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(actions))
    
    async def _evaluate_rule_conditions(self, rule: FatigueRule, group: AlertGroup, 
                                      alert_data: Dict[str, Any]) -> bool:
        """Evaluate if rule conditions are met"""
        
        conditions = rule.conditions
        
        # Check duplicate count condition
        if "duplicate_count" in conditions:
            threshold = list(conditions["duplicate_count"].values())[0]
            operator = list(conditions["duplicate_count"].keys())[0]
            
            if not self._compare_values(group.count, threshold, operator):
                return False
        
        # Check alerts per minute condition
        if "alerts_per_minute" in conditions:
            threshold = list(conditions["alerts_per_minute"].values())[0]
            operator = list(conditions["alerts_per_minute"].keys())[0]
            
            time_window = conditions.get("time_window_minutes", 10)
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window)
            
            recent_alerts = sum(1 for alert in group.alerts 
                              if alert.get('timestamp', datetime.utcnow()) > cutoff_time)
            alerts_per_minute = recent_alerts / time_window
            
            if not self._compare_values(alerts_per_minute, threshold, operator):
                return False
        
        # Check similarity score condition
        if "similarity_score" in conditions:
            threshold = list(conditions["similarity_score"].values())[0]
            operator = list(conditions["similarity_score"].keys())[0]
            
            # Calculate average similarity within group
            if len(group.alerts) > 1:
                similarities = []
                for i in range(len(group.alerts) - 1):
                    fp1 = self.create_alert_fingerprint(group.alerts[i])
                    fp2 = self.create_alert_fingerprint(group.alerts[i + 1])
                    similarities.append(self._calculate_similarity(fp1, fp2))
                
                avg_similarity = sum(similarities) / len(similarities)
                if not self._compare_values(avg_similarity, threshold, operator):
                    return False
        
        # Check group count condition
        if "group_count" in conditions:
            threshold = list(conditions["group_count"].values())[0]
            operator = list(conditions["group_count"].keys())[0]
            
            if not self._compare_values(group.count, threshold, operator):
                return False
        
        # Check maintenance mode condition
        if "maintenance_mode" in conditions:
            # This would be checked against a maintenance mode flag
            # For now, assume maintenance mode is False
            maintenance_mode = False  # This should come from configuration
            if conditions["maintenance_mode"] != maintenance_mode:
                return False
        
        # Check alert types condition
        if "alert_types" in conditions:
            allowed_types = conditions["alert_types"]
            alert_type = alert_data.get('alert_type', alert_data.get('title', ''))
            if alert_type not in allowed_types:
                return False
        
        return True
    
    def _compare_values(self, value: float, threshold: float, operator: str) -> bool:
        """Compare values using the specified operator"""
        if operator == ">=":
            return value >= threshold
        elif operator == ">":
            return value > threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "<":
            return value < threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False
    
    async def _update_group_state(self, group: AlertGroup, actions: List[str]):
        """Update group state based on applied actions"""
        
        if "suppress" in actions:
            group.state = AlertState.SUPPRESSED
            group.suppression_count += 1
        elif "escalate" in actions:
            group.state = AlertState.ESCALATED
            group.escalation_level += 1
        elif "group" in actions:
            group.state = AlertState.GROUPED
        
        # Update metadata
        group.metadata.update({
            "last_actions": actions,
            "last_updated": datetime.utcnow().isoformat()
        })
    
    async def _persist_group(self, group: AlertGroup):
        """Persist group data to database"""
        
        with sqlite3.connect(self.db_path) as conn:
            # Update or insert group
            conn.execute("""
                INSERT OR REPLACE INTO alert_groups 
                (group_id, fingerprint_hash, source, alert_type, severity,
                 first_seen, last_seen, count, state, suppression_count,
                 escalation_level, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                group.group_id, group.fingerprint.fingerprint_hash,
                group.fingerprint.source, group.fingerprint.alert_type,
                group.fingerprint.severity, group.first_seen, group.last_seen,
                group.count, group.state.value, group.suppression_count,
                group.escalation_level, json.dumps(group.metadata)
            ))
            
            # Insert new alerts
            for alert in group.alerts[-1:]:  # Only insert the latest alert
                conn.execute("""
                    INSERT INTO grouped_alerts (group_id, alert_data)
                    VALUES (?, ?)
                """, (group.group_id, json.dumps(alert)))
    
    async def _log_rule_application(self, rule: FatigueRule, group: AlertGroup, 
                                  actions: List[str]):
        """Log the application of a fatigue rule"""
        
        with sqlite3.connect(self.db_path) as conn:
            for action in actions:
                conn.execute("""
                    INSERT INTO suppression_history 
                    (group_id, rule_id, action, reason)
                    VALUES (?, ?, ?, ?)
                """, (group.group_id, rule.rule_id, action, rule.description))
    
    async def get_group_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of alert groups in the specified time period"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get group statistics
            stats = conn.execute("""
                SELECT state, COUNT(*) as count, SUM(count) as total_alerts
                FROM alert_groups 
                WHERE last_seen > ?
                GROUP BY state
            """, (cutoff_time.isoformat(),)).fetchall()
            
            # Get top sources
            top_sources = conn.execute("""
                SELECT source, COUNT(*) as group_count, SUM(count) as alert_count
                FROM alert_groups 
                WHERE last_seen > ?
                GROUP BY source
                ORDER BY alert_count DESC
                LIMIT 10
            """, (cutoff_time.isoformat(),)).fetchall()
            
            # Get suppression statistics
            suppressions = conn.execute("""
                SELECT rule_id, action, COUNT(*) as count
                FROM suppression_history 
                WHERE suppressed_at > ?
                GROUP BY rule_id, action
                ORDER BY count DESC
            """, (cutoff_time.isoformat(),)).fetchall()
        
        return {
            "time_period_hours": hours,
            "group_statistics": [{"state": s[0], "groups": s[1], "total_alerts": s[2]} for s in stats],
            "top_sources": [{"source": s[0], "groups": s[1], "alerts": s[2]} for s in top_sources],
            "suppression_actions": [{"rule": s[0], "action": s[1], "count": s[2]} for s in suppressions],
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def cleanup_old_groups(self, days: int = 7):
        """Clean up old resolved groups"""
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Remove old groups and their alerts
            removed_groups = conn.execute("""
                DELETE FROM alert_groups 
                WHERE state = 'resolved' AND last_seen < ?
            """, (cutoff_time.isoformat(),)).rowcount
            
            # Clean up orphaned alerts
            conn.execute("""
                DELETE FROM grouped_alerts 
                WHERE group_id NOT IN (SELECT group_id FROM alert_groups)
            """)
            
            # Clean up old suppression history
            conn.execute("""
                DELETE FROM suppression_history 
                WHERE suppressed_at < ?
            """, (cutoff_time.isoformat(),))
        
        # Remove from active groups
        to_remove = []
        for fingerprint_hash, group in self.active_groups.items():
            if group.state == AlertState.RESOLVED and group.last_seen < cutoff_time:
                to_remove.append(fingerprint_hash)
        
        for fingerprint_hash in to_remove:
            del self.active_groups[fingerprint_hash]
        
        logger.info(f"Cleaned up {removed_groups} old alert groups")
        return removed_groups

# Example usage and testing
async def example_usage():
    """Example of how to use the alert fatigue prevention system"""
    
    # Initialize system
    afp = AlertFatiguePreventionSystem()
    
    # Process some example alerts
    test_alerts = [
        {
            "title": "High CPU Usage",
            "message": "CPU usage is 87% on server-01",
            "priority": "medium",
            "source": "monitoring",
            "metadata": {"server": "server-01", "cpu_usage": 87}
        },
        {
            "title": "High CPU Usage", 
            "message": "CPU usage is 89% on server-01",
            "priority": "medium",
            "source": "monitoring",
            "metadata": {"server": "server-01", "cpu_usage": 89}
        },
        {
            "title": "High CPU Usage",
            "message": "CPU usage is 91% on server-01", 
            "priority": "high",
            "source": "monitoring",
            "metadata": {"server": "server-01", "cpu_usage": 91}
        }
    ]
    
    # Process alerts
    for i, alert in enumerate(test_alerts):
        result = await afp.process_alert(alert)
        print(f"Alert {i+1} processed:")
        print(f"  Group ID: {result['group_id']}")
        print(f"  Actions: {result['actions_taken']}")
        print(f"  Should notify: {result['should_notify']}")
        print(f"  Group count: {result['group_count']}")
        print()
    
    # Get summary
    summary = await afp.get_group_summary(hours=1)
    print("Alert Group Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    asyncio.run(example_usage())
