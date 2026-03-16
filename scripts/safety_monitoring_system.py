"""
Safety Monitoring and Incident Response System
Part of Task 103: Safety Validation Certification

This module provides real-time safety monitoring and incident response capabilities:
- Real-time crisis detection monitoring
- Automated incident response
- Safety alert system
- Crisis intervention protocols
- Safety metrics tracking
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ResponseAction(Enum):
    """Automated response actions"""
    LOG_ONLY = "log_only"
    ALERT_STAFF = "alert_staff"
    IMMEDIATE_INTERVENTION = "immediate_intervention"
    EMERGENCY_SERVICES = "emergency_services"
    ESCALATE_TO_HUMAN = "escalate_to_human"

@dataclass
class SafetyIncident:
    """Safety incident record"""
    incident_id: str
    timestamp: datetime
    severity: IncidentSeverity
    crisis_level: str
    user_input: str
    detected_patterns: List[str]
    response_actions: List[ResponseAction]
    human_notified: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    notes: str = ""

@dataclass
class SafetyMetrics:
    """Real-time safety metrics"""
    total_interactions: int = 0
    crisis_detections: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    response_time_avg: float = 0.0
    incidents_resolved: int = 0
    human_interventions: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)

class SafetyMonitoringSystem:
    """
    Real-time Safety Monitoring and Incident Response System
    
    Provides continuous safety monitoring with automated response capabilities
    """
    
    def __init__(self):
        self.incidents: List[SafetyIncident] = []
        self.metrics = SafetyMetrics()
        self.monitoring_active = False
        self.alert_callbacks: List[Callable] = []
        
        # Response protocols by severity
        self.response_protocols = {
            IncidentSeverity.EMERGENCY: [
                ResponseAction.IMMEDIATE_INTERVENTION,
                ResponseAction.EMERGENCY_SERVICES,
                ResponseAction.ALERT_STAFF
            ],
            IncidentSeverity.CRITICAL: [
                ResponseAction.IMMEDIATE_INTERVENTION,
                ResponseAction.ESCALATE_TO_HUMAN,
                ResponseAction.ALERT_STAFF
            ],
            IncidentSeverity.HIGH: [
                ResponseAction.ESCALATE_TO_HUMAN,
                ResponseAction.ALERT_STAFF
            ],
            IncidentSeverity.MEDIUM: [
                ResponseAction.ALERT_STAFF,
                ResponseAction.LOG_ONLY
            ],
            IncidentSeverity.LOW: [
                ResponseAction.LOG_ONLY
            ]
        }
        
        # Crisis intervention resources
        self.crisis_resources = {
            "suicide_prevention": {
                "hotline": "988",
                "text": "Text HOME to 741741",
                "chat": "suicidepreventionlifeline.org/chat"
            },
            "domestic_violence": {
                "hotline": "1-800-799-7233",
                "text": "Text START to 88788",
                "chat": "thehotline.org"
            },
            "mental_health": {
                "hotline": "1-800-662-4357",
                "text": "Text MHA to 741741",
                "online": "mhanational.org/finding-help"
            }
        }
        
        logger.info("Safety monitoring system initialized")
    
    def start_monitoring(self):
        """Start real-time safety monitoring"""
        self.monitoring_active = True
        logger.info("Safety monitoring started")
    
    def stop_monitoring(self):
        """Stop safety monitoring"""
        self.monitoring_active = False
        logger.info("Safety monitoring stopped")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for safety alerts"""
        self.alert_callbacks.append(callback)
        logger.info("Alert callback added")
    
    async def process_user_input(self, user_input: str, user_id: str = None) -> Dict[str, Any]:
        """
        Process user input for safety monitoring
        Returns safety assessment and response recommendations
        """
        if not self.monitoring_active:
            return {"status": "monitoring_disabled"}
        
        start_time = time.time()
        self.metrics.total_interactions += 1
        
        # Detect crisis level and patterns
        crisis_assessment = await self._assess_crisis_level(user_input)
        
        # Determine incident severity
        incident_severity = self._determine_incident_severity(crisis_assessment)
        
        # Create incident if necessary
        incident = None
        if incident_severity != IncidentSeverity.LOW or crisis_assessment["crisis_level"] != "none":
            incident = await self._create_incident(
                user_input, crisis_assessment, incident_severity, user_id
            )
            
            # Execute response protocol
            await self._execute_response_protocol(incident)
        
        processing_time = time.time() - start_time
        
        # Update metrics
        self._update_metrics(crisis_assessment, processing_time)
        
        # Generate response
        response = self._generate_safety_response(crisis_assessment, incident)
        
        return response
    
    async def _assess_crisis_level(self, user_input: str) -> Dict[str, Any]:
        """Assess crisis level in user input"""
        
        # Crisis detection patterns (enhanced from safety validation system)
        crisis_patterns = {
            "emergency": [
                r'\b(took pills|overdose|bleeding|can\'t breathe)\b',
                r'\b(911|emergency|ambulance|hospital)\b'
            ],
            "critical": [
                r'\b(suicide|kill myself|end my life|want to die)\b',
                r'\b(self harm|cut myself|hurt myself)\b',
                r'\b(gun|knife|rope|bridge)\b.*\b(kill|die|end)\b'
            ],
            "high": [
                r'\b(depressed|hopeless|worthless|useless)\b',
                r'\b(can\'t go on|give up|no point|no hope)\b',
                r'\b(abuse|violence|threat|hurt me)\b'
            ],
            "medium": [
                r'\b(sad|upset|anxious|worried|scared)\b',
                r'\b(stressed|overwhelmed|struggling|difficult)\b',
                r'\b(lonely|isolated|alone|abandoned)\b'
            ],
            "low": [
                r'\b(tired|frustrated|annoyed|bothered)\b',
                r'\b(confused|uncertain|unsure|lost)\b'
            ]
        }
        
        detected_patterns = []
        crisis_level = "none"
        confidence = 0.0
        
        user_input_lower = user_input.lower()
        
        # Check patterns in order of severity
        for level in ["emergency", "critical", "high", "medium", "low"]:
            patterns = crisis_patterns.get(level, [])
            for pattern in patterns:
                import re
                if re.search(pattern, user_input_lower, re.IGNORECASE):
                    detected_patterns.append(pattern)
                    crisis_level = level
                    confidence = 0.95 if level in ["emergency", "critical"] else 0.8
                    break
            if crisis_level != "none":
                break
        
        # Additional context analysis
        context_indicators = self._analyze_context_indicators(user_input)
        
        return {
            "crisis_level": crisis_level,
            "confidence": confidence,
            "detected_patterns": detected_patterns,
            "context_indicators": context_indicators,
            "requires_intervention": crisis_level in ["emergency", "critical", "high"]
        }
    
    def _analyze_context_indicators(self, user_input: str) -> Dict[str, Any]:
        """Analyze additional context indicators"""
        
        indicators = {
            "urgency_words": 0,
            "negative_emotion_words": 0,
            "help_seeking": False,
            "past_tense_harm": False,
            "future_tense_harm": False,
            "specific_methods": False
        }
        
        user_input_lower = user_input.lower()
        
        # Count urgency indicators
        urgency_words = ["now", "today", "tonight", "immediately", "urgent", "emergency"]
        indicators["urgency_words"] = sum(1 for word in urgency_words if word in user_input_lower)
        
        # Count negative emotion words
        negative_emotions = ["sad", "depressed", "hopeless", "worthless", "scared", "terrified"]
        indicators["negative_emotion_words"] = sum(1 for word in negative_emotions if word in user_input_lower)
        
        # Check for help-seeking behavior
        help_words = ["help", "support", "talk", "listen", "advice"]
        indicators["help_seeking"] = any(word in user_input_lower for word in help_words)
        
        # Check temporal indicators
        indicators["past_tense_harm"] = any(phrase in user_input_lower for phrase in ["hurt myself", "cut myself", "tried to"])
        indicators["future_tense_harm"] = any(phrase in user_input_lower for phrase in ["going to", "will", "plan to"])
        
        # Check for specific methods
        methods = ["pills", "knife", "gun", "rope", "bridge", "car", "train"]
        indicators["specific_methods"] = any(method in user_input_lower for method in methods)
        
        return indicators
    
    def _determine_incident_severity(self, crisis_assessment: Dict[str, Any]) -> IncidentSeverity:
        """Determine incident severity based on crisis assessment"""
        
        crisis_level = crisis_assessment["crisis_level"]
        context = crisis_assessment["context_indicators"]
        
        # Emergency level
        if crisis_level == "emergency":
            return IncidentSeverity.EMERGENCY
        
        # Critical level
        if crisis_level == "critical":
            # Escalate to emergency if specific methods mentioned
            if context["specific_methods"] or context["urgency_words"] > 2:
                return IncidentSeverity.EMERGENCY
            return IncidentSeverity.CRITICAL
        
        # High level
        if crisis_level == "high":
            # Escalate if urgent or specific methods
            if context["urgency_words"] > 1 or context["specific_methods"]:
                return IncidentSeverity.CRITICAL
            return IncidentSeverity.HIGH
        
        # Medium level
        if crisis_level == "medium":
            return IncidentSeverity.MEDIUM
        
        # Low level
        if crisis_level == "low":
            return IncidentSeverity.LOW
        
        return IncidentSeverity.LOW
    
    async def _create_incident(self, user_input: str, crisis_assessment: Dict[str, Any], 
                              severity: IncidentSeverity, user_id: str = None) -> SafetyIncident:
        """Create safety incident record"""
        
        incident_id = str(uuid.uuid4())[:8]
        
        incident = SafetyIncident(
            incident_id=incident_id,
            timestamp=datetime.utcnow(),
            severity=severity,
            crisis_level=crisis_assessment["crisis_level"],
            user_input=user_input[:500],  # Truncate for privacy
            detected_patterns=crisis_assessment["detected_patterns"],
            response_actions=self.response_protocols.get(severity, [ResponseAction.LOG_ONLY])
        )
        
        self.incidents.append(incident)
        self.metrics.crisis_detections += 1
        
        logger.warning(f"Safety incident created: {incident_id} (Severity: {severity.value})")
        
        return incident
    
    async def _execute_response_protocol(self, incident: SafetyIncident):
        """Execute automated response protocol for incident"""
        
        for action in incident.response_actions:
            try:
                if action == ResponseAction.LOG_ONLY:
                    logger.info(f"Incident {incident.incident_id} logged")
                
                elif action == ResponseAction.ALERT_STAFF:
                    await self._alert_staff(incident)
                
                elif action == ResponseAction.IMMEDIATE_INTERVENTION:
                    await self._trigger_immediate_intervention(incident)
                
                elif action == ResponseAction.EMERGENCY_SERVICES:
                    await self._contact_emergency_services(incident)
                
                elif action == ResponseAction.ESCALATE_TO_HUMAN:
                    await self._escalate_to_human(incident)
                
            except Exception as e:
                logger.error(f"Failed to execute response action {action.value}: {e}")
    
    async def _alert_staff(self, incident: SafetyIncident):
        """Alert staff about safety incident"""
        
        alert_message = {
            "type": "safety_alert",
            "incident_id": incident.incident_id,
            "severity": incident.severity.value,
            "crisis_level": incident.crisis_level,
            "timestamp": incident.timestamp.isoformat(),
            "requires_immediate_attention": incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.EMERGENCY]
        }
        
        # Send alerts to registered callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert_message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        incident.human_notified = True
        logger.info(f"Staff alerted for incident {incident.incident_id}")
    
    async def _trigger_immediate_intervention(self, incident: SafetyIncident):
        """Trigger immediate intervention protocol"""
        
        # This would integrate with crisis intervention systems
        logger.critical(f"IMMEDIATE INTERVENTION REQUIRED: Incident {incident.incident_id}")
        
        # In a real system, this would:
        # 1. Connect user to crisis counselor
        # 2. Provide immediate crisis resources
        # 3. Initiate safety check protocols
        
        self.metrics.human_interventions += 1
    
    async def _contact_emergency_services(self, incident: SafetyIncident):
        """Contact emergency services (simulation)"""
        
        # This would integrate with emergency services in a real system
        logger.critical(f"EMERGENCY SERVICES CONTACT: Incident {incident.incident_id}")
        
        # In a real system, this would:
        # 1. Contact local emergency services
        # 2. Provide incident details
        # 3. Coordinate response
    
    async def _escalate_to_human(self, incident: SafetyIncident):
        """Escalate incident to human operator"""
        
        logger.warning(f"Human escalation for incident {incident.incident_id}")
        
        # This would route the conversation to a human operator
        self.metrics.human_interventions += 1
    
    def _generate_safety_response(self, crisis_assessment: Dict[str, Any], 
                                 incident: Optional[SafetyIncident]) -> Dict[str, Any]:
        """Generate appropriate safety response"""
        
        crisis_level = crisis_assessment["crisis_level"]
        
        response = {
            "safety_status": "monitored",
            "crisis_level": crisis_level,
            "requires_intervention": crisis_assessment["requires_intervention"],
            "incident_created": incident is not None,
            "response_type": "normal"
        }
        
        if incident:
            response["incident_id"] = incident.incident_id
            response["severity"] = incident.severity.value
        
        # Generate appropriate response based on crisis level
        if crisis_level == "emergency":
            response.update({
                "response_type": "emergency",
                "message": "I'm very concerned about your safety. Please contact emergency services immediately at 911 or go to your nearest emergency room.",
                "resources": self.crisis_resources["suicide_prevention"],
                "immediate_action_required": True
            })
        
        elif crisis_level == "critical":
            response.update({
                "response_type": "crisis_intervention",
                "message": "I'm concerned about what you're going through. You don't have to face this alone. Please reach out to a crisis counselor who can help.",
                "resources": self.crisis_resources["suicide_prevention"],
                "human_support_available": True
            })
        
        elif crisis_level == "high":
            response.update({
                "response_type": "supportive_intervention",
                "message": "It sounds like you're going through a really difficult time. I want you to know that support is available.",
                "resources": self.crisis_resources["mental_health"],
                "professional_help_recommended": True
            })
        
        elif crisis_level == "medium":
            response.update({
                "response_type": "empathetic_support",
                "message": "I hear that you're struggling right now. It's okay to feel this way, and there are people who can help.",
                "coping_strategies": True
            })
        
        return response
    
    def _update_metrics(self, crisis_assessment: Dict[str, Any], processing_time: float):
        """Update safety monitoring metrics"""
        
        # Update response time average
        total_time = self.metrics.response_time_avg * (self.metrics.total_interactions - 1)
        self.metrics.response_time_avg = (total_time + processing_time) / self.metrics.total_interactions
        
        self.metrics.last_updated = datetime.utcnow()
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get current safety monitoring metrics"""
        
        return {
            "total_interactions": self.metrics.total_interactions,
            "crisis_detections": self.metrics.crisis_detections,
            "detection_rate": (self.metrics.crisis_detections / self.metrics.total_interactions * 100) if self.metrics.total_interactions > 0 else 0,
            "false_positives": self.metrics.false_positives,
            "false_negatives": self.metrics.false_negatives,
            "average_response_time": self.metrics.response_time_avg,
            "incidents_resolved": self.metrics.incidents_resolved,
            "human_interventions": self.metrics.human_interventions,
            "monitoring_active": self.monitoring_active,
            "last_updated": self.metrics.last_updated.isoformat()
        }
    
    def get_recent_incidents(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent safety incidents"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_incidents = [
            incident for incident in self.incidents 
            if incident.timestamp >= cutoff_time
        ]
        
        return [
            {
                "incident_id": incident.incident_id,
                "timestamp": incident.timestamp.isoformat(),
                "severity": incident.severity.value,
                "crisis_level": incident.crisis_level,
                "resolved": incident.resolved,
                "human_notified": incident.human_notified,
                "response_actions": [action.value for action in incident.response_actions]
            }
            for incident in recent_incidents
        ]
    
    async def resolve_incident(self, incident_id: str, resolution_notes: str = ""):
        """Mark incident as resolved"""
        
        for incident in self.incidents:
            if incident.incident_id == incident_id:
                incident.resolved = True
                incident.resolution_time = datetime.utcnow()
                incident.notes = resolution_notes
                self.metrics.incidents_resolved += 1
                
                logger.info(f"Incident {incident_id} resolved")
                return True
        
        return False

# Example usage and testing
if __name__ == "__main__":
    async def test_safety_monitoring():
        # Initialize monitoring system
        monitor = SafetyMonitoringSystem()
        
        # Add alert callback
        async def alert_callback(alert):
            print(f"🚨 SAFETY ALERT: {alert}")
        
        monitor.add_alert_callback(alert_callback)
        monitor.start_monitoring()
        
        # Test various inputs
        test_inputs = [
            "I want to kill myself",
            "I'm feeling really depressed and hopeless",
            "I'm having a bad day at work",
            "What's the weather like?",
            "I took too many pills and I'm scared"
        ]
        
        print("Testing Safety Monitoring System")
        print("=" * 50)
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\nTest {i}: {test_input}")
            response = await monitor.process_user_input(test_input, f"user_{i}")
            print(f"Response: {response['response_type']} (Crisis Level: {response['crisis_level']})")
            
            if response.get("incident_created"):
                print(f"Incident Created: {response['incident_id']} (Severity: {response['severity']})")
        
        # Print metrics
        print(f"\nSafety Metrics:")
        metrics = monitor.get_safety_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Print recent incidents
        print(f"\nRecent Incidents:")
        incidents = monitor.get_recent_incidents()
        for incident in incidents:
            print(f"  {incident['incident_id']}: {incident['severity']} - {incident['crisis_level']}")
    
    asyncio.run(test_safety_monitoring())
