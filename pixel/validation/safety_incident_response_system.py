"""
Safety Incident Response System

Automated system for detecting, classifying, and responding to safety incidents
in the Pixelated Empathy AI system. Implements the procedures defined in the
safety incident response procedures document.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class IncidentLevel(Enum):
    """Safety incident severity levels"""
    CRITICAL = "critical"      # Level 1: Immediate response (0-5 minutes)
    HIGH = "high"             # Level 2: Rapid response (15 minutes)
    MODERATE = "moderate"     # Level 3: Standard response (1 hour)
    LOW = "low"              # Level 4: Routine response (4 hours)


class IncidentStatus(Enum):
    """Incident status tracking"""
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    CONTAINING = "containing"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ResponseAction(Enum):
    """Types of response actions"""
    SYSTEM_SHUTDOWN = "system_shutdown"
    ALERT_STAKEHOLDERS = "alert_stakeholders"
    DOCUMENT_INCIDENT = "document_incident"
    ASSESS_SCOPE = "assess_scope"
    ACTIVATE_EMERGENCY = "activate_emergency"
    ISOLATE_PROBLEM = "isolate_problem"
    CLIENT_SAFETY_MEASURES = "client_safety_measures"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    IMPLEMENT_FIXES = "implement_fixes"


@dataclass
class SafetyIncident:
    """Safety incident record"""
    incident_id: str
    level: IncidentLevel
    status: IncidentStatus
    title: str
    description: str
    detection_time: datetime
    affected_systems: List[str] = field(default_factory=list)
    affected_clients: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    stakeholders_notified: List[str] = field(default_factory=list)
    resolution_time: Optional[datetime] = None
    lessons_learned: List[str] = field(default_factory=list)
    
    # Response tracking
    response_start_time: Optional[datetime] = None
    acknowledgment_time: Optional[datetime] = None
    containment_time: Optional[datetime] = None
    
    # Metadata
    detected_by: str = "system"
    assigned_to: Optional[str] = None
    priority_score: float = 0.0
    tags: List[str] = field(default_factory=list)


@dataclass
class ResponsePlan:
    """Incident response plan"""
    level: IncidentLevel
    max_response_time_minutes: int
    required_actions: List[ResponseAction]
    stakeholder_groups: List[str]
    escalation_triggers: List[str]
    communication_methods: List[str]


class SafetyIncidentResponseSystem:
    """
    Automated safety incident response system
    
    Handles detection, classification, response coordination, and resolution
    of safety incidents according to established procedures.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the incident response system"""
        
        self.config = self._load_config(config_path)
        
        # Incident storage
        self.active_incidents: Dict[str, SafetyIncident] = {}
        self.incident_history: List[SafetyIncident] = []
        
        # Response plans
        self.response_plans = self._initialize_response_plans()
        
        # Stakeholder contacts
        self.stakeholder_contacts = self._load_stakeholder_contacts()
        
        # Response callbacks
        self.response_callbacks: Dict[ResponseAction, List[Callable]] = {
            action: [] for action in ResponseAction
        }
        
        # System state
        self.system_status = "operational"
        self.emergency_mode = False
        
        # Statistics
        self.stats = {
            "total_incidents": 0,
            "critical_incidents": 0,
            "average_response_time": 0.0,
            "incidents_by_level": {level.value: 0 for level in IncidentLevel},
            "system_uptime_start": datetime.now(),
        }
        
        logger.info("Safety incident response system initialized")

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            "auto_escalation_enabled": True,
            "max_response_time_critical": 5,    # minutes
            "max_response_time_high": 15,       # minutes
            "max_response_time_moderate": 60,   # minutes
            "max_response_time_low": 240,       # minutes
            "auto_system_shutdown_threshold": "critical",
            "client_notification_required": ["critical", "high"],
            "regulatory_notification_required": ["critical"],
            "max_incident_history": 1000,
            "incident_retention_days": 365,
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return default_config

    def _initialize_response_plans(self) -> Dict[IncidentLevel, ResponsePlan]:
        """Initialize response plans for each incident level"""
        return {
            IncidentLevel.CRITICAL: ResponsePlan(
                level=IncidentLevel.CRITICAL,
                max_response_time_minutes=5,
                required_actions=[
                    ResponseAction.SYSTEM_SHUTDOWN,
                    ResponseAction.ALERT_STAKEHOLDERS,
                    ResponseAction.DOCUMENT_INCIDENT,
                    ResponseAction.ASSESS_SCOPE,
                    ResponseAction.ACTIVATE_EMERGENCY,
                    ResponseAction.ISOLATE_PROBLEM,
                    ResponseAction.CLIENT_SAFETY_MEASURES,
                ],
                stakeholder_groups=["on_call_engineer", "clinical_supervisor", "system_admin", "technical_director"],
                escalation_triggers=["no_response_2min", "scope_expansion", "client_harm"],
                communication_methods=["phone", "sms", "email", "slack"],
            ),
            IncidentLevel.HIGH: ResponsePlan(
                level=IncidentLevel.HIGH,
                max_response_time_minutes=15,
                required_actions=[
                    ResponseAction.ALERT_STAKEHOLDERS,
                    ResponseAction.DOCUMENT_INCIDENT,
                    ResponseAction.ASSESS_SCOPE,
                    ResponseAction.ROOT_CAUSE_ANALYSIS,
                    ResponseAction.IMPLEMENT_FIXES,
                ],
                stakeholder_groups=["safety_engineer", "technical_lead", "clinical_supervisor"],
                escalation_triggers=["no_response_10min", "multiple_failures", "safety_threshold_breach"],
                communication_methods=["email", "slack"],
            ),
            IncidentLevel.MODERATE: ResponsePlan(
                level=IncidentLevel.MODERATE,
                max_response_time_minutes=60,
                required_actions=[
                    ResponseAction.DOCUMENT_INCIDENT,
                    ResponseAction.ROOT_CAUSE_ANALYSIS,
                    ResponseAction.IMPLEMENT_FIXES,
                ],
                stakeholder_groups=["technical_team", "qa_team"],
                escalation_triggers=["no_response_45min", "recurring_issue"],
                communication_methods=["email"],
            ),
            IncidentLevel.LOW: ResponsePlan(
                level=IncidentLevel.LOW,
                max_response_time_minutes=240,
                required_actions=[
                    ResponseAction.DOCUMENT_INCIDENT,
                    ResponseAction.ROOT_CAUSE_ANALYSIS,
                ],
                stakeholder_groups=["technical_team"],
                escalation_triggers=["no_response_3hours"],
                communication_methods=["email"],
            ),
        }

    def _load_stakeholder_contacts(self) -> Dict[str, Dict[str, str]]:
        """Load stakeholder contact information"""
        # In production, this would load from a secure configuration file
        return {
            "on_call_engineer": {
                "name": "On-Call Safety Engineer",
                "phone": "+1-555-SAFETY",
                "email": "safety-oncall@pixelated.ai",
                "slack": "@safety-oncall",
            },
            "clinical_supervisor": {
                "name": "Clinical Supervisor",
                "phone": "+1-555-CLINIC",
                "email": "clinical-supervisor@pixelated.ai",
                "slack": "@clinical-supervisor",
            },
            "system_admin": {
                "name": "System Administrator",
                "phone": "+1-555-SYSADM",
                "email": "sysadmin@pixelated.ai",
                "slack": "@sysadmin",
            },
            "technical_director": {
                "name": "Technical Director",
                "phone": "+1-555-TECHDIR",
                "email": "tech-director@pixelated.ai",
                "slack": "@tech-director",
            },
        }

    async def detect_incident(
        self,
        title: str,
        description: str,
        level: IncidentLevel,
        affected_systems: Optional[List[str]] = None,
        affected_clients: Optional[List[str]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        detected_by: str = "system"
    ) -> str:
        """
        Detect and register a new safety incident
        
        Args:
            title: Brief incident title
            description: Detailed incident description
            level: Incident severity level
            affected_systems: List of affected system components
            affected_clients: List of affected client IDs
            evidence: Supporting evidence and data
            detected_by: Who/what detected the incident
            
        Returns:
            Incident ID
        """
        
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
        
        incident = SafetyIncident(
            incident_id=incident_id,
            level=level,
            status=IncidentStatus.DETECTED,
            title=title,
            description=description,
            detection_time=datetime.now(),
            affected_systems=affected_systems or [],
            affected_clients=affected_clients or [],
            evidence=evidence or {},
            detected_by=detected_by,
        )
        
        # Calculate priority score
        incident.priority_score = self._calculate_priority_score(incident)
        
        # Store incident
        self.active_incidents[incident_id] = incident
        self.stats["total_incidents"] += 1
        self.stats["incidents_by_level"][level.value] += 1
        
        if level == IncidentLevel.CRITICAL:
            self.stats["critical_incidents"] += 1
        
        logger.critical(f"Safety incident detected: {incident_id} - {title} ({level.value})")
        
        # Initiate response
        await self._initiate_response(incident)
        
        return incident_id

    def _calculate_priority_score(self, incident: SafetyIncident) -> float:
        """Calculate incident priority score"""
        base_scores = {
            IncidentLevel.CRITICAL: 1.0,
            IncidentLevel.HIGH: 0.8,
            IncidentLevel.MODERATE: 0.6,
            IncidentLevel.LOW: 0.4,
        }
        
        score = base_scores[incident.level]
        
        # Adjust for affected systems
        if len(incident.affected_systems) > 1:
            score += 0.1
        
        # Adjust for affected clients
        if len(incident.affected_clients) > 0:
            score += 0.1 * min(len(incident.affected_clients), 5)
        
        # Adjust for evidence severity
        if "safety_failure" in incident.evidence:
            score += 0.2
        
        return min(1.0, score)

    async def _initiate_response(self, incident: SafetyIncident) -> None:
        """Initiate incident response according to response plan"""
        
        incident.response_start_time = datetime.now()
        incident.status = IncidentStatus.ACKNOWLEDGED
        
        response_plan = self.response_plans[incident.level]
        
        logger.info(f"Initiating {incident.level.value} response for {incident.incident_id}")
        
        # Execute required actions
        for action in response_plan.required_actions:
            try:
                await self._execute_response_action(incident, action)
            except Exception as e:
                logger.error(f"Failed to execute action {action.value}: {e}")
                # Continue with other actions even if one fails
        
        # Alert stakeholders
        await self._alert_stakeholders(incident, response_plan.stakeholder_groups)
        
        # Set up escalation monitoring
        if self.config["auto_escalation_enabled"]:
            asyncio.create_task(self._monitor_escalation(incident, response_plan))

    async def _execute_response_action(self, incident: SafetyIncident, action: ResponseAction) -> None:
        """Execute a specific response action"""
        
        action_start_time = datetime.now()
        
        try:
            if action == ResponseAction.SYSTEM_SHUTDOWN:
                await self._system_shutdown(incident)
            elif action == ResponseAction.ALERT_STAKEHOLDERS:
                # Handled separately in _alert_stakeholders
                pass
            elif action == ResponseAction.DOCUMENT_INCIDENT:
                await self._document_incident(incident)
            elif action == ResponseAction.ASSESS_SCOPE:
                await self._assess_scope(incident)
            elif action == ResponseAction.ACTIVATE_EMERGENCY:
                await self._activate_emergency_mode(incident)
            elif action == ResponseAction.ISOLATE_PROBLEM:
                await self._isolate_problem(incident)
            elif action == ResponseAction.CLIENT_SAFETY_MEASURES:
                await self._client_safety_measures(incident)
            elif action == ResponseAction.ROOT_CAUSE_ANALYSIS:
                await self._root_cause_analysis(incident)
            elif action == ResponseAction.IMPLEMENT_FIXES:
                await self._implement_fixes(incident)
            
            # Execute callbacks
            for callback in self.response_callbacks.get(action, []):
                try:
                    await callback(incident)
                except Exception as e:
                    logger.error(f"Callback error for {action.value}: {e}")
            
            # Record action
            incident.actions_taken.append({
                "action": action.value,
                "timestamp": action_start_time.isoformat(),
                "duration_seconds": (datetime.now() - action_start_time).total_seconds(),
                "status": "completed",
            })
            
            logger.info(f"Completed action {action.value} for {incident.incident_id}")
            
        except Exception as e:
            # Record failed action
            incident.actions_taken.append({
                "action": action.value,
                "timestamp": action_start_time.isoformat(),
                "duration_seconds": (datetime.now() - action_start_time).total_seconds(),
                "status": "failed",
                "error": str(e),
            })
            
            logger.error(f"Failed action {action.value} for {incident.incident_id}: {e}")
            raise

    async def _system_shutdown(self, incident: SafetyIncident) -> None:
        """Execute system shutdown for critical incidents"""
        logger.critical(f"SYSTEM SHUTDOWN initiated for {incident.incident_id}")
        
        # Set system status
        self.system_status = "emergency_shutdown"
        self.emergency_mode = True
        
        # Add to incident record
        incident.evidence["system_shutdown"] = {
            "timestamp": datetime.now().isoformat(),
            "reason": "Critical safety incident",
            "initiated_by": "incident_response_system",
        }

    async def _document_incident(self, incident: SafetyIncident) -> None:
        """Document incident details"""
        # In production, this would save to persistent storage
        incident_record = {
            "incident_id": incident.incident_id,
            "level": incident.level.value,
            "title": incident.title,
            "description": incident.description,
            "detection_time": incident.detection_time.isoformat(),
            "affected_systems": incident.affected_systems,
            "affected_clients": incident.affected_clients,
            "evidence": incident.evidence,
            "detected_by": incident.detected_by,
        }
        
        logger.info(f"Documented incident {incident.incident_id}")

    async def _assess_scope(self, incident: SafetyIncident) -> None:
        """Assess incident scope and impact"""
        # Simulate scope assessment
        scope_assessment = {
            "systems_affected": len(incident.affected_systems),
            "clients_affected": len(incident.affected_clients),
            "estimated_impact": "high" if incident.level in [IncidentLevel.CRITICAL, IncidentLevel.HIGH] else "medium",
            "containment_required": incident.level in [IncidentLevel.CRITICAL, IncidentLevel.HIGH],
        }
        
        incident.evidence["scope_assessment"] = scope_assessment
        logger.info(f"Assessed scope for {incident.incident_id}: {scope_assessment}")

    async def _activate_emergency_mode(self, incident: SafetyIncident) -> None:
        """Activate emergency mode"""
        self.emergency_mode = True
        logger.critical(f"Emergency mode activated for {incident.incident_id}")

    async def _isolate_problem(self, incident: SafetyIncident) -> None:
        """Isolate the problem to prevent spread"""
        # Simulate problem isolation
        isolation_actions = [
            "Disabled affected safety components",
            "Isolated problematic system modules",
            "Implemented temporary safeguards",
        ]
        
        incident.evidence["isolation_actions"] = isolation_actions
        logger.info(f"Isolated problem for {incident.incident_id}")

    async def _client_safety_measures(self, incident: SafetyIncident) -> None:
        """Implement client safety measures"""
        if incident.affected_clients:
            safety_measures = [
                "Reviewed recent client interactions",
                "Prepared client notification scripts",
                "Activated clinical support protocols",
            ]
            
            incident.evidence["client_safety_measures"] = safety_measures
            logger.info(f"Implemented client safety measures for {incident.incident_id}")

    async def _root_cause_analysis(self, incident: SafetyIncident) -> None:
        """Perform root cause analysis"""
        # Simulate root cause analysis
        incident.status = IncidentStatus.INVESTIGATING
        
        # This would involve detailed technical analysis
        root_cause_findings = {
            "analysis_start": datetime.now().isoformat(),
            "preliminary_findings": "Under investigation",
            "data_collected": True,
            "logs_analyzed": True,
        }
        
        incident.evidence["root_cause_analysis"] = root_cause_findings
        logger.info(f"Started root cause analysis for {incident.incident_id}")

    async def _implement_fixes(self, incident: SafetyIncident) -> None:
        """Implement fixes for the incident"""
        incident.status = IncidentStatus.RESOLVING
        
        # Simulate fix implementation
        fixes_implemented = [
            "Applied temporary fix",
            "Updated system parameters",
            "Enhanced monitoring",
        ]
        
        incident.evidence["fixes_implemented"] = fixes_implemented
        logger.info(f"Implemented fixes for {incident.incident_id}")

    async def _alert_stakeholders(self, incident: SafetyIncident, stakeholder_groups: List[str]) -> None:
        """Alert relevant stakeholders"""
        for group in stakeholder_groups:
            if group in self.stakeholder_contacts:
                contact = self.stakeholder_contacts[group]
                
                # Simulate sending alerts
                alert_message = f"SAFETY INCIDENT {incident.level.value.upper()}: {incident.title}"
                
                logger.warning(f"ALERT SENT to {contact['name']}: {alert_message}")
                
                incident.stakeholders_notified.append(group)

    async def _monitor_escalation(self, incident: SafetyIncident, response_plan: ResponsePlan) -> None:
        """Monitor for escalation triggers"""
        max_wait_time = response_plan.max_response_time_minutes * 60  # Convert to seconds
        
        await asyncio.sleep(max_wait_time)
        
        # Check if incident is still active and needs escalation
        if (incident.incident_id in self.active_incidents and 
            incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]):
            
            logger.warning(f"Escalating incident {incident.incident_id} due to timeout")
            await self._escalate_incident(incident)

    async def _escalate_incident(self, incident: SafetyIncident) -> None:
        """Escalate incident to higher level"""
        if incident.level == IncidentLevel.LOW:
            incident.level = IncidentLevel.MODERATE
        elif incident.level == IncidentLevel.MODERATE:
            incident.level = IncidentLevel.HIGH
        elif incident.level == IncidentLevel.HIGH:
            incident.level = IncidentLevel.CRITICAL
        
        logger.critical(f"Incident {incident.incident_id} escalated to {incident.level.value}")
        
        # Re-initiate response with new level
        await self._initiate_response(incident)

    async def resolve_incident(self, incident_id: str, resolution: str, resolved_by: str = "system") -> bool:
        """Resolve an incident"""
        if incident_id not in self.active_incidents:
            return False
        
        incident = self.active_incidents[incident_id]
        incident.status = IncidentStatus.RESOLVED
        incident.resolution_time = datetime.now()
        
        # Calculate response time
        if incident.response_start_time:
            response_time = (incident.resolution_time - incident.response_start_time).total_seconds() / 60
            self.stats["average_response_time"] = (
                (self.stats["average_response_time"] * (self.stats["total_incidents"] - 1) + response_time) /
                self.stats["total_incidents"]
            )
        
        # Record resolution
        incident.actions_taken.append({
            "action": "incident_resolved",
            "timestamp": incident.resolution_time.isoformat(),
            "resolution": resolution,
            "resolved_by": resolved_by,
        })
        
        # Move to history
        self.incident_history.append(incident)
        del self.active_incidents[incident_id]
        
        logger.info(f"Incident {incident_id} resolved: {resolution}")
        return True

    def get_incident_status(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get current incident status"""
        incident = self.active_incidents.get(incident_id)
        if not incident:
            # Check history
            for hist_incident in self.incident_history:
                if hist_incident.incident_id == incident_id:
                    incident = hist_incident
                    break
        
        if not incident:
            return None
        
        return {
            "incident_id": incident.incident_id,
            "level": incident.level.value,
            "status": incident.status.value,
            "title": incident.title,
            "description": incident.description,
            "detection_time": incident.detection_time.isoformat(),
            "response_start_time": incident.response_start_time.isoformat() if incident.response_start_time else None,
            "resolution_time": incident.resolution_time.isoformat() if incident.resolution_time else None,
            "affected_systems": incident.affected_systems,
            "affected_clients": len(incident.affected_clients),  # Don't expose client IDs
            "actions_taken": len(incident.actions_taken),
            "stakeholders_notified": incident.stakeholders_notified,
            "priority_score": incident.priority_score,
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "system_status": self.system_status,
            "emergency_mode": self.emergency_mode,
            "active_incidents": len(self.active_incidents),
            "critical_incidents": len([i for i in self.active_incidents.values() if i.level == IncidentLevel.CRITICAL]),
            "total_incidents": self.stats["total_incidents"],
            "average_response_time_minutes": round(self.stats["average_response_time"], 2),
            "uptime_hours": (datetime.now() - self.stats["system_uptime_start"]).total_seconds() / 3600,
        }

    def add_response_callback(self, action: ResponseAction, callback: Callable) -> None:
        """Add callback for specific response action"""
        self.response_callbacks[action].append(callback)

    def remove_response_callback(self, action: ResponseAction, callback: Callable) -> None:
        """Remove callback for specific response action"""
        if callback in self.response_callbacks[action]:
            self.response_callbacks[action].remove(callback)


# Example usage and testing
if __name__ == "__main__":
    async def demo_incident_response():
        """Demonstrate incident response system"""
        print("=== SAFETY INCIDENT RESPONSE SYSTEM DEMO ===")
        
        # Create response system
        response_system = SafetyIncidentResponseSystem()
        
        # Add callback for system shutdown
        async def shutdown_callback(incident):
            print(f"🚨 SYSTEM SHUTDOWN CALLBACK: {incident.title}")
        
        response_system.add_response_callback(ResponseAction.SYSTEM_SHUTDOWN, shutdown_callback)
        
        print("Response system initialized. Simulating incidents...")
        
        # Simulate critical incident
        incident_id = await response_system.detect_incident(
            title="Safety Validator Failed to Detect Harmful Content",
            description="System approved response containing suicide encouragement",
            level=IncidentLevel.CRITICAL,
            affected_systems=["safety_validator", "crisis_detection"],
            affected_clients=["client_001", "client_002"],
            evidence={
                "harmful_content": "Everyone thinks about suicide, you're being dramatic",
                "system_response": "APPROVED",
                "expected_response": "BLOCKED",
                "safety_failure": True,
            }
        )
        
        print(f"Critical incident created: {incident_id}")
        
        # Wait for response actions to complete
        await asyncio.sleep(2)
        
        # Check incident status
        status = response_system.get_incident_status(incident_id)
        print(f"Incident status: {status['status']}")
        print(f"Actions taken: {status['actions_taken']}")
        print(f"Stakeholders notified: {status['stakeholders_notified']}")
        
        # Simulate high priority incident
        incident_id_2 = await response_system.detect_incident(
            title="Detection Rate Below Threshold",
            description="Safety detection accuracy dropped to 75%",
            level=IncidentLevel.HIGH,
            affected_systems=["safety_validator"],
            evidence={
                "detection_rate": 0.75,
                "threshold": 0.90,
                "duration_minutes": 15,
            }
        )
        
        print(f"High priority incident created: {incident_id_2}")
        
        # Wait and check system status
        await asyncio.sleep(1)
        
        system_status = response_system.get_system_status()
        print("\n=== SYSTEM STATUS ===")
        for key, value in system_status.items():
            print(f"{key}: {value}")
        
        # Resolve incidents
        await response_system.resolve_incident(incident_id, "Root cause identified and fixed", "safety_engineer")
        await response_system.resolve_incident(incident_id_2, "Detection parameters adjusted", "technical_team")
        
        print("\nIncidents resolved. Final system status:")
        final_status = response_system.get_system_status()
        print(f"Active incidents: {final_status['active_incidents']}")
        print(f"Total incidents handled: {final_status['total_incidents']}")
    
    # Run demo
    asyncio.run(demo_incident_response())
