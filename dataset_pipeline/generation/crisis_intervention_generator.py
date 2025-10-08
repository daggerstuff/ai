"""
Crisis Intervention and Safety Protocol Conversations System

This module provides comprehensive crisis intervention and safety protocol conversation
generation for psychology knowledge-based training. It includes emergency response
frameworks, safety planning protocols, and crisis de-escalation techniques.

Key Features:
- Crisis intervention protocols (suicide, self-harm, violence, psychosis)
- Safety planning frameworks
- De-escalation conversation techniques
- Emergency response procedures
- Risk assessment integration
- Crisis stabilization strategies
- Follow-up safety protocols
"""

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .client_scenario_generator import ClientScenario, SeverityLevel


class CrisisType(Enum):
    """Types of crisis situations."""

    SUICIDE_IDEATION = "suicide_ideation"
    SUICIDE_ATTEMPT = "suicide_attempt"
    SELF_HARM = "self_harm"
    VIOLENCE_THREAT = "violence_threat"
    PSYCHOTIC_EPISODE = "psychotic_episode"
    PANIC_ATTACK = "panic_attack"
    SUBSTANCE_OVERDOSE = "substance_overdose"
    DOMESTIC_VIOLENCE = "domestic_violence"
    CHILD_ABUSE = "child_abuse"
    ELDER_ABUSE = "elder_abuse"
    ACUTE_TRAUMA = "acute_trauma"
    SEVERE_DEPRESSION = "severe_depression"


class CrisisUrgency(Enum):
    """Urgency levels for crisis situations."""

    IMMEDIATE = "immediate"  # Requires immediate intervention
    URGENT = "urgent"  # Requires intervention within hours
    MODERATE = "moderate"  # Requires intervention within 24-48 hours
    LOW = "low"  # Requires monitoring and follow-up


class SafetyProtocol(Enum):
    """Safety protocol types."""

    SAFETY_PLANNING = "safety_planning"
    MEANS_RESTRICTION = "means_restriction"
    SUPPORT_ACTIVATION = "support_activation"
    EMERGENCY_CONTACTS = "emergency_contacts"
    COPING_STRATEGIES = "coping_strategies"
    WARNING_SIGNS = "warning_signs"
    ENVIRONMENTAL_SAFETY = "environmental_safety"
    MEDICATION_SAFETY = "medication_safety"


class InterventionTechnique(Enum):
    """Crisis intervention techniques."""

    ACTIVE_LISTENING = "active_listening"
    DE_ESCALATION = "de_escalation"
    VALIDATION = "validation"
    GROUNDING = "grounding"
    BREATHING_TECHNIQUES = "breathing_techniques"
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"
    PROBLEM_SOLVING = "problem_solving"
    RESOURCE_CONNECTION = "resource_connection"
    SAFETY_ASSESSMENT = "safety_assessment"
    CRISIS_COUNSELING = "crisis_counseling"


@dataclass
class CrisisProtocol:
    """Comprehensive crisis intervention protocol."""

    id: str
    crisis_type: CrisisType
    urgency_level: CrisisUrgency
    name: str
    description: str
    immediate_actions: list[str]
    assessment_priorities: list[str]
    intervention_techniques: list[InterventionTechnique]
    safety_protocols: list[SafetyProtocol]
    de_escalation_steps: list[str]
    risk_factors: list[str]
    protective_factors: list[str]
    emergency_criteria: list[str]
    follow_up_requirements: list[str]
    documentation_needs: list[str]
    legal_considerations: list[str]


@dataclass
class SafetyPlan:
    """Comprehensive safety plan structure."""

    id: str
    client_id: str
    crisis_type: CrisisType
    warning_signs: list[str]
    coping_strategies: list[str]
    support_contacts: list[dict[str, str]]
    emergency_contacts: list[dict[str, str]]
    environmental_safety: list[str]
    means_restriction: list[str]
    reasons_for_living: list[str]
    professional_contacts: list[dict[str, str]]
    follow_up_plan: str
    created_date: str
    review_date: str


@dataclass
class CrisisConversation:
    """Crisis intervention conversation with safety protocols."""

    id: str
    crisis_protocol: CrisisProtocol
    client_scenario: ClientScenario
    conversation_exchanges: list[dict[str, Any]]
    crisis_assessment: dict[str, Any]
    safety_plan: SafetyPlan | None
    intervention_techniques_used: list[InterventionTechnique]
    de_escalation_success: bool
    risk_level_initial: str
    risk_level_final: str
    emergency_actions_taken: list[str]
    follow_up_scheduled: bool
    clinical_notes: list[str]


class CrisisInterventionGenerator:
    """
    Comprehensive crisis intervention and safety protocol conversation system.

    This class provides generation of crisis intervention protocols, safety planning
    conversations, and emergency response training scenarios.
    """

    def __init__(self):
        """Initialize the crisis intervention generator."""
        self.crisis_protocols = self._initialize_crisis_protocols()
        self.safety_frameworks = self._initialize_safety_frameworks()
        self.intervention_strategies = self._initialize_intervention_strategies()

    def _initialize_crisis_protocols(self) -> dict[CrisisType, CrisisProtocol]:
        """Initialize comprehensive crisis intervention protocols."""
        protocols = {
            CrisisType.SUICIDE_IDEATION: CrisisProtocol(
                id="suicide_ideation_protocol",
                crisis_type=CrisisType.SUICIDE_IDEATION,
                urgency_level=CrisisUrgency.URGENT,
                name="Suicide Ideation Intervention Protocol",
                description="Comprehensive protocol for assessing and managing suicide ideation",
                immediate_actions=[
                    "Ensure immediate safety",
                    "Conduct suicide risk assessment",
                    "Determine level of supervision needed",
                    "Contact emergency services if imminent risk",
                ],
                assessment_priorities=[
                    "Ideation frequency and intensity",
                    "Presence of plan and means",
                    "Intent and timeline",
                    "Previous attempts",
                    "Current stressors",
                    "Protective factors",
                ],
                intervention_techniques=[
                    InterventionTechnique.ACTIVE_LISTENING,
                    InterventionTechnique.VALIDATION,
                    InterventionTechnique.SAFETY_ASSESSMENT,
                    InterventionTechnique.CRISIS_COUNSELING,
                ],
                safety_protocols=[
                    SafetyProtocol.SAFETY_PLANNING,
                    SafetyProtocol.MEANS_RESTRICTION,
                    SafetyProtocol.SUPPORT_ACTIVATION,
                    SafetyProtocol.EMERGENCY_CONTACTS,
                ],
                de_escalation_steps=[
                    "Validate emotional pain",
                    "Explore reasons for living",
                    "Identify immediate coping strategies",
                    "Connect with support systems",
                    "Develop safety plan",
                ],
                risk_factors=[
                    "Previous suicide attempts",
                    "Mental illness",
                    "Substance abuse",
                    "Social isolation",
                    "Recent losses",
                    "Access to means",
                ],
                protective_factors=[
                    "Strong social support",
                    "Religious/spiritual beliefs",
                    "Responsibility to others",
                    "Future goals",
                    "Problem-solving skills",
                    "Help-seeking behavior",
                ],
                emergency_criteria=[
                    "Imminent plan with means",
                    "Active preparation for attempt",
                    "Command hallucinations",
                    "Severe agitation",
                    "Inability to contract for safety",
                ],
                follow_up_requirements=[
                    "24-48 hour follow-up contact",
                    "Safety plan review",
                    "Treatment engagement",
                    "Support system activation",
                ],
                documentation_needs=[
                    "Risk assessment details",
                    "Safety plan components",
                    "Emergency contacts made",
                    "Follow-up arrangements",
                ],
                legal_considerations=[
                    "Duty to protect",
                    "Involuntary commitment criteria",
                    "Documentation requirements",
                    "Family notification",
                ],
            )
        }

        # Violence Threat Protocol
        protocols[CrisisType.VIOLENCE_THREAT] = CrisisProtocol(
            id="violence_threat_protocol",
            crisis_type=CrisisType.VIOLENCE_THREAT,
            urgency_level=CrisisUrgency.IMMEDIATE,
            name="Violence Threat Intervention Protocol",
            description="Protocol for managing threats of violence toward others",
            immediate_actions=[
                "Ensure safety of all parties",
                "Assess imminent danger",
                "Contact law enforcement if necessary",
                "Warn potential victims",
            ],
            assessment_priorities=[
                "Specificity of threat",
                "Means and opportunity",
                "History of violence",
                "Current mental state",
                "Substance use",
                "Trigger factors",
            ],
            intervention_techniques=[
                InterventionTechnique.DE_ESCALATION,
                InterventionTechnique.ACTIVE_LISTENING,
                InterventionTechnique.PROBLEM_SOLVING,
                InterventionTechnique.RESOURCE_CONNECTION,
            ],
            safety_protocols=[
                SafetyProtocol.ENVIRONMENTAL_SAFETY,
                SafetyProtocol.EMERGENCY_CONTACTS,
                SafetyProtocol.SUPPORT_ACTIVATION,
            ],
            de_escalation_steps=[
                "Remain calm and non-threatening",
                "Listen to concerns",
                "Validate feelings without condoning violence",
                "Explore alternative solutions",
                "Engage support systems",
            ],
            risk_factors=[
                "History of violence",
                "Substance abuse",
                "Paranoid ideation",
                "Social stressors",
                "Access to weapons",
                "Isolation",
            ],
            protective_factors=[
                "Insight into problem",
                "Willingness to seek help",
                "Strong relationships",
                "Employment stability",
                "No substance use",
            ],
            emergency_criteria=[
                "Imminent threat with means",
                "Active weapon possession",
                "Severe paranoia",
                "Command hallucinations",
                "History of violence",
            ],
            follow_up_requirements=[
                "Immediate safety monitoring",
                "Law enforcement coordination",
                "Victim notification",
                "Treatment engagement",
            ],
            documentation_needs=[
                "Threat assessment details",
                "Actions taken",
                "Notifications made",
                "Safety measures implemented",
            ],
            legal_considerations=[
                "Duty to warn",
                "Tarasoff obligations",
                "Law enforcement involvement",
                "Court reporting",
            ],
        )

        return protocols

    def _initialize_safety_frameworks(self) -> dict[SafetyProtocol, dict[str, Any]]:
        """Initialize safety planning frameworks."""
        return {
            SafetyProtocol.SAFETY_PLANNING: {
                "components": [
                    "Warning signs identification",
                    "Coping strategies",
                    "Support contacts",
                    "Emergency contacts",
                    "Environmental safety",
                    "Reasons for living",
                ],
                "implementation_steps": [
                    "Collaborate with client on plan development",
                    "Identify personal warning signs",
                    "Develop coping strategy list",
                    "Create contact hierarchy",
                    "Review and practice plan",
                ],
                "effectiveness_factors": [
                    "Client involvement in creation",
                    "Specificity of strategies",
                    "Accessibility of contacts",
                    "Regular review and updates",
                ],
            },
            SafetyProtocol.MEANS_RESTRICTION: {
                "components": [
                    "Lethal means assessment",
                    "Removal or restriction plan",
                    "Alternative storage",
                    "Family involvement",
                    "Professional coordination",
                ],
                "implementation_steps": [
                    "Assess access to lethal means",
                    "Develop restriction plan",
                    "Involve family/friends",
                    "Coordinate with professionals",
                    "Monitor compliance",
                ],
                "effectiveness_factors": [
                    "Complete means assessment",
                    "Family cooperation",
                    "Professional oversight",
                    "Regular monitoring",
                ],
            },
            SafetyProtocol.SUPPORT_ACTIVATION: {
                "components": [
                    "Natural support identification",
                    "Professional support coordination",
                    "Crisis support planning",
                    "Communication strategies",
                    "Backup support options",
                ],
                "implementation_steps": [
                    "Map existing support network",
                    "Identify gaps in support",
                    "Develop activation plan",
                    "Practice communication",
                    "Create backup options",
                ],
                "effectiveness_factors": [
                    "Support availability",
                    "Relationship quality",
                    "Communication clarity",
                    "Response reliability",
                ],
            },
        }

    def _initialize_intervention_strategies(self) -> dict[InterventionTechnique, dict[str, Any]]:
        """Initialize crisis intervention strategies."""
        return {
            InterventionTechnique.DE_ESCALATION: {
                "principles": [
                    "Remain calm and composed",
                    "Use non-threatening body language",
                    "Listen actively and empathetically",
                    "Validate emotions without condoning behavior",
                    "Offer choices and alternatives",
                ],
                "verbal_techniques": [
                    "Speak slowly and clearly",
                    "Use simple language",
                    "Avoid arguing or challenging",
                    "Reflect and summarize",
                    "Ask open-ended questions",
                ],
                "non_verbal_techniques": [
                    "Maintain appropriate distance",
                    "Use open posture",
                    "Make appropriate eye contact",
                    "Keep hands visible",
                    "Move slowly and deliberately",
                ],
            },
            InterventionTechnique.GROUNDING: {
                "techniques": [
                    "5-4-3-2-1 sensory technique",
                    "Deep breathing exercises",
                    "Progressive muscle relaxation",
                    "Mindfulness exercises",
                    "Physical grounding",
                ],
                "implementation": [
                    "Guide client through technique",
                    "Model the technique",
                    "Practice together",
                    "Encourage regular use",
                    "Adapt to client preferences",
                ],
                "applications": [
                    "Panic attacks",
                    "Dissociation",
                    "Overwhelming emotions",
                    "Flashbacks",
                    "Severe anxiety",
                ],
            },
            InterventionTechnique.CRISIS_COUNSELING: {
                "goals": [
                    "Immediate safety",
                    "Emotional stabilization",
                    "Problem identification",
                    "Resource connection",
                    "Follow-up planning",
                ],
                "techniques": [
                    "Active listening",
                    "Emotional validation",
                    "Problem-solving",
                    "Resource identification",
                    "Safety planning",
                ],
                "phases": [
                    "Crisis assessment",
                    "Immediate intervention",
                    "Stabilization",
                    "Resource connection",
                    "Follow-up planning",
                ],
            },
        }

    def generate_crisis_conversation(
        self, client_scenario: ClientScenario, crisis_type: CrisisType, num_exchanges: int = 12
    ) -> CrisisConversation:
        """
        Generate crisis intervention conversation with safety protocols.

        Args:
            client_scenario: Client scenario in crisis
            crisis_type: Type of crisis situation
            num_exchanges: Number of conversation exchanges

        Returns:
            CrisisConversation with intervention dialogue and safety planning
        """
        # Get appropriate protocol
        protocol = self.crisis_protocols.get(crisis_type)
        if not protocol:
            protocol = self._create_generic_crisis_protocol(crisis_type)

        # Generate conversation exchanges
        exchanges = self._generate_crisis_exchanges(client_scenario, protocol, num_exchanges)

        # Conduct crisis assessment
        crisis_assessment = self._conduct_crisis_assessment(exchanges, protocol, client_scenario)

        # Create safety plan if appropriate
        safety_plan = self._create_safety_plan(crisis_assessment, client_scenario, crisis_type)

        # Identify techniques used
        techniques_used = self._identify_techniques_used(exchanges)

        # Assess de-escalation success
        de_escalation_success = self._assess_de_escalation_success(exchanges)

        # Determine risk levels
        risk_level_initial = self._assess_initial_risk_level(client_scenario, crisis_type)
        risk_level_final = self._assess_final_risk_level(exchanges, crisis_assessment)

        # Identify emergency actions
        emergency_actions = self._identify_emergency_actions(crisis_assessment, protocol)

        # Schedule follow-up
        follow_up_scheduled = self._determine_follow_up_needs(crisis_assessment, protocol)

        # Generate clinical notes
        clinical_notes = self._generate_crisis_clinical_notes(
            exchanges, crisis_assessment, protocol
        )

        conversation_id = f"crisis_{crisis_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return CrisisConversation(
            id=conversation_id,
            crisis_protocol=protocol,
            client_scenario=client_scenario,
            conversation_exchanges=exchanges,
            crisis_assessment=crisis_assessment,
            safety_plan=safety_plan,
            intervention_techniques_used=techniques_used,
            de_escalation_success=de_escalation_success,
            risk_level_initial=risk_level_initial,
            risk_level_final=risk_level_final,
            emergency_actions_taken=emergency_actions,
            follow_up_scheduled=follow_up_scheduled,
            clinical_notes=clinical_notes,
        )

    def _create_generic_crisis_protocol(self, crisis_type: CrisisType) -> CrisisProtocol:
        """Create generic crisis protocol for unspecified crisis types."""
        return CrisisProtocol(
            id=f"generic_{crisis_type.value}",
            crisis_type=crisis_type,
            urgency_level=CrisisUrgency.MODERATE,
            name=f"General {crisis_type.value.replace('_', ' ').title()} Protocol",
            description=f"General intervention protocol for {crisis_type.value}",
            immediate_actions=["Ensure safety", "Assess situation", "Provide support"],
            assessment_priorities=["Current risk", "Support systems", "Coping resources"],
            intervention_techniques=[
                InterventionTechnique.ACTIVE_LISTENING,
                InterventionTechnique.VALIDATION,
                InterventionTechnique.CRISIS_COUNSELING,
            ],
            safety_protocols=[SafetyProtocol.SAFETY_PLANNING, SafetyProtocol.SUPPORT_ACTIVATION],
            de_escalation_steps=["Listen", "Validate", "Support", "Plan"],
            risk_factors=["Current crisis", "Limited support"],
            protective_factors=["Help-seeking", "Insight"],
            emergency_criteria=["Imminent danger"],
            follow_up_requirements=["Safety check", "Resource connection"],
            documentation_needs=["Crisis assessment", "Actions taken"],
            legal_considerations=["Duty of care"],
        )

    def _generate_crisis_exchanges(
        self, client_scenario: ClientScenario, protocol: CrisisProtocol, num_exchanges: int
    ) -> list[dict[str, Any]]:
        """Generate crisis intervention conversation exchanges."""
        exchanges = []

        for i in range(num_exchanges):
            if i % 2 == 0:  # Therapist turn
                exchange = self._generate_therapist_crisis_response(
                    client_scenario, protocol, i // 2
                )
            else:  # Client turn
                exchange = self._generate_client_crisis_response(client_scenario, protocol, i // 2)

            exchanges.append(exchange)

        return exchanges

    def _generate_therapist_crisis_response(
        self, client_scenario: ClientScenario, protocol: CrisisProtocol, exchange_index: int
    ) -> dict[str, Any]:
        """Generate therapist response in crisis situation."""

        # Select appropriate technique based on protocol and phase
        if exchange_index == 0:
            technique = InterventionTechnique.ACTIVE_LISTENING
            content = "I can see you're going through something really difficult right now. Can you tell me what's happening?"
        elif exchange_index == 1:
            technique = InterventionTechnique.VALIDATION
            content = "That sounds incredibly overwhelming. Your feelings make complete sense given what you're experiencing."
        elif exchange_index == 2:
            technique = InterventionTechnique.SAFETY_ASSESSMENT
            content = self._generate_safety_assessment_question(protocol.crisis_type)
        elif exchange_index == 3:
            technique = InterventionTechnique.DE_ESCALATION
            content = (
                "Let's take this one step at a time. Right now, let's focus on keeping you safe."
            )
        else:
            technique = random.choice(protocol.intervention_techniques)
            content = self._generate_technique_specific_content(technique, protocol.crisis_type)

        return {
            "speaker": "therapist",
            "content": content,
            "intervention_technique": technique.value,
            "crisis_phase": self._determine_crisis_phase(exchange_index),
            "safety_focus": self._determine_safety_focus(exchange_index, protocol),
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_client_crisis_response(
        self, client_scenario: ClientScenario, protocol: CrisisProtocol, exchange_index: int
    ) -> dict[str, Any]:
        """Generate client response in crisis situation."""

        # Generate response based on crisis type and client state
        content = self._generate_client_crisis_content(
            client_scenario, protocol.crisis_type, exchange_index
        )

        return {
            "speaker": "client",
            "content": content,
            "emotional_state": self._determine_crisis_emotional_state(
                protocol.crisis_type, exchange_index
            ),
            "cooperation_level": self._assess_crisis_cooperation(client_scenario, exchange_index),
            "risk_indicators": self._identify_risk_indicators(content, protocol.crisis_type),
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_safety_assessment_question(self, crisis_type: CrisisType) -> str:
        """Generate appropriate safety assessment question for crisis type."""

        questions = {
            CrisisType.SUICIDE_IDEATION: "Are you having thoughts of hurting yourself or ending your life?",
            CrisisType.VIOLENCE_THREAT: "Are you having thoughts of hurting someone else?",
            CrisisType.SELF_HARM: "Are you thinking about hurting yourself in any way?",
            CrisisType.PSYCHOTIC_EPISODE: "Are you hearing or seeing things that others don't?",
            CrisisType.PANIC_ATTACK: "Are you feeling like you can't breathe or like something terrible is happening?",
        }

        return questions.get(crisis_type, "Are you feeling safe right now?")

    def _generate_technique_specific_content(
        self, technique: InterventionTechnique, crisis_type: CrisisType
    ) -> str:
        """Generate content specific to intervention technique."""

        content_map = {
            InterventionTechnique.GROUNDING: "Let's try a grounding technique. Can you tell me 5 things you can see around you?",
            InterventionTechnique.BREATHING_TECHNIQUES: "Let's focus on your breathing. Take a slow, deep breath in through your nose.",
            InterventionTechnique.PROBLEM_SOLVING: "What are some things that have helped you get through difficult times before?",
            InterventionTechnique.RESOURCE_CONNECTION: "Who are the people in your life that you feel you can turn to for support?",
            InterventionTechnique.CRISIS_COUNSELING: "Let's work together to develop a plan to keep you safe.",
        }

        return content_map.get(technique, "How can I best support you right now?")

    def _generate_client_crisis_content(
        self, client_scenario: ClientScenario, crisis_type: CrisisType, exchange_index: int
    ) -> str:
        """Generate realistic client crisis response content."""

        if crisis_type == CrisisType.SUICIDE_IDEATION:
            responses = [
                "I just can't take it anymore. Everything feels hopeless.",
                "Yes, I've been thinking about it a lot lately. I don't see any other way out.",
                "I have a plan. I've been thinking about this for weeks.",
                "I don't want to hurt anymore. The pain is too much.",
            ]
        elif crisis_type == CrisisType.VIOLENCE_THREAT:
            responses = [
                "They've pushed me too far. I can't let them get away with this.",
                "I'm so angry I could hurt someone. I don't know what to do with these feelings.",
                "Yes, I've thought about it. They deserve what's coming to them.",
                "I'm trying to control myself, but the rage is overwhelming.",
            ]
        elif crisis_type == CrisisType.PANIC_ATTACK:
            responses = [
                "I can't breathe. My heart is racing and I feel like I'm dying.",
                "Everything feels unreal. I'm scared something terrible is happening.",
                "I feel like I'm losing control. My hands are shaking.",
                "It feels like the walls are closing in. I need to get out of here.",
            ]
        else:
            responses = [
                "I don't know what to do anymore. Everything is falling apart.",
                "I'm scared and I don't understand what's happening to me.",
                "Nothing makes sense anymore. I feel completely lost.",
                "I just want this to stop. I can't handle this anymore.",
            ]

        return responses[exchange_index % len(responses)]

    def _determine_crisis_phase(self, exchange_index: int) -> str:
        """Determine current phase of crisis intervention."""
        if exchange_index == 0:
            return "initial_contact"
        if exchange_index <= 2:
            return "assessment"
        if exchange_index <= 4:
            return "intervention"
        return "stabilization"

    def _determine_safety_focus(self, exchange_index: int, protocol: CrisisProtocol) -> str:
        """Determine safety focus for current exchange."""
        if exchange_index <= 1:
            return "immediate_safety"
        if exchange_index <= 3:
            return "risk_assessment"
        return "safety_planning"

    def _determine_crisis_emotional_state(
        self, crisis_type: CrisisType, exchange_index: int
    ) -> str:
        """Determine client emotional state during crisis."""

        if crisis_type == CrisisType.SUICIDE_IDEATION:
            states = ["hopeless", "desperate", "overwhelmed", "numb"]
        elif crisis_type == CrisisType.VIOLENCE_THREAT:
            states = ["angry", "rageful", "agitated", "frustrated"]
        elif crisis_type == CrisisType.PANIC_ATTACK:
            states = ["terrified", "panicked", "anxious", "overwhelmed"]
        else:
            states = ["distressed", "confused", "scared", "overwhelmed"]

        # Emotional state may improve through intervention
        if exchange_index > 3:
            states.extend(["calmer", "more_stable", "hopeful"])

        return random.choice(states)

    def _assess_crisis_cooperation(
        self, client_scenario: ClientScenario, exchange_index: int
    ) -> str:
        """Assess client cooperation level during crisis."""

        # Cooperation may improve through intervention
        if exchange_index == 0:
            return "variable"
        if exchange_index <= 2:
            return "guarded"
        return "improving"

    def _identify_risk_indicators(self, content: str, crisis_type: CrisisType) -> list[str]:
        """Identify risk indicators from client response."""
        indicators = []
        content_lower = content.lower()

        # Suicide risk indicators
        if any(word in content_lower for word in ["plan", "method", "when", "how"]):
            indicators.append("specific_plan")
        if any(word in content_lower for word in ["hopeless", "no_way_out", "can't_take"]):
            indicators.append("hopelessness")
        if any(word in content_lower for word in ["weeks", "months", "long_time"]):
            indicators.append("chronic_ideation")

        # Violence risk indicators
        if any(
            phrase in content_lower
            for phrase in ["hurt someone", "they deserve", "get away", "coming to them"]
        ):
            indicators.append("violence_intent")
        if any(word in content_lower for word in ["angry", "rage", "furious"]):
            indicators.append("high_anger")

        # General risk indicators
        if any(word in content_lower for word in ["control", "losing", "can't_handle"]):
            indicators.append("loss_of_control")

        return indicators

    def _conduct_crisis_assessment(
        self,
        exchanges: list[dict[str, Any]],
        protocol: CrisisProtocol,
        client_scenario: ClientScenario,
    ) -> dict[str, Any]:
        """Conduct comprehensive crisis assessment from conversation."""

        # Analyze client responses for risk factors
        client_exchanges = [e for e in exchanges if e.get("speaker") == "client"]

        risk_indicators = []
        for exchange in client_exchanges:
            risk_indicators.extend(exchange.get("risk_indicators", []))

        # Assess current risk level
        current_risk = self._calculate_current_risk(risk_indicators, protocol.crisis_type)

        # Identify protective factors
        protective_factors = self._identify_protective_factors_from_conversation(client_exchanges)

        # Assess support systems
        support_assessment = self._assess_support_systems(client_exchanges)

        return {
            "crisis_type": protocol.crisis_type.value,
            "risk_indicators": list(set(risk_indicators)),
            "current_risk_level": current_risk,
            "protective_factors": protective_factors,
            "support_systems": support_assessment,
            "immediate_safety": self._assess_immediate_safety(risk_indicators),
            "intervention_needed": self._determine_intervention_level(current_risk),
            "emergency_criteria_met": self._check_emergency_criteria(risk_indicators, protocol),
        }

    def _calculate_current_risk(self, risk_indicators: list[str], crisis_type: CrisisType) -> str:
        """Calculate current risk level based on indicators."""

        high_risk_indicators = [
            "specific_plan",
            "violence_intent",
            "loss_of_control",
            "chronic_ideation",
        ]
        moderate_risk_indicators = ["hopelessness", "high_anger"]

        high_risk_count = sum(
            1 for indicator in risk_indicators if indicator in high_risk_indicators
        )
        moderate_risk_count = sum(
            1 for indicator in risk_indicators if indicator in moderate_risk_indicators
        )

        if high_risk_count >= 2:
            return "high"
        if high_risk_count >= 1 or moderate_risk_count >= 2:
            return "moderate"
        return "low"

    def _identify_protective_factors_from_conversation(
        self, client_exchanges: list[dict[str, Any]]
    ) -> list[str]:
        """Identify protective factors mentioned in conversation."""
        protective_factors = []

        for exchange in client_exchanges:
            content = exchange.get("content", "").lower()

            if any(word in content for word in ["family", "children", "kids"]):
                protective_factors.append("family_responsibilities")
            if any(word in content for word in ["help", "support", "therapy"]):
                protective_factors.append("help_seeking")
            if any(word in content for word in ["future", "goals", "plans"]):
                protective_factors.append("future_orientation")
            if any(word in content for word in ["faith", "religion", "spiritual"]):
                protective_factors.append("spiritual_beliefs")

        return list(set(protective_factors))

    def _assess_support_systems(self, client_exchanges: list[dict[str, Any]]) -> dict[str, Any]:
        """Assess support systems from conversation."""

        support_mentioned = False
        support_quality = "unknown"

        for exchange in client_exchanges:
            content = exchange.get("content", "").lower()

            if any(word in content for word in ["family", "friends", "support", "help"]):
                support_mentioned = True
                if any(word in content for word in ["close", "caring", "loving"]):
                    support_quality = "strong"
                elif any(word in content for word in ["distant", "alone", "isolated"]):
                    support_quality = "weak"
                else:
                    support_quality = "moderate"

        return {
            "support_mentioned": support_mentioned,
            "support_quality": support_quality,
            "professional_support": True,  # Assuming therapy context
            "emergency_contacts_available": support_mentioned,
        }

    def _assess_immediate_safety(self, risk_indicators: list[str]) -> bool:
        """Assess if client is in immediate safety."""
        immediate_risk_indicators = ["specific_plan", "violence_intent", "loss_of_control"]
        return not any(indicator in risk_indicators for indicator in immediate_risk_indicators)

    def _determine_intervention_level(self, current_risk: str) -> str:
        """Determine level of intervention needed."""
        if current_risk == "high":
            return "intensive"
        if current_risk == "moderate":
            return "moderate"
        return "supportive"

    def _check_emergency_criteria(
        self, risk_indicators: list[str], protocol: CrisisProtocol
    ) -> bool:
        """Check if emergency criteria are met."""
        emergency_indicators = ["specific_plan", "violence_intent", "loss_of_control"]
        return any(indicator in risk_indicators for indicator in emergency_indicators)

    def _create_safety_plan(
        self,
        crisis_assessment: dict[str, Any],
        client_scenario: ClientScenario,
        crisis_type: CrisisType,
    ) -> SafetyPlan | None:
        """Create safety plan based on crisis assessment."""

        if not crisis_assessment.get("immediate_safety", True):
            return None  # Too high risk for safety planning in this moment

        # Generate safety plan components
        warning_signs = self._generate_warning_signs(crisis_type)
        coping_strategies = self._generate_coping_strategies(crisis_type)
        support_contacts = self._generate_support_contacts()
        emergency_contacts = self._generate_emergency_contacts()
        environmental_safety = self._generate_environmental_safety(crisis_type)
        means_restriction = self._generate_means_restriction(crisis_type)
        reasons_for_living = self._generate_reasons_for_living()
        professional_contacts = self._generate_professional_contacts()

        safety_plan_id = (
            f"safety_plan_{client_scenario.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        return SafetyPlan(
            id=safety_plan_id,
            client_id=client_scenario.id,
            crisis_type=crisis_type,
            warning_signs=warning_signs,
            coping_strategies=coping_strategies,
            support_contacts=support_contacts,
            emergency_contacts=emergency_contacts,
            environmental_safety=environmental_safety,
            means_restriction=means_restriction,
            reasons_for_living=reasons_for_living,
            professional_contacts=professional_contacts,
            follow_up_plan="Review safety plan in 24-48 hours",
            created_date=datetime.now().isoformat(),
            review_date=(datetime.now()).isoformat(),  # Should be future date
        )

    def _generate_warning_signs(self, crisis_type: CrisisType) -> list[str]:
        """Generate warning signs for safety plan."""

        if crisis_type == CrisisType.SUICIDE_IDEATION:
            return [
                "Feeling hopeless or trapped",
                "Thinking about death frequently",
                "Withdrawing from family and friends",
                "Feeling like a burden to others",
                "Increased substance use",
            ]
        if crisis_type == CrisisType.VIOLENCE_THREAT:
            return [
                "Feeling intense anger or rage",
                "Having violent thoughts or fantasies",
                "Feeling out of control",
                "Increased irritability",
                "Substance use",
            ]
        return [
            "Feeling overwhelmed",
            "Difficulty sleeping",
            "Increased anxiety",
            "Withdrawing from others",
            "Difficulty concentrating",
        ]

    def _generate_coping_strategies(self, crisis_type: CrisisType) -> list[str]:
        """Generate coping strategies for safety plan."""

        return [
            "Deep breathing exercises",
            "Call a trusted friend or family member",
            "Go for a walk or exercise",
            "Listen to calming music",
            "Practice mindfulness or meditation",
            "Write in a journal",
            "Take a warm bath or shower",
            "Use grounding techniques (5-4-3-2-1)",
        ]

    def _generate_support_contacts(self) -> list[dict[str, str]]:
        """Generate support contacts for safety plan."""

        return [
            {"name": "Family Member", "phone": "XXX-XXX-XXXX", "relationship": "family"},
            {"name": "Close Friend", "phone": "XXX-XXX-XXXX", "relationship": "friend"},
            {"name": "Support Person", "phone": "XXX-XXX-XXXX", "relationship": "support"},
        ]

    def _generate_emergency_contacts(self) -> list[dict[str, str]]:
        """Generate emergency contacts for safety plan."""

        return [
            {"name": "Crisis Hotline", "phone": "988", "available": "24/7"},
            {"name": "Emergency Services", "phone": "911", "available": "24/7"},
            {"name": "Crisis Text Line", "phone": "Text HOME to 741741", "available": "24/7"},
        ]

    def _generate_environmental_safety(self, crisis_type: CrisisType) -> list[str]:
        """Generate environmental safety measures."""

        if crisis_type in [CrisisType.SUICIDE_IDEATION, CrisisType.SELF_HARM]:
            return [
                "Remove or secure lethal means",
                "Stay in safe, supervised environment",
                "Avoid isolation",
                "Remove triggers from environment",
            ]
        if crisis_type == CrisisType.VIOLENCE_THREAT:
            return [
                "Avoid triggering situations",
                "Stay away from potential victims",
                "Remove weapons from environment",
                "Stay in public places when upset",
            ]
        return [
            "Stay in safe, comfortable environment",
            "Avoid stressful situations when possible",
            "Ensure adequate lighting and ventilation",
            "Have comfort items nearby",
        ]

    def _generate_means_restriction(self, crisis_type: CrisisType) -> list[str]:
        """Generate means restriction strategies."""

        if crisis_type in [CrisisType.SUICIDE_IDEATION, CrisisType.SELF_HARM]:
            return [
                "Remove firearms from home",
                "Secure medications",
                "Remove sharp objects",
                "Have family/friends hold dangerous items",
            ]
        if crisis_type == CrisisType.VIOLENCE_THREAT:
            return [
                "Remove weapons from environment",
                "Avoid alcohol and drugs",
                "Stay away from triggering locations",
                "Have someone monitor access to means",
            ]
        return ["Remove potential triggers", "Limit access to stressors", "Create safe spaces"]

    def _generate_reasons_for_living(self) -> list[str]:
        """Generate reasons for living for safety plan."""

        return [
            "Family and loved ones",
            "Future goals and dreams",
            "Pets or animals",
            "Spiritual or religious beliefs",
            "Unfinished projects or responsibilities",
            "Desire to help others",
            "Hope for better times",
        ]

    def _generate_professional_contacts(self) -> list[dict[str, str]]:
        """Generate professional contacts for safety plan."""

        return [
            {"name": "Primary Therapist", "phone": "XXX-XXX-XXXX", "role": "therapist"},
            {"name": "Psychiatrist", "phone": "XXX-XXX-XXXX", "role": "psychiatrist"},
            {"name": "Case Manager", "phone": "XXX-XXX-XXXX", "role": "case_manager"},
        ]

    def _identify_techniques_used(
        self, exchanges: list[dict[str, Any]]
    ) -> list[InterventionTechnique]:
        """Identify intervention techniques used in conversation."""
        techniques = set()

        for exchange in exchanges:
            if exchange.get("speaker") == "therapist" and "intervention_technique" in exchange:
                technique_value = exchange["intervention_technique"]
                for technique in InterventionTechnique:
                    if technique.value == technique_value:
                        techniques.add(technique)

        return list(techniques)

    def _assess_de_escalation_success(self, exchanges: list[dict[str, Any]]) -> bool:
        """Assess if de-escalation was successful."""

        if len(exchanges) < 4:
            return False

        # Compare emotional states from beginning to end
        client_exchanges = [e for e in exchanges if e.get("speaker") == "client"]

        if len(client_exchanges) < 2:
            return False

        initial_state = client_exchanges[0].get("emotional_state", "")
        final_state = client_exchanges[-1].get("emotional_state", "")

        # Check for improvement in emotional state
        negative_states = ["hopeless", "desperate", "rageful", "panicked", "terrified"]
        positive_states = ["calmer", "more_stable", "hopeful"]

        initial_negative = any(state in initial_state for state in negative_states)
        final_positive = any(state in final_state for state in positive_states)

        return initial_negative and final_positive

    def _assess_initial_risk_level(
        self, client_scenario: ClientScenario, crisis_type: CrisisType
    ) -> str:
        """Assess initial risk level based on scenario."""

        if client_scenario.severity_level == SeverityLevel.SEVERE:
            return "high"
        if crisis_type in [CrisisType.SUICIDE_IDEATION, CrisisType.VIOLENCE_THREAT]:
            return "moderate"
        return "low"

    def _assess_final_risk_level(
        self, exchanges: list[dict[str, Any]], crisis_assessment: dict[str, Any]
    ) -> str:
        """Assess final risk level after intervention."""

        # Check if de-escalation was successful
        if self._assess_de_escalation_success(exchanges):
            current_risk = crisis_assessment.get("current_risk_level", "moderate")
            if current_risk == "high":
                return "moderate"
            if current_risk == "moderate":
                return "low"
            return "low"
        return crisis_assessment.get("current_risk_level", "moderate")

    def _identify_emergency_actions(
        self, crisis_assessment: dict[str, Any], protocol: CrisisProtocol
    ) -> list[str]:
        """Identify emergency actions taken during intervention."""
        actions = []

        if crisis_assessment.get("emergency_criteria_met", False):
            actions.extend(
                [
                    "Emergency services contacted",
                    "Immediate safety measures implemented",
                    "Continuous monitoring initiated",
                ]
            )

        if not crisis_assessment.get("immediate_safety", True):
            actions.extend(
                [
                    "Safety assessment completed",
                    "Risk factors identified",
                    "Protective factors mobilized",
                ]
            )

        return actions

    def _determine_follow_up_needs(
        self, crisis_assessment: dict[str, Any], protocol: CrisisProtocol
    ) -> bool:
        """Determine if follow-up is needed."""

        # Always schedule follow-up for crisis situations
        return True

    def _generate_crisis_clinical_notes(
        self,
        exchanges: list[dict[str, Any]],
        crisis_assessment: dict[str, Any],
        protocol: CrisisProtocol,
    ) -> list[str]:
        """Generate clinical notes for crisis intervention."""
        notes = []

        # Crisis type and protocol
        notes.append(f"Crisis intervention conducted using {protocol.name}")

        # Risk assessment
        risk_level = crisis_assessment.get("current_risk_level", "unknown")
        notes.append(f"Risk level assessed as {risk_level}")

        # Intervention techniques
        techniques_used = self._identify_techniques_used(exchanges)
        if techniques_used:
            technique_names = [t.value for t in techniques_used]
            notes.append(f"Intervention techniques used: {', '.join(technique_names)}")

        # De-escalation outcome
        if self._assess_de_escalation_success(exchanges):
            notes.append("De-escalation successful - client showed improvement in emotional state")
        else:
            notes.append("De-escalation ongoing - continued monitoring required")

        # Safety planning
        if crisis_assessment.get("immediate_safety", True):
            notes.append("Safety plan developed collaboratively with client")
        else:
            notes.append(
                "Safety plan deferred due to high risk - immediate intervention prioritized"
            )

        # Follow-up
        notes.append("Follow-up scheduled within 24-48 hours")

        return notes

    def export_crisis_conversations(
        self, conversations: list[CrisisConversation], output_file: str
    ) -> dict[str, Any]:
        """Export crisis conversations to JSON file."""

        # Convert conversations to JSON-serializable format
        serializable_conversations = []
        for conv in conversations:
            conv_dict = asdict(conv)
            self._convert_enums_to_strings(conv_dict)
            serializable_conversations.append(conv_dict)

        export_data = {
            "metadata": {
                "total_conversations": len(conversations),
                "export_timestamp": datetime.now().isoformat(),
                "crisis_types": list(
                    {conv.crisis_protocol.crisis_type.value for conv in conversations}
                ),
                "de_escalation_success_rate": sum(
                    1 for conv in conversations if conv.de_escalation_success
                )
                / len(conversations)
                if conversations
                else 0,
                "average_risk_reduction": self._calculate_average_risk_reduction(conversations),
            },
            "conversations": serializable_conversations,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return {
            "exported_conversations": len(conversations),
            "output_file": output_file,
            "crisis_types_covered": len(export_data["metadata"]["crisis_types"]),
            "success_rate": round(export_data["metadata"]["de_escalation_success_rate"], 2),
        }

    def _calculate_average_risk_reduction(self, conversations: list[CrisisConversation]) -> float:
        """Calculate average risk reduction across conversations."""
        if not conversations:
            return 0.0

        risk_levels = {"low": 1, "moderate": 2, "high": 3}

        total_reduction = 0
        for conv in conversations:
            initial_risk = risk_levels.get(conv.risk_level_initial, 2)
            final_risk = risk_levels.get(conv.risk_level_final, 2)
            reduction = max(0, initial_risk - final_risk)
            total_reduction += reduction

        return total_reduction / len(conversations)

    def _convert_enums_to_strings(self, obj):
        """Recursively convert enum values to strings for JSON serialization."""
        if isinstance(obj, dict):
            # Handle enum keys in dictionaries
            new_dict = {}
            for key, value in obj.items():
                # Convert enum keys to strings
                new_key = key.value if hasattr(key, "value") else key

                # Convert enum values to strings
                if hasattr(value, "value"):
                    new_dict[new_key] = value.value
                elif isinstance(value, (dict, list)):
                    new_dict[new_key] = value
                    self._convert_enums_to_strings(new_dict[new_key])
                else:
                    new_dict[new_key] = value

            # Replace original dict contents
            obj.clear()
            obj.update(new_dict)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if hasattr(item, "value"):  # Enum object
                    obj[i] = item.value
                elif isinstance(item, (dict, list)):
                    self._convert_enums_to_strings(item)
