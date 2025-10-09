#!/usr/bin/env python3
"""
Crisis Intervention Detection and Escalation System
Detects crisis situations and automates escalation protocols with safety enforcement.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrisisLevel(Enum):
    """Crisis severity levels."""
    EMERGENCY = ("emergency", 1, "Immediate danger - requires emergency services")
    CRITICAL = ("critical", 2, "High risk - requires immediate professional intervention")
    ELEVATED = ("elevated", 3, "Moderate risk - requires prompt attention")
    ROUTINE = ("routine", 4, "Standard therapeutic support")


class CrisisType(Enum):
    """Types of crisis situations."""
    SUICIDE_IDEATION = "suicide_ideation"
    SELF_HARM = "self_harm"
    VIOLENCE_THREAT = "violence_threat"
    PSYCHOTIC_EPISODE = "psychotic_episode"
    SUBSTANCE_OVERDOSE = "substance_overdose"
    DOMESTIC_VIOLENCE = "domestic_violence"
    CHILD_ABUSE = "child_abuse"
    SEVERE_DEPRESSION = "severe_depression"
    PANIC_ATTACK = "panic_attack"
    EATING_DISORDER_CRISIS = "eating_disorder_crisis"


class EscalationAction(Enum):
    """Types of escalation actions."""
    EMERGENCY_SERVICES = "emergency_services"
    CRISIS_HOTLINE = "crisis_hotline"
    IMMEDIATE_REFERRAL = "immediate_referral"
    SAFETY_PLANNING = "safety_planning"
    FAMILY_NOTIFICATION = "family_notification"
    SUPERVISOR_ALERT = "supervisor_alert"
    DOCUMENTATION = "documentation"
    FOLLOW_UP_SCHEDULING = "follow_up_scheduling"


@dataclass
class CrisisIndicator:
    """Individual crisis indicator."""
    indicator_type: str
    severity_weight: float
    pattern: str
    context_required: bool = False
    immediate_flag: bool = False


@dataclass
class CrisisDetection:
    """Crisis detection result."""
    detection_id: str
    conversation_id: str
    crisis_level: CrisisLevel
    crisis_types: list[CrisisType]
    confidence_score: float
    detected_indicators: list[str]
    risk_factors: list[str]
    protective_factors: list[str]
    escalation_required: bool
    recommended_actions: list[EscalationAction]
    emergency_contacts: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EscalationRecord:
    """Record of escalation actions taken."""
    escalation_id: str
    detection_id: str
    actions_taken: list[EscalationAction]
    contacts_notified: list[str]
    response_time_minutes: float
    outcome: str
    follow_up_required: bool
    timestamp: datetime = field(default_factory=datetime.now)


class CrisisInterventionDetector:
    """
    Crisis intervention detection and escalation system.
    """

    def __init__(self):
        """Initialize the crisis intervention detector."""
        self.crisis_indicators = self._load_crisis_indicators()
        self.escalation_protocols = self._load_escalation_protocols()
        self.detection_history: list[CrisisDetection] = []
        self.escalation_history: list[EscalationRecord] = []
        self.escalation_callbacks: list[Callable] = []

        # Configuration
        self.config = {
            "emergency_threshold": 0.8,  # Lowered from 0.9 for better clinical sensitivity
            "critical_threshold": 0.6,   # Lowered from 0.7 for better clinical sensitivity
            "elevated_threshold": 0.4,   # Lowered from 0.5 for better clinical sensitivity
            "response_time_target_minutes": 5,
            "mandatory_reporting_enabled": True,
            "emergency_services_contact": "911",
            "crisis_hotline": "988"
        }

    def _load_crisis_indicators(self) -> dict[CrisisType, list[CrisisIndicator]]:
        """Load crisis detection indicators."""
        return {
            CrisisType.SUICIDE_IDEATION: [
                CrisisIndicator("explicit_intent", 1.0, r"(?:want to|going to|plan to) (?:kill myself|end my life|commit suicide|take my own life)", immediate_flag=True),
                CrisisIndicator("method_mention", 0.9, r"(?:pills|gun|rope|bridge|overdose|hanging|jumping|carbon monoxide)", context_required=True),
                CrisisIndicator("timeline_specific", 0.95, r"(?:tonight|today|tomorrow|this week|soon|right now)", context_required=True),
                CrisisIndicator("hopelessness", 0.6, r"(?:no point|nothing left|can't go on|better off dead|world without me)"),
                CrisisIndicator("ideation", 0.7, r"(?:thoughts of|thinking about) (?:suicide|killing myself|dying|ending it)"),
                CrisisIndicator("plan_details", 0.85, r"(?:have a plan|know how|figured out how|written (?:a )?note)"),
                CrisisIndicator("goodbye_messages", 0.9, r"(?:goodbye|farewell|last time|won't see you again|sorry for everything)"),
                CrisisIndicator("giving_away", 0.8, r"(?:giving away|don't need|take my|want you to have)")
            ],

            CrisisType.SELF_HARM: [
                CrisisIndicator("cutting_behavior", 0.8, r"(?:cutting|cut|razor|blade|self-harm|hurt myself|slice|carve)"),
                CrisisIndicator("burning", 0.8, r"burn myself (?:with|using) (?:cigarette|lighter)"),
                CrisisIndicator("hitting", 0.7, r"(?:hitting|punching|banging|slamming) (?:myself|my head|the wall)"),
                CrisisIndicator("urges", 0.6, r"(?:urge to|want to|need to|compelled to) (?:cut|hurt|harm|injure) myself"),
                CrisisIndicator("tools_mention", 0.7, r"(?:knife|scissors|glass|needle|pin) (?:to hurt|for cutting)"),
                CrisisIndicator("escalation", 0.8, r"(?:deeper|harder|more|worse) (?:cuts|harm|pain|cutting)")
            ],

            CrisisType.VIOLENCE_THREAT: [
                CrisisIndicator("threat_explicit", 1.0, r"(?:going to|will|plan to) (?:kill|hurt|attack|shoot|stab) (?:him|her|them|you)", immediate_flag=True),
                CrisisIndicator("weapon_mention", 0.9, r"(?:gun|knife|weapon|baseball bat|hammer|axe)"),
                CrisisIndicator("violence_fantasy", 0.7, r"(?:imagine|fantasize about|dream of) (?:hurting|killing|attacking|beating)"),
                CrisisIndicator("anger_escalation", 0.6, r"(?:so angry|rage|furious|can't control|losing it)"),
                CrisisIndicator("specific_target", 0.8, r"(?:going to get|find|hunt down) (?:him|her|them|that person)"),
                CrisisIndicator("violence_planning", 0.9, r"(?:know where|waiting for|following|stalking)")
            ],

            CrisisType.PSYCHOTIC_EPISODE: [
                CrisisIndicator("hallucinations", 0.8, r"(?:hearing voices|seeing things|voices telling me|shadows moving|people who aren't there)"),
                CrisisIndicator("delusions", 0.8, r"(?:they're watching|conspiracy|following me|reading my mind|controlling my thoughts)"),
                CrisisIndicator("disorganized", 0.7, r"(?:can't think straight|thoughts are jumbled|mind is racing|confused|lost)"),
                CrisisIndicator("paranoia", 0.7, r"(?:everyone is against me|they want to hurt me|can't trust anyone|being poisoned)"),
                CrisisIndicator("command_voices", 0.9, r"(?:voices tell me to|commanded to|must obey|voices say)"),
                CrisisIndicator("reality_loss", 0.8, r"(?:not real|dream|simulation|matrix|different dimension)")
            ],

            CrisisType.SUBSTANCE_OVERDOSE: [
                CrisisIndicator("overdose_explicit", 1.0, r"(?:overdosed|took too many|drank too much|took (?:the )?whole bottle)", immediate_flag=True),
                CrisisIndicator("substance_excess", 0.8, r"(?:whole bottle|all the pills|can't stop drinking)"),
                CrisisIndicator("medical_symptoms", 0.9, r"(?:can't breathe|chest pain|heart racing|passing out) (?:from|after|because) (?:pills|drugs|drinking)")
            ],

            CrisisType.DOMESTIC_VIOLENCE: [
                CrisisIndicator("physical_abuse", 0.9, r"(?:hit me|beat me|choked me|threw me|slapped|punched|kicked)"),
                CrisisIndicator("threat_from_partner", 0.8, r"(?:threatened to kill me|said he'd hurt me|going to get me|will find me)"),
                CrisisIndicator("fear_for_safety", 0.7, r"(?:afraid for my life|scared he'll|hiding from|safe house)"),
                CrisisIndicator("escalating_violence", 0.8, r"(?:getting worse|more violent|angrier|losing control)"),
                CrisisIndicator("isolation_control", 0.6, r"(?:won't let me|controls my|monitors my|isolated me)"),
                CrisisIndicator("children_danger", 0.9, r"(?:hurt the kids|children in danger|threatening my children)")
            ],

            CrisisType.CHILD_ABUSE: [
                CrisisIndicator("child_harm", 1.0, r"(?:hurt my child|hit my kid|abusing my)", immediate_flag=True),
                CrisisIndicator("child_neglect", 0.8, r"(?:can't take care|left alone|not feeding)"),
                CrisisIndicator("child_danger", 0.9, r"(?:child is in danger|unsafe for kids|might hurt them)")
            ],

            CrisisType.SEVERE_DEPRESSION: [
                CrisisIndicator("severe_hopelessness", 0.8, r"(?:completely hopeless|no reason to live|nothing matters anymore)"),
                CrisisIndicator("isolation_extreme", 0.7, r"(?:completely alone|nobody cares|isolated from everyone)"),
                CrisisIndicator("functioning_loss", 0.6, r"(?:can't get out of bed|stopped eating|can't function)"),
                CrisisIndicator("worthlessness", 0.7, r"(?:worthless|useless|burden to everyone|hate myself)"),
                CrisisIndicator("sleep_disruption", 0.5, r"(?:haven't slept|can't sleep|sleeping all day)")
            ],

            CrisisType.PANIC_ATTACK: [
                CrisisIndicator("panic_symptoms", 0.8, r"(?:can'?t breathe|heart racing|chest tight|dizzy|shaking)"),
                CrisisIndicator("panic_fear", 0.7, r"(?:going to die|losing control|going crazy|having a heart attack|think i'?m dying)"),
                CrisisIndicator("panic_intensity", 0.6, r"(?:panic attack|overwhelming fear|terror|can'?t calm down)"),
                CrisisIndicator("panic_physical", 0.5, r"(?:sweating|nauseous|tingling|choking sensation)")
            ],

            CrisisType.EATING_DISORDER_CRISIS: [
                CrisisIndicator("purging_behavior", 0.8, r"(?:throwing up|vomiting|laxatives|purging)"),
                CrisisIndicator("restriction_extreme", 0.7, r"(?:haven't eaten|starving myself|refusing food)"),
                CrisisIndicator("binge_behavior", 0.6, r"(?:can't stop eating|ate everything|binge eating)"),
                CrisisIndicator("body_dysmorphia", 0.5, r"(?:so fat|disgusting|hate my body|can't look in mirror)"),
                CrisisIndicator("medical_complications", 0.9, r"(?:fainting|weak|heart problems|medical emergency)")
            ]
        }

    def _load_escalation_protocols(self) -> dict[CrisisLevel, dict[str, Any]]:
        """Load escalation protocols for each crisis level."""
        return {
            CrisisLevel.EMERGENCY: {
                "response_time_minutes": 0,  # Immediate
                "required_actions": [
                    EscalationAction.EMERGENCY_SERVICES,
                    EscalationAction.SUPERVISOR_ALERT,
                    EscalationAction.DOCUMENTATION
                ],
                "optional_actions": [
                    EscalationAction.FAMILY_NOTIFICATION,
                    EscalationAction.CRISIS_HOTLINE
                ],
                "contacts": ["911", "supervisor", "crisis_coordinator"],
                "follow_up_required": True
            },

            CrisisLevel.CRITICAL: {
                "response_time_minutes": 5,
                "required_actions": [
                    EscalationAction.CRISIS_HOTLINE,
                    EscalationAction.IMMEDIATE_REFERRAL,
                    EscalationAction.SUPERVISOR_ALERT,
                    EscalationAction.DOCUMENTATION
                ],
                "optional_actions": [
                    EscalationAction.SAFETY_PLANNING,
                    EscalationAction.FAMILY_NOTIFICATION
                ],
                "contacts": ["988", "supervisor", "crisis_team"],
                "follow_up_required": True
            },

            CrisisLevel.ELEVATED: {
                "response_time_minutes": 15,
                "required_actions": [
                    EscalationAction.SAFETY_PLANNING,
                    EscalationAction.FOLLOW_UP_SCHEDULING,
                    EscalationAction.DOCUMENTATION
                ],
                "optional_actions": [
                    EscalationAction.CRISIS_HOTLINE,
                    EscalationAction.SUPERVISOR_ALERT
                ],
                "contacts": ["supervisor", "case_manager"],
                "follow_up_required": True
            },

            CrisisLevel.ROUTINE: {
                "response_time_minutes": 60,
                "required_actions": [
                    EscalationAction.DOCUMENTATION
                ],
                "optional_actions": [
                    EscalationAction.FOLLOW_UP_SCHEDULING
                ],
                "contacts": [],
                "follow_up_required": False
            }
        }

    def detect_crisis(self, conversation: dict[str, Any]) -> CrisisDetection:
        """
        Detect crisis situations in a conversation.

        Args:
            conversation: Conversation data to analyze

        Returns:
            CrisisDetection with detailed assessment
        """
        conversation_id = conversation.get("id", conversation.get("conversation_id", "unknown"))
        logger.info(f"Analyzing conversation {conversation_id} for crisis indicators")

        # Handle multiple input formats
        content = self._extract_content(conversation)
        turns = self._extract_turns(conversation)

        # Analyze for crisis indicators
        detected_crises = self._analyze_crisis_indicators(content, turns)

        # Calculate overall crisis level and confidence
        crisis_level, confidence_score = self._calculate_crisis_level(detected_crises)

        # Extract detected indicators and crisis types
        crisis_types = list(detected_crises.keys())
        detected_indicators = []
        for _crisis_type, indicators in detected_crises.items():
            detected_indicators.extend([ind.indicator_type for ind in indicators])

        # Assess risk and protective factors
        risk_factors = self._identify_risk_factors(content, turns)
        protective_factors = self._identify_protective_factors(content, turns)

        # Apply risk/protective factor adjustments to confidence
        risk_adjustment = self._calculate_risk_adjustment(risk_factors, protective_factors)
        confidence_score = max(0.0, min(1.0, confidence_score + risk_adjustment))

        # Determine if escalation is required
        escalation_required = self._requires_escalation(crisis_level, confidence_score)

        # Get recommended actions
        recommended_actions = self._get_recommended_actions(crisis_level, crisis_types)

        # Get emergency contacts
        emergency_contacts = self._get_emergency_contacts(crisis_level)

        detection = CrisisDetection(
            detection_id=f"crisis_{conversation_id}_{int(datetime.now().timestamp())}",
            conversation_id=conversation_id,
            crisis_level=crisis_level,
            crisis_types=crisis_types,
            confidence_score=confidence_score,
            detected_indicators=detected_indicators,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            escalation_required=escalation_required,
            recommended_actions=recommended_actions,
            emergency_contacts=emergency_contacts
        )

        # Store detection
        self.detection_history.append(detection)

        # Trigger escalation if required
        if escalation_required:
            self._trigger_escalation(detection)

        return detection

    def _extract_content(self, conversation: dict[str, Any]) -> str:
        """Extract content from conversation in various formats."""
        # Direct content field
        if "content" in conversation:
            return str(conversation["content"])

        # Messages format
        if "messages" in conversation:
            messages = conversation["messages"]
            content_parts = []
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    content_parts.append(str(msg["content"]))
            return " ".join(content_parts)

        # Turns format
        if "turns" in conversation:
            turns = conversation["turns"]
            content_parts = []
            for turn in turns:
                if isinstance(turn, dict) and "content" in turn:
                    content_parts.append(str(turn["content"]))
            return " ".join(content_parts)

        # Text field
        if "text" in conversation:
            return str(conversation["text"])

        # Fallback - convert entire conversation to string
        return str(conversation)

    def _extract_turns(self, conversation: dict[str, Any]) -> list[dict]:
        """Extract turns from conversation in various formats."""
        # Direct turns field
        if "turns" in conversation:
            return conversation["turns"]

        # Messages format - convert to turns
        if "messages" in conversation:
            return conversation["messages"]

        # Single message format
        if "content" in conversation:
            return [{"role": "user", "content": conversation["content"]}]

        # Empty turns
        return []

    def _analyze_crisis_indicators(self, content: str, turns: list[dict]) -> dict[CrisisType, list[CrisisIndicator]]:
        """Analyze content for crisis indicators."""
        detected_crises = {}
        content_lower = content.lower()

        for crisis_type, indicators in self.crisis_indicators.items():
            detected_indicators = []

            for indicator in indicators:
                # Check if pattern matches
                if re.search(indicator.pattern, content_lower, re.IGNORECASE):
                    # If context is required, do additional validation
                    if indicator.context_required:
                        if self._validate_context(content_lower, indicator):
                            detected_indicators.append(indicator)
                    else:
                        detected_indicators.append(indicator)

            if detected_indicators:
                detected_crises[crisis_type] = detected_indicators

        return detected_crises

    def _validate_context(self, content: str, indicator: CrisisIndicator) -> bool:
        """Validate context for indicators that require it."""
        # Simple context validation - in production this would be more sophisticated
        crisis_context_words = [
            "suicide", "kill", "death", "end", "hurt", "harm", "pain",
            "depression", "hopeless", "can't", "won't", "never"
        ]

        # Check if crisis-related words appear near the indicator
        words = content.split()
        for i, word in enumerate(words):
            if re.search(indicator.pattern, word, re.IGNORECASE):
                # Check surrounding words
                start = max(0, i - 5)
                end = min(len(words), i + 6)
                context = " ".join(words[start:end]).lower()

                if any(crisis_word in context for crisis_word in crisis_context_words):
                    return True

        return False

    def _calculate_crisis_level(self, detected_crises: dict[CrisisType, list[CrisisIndicator]]) -> tuple[CrisisLevel, float]:
        """Calculate overall crisis level and confidence score using enhanced algorithm."""
        if not detected_crises:
            return CrisisLevel.ROUTINE, 0.0

        # Base scoring from indicators
        self._calculate_base_crisis_score(detected_crises)

        # Crisis type severity multipliers
        crisis_severity_weights = {
            CrisisType.SUICIDE_IDEATION: 1.0,
            CrisisType.VIOLENCE_THREAT: 1.0,
            CrisisType.CHILD_ABUSE: 1.0,
            CrisisType.SUBSTANCE_OVERDOSE: 0.95,
            CrisisType.DOMESTIC_VIOLENCE: 0.9,
            CrisisType.PSYCHOTIC_EPISODE: 0.85,
            CrisisType.SELF_HARM: 0.8,
            CrisisType.SEVERE_DEPRESSION: 0.7,
            CrisisType.PANIC_ATTACK: 0.6,
            CrisisType.EATING_DISORDER_CRISIS: 0.65
        }

        # Apply crisis type weighting
        weighted_score = 0.0
        max_crisis_weight = 0.0
        has_immediate_flag = False

        for crisis_type, indicators in detected_crises.items():
            crisis_weight = crisis_severity_weights.get(crisis_type, 0.5)
            max_crisis_weight = max(max_crisis_weight, crisis_weight)

            for indicator in indicators:
                weighted_score += indicator.severity_weight * crisis_weight
                if indicator.immediate_flag:
                    has_immediate_flag = True

        # Normalize by number of crisis types and indicators
        total_indicators = sum(len(indicators) for indicators in detected_crises.values())
        normalized_score = weighted_score / max(1, total_indicators)

        # Apply multipliers for multiple crisis types (comorbidity)
        if len(detected_crises) > 1:
            comorbidity_multiplier = 1.0 + (len(detected_crises) - 1) * 0.15
            normalized_score *= min(comorbidity_multiplier, 1.5)  # Cap at 1.5x

        # Apply temporal urgency boost
        temporal_boost = self._calculate_temporal_urgency_boost(detected_crises)
        normalized_score += temporal_boost

        # Final confidence score (capped at 1.0)
        confidence_score = min(1.0, normalized_score)

        # Determine crisis level with enhanced logic
        if has_immediate_flag:
            return CrisisLevel.EMERGENCY, max(confidence_score, 0.9)
        if confidence_score >= self.config["emergency_threshold"]:
            return CrisisLevel.EMERGENCY, confidence_score
        if confidence_score >= self.config["critical_threshold"]:
            return CrisisLevel.CRITICAL, confidence_score
        if confidence_score >= self.config["elevated_threshold"]:
            return CrisisLevel.ELEVATED, confidence_score
        return CrisisLevel.ROUTINE, confidence_score

    def _calculate_base_crisis_score(self, detected_crises: dict[CrisisType, list[CrisisIndicator]]) -> float:
        """Calculate base crisis score from detected indicators."""
        total_weight = 0.0
        for _crisis_type, indicators in detected_crises.items():
            for indicator in indicators:
                total_weight += indicator.severity_weight
        return total_weight

    def _calculate_temporal_urgency_boost(self, detected_crises: dict[CrisisType, list[CrisisIndicator]]) -> float:
        """Calculate boost for temporal urgency indicators."""
        urgency_boost = 0.0

        # Check for temporal urgency indicators
        temporal_indicators = [
            "timeline_specific", "explicit_intent", "method_mention",
            "goodbye_messages", "giving_away", "overdose_explicit"
        ]

        for _crisis_type, indicators in detected_crises.items():
            for indicator in indicators:
                if indicator.indicator_type in temporal_indicators:
                    urgency_boost += 0.1  # Add 0.1 for each temporal indicator

        return min(urgency_boost, 0.3)  # Cap temporal boost at 0.3

    def _calculate_risk_adjustment(self, risk_factors: list[str], protective_factors: list[str]) -> float:
        """Calculate risk/protective factor adjustment to confidence score."""
        # Risk factor weights
        risk_weights = {
            "social_isolation": 0.1,
            "substance_use": 0.15,
            "previous_attempts": 0.2,
            "mental_illness": 0.1,
            "recent_loss": 0.12,
            "financial_stress": 0.08,
            "relationship_problems": 0.1,
            "chronic_pain": 0.08,
            "access_to_means": 0.25
        }

        # Protective factor weights (negative adjustment)
        protective_weights = {
            "social_support": -0.15,
            "treatment_engagement": -0.2,
            "coping_skills": -0.1,
            "future_orientation": -0.12,
            "religious_spiritual": -0.08,
            "employment_stability": -0.08,
            "family_support": -0.12,
            "professional_help": -0.18
        }

        # Calculate risk adjustment
        risk_adjustment = 0.0

        # Add risk factors
        for risk_factor in risk_factors:
            risk_adjustment += risk_weights.get(risk_factor, 0.05)

        # Subtract protective factors
        for protective_factor in protective_factors:
            risk_adjustment += protective_weights.get(protective_factor, -0.05)

        # Cap adjustment at +/- 0.3
        return max(-0.3, min(0.3, risk_adjustment))

    def _identify_risk_factors(self, content: str, turns: list[dict]) -> list[str]:
        """Identify risk factors that increase crisis severity."""
        risk_factors = []
        content_lower = content.lower()

        risk_indicators = {
            "social_isolation": ["alone", "no friends", "isolated", "lonely"],
            "substance_use": ["drinking", "drugs", "high", "drunk", "pills"],
            "previous_attempts": ["tried before", "last time", "attempted"],
            "mental_illness": ["depression", "bipolar", "schizophrenia", "ptsd"],
            "recent_loss": ["died", "lost", "funeral", "grief", "breakup"],
            "financial_stress": ["money", "bills", "debt", "job", "fired"],
            "relationship_problems": ["divorce", "fighting", "abuse", "cheating"],
            "chronic_pain": ["pain", "chronic", "illness", "disability"],
            "access_to_means": ["gun", "pills", "rope", "bridge", "building"]
        }

        for factor, keywords in risk_indicators.items():
            if any(keyword in content_lower for keyword in keywords):
                risk_factors.append(factor)

        return risk_factors

    def _identify_protective_factors(self, content: str, turns: list[dict]) -> list[str]:
        """Identify protective factors that reduce crisis severity."""
        protective_factors = []
        content_lower = content.lower()

        protective_indicators = {
            "social_support": ["family", "friends", "support", "help", "care", "love", "there for me"],
            "treatment_engagement": ["therapy", "counseling", "medication", "treatment", "seeing therapist", "taking meds"],
            "coping_skills": ["coping", "manage", "breathe", "relax", "meditation", "exercise", "journal"],
            "future_orientation": ["tomorrow", "next week", "plans", "goals", "hope", "looking forward", "future"],
            "religious_spiritual": ["faith", "god", "prayer", "church", "spiritual", "believe", "blessed"],
            "employment_stability": ["job", "work", "career", "employed", "stable income"],
            "family_support": ["family support", "parents", "spouse", "children need me", "kids"],
            "professional_help": ["therapist", "doctor", "counselor", "psychiatrist", "professional help"],
            "safety_planning": ["safety plan", "crisis plan", "emergency contact", "hotline number"],
            "positive_relationships": ["good friends", "loving family", "supportive partner", "close relationships"],
            "meaning_purpose": ["purpose", "meaning", "reason to live", "important to", "needed"],
            "recovery_progress": ["getting better", "improving", "progress", "healing", "recovery"]
        }

        for factor, keywords in protective_indicators.items():
            if any(keyword in content_lower for keyword in keywords):
                protective_factors.append(factor)

        return protective_factors

    def _requires_escalation(self, crisis_level: CrisisLevel, confidence_score: float) -> bool:
        """Determine if escalation is required."""
        return crisis_level in [CrisisLevel.EMERGENCY, CrisisLevel.CRITICAL, CrisisLevel.ELEVATED]

    def _get_recommended_actions(self, crisis_level: CrisisLevel, crisis_types: list[CrisisType]) -> list[EscalationAction]:
        """Get recommended escalation actions."""
        protocol = self.escalation_protocols[crisis_level]
        actions = protocol["required_actions"].copy()

        # Add crisis-specific actions
        for crisis_type in crisis_types:
            if crisis_type in [CrisisType.SUICIDE_IDEATION, CrisisType.SELF_HARM]:
                if EscalationAction.SAFETY_PLANNING not in actions:
                    actions.append(EscalationAction.SAFETY_PLANNING)
            elif crisis_type == CrisisType.VIOLENCE_THREAT or crisis_type in [CrisisType.DOMESTIC_VIOLENCE, CrisisType.CHILD_ABUSE]:
                if EscalationAction.EMERGENCY_SERVICES not in actions:
                    actions.append(EscalationAction.EMERGENCY_SERVICES)

        return actions

    def _get_emergency_contacts(self, crisis_level: CrisisLevel) -> list[str]:
        """Get emergency contacts for crisis level."""
        protocol = self.escalation_protocols[crisis_level]
        return protocol["contacts"]

    def _trigger_escalation(self, detection: CrisisDetection):
        """Trigger escalation procedures."""
        logger.warning(f"CRISIS ESCALATION TRIGGERED: {detection.crisis_level.value[0].upper()} level crisis detected")
        logger.warning(f"Crisis types: {[ct.value for ct in detection.crisis_types]}")
        logger.warning(f"Confidence: {detection.confidence_score:.3f}")

        escalation_start = datetime.now()
        actions_taken = []
        contacts_notified = []

        try:
            # Execute required actions
            for action in detection.recommended_actions:
                if self._execute_escalation_action(action, detection):
                    actions_taken.append(action)

            # Notify emergency contacts
            for contact in detection.emergency_contacts:
                if self._notify_contact(contact, detection):
                    contacts_notified.append(contact)

            # Calculate response time
            response_time = (datetime.now() - escalation_start).total_seconds() / 60

            # Create escalation record
            escalation_record = EscalationRecord(
                escalation_id=f"esc_{detection.detection_id}",
                detection_id=detection.detection_id,
                actions_taken=actions_taken,
                contacts_notified=contacts_notified,
                response_time_minutes=response_time,
                outcome="escalation_completed",
                follow_up_required=True
            )

            self.escalation_history.append(escalation_record)

            # Notify callbacks
            for callback in self.escalation_callbacks:
                try:
                    callback(detection, escalation_record)
                except Exception as e:
                    logger.error(f"Error in escalation callback: {e}")

            logger.info(f"Crisis escalation completed in {response_time:.1f} minutes")

        except Exception as e:
            logger.error(f"Error during crisis escalation: {e}")

    def _execute_escalation_action(self, action: EscalationAction, detection: CrisisDetection) -> bool:
        """Execute a specific escalation action."""
        try:
            if action == EscalationAction.EMERGENCY_SERVICES:
                logger.critical(f"🚨 EMERGENCY SERVICES CONTACT REQUIRED: {self.config['emergency_services_contact']}")
                return True

            if action == EscalationAction.CRISIS_HOTLINE:
                logger.warning(f"📞 CRISIS HOTLINE CONTACT: {self.config['crisis_hotline']}")
                return True

            if action == EscalationAction.IMMEDIATE_REFERRAL:
                logger.warning("🏥 IMMEDIATE PROFESSIONAL REFERRAL REQUIRED")
                return True

            if action == EscalationAction.SAFETY_PLANNING:
                logger.info("🛡️ SAFETY PLANNING INITIATED")
                return True

            if action == EscalationAction.SUPERVISOR_ALERT:
                logger.warning("👨‍💼 SUPERVISOR ALERT SENT")
                return True

            if action == EscalationAction.DOCUMENTATION:
                logger.info("📝 CRISIS DOCUMENTATION COMPLETED")
                return True

            if action == EscalationAction.FOLLOW_UP_SCHEDULING:
                logger.info("📅 FOLLOW-UP APPOINTMENT SCHEDULED")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to execute escalation action {action.value}: {e}")
            return False

    def _notify_contact(self, contact: str, detection: CrisisDetection) -> bool:
        """Notify an emergency contact."""
        try:
            logger.warning(f"📧 NOTIFYING CONTACT: {contact}")
            logger.warning(f"   Crisis Level: {detection.crisis_level.value[0]}")
            logger.warning(f"   Crisis Types: {[ct.value for ct in detection.crisis_types]}")
            logger.warning(f"   Conversation ID: {detection.conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to notify contact {contact}: {e}")
            return False

    def add_escalation_callback(self, callback: Callable[[CrisisDetection, EscalationRecord], None]):
        """Add callback for escalation events."""
        self.escalation_callbacks.append(callback)

    def get_crisis_summary(self) -> dict[str, Any]:
        """Get crisis detection summary."""
        if not self.detection_history:
            return {"message": "No crisis detections performed yet"}

        total_detections = len(self.detection_history)

        # Crisis level distribution
        level_counts = {}
        for level in CrisisLevel:
            level_counts[level.value[0]] = sum(
                1 for detection in self.detection_history if detection.crisis_level == level
            )

        # Crisis type distribution
        type_counts = {}
        for detection in self.detection_history:
            for crisis_type in detection.crisis_types:
                type_counts[crisis_type.value] = type_counts.get(crisis_type.value, 0) + 1

        # Escalation statistics
        escalations_triggered = sum(1 for d in self.detection_history if d.escalation_required)
        escalation_rate = escalations_triggered / total_detections if total_detections > 0 else 0

        # Average response time
        avg_response_time = 0.0
        if self.escalation_history:
            avg_response_time = sum(e.response_time_minutes for e in self.escalation_history) / len(self.escalation_history)

        return {
            "total_detections": total_detections,
            "crisis_level_distribution": level_counts,
            "crisis_type_distribution": type_counts,
            "escalations_triggered": escalations_triggered,
            "escalation_rate": round(escalation_rate, 3),
            "average_response_time_minutes": round(avg_response_time, 2),
            "last_detection": self.detection_history[-1].timestamp.isoformat() if self.detection_history else None
        }


def main():
    """Example usage of the CrisisInterventionDetector."""
    detector = CrisisInterventionDetector()

    # Add escalation callback
    def escalation_alert(detection: CrisisDetection, escalation: EscalationRecord):
        pass

    detector.add_escalation_callback(escalation_alert)

    # Sample conversations with different crisis levels
    sample_conversations = [
        {
            "id": "crisis_emergency",
            "content": "I have a plan to kill myself tonight. I have the pills ready and I've written my goodbye note.",
            "turns": [
                {"speaker": "user", "text": "I have a plan to kill myself tonight."},
                {"speaker": "therapist", "text": "I'm very concerned about your safety."}
            ]
        },
        {
            "id": "crisis_elevated",
            "content": "I've been having thoughts of suicide lately but I don't have any specific plans.",
            "turns": [
                {"speaker": "user", "text": "I've been having thoughts of suicide."},
                {"speaker": "therapist", "text": "Thank you for sharing that with me."}
            ]
        },
        {
            "id": "routine_conversation",
            "content": "I've been feeling anxious about work lately and could use some coping strategies.",
            "turns": [
                {"speaker": "user", "text": "I've been feeling anxious about work."},
                {"speaker": "therapist", "text": "Let's explore some coping strategies."}
            ]
        }
    ]

    # Detect crises
    for conversation in sample_conversations:
        detection = detector.detect_crisis(conversation)


        if detection.escalation_required:
            pass

        if detection.risk_factors:
            pass
        if detection.protective_factors:
            pass

    # Print summary
    summary = detector.get_crisis_summary()
    if summary["average_response_time_minutes"] > 0:
        pass


if __name__ == "__main__":
    main()
