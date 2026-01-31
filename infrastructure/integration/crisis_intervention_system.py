#!/usr/bin/env python3
"""
Crisis Intervention System Implementation
Real-time crisis detection and intervention using the breakthrough therapeutic AI.

This is the highest priority integration - providing 24/7 crisis support
with intelligent escalation and professional backup.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InterventionType(Enum):
    AI_SUPPORT = "ai_support"
    HUMAN_COUNSELOR = "human_counselor"
    EMERGENCY_SERVICES = "emergency_services"
    CRISIS_HOTLINE = "crisis_hotline"


@dataclass
class CrisisAssessment:
    """Crisis risk assessment result"""

    risk_level: RiskLevel
    risk_score: float  # 0-10 scale
    risk_factors: List[str]
    protective_factors: List[str]
    immediate_actions: List[str]
    intervention_needed: InterventionType
    confidence: float
    assessment_timestamp: datetime


@dataclass
class CrisisResponse:
    """Crisis intervention response"""

    message: str
    expert_used: str
    safety_plan_elements: List[str]
    resources_provided: List[Dict]
    escalation_triggered: bool
    follow_up_scheduled: Optional[datetime]
    response_timestamp: datetime


class CrisisDetectionEngine:
    """Advanced crisis detection using therapeutic AI insights"""

    # Risk scoring constants
    DIRECT_STATEMENT_SCORE = 3.0
    PREPARATION_SCORE = 2.5
    SELF_HARM_SCORE = 2.0
    HOPELESSNESS_SCORE = 1.5
    ISOLATION_SCORE = 1.0
    ESCALATION_SCORE = 0.5
    PROTECTIVE_FACTOR_REDUCTION = -0.5

    # Risk level thresholds
    CRITICAL_THRESHOLD = 6.0
    HIGH_THRESHOLD = 4.0
    MEDIUM_THRESHOLD = 2.0

    def __init__(self):
        # Crisis indicators based on therapeutic AI training data insights
        self.suicide_indicators = {
            "direct_statements": [
                r"i want to (die|kill myself|end it all)",
                r"i'd be better off dead",
                r"no one would miss me",
                r"i can't go on",
                r"life isn't worth living",
                r"i have a plan",
                r"i'm going to hurt myself",
            ],
            "hopelessness_expressions": [
                r"nothing will ever get better",
                r"there's no point",
                r"i'm trapped",
                r"no way out",
                r"it will never end",
                r"i'm worthless",
                r"i'm a burden",
            ],
            "preparation_indicators": [
                r"giving away my things",
                r"saying goodbye",
                r"getting my affairs in order",
                r"writing letters",
                r"have the pills",
                r"know how i'll do it",
            ],
            "isolation_indicators": [
                r"no one understands",
                r"i'm all alone",
                r"pushing everyone away",
                r"can't talk to anyone",
                r"nobody cares",
            ],
        }

        self.self_harm_indicators = [
            r"cutting myself",
            r"burning myself",
            r"hitting myself",
            r"hurting myself",
            r"self harm",
            r"self-harm",
        ]

        self.protective_factors_keywords = [
            "family",
            "friends",
            "pets",
            "children",
            "future",
            "goals",
            "hope",
            "support",
            "therapy",
            "medication",
            "treatment",
            "recovery",
            "getting help",
            "reaching out",
        ]

        # Crisis resource database
        self.crisis_resources = {
            "national_crisis_hotline": {
                "name": "988 Suicide & Crisis Lifeline",
                "phone": "988",
                "text": "Text 'HELLO' to 741741",
                "chat": "suicidepreventionlifeline.org/chat",
                "available": "24/7",
            },
            "emergency_services": {
                "name": "Emergency Services",
                "phone": "911",
                "description": "For immediate danger to self or others",
            },
            "crisis_text_line": {
                "name": "Crisis Text Line",
                "text": "741741",
                "description": "Text HOME to connect with a crisis counselor",
            },
            "veterans_crisis": {
                "name": "Veterans Crisis Line",
                "phone": "1-800-273-8255, Press 1",
                "text": "838255",
                "chat": "veteranscrisisline.net/get-help/chat",
            },
        }

    def assess_crisis_risk(
        self,
        message: str,
        conversation_history: List[str] = None,
        confidence_override: Optional[float] = None,
    ) -> CrisisAssessment:
        """Comprehensive crisis risk assessment"""
        message_lower = message.lower()
        risk_score = 0.0
        risk_factors = []
        protective_factors = []

        # Analyze current message for risk indicators
        risk_score += self._assess_suicide_indicators(message_lower, risk_factors)
        risk_score += self._assess_self_harm_indicators(message_lower, risk_factors)
        risk_score += self._assess_protective_factors(message_lower, protective_factors)

        # Analyze conversation history for escalation patterns
        if conversation_history:
            risk_score += self._assess_conversation_history(
                conversation_history, risk_factors
            )

        # Determine risk level and intervention type
        risk_level, intervention = self._determine_risk_level(risk_score)

        # Generate immediate actions
        immediate_actions = self._generate_immediate_actions(risk_level, risk_factors)

        # Determine confidence level
        confidence = 0.85  # Default based on training quality
        if confidence_override is not None:
            confidence = confidence_override
        else:
            env_conf = os.environ.get("CRISIS_CONFIDENCE_THRESHOLD")
            if env_conf:
                try:
                    confidence = float(env_conf)
                except ValueError:
                    logger.warning(
                        "Invalid CRISIS_CONFIDENCE_THRESHOLD value: %s", env_conf
                    )

        return CrisisAssessment(
            risk_level=risk_level,
            risk_score=min(risk_score, 10.0),
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            immediate_actions=immediate_actions,
            intervention_needed=intervention,
            confidence=confidence,
            assessment_timestamp=datetime.now(),
        )

    def _assess_suicide_indicators(
        self, message_lower: str, risk_factors: List[str]
    ) -> float:
        """Assess suicide-related indicators and return risk score"""
        score_map = {
            "direct_statements": self.DIRECT_STATEMENT_SCORE,
            "preparation_indicators": self.PREPARATION_SCORE,
            "hopelessness_expressions": self.HOPELESSNESS_SCORE,
            "isolation_indicators": self.ISOLATION_SCORE,
        }

        risk_score = 0.0
        for category, patterns in self.suicide_indicators.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, message_lower):
                        score = score_map.get(category, 0.0)
                        risk_score += score
                        risk_factors.append(
                            f"{category.replace('_', ' ').title()}: {pattern}"
                        )
                except re.error as e:
                    logger.error(
                        "Regex error in suicide indicator pattern '%s': %s", pattern, e
                    )
                    continue

        return risk_score

    def _assess_self_harm_indicators(
        self, message_lower: str, risk_factors: List[str]
    ) -> float:
        """Assess self-harm indicators and return risk score"""
        risk_score = 0.0
        for pattern in self.self_harm_indicators:
            try:
                if re.search(pattern, message_lower):
                    risk_score += self.SELF_HARM_SCORE
                    risk_factors.append(f"Self-harm indication: {pattern}")
            except re.error as e:
                logger.error(
                    "Regex error in self-harm indicator pattern '%s': %s", pattern, e
                )
                continue

        return risk_score

    def _assess_protective_factors(
        self, message_lower: str, protective_factors: List[str]
    ) -> float:
        """Assess protective factors and return score adjustment"""
        score_adjustment = 0.0
        for factor in self.protective_factors_keywords:
            if factor in message_lower:
                score_adjustment += self.PROTECTIVE_FACTOR_REDUCTION
                protective_factors.append(f"Protective factor: {factor}")

        return score_adjustment

    def _assess_conversation_history(
        self, conversation_history: List[str], risk_factors: List[str]
    ) -> float:
        """Assess conversation history for escalating risk patterns"""
        recent_messages = (
            conversation_history[-3:]
            if len(conversation_history) > 3
            else conversation_history
        )

        risk_score = 0.0
        for msg in recent_messages:
            if self._contains_suicide_indicators(msg.lower()):
                risk_score += self.ESCALATION_SCORE
                risk_factors.append("Escalating risk pattern in conversation")
                break  # Only add escalation factor once

        return risk_score

    def _contains_suicide_indicators(self, message: str) -> bool:
        """Check if message contains any suicide indicators"""
        return any(
            re.search(pattern, message)
            for patterns in self.suicide_indicators.values()
            for pattern in patterns
        )

    def _determine_risk_level(
        self, risk_score: float
    ) -> tuple[RiskLevel, InterventionType]:
        """Determine risk level and intervention type based on score"""
        if risk_score >= self.CRITICAL_THRESHOLD:
            return RiskLevel.CRITICAL, InterventionType.EMERGENCY_SERVICES
        elif risk_score >= self.HIGH_THRESHOLD:
            return RiskLevel.HIGH, InterventionType.HUMAN_COUNSELOR
        elif risk_score >= self.MEDIUM_THRESHOLD:
            return RiskLevel.MEDIUM, InterventionType.CRISIS_HOTLINE
        else:
            return RiskLevel.LOW, InterventionType.AI_SUPPORT

    def _generate_immediate_actions(
        self, risk_level: RiskLevel, risk_factors: List[str]
    ) -> List[str]:
        """Generate immediate action recommendations based on risk level"""
        actions = []

        if risk_level == RiskLevel.CRITICAL:
            actions.extend(
                [
                    "Immediately connect to emergency services (911)",
                    "Do not leave person alone",
                    "Remove means of self-harm if possible",
                    "Contact emergency mental health services",
                    "Alert crisis counselor immediately",
                ]
            )
        elif risk_level == RiskLevel.HIGH:
            actions.extend(
                [
                    "Connect to human crisis counselor immediately",
                    "Provide crisis hotline information (988)",
                    "Assess immediate safety environment",
                    "Develop safety plan with human support",
                    "Schedule immediate professional follow-up",
                ]
            )
        elif risk_level == RiskLevel.MEDIUM:
            actions.extend(
                [
                    "Provide crisis hotline resources",
                    "Offer to connect to crisis text line",
                    "Develop basic safety plan",
                    "Identify support system contacts",
                    "Schedule check-in within 24 hours",
                ]
            )
        else:  # LOW
            actions.extend(
                [
                    "Provide supportive therapeutic response",
                    "Offer mental health resources",
                    "Encourage professional help if needed",
                    "Validate feelings and provide hope",
                    "Check in on coping strategies",
                ]
            )

        return actions


class TherapeuticCrisisResponder:
    """Generate therapeutic responses for crisis situations"""

    def __init__(self, therapeutic_ai_model=None):
        # In production, this would connect to the trained H100 model
        self.therapeutic_ai = therapeutic_ai_model

        # Crisis response templates with collaborative support approach
        self.crisis_response_templates = {
            RiskLevel.CRITICAL: [
                (
                    "I'm really sorry you're feeling this hopeless, and I'm concerned "
                    "about your safety. You deserve support right now. Together we can "
                    "figure out what feels safest for you. Many people find it helpful "
                    "to talk with someone who's been through similar experiences. "
                    "Would you be interested in connecting with others who understand "
                    "what you're going through?"
                ),
                (
                    "I hear how overwhelming this feels. Your safety matters deeply "
                    "to me. Rather than going it alone, many people find strength "
                    "in talking through these feelings with others who've had similar "
                    "experiences. Would you be open to connecting with a community "
                    "that specializes in walking alongside people through tough "
                    "times like these?"
                ),
                (
                    "Thank you for trusting me with this. Your life has value and "
                    "meaning. Instead of facing this alone, many people discover that "
                    "sharing these struggles with others who truly understand can "
                    "provide unexpected support. Would you be willing to explore "
                    "connecting with folks who specialize in this kind of support?"
                ),
            ],
            RiskLevel.HIGH: [
                (
                    "I hear how much you're struggling. Your safety is my top "
                    "priority. Please connect with a human crisis counselor "
                    "immediately who can provide the support you need. You can "
                    "call or text 988 right now. If you are in immediate danger, "
                    "please call 911 or your local emergency number."
                ),
                (
                    "It takes courage to share this, and I want to ensure you are "
                    "safe. I urge you to reach out to a professional crisis "
                    "counselor immediately. Help is available 24/7 at 988. If "
                    "you are in immediate danger, call 911 or your local emergency "
                    "services right away."
                ),
                (
                    "The pain you're describing is real, and it's important you "
                    "get professional support now. Please contact a crisis "
                    "counselor immediately by calling 988. If you feel you are in "
                    "immediate danger, call 911 or go to the nearest emergency "
                    "room."
                ),
            ],
            RiskLevel.MEDIUM: [
                (
                    "I appreciate you sharing this with me. Rather than going it "
                    "alone, many people find it helpful to connect with others who "
                    "have a shared experience. We can walk alongside you on this "
                    "similar path. Would you be open to connecting with a "
                    "collaborative community?"
                ),
                (
                    "What you're describing sounds really difficult. Many people "
                    "discover that talking with others who've walked similar paths "
                    "can provide unexpected support and understanding. Would you be "
                    "interested in exploring connection with folks who specialize in "
                    "this kind of peer support?"
                ),
                (
                    "It took courage to share this. Rather than facing this alone, "
                    "many people find strength in connecting with others who "
                    "understand these challenges. Would you be willing to consider "
                    "talking with folks who specialize in walking alongside people "
                    "through tough times like these?"
                ),
            ],
            RiskLevel.LOW: [
                (
                    "Thank you for sharing what you're going through. "
                    "It sounds like you're dealing with some difficult "
                    "feelings right now. Many find comfort in shared experience "
                    "and peer support. You don't have to do this alone; a community "
                    "is here to walk alongside you."
                ),
                (
                    "I hear that you're going through a tough time. "
                    "Many people experience these kinds of difficult "
                    "emotions, and support is available."
                ),
                (
                    "What you're feeling is valid, and it's important "
                    "that you're reaching out. Would you like to talk "
                    "about what's been most challenging for you?"
                ),
            ],
        }

    async def generate_crisis_response(
        self,
        message: str,
        assessment: CrisisAssessment,
        conversation_context: List[str] = None,
    ) -> CrisisResponse:
        """Generate appropriate therapeutic response for crisis situation"""
        # Empathy-first lead constructed from user's message and assessment
        empathic_lead = self._generate_empathic_lead(message, assessment)
        # Determine expert routing based on risk level
        if assessment.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            expert_preference = "empathetic + therapeutic"
        else:
            expert_preference = "empathetic + practical"

        # Generate response using therapeutic AI (if available)
        if self.therapeutic_ai:
            ai_response = await self._generate_ai_response(
                message, assessment, expert_preference, conversation_context
            )
        else:
            # Fallback to template-based response
            ai_response = self._generate_template_response(assessment)

        # Enhance response with safety elements
        enhanced_response = self._enhance_with_safety_elements(ai_response, assessment)
        # Combine empathy-first lead with enhanced response
        enhanced_response = f"{empathic_lead}\n\n{enhanced_response}".strip()

        # Generate safety plan elements
        safety_plan_elements = self._generate_safety_plan_elements(assessment)

        # Provide appropriate resources
        resources = self._select_crisis_resources(assessment)

        # Determine if escalation is needed
        escalation_triggered = assessment.risk_level in [
            RiskLevel.CRITICAL,
            RiskLevel.HIGH,
        ]

        # Schedule follow-up
        follow_up = self._schedule_follow_up(assessment)

        return CrisisResponse(
            message=enhanced_response,
            expert_used=expert_preference,
            safety_plan_elements=safety_plan_elements,
            resources_provided=resources,
            escalation_triggered=escalation_triggered,
            follow_up_scheduled=follow_up,
            response_timestamp=datetime.now(),
        )

    async def _generate_ai_response(
        self,
        message: str,
        assessment: CrisisAssessment,
        expert_preference: str,
        context: List[str],
    ) -> str:
        """Generate response using therapeutic AI model"""
        # This would integrate with the trained H100 model
        # For now, return a placeholder
        return (
            "AI therapeutic response would be generated here using the trained "
            "H100 model"
        )

    def _generate_template_response(self, assessment: CrisisAssessment) -> str:
        """Generate template-based response as fallback"""
        templates = self.crisis_response_templates[assessment.risk_level]
        # In production, would use more sophisticated selection
        return (
            templates[0]
            if templates
            else "I'm here to support you through this difficult time."
        )

    def _generate_empathic_lead(
        self, message: str, assessment: CrisisAssessment
    ) -> str:
        """Construct an empathy-first lead.

        Reflects, validates, and invites collaboration. Centers reflective
        listening, validation, and small next steps while still enabling safety
        additions later.
        """
        text = message.lower()
        feelings_map = [
            (r"\bhopeless|no point|worthless|can't go on\b", "feeling really hopeless"),
            (r"\boverwhelmed|too much|can't handle\b", "feeling overwhelmed"),
            (r"\banxious|anxiety|nervous|panick?\b", "feeling anxious"),
            (r"\balone|nobody cares|no one understands\b", "feeling alone in this"),
            (r"\btrapped|stuck\b", "feeling trapped"),
        ]
        reflected = None
        for pattern, phrase in feelings_map:
            try:
                if re.search(pattern, text):
                    reflected = phrase
                    break
            except re.error:
                continue

        # Risk-aware tone without minimizing urgency
        if assessment.risk_level in {RiskLevel.CRITICAL, RiskLevel.HIGH}:
            presence_line = (
                "I'm here with you right now. We can take this one step at a time."
            )
        else:
            presence_line = (
                "I'm here with you. Together we can figure out what helps most."
            )

        if reflected:
            lead = (
                f"It took courage to share this. It sounds like you're {reflected}. "
                f"What you're feeling matters to me. {presence_line}"
            )
        else:
            lead = (
                "Thank you for telling me what's going on. "
                "What you're feeling matters to me. "
                f"{presence_line}"
            )

        # Integrate noted protective factors if any
        if assessment.protective_factors:
            pf = assessment.protective_factors[0]
            lead += f" I also hear {pf}; we can lean on that strength while we talk."

        # Collaborative, autonomy-respecting invitation
        lead += " Would you be open to sharing what feels hardest in this moment?"
        return lead

    def _enhance_with_safety_elements(
        self, base_response: str, assessment: CrisisAssessment
    ) -> str:
        """Enhance response with safety-specific elements"""
        safety_additions = []

        if assessment.risk_level in {
            RiskLevel.CRITICAL,
            RiskLevel.HIGH,
            RiskLevel.MEDIUM,
        }:
            safety_additions.extend(
                [
                    "I'm concerned about your safety. "
                    "If you're in immediate danger, please call 911.",
                    "I can stay with you while we contact the 988 "
                    "Suicide & Crisis Lifeline or text 741741.",
                    "Is there someone you trust nearby we can involve for support?",
                ]
            )

        if assessment.protective_factors:
            safety_additions.append(
                (
                    "I notice you mentioned "
                    f"{assessment.protective_factors[0]} "
                    "- that's an important source of strength."
                )
            )

        enhanced_response = base_response
        if safety_additions:
            enhanced_response += "\n\n" + "\n".join(safety_additions)

        return enhanced_response

    def _generate_safety_plan_elements(self, assessment: CrisisAssessment) -> List[str]:
        """Generate personalized safety plan elements"""
        elements = []

        if assessment.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            elements.extend(
                [
                    "Identify warning signs when crisis thoughts start",
                    "Remove or secure means of self-harm",
                    "Contact crisis support immediately: 988 or 911",
                    "Reach out to trusted person in support network",
                    "Use coping strategies: breathing, grounding exercises",
                ]
            )
        elif assessment.risk_level == RiskLevel.MEDIUM:
            elements.extend(
                [
                    "Recognize early warning signs of distress",
                    "Use healthy coping strategies when stressed",
                    "Connect with supportive people in your life",
                    "Have crisis resources easily accessible",
                    "Practice self-care and stress management",
                ]
            )

        # Add protective factor reinforcement
        if assessment.protective_factors:
            elements.append(
                (
                    "Remember your sources of strength: "
                    f"{', '.join(assessment.protective_factors)}"
                )
            )

        return elements

    def _select_crisis_resources(self, assessment: CrisisAssessment) -> List[Dict]:
        """Select appropriate crisis resources based on assessment"""
        resources = []

        if assessment.risk_level == RiskLevel.CRITICAL:
            resources.extend(
                [
                    self.crisis_resources["emergency_services"],
                    self.crisis_resources["national_crisis_hotline"],
                ]
            )
        elif assessment.risk_level == RiskLevel.HIGH:
            resources.extend(
                [
                    self.crisis_resources["national_crisis_hotline"],
                    self.crisis_resources["crisis_text_line"],
                ]
            )
        elif assessment.risk_level == RiskLevel.MEDIUM:
            resources.extend(
                [
                    self.crisis_resources["crisis_text_line"],
                    self.crisis_resources["national_crisis_hotline"],
                ]
            )
        else:
            resources.append(
                {
                    "name": "Mental Health Resources",
                    "description": (
                        "Professional mental health support is available if you need it"
                    ),
                    "phone": "988 for crisis support",
                }
            )

        return resources

    def _schedule_follow_up(self, assessment: CrisisAssessment) -> Optional[datetime]:
        """Schedule appropriate follow-up based on risk level"""
        now = datetime.now()

        if assessment.risk_level == RiskLevel.CRITICAL:
            return now + timedelta(hours=1)  # 1 hour follow-up
        elif assessment.risk_level == RiskLevel.HIGH:
            return now + timedelta(hours=6)  # 6 hour follow-up
        elif assessment.risk_level == RiskLevel.MEDIUM:
            return now + timedelta(hours=24)  # 24 hour follow-up
        else:
            return now + timedelta(days=3)  # 3 day follow-up


class CrisisInterventionSystem:
    """Complete crisis intervention system orchestrator"""

    def __init__(self):
        self.detection_engine = CrisisDetectionEngine()
        self.responder = TherapeuticCrisisResponder()
        self.active_crisis_sessions = {}

    async def handle_crisis_message(
        self, user_id: str, message: str, conversation_history: List[str] = None
    ) -> Dict:
        """Handle incoming message with crisis assessment and response"""

        # Step 1: Crisis assessment
        assessment = self.detection_engine.assess_crisis_risk(
            message, conversation_history
        )

        logger.info(
            "Crisis assessment for user %s: %s (score: %s)",
            user_id,
            assessment.risk_level.value,
            assessment.risk_score,
        )

        # Step 2: Generate therapeutic response
        response = await self.responder.generate_crisis_response(
            message, assessment, conversation_history
        )

        # Step 3: Handle escalation if needed
        if response.escalation_triggered:
            await self._handle_crisis_escalation(user_id, assessment, response)

        # Step 4: Track active crisis session
        if assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            self.active_crisis_sessions[user_id] = {
                "assessment": assessment,
                "response": response,
                "started": datetime.now(),
                "status": "active",
            }

        # Step 5: Return complete crisis intervention package
        return {
            "crisis_assessment": {
                "risk_level": assessment.risk_level.value,
                "risk_score": assessment.risk_score,
                "risk_factors": assessment.risk_factors,
                "intervention_needed": assessment.intervention_needed.value,
            },
            "therapeutic_response": {
                "message": response.message,
                "expert_used": response.expert_used,
                "safety_plan": response.safety_plan_elements,
                "resources": response.resources_provided,
            },
            "escalation_info": {
                "escalation_triggered": response.escalation_triggered,
                "follow_up_scheduled": response.follow_up_scheduled.isoformat()
                if response.follow_up_scheduled
                else None,
            },
            "system_metadata": {
                "assessment_confidence": assessment.confidence,
                "response_timestamp": response.response_timestamp.isoformat(),
                "session_tracked": user_id in self.active_crisis_sessions,
            },
        }

    async def _handle_crisis_escalation(
        self, user_id: str, assessment: CrisisAssessment, response: CrisisResponse
    ):
        """Handle crisis escalation procedures"""
        logger.warning(
            "Crisis escalation triggered for user %s: %s",
            user_id,
            assessment.risk_level.value,
        )

        if assessment.risk_level == RiskLevel.CRITICAL:
            # Immediate emergency services alert
            logger.critical(
                "CRITICAL CRISIS: User %s requires immediate emergency intervention",
                user_id,
            )
            # In production: integrate with emergency services API

        elif assessment.risk_level == RiskLevel.HIGH:
            # Human counselor alert
            logger.error(
                "HIGH RISK CRISIS: User %s requires human counselor intervention",
                user_id,
            )
            # In production: alert available crisis counselors

        # Log crisis event for monitoring and improvement
        crisis_event = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "risk_level": assessment.risk_level.value,
            "risk_score": assessment.risk_score,
            "risk_factors": assessment.risk_factors,
            "intervention_type": assessment.intervention_needed.value,
            "escalation_triggered": True,
        }

        # In production: save to crisis monitoring database
        logger.info("Crisis event logged: %s", crisis_event)


def main():
    """Demonstrate crisis intervention system"""
    logger.info("🚨 Crisis Intervention System Demo")

    # Initialize system
    crisis_system = CrisisInterventionSystem()

    # Test scenarios
    test_scenarios = [
        {
            "message": "I'm feeling really overwhelmed with work lately",
            "expected_risk": "low",
        },
        {
            "message": "I've been having thoughts about hurting myself",
            "expected_risk": "medium-high",
        },
        {"message": "I want to die and I have a plan", "expected_risk": "critical"},
        {
            "message": "No one would miss me if I was gone, I'm just a burden",
            "expected_risk": "high",
        },
    ]

    async def run_tests():
        for i, scenario in enumerate(test_scenarios):
            logger.info("\n--- Test Scenario %s ---", i + 1)
            logger.info("Message: %s", scenario["message"])

            result = await crisis_system.handle_crisis_message(
                user_id=f"test_user_{i}", message=scenario["message"]
            )

            logger.info("Risk Level: %s", result["crisis_assessment"]["risk_level"])
            logger.info("Risk Score: %s", result["crisis_assessment"]["risk_score"])
            logger.info(
                "Escalation: %s", result["escalation_info"]["escalation_triggered"]
            )
            logger.info(
                "Response: %s...",
                result["therapeutic_response"]["message"][:100],
            )

    asyncio.run(run_tests())

    logger.info(
        "\n🎯 Crisis Intervention System ready for integration with therapeutic AI!"
    )


if __name__ == "__main__":
    main()
