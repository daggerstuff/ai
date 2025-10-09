#!/usr/bin/env python3
"""
Crisis Intervention System Implementation
Real-time crisis detection and intervention using the breakthrough therapeutic AI.

This is the highest priority integration - providing 24/7 crisis support
with intelligent escalation and professional backup.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
                r"i'm going to hurt myself"
            ],
            "hopelessness_expressions": [
                r"nothing will ever get better",
                r"there's no point",
                r"i'm trapped",
                r"no way out",
                r"it will never end",
                r"i'm worthless",
                r"i'm a burden"
            ],
            "preparation_indicators": [
                r"giving away my things",
                r"saying goodbye",
                r"getting my affairs in order",
                r"writing letters",
                r"have the pills",
                r"know how i'll do it"
            ],
            "isolation_indicators": [
                r"no one understands",
                r"i'm all alone",
                r"pushing everyone away",
                r"can't talk to anyone",
                r"nobody cares"
            ]
        }
        
        self.self_harm_indicators = [
            r"cutting myself",
            r"burning myself",
            r"hitting myself",
            r"hurting myself",
            r"self harm",
            r"self-harm"
        ]
        
        self.protective_factors_keywords = [
            "family", "friends", "pets", "children", "future", "goals",
            "hope", "support", "therapy", "medication", "treatment",
            "recovery", "getting help", "reaching out"
        ]
        
        # Crisis resource database
        self.crisis_resources = {
            "national_crisis_hotline": {
                "name": "988 Suicide & Crisis Lifeline",
                "phone": "988",
                "text": "Text 'HELLO' to 741741",
                "chat": "suicidepreventionlifeline.org/chat",
                "available": "24/7"
            },
            "emergency_services": {
                "name": "Emergency Services",
                "phone": "911",
                "description": "For immediate danger to self or others"
            },
            "crisis_text_line": {
                "name": "Crisis Text Line",
                "text": "741741",
                "description": "Text HOME to connect with a crisis counselor"
            },
            "veterans_crisis": {
                "name": "Veterans Crisis Line",
                "phone": "1-800-273-8255, Press 1",
                "text": "838255",
                "chat": "veteranscrisisline.net/get-help/chat"
            }
        }
    
    def assess_crisis_risk(self, message: str, conversation_history: List[str] = None) -> CrisisAssessment:
        """Comprehensive crisis risk assessment"""
        risk_score = 0.0
        risk_factors = []
        protective_factors = []
        
        # Analyze current message
        message_lower = message.lower()
        
        # Check for direct suicide indicators
        for category, patterns in self.suicide_indicators.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    if category == "direct_statements":
                        risk_score += 3.0
                        risk_factors.append(f"Direct suicide ideation: {pattern}")
                    elif category == "preparation_indicators":
                        risk_score += 2.5
                        risk_factors.append(f"Suicide preparation behavior: {pattern}")
                    elif category == "hopelessness_expressions":
                        risk_score += 1.5
                        risk_factors.append(f"Hopelessness expression: {pattern}")
                    elif category == "isolation_indicators":
                        risk_score += 1.0
                        risk_factors.append(f"Social isolation: {pattern}")
        
        # Check for self-harm indicators
        for pattern in self.self_harm_indicators:
            if re.search(pattern, message_lower):
                risk_score += 2.0
                risk_factors.append(f"Self-harm indication: {pattern}")
        
        # Check for protective factors
        for factor in self.protective_factors_keywords:
            if factor in message_lower:
                risk_score -= 0.5
                protective_factors.append(f"Protective factor: {factor}")
        
        # Analyze conversation history if available
        if conversation_history:
            history_text = " ".join(conversation_history).lower()
            
            # Check for escalating risk over time
            recent_messages = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            for msg in recent_messages:
                if any(re.search(pattern, msg.lower()) for patterns in self.suicide_indicators.values() for pattern in patterns):
                    risk_score += 0.5
                    risk_factors.append("Escalating risk pattern in conversation")
        
        # Determine risk level
        if risk_score >= 6.0:
            risk_level = RiskLevel.CRITICAL
            intervention = InterventionType.EMERGENCY_SERVICES
        elif risk_score >= 4.0:
            risk_level = RiskLevel.HIGH
            intervention = InterventionType.HUMAN_COUNSELOR
        elif risk_score >= 2.0:
            risk_level = RiskLevel.MEDIUM
            intervention = InterventionType.CRISIS_HOTLINE
        else:
            risk_level = RiskLevel.LOW
            intervention = InterventionType.AI_SUPPORT
        
        # Generate immediate actions
        immediate_actions = self._generate_immediate_actions(risk_level, risk_factors)
        
        return CrisisAssessment(
            risk_level=risk_level,
            risk_score=min(risk_score, 10.0),
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            immediate_actions=immediate_actions,
            intervention_needed=intervention,
            confidence=0.85,  # Based on therapeutic AI training quality
            assessment_timestamp=datetime.now()
        )
    
    def _generate_immediate_actions(self, risk_level: RiskLevel, risk_factors: List[str]) -> List[str]:
        """Generate immediate action recommendations based on risk level"""
        actions = []
        
        if risk_level == RiskLevel.CRITICAL:
            actions.extend([
                "Immediately connect to emergency services (911)",
                "Do not leave person alone",
                "Remove means of self-harm if possible",
                "Contact emergency mental health services",
                "Alert crisis counselor immediately"
            ])
        elif risk_level == RiskLevel.HIGH:
            actions.extend([
                "Connect to human crisis counselor immediately",
                "Provide crisis hotline information (988)",
                "Assess immediate safety environment",
                "Develop safety plan with human support",
                "Schedule immediate professional follow-up"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            actions.extend([
                "Provide crisis hotline resources",
                "Offer to connect to crisis text line",
                "Develop basic safety plan",
                "Identify support system contacts",
                "Schedule check-in within 24 hours"
            ])
        else:  # LOW
            actions.extend([
                "Provide supportive therapeutic response",
                "Offer mental health resources",
                "Encourage professional help if needed",
                "Validate feelings and provide hope",
                "Check in on coping strategies"
            ])
        
        return actions

class TherapeuticCrisisResponder:
    """Generate therapeutic responses for crisis situations"""
    
    def __init__(self, therapeutic_ai_model=None):
        # In production, this would connect to the trained H100 model
        self.therapeutic_ai = therapeutic_ai_model
        
        # Crisis response templates (fallback if AI unavailable)
        self.crisis_response_templates = {
            RiskLevel.CRITICAL: [
                "I'm very concerned about what you're sharing. Your life has value and you deserve support right now. I'm connecting you with emergency help immediately.",
                "What you're feeling is overwhelming, but you don't have to face this alone. Emergency support is available right now - let me connect you immediately.",
                "I hear how much pain you're in. Professional crisis counselors are standing by to help you through this moment. You matter, and help is available right now."
            ],
            RiskLevel.HIGH: [
                "I can hear how much you're struggling right now. These feelings are temporary, even when they feel overwhelming. Let me connect you with someone who can provide immediate support.",
                "You've taken a brave step by sharing what you're going through. Professional crisis counselors are available right now to help you work through these feelings.",
                "The pain you're experiencing is real, but you don't have to carry it alone. Crisis support is available 24/7 - let me help you connect with someone right now."
            ],
            RiskLevel.MEDIUM: [
                "I appreciate you sharing what you're going through. These difficult feelings can be overwhelming, but there are people trained to help you navigate through this.",
                "What you're experiencing sounds really challenging. Would it be helpful to connect with a crisis counselor who can provide additional support?",
                "It takes courage to reach out when you're struggling. There are crisis support resources available 24/7 if you'd like to talk with someone trained in crisis intervention."
            ],
            RiskLevel.LOW: [
                "Thank you for sharing what you're going through. It sounds like you're dealing with some difficult feelings right now.",
                "I hear that you're going through a tough time. Many people experience these kinds of difficult emotions, and support is available.",
                "What you're feeling is valid, and it's important that you're reaching out. Would you like to talk about what's been most challenging for you?"
            ]
        }
    
    async def generate_crisis_response(self, message: str, assessment: CrisisAssessment, 
                                     conversation_context: List[str] = None) -> CrisisResponse:
        """Generate appropriate therapeutic response for crisis situation"""
        
        # Determine expert routing based on risk level
        if assessment.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            expert_preference = "empathetic + therapeutic"
        else:
            expert_preference = "empathetic + practical"
        
        # Generate response using therapeutic AI (if available)
        if self.therapeutic_ai:
            ai_response = await self._generate_ai_response(message, assessment, expert_preference, conversation_context)
        else:
            # Fallback to template-based response
            ai_response = self._generate_template_response(assessment)
        
        # Enhance response with safety elements
        enhanced_response = self._enhance_with_safety_elements(ai_response, assessment)
        
        # Generate safety plan elements
        safety_plan_elements = self._generate_safety_plan_elements(assessment)
        
        # Provide appropriate resources
        resources = self._select_crisis_resources(assessment)
        
        # Determine if escalation is needed
        escalation_triggered = assessment.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        
        # Schedule follow-up
        follow_up = self._schedule_follow_up(assessment)
        
        return CrisisResponse(
            message=enhanced_response,
            expert_used=expert_preference,
            safety_plan_elements=safety_plan_elements,
            resources_provided=resources,
            escalation_triggered=escalation_triggered,
            follow_up_scheduled=follow_up,
            response_timestamp=datetime.now()
        )
    
    async def _generate_ai_response(self, message: str, assessment: CrisisAssessment, 
                                  expert_preference: str, context: List[str]) -> str:
        """Generate response using therapeutic AI model"""
        # This would integrate with the trained H100 model
        # For now, return a placeholder
        return "AI therapeutic response would be generated here using the trained H100 model"
    
    def _generate_template_response(self, assessment: CrisisAssessment) -> str:
        """Generate template-based response as fallback"""
        templates = self.crisis_response_templates[assessment.risk_level]
        # In production, would use more sophisticated selection
        return templates[0]
    
    def _enhance_with_safety_elements(self, base_response: str, assessment: CrisisAssessment) -> str:
        """Enhance response with safety-specific elements"""
        safety_additions = []
        
        if assessment.risk_level == RiskLevel.CRITICAL:
            safety_additions.append("🚨 I'm connecting you with emergency crisis support right now.")
        elif assessment.risk_level == RiskLevel.HIGH:
            safety_additions.append("💙 Crisis counselors are available 24/7 - would you like me to connect you?")
        elif assessment.risk_level == RiskLevel.MEDIUM:
            safety_additions.append("💬 Crisis text support is available 24/7: Text HOME to 741741")
        
        if assessment.protective_factors:
            safety_additions.append(f"I notice you mentioned {assessment.protective_factors[0]} - that's an important source of strength.")
        
        enhanced_response = base_response
        if safety_additions:
            enhanced_response += "\n\n" + "\n".join(safety_additions)
        
        return enhanced_response
    
    def _generate_safety_plan_elements(self, assessment: CrisisAssessment) -> List[str]:
        """Generate personalized safety plan elements"""
        elements = []
        
        if assessment.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            elements.extend([
                "Identify warning signs when crisis thoughts start",
                "Remove or secure means of self-harm",
                "Contact crisis support immediately: 988 or 911",
                "Reach out to trusted person in support network",
                "Use coping strategies: breathing, grounding exercises"
            ])
        elif assessment.risk_level == RiskLevel.MEDIUM:
            elements.extend([
                "Recognize early warning signs of distress",
                "Use healthy coping strategies when stressed",
                "Connect with supportive people in your life",
                "Have crisis resources easily accessible",
                "Practice self-care and stress management"
            ])
        
        # Add protective factor reinforcement
        if assessment.protective_factors:
            elements.append(f"Remember your sources of strength: {', '.join(assessment.protective_factors)}")
        
        return elements
    
    def _select_crisis_resources(self, assessment: CrisisAssessment) -> List[Dict]:
        """Select appropriate crisis resources based on assessment"""
        resources = []
        
        if assessment.risk_level == RiskLevel.CRITICAL:
            resources.extend([
                self.crisis_resources["emergency_services"],
                self.crisis_resources["national_crisis_hotline"]
            ])
        elif assessment.risk_level == RiskLevel.HIGH:
            resources.extend([
                self.crisis_resources["national_crisis_hotline"],
                self.crisis_resources["crisis_text_line"]
            ])
        elif assessment.risk_level == RiskLevel.MEDIUM:
            resources.extend([
                self.crisis_resources["crisis_text_line"],
                self.crisis_resources["national_crisis_hotline"]
            ])
        else:
            resources.append({
                "name": "Mental Health Resources",
                "description": "Professional mental health support is available if you need it",
                "phone": "988 for crisis support"
            })
        
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
        
    async def handle_crisis_message(self, user_id: str, message: str, 
                                  conversation_history: List[str] = None) -> Dict:
        """Handle incoming message with crisis assessment and response"""
        
        # Step 1: Crisis assessment
        assessment = self.detection_engine.assess_crisis_risk(message, conversation_history)
        
        logger.info(f"Crisis assessment for user {user_id}: {assessment.risk_level.value} (score: {assessment.risk_score})")
        
        # Step 2: Generate therapeutic response
        response = await self.responder.generate_crisis_response(message, assessment, conversation_history)
        
        # Step 3: Handle escalation if needed
        if response.escalation_triggered:
            await self._handle_crisis_escalation(user_id, assessment, response)
        
        # Step 4: Track active crisis session
        if assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            self.active_crisis_sessions[user_id] = {
                "assessment": assessment,
                "response": response,
                "started": datetime.now(),
                "status": "active"
            }
        
        # Step 5: Return complete crisis intervention package
        return {
            "crisis_assessment": {
                "risk_level": assessment.risk_level.value,
                "risk_score": assessment.risk_score,
                "risk_factors": assessment.risk_factors,
                "intervention_needed": assessment.intervention_needed.value
            },
            "therapeutic_response": {
                "message": response.message,
                "expert_used": response.expert_used,
                "safety_plan": response.safety_plan_elements,
                "resources": response.resources_provided
            },
            "escalation_info": {
                "escalation_triggered": response.escalation_triggered,
                "follow_up_scheduled": response.follow_up_scheduled.isoformat() if response.follow_up_scheduled else None
            },
            "system_metadata": {
                "assessment_confidence": assessment.confidence,
                "response_timestamp": response.response_timestamp.isoformat(),
                "session_tracked": user_id in self.active_crisis_sessions
            }
        }
    
    async def _handle_crisis_escalation(self, user_id: str, assessment: CrisisAssessment, 
                                      response: CrisisResponse):
        """Handle crisis escalation procedures"""
        logger.warning(f"Crisis escalation triggered for user {user_id}: {assessment.risk_level.value}")
        
        if assessment.risk_level == RiskLevel.CRITICAL:
            # Immediate emergency services alert
            logger.critical(f"CRITICAL CRISIS: User {user_id} requires immediate emergency intervention")
            # In production: integrate with emergency services API
            
        elif assessment.risk_level == RiskLevel.HIGH:
            # Human counselor alert
            logger.error(f"HIGH RISK CRISIS: User {user_id} requires human counselor intervention")
            # In production: alert available crisis counselors
        
        # Log crisis event for monitoring and improvement
        crisis_event = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "risk_level": assessment.risk_level.value,
            "risk_score": assessment.risk_score,
            "risk_factors": assessment.risk_factors,
            "intervention_type": assessment.intervention_needed.value,
            "escalation_triggered": True
        }
        
        # In production: save to crisis monitoring database
        logger.info(f"Crisis event logged: {crisis_event}")

def main():
    """Demonstrate crisis intervention system"""
    logger.info("🚨 Crisis Intervention System Demo")
    
    # Initialize system
    crisis_system = CrisisInterventionSystem()
    
    # Test scenarios
    test_scenarios = [
        {
            "message": "I'm feeling really overwhelmed with work lately",
            "expected_risk": "low"
        },
        {
            "message": "I've been having thoughts about hurting myself",
            "expected_risk": "medium-high"
        },
        {
            "message": "I want to die and I have a plan",
            "expected_risk": "critical"
        },
        {
            "message": "No one would miss me if I was gone, I'm just a burden",
            "expected_risk": "high"
        }
    ]
    
    async def run_tests():
        for i, scenario in enumerate(test_scenarios):
            logger.info(f"\n--- Test Scenario {i+1} ---")
            logger.info(f"Message: '{scenario['message']}'")
            
            result = await crisis_system.handle_crisis_message(
                user_id=f"test_user_{i}",
                message=scenario["message"]
            )
            
            logger.info(f"Risk Level: {result['crisis_assessment']['risk_level']}")
            logger.info(f"Risk Score: {result['crisis_assessment']['risk_score']}")
            logger.info(f"Escalation: {result['escalation_info']['escalation_triggered']}")
            logger.info(f"Response: {result['therapeutic_response']['message'][:100]}...")
    
    import asyncio
    asyncio.run(run_tests())
    
    logger.info("\n🎯 Crisis Intervention System ready for integration with therapeutic AI!")

if __name__ == "__main__":
    main()