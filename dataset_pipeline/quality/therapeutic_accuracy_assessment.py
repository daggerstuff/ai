"""
Therapeutic accuracy assessment system for mental health conversation validation.

This module provides tools to assess the therapeutic accuracy and clinical appropriateness
of mental health conversations, ensuring they meet professional standards and safety requirements.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from conversation_schema import Conversation, Message

# Set up logging
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for therapeutic content."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TherapeuticApproach(Enum):
    """Common therapeutic approaches."""
    CBT = "cognitive_behavioral_therapy"
    DBT = "dialectical_behavior_therapy"
    HUMANISTIC = "humanistic"
    PSYCHODYNAMIC = "psychodynamic"
    SOLUTION_FOCUSED = "solution_focused"
    MINDFULNESS = "mindfulness_based"


@dataclass
class TherapeuticAccuracyMetrics:
    """Metrics for therapeutic accuracy assessment."""
    overall_score: float
    clinical_appropriateness_score: float
    safety_compliance_score: float
    therapeutic_technique_score: float
    professional_boundaries_score: float
    crisis_handling_score: float
    ethical_standards_score: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    critical_issues: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    quality_level: str = ""


class TherapeuticAccuracyAssessor:
    """
    Comprehensive therapeutic accuracy assessment system.

    Evaluates mental health conversations across multiple dimensions:
    1. Clinical appropriateness - proper therapeutic responses
    2. Safety compliance - crisis detection and appropriate responses
    3. Therapeutic technique - evidence-based interventions
    4. Professional boundaries - maintaining appropriate therapeutic relationship
    5. Crisis handling - appropriate responses to crisis situations
    6. Ethical standards - adherence to professional ethics
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the assessor with configuration."""
        self.config = config or {}

        # Default weights for different assessment dimensions
        self.weights = self.config.get("weights", {
            "clinical_appropriateness": 0.25,
            "safety_compliance": 0.20,
            "therapeutic_technique": 0.18,
            "professional_boundaries": 0.15,
            "crisis_handling": 0.12,
            "ethical_standards": 0.10
        })

        # Quality thresholds
        self.thresholds = self.config.get("thresholds", {
            "excellent": 0.90,
            "good": 0.75,
            "acceptable": 0.60,
            "poor": 0.45
        })

        # Initialize therapeutic knowledge bases
        self._initialize_therapeutic_knowledge()

        # Initialize safety and crisis indicators
        self._initialize_safety_indicators()

        # Initialize professional standards
        self._initialize_professional_standards()

    def _initialize_therapeutic_knowledge(self):
        """Initialize therapeutic techniques and interventions."""
        self.therapeutic_techniques = {
            "cbt_techniques": [
                "cognitive restructuring", "thought challenging", "behavioral activation",
                "exposure therapy", "homework assignments", "thought records",
                "activity scheduling", "graded exposure", "cognitive defusion"
            ],
            "dbt_techniques": [
                "distress tolerance", "emotion regulation", "mindfulness",
                "interpersonal effectiveness", "wise mind", "radical acceptance",
                "distraction techniques", "self-soothing"
            ],
            "humanistic_techniques": [
                "active listening", "empathetic responding", "unconditional positive regard",
                "reflection", "clarification", "summarizing", "paraphrasing"
            ],
            "solution_focused": [
                "scaling questions", "miracle question", "exception finding",
                "goal setting", "strength identification", "resource mapping"
            ]
        }

        self.therapeutic_language = {
            "validation": [
                "that makes sense", "i can understand", "your feelings are valid",
                "that sounds difficult", "i hear you", "that's understandable"
            ],
            "exploration": [
                "tell me more about", "what was that like", "how did that feel",
                "what comes up for you", "what do you notice", "help me understand"
            ],
            "reframing": [
                "another way to look at", "what if we considered", "i wonder if",
                "have you thought about", "what would it mean if", "alternatively"
            ],
            "psychoeducation": [
                "it's common to", "many people experience", "research shows",
                "this is a normal response", "what we know about", "studies indicate"
            ]
        }

    def _initialize_safety_indicators(self):
        """Initialize crisis and safety indicators."""
        self.crisis_indicators = {
            "suicide_risk": [
                "kill myself", "end my life", "suicide", "suicidal", "want to die",
                "better off dead", "no point living", "end it all", "take my life"
            ],
            "self_harm": [
                "hurt myself", "cut myself", "self-harm", "self harm", "cutting",
                "burning myself", "hitting myself", "punish myself"
            ],
            "violence_risk": [
                "hurt someone", "kill them", "violent thoughts", "want to hurt",
                "make them pay", "get revenge", "violent urges"
            ],
            "substance_abuse": [
                "drinking too much", "using drugs", "overdose", "getting high",
                "can't stop drinking", "need drugs", "substance abuse"
            ],
            "psychosis_indicators": [
                "hearing voices", "seeing things", "paranoid", "conspiracy",
                "people watching me", "voices telling me", "hallucinations"
            ]
        }

        self.appropriate_crisis_responses = [
            "safety is my primary concern", "let's talk about safety",
            "i'm concerned about your safety", "we need to address this immediately",
            "have you thought about getting help", "emergency services",
            "crisis hotline", "immediate support", "safety plan"
        ]

    def _initialize_professional_standards(self):
        """Initialize professional boundaries and ethical standards."""
        self.boundary_violations = {
            "dual_relationships": [
                "we should be friends", "let's meet outside", "personal relationship",
                "date me", "romantic feelings", "attracted to you"
            ],
            "inappropriate_disclosure": [
                "my personal problems", "my relationship issues", "my trauma",
                "when i was depressed", "my therapist told me", "my own experience"
            ],
            "advice_giving": [
                "you should leave him", "you need to quit", "just do this",
                "my advice is", "if i were you", "you have to"
            ],
            "diagnosis_claims": [
                "you have depression", "you're bipolar", "you have ptsd",
                "i diagnose you", "you definitely have", "your diagnosis is"
            ]
        }

        self.ethical_violations = [
            "confidentiality breach", "tell your family", "share this information",
            "report to authorities", "mandatory reporting", "break confidentiality"
        ]

        self.professional_language = [
            "in our therapeutic relationship", "within our sessions",
            "professional boundaries", "therapeutic alliance",
            "clinical assessment", "treatment planning"
        ]

    def assess_therapeutic_accuracy(self, conversation: Conversation) -> TherapeuticAccuracyMetrics:
        """
        Assess the therapeutic accuracy of a mental health conversation.

        Args:
            conversation: The conversation to assess

        Returns:
            TherapeuticAccuracyMetrics with detailed assessment results
        """
        logger.info(f"Assessing therapeutic accuracy for conversation {conversation.id}")

        if len(conversation.messages) < 2:
            return TherapeuticAccuracyMetrics(
                overall_score=0.0,
                clinical_appropriateness_score=0.0,
                safety_compliance_score=0.0,
                therapeutic_technique_score=0.0,
                professional_boundaries_score=0.0,
                crisis_handling_score=0.0,
                ethical_standards_score=0.0,
                issues=["Insufficient messages for therapeutic accuracy assessment"],
                risk_level=RiskLevel.LOW,
                details={"conversation_length": len(conversation.messages)},
                quality_level="very_poor"
            )

        # Assess different dimensions
        clinical_appropriateness = self._assess_clinical_appropriateness(conversation.messages)
        safety_compliance = self._assess_safety_compliance(conversation.messages)
        therapeutic_technique = self._assess_therapeutic_technique(conversation.messages)
        professional_boundaries = self._assess_professional_boundaries(conversation.messages)
        crisis_handling = self._assess_crisis_handling(conversation.messages)
        ethical_standards = self._assess_ethical_standards(conversation.messages)

        # Calculate weighted overall score
        overall_score = (
            clinical_appropriateness["score"] * self.weights["clinical_appropriateness"] +
            safety_compliance["score"] * self.weights["safety_compliance"] +
            therapeutic_technique["score"] * self.weights["therapeutic_technique"] +
            professional_boundaries["score"] * self.weights["professional_boundaries"] +
            crisis_handling["score"] * self.weights["crisis_handling"] +
            ethical_standards["score"] * self.weights["ethical_standards"]
        )

        # Compile all issues
        all_issues = []
        all_warnings = []
        all_critical_issues = []

        for assessment in [clinical_appropriateness, safety_compliance, therapeutic_technique,
                          professional_boundaries, crisis_handling, ethical_standards]:
            all_issues.extend(assessment.get("issues", []))
            all_warnings.extend(assessment.get("warnings", []))
            all_critical_issues.extend(assessment.get("critical_issues", []))

        # Determine risk level
        risk_level = self._determine_risk_level(all_critical_issues, all_warnings, overall_score)

        # Compile detailed results
        details = {
            "conversation_length": len(conversation.messages),
            "unique_roles": len({msg.role for msg in conversation.messages}),
            "clinical_appropriateness_details": clinical_appropriateness.get("details", {}),
            "safety_compliance_details": safety_compliance.get("details", {}),
            "therapeutic_technique_details": therapeutic_technique.get("details", {}),
            "professional_boundaries_details": professional_boundaries.get("details", {}),
            "crisis_handling_details": crisis_handling.get("details", {}),
            "ethical_standards_details": ethical_standards.get("details", {}),
            "quality_level": self._determine_quality_level(overall_score)
        }

        return TherapeuticAccuracyMetrics(
            overall_score=overall_score,
            clinical_appropriateness_score=clinical_appropriateness["score"],
            safety_compliance_score=safety_compliance["score"],
            therapeutic_technique_score=therapeutic_technique["score"],
            professional_boundaries_score=professional_boundaries["score"],
            crisis_handling_score=crisis_handling["score"],
            ethical_standards_score=ethical_standards["score"],
            issues=all_issues,
            warnings=all_warnings,
            critical_issues=all_critical_issues,
            details=details,
            risk_level=risk_level,
            quality_level=self._determine_quality_level(overall_score)
        )

    def _determine_risk_level(self, critical_issues: list[str], warnings: list[str], overall_score: float) -> RiskLevel:
        """Determine the risk level based on issues and score."""
        if critical_issues:
            return RiskLevel.CRITICAL
        if len(warnings) > 3 or overall_score < 0.4:
            return RiskLevel.HIGH
        if len(warnings) > 1 or overall_score < 0.6:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        if score >= self.thresholds["excellent"]:
            return "excellent"
        if score >= self.thresholds["good"]:
            return "good"
        if score >= self.thresholds["acceptable"]:
            return "acceptable"
        if score >= self.thresholds["poor"]:
            return "poor"
        return "very_poor"

    def _assess_clinical_appropriateness(self, messages: list[Message]) -> dict[str, Any]:
        """Assess clinical appropriateness of therapeutic responses."""
        score = 1.0
        issues = []
        warnings = []
        critical_issues = []

        therapeutic_responses = 0
        appropriate_responses = 0

        for i in range(1, len(messages)):
            current_msg = messages[i]
            previous_msg = messages[i-1]

            # Skip if not a therapeutic response (assistant/therapist role)
            if current_msg.role not in ["assistant", "therapist"]:
                continue

            therapeutic_responses += 1
            current_content = current_msg.content.lower()
            previous_content = previous_msg.content.lower()

            # Check for appropriate therapeutic language
            has_therapeutic_language = any(
                any(phrase in current_content for phrase in phrases)
                for phrases in self.therapeutic_language.values()
            )

            if has_therapeutic_language:
                appropriate_responses += 1
            else:
                issues.append(f"Lack of therapeutic language in response at turn {i}")

            # Check for inappropriate medical advice
            medical_advice_indicators = [
                "take this medication", "stop your medication", "increase dosage",
                "medical diagnosis", "you need surgery", "medical treatment"
            ]
            if any(indicator in current_content for indicator in medical_advice_indicators):
                critical_issues.append(f"Inappropriate medical advice given at turn {i}")
                score -= 0.3

            # Check for premature problem-solving
            if len(previous_content) > 100 and any(emotion in previous_content for emotion in
                ["sad", "depressed", "anxious", "worried", "scared", "angry"]):
                if any(phrase in current_content for phrase in
                    ["just do this", "simple solution", "easy fix", "just think positive"]):
                    warnings.append(f"Premature problem-solving without exploration at turn {i}")
                    score -= 0.1

        # Calculate appropriateness ratio
        if therapeutic_responses > 0:
            appropriateness_ratio = appropriate_responses / therapeutic_responses
            if appropriateness_ratio == 0.0:
                issues.append("No therapeutic language usage")
                score -= 0.6
            elif appropriateness_ratio < 0.3:
                issues.append("Very low therapeutic language usage")
                score -= 0.4
            elif appropriateness_ratio < 0.7:
                warnings.append("Moderate therapeutic language usage - could be improved")
                score -= 0.1

        return {
            "score": max(0.0, score),
            "issues": issues,
            "warnings": warnings,
            "critical_issues": critical_issues,
            "details": {
                "therapeutic_responses": therapeutic_responses,
                "appropriate_responses": appropriate_responses,
                "appropriateness_ratio": appropriate_responses / max(1, therapeutic_responses)
            }
        }

    def _assess_safety_compliance(self, messages: list[Message]) -> dict[str, Any]:
        """Assess safety compliance and crisis response appropriateness."""
        score = 1.0
        issues = []
        warnings = []
        critical_issues = []

        crisis_situations_detected = 0
        appropriate_crisis_responses = 0

        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1] if i + 1 < len(messages) else None

            current_content = current_msg.content.lower()

            # Detect crisis indicators
            crisis_detected = False
            crisis_type = None

            for crisis_category, indicators in self.crisis_indicators.items():
                if any(indicator in current_content for indicator in indicators):
                    crisis_detected = True
                    crisis_type = crisis_category
                    crisis_situations_detected += 1
                    break

            if crisis_detected and next_msg and next_msg.role in ["assistant", "therapist"]:
                next_content = next_msg.content.lower()

                # Check for appropriate crisis response
                has_appropriate_response = any(
                    response in next_content for response in self.appropriate_crisis_responses
                )

                if has_appropriate_response:
                    appropriate_crisis_responses += 1
                elif crisis_type in ["suicide_risk", "violence_risk"]:
                    critical_issues.append(f"Inadequate response to {crisis_type} at turn {i+1}")
                    score -= 0.4
                else:
                    warnings.append(f"Suboptimal response to {crisis_type} at turn {i+1}")
                    score -= 0.2

                # Check for dismissive responses to crisis
                dismissive_phrases = [
                    "you'll be fine", "don't worry about it", "it's not that bad",
                    "just relax", "calm down", "get over it"
                ]
                if any(phrase in next_content for phrase in dismissive_phrases):
                    critical_issues.append(f"Dismissive response to crisis situation at turn {i+1}")
                    score -= 0.3

        # Assess overall crisis handling
        if crisis_situations_detected > 0:
            crisis_response_ratio = appropriate_crisis_responses / crisis_situations_detected
            if crisis_response_ratio < 0.8:
                issues.append("Inconsistent crisis response quality")
                score -= 0.2

        return {
            "score": max(0.0, score),
            "issues": issues,
            "warnings": warnings,
            "critical_issues": critical_issues,
            "details": {
                "crisis_situations_detected": crisis_situations_detected,
                "appropriate_crisis_responses": appropriate_crisis_responses,
                "crisis_response_ratio": appropriate_crisis_responses / max(1, crisis_situations_detected)
            }
        }

    def _assess_therapeutic_technique(self, messages: list[Message]) -> dict[str, Any]:
        """Assess use of evidence-based therapeutic techniques."""
        score = 1.0
        issues = []
        warnings = []
        critical_issues = []

        technique_usage = Counter()
        therapeutic_responses = 0

        for _i, message in enumerate(messages):
            if message.role not in ["assistant", "therapist"]:
                continue

            therapeutic_responses += 1
            content = message.content.lower()

            # Count therapeutic techniques used
            for technique_category, techniques in self.therapeutic_techniques.items():
                for technique in techniques:
                    if technique in content:
                        technique_usage[technique_category] += 1

            # Check for specific therapeutic language patterns
            for language_type, phrases in self.therapeutic_language.items():
                for phrase in phrases:
                    if phrase in content:
                        technique_usage[language_type] += 1

        # Assess technique diversity
        unique_techniques = len(technique_usage)
        if unique_techniques == 0:
            issues.append("No identifiable therapeutic techniques used")
            score -= 0.4
        elif unique_techniques < 3:
            warnings.append("Limited variety of therapeutic techniques")
            score -= 0.2

        # Assess technique appropriateness
        total_technique_usage = sum(technique_usage.values())
        if therapeutic_responses > 0:
            technique_ratio = total_technique_usage / therapeutic_responses
            if technique_ratio < 0.3:
                issues.append("Low therapeutic technique usage")
                score -= 0.3
            elif technique_ratio < 0.5:
                warnings.append("Moderate therapeutic technique usage")
                score -= 0.1

        return {
            "score": max(0.0, score),
            "issues": issues,
            "warnings": warnings,
            "critical_issues": critical_issues,
            "details": {
                "technique_usage": dict(technique_usage),
                "unique_techniques": unique_techniques,
                "therapeutic_responses": therapeutic_responses,
                "technique_ratio": total_technique_usage / max(1, therapeutic_responses)
            }
        }

    def _assess_professional_boundaries(self, messages: list[Message]) -> dict[str, Any]:
        """Assess maintenance of professional boundaries."""
        score = 1.0
        issues = []
        warnings = []
        critical_issues = []

        boundary_violations = 0

        for i, message in enumerate(messages):
            if message.role not in ["assistant", "therapist"]:
                continue

            content = message.content.lower()

            # Check for boundary violations
            for violation_type, indicators in self.boundary_violations.items():
                for indicator in indicators:
                    if indicator in content:
                        boundary_violations += 1
                        if violation_type in ["dual_relationships", "diagnosis_claims"]:
                            critical_issues.append(f"{violation_type} violation at turn {i}")
                            score -= 0.4
                        else:
                            warnings.append(f"{violation_type} concern at turn {i}")
                            score -= 0.2
                        break

            # Check for ethical violations
            for violation in self.ethical_violations:
                if violation in content:
                    critical_issues.append(f"Ethical violation: {violation} at turn {i}")
                    score -= 0.5

            # Reward professional language
            professional_language_count = sum(1 for phrase in self.professional_language if phrase in content)
            if professional_language_count > 0:
                score += 0.05  # Small bonus for professional language

        return {
            "score": min(1.0, max(0.0, score)),
            "issues": issues,
            "warnings": warnings,
            "critical_issues": critical_issues,
            "details": {
                "boundary_violations": boundary_violations,
                "professional_language_usage": sum(1 for msg in messages
                    if msg.role in ["assistant", "therapist"] and
                    any(phrase in msg.content.lower() for phrase in self.professional_language))
            }
        }

    def _assess_crisis_handling(self, messages: list[Message]) -> dict[str, Any]:
        """Assess specific crisis handling capabilities."""
        score = 1.0
        issues = []
        warnings = []
        critical_issues = []

        crisis_situations = []
        crisis_responses = []

        # Identify crisis situations and responses
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]

            current_content = current_msg.content.lower()

            # Check for crisis indicators
            for crisis_type, indicators in self.crisis_indicators.items():
                if any(indicator in current_content for indicator in indicators):
                    crisis_situations.append({
                        "type": crisis_type,
                        "turn": i,
                        "severity": self._assess_crisis_severity(current_content, crisis_type)
                    })

                    if next_msg.role in ["assistant", "therapist"]:
                        crisis_responses.append({
                            "crisis_turn": i,
                            "response_turn": i + 1,
                            "response_content": next_msg.content.lower(),
                            "crisis_type": crisis_type
                        })

        # Evaluate crisis responses
        for response in crisis_responses:
            response_quality = self._evaluate_crisis_response(
                response["response_content"],
                response["crisis_type"]
            )

            if response_quality["score"] < 0.3:
                if response["crisis_type"] in ["suicide_risk", "violence_risk"]:
                    critical_issues.append(f"Inadequate {response['crisis_type']} response at turn {response['response_turn']}")
                    score -= 0.8
                else:
                    warnings.append(f"Suboptimal {response['crisis_type']} response at turn {response['response_turn']}")
                    score -= 0.4
            elif response_quality["score"] < 0.7:
                warnings.append(f"Marginal {response['crisis_type']} response at turn {response['response_turn']}")
                score -= 0.2

        return {
            "score": max(0.0, score),
            "issues": issues,
            "warnings": warnings,
            "critical_issues": critical_issues,
            "details": {
                "crisis_situations": len(crisis_situations),
                "crisis_responses": len(crisis_responses),
                "crisis_types_detected": list({cs["type"] for cs in crisis_situations})
            }
        }

    def _assess_ethical_standards(self, messages: list[Message]) -> dict[str, Any]:
        """Assess adherence to ethical standards."""
        score = 1.0
        issues = []
        warnings = []
        critical_issues = []

        for i, message in enumerate(messages):
            if message.role not in ["assistant", "therapist"]:
                continue

            content = message.content.lower()

            # Check for confidentiality concerns
            confidentiality_breaches = [
                "tell your family", "share with others", "report to",
                "inform authorities", "break confidentiality"
            ]

            for breach in confidentiality_breaches:
                if breach in content:
                    # Check if it's appropriate (mandatory reporting situations)
                    if any(situation in content for situation in ["child abuse", "elder abuse", "imminent danger"]):
                        warnings.append(f"Appropriate mandatory reporting mentioned at turn {i}")
                    else:
                        critical_issues.append(f"Inappropriate confidentiality breach at turn {i}")
                        score -= 0.6

            # Check for informed consent discussions
            consent_indicators = [
                "informed consent", "limits of confidentiality", "mandatory reporting",
                "treatment goals", "therapeutic process"
            ]

            if any(indicator in content for indicator in consent_indicators):
                score += 0.05  # Small bonus for ethical practice

        return {
            "score": min(1.0, max(0.0, score)),
            "issues": issues,
            "warnings": warnings,
            "critical_issues": critical_issues,
            "details": {
                "ethical_discussions": sum(1 for msg in messages
                    if msg.role in ["assistant", "therapist"] and
                    any(indicator in msg.content.lower() for indicator in
                        ["informed consent", "confidentiality", "ethical"]))
            }
        }

    def _assess_crisis_severity(self, content: str, crisis_type: str) -> str:
        """Assess the severity of a crisis situation."""
        high_severity_indicators = {
            "suicide_risk": ["plan to", "tonight", "today", "right now", "immediately"],
            "self_harm": ["cutting now", "hurting myself now", "bleeding"],
            "violence_risk": ["going to hurt", "plan to", "weapon", "tonight"],
            "substance_abuse": ["overdosed", "too much", "can't stop"],
            "psychosis_indicators": ["right now", "telling me to", "commanding"]
        }

        if crisis_type in high_severity_indicators:
            if any(indicator in content for indicator in high_severity_indicators[crisis_type]):
                return "high"

        return "moderate"

    def _evaluate_crisis_response(self, response_content: str, crisis_type: str) -> dict[str, Any]:
        """Evaluate the quality of a crisis response."""
        score = 0.0

        # Check for appropriate crisis response elements
        if any(phrase in response_content for phrase in self.appropriate_crisis_responses):
            score += 0.5

        # Check for safety planning
        safety_planning_indicators = [
            "safety plan", "coping strategies", "support system",
            "emergency contacts", "crisis hotline"
        ]
        if any(indicator in response_content for indicator in safety_planning_indicators):
            score += 0.3

        # Check for immediate action when needed
        if crisis_type in ["suicide_risk", "violence_risk"]:
            immediate_action_indicators = [
                "emergency services", "911", "crisis line", "immediate help",
                "go to emergency room", "call for help"
            ]
            if any(indicator in response_content for indicator in immediate_action_indicators):
                score += 0.2

        return {"score": min(1.0, score)}


# Backward compatibility function
def assess_therapeutic_accuracy(conversation: Conversation, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Backward compatibility function for therapeutic accuracy assessment.

    Args:
        conversation: The conversation to assess
        config: Optional configuration dictionary

    Returns:
        Dictionary with 'score', 'issues', 'warnings', 'critical_issues', 'risk_level', and 'details' keys
    """
    assessor = TherapeuticAccuracyAssessor(config)
    metrics = assessor.assess_therapeutic_accuracy(conversation)

    return {
        "score": metrics.overall_score,
        "issues": metrics.issues,
        "warnings": metrics.warnings,
        "critical_issues": metrics.critical_issues,
        "risk_level": metrics.risk_level.value,
        "details": metrics.details
    }
