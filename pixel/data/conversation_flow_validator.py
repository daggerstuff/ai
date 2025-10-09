"""
Conversation Flow Validator

Validates therapeutic conversation flow for appropriateness, coherence,
and adherence to therapeutic principles.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .therapeutic_conversation_schema import (
    TherapeuticConversation,
    ConversationTurn,
    ConversationRole,
    TherapeuticModality
)


class FlowViolationType(Enum):
    """Types of conversation flow violations."""
    INAPPROPRIATE_DISCLOSURE = "inappropriate_disclosure"
    BOUNDARY_VIOLATION = "boundary_violation"
    PREMATURE_INTERPRETATION = "premature_interpretation"
    MISSED_CRISIS_INDICATOR = "missed_crisis_indicator"
    INCONSISTENT_APPROACH = "inconsistent_approach"
    POOR_TIMING = "poor_timing"
    LACK_OF_EMPATHY = "lack_of_empathy"
    THERAPEUTIC_RUPTURE = "therapeutic_rupture"


@dataclass
class FlowViolation:
    """Represents a conversation flow violation."""
    violation_type: FlowViolationType
    turn_index: int
    severity: str  # "low", "medium", "high"
    description: str
    recommendation: str


@dataclass
class FlowValidationResult:
    """Result of conversation flow validation."""
    is_appropriate: bool
    overall_score: float
    violations: List[FlowViolation]
    strengths: List[str]
    recommendations: List[str]


class ConversationFlowValidator:
    """Validates therapeutic conversation flow."""

    def __init__(self):
        self.crisis_keywords = [
            "kill myself", "end it all", "not worth living", "want to die",
            "hurt myself", "cut myself", "suicide", "overdose"
        ]

        self.boundary_violations = [
            "let me tell you about my", "i had the same", "you should",
            "if i were you", "my personal experience", "my own"
        ]

    def validate_conversation_flow(
        self, conversation: TherapeuticConversation
    ) -> FlowValidationResult:
        """Validate complete conversation flow."""
        violations = []
        strengths = []

        # Check each turn for violations
        for i, turn in enumerate(conversation.turns):
            turn_violations = self._validate_turn(turn, i, conversation)
            violations.extend(turn_violations)

        # Check overall flow patterns
        flow_violations = self._validate_overall_flow(conversation)
        violations.extend(flow_violations)

        # Identify strengths
        strengths = self._identify_strengths(conversation)

        # Calculate overall score
        score = self._calculate_flow_score(conversation, violations)

        # Generate recommendations
        recommendations = self._generate_recommendations(violations)

        return FlowValidationResult(
            is_appropriate=(
                score >= 0.7 and all(v.severity != "high" for v in violations)
            ),
            overall_score=score,
            violations=violations,
            strengths=strengths,
            recommendations=recommendations
        )

    def _validate_turn(
        self, turn: ConversationTurn, index: int,
        conversation: TherapeuticConversation
    ) -> List[FlowViolation]:
        """Validate individual conversation turn."""
        violations = []
        content_lower = turn.content.lower()

        if turn.role == ConversationRole.THERAPIST:
            # Check for boundary violations
            violations.extend(
                FlowViolation(
                    violation_type=FlowViolationType.BOUNDARY_VIOLATION,
                    turn_index=index,
                    severity="high",
                    description=f"Therapist self-disclosure or advice-giving detected: '{boundary_phrase}'",
                    recommendation="Maintain professional boundaries and avoid personal disclosure",
                )
                for boundary_phrase in self.boundary_violations
                if boundary_phrase in content_lower
            )
            # Check for premature interpretation (early in conversation)
            early_interp_words = [
                "you always",
                "your pattern",
                "this stems from"]
            if index < 4 and any(
                    word in content_lower for word in early_interp_words):
                violations.append(
                    FlowViolation(
                        violation_type=FlowViolationType.PREMATURE_INTERPRETATION,
                        turn_index=index,
                        severity="medium",
                        description="Interpretation offered too early in therapeutic relationship",
                        recommendation="Build rapport and gather more information before offering interpretations"))

        elif turn.role == ConversationRole.CLIENT:
            # Check for missed crisis indicators
            crisis_detected = any(
                keyword in content_lower for keyword in self.crisis_keywords
            )
            if crisis_detected:
                # Check if next therapist turn addresses crisis
                next_therapist_turn = self._get_next_therapist_turn(
                    conversation.turns, index
                )
                if (next_therapist_turn and not self._addresses_crisis(
                        next_therapist_turn.content)):
                    violations.append(
                        FlowViolation(
                            violation_type=FlowViolationType.MISSED_CRISIS_INDICATOR,
                            turn_index=index + 1,
                            severity="high",
                            description="Crisis indicator not addressed by therapist",
                            recommendation="Immediately assess safety and address crisis indicators"))

        return violations

    def _validate_overall_flow(
            self,
            conversation: TherapeuticConversation) -> List[FlowViolation]:
        """Validate overall conversation flow patterns."""
        violations = []

        # Check for consistent therapeutic approach
        therapist_turns = [
            t for t in conversation.turns if t.role == ConversationRole.THERAPIST]

        if len(therapist_turns) > 3:
            techniques = [
                t.therapeutic_technique for t in therapist_turns
                if t.therapeutic_technique
            ]
            # Too many different techniques
            if len(set(techniques)) > len(techniques) * 0.8:
                violations.append(
                    FlowViolation(
                        violation_type=FlowViolationType.INCONSISTENT_APPROACH,
                        turn_index=-1,
                        severity="medium",
                        description="Inconsistent therapeutic approach across session",
                        recommendation="Maintain consistent therapeutic modality and approach"))

        # Check for empathy and validation
        empathy_count = len([
            turn for turn in therapist_turns
            if self._contains_empathy(turn.content)
        ])

        if therapist_turns and empathy_count / len(therapist_turns) < 0.3:
            violations.append(
                FlowViolation(
                    violation_type=FlowViolationType.LACK_OF_EMPATHY,
                    turn_index=-1,
                    severity="medium",
                    description="Insufficient empathetic responses throughout conversation",
                    recommendation="Increase validation and empathetic responding"))

        return violations

    def _get_next_therapist_turn(
        self, turns: List[ConversationTurn],
        current_index: int
    ) -> Optional[ConversationTurn]:
        """Get the next therapist turn after current index."""
        return next(
            (turns[i] for i in range(current_index + 1, len(turns))
             if turns[i].role == ConversationRole.THERAPIST),
            None
        )

    def _addresses_crisis(self, content: str) -> bool:
        """Check if therapist response addresses crisis."""
        crisis_responses = [
            "safety", "safe", "harm", "hurt", "crisis", "emergency",
            "plan", "support", "help", "resources", "concerned"
        ]
        content_lower = content.lower()
        return any(word in content_lower for word in crisis_responses)

    def _contains_empathy(self, content: str) -> bool:
        """Check if content contains empathetic language."""
        empathy_indicators = [
            "i hear", "i understand", "that sounds", "i can see",
            "that must be", "i imagine", "it seems like", "that's difficult"
        ]
        content_lower = content.lower()
        return any(phrase in content_lower for phrase in empathy_indicators)

    def _identify_strengths(
            self,
            conversation: TherapeuticConversation) -> List[str]:
        """Identify conversation strengths."""
        strengths = []
        therapist_turns = [
            t for t in conversation.turns if t.role == ConversationRole.THERAPIST]

        # Check for good therapeutic qualities
        if any(self._contains_empathy(turn.content)
               for turn in therapist_turns):
            strengths.append("Demonstrates empathetic responding")

        if any(turn.clinical_rationale for turn in therapist_turns):
            strengths.append("Clear clinical rationale for interventions")

        if (
            len(
                {
                    t.therapeutic_technique
                    for t in therapist_turns
                    if t.therapeutic_technique
                }
            )
            <= 3
        ):
            strengths.append("Consistent therapeutic approach")

        # Check for appropriate pacing
        if len(conversation.turns) >= 6:
            strengths.append("Appropriate conversation length and pacing")

        return strengths

    def _calculate_flow_score(
        self, conversation: TherapeuticConversation,
        violations: List[FlowViolation]
    ) -> float:
        """Calculate overall flow appropriateness score."""
        base_score = 1.0

        # Deduct points for violations
        for violation in violations:
            if violation.severity == "high":
                base_score -= 0.3
            elif violation.severity == "medium":
                base_score -= 0.15
            else:  # low
                base_score -= 0.05

        # Bonus for good practices
        therapist_turns = [
            t for t in conversation.turns if t.role == ConversationRole.THERAPIST]

        # Bonus for clinical rationale
        rationale_ratio = (
            len([t for t in therapist_turns if t.clinical_rationale]) /
            max(1, len(therapist_turns))
        )
        base_score += rationale_ratio * 0.1

        # Bonus for empathy
        empathy_ratio = (
            len([t for t in therapist_turns if self._contains_empathy(t.content)]) /
            max(1, len(therapist_turns))
        )
        base_score += empathy_ratio * 0.1

        return max(0.0, min(1.0, base_score))

    def _generate_recommendations(
            self, violations: List[FlowViolation]) -> List[str]:
        """Generate recommendations based on violations."""

        # Get unique recommendations from violations
        seen = set()
        recommendations = [
            v.recommendation for v in violations
            if v.recommendation not in seen and not seen.add(v.recommendation)
        ]

        # Add general recommendations
        boundary_violations = any(
            v.violation_type == FlowViolationType.BOUNDARY_VIOLATION
            for v in violations
        )
        if boundary_violations:
            recommendations.append(
                "Review professional boundaries and therapeutic frame"
            )

        crisis_violations = any(
            v.violation_type == FlowViolationType.MISSED_CRISIS_INDICATOR
            for v in violations
        )
        if crisis_violations:
            recommendations.append("Complete crisis intervention training")

        return recommendations


class ConversationQualityScorer:
    """Scores conversation quality across multiple dimensions."""

    def __init__(self):
        self.flow_validator = ConversationFlowValidator()

    def score_conversation(
            self, conversation: TherapeuticConversation) -> Dict[str, Any]:
        """Score conversation across multiple quality dimensions."""

        # Flow validation
        flow_result = self.flow_validator.validate_conversation_flow(
            conversation)

        # Clinical appropriateness
        clinical_score = self._score_clinical_appropriateness(conversation)

        # Therapeutic alliance
        alliance_score = self._score_therapeutic_alliance(conversation)

        # Coherence and structure
        coherence_score = self._score_coherence(conversation)

        # Overall quality
        overall_score = (
            flow_result.overall_score * 0.3 +
            clinical_score * 0.3 +
            alliance_score * 0.2 +
            coherence_score * 0.2
        )

        return {
            "overall_score": overall_score,
            "flow_score": flow_result.overall_score,
            "clinical_score": clinical_score,
            "alliance_score": alliance_score,
            "coherence_score": coherence_score,
            "is_appropriate": flow_result.is_appropriate and overall_score >= 0.7,
            "violations": [
                {
                    "type": v.violation_type.value,
                    "severity": v.severity,
                    "description": v.description,
                    "recommendation": v.recommendation
                }
                for v in flow_result.violations
            ],
            "strengths": flow_result.strengths,
            "recommendations": flow_result.recommendations
        }

    def _score_clinical_appropriateness(
            self, conversation: TherapeuticConversation) -> float:
        """Score clinical appropriateness of interventions."""
        therapist_turns = [
            t for t in conversation.turns if t.role == ConversationRole.THERAPIST]

        if not therapist_turns:
            return 0.0

        # Check for clinical rationale
        rationale_score = (
            len([t for t in therapist_turns if t.clinical_rationale]) /
            len(therapist_turns)
        )

        # Check for appropriate techniques
        technique_score = (
            len([t for t in therapist_turns if t.therapeutic_technique]) /
            len(therapist_turns)
        )

        # Check modality consistency
        modality_score = 1.0 if conversation.modality else 0.5

        return (
            rationale_score *
            0.4 +
            technique_score *
            0.4 +
            modality_score *
            0.2)

    def _score_therapeutic_alliance(
            self, conversation: TherapeuticConversation) -> float:
        """Score therapeutic alliance indicators."""
        therapist_turns = [
            t for t in conversation.turns if t.role == ConversationRole.THERAPIST]

        if not therapist_turns:
            return 0.0

        # Check for empathy
        empathy_count = len([
            t for t in therapist_turns
            if self.flow_validator._contains_empathy(t.content)
        ])
        empathy_score = empathy_count / len(therapist_turns)

        # Check for validation
        validation_words = [
            "understand",
            "hear",
            "valid",
            "makes sense",
            "difficult"]
        validation_count = len([
            t for t in therapist_turns
            if any(word in t.content.lower() for word in validation_words)
        ])
        validation_score = min(1.0, validation_count / len(therapist_turns))

        return (empathy_score * 0.6 + validation_score * 0.4)

    def _score_coherence(self, conversation: TherapeuticConversation) -> float:
        """Score conversation coherence and structure."""
        if len(conversation.turns) < 2:
            return 0.0

        # Check alternating pattern
        alternating_score = 1.0
        for i in range(1, len(conversation.turns)):
            if conversation.turns[i].role == conversation.turns[i - 1].role:
                alternating_score -= 0.1

        # Check appropriate length
        length_score = 1.0 if 4 <= len(conversation.turns) <= 20 else 0.7

        # Check for logical flow
        flow_score = 0.8  # Default assumption of reasonable flow

        return max(
            0.0,
            (alternating_score * 0.4 + length_score * 0.3 + flow_score * 0.3)
        )


def main():
    """Test conversation flow validation."""
    from .therapeutic_conversation_schema import ClinicalContext, ClinicalSeverity

    # Create test conversation
    from .therapeutic_conversation_schema import ClinicalContext, ClinicalSeverity
    conversation = TherapeuticConversation(
        title="Test Session",
        modality=TherapeuticModality.CBT,
        clinical_context=ClinicalContext(
            dsm5_categories=["Major Depressive Disorder"],
            severity_level=ClinicalSeverity.MODERATE
        )
    )

    # Add turns
    conversation.add_turn(
        ConversationRole.CLIENT,
        "I've been feeling really depressed and sometimes think about ending it all.")

    conversation.add_turn(
        ConversationRole.THERAPIST,
        "I'm concerned about your safety. Let's talk about these thoughts and create a safety plan.",
        clinical_rationale="Crisis intervention required for suicidal ideation",
        therapeutic_technique="crisis_assessment")

    # Validate
    validator = ConversationFlowValidator()
    result = validator.validate_conversation_flow(conversation)

    print("Flow Validation Results:")
    print(f"Appropriate: {result.is_appropriate}")
    print(f"Score: {result.overall_score:.2f}")
    print(f"Violations: {len(result.violations)}")
    print(f"Strengths: {result.strengths}")

    # Score quality
    scorer = ConversationQualityScorer()
    quality_result = scorer.score_conversation(conversation)

    print("\nQuality Scoring Results:")
    print(f"Overall Score: {quality_result['overall_score']:.2f}")
    print(f"Clinical Score: {quality_result['clinical_score']:.2f}")
    print(f"Alliance Score: {quality_result['alliance_score']:.2f}")


if __name__ == "__main__":
    main()
