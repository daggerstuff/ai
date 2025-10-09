"""
Therapist Response Generator

Generates clinically appropriate therapist responses with detailed clinical rationale
based on therapeutic modalities, client presentations, and clinical context.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import random

from .therapeutic_conversation_schema import (
    TherapeuticModality,
    ClinicalContext,
    ClinicalSeverity,
    ConversationRole
)


class InterventionType(Enum):
    """Types of therapeutic interventions."""
    ASSESSMENT = "assessment"
    EXPLORATION = "exploration"
    VALIDATION = "validation"
    PSYCHOEDUCATION = "psychoeducation"
    SKILL_BUILDING = "skill_building"
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"
    BEHAVIORAL_ACTIVATION = "behavioral_activation"
    CRISIS_INTERVENTION = "crisis_intervention"
    INTERPRETATION = "interpretation"
    REFLECTION = "reflection"


@dataclass
class TherapistResponse:
    """Complete therapist response with clinical rationale."""
    content: str
    clinical_rationale: str
    therapeutic_technique: str
    intervention_type: InterventionType
    confidence_score: float
    contraindications: List[str]
    follow_up_suggestions: List[str]


class TherapistResponseGenerator:
    """Generates therapeutic responses with clinical rationale."""

    def __init__(self):
        self.response_templates = self._initialize_templates()
        self.technique_rationales = self._initialize_rationales()

    def _initialize_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize response templates by modality and intervention type."""
        return {
            "CBT": {
                "assessment": [
                    "Can you tell me more about when these feelings are strongest?",
                    "What thoughts go through your mind when you experience this?",
                    "How long have you been noticing these patterns?",
                    "What situations tend to trigger these responses?"
                ],
                "cognitive_restructuring": [
                    "Let's examine the evidence for and against that thought.",
                    "What would you tell a friend who had this same thought?",
                    "Is there another way to look at this situation?",
                    "How realistic is this thought when you really examine it?"
                ],
                "behavioral_activation": [
                    "What activities used to bring you joy or satisfaction?",
                    "Let's identify some small, manageable steps you could take.",
                    "How might we schedule some pleasant activities into your week?",
                    "What would be one small thing you could do differently today?"
                ],
                "psychoeducation": [
                    "It's common for people with depression to experience these thoughts.",
                    "What you're describing sounds like a cognitive distortion called...",
                    "Understanding the connection between thoughts, feelings, and behaviors can be helpful.",
                    "Research shows that these patterns are very treatable with the right approach."
                ]
            },
            "DBT": {
                "validation": [
                    "That sounds incredibly difficult and painful.",
                    "Your feelings make complete sense given what you've experienced.",
                    "I can see how much you're struggling right now.",
                    "It takes courage to share something so personal."
                ],
                "skill_building": [
                    "Let's practice the TIPP skill for managing intense emotions.",
                    "Have you tried using the STOP technique when you notice these urges?",
                    "Let's work on some distress tolerance skills for moments like these.",
                    "The wise mind concept might be helpful here - let me explain."
                ],
                "crisis_intervention": [
                    "Right now, let's focus on keeping you safe. What coping skills have helped before?",
                    "I'm concerned about your safety. Let's create a plan together.",
                    "These feelings are temporary, even though they feel overwhelming right now.",
                    "Let's use some grounding techniques to help you feel more stable."
                ]
            },
            "PSYCHODYNAMIC": {
                "exploration": [
                    "I'm curious about what this reminds you of from your past.",
                    "What comes to mind when you think about this pattern?",
                    "Tell me more about what this brings up for you.",
                    "I wonder if there's a deeper meaning to this experience."
                ],
                "interpretation": [
                    "I notice a pattern in how you relate to authority figures.",
                    "It seems like you might be experiencing some of those early feelings here with me.",
                    "This pattern of pushing people away when you need them most - where have we seen this before?",
                    "I wonder if this defense mechanism served you well in the past but might not be helpful now."
                ],
                "reflection": [
                    "You seem to have mixed feelings about that.",
                    "I hear both anger and sadness in what you're sharing.",
                    "There's something important in what you just said.",
                    "I notice you became quiet when we touched on that topic."
                ]
            },
            "HUMANISTIC": {
                "validation": [
                    "I hear how much pain you're in right now.",
                    "Your experience is valid and important.",
                    "Thank you for trusting me with something so personal.",
                    "I can see your strength even in this difficult moment."
                ],
                "reflection": [
                    "It sounds like you're feeling...",
                    "What I'm hearing is...",
                    "You seem to be saying that...",
                    "If I understand correctly, you're experiencing..."
                ],
                "exploration": [
                    "What does that mean to you?",
                    "How does that sit with you?",
                    "What's that like for you?",
                    "Tell me more about that experience."
                ]
            }
        }

    def _initialize_rationales(self) -> Dict[str, Dict[str, str]]:
        """Initialize clinical rationales for therapeutic techniques."""
        return {
            "CBT": {
                "assessment": "Gathering specific information about symptom patterns, triggers, and duration to inform case conceptualization and treatment planning.",
                "cognitive_restructuring": "Challenging cognitive distortions and helping client develop more balanced, realistic thinking patterns consistent with CBT principles.",
                "behavioral_activation": "Increasing engagement in meaningful activities to combat depression and improve mood through behavioral change.",
                "psychoeducation": "Providing evidence-based information to normalize experience and increase understanding of symptoms and treatment."},
            "DBT": {
                "validation": "Using radical acceptance and validation to reduce emotional dysregulation and build therapeutic alliance.",
                "skill_building": "Teaching concrete DBT skills for distress tolerance, emotion regulation, and interpersonal effectiveness.",
                "crisis_intervention": "Prioritizing safety and stabilization using DBT crisis survival strategies and grounding techniques."},
            "PSYCHODYNAMIC": {
                "exploration": "Encouraging free association and exploration of unconscious material to increase insight and self-awareness.",
                "interpretation": "Offering interpretations of unconscious patterns, defenses, and transference to promote psychological insight.",
                "reflection": "Reflecting emotional content and process to deepen awareness of internal experience."},
            "HUMANISTIC": {
                "validation": "Providing unconditional positive regard and empathetic understanding to facilitate self-acceptance.",
                "reflection": "Using accurate empathy and reflection to help client feel heard and understood.",
                "exploration": "Facilitating client's own exploration and discovery through open-ended, non-directive questioning."}}

    def _analyze_client_content(self, content: str) -> Dict[str, any]:
        """Analyze client content to determine appropriate intervention."""
        analysis = {
            "emotional_intensity": self._assess_emotional_intensity(content),
            "crisis_indicators": self._detect_crisis_indicators(content),
            "cognitive_distortions": self._identify_cognitive_distortions(content),
            "emotional_themes": self._extract_emotional_themes(content),
            "behavioral_patterns": self._identify_behavioral_patterns(content)}
        return analysis

    def _assess_emotional_intensity(self, content: str) -> str:
        """Assess emotional intensity from client content."""
        high_intensity_words = [
            "overwhelming",
            "unbearable",
            "can't take",
            "desperate",
            "hopeless"]
        moderate_intensity_words = [
            "difficult",
            "struggling",
            "hard",
            "challenging",
            "upset"]

        content_lower = content.lower()

        if any(word in content_lower for word in high_intensity_words):
            return "high"
        elif any(word in content_lower for word in moderate_intensity_words):
            return "moderate"
        else:
            return "low"

    def _detect_crisis_indicators(self, content: str) -> List[str]:
        """Detect crisis indicators in client content."""
        crisis_patterns = {
            "suicidal_ideation": [
                "want to die",
                "kill myself",
                "end it all",
                "not worth living"],
            "self_harm": [
                "hurt myself",
                "cut myself",
                "self-harm",
                "punish myself"],
            "substance_abuse": [
                "drinking too much",
                "using drugs",
                "getting high",
                "numbing"],
            "psychosis": [
                "hearing voices",
                "seeing things",
                "not real",
                "paranoid"]}

        indicators = []
        content_lower = content.lower()

        for indicator_type, patterns in crisis_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                indicators.append(indicator_type)

        return indicators

    def _identify_cognitive_distortions(self, content: str) -> List[str]:
        """Identify cognitive distortions in client content."""
        distortion_patterns = {
            "all_or_nothing": [
                "always",
                "never",
                "completely",
                "totally",
                "everything",
                "nothing"],
            "catastrophizing": [
                "disaster",
                "terrible",
                "awful",
                "worst",
                "catastrophe"],
            "mind_reading": [
                "they think",
                "he thinks",
                "she thinks",
                "everyone thinks"],
            "fortune_telling": [
                "will never",
                "going to fail",
                "won't work",
                "bound to"]}

        distortions = []
        content_lower = content.lower()

        for distortion_type, patterns in distortion_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                distortions.append(distortion_type)

        return distortions

    def _extract_emotional_themes(self, content: str) -> List[str]:
        """Extract primary emotional themes from content."""
        emotion_patterns = {
            "depression": ["sad", "down", "depressed", "hopeless", "empty", "worthless"],
            "anxiety": ["worried", "anxious", "nervous", "scared", "panic", "fear"],
            "anger": ["angry", "mad", "furious", "rage", "irritated", "frustrated"],
            "shame": ["ashamed", "embarrassed", "humiliated", "guilty", "bad person"],
            "grief": ["loss", "miss", "gone", "died", "grief", "mourning"]
        }

        themes = []
        content_lower = content.lower()

        for theme, patterns in emotion_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                themes.append(theme)

        return themes

    def _identify_behavioral_patterns(self, content: str) -> List[str]:
        """Identify behavioral patterns mentioned in content."""
        behavior_patterns = {
            "avoidance": [
                "avoid",
                "stay away",
                "don't go",
                "skip",
                "cancel"],
            "isolation": [
                "alone",
                "isolate",
                "withdraw",
                "don't see people",
                "stay home"],
            "rumination": [
                "can't stop thinking",
                "keep thinking",
                "obsessing",
                "replaying"],
            "procrastination": [
                "put off",
                "delay",
                "postpone",
                "can't start",
                "avoiding"]}

        patterns = []
        content_lower = content.lower()

        for pattern_type, indicators in behavior_patterns.items():
            if any(indicator in content_lower for indicator in indicators):
                patterns.append(pattern_type)

        return patterns

    def _select_intervention_type(self,
                                  analysis: Dict[str,
                                                 any],
                                  clinical_context: ClinicalContext,
                                  modality: TherapeuticModality) -> InterventionType:
        """Select appropriate intervention type based on analysis."""

        # Crisis intervention takes priority
        if analysis["crisis_indicators"]:
            return InterventionType.CRISIS_INTERVENTION

        # High emotional intensity may need validation first
        if analysis["emotional_intensity"] == "high":
            return InterventionType.VALIDATION

        # Modality-specific intervention selection
        if modality == TherapeuticModality.CBT:
            if analysis["cognitive_distortions"]:
                return InterventionType.COGNITIVE_RESTRUCTURING
            elif analysis["behavioral_patterns"]:
                return InterventionType.BEHAVIORAL_ACTIVATION
            else:
                return InterventionType.ASSESSMENT

        elif modality == TherapeuticModality.DBT:
            if analysis["emotional_intensity"] in ["high", "moderate"]:
                return InterventionType.SKILL_BUILDING
            else:
                return InterventionType.VALIDATION

        elif modality == TherapeuticModality.PSYCHODYNAMIC:
            return InterventionType.EXPLORATION

        else:  # Humanistic
            return InterventionType.REFLECTION

    def generate_response(self, client_content: str,
                          clinical_context: ClinicalContext,
                          modality: TherapeuticModality,
                          session_number: int = 1) -> TherapistResponse:
        """Generate complete therapist response with clinical rationale."""

        # Analyze client content
        analysis = self._analyze_client_content(client_content)

        # Select intervention type
        intervention_type = self._select_intervention_type(
            analysis, clinical_context, modality)

        # Get modality key
        modality_key = modality.value.upper().replace("_", "")
        if modality_key not in self.response_templates:
            modality_key = "CBT"  # Default fallback

        # Get intervention key
        intervention_key = intervention_type.value
        if intervention_key not in self.response_templates[modality_key]:
            intervention_key = list(
                self.response_templates[modality_key].keys())[0]

        # Select response template
        templates = self.response_templates[modality_key][intervention_key]
        response_content = random.choice(templates)

        # Get clinical rationale
        rationale = self.technique_rationales.get(
            modality_key,
            {}).get(
            intervention_key,
            "Therapeutic intervention selected based on client presentation and clinical context.")

        # Enhance rationale with specific analysis
        enhanced_rationale = self._enhance_rationale(
            rationale, analysis, clinical_context)

        # Generate technique name
        technique = f"{modality_key.lower()}_{intervention_key}"

        # Calculate confidence score
        confidence = self._calculate_confidence(
            analysis, clinical_context, modality)

        # Identify contraindications
        contraindications = self._identify_contraindications(
            analysis, intervention_type)

        # Generate follow-up suggestions
        follow_ups = self._generate_follow_ups(analysis, intervention_type)

        return TherapistResponse(
            content=response_content,
            clinical_rationale=enhanced_rationale,
            therapeutic_technique=technique,
            intervention_type=intervention_type,
            confidence_score=confidence,
            contraindications=contraindications,
            follow_up_suggestions=follow_ups
        )

    def _enhance_rationale(self, base_rationale: str, analysis: Dict[str, any],
                           clinical_context: ClinicalContext) -> str:
        """Enhance rationale with specific analysis details."""
        enhancements = []

        if analysis["crisis_indicators"]:
            enhancements.append(
                f"Crisis indicators detected: {', '.join(analysis['crisis_indicators'])}")

        if analysis["emotional_intensity"] == "high":
            enhancements.append(
                "High emotional intensity requires stabilization and validation")

        if analysis["cognitive_distortions"]:
            enhancements.append(
                f"Cognitive distortions identified: {', '.join(analysis['cognitive_distortions'])}")

        if clinical_context.severity_level == ClinicalSeverity.SEVERE:
            enhancements.append(
                "Severe symptom presentation requires careful intervention selection")

        if enhancements:
            return f"{base_rationale} Specific considerations: {'; '.join(enhancements)}."

        return base_rationale

    def _calculate_confidence(self, analysis: Dict[str, any],
                              clinical_context: ClinicalContext,
                              modality: TherapeuticModality) -> float:
        """Calculate confidence score for response appropriateness."""
        base_confidence = 0.7

        # Increase confidence for clear indicators
        if analysis["crisis_indicators"]:
            base_confidence += 0.2  # Crisis intervention is clearly indicated

        if analysis["cognitive_distortions"] and modality == TherapeuticModality.CBT:
            base_confidence += 0.1  # CBT well-suited for cognitive distortions

        if analysis["emotional_intensity"] == "high" and modality == TherapeuticModality.DBT:
            base_confidence += 0.1  # DBT good for emotional dysregulation

        # Decrease confidence for complexity
        if len(clinical_context.dsm5_categories) > 2:
            base_confidence -= 0.1  # Multiple diagnoses increase complexity

        if clinical_context.severity_level == ClinicalSeverity.CRISIS:
            base_confidence -= 0.1  # Crisis situations are inherently complex

        return min(1.0, max(0.3, base_confidence))

    def _identify_contraindications(self, analysis: Dict[str, any],
                                    intervention_type: InterventionType) -> List[str]:
        """Identify contraindications for selected intervention."""
        contraindications = []

        if intervention_type == InterventionType.COGNITIVE_RESTRUCTURING:
            if analysis["emotional_intensity"] == "high":
                contraindications.append(
                    "High emotional intensity may interfere with cognitive work")
            if "psychosis" in analysis["crisis_indicators"]:
                contraindications.append(
                    "Psychotic symptoms present - reality testing impaired")

        if intervention_type == InterventionType.EXPLORATION:
            if analysis["crisis_indicators"]:
                contraindications.append(
                    "Crisis indicators present - stabilization needed first")

        return contraindications

    def _generate_follow_ups(self, analysis: Dict[str, any],
                             intervention_type: InterventionType) -> List[str]:
        """Generate follow-up suggestions."""
        follow_ups = []

        if analysis["crisis_indicators"]:
            follow_ups.append("Assess safety and create safety plan")
            follow_ups.append("Consider need for higher level of care")

        if analysis["emotional_intensity"] == "high":
            follow_ups.append(
                "Monitor emotional regulation throughout session")
            follow_ups.append("Teach grounding techniques if needed")

        if intervention_type == InterventionType.COGNITIVE_RESTRUCTURING:
            follow_ups.append("Assign thought record homework")
            follow_ups.append("Practice identifying cognitive distortions")

        if intervention_type == InterventionType.SKILL_BUILDING:
            follow_ups.append("Practice skills between sessions")
            follow_ups.append("Review skill effectiveness next session")

        return follow_ups


def main():
    """Test the therapist response generator."""
    generator = TherapistResponseGenerator()

    # Test scenarios
    test_scenarios = [
        {
            "client_content": "I've been feeling really depressed lately. Nothing seems to matter anymore and I can't get out of bed.",
            "modality": TherapeuticModality.CBT,
            "clinical_context": ClinicalContext(
                dsm5_categories=["Major Depressive Disorder"],
                severity_level=ClinicalSeverity.MODERATE)},
        {
            "client_content": "I'm so angry I could explode! I want to hurt myself when I feel this way.",
            "modality": TherapeuticModality.DBT,
            "clinical_context": ClinicalContext(
                dsm5_categories=["Borderline Personality Disorder"],
                severity_level=ClinicalSeverity.SEVERE,
                risk_factors=[
                    "self-harm",
                    "emotional dysregulation"])}]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n=== Test Scenario {i} ===")
        print(f"Client: {scenario['client_content']}")

        response = generator.generate_response(
            scenario["client_content"],
            scenario["clinical_context"],
            scenario["modality"]
        )

        print(f"\nTherapist: {response.content}")
        print(f"Technique: {response.therapeutic_technique}")
        print(f"Intervention: {response.intervention_type.value}")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Rationale: {response.clinical_rationale}")

        if response.contraindications:
            print(
                f"Contraindications: {', '.join(response.contraindications)}")

        if response.follow_up_suggestions:
            print(f"Follow-ups: {', '.join(response.follow_up_suggestions)}")


if __name__ == "__main__":
    main()
