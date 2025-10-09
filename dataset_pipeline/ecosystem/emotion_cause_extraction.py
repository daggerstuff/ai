#!/usr/bin/env python3
"""
Task 6.15: Emotion Cause Extraction and Therapeutic Intervention Mapping (RECCON)

This module implements sophisticated emotion cause extraction and maps identified
causes to specific therapeutic interventions for targeted treatment approaches.

Strategic Goal: Identify root causes of emotional distress and provide precise
intervention recommendations for maximum therapeutic effectiveness.
"""

import logging
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class EmotionType(Enum):
    """Types of emotions for cause extraction."""
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    ANXIETY = "anxiety"
    GUILT = "guilt"
    SHAME = "shame"
    FRUSTRATION = "frustration"
    HOPELESSNESS = "hopelessness"
    LONELINESS = "loneliness"
    OVERWHELM = "overwhelm"


class CauseCategory(Enum):
    """Categories of emotion causes."""
    INTERPERSONAL = "interpersonal"
    SITUATIONAL = "situational"
    COGNITIVE = "cognitive"
    PHYSIOLOGICAL = "physiological"
    BEHAVIORAL = "behavioral"
    ENVIRONMENTAL = "environmental"
    EXISTENTIAL = "existential"
    TRAUMATIC = "traumatic"


class InterventionType(Enum):
    """Types of therapeutic interventions."""
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"
    BEHAVIORAL_ACTIVATION = "behavioral_activation"
    EXPOSURE_THERAPY = "exposure_therapy"
    MINDFULNESS = "mindfulness"
    INTERPERSONAL_SKILLS = "interpersonal_skills"
    EMOTION_REGULATION = "emotion_regulation"
    TRAUMA_PROCESSING = "trauma_processing"
    PROBLEM_SOLVING = "problem_solving"
    RELAXATION_TECHNIQUES = "relaxation_techniques"
    SOCIAL_SUPPORT = "social_support"


@dataclass
class EmotionCause:
    """Represents an identified cause of an emotion."""
    emotion: EmotionType
    cause_text: str
    cause_category: CauseCategory
    confidence_score: float
    contextual_factors: list[str]
    severity_indicators: list[str]
    temporal_markers: list[str]


@dataclass
class InterventionMapping:
    """Maps emotion causes to specific interventions."""
    cause: EmotionCause
    primary_intervention: InterventionType
    secondary_interventions: list[InterventionType]
    intervention_rationale: str
    expected_effectiveness: float
    implementation_priority: str  # immediate, short_term, long_term
    session_recommendations: list[str]


@dataclass
class EmotionCauseAnalysis:
    """Complete emotion cause extraction and intervention analysis."""
    conversation_id: str
    identified_emotions: list[EmotionType]
    emotion_causes: list[EmotionCause]
    intervention_mappings: list[InterventionMapping]
    cause_interaction_patterns: dict[str, Any]
    therapeutic_focus_areas: list[str]
    treatment_sequence: list[str]
    analysis_confidence: float
    timestamp: str


class EmotionCauseExtractor:
    """Extracts emotion causes from therapeutic conversations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Emotion detection patterns
        self.emotion_patterns = {
            EmotionType.ANGER: [
                r"(angry|mad|furious|rage|irritated|pissed|frustrated with)",
                r"(makes me angry|so mad|really irritated)"
            ],
            EmotionType.SADNESS: [
                r"(sad|depressed|down|blue|melancholy|heartbroken)",
                r"(makes me sad|feel down|brings me down)"
            ],
            EmotionType.FEAR: [
                r"(afraid|scared|terrified|frightened|fearful)",
                r"(makes me afraid|scares me|frightens me)"
            ],
            EmotionType.ANXIETY: [
                r"(anxious|worried|nervous|stressed|panicked)",
                r"(makes me anxious|worry about|stressed about)"
            ],
            EmotionType.GUILT: [
                r"(guilty|ashamed|regret|should have|shouldn\'t have)",
                r"(feel guilty|makes me feel bad|regret that)"
            ],
            EmotionType.HOPELESSNESS: [
                r"(hopeless|helpless|pointless|no point|give up)",
                r"(feel hopeless|seems pointless|no hope)"
            ]
        }

        # Cause extraction patterns
        self.cause_patterns = {
            CauseCategory.INTERPERSONAL: [
                r"(because|when|after) (he|she|they|my \w+) (said|did|didn\'t)",
                r"(relationship|friend|family|partner|spouse) (problems|issues|conflict)",
                r"(argument|fight|disagreement) with"
            ],
            CauseCategory.SITUATIONAL: [
                r"(because of|due to|after) (work|job|school|money|financial)",
                r"(situation|circumstances|events) (that|which|where)",
                r"(happened|occurred|took place) (when|where|that)"
            ],
            CauseCategory.COGNITIVE: [
                r"(thinking|thoughts|believe|convinced) (that|about)",
                r"(keep thinking|can\'t stop thinking|thoughts about)",
                r"(worry|worried|concern) (that|about)"
            ],
            CauseCategory.BEHAVIORAL: [
                r"(because I|when I|after I) (did|didn\'t|said|acted)",
                r"(my behavior|what I did|how I acted)",
                r"(habit|pattern|tendency) (of|to)"
            ],
            CauseCategory.TRAUMATIC: [
                r"(trauma|abuse|accident|attack|violence)",
                r"(happened to me|experienced|went through)",
                r"(flashback|nightmare|triggered by)"
            ]
        }

        # Intervention mapping rules
        self.intervention_mappings = self._initialize_intervention_mappings()

        # Analysis tracking
        self.extraction_stats = {
            "total_extractions": 0,
            "emotion_distribution": defaultdict(int),
            "cause_distribution": defaultdict(int),
            "intervention_recommendations": defaultdict(int)
        }

    def _initialize_intervention_mappings(self) -> dict[tuple[EmotionType, CauseCategory], dict[str, Any]]:
        """Initialize intervention mapping rules."""
        return {
            (EmotionType.ANGER, CauseCategory.INTERPERSONAL): {
                "primary": InterventionType.INTERPERSONAL_SKILLS,
                "secondary": [InterventionType.EMOTION_REGULATION, InterventionType.PROBLEM_SOLVING],
                "effectiveness": 0.8,
                "priority": "immediate"
            },
            (EmotionType.ANGER, CauseCategory.COGNITIVE): {
                "primary": InterventionType.COGNITIVE_RESTRUCTURING,
                "secondary": [InterventionType.EMOTION_REGULATION],
                "effectiveness": 0.85,
                "priority": "short_term"
            },
            (EmotionType.SADNESS, CauseCategory.SITUATIONAL): {
                "primary": InterventionType.PROBLEM_SOLVING,
                "secondary": [InterventionType.BEHAVIORAL_ACTIVATION, InterventionType.SOCIAL_SUPPORT],
                "effectiveness": 0.75,
                "priority": "short_term"
            },
            (EmotionType.SADNESS, CauseCategory.COGNITIVE): {
                "primary": InterventionType.COGNITIVE_RESTRUCTURING,
                "secondary": [InterventionType.BEHAVIORAL_ACTIVATION],
                "effectiveness": 0.8,
                "priority": "short_term"
            },
            (EmotionType.ANXIETY, CauseCategory.COGNITIVE): {
                "primary": InterventionType.COGNITIVE_RESTRUCTURING,
                "secondary": [InterventionType.RELAXATION_TECHNIQUES, InterventionType.MINDFULNESS],
                "effectiveness": 0.85,
                "priority": "immediate"
            },
            (EmotionType.ANXIETY, CauseCategory.SITUATIONAL): {
                "primary": InterventionType.EXPOSURE_THERAPY,
                "secondary": [InterventionType.RELAXATION_TECHNIQUES, InterventionType.PROBLEM_SOLVING],
                "effectiveness": 0.8,
                "priority": "short_term"
            },
            (EmotionType.FEAR, CauseCategory.TRAUMATIC): {
                "primary": InterventionType.TRAUMA_PROCESSING,
                "secondary": [InterventionType.RELAXATION_TECHNIQUES, InterventionType.MINDFULNESS],
                "effectiveness": 0.75,
                "priority": "long_term"
            },
            (EmotionType.GUILT, CauseCategory.BEHAVIORAL): {
                "primary": InterventionType.COGNITIVE_RESTRUCTURING,
                "secondary": [InterventionType.PROBLEM_SOLVING],
                "effectiveness": 0.8,
                "priority": "short_term"
            },
            (EmotionType.HOPELESSNESS, CauseCategory.COGNITIVE): {
                "primary": InterventionType.COGNITIVE_RESTRUCTURING,
                "secondary": [InterventionType.BEHAVIORAL_ACTIVATION, InterventionType.SOCIAL_SUPPORT],
                "effectiveness": 0.7,
                "priority": "immediate"
            }
        }

    def extract_emotion_causes(self, conversation: dict[str, Any]) -> EmotionCauseAnalysis:
        """Extract emotion causes and map to interventions."""
        conversation_id = conversation.get("id", "unknown")

        # Extract conversation text
        text = self._extract_conversation_text(conversation)

        # Step 1: Identify emotions
        identified_emotions = self._identify_emotions(text)

        # Step 2: Extract causes for each emotion
        emotion_causes = []
        for emotion in identified_emotions:
            causes = self._extract_causes_for_emotion(text, emotion)
            emotion_causes.extend(causes)

        # Step 3: Map causes to interventions
        intervention_mappings = []
        for cause in emotion_causes:
            mapping = self._map_cause_to_intervention(cause)
            if mapping:
                intervention_mappings.append(mapping)

        # Step 4: Analyze cause interaction patterns
        interaction_patterns = self._analyze_cause_interactions(emotion_causes)

        # Step 5: Identify therapeutic focus areas
        focus_areas = self._identify_focus_areas(emotion_causes, intervention_mappings)

        # Step 6: Generate treatment sequence
        treatment_sequence = self._generate_treatment_sequence(intervention_mappings)

        # Step 7: Calculate analysis confidence
        analysis_confidence = self._calculate_analysis_confidence(
            emotion_causes, intervention_mappings
        )

        analysis = EmotionCauseAnalysis(
            conversation_id=conversation_id,
            identified_emotions=identified_emotions,
            emotion_causes=emotion_causes,
            intervention_mappings=intervention_mappings,
            cause_interaction_patterns=interaction_patterns,
            therapeutic_focus_areas=focus_areas,
            treatment_sequence=treatment_sequence,
            analysis_confidence=analysis_confidence,
            timestamp=datetime.now().isoformat()
        )

        # Update statistics
        self._update_extraction_stats(analysis)

        return analysis

    def _extract_conversation_text(self, conversation: dict[str, Any]) -> str:
        """Extract text content from conversation structure."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])
        elif "input" in conversation and "output" in conversation:
            text_parts.extend([conversation["input"], conversation["output"]])
        elif "text" in conversation:
            text_parts.append(conversation["text"])

        return " ".join(text_parts).lower()

    def _identify_emotions(self, text: str) -> list[EmotionType]:
        """Identify emotions present in the text."""
        identified_emotions = []

        for emotion, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    identified_emotions.append(emotion)
                    break  # Found this emotion, move to next

        return identified_emotions

    def _extract_causes_for_emotion(self, text: str, emotion: EmotionType) -> list[EmotionCause]:
        """Extract causes for a specific emotion."""
        causes = []

        # Look for cause patterns near emotion mentions
        emotion_patterns = self.emotion_patterns[emotion]

        for emotion_pattern in emotion_patterns:
            emotion_matches = list(re.finditer(emotion_pattern, text, re.IGNORECASE))

            for emotion_match in emotion_matches:
                # Look for causes in surrounding context
                start_pos = max(0, emotion_match.start() - 100)
                end_pos = min(len(text), emotion_match.end() + 100)
                context = text[start_pos:end_pos]

                # Check each cause category
                for cause_category, cause_patterns in self.cause_patterns.items():
                    for cause_pattern in cause_patterns:
                        cause_match = re.search(cause_pattern, context, re.IGNORECASE)
                        if cause_match:
                            # Extract the cause text
                            cause_text = cause_match.group(0)

                            # Calculate confidence based on proximity and pattern strength
                            distance = abs(emotion_match.start() - (start_pos + cause_match.start()))
                            confidence = max(0.3, 1.0 - (distance / 200))

                            # Extract contextual factors
                            contextual_factors = self._extract_contextual_factors(context)

                            # Extract severity indicators
                            severity_indicators = self._extract_severity_indicators(context)

                            # Extract temporal markers
                            temporal_markers = self._extract_temporal_markers(context)

                            cause = EmotionCause(
                                emotion=emotion,
                                cause_text=cause_text,
                                cause_category=cause_category,
                                confidence_score=confidence,
                                contextual_factors=contextual_factors,
                                severity_indicators=severity_indicators,
                                temporal_markers=temporal_markers
                            )

                            causes.append(cause)

        return causes

    def _extract_contextual_factors(self, context: str) -> list[str]:
        """Extract contextual factors from the surrounding text."""
        factors = []

        factor_patterns = {
            "frequency": r"(always|often|sometimes|rarely|never|constantly)",
            "duration": r"(for \w+ (days|weeks|months|years)|since|long time)",
            "intensity": r"(very|extremely|really|quite|somewhat|a little)",
            "social": r"(alone|with others|in public|at home|at work)",
            "timing": r"(morning|afternoon|evening|night|weekend|weekday)"
        }

        for factor_type, pattern in factor_patterns.items():
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                factors.extend([f"{factor_type}: {match[0] if isinstance(match, tuple) else match}" for match in matches[:2]])

        return factors

    def _extract_severity_indicators(self, context: str) -> list[str]:
        """Extract severity indicators from the context."""
        severity_patterns = [
            r"(unbearable|overwhelming|intense|severe|extreme)",
            r"(can\'t handle|too much|breaking point|crisis)",
            r"(getting worse|escalating|out of control)"
        ]

        indicators = []
        for pattern in severity_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            indicators.extend(matches)

        return indicators[:3]  # Limit to top 3

    def _extract_temporal_markers(self, context: str) -> list[str]:
        """Extract temporal markers from the context."""
        temporal_patterns = [
            r"(yesterday|today|tomorrow|last week|next week)",
            r"(when|after|before|during|since|until)",
            r"(started|began|happened|occurred) (yesterday|today|last \w+)"
        ]

        markers = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            markers.extend([match[0] if isinstance(match, tuple) else match for match in matches])

        return markers[:3]  # Limit to top 3

    def _map_cause_to_intervention(self, cause: EmotionCause) -> InterventionMapping | None:
        """Map an emotion cause to appropriate interventions."""
        mapping_key = (cause.emotion, cause.cause_category)

        if mapping_key in self.intervention_mappings:
            mapping_info = self.intervention_mappings[mapping_key]

            # Generate intervention rationale
            rationale = f"Addressing {cause.emotion.value} caused by {cause.cause_category.value} factors through {mapping_info['primary'].value}"

            # Generate session recommendations
            session_recommendations = self._generate_session_recommendations(
                cause, mapping_info["primary"], mapping_info["secondary"]
            )

            return InterventionMapping(
                cause=cause,
                primary_intervention=mapping_info["primary"],
                secondary_interventions=mapping_info["secondary"],
                intervention_rationale=rationale,
                expected_effectiveness=mapping_info["effectiveness"],
                implementation_priority=mapping_info["priority"],
                session_recommendations=session_recommendations
            )

        return None

    def _generate_session_recommendations(self, cause: EmotionCause,
                                        primary: InterventionType,
                                        secondary: list[InterventionType]) -> list[str]:
        """Generate specific session recommendations."""
        recommendations = []

        # Primary intervention recommendations
        primary_recs = {
            InterventionType.COGNITIVE_RESTRUCTURING: [
                "Identify and challenge negative thought patterns",
                "Practice thought record exercises",
                "Develop balanced thinking strategies"
            ],
            InterventionType.BEHAVIORAL_ACTIVATION: [
                "Schedule pleasant activities",
                "Increase activity levels gradually",
                "Monitor mood-activity relationships"
            ],
            InterventionType.EMOTION_REGULATION: [
                "Practice emotion identification skills",
                "Learn distress tolerance techniques",
                "Develop healthy coping strategies"
            ],
            InterventionType.INTERPERSONAL_SKILLS: [
                "Practice assertiveness techniques",
                "Improve communication skills",
                "Address relationship conflicts"
            ],
            InterventionType.RELAXATION_TECHNIQUES: [
                "Learn progressive muscle relaxation",
                "Practice deep breathing exercises",
                "Develop mindfulness skills"
            ]
        }

        if primary in primary_recs:
            recommendations.extend(primary_recs[primary][:2])

        # Add secondary intervention recommendations
        for intervention in secondary[:1]:  # Just first secondary
            if intervention in primary_recs:
                recommendations.append(primary_recs[intervention][0])

        return recommendations

    def _analyze_cause_interactions(self, emotion_causes: list[EmotionCause]) -> dict[str, Any]:
        """Analyze interactions between different emotion causes."""
        if len(emotion_causes) < 2:
            return {"interaction_complexity": "simple", "cause_clusters": []}

        # Group causes by category
        cause_categories = defaultdict(list)
        for cause in emotion_causes:
            cause_categories[cause.cause_category.value].append(cause)

        # Identify cause clusters
        cause_clusters = []
        for category, causes in cause_categories.items():
            if len(causes) > 1:
                cause_clusters.append({
                    "category": category,
                    "cause_count": len(causes),
                    "emotions_involved": list({c.emotion.value for c in causes})
                })

        # Determine interaction complexity
        complexity = "simple"
        if len(cause_categories) > 2:
            complexity = "moderate"
        if len(cause_categories) > 3 or any(len(causes) > 2 for causes in cause_categories.values()):
            complexity = "complex"

        return {
            "interaction_complexity": complexity,
            "cause_clusters": cause_clusters,
            "category_distribution": {cat: len(causes) for cat, causes in cause_categories.items()}
        }

    def _identify_focus_areas(self, emotion_causes: list[EmotionCause],
                            intervention_mappings: list[InterventionMapping]) -> list[str]:
        """Identify primary therapeutic focus areas."""

        # Count intervention types
        intervention_counts = defaultdict(int)
        for mapping in intervention_mappings:
            intervention_counts[mapping.primary_intervention.value] += 1
            for secondary in mapping.secondary_interventions:
                intervention_counts[secondary.value] += 0.5

        # Get top focus areas
        sorted_interventions = sorted(intervention_counts.items(), key=lambda x: x[1], reverse=True)
        return [intervention for intervention, count in sorted_interventions[:3]]


    def _generate_treatment_sequence(self, intervention_mappings: list[InterventionMapping]) -> list[str]:
        """Generate recommended treatment sequence."""
        if not intervention_mappings:
            return []

        # Group by priority
        immediate = [m for m in intervention_mappings if m.implementation_priority == "immediate"]
        short_term = [m for m in intervention_mappings if m.implementation_priority == "short_term"]
        long_term = [m for m in intervention_mappings if m.implementation_priority == "long_term"]

        sequence = []

        # Add immediate interventions
        if immediate:
            sequence.append(f"Immediate: {immediate[0].primary_intervention.value}")

        # Add short-term interventions
        if short_term:
            sequence.append(f"Short-term: {short_term[0].primary_intervention.value}")

        # Add long-term interventions
        if long_term:
            sequence.append(f"Long-term: {long_term[0].primary_intervention.value}")

        return sequence

    def _calculate_analysis_confidence(self, emotion_causes: list[EmotionCause],
                                     intervention_mappings: list[InterventionMapping]) -> float:
        """Calculate overall analysis confidence."""
        if not emotion_causes:
            return 0.0

        # Average cause confidence
        cause_confidence = statistics.mean([c.confidence_score for c in emotion_causes])

        # Mapping coverage (how many causes have interventions)
        mapping_coverage = len(intervention_mappings) / len(emotion_causes) if emotion_causes else 0

        # Overall confidence
        overall_confidence = (cause_confidence * 0.7) + (mapping_coverage * 0.3)

        return min(overall_confidence, 1.0)

    def _update_extraction_stats(self, analysis: EmotionCauseAnalysis):
        """Update extraction statistics."""
        self.extraction_stats["total_extractions"] += 1

        for emotion in analysis.identified_emotions:
            self.extraction_stats["emotion_distribution"][emotion.value] += 1

        for cause in analysis.emotion_causes:
            self.extraction_stats["cause_distribution"][cause.cause_category.value] += 1

        for mapping in analysis.intervention_mappings:
            self.extraction_stats["intervention_recommendations"][mapping.primary_intervention.value] += 1

    def get_extraction_statistics(self) -> dict[str, Any]:
        """Get emotion cause extraction statistics."""
        total = self.extraction_stats["total_extractions"]
        if total == 0:
            return {}

        return {
            "total_extractions": total,
            "emotion_distribution": dict(self.extraction_stats["emotion_distribution"]),
            "cause_distribution": dict(self.extraction_stats["cause_distribution"]),
            "intervention_recommendations": dict(self.extraction_stats["intervention_recommendations"])
        }


# Example usage and testing
def main():
    """Example usage of the emotion cause extraction system."""

    # Create extractor
    extractor = EmotionCauseExtractor()

    # Example conversation with clear emotion-cause relationships
    test_conversation = {
        "id": "emotion_cause_test",
        "messages": [
            {
                "role": "client",
                "content": "I feel so angry because my partner never listens to me when I try to talk about our problems. It makes me sad that we keep having the same arguments over and over. I'm also anxious about what this means for our relationship because I keep thinking we might break up."
            },
            {
                "role": "therapist",
                "content": "I can hear how frustrated and worried you are about the communication patterns in your relationship. Let's explore what's happening and work on some strategies to improve this situation."
            }
        ]
    }


    # Perform analysis
    analysis = extractor.extract_emotion_causes(test_conversation)

    # Display results

    for _emotion in analysis.identified_emotions:
        pass

    for cause in analysis.emotion_causes:
        if cause.contextual_factors:
            pass

    for mapping in analysis.intervention_mappings:
        if mapping.session_recommendations:
            pass

    for _area in analysis.therapeutic_focus_areas:
        pass

    for _step in analysis.treatment_sequence:
        pass

    patterns = analysis.cause_interaction_patterns
    if patterns["cause_clusters"]:
        pass

    # Show statistics
    extractor.get_extraction_statistics()


if __name__ == "__main__":
    main()
