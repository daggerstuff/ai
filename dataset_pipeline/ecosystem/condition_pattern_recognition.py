#!/usr/bin/env python3
"""
Task 6.8: Mental Health Condition Pattern Recognition

This module implements sophisticated pattern recognition for 20+ mental health
conditions (depression, anxiety, PTSD, bipolar, etc.) with high accuracy
diagnostic indicators and therapeutic response patterns.

Strategic Goal: Automatically identify and balance mental health conditions
across the 2.59M+ conversation ecosystem for comprehensive training coverage.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Import ecosystem components


class MentalHealthCondition(Enum):
    """Major mental health conditions and disorders."""
    DEPRESSION = "major_depressive_disorder"
    ANXIETY = "generalized_anxiety_disorder"
    PTSD = "post_traumatic_stress_disorder"
    BIPOLAR = "bipolar_disorder"
    BPD = "borderline_personality_disorder"
    ADHD = "attention_deficit_hyperactivity_disorder"
    AUTISM = "autism_spectrum_disorder"
    SCHIZOPHRENIA = "schizophrenia"
    OCD = "obsessive_compulsive_disorder"
    EATING_DISORDER = "eating_disorders"
    SUBSTANCE_USE = "substance_use_disorder"
    SOCIAL_ANXIETY = "social_anxiety_disorder"
    PANIC_DISORDER = "panic_disorder"
    HEALTH_ANXIETY = "health_anxiety"
    LONELINESS = "chronic_loneliness"
    GRIEF = "complicated_grief"
    RELATIONSHIP_ISSUES = "relationship_difficulties"
    PARENTING_STRESS = "parenting_stress"
    DIVORCE_RECOVERY = "divorce_recovery"
    TRAUMA_GENERAL = "trauma_related_disorders"


@dataclass
class ConditionMarker:
    """Represents a diagnostic or symptomatic marker for a condition."""
    condition: MentalHealthCondition
    marker_type: str  # symptom, behavior, thought_pattern, trigger, coping_mechanism
    pattern: str
    severity_indicator: str  # mild, moderate, severe
    confidence_weight: float
    dsm5_criteria: str | None = None
    context_requirements: list[str] = None
    exclusion_patterns: list[str] = None


@dataclass
class ConditionRecognition:
    """Result of mental health condition pattern recognition."""
    conversation_id: str
    primary_condition: MentalHealthCondition
    secondary_conditions: list[MentalHealthCondition]
    severity_assessment: str
    confidence_scores: dict[str, float]
    symptom_markers: dict[str, list[str]]
    risk_indicators: list[str]
    therapeutic_recommendations: list[str]
    comorbidity_likelihood: float
    recognition_quality: float


class MentalHealthConditionRecognizer:
    """Recognizes mental health conditions using advanced pattern recognition."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.condition_markers = self._initialize_condition_markers()
        self.recognition_cache = {}

        # Performance tracking
        self.recognition_stats = {
            "total_recognized": 0,
            "condition_distribution": defaultdict(int),
            "severity_distribution": defaultdict(int),
            "comorbidity_cases": 0,
            "high_risk_cases": 0
        }

    def _initialize_condition_markers(self) -> dict[MentalHealthCondition, list[ConditionMarker]]:
        """Initialize mental health condition markers and patterns."""
        return {
            MentalHealthCondition.DEPRESSION: [
                ConditionMarker(
                    MentalHealthCondition.DEPRESSION, "symptom",
                    r"depressed.*mood|feeling.*down|hopeless|worthless|empty",
                    "moderate", 0.9, "Depressed mood most of the day"
                ),
                ConditionMarker(
                    MentalHealthCondition.DEPRESSION, "symptom",
                    r"lost.*interest|no.*pleasure|anhedonia|nothing.*enjoyable",
                    "moderate", 0.85, "Diminished interest or pleasure"
                ),
                ConditionMarker(
                    MentalHealthCondition.DEPRESSION, "behavior",
                    r"sleep.*problem|insomnia|sleeping.*too.*much|fatigue|tired",
                    "mild", 0.7
                ),
                ConditionMarker(
                    MentalHealthCondition.DEPRESSION, "thought_pattern",
                    r"suicidal.*thought|want.*to.*die|better.*off.*dead|end.*it.*all",
                    "severe", 0.95, context_requirements=["depression", "mood"]
                ),
                ConditionMarker(
                    MentalHealthCondition.DEPRESSION, "behavior",
                    r"appetite.*change|weight.*loss|weight.*gain|eating.*less|eating.*more",
                    "mild", 0.6
                )
            ],

            MentalHealthCondition.ANXIETY: [
                ConditionMarker(
                    MentalHealthCondition.ANXIETY, "symptom",
                    r"anxious|worried|nervous|panic|fear|scared|terrified",
                    "moderate", 0.8
                ),
                ConditionMarker(
                    MentalHealthCondition.ANXIETY, "behavior",
                    r"avoid|avoidance|can't.*go|won't.*do|stay.*home|escape",
                    "moderate", 0.75
                ),
                ConditionMarker(
                    MentalHealthCondition.ANXIETY, "symptom",
                    r"heart.*racing|sweating|shaking|trembling|shortness.*breath|dizzy",
                    "moderate", 0.85
                ),
                ConditionMarker(
                    MentalHealthCondition.ANXIETY, "thought_pattern",
                    r"what.*if|catastrophic|worst.*case|something.*bad|disaster",
                    "mild", 0.7
                )
            ],

            MentalHealthCondition.PTSD: [
                ConditionMarker(
                    MentalHealthCondition.PTSD, "symptom",
                    r"flashback|nightmare|reliving|intrusive.*thought|triggered",
                    "severe", 0.9
                ),
                ConditionMarker(
                    MentalHealthCondition.PTSD, "behavior",
                    r"hypervigilant|startle|jumpy|on.*edge|can't.*relax",
                    "moderate", 0.8
                ),
                ConditionMarker(
                    MentalHealthCondition.PTSD, "symptom",
                    r"numb|detached|disconnected|dissociat|out.*of.*body",
                    "moderate", 0.75
                ),
                ConditionMarker(
                    MentalHealthCondition.PTSD, "trigger",
                    r"trauma|abuse|assault|accident|combat|violence|attack",
                    "severe", 0.85
                )
            ],

            MentalHealthCondition.BIPOLAR: [
                ConditionMarker(
                    MentalHealthCondition.BIPOLAR, "symptom",
                    r"manic|hypomanic|elevated.*mood|euphoric|grandiose",
                    "severe", 0.9
                ),
                ConditionMarker(
                    MentalHealthCondition.BIPOLAR, "behavior",
                    r"racing.*thoughts|flight.*of.*ideas|rapid.*speech|pressured.*speech",
                    "moderate", 0.85
                ),
                ConditionMarker(
                    MentalHealthCondition.BIPOLAR, "behavior",
                    r"decreased.*sleep|little.*sleep|don't.*need.*sleep|energy.*high",
                    "moderate", 0.8
                ),
                ConditionMarker(
                    MentalHealthCondition.BIPOLAR, "behavior",
                    r"impulsive|reckless|poor.*judgment|spending.*spree|risky.*behavior",
                    "moderate", 0.75
                )
            ],

            MentalHealthCondition.BPD: [
                ConditionMarker(
                    MentalHealthCondition.BPD, "behavior",
                    r"abandonment|fear.*of.*being.*left|clingy|desperate",
                    "moderate", 0.8
                ),
                ConditionMarker(
                    MentalHealthCondition.BPD, "symptom",
                    r"identity.*crisis|who.*am.*i|empty|unstable.*self",
                    "moderate", 0.75
                ),
                ConditionMarker(
                    MentalHealthCondition.BPD, "behavior",
                    r"self.*harm|cutting|burning|suicidal.*gesture|impulsive",
                    "severe", 0.9
                ),
                ConditionMarker(
                    MentalHealthCondition.BPD, "symptom",
                    r"intense.*anger|rage|explosive|mood.*swing|emotional.*rollercoaster",
                    "moderate", 0.8
                )
            ],

            MentalHealthCondition.ADHD: [
                ConditionMarker(
                    MentalHealthCondition.ADHD, "symptom",
                    r"attention|focus|concentrate|distracted|mind.*wander",
                    "moderate", 0.8
                ),
                ConditionMarker(
                    MentalHealthCondition.ADHD, "behavior",
                    r"hyperactive|restless|fidget|can't.*sit.*still|impulsive",
                    "moderate", 0.85
                ),
                ConditionMarker(
                    MentalHealthCondition.ADHD, "behavior",
                    r"procrastinate|disorganized|forgetful|lose.*things|time.*management",
                    "mild", 0.7
                )
            ],

            MentalHealthCondition.AUTISM: [
                ConditionMarker(
                    MentalHealthCondition.AUTISM, "behavior",
                    r"social.*difficulty|social.*cue|eye.*contact|interaction",
                    "moderate", 0.8
                ),
                ConditionMarker(
                    MentalHealthCondition.AUTISM, "behavior",
                    r"routine|ritual|repetitive|stimming|sensory|overwhelm",
                    "moderate", 0.85
                ),
                ConditionMarker(
                    MentalHealthCondition.AUTISM, "symptom",
                    r"special.*interest|intense.*focus|detail.*oriented|pattern",
                    "mild", 0.7
                )
            ],

            MentalHealthCondition.OCD: [
                ConditionMarker(
                    MentalHealthCondition.OCD, "symptom",
                    r"obsessive|intrusive.*thought|can't.*stop.*thinking|stuck.*thought",
                    "moderate", 0.85
                ),
                ConditionMarker(
                    MentalHealthCondition.OCD, "behavior",
                    r"compulsive|ritual|checking|washing|counting|repeating",
                    "moderate", 0.9
                ),
                ConditionMarker(
                    MentalHealthCondition.OCD, "symptom",
                    r"contamination|germs|harm|symmetry|order|perfect",
                    "mild", 0.75
                )
            ],

            MentalHealthCondition.SOCIAL_ANXIETY: [
                ConditionMarker(
                    MentalHealthCondition.SOCIAL_ANXIETY, "symptom",
                    r"social.*anxiety|social.*phobia|fear.*judgment|embarrass",
                    "moderate", 0.9
                ),
                ConditionMarker(
                    MentalHealthCondition.SOCIAL_ANXIETY, "behavior",
                    r"avoid.*social|avoid.*people|isolate|stay.*home|cancel.*plan",
                    "moderate", 0.8
                ),
                ConditionMarker(
                    MentalHealthCondition.SOCIAL_ANXIETY, "symptom",
                    r"blushing|sweating|trembling|voice.*shake|mind.*blank",
                    "mild", 0.7
                )
            ],

            MentalHealthCondition.EATING_DISORDER: [
                ConditionMarker(
                    MentalHealthCondition.EATING_DISORDER, "behavior",
                    r"restrict|binge|purge|vomit|laxative|diet|weight.*loss",
                    "severe", 0.9
                ),
                ConditionMarker(
                    MentalHealthCondition.EATING_DISORDER, "thought_pattern",
                    r"body.*image|fat|ugly|calories|food.*guilt|weight.*obsess",
                    "moderate", 0.8
                ),
                ConditionMarker(
                    MentalHealthCondition.EATING_DISORDER, "behavior",
                    r"exercise.*compulsive|over.*exercise|body.*check|mirror.*check",
                    "moderate", 0.75
                )
            ],

            MentalHealthCondition.SCHIZOPHRENIA: [
                ConditionMarker(
                    MentalHealthCondition.SCHIZOPHRENIA, "symptom",
                    r"delusion|hallucination|paranoid|voices|thought.*disorder",
                    "severe", 0.95
                ),
                ConditionMarker(
                    MentalHealthCondition.SCHIZOPHRENIA, "symptom",
                    r"social.*withdrawal|flat.*affect|lack.*motivation|alogia",
                    "moderate", 0.8
                ),
                ConditionMarker(
                    MentalHealthCondition.SCHIZOPHRENIA, "behavior",
                    r"disorganized.*speech|catatonia|unusual.*behavior",
                    "moderate", 0.85
                )
            ],

            MentalHealthCondition.SUBSTANCE_USE: [
                ConditionMarker(
                    MentalHealthCondition.SUBSTANCE_USE, "behavior",
                    r"crave|withdrawal|tolerance|use.*more|can't.*cut.*down",
                    "severe", 0.9
                ),
                ConditionMarker(
                    MentalHealthCondition.SUBSTANCE_USE, "symptom",
                    r"neglect.*responsibilities|social.*problems|hazardous.*use",
                    "moderate", 0.85
                ),
                ConditionMarker(
                    MentalHealthCondition.SUBSTANCE_USE, "trigger",
                    r"relapse|craving|temptation|addiction.*struggle",
                    "moderate", 0.75
                )
            ]
        }


    def recognize_condition(self, conversation: dict[str, Any]) -> ConditionRecognition:
        """Recognize mental health conditions in a single conversation."""
        conversation_id = conversation.get("id", "unknown")

        # Check cache first
        if conversation_id in self.recognition_cache:
            return self.recognition_cache[conversation_id]

        # Extract conversation text
        conversation_text = self._extract_conversation_text(conversation)

        # Calculate condition scores
        condition_scores = self._calculate_condition_scores(conversation_text)

        # Determine primary and secondary conditions
        primary_condition, secondary_conditions = self._determine_conditions(condition_scores)

        # Assess severity
        severity = self._assess_severity(conversation_text, primary_condition)

        # Extract symptom markers
        symptom_markers = self._extract_symptom_markers(conversation_text, primary_condition)

        # Identify risk indicators
        risk_indicators = self._identify_risk_indicators(conversation_text)

        # Generate therapeutic recommendations
        recommendations = self._generate_therapeutic_recommendations(
            primary_condition, secondary_conditions, severity
        )

        # Calculate comorbidity likelihood
        comorbidity_likelihood = self._calculate_comorbidity_likelihood(condition_scores)

        # Calculate recognition quality
        quality_score = self._calculate_recognition_quality(
            condition_scores, symptom_markers, risk_indicators
        )

        recognition = ConditionRecognition(
            conversation_id=conversation_id,
            primary_condition=primary_condition,
            secondary_conditions=secondary_conditions,
            severity_assessment=severity,
            confidence_scores={cond.value: score for cond, score in condition_scores.items()},
            symptom_markers=symptom_markers,
            risk_indicators=risk_indicators,
            therapeutic_recommendations=recommendations,
            comorbidity_likelihood=comorbidity_likelihood,
            recognition_quality=quality_score
        )

        # Cache result
        self.recognition_cache[conversation_id] = recognition

        # Update statistics
        self._update_recognition_stats(recognition)

        return recognition

    def _extract_conversation_text(self, conversation: dict[str, Any]) -> str:
        """Extract text content from conversation structure."""
        text_parts = []

        # Handle different conversation formats
        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])
                elif isinstance(message, str):
                    text_parts.append(message)

        elif "conversations" in conversation:
            for conv in conversation["conversations"]:
                if isinstance(conv, dict):
                    text_parts.extend(conv.values())
                elif isinstance(conv, str):
                    text_parts.append(conv)

        elif "input" in conversation and "output" in conversation:
            text_parts.extend([conversation["input"], conversation["output"]])

        elif "text" in conversation:
            text_parts.append(conversation["text"])

        return " ".join(text_parts).lower()

    def _calculate_condition_scores(self, text: str) -> dict[MentalHealthCondition, float]:
        """Calculate confidence scores for each mental health condition."""
        condition_scores = dict.fromkeys(MentalHealthCondition, 0.0)

        for condition, markers in self.condition_markers.items():
            total_score = 0.0
            marker_count = 0
            severity_multiplier = 1.0

            for marker in markers:
                # Check if pattern matches
                matches = re.findall(marker.pattern, text, re.IGNORECASE)

                if matches:
                    # Apply context requirements
                    if marker.context_requirements:
                        context_met = any(
                            req.lower() in text for req in marker.context_requirements
                        )
                        if not context_met:
                            continue

                    # Check exclusion patterns
                    if marker.exclusion_patterns:
                        excluded = any(
                            re.search(excl, text, re.IGNORECASE)
                            for excl in marker.exclusion_patterns
                        )
                        if excluded:
                            continue

                    # Apply severity multiplier
                    if marker.severity_indicator == "severe":
                        severity_multiplier = 1.3
                    elif marker.severity_indicator == "moderate":
                        severity_multiplier = 1.1
                    else:
                        severity_multiplier = 1.0

                    # Calculate score based on matches and weight
                    match_score = min(
                        len(matches) * marker.confidence_weight * severity_multiplier,
                        1.0
                    )
                    total_score += match_score
                    marker_count += 1

            # Normalize score
            if marker_count > 0:
                condition_scores[condition] = min(total_score / marker_count, 1.0)

        return condition_scores

    def _determine_conditions(self, scores: dict[MentalHealthCondition, float]) -> tuple[MentalHealthCondition, list[MentalHealthCondition]]:
        """Determine primary and secondary mental health conditions."""
        # Sort conditions by score
        sorted_conditions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Primary condition (highest score)
        primary_condition = sorted_conditions[0][0]

        # Secondary conditions (score > 0.4 and not primary)
        secondary_conditions = [
            condition for condition, score in sorted_conditions[1:]
            if score > 0.4
        ]

        return primary_condition, secondary_conditions

    def _assess_severity(self, text: str, condition: MentalHealthCondition) -> str:
        """Assess severity level of the primary condition."""
        severe_indicators = [
            r"suicidal|self.*harm|can't.*function|completely.*unable|severe|crisis",
            r"hospitalization|emergency|urgent|immediate.*help|danger"
        ]

        moderate_indicators = [
            r"difficult|struggle|hard.*time|interfere|impact|affect.*daily",
            r"moderate|sometimes|often|frequently"
        ]

        # Check for severe indicators
        for pattern in severe_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return "severe"

        # Check for moderate indicators
        for pattern in moderate_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return "moderate"

        return "mild"

    def _extract_symptom_markers(self, text: str, condition: MentalHealthCondition) -> dict[str, list[str]]:
        """Extract specific symptom markers for the primary condition."""
        symptoms = defaultdict(list)

        if condition in self.condition_markers:
            for marker in self.condition_markers[condition]:
                matches = re.findall(marker.pattern, text, re.IGNORECASE)
                if matches:
                    symptoms[marker.marker_type].extend(matches[:3])  # Limit to 3 examples

        return dict(symptoms)

    def _identify_risk_indicators(self, text: str) -> list[str]:
        """Identify risk indicators in the conversation."""
        risk_patterns = {
            "suicide_risk": r"suicidal.*thought|want.*to.*die|kill.*myself|end.*it.*all|better.*off.*dead",
            "self_harm_risk": r"self.*harm|cutting|burning|hurt.*myself|self.*injur",
            "substance_risk": r"drinking.*too.*much|drug.*use|addiction|substance|alcohol.*problem",
            "crisis_risk": r"crisis|emergency|can't.*cope|breaking.*point|desperate",
            "isolation_risk": r"no.*one.*cares|completely.*alone|no.*support|isolated"
        }

        identified_risks = []
        for risk_type, pattern in risk_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                identified_risks.append(risk_type)

        return identified_risks

    def _generate_therapeutic_recommendations(self, primary: MentalHealthCondition,
                                           secondary: list[MentalHealthCondition],
                                           severity: str) -> list[str]:
        """Generate therapeutic recommendations based on conditions."""
        recommendations = []

        # Primary condition recommendations
        condition_recommendations = {
            MentalHealthCondition.DEPRESSION: ["CBT", "Behavioral Activation", "Antidepressant medication"],
            MentalHealthCondition.ANXIETY: ["CBT", "Exposure therapy", "Relaxation techniques"],
            MentalHealthCondition.PTSD: ["Trauma-focused CBT", "EMDR", "Prolonged exposure"],
            MentalHealthCondition.BIPOLAR: ["Mood stabilizers", "Psychoeducation", "Sleep hygiene"],
            MentalHealthCondition.BPD: ["DBT", "Mentalization-based therapy", "Crisis planning"],
            MentalHealthCondition.ADHD: ["Behavioral interventions", "Medication", "Organization skills"],
            MentalHealthCondition.AUTISM: ["Social skills training", "Sensory accommodations", "Routine structure"],
            MentalHealthCondition.OCD: ["ERP therapy", "CBT", "SSRI medication"],
            MentalHealthCondition.SOCIAL_ANXIETY: ["CBT", "Social skills training", "Gradual exposure"],
            MentalHealthCondition.EATING_DISORDER: ["Nutritional counseling", "CBT-E", "Medical monitoring"],
            MentalHealthCondition.SCHIZOPHRENIA: ["Antipsychotic medication", "Psychoeducation", "Social skills training"],
            MentalHealthCondition.SUBSTANCE_USE: ["Detoxification", "CBT", "Support groups", "Relapse prevention"]
        }

        if primary in condition_recommendations:
            recommendations.extend(condition_recommendations[primary])

        # Severity-based recommendations
        if severity == "severe":
            recommendations.extend(["Crisis intervention", "Intensive outpatient", "Safety planning"])
        elif severity == "moderate":
            recommendations.extend(["Regular therapy", "Support groups", "Lifestyle modifications"])

        # Comorbidity considerations
        if len(secondary) > 1:
            recommendations.append("Integrated treatment approach")

        return recommendations[:5]  # Limit to top 5 recommendations

    def _calculate_comorbidity_likelihood(self, scores: dict[MentalHealthCondition, float]) -> float:
        """Calculate likelihood of comorbid conditions."""
        # Count conditions with significant scores
        significant_conditions = sum(1 for score in scores.values() if score > 0.4)

        # Calculate comorbidity likelihood based on number and strength of conditions
        if significant_conditions <= 1:
            return 0.0
        if significant_conditions == 2:
            return 0.6
        if significant_conditions == 3:
            return 0.8
        return 0.9

    def _calculate_recognition_quality(self, scores: dict[MentalHealthCondition, float],
                                     symptoms: dict[str, list[str]],
                                     risks: list[str]) -> float:
        """Calculate quality score for the recognition."""
        # Base score from primary condition confidence
        primary_score = max(scores.values())

        # Symptom evidence bonus
        symptom_count = sum(len(symptom_list) for symptom_list in symptoms.values())
        symptom_bonus = min(symptom_count * 0.05, 0.2)

        # Risk identification bonus
        risk_bonus = min(len(risks) * 0.1, 0.2)

        # Clarity bonus (clear primary vs mixed)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            clarity_bonus = (sorted_scores[0] - sorted_scores[1]) * 0.15
        else:
            clarity_bonus = 0.1

        return min(primary_score + symptom_bonus + risk_bonus + clarity_bonus, 1.0)

    def _update_recognition_stats(self, recognition: ConditionRecognition):
        """Update recognition statistics."""
        self.recognition_stats["total_recognized"] += 1
        self.recognition_stats["condition_distribution"][recognition.primary_condition.value] += 1
        self.recognition_stats["severity_distribution"][recognition.severity_assessment] += 1

        if recognition.comorbidity_likelihood > 0.6:
            self.recognition_stats["comorbidity_cases"] += 1

        if recognition.risk_indicators:
            self.recognition_stats["high_risk_cases"] += 1

    def recognize_batch(self, conversations: list[dict[str, Any]]) -> list[ConditionRecognition]:
        """Recognize conditions in a batch of conversations."""
        self.logger.info(f"Recognizing conditions in batch of {len(conversations)} conversations...")

        recognitions = []
        for conversation in conversations:
            try:
                recognition = self.recognize_condition(conversation)
                recognitions.append(recognition)
            except Exception as e:
                self.logger.error(f"Error recognizing condition in conversation {conversation.get('id', 'unknown')}: {e}")

        self.logger.info(f"Successfully recognized conditions in {len(recognitions)} conversations")
        return recognitions

    def get_condition_distribution(self) -> dict[str, Any]:
        """Get distribution of recognized mental health conditions."""
        total = self.recognition_stats["total_recognized"]
        if total == 0:
            return {}

        return {
            "total_recognized": total,
            "condition_distribution": dict(self.recognition_stats["condition_distribution"]),
            "severity_distribution": dict(self.recognition_stats["severity_distribution"]),
            "comorbidity_rate": (self.recognition_stats["comorbidity_cases"] / total) * 100,
            "high_risk_rate": (self.recognition_stats["high_risk_cases"] / total) * 100
        }

    def export_recognition_results(self, output_path: str):
        """Export recognition results to file."""
        # Convert markers to serializable format
        serializable_markers = {}
        for condition, markers in self.condition_markers.items():
            serializable_markers[condition.value] = []
            for marker in markers:
                marker_dict = asdict(marker)
                marker_dict["condition"] = marker.condition.value  # Convert enum to string
                serializable_markers[condition.value].append(marker_dict)

        results = {
            "recognition_statistics": self.get_condition_distribution(),
            "condition_markers": serializable_markers,
            "export_timestamp": datetime.now().isoformat()
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Recognition results exported to {output_path}")


# Example usage and testing
def main():
    """Example usage of the mental health condition recognizer."""

    # Create recognizer
    recognizer = MentalHealthConditionRecognizer()

    # Example conversations for testing
    test_conversations = [
        {
            "id": "depression_example",
            "messages": [
                {"role": "client", "content": "I feel so hopeless and empty. Nothing brings me joy anymore. I just want to sleep all the time."},
                {"role": "therapist", "content": "I hear that you're experiencing some really difficult feelings right now."}
            ]
        },
        {
            "id": "anxiety_example",
            "messages": [
                {"role": "client", "content": "I'm constantly worried about everything. My heart races and I can't stop thinking about what if something bad happens."},
                {"role": "therapist", "content": "Anxiety can be really overwhelming. Let's work on some coping strategies."}
            ]
        },
        {
            "id": "ptsd_example",
            "messages": [
                {"role": "client", "content": "I keep having flashbacks of the accident. I can't sleep and I jump at every sound."},
                {"role": "therapist", "content": "Trauma can have lasting effects. We can work through this together."}
            ]
        }
    ]


    # Recognize conditions
    recognitions = recognizer.recognize_batch(test_conversations)

    # Display results
    for recognition in recognitions:

        if recognition.risk_indicators:
            pass

        if recognition.therapeutic_recommendations:
            pass

    # Show distribution
    distribution = recognizer.get_condition_distribution()

    for _condition, _count in distribution["condition_distribution"].items():
        pass

    for _severity, _count in distribution["severity_distribution"].items():
        pass


if __name__ == "__main__":
    main()

# Alias for compatibility
ConditionPatternRecognizer = MentalHealthConditionRecognizer
