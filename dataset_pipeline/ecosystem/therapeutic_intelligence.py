#!/usr/bin/env python3
"""
Task 6.7: Comprehensive Therapeutic Approach Classification System

This module implements a sophisticated system to classify therapeutic conversations
across major therapeutic modalities (CBT, DBT, Psychodynamic, Humanistic, etc.)
with high accuracy and detailed analysis.

Strategic Goal: Automatically classify and balance therapeutic approaches across
the 2.59M+ conversation ecosystem to ensure comprehensive training coverage.
"""

import json
import logging
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Import ecosystem components


class TherapeuticApproach(Enum):
    """Major therapeutic approaches and modalities."""
    CBT = "cognitive_behavioral_therapy"
    DBT = "dialectical_behavior_therapy"
    PSYCHODYNAMIC = "psychodynamic_therapy"
    HUMANISTIC = "humanistic_therapy"
    GESTALT = "gestalt_therapy"
    SOLUTION_FOCUSED = "solution_focused_therapy"
    NARRATIVE = "narrative_therapy"
    FAMILY_SYSTEMS = "family_systems_therapy"
    TRAUMA_INFORMED = "trauma_informed_therapy"
    MINDFULNESS_BASED = "mindfulness_based_therapy"
    ACCEPTANCE_COMMITMENT = "acceptance_commitment_therapy"
    INTERPERSONAL = "interpersonal_therapy"
    EXISTENTIAL = "existential_therapy"
    INTEGRATIVE = "integrative_therapy"
    ECLECTIC = "eclectic_approach"


@dataclass
class TherapeuticMarker:
    """Represents a marker that indicates a specific therapeutic approach."""
    approach: TherapeuticApproach
    marker_type: str  # keyword, phrase, technique, intervention
    pattern: str
    confidence_weight: float
    context_requirements: list[str] = None
    exclusion_patterns: list[str] = None


@dataclass
class ApproachClassification:
    """Result of therapeutic approach classification."""
    conversation_id: str
    primary_approach: TherapeuticApproach
    secondary_approaches: list[TherapeuticApproach]
    confidence_scores: dict[str, float]
    evidence_markers: dict[str, list[str]]
    classification_rationale: str
    mixed_approach_indicator: bool
    quality_score: float


class TherapeuticApproachClassifier:
    """Classifies conversations by therapeutic approach using pattern recognition."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.therapeutic_markers = self._initialize_therapeutic_markers()
        self.classification_cache = {}

        # Performance tracking
        self.classification_stats = {
            "total_classified": 0,
            "approach_distribution": defaultdict(int),
            "confidence_scores": [],
            "mixed_approaches": 0
        }

    def _initialize_therapeutic_markers(self) -> dict[TherapeuticApproach, list[TherapeuticMarker]]:
        """Initialize therapeutic approach markers and patterns."""
        return {
            TherapeuticApproach.CBT: [
                TherapeuticMarker(
                    TherapeuticApproach.CBT, "technique",
                    r"thought.*record|cognitive.*restructur|automatic.*thought",
                    0.9, ["therapy", "counseling"]
                ),
                TherapeuticMarker(
                    TherapeuticApproach.CBT, "intervention",
                    r"behavioral.*experiment|exposure.*therapy|homework.*assignment",
                    0.85, ["therapeutic"]
                ),
                TherapeuticMarker(
                    TherapeuticApproach.CBT, "concept",
                    r"cognitive.*distortion|thinking.*pattern|belief.*system",
                    0.8
                ),
                TherapeuticMarker(
                    TherapeuticApproach.CBT, "phrase",
                    r"what.*evidence|alternative.*thought|balanced.*thinking",
                    0.75
                )
            ],

            TherapeuticApproach.DBT: [
                TherapeuticMarker(
                    TherapeuticApproach.DBT, "skill",
                    r"distress.*tolerance|emotion.*regulation|interpersonal.*effectiveness",
                    0.95
                ),
                TherapeuticMarker(
                    TherapeuticApproach.DBT, "technique",
                    r"mindfulness|wise.*mind|radical.*acceptance",
                    0.85
                ),
                TherapeuticMarker(
                    TherapeuticApproach.DBT, "concept",
                    r"dialectical|opposite.*action|TIPP|PLEASE",
                    0.9
                )
            ],

            TherapeuticApproach.PSYCHODYNAMIC: [
                TherapeuticMarker(
                    TherapeuticApproach.PSYCHODYNAMIC, "concept",
                    r"unconscious|transference|defense.*mechanism|attachment.*style",
                    0.9
                ),
                TherapeuticMarker(
                    TherapeuticApproach.PSYCHODYNAMIC, "technique",
                    r"free.*association|dream.*analysis|interpretation",
                    0.85
                ),
                TherapeuticMarker(
                    TherapeuticApproach.PSYCHODYNAMIC, "phrase",
                    r"early.*childhood|relationship.*pattern|inner.*conflict",
                    0.75
                )
            ],

            TherapeuticApproach.HUMANISTIC: [
                TherapeuticMarker(
                    TherapeuticApproach.HUMANISTIC, "concept",
                    r"unconditional.*positive.*regard|self.*actualization|person.*centered",
                    0.9
                ),
                TherapeuticMarker(
                    TherapeuticApproach.HUMANISTIC, "technique",
                    r"active.*listening|empathic.*reflection|genuineness",
                    0.8
                ),
                TherapeuticMarker(
                    TherapeuticApproach.HUMANISTIC, "phrase",
                    r"how.*does.*that.*feel|what.*comes.*up|your.*experience",
                    0.7
                )
            ],

            TherapeuticApproach.SOLUTION_FOCUSED: [
                TherapeuticMarker(
                    TherapeuticApproach.SOLUTION_FOCUSED, "technique",
                    r"miracle.*question|scaling.*question|exception.*finding",
                    0.95
                ),
                TherapeuticMarker(
                    TherapeuticApproach.SOLUTION_FOCUSED, "phrase",
                    r"what.*would.*be.*different|when.*was.*it.*better|small.*step",
                    0.8
                )
            ],

            TherapeuticApproach.TRAUMA_INFORMED: [
                TherapeuticMarker(
                    TherapeuticApproach.TRAUMA_INFORMED, "concept",
                    r"trauma.*informed|EMDR|somatic|body.*awareness",
                    0.9
                ),
                TherapeuticMarker(
                    TherapeuticApproach.TRAUMA_INFORMED, "technique",
                    r"grounding.*technique|bilateral.*stimulation|window.*of.*tolerance",
                    0.85
                )
            ],

            TherapeuticApproach.MINDFULNESS_BASED: [
                TherapeuticMarker(
                    TherapeuticApproach.MINDFULNESS_BASED, "technique",
                    r"mindfulness.*meditation|body.*scan|breathing.*exercise",
                    0.9
                ),
                TherapeuticMarker(
                    TherapeuticApproach.MINDFULNESS_BASED, "concept",
                    r"present.*moment|non.*judgmental|awareness",
                    0.8
                )
            ]
        }


    def classify_conversation(self, conversation: dict[str, Any]) -> ApproachClassification:
        """Classify a single conversation by therapeutic approach."""
        conversation_id = conversation.get("id", "unknown")

        # Check cache first
        if conversation_id in self.classification_cache:
            return self.classification_cache[conversation_id]

        # Extract conversation text
        conversation_text = self._extract_conversation_text(conversation)

        # Calculate approach scores
        approach_scores = self._calculate_approach_scores(conversation_text)

        # Determine primary and secondary approaches
        primary_approach, secondary_approaches = self._determine_approaches(approach_scores)

        # Extract evidence markers
        evidence_markers = self._extract_evidence_markers(conversation_text, primary_approach)

        # Generate classification rationale
        rationale = self._generate_classification_rationale(
            primary_approach, approach_scores, evidence_markers
        )

        # Calculate quality score
        quality_score = self._calculate_classification_quality(approach_scores, evidence_markers)

        # Check for mixed approach
        mixed_approach = len(secondary_approaches) > 1 and max(
            approach_scores[app.value] for app in secondary_approaches
        ) > 0.6

        classification = ApproachClassification(
            conversation_id=conversation_id,
            primary_approach=primary_approach,
            secondary_approaches=secondary_approaches,
            confidence_scores={app.value: score for app, score in approach_scores.items()},
            evidence_markers=evidence_markers,
            classification_rationale=rationale,
            mixed_approach_indicator=mixed_approach,
            quality_score=quality_score
        )

        # Cache result
        self.classification_cache[conversation_id] = classification

        # Update statistics
        self._update_classification_stats(classification)

        return classification

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

    def _calculate_approach_scores(self, text: str) -> dict[TherapeuticApproach, float]:
        """Calculate confidence scores for each therapeutic approach."""
        approach_scores = dict.fromkeys(TherapeuticApproach, 0.0)

        for approach, markers in self.therapeutic_markers.items():
            total_score = 0.0
            marker_count = 0

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

                    # Calculate score based on matches and weight
                    match_score = min(len(matches) * marker.confidence_weight, 1.0)
                    total_score += match_score
                    marker_count += 1

            # Normalize score
            if marker_count > 0:
                approach_scores[approach] = min(total_score / marker_count, 1.0)

        return approach_scores

    def _determine_approaches(self, scores: dict[TherapeuticApproach, float]) -> tuple[TherapeuticApproach, list[TherapeuticApproach]]:
        """Determine primary and secondary therapeutic approaches."""
        # Sort approaches by score
        sorted_approaches = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Primary approach (highest score)
        primary_approach = sorted_approaches[0][0]

        # Secondary approaches (score > 0.3 and not primary)
        secondary_approaches = [
            approach for approach, score in sorted_approaches[1:]
            if score > 0.3
        ]

        return primary_approach, secondary_approaches

    def _extract_evidence_markers(self, text: str, primary_approach: TherapeuticApproach) -> dict[str, list[str]]:
        """Extract specific evidence markers for the classification."""
        evidence = defaultdict(list)

        if primary_approach in self.therapeutic_markers:
            for marker in self.therapeutic_markers[primary_approach]:
                matches = re.findall(marker.pattern, text, re.IGNORECASE)
                if matches:
                    evidence[marker.marker_type].extend(matches[:3])  # Limit to 3 examples

        return dict(evidence)

    def _generate_classification_rationale(self, primary_approach: TherapeuticApproach,
                                         scores: dict[TherapeuticApproach, float],
                                         evidence: dict[str, list[str]]) -> str:
        """Generate human-readable classification rationale."""
        primary_score = scores[primary_approach]

        rationale_parts = [
            f"Classified as {primary_approach.value} with confidence {primary_score:.2f}"
        ]

        if evidence:
            evidence_summary = []
            for marker_type, markers in evidence.items():
                if markers:
                    evidence_summary.append(f"{marker_type}: {', '.join(markers[:2])}")

            if evidence_summary:
                rationale_parts.append(f"Evidence: {'; '.join(evidence_summary)}")

        # Add secondary approaches if significant
        secondary_scores = [
            (app, score) for app, score in scores.items()
            if app != primary_approach and score > 0.4
        ]

        if secondary_scores:
            secondary_info = [f"{app.value} ({score:.2f})" for app, score in secondary_scores[:2]]
            rationale_parts.append(f"Secondary approaches: {', '.join(secondary_info)}")

        return ". ".join(rationale_parts)

    def _calculate_classification_quality(self, scores: dict[TherapeuticApproach, float],
                                        evidence: dict[str, list[str]]) -> float:
        """Calculate quality score for the classification."""
        # Base score from primary approach confidence
        primary_score = max(scores.values())

        # Evidence quality bonus
        evidence_count = sum(len(markers) for markers in evidence.values())
        evidence_bonus = min(evidence_count * 0.15, 0.4)  # Increased bonus

        # Clarity bonus (clear primary vs mixed)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            clarity_bonus = (sorted_scores[0] - sorted_scores[1]) * 0.3  # Increased bonus
        else:
            clarity_bonus = 0.2  # Increased base bonus

        # Minimum quality floor for valid classifications
        base_quality = 0.4 if primary_score > 0 else 0.1

        return min(base_quality + primary_score + evidence_bonus + clarity_bonus, 1.0)

    def _update_classification_stats(self, classification: ApproachClassification):
        """Update classification statistics."""
        self.classification_stats["total_classified"] += 1
        self.classification_stats["approach_distribution"][classification.primary_approach.value] += 1
        self.classification_stats["confidence_scores"].append(classification.quality_score)

        if classification.mixed_approach_indicator:
            self.classification_stats["mixed_approaches"] += 1

    def classify_batch(self, conversations: list[dict[str, Any]]) -> list[ApproachClassification]:
        """Classify a batch of conversations."""
        self.logger.info(f"Classifying batch of {len(conversations)} conversations...")

        classifications = []
        for conversation in conversations:
            try:
                classification = self.classify_conversation(conversation)
                classifications.append(classification)
            except Exception as e:
                self.logger.error(f"Error classifying conversation {conversation.get('id', 'unknown')}: {e}")

        self.logger.info(f"Successfully classified {len(classifications)} conversations")
        return classifications

    def get_approach_distribution(self) -> dict[str, Any]:
        """Get distribution of therapeutic approaches."""
        total = self.classification_stats["total_classified"]
        if total == 0:
            return {}

        distribution = {}
        for approach, count in self.classification_stats["approach_distribution"].items():
            distribution[approach] = {
                "count": count,
                "percentage": (count / total) * 100
            }

        return {
            "total_classified": total,
            "approach_distribution": distribution,
            "average_confidence": statistics.mean(self.classification_stats["confidence_scores"]) if self.classification_stats["confidence_scores"] else 0,
            "mixed_approaches_percentage": (self.classification_stats["mixed_approaches"] / total) * 100 if total > 0 else 0
        }

    def export_classification_results(self, output_path: str):
        """Export classification results to file."""
        # Convert markers to serializable format
        serializable_markers = {}
        for approach, markers in self.therapeutic_markers.items():
            serializable_markers[approach.value] = []
            for marker in markers:
                marker_dict = asdict(marker)
                marker_dict["approach"] = marker.approach.value  # Convert enum to string
                serializable_markers[approach.value].append(marker_dict)

        results = {
            "classification_statistics": self.get_approach_distribution(),
            "therapeutic_markers": serializable_markers,
            "export_timestamp": datetime.now().isoformat()
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Classification results exported to {output_path}")


# Example usage and testing
def main():
    """Example usage of the therapeutic approach classifier."""

    # Create classifier
    classifier = TherapeuticApproachClassifier()

    # Example conversations for testing
    test_conversations = [
        {
            "id": "cbt_example",
            "messages": [
                {"role": "client", "content": "I keep having these negative thoughts about myself"},
                {"role": "therapist", "content": "Let's examine the evidence for these automatic thoughts. What proof do you have that these thoughts are true?"}
            ]
        },
        {
            "id": "dbt_example",
            "messages": [
                {"role": "client", "content": "I feel so overwhelmed with emotions"},
                {"role": "therapist", "content": "Let's practice some distress tolerance skills. Try the TIPP technique - temperature, intense exercise, paced breathing, and paired muscle relaxation."}
            ]
        },
        {
            "id": "humanistic_example",
            "messages": [
                {"role": "client", "content": "I don't know what I want in life"},
                {"role": "therapist", "content": "How does that uncertainty feel for you right now? What comes up when you sit with that not knowing?"}
            ]
        }
    ]


    # Classify conversations
    classifications = classifier.classify_batch(test_conversations)

    # Display results
    for classification in classifications:

        if classification.secondary_approaches:
            [app.value for app in classification.secondary_approaches]

    # Show distribution
    distribution = classifier.get_approach_distribution()

    for _approach, _stats in distribution["approach_distribution"].items():
        pass


if __name__ == "__main__":
    main()
