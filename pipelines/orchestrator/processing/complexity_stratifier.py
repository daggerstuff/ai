#!/usr/bin/env python3
"""
Conversation Complexity Stratification for Task 6.23
Classifies therapeutic conversations into beginner, intermediate, and advanced levels
based on emotional intensity, topic difficulty, and intervention complexity.
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ComplexityConfig:
    """Configuration for complexity levels"""
    level: str
    weight: float  # Target proportion
    min_samples: int
    complexity_range: tuple[float, float]  # Min, max complexity scores
    max_samples: int | None = None
    characteristics: list[str] = None

@dataclass
class ComplexityResult:
    """Result of complexity stratification"""
    level: str
    target_samples: int
    actual_samples: int
    avg_complexity: float
    complexity_range: tuple[float, float]
    conversations: list[dict[str, Any]]
    metadata: dict[str, Any]

class ComplexityStratifier:
    """
    Advanced conversation complexity stratification system.
    Classifies conversations by therapeutic complexity for progressive training.
    """

    def __init__(self, config_path: str | None = None):
        """Initialize the complexity stratifier"""
        self.complexity_configs = self._load_complexity_configs(config_path)
        self.stratification_history = []

    def _load_complexity_configs(self, config_path: str | None = None) -> dict[str, ComplexityConfig]:
        """Load complexity level configurations"""
        default_configs = {
            "beginner": ComplexityConfig(
                level="Beginner",
                weight=0.40,  # 40% - foundational conversations
                min_samples=1000,
                max_samples=15000,
                complexity_range=(0.0, 0.4),
                characteristics=[
                    "Simple emotional expressions",
                    "Basic therapeutic techniques",
                    "Clear, straightforward issues",
                    "Single-topic focus",
                    "Minimal comorbidity",
                    "Standard interventions"
                ]
            ),
            "intermediate": ComplexityConfig(
                level="Intermediate",
                weight=0.45,  # 45% - moderate complexity
                min_samples=1200,
                max_samples=18000,
                complexity_range=(0.4, 0.7),
                characteristics=[
                    "Moderate emotional intensity",
                    "Multiple therapeutic techniques",
                    "Interconnected issues",
                    "Some comorbidity",
                    "Nuanced interventions",
                    "Relationship dynamics"
                ]
            ),
            "advanced": ComplexityConfig(
                level="Advanced",
                weight=0.15,  # 15% - high complexity scenarios
                min_samples=400,
                max_samples=6000,
                complexity_range=(0.7, 1.0),
                characteristics=[
                    "High emotional intensity",
                    "Complex therapeutic approaches",
                    "Multiple interconnected issues",
                    "Significant comorbidity",
                    "Crisis intervention elements",
                    "Advanced clinical skills required"
                ]
            )
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    custom_config = json.load(f)
                # Update configurations
                for level, config_data in custom_config.items():
                    if level in default_configs:
                        for key, value in config_data.items():
                            setattr(default_configs[level], key, value)
            except Exception as e:
                logger.warning(f"Failed to load custom config: {e}. Using defaults.")

        return default_configs

    def calculate_complexity_score(self, conversation: dict[str, Any]) -> float:
        """
        Calculate overall complexity score for a conversation.
        Combines emotional intensity, topic difficulty, and intervention complexity.
        """
        complexity_factors = {
            "emotional_intensity": 0.35,
            "topic_difficulty": 0.30,
            "intervention_complexity": 0.25,
            "linguistic_complexity": 0.10
        }

        total_score = 0.0

        # Emotional intensity (0.0-1.0)
        emotional_score = self._assess_emotional_intensity(conversation)
        total_score += emotional_score * complexity_factors["emotional_intensity"]

        # Topic difficulty (0.0-1.0)
        topic_score = self._assess_topic_difficulty(conversation)
        total_score += topic_score * complexity_factors["topic_difficulty"]

        # Intervention complexity (0.0-1.0)
        intervention_score = self._assess_intervention_complexity(conversation)
        total_score += intervention_score * complexity_factors["intervention_complexity"]

        # Linguistic complexity (0.0-1.0)
        linguistic_score = self._assess_linguistic_complexity(conversation)
        total_score += linguistic_score * complexity_factors["linguistic_complexity"]

        return min(1.0, total_score)

    def _assess_emotional_intensity(self, conversation: dict[str, Any]) -> float:
        """Assess emotional intensity of the conversation"""
        content = str(conversation).lower()

        # High intensity emotional words
        high_intensity = [
            "overwhelming", "devastating", "unbearable", "excruciating", "terrifying",
            "panic", "crisis", "emergency", "suicidal", "hopeless", "desperate",
            "rage", "fury", "anguish", "torment", "agony", "breakdown"
        ]

        # Moderate intensity emotional words
        moderate_intensity = [
            "anxious", "worried", "sad", "upset", "frustrated", "angry",
            "depressed", "stressed", "concerned", "troubled", "disturbed",
            "uncomfortable", "difficult", "challenging", "struggling"
        ]

        # Low intensity emotional words
        low_intensity = [
            "concerned", "bothered", "uneasy", "unsure", "confused",
            "disappointed", "tired", "restless", "mild", "slight"
        ]

        # Count emotional indicators
        high_count = sum(1 for word in high_intensity if word in content)
        moderate_count = sum(1 for word in moderate_intensity if word in content)
        low_count = sum(1 for word in low_intensity if word in content)

        # Calculate intensity score
        if high_count > 0:
            base_score = 0.8 + (high_count * 0.05)
        elif moderate_count > 0:
            base_score = 0.4 + (moderate_count * 0.03)
        elif low_count > 0:
            base_score = 0.1 + (low_count * 0.02)
        else:
            base_score = 0.2  # Default moderate-low

        return min(1.0, base_score)

    def _assess_topic_difficulty(self, conversation: dict[str, Any]) -> float:
        """Assess difficulty/complexity of topics discussed"""
        content = str(conversation).lower()

        # Complex/difficult topics
        complex_topics = [
            "trauma", "abuse", "suicide", "self-harm", "addiction", "psychosis",
            "personality disorder", "dissociation", "complex ptsd", "bipolar",
            "schizophrenia", "eating disorder", "borderline", "narcissistic",
            "antisocial", "paranoid", "multiple diagnoses", "comorbid"
        ]

        # Moderate difficulty topics
        moderate_topics = [
            "depression", "anxiety", "panic", "ocd", "phobia", "grief",
            "relationship", "family", "work stress", "social anxiety",
            "adjustment", "sleep", "anger", "communication", "boundaries"
        ]

        # Basic topics
        basic_topics = [
            "stress", "worry", "sadness", "loneliness", "confidence",
            "motivation", "habits", "goals", "self-care", "coping",
            "support", "wellness", "mindfulness", "relaxation"
        ]

        # Count topic indicators
        complex_count = sum(1 for topic in complex_topics if topic in content)
        moderate_count = sum(1 for topic in moderate_topics if topic in content)
        basic_count = sum(1 for topic in basic_topics if topic in content)

        # Calculate difficulty score
        if complex_count > 0:
            base_score = 0.7 + (complex_count * 0.08)
        elif moderate_count > 0:
            base_score = 0.3 + (moderate_count * 0.05)
        elif basic_count > 0:
            base_score = 0.1 + (basic_count * 0.03)
        else:
            base_score = 0.25  # Default moderate-low

        return min(1.0, base_score)

    def _assess_intervention_complexity(self, conversation: dict[str, Any]) -> float:
        """Assess complexity of therapeutic interventions used"""
        content = str(conversation).lower()

        # Advanced/complex interventions
        advanced_interventions = [
            "emdr", "exposure therapy", "dialectical behavior", "schema therapy",
            "psychodynamic", "transference", "countertransference", "interpretation",
            "family systems", "structural", "strategic", "narrative therapy",
            "somatic", "gestalt", "existential", "crisis intervention"
        ]

        # Intermediate interventions
        intermediate_interventions = [
            "cognitive behavioral", "cbt", "mindfulness", "acceptance commitment",
            "interpersonal therapy", "solution focused", "motivational interviewing",
            "behavioral activation", "cognitive restructuring", "thought records",
            "relaxation", "grounding", "coping skills", "problem solving"
        ]

        # Basic interventions
        basic_interventions = [
            "active listening", "reflection", "empathy", "validation",
            "support", "encouragement", "psychoeducation", "breathing",
            "self-care", "scheduling", "goal setting", "homework"
        ]

        # Count intervention indicators
        advanced_count = sum(1 for intervention in advanced_interventions if intervention in content)
        intermediate_count = sum(1 for intervention in intermediate_interventions if intervention in content)
        basic_count = sum(1 for intervention in basic_interventions if intervention in content)

        # Calculate complexity score
        if advanced_count > 0:
            base_score = 0.8 + (advanced_count * 0.05)
        elif intermediate_count > 0:
            base_score = 0.4 + (intermediate_count * 0.04)
        elif basic_count > 0:
            base_score = 0.1 + (basic_count * 0.02)
        else:
            base_score = 0.3  # Default moderate

        return min(1.0, base_score)

    def _assess_linguistic_complexity(self, conversation: dict[str, Any]) -> float:
        """Assess linguistic complexity of the conversation"""
        messages = conversation.get("messages", [])
        if not messages:
            return 0.0

        total_complexity = 0.0
        message_count = 0

        for message in messages:
            content = message.get("content", "")
            if not content:
                continue

            # Calculate various linguistic metrics
            words = content.split()
            sentences = re.split(r"[.!?]+", content)

            if not words:
                continue

            # Average word length
            avg_word_length = np.mean([len(word) for word in words])
            word_length_score = min(1.0, (avg_word_length - 3) / 5)  # Normalize around 3-8 chars

            # Sentence complexity
            if sentences:
                avg_sentence_length = len(words) / len([s for s in sentences if s.strip()])
                sentence_complexity = min(1.0, (avg_sentence_length - 5) / 20)  # Normalize around 5-25 words
            else:
                sentence_complexity = 0.0

            # Technical/clinical vocabulary
            clinical_terms = [
                "therapeutic", "intervention", "assessment", "diagnosis", "treatment",
                "cognitive", "behavioral", "psychodynamic", "systemic", "clinical",
                "psychological", "psychiatric", "neurological", "pharmacological"
            ]

            clinical_count = sum(1 for term in clinical_terms if term.lower() in content.lower())
            clinical_score = min(1.0, clinical_count / 5)

            # Combined linguistic complexity
            message_complexity = (word_length_score * 0.3 +
                                sentence_complexity * 0.4 +
                                clinical_score * 0.3)

            total_complexity += message_complexity
            message_count += 1

        return total_complexity / message_count if message_count > 0 else 0.0

    def classify_complexity_level(self, complexity_score: float) -> str:
        """Classify complexity score into beginner/intermediate/advanced"""
        for level_id, config in self.complexity_configs.items():
            min_score, max_score = config.complexity_range
            if min_score <= complexity_score < max_score:
                return level_id

        # Default to advanced if score is very high
        return "advanced"

    def stratify_conversations(self, conversations: list[dict[str, Any]],
                             target_total: int = 10000) -> list[ComplexityResult]:
        """
        Main method to stratify conversations by complexity level.
        """
        logger.info(f"Starting complexity stratification for {len(conversations)} conversations")

        # Calculate complexity scores and classify
        complexity_conversations = defaultdict(list)
        complexity_scores = {}

        for conv in conversations:
            complexity_score = self.calculate_complexity_score(conv)
            complexity_level = self.classify_complexity_level(complexity_score)

            complexity_conversations[complexity_level].append(conv)
            complexity_scores[conv.get("id", str(hash(str(conv))))] = complexity_score

        # Calculate target distribution
        target_distribution = {}
        total_weight = sum(config.weight for config in self.complexity_configs.values())

        for level_id, config in self.complexity_configs.items():
            # Base samples from weight
            weight_ratio = config.weight / total_weight
            base_samples = int(target_total * weight_ratio)

            # Ensure minimum samples
            base_samples = max(base_samples, config.min_samples)

            # Ensure maximum samples
            if config.max_samples:
                base_samples = min(base_samples, config.max_samples)

            # Ensure we don't exceed available data
            available = len(complexity_conversations.get(level_id, []))
            base_samples = min(base_samples, available)

            target_distribution[level_id] = base_samples

            logger.info(f"{config.level}: {base_samples} samples "
                       f"(weight: {config.weight:.1%}, available: {available})")

        # Stratify each complexity level
        results = []
        total_stratified = 0

        for level_id, target_count in target_distribution.items():
            if target_count <= 0:
                continue

            config = self.complexity_configs[level_id]
            available_conversations = complexity_conversations.get(level_id, [])

            logger.info(f"Stratifying {config.level}: {target_count} target, "
                       f"{len(available_conversations)} available")

            # Select conversations with complexity-based quality weighting
            selected = self._select_complexity_balanced_conversations(
                available_conversations, target_count, level_id, complexity_scores
            )

            if selected:
                # Calculate metrics
                selected_scores = [complexity_scores.get(conv.get("id", str(hash(str(conv)))), 0.0)
                                 for conv in selected]
                avg_complexity = np.mean(selected_scores)
                complexity_range = (min(selected_scores), max(selected_scores))

                result = ComplexityResult(
                    level=config.level,
                    target_samples=target_count,
                    actual_samples=len(selected),
                    avg_complexity=avg_complexity,
                    complexity_range=complexity_range,
                    conversations=selected,
                    metadata={
                        "level_id": level_id,
                        "weight": config.weight,
                        "target_complexity_range": config.complexity_range,
                        "available_count": len(available_conversations),
                        "characteristics": config.characteristics
                    }
                )

                results.append(result)
                total_stratified += len(selected)

                logger.info(f"Stratified {config.level}: {len(selected)} conversations "
                           f"(avg complexity: {avg_complexity:.3f})")

        logger.info(f"Total stratified: {total_stratified} conversations across "
                   f"{len(results)} complexity levels")

        # Store stratification history
        self.stratification_history.append({
            "timestamp": str(np.datetime64("now")),
            "target_total": target_total,
            "actual_total": total_stratified,
            "complexity_levels": len(results),
            "level_results": {r.level: r.metadata for r in results}
        })

        return results

    def _select_complexity_balanced_conversations(self, conversations: list[dict[str, Any]],
                                                target_count: int, level_id: str,
                                                complexity_scores: dict[str, float]) -> list[dict[str, Any]]:
        """Select conversations with complexity-based balancing"""
        if not conversations or target_count <= 0:
            return []

        config = self.complexity_configs[level_id]
        min_complexity, max_complexity = config.complexity_range

        # Score conversations for selection
        scored_conversations = []
        for conv in conversations:
            conv_id = conv.get("id", str(hash(str(conv))))
            complexity_score = complexity_scores.get(conv_id, 0.0)

            # Quality score based on how well it fits the complexity range
            if min_complexity <= complexity_score < max_complexity:
                # Perfect fit
                range_fit_score = 1.0
            else:
                # Calculate distance from ideal range
                if complexity_score < min_complexity:
                    distance = min_complexity - complexity_score
                else:
                    distance = complexity_score - max_complexity
                range_fit_score = max(0.0, 1.0 - distance)

            # Diversity score (prefer varied complexity within range)
            diversity_score = self._assess_complexity_diversity(conv)

            # Combined selection score
            selection_score = range_fit_score * 0.7 + diversity_score * 0.3

            scored_conversations.append((conv, selection_score, complexity_score))

        # Sort by selection score
        scored_conversations.sort(key=lambda x: x[1], reverse=True)

        # Select conversations ensuring complexity distribution within range
        selected = []
        selected_count = min(target_count, len(scored_conversations))

        # Stratified selection within the complexity range
        if selected_count > 0:
            # Divide range into sub-ranges for better distribution
            num_sub_ranges = min(3, selected_count)
            sub_range_size = (max_complexity - min_complexity) / num_sub_ranges

            conversations_by_sub_range = [[] for _ in range(num_sub_ranges)]

            for conv, score, complexity in scored_conversations:
                if min_complexity <= complexity < max_complexity:
                    sub_range_idx = min(num_sub_ranges - 1,
                                      int((complexity - min_complexity) / sub_range_size))
                    conversations_by_sub_range[sub_range_idx].append((conv, score, complexity))

            # Select proportionally from each sub-range
            samples_per_sub_range = selected_count // num_sub_ranges
            remainder = selected_count % num_sub_ranges

            for i, sub_range_conversations in enumerate(conversations_by_sub_range):
                sub_range_target = samples_per_sub_range + (1 if i < remainder else 0)
                sub_range_selected = min(sub_range_target, len(sub_range_conversations))

                # Sort by score and select top conversations
                sub_range_conversations.sort(key=lambda x: x[1], reverse=True)
                selected.extend([conv for conv, _, _ in sub_range_conversations[:sub_range_selected]])

        return selected

    def _assess_complexity_diversity(self, conversation: dict[str, Any]) -> float:
        """Assess diversity of complexity factors within conversation"""
        # Look for indicators of multi-faceted complexity
        content = str(conversation).lower()

        diversity_indicators = [
            "multiple", "various", "different", "complex", "interconnected",
            "layered", "multifaceted", "comprehensive", "holistic", "integrated"
        ]

        diversity_count = sum(1 for indicator in diversity_indicators if indicator in content)
        return min(1.0, diversity_count / len(diversity_indicators) * 2)

    def get_stratification_statistics(self) -> dict[str, Any]:
        """Get comprehensive stratification statistics"""
        if not self.stratification_history:
            return {"error": "No stratification history available"}

        latest = self.stratification_history[-1]

        return {
            "total_stratification_runs": len(self.stratification_history),
            "latest_run": latest,
            "complexity_configurations": {
                level_id: {
                    "level_name": config.level,
                    "weight": config.weight,
                    "complexity_range": config.complexity_range,
                    "characteristics": config.characteristics
                }
                for level_id, config in self.complexity_configs.items()
            },
            "total_complexity_levels": len(self.complexity_configs)
        }


    def export_complexity_config(self, output_path: str):
        """Export current complexity configuration"""
        config_data = {}
        for level_id, config in self.complexity_configs.items():
            config_data[level_id] = {
                "level": config.level,
                "weight": config.weight,
                "min_samples": config.min_samples,
                "max_samples": config.max_samples,
                "complexity_range": config.complexity_range,
                "characteristics": config.characteristics
            }

        with open(output_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Complexity configuration exported to {output_path}")

def main():
    """Example usage of the Complexity Stratifier"""
    # Initialize stratifier
    stratifier = ComplexityStratifier()

    # Example conversations with varying complexity
    example_conversations = [
        {
            "id": "conv_1",
            "messages": [
                {"content": "I feel a bit stressed about work lately. Can you help me with some relaxation techniques?", "role": "client"},
                {"content": "Of course. Let's start with some simple breathing exercises and self-care strategies.", "role": "therapist"}
            ]
        },
        {
            "id": "conv_2",
            "messages": [
                {"content": "I'm struggling with depression and anxiety, and it's affecting my relationships and work performance.", "role": "client"},
                {"content": "That sounds challenging. Let's explore cognitive behavioral techniques and develop coping strategies.", "role": "therapist"}
            ]
        },
        {
            "id": "conv_3",
            "messages": [
                {"content": "I have complex PTSD from childhood trauma, borderline personality disorder, and I'm having suicidal thoughts. Everything feels overwhelming.", "role": "client"},
                {"content": "I hear how much pain you're in. Let's focus on crisis intervention and dialectical behavior therapy techniques for emotional regulation.", "role": "therapist"}
            ]
        }
    ] * 50  # Simulate larger dataset

    # Perform complexity stratification
    results = stratifier.stratify_conversations(example_conversations, target_total=300)

    # Display results
    for _result in results:
        pass

    # Export configuration
    stratifier.export_complexity_config("complexity_config.json")

    # Get statistics
    stratifier.get_stratification_statistics()

if __name__ == "__main__":
    main()
