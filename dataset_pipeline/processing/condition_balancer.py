#!/usr/bin/env python3
"""
Condition-Specific Balancing System for Task 6.20
Balances conversations across 20+ mental health conditions based on real-world prevalence.

Handles comorbidity, ensures minimum representation for rare conditions,
and implements quality-weighted condition balancing.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConditionConfig:
    """Configuration for each mental health condition"""
    name: str
    prevalence: float  # Real-world prevalence (0.0-1.0)
    min_samples: int
    max_samples: int | None = None
    aliases: list[str] = None  # Alternative names/keywords
    comorbid_conditions: list[str] = None  # Common comorbidities
    severity_levels: list[str] = None  # Mild, moderate, severe

@dataclass
class ConditionBalance:
    """Result of condition balancing"""
    condition: str
    target_samples: int
    actual_samples: int
    prevalence_weight: float
    quality_adjustment: float
    conversations: list[dict[str, Any]]
    metadata: dict[str, Any]

class ConditionBalancer:
    """
    Advanced condition-specific balancing system for therapeutic conversations.
    Ensures balanced representation across 20+ mental health conditions.
    """

    def __init__(self, config_path: str | None = None):
        """Initialize the condition balancer"""
        self.condition_configs = self._load_condition_configs(config_path)
        self.condition_keywords = self._build_keyword_mapping()
        self.balancing_history = []

    def _load_condition_configs(self, config_path: str | None = None) -> dict[str, ConditionConfig]:
        """Load condition configurations with real-world prevalence data"""
        # Based on NIMH and WHO prevalence data
        default_configs = {
            "depression": ConditionConfig(
                name="Major Depressive Disorder",
                prevalence=0.084,  # 8.4% annual prevalence
                min_samples=500,
                max_samples=8000,
                aliases=["depression", "depressed", "major depression", "mdd", "sad", "sadness"],
                comorbid_conditions=["anxiety", "ptsd", "substance_abuse"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "anxiety": ConditionConfig(
                name="Generalized Anxiety Disorder",
                prevalence=0.031,  # 3.1% annual prevalence
                min_samples=400,
                max_samples=6000,
                aliases=["anxiety", "anxious", "gad", "worry", "worried", "panic"],
                comorbid_conditions=["depression", "ptsd", "ocd"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "ptsd": ConditionConfig(
                name="Post-Traumatic Stress Disorder",
                prevalence=0.037,  # 3.7% lifetime prevalence
                min_samples=300,
                max_samples=4000,
                aliases=["ptsd", "trauma", "traumatic", "flashback", "nightmares"],
                comorbid_conditions=["depression", "anxiety", "substance_abuse"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "bipolar": ConditionConfig(
                name="Bipolar Disorder",
                prevalence=0.028,  # 2.8% lifetime prevalence
                min_samples=250,
                max_samples=3000,
                aliases=["bipolar", "manic", "mania", "mood swings", "hypomania"],
                comorbid_conditions=["anxiety", "substance_abuse", "adhd"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "adhd": ConditionConfig(
                name="Attention-Deficit/Hyperactivity Disorder",
                prevalence=0.041,  # 4.1% adult prevalence
                min_samples=300,
                max_samples=4000,
                aliases=["adhd", "add", "attention deficit", "hyperactive", "inattentive"],
                comorbid_conditions=["anxiety", "depression", "bipolar"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "ocd": ConditionConfig(
                name="Obsessive-Compulsive Disorder",
                prevalence=0.012,  # 1.2% annual prevalence
                min_samples=150,
                max_samples=2000,
                aliases=["ocd", "obsessive", "compulsive", "intrusive thoughts", "rituals"],
                comorbid_conditions=["anxiety", "depression", "tics"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "autism": ConditionConfig(
                name="Autism Spectrum Disorder",
                prevalence=0.016,  # 1.6% prevalence
                min_samples=200,
                max_samples=2500,
                aliases=["autism", "asd", "asperger", "autistic", "spectrum"],
                comorbid_conditions=["anxiety", "depression", "adhd"],
                severity_levels=["level 1", "level 2", "level 3"]
            ),
            "bpd": ConditionConfig(
                name="Borderline Personality Disorder",
                prevalence=0.014,  # 1.4% prevalence
                min_samples=150,
                max_samples=2000,
                aliases=["bpd", "borderline", "personality disorder", "emotional dysregulation"],
                comorbid_conditions=["depression", "anxiety", "ptsd", "substance_abuse"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "schizophrenia": ConditionConfig(
                name="Schizophrenia",
                prevalence=0.011,  # 1.1% lifetime prevalence
                min_samples=100,
                max_samples=1500,
                aliases=["schizophrenia", "psychosis", "hallucinations", "delusions"],
                comorbid_conditions=["depression", "anxiety", "substance_abuse"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "eating_disorders": ConditionConfig(
                name="Eating Disorders",
                prevalence=0.009,  # 0.9% prevalence (anorexia + bulimia)
                min_samples=100,
                max_samples=1500,
                aliases=["anorexia", "bulimia", "binge eating", "eating disorder", "body image"],
                comorbid_conditions=["depression", "anxiety", "ocd"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "substance_abuse": ConditionConfig(
                name="Substance Use Disorders",
                prevalence=0.104,  # 10.4% annual prevalence
                min_samples=400,
                max_samples=6000,
                aliases=["addiction", "substance abuse", "alcoholism", "drug abuse", "dependency"],
                comorbid_conditions=["depression", "anxiety", "ptsd", "bipolar"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "social_anxiety": ConditionConfig(
                name="Social Anxiety Disorder",
                prevalence=0.073,  # 7.3% annual prevalence
                min_samples=300,
                max_samples=4000,
                aliases=["social anxiety", "social phobia", "shy", "shyness", "social fear"],
                comorbid_conditions=["depression", "anxiety", "avoidant_personality"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "panic_disorder": ConditionConfig(
                name="Panic Disorder",
                prevalence=0.028,  # 2.8% annual prevalence
                min_samples=200,
                max_samples=3000,
                aliases=["panic disorder", "panic attacks", "agoraphobia", "panic"],
                comorbid_conditions=["anxiety", "depression", "substance_abuse"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "insomnia": ConditionConfig(
                name="Insomnia and Sleep Disorders",
                prevalence=0.060,  # 6.0% prevalence
                min_samples=250,
                max_samples=3500,
                aliases=["insomnia", "sleep disorder", "sleepless", "sleep problems"],
                comorbid_conditions=["depression", "anxiety", "bipolar"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "chronic_pain": ConditionConfig(
                name="Chronic Pain and Mental Health",
                prevalence=0.050,  # 5.0% with mental health comorbidity
                min_samples=200,
                max_samples=3000,
                aliases=["chronic pain", "fibromyalgia", "pain", "chronic illness"],
                comorbid_conditions=["depression", "anxiety", "ptsd"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "grief": ConditionConfig(
                name="Grief and Bereavement",
                prevalence=0.035,  # 3.5% complicated grief
                min_samples=150,
                max_samples=2500,
                aliases=["grief", "bereavement", "loss", "mourning", "death"],
                comorbid_conditions=["depression", "anxiety", "ptsd"],
                severity_levels=["normal", "complicated", "prolonged"]
            ),
            "relationship_issues": ConditionConfig(
                name="Relationship and Interpersonal Issues",
                prevalence=0.080,  # 8.0% seeking help
                min_samples=300,
                max_samples=4500,
                aliases=["relationship", "marriage", "divorce", "breakup", "interpersonal"],
                comorbid_conditions=["depression", "anxiety", "attachment_issues"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "work_stress": ConditionConfig(
                name="Work-Related Stress and Burnout",
                prevalence=0.070,  # 7.0% severe work stress
                min_samples=250,
                max_samples=3500,
                aliases=["work stress", "burnout", "job stress", "workplace", "career"],
                comorbid_conditions=["depression", "anxiety", "insomnia"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "parenting_stress": ConditionConfig(
                name="Parenting Stress and Family Issues",
                prevalence=0.045,  # 4.5% severe parenting stress
                min_samples=200,
                max_samples=3000,
                aliases=["parenting", "family stress", "child behavior", "parental stress"],
                comorbid_conditions=["depression", "anxiety", "relationship_issues"],
                severity_levels=["mild", "moderate", "severe"]
            ),
            "loneliness": ConditionConfig(
                name="Loneliness and Social Isolation",
                prevalence=0.055,  # 5.5% severe loneliness
                min_samples=200,
                max_samples=3000,
                aliases=["loneliness", "lonely", "isolated", "social isolation", "alone"],
                comorbid_conditions=["depression", "anxiety", "social_anxiety"],
                severity_levels=["mild", "moderate", "severe"]
            )
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    custom_config = json.load(f)
                # Update default configs with custom values
                for condition_id, config_data in custom_config.items():
                    if condition_id in default_configs:
                        for key, value in config_data.items():
                            setattr(default_configs[condition_id], key, value)
            except Exception as e:
                logger.warning(f"Failed to load custom config: {e}. Using defaults.")

        return default_configs

    def _build_keyword_mapping(self) -> dict[str, list[str]]:
        """Build mapping from keywords to conditions"""
        keyword_mapping = defaultdict(list)

        for condition_id, config in self.condition_configs.items():
            # Add main condition name
            for word in config.name.lower().split():
                keyword_mapping[word].append(condition_id)

            # Add aliases
            if config.aliases:
                for alias in config.aliases:
                    for word in alias.lower().split():
                        keyword_mapping[word].append(condition_id)

        return dict(keyword_mapping)

    def detect_conditions(self, conversation: dict[str, Any]) -> dict[str, float]:
        """
        Detect mental health conditions mentioned in a conversation.
        Returns condition scores (0.0-1.0) for each detected condition.
        """
        content = str(conversation).lower()
        condition_scores = defaultdict(float)

        # Count keyword matches
        for keyword, conditions in self.condition_keywords.items():
            if keyword in content:
                # Weight by keyword specificity (fewer conditions = higher weight)
                weight = 1.0 / len(conditions)
                for condition in conditions:
                    condition_scores[condition] += weight

        # Normalize scores
        if condition_scores:
            max_score = max(condition_scores.values())
            for condition in condition_scores:
                condition_scores[condition] /= max_score

        # Filter out very low scores
        return {k: v for k, v in condition_scores.items() if v >= 0.1}

    def assess_condition_severity(self, conversation: dict[str, Any],
                                condition: str) -> str:
        """Assess the severity level of a condition in the conversation"""
        content = str(conversation).lower()

        # Severity indicators
        severe_indicators = [
            "severe", "extreme", "unbearable", "crisis", "emergency",
            "suicidal", "can't function", "completely unable", "hospitalized"
        ]

        moderate_indicators = [
            "moderate", "significant", "interfering", "difficult",
            "struggling", "impacting", "affecting daily life"
        ]

        mild_indicators = [
            "mild", "slight", "manageable", "occasional", "sometimes",
            "minor", "little bit", "somewhat"
        ]

        # Count indicators
        severe_count = sum(1 for indicator in severe_indicators if indicator in content)
        moderate_count = sum(1 for indicator in moderate_indicators if indicator in content)
        mild_count = sum(1 for indicator in mild_indicators if indicator in content)

        # Determine severity
        if severe_count > 0:
            return "severe"
        if moderate_count > 0:
            return "moderate"
        if mild_count > 0:
            return "mild"
        return "moderate"  # Default to moderate

    def handle_comorbidity(self, conversation: dict[str, Any],
                          detected_conditions: dict[str, float]) -> dict[str, float]:
        """
        Handle comorbid conditions by adjusting scores based on known comorbidities.
        """
        adjusted_scores = detected_conditions.copy()

        for condition, score in detected_conditions.items():
            config = self.condition_configs.get(condition)
            if not config or not config.comorbid_conditions:
                continue

            # Boost scores for known comorbid conditions
            for comorbid in config.comorbid_conditions:
                if comorbid in detected_conditions:
                    # Increase both conditions' scores slightly
                    boost = 0.1 * min(score, detected_conditions[comorbid])
                    adjusted_scores[condition] += boost
                    adjusted_scores[comorbid] += boost

        # Normalize again
        if adjusted_scores:
            max_score = max(adjusted_scores.values())
            for condition in adjusted_scores:
                adjusted_scores[condition] = min(1.0, adjusted_scores[condition] / max_score)

        return adjusted_scores

    def calculate_target_distribution(self, total_samples: int,
                                    available_data: dict[str, list[dict]]) -> dict[str, int]:
        """
        Calculate target sample distribution based on prevalence and available data.
        """
        logger.info(f"Calculating target distribution for {total_samples} samples")

        # Calculate base distribution from prevalence
        total_prevalence = sum(config.prevalence for config in self.condition_configs.values())
        base_distribution = {}

        for condition_id, config in self.condition_configs.items():
            # Base samples from prevalence
            prevalence_ratio = config.prevalence / total_prevalence
            base_samples = int(total_samples * prevalence_ratio)

            # Ensure minimum samples
            base_samples = max(base_samples, config.min_samples)

            # Ensure maximum samples
            if config.max_samples:
                base_samples = min(base_samples, config.max_samples)

            # Ensure we don't exceed available data
            available = len(available_data.get(condition_id, []))
            base_samples = min(base_samples, available)

            base_distribution[condition_id] = base_samples

            logger.info(f"{config.name}: {base_samples} samples "
                       f"(prevalence: {config.prevalence:.1%}, available: {available})")

        return base_distribution

    def quality_weighted_selection(self, conversations: list[dict[str, Any]],
                                 target_count: int, condition: str) -> list[dict[str, Any]]:
        """
        Select conversations with quality weighting for a specific condition.
        """
        if not conversations or target_count <= 0:
            return []

        # Score conversations for this condition
        scored_conversations = []
        for conv in conversations:
            # Detect condition strength
            condition_scores = self.detect_conditions(conv)
            condition_strength = condition_scores.get(condition, 0.0)

            # Assess severity appropriateness
            severity = self.assess_condition_severity(conv, condition)
            severity_weight = {"mild": 0.8, "moderate": 1.0, "severe": 1.2}.get(severity, 1.0)

            # Calculate overall quality score
            quality_score = condition_strength * severity_weight

            scored_conversations.append((conv, quality_score))

        # Sort by quality score
        scored_conversations.sort(key=lambda x: x[1], reverse=True)

        # Select top conversations
        selected_count = min(target_count, len(scored_conversations))
        return [conv for conv, _ in scored_conversations[:selected_count]]


    def balance_conditions(self, conversations: list[dict[str, Any]],
                          target_total: int = 10000) -> list[ConditionBalance]:
        """
        Main method to balance conversations across mental health conditions.
        """
        logger.info(f"Starting condition balancing for {len(conversations)} conversations")

        # Categorize conversations by detected conditions
        condition_conversations = defaultdict(list)

        for conv in conversations:
            detected = self.detect_conditions(conv)
            adjusted = self.handle_comorbidity(conv, detected)

            # Assign to primary condition (highest score)
            if adjusted:
                primary_condition = max(adjusted.items(), key=lambda x: x[1])[0]
                condition_conversations[primary_condition].append(conv)

        # Calculate target distribution
        target_distribution = self.calculate_target_distribution(
            target_total, condition_conversations
        )

        # Balance each condition
        results = []
        total_balanced = 0

        for condition_id, target_count in target_distribution.items():
            if target_count <= 0:
                continue

            config = self.condition_configs[condition_id]
            available_conversations = condition_conversations.get(condition_id, [])

            logger.info(f"Balancing {config.name}: {target_count} target, "
                       f"{len(available_conversations)} available")

            # Select quality-weighted conversations
            selected = self.quality_weighted_selection(
                available_conversations, target_count, condition_id
            )

            if selected:
                # Calculate metrics
                prevalence_weight = config.prevalence
                quality_adjustment = len(selected) / max(1, len(available_conversations))

                result = ConditionBalance(
                    condition=config.name,
                    target_samples=target_count,
                    actual_samples=len(selected),
                    prevalence_weight=prevalence_weight,
                    quality_adjustment=quality_adjustment,
                    conversations=selected,
                    metadata={
                        "condition_id": condition_id,
                        "prevalence": config.prevalence,
                        "available_count": len(available_conversations),
                        "min_samples": config.min_samples,
                        "max_samples": config.max_samples,
                        "aliases": config.aliases,
                        "comorbid_conditions": config.comorbid_conditions
                    }
                )

                results.append(result)
                total_balanced += len(selected)

                logger.info(f"Balanced {config.name}: {len(selected)} conversations "
                           f"(quality: {quality_adjustment:.3f})")

        logger.info(f"Total balanced: {total_balanced} conversations across "
                   f"{len(results)} conditions")

        # Store balancing history
        self.balancing_history.append({
            "timestamp": str(np.datetime64("now")),
            "target_total": target_total,
            "actual_total": total_balanced,
            "conditions_balanced": len(results),
            "condition_results": {r.condition: r.metadata for r in results}
        })

        return results

    def get_condition_statistics(self) -> dict[str, Any]:
        """Get comprehensive condition balancing statistics"""
        if not self.balancing_history:
            return {"error": "No balancing history available"}

        latest = self.balancing_history[-1]

        return {
            "total_balancing_runs": len(self.balancing_history),
            "latest_run": latest,
            "condition_configurations": {
                condition_id: {
                    "name": config.name,
                    "prevalence": config.prevalence,
                    "min_samples": config.min_samples,
                    "max_samples": config.max_samples
                }
                for condition_id, config in self.condition_configs.items()
            },
            "total_conditions": len(self.condition_configs),
            "keyword_mappings": len(self.condition_keywords)
        }


    def export_condition_config(self, output_path: str):
        """Export current condition configuration"""
        config_data = {}
        for condition_id, config in self.condition_configs.items():
            config_data[condition_id] = {
                "name": config.name,
                "prevalence": config.prevalence,
                "min_samples": config.min_samples,
                "max_samples": config.max_samples,
                "aliases": config.aliases,
                "comorbid_conditions": config.comorbid_conditions,
                "severity_levels": config.severity_levels
            }

        with open(output_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Condition configuration exported to {output_path}")

def main():
    """Example usage of the Condition Balancer"""
    # Initialize balancer
    balancer = ConditionBalancer()

    # Example conversations (would be loaded from actual datasets)
    example_conversations = [
        {
            "id": "conv_1",
            "messages": [
                {"content": "I have been feeling very depressed and anxious lately. I can barely get out of bed.", "role": "user"},
                {"content": "I understand you're struggling with depression and anxiety. Let's work on some coping strategies.", "role": "therapist"}
            ]
        },
        {
            "id": "conv_2",
            "messages": [
                {"content": "I keep having panic attacks and I'm afraid to leave my house.", "role": "user"},
                {"content": "Panic disorder with agoraphobia can be very challenging. We can use exposure therapy techniques.", "role": "therapist"}
            ]
        },
        {
            "id": "conv_3",
            "messages": [
                {"content": "My ADHD makes it impossible to focus at work. I'm also feeling really anxious about my performance.", "role": "user"},
                {"content": "ADHD and anxiety often occur together. Let's develop strategies for both conditions.", "role": "therapist"}
            ]
        }
    ] * 100  # Simulate larger dataset

    # Perform condition balancing
    results = balancer.balance_conditions(example_conversations, target_total=500)

    # Display results
    for _result in results:
        pass

    # Export configuration
    balancer.export_condition_config("condition_config.json")

    # Get statistics
    balancer.get_condition_statistics()

if __name__ == "__main__":
    main()
