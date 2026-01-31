#!/usr/bin/env python3
"""
Priority-Weighted Sampling Algorithms for Task 6.19
Implements tiered sampling across the 6-tier data ecosystem with dynamic weight adjustment.

Tier Distribution:
- Tier 1 (Priority): 40% - Highest quality curated datasets
- Tier 2 (Professional): 25% - Clinical-grade therapeutic data
- Tier 3 (CoT Reasoning): 20% - Advanced reasoning patterns
- Tier 4 (Reddit Archive): 10% - Real-world mental health data
- Tier 5 (Research): 4% - Academic research datasets
- Tier 6 (Knowledge Base): 1% - Reference materials
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Import sampling validation
try:
    from .sampling_validation import SamplingValidationReport, SamplingValidator
except ImportError:
    from sampling_validation import SamplingValidationReport, SamplingValidator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TierConfig:
    """Configuration for each data tier"""
    name: str
    weight: float
    quality_threshold: float
    min_samples: int
    max_samples: int | None = None

@dataclass
class SamplingResult:
    """Result of sampling operation"""
    tier: str
    samples: list[dict[str, Any]]
    actual_weight: float
    quality_score: float
    metadata: dict[str, Any]

class PriorityWeightedSampler:
    """
    Advanced sampling system for the 6-tier therapeutic conversation ecosystem.
    Implements intelligent sampling with quality-based weight adjustment.
    """

    def __init__(self, config_path: str | None = None):
        """Initialize the priority-weighted sampler"""
        self.tier_configs = self._load_tier_configs(config_path)
        self.sampling_history = []
        self.quality_cache = {}

        # Initialize sampling validator
        self.validator = SamplingValidator()
        logger.info("Priority-weighted sampler initialized with validation system")

    def _load_tier_configs(self, config_path: str | None = None) -> dict[str, TierConfig]:
        """Load tier configurations with default values"""
        default_configs = {
            "tier_1_priority": TierConfig(
                name="Priority Datasets",
                weight=0.40,
                quality_threshold=0.70,  # Reduced from 0.85
                min_samples=1000,
                max_samples=40000
            ),
            "tier_2_professional": TierConfig(
                name="Professional Therapeutic",
                weight=0.25,
                quality_threshold=0.65,  # Reduced from 0.80
                min_samples=500,
                max_samples=25000
            ),
            "tier_3_cot": TierConfig(
                name="Chain-of-Thought Reasoning",
                weight=0.20,
                quality_threshold=0.60,  # Reduced from 0.75
                min_samples=400,
                max_samples=20000
            ),
            "tier_4_reddit": TierConfig(
                name="Reddit Mental Health Archive",
                weight=0.10,
                quality_threshold=0.55,  # Reduced from 0.70
                min_samples=200,
                max_samples=10000
            ),
            "tier_5_research": TierConfig(
                name="Research Datasets",
                weight=0.04,
                quality_threshold=0.50,  # Reduced from 0.65
                min_samples=100,
                max_samples=4000
            ),
            "tier_6_knowledge": TierConfig(
                name="Knowledge Base",
                weight=0.01,
                quality_threshold=0.45,  # Reduced from 0.60
                min_samples=50,
                max_samples=1000
            )
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    custom_config = json.load(f)
                # Update default configs with custom values
                for tier_id, config_data in custom_config.items():
                    if tier_id in default_configs:
                        for key, value in config_data.items():
                            setattr(default_configs[tier_id], key, value)
            except Exception as e:
                logger.warning(f"Failed to load custom config: {e}. Using defaults.")

        return default_configs

    def calculate_quality_score(self, conversation: dict[str, Any]) -> float:
        """
        Calculate quality score for a conversation.
        Uses multiple factors: coherence, therapeutic accuracy, safety, etc.
        """
        # Check cache first
        conv_id = conversation.get("id", str(hash(str(conversation))))
        if conv_id in self.quality_cache:
            return self.quality_cache[conv_id]

        quality_factors = {
            "coherence": 0.25,
            "therapeutic_accuracy": 0.30,
            "emotional_authenticity": 0.20,
            "safety_compliance": 0.15,
            "language_quality": 0.10
        }

        total_score = 0.0

        # Coherence scoring (0.0-1.0)
        coherence = self._assess_coherence(conversation)
        total_score += coherence * quality_factors["coherence"]

        # Therapeutic accuracy (0.0-1.0)
        therapeutic = self._assess_therapeutic_accuracy(conversation)
        total_score += therapeutic * quality_factors["therapeutic_accuracy"]

        # Emotional authenticity (0.0-1.0)
        emotional = self._assess_emotional_authenticity(conversation)
        total_score += emotional * quality_factors["emotional_authenticity"]

        # Safety compliance (0.0-1.0)
        safety = self._assess_safety_compliance(conversation)
        total_score += safety * quality_factors["safety_compliance"]

        # Language quality (0.0-1.0)
        language = self._assess_language_quality(conversation)
        total_score += language * quality_factors["language_quality"]

        # Cache the result
        self.quality_cache[conv_id] = total_score

        return total_score

    def _assess_coherence(self, conversation: dict[str, Any]) -> float:
        """Assess conversation coherence"""
        messages = conversation.get("messages", [])
        if len(messages) < 2:
            return 0.5

        # Simple coherence metrics
        coherence_score = 0.8  # Base score

        # Check for topic consistency
        topics = [msg.get("topic", "") for msg in messages if msg.get("topic")]
        if topics and len(set(topics)) / len(topics) > 0.7:
            coherence_score -= 0.2  # Penalty for topic jumping

        # Check for appropriate response lengths
        lengths = [len(msg.get("content", "")) for msg in messages]
        if lengths:
            avg_length = np.mean(lengths)
            if avg_length < 20 or avg_length > 1000:
                coherence_score -= 0.1

        return max(0.0, min(1.0, coherence_score))

    def _assess_therapeutic_accuracy(self, conversation: dict[str, Any]) -> float:
        """Assess therapeutic accuracy"""
        # Look for therapeutic indicators
        therapeutic_keywords = [
            "cbt", "cognitive", "behavioral", "therapy", "therapeutic",
            "mindfulness", "coping", "strategy", "intervention", "treatment",
            "anxiety", "depression", "feel", "emotion", "help", "better"
        ]

        content = str(conversation).lower()
        keyword_count = sum(1 for keyword in therapeutic_keywords if keyword in content)

        # More discriminating base score
        if keyword_count == 0:
            base_score = 0.2  # Very low for no therapeutic content
        elif keyword_count <= 2:
            base_score = 0.4  # Low for minimal therapeutic content
        else:
            base_score = 0.6  # Higher base only with substantial therapeutic content

        keyword_bonus = min(0.4, keyword_count * 0.05)  # Bonus for additional keywords

        return min(1.0, base_score + keyword_bonus)

    def _assess_emotional_authenticity(self, conversation: dict[str, Any]) -> float:
        """Assess emotional authenticity"""
        # Look for emotional indicators
        emotion_keywords = [
            "feel", "emotion", "sad", "happy", "angry", "anxious",
            "depressed", "worried", "stressed", "overwhelmed"
        ]

        content = str(conversation).lower()
        emotion_count = sum(1 for keyword in emotion_keywords if keyword in content)

        # Emotional authenticity based on presence of emotional language
        if emotion_count == 0:
            return 0.4  # Low authenticity without emotional content
        if emotion_count <= 3:
            return 0.7  # Moderate authenticity
        return 0.9  # High authenticity with rich emotional content

    def _assess_safety_compliance(self, conversation: dict[str, Any]) -> float:
        """Assess safety compliance"""
        # Check for harmful content indicators
        harmful_keywords = [
            "suicide", "kill", "harm", "hurt", "violence", "abuse",
            "dangerous", "illegal", "drug", "weapon"
        ]

        content = str(conversation).lower()
        harmful_count = sum(1 for keyword in harmful_keywords if keyword in content)

        # Safety compliance decreases with harmful content
        if harmful_count == 0:
            return 1.0  # Perfect safety compliance
        if harmful_count <= 2:
            return 0.8  # Acceptable with context (e.g., discussing safety)
        return 0.3  # Poor safety compliance

    def _assess_language_quality(self, conversation: dict[str, Any]) -> float:
        """Assess language quality"""
        messages = conversation.get("messages", [])
        if not messages:
            return 0.0

        total_quality = 0.0
        for message in messages:
            content = message.get("content", "")
            if not content:
                continue

            # Basic language quality metrics
            word_count = len(content.split())
            sentence_count = content.count(".") + content.count("!") + content.count("?")

            if word_count == 0:
                quality = 0.0
            elif sentence_count == 0:
                quality = 0.5  # No punctuation
            else:
                avg_words_per_sentence = word_count / sentence_count
                if 5 <= avg_words_per_sentence <= 25:
                    quality = 0.9  # Good sentence structure
                else:
                    quality = 0.6  # Suboptimal sentence structure

            total_quality += quality

        return total_quality / len(messages) if messages else 0.0

    def adjust_weights_by_quality(self, tier_data: dict[str, list[dict]],
                                 target_total: int) -> dict[str, int]:
        """
        Dynamically adjust sampling weights based on available data quality.
        """
        logger.info("Adjusting weights based on quality assessment...")

        # Calculate average quality for each tier
        tier_qualities = {}
        tier_counts = {}

        for tier_id, conversations in tier_data.items():
            if not conversations:
                tier_qualities[tier_id] = 0.0
                tier_counts[tier_id] = 0
                continue

            # Sample a subset for quality assessment (performance optimization)
            sample_size = min(100, len(conversations))
            sample_conversations = random.sample(conversations, sample_size)

            qualities = [self.calculate_quality_score(conv) for conv in sample_conversations]
            tier_qualities[tier_id] = np.mean(qualities)
            tier_counts[tier_id] = len(conversations)

            logger.info(f"Tier {tier_id}: {len(conversations)} conversations, "
                       f"avg quality: {tier_qualities[tier_id]:.3f}")

        # Adjust weights based on quality and availability
        adjusted_weights = {}
        total_adjusted_weight = 0.0

        for tier_id, config in self.tier_configs.items():
            base_weight = config.weight
            quality = tier_qualities.get(tier_id, 0.0)
            count = tier_counts.get(tier_id, 0)

            # Quality adjustment factor (0.5 to 1.5 multiplier)
            quality_factor = 0.5 + quality

            # Availability adjustment (reduce weight if insufficient data)
            min_required = config.min_samples
            availability_factor = min(1.0, count / min_required) if min_required > 0 else 1.0

            adjusted_weight = base_weight * quality_factor * availability_factor
            adjusted_weights[tier_id] = adjusted_weight
            total_adjusted_weight += adjusted_weight

        # Normalize weights and calculate sample counts
        sample_counts = {}
        total_allocated = 0

        for tier_id, adjusted_weight in adjusted_weights.items():
            normalized_weight = adjusted_weight / total_adjusted_weight
            sample_count = int(target_total * normalized_weight)

            # Ensure minimum samples (but scale with target_total)
            config = self.tier_configs[tier_id]
            # Scale minimum samples proportionally to target_total
            scaled_min_samples = max(1, int(config.min_samples * target_total / 100000))
            sample_count = max(scaled_min_samples, sample_count)

            # Ensure maximum samples
            if config.max_samples:
                scaled_max_samples = int(config.max_samples * target_total / 100000)
                sample_count = min(scaled_max_samples, sample_count)

            # Ensure we don't exceed available data
            available = tier_counts.get(tier_id, 0)
            sample_count = min(sample_count, available)

            sample_counts[tier_id] = sample_count
            total_allocated += sample_count

        # If we're still over target, proportionally reduce all tiers
        if total_allocated > target_total:
            reduction_factor = target_total / total_allocated
            for tier_id in sample_counts:
                sample_counts[tier_id] = max(1, int(sample_counts[tier_id] * reduction_factor))

        # Final check to ensure we don't exceed target
        total_final = sum(sample_counts.values())
        if total_final > target_total:
            # Remove excess from largest tiers first
            excess = total_final - target_total
            sorted_tiers = sorted(sample_counts.items(), key=lambda x: x[1], reverse=True)
            for tier_id, count in sorted_tiers:
                if excess <= 0:
                    break
                reduction = min(excess, count - 1)  # Keep at least 1 sample
                sample_counts[tier_id] -= reduction
                excess -= reduction

            logger.info(f"Tier {tier_id}: {sample_count} samples "
                       f"({normalized_weight:.1%} weight)")

        return sample_counts

    def stratified_sample(self, conversations: list[dict], target_count: int,
                         quality_threshold: float) -> list[dict]:
        """
        Perform stratified sampling ensuring quality and diversity.
        """
        if not conversations or target_count <= 0:
            return []

        # Filter by quality threshold
        quality_conversations = []
        for conv in conversations:
            quality = self.calculate_quality_score(conv)
            if quality >= quality_threshold:
                quality_conversations.append((conv, quality))

        if not quality_conversations:
            logger.warning(f"No conversations meet quality threshold {quality_threshold}")
            return []

        # Sort by quality (descending)
        quality_conversations.sort(key=lambda x: x[1], reverse=True)

        # Stratified sampling: take from different quality bands
        sample_size = min(target_count, len(quality_conversations))

        if sample_size == len(quality_conversations):
            return [conv for conv, _ in quality_conversations]

        # Create quality bands
        band_size = len(quality_conversations) // 3
        high_quality = quality_conversations[:band_size]
        mid_quality = quality_conversations[band_size:2*band_size]
        low_quality = quality_conversations[2*band_size:]

        # Sample proportionally from each band (70% high, 20% mid, 10% low)
        high_samples = int(sample_size * 0.7)
        mid_samples = int(sample_size * 0.2)
        low_samples = sample_size - high_samples - mid_samples

        sampled = []

        # Sample from high quality band
        if high_quality and high_samples > 0:
            high_sample_size = min(high_samples, len(high_quality))
            sampled.extend(random.sample(high_quality, high_sample_size))

        # Sample from mid quality band
        if mid_quality and mid_samples > 0:
            mid_sample_size = min(mid_samples, len(mid_quality))
            sampled.extend(random.sample(mid_quality, mid_sample_size))

        # Sample from low quality band
        if low_quality and low_samples > 0:
            low_sample_size = min(low_samples, len(low_quality))
            sampled.extend(random.sample(low_quality, low_sample_size))

        # Return just the conversations (not the quality scores)
        return [conv for conv, _ in sampled]

    def sample_from_tiers(self, tier_data: dict[str, list[dict]],
                         target_total: int = 100000) -> list[SamplingResult]:
        """
        Main sampling method that orchestrates the entire process.
        """
        logger.info(f"Starting priority-weighted sampling for {target_total} conversations")

        # Adjust weights based on quality and availability
        sample_counts = self.adjust_weights_by_quality(tier_data, target_total)

        results = []
        total_sampled = 0

        for tier_id, target_count in sample_counts.items():
            if target_count <= 0:
                continue

            config = self.tier_configs[tier_id]
            conversations = tier_data.get(tier_id, [])

            logger.info(f"Sampling {target_count} conversations from {config.name}")

            # Perform stratified sampling
            sampled = self.stratified_sample(
                conversations, target_count, config.quality_threshold
            )

            if sampled:
                # Calculate actual metrics
                actual_weight = len(sampled) / target_total
                avg_quality = np.mean([self.calculate_quality_score(conv) for conv in sampled])

                result = SamplingResult(
                    tier=config.name,
                    samples=sampled,
                    actual_weight=actual_weight,
                    quality_score=avg_quality,
                    metadata={
                        "tier_id": tier_id,
                        "target_count": target_count,
                        "actual_count": len(sampled),
                        "quality_threshold": config.quality_threshold,
                        "source_count": len(conversations)
                    }
                )

                results.append(result)
                total_sampled += len(sampled)

                logger.info(f"Sampled {len(sampled)} from {config.name} "
                           f"(quality: {avg_quality:.3f})")

        logger.info(f"Total sampled: {total_sampled} conversations")

        # Store sampling history
        self.sampling_history.append({
            "timestamp": str(np.datetime64("now")),
            "target_total": target_total,
            "actual_total": total_sampled,
            "tier_results": {r.tier: r.metadata for r in results}
        })

        return results

    def sample_with_validation(self, tier_data: dict[str, list[dict]],
                              target_total: int = 100000,
                              validation_threshold: float = 0.8) -> tuple[list[SamplingResult], SamplingValidationReport]:
        """
        Perform sampling with comprehensive validation.

        Args:
            tier_data: Dictionary of tier data to sample from
            target_total: Target total number of samples
            validation_threshold: Minimum validation score to accept results

        Returns:
            Tuple of (sampling_results, validation_report)
        """
        logger.info(f"Starting validated sampling for {target_total} conversations")

        # Perform sampling
        sampling_results = self.sample_from_tiers(tier_data, target_total)

        # Validate results
        validation_report = self.validator.validate_sampling_results(
            sampling_results, tier_data, target_total
        )

        # Check if validation passes threshold
        if validation_report.overall_score < validation_threshold:
            logger.warning(f"Sampling validation score {validation_report.overall_score:.3f} "
                          f"below threshold {validation_threshold}")
            logger.warning(f"Critical issues: {len(validation_report.critical_issues)}")
            for issue in validation_report.critical_issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info(f"Sampling validation passed with score {validation_report.overall_score:.3f}")

        return sampling_results, validation_report

    def get_validation_history(self) -> list[SamplingValidationReport]:
        """Get history of validation reports."""
        return self.validator.validation_history

    def export_validation_report(self, validation_report: SamplingValidationReport,
                                output_path: str) -> None:
        """Export validation report to file."""
        self.validator.export_validation_report(validation_report, output_path)
        logger.info(f"Validation report exported to {output_path}")

    def export_sampling_config(self, output_path: str):
        """Export current sampling configuration"""
        config_data = {}
        for tier_id, config in self.tier_configs.items():
            config_data[tier_id] = {
                "name": config.name,
                "weight": config.weight,
                "quality_threshold": config.quality_threshold,
                "min_samples": config.min_samples,
                "max_samples": config.max_samples
            }

        with open(output_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Sampling configuration exported to {output_path}")

    def get_sampling_statistics(self) -> dict[str, Any]:
        """Get comprehensive sampling statistics"""
        if not self.sampling_history:
            return {"error": "No sampling history available"}

        latest = self.sampling_history[-1]

        return {
            "total_sampling_runs": len(self.sampling_history),
            "latest_run": latest,
            "tier_configurations": {
                tier_id: {
                    "name": config.name,
                    "weight": config.weight,
                    "quality_threshold": config.quality_threshold
                }
                for tier_id, config in self.tier_configs.items()
            },
            "quality_cache_size": len(self.quality_cache)
        }


def main():
    """Example usage of the Priority-Weighted Sampler"""
    # Initialize sampler
    sampler = PriorityWeightedSampler()

    # Example tier data (would be loaded from actual datasets)
    example_tier_data = {
        "tier_1_priority": [
            {"id": f"priority_{i}", "messages": [
                {"content": f"High quality therapeutic conversation {i}", "role": "therapist"},
                {"content": f"Patient response {i}", "role": "patient"}
            ]} for i in range(1000)
        ],
        "tier_2_professional": [
            {"id": f"professional_{i}", "messages": [
                {"content": f"Professional therapeutic dialogue {i}", "role": "therapist"},
                {"content": f"Client response {i}", "role": "client"}
            ]} for i in range(800)
        ],
        "tier_3_cot": [
            {"id": f"cot_{i}", "messages": [
                {"content": f"Chain of thought reasoning {i}", "role": "assistant"},
                {"content": f"Reasoning response {i}", "role": "user"}
            ]} for i in range(600)
        ]
    }

    # Perform sampling
    results = sampler.sample_from_tiers(example_tier_data, target_total=1000)

    # Display results
    for _result in results:
        pass

    # Export configuration
    sampler.export_sampling_config("sampling_config.json")

    # Get statistics
    sampler.get_sampling_statistics()

if __name__ == "__main__":
    main()
