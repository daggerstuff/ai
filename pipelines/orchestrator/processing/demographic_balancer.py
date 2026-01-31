#!/usr/bin/env python3
"""
Demographic and Cultural Diversity Balancing for Task 6.22
Balances conversations across age, gender, cultural background, socioeconomic status,
and geographic distribution with bias detection and mitigation algorithms.
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
class DemographicConfig:
    """Configuration for demographic categories"""
    category: str
    subcategories: dict[str, float]  # subcategory -> target proportion
    keywords: dict[str, list[str]]  # subcategory -> keywords
    min_samples_per_subcategory: int = 50
    bias_indicators: list[str] = None

@dataclass
class DemographicBalance:
    """Result of demographic balancing"""
    category: str
    subcategory: str
    target_samples: int
    actual_samples: int
    proportion: float
    conversations: list[dict[str, Any]]
    bias_score: float
    metadata: dict[str, Any]

class DemographicBalancer:
    """
    Advanced demographic and cultural diversity balancing system.
    Ensures representative distribution across demographic dimensions.
    """

    def __init__(self, config_path: str | None = None):
        """Initialize the demographic balancer"""
        self.demographic_configs = self._load_demographic_configs(config_path)
        self.balancing_history = []
        self.bias_patterns = self._load_bias_patterns()

    def _load_demographic_configs(self, config_path: str | None = None) -> dict[str, DemographicConfig]:
        """Load demographic configurations based on census and research data"""
        default_configs = {
            "age": DemographicConfig(
                category="Age Groups",
                subcategories={
                    "young_adult": 0.25,    # 18-29
                    "adult": 0.35,          # 30-49
                    "middle_aged": 0.25,    # 50-64
                    "older_adult": 0.15     # 65+
                },
                keywords={
                    "young_adult": ["young", "college", "university", "twenties", "early career", "student"],
                    "adult": ["adult", "career", "thirties", "forties", "working", "professional"],
                    "middle_aged": ["middle-aged", "fifties", "midlife", "experienced", "senior"],
                    "older_adult": ["elderly", "senior", "retired", "older", "sixties", "seventies"]
                },
                min_samples_per_subcategory=100,
                bias_indicators=["ageism", "age discrimination", "generational bias"]
            ),
            "gender": DemographicConfig(
                category="Gender Identity",
                subcategories={
                    "female": 0.45,
                    "male": 0.45,
                    "non_binary": 0.05,
                    "other": 0.05
                },
                keywords={
                    "female": ["woman", "female", "she", "her", "mother", "daughter", "wife"],
                    "male": ["man", "male", "he", "him", "father", "son", "husband"],
                    "non_binary": ["non-binary", "nonbinary", "they", "them", "genderqueer"],
                    "other": ["transgender", "trans", "gender fluid", "questioning"]
                },
                min_samples_per_subcategory=80,
                bias_indicators=["sexism", "gender bias", "misogyny", "transphobia"]
            ),
            "cultural_background": DemographicConfig(
                category="Cultural Background",
                subcategories={
                    "western": 0.40,
                    "hispanic_latino": 0.20,
                    "african": 0.15,
                    "asian": 0.15,
                    "indigenous": 0.05,
                    "middle_eastern": 0.05
                },
                keywords={
                    "western": ["american", "european", "western", "caucasian", "white"],
                    "hispanic_latino": ["hispanic", "latino", "latina", "mexican", "spanish", "puerto rican"],
                    "african": ["african", "black", "african american", "caribbean"],
                    "asian": ["asian", "chinese", "japanese", "korean", "indian", "vietnamese"],
                    "indigenous": ["native", "indigenous", "tribal", "first nations"],
                    "middle_eastern": ["middle eastern", "arab", "persian", "turkish"]
                },
                min_samples_per_subcategory=60,
                bias_indicators=["racism", "cultural bias", "xenophobia", "stereotyping"]
            ),
            "socioeconomic": DemographicConfig(
                category="Socioeconomic Status",
                subcategories={
                    "low_income": 0.25,
                    "middle_income": 0.50,
                    "high_income": 0.25
                },
                keywords={
                    "low_income": ["poor", "poverty", "low income", "struggling financially", "unemployed", "welfare"],
                    "middle_income": ["middle class", "working class", "average income", "stable job"],
                    "high_income": ["wealthy", "rich", "high income", "affluent", "executive", "professional"]
                },
                min_samples_per_subcategory=70,
                bias_indicators=["classism", "economic bias", "poverty stigma"]
            ),
            "geographic": DemographicConfig(
                category="Geographic Distribution",
                subcategories={
                    "urban": 0.45,
                    "suburban": 0.35,
                    "rural": 0.20
                },
                keywords={
                    "urban": ["city", "urban", "downtown", "metropolitan", "apartment"],
                    "suburban": ["suburban", "suburbs", "neighborhood", "community"],
                    "rural": ["rural", "country", "farm", "small town", "village"]
                },
                min_samples_per_subcategory=60,
                bias_indicators=["urban bias", "rural stereotypes", "geographic discrimination"]
            )
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    custom_config = json.load(f)
                # Update configurations
                for category, config_data in custom_config.items():
                    if category in default_configs:
                        for key, value in config_data.items():
                            setattr(default_configs[category], key, value)
            except Exception as e:
                logger.warning(f"Failed to load custom config: {e}. Using defaults.")

        return default_configs

    def _load_bias_patterns(self) -> dict[str, list[str]]:
        """Load patterns that indicate demographic bias"""
        return {
            "age_bias": [
                "too old", "too young", "generational gap", "outdated", "inexperienced",
                "over the hill", "past their prime", "millennial", "boomer"
            ],
            "gender_bias": [
                "typical woman", "typical man", "act like a lady", "man up",
                "emotional female", "weak man", "bossy woman"
            ],
            "cultural_bias": [
                "those people", "their kind", "not from here", "foreign",
                "exotic", "primitive", "uncivilized", "different culture"
            ],
            "socioeconomic_bias": [
                "poor people", "rich snob", "trailer trash", "ghetto",
                "privileged", "entitled", "welfare queen", "trust fund"
            ],
            "geographic_bias": [
                "city folk", "country bumpkin", "hillbilly", "urban elite",
                "small town mentality", "big city problems"
            ]
        }

    def detect_demographics(self, conversation: dict[str, Any]) -> dict[str, dict[str, float]]:
        """
        Detect demographic indicators in a conversation.
        Returns category -> subcategory -> score mapping.
        """
        content = str(conversation).lower()
        demographic_scores = {}

        for category, config in self.demographic_configs.items():
            category_scores = {}

            for subcategory, keywords in config.keywords.items():
                score = 0.0
                for keyword in keywords:
                    if keyword.lower() in content:
                        # Weight by keyword specificity
                        score += 1.0 / len(keywords)

                if score > 0:
                    category_scores[subcategory] = min(1.0, score)

            if category_scores:
                demographic_scores[category] = category_scores

        return demographic_scores

    def assess_bias_score(self, conversation: dict[str, Any]) -> float:
        """
        Assess the level of demographic bias in a conversation.
        Returns bias score (0.0 = no bias, 1.0 = high bias).
        """
        content = str(conversation).lower()
        total_bias = 0.0
        bias_count = 0

        for _bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    total_bias += 1.0
                    bias_count += 1

        # Normalize bias score
        if bias_count > 0:
            return min(1.0, total_bias / len(self.bias_patterns))

        return 0.0

    def detect_bias_mitigation_opportunities(self, conversations: list[dict[str, Any]]) -> dict[str, list[str]]:
        """Identify opportunities for bias mitigation"""
        mitigation_opportunities = defaultdict(list)

        for conv in conversations:
            bias_score = self.assess_bias_score(conv)
            if bias_score > 0.3:  # Threshold for concerning bias
                content = str(conv).lower()

                for bias_type, patterns in self.bias_patterns.items():
                    for pattern in patterns:
                        if pattern in content:
                            mitigation_opportunities[bias_type].append({
                                "conversation_id": conv.get("id", "unknown"),
                                "bias_pattern": pattern,
                                "bias_score": bias_score,
                                "suggestion": f"Consider rephrasing to avoid '{pattern}' stereotype"
                            })

        return dict(mitigation_opportunities)

    def balance_demographics(self, conversations: list[dict[str, Any]],
                           target_total: int = 10000) -> list[DemographicBalance]:
        """
        Main method to balance conversations across demographic dimensions.
        """
        logger.info(f"Starting demographic balancing for {len(conversations)} conversations")

        # Categorize conversations by demographics
        demographic_conversations = defaultdict(lambda: defaultdict(list))

        for conv in conversations:
            detected = self.detect_demographics(conv)
            bias_score = self.assess_bias_score(conv)

            # Only include conversations with low bias
            if bias_score < 0.5:
                for category, subcategory_scores in detected.items():
                    if subcategory_scores:
                        # Assign to highest scoring subcategory
                        primary_subcategory = max(subcategory_scores.items(), key=lambda x: x[1])[0]
                        demographic_conversations[category][primary_subcategory].append(conv)

        # Balance each demographic category
        results = []

        for category, config in self.demographic_configs.items():
            logger.info(f"Balancing {config.category}")

            category_conversations = demographic_conversations[category]
            category_total = int(target_total / len(self.demographic_configs))

            for subcategory, target_proportion in config.subcategories.items():
                target_count = int(category_total * target_proportion)
                target_count = max(target_count, config.min_samples_per_subcategory)

                available_conversations = category_conversations.get(subcategory, [])

                # Select conversations with bias filtering
                selected = self._select_diverse_conversations(
                    available_conversations, target_count, subcategory
                )

                if selected:
                    # Calculate metrics
                    avg_bias_score = np.mean([self.assess_bias_score(conv) for conv in selected])

                    result = DemographicBalance(
                        category=config.category,
                        subcategory=subcategory,
                        target_samples=target_count,
                        actual_samples=len(selected),
                        proportion=len(selected) / category_total if category_total > 0 else 0,
                        conversations=selected,
                        bias_score=avg_bias_score,
                        metadata={
                            "category_id": category,
                            "target_proportion": target_proportion,
                            "available_count": len(available_conversations),
                            "keywords": config.keywords.get(subcategory, []),
                            "bias_indicators": config.bias_indicators
                        }
                    )

                    results.append(result)

                    logger.info(f"Balanced {subcategory}: {len(selected)} conversations "
                               f"(bias score: {avg_bias_score:.3f})")

        # Detect bias mitigation opportunities
        mitigation_opportunities = self.detect_bias_mitigation_opportunities(conversations)

        total_balanced = sum(r.actual_samples for r in results)
        logger.info(f"Total balanced: {total_balanced} conversations across "
                   f"{len(results)} demographic groups")

        # Store balancing history
        self.balancing_history.append({
            "timestamp": str(np.datetime64("now")),
            "target_total": target_total,
            "actual_total": total_balanced,
            "demographic_groups": len(results),
            "mitigation_opportunities": {k: len(v) for k, v in mitigation_opportunities.items()},
            "results": {f"{r.category}_{r.subcategory}": r.metadata for r in results}
        })

        return results

    def _select_diverse_conversations(self, conversations: list[dict[str, Any]],
                                    target_count: int, subcategory: str) -> list[dict[str, Any]]:
        """Select conversations with diversity optimization"""
        if not conversations or target_count <= 0:
            return []

        # Score conversations for diversity and quality
        scored_conversations = []
        for conv in conversations:
            # Bias score (lower is better)
            bias_score = self.assess_bias_score(conv)

            # Diversity score (variety of topics, perspectives)
            diversity_score = self._assess_diversity_score(conv)

            # Combined score (higher is better)
            combined_score = diversity_score * (1.0 - bias_score)

            scored_conversations.append((conv, combined_score))

        # Sort by combined score
        scored_conversations.sort(key=lambda x: x[1], reverse=True)

        # Select top conversations
        selected_count = min(target_count, len(scored_conversations))
        return [conv for conv, _ in scored_conversations[:selected_count]]


    def _assess_diversity_score(self, conversation: dict[str, Any]) -> float:
        """Assess diversity of perspectives and topics in conversation"""
        content = str(conversation).lower()

        # Diversity indicators
        diversity_indicators = [
            "perspective", "viewpoint", "different", "various", "multiple",
            "diverse", "inclusive", "variety", "range", "spectrum"
        ]

        # Topic variety indicators
        topic_indicators = [
            "experience", "background", "culture", "tradition", "belief",
            "value", "opinion", "approach", "method", "style"
        ]

        diversity_count = sum(1 for indicator in diversity_indicators if indicator in content)
        topic_count = sum(1 for indicator in topic_indicators if indicator in content)

        # Calculate diversity score
        diversity_score = (diversity_count + topic_count) / (len(diversity_indicators) + len(topic_indicators))

        return min(1.0, diversity_score * 2)  # Scale up and cap at 1.0

    def get_balancing_statistics(self) -> dict[str, Any]:
        """Get comprehensive balancing statistics"""
        if not self.balancing_history:
            return {"error": "No balancing history available"}

        latest = self.balancing_history[-1]

        return {
            "total_balancing_runs": len(self.balancing_history),
            "latest_run": latest,
            "demographic_configurations": {
                category: {
                    "category_name": config.category,
                    "subcategories": config.subcategories,
                    "min_samples_per_subcategory": config.min_samples_per_subcategory
                }
                for category, config in self.demographic_configs.items()
            },
            "total_demographic_categories": len(self.demographic_configs),
            "bias_patterns_tracked": len(self.bias_patterns)
        }


    def export_demographic_config(self, output_path: str):
        """Export current demographic configuration"""
        config_data = {}
        for category, config in self.demographic_configs.items():
            config_data[category] = {
                "category": config.category,
                "subcategories": config.subcategories,
                "keywords": config.keywords,
                "min_samples_per_subcategory": config.min_samples_per_subcategory,
                "bias_indicators": config.bias_indicators
            }

        with open(output_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Demographic configuration exported to {output_path}")

def main():
    """Example usage of the Demographic Balancer"""
    # Initialize balancer
    balancer = DemographicBalancer()

    # Example conversations with demographic indicators
    example_conversations = [
        {
            "id": "conv_1",
            "messages": [
                {"content": "As a young woman in her twenties, I struggle with anxiety about my career.", "role": "client"},
                {"content": "Many young adults face career anxiety. Let's explore your specific concerns.", "role": "therapist"}
            ]
        },
        {
            "id": "conv_2",
            "messages": [
                {"content": "I'm a middle-aged man dealing with depression after losing my job.", "role": "client"},
                {"content": "Job loss can be particularly challenging for men in midlife. How are you coping?", "role": "therapist"}
            ]
        },
        {
            "id": "conv_3",
            "messages": [
                {"content": "Growing up in a Hispanic family, mental health was never discussed.", "role": "client"},
                {"content": "Cultural attitudes toward mental health vary. How does your background influence your perspective?", "role": "therapist"}
            ]
        }
    ] * 50  # Simulate larger dataset

    # Perform demographic balancing
    results = balancer.balance_demographics(example_conversations, target_total=500)

    # Display results
    for _result in results:
        pass

    # Export configuration
    balancer.export_demographic_config("demographic_config.json")

    # Get statistics
    balancer.get_balancing_statistics()

if __name__ == "__main__":
    main()
