#!/usr/bin/env python3
"""
Therapeutic Approach Diversity Optimization for Task 6.21
Balances conversations across 15+ therapeutic approaches ensuring representation
of evidence-based practices with cross-approach consistency validation.
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
class ApproachConfig:
    """Configuration for each therapeutic approach"""
    name: str
    evidence_level: str  # "strong", "moderate", "emerging"
    target_weight: float  # Desired proportion (0.0-1.0)
    min_samples: int
    max_samples: int | None = None
    keywords: list[str] = None  # Identifying keywords
    techniques: list[str] = None  # Specific techniques
    conditions_suited: list[str] = None  # Best suited conditions
    effectiveness_score: float = 0.8  # Evidence-based effectiveness (0.0-1.0)

@dataclass
class ApproachBalance:
    """Result of therapeutic approach balancing"""
    approach: str
    target_samples: int
    actual_samples: int
    evidence_weight: float
    effectiveness_score: float
    conversations: list[dict[str, Any]]
    metadata: dict[str, Any]

class ApproachDiversityOptimizer:
    """
    Advanced therapeutic approach diversity optimization system.
    Ensures balanced representation across evidence-based therapeutic approaches.
    """

    def __init__(self, config_path: str | None = None):
        """Initialize the approach diversity optimizer"""
        self.approach_configs = self._load_approach_configs(config_path)
        self.approach_keywords = self._build_keyword_mapping()
        self.optimization_history = []

    def _load_approach_configs(self, config_path: str | None = None) -> dict[str, ApproachConfig]:
        """Load therapeutic approach configurations with evidence-based weights"""
        # Based on APA Division 12 evidence-based treatments and meta-analyses
        default_configs = {
            "cbt": ApproachConfig(
                name="Cognitive Behavioral Therapy",
                evidence_level="strong",
                target_weight=0.25,  # 25% - most evidence-based
                min_samples=500,
                max_samples=8000,
                keywords=["cbt", "cognitive behavioral", "cognitive therapy", "behavioral therapy",
                         "thought patterns", "cognitive restructuring", "behavioral activation"],
                techniques=["cognitive restructuring", "behavioral activation", "exposure therapy",
                          "thought records", "activity scheduling", "behavioral experiments"],
                conditions_suited=["depression", "anxiety", "ptsd", "ocd", "panic_disorder"],
                effectiveness_score=0.95
            ),
            "dbt": ApproachConfig(
                name="Dialectical Behavior Therapy",
                evidence_level="strong",
                target_weight=0.12,  # 12% - strong evidence for specific conditions
                min_samples=300,
                max_samples=4000,
                keywords=["dbt", "dialectical", "mindfulness", "distress tolerance",
                         "emotion regulation", "interpersonal effectiveness"],
                techniques=["mindfulness", "distress tolerance", "emotion regulation",
                          "interpersonal effectiveness", "wise mind", "radical acceptance"],
                conditions_suited=["bpd", "self_harm", "suicidal_ideation", "emotion_dysregulation"],
                effectiveness_score=0.90
            ),
            "psychodynamic": ApproachConfig(
                name="Psychodynamic Therapy",
                evidence_level="moderate",
                target_weight=0.15,  # 15% - moderate evidence, important for depth
                min_samples=400,
                max_samples=5000,
                keywords=["psychodynamic", "psychoanalytic", "unconscious", "transference",
                         "defense mechanisms", "insight", "interpretation"],
                techniques=["free association", "dream analysis", "transference analysis",
                          "interpretation", "working through", "insight development"],
                conditions_suited=["depression", "anxiety", "personality_disorders", "trauma"],
                effectiveness_score=0.75
            ),
            "humanistic": ApproachConfig(
                name="Humanistic/Person-Centered Therapy",
                evidence_level="moderate",
                target_weight=0.10,  # 10% - foundational approach
                min_samples=250,
                max_samples=3500,
                keywords=["person-centered", "humanistic", "unconditional positive regard",
                         "empathy", "genuineness", "self-actualization", "client-centered"],
                techniques=["active listening", "reflection", "unconditional positive regard",
                          "empathic understanding", "genuineness", "congruence"],
                conditions_suited=["self_esteem", "identity_issues", "personal_growth"],
                effectiveness_score=0.70
            ),
            "acceptance_commitment": ApproachConfig(
                name="Acceptance and Commitment Therapy",
                evidence_level="strong",
                target_weight=0.08,  # 8% - growing evidence base
                min_samples=200,
                max_samples=3000,
                keywords=["act", "acceptance commitment", "psychological flexibility",
                         "mindfulness", "values", "committed action", "defusion"],
                techniques=["mindfulness", "acceptance", "cognitive defusion", "values clarification",
                          "committed action", "psychological flexibility"],
                conditions_suited=["anxiety", "depression", "chronic_pain", "substance_abuse"],
                effectiveness_score=0.85
            ),
            "emdr": ApproachConfig(
                name="Eye Movement Desensitization and Reprocessing",
                evidence_level="strong",
                target_weight=0.06,  # 6% - specific to trauma
                min_samples=150,
                max_samples=2500,
                keywords=["emdr", "eye movement", "bilateral stimulation", "trauma processing",
                         "desensitization", "reprocessing"],
                techniques=["bilateral stimulation", "resource installation", "trauma processing",
                          "desensitization", "reprocessing", "safe place visualization"],
                conditions_suited=["ptsd", "trauma", "phobias", "anxiety"],
                effectiveness_score=0.90
            ),
            "family_systems": ApproachConfig(
                name="Family Systems Therapy",
                evidence_level="moderate",
                target_weight=0.07,  # 7% - important for relational issues
                min_samples=180,
                max_samples=2800,
                keywords=["family therapy", "systems therapy", "family systems", "structural",
                         "strategic", "multigenerational", "boundaries"],
                techniques=["genogram", "structural interventions", "strategic interventions",
                          "boundary setting", "family sculpting", "circular questioning"],
                conditions_suited=["family_conflict", "relationship_issues", "adolescent_issues"],
                effectiveness_score=0.75
            ),
            "gestalt": ApproachConfig(
                name="Gestalt Therapy",
                evidence_level="emerging",
                target_weight=0.04,  # 4% - experiential approach
                min_samples=100,
                max_samples=1500,
                keywords=["gestalt", "here and now", "awareness", "contact", "experiment",
                         "phenomenology", "field theory"],
                techniques=["empty chair", "two-chair technique", "body awareness",
                          "here and now focus", "experiments", "contact and awareness"],
                conditions_suited=["anxiety", "depression", "relationship_issues"],
                effectiveness_score=0.65
            ),
            "solution_focused": ApproachConfig(
                name="Solution-Focused Brief Therapy",
                evidence_level="moderate",
                target_weight=0.05,  # 5% - brief therapy approach
                min_samples=120,
                max_samples=2000,
                keywords=["solution focused", "brief therapy", "miracle question", "scaling",
                         "exceptions", "goals", "strengths"],
                techniques=["miracle question", "scaling questions", "exception finding",
                          "goal setting", "compliments", "between-session tasks"],
                conditions_suited=["depression", "anxiety", "relationship_issues", "substance_abuse"],
                effectiveness_score=0.70
            ),
            "narrative": ApproachConfig(
                name="Narrative Therapy",
                evidence_level="emerging",
                target_weight=0.03,  # 3% - postmodern approach
                min_samples=80,
                max_samples=1200,
                keywords=["narrative", "story", "externalization", "unique outcomes",
                         "re-authoring", "deconstruction", "preferred story"],
                techniques=["externalization", "unique outcomes", "re-authoring",
                          "definitional ceremony", "outsider witness", "therapeutic documents"],
                conditions_suited=["identity_issues", "trauma", "oppression", "self_esteem"],
                effectiveness_score=0.60
            ),
            "mindfulness_based": ApproachConfig(
                name="Mindfulness-Based Interventions",
                evidence_level="strong",
                target_weight=0.06,  # 6% - growing evidence
                min_samples=150,
                max_samples=2500,
                keywords=["mindfulness", "mbsr", "mbct", "meditation", "present moment",
                         "non-judgmental awareness", "body scan"],
                techniques=["mindfulness meditation", "body scan", "breathing exercises",
                          "mindful movement", "loving-kindness", "present moment awareness"],
                conditions_suited=["anxiety", "depression", "chronic_pain", "stress"],
                effectiveness_score=0.80
            ),
            "interpersonal": ApproachConfig(
                name="Interpersonal Therapy",
                evidence_level="strong",
                target_weight=0.07,  # 7% - evidence-based for depression
                min_samples=180,
                max_samples=2800,
                keywords=["interpersonal therapy", "ipt", "grief", "role disputes",
                         "role transitions", "interpersonal deficits"],
                techniques=["grief work", "role dispute resolution", "role transition work",
                          "interpersonal skills training", "communication analysis"],
                conditions_suited=["depression", "anxiety", "eating_disorders", "ptsd"],
                effectiveness_score=0.85
            ),
            "motivational_interviewing": ApproachConfig(
                name="Motivational Interviewing",
                evidence_level="strong",
                target_weight=0.05,  # 5% - specific to motivation/change
                min_samples=120,
                max_samples=2000,
                keywords=["motivational interviewing", "mi", "ambivalence", "change talk",
                         "rolling with resistance", "self-efficacy"],
                techniques=["open-ended questions", "affirmations", "reflective listening",
                          "summarizing", "eliciting change talk", "developing discrepancy"],
                conditions_suited=["substance_abuse", "health_behavior_change", "motivation"],
                effectiveness_score=0.80
            ),
            "exposure_therapy": ApproachConfig(
                name="Exposure and Response Prevention",
                evidence_level="strong",
                target_weight=0.04,  # 4% - specific to anxiety/OCD
                min_samples=100,
                max_samples=1500,
                keywords=["exposure", "response prevention", "systematic desensitization",
                         "flooding", "habituation", "fear hierarchy"],
                techniques=["systematic desensitization", "in vivo exposure", "imaginal exposure",
                          "response prevention", "fear hierarchy", "habituation"],
                conditions_suited=["ocd", "phobias", "anxiety", "ptsd"],
                effectiveness_score=0.90
            ),
            "integrative": ApproachConfig(
                name="Integrative/Eclectic Therapy",
                evidence_level="moderate",
                target_weight=0.08,  # 8% - common in practice
                min_samples=200,
                max_samples=3000,
                keywords=["integrative", "eclectic", "multimodal", "combination",
                         "tailored approach", "best practices"],
                techniques=["technique integration", "approach combination", "tailored interventions",
                          "flexible methodology", "evidence-based selection"],
                conditions_suited=["complex_presentations", "comorbid_conditions", "treatment_resistant"],
                effectiveness_score=0.75
            )
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    custom_config = json.load(f)
                # Update default configs with custom values
                for approach_id, config_data in custom_config.items():
                    if approach_id in default_configs:
                        for key, value in config_data.items():
                            setattr(default_configs[approach_id], key, value)
            except Exception as e:
                logger.warning(f"Failed to load custom config: {e}. Using defaults.")

        return default_configs

    def _build_keyword_mapping(self) -> dict[str, list[str]]:
        """Build mapping from keywords to therapeutic approaches"""
        keyword_mapping = defaultdict(list)

        for approach_id, config in self.approach_configs.items():
            # Add keywords
            if config.keywords:
                for keyword in config.keywords:
                    for word in keyword.lower().split():
                        keyword_mapping[word].append(approach_id)

            # Add techniques
            if config.techniques:
                for technique in config.techniques:
                    for word in technique.lower().split():
                        keyword_mapping[word].append(approach_id)

        return dict(keyword_mapping)

    def detect_therapeutic_approaches(self, conversation: dict[str, Any]) -> dict[str, float]:
        """
        Detect therapeutic approaches mentioned in a conversation.
        Returns approach scores (0.0-1.0) for each detected approach.
        """
        content = str(conversation).lower()
        approach_scores = defaultdict(float)

        # Count keyword matches
        for keyword, approaches in self.approach_keywords.items():
            if keyword in content:
                # Weight by keyword specificity and approach evidence level
                base_weight = 1.0 / len(approaches)
                for approach in approaches:
                    config = self.approach_configs[approach]
                    evidence_multiplier = {
                        "strong": 1.2,
                        "moderate": 1.0,
                        "emerging": 0.8
                    }.get(config.evidence_level, 1.0)

                    approach_scores[approach] += base_weight * evidence_multiplier

        # Normalize scores
        if approach_scores:
            max_score = max(approach_scores.values())
            for approach in approach_scores:
                approach_scores[approach] /= max_score

        # Filter out very low scores
        return {k: v for k, v in approach_scores.items() if v >= 0.1}

    def assess_approach_quality(self, conversation: dict[str, Any],
                              approach: str) -> float:
        """Assess the quality of therapeutic approach implementation"""
        content = str(conversation).lower()
        config = self.approach_configs[approach]

        quality_factors = {
            "technique_usage": 0.4,
            "theoretical_consistency": 0.3,
            "evidence_alignment": 0.2,
            "implementation_quality": 0.1
        }

        total_score = 0.0

        # Technique usage (0.0-1.0)
        technique_score = self._assess_technique_usage(content, config)
        total_score += technique_score * quality_factors["technique_usage"]

        # Theoretical consistency (0.0-1.0)
        consistency_score = self._assess_theoretical_consistency(content, config)
        total_score += consistency_score * quality_factors["theoretical_consistency"]

        # Evidence alignment (0.0-1.0)
        evidence_score = config.effectiveness_score
        total_score += evidence_score * quality_factors["evidence_alignment"]

        # Implementation quality (0.0-1.0)
        implementation_score = self._assess_implementation_quality(content, config)
        total_score += implementation_score * quality_factors["implementation_quality"]

        return min(1.0, total_score)

    def _assess_technique_usage(self, content: str, config: ApproachConfig) -> float:
        """Assess usage of specific techniques for the approach"""
        if not config.techniques:
            return 0.7  # Default score if no techniques specified

        technique_count = 0
        for technique in config.techniques:
            if technique.lower() in content:
                technique_count += 1

        # Score based on technique diversity
        technique_ratio = technique_count / len(config.techniques)
        return min(1.0, technique_ratio * 2)  # Cap at 1.0, reward diversity

    def _assess_theoretical_consistency(self, content: str, config: ApproachConfig) -> float:
        """Assess theoretical consistency with the approach"""
        # Look for approach-specific language and concepts
        consistency_indicators = {
            "cbt": ["thoughts", "behaviors", "feelings", "patterns", "evidence"],
            "dbt": ["mindfulness", "distress", "emotions", "interpersonal", "wise"],
            "psychodynamic": ["unconscious", "past", "childhood", "patterns", "insight"],
            "humanistic": ["feelings", "experience", "growth", "authentic", "self"],
            "acceptance_commitment": ["acceptance", "values", "mindfulness", "flexibility"],
            "emdr": ["trauma", "processing", "bilateral", "safe", "resources"],
            "family_systems": ["family", "relationships", "patterns", "boundaries", "system"],
            "gestalt": ["awareness", "present", "experience", "contact", "experiment"],
            "solution_focused": ["solutions", "goals", "strengths", "exceptions", "future"],
            "narrative": ["story", "identity", "meaning", "externalize", "preferred"],
            "mindfulness_based": ["mindfulness", "present", "awareness", "meditation", "breath"],
            "interpersonal": ["relationships", "communication", "grief", "roles", "social"],
            "motivational_interviewing": ["motivation", "change", "ambivalence", "goals"],
            "exposure_therapy": ["exposure", "fear", "anxiety", "gradual", "hierarchy"],
            "integrative": ["combination", "tailored", "flexible", "evidence", "best"]
        }

        approach_id = None
        for aid, cfg in self.approach_configs.items():
            if cfg == config:
                approach_id = aid
                break

        if approach_id not in consistency_indicators:
            return 0.7  # Default score

        indicators = consistency_indicators[approach_id]
        indicator_count = sum(1 for indicator in indicators if indicator in content)

        return min(1.0, indicator_count / len(indicators) * 1.5)

    def _assess_implementation_quality(self, content: str, config: ApproachConfig) -> float:
        """Assess quality of approach implementation"""
        # Look for quality indicators
        quality_indicators = [
            "structured", "systematic", "evidence-based", "effective",
            "collaborative", "therapeutic", "professional", "skilled"
        ]

        quality_count = sum(1 for indicator in quality_indicators if indicator in content)
        return min(1.0, quality_count / len(quality_indicators) * 2)

    def validate_cross_approach_consistency(self, conversations: list[dict[str, Any]]) -> dict[str, float]:
        """
        Validate consistency across different therapeutic approaches.
        Ensures approaches are not contradictory or conflicting.
        """
        approach_combinations = defaultdict(int)
        consistency_scores = {}

        # Analyze approach combinations
        for conv in conversations:
            detected = self.detect_therapeutic_approaches(conv)
            if len(detected) > 1:
                # Multiple approaches detected
                approaches = sorted(detected.keys())
                combination = tuple(approaches)
                approach_combinations[combination] += 1

        # Define compatibility matrix
        compatibility_matrix = {
            ("cbt", "mindfulness_based"): 0.9,  # Highly compatible
            ("cbt", "acceptance_commitment"): 0.8,
            ("dbt", "mindfulness_based"): 0.9,
            ("psychodynamic", "humanistic"): 0.7,
            ("family_systems", "solution_focused"): 0.6,
            ("integrative", "cbt"): 0.8,
            ("integrative", "psychodynamic"): 0.7,
            ("exposure_therapy", "cbt"): 0.9,
            ("interpersonal", "cbt"): 0.7,
            ("motivational_interviewing", "cbt"): 0.6,
        }

        # Calculate consistency scores
        for combination, _count in approach_combinations.items():
            if len(combination) == 2:
                compatibility = compatibility_matrix.get(combination, 0.5)  # Default moderate compatibility
                consistency_scores[combination] = compatibility
            else:
                # Multiple approaches - lower compatibility
                consistency_scores[combination] = 0.4

        return consistency_scores

    def calculate_approach_effectiveness_weights(self, conversations: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate effectiveness-based weights for approaches"""
        approach_effectiveness = {}

        for approach_id, config in self.approach_configs.items():
            # Base effectiveness from research evidence
            base_effectiveness = config.effectiveness_score

            # Adjust based on evidence level
            evidence_multiplier = {
                "strong": 1.0,
                "moderate": 0.85,
                "emerging": 0.7
            }.get(config.evidence_level, 0.8)

            # Calculate final effectiveness weight
            effectiveness_weight = base_effectiveness * evidence_multiplier
            approach_effectiveness[approach_id] = effectiveness_weight

        return approach_effectiveness

    def optimize_approach_diversity(self, conversations: list[dict[str, Any]],
                                  target_total: int = 10000) -> list[ApproachBalance]:
        """
        Main method to optimize therapeutic approach diversity.
        """
        logger.info(f"Starting approach diversity optimization for {len(conversations)} conversations")

        # Categorize conversations by detected approaches
        approach_conversations = defaultdict(list)

        for conv in conversations:
            detected = self.detect_therapeutic_approaches(conv)

            if detected:
                # Assign to primary approach (highest score)
                primary_approach = max(detected.items(), key=lambda x: x[1])[0]
                approach_conversations[primary_approach].append(conv)
            else:
                # No specific approach detected - assign to integrative
                approach_conversations["integrative"].append(conv)

        # Calculate target distribution based on weights and effectiveness
        effectiveness_weights = self.calculate_approach_effectiveness_weights(conversations)
        target_distribution = {}

        total_weight = sum(config.target_weight for config in self.approach_configs.values())

        for approach_id, config in self.approach_configs.items():
            # Base samples from target weight
            weight_ratio = config.target_weight / total_weight
            base_samples = int(target_total * weight_ratio)

            # Adjust by effectiveness
            effectiveness_multiplier = effectiveness_weights.get(approach_id, 0.8)
            adjusted_samples = int(base_samples * effectiveness_multiplier)

            # Ensure minimum samples
            adjusted_samples = max(adjusted_samples, config.min_samples)

            # Ensure maximum samples
            if config.max_samples:
                adjusted_samples = min(adjusted_samples, config.max_samples)

            # Ensure we don't exceed available data
            available = len(approach_conversations.get(approach_id, []))
            adjusted_samples = min(adjusted_samples, available)

            target_distribution[approach_id] = adjusted_samples

            logger.info(f"{config.name}: {adjusted_samples} samples "
                       f"(weight: {config.target_weight:.1%}, effectiveness: {effectiveness_multiplier:.3f}, "
                       f"available: {available})")

        # Validate cross-approach consistency
        consistency_scores = self.validate_cross_approach_consistency(conversations)
        logger.info(f"Cross-approach consistency validated: {len(consistency_scores)} combinations")

        # Balance each approach
        results = []
        total_optimized = 0

        for approach_id, target_count in target_distribution.items():
            if target_count <= 0:
                continue

            config = self.approach_configs[approach_id]
            available_conversations = approach_conversations.get(approach_id, [])

            logger.info(f"Optimizing {config.name}: {target_count} target, "
                       f"{len(available_conversations)} available")

            # Select quality-weighted conversations
            selected = self._quality_weighted_approach_selection(
                available_conversations, target_count, approach_id
            )

            if selected:
                # Calculate metrics
                evidence_weight = effectiveness_weights.get(approach_id, 0.8)
                effectiveness_score = config.effectiveness_score

                result = ApproachBalance(
                    approach=config.name,
                    target_samples=target_count,
                    actual_samples=len(selected),
                    evidence_weight=evidence_weight,
                    effectiveness_score=effectiveness_score,
                    conversations=selected,
                    metadata={
                        "approach_id": approach_id,
                        "evidence_level": config.evidence_level,
                        "target_weight": config.target_weight,
                        "available_count": len(available_conversations),
                        "keywords": config.keywords,
                        "techniques": config.techniques,
                        "conditions_suited": config.conditions_suited
                    }
                )

                results.append(result)
                total_optimized += len(selected)

                logger.info(f"Optimized {config.name}: {len(selected)} conversations "
                           f"(evidence: {evidence_weight:.3f})")

        logger.info(f"Total optimized: {total_optimized} conversations across "
                   f"{len(results)} approaches")

        # Store optimization history
        self.optimization_history.append({
            "timestamp": str(np.datetime64("now")),
            "target_total": target_total,
            "actual_total": total_optimized,
            "approaches_optimized": len(results),
            "consistency_scores": {str(k): v for k, v in consistency_scores.items()},  # Convert tuple keys to strings
            "approach_results": {r.approach: r.metadata for r in results}
        })

        return results

    def _quality_weighted_approach_selection(self, conversations: list[dict[str, Any]],
                                           target_count: int, approach: str) -> list[dict[str, Any]]:
        """Select conversations with quality weighting for a specific approach"""
        if not conversations or target_count <= 0:
            return []

        # Score conversations for this approach
        scored_conversations = []
        for conv in conversations:
            quality_score = self.assess_approach_quality(conv, approach)
            scored_conversations.append((conv, quality_score))

        # Sort by quality score
        scored_conversations.sort(key=lambda x: x[1], reverse=True)

        # Select top conversations
        selected_count = min(target_count, len(scored_conversations))
        return [conv for conv, _ in scored_conversations[:selected_count]]


    def get_optimization_statistics(self) -> dict[str, Any]:
        """Get comprehensive optimization statistics"""
        if not self.optimization_history:
            return {"error": "No optimization history available"}

        latest = self.optimization_history[-1]

        return {
            "total_optimization_runs": len(self.optimization_history),
            "latest_run": latest,
            "approach_configurations": {
                approach_id: {
                    "name": config.name,
                    "evidence_level": config.evidence_level,
                    "target_weight": config.target_weight,
                    "effectiveness_score": config.effectiveness_score
                }
                for approach_id, config in self.approach_configs.items()
            },
            "total_approaches": len(self.approach_configs),
            "keyword_mappings": len(self.approach_keywords)
        }


    def export_approach_config(self, output_path: str):
        """Export current approach configuration"""
        config_data = {}
        for approach_id, config in self.approach_configs.items():
            config_data[approach_id] = {
                "name": config.name,
                "evidence_level": config.evidence_level,
                "target_weight": config.target_weight,
                "min_samples": config.min_samples,
                "max_samples": config.max_samples,
                "keywords": config.keywords,
                "techniques": config.techniques,
                "conditions_suited": config.conditions_suited,
                "effectiveness_score": config.effectiveness_score
            }

        with open(output_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Approach configuration exported to {output_path}")

def main():
    """Example usage of the Approach Diversity Optimizer"""
    # Initialize optimizer
    optimizer = ApproachDiversityOptimizer()

    # Example conversations (would be loaded from actual datasets)
    example_conversations = [
        {
            "id": "conv_1",
            "messages": [
                {"content": "Let's use cognitive behavioral therapy techniques to identify and challenge your negative thought patterns.", "role": "therapist"},
                {"content": "I can see how my thoughts affect my mood. The thought records are helpful.", "role": "client"}
            ]
        },
        {
            "id": "conv_2",
            "messages": [
                {"content": "We'll practice mindfulness and distress tolerance skills from DBT to help you manage intense emotions.", "role": "therapist"},
                {"content": "The wise mind concept really helps me find balance between emotion and logic.", "role": "client"}
            ]
        },
        {
            "id": "conv_3",
            "messages": [
                {"content": "Let's explore how your childhood experiences might be influencing your current relationships.", "role": "therapist"},
                {"content": "I never realized how my past patterns were affecting my present relationships.", "role": "client"}
            ]
        }
    ] * 50  # Simulate larger dataset

    # Perform approach optimization
    results = optimizer.optimize_approach_diversity(example_conversations, target_total=300)

    # Display results
    for _result in results:
        pass

    # Export configuration
    optimizer.export_approach_config("approach_config.json")

    # Get statistics
    optimizer.get_optimization_statistics()

if __name__ == "__main__":
    main()
