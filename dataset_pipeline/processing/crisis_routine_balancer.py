#!/usr/bin/env python3
"""
Crisis-to-Routine Conversation Ratio Optimization for Task 6.24
Optimizes the ratio between crisis and routine therapeutic conversations
with crisis severity classification and safety protocol integration.
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
class CrisisConfig:
    """Configuration for crisis levels"""
    level: str
    severity: str  # "low", "moderate", "high", "extreme"
    target_ratio: float  # Proportion of total conversations
    min_samples: int
    max_samples: int | None = None
    risk_indicators: list[str] = None
    safety_protocols: list[str] = None

@dataclass
class CrisisBalance:
    """Result of crisis-routine balancing"""
    level: str
    severity: str
    target_samples: int
    actual_samples: int
    risk_score: float
    safety_protocols_triggered: int
    conversations: list[dict[str, Any]]
    metadata: dict[str, Any]

class CrisisRoutineBalancer:
    """
    Advanced crisis-to-routine conversation ratio optimization system.
    Ensures appropriate balance between crisis and routine therapeutic scenarios.
    """

    def __init__(self, config_path: str | None = None):
        """Initialize the crisis-routine balancer"""
        self.crisis_configs = self._load_crisis_configs(config_path)
        self.safety_protocols = self._load_safety_protocols()
        self.balancing_history = []

    def _load_crisis_configs(self, config_path: str | None = None) -> dict[str, CrisisConfig]:
        """Load crisis level configurations based on clinical guidelines"""
        default_configs = {
            "routine": CrisisConfig(
                level="Routine",
                severity="low",
                target_ratio=0.75,  # 75% - majority of therapeutic conversations
                min_samples=2000,
                max_samples=30000,
                risk_indicators=[
                    "mild stress", "general anxiety", "relationship concerns",
                    "work stress", "adjustment issues", "self-improvement"
                ],
                safety_protocols=["standard_care", "regular_monitoring"]
            ),
            "elevated_concern": CrisisConfig(
                level="Elevated Concern",
                severity="moderate",
                target_ratio=0.15,  # 15% - moderate risk situations
                min_samples=400,
                max_samples=6000,
                risk_indicators=[
                    "moderate depression", "significant anxiety", "substance use",
                    "relationship breakdown", "job loss", "financial stress",
                    "family conflict", "health concerns"
                ],
                safety_protocols=["increased_monitoring", "safety_planning", "resource_referral"]
            ),
            "crisis": CrisisConfig(
                level="Crisis",
                severity="high",
                target_ratio=0.08,  # 8% - high-risk crisis situations
                min_samples=200,
                max_samples=3000,
                risk_indicators=[
                    "suicidal ideation", "self-harm", "severe depression",
                    "panic attacks", "psychotic symptoms", "domestic violence",
                    "substance abuse", "eating disorder", "trauma response"
                ],
                safety_protocols=[
                    "immediate_assessment", "safety_planning", "crisis_intervention",
                    "emergency_contacts", "hospitalization_consideration"
                ]
            ),
            "emergency": CrisisConfig(
                level="Emergency",
                severity="extreme",
                target_ratio=0.02,  # 2% - immediate danger situations
                min_samples=50,
                max_samples=800,
                risk_indicators=[
                    "active suicidal plan", "suicide attempt", "homicidal ideation",
                    "severe psychosis", "acute intoxication", "imminent danger",
                    "medical emergency", "severe self-harm", "complete breakdown"
                ],
                safety_protocols=[
                    "immediate_intervention", "emergency_services", "hospitalization",
                    "crisis_team_activation", "family_notification", "continuous_monitoring"
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

    def _load_safety_protocols(self) -> dict[str, dict[str, Any]]:
        """Load safety protocols for different crisis levels"""
        return {
            "standard_care": {
                "description": "Standard therapeutic care",
                "actions": ["regular_sessions", "homework_assignments", "progress_monitoring"],
                "urgency": "low",
                "follow_up": "weekly"
            },
            "increased_monitoring": {
                "description": "Enhanced monitoring and support",
                "actions": ["frequent_check_ins", "safety_assessment", "support_system_activation"],
                "urgency": "moderate",
                "follow_up": "bi_weekly"
            },
            "safety_planning": {
                "description": "Comprehensive safety planning",
                "actions": ["safety_plan_creation", "coping_strategies", "emergency_contacts"],
                "urgency": "moderate",
                "follow_up": "weekly"
            },
            "crisis_intervention": {
                "description": "Active crisis intervention",
                "actions": ["immediate_assessment", "risk_mitigation", "intensive_support"],
                "urgency": "high",
                "follow_up": "daily"
            },
            "emergency_services": {
                "description": "Emergency services activation",
                "actions": ["911_call", "emergency_room", "crisis_team", "family_notification"],
                "urgency": "extreme",
                "follow_up": "continuous"
            },
            "hospitalization": {
                "description": "Psychiatric hospitalization",
                "actions": ["involuntary_hold", "medical_evaluation", "medication_review"],
                "urgency": "extreme",
                "follow_up": "continuous"
            }
        }

    def assess_crisis_level(self, conversation: dict[str, Any]) -> tuple[str, float]:
        """
        Assess crisis level and risk score for a conversation.
        Returns (crisis_level, risk_score).
        """
        content = str(conversation).lower()

        # Calculate risk scores for each level
        level_scores = {}

        for level_id, config in self.crisis_configs.items():
            score = 0.0

            # Count risk indicators
            for indicator in config.risk_indicators:
                if indicator.lower() in content:
                    # Weight by severity
                    severity_weight = {
                        "low": 0.25,
                        "moderate": 0.5,
                        "high": 0.75,
                        "extreme": 1.0
                    }.get(config.severity, 0.5)

                    score += severity_weight

            # Normalize by number of indicators
            if config.risk_indicators:
                score /= len(config.risk_indicators)

            level_scores[level_id] = score

        # Determine primary crisis level
        if level_scores:
            primary_level = max(level_scores.items(), key=lambda x: x[1])[0]
            risk_score = level_scores[primary_level]
        else:
            primary_level = "routine"
            risk_score = 0.1

        return primary_level, risk_score

    def assess_safety_protocols_needed(self, conversation: dict[str, Any],
                                     crisis_level: str) -> list[str]:
        """Assess which safety protocols should be triggered"""
        config = self.crisis_configs.get(crisis_level)
        if not config or not config.safety_protocols:
            return ["standard_care"]

        content = str(conversation).lower()
        triggered_protocols = []

        # Check for specific protocol triggers
        protocol_triggers = {
            "safety_planning": ["plan", "safety", "coping", "emergency"],
            "crisis_intervention": ["crisis", "immediate", "urgent", "help"],
            "emergency_services": ["emergency", "911", "hospital", "danger"],
            "hospitalization": ["admit", "inpatient", "psychiatric", "hold"]
        }

        for protocol in config.safety_protocols:
            if protocol in protocol_triggers:
                triggers = protocol_triggers[protocol]
                if any(trigger in content for trigger in triggers):
                    triggered_protocols.append(protocol)
            else:
                # Default protocols are always triggered
                triggered_protocols.append(protocol)

        return triggered_protocols if triggered_protocols else ["standard_care"]

    def calculate_optimal_ratios(self, total_conversations: int) -> dict[str, int]:
        """Calculate optimal conversation distribution based on clinical guidelines"""
        optimal_distribution = {}

        # Calculate base distribution from target ratios
        total_ratio = sum(config.target_ratio for config in self.crisis_configs.values())

        for level_id, config in self.crisis_configs.items():
            # Base samples from target ratio
            ratio_proportion = config.target_ratio / total_ratio
            base_samples = int(total_conversations * ratio_proportion)

            # Ensure minimum samples
            base_samples = max(base_samples, config.min_samples)

            # Ensure maximum samples
            if config.max_samples:
                base_samples = min(base_samples, config.max_samples)

            optimal_distribution[level_id] = base_samples

        return optimal_distribution

    def balance_crisis_routine_ratio(self, conversations: list[dict[str, Any]],
                                   target_total: int = 10000) -> list[CrisisBalance]:
        """
        Main method to balance crisis-to-routine conversation ratios.
        """
        logger.info(f"Starting crisis-routine balancing for {len(conversations)} conversations")

        # Classify conversations by crisis level
        crisis_conversations = defaultdict(list)
        conversation_assessments = {}

        for conv in conversations:
            crisis_level, risk_score = self.assess_crisis_level(conv)
            safety_protocols = self.assess_safety_protocols_needed(conv, crisis_level)

            crisis_conversations[crisis_level].append(conv)
            conversation_assessments[conv.get("id", str(hash(str(conv))))] = {
                "crisis_level": crisis_level,
                "risk_score": risk_score,
                "safety_protocols": safety_protocols
            }

        # Calculate optimal distribution
        optimal_distribution = self.calculate_optimal_ratios(target_total)

        # Log distribution analysis
        logger.info("Crisis-Routine Distribution Analysis:")
        for level_id, target_count in optimal_distribution.items():
            config = self.crisis_configs[level_id]
            available = len(crisis_conversations.get(level_id, []))
            logger.info(f"{config.level} ({config.severity}): {target_count} target "
                       f"({config.target_ratio:.1%}), {available} available")

        # Balance each crisis level
        results = []
        total_balanced = 0

        for level_id, target_count in optimal_distribution.items():
            if target_count <= 0:
                continue

            config = self.crisis_configs[level_id]
            available_conversations = crisis_conversations.get(level_id, [])

            logger.info(f"Balancing {config.level}: {target_count} target, "
                       f"{len(available_conversations)} available")

            # Select conversations with risk-based prioritization
            selected = self._select_risk_balanced_conversations(
                available_conversations, target_count, level_id, conversation_assessments
            )

            if selected:
                # Calculate metrics
                selected_assessments = [
                    conversation_assessments.get(conv.get("id", str(hash(str(conv)))), {})
                    for conv in selected
                ]

                avg_risk_score = np.mean([
                    assessment.get("risk_score", 0.0)
                    for assessment in selected_assessments
                ])

                total_protocols_triggered = sum([
                    len(assessment.get("safety_protocols", []))
                    for assessment in selected_assessments
                ])

                result = CrisisBalance(
                    level=config.level,
                    severity=config.severity,
                    target_samples=target_count,
                    actual_samples=len(selected),
                    risk_score=avg_risk_score,
                    safety_protocols_triggered=total_protocols_triggered,
                    conversations=selected,
                    metadata={
                        "level_id": level_id,
                        "target_ratio": config.target_ratio,
                        "available_count": len(available_conversations),
                        "risk_indicators": config.risk_indicators,
                        "safety_protocols": config.safety_protocols,
                        "protocol_details": [
                            self.safety_protocols.get(protocol, {})
                            for protocol in config.safety_protocols
                        ]
                    }
                )

                results.append(result)
                total_balanced += len(selected)

                logger.info(f"Balanced {config.level}: {len(selected)} conversations "
                           f"(avg risk: {avg_risk_score:.3f}, protocols: {total_protocols_triggered})")

        # Validate crisis-routine ratio
        crisis_count = sum(r.actual_samples for r in results
                          if r.severity in ["high", "extreme"])
        routine_count = sum(r.actual_samples for r in results
                           if r.severity in ["low", "moderate"])

        if total_balanced > 0:
            crisis_ratio = crisis_count / total_balanced
            routine_ratio = routine_count / total_balanced
            logger.info(f"Final ratio - Crisis: {crisis_ratio:.1%}, Routine: {routine_ratio:.1%}")

        logger.info(f"Total balanced: {total_balanced} conversations across "
                   f"{len(results)} crisis levels")

        # Store balancing history
        self.balancing_history.append({
            "timestamp": str(np.datetime64("now")),
            "target_total": target_total,
            "actual_total": total_balanced,
            "crisis_levels": len(results),
            "crisis_ratio": crisis_count / total_balanced if total_balanced > 0 else 0,
            "routine_ratio": routine_count / total_balanced if total_balanced > 0 else 0,
            "level_results": {r.level: r.metadata for r in results}
        })

        return results

    def _select_risk_balanced_conversations(self, conversations: list[dict[str, Any]],
                                          target_count: int, level_id: str,
                                          assessments: dict[str, dict]) -> list[dict[str, Any]]:
        """Select conversations with risk-based balancing"""
        if not conversations or target_count <= 0:
            return []

        config = self.crisis_configs[level_id]

        # Score conversations for selection
        scored_conversations = []
        for conv in conversations:
            conv_id = conv.get("id", str(hash(str(conv))))
            assessment = assessments.get(conv_id, {})

            risk_score = assessment.get("risk_score", 0.0)
            safety_protocols = assessment.get("safety_protocols", [])

            # Selection score based on risk appropriateness and protocol coverage
            risk_appropriateness = self._assess_risk_appropriateness(risk_score, config.severity)
            protocol_coverage = len(safety_protocols) / max(1, len(config.safety_protocols))

            # Diversity score for varied scenarios
            diversity_score = self._assess_crisis_diversity(conv)

            # Combined selection score
            selection_score = (risk_appropriateness * 0.5 +
                             protocol_coverage * 0.3 +
                             diversity_score * 0.2)

            scored_conversations.append((conv, selection_score, risk_score))

        # Sort by selection score
        scored_conversations.sort(key=lambda x: x[1], reverse=True)

        # Select top conversations
        selected_count = min(target_count, len(scored_conversations))
        return [conv for conv, _, _ in scored_conversations[:selected_count]]


    def _assess_risk_appropriateness(self, risk_score: float, severity: str) -> float:
        """Assess how well risk score matches expected severity level"""
        severity_ranges = {
            "low": (0.0, 0.3),
            "moderate": (0.2, 0.6),
            "high": (0.5, 0.8),
            "extreme": (0.7, 1.0)
        }

        min_risk, max_risk = severity_ranges.get(severity, (0.0, 1.0))

        if min_risk <= risk_score <= max_risk:
            return 1.0  # Perfect match
        # Calculate distance from ideal range
        distance = min_risk - risk_score if risk_score < min_risk else risk_score - max_risk

        return max(0.0, 1.0 - distance)

    def _assess_crisis_diversity(self, conversation: dict[str, Any]) -> float:
        """Assess diversity of crisis scenarios and interventions"""
        content = str(conversation).lower()

        # Diversity indicators for crisis scenarios
        diversity_indicators = [
            "multiple", "complex", "various", "different", "range",
            "spectrum", "variety", "diverse", "comprehensive", "holistic"
        ]

        # Crisis intervention variety
        intervention_indicators = [
            "intervention", "strategy", "approach", "technique", "method",
            "protocol", "procedure", "plan", "response", "action"
        ]

        diversity_count = sum(1 for indicator in diversity_indicators if indicator in content)
        intervention_count = sum(1 for indicator in intervention_indicators if indicator in content)

        total_indicators = len(diversity_indicators) + len(intervention_indicators)
        diversity_score = (diversity_count + intervention_count) / total_indicators

        return min(1.0, diversity_score * 2)  # Scale up and cap at 1.0

    def get_balancing_statistics(self) -> dict[str, Any]:
        """Get comprehensive balancing statistics"""
        if not self.balancing_history:
            return {"error": "No balancing history available"}

        latest = self.balancing_history[-1]

        return {
            "total_balancing_runs": len(self.balancing_history),
            "latest_run": latest,
            "crisis_configurations": {
                level_id: {
                    "level_name": config.level,
                    "severity": config.severity,
                    "target_ratio": config.target_ratio,
                    "safety_protocols": config.safety_protocols
                }
                for level_id, config in self.crisis_configs.items()
            },
            "safety_protocols": self.safety_protocols,
            "total_crisis_levels": len(self.crisis_configs)
        }


    def export_crisis_config(self, output_path: str):
        """Export current crisis configuration"""
        config_data = {}
        for level_id, config in self.crisis_configs.items():
            config_data[level_id] = {
                "level": config.level,
                "severity": config.severity,
                "target_ratio": config.target_ratio,
                "min_samples": config.min_samples,
                "max_samples": config.max_samples,
                "risk_indicators": config.risk_indicators,
                "safety_protocols": config.safety_protocols
            }

        with open(output_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Crisis configuration exported to {output_path}")

def main():
    """Example usage of the Crisis-Routine Balancer"""
    # Initialize balancer
    balancer = CrisisRoutineBalancer()

    # Example conversations with varying crisis levels
    example_conversations = [
        {
            "id": "conv_1",
            "messages": [
                {"content": "I've been feeling a bit stressed at work lately. Can you help me with some coping strategies?", "role": "client"},
                {"content": "Of course. Let's explore some stress management techniques and work-life balance strategies.", "role": "therapist"}
            ]
        },
        {
            "id": "conv_2",
            "messages": [
                {"content": "I'm going through a difficult divorce and feeling very depressed. I'm having trouble sleeping and eating.", "role": "client"},
                {"content": "I understand this is a very challenging time. Let's create a safety plan and discuss support resources.", "role": "therapist"}
            ]
        },
        {
            "id": "conv_3",
            "messages": [
                {"content": "I've been having suicidal thoughts and I have a plan. I don't think I can keep going.", "role": "client"},
                {"content": "I'm very concerned about your safety. We need to get you immediate help. Let's call emergency services right now.", "role": "therapist"}
            ]
        }
    ] * 50  # Simulate larger dataset

    # Perform crisis-routine balancing
    results = balancer.balance_crisis_routine_ratio(example_conversations, target_total=300)

    # Display results
    for _result in results:
        pass

    # Export configuration
    balancer.export_crisis_config("crisis_config.json")

    # Get statistics
    balancer.get_balancing_statistics()

if __name__ == "__main__":
    main()
