#!/usr/bin/env python3
"""
Conversation Effectiveness Feedback Loops
Tracks real-world effectiveness and feeds back into dataset improvement.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Types of therapeutic outcomes."""
    SYMPTOM_IMPROVEMENT = "symptom_improvement"
    ENGAGEMENT_INCREASE = "engagement_increase"
    CRISIS_RESOLUTION = "crisis_resolution"
    SKILL_ACQUISITION = "skill_acquisition"
    THERAPEUTIC_ALLIANCE = "therapeutic_alliance"
    TREATMENT_COMPLETION = "treatment_completion"


@dataclass
class EffectivenessFeedback:
    """Real-world effectiveness feedback."""
    feedback_id: str
    conversation_id: str
    outcome_type: OutcomeType
    effectiveness_score: float
    user_rating: int | None = None
    therapist_rating: int | None = None
    outcome_details: dict[str, Any] = field(default_factory=dict)
    follow_up_data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementAction:
    """Dataset improvement action based on feedback."""
    action_id: str
    conversation_id: str
    improvement_type: str
    original_score: float
    target_score: float
    modifications: list[str] = field(default_factory=list)
    rationale: str = ""
    implemented: bool = False
    results: dict[str, Any] | None = None


class FeedbackLoops:
    """
    Conversation effectiveness feedback loops system.
    """

    def __init__(self):
        """Initialize the feedback loops system."""
        self.feedback_history: list[EffectivenessFeedback] = []
        self.improvement_actions: list[ImprovementAction] = []
        self.effectiveness_patterns: dict[str, Any] = {}

        # Configuration
        self.config = {
            "min_feedback_threshold": 5,
            "effectiveness_threshold": 0.7,
            "improvement_target": 0.1,
            "feedback_window_days": 30
        }

    def submit_effectiveness_feedback(self, feedback: EffectivenessFeedback):
        """Submit real-world effectiveness feedback."""
        logger.info(f"Received effectiveness feedback for conversation {feedback.conversation_id}")

        # Store feedback
        self.feedback_history.append(feedback)

        # Analyze for improvement opportunities
        self._analyze_feedback_for_improvements(feedback)

        # Update effectiveness patterns
        self._update_effectiveness_patterns(feedback)

    def _analyze_feedback_for_improvements(self, feedback: EffectivenessFeedback):
        """Analyze feedback to identify improvement opportunities."""
        # Check if conversation needs improvement
        if feedback.effectiveness_score < self.config["effectiveness_threshold"]:
            improvement_action = self._create_improvement_action(feedback)
            if improvement_action:
                self.improvement_actions.append(improvement_action)

    def _create_improvement_action(self, feedback: EffectivenessFeedback) -> ImprovementAction | None:
        """Create improvement action based on feedback."""
        conversation_id = feedback.conversation_id
        current_score = feedback.effectiveness_score
        target_score = min(1.0, current_score + self.config["improvement_target"])

        # Determine improvement type based on outcome
        improvement_type = self._determine_improvement_type(feedback)

        # Generate specific modifications
        modifications = self._generate_modifications(feedback, improvement_type)

        if not modifications:
            return None

        return ImprovementAction(
            action_id=f"improve_{conversation_id}_{int(datetime.now().timestamp())}",
            conversation_id=conversation_id,
            improvement_type=improvement_type,
            original_score=current_score,
            target_score=target_score,
            modifications=modifications,
            rationale=f"Low effectiveness score ({current_score:.3f}) for {feedback.outcome_type.value}"
        )

    def _determine_improvement_type(self, feedback: EffectivenessFeedback) -> str:
        """Determine the type of improvement needed."""
        outcome_type = feedback.outcome_type
        effectiveness_score = feedback.effectiveness_score

        if outcome_type == OutcomeType.SYMPTOM_IMPROVEMENT and effectiveness_score < 0.6:
            return "therapeutic_technique_enhancement"
        if outcome_type == OutcomeType.ENGAGEMENT_INCREASE and effectiveness_score < 0.7:
            return "engagement_optimization"
        if outcome_type == OutcomeType.CRISIS_RESOLUTION and effectiveness_score < 0.8:
            return "crisis_response_improvement"
        if outcome_type == OutcomeType.THERAPEUTIC_ALLIANCE and effectiveness_score < 0.7:
            return "alliance_building_enhancement"
        return "general_quality_improvement"

    def _generate_modifications(self, feedback: EffectivenessFeedback, improvement_type: str) -> list[str]:
        """Generate specific modifications for improvement."""
        modifications = []

        if improvement_type == "therapeutic_technique_enhancement":
            modifications.extend([
                "Add more specific therapeutic techniques",
                "Include evidence-based interventions",
                "Enhance clinical accuracy",
                "Improve intervention sequencing"
            ])

        elif improvement_type == "engagement_optimization":
            modifications.extend([
                "Increase empathetic responses",
                "Add more validation statements",
                "Improve motivational interviewing techniques",
                "Enhance collaborative language"
            ])

        elif improvement_type == "crisis_response_improvement":
            modifications.extend([
                "Strengthen safety assessment",
                "Add crisis intervention protocols",
                "Include emergency resource information",
                "Improve risk management strategies"
            ])

        elif improvement_type == "alliance_building_enhancement":
            modifications.extend([
                "Increase warmth and genuineness",
                "Add more collaborative planning",
                "Improve cultural sensitivity",
                "Enhance trust-building elements"
            ])

        else:  # general_quality_improvement
            modifications.extend([
                "Improve overall conversation flow",
                "Enhance therapeutic coherence",
                "Add more personalized responses",
                "Strengthen professional boundaries"
            ])

        # Filter based on specific feedback details
        if feedback.outcome_details:
            specific_issues = feedback.outcome_details.get("issues", [])
            for issue in specific_issues:
                if "unclear" in issue.lower():
                    modifications.append("Improve clarity and communication")
                elif "rushed" in issue.lower():
                    modifications.append("Allow more time for processing")
                elif "impersonal" in issue.lower():
                    modifications.append("Add more personalized elements")

        return modifications[:4]  # Limit to top 4 modifications

    def _update_effectiveness_patterns(self, feedback: EffectivenessFeedback):
        """Update effectiveness patterns based on feedback."""
        outcome_type = feedback.outcome_type.value

        if outcome_type not in self.effectiveness_patterns:
            self.effectiveness_patterns[outcome_type] = {
                "total_feedback": 0,
                "average_effectiveness": 0.0,
                "effectiveness_trend": [],
                "common_issues": {},
                "success_factors": {}
            }

        pattern = self.effectiveness_patterns[outcome_type]

        # Update statistics
        pattern["total_feedback"] += 1
        pattern["effectiveness_trend"].append(feedback.effectiveness_score)

        # Keep only recent trend data
        if len(pattern["effectiveness_trend"]) > 50:
            pattern["effectiveness_trend"] = pattern["effectiveness_trend"][-50:]

        # Update average
        pattern["average_effectiveness"] = sum(pattern["effectiveness_trend"]) / len(pattern["effectiveness_trend"])

        # Track issues and success factors
        if feedback.outcome_details:
            issues = feedback.outcome_details.get("issues", [])
            for issue in issues:
                pattern["common_issues"][issue] = pattern["common_issues"].get(issue, 0) + 1

            if feedback.effectiveness_score > 0.8:
                success_factors = feedback.outcome_details.get("success_factors", [])
                for factor in success_factors:
                    pattern["success_factors"][factor] = pattern["success_factors"].get(factor, 0) + 1

    def get_improvement_recommendations(self) -> list[dict[str, Any]]:
        """Get data-driven improvement recommendations."""
        recommendations = []

        # Analyze effectiveness patterns
        for outcome_type, pattern in self.effectiveness_patterns.items():
            if pattern["total_feedback"] >= self.config["min_feedback_threshold"]:
                avg_effectiveness = pattern["average_effectiveness"]

                if avg_effectiveness < self.config["effectiveness_threshold"]:
                    # Get most common issues
                    common_issues = sorted(
                        pattern["common_issues"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]

                    # Get top success factors
                    success_factors = sorted(
                        pattern["success_factors"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]

                    recommendations.append({
                        "outcome_type": outcome_type,
                        "current_effectiveness": avg_effectiveness,
                        "improvement_needed": self.config["effectiveness_threshold"] - avg_effectiveness,
                        "common_issues": [issue for issue, count in common_issues],
                        "success_factors": [factor for factor, count in success_factors],
                        "priority": "high" if avg_effectiveness < 0.6 else "medium"
                    })

        # Sort by priority and improvement needed
        recommendations.sort(key=lambda x: (x["priority"] == "high", x["improvement_needed"]), reverse=True)

        return recommendations

    def implement_improvement_action(self, action_id: str) -> bool:
        """Implement a specific improvement action."""
        action = next((a for a in self.improvement_actions if a.action_id == action_id), None)

        if not action:
            logger.warning(f"Improvement action {action_id} not found")
            return False

        if action.implemented:
            logger.info(f"Improvement action {action_id} already implemented")
            return True

        try:
            # This would integrate with the actual dataset modification system
            logger.info(f"Implementing improvement action: {action.improvement_type}")
            logger.info(f"Modifications: {', '.join(action.modifications)}")

            # Simulate implementation
            action.implemented = True
            action.results = {
                "implementation_date": datetime.now().isoformat(),
                "modifications_applied": len(action.modifications),
                "expected_improvement": action.target_score - action.original_score
            }

            logger.info(f"Successfully implemented improvement action {action_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to implement improvement action {action_id}: {e}")
            return False

    def get_feedback_summary(self) -> dict[str, Any]:
        """Get comprehensive feedback summary."""
        if not self.feedback_history:
            return {"message": "No feedback data available"}

        total_feedback = len(self.feedback_history)

        # Calculate overall effectiveness
        overall_effectiveness = sum(f.effectiveness_score for f in self.feedback_history) / total_feedback

        # Outcome type distribution
        outcome_distribution = {}
        for feedback in self.feedback_history:
            outcome_type = feedback.outcome_type.value
            outcome_distribution[outcome_type] = outcome_distribution.get(outcome_type, 0) + 1

        # Recent trends (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_feedback = [f for f in self.feedback_history if f.timestamp >= recent_cutoff]

        recent_effectiveness = 0.0
        if recent_feedback:
            recent_effectiveness = sum(f.effectiveness_score for f in recent_feedback) / len(recent_feedback)

        # Improvement actions summary
        total_actions = len(self.improvement_actions)
        implemented_actions = len([a for a in self.improvement_actions if a.implemented])

        return {
            "total_feedback_received": total_feedback,
            "overall_effectiveness": round(overall_effectiveness, 3),
            "recent_effectiveness_30d": round(recent_effectiveness, 3),
            "outcome_distribution": outcome_distribution,
            "effectiveness_patterns": {
                outcome: {
                    "average": round(pattern["average_effectiveness"], 3),
                    "total_samples": pattern["total_feedback"]
                }
                for outcome, pattern in self.effectiveness_patterns.items()
            },
            "improvement_actions": {
                "total_created": total_actions,
                "implemented": implemented_actions,
                "implementation_rate": round(implemented_actions / max(total_actions, 1), 3)
            },
            "recommendations_available": len(self.get_improvement_recommendations())
        }

    def export_feedback_data(self, filepath: str):
        """Export feedback data for analysis."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "feedback_history": [
                {
                    "feedback_id": f.feedback_id,
                    "conversation_id": f.conversation_id,
                    "outcome_type": f.outcome_type.value,
                    "effectiveness_score": f.effectiveness_score,
                    "user_rating": f.user_rating,
                    "therapist_rating": f.therapist_rating,
                    "outcome_details": f.outcome_details,
                    "timestamp": f.timestamp.isoformat()
                }
                for f in self.feedback_history
            ],
            "improvement_actions": [
                {
                    "action_id": a.action_id,
                    "conversation_id": a.conversation_id,
                    "improvement_type": a.improvement_type,
                    "original_score": a.original_score,
                    "target_score": a.target_score,
                    "modifications": a.modifications,
                    "implemented": a.implemented,
                    "results": a.results
                }
                for a in self.improvement_actions
            ],
            "effectiveness_patterns": self.effectiveness_patterns,
            "summary": self.get_feedback_summary()
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Feedback data exported to {filepath}")


def main():
    """Example usage of the FeedbackLoops system."""
    feedback_system = FeedbackLoops()

    # Simulate effectiveness feedback
    sample_feedback = [
        EffectivenessFeedback(
            feedback_id="fb_001",
            conversation_id="conv_001",
            outcome_type=OutcomeType.SYMPTOM_IMPROVEMENT,
            effectiveness_score=0.65,
            user_rating=3,
            therapist_rating=4,
            outcome_details={
                "issues": ["unclear instructions", "rushed pace"],
                "success_factors": ["empathetic responses"]
            }
        ),
        EffectivenessFeedback(
            feedback_id="fb_002",
            conversation_id="conv_002",
            outcome_type=OutcomeType.THERAPEUTIC_ALLIANCE,
            effectiveness_score=0.85,
            user_rating=5,
            therapist_rating=5,
            outcome_details={
                "success_factors": ["good rapport", "collaborative approach", "cultural sensitivity"]
            }
        ),
        EffectivenessFeedback(
            feedback_id="fb_003",
            conversation_id="conv_003",
            outcome_type=OutcomeType.CRISIS_RESOLUTION,
            effectiveness_score=0.55,
            user_rating=2,
            therapist_rating=3,
            outcome_details={
                "issues": ["inadequate safety assessment", "missing crisis resources"]
            }
        )
    ]

    # Submit feedback
    for feedback in sample_feedback:
        feedback_system.submit_effectiveness_feedback(feedback)

    # Get improvement recommendations
    recommendations = feedback_system.get_improvement_recommendations()
    for rec in recommendations:
        if rec["success_factors"]:
            pass

    # Get feedback summary
    feedback_system.get_feedback_summary()

    # Export feedback data
    feedback_system.export_feedback_data("feedback_export.json")


if __name__ == "__main__":
    main()
