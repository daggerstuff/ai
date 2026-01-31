#!/usr/bin/env python3
"""
Therapeutic Effectiveness Prediction System
Predicts therapeutic effectiveness using longitudinal data and evidence-based metrics.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Types of therapeutic outcomes."""
    SYMPTOM_REDUCTION = "symptom_reduction"
    FUNCTIONAL_IMPROVEMENT = "functional_improvement"
    QUALITY_OF_LIFE = "quality_of_life"
    TREATMENT_ENGAGEMENT = "treatment_engagement"
    RELAPSE_PREVENTION = "relapse_prevention"
    CRISIS_REDUCTION = "crisis_reduction"


class EffectivenessLevel(Enum):
    """Levels of therapeutic effectiveness."""
    HIGHLY_EFFECTIVE = "highly_effective"
    MODERATELY_EFFECTIVE = "moderately_effective"
    MINIMALLY_EFFECTIVE = "minimally_effective"
    INEFFECTIVE = "ineffective"
    POTENTIALLY_HARMFUL = "potentially_harmful"


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


@dataclass
class OutcomeMetric:
    """Individual outcome metric."""
    metric_type: OutcomeType
    baseline_score: float
    predicted_score: float
    improvement_percentage: float
    confidence: PredictionConfidence
    evidence_strength: str
    timeframe_weeks: int


@dataclass
class EffectivenessPrediction:
    """Complete effectiveness prediction result."""
    conversation_id: str
    overall_effectiveness: EffectivenessLevel
    success_probability: float
    outcome_metrics: list[OutcomeMetric] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    protective_factors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    evidence_base: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class TherapeuticEffectivenessPredictor:
    """
    Therapeutic effectiveness prediction system.
    """

    def __init__(self):
        """Initialize the effectiveness predictor."""
        self.prediction_history: list[EffectivenessPrediction] = []
        self.evidence_base = self._load_evidence_base()
        self.outcome_models = self._initialize_outcome_models()

    def _load_evidence_base(self) -> dict[str, Any]:
        """Load evidence-based effectiveness data."""
        return {
            "intervention_effectiveness": {
                "CBT": {
                    "depression": {"effect_size": 0.68, "success_rate": 0.75, "evidence": "strong"},
                    "anxiety": {"effect_size": 0.71, "success_rate": 0.78, "evidence": "strong"},
                    "ptsd": {"effect_size": 0.65, "success_rate": 0.72, "evidence": "strong"}
                },
                "DBT": {
                    "borderline_pd": {"effect_size": 0.58, "success_rate": 0.68, "evidence": "strong"},
                    "self_harm": {"effect_size": 0.62, "success_rate": 0.70, "evidence": "strong"}
                },
                "mindfulness": {
                    "anxiety": {"effect_size": 0.45, "success_rate": 0.60, "evidence": "moderate"},
                    "depression": {"effect_size": 0.42, "success_rate": 0.58, "evidence": "moderate"}
                }
            },
            "outcome_predictors": {
                "positive": [
                    "therapeutic alliance", "motivation", "social support", "insight",
                    "homework compliance", "session attendance", "coping skills"
                ],
                "negative": [
                    "substance use", "personality disorder", "trauma history", "social isolation",
                    "financial stress", "medical comorbidity", "treatment resistance"
                ]
            },
            "timeframe_expectations": {
                "acute_symptoms": {"weeks": 4, "improvement_threshold": 0.25},
                "moderate_improvement": {"weeks": 8, "improvement_threshold": 0.50},
                "significant_improvement": {"weeks": 16, "improvement_threshold": 0.70},
                "maintenance": {"weeks": 52, "improvement_threshold": 0.80}
            }
        }

    def _initialize_outcome_models(self) -> dict[OutcomeType, dict[str, Any]]:
        """Initialize outcome prediction models."""
        return {
            OutcomeType.SYMPTOM_REDUCTION: {
                "weight": 0.30,
                "baseline_factors": ["symptom_severity", "duration", "comorbidity"],
                "intervention_factors": ["technique_match", "dosage", "adherence"]
            },
            OutcomeType.FUNCTIONAL_IMPROVEMENT: {
                "weight": 0.25,
                "baseline_factors": ["functional_impairment", "social_support", "resources"],
                "intervention_factors": ["skill_building", "behavioral_activation", "exposure"]
            },
            OutcomeType.QUALITY_OF_LIFE: {
                "weight": 0.20,
                "baseline_factors": ["life_satisfaction", "relationships", "meaning"],
                "intervention_factors": ["values_work", "goal_setting", "mindfulness"]
            },
            OutcomeType.TREATMENT_ENGAGEMENT: {
                "weight": 0.15,
                "baseline_factors": ["motivation", "previous_experience", "stigma"],
                "intervention_factors": ["therapeutic_alliance", "cultural_fit", "accessibility"]
            },
            OutcomeType.RELAPSE_PREVENTION: {
                "weight": 0.05,
                "baseline_factors": ["relapse_history", "triggers", "support_system"],
                "intervention_factors": ["relapse_plan", "maintenance_skills", "monitoring"]
            },
            OutcomeType.CRISIS_REDUCTION: {
                "weight": 0.05,
                "baseline_factors": ["crisis_history", "risk_factors", "safety_planning"],
                "intervention_factors": ["crisis_intervention", "safety_skills", "emergency_planning"]
            }
        }

    def predict_effectiveness(self, conversation: dict[str, Any]) -> EffectivenessPrediction:
        """
        Predict therapeutic effectiveness for a conversation.

        Args:
            conversation: Conversation data to analyze

        Returns:
            EffectivenessPrediction with detailed assessment
        """
        conversation_id = conversation.get("id", "unknown")
        logger.info(f"Predicting effectiveness for conversation {conversation_id}")

        content = str(conversation.get("content", ""))
        turns = conversation.get("turns", [])
        metadata = conversation.get("metadata", {})

        # Extract therapeutic elements
        therapeutic_elements = self._extract_therapeutic_elements(content, turns)

        # Identify risk and protective factors
        risk_factors = self._identify_risk_factors(content, turns, metadata)
        protective_factors = self._identify_protective_factors(content, turns, metadata)

        # Predict outcome metrics
        outcome_metrics = self._predict_outcome_metrics(
            therapeutic_elements, risk_factors, protective_factors
        )

        # Calculate overall effectiveness
        overall_effectiveness = self._calculate_overall_effectiveness(outcome_metrics)

        # Calculate success probability
        success_probability = self._calculate_success_probability(
            outcome_metrics, risk_factors, protective_factors
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            therapeutic_elements, outcome_metrics, risk_factors, protective_factors
        )

        # Compile evidence base
        evidence_base = self._compile_evidence_base(therapeutic_elements)

        prediction = EffectivenessPrediction(
            conversation_id=conversation_id,
            overall_effectiveness=overall_effectiveness,
            success_probability=success_probability,
            outcome_metrics=outcome_metrics,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            recommendations=recommendations,
            evidence_base=evidence_base
        )

        self.prediction_history.append(prediction)
        return prediction

    def _extract_therapeutic_elements(self, content: str, turns: list[dict]) -> dict[str, Any]:
        """Extract therapeutic elements from conversation."""
        content_lower = content.lower()

        elements = {
            "interventions": [],
            "techniques": [],
            "therapeutic_alliance": 0.0,
            "client_engagement": 0.0,
            "skill_building": 0.0
        }

        # Identify interventions
        intervention_keywords = {
            "CBT": ["cognitive behavioral", "cbt", "thought challenging", "cognitive restructuring"],
            "DBT": ["dialectical behavior", "dbt", "distress tolerance", "emotion regulation"],
            "mindfulness": ["mindfulness", "meditation", "breathing exercises", "present moment"],
            "psychodynamic": ["unconscious", "childhood", "transference", "insight"],
            "humanistic": ["empathy", "unconditional", "self-actualization", "person-centered"]
        }

        for intervention, keywords in intervention_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                elements["interventions"].append(intervention)

        # Assess therapeutic alliance indicators
        alliance_indicators = [
            "understand", "support", "together", "partnership", "trust",
            "safe", "comfortable", "heard", "validated"
        ]
        alliance_score = sum(1 for indicator in alliance_indicators if indicator in content_lower)
        elements["therapeutic_alliance"] = min(1.0, alliance_score / 5.0)

        # Assess client engagement
        engagement_indicators = [
            "willing", "motivated", "ready", "committed", "interested",
            "homework", "practice", "try", "work on"
        ]
        engagement_score = sum(1 for indicator in engagement_indicators if indicator in content_lower)
        elements["client_engagement"] = min(1.0, engagement_score / 5.0)

        # Assess skill building
        skill_indicators = [
            "coping", "strategy", "technique", "skill", "tool",
            "practice", "exercise", "homework", "between sessions"
        ]
        skill_score = sum(1 for indicator in skill_indicators if indicator in content_lower)
        elements["skill_building"] = min(1.0, skill_score / 5.0)

        return elements

    def _identify_risk_factors(self, content: str, turns: list[dict], metadata: dict) -> list[str]:
        """Identify risk factors that may impact effectiveness."""
        risk_factors = []
        content_lower = content.lower()

        risk_indicators = {
            "substance_use": ["drinking", "drugs", "alcohol", "substance", "addiction"],
            "trauma_history": ["trauma", "abuse", "assault", "ptsd", "flashbacks"],
            "social_isolation": ["alone", "isolated", "no friends", "lonely", "withdrawn"],
            "financial_stress": ["money", "financial", "job loss", "unemployed", "bills"],
            "medical_comorbidity": ["chronic pain", "illness", "medical", "disability"],
            "treatment_resistance": ["tried everything", "nothing works", "hopeless", "given up"],
            "personality_disorder": ["borderline", "narcissistic", "antisocial", "personality"],
            "severe_symptoms": ["severe", "extreme", "overwhelming", "unbearable", "crisis"]
        }

        for factor, keywords in risk_indicators.items():
            if any(keyword in content_lower for keyword in keywords):
                risk_factors.append(factor)

        return risk_factors

    def _identify_protective_factors(self, content: str, turns: list[dict], metadata: dict) -> list[str]:
        """Identify protective factors that may enhance effectiveness."""
        protective_factors = []
        content_lower = content.lower()

        protective_indicators = {
            "social_support": ["family", "friends", "support", "partner", "spouse"],
            "motivation": ["motivated", "ready", "committed", "determined", "willing"],
            "insight": ["understand", "realize", "aware", "insight", "recognize"],
            "coping_skills": ["coping", "manage", "handle", "deal with", "strategies"],
            "stable_housing": ["home", "stable", "housing", "place to live"],
            "employment": ["job", "work", "employed", "career", "income"],
            "education": ["education", "school", "learning", "knowledge", "skills"],
            "spiritual_resources": ["faith", "spiritual", "religion", "meaning", "purpose"],
            "previous_success": ["helped before", "worked", "successful", "improved"]
        }

        for factor, keywords in protective_indicators.items():
            if any(keyword in content_lower for keyword in keywords):
                protective_factors.append(factor)

        return protective_factors

    def _predict_outcome_metrics(
        self,
        therapeutic_elements: dict[str, Any],
        risk_factors: list[str],
        protective_factors: list[str]
    ) -> list[OutcomeMetric]:
        """Predict specific outcome metrics."""
        metrics = []

        for outcome_type, _model in self.outcome_models.items():
            # Base prediction from therapeutic elements
            base_score = self._calculate_base_outcome_score(outcome_type, therapeutic_elements)

            # Adjust for risk factors
            risk_adjustment = len(risk_factors) * -0.1

            # Adjust for protective factors
            protective_adjustment = len(protective_factors) * 0.1

            # Calculate predicted score
            predicted_score = max(0.0, min(1.0, base_score + risk_adjustment + protective_adjustment))

            # Calculate improvement percentage
            baseline_score = 0.3  # Assume moderate baseline impairment
            improvement_percentage = max(0.0, (predicted_score - baseline_score) / baseline_score * 100)

            # Determine confidence
            confidence = self._determine_prediction_confidence(
                therapeutic_elements, risk_factors, protective_factors
            )

            # Determine evidence strength
            evidence_strength = self._assess_evidence_strength(therapeutic_elements, outcome_type)

            metric = OutcomeMetric(
                metric_type=outcome_type,
                baseline_score=baseline_score,
                predicted_score=predicted_score,
                improvement_percentage=improvement_percentage,
                confidence=confidence,
                evidence_strength=evidence_strength,
                timeframe_weeks=16  # Standard 16-week timeframe
            )
            metrics.append(metric)

        return metrics

    def _calculate_base_outcome_score(self, outcome_type: OutcomeType, elements: dict[str, Any]) -> float:
        """Calculate base outcome score from therapeutic elements."""
        base_score = 0.5  # Neutral starting point

        # Adjust based on interventions
        if elements["interventions"]:
            intervention_bonus = len(elements["interventions"]) * 0.1
            base_score += min(0.3, intervention_bonus)

        # Adjust based on therapeutic alliance
        base_score += elements["therapeutic_alliance"] * 0.2

        # Adjust based on client engagement
        base_score += elements["client_engagement"] * 0.2

        # Adjust based on skill building
        base_score += elements["skill_building"] * 0.1

        return min(1.0, base_score)

    def _determine_prediction_confidence(
        self,
        therapeutic_elements: dict[str, Any],
        risk_factors: list[str],
        protective_factors: list[str]
    ) -> PredictionConfidence:
        """Determine confidence level for prediction."""
        # High confidence: strong therapeutic elements, clear factors
        if (len(therapeutic_elements["interventions"]) >= 2 and
            therapeutic_elements["therapeutic_alliance"] > 0.7 and
            len(risk_factors) + len(protective_factors) >= 3):
            return PredictionConfidence.HIGH

        # Moderate confidence: some therapeutic elements, some factors
        if (len(therapeutic_elements["interventions"]) >= 1 and
              therapeutic_elements["therapeutic_alliance"] > 0.4 and
              len(risk_factors) + len(protective_factors) >= 1):
            return PredictionConfidence.MODERATE

        # Low confidence: limited information
        return PredictionConfidence.LOW

    def _assess_evidence_strength(self, elements: dict[str, Any], outcome_type: OutcomeType) -> str:
        """Assess strength of evidence for predicted outcome."""
        interventions = elements["interventions"]

        if not interventions:
            return "limited"

        # Check evidence base for interventions
        strong_evidence_count = 0
        for intervention in interventions:
            if intervention in self.evidence_base["intervention_effectiveness"]:
                # Check if any condition has strong evidence
                conditions = self.evidence_base["intervention_effectiveness"][intervention]
                if any(data.get("evidence") == "strong" for data in conditions.values()):
                    strong_evidence_count += 1

        if strong_evidence_count >= 2:
            return "strong"
        if strong_evidence_count >= 1:
            return "moderate"
        return "limited"

    def _calculate_overall_effectiveness(self, outcome_metrics: list[OutcomeMetric]) -> EffectivenessLevel:
        """Calculate overall effectiveness level."""
        if not outcome_metrics:
            return EffectivenessLevel.INEFFECTIVE

        # Weighted average of predicted scores
        weighted_score = sum(
            metric.predicted_score * self.outcome_models[metric.metric_type]["weight"]
            for metric in outcome_metrics
        )

        if weighted_score >= 0.8:
            return EffectivenessLevel.HIGHLY_EFFECTIVE
        if weighted_score >= 0.6:
            return EffectivenessLevel.MODERATELY_EFFECTIVE
        if weighted_score >= 0.4:
            return EffectivenessLevel.MINIMALLY_EFFECTIVE
        return EffectivenessLevel.INEFFECTIVE

    def _calculate_success_probability(
        self,
        outcome_metrics: list[OutcomeMetric],
        risk_factors: list[str],
        protective_factors: list[str]
    ) -> float:
        """Calculate probability of successful treatment outcome."""
        if not outcome_metrics:
            return 0.0

        # Base probability from outcome metrics
        base_probability = sum(metric.predicted_score for metric in outcome_metrics) / len(outcome_metrics)

        # Adjust for risk factors (each reduces probability by 5%)
        risk_adjustment = len(risk_factors) * -0.05

        # Adjust for protective factors (each increases probability by 5%)
        protective_adjustment = len(protective_factors) * 0.05

        # Calculate final probability
        success_probability = base_probability + risk_adjustment + protective_adjustment

        return max(0.0, min(1.0, success_probability))

    def _generate_recommendations(
        self,
        therapeutic_elements: dict[str, Any],
        outcome_metrics: list[OutcomeMetric],
        risk_factors: list[str],
        protective_factors: list[str]
    ) -> list[str]:
        """Generate recommendations to improve effectiveness."""
        recommendations = []

        # Intervention recommendations
        if not therapeutic_elements["interventions"]:
            recommendations.append("Incorporate evidence-based therapeutic interventions")
        elif len(therapeutic_elements["interventions"]) == 1:
            recommendations.append("Consider integrating additional therapeutic approaches")

        # Alliance recommendations
        if therapeutic_elements["therapeutic_alliance"] < 0.5:
            recommendations.append("Focus on strengthening therapeutic alliance")
            recommendations.append("Increase empathy and validation in responses")

        # Engagement recommendations
        if therapeutic_elements["client_engagement"] < 0.5:
            recommendations.append("Enhance client motivation and engagement")
            recommendations.append("Explore barriers to treatment participation")

        # Risk factor recommendations
        if "substance_use" in risk_factors:
            recommendations.append("Address substance use as priority treatment target")
        if "trauma_history" in risk_factors:
            recommendations.append("Consider trauma-informed treatment approaches")
        if "social_isolation" in risk_factors:
            recommendations.append("Incorporate social support building interventions")

        # Protective factor recommendations
        if "social_support" not in protective_factors:
            recommendations.append("Assess and strengthen social support network")
        if "coping_skills" not in protective_factors:
            recommendations.append("Focus on coping skills development")

        # Outcome-specific recommendations
        low_outcome_metrics = [m for m in outcome_metrics if m.predicted_score < 0.5]
        for metric in low_outcome_metrics:
            if metric.metric_type == OutcomeType.SYMPTOM_REDUCTION:
                recommendations.append("Increase focus on symptom-specific interventions")
            elif metric.metric_type == OutcomeType.FUNCTIONAL_IMPROVEMENT:
                recommendations.append("Incorporate behavioral activation and skill building")

        return list(set(recommendations))  # Remove duplicates

    def _compile_evidence_base(self, therapeutic_elements: dict[str, Any]) -> dict[str, Any]:
        """Compile relevant evidence base information."""
        evidence = {
            "interventions_used": therapeutic_elements["interventions"],
            "evidence_ratings": {},
            "effect_sizes": {},
            "success_rates": {}
        }

        for intervention in therapeutic_elements["interventions"]:
            if intervention in self.evidence_base["intervention_effectiveness"]:
                intervention_data = self.evidence_base["intervention_effectiveness"][intervention]
                evidence["evidence_ratings"][intervention] = {}
                evidence["effect_sizes"][intervention] = {}
                evidence["success_rates"][intervention] = {}

                for condition, data in intervention_data.items():
                    evidence["evidence_ratings"][intervention][condition] = data.get("evidence", "unknown")
                    evidence["effect_sizes"][intervention][condition] = data.get("effect_size", 0.0)
                    evidence["success_rates"][intervention][condition] = data.get("success_rate", 0.0)

        return evidence

    def get_prediction_summary(self) -> dict[str, Any]:
        """Get prediction summary statistics."""
        if not self.prediction_history:
            return {"message": "No predictions performed yet"}

        total_predictions = len(self.prediction_history)

        # Effectiveness distribution
        effectiveness_counts = {}
        for level in EffectivenessLevel:
            effectiveness_counts[level.value] = sum(
                1 for p in self.prediction_history if p.overall_effectiveness == level
            )

        # Average success probability
        avg_success_probability = sum(p.success_probability for p in self.prediction_history) / total_predictions

        # Most common risk factors
        all_risk_factors = [factor for p in self.prediction_history for factor in p.risk_factors]
        risk_factor_counts = {}
        for factor in set(all_risk_factors):
            risk_factor_counts[factor] = all_risk_factors.count(factor)

        # Most common protective factors
        all_protective_factors = [factor for p in self.prediction_history for factor in p.protective_factors]
        protective_factor_counts = {}
        for factor in set(all_protective_factors):
            protective_factor_counts[factor] = all_protective_factors.count(factor)

        return {
            "total_predictions": total_predictions,
            "effectiveness_distribution": effectiveness_counts,
            "average_success_probability": avg_success_probability,
            "common_risk_factors": dict(sorted(risk_factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "common_protective_factors": dict(sorted(protective_factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "last_prediction": self.prediction_history[-1].timestamp.isoformat()
        }


def main():
    """Example usage of the TherapeuticEffectivenessPredictor."""
    predictor = TherapeuticEffectivenessPredictor()

    # Example conversations
    sample_conversations = [
        {
            "id": "effectiveness_001",
            "content": "I understand you're struggling with depression. Let's work together using cognitive behavioral techniques to identify and challenge negative thought patterns. I believe we can make significant progress.",
            "turns": [
                {"speaker": "user", "text": "I've been depressed for months and nothing helps."},
                {"speaker": "therapist", "text": "Let's work together using CBT techniques. I believe we can make progress."}
            ],
            "metadata": {"condition": "depression", "session_number": 1}
        },
        {
            "id": "effectiveness_002",
            "content": "You mentioned having good family support and being motivated to change. These are great strengths we can build on using mindfulness and coping strategies.",
            "turns": [
                {"speaker": "user", "text": "I have good family support and want to get better."},
                {"speaker": "therapist", "text": "These are great strengths. Let's use mindfulness and coping strategies."}
            ],
            "metadata": {"condition": "anxiety", "session_number": 3}
        }
    ]

    # Predict effectiveness
    for conversation in sample_conversations:
        prediction = predictor.predict_effectiveness(conversation)


        for _metric in prediction.outcome_metrics:
            pass

        if prediction.recommendations:
            for _rec in prediction.recommendations[:3]:
                pass

    # Print summary
    predictor.get_prediction_summary()


if __name__ == "__main__":
    main()

# Alias for compatibility
EffectivenessPredictor = TherapeuticEffectivenessPredictor
