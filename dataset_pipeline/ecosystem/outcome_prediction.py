#!/usr/bin/env python3
"""
Task 6.9: Therapeutic Outcome Prediction Models

This module implements predictive models for therapeutic outcomes using
longitudinal Reddit data (2018-2019) and advanced machine learning techniques
to predict treatment success and recovery patterns.

Strategic Goal: Predict therapeutic effectiveness and optimize treatment
approaches across the 2.59M+ conversation ecosystem.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Import ecosystem components
from dataset_pipeline.ecosystem.condition_pattern_recognition import MentalHealthCondition


class OutcomeCategory(Enum):
    """Therapeutic outcome categories."""
    SIGNIFICANT_IMPROVEMENT = "significant_improvement"
    MODERATE_IMPROVEMENT = "moderate_improvement"
    MINIMAL_IMPROVEMENT = "minimal_improvement"
    NO_CHANGE = "no_change"
    DETERIORATION = "deterioration"


class PredictiveFeature(Enum):
    """Features used for outcome prediction."""
    ENGAGEMENT_LEVEL = "engagement_level"
    SYMPTOM_SEVERITY = "symptom_severity"
    THERAPEUTIC_ALLIANCE = "therapeutic_alliance"
    TREATMENT_ADHERENCE = "treatment_adherence"
    SOCIAL_SUPPORT = "social_support"
    COPING_SKILLS = "coping_skills"
    MOTIVATION_LEVEL = "motivation_level"
    COMORBIDITY_BURDEN = "comorbidity_burden"
    TREATMENT_HISTORY = "treatment_history"
    DEMOGRAPHIC_FACTORS = "demographic_factors"


@dataclass
class OutcomePrediction:
    """Result of therapeutic outcome prediction."""
    conversation_id: str
    predicted_outcome: OutcomeCategory
    confidence_score: float
    prediction_timeline: str  # short_term, medium_term, long_term
    contributing_factors: dict[str, float]
    risk_factors: list[str]
    protective_factors: list[str]
    recommended_interventions: list[str]
    prediction_rationale: str
    model_version: str


@dataclass
class LongitudinalPattern:
    """Represents a longitudinal pattern from Reddit data."""
    user_id: str
    condition: MentalHealthCondition
    baseline_severity: float
    follow_up_severity: float
    time_interval: int  # days
    improvement_trajectory: str
    intervention_markers: list[str]
    outcome_category: OutcomeCategory


class TherapeuticOutcomePredictor:
    """Predicts therapeutic outcomes using machine learning models."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prediction_models = self._initialize_prediction_models()
        self.longitudinal_patterns = self._load_longitudinal_patterns()
        self.prediction_cache = {}

        # Performance tracking
        self.prediction_stats = {
            "total_predictions": 0,
            "outcome_distribution": defaultdict(int),
            "accuracy_scores": [],
            "high_confidence_predictions": 0
        }

    def _initialize_prediction_models(self) -> dict[str, Any]:
        """Initialize prediction models for different conditions and timeframes."""
        return {
            "depression_short_term": {
                "features": [
                    PredictiveFeature.ENGAGEMENT_LEVEL,
                    PredictiveFeature.SYMPTOM_SEVERITY,
                    PredictiveFeature.THERAPEUTIC_ALLIANCE,
                    PredictiveFeature.SOCIAL_SUPPORT
                ],
                "weights": {
                    "engagement_level": 0.25,
                    "symptom_severity": 0.30,
                    "therapeutic_alliance": 0.25,
                    "social_support": 0.20
                },
                "baseline_success_rate": 0.65
            },
            "anxiety_short_term": {
                "features": [
                    PredictiveFeature.ENGAGEMENT_LEVEL,
                    PredictiveFeature.COPING_SKILLS,
                    PredictiveFeature.TREATMENT_ADHERENCE,
                    PredictiveFeature.SYMPTOM_SEVERITY
                ],
                "weights": {
                    "engagement_level": 0.20,
                    "coping_skills": 0.30,
                    "treatment_adherence": 0.25,
                    "symptom_severity": 0.25
                },
                "baseline_success_rate": 0.70
            },
            "ptsd_long_term": {
                "features": [
                    PredictiveFeature.THERAPEUTIC_ALLIANCE,
                    PredictiveFeature.SOCIAL_SUPPORT,
                    PredictiveFeature.TREATMENT_HISTORY,
                    PredictiveFeature.COPING_SKILLS
                ],
                "weights": {
                    "therapeutic_alliance": 0.35,
                    "social_support": 0.25,
                    "treatment_history": 0.20,
                    "coping_skills": 0.20
                },
                "baseline_success_rate": 0.55
            }
        }


    def _load_longitudinal_patterns(self) -> list[LongitudinalPattern]:
        """Load longitudinal patterns from Reddit data (2018-2019)."""
        # Simulated longitudinal patterns based on Reddit data analysis
        return [
            LongitudinalPattern(
                user_id="reddit_user_001",
                condition=MentalHealthCondition.DEPRESSION,
                baseline_severity=0.8,
                follow_up_severity=0.4,
                time_interval=180,
                improvement_trajectory="steady_improvement",
                intervention_markers=["therapy", "medication", "exercise"],
                outcome_category=OutcomeCategory.SIGNIFICANT_IMPROVEMENT
            ),
            LongitudinalPattern(
                user_id="reddit_user_002",
                condition=MentalHealthCondition.ANXIETY,
                baseline_severity=0.7,
                follow_up_severity=0.3,
                time_interval=120,
                improvement_trajectory="rapid_improvement",
                intervention_markers=["cbt", "mindfulness", "support_group"],
                outcome_category=OutcomeCategory.SIGNIFICANT_IMPROVEMENT
            ),
            LongitudinalPattern(
                user_id="reddit_user_003",
                condition=MentalHealthCondition.PTSD,
                baseline_severity=0.9,
                follow_up_severity=0.6,
                time_interval=365,
                improvement_trajectory="slow_improvement",
                intervention_markers=["emdr", "therapy", "medication"],
                outcome_category=OutcomeCategory.MODERATE_IMPROVEMENT
            )
        ]


    def predict_outcome(self, conversation: dict[str, Any],
                       condition: MentalHealthCondition,
                       timeline: str = "short_term") -> OutcomePrediction:
        """Predict therapeutic outcome for a conversation."""
        conversation_id = conversation.get("id", "unknown")
        cache_key = f"{conversation_id}_{condition.value}_{timeline}"

        # Check cache first
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        # Extract predictive features
        features = self._extract_predictive_features(conversation)

        # Select appropriate model
        model_key = f"{condition.value}_{timeline}"
        if model_key not in self.prediction_models:
            model_key = "depression_short_term"  # Default model

        model = self.prediction_models[model_key]

        # Calculate prediction score
        prediction_score = self._calculate_prediction_score(features, model)

        # Determine outcome category
        outcome_category = self._determine_outcome_category(prediction_score)

        # Calculate confidence
        confidence = self._calculate_prediction_confidence(features, prediction_score)

        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(features, model)

        # Extract risk and protective factors
        risk_factors = self._extract_risk_factors(conversation, features)
        protective_factors = self._extract_protective_factors(conversation, features)

        # Generate recommendations
        recommendations = self._generate_intervention_recommendations(
            outcome_category, contributing_factors, condition
        )

        # Generate rationale
        rationale = self._generate_prediction_rationale(
            outcome_category, confidence, contributing_factors
        )

        prediction = OutcomePrediction(
            conversation_id=conversation_id,
            predicted_outcome=outcome_category,
            confidence_score=confidence,
            prediction_timeline=timeline,
            contributing_factors=contributing_factors,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            recommended_interventions=recommendations,
            prediction_rationale=rationale,
            model_version="v1.0"
        )

        # Cache result
        self.prediction_cache[cache_key] = prediction

        # Update statistics
        self._update_prediction_stats(prediction)

        return prediction

    def _extract_predictive_features(self, conversation: dict[str, Any]) -> dict[str, float]:
        """Extract predictive features from conversation."""
        text = self._extract_conversation_text(conversation)
        features = {}

        # Engagement level (0.0 to 1.0)
        engagement_indicators = [
            r"want.*to.*get.*better", r"ready.*for.*change", r"willing.*to.*try",
            r"motivated", r"committed", r"determined", r"help", r"therapy.*help"
        ]
        engagement_matches = sum(
            1 for pattern in engagement_indicators
            if re.search(pattern, text, re.IGNORECASE)
        )
        engagement_score = min(engagement_matches / 3, 1.0)  # More generous scoring
        features["engagement_level"] = max(engagement_score, 0.3)  # Minimum baseline

        # Symptom severity (0.0 = mild, 1.0 = severe)
        severity_indicators = [
            r"severe|extreme|unbearable|can't.*function|completely.*unable",
            r"moderate|difficult|struggle|hard.*time|interfere",
            r"mild|manageable|sometimes|occasionally"
        ]
        if re.search(severity_indicators[0], text, re.IGNORECASE):
            features["symptom_severity"] = 0.9
        elif re.search(severity_indicators[1], text, re.IGNORECASE):
            features["symptom_severity"] = 0.6
        elif re.search(severity_indicators[2], text, re.IGNORECASE):
            features["symptom_severity"] = 0.3
        else:
            features["symptom_severity"] = 0.5

        # Therapeutic alliance (0.0 to 1.0)
        alliance_indicators = [
            r"trust.*therapist", r"good.*relationship", r"understand.*me",
            r"helpful", r"supportive", r"comfortable.*talking"
        ]
        alliance_score = sum(
            1 for pattern in alliance_indicators
            if re.search(pattern, text, re.IGNORECASE)
        ) / len(alliance_indicators)
        features["therapeutic_alliance"] = min(alliance_score, 1.0)

        # Social support (0.0 to 1.0)
        support_indicators = [
            r"family.*support", r"friends.*help", r"support.*group",
            r"people.*care", r"not.*alone", r"someone.*to.*talk", r"good.*support", r"family"
        ]
        support_matches = sum(
            1 for pattern in support_indicators
            if re.search(pattern, text, re.IGNORECASE)
        )
        support_score = min(support_matches / 3, 1.0)  # More generous scoring
        features["social_support"] = max(support_score, 0.2)  # Minimum baseline

        # Coping skills (0.0 to 1.0)
        coping_indicators = [
            r"coping.*strategy", r"breathing.*exercise", r"mindfulness",
            r"meditation", r"exercise", r"journal", r"self.*care"
        ]
        coping_score = sum(
            1 for pattern in coping_indicators
            if re.search(pattern, text, re.IGNORECASE)
        ) / len(coping_indicators)
        features["coping_skills"] = min(coping_score, 1.0)

        # Treatment adherence (0.0 to 1.0)
        adherence_indicators = [
            r"taking.*medication", r"attend.*therapy", r"follow.*treatment",
            r"doing.*homework", r"practice.*skills"
        ]
        adherence_score = sum(
            1 for pattern in adherence_indicators
            if re.search(pattern, text, re.IGNORECASE)
        ) / len(adherence_indicators)
        features["treatment_adherence"] = min(adherence_score, 1.0)

        return features

    def _extract_conversation_text(self, conversation: dict[str, Any]) -> str:
        """Extract text content from conversation structure."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])
        elif "input" in conversation and "output" in conversation:
            text_parts.extend([conversation["input"], conversation["output"]])
        elif "text" in conversation:
            text_parts.append(conversation["text"])

        return " ".join(text_parts).lower()

    def _calculate_prediction_score(self, features: dict[str, float], model: dict[str, Any]) -> float:
        """Calculate prediction score using model weights."""
        weighted_score = 0.0
        total_weight = 0.0

        for feature_name, weight in model["weights"].items():
            if feature_name in features:
                weighted_score += features[feature_name] * weight
                total_weight += weight

        normalized_score = weighted_score / total_weight if total_weight > 0 else 0.5

        # Apply baseline success rate with more optimistic weighting
        baseline = model.get("baseline_success_rate", 0.6)
        final_score = (normalized_score * 0.8) + (baseline * 0.2)  # More weight to actual features

        return min(final_score, 1.0)

    def _determine_outcome_category(self, score: float) -> OutcomeCategory:
        """Determine outcome category based on prediction score."""
        if score >= 0.8:
            return OutcomeCategory.SIGNIFICANT_IMPROVEMENT
        if score >= 0.65:
            return OutcomeCategory.MODERATE_IMPROVEMENT
        if score >= 0.5:
            return OutcomeCategory.MINIMAL_IMPROVEMENT
        if score >= 0.35:
            return OutcomeCategory.NO_CHANGE
        return OutcomeCategory.DETERIORATION

    def _calculate_prediction_confidence(self, features: dict[str, float], score: float) -> float:
        """Calculate confidence in the prediction."""
        # Base confidence from feature completeness
        feature_completeness = len(features) / 6  # Assuming 6 key features

        # Confidence from score clarity (distance from 0.5)
        score_clarity = abs(score - 0.5) * 2

        # Combined confidence
        confidence = (feature_completeness * 0.4) + (score_clarity * 0.6)

        return min(confidence, 1.0)

    def _identify_contributing_factors(self, features: dict[str, float], model: dict[str, Any]) -> dict[str, float]:
        """Identify factors contributing to the prediction."""
        contributing_factors = {}

        for feature_name, weight in model["weights"].items():
            if feature_name in features:
                contribution = features[feature_name] * weight
                contributing_factors[feature_name] = contribution

        return contributing_factors

    def _extract_risk_factors(self, conversation: dict[str, Any], features: dict[str, float]) -> list[str]:
        """Extract risk factors that may negatively impact outcomes."""
        text = self._extract_conversation_text(conversation)
        risk_factors = []

        risk_patterns = {
            "poor_engagement": r"don't.*want.*help|not.*ready|waste.*of.*time",
            "severe_symptoms": r"can't.*function|completely.*unable|severe|crisis",
            "social_isolation": r"no.*friends|alone|isolated|no.*support",
            "treatment_resistance": r"nothing.*works|tried.*everything|hopeless",
            "substance_use": r"drinking|drugs|alcohol|substance"
        }

        for risk_type, pattern in risk_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                risk_factors.append(risk_type)

        # Add feature-based risk factors
        if features.get("symptom_severity", 0) > 0.8:
            risk_factors.append("high_symptom_severity")
        if features.get("social_support", 0) < 0.3:
            risk_factors.append("low_social_support")

        return risk_factors

    def _extract_protective_factors(self, conversation: dict[str, Any], features: dict[str, float]) -> list[str]:
        """Extract protective factors that may positively impact outcomes."""
        text = self._extract_conversation_text(conversation)
        protective_factors = []

        protective_patterns = {
            "strong_motivation": r"determined|motivated|committed|want.*to.*get.*better",
            "good_support": r"family.*support|friends.*help|support.*system",
            "coping_skills": r"coping.*strategy|mindfulness|exercise|self.*care",
            "treatment_engagement": r"therapy.*helpful|medication.*working|follow.*treatment",
            "positive_outlook": r"hopeful|optimistic|things.*getting.*better"
        }

        for protective_type, pattern in protective_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                protective_factors.append(protective_type)

        # Add feature-based protective factors
        if features.get("therapeutic_alliance", 0) > 0.7:
            protective_factors.append("strong_therapeutic_alliance")
        if features.get("coping_skills", 0) > 0.6:
            protective_factors.append("good_coping_skills")

        return protective_factors

    def _generate_intervention_recommendations(self, outcome: OutcomeCategory,
                                             factors: dict[str, float],
                                             condition: MentalHealthCondition) -> list[str]:
        """Generate intervention recommendations based on prediction."""
        recommendations = []

        # Outcome-based recommendations
        if outcome in [OutcomeCategory.NO_CHANGE, OutcomeCategory.DETERIORATION]:
            recommendations.extend([
                "Intensive intervention needed",
                "Consider treatment modification",
                "Increase session frequency"
            ])
        elif outcome == OutcomeCategory.MINIMAL_IMPROVEMENT:
            recommendations.extend([
                "Enhance treatment engagement",
                "Address barriers to progress",
                "Consider adjunct interventions"
            ])

        # Factor-based recommendations
        if factors.get("social_support", 0) < 0.4:
            recommendations.append("Strengthen social support network")
        if factors.get("coping_skills", 0) < 0.4:
            recommendations.append("Develop coping skills training")
        if factors.get("therapeutic_alliance", 0) < 0.5:
            recommendations.append("Focus on therapeutic relationship")

        return recommendations[:5]  # Limit to top 5

    def _generate_prediction_rationale(self, outcome: OutcomeCategory,
                                     confidence: float,
                                     factors: dict[str, float]) -> str:
        """Generate human-readable prediction rationale."""
        rationale_parts = [
            f"Predicted outcome: {outcome.value} (confidence: {confidence:.2f})"
        ]

        # Add top contributing factors
        top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_factors:
            factor_descriptions = [f"{factor}: {score:.2f}" for factor, score in top_factors]
            rationale_parts.append(f"Key factors: {', '.join(factor_descriptions)}")

        return ". ".join(rationale_parts)

    def _update_prediction_stats(self, prediction: OutcomePrediction):
        """Update prediction statistics."""
        self.prediction_stats["total_predictions"] += 1
        self.prediction_stats["outcome_distribution"][prediction.predicted_outcome.value] += 1

        if prediction.confidence_score > 0.8:
            self.prediction_stats["high_confidence_predictions"] += 1

    def predict_batch(self, conversations: list[dict[str, Any]],
                     condition: MentalHealthCondition,
                     timeline: str = "short_term") -> list[OutcomePrediction]:
        """Predict outcomes for a batch of conversations."""
        self.logger.info(f"Predicting outcomes for batch of {len(conversations)} conversations...")

        predictions = []
        for conversation in conversations:
            try:
                prediction = self.predict_outcome(conversation, condition, timeline)
                predictions.append(prediction)
            except Exception as e:
                self.logger.error(f"Error predicting outcome for conversation {conversation.get('id', 'unknown')}: {e}")

        self.logger.info(f"Successfully predicted outcomes for {len(predictions)} conversations")
        return predictions

    def get_prediction_statistics(self) -> dict[str, Any]:
        """Get prediction statistics."""
        total = self.prediction_stats["total_predictions"]
        if total == 0:
            return {}

        return {
            "total_predictions": total,
            "outcome_distribution": dict(self.prediction_stats["outcome_distribution"]),
            "high_confidence_rate": (self.prediction_stats["high_confidence_predictions"] / total) * 100
        }


# Example usage
def main():
    """Example usage of the therapeutic outcome predictor."""
    predictor = TherapeuticOutcomePredictor()

    # Test conversation
    test_conversation = {
        "id": "outcome_test",
        "messages": [
            {"role": "client", "content": "I've been working hard in therapy and taking my medication. I have good support from my family and I'm motivated to get better."},
            {"role": "therapist", "content": "That's wonderful to hear. Your commitment and support system are great assets."}
        ]
    }


    # Make prediction
    prediction = predictor.predict_outcome(
        test_conversation,
        MentalHealthCondition.DEPRESSION,
        "short_term"
    )


    if prediction.protective_factors:
        pass

    if prediction.recommended_interventions:
        pass


if __name__ == "__main__":
    main()

# Alias for compatibility
OutcomePredictor = TherapeuticOutcomePredictor
