"""
EvaluationMetricsAggregator: Aggregates results from all core evaluation modules for Pixel Evaluation Framework

Integrates:
- EQEvaluator
- PersonaSwitchingEvaluator
- ClinicalAccuracyEvaluator
- EmpathyEvaluator
- ConversationalQualityEvaluator
- TherapeuticAppropriatenessEvaluator

Provides unified metrics and scoring for model evaluation.
"""

from typing import Any

from .clinical_accuracy_evaluator import ClinicalAccuracyEvaluator
from .conversational_quality_evaluator import ConversationalQualityEvaluator
from .empathy_evaluator import EmpathyEvaluator
from .eq_evaluator import EQEvaluator
from .persona_switching_evaluator import PersonaSwitchingEvaluator
from .therapeutic_appropriateness_evaluator import TherapeuticAppropriatenessEvaluator


class EvaluationMetricsAggregator:
    def __init__(self):
        self.eq_evaluator = EQEvaluator()
        self.persona_evaluator = PersonaSwitchingEvaluator()
        self.clinical_evaluator = ClinicalAccuracyEvaluator()
        self.empathy_evaluator = EmpathyEvaluator()
        self.quality_evaluator = ConversationalQualityEvaluator()
        self.therapeutic_evaluator = TherapeuticAppropriatenessEvaluator()

    def aggregate(self, conversation: Any) -> dict[str, float]:
        """
        Run all evaluators and aggregate their results into a unified metrics dictionary.
        """
        metrics = {}
        metrics.update({"eq_" + k: v for k, v in self.eq_evaluator.evaluate(conversation).items()})
        metrics.update(
            {"persona_" + k: v for k, v in self.persona_evaluator.evaluate(conversation).items()}
        )
        metrics.update(
            {"clinical_" + k: v for k, v in self.clinical_evaluator.evaluate(conversation).items()}
        )
        metrics.update(
            {"empathy_" + k: v for k, v in self.empathy_evaluator.evaluate(conversation).items()}
        )
        metrics.update(
            {"quality_" + k: v for k, v in self.quality_evaluator.evaluate(conversation).items()}
        )
        metrics.update(
            {
                "therapeutic_" + k: v
                for k, v in self.therapeutic_evaluator.evaluate(conversation).items()
            }
        )
        # Optionally, add overall scoring/aggregation here
        return metrics
