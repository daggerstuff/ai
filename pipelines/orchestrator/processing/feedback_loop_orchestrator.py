import logging
from typing import Any, Dict, List

from ai.pipelines.orchestrator.processing.nvidia_clients import (
    NemoCustomizerClient,
    NemoEvaluatorClient,
)

logger = logging.getLogger(__name__)


class FeedbackLoopOrchestrator:
    """
    Orchestrates the feedback loop between NeMo Evaluator and NeMo Customizer.
    This implements 'Resonance-Optimal Tuning' by adjusting LoRA based on
    empathic resonance scores.
    """

    def __init__(self):
        self.evaluator = NemoEvaluatorClient()
        self.customizer = NemoCustomizerClient()

    def optimize_training_resonance(
        self, model_id: str, sample_interactions: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        1. Evaluates samples for resonance.
        2. Dispatches an optimization job to Customizer.
        """
        logger.info(f"Starting resonance optimization loop for model: {model_id}")

        resonance_scores = []
        for interaction in sample_interactions:
            try:
                res = self.evaluator.measure_empathic_resonance(
                    user_utterance=interaction["user"], bot_response=interaction["bot"]
                )
                resonance_scores.append(res.get("score", 0.0))
            except Exception as e:
                logger.error(f"Failed to measure resonance for sample: {e}")

        if not resonance_scores:
            return {"status": "skipped", "reason": "no resonance scores collected"}

        avg_resonance = sum(resonance_scores) / len(resonance_scores)
        logger.info(f"Average resonance score: {avg_resonance:.4f}")

        # Dispatch to Customizer for resonance-optimal tuning
        try:
            result = self.customizer.resonance_optimal_tuning(
                model_id=model_id, resonance_scores=resonance_scores
            )
            logger.info(f"Optimization job dispatched: {result.get('id')}")
            return result
        except Exception as e:
            logger.error(f"Failed to dispatch optimization: {e}")
            return {"status": "error", "message": str(e)}

    def detect_and_fix_drift(
        self, model_id: str, session_history: List[str], persona: str
    ):
        """
        Detects therapeutic drift and triggers a 'therapeutic essence distillation'
        if the drift is too high.
        """
        try:
            drift_result = self.evaluator.detect_therapeutic_drift(
                session_history, persona
            )
            if drift_result.get("drift_detected", False):
                logger.warning(
                    f"🚨 THERAPEUTIC DRIFT DETECTED for {model_id}! "
                    "Triggering essence distillation."
                )
                return self.customizer.distill_therapeutic_essence(
                    teacher_id="expert_therapeutic_teacher_v1"
                )
            return {"status": "stable"}
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Example usage
    orchestrator = FeedbackLoopOrchestrator()
    samples = [
        {
            "user": "I feel lonely.",
            "bot": "I'm sorry you're feeling that way. I'm here.",
        },
        {
            "user": "It's hard to get through the day.",
            "bot": "I hear how much of a struggle it is for you.",
        },
    ]
    orchestrator.optimize_training_resonance("base_therapeutic_model_v1", samples)
