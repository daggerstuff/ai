import logging
from typing import Any, Dict, List, Optional

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
        self,
        model_id: str,
        session_history: List[str],
        persona: str,
        current_interactions: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Detects therapeutic drift using both formal narrative analysis and
        empathic resonance feedback (Resonance-Optimal Tuning).
        """
        try:
            # 1. Check formal drift (narrative/framework consistency)
            drift_result = self.evaluator.detect_therapeutic_drift(
                session_history, persona
            )

            # 2. Check resonance if current interactions are provided
            resonance_drift = False
            avg_resonance = 0.0
            resonance_scores = []

            if current_interactions:
                for interaction in current_interactions:
                    try:
                        res = self.evaluator.measure_empathic_resonance(
                            user_utterance=interaction.get("user", ""),
                            bot_response=interaction.get("bot", ""),
                        )
                        score = res.get("score", 0.0)
                        resonance_scores.append(score)
                    except Exception as e:
                        logger.error(f"Failed to measure resonance for segment: {e}")

                if resonance_scores:
                    avg_resonance = sum(resonance_scores) / len(resonance_scores)
                    # Drift threshold: resonance < 0.6 indicates empathic disconnection
                    if avg_resonance < 0.6:
                        resonance_drift = True

            if drift_result.get("drift_detected", False) or resonance_drift:
                cause = []
                if drift_result.get("drift_detected"):
                    cause.append("Narrative Shift")
                if resonance_drift:
                    cause.append(f"Low Resonance ({avg_resonance:.2f})")

                logger.warning(
                    f"🚨 THERAPEUTIC DRIFT DETECTED for {model_id}! "
                    f"Causes: {', '.join(cause)}"
                )

                # Use Resonance-Optimal Tuning as the primary fix if we have scores
                if resonance_scores:
                    logger.info("Triggering Resonance-Optimal Tuning feedback loop.")
                    return self.customizer.resonance_optimal_tuning(
                        model_id=model_id, resonance_scores=resonance_scores
                    )
                else:
                    logger.info("Falling back to expert therapeutic distillation.")
                    return self.customizer.distill_therapeutic_essence(
                        teacher_id="expert_therapeutic_teacher_v1"
                    )

            return {"status": "stable", "avg_resonance": avg_resonance}
        except Exception as e:
            logger.error(f"Drift detection pipeline failed: {e}")
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
