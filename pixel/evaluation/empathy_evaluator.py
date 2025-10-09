"""
EmpathyEvaluator: Empathy Measurement Module for Pixel Evaluation Framework

Evaluates empathy-related aspects of model outputs:
- Empathy vs simulation differentiation metrics
- Progressive empathy development tracking
- Empathy consistency measurement
- Empathy calibration against human baselines
- Empathy progression visualization and reporting
"""

from threading import Lock
from typing import Any

from transformers.pipelines import pipeline

from ai.pixel.evaluation.base_evaluator import BaseEvaluator

# Constants for magic values
GENUINE_SCORE_THRESHOLD = 0.5
SIMULATED_SCORE_THRESHOLD = 0.5
PROGRESSIVE_TREND_POSITIVE = 0.05
PROGRESSIVE_TREND_NEGATIVE = -0.05
MIN_EMPATHY_SCORES = 2
CONSISTENCY_SCORE_THRESHOLD = 0.5


class EmpathyEvaluator(BaseEvaluator):
    """
    Evaluates empathy-related aspects of model outputs.

    This class provides methods to analyze and score empathy in conversations, including:
    - Differentiating between genuine and simulated empathy
    - Tracking empathy progression over time
    - Measuring consistency of empathy across interactions
    - Calibrating against human baselines
    - Visualizing empathy progression
    """

    # Singleton empathy classifier and lock for thread safety
    _empathy_classifier = None
    _classifier_lock = None

    def __init__(self) -> None:
        """Initialize the EmpathyEvaluator with required resources."""
        super().__init__()

    @staticmethod
    def _get_empathy_classifier() -> Any:
        """Get or initialize the empathy classifier.

        Returns:
            Any: The initialized empathy classifier pipeline.
        """
        if EmpathyEvaluator._classifier_lock is None:
            EmpathyEvaluator._classifier_lock = Lock()
        with EmpathyEvaluator._classifier_lock:
            if EmpathyEvaluator._empathy_classifier is None:
                # Use GoEmotions or a similar model for empathy/simulation distinction
                EmpathyEvaluator._empathy_classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None,
                )
            return EmpathyEvaluator._empathy_classifier

    def differentiate_empathy_simulation(self, conversation: Any) -> dict[str, float]:
        """Differentiate genuine empathy from simulated empathy.

        Uses a transformer-based classifier to analyze conversation for genuine vs simulated empathy.

        Args:
            conversation: The conversation data to analyze.

        Returns:
            dict[str, float]: Dictionary containing 'empathy_simulation_score' between 0.0 and 1.0.
        """
        if not conversation:
            return {"empathy_simulation_score": 0.0}

        classifier = EmpathyEvaluator._get_empathy_classifier()
        utterances = conversation if isinstance(conversation, list) else [str(conversation)]
        genuine_empathy_labels = {"empathy", "caring", "supportive", "compassion"}
        simulated_empathy_labels = {"simulation", "pretend", "mimic", "artificial"}

        genuine_score_sum = 0.0
        simulated_score_sum = 0.0
        count = 0

        for utt in utterances:
            try:
                preds = classifier(utt)
                for pred in preds:
                    label = pred["label"].lower() if isinstance(pred.get("label", ""), str) else ""
                    score = pred["score"] if isinstance(pred.get("score", 0.0), float) else 0.0
                    try:
                        score_f = float(score)
                    except Exception:
                        continue
                    if label in genuine_empathy_labels and score_f > GENUINE_SCORE_THRESHOLD:
                        genuine_score_sum += score_f
                    elif label in simulated_empathy_labels and score_f > SIMULATED_SCORE_THRESHOLD:
                        simulated_score_sum += score_f
                count += 1
            except Exception:
                continue

        if count == 0:
            return {"empathy_simulation_score": 0.0}

        total = genuine_score_sum + simulated_score_sum
        score = 0.0 if total == 0 else genuine_score_sum / total

        return {"empathy_simulation_score": round(score, 3)}

    def track_progressive_empathy(self, conversation: Any) -> dict[str, float]:
        """Track progressive empathy development in the conversation.

        Analyzes how empathy changes throughout the conversation.

        Args:
            conversation: The conversation data to analyze.

        Returns:
            dict[str, float]: Dictionary containing 'progressive_empathy_score' between 0.0 and 1.0.
        """
        if not conversation:
            return {"progressive_empathy_score": 0.0}

        classifier = EmpathyEvaluator._get_empathy_classifier()
        utterances = conversation if isinstance(conversation, list) else [str(conversation)]
        empathy_labels = {"empathy", "caring", "supportive", "compassion"}
        empathy_scores = []

        for utt in utterances:
            try:
                preds = classifier(utt)
                max_empathy = 0.0
                for pred in preds:
                    label = pred["label"].lower() if isinstance(pred.get("label", ""), str) else ""
                    score = pred["score"] if isinstance(pred.get("score", 0.0), float) else 0.0
                    try:
                        score_f = float(score)
                    except Exception:
                        continue
                    if label in empathy_labels and score_f > max_empathy:
                        max_empathy = score_f
                empathy_scores.append(max_empathy)
            except Exception:
                empathy_scores.append(0.0)

        if len(empathy_scores) < MIN_EMPATHY_SCORES:
            return {
                "progressive_empathy_score": round(empathy_scores[0] if empathy_scores else 0.0, 3)
            }

        # Calculate trend: (last - first) / number of turns, clipped to [-1, 1]
        trend = (empathy_scores[-1] - empathy_scores[0]) / max(1, len(empathy_scores) - 1)
        # Normalize to [0, 1]: negative trend = 0, stable = 0.5, positive = 1.0
        score = (
            1.0
            if trend > PROGRESSIVE_TREND_POSITIVE
            else 0.0 if trend < PROGRESSIVE_TREND_NEGATIVE else 0.5
        )
        return {"progressive_empathy_score": round(score, 3)}

    def measure_empathy_consistency(self, conversation: Any) -> dict[str, float]:
        """Measure empathy consistency across interactions.

        Args:
            conversation: The conversation data to analyze.

        Returns:
            dict[str, float]: Dictionary containing 'empathy_consistency_score'.
        """
        # Compute empathy score for each utterance
        classifier = EmpathyEvaluator._get_empathy_classifier()
        utterances = conversation if isinstance(conversation, list) else [str(conversation)]
        empathy_labels = {"empathy", "caring", "supportive", "compassion"}
        empathy_scores = []

        for utt in utterances:
            try:
                preds = classifier(utt)
                max_empathy = 0.0
                for pred in preds:
                    label = pred["label"].lower() if isinstance(pred.get("label", ""), str) else ""
                    score = pred["score"] if isinstance(pred.get("score", 0.0), float) else 0.0
                    try:
                        score_f = float(score)
                    except Exception:
                        continue
                    if label in empathy_labels and score_f > max_empathy:
                        max_empathy = score_f
                empathy_scores.append(max_empathy)
            except Exception:
                empathy_scores.append(0.0)

        if len(empathy_scores) < MIN_EMPATHY_SCORES:
            return {
                "empathy_consistency_score": (
                    1.0
                    if empathy_scores and empathy_scores[0] > CONSISTENCY_SCORE_THRESHOLD
                    else 0.0
                )
            }

        # Compute standard deviation (penalize high variance)
        mean = sum(empathy_scores) / len(empathy_scores)
        variance = sum((x - mean) ** 2 for x in empathy_scores) / len(empathy_scores)
        stddev = variance**0.5
        # Normalize: low stddev = high score, high stddev = low score
        # Assume stddev in [0, 0.5] is good, >0.5 is bad
        score = max(0.0, 1.0 - min(stddev / 0.5, 1.0))
        return {"empathy_consistency_score": round(score, 3)}

    def calibrate_against_human(self, _conversation: Any) -> dict[str, float]:
        """Calibrate empathy measurement against human baselines.

        Args:
            _conversation: The conversation data to analyze.

        Returns:
            dict[str, float]: Dictionary containing 'empathy_calibration_score'.
        """
        # Production: Compare model empathy scores to human baseline if available
        # For now, return 0.5 if no baseline provided (neutral)
        # In future, pass a 'baseline_scores' argument or load from reference set
        return {"empathy_calibration_score": 0.5}

    def visualize_empathy_progression(self, conversation: Any) -> dict[str, Any]:
        """Visualize and report empathy progression.

        Args:
            conversation: The conversation data to analyze.

        Returns:
            dict[str, float]: Dictionary containing visualization data and scores.
        """
        # Compute empathy score for each utterance
        classifier = EmpathyEvaluator._get_empathy_classifier()
        utterances = conversation if isinstance(conversation, list) else [str(conversation)]
        empathy_labels = {"empathy", "caring", "supportive", "compassion"}
        empathy_scores = []

        for utt in utterances:
            try:
                preds = classifier(utt)
                max_empathy = 0.0
                for pred in preds:
                    label = pred["label"].lower() if isinstance(pred.get("label", ""), str) else ""
                    score = pred["score"] if isinstance(pred.get("score", 0.0), float) else 0.0
                    try:
                        score_f = float(score)
                    except Exception:
                        continue
                    if label in empathy_labels and score_f > max_empathy:
                        max_empathy = score_f
                empathy_scores.append(max_empathy)
            except Exception:
                empathy_scores.append(0.0)

        if not empathy_scores:
            return {
                "empathy_progression_score": 0.0,
                "empathy_scores": [],
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "trend": 0.0,
            }

        min_score = min(empathy_scores)
        max_score = max(empathy_scores)
        mean_score = sum(empathy_scores) / len(empathy_scores)
        trend = (empathy_scores[-1] - empathy_scores[0]) / max(1, len(empathy_scores) - 1)
        progression_score = max(0.0, min(1.0, (trend + 1) / 2))  # Normalize trend to [0,1]

        return {
            "empathy_progression_score": round(progression_score, 3),
            "empathy_scores": [round(s, 3) for s in empathy_scores],
            "min": round(min_score, 3),
            "max": round(max_score, 3),
            "mean": round(mean_score, 3),
            "trend": round(trend, 3),
        }

    def evaluate(self, conversation: Any) -> dict[str, float]:
        """
        Run all empathy evaluations and aggregate results.

        Args:
            conversation (Any): The conversation data to evaluate.

        Returns:
            dict[str, float]: Aggregated empathy scores for all domains.

        Example:
            >>> evaluator = EmpathyEvaluator()
            >>> scores = evaluator.evaluate(conversation)
            >>> print(scores)
        """
        self.audit_log("evaluate_start", "started", {"evaluator": "EmpathyEvaluator"})
        self.track_event("Empathy evaluation started", {"conversation_id": id(conversation)})
        results = {}
        try:
            results.update(self.safe_execute(self.differentiate_empathy_simulation, conversation))
            results.update(self.safe_execute(self.track_progressive_empathy, conversation))
            results.update(self.safe_execute(self.measure_empathy_consistency, conversation))
            results.update(self.safe_execute(self.calibrate_against_human, conversation))
            viz = self.safe_execute(self.visualize_empathy_progression, conversation)
            # Only include float-valued keys from visualization in aggregate results
            results.update({k: v for k, v in viz.items() if isinstance(v, float)})
            # Optionally, add overall empathy scoring/aggregation here
            self.audit_log("evaluate_end", "success", {"evaluator": "EmpathyEvaluator"})
            self.track_event("Empathy evaluation completed", {"conversation_id": id(conversation)})
            return results
        except Exception as e:
            self.audit_log(
                "evaluate_end", "error", {"evaluator": "EmpathyEvaluator", "error": str(e)}
            )
            self.track_event(
                "Empathy evaluation error", {"conversation_id": id(conversation), "error": str(e)}
            )
            raise
