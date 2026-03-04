import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class AdaptiveLearningStrategy(ABC):
    """
    Abstract base class for adaptive learning strategies.
    Defines the contract for implementing different learning algorithms
    (e.g., active learning, curriculum learning).
    """

    @abstractmethod
    def evaluate_sample(self, sample: Dict[str, Any]) -> float:
        """Evaluate the value of a sample for learning."""
        pass

    @abstractmethod
    def update_model(self, metrics: Dict[str, Any]) -> None:
        """Update the strategy based on feedback metrics."""
        pass


class ActiveLearningStrategy(AdaptiveLearningStrategy):
    """
    Implements active learning by selecting samples with the highest uncertainty
    or those that are expected to yield the most information.
    """

    def __init__(self, uncertainty_threshold: float = 0.5):
        """Initialize with a specific uncertainty threshold."""
        self.uncertainty_threshold = uncertainty_threshold
        self.total_evaluated = 0
        self.selected_count = 0

    def evaluate_sample(self, sample: Dict[str, Any]) -> float:
        """Compute an uncertainty score for the sample."""
        if not isinstance(sample, dict):
            raise ValueError("Sample must be a dictionary.")
        # Mock calculation: return a simulated uncertainty score based on text length or metadata
        text = sample.get("text", "")
        # Dummy math for text complexity representing uncertainty
        score = min(1.0, len(text) / 1000.0)
        self.total_evaluated += 1
        return score

    def update_model(self, metrics: Dict[str, Any]) -> None:
        """Adjust thresholds based on incoming performance metrics."""
        success_rate = metrics.get("success_rate", 0.0)
        try:
            if success_rate > 0.8:
                self.uncertainty_threshold = min(0.9, self.uncertainty_threshold + 0.05)
            else:
                self.uncertainty_threshold = max(0.1, self.uncertainty_threshold - 0.05)
            logger.info(
                f"Updated ActiveLearningStrategy threshold to {self.uncertainty_threshold}"
            )
        except Exception as e:
            logger.error(f"Failed to update ActiveLearningStrategy: {e}")


class AdaptiveLearner:
    """
    The Adaptive Learner orchestrates the continuous improvement of the models.

    It consumes streaming feedback and dataset inputs, evaluates them via a
    configured learning strategy, and iteratively updates training sets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AdaptiveLearner with comprehensive configuration.
        """
        self.config = config or {
            "learning_rate_adjustment": True,
            "min_samples_per_batch": 100,
            "max_samples_per_batch": 1000,
            "strategy": "active_learning",
        }
        self.state = {"is_running": False, "iterations": 0, "last_error": None}
        self._strategy = self._initialize_strategy()
        logger.info("AdaptiveLearner initialized robustly.")

    def _initialize_strategy(self) -> AdaptiveLearningStrategy:
        """Internal factory for learning strategies based on config."""
        strategy_name = self.config.get("strategy", "active_learning")
        if strategy_name == "active_learning":
            return ActiveLearningStrategy()
        else:
            raise ValueError(f"Unknown learning strategy: {strategy_name}")

    def validate_inputs(self, dataset_stream: Iterator[Dict[str, Any]]) -> bool:
        """
        Validate incoming streams before processing to prevent pipeline failures.
        """
        if dataset_stream is None:
            raise ValueError("Dataset stream cannot be None.")
        if not hasattr(dataset_stream, "__iter__"):
            raise ValueError("Dataset stream must be an iterator or generator.")
        return True

    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of samples, selecting those that pass the active learning threshold.
        """
        selected_samples = []
        try:
            for sample in batch:
                if not isinstance(sample, dict):
                    logger.warning("Skipping invalid sample format.")
                    continue

                score = self._strategy.evaluate_sample(sample)
                # If score beats threshold, we select this sample for adaptive training
                threshold = getattr(self._strategy, "uncertainty_threshold", 0.5)
                if score >= threshold:
                    sample["adaptive_score"] = score
                    selected_samples.append(sample)
        except Exception as e:
            logger.error(f"Error processing batch in AdaptiveLearner: {e}")
            self.state["last_error"] = str(e)

        return selected_samples

    def start_adaptive_learning(
        self, dataset_stream: Iterator[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Main entry point to start the continuous adaptive learning loop.
        Processes streams in memory.
        """
        self.validate_inputs(dataset_stream)
        self.state["is_running"] = True
        logger.info("Starting adaptive learning pipeline.")

        results = {"total_processed": 0, "total_selected": 0, "iterations": 0}

        batch = []
        try:
            for sample in dataset_stream:
                batch.append(sample)
                results["total_processed"] += 1

                if len(batch) >= self.config["max_samples_per_batch"]:
                    selected = self.process_batch(batch)
                    results["total_selected"] += len(selected)
                    batch = []
                    self.state["iterations"] += 1
                    results["iterations"] = self.state["iterations"]

                    # Mock updating model
                    self._strategy.update_model({"success_rate": 0.85})

            # Process remaining items in the final batch
            if batch:
                selected = self.process_batch(batch)
                results["total_selected"] += len(selected)
                self.state["iterations"] += 1
                results["iterations"] = self.state["iterations"]

        except Exception as e:
            logger.error(f"Critical failure during adaptive learning: {e}")
            self.state["is_running"] = False
            raise

        self.state["is_running"] = False
        logger.info(
            f"Adaptive learning completed. Selected {results['total_selected']}/{results['total_processed']} samples."
        )
        return results


def test_adaptive_learner():
    """Unit tests to verify functionality and ensure the component meets enterprise standards."""
    learner = AdaptiveLearner({"max_samples_per_batch": 2})

    # Create a mock stream
    def mock_stream():
        yield {"text": "Hello world", "label": 1}
        yield {"text": "A very long text " * 100, "label": 0}
        yield {"text": "Another text", "label": 1}

    results = learner.start_adaptive_learning(mock_stream())

    assert results["total_processed"] == 3
    assert results["iterations"] >= 1
    assert learner.state["is_running"] is False
    print("AdaptiveLearner test passed successfully.")


if __name__ == "__main__":
    test_adaptive_learner()
