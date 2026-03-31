import json
import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeedbackEntry:
    """
    Represents a single atomic unit of feedback for a conversation or generation.
    Used to track human-in-the-loop (HITL) corrections or automated quality gate rejections.
    """

    def __init__(
        self, item_id: str, rating: float, context: str, source: str = "automated"
    ):
        """Initialize the feedback entry."""
        if not isinstance(rating, (int, float)):
            raise ValueError("Rating must be numeric.")

        self.item_id = item_id
        self.rating = rating
        self.context = context
        self.source = source
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize feedback entry to dictionary."""
        return {
            "item_id": self.item_id,
            "rating": self.rating,
            "context": self.context,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }


class FeedbackLoops:
    """
    Conversation Effectiveness Feedback Loops.

    Collects, aggregates, and acts upon qualitative and quantitative feedback
    directed at the training datasets. Feeds back into `AdaptiveLearner` and
    generation pipelines to correct repeated mistakes structurally.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FeedbackLoops processor.
        """
        self.config = config or {
            "ingestion_batch_size": 100,
            "confidence_threshold": 0.85,
            "max_memory_buffer": 10000,
            "auto_correct_patterns": True,
        }

        # We use a deque for fast FIFO operations bounded by our buffer size
        self.memory_buffer: deque = deque(
            maxlen=self.config.get("max_memory_buffer", 1000)
        )
        self.aggregate_metrics = {
            "total_received": 0,
            "average_rating": 0.0,
            "source_distribution": {},
        }

        logger.info("FeedbackLoops initialized cleanly.")

    def ingest_feedback(self, entry: Dict[str, Any]) -> bool:
        """
        Accept a raw dictionary feedback entry, parse it, and put it
        into the internal processing queue.
        """
        if not isinstance(entry, dict):
            raise ValueError("Feedback entry must be a dictionary.")

        try:
            item_id = entry.get("item_id", "unknown")
            rating = float(entry.get("rating", 0.0))
            context = entry.get("context", "")
            source = entry.get("source", "automated")

            fb = FeedbackEntry(item_id, rating, context, source)
            self.memory_buffer.append(fb)

            # Streaming updates to metrics
            self.aggregate_metrics["total_received"] += 1

            n = self.aggregate_metrics["total_received"]
            old_avg = self.aggregate_metrics["average_rating"]
            self.aggregate_metrics["average_rating"] = old_avg + (
                (rating - old_avg) / n
            )

            # Distribution tracking
            dist = self.aggregate_metrics["source_distribution"]
            dist[source] = dist.get(source, 0) + 1

            return True
        except Exception as e:
            logger.error(f"Failed to ingest feedback entry: {e}")
            return False

    def identify_anti_patterns(self) -> List[Dict[str, Any]]:
        """
        Analyze the memory buffer to find common features in highly requested/poorly rated prompts.
        This provides structured directives to the generation layer to stop
        producing conversational dead-ends.
        """
        logger.info("Analyzing feedback buffer for anti-patterns...")

        # Normally would cluster the context string embeddings using NLP
        # Here we mock finding string commonalities
        anti_patterns = []
        failure_contexts = [
            fb.context
            for fb in self.memory_buffer
            if fb.rating < self.config["confidence_threshold"]
        ]

        try:
            # Mocked naive pattern extraction
            dummy_keywords = ["toxic positivity", "abrupt ending", "unhelpful generic"]
            # ⚡ Bolt: Prevent O(n*m) complexity by compiling a single regex pattern for O(N) searching
            import re
            pattern = re.compile("|".join(map(re.escape, dummy_keywords)))
            counts = {k: 0 for k in dummy_keywords}
            for c in failure_contexts:
                for match in set(pattern.findall(c)):
                    counts[match] += 1
            for keyword, matches in counts.items():
                if matches > 5:
                    anti_patterns.append(
                        {
                            "pattern": keyword,
                            "frequency": matches,
                            "severity": "high" if matches > 20 else "medium",
                        }
                    )
        except Exception as e:
            logger.error(f"Error during anti-pattern extraction: {e}")

        return anti_patterns

    def generate_correction_directives(self) -> str:
        """
        Transforms identified anti-patterns into system-prompt modifications
        to be utilized by the Datagen API or NeMo orchestrator.
        """
        patterns = self.identify_anti_patterns()

        if not patterns:
            return (
                "Feedback nominal. No systemic prompt corrections advised at this time."
            )

        bases = ["CRITICAL INSTRUCTIONS ADDED BY FEEDBACK LOOP:"]
        for p in patterns:
            bases.append(
                f"- AVOID pattern '{p['pattern']}' (Severity: {p['severity']} frequency: {p['frequency']})"
            )

        compiled = "\n".join(bases)
        logger.info(
            f"Generated correction directives based on {len(patterns)} patterns."
        )

        return compiled

    def export_feedback_logs(self, filepath: str) -> None:
        """
        Export current buffer state for manual offline analysis.
        """
        if not filepath:
            raise ValueError("Filepath required to export logs.")

        try:
            with open(filepath, "w") as f:
                for fb in self.memory_buffer:
                    f.write(json.dumps(fb.to_dict()) + "\n")
            logger.info(
                f"Exported {len(self.memory_buffer)} feedback components to {filepath}"
            )
        except Exception as e:
            logger.error(f"Export failed: {e}")


def test_feedback_loops():
    """Verify that feedback can be ingested, averaged, and analyzed."""
    floops = FeedbackLoops()

    # Ingest mix of feedback
    floops.ingest_feedback(
        {"item_id": "a", "rating": 0.9, "context": "great", "source": "hitl"}
    )
    floops.ingest_feedback(
        {
            "item_id": "b",
            "rating": 0.2,
            "context": "toxic positivity used",
            "source": "automated",
        }
    )
    floops.ingest_feedback(
        {
            "item_id": "c",
            "rating": 0.1,
            "context": "extremely abrupt ending. toxic positivity.",
            "source": "hitl",
        }
    )

    for _ in range(6):
        floops.ingest_feedback(
            {
                "item_id": "d",
                "rating": 0.3,
                "context": "toxic positivity detected in transcript",
                "source": "automated",
            }
        )

    assert floops.aggregate_metrics["total_received"] == 9
    assert len(floops.memory_buffer) == 9

    # Analyze
    patterns = floops.identify_anti_patterns()
    assert any(p["pattern"] == "toxic positivity" for p in patterns)

    directives = floops.generate_correction_directives()
    assert "AVOID pattern" in directives

    print("FeedbackLoops passed enterprise checks.")


if __name__ == "__main__":
    test_feedback_loops()
