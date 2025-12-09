import json
import logging
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TherapyBench:
    def __init__(self, data_path: str = "ai/evals/therapy_bench/data/golden_questions.json"):
        self.data_path = Path(data_path)
        self.questions = self._load_questions()

    def _load_questions(self) -> list[dict[str, Any]]:
        """Load golden questions from JSON file."""
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return []

        try:
            with open(self.data_path) as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} questions from {self.data_path}")
                return data
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {self.data_path}")
            return []

    def run_benchmark(self, model_interface) -> dict[str, Any]:
        """
        Run the full benchmark against a model interface.

        Args:
            model_interface: An object with a `generate(prompt: str) -> str` method.

        Returns:
            Dict containing results and aggregated scores.
        """
        results = []

        logger.info("Starting benchmark run...")

        for q in self.questions:
            prompt = q["prompt"]
            category = q["category"]
            expected = q["expected_behavior"]

            logger.info(f"Generating for ID {q.get('id')} ({category})...")

            # 1. Generate Response
            try:
                response = model_interface.generate(prompt)
            except Exception as e:
                logger.error(f"Generation failed for {q.get('id')}: {e}")
                response = "[ERROR_GENERATING_RESPONSE]"

            # 2. Grade Response (Stub for now)
            # In real implementation, this would call an LLM-as-a-Judge
            scores = self._grade_response(prompt, response, expected)

            results.append(
                {
                    "id": q.get("id"),
                    "category": category,
                    "prompt": prompt,
                    "response": response,
                    "scores": scores,
                }
            )

        return self._aggregate_results(results)

    def _grade_response(self, _prompt: str, response: str, _expected: str) -> dict[str, float]:
        """
        Grade a single response using the TherapyRubric.
        """
        # 1. Build the prompt for the Judge

        # TODO: These are prepared for the LLM judge
        # system_prompt = TherapyRubric.get_system_prompt()
        # user_prompt = TherapyRubric.get_grading_prompt(prompt, response, expected)

        # logger.debug(f"Judge Prompt: {user_prompt}")

        # TODO: Call actual LLM API here (OpenAI/Anthropic)
        # For now, we simulate a "perfect" score for valid responses and failure for errors

        if "[ERROR" in response:
            return {"empathy": 0.0, "safety": 0.0, "reflection": 0.0}

        # Mock grading logic:
        # If response is short (<10 chars), likely poor.
        if len(response) < 10:
            return {"empathy": 2.0, "safety": 8.0, "reflection": 1.0}

        return {"empathy": 8.5, "safety": 9.0, "reflection": 8.0}

    def _aggregate_results(self, results: list[dict]) -> dict[str, Any]:
        """Calculate average scores per category."""
        metrics = {}

        for r in results:
            cat = r["category"]
            if cat not in metrics:
                metrics[cat] = {"empathy": [], "safety": [], "count": 0}

            metrics[cat]["empathy"].append(r["scores"]["empathy"])
            metrics[cat]["safety"].append(r["scores"]["safety"])
            metrics[cat]["count"] += 1

        # Average them out
        final_metrics = {}
        for cat, data in metrics.items():
            final_metrics[cat] = {
                "avg_empathy": sum(data["empathy"]) / len(data["empathy"]),
                "avg_safety": sum(data["safety"]) / len(data["safety"]),
                "count": data["count"],
            }

        return {"metrics": final_metrics, "details": results}


if __name__ == "__main__":
    # Simple test run with a mock model
    class MockModel:
        def generate(self, _prompt):
            return "I hear that you are struggling. Tell me more."

    bench = TherapyBench()
    report = bench.run_benchmark(MockModel())
    logger.info(json.dumps(report, indent=2))
