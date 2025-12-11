import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TherapyBench:
    def __init__(
        self,
        data_path: str = "ai/evals/therapy_bench/data/golden_questions.json",
        results_dir: str | Path | None = None,
        judge_driver: str = "openai",
        judge_config: dict[str, Any] | None = None,
    ):
        self.data_path = Path(data_path)
        self.results_dir = (
            Path(results_dir) if results_dir is not None else Path("ai/evals/therapy_bench/results")
        )
        self.questions = self._load_questions()

        # Initialize Judge Client (defaults to OpenAI-compatible driver; can be overridden)
        from ai.common.llm_client import LLMClient

        self.judge_client = LLMClient(driver=judge_driver, config=judge_config)

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

    def run_benchmark(
        self,
        model_interface,
        *,
        persist: bool = True,
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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

            # 2. Grade Response
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

        aggregated = self._aggregate_results(results)

        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": getattr(model_interface, "name", model_interface.__class__.__name__),
            "driver": type(getattr(self.judge_client, "driver", object())).__name__,
            "question_count": len(self.questions),
            "data_path": str(self.data_path),
        }
        if run_metadata:
            metadata |= run_metadata

        aggregated["run_metadata"] = metadata

        if persist and (persisted_path := self._persist_results(aggregated, metadata)):
            aggregated["persisted_path"] = str(persisted_path)

        return aggregated

    def _grade_response(self, prompt: str, response: str, expected: str) -> dict[str, float]:
        """
        Grade a single response using the TherapyRubric via LLM.
        """
        from ai.evals.therapy_bench.rubric import TherapyRubric

        if "[ERROR" in response:
            return {"empathy": 0.0, "safety": 0.0, "reflection": 0.0}

        system_prompt = TherapyRubric.get_system_prompt()
        user_prompt = TherapyRubric.get_grading_prompt(prompt, response, expected)

        # Define expected schema for structured output
        schema = {
            "safety_score": 0.0,
            "empathy_score": 0.0,
            "reflection_score": 0.0,
            "reasoning": "string"
        }

        try:
            # Call the Judge
            result = self.judge_client.generate_structured(
                prompt=user_prompt,
                schema=schema,
                system_prompt=system_prompt
            )

            # Extract scores (handling potential key variations or strings)
            return {
                "empathy": float(result.get("empathy_score", 0)),
                "safety": float(result.get("safety_score", 0)),
                "reflection": float(result.get("reflection_score", 0))
            }
        except Exception as e:
            logger.error(f"Grading failed: {e}")
            # Fallback for failure
            return {"empathy": 0.0, "safety": 0.0, "reflection": 0.0}

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
        final_metrics = {
            cat: {
                "avg_empathy": sum(data["empathy"]) / len(data["empathy"]),
                "avg_safety": sum(data["safety"]) / len(data["safety"]),
                "count": data["count"],
            }
            for cat, data in metrics.items()
        }

        return {"metrics": final_metrics, "details": results}

    def _persist_results(self, aggregated_results: dict[str, Any], metadata: dict[str, Any]) -> Path | None:
        """
        Persist benchmark run to disk (atomic write).
        """
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)

            timestamp = metadata.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"))
            model_fragment = metadata.get("model_name", "model")
            safe_model_fragment = "".join(
                c if c.isalnum() or c in {"-", "_"} else "-" for c in model_fragment
            )[:64]

            filename = f"{timestamp}_{safe_model_fragment or 'model'}.json"
            destination = self.results_dir / filename
            tmp_path = destination.with_suffix(".tmp")

            payload = {
                "run_metadata": metadata,
                "results": aggregated_results,
            }

            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(destination)

            logger.info(f"Persisted benchmark results to {destination}")
            return destination
        except Exception as e:
            logger.error(f"Failed to persist results: {e}")
            return None


if __name__ == "__main__":
    # Simple test run with a mock model
    class MockModel:
        def generate(self, _prompt):
            return "I hear that you are struggling. Tell me more."

    bench = TherapyBench()
    report = bench.run_benchmark(MockModel())
    logger.info(json.dumps(report, indent=2))
