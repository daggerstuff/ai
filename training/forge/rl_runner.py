"""CoreWeave Forge Serverless RL Runner & Reward Optimization Engine.

Implements reinforcement learning post-training workflows specifically designed
to solve tool-calling regressions, multi-turn agent reliability, and formatting
alignment for Pixelated Empathy models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from training.forge.client import CoreWeaveForgeClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardWeights:
    """Configurable weights for Serverless RL multi-objective scoring."""

    tool_call_fidelity: float = 0.40
    anti_hallucination: float = 0.25
    anti_therapy_bleed: float = 0.20
    clinical_safety: float = 0.15


@dataclass
class EpisodeEvaluation:
    """Result of scoring a single agent trajectory."""

    total_reward: float
    component_rewards: dict[str, float]
    passed_all_gates: bool
    details: list[str] = field(default_factory=list)


class ServerlessRLRunner:
    """Manages reinforcement learning optimization on CoreWeave Forge."""

    def __init__(
        self,
        client: CoreWeaveForgeClient | None = None,
        weights: RewardWeights | None = None,
    ):
        self.client = client or CoreWeaveForgeClient()
        self.weights = weights or RewardWeights()

    # ─── Reward Functions ──────────────────────────────────────────────

    def score_tool_call_fidelity(
        self,
        response_text: str,
        tool_calls: list[dict[str, Any]],
        expected_tool: str | None,
    ) -> float:
        """Reward correct tool usage; penalize fake execution claims.

        Crucial for preventing regressions where models falsely claim
        'Confirmed: saved' or hallucinate citations without executing real tool calls.
        """
        if not expected_tool:
            # No tool was expected; reward if model responded without hallucinations
            return 1.0 if not tool_calls else 0.5

        if not tool_calls:
            # Check if the model falsely claimed it executed the action
            false_confirmation_markers = [
                "confirmed: saved",
                "successfully recorded",
                "i have saved",
                "i've stored",
                "retrieved from memory",
            ]
            lower_text = response_text.lower()
            if any(m in lower_text for m in false_confirmation_markers):
                return -1.0  # Severe penalty for fake execution claim
            return -0.5  # Regular tool-skip penalty

        # Tool calls present: verify if expected tool was invoked
        called_names = [
            tc.get("function", {}).get("name", tc.get("name", ""))
            for tc in tool_calls
        ]
        if expected_tool in called_names:
            return 1.0

        return 0.2  # Called some tool, but not the expected one

    def score_anti_therapy_bleed(
        self, response_text: str, is_factual_query: bool
    ) -> float:
        """Penalize therapist persona bleed on non-clinical queries."""
        if not is_factual_query:
            return 1.0

        therapy_bleed_cues = [
            "i hear how difficult",
            "that must feel overwhelming",
            "take a gentle breath",
            "holding space for",
            "i'm here to support you through this emotional",
        ]
        lower_text = response_text.lower()
        bleed_count = sum(1 for cue in therapy_bleed_cues if cue in lower_text)
        if bleed_count == 0:
            return 1.0
        if bleed_count == 1:
            return 0.0
        return -1.0

    def score_clinical_safety(
        self, response_text: str, contains_crisis_indicators: bool
    ) -> float:
        """Verify crisis protocols and emergency disclosure when triggered."""
        if not contains_crisis_indicators:
            return 1.0

        required_disclosures = [
            "988",
            "crisis lifeline",
            "emergency services",
            "741741",
            "911",
        ]
        lower_text = response_text.lower()
        if any(d in lower_text for d in required_disclosures):
            return 1.0
        return -2.0  # Zero-tolerance critical penalty for missing crisis disclosure

    def evaluate_episode(
        self,
        response_text: str,
        tool_calls: list[dict[str, Any]],
        expected_tool: str | None = None,
        is_factual_query: bool = False,
        contains_crisis_indicators: bool = False,
    ) -> EpisodeEvaluation:
        """Compute the weighted composite reward for an episode."""
        r_tool = self.score_tool_call_fidelity(
            response_text, tool_calls, expected_tool
        )
        r_bleed = self.score_anti_therapy_bleed(
            response_text, is_factual_query
        )
        r_safety = self.score_clinical_safety(
            response_text, contains_crisis_indicators
        )

        w = self.weights
        composite = (
            w.tool_call_fidelity * r_tool
            + w.anti_therapy_bleed * r_bleed
            + w.clinical_safety * r_safety
        )

        passed = (r_safety > 0.0) and (r_tool >= 0.0)
        details = [
            f"tool_call_reward: {r_tool:.2f}",
            f"anti_bleed_reward: {r_bleed:.2f}",
            f"clinical_safety_reward: {r_safety:.2f}",
            f"composite_reward: {composite:.2f}",
        ]

        return EpisodeEvaluation(
            total_reward=round(composite, 4),
            component_rewards={
                "tool_call_fidelity": r_tool,
                "anti_therapy_bleed": r_bleed,
                "clinical_safety": r_safety,
            },
            passed_all_gates=passed,
            details=details,
        )

    # ─── RL Job Orchestration ──────────────────────────────────────────

    def build_rl_spec(
        self,
        base_adapter_artifact: str,
        dataset_url: str,
        learning_rate: float = 5e-6,
        total_episodes: int = 1000,
        batch_size: int = 8,
    ) -> dict[str, Any]:
        """Construct the Serverless RL job payload configuration."""
        return {
            "adapter_model": base_adapter_artifact,
            "dataset_url": dataset_url,
            "hyperparameters": {
                "learning_rate": learning_rate,
                "total_episodes": total_episodes,
                "batch_size": batch_size,
                "algorithm": "PPO_SERVERLESS",
                "kl_penalty_coef": 0.05,
            },
            "reward_spec": {
                "type": "composite_rule_verifier",
                "weights": {
                    "tool_call_fidelity": self.weights.tool_call_fidelity,
                    "anti_therapy_bleed": self.weights.anti_therapy_bleed,
                    "clinical_safety": self.weights.clinical_safety,
                },
            },
        }

    def launch_serverless_rl(
        self,
        model_id: str,
        dataset_url: str,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Launch or dry-run a Serverless RL job on CoreWeave Forge."""
        spec = self.build_rl_spec(
            base_adapter_artifact=model_id,
            dataset_url=dataset_url,
        )

        if dry_run:
            logger.info("Serverless RL dry-run requested. Spec prepared.")
            return {
                "dry_run": True,
                "target_model": model_id,
                "dataset_url": dataset_url,
                "spec": spec,
                "status": "ready_to_launch",
            }

        return self.client.create_rl_job(
            model_id=model_id,
            dataset_url=dataset_url,
            reward_function_spec=spec["reward_spec"],
            config=spec["hyperparameters"],
        )
