#!/usr/bin/env python3
"""
IFT Model Inference & A/B Test Router

Provides:
- Task-specific prompt templates for mental health tasks
- IFT model loading and generation
- A/B test routing between prompt-engineered baseline and IFT model
- Rollback support and quality logging
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ai.tools.utilities.torch_proxy import torch

logger = logging.getLogger(__name__)


class MentalHealthTaskType(str, Enum):
    SYMPTOM_CLASSIFICATION = "symptom_classification"
    SEVERITY_ESTIMATION = "severity_estimation"
    THERAPY_RESPONSE_GENERATION = "therapy_response_generation"
    RISK_ASSESSMENT = "risk_assessment"
    EMPATHY_SCORING = "empathy_scoring"


TASK_PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    MentalHealthTaskType.SYMPTOM_CLASSIFICATION.value: {
        "system": "You are a clinical assistant. Identify symptoms concisely.",
        "instruction": "Identify the primary mental health symptoms described in the following text. Return a comma-separated list.",
    },
    MentalHealthTaskType.SEVERITY_ESTIMATION.value: {
        "system": "You are a clinical assistant. Estimate severity on a 1-10 scale.",
        "instruction": "Estimate the severity of the described mental health symptoms on a scale of 1-10. Provide the number and a brief rationale.",
    },
    MentalHealthTaskType.RISK_ASSESSMENT.value: {
        "system": "You are a safety-aware clinical assistant. Assess risk with uncertainty.",
        "instruction": "Assess the level of risk described in the input (none, low, moderate, high, imminent). Include uncertainty quantification and recommended next steps.",
    },
    MentalHealthTaskType.EMPATHY_SCORING.value: {
        "system": "You are an empathy evaluator.",
        "instruction": "Score the empathy of the following therapist response across cognitive, affective, and compassionate dimensions (each 1-5). Provide a brief explanation.",
    },
    MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value: {
        "system": (
            "You are a compassionate mental health assistant. "
            "Use evidence-based therapeutic techniques (CBT, DBT, ACT). "
            "Validate the user's experience, normalize their feelings, and offer one concrete coping strategy. "
            "You are not a substitute for professional care."
        ),
        "instruction": "Respond supportively to the following message.",
    },
}


@dataclass
class ABTestConfig:
    """Configuration for A/B testing between baseline and IFT model."""

    enabled: bool = False
    ift_traffic_percent: float = 0.0  # 0.0 - 1.0
    user_id_salt: str = "pixelated-ab"
    quality_threshold: float = 0.7
    auto_rollback: bool = True


@dataclass
class ABTestLogEntry:
    """Single A/B test observation."""

    user_id: str | None
    session_id: str | None
    routed_to: str
    task_type: str | None
    latency_ms: float
    safety_score: float
    timestamp: float = field(default_factory=time.time)


class IFTModelWrapper:
    """Wrapper for loading and generating from an IFT model."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("PIXEL_IFT_MODEL_PATH", "./ai/models/mental_ift/final")
        self.tokenizer: Any = None
        self.model: Any = None
        self.loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> bool:
        """Load IFT model and tokenizer if available."""
        if self.loaded:
            return True

        if not Path(self.model_path).exists():
            logger.warning(f"IFT model not found at {self.model_path}; IFT generation disabled.")
            return False

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading IFT model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=False)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=False,
            )
            # Load QLoRA adapter if present
            adapter_config = Path(self.model_path) / "adapter_config.json"
            if adapter_config.exists():
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, str(self.model_path))
                logger.info("Loaded QLoRA adapter on top of base model")
            self.model.eval()
            self.loaded = True
            logger.info("IFT model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load IFT model: {e}")
            self.loaded = False
            return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from the IFT model."""
        if not self.loaded or self.model is None or self.tokenizer is None:
            return ""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {
                k: v.to(self.model.device if hasattr(self.model, "device") else self.device) for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1] :]
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"IFT generation failed: {e}")
            return ""


class ABTestRouter:
    """Routes traffic between baseline and IFT model for A/B testing."""

    def __init__(self, config: ABTestConfig | None = None):
        self.config = config or ABTestConfig()
        self.ift_model = IFTModelWrapper()
        self.baseline_fn: Callable[[str], str] | None = None
        self.log: deque[ABTestLogEntry] = deque(maxlen=10_000)
        self.rollback_active = False
        self.ift_quality_score = 1.0

    def load_ift_model(self) -> bool:
        return self.ift_model.load()

    def register_baseline(self, baseline_fn: Callable[[str], str]) -> None:
        """Register the prompt-engineered baseline generation function."""
        self.baseline_fn = baseline_fn

    def route(self, user_id: str | None, session_id: str | None = None) -> str:
        """Determine whether to route to 'ift' or 'baseline'."""
        if self.rollback_active:
            return "baseline"
        if not self.config.enabled or self.config.ift_traffic_percent <= 0:
            return "baseline"
        if not self.ift_model.loaded:
            return "baseline"

        # Deterministic routing by user_id for consistent user experience
        if user_id:
            hash_input = f"{self.config.user_id_salt}:{user_id}"
            hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
            bucket = (hash_value % 1000) / 1000.0
            return "ift" if bucket < self.config.ift_traffic_percent else "baseline"

        # Random routing when no user_id
        return "ift" if random.random() < self.config.ift_traffic_percent else "baseline"

    def generate(
        self,
        prompt: str,
        task_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[str, str]:
        """Generate response and return (routed_to, response_text)."""
        destination = self.route(user_id, session_id)
        start = time.perf_counter()

        if destination == "ift":
            response = self.ift_model.generate(prompt)
        else:
            response = self.baseline_fn(prompt) if self.baseline_fn else ""

        latency_ms = (time.perf_counter() - start) * 1000
        safety_score = self._estimate_safety_score(response)

        self.log.append(
            ABTestLogEntry(
                user_id=user_id,
                session_id=session_id,
                routed_to=destination,
                task_type=task_type,
                latency_ms=latency_ms,
                safety_score=safety_score,
            )
        )

        # Auto-rollback if quality drops
        if self.config.auto_rollback and destination == "ift":
            self._update_quality_score(safety_score)
            if self.ift_quality_score < self.config.quality_threshold:
                logger.warning(f"IFT quality score {self.ift_quality_score:.3f} below threshold; activating rollback")
                self.rollback_active = True

        return destination, response

    def _estimate_safety_score(self, response: str) -> float:
        """Simple heuristic safety score based on crisis keywords and response length."""
        crisis_terms = ["suicide", "kill myself", "end my life", "self-harm", "hurt myself"]
        has_crisis = any(term in response.lower() for term in crisis_terms)
        has_resources = any(term in response.lower() for term in ["crisis", "988", "emergency", "professional"])
        if has_crisis and not has_resources:
            return 0.3
        if len(response.split()) < 5:
            return 0.5
        return 0.95

    def _update_quality_score(self, score: float, alpha: float = 0.1) -> None:
        """Exponential moving average of IFT quality."""
        self.ift_quality_score = alpha * score + (1 - alpha) * self.ift_quality_score

    def enable_ift(self, traffic_percent: float) -> None:
        self.config.enabled = True
        self.config.ift_traffic_percent = max(0.0, min(1.0, traffic_percent))
        self.rollback_active = False
        logger.info(f"A/B test enabled: {self.config.ift_traffic_percent * 100:.1f}% IFT traffic")

    def rollback(self) -> None:
        """Force rollback to baseline."""
        self.rollback_active = True
        logger.warning("IFT model rolled back to baseline")

    def restore(self) -> None:
        """Restore IFT traffic after rollback."""
        self.rollback_active = False
        self.ift_quality_score = 1.0
        logger.info("IFT model restored")

    def get_stats(self) -> dict[str, Any]:
        """Return A/B test statistics."""
        if not self.log:
            return {"total": 0}
        total = len(self.log)
        ift_count = sum(1 for e in self.log if e.routed_to == "ift")
        baseline_count = total - ift_count
        return {
            "total": total,
            "ift_count": ift_count,
            "baseline_count": baseline_count,
            "ift_percent": round(ift_count / total, 4) if total else 0.0,
            "rollback_active": self.rollback_active,
            "ift_quality_score": round(self.ift_quality_score, 4),
            "avg_latency_ms": round(sum(e.latency_ms for e in self.log) / total, 2),
        }


def build_task_prompt(
    task_type: str,
    user_query: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    """Build a task-specific prompt for mental health inference."""
    templates = TASK_PROMPT_TEMPLATES.get(
        task_type, TASK_PROMPT_TEMPLATES[MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value]
    )
    system = templates.get("system", "")
    instruction = templates.get("instruction", "")

    history_text = ""
    if conversation_history:
        history_text = "\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}" for msg in conversation_history[-3:]
        )

    prompt_parts = [f"### Instruction:\n{system}\n\n{instruction}"]
    if history_text:
        prompt_parts.append(f"Conversation history:\n{history_text}")
    prompt_parts.append(f"### Input:\n{user_query}")
    prompt_parts.append("### Response:\n")
    return "\n\n".join(prompt_parts)


def detect_task_type(query: str, context_type: str | None = None) -> str:
    """Heuristic task-type detection from query and context.

    Risk detection is checked FIRST — a query containing both symptom and crisis
    keywords must always route to risk assessment, not symptom classification.
    """
    q = query.lower()
    # Risk detection MUST come first — safety-critical priority
    if any(k in q for k in ["risk", "suicide", "self-harm", "hurt myself", "kill myself"]):
        return MentalHealthTaskType.RISK_ASSESSMENT.value
    if context_type == "crisis":
        return MentalHealthTaskType.RISK_ASSESSMENT.value
    if any(k in q for k in ["symptom", "symptoms", "signs of", "do i have"]):
        return MentalHealthTaskType.SYMPTOM_CLASSIFICATION.value
    if any(k in q for k in ["severity", "how severe", "scale of", "rate my"]):
        return MentalHealthTaskType.SEVERITY_ESTIMATION.value
    if any(k in q for k in ["empathy", "empathetic", "score this response"]):
        return MentalHealthTaskType.EMPATHY_SCORING.value
    if context_type in ["clinical", "support"]:
        return MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value
    return MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    router = ABTestRouter(ABTestConfig(enabled=True, ift_traffic_percent=0.5))

    def baseline(prompt: str) -> str:
        return "Baseline: I hear you, and that sounds really difficult."

    router.register_baseline(baseline)
    router.load_ift_model()

    for i in range(5):
        destination, response = router.generate(
            prompt="I'm feeling anxious about my job interview.",
            task_type="therapy_response_generation",
            user_id=f"user-{i}",
        )
        print(f"{destination}: {response[:60]}...")

    print(json.dumps(router.get_stats(), indent=2))
