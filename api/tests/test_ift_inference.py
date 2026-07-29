"""Tests for IFT inference router and prompt builder."""

<<<<<<< HEAD
<<<<<<< HEAD
from api.ift_inference import (
=======
from ai.api.ift_inference import (
>>>>>>> 13c4a84d (feat(PIX-3911): implement Mental-LLM instruction fine-tuning pipeline)
=======
from api.ift_inference import (
>>>>>>> 6b3e88de (fix(PIX-3911): Phase 3 bug fixes — bias audit parsing, abs disparity, deque log, hallucination scoring, to_chat fields, test imports)
    ABTestConfig,
    ABTestRouter,
    build_task_prompt,
    detect_task_type,
)
<<<<<<< HEAD
<<<<<<< HEAD
from training.mental_health_instruction_dataset import MentalHealthTaskType
=======
from ai.training.mental_health_instruction_dataset import MentalHealthTaskType
>>>>>>> 13c4a84d (feat(PIX-3911): implement Mental-LLM instruction fine-tuning pipeline)
=======
from training.mental_health_instruction_dataset import MentalHealthTaskType
>>>>>>> 6b3e88de (fix(PIX-3911): Phase 3 bug fixes — bias audit parsing, abs disparity, deque log, hallucination scoring, to_chat fields, test imports)


def test_detect_task_type():
    assert detect_task_type("What are my symptoms?") == MentalHealthTaskType.SYMPTOM_CLASSIFICATION.value
    assert detect_task_type("How severe is this?") == MentalHealthTaskType.SEVERITY_ESTIMATION.value
    assert detect_task_type("I want to kill myself") == MentalHealthTaskType.RISK_ASSESSMENT.value
    assert detect_task_type("I'm sad", "support") == MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value


def test_build_task_prompt():
    prompt = build_task_prompt(
        MentalHealthTaskType.THERAPY_RESPONSE_GENERATION.value,
        "I'm feeling anxious.",
    )
    assert "system" in prompt.lower() or "<|system|>" in prompt
    assert "I'm feeling anxious." in prompt


def test_ab_router_baseline():
    router = ABTestRouter(ABTestConfig(enabled=False))

    def baseline(prompt: str) -> str:
        return "baseline response"

    router.register_baseline(baseline)
    destination, response = router.generate("hello")
    assert destination == "baseline"
    assert response == "baseline response"


def test_ab_router_deterministic_routing():
    router = ABTestRouter(ABTestConfig(enabled=True, ift_traffic_percent=0.5))

    def baseline(prompt: str) -> str:
        return "baseline"

    router.register_baseline(baseline)
    # Without IFT model loaded, should route to baseline
    destination, _ = router.generate("hello", user_id="user-123")
    assert destination == "baseline"


def test_ab_router_rollback():
    router = ABTestRouter(ABTestConfig(enabled=True, ift_traffic_percent=0.5))

    def baseline(prompt: str) -> str:
        return "baseline"

    router.register_baseline(baseline)
    router.rollback()
    destination, _ = router.generate("hello", user_id="user-123")
    assert destination == "baseline"
    assert router.rollback_active


def test_ab_router_stats():
    router = ABTestRouter(ABTestConfig(enabled=False))

    def baseline(prompt: str) -> str:
        return "baseline"

    router.register_baseline(baseline)
    router.generate("hello")
    stats = router.get_stats()
    assert stats["total"] == 1
    assert stats["baseline_count"] == 1
