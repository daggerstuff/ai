"""Tests for IFT vs prompt engineering comparison study."""

<<<<<<< HEAD
<<<<<<< HEAD
from evals.ift_comparison import IFTComparisonStudy
from training.mental_health_instruction_dataset import MentalHealthTaskType
=======
from ai.evals.ift_comparison import IFTComparisonStudy
from ai.training.mental_health_instruction_dataset import MentalHealthTaskType
>>>>>>> 13c4a84d (feat(PIX-3911): implement Mental-LLM instruction fine-tuning pipeline)
=======
from evals.ift_comparison import IFTComparisonStudy
from training.mental_health_instruction_dataset import MentalHealthTaskType
>>>>>>> 6b3e88de (fix(PIX-3911): Phase 3 bug fixes — bias audit parsing, abs disparity, deque log, hallucination scoring, to_chat fields, test imports)


def dummy_inference(prompt: str) -> str:
    if "symptom" in prompt.lower():
        return "anxiety, low mood"
    if "severity" in prompt.lower():
        return "6"
    if "risk" in prompt.lower():
        return "moderate"
    if "empathy" in prompt.lower():
        return "4"
    return "I hear you, and that sounds really difficult."


def test_comparison_study():
    examples = [
        {
            "task_type": MentalHealthTaskType.SYMPTOM_CLASSIFICATION.value,
            "input": "I feel anxious and sad.",
            "output": "anxiety, low mood",
        },
        {
            "task_type": MentalHealthTaskType.SEVERITY_ESTIMATION.value,
            "input": "I can't function.",
            "output": "8",
        },
    ]

    study = IFTComparisonStudy(
        zero_shot_fn=dummy_inference,
        few_shot_fn=dummy_inference,
        ift_fn=dummy_inference,
    )
    report = study.evaluate(examples, model_name="test")
    assert len(report.results) == 6  # 2 tasks x 3 approaches
    assert report.best_approach_per_task()


def test_improvement_over_baseline():
    examples = [
        {
            "task_type": MentalHealthTaskType.SYMPTOM_CLASSIFICATION.value,
            "input": "I feel anxious.",
            "output": "anxiety",
        },
    ]

    def zero_shot(prompt: str) -> str:
        return "anxiety"

    def ift(prompt: str) -> str:
        return "anxiety, low mood"

    study = IFTComparisonStudy(zero_shot_fn=zero_shot, ift_fn=ift)
    report = study.evaluate(examples, model_name="test")
    improvements = report.improvement_over_baseline()
    assert MentalHealthTaskType.SYMPTOM_CLASSIFICATION.value in improvements
