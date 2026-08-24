"""Tests for Mental-LLM IFT orchestrator."""

import tempfile

from ai.tools.utilities.platform.mental_ift_orchestrator import MentalHealthIFTOrchestrator


def test_orchestrator_initialization():
    with tempfile.TemporaryDirectory() as tmp:
        core = MentalHealthIFTOrchestrator(config={"output_dir": tmp})
        assert str(core.output_dir) == tmp
        assert core.stage_results == []


def test_dataset_curation_stage():
    with tempfile.TemporaryDirectory() as tmp:
        core = MentalHealthIFTOrchestrator(config={"output_dir": tmp, "min_examples": 100})
        core._run_stage("dataset_curation", core._curate_dataset)
        assert core.stage_results[0].status == "success"
        assert core.stage_results[0].metrics["total_examples"] >= 100


def test_bias_audit_stage():
    with tempfile.TemporaryDirectory() as tmp:
        core = MentalHealthIFTOrchestrator(config={"output_dir": tmp})
        core._run_stage("bias_audit", core._run_bias_audit)
        assert core.stage_results[0].status == "success"
        assert "max_disparity" in core.stage_results[0].metrics


def test_comparison_study_stage():
    with tempfile.TemporaryDirectory() as tmp:
        core = MentalHealthIFTOrchestrator(config={"output_dir": tmp})
        core._run_stage("comparison_study", core._run_comparison_study)
        assert core.stage_results[0].status == "success"
        assert "best_approach_per_task" in core.stage_results[0].metrics


def test_continuous_fine_tuning_skip():
    with tempfile.TemporaryDirectory() as tmp:
        core = MentalHealthIFTOrchestrator(config={"output_dir": tmp, "min_continuous_examples": 100})
        result = core.run_continuous_fine_tuning([{"therapist_approved": True}])
        assert result["status"] == "skipped"


def test_continuous_fine_tuning_run():
    with tempfile.TemporaryDirectory() as tmp:
        core = MentalHealthIFTOrchestrator(config={"output_dir": tmp, "min_continuous_examples": 2})
        feedback = [
            {"therapist_approved": True, "client": "I'm sad", "therapist": "I hear you."},
            {"therapist_approved": True, "client": "I'm anxious", "therapist": "That makes sense."},
        ]
        result = core.run_continuous_fine_tuning(feedback)
        assert result["status"] == "success"
        assert result["approved_examples"] == 2
