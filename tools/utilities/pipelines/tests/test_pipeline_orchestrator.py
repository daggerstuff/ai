"""Tests for the pipeline orchestrator."""

import pytest

from ai.tools.utilities.pipelines.pipeline_orchestrator import PipelineOrchestrator


@pytest.fixture
def orchestrator() -> PipelineOrchestrator:
    return PipelineOrchestrator()


def test_the_default_pipeline_runs_its_stages(orchestrator: PipelineOrchestrator) -> None:
    """A payload flows through every registered stage and succeeds."""
    result = orchestrator.process({"text": "hello"})

    assert result.success is True
    assert result.errors == []
    assert len(result.stages) == len(orchestrator.stages)


def test_the_result_is_fingerprinted(orchestrator: PipelineOrchestrator) -> None:
    """Input and output hashes are recorded for the run."""
    result = orchestrator.process({"text": "hello"})

    assert result.input_hash
    assert result.output_hash
    assert result.started_at is not None
    assert result.finished_at is not None


def test_a_registered_stage_joins_the_pipeline(orchestrator: PipelineOrchestrator) -> None:
    """A stage added by the caller runs with the rest."""
    seen = []

    def record(payload):
        seen.append(payload)
        return payload

    orchestrator.register_stage("record", record)
    result = orchestrator.process({"text": "hello"})

    assert "record" in orchestrator.stages
    assert len(result.stages) == len(orchestrator.stages)
    assert seen


def test_a_failing_stage_is_reported_not_raised(orchestrator: PipelineOrchestrator) -> None:
    """A stage that raises is recorded as an error on the result."""

    def explode(_payload):
        raise RuntimeError("stage exploded")

    orchestrator.register_stage("explode", explode)
    result = orchestrator.process({"text": "hello"})

    assert result.success is False
    assert result.errors
