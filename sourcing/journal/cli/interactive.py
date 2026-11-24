"""Compatibility wrapper for the pipelines data-processing journal interactive prompts."""

from __future__ import annotations

from ai.pipelines.data_processing.journal.cli.interactive import (
    prompt_for_acquisition_approval,
    prompt_for_dataset_review,
    prompt_for_integration_approval,
    prompt_for_manual_evaluation_override,
    prompt_for_phase_transition,
    prompt_for_session_config,
)

__all__ = [
    "prompt_for_acquisition_approval",
    "prompt_for_dataset_review",
    "prompt_for_integration_approval",
    "prompt_for_manual_evaluation_override",
    "prompt_for_phase_transition",
    "prompt_for_session_config",
]
