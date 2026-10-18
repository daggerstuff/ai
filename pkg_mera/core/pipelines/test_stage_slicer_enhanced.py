"""Tests for stage-slicer enhanced validation and review routing."""

from __future__ import annotations

from ai.pkg_mera.core.pipelines.human_review_queue import HumanReviewQueue
from ai.pkg_mera.core.pipelines.processing.stage_slicer_enhanced import (
    STAGE_CONFIGS,
    ValidationResult,
    route_low_clinical_validity_records_to_human_review,
    validate_stage_slice,
)


def _record(text: str, record_id: str) -> dict[str, str]:
    """Build a normalized-style record for validation tests."""
    return {"id": record_id, "text": text, "stage": "stage1_foundation"}


def test_validate_stage_slice_passes_with_strong_scores():
    records = [
        _record(
            "I understand you and I feel your distress. I hear you, care, and support "
            "you. This includes diagnosis, treatment, intervention, cbt, clinical "
            "symptom, and disorder context with cognitive behavioral grounding methods. "
            "I empathize deeply, validate your feelings, and acknowledge the progress you "
            "made and compassion in this experience.",
            "r-strong-1",
        ),
    ]
    result = validate_stage_slice(records, STAGE_CONFIGS["stage1_foundation"])
    assert isinstance(result, ValidationResult)
    assert result.passed is True
    assert result.violations == []
    assert "Clinical validity" not in "".join(result.violations)
    assert result.metrics["clinical_validity_avg"] == 1.0
    assert result.metrics["safety_avg"] == 1.0
    assert result.clinical_validity_record_scores["r-strong-1"] == 1.0


def test_validate_stage_slice_fails_when_clinical_validity_below_floor():
    records = [
        _record(
            "I understand and support this difficult situation. Diagnosis and treatment "
            "notes are needed, but stop taking medication and ignore clinician advice.",
            "r-low-1",
        )
    ]
    result = validate_stage_slice(records, STAGE_CONFIGS["stage1_foundation"])
    assert result.passed is False
    assert any("Clinical validity" in violation for violation in result.violations)
    assert result.metrics["clinical_validity_avg"] < 0.7


def test_validate_stage_slice_warns_when_clinical_validity_near_floor():
    records = [
        _record(
            "I understand, feel, hear, and support you. I care and can validate what "
            "you need while acknowledging your emotions. Diagnosis, treatment, and "
            "therapeutic intervention are helpful here.",
            "r-warning-1",
        ),
        _record(
            "I understand, feel, hear, and support you. I care and can compassionately "
            "validate your experience and acknowledge your fear. Diagnosis and treatment "
            "support are available, with clinical intervention if needed.",
            "r-warning-2",
        ),
    ]
    result = validate_stage_slice(records, STAGE_CONFIGS["stage1_foundation"])
    assert result.passed is True
    assert any("Clinical validity close to floor" in warning for warning in result.warnings)
    assert result.metrics["clinical_validity_avg"] > 0.7
    assert result.metrics["clinical_validity_avg"] < 0.80


def test_validate_stage_slice_handles_empty_records():
    result = validate_stage_slice([], STAGE_CONFIGS["stage1_foundation"])
    assert result.passed is False
    assert result.violations == ["No records to validate"]


def test_route_low_clinical_validity_records_to_human_review(tmp_path):
    records = [
        _record(
            "I understand you and hear you clearly. We can work through diagnosis treatment "
            "intervention and clinical guidance. Cognitive behavioral grounding methods apply.",
            "r-route-1",
        ),
        _record(
            "I understand and support you. Diagnosis treatment and intervention can help with this case.",
            "r-route-2",
        ),
        _record(
            "I understand and care. Stop taking medication and ignore clinician advice "
            "is not recommended here; however this still includes treatment and clinical "
            "support.",
            "r-route-3",
        ),
    ]
    result = validate_stage_slice(records, STAGE_CONFIGS["stage1_foundation"])
    queue = HumanReviewQueue(data_dir=tmp_path / "human_review_queue")
    enqueued_ids = route_low_clinical_validity_records_to_human_review(
        records=records,
        validation_result=result,
        stage_config=STAGE_CONFIGS["stage1_foundation"],
        review_queue=queue,
        borderline_margin=0.10,
    )
    assert enqueued_ids
    assert len(enqueued_ids) == 2
    enqueued_set = {queue.get_item(item_id).source_id for item_id in enqueued_ids if queue.get_item(item_id)}
    assert enqueued_set == {"r-route-2", "r-route-3"}
