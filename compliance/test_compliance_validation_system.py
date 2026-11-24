"""Tests for the compliance validation system."""

import asyncio

import pytest

from ai.compliance.compliance_validation_system import (
    ComplianceControl,
    ComplianceStandard,
    ComplianceValidationSystem,
)


@pytest.fixture
def system() -> ComplianceValidationSystem:
    return ComplianceValidationSystem()


def test_registry_covers_every_standard(system: ComplianceValidationSystem) -> None:
    """Every control is registered and every standard is represented."""
    assert system.controls
    registered = {control.standard for control in system.controls.values()}
    assert registered
    assert registered.issubset(set(ComplianceStandard))
    for control_id, control in system.controls.items():
        assert isinstance(control, ComplianceControl)
        assert control_id


def test_assessment_reports_every_control(system: ComplianceValidationSystem) -> None:
    """A full assessment produces one detailed result per registered control."""
    report = asyncio.run(system.run_comprehensive_compliance_assessment())

    summary = report["compliance_assessment_summary"]
    assert 0.0 <= summary["overall_compliance_score"] <= 100.0
    assert isinstance(summary["production_ready"], bool)
    assert summary["certification_status"]

    assert len(report["detailed_assessments"]) == len(system.controls)
    assert report["compliance_metrics"]["total_evidence_collected"] >= 0


def test_assessment_metrics_partition_the_controls(system: ComplianceValidationSystem) -> None:
    """Compliant, partial and non-compliant counts partition the control set."""
    report = asyncio.run(system.run_comprehensive_compliance_assessment())

    metrics = report["compliance_metrics"]
    counted = (
        metrics["compliant_controls"] + metrics["partially_compliant_controls"] + metrics["non_compliant_controls"]
    )
    assert counted == len(system.controls)


def test_production_readiness_rejects_a_low_score(system: ComplianceValidationSystem) -> None:
    """Readiness is denied when the overall score falls below the threshold."""
    below_threshold = {
        "overall_score": 10.0,
        "standard_scores": {standard.value: 10.0 for standard in ComplianceStandard},
        "compliance_levels": {"non_compliant": 0},
    }
    assert system._determine_production_readiness(below_threshold) is False


def test_empty_registry_produces_an_empty_assessment(system: ComplianceValidationSystem) -> None:
    """An empty registry yields no detailed results instead of raising."""
    system.controls = {}
    report = asyncio.run(system.run_comprehensive_compliance_assessment())
    assert report["detailed_assessments"] == []
