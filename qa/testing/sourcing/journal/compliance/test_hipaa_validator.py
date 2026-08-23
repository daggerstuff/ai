"""Tests for HIPAA compliance validator."""

import sys
from pathlib import Path

import pytest

# Import directly from the module file to avoid broken package __init__.py
_compliance_dir = Path(__file__).resolve().parents[4] / "sourcing" / "journal" / "compliance"
sys.path.insert(0, str(_compliance_dir))

from hipaa_validator import (  # noqa: E402
    HIPAAComplianceError,
    HIPAAComplianceStatus,
    HIPAAValidator,
)


@pytest.fixture
def validator():
    return HIPAAValidator()


@pytest.fixture
def phi_description():
    return "Dataset containing patient medical records and clinical data"


@pytest.fixture
def non_phi_description():
    return "Dataset containing weather sensor readings and temperature logs"


@pytest.fixture
def full_compliance_args():
    return {
        "encryption_status": {"at_rest": True, "in_transit": True},
        "access_control_status": {"implemented": True},
        "audit_logging_status": True,
    }


class TestPHIDetection:
    def test_detects_phi_from_description(self, validator, phi_description):
        result = validator.validate_hipaa_compliance("ds-1", phi_description)
        assert result.contains_phi is True

    def test_detects_phi_from_metadata(self, validator):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            metadata={"category": "Mental Health therapy sessions"},
        )
        assert result.contains_phi is True

    def test_no_phi_returns_not_applicable(self, validator, non_phi_description):
        result = validator.validate_hipaa_compliance("ds-1", non_phi_description)
        assert result.compliance_status == HIPAAComplianceStatus.NOT_APPLICABLE
        assert result.contains_phi is False
        assert result.encryption_required is False


class TestEncryptionAtRest:
    def test_encryption_at_rest_passes(self, validator, phi_description):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            phi_description,
            encryption_status={"at_rest": True, "in_transit": True},
            access_control_status={"implemented": True},
            audit_logging_status=True,
        )
        assert result.checklist_items["encryption_at_rest"] is True
        assert result.encryption_implemented is True

    def test_encryption_at_rest_fails(self, validator, phi_description):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            phi_description,
            encryption_status={"at_rest": False, "in_transit": True},
            access_control_status={"implemented": True},
            audit_logging_status=True,
        )
        assert result.checklist_items["encryption_at_rest"] is False
        assert result.encryption_implemented is False
        assert any("Encryption at rest not implemented" in i for i in result.issues)

    def test_missing_encryption_status_fails_closed(self, validator, phi_description):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            phi_description,
            access_control_status={"implemented": True},
            audit_logging_status=True,
        )
        assert result.checklist_items["encryption_at_rest"] is False
        assert result.checklist_items["encryption_in_transit"] is False


class TestEncryptionInTransit:
    def test_encryption_in_transit_passes(self, validator, phi_description):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            phi_description,
            encryption_status={"at_rest": True, "in_transit": True},
            access_control_status={"implemented": True},
            audit_logging_status=True,
        )
        assert result.checklist_items["encryption_in_transit"] is True

    def test_encryption_in_transit_fails(self, validator, phi_description):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            phi_description,
            encryption_status={"at_rest": True, "in_transit": False},
            access_control_status={"implemented": True},
            audit_logging_status=True,
        )
        assert result.checklist_items["encryption_in_transit"] is False
        assert any("Encryption in transit not implemented" in i for i in result.issues)


class TestAccessControls:
    def test_access_controls_pass(self, validator, phi_description):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            phi_description,
            encryption_status={"at_rest": True, "in_transit": True},
            access_control_status={"implemented": True},
            audit_logging_status=True,
        )
        assert result.checklist_items["access_controls"] is True
        assert result.access_controls_implemented is True

    def test_access_controls_fail(self, validator, phi_description):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            phi_description,
            encryption_status={"at_rest": True, "in_transit": True},
            access_control_status={"implemented": False},
            audit_logging_status=True,
        )
        assert result.checklist_items["access_controls"] is False
        assert result.access_controls_implemented is False
        assert any("Access controls not implemented" in i for i in result.issues)


class TestAuditLogging:
    def test_audit_logging_pass(self, validator, phi_description):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            phi_description,
            encryption_status={"at_rest": True, "in_transit": True},
            access_control_status={"implemented": True},
            audit_logging_status=True,
        )
        assert result.checklist_items["audit_logging"] is True
        assert result.audit_logging_implemented is True

    def test_audit_logging_fail(self, validator, phi_description):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            phi_description,
            encryption_status={"at_rest": True, "in_transit": True},
            access_control_status={"implemented": True},
            audit_logging_status=False,
        )
        assert result.checklist_items["audit_logging"] is False
        assert result.audit_logging_implemented is False
        assert any("Audit logging not implemented" in i for i in result.issues)


class TestComplianceStatus:
    def test_full_compliance(self, validator, phi_description, full_compliance_args):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            phi_description,
            **full_compliance_args,
        )
        assert result.compliance_status == HIPAAComplianceStatus.PARTIALLY_COMPLIANT
        assert result.compliance_score > 0.5

    def test_non_compliant_with_no_controls(self, validator, phi_description):
        result = validator.validate_hipaa_compliance("ds-1", phi_description)
        assert result.compliance_status == HIPAAComplianceStatus.NON_COMPLIANT
        assert result.compliance_score < 0.5

    def test_is_compliant_threshold(self, validator, phi_description):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            phi_description,
            encryption_status={"at_rest": True, "in_transit": True},
            access_control_status={"implemented": True},
            audit_logging_status=True,
        )
        assert result.is_compliant(threshold=0.5) is False


class TestFailClosed:
    def test_raises_on_non_compliant(self, validator, phi_description):
        with pytest.raises(HIPAAComplianceError) as exc_info:
            validator.validate_hipaa_compliance(
                "ds-1",
                phi_description,
                fail_closed=True,
            )
        assert "failed HIPAA compliance" in str(exc_info.value)

    def test_raises_on_partially_compliant(self, validator, phi_description, full_compliance_args):
        with pytest.raises(HIPAAComplianceError):
            validator.validate_hipaa_compliance(
                "ds-1",
                phi_description,
                fail_closed=True,
                **full_compliance_args,
            )

    def test_no_raise_when_not_applicable(self, validator, non_phi_description):
        result = validator.validate_hipaa_compliance(
            "ds-1",
            non_phi_description,
            fail_closed=True,
        )
        assert result.compliance_status == HIPAAComplianceStatus.NOT_APPLICABLE

    def test_error_message_includes_issues(self, validator, phi_description):
        with pytest.raises(HIPAAComplianceError) as exc_info:
            validator.validate_hipaa_compliance(
                "ds-1",
                phi_description,
                fail_closed=True,
            )
        msg = str(exc_info.value)
        assert "encryption" in msg.lower()
