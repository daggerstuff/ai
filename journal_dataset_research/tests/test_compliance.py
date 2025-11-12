"""
Tests for compliance and security module.

Tests license checking, privacy verification, HIPAA validation, audit logging,
and encryption functionality.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from ai.journal_dataset_research.compliance.audit_logger import (
    AuditEventType,
    AuditLogger,
)
from ai.journal_dataset_research.compliance.compliance_checker import (
    ComplianceChecker,
)
from ai.journal_dataset_research.compliance.encryption_manager import (
    EncryptionManager,
)
from ai.journal_dataset_research.compliance.hipaa_validator import (
    HIPAAComplianceStatus,
    HIPAAValidator,
)
from ai.journal_dataset_research.compliance.license_checker import (
    LicenseChecker,
    LicenseCompatibility,
)
from ai.journal_dataset_research.compliance.privacy_verifier import (
    AnonymizationQuality,
    PrivacyRiskLevel,
    PrivacyVerifier,
)
from ai.journal_dataset_research.models.dataset_models import DatasetSource


class TestLicenseChecker:
    """Tests for license checker."""

    def test_mit_license(self):
        """Test MIT license detection and compatibility."""
        checker = LicenseChecker()
        result = checker.check_license("MIT License")

        assert result.license_name == "MIT"
        assert result.ai_training_compatible == LicenseCompatibility.COMPATIBLE
        assert result.commercial_use_compatible == LicenseCompatibility.COMPATIBLE
        assert result.is_usable()

    def test_cc_by_nc_license(self):
        """Test CC-BY-NC license (non-commercial)."""
        checker = LicenseChecker()
        result = checker.check_license("CC-BY-NC 4.0")

        assert result.license_name == "CC-BY-NC"
        assert result.ai_training_compatible == LicenseCompatibility.INCOMPATIBLE
        assert result.commercial_use_compatible == LicenseCompatibility.INCOMPATIBLE
        assert not result.is_usable()

    def test_gpl_license(self):
        """Test GPL license (requires review)."""
        checker = LicenseChecker()
        result = checker.check_license("GPL-3.0")

        assert result.license_name == "GPL-3.0"
        assert result.ai_training_compatible == LicenseCompatibility.REQUIRES_REVIEW
        assert checker.flag_incompatible_licenses(result)

    def test_unknown_license(self):
        """Test unknown license handling."""
        checker = LicenseChecker()
        result = checker.check_license("Custom License Agreement")

        assert result.license_name == "Unknown"
        assert result.ai_training_compatible in [
            LicenseCompatibility.UNKNOWN,
            LicenseCompatibility.REQUIRES_REVIEW,
        ]

    def test_ai_training_restriction(self):
        """Test detection of AI training restrictions."""
        checker = LicenseChecker()
        result = checker.check_license("License prohibits AI training and ML use")

        assert result.ai_training_compatible == LicenseCompatibility.INCOMPATIBLE
        issues_text = " ".join(result.issues).lower()
        assert "ai training" in issues_text or "training" in issues_text


class TestPrivacyVerifier:
    """Tests for privacy verifier."""

    def test_no_pii(self):
        """Test dataset with no PII."""
        verifier = PrivacyVerifier()
        assessment = verifier.verify_privacy(
            source_id="test-1",
            dataset_sample="This is a sample therapeutic conversation about general topics.",
        )

        assert not assessment.pii_detected
        assert len(assessment.pii_types) == 0
        assert assessment.anonymization_quality in [
            AnonymizationQuality.EXCELLENT,
            AnonymizationQuality.GOOD,
        ]
        assert assessment.re_identification_risk == PrivacyRiskLevel.LOW

    def test_email_detection(self):
        """Test email address detection."""
        verifier = PrivacyVerifier()
        assessment = verifier.verify_privacy(
            source_id="test-2",
            dataset_sample="Contact patient at john.doe@example.com for follow-up.",
        )

        assert assessment.pii_detected
        assert "email" in assessment.pii_types
        assert assessment.pii_count > 0

    def test_phone_detection(self):
        """Test phone number detection."""
        verifier = PrivacyVerifier()
        assessment = verifier.verify_privacy(
            source_id="test-3",
            dataset_sample="Patient phone: (555) 123-4567",
        )

        assert assessment.pii_detected
        assert "phone" in assessment.pii_types

    def test_medical_id_detection(self):
        """Test medical identifier detection."""
        verifier = PrivacyVerifier()
        assessment = verifier.verify_privacy(
            source_id="test-4",
            dataset_sample="Patient MRN: 12345678",
        )

        assert assessment.pii_detected
        assert "medical_id" in assessment.pii_types
        assert assessment.re_identification_risk in [
            PrivacyRiskLevel.HIGH,
            PrivacyRiskLevel.CRITICAL,
        ]

    def test_anonymization_quality(self):
        """Test anonymization quality assessment."""
        verifier = PrivacyVerifier()
        assessment = verifier.verify_privacy(
            source_id="test-5",
            dataset_sample="Patient [redacted] reported symptoms. Client P1 mentioned concerns.",
        )

        # Should detect anonymization indicators
        assert assessment.anonymization_quality in [
            AnonymizationQuality.EXCELLENT,
            AnonymizationQuality.GOOD,
        ]


class TestHIPAAValidator:
    """Tests for HIPAA validator."""

    def test_no_phi(self):
        """Test dataset with no PHI."""
        validator = HIPAAValidator()
        result = validator.validate_hipaa_compliance(
            source_id="test-1",
            dataset_description="General therapeutic techniques and approaches",
        )

        assert not result.contains_phi
        assert result.compliance_status == HIPAAComplianceStatus.NOT_APPLICABLE

    def test_phi_detection(self):
        """Test PHI detection."""
        validator = HIPAAValidator()
        result = validator.validate_hipaa_compliance(
            source_id="test-2",
            dataset_description="Patient therapy session transcripts with medical records",
        )

        assert result.contains_phi
        assert result.compliance_status != HIPAAComplianceStatus.NOT_APPLICABLE

    def test_encryption_requirements(self):
        """Test encryption requirement checking."""
        validator = HIPAAValidator()
        result = validator.validate_hipaa_compliance(
            source_id="test-3",
            dataset_description="Clinical trial data with patient information",
            encryption_status={"at_rest": True, "in_transit": True},
            access_control_status={"implemented": True},
            audit_logging_status=True,
        )

        assert result.contains_phi
        assert result.encryption_implemented
        assert result.access_controls_implemented
        assert result.audit_logging_implemented


class TestAuditLogger:
    """Tests for audit logger."""

    def test_log_event(self):
        """Test logging an event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_directory=tmpdir, enable_hash_chain=True)
            entry = logger.log_event(
                event_type=AuditEventType.DATASET_ACCESS,
                source_id="test-1",
                user_id="user-1",
                action="access",
                outcome="success",
            )

            assert entry.source_id == "test-1"
            assert entry.user_id == "user-1"
            assert entry.entry_hash is not None

    def test_log_dataset_access(self):
        """Test logging dataset access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_directory=tmpdir)
            entry = logger.log_dataset_access(
                source_id="test-1",
                user_id="user-1",
                action="download",
                outcome="success",
            )

            assert entry.event_type == AuditEventType.DATASET_ACCESS
            assert entry.source_id == "test-1"

    def test_query_logs(self):
        """Test querying logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_directory=tmpdir)
            logger.log_event(
                event_type=AuditEventType.DATASET_ACCESS,
                source_id="test-1",
                action="access",
            )
            logger.log_event(
                event_type=AuditEventType.DATASET_DOWNLOAD,
                source_id="test-2",
                action="download",
            )

            entries = logger.query_logs(source_id="test-1")
            assert len(entries) >= 1
            assert all(entry.source_id == "test-1" for entry in entries)

    def test_verify_log_integrity(self):
        """Test log integrity verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(log_directory=tmpdir, enable_hash_chain=True)
            logger.log_event(
                event_type=AuditEventType.DATASET_ACCESS,
                source_id="test-1",
                action="access",
            )

            results = logger.verify_log_integrity()
            assert results["integrity_check_enabled"]
            assert results["hash_chain_valid"]


class TestEncryptionManager:
    """Tests for encryption manager."""

    def test_encrypt_decrypt_string(self):
        """Test string encryption and decryption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EncryptionManager(key_directory=tmpdir)
            original = "This is a test string"
            encrypted = manager.encrypt_string(original)
            decrypted = manager.decrypt_string(encrypted)

            assert encrypted != original
            assert decrypted == original

    def test_encrypt_decrypt_file(self):
        """Test file encryption and decryption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EncryptionManager(key_directory=tmpdir)

            # Create test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("This is a test file")

            # Encrypt
            encrypted_file = manager.encrypt_file(str(test_file))
            assert Path(encrypted_file).exists()
            assert encrypted_file != str(test_file)

            # Decrypt
            decrypted_file = manager.decrypt_file(encrypted_file)
            assert Path(decrypted_file).exists()
            assert Path(decrypted_file).read_text() == "This is a test file"

    def test_encrypt_decrypt_data(self):
        """Test data encryption and decryption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EncryptionManager(key_directory=tmpdir)
            original = b"This is test binary data"
            encrypted = manager.encrypt_data(original)
            decrypted = manager.decrypt_data(encrypted)

            assert encrypted != original
            assert decrypted == original

    def test_generate_key_pair(self):
        """Test RSA key pair generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EncryptionManager(key_directory=tmpdir)
            private_key, public_key = manager.generate_key_pair()

            assert private_key is not None
            assert public_key is not None
            assert b"PRIVATE KEY" in private_key
            assert b"PUBLIC KEY" in public_key


class TestComplianceChecker:
    """Tests for compliance checker."""

    def test_check_compliance_basic(self):
        """Test basic compliance check."""
        source = DatasetSource(
            source_id="test-1",
            title="Test Dataset",
            authors=["Author 1"],
            publication_date=datetime.now(),
            source_type="journal",
            url="https://example.com/dataset",
            open_access=True,
            data_availability="available",
        )

        checker = ComplianceChecker()
        result = checker.check_compliance(source=source)

        assert result.source_id == "test-1"
        assert result.license_check is not None
        assert result.privacy_assessment is not None
        assert result.hipaa_compliance is not None
        assert 0.0 <= result.overall_compliance_score <= 1.0

    def test_check_compliance_with_audit_logging(self):
        """Test compliance check with audit logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_logger = AuditLogger(log_directory=tmpdir)
            source = DatasetSource(
                source_id="test-2",
                title="Test Dataset",
                authors=["Author 1"],
                publication_date=datetime.now(),
                source_type="clinical_trial",
                url="https://example.com/dataset",
                abstract="Patient therapy session data",
            )

            checker = ComplianceChecker(audit_logger=audit_logger)
            result = checker.check_compliance(source=source)

            assert result.source_id == "test-2"
            # Verify audit logs were created
            logs = audit_logger.query_logs(source_id="test-2")
            assert len(logs) > 0

    def test_check_compliance_with_encryption(self):
        """Test compliance check with encryption manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            encryption_manager = EncryptionManager(key_directory=tmpdir)
            source = DatasetSource(
                source_id="test-3",
                title="Test Dataset",
                authors=["Author 1"],
                publication_date=datetime.now(),
                source_type="repository",
                url="https://example.com/dataset",
            )

            checker = ComplianceChecker(encryption_manager=encryption_manager)
            result = checker.check_compliance(source=source)

            assert result.source_id == "test-3"
            # Check that encryption status is considered
            if result.hipaa_compliance and result.hipaa_compliance.contains_phi:
                assert result.hipaa_compliance.encryption_implemented


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

