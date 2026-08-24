"""
Compliance and Security Module for Journal Dataset Research System.

This module provides comprehensive compliance checking, privacy verification,
HIPAA validation, audit logging, and data encryption capabilities.
"""

from ai.pipelines.data_processing.journal.compliance.audit_logger import AuditLogger
from ai.pipelines.data_processing.journal.compliance.compliance_checker import (
    ComplianceChecker,
    ComplianceResult,
)
from ai.pipelines.data_processing.journal.compliance.encryption_manager import (
    EncryptionManager,
)
from ai.pipelines.data_processing.journal.compliance.hipaa_validator import (
    HIPAAComplianceResult,
    HIPAAValidator,
)
from ai.pipelines.data_processing.journal.compliance.license_checker import (
    LicenseChecker,
    LicenseCompatibility,
)
from ai.pipelines.data_processing.journal.compliance.privacy_verifier import (
    PrivacyAssessment,
    PrivacyVerifier,
)

__all__ = [
    "AuditLogger",
    "ComplianceChecker",
    "ComplianceResult",
    "EncryptionManager",
    "HIPAAComplianceResult",
    "HIPAAValidator",
    "LicenseChecker",
    "LicenseCompatibility",
    "PrivacyAssessment",
    "PrivacyVerifier",
]
