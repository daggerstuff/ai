"""
Compliance and Security Module for Journal Dataset Research System.

This module provides comprehensive compliance checking, privacy verification,
HIPAA validation, audit logging, and data encryption capabilities.
"""

from ai.core.sourcing.journal.compliance.audit_logger import AuditLogger
from ai.core.sourcing.journal.compliance.compliance_checker import (
    ComplianceChecker,
    ComplianceResult,
)
from ai.core.sourcing.journal.compliance.encryption_manager import (
    EncryptionManager,
)
from ai.core.sourcing.journal.compliance.hipaa_validator import (
    HIPAAComplianceResult,
    HIPAAValidator,
)
from ai.core.sourcing.journal.compliance.license_checker import (
    LicenseChecker,
    LicenseCompatibility,
)
from ai.core.sourcing.journal.compliance.privacy_verifier import (
    PrivacyAssessment,
    PrivacyVerifier,
)

__all__ = [
    "AuditLogger",
    "ComplianceChecker",
    "ComplianceResult",
    "EncryptionManager",
    "HIPAAValidator",
    "HIPAAComplianceResult",
    "LicenseChecker",
    "LicenseCompatibility",
    "PrivacyVerifier",
    "PrivacyAssessment",
]
