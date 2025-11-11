"""
Compliance and Security Module for Journal Dataset Research System.

This module provides comprehensive compliance checking, privacy verification,
HIPAA validation, audit logging, and data encryption capabilities.
"""

from ai.journal_dataset_research.compliance.audit_logger import AuditLogger
from ai.journal_dataset_research.compliance.compliance_checker import (
    ComplianceChecker,
    ComplianceResult,
)
from ai.journal_dataset_research.compliance.encryption_manager import (
    EncryptionManager,
)
from ai.journal_dataset_research.compliance.hipaa_validator import (
    HIPAAValidator,
    HIPAAComplianceResult,
)
from ai.journal_dataset_research.compliance.license_checker import (
    LicenseChecker,
    LicenseCompatibility,
)
from ai.journal_dataset_research.compliance.privacy_verifier import (
    PrivacyVerifier,
    PrivacyAssessment,
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

