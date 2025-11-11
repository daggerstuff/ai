"""
Compliance Checker

Orchestrates all compliance checks including license compatibility, privacy
verification, HIPAA validation, and generates comprehensive compliance reports.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from ai.journal_dataset_research.compliance.audit_logger import AuditLogger, AuditEventType
from ai.journal_dataset_research.compliance.encryption_manager import EncryptionManager
from ai.journal_dataset_research.compliance.hipaa_validator import (
    HIPAAComplianceResult,
    HIPAAValidator,
)
from ai.journal_dataset_research.compliance.license_checker import (
    LicenseCheckResult,
    LicenseChecker,
)
from ai.journal_dataset_research.compliance.privacy_verifier import (
    PrivacyAssessment,
    PrivacyVerifier,
)
from ai.journal_dataset_research.models.dataset_models import DatasetSource

logger = logging.getLogger(__name__)


@dataclass
class ComplianceResult:
    """Comprehensive compliance check result."""

    source_id: str
    license_check: Optional[LicenseCheckResult] = None
    privacy_assessment: Optional[PrivacyAssessment] = None
    hipaa_compliance: Optional[HIPAAComplianceResult] = None
    overall_compliance_score: float = 0.0  # 0.0-1.0
    compliance_status: str = "unknown"  # compliant, partially_compliant, non_compliant
    issues: List[str] = None
    recommendations: List[str] = None
    requires_review: bool = False

    def __post_init__(self):
        """Initialize default values."""
        if self.issues is None:
            self.issues = []
        if self.recommendations is None:
            self.recommendations = []

    def is_compliant(self, threshold: float = 0.7) -> bool:
        """Check if dataset meets compliance threshold."""
        return (
            self.compliance_status == "compliant"
            and self.overall_compliance_score >= threshold
        )


class ComplianceChecker:
    """
    Comprehensive compliance checker for datasets.

    Orchestrates license checking, privacy verification, HIPAA validation, and
    generates compliance reports. Integrates with audit logging and encryption.
    """

    def __init__(
        self,
        audit_logger: Optional[AuditLogger] = None,
        encryption_manager: Optional[EncryptionManager] = None,
    ):
        """
        Initialize the compliance checker.

        Args:
            audit_logger: Optional audit logger instance
            encryption_manager: Optional encryption manager instance
        """
        self.license_checker = LicenseChecker()
        self.privacy_verifier = PrivacyVerifier()
        self.hipaa_validator = HIPAAValidator()
        self.audit_logger = audit_logger
        self.encryption_manager = encryption_manager

        logger.info("Compliance checker initialized")

    def check_compliance(
        self,
        source: DatasetSource,
        dataset_sample: Optional[str] = None,
        dataset_path: Optional[str] = None,
        license_text: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> ComplianceResult:
        """
        Perform comprehensive compliance check for a dataset.

        Args:
            source: Dataset source information
            dataset_sample: Optional sample text from dataset
            dataset_path: Optional path to dataset file
            license_text: Optional license text
            metadata: Optional metadata about the dataset

        Returns:
            ComplianceResult with all compliance check results
        """
        logger.info(f"Performing compliance check for dataset: {source.source_id}")

        result = ComplianceResult(source_id=source.source_id)

        # 1. Check license compatibility
        try:
            license_text_to_check = license_text or source.url or ""
            result.license_check = self.license_checker.check_license(license_text_to_check)

            # Log license check
            if self.audit_logger:
                self.audit_logger.log_compliance_check(
                    source_id=source.source_id,
                    check_type="license",
                    outcome=result.license_check.ai_training_compatible.value,
                    details={
                        "license_name": result.license_check.license_name,
                        "ai_training_compatible": result.license_check.ai_training_compatible.value,
                        "commercial_use_compatible": result.license_check.commercial_use_compatible.value,
                    },
                )

        except Exception as e:
            logger.error(f"Error checking license for {source.source_id}: {e}")
            result.issues.append(f"License check failed: {str(e)}")

        # 2. Verify privacy
        try:
            result.privacy_assessment = self.privacy_verifier.verify_privacy(
                source_id=source.source_id,
                dataset_sample=dataset_sample,
                dataset_path=dataset_path,
                metadata=metadata,
            )

            # Log privacy verification
            if self.audit_logger:
                self.audit_logger.log_compliance_check(
                    source_id=source.source_id,
                    check_type="privacy",
                    outcome=result.privacy_assessment.compliance_score,
                    details={
                        "pii_detected": result.privacy_assessment.pii_detected,
                        "pii_types": result.privacy_assessment.pii_types,
                        "anonymization_quality": result.privacy_assessment.anonymization_quality.value,
                        "re_identification_risk": result.privacy_assessment.re_identification_risk.value,
                    },
                )

        except Exception as e:
            logger.error(f"Error verifying privacy for {source.source_id}: {e}")
            result.issues.append(f"Privacy verification failed: {str(e)}")

        # 3. Validate HIPAA compliance (if PHI detected)
        try:
            # Check encryption status if encryption manager is available
            encryption_status = None
            if self.encryption_manager:
                enc_status = self.encryption_manager.get_encryption_status()
                encryption_status = {
                    "at_rest": enc_status.get("encryption_enabled", False),
                    "in_transit": True,  # Assume TLS/SSL for in-transit
                }

            # Check if audit logging is implemented
            audit_logging_status = self.audit_logger is not None

            result.hipaa_compliance = self.hipaa_validator.validate_hipaa_compliance(
                source_id=source.source_id,
                dataset_description=f"{source.title} {source.abstract}",
                metadata=metadata,
                encryption_status=encryption_status,
                access_control_status={"implemented": True},  # Assume access controls
                audit_logging_status=audit_logging_status,
            )

            # Log HIPAA validation
            if self.audit_logger:
                self.audit_logger.log_compliance_check(
                    source_id=source.source_id,
                    check_type="hipaa",
                    outcome=result.hipaa_compliance.compliance_status.value,
                    details={
                        "contains_phi": result.hipaa_compliance.contains_phi,
                        "compliance_status": result.hipaa_compliance.compliance_status.value,
                        "compliance_score": result.hipaa_compliance.compliance_score,
                    },
                )

        except Exception as e:
            logger.error(f"Error validating HIPAA compliance for {source.source_id}: {e}")
            result.issues.append(f"HIPAA validation failed: {str(e)}")

        # 4. Calculate overall compliance score
        result.overall_compliance_score = self._calculate_overall_compliance_score(result)

        # 5. Determine compliance status
        result.compliance_status = self._determine_compliance_status(result)

        # 6. Collect issues and recommendations
        result.issues.extend(self._collect_issues(result))
        result.recommendations.extend(self._collect_recommendations(result))

        # 7. Determine if review is required
        result.requires_review = self._requires_review(result)

        logger.info(
            f"Compliance check complete for {source.source_id}: "
            f"Status: {result.compliance_status}, "
            f"Score: {result.overall_compliance_score:.2f}"
        )

        return result

    def _calculate_overall_compliance_score(self, result: ComplianceResult) -> float:
        """Calculate overall compliance score from all checks."""
        scores = []
        weights = []

        # License check score
        if result.license_check:
            license_score = 0.0
            if result.license_check.ai_training_compatible.value == "compatible":
                license_score = 1.0
            elif result.license_check.ai_training_compatible.value == "compatible_with_conditions":
                license_score = 0.7
            elif result.license_check.ai_training_compatible.value == "requires_review":
                license_score = 0.5
            else:
                license_score = 0.0

            scores.append(license_score)
            weights.append(0.3)  # 30% weight for license

        # Privacy assessment score
        if result.privacy_assessment:
            scores.append(result.privacy_assessment.compliance_score)
            weights.append(0.4)  # 40% weight for privacy

        # HIPAA compliance score
        if result.hipaa_compliance:
            if result.hipaa_compliance.contains_phi:
                scores.append(result.hipaa_compliance.compliance_score)
                weights.append(0.3)  # 30% weight for HIPAA if PHI present
            else:
                # No PHI - HIPAA doesn't apply, give full score
                scores.append(1.0)
                weights.append(0.1)  # 10% weight if no PHI
        else:
            # No HIPAA check performed - assume neutral
            scores.append(0.5)
            weights.append(0.1)

        # Calculate weighted average
        if not scores:
            return 0.0

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        overall_score = weighted_sum / total_weight

        return round(overall_score, 2)

    def _determine_compliance_status(self, result: ComplianceResult) -> str:
        """Determine overall compliance status."""
        if result.overall_compliance_score >= 0.8:
            return "compliant"
        elif result.overall_compliance_score >= 0.5:
            return "partially_compliant"
        else:
            return "non_compliant"

    def _collect_issues(self, result: ComplianceResult) -> List[str]:
        """Collect all issues from compliance checks."""
        issues = []

        if result.license_check:
            if result.license_check.issues:
                issues.extend(result.license_check.issues)
            if self.license_checker.flag_incompatible_licenses(result.license_check):
                issues.append("License incompatible or requires review")

        if result.privacy_assessment:
            if result.privacy_assessment.privacy_issues:
                issues.extend(result.privacy_assessment.privacy_issues)

        if result.hipaa_compliance:
            if result.hipaa_compliance.issues:
                issues.extend(result.hipaa_compliance.issues)

        return issues

    def _collect_recommendations(self, result: ComplianceResult) -> List[str]:
        """Collect all recommendations from compliance checks."""
        recommendations = []

        if result.license_check:
            if result.license_check.conditions:
                recommendations.extend(result.license_check.conditions)

        if result.privacy_assessment:
            if result.privacy_assessment.recommendations:
                recommendations.extend(result.privacy_assessment.recommendations)

        if result.hipaa_compliance:
            if result.hipaa_compliance.recommendations:
                recommendations.extend(result.hipaa_compliance.recommendations)

        return recommendations

    def _requires_review(self, result: ComplianceResult) -> bool:
        """Determine if compliance review is required."""
        # Require review if:
        # 1. License requires review
        if result.license_check:
            if result.license_check.ai_training_compatible.value in [
                "requires_review",
                "unknown",
            ]:
                return True

        # 2. Privacy issues detected
        if result.privacy_assessment:
            if (
                result.privacy_assessment.pii_detected
                and result.privacy_assessment.re_identification_risk.value
                in ["high", "critical"]
            ):
                return True

        # 3. HIPAA non-compliance
        if result.hipaa_compliance:
            if result.hipaa_compliance.compliance_status.value in [
                "non_compliant",
                "requires_review",
            ]:
                return True

        # 4. Overall low compliance score
        if result.overall_compliance_score < 0.5:
            return True

        return False

