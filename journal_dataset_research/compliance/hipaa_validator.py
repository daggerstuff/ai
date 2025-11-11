"""
HIPAA Compliance Validator

Implements HIPAA compliance checking for datasets containing protected health
information (PHI). Validates encryption, access controls, and audit logging requirements.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HIPAAComplianceStatus(Enum):
    """HIPAA compliance status."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class HIPAAComplianceResult:
    """HIPAA compliance validation result."""

    source_id: str
    contains_phi: bool
    compliance_status: HIPAAComplianceStatus
    checklist_items: Dict[str, bool]  # Checklist item -> compliance status
    encryption_required: bool
    encryption_implemented: bool
    access_controls_implemented: bool
    audit_logging_implemented: bool
    issues: List[str]
    recommendations: List[str]
    compliance_score: float  # 0.0-1.0

    def is_compliant(self, threshold: float = 0.8) -> bool:
        """Check if dataset meets HIPAA compliance threshold."""
        return (
            self.compliance_status == HIPAAComplianceStatus.COMPLIANT
            and self.compliance_score >= threshold
        )


class HIPAAValidator:
    """
    Validates HIPAA compliance for datasets containing PHI.

    Checks encryption requirements, access controls, audit logging, and other
    HIPAA security and privacy requirements.
    """

    # PHI indicators
    PHI_INDICATORS = [
        "patient",
        "medical record",
        "health information",
        "clinical data",
        "therapy session",
        "counseling session",
        "mental health",
        "diagnosis",
        "treatment",
        "prescription",
        "symptoms",
        "medical history",
    ]

    # Required HIPAA checklist items
    REQUIRED_CHECKLIST_ITEMS = [
        "encryption_at_rest",
        "encryption_in_transit",
        "access_controls",
        "audit_logging",
        "data_backup",
        "secure_disposal",
        "business_associate_agreement",
        "risk_assessment",
        "incident_response_plan",
        "workforce_training",
    ]

    def __init__(self):
        """Initialize the HIPAA validator."""
        pass

    def validate_hipaa_compliance(
        self,
        source_id: str,
        dataset_description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        encryption_status: Optional[Dict] = None,
        access_control_status: Optional[Dict] = None,
        audit_logging_status: Optional[bool] = None,
    ) -> HIPAAComplianceResult:
        """
        Validate HIPAA compliance for a dataset.

        Args:
            source_id: Unique identifier for the dataset source
            dataset_description: Description of the dataset
            metadata: Optional metadata about the dataset
            encryption_status: Dict with encryption status (at_rest, in_transit)
            access_control_status: Dict with access control status
            audit_logging_status: Whether audit logging is implemented

        Returns:
            HIPAAComplianceResult with compliance validation results
        """
        logger.info(f"Validating HIPAA compliance for dataset: {source_id}")

        # Check if dataset contains PHI
        contains_phi = self._check_contains_phi(dataset_description, metadata)

        if not contains_phi:
            # No PHI - HIPAA may not apply
            return HIPAAComplianceResult(
                source_id=source_id,
                contains_phi=False,
                compliance_status=HIPAAComplianceStatus.NOT_APPLICABLE,
                checklist_items={},
                encryption_required=False,
                encryption_implemented=False,
                access_controls_implemented=False,
                audit_logging_implemented=False,
                issues=[],
                recommendations=["Dataset does not appear to contain PHI - HIPAA may not apply"],
                compliance_score=1.0,
            )

        # Dataset contains PHI - validate compliance
        checklist_items = self._check_compliance_checklist(
            encryption_status, access_control_status, audit_logging_status
        )

        # Determine compliance status
        compliance_status = self._determine_compliance_status(checklist_items)

        # Generate issues and recommendations
        issues = self._generate_issues(checklist_items, compliance_status)
        recommendations = self._generate_recommendations(
            checklist_items, compliance_status
        )

        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(checklist_items)

        result = HIPAAComplianceResult(
            source_id=source_id,
            contains_phi=True,
            compliance_status=compliance_status,
            checklist_items=checklist_items,
            encryption_required=True,
            encryption_implemented=checklist_items.get("encryption_at_rest", False)
            and checklist_items.get("encryption_in_transit", False),
            access_controls_implemented=checklist_items.get("access_controls", False),
            audit_logging_implemented=checklist_items.get("audit_logging", False),
            issues=issues,
            recommendations=recommendations,
            compliance_score=compliance_score,
        )

        logger.info(
            f"HIPAA validation complete for {source_id}: "
            f"Status: {result.compliance_status.value}, "
            f"Score: {result.compliance_score:.2f}"
        )

        return result

    def _check_contains_phi(
        self,
        dataset_description: Optional[str],
        metadata: Optional[Dict],
    ) -> bool:
        """Check if dataset contains PHI indicators."""
        text_to_check = ""
        if dataset_description:
            text_to_check = dataset_description.lower()
        if metadata:
            # Check metadata fields
            for key, value in metadata.items():
                if isinstance(value, str):
                    text_to_check += f" {value.lower()}"

        # Check for PHI indicators
        for indicator in self.PHI_INDICATORS:
            if indicator in text_to_check:
                return True

        return False

    def _check_compliance_checklist(
        self,
        encryption_status: Optional[Dict],
        access_control_status: Optional[Dict],
        audit_logging_status: Optional[bool],
    ) -> Dict[str, bool]:
        """Check HIPAA compliance checklist items."""
        checklist = {}

        # Encryption at rest
        checklist["encryption_at_rest"] = (
            encryption_status is not None
            and encryption_status.get("at_rest", False)
            if encryption_status
            else False
        )

        # Encryption in transit
        checklist["encryption_in_transit"] = (
            encryption_status is not None
            and encryption_status.get("in_transit", False)
            if encryption_status
            else False
        )

        # Access controls
        checklist["access_controls"] = (
            access_control_status is not None
            and access_control_status.get("implemented", False)
            if access_control_status
            else False
        )

        # Audit logging
        checklist["audit_logging"] = audit_logging_status is True

        # Other items - assume not implemented unless explicitly provided
        for item in self.REQUIRED_CHECKLIST_ITEMS:
            if item not in checklist:
                checklist[item] = False

        return checklist

    def _determine_compliance_status(
        self, checklist_items: Dict[str, bool]
    ) -> HIPAAComplianceStatus:
        """Determine overall HIPAA compliance status."""
        required_items = [
            "encryption_at_rest",
            "encryption_in_transit",
            "access_controls",
            "audit_logging",
        ]

        implemented_count = sum(
            1 for item in required_items if checklist_items.get(item, False)
        )
        total_required = len(required_items)

        if implemented_count == total_required:
            # Check other important items
            other_items = [
                "data_backup",
                "secure_disposal",
                "risk_assessment",
            ]
            other_implemented = sum(
                1 for item in other_items if checklist_items.get(item, False)
            )
            if other_implemented >= 2:
                return HIPAAComplianceStatus.COMPLIANT
            else:
                return HIPAAComplianceStatus.PARTIALLY_COMPLIANT
        elif implemented_count >= total_required * 0.5:
            return HIPAAComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return HIPAAComplianceStatus.NON_COMPLIANT

    def _generate_issues(
        self,
        checklist_items: Dict[str, bool],
        compliance_status: HIPAAComplianceStatus,
    ) -> List[str]:
        """Generate list of compliance issues."""
        issues = []

        if compliance_status == HIPAAComplianceStatus.NON_COMPLIANT:
            issues.append("Dataset does not meet HIPAA compliance requirements")

        # Check critical items
        if not checklist_items.get("encryption_at_rest", False):
            issues.append("Encryption at rest not implemented - required for PHI")

        if not checklist_items.get("encryption_in_transit", False):
            issues.append("Encryption in transit not implemented - required for PHI")

        if not checklist_items.get("access_controls", False):
            issues.append("Access controls not implemented - required for PHI")

        if not checklist_items.get("audit_logging", False):
            issues.append("Audit logging not implemented - required for PHI")

        # Check other important items
        if not checklist_items.get("data_backup", False):
            issues.append("Data backup procedures not documented")

        if not checklist_items.get("secure_disposal", False):
            issues.append("Secure disposal procedures not documented")

        if not checklist_items.get("risk_assessment", False):
            issues.append("Risk assessment not conducted")

        return issues

    def _generate_recommendations(
        self,
        checklist_items: Dict[str, bool],
        compliance_status: HIPAAComplianceStatus,
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []

        if compliance_status != HIPAAComplianceStatus.COMPLIANT:
            recommendations.append(
                "Implement all required HIPAA security and privacy safeguards"
            )

        if not checklist_items.get("encryption_at_rest", False):
            recommendations.append(
                "Implement encryption at rest for all PHI storage"
            )

        if not checklist_items.get("encryption_in_transit", False):
            recommendations.append(
                "Implement encryption in transit (TLS/SSL) for all PHI transmission"
            )

        if not checklist_items.get("access_controls", False):
            recommendations.append(
                "Implement access controls (authentication, authorization, role-based access)"
            )

        if not checklist_items.get("audit_logging", False):
            recommendations.append(
                "Implement comprehensive audit logging for all PHI access"
            )

        if not checklist_items.get("data_backup", False):
            recommendations.append(
                "Establish data backup and recovery procedures"
            )

        if not checklist_items.get("secure_disposal", False):
            recommendations.append(
                "Establish secure data disposal procedures"
            )

        if not checklist_items.get("risk_assessment", False):
            recommendations.append("Conduct comprehensive risk assessment")

        if not checklist_items.get("business_associate_agreement", False):
            recommendations.append(
                "Establish business associate agreements if using third-party services"
            )

        if not checklist_items.get("incident_response_plan", False):
            recommendations.append("Develop incident response plan")

        if not checklist_items.get("workforce_training", False):
            recommendations.append(
                "Provide HIPAA training to all workforce members"
            )

        return recommendations

    def _calculate_compliance_score(
        self, checklist_items: Dict[str, bool]
    ) -> float:
        """Calculate HIPAA compliance score (0.0-1.0)."""
        if not checklist_items:
            return 0.0

        # Weight critical items more heavily
        critical_items = [
            "encryption_at_rest",
            "encryption_in_transit",
            "access_controls",
            "audit_logging",
        ]
        critical_weight = 0.6
        other_weight = 0.4

        critical_score = sum(
            1 for item in critical_items if checklist_items.get(item, False)
        ) / len(critical_items)

        other_items = [
            item
            for item in self.REQUIRED_CHECKLIST_ITEMS
            if item not in critical_items
        ]
        other_score = (
            sum(1 for item in other_items if checklist_items.get(item, False))
            / len(other_items)
            if other_items
            else 0.0
        )

        compliance_score = (
            critical_score * critical_weight + other_score * other_weight
        )

        return compliance_score

