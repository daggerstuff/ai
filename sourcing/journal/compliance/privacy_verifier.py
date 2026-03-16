"""
Privacy Verification Module

Implements PII detection, anonymization quality assessment, and re-identification
risk assessment for therapeutic datasets.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PrivacyRiskLevel(Enum):
    """Privacy risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AnonymizationQuality(Enum):
    """Anonymization quality levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    NONE = "none"


@dataclass
class PrivacyAssessment:
    """Privacy assessment result for a dataset."""

    source_id: str
    pii_detected: bool
    pii_types: List[str]  # email, phone, ssn, name, address, etc.
    pii_count: int
    anonymization_quality: AnonymizationQuality
    re_identification_risk: PrivacyRiskLevel
    privacy_issues: List[str]
    recommendations: List[str]
    compliance_score: float  # 0.0-1.0

    def is_compliant(self, threshold: float = 0.7) -> bool:
        """Check if dataset meets privacy compliance threshold."""
        return self.compliance_score >= threshold


class PrivacyVerifier:
    """
    Verifies privacy and anonymization quality of datasets.

    Detects PII, assesses anonymization quality, and evaluates re-identification risks.
    """

    # PII patterns
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )
    PHONE_PATTERN = re.compile(
        r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b")
    CREDIT_CARD_PATTERN = re.compile(
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
    )
    IP_ADDRESS_PATTERN = re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    )
    DATE_PATTERN = re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"
    )

    # Common name patterns (first/last names)
    NAME_PATTERNS = [
        r"\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
        r"\b(?:patient|client|subject)\s+[A-Z][a-z]+\b",
    ]

    # Address patterns
    ADDRESS_PATTERNS = [
        r"\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b",
        r"\b(?:PO\s+Box|P\.O\.\s+Box)\s+\d+\b",
    ]

    # Medical identifiers
    MEDICAL_ID_PATTERNS = [
        r"\b(?:MRN|Medical\s+Record\s+Number)[:]\s*\d+\b",
        r"\b(?:Patient\s+ID|PAT\s+ID)[:]\s*\w+\b",
        r"\b(?:Case\s+Number|Case\s+#)[:]\s*\d+\b",
    ]

    # Anonymization indicators
    ANONYMIZATION_PATTERNS = [
        r"\b(?:redacted|\[redacted\]|\[removed\]|\[deleted\])\b",
        r"\b(?:xxx|XXX|\*{3,})\b",
        r"\b(?:patient\s+[A-Z]|client\s+[A-Z]|subject\s+[A-Z])\b",  # Generic identifiers
        r"\b(?:p\d+|c\d+|s\d+)\b",  # Pattern like P1, C2, S3
    ]

    def __init__(self, sample_size: int = 1000):
        """
        Initialize the privacy verifier.

        Args:
            sample_size: Number of samples to analyze for PII detection
        """
        self.sample_size = sample_size
        self._compiled_name_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.NAME_PATTERNS
        ]
        self._compiled_address_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.ADDRESS_PATTERNS
        ]
        self._compiled_medical_id_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.MEDICAL_ID_PATTERNS
        ]
        self._compiled_anonymization_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.ANONYMIZATION_PATTERNS
        ]

    def verify_privacy(
        self,
        source_id: str,
        dataset_sample: Optional[str] = None,
        dataset_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> PrivacyAssessment:
        """
        Verify privacy and anonymization quality of a dataset.

        Args:
            source_id: Unique identifier for the dataset source
            dataset_sample: Sample text from the dataset
            dataset_path: Path to dataset file (for file-based analysis)
            metadata: Optional metadata about the dataset

        Returns:
            PrivacyAssessment with privacy verification results
        """
        logger.info(f"Verifying privacy for dataset: {source_id}")

        # Collect text to analyze
        text_to_analyze = ""
        if dataset_sample:
            text_to_analyze = dataset_sample
        elif dataset_path:
            # In a real implementation, would read and sample from file
            text_to_analyze = self._sample_from_file(dataset_path)

        if not text_to_analyze:
            # No data to analyze - return unknown assessment
            return PrivacyAssessment(
                source_id=source_id,
                pii_detected=False,
                pii_types=[],
                pii_count=0,
                anonymization_quality=AnonymizationQuality.NONE,
                re_identification_risk=PrivacyRiskLevel.UNKNOWN,
                privacy_issues=["No dataset sample available for analysis"],
                recommendations=["Provide dataset sample for privacy analysis"],
                compliance_score=0.0,
            )

        # Detect PII
        pii_types, pii_count = self._detect_pii(text_to_analyze)

        # Assess anonymization quality
        anonymization_quality = self._assess_anonymization_quality(
            text_to_analyze, pii_types
        )

        # Assess re-identification risk
        re_identification_risk = self._assess_re_identification_risk(
            text_to_analyze, pii_types, anonymization_quality
        )

        # Generate issues and recommendations
        privacy_issues = self._generate_privacy_issues(
            pii_types, pii_count, anonymization_quality, re_identification_risk
        )
        recommendations = self._generate_recommendations(
            pii_types, anonymization_quality, re_identification_risk
        )

        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(
            pii_count, anonymization_quality, re_identification_risk
        )

        assessment = PrivacyAssessment(
            source_id=source_id,
            pii_detected=len(pii_types) > 0,
            pii_types=list(pii_types),
            pii_count=pii_count,
            anonymization_quality=anonymization_quality,
            re_identification_risk=re_identification_risk,
            privacy_issues=privacy_issues,
            recommendations=recommendations,
            compliance_score=compliance_score,
        )

        logger.info(
            f"Privacy verification complete for {source_id}: "
            f"PII detected: {assessment.pii_detected}, "
            f"Risk: {assessment.re_identification_risk.value}, "
            f"Compliance: {assessment.compliance_score:.2f}"
        )

        return assessment

    def _detect_pii(self, text: str) -> Tuple[Set[str], int]:
        """Detect PII in text and return types and count."""
        pii_types = set()
        pii_count = 0

        # Check for email addresses
        if self.EMAIL_PATTERN.search(text):
            pii_types.add("email")
            pii_count += len(self.EMAIL_PATTERN.findall(text))

        # Check for phone numbers
        if self.PHONE_PATTERN.search(text):
            pii_types.add("phone")
            pii_count += len(self.PHONE_PATTERN.findall(text))

        # Check for SSN
        if self.SSN_PATTERN.search(text):
            pii_types.add("ssn")
            pii_count += len(self.SSN_PATTERN.findall(text))

        # Check for credit card numbers
        if self.CREDIT_CARD_PATTERN.search(text):
            pii_types.add("credit_card")
            pii_count += len(self.CREDIT_CARD_PATTERN.findall(text))

        # Check for IP addresses
        if self.IP_ADDRESS_PATTERN.search(text):
            pii_types.add("ip_address")
            pii_count += len(self.IP_ADDRESS_PATTERN.findall(text))

        # Check for dates (potential birth dates)
        dates = self.DATE_PATTERN.findall(text)
        if len(dates) > 5:  # Multiple dates might indicate personal information
            pii_types.add("dates")
            pii_count += len(dates)

        # Check for names
        for pattern in self._compiled_name_patterns:
            if pattern.search(text):
                pii_types.add("name")
                pii_count += len(pattern.findall(text))
                break

        # Check for addresses
        for pattern in self._compiled_address_patterns:
            if pattern.search(text):
                pii_types.add("address")
                pii_count += len(pattern.findall(text))
                break

        # Check for medical identifiers
        for pattern in self._compiled_medical_id_patterns:
            if pattern.search(text):
                pii_types.add("medical_id")
                pii_count += len(pattern.findall(text))
                break

        return pii_types, pii_count

    def _assess_anonymization_quality(
        self, text: str, pii_types: Set[str]
    ) -> AnonymizationQuality:
        """Assess anonymization quality of the dataset."""
        if not pii_types:
            # No PII detected - could be well anonymized or no PII present
            # Check for anonymization indicators
            anonymization_indicators = sum(
                1
                for pattern in self._compiled_anonymization_patterns
                if pattern.search(text)
            )
            if anonymization_indicators > 0:
                return AnonymizationQuality.EXCELLENT
            return AnonymizationQuality.GOOD

        # PII detected - check if it's anonymized
        anonymization_indicators = sum(
            1
            for pattern in self._compiled_anonymization_patterns
            if pattern.search(text)
        )

        pii_ratio = len(pii_types) / 10.0  # Normalize to 0-1
        anonymization_ratio = anonymization_indicators / max(1, len(pii_types))

        if anonymization_ratio > 0.7:
            return AnonymizationQuality.GOOD
        elif anonymization_ratio > 0.4:
            return AnonymizationQuality.ADEQUATE
        elif pii_ratio < 0.3:
            return AnonymizationQuality.POOR
        else:
            return AnonymizationQuality.NONE

    def _assess_re_identification_risk(
        self,
        text: str,
        pii_types: Set[str],
        anonymization_quality: AnonymizationQuality,
    ) -> PrivacyRiskLevel:
        """Assess re-identification risk."""
        if not pii_types:
            if anonymization_quality in [
                AnonymizationQuality.EXCELLENT,
                AnonymizationQuality.GOOD,
            ]:
                return PrivacyRiskLevel.LOW
            return PrivacyRiskLevel.MEDIUM

        # High-risk PII types
        high_risk_types = {"ssn", "credit_card", "medical_id", "email", "address"}
        if high_risk_types.intersection(pii_types):
            if anonymization_quality == AnonymizationQuality.NONE:
                return PrivacyRiskLevel.CRITICAL
            elif anonymization_quality == AnonymizationQuality.POOR:
                return PrivacyRiskLevel.HIGH
            elif anonymization_quality == AnonymizationQuality.ADEQUATE:
                return PrivacyRiskLevel.MEDIUM
            else:
                return PrivacyRiskLevel.LOW

        # Medium-risk PII types
        medium_risk_types = {"phone", "name", "dates"}
        if medium_risk_types.intersection(pii_types):
            if anonymization_quality in [
                AnonymizationQuality.NONE,
                AnonymizationQuality.POOR,
            ]:
                return PrivacyRiskLevel.HIGH
            elif anonymization_quality == AnonymizationQuality.ADEQUATE:
                return PrivacyRiskLevel.MEDIUM
            else:
                return PrivacyRiskLevel.LOW

        # Low-risk or unknown
        return PrivacyRiskLevel.MEDIUM

    def _generate_privacy_issues(
        self,
        pii_types: Set[str],
        pii_count: int,
        anonymization_quality: AnonymizationQuality,
        re_identification_risk: PrivacyRiskLevel,
    ) -> List[str]:
        """Generate list of privacy issues."""
        issues = []

        if pii_types:
            issues.append(
                f"PII detected: {', '.join(sorted(pii_types))} ({pii_count} instances)"
            )

        if anonymization_quality in [AnonymizationQuality.POOR, AnonymizationQuality.NONE]:
            issues.append(f"Poor anonymization quality: {anonymization_quality.value}")

        if re_identification_risk in [PrivacyRiskLevel.HIGH, PrivacyRiskLevel.CRITICAL]:
            issues.append(
                f"High re-identification risk: {re_identification_risk.value}"
            )

        if "ssn" in pii_types or "credit_card" in pii_types:
            issues.append("Sensitive financial identifiers detected")

        if "medical_id" in pii_types:
            issues.append("Medical identifiers detected - HIPAA compliance concern")

        return issues

    def _generate_recommendations(
        self,
        pii_types: Set[str],
        anonymization_quality: AnonymizationQuality,
        re_identification_risk: PrivacyRiskLevel,
    ) -> List[str]:
        """Generate privacy recommendations."""
        recommendations = []

        if pii_types:
            recommendations.append(
                f"Remove or anonymize detected PII types: {', '.join(sorted(pii_types))}"
            )

        if anonymization_quality in [AnonymizationQuality.POOR, AnonymizationQuality.NONE]:
            recommendations.append(
                "Implement proper anonymization techniques (redaction, pseudonymization, generalization)"
            )

        if re_identification_risk in [PrivacyRiskLevel.HIGH, PrivacyRiskLevel.CRITICAL]:
            recommendations.append(
                "High re-identification risk - implement additional privacy protections"
            )
            recommendations.append("Consider differential privacy or k-anonymity techniques")

        if "medical_id" in pii_types:
            recommendations.append(
                "Medical identifiers detected - ensure HIPAA compliance before use"
            )

        if not pii_types and anonymization_quality == AnonymizationQuality.EXCELLENT:
            recommendations.append("Dataset appears well-anonymized - verify with full dataset sample")

        return recommendations

    def _calculate_compliance_score(
        self,
        pii_count: int,
        anonymization_quality: AnonymizationQuality,
        re_identification_risk: PrivacyRiskLevel,
    ) -> float:
        """Calculate privacy compliance score (0.0-1.0)."""
        score = 1.0

        # Penalize for PII
        if pii_count > 0:
            pii_penalty = min(0.5, pii_count / 100.0)  # Max 0.5 penalty
            score -= pii_penalty

        # Penalize for poor anonymization
        quality_penalties = {
            AnonymizationQuality.EXCELLENT: 0.0,
            AnonymizationQuality.GOOD: 0.1,
            AnonymizationQuality.ADEQUATE: 0.2,
            AnonymizationQuality.POOR: 0.4,
            AnonymizationQuality.NONE: 0.5,
        }
        score -= quality_penalties.get(anonymization_quality, 0.5)

        # Penalize for re-identification risk
        risk_penalties = {
            PrivacyRiskLevel.LOW: 0.0,
            PrivacyRiskLevel.MEDIUM: 0.1,
            PrivacyRiskLevel.HIGH: 0.3,
            PrivacyRiskLevel.CRITICAL: 0.5,
        }
        score -= risk_penalties.get(re_identification_risk, 0.3)

        return max(0.0, min(1.0, score))

    def _sample_from_file(self, file_path: str) -> str:
        """Sample text from a dataset file (placeholder implementation)."""
        # In a real implementation, would:
        # 1. Detect file format (CSV, JSON, XML, etc.)
        # 2. Read and sample random rows/entries
        # 3. Extract text fields
        # 4. Return combined sample text
        logger.warning(f"File sampling not yet implemented for {file_path}")
        return ""

