"""
License Compatibility Checker

Implements license parsing, classification, and compatibility checking for AI training
and commercial use. Supports common open-source and academic licenses.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LicenseCompatibility(Enum):
    """License compatibility status for different use cases."""

    COMPATIBLE = "compatible"
    COMPATIBLE_WITH_CONDITIONS = "compatible_with_conditions"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class LicenseCheckResult:
    """Result of a license compatibility check."""

    license_name: str
    license_text: str
    ai_training_compatible: LicenseCompatibility
    commercial_use_compatible: LicenseCompatibility
    attribution_required: bool
    conditions: List[str]
    issues: List[str]
    confidence: float  # 0.0-1.0

    def is_usable(self) -> bool:
        """Check if license is usable for AI training and commercial use."""
        return (
            self.ai_training_compatible
            in [LicenseCompatibility.COMPATIBLE, LicenseCompatibility.COMPATIBLE_WITH_CONDITIONS]
            and self.commercial_use_compatible
            in [LicenseCompatibility.COMPATIBLE, LicenseCompatibility.COMPATIBLE_WITH_CONDITIONS]
        )


class LicenseChecker:
    """
    Checks license compatibility for AI training and commercial use.

    Supports common open-source licenses (MIT, Apache, BSD, GPL, etc.) and
    academic licenses (CC-BY, CC-BY-SA, etc.).
    """

    # License patterns and their compatibility
    LICENSE_PATTERNS: Dict[str, Dict[str, any]] = {
        # Permissive licenses - fully compatible
        "MIT": {
            "patterns": [r"\bmit\b", r"mit\s+license"],
            "ai_training": LicenseCompatibility.COMPATIBLE,
            "commercial": LicenseCompatibility.COMPATIBLE,
            "attribution": False,
            "conditions": [],
        },
        "Apache-2.0": {
            "patterns": [r"\bapache\s*[-]?\s*2\.?0\b", r"apache\s+license\s+version\s+2"],
            "ai_training": LicenseCompatibility.COMPATIBLE,
            "commercial": LicenseCompatibility.COMPATIBLE,
            "attribution": True,
            "conditions": ["Requires attribution", "Requires license notice"],
        },
        "BSD-3-Clause": {
            "patterns": [r"\bbsd\s*[-]?\s*3\b", r"bsd\s+3[\s-]clause"],
            "ai_training": LicenseCompatibility.COMPATIBLE,
            "commercial": LicenseCompatibility.COMPATIBLE,
            "attribution": True,
            "conditions": ["Requires attribution"],
        },
        "BSD-2-Clause": {
            "patterns": [r"\bbsd\s*[-]?\s*2\b", r"bsd\s+2[\s-]clause"],
            "ai_training": LicenseCompatibility.COMPATIBLE,
            "commercial": LicenseCompatibility.COMPATIBLE,
            "attribution": True,
            "conditions": ["Requires attribution"],
        },
        # Creative Commons - generally compatible with attribution
        "CC-BY": {
            "patterns": [
                r"\bcc[\s-]?by\b",
                r"creative\s+commons\s+attribution",
                r"cc[\s-]?by[\s-]?4\.?0",
            ],
            "ai_training": LicenseCompatibility.COMPATIBLE_WITH_CONDITIONS,
            "commercial": LicenseCompatibility.COMPATIBLE_WITH_CONDITIONS,
            "attribution": True,
            "conditions": ["Requires attribution", "Must indicate if changes were made"],
        },
        "CC-BY-SA": {
            "patterns": [
                r"\bcc[\s-]?by[\s-]?sa\b",
                r"creative\s+commons\s+attribution[\s-]?sharealike",
                r"cc[\s-]?by[\s-]?sa[\s-]?4\.?0",
            ],
            "ai_training": LicenseCompatibility.COMPATIBLE_WITH_CONDITIONS,
            "commercial": LicenseCompatibility.COMPATIBLE_WITH_CONDITIONS,
            "attribution": True,
            "conditions": [
                "Requires attribution",
                "Derivatives must be licensed under same license (copyleft)",
            ],
        },
        "CC0": {
            "patterns": [r"\bcc0\b", r"creative\s+commons\s+zero", r"public\s+domain"],
            "ai_training": LicenseCompatibility.COMPATIBLE,
            "commercial": LicenseCompatibility.COMPATIBLE,
            "attribution": False,
            "conditions": [],
        },
        # GPL licenses - copyleft, may have restrictions
        "GPL-3.0": {
            "patterns": [r"\bgpl[\s-]?v?3\b", r"gnu\s+general\s+public\s+license\s+v?3"],
            "ai_training": LicenseCompatibility.REQUIRES_REVIEW,
            "commercial": LicenseCompatibility.COMPATIBLE_WITH_CONDITIONS,
            "attribution": True,
            "conditions": [
                "Copyleft license - derivatives must be GPL-3.0",
                "Source code must be available",
                "AI training may create derivative works - review required",
            ],
        },
        "GPL-2.0": {
            "patterns": [r"\bgpl[\s-]?v?2\b", r"gnu\s+general\s+public\s+license\s+v?2"],
            "ai_training": LicenseCompatibility.REQUIRES_REVIEW,
            "commercial": LicenseCompatibility.COMPATIBLE_WITH_CONDITIONS,
            "attribution": True,
            "conditions": [
                "Copyleft license - derivatives must be GPL-2.0",
                "Source code must be available",
                "AI training may create derivative works - review required",
            ],
        },
        "AGPL-3.0": {
            "patterns": [
                r"\bagpl[\s-]?v?3\b",
                r"affero\s+gnu\s+general\s+public\s+license",
            ],
            "ai_training": LicenseCompatibility.REQUIRES_REVIEW,
            "commercial": LicenseCompatibility.REQUIRES_REVIEW,
            "attribution": True,
            "conditions": [
                "Strong copyleft - network use triggers license",
                "Source code must be available",
                "AI training and SaaS use may trigger requirements",
            ],
        },
        # LGPL - less restrictive copyleft
        "LGPL-3.0": {
            "patterns": [
                r"\blgpl[\s-]?v?3\b",
                r"gnu\s+lesser\s+general\s+public\s+license",
            ],
            "ai_training": LicenseCompatibility.COMPATIBLE_WITH_CONDITIONS,
            "commercial": LicenseCompatibility.COMPATIBLE_WITH_CONDITIONS,
            "attribution": True,
            "conditions": [
                "Modifications to library must be LGPL",
                "Linking with library is generally allowed",
            ],
        },
        # Non-commercial licenses - incompatible for commercial use
        "CC-BY-NC": {
            "patterns": [
                r"\bcc[\s-]?by[\s-]?nc\b",
                r"creative\s+commons\s+attribution[\s-]?non[\s-]?commercial",
            ],
            "ai_training": LicenseCompatibility.INCOMPATIBLE,
            "commercial": LicenseCompatibility.INCOMPATIBLE,
            "attribution": True,
            "conditions": ["Non-commercial use only"],
        },
        "CC-BY-NC-SA": {
            "patterns": [
                r"\bcc[\s-]?by[\s-]?nc[\s-]?sa\b",
                r"creative\s+commons\s+attribution[\s-]?non[\s-]?commercial[\s-]?sharealike",
            ],
            "ai_training": LicenseCompatibility.INCOMPATIBLE,
            "commercial": LicenseCompatibility.INCOMPATIBLE,
            "attribution": True,
            "conditions": ["Non-commercial use only", "Share-alike applies"],
        },
        # No derivatives - may restrict AI training
        "CC-BY-ND": {
            "patterns": [
                r"\bcc[\s-]?by[\s-]?nd\b",
                r"creative\s+commons\s+attribution[\s-]?no[\s-]?derivatives",
            ],
            "ai_training": LicenseCompatibility.REQUIRES_REVIEW,
            "commercial": LicenseCompatibility.COMPATIBLE_WITH_CONDITIONS,
            "attribution": True,
            "conditions": [
                "No derivatives allowed",
                "AI training may be considered derivative - review required",
            ],
        },
        # Academic/custom licenses
        "Academic": {
            "patterns": [
                r"\bacademic\s+license",
                r"research\s+use\s+only",
                r"educational\s+use\s+only",
            ],
            "ai_training": LicenseCompatibility.REQUIRES_REVIEW,
            "commercial": LicenseCompatibility.REQUIRES_REVIEW,
            "attribution": True,
            "conditions": ["Review terms carefully", "May restrict commercial use"],
        },
        # Proprietary/All Rights Reserved
        "Proprietary": {
            "patterns": [
                r"\ball\s+rights\s+reserved",
                r"proprietary",
                r"copyright",
                r"©",
            ],
            "ai_training": LicenseCompatibility.INCOMPATIBLE,
            "commercial": LicenseCompatibility.INCOMPATIBLE,
            "attribution": False,
            "conditions": ["All rights reserved", "Requires explicit permission"],
        },
    }

    # Keywords that indicate AI training restrictions
    AI_TRAINING_RESTRICTIONS = [
        r"no\s+ai\s+training",
        r"prohibits?\s+ai\s+training",
        r"restricts?\s+machine\s+learning",
        r"no\s+machine\s+learning",
        r"no\s+ml\s+training",
        r"prohibits?\s+training",
    ]

    # Keywords that indicate commercial use restrictions
    COMMERCIAL_RESTRICTIONS = [
        r"non[\s-]?commercial",
        r"no\s+commercial\s+use",
        r"research\s+use\s+only",
        r"educational\s+use\s+only",
        r"personal\s+use\s+only",
    ]

    def __init__(self):
        """Initialize the license checker."""
        # Compile regex patterns for performance
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for license_name, license_info in self.LICENSE_PATTERNS.items():
            self._compiled_patterns[license_name] = [
                re.compile(pattern, re.IGNORECASE) for pattern in license_info["patterns"]
            ]

        self._ai_restriction_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.AI_TRAINING_RESTRICTIONS
        ]
        self._commercial_restriction_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.COMMERCIAL_RESTRICTIONS
        ]

    def check_license(
        self, license_text: Optional[str], license_name: Optional[str] = None
    ) -> LicenseCheckResult:
        """
        Check license compatibility for AI training and commercial use.

        Args:
            license_text: Full license text or description
            license_name: Optional license name/identifier

        Returns:
            LicenseCheckResult with compatibility information
        """
        if not license_text and not license_name:
            return LicenseCheckResult(
                license_name="Unknown",
                license_text="",
                ai_training_compatible=LicenseCompatibility.UNKNOWN,
                commercial_use_compatible=LicenseCompatibility.UNKNOWN,
                attribution_required=False,
                conditions=["No license information provided"],
                issues=["License information not available"],
                confidence=0.0,
            )

        # Normalize license text
        normalized_text = ""
        if license_text:
            normalized_text = license_text.lower().strip()
        if license_name:
            normalized_text = f"{license_name.lower()} {normalized_text}"

        # Try to identify license
        identified_license = self._identify_license(normalized_text)
        if identified_license:
            result = self._check_known_license(identified_license, normalized_text)
        else:
            # Unknown license - analyze text for restrictions
            result = self._analyze_unknown_license(normalized_text)

        # Check for explicit restrictions in text
        result = self._apply_restriction_checks(result, normalized_text)

        logger.info(
            f"License check: {result.license_name}, "
            f"AI training: {result.ai_training_compatible.value}, "
            f"Commercial: {result.commercial_use_compatible.value}"
        )

        return result

    def _identify_license(self, text: str) -> Optional[str]:
        """Identify license from text patterns."""
        for license_name, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return license_name
        return None

    def _check_known_license(
        self, license_name: str, license_text: str
    ) -> LicenseCheckResult:
        """Check compatibility for a known license."""
        license_info = self.LICENSE_PATTERNS[license_name]

        return LicenseCheckResult(
            license_name=license_name,
            license_text=license_text,
            ai_training_compatible=license_info["ai_training"],
            commercial_use_compatible=license_info["commercial"],
            attribution_required=license_info["attribution"],
            conditions=license_info["conditions"].copy(),
            issues=[],
            confidence=0.9,  # High confidence for known licenses
        )

    def _analyze_unknown_license(self, license_text: str) -> LicenseCheckResult:
        """Analyze unknown license for compatibility indicators."""
        issues = []
        conditions = []
        ai_compatible = LicenseCompatibility.UNKNOWN
        commercial_compatible = LicenseCompatibility.UNKNOWN
        attribution_required = False
        confidence = 0.3  # Low confidence for unknown licenses

        # Check for permissive indicators
        if any(
            keyword in license_text
            for keyword in [
                "permit",
                "allow",
                "free",
                "open",
                "distribute",
                "modify",
            ]
        ):
            # May be permissive, but need review
            ai_compatible = LicenseCompatibility.REQUIRES_REVIEW
            commercial_compatible = LicenseCompatibility.REQUIRES_REVIEW
            conditions.append("License terms unclear - manual review required")
            confidence = 0.5

        # Check for restrictive indicators
        if any(
            keyword in license_text
            for keyword in [
                "prohibit",
                "forbid",
                "restrict",
                "reserved",
                "copyright",
            ]
        ):
            issues.append("License contains restrictive language")
            conditions.append("Review license terms carefully")

        return LicenseCheckResult(
            license_name="Unknown",
            license_text=license_text,
            ai_training_compatible=ai_compatible,
            commercial_use_compatible=commercial_compatible,
            attribution_required=attribution_required,
            conditions=conditions,
            issues=issues,
            confidence=confidence,
        )

    def _apply_restriction_checks(
        self, result: LicenseCheckResult, license_text: str
    ) -> LicenseCheckResult:
        """Apply additional restriction checks to result."""
        # Check for AI training restrictions
        for pattern in self._ai_restriction_patterns:
            if pattern.search(license_text):
                result.ai_training_compatible = LicenseCompatibility.INCOMPATIBLE
                result.issues.append("License explicitly prohibits AI training")
                result.conditions.append("AI training explicitly prohibited")
                break

        # Check for commercial use restrictions
        for pattern in self._commercial_restriction_patterns:
            if pattern.search(license_text):
                result.commercial_use_compatible = LicenseCompatibility.INCOMPATIBLE
                result.issues.append("License prohibits commercial use")
                result.conditions.append("Commercial use prohibited")
                break

        return result

    def flag_incompatible_licenses(self, result: LicenseCheckResult) -> bool:
        """
        Flag licenses that are incompatible for review.

        Args:
            result: License check result

        Returns:
            True if license should be flagged for review
        """
        return (
            result.ai_training_compatible == LicenseCompatibility.INCOMPATIBLE
            or result.commercial_use_compatible == LicenseCompatibility.INCOMPATIBLE
            or result.ai_training_compatible == LicenseCompatibility.REQUIRES_REVIEW
            or result.commercial_use_compatible == LicenseCompatibility.REQUIRES_REVIEW
            or result.confidence < 0.5
        )

