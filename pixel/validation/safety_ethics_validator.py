"""
Safety and Ethics Compliance Validation System

This module provides comprehensive validation of safety and ethics compliance
for therapeutic responses, including crisis detection, ethical guideline
adherence, legal compliance, and professional standards verification.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .clinical_accuracy_validator import (
    ClinicalContext,
    SafetyRiskLevel,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Safety and ethics compliance levels"""

    FULLY_COMPLIANT = "fully_compliant"
    MOSTLY_COMPLIANT = "mostly_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    CRITICALLY_NON_COMPLIANT = "critically_non_compliant"


class SafetyCategory(Enum):
    """Safety assessment categories"""

    SUICIDE_RISK = "suicide_risk"
    SELF_HARM_RISK = "self_harm_risk"
    VIOLENCE_RISK = "violence_risk"
    CHILD_SAFETY = "child_safety"
    ELDER_ABUSE = "elder_abuse"
    SUBSTANCE_ABUSE = "substance_abuse"
    PSYCHOSIS_RISK = "psychosis_risk"
    CRISIS_INTERVENTION = "crisis_intervention"


class EthicsCategory(Enum):
    """Ethics compliance categories"""

    INFORMED_CONSENT = "informed_consent"
    CONFIDENTIALITY = "confidentiality"
    DUAL_RELATIONSHIPS = "dual_relationships"
    COMPETENCE = "competence"
    INTEGRITY = "integrity"
    PROFESSIONAL_RESPONSIBILITY = "professional_responsibility"
    RESPECT_FOR_RIGHTS = "respect_for_rights"
    SOCIAL_RESPONSIBILITY = "social_responsibility"


class LegalCategory(Enum):
    """Legal compliance categories"""

    MANDATORY_REPORTING = "mandatory_reporting"
    DUTY_TO_WARN = "duty_to_warn"
    HIPAA_COMPLIANCE = "hipaa_compliance"
    SCOPE_OF_PRACTICE = "scope_of_practice"
    DOCUMENTATION = "documentation"
    LICENSING_REQUIREMENTS = "licensing_requirements"


class ViolationSeverity(Enum):
    """Violation severity levels"""

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class SafetyIndicator:
    """Safety risk indicator"""

    indicator_id: str
    category: SafetyCategory
    description: str
    risk_level: SafetyRiskLevel
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    context_requirements: List[str] = field(default_factory=list)
    immediate_action_required: bool = False


@dataclass
class EthicsGuideline:
    """Ethics compliance guideline"""

    guideline_id: str
    category: EthicsCategory
    title: str
    description: str
    requirements: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    severity: ViolationSeverity = ViolationSeverity.MODERATE


@dataclass
class LegalRequirement:
    """Legal compliance requirement"""

    requirement_id: str
    category: LegalCategory
    title: str
    description: str
    jurisdiction: str = "US"
    mandatory: bool = True
    penalties: List[str] = field(default_factory=list)
    compliance_indicators: List[str] = field(default_factory=list)


@dataclass
class ComplianceViolation:
    """Safety or ethics compliance violation"""

    violation_id: str
    category: str  # SafetyCategory, EthicsCategory, or LegalCategory
    severity: ViolationSeverity
    title: str
    description: str
    evidence: str
    recommendations: List[str] = field(default_factory=list)
    immediate_action_required: bool = False
    legal_implications: List[str] = field(default_factory=list)
    confidence: float = 0.0
    location: Dict[str, int] = field(default_factory=dict)


@dataclass
class SafetyAssessmentResult:
    """Safety assessment result"""

    overall_risk: SafetyRiskLevel
    category_risks: Dict[SafetyCategory, SafetyRiskLevel] = field(default_factory=dict)
    indicators_detected: List[str] = field(default_factory=list)
    immediate_interventions: List[str] = field(default_factory=list)
    safety_plan_needed: bool = False
    crisis_protocol_triggered: bool = False


@dataclass
class EthicsComplianceResult:
    """Ethics compliance assessment result"""

    overall_compliance: ComplianceLevel
    category_compliance: Dict[EthicsCategory, ComplianceLevel] = field(default_factory=dict)
    guidelines_met: List[str] = field(default_factory=list)
    guidelines_violated: List[str] = field(default_factory=list)
    ethical_concerns: List[str] = field(default_factory=list)


@dataclass
class LegalComplianceResult:
    """Legal compliance assessment result"""

    overall_compliance: ComplianceLevel
    category_compliance: Dict[LegalCategory, ComplianceLevel] = field(default_factory=dict)
    requirements_met: List[str] = field(default_factory=list)
    requirements_violated: List[str] = field(default_factory=list)
    legal_risks: List[str] = field(default_factory=list)
    mandatory_actions: List[str] = field(default_factory=list)


@dataclass
class SafetyEthicsComplianceResult:
    """Complete safety and ethics compliance assessment result"""

    assessment_id: str
    timestamp: datetime
    response_text: str
    clinical_context: ClinicalContext
    overall_compliance: ComplianceLevel
    confidence_score: float
    safety_result: SafetyAssessmentResult
    ethics_result: EthicsComplianceResult
    legal_result: LegalComplianceResult
    violations: List[ComplianceViolation] = field(default_factory=list)
    immediate_actions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafetyEthicsValidator:
    """
    Comprehensive safety and ethics compliance validation system

    This class provides validation of therapeutic responses against safety
    protocols, ethical guidelines, and legal requirements.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the safety and ethics validator"""
        self.config = self._load_config(config_path)
        self.safety_indicators: Dict[str, SafetyIndicator] = {}
        self.ethics_guidelines: Dict[str, EthicsGuideline] = {}
        self.legal_requirements: Dict[str, LegalRequirement] = {}

        # Load validation rules
        self._load_safety_indicators()
        self._load_ethics_guidelines()
        self._load_legal_requirements()

        # Initialize statistics
        self.validation_stats = {
            "total_validations": 0,
            "safety_violations": 0,
            "ethics_violations": 0,
            "legal_violations": 0,
            "critical_violations": 0,
        }

        logger.info("Safety and ethics validator initialized")

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration settings"""
        default_config = {
            "crisis_threshold": 0.7,
            "immediate_action_threshold": 0.8,
            "legal_compliance_threshold": 0.9,
            "ethics_compliance_threshold": 0.8,
            "safety_compliance_threshold": 0.85,
            "enable_crisis_protocols": True,
            "enable_mandatory_reporting": True,
        }

        if config_path and config_path.exists():
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return default_config

    def _load_safety_indicators(self) -> None:
        """Load safety risk indicators"""
        indicators = [
            SafetyIndicator(
                indicator_id="SI001",
                category=SafetyCategory.SUICIDE_RISK,
                description="Direct suicidal ideation expression",
                risk_level=SafetyRiskLevel.HIGH,
                keywords=[
                    "kill myself",
                    "end my life",
                    "suicide",
                    "want to die",
                    "better off dead",
                    "not worth living",
                ],
                patterns=[
                    r"(want to|going to|plan to) (die|kill myself)",
                    r"(life is|I am) not worth living",
                    r"better off dead",
                    r"end it all",
                ],
                immediate_action_required=True,
            ),
            SafetyIndicator(
                indicator_id="SI002",
                category=SafetyCategory.SELF_HARM_RISK,
                description="Self-harm behavior indicators",
                risk_level=SafetyRiskLevel.MODERATE,
                keywords=[
                    "cut myself",
                    "hurt myself",
                    "self-harm",
                    "cutting",
                    "burning",
                    "hitting myself",
                ],
                patterns=[r"(cut|hurt|harm) myself", r"self.harm", r"(burning|hitting) myself"],
                immediate_action_required=True,
            ),
            SafetyIndicator(
                indicator_id="SI003",
                category=SafetyCategory.VIOLENCE_RISK,
                description="Violence toward others indicators",
                risk_level=SafetyRiskLevel.HIGH,
                keywords=["hurt someone", "kill them", "violence", "attack", "weapon", "revenge"],
                patterns=[
                    r"(hurt|kill|attack) (someone|them|him|her)",
                    r"get (revenge|back at)",
                    r"use a weapon",
                ],
                immediate_action_required=True,
            ),
            SafetyIndicator(
                indicator_id="SI004",
                category=SafetyCategory.CHILD_SAFETY,
                description="Child abuse or neglect indicators",
                risk_level=SafetyRiskLevel.CRITICAL,
                keywords=[
                    "child abuse",
                    "hit my child",
                    "neglect",
                    "inappropriate touch",
                    "sexual abuse",
                ],
                patterns=[
                    r"(hit|abuse|neglect) (my|the) child",
                    r"inappropriate (touch|contact)",
                    r"sexual (abuse|contact) with (child|minor)",
                ],
                immediate_action_required=True,
            ),
        ]

        for indicator in indicators:
            self.safety_indicators[indicator.indicator_id] = indicator

    def _load_ethics_guidelines(self) -> None:
        """Load ethics compliance guidelines"""
        guidelines = [
            EthicsGuideline(
                guideline_id="EG001",
                category=EthicsCategory.CONFIDENTIALITY,
                title="Client Confidentiality",
                description="Maintain strict client confidentiality",
                requirements=[
                    "Do not share client information without consent",
                    "Protect client identity and personal details",
                    "Secure storage and transmission of client data",
                ],
                violations=[
                    "sharing client details with others",
                    "discussing client cases inappropriately",
                    "breaching confidentiality agreements",
                ],
                severity=ViolationSeverity.SEVERE,
            ),
            EthicsGuideline(
                guideline_id="EG002",
                category=EthicsCategory.DUAL_RELATIONSHIPS,
                title="Dual Relationship Boundaries",
                description="Avoid dual relationships with clients",
                requirements=[
                    "Maintain professional boundaries",
                    "Avoid personal relationships with clients",
                    "No business relationships with clients",
                ],
                violations=[
                    "personal friendship with client",
                    "romantic involvement with client",
                    "business dealings with client",
                ],
                severity=ViolationSeverity.SEVERE,
            ),
            EthicsGuideline(
                guideline_id="EG003",
                category=EthicsCategory.COMPETENCE,
                title="Professional Competence",
                description="Practice within scope of competence",
                requirements=[
                    "Stay within areas of training and expertise",
                    "Seek consultation when needed",
                    "Maintain professional development",
                ],
                violations=[
                    "practicing outside scope of competence",
                    "providing services without proper training",
                    "failing to seek appropriate consultation",
                ],
                severity=ViolationSeverity.MAJOR,
            ),
            EthicsGuideline(
                guideline_id="EG004",
                category=EthicsCategory.INFORMED_CONSENT,
                title="Informed Consent",
                description="Ensure proper informed consent",
                requirements=[
                    "Explain treatment procedures and risks",
                    "Obtain voluntary consent",
                    "Respect client's right to refuse treatment",
                ],
                violations=[
                    "proceeding without proper consent",
                    "coercing client into treatment",
                    "failing to explain risks and procedures",
                ],
                severity=ViolationSeverity.MAJOR,
            ),
        ]

        for guideline in guidelines:
            self.ethics_guidelines[guideline.guideline_id] = guideline

    def _load_legal_requirements(self) -> None:
        """Load legal compliance requirements"""
        requirements = [
            LegalRequirement(
                requirement_id="LR001",
                category=LegalCategory.MANDATORY_REPORTING,
                title="Mandatory Reporting of Child Abuse",
                description="Legal requirement to report suspected child abuse",
                mandatory=True,
                penalties=[
                    "Criminal charges for failure to report",
                    "Professional license suspension",
                    "Civil liability",
                ],
                compliance_indicators=[
                    "immediate reporting to authorities",
                    "documentation of report",
                    "follow-up as required",
                ],
            ),
            LegalRequirement(
                requirement_id="LR002",
                category=LegalCategory.DUTY_TO_WARN,
                title="Duty to Warn/Protect",
                description="Legal duty to warn potential victims of violence",
                mandatory=True,
                penalties=[
                    "Civil liability for failure to warn",
                    "Professional malpractice claims",
                    "License disciplinary action",
                ],
                compliance_indicators=[
                    "warning potential victims",
                    "contacting law enforcement",
                    "documentation of actions taken",
                ],
            ),
            LegalRequirement(
                requirement_id="LR003",
                category=LegalCategory.HIPAA_COMPLIANCE,
                title="HIPAA Privacy Protection",
                description="Compliance with HIPAA privacy regulations",
                mandatory=True,
                penalties=[
                    "Federal fines up to $1.5 million",
                    "Criminal charges for willful violations",
                    "Professional sanctions",
                ],
                compliance_indicators=[
                    "proper authorization for disclosures",
                    "minimum necessary standard",
                    "secure handling of PHI",
                ],
            ),
        ]

        for requirement in requirements:
            self.legal_requirements[requirement.requirement_id] = requirement

    async def validate_compliance(
        self, response_text: str, clinical_context: ClinicalContext
    ) -> SafetyEthicsComplianceResult:
        """
        Perform comprehensive safety and ethics compliance validation

        Args:
            response_text: The therapeutic response to validate
            clinical_context: Clinical context for the response

        Returns:
            Complete compliance validation result
        """
        assessment_id = f"sec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Perform individual assessments
        safety_result = await self._assess_safety_compliance(response_text, clinical_context)
        ethics_result = await self._assess_ethics_compliance(response_text, clinical_context)
        legal_result = await self._assess_legal_compliance(response_text, clinical_context)

        # Collect all violations
        violations = []
        violations.extend(
            await self._detect_safety_violations(response_text, clinical_context, safety_result)
        )
        violations.extend(
            await self._detect_ethics_violations(response_text, clinical_context, ethics_result)
        )
        violations.extend(
            await self._detect_legal_violations(response_text, clinical_context, legal_result)
        )

        # Calculate overall compliance
        overall_compliance, confidence_score = self._calculate_overall_compliance(
            safety_result, ethics_result, legal_result, violations
        )

        # Generate immediate actions, recommendations, and warnings
        immediate_actions = self._generate_immediate_actions(violations, safety_result)
        recommendations = self._generate_recommendations(violations)
        warnings = self._generate_warnings(violations, safety_result)

        # Create result
        result = SafetyEthicsComplianceResult(
            assessment_id=assessment_id,
            timestamp=datetime.now(),
            response_text=response_text,
            clinical_context=clinical_context,
            overall_compliance=overall_compliance,
            confidence_score=confidence_score,
            safety_result=safety_result,
            ethics_result=ethics_result,
            legal_result=legal_result,
            violations=violations,
            immediate_actions=immediate_actions,
            recommendations=recommendations,
            warnings=warnings,
        )

        # Update statistics
        self._update_statistics(result)

        logger.info(f"Safety and ethics compliance validation completed: {assessment_id}")
        return result

    async def _assess_safety_compliance(
        self, response_text: str, clinical_context: ClinicalContext
    ) -> SafetyAssessmentResult:
        """Assess safety compliance"""
        category_risks = {}
        indicators_detected = []
        immediate_interventions = []
        crisis_protocol_triggered = False
        safety_plan_needed = False

        # Check each safety indicator
        for indicator in self.safety_indicators.values():
            detected = await self._check_safety_indicator(
                indicator, response_text, clinical_context
            )

            if detected:
                indicators_detected.append(indicator.indicator_id)
                category_risks[indicator.category] = indicator.risk_level

                if indicator.immediate_action_required:
                    immediate_interventions.append(
                        f"Immediate intervention required for {indicator.description}"
                    )

                if indicator.risk_level in [SafetyRiskLevel.HIGH, SafetyRiskLevel.CRITICAL]:
                    crisis_protocol_triggered = True
                    safety_plan_needed = True

        # Determine overall risk
        if not category_risks:
            overall_risk = SafetyRiskLevel.MINIMAL
        else:
            risk_values = {
                SafetyRiskLevel.MINIMAL: 0,
                SafetyRiskLevel.LOW: 1,
                SafetyRiskLevel.MODERATE: 2,
                SafetyRiskLevel.HIGH: 3,
                SafetyRiskLevel.CRITICAL: 4,
            }
            max_risk_value = max(risk_values[risk] for risk in category_risks.values())
            overall_risk = next(
                risk for risk, value in risk_values.items() if value == max_risk_value
            )

        return SafetyAssessmentResult(
            overall_risk=overall_risk,
            category_risks=category_risks,
            indicators_detected=indicators_detected,
            immediate_interventions=immediate_interventions,
            safety_plan_needed=safety_plan_needed,
            crisis_protocol_triggered=crisis_protocol_triggered,
        )

    async def _check_safety_indicator(
        self, indicator: SafetyIndicator, response_text: str, clinical_context: ClinicalContext
    ) -> bool:
        """Check if a safety indicator is present"""
        response_lower = response_text.lower()

        # Check keywords
        for keyword in indicator.keywords:
            if keyword.lower() in response_lower:
                return True

        # Check patterns
        for pattern in indicator.patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                return True

        return False

    async def _assess_ethics_compliance(
        self, response_text: str, clinical_context: ClinicalContext
    ) -> EthicsComplianceResult:
        """Assess ethics compliance"""
        category_compliance = {}
        guidelines_met = []
        guidelines_violated = []
        ethical_concerns = []

        # Check each ethics guideline
        for guideline in self.ethics_guidelines.values():
            compliance_level = await self._check_ethics_guideline(
                guideline, response_text, clinical_context
            )

            category_compliance[guideline.category] = compliance_level

            if compliance_level in [
                ComplianceLevel.FULLY_COMPLIANT,
                ComplianceLevel.MOSTLY_COMPLIANT,
            ]:
                guidelines_met.append(guideline.guideline_id)
            else:
                guidelines_violated.append(guideline.guideline_id)
                ethical_concerns.append(f"{guideline.title}: {guideline.description}")

        # Calculate overall compliance
        if not category_compliance:
            overall_compliance = ComplianceLevel.FULLY_COMPLIANT
        else:
            compliance_values = {
                ComplianceLevel.FULLY_COMPLIANT: 4,
                ComplianceLevel.MOSTLY_COMPLIANT: 3,
                ComplianceLevel.PARTIALLY_COMPLIANT: 2,
                ComplianceLevel.NON_COMPLIANT: 1,
                ComplianceLevel.CRITICALLY_NON_COMPLIANT: 0,
            }
            avg_compliance = sum(
                compliance_values[comp] for comp in category_compliance.values()
            ) / len(category_compliance)

            overall_compliance = next(
                comp for comp, value in compliance_values.items() if value <= avg_compliance
            )

        return EthicsComplianceResult(
            overall_compliance=overall_compliance,
            category_compliance=category_compliance,
            guidelines_met=guidelines_met,
            guidelines_violated=guidelines_violated,
            ethical_concerns=ethical_concerns,
        )

    async def _check_ethics_guideline(
        self, guideline: EthicsGuideline, response_text: str, clinical_context: ClinicalContext
    ) -> ComplianceLevel:
        """Check compliance with an ethics guideline"""
        response_lower = response_text.lower()

        # Check for violations
        violation_count = 0
        for violation in guideline.violations:
            if violation.lower() in response_lower:
                violation_count += 1

        # Determine compliance level based on violations
        if violation_count == 0:
            return ComplianceLevel.FULLY_COMPLIANT
        elif violation_count == 1:
            return ComplianceLevel.MOSTLY_COMPLIANT
        elif violation_count == 2:
            return ComplianceLevel.PARTIALLY_COMPLIANT
        elif violation_count <= 3:
            return ComplianceLevel.NON_COMPLIANT
        else:
            return ComplianceLevel.CRITICALLY_NON_COMPLIANT

    async def _assess_legal_compliance(
        self, response_text: str, clinical_context: ClinicalContext
    ) -> LegalComplianceResult:
        """Assess legal compliance"""
        category_compliance = {}
        requirements_met = []
        requirements_violated = []
        legal_risks = []
        mandatory_actions = []

        # Check each legal requirement
        for requirement in self.legal_requirements.values():
            compliance_level = await self._check_legal_requirement(
                requirement, response_text, clinical_context
            )

            category_compliance[requirement.category] = compliance_level

            if compliance_level in [
                ComplianceLevel.FULLY_COMPLIANT,
                ComplianceLevel.MOSTLY_COMPLIANT,
            ]:
                requirements_met.append(requirement.requirement_id)
            else:
                requirements_violated.append(requirement.requirement_id)
                legal_risks.extend(requirement.penalties)

                if requirement.mandatory:
                    mandatory_actions.extend(requirement.compliance_indicators)

        # Calculate overall compliance
        if not category_compliance:
            overall_compliance = ComplianceLevel.FULLY_COMPLIANT
        else:
            compliance_values = {
                ComplianceLevel.FULLY_COMPLIANT: 4,
                ComplianceLevel.MOSTLY_COMPLIANT: 3,
                ComplianceLevel.PARTIALLY_COMPLIANT: 2,
                ComplianceLevel.NON_COMPLIANT: 1,
                ComplianceLevel.CRITICALLY_NON_COMPLIANT: 0,
            }
            avg_compliance = sum(
                compliance_values[comp] for comp in category_compliance.values()
            ) / len(category_compliance)

            overall_compliance = next(
                comp for comp, value in compliance_values.items() if value <= avg_compliance
            )

        return LegalComplianceResult(
            overall_compliance=overall_compliance,
            category_compliance=category_compliance,
            requirements_met=requirements_met,
            requirements_violated=requirements_violated,
            legal_risks=list(set(legal_risks)),  # Remove duplicates
            mandatory_actions=list(set(mandatory_actions)),  # Remove duplicates
        )

    async def _check_legal_requirement(
        self, requirement: LegalRequirement, response_text: str, clinical_context: ClinicalContext
    ) -> ComplianceLevel:
        """Check compliance with a legal requirement"""
        # This is a simplified implementation
        # In production, this would involve more sophisticated analysis

        response_lower = response_text.lower()

        # Check for compliance indicators
        compliance_indicators_found = 0
        for indicator in requirement.compliance_indicators:
            if any(word in response_lower for word in indicator.lower().split()):
                compliance_indicators_found += 1

        # Determine compliance level
        total_indicators = len(requirement.compliance_indicators)
        if total_indicators == 0:
            return ComplianceLevel.FULLY_COMPLIANT

        compliance_ratio = compliance_indicators_found / total_indicators

        if compliance_ratio >= 0.9:
            return ComplianceLevel.FULLY_COMPLIANT
        elif compliance_ratio >= 0.7:
            return ComplianceLevel.MOSTLY_COMPLIANT
        elif compliance_ratio >= 0.5:
            return ComplianceLevel.PARTIALLY_COMPLIANT
        elif compliance_ratio >= 0.3:
            return ComplianceLevel.NON_COMPLIANT
        else:
            return ComplianceLevel.CRITICALLY_NON_COMPLIANT

    async def _detect_safety_violations(
        self,
        response_text: str,
        clinical_context: ClinicalContext,
        safety_result: SafetyAssessmentResult,
    ) -> List[ComplianceViolation]:
        """Detect safety compliance violations"""
        violations = []

        for indicator_id in safety_result.indicators_detected:
            indicator = self.safety_indicators[indicator_id]

            violation = ComplianceViolation(
                violation_id=f"sv_{indicator_id}_{len(violations)}",
                category=indicator.category.value,
                severity=self._map_risk_to_severity(indicator.risk_level),
                title=f"Safety Risk: {indicator.description}",
                description=f"Detected safety indicator: {indicator.description}",
                evidence=self._extract_evidence(response_text, indicator),
                immediate_action_required=indicator.immediate_action_required,
                confidence=0.85,
            )

            violations.append(violation)

        return violations

    async def _detect_ethics_violations(
        self,
        response_text: str,
        clinical_context: ClinicalContext,
        ethics_result: EthicsComplianceResult,
    ) -> List[ComplianceViolation]:
        """Detect ethics compliance violations"""
        violations = []

        for guideline_id in ethics_result.guidelines_violated:
            guideline = self.ethics_guidelines[guideline_id]

            violation = ComplianceViolation(
                violation_id=f"ev_{guideline_id}_{len(violations)}",
                category=guideline.category.value,
                severity=guideline.severity,
                title=f"Ethics Violation: {guideline.title}",
                description=guideline.description,
                evidence=self._extract_ethics_evidence(response_text, guideline),
                confidence=0.8,
            )

            violations.append(violation)

        return violations

    async def _detect_legal_violations(
        self,
        response_text: str,
        clinical_context: ClinicalContext,
        legal_result: LegalComplianceResult,
    ) -> List[ComplianceViolation]:
        """Detect legal compliance violations"""
        violations = []

        for requirement_id in legal_result.requirements_violated:
            requirement = self.legal_requirements[requirement_id]

            violation = ComplianceViolation(
                violation_id=f"lv_{requirement_id}_{len(violations)}",
                category=requirement.category.value,
                severity=(
                    ViolationSeverity.SEVERE if requirement.mandatory else ViolationSeverity.MAJOR
                ),
                title=f"Legal Violation: {requirement.title}",
                description=requirement.description,
                evidence=self._extract_legal_evidence(response_text, requirement),
                legal_implications=requirement.penalties,
                immediate_action_required=requirement.mandatory,
                confidence=0.75,
            )

            violations.append(violation)

        return violations

    def _map_risk_to_severity(self, risk_level: SafetyRiskLevel) -> ViolationSeverity:
        """Map safety risk level to violation severity"""
        mapping = {
            SafetyRiskLevel.MINIMAL: ViolationSeverity.MINOR,
            SafetyRiskLevel.LOW: ViolationSeverity.MODERATE,
            SafetyRiskLevel.MODERATE: ViolationSeverity.MAJOR,
            SafetyRiskLevel.HIGH: ViolationSeverity.SEVERE,
            SafetyRiskLevel.CRITICAL: ViolationSeverity.CRITICAL,
        }
        return mapping.get(risk_level, ViolationSeverity.MODERATE)

    def _extract_evidence(self, response_text: str, indicator: SafetyIndicator) -> str:
        """Extract evidence for safety indicator"""
        response_lower = response_text.lower()

        # Find matching keywords or patterns
        for keyword in indicator.keywords:
            if keyword.lower() in response_lower:
                start_pos = response_lower.find(keyword.lower())
                context_start = max(0, start_pos - 50)
                context_end = min(len(response_text), start_pos + len(keyword) + 50)
                return response_text[context_start:context_end]

        return "Evidence found through safety analysis"

    def _extract_ethics_evidence(self, response_text: str, guideline: EthicsGuideline) -> str:
        """Extract evidence for ethics violation"""
        response_lower = response_text.lower()

        for violation in guideline.violations:
            if violation.lower() in response_lower:
                start_pos = response_lower.find(violation.lower())
                context_start = max(0, start_pos - 50)
                context_end = min(len(response_text), start_pos + len(violation) + 50)
                return response_text[context_start:context_end]

        return "Ethics violation detected through pattern analysis"

    def _extract_legal_evidence(self, response_text: str, requirement: LegalRequirement) -> str:
        """Extract evidence for legal violation"""
        return "Legal compliance analysis indicates potential violation"

    def _calculate_overall_compliance(
        self,
        safety_result: SafetyAssessmentResult,
        ethics_result: EthicsComplianceResult,
        legal_result: LegalComplianceResult,
        violations: List[ComplianceViolation],
    ) -> Tuple[ComplianceLevel, float]:
        """Calculate overall compliance level and confidence"""

        # Map results to numeric values
        compliance_values = {
            ComplianceLevel.FULLY_COMPLIANT: 4,
            ComplianceLevel.MOSTLY_COMPLIANT: 3,
            ComplianceLevel.PARTIALLY_COMPLIANT: 2,
            ComplianceLevel.NON_COMPLIANT: 1,
            ComplianceLevel.CRITICALLY_NON_COMPLIANT: 0,
        }

        risk_values = {
            SafetyRiskLevel.MINIMAL: 4,
            SafetyRiskLevel.LOW: 3,
            SafetyRiskLevel.MODERATE: 2,
            SafetyRiskLevel.HIGH: 1,
            SafetyRiskLevel.CRITICAL: 0,
        }

        # Calculate weighted score
        safety_score = risk_values[safety_result.overall_risk]
        ethics_score = compliance_values[ethics_result.overall_compliance]
        legal_score = compliance_values[legal_result.overall_compliance]

        # Weight the scores
        weighted_score = (
            safety_score * 0.4  # Safety is most important
            + ethics_score * 0.35
            + legal_score * 0.25
        )

        # Map back to compliance level
        if weighted_score >= 3.5:
            overall_compliance = ComplianceLevel.FULLY_COMPLIANT
            confidence = 0.9
        elif weighted_score >= 2.5:
            overall_compliance = ComplianceLevel.MOSTLY_COMPLIANT
            confidence = 0.8
        elif weighted_score >= 1.5:
            overall_compliance = ComplianceLevel.PARTIALLY_COMPLIANT
            confidence = 0.7
        elif weighted_score >= 0.5:
            overall_compliance = ComplianceLevel.NON_COMPLIANT
            confidence = 0.6
        else:
            overall_compliance = ComplianceLevel.CRITICALLY_NON_COMPLIANT
            confidence = 0.5

        return overall_compliance, confidence

    def _generate_immediate_actions(
        self, violations: List[ComplianceViolation], safety_result: SafetyAssessmentResult
    ) -> List[str]:
        """Generate immediate actions required"""
        actions = []

        # Add safety interventions
        actions.extend(safety_result.immediate_interventions)

        # Add violation-specific actions
        for violation in violations:
            if violation.immediate_action_required:
                actions.append(f"Immediate action required: {violation.title}")

        return list(set(actions))  # Remove duplicates

    def _generate_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []

        if violations:
            recommendations.extend(
                ["Review safety and ethics protocols", "Seek supervision and consultation"]
            )

        return recommendations

    def _generate_warnings(
        self, violations: List[ComplianceViolation], safety_result: SafetyAssessmentResult
    ) -> List[str]:
        """Generate warnings based on violations and safety assessment"""
        warnings = []

        # Critical safety warnings
        if safety_result.overall_risk in [SafetyRiskLevel.HIGH, SafetyRiskLevel.CRITICAL]:
            warnings.append("CRITICAL SAFETY RISK: Immediate intervention required")

        # Critical violations
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical_violations:
            warnings.append("CRITICAL VIOLATIONS DETECTED: Immediate action required")

        return warnings

    def _update_statistics(self, result: SafetyEthicsComplianceResult) -> None:
        """Update validation statistics"""
        self.validation_stats["total_validations"] += 1

        # Count violations by type
        for violation in result.violations:
            if "safety" in violation.category:
                self.validation_stats["safety_violations"] += 1
            elif "ethics" in violation.category:
                self.validation_stats["ethics_violations"] += 1
            elif "legal" in violation.category:
                self.validation_stats["legal_violations"] += 1

            if violation.severity == ViolationSeverity.CRITICAL:
                self.validation_stats["critical_violations"] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self.validation_stats.copy()


# Example usage and testing
if __name__ == "__main__":

    async def main():
        # Initialize validator
        validator = SafetyEthicsValidator()

        # Example clinical context
        from .clinical_accuracy_validator import ClinicalContext, TherapeuticModality

        context = ClinicalContext(
            client_presentation="Client expressing suicidal thoughts",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="crisis",
            crisis_indicators=["suicidal_ideation"],
        )

        # Example response with safety and ethics issues
        problematic_response = """
        I understand you want to kill yourself, but everyone feels that way sometimes.
        Let me tell you about my personal experience with depression when my wife
        left me. You don't really mean it, you're just seeking attention. We can
        talk about this more when we grab coffee outside of therapy next week.
        """

        # Validate compliance
        result = await validator.validate_compliance(problematic_response, context)

        # Print results
        print(f"Assessment ID: {result.assessment_id}")
        print(f"Overall Compliance: {result.overall_compliance.value}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Safety Risk: {result.safety_result.overall_risk.value}")
        print(f"Violations Found: {len(result.violations)}")

        for violation in result.violations:
            print(f"\n- {violation.title} ({violation.severity.value})")
            print(f"  Evidence: {violation.evidence[:100]}...")

        print(f"\nImmediate Actions: {result.immediate_actions}")
        print(f"Warnings: {result.warnings}")
        print(f"Recommendations: {result.recommendations}")

        # Get statistics
        stats = validator.get_statistics()
        print(f"\nStatistics: {stats}")

    # Run example
    asyncio.run(main())
