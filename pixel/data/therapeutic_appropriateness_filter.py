"""
Therapeutic Appropriateness Filtering System

Filters therapeutic conversations and responses based on clinical appropriateness,
safety considerations, ethical guidelines, and therapeutic best practices.
Ensures only clinically sound content is used for training and generation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilterDecision(Enum):
    """Filter decision outcomes"""
    ACCEPT = "accept"
    REJECT = "reject"
    MODIFY = "modify"
    REVIEW = "review"


class FilterReason(Enum):
    """Reasons for filtering decisions"""
    CLINICALLY_APPROPRIATE = "clinically_appropriate"
    SAFETY_CONCERN = "safety_concern"
    ETHICAL_VIOLATION = "ethical_violation"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    CONTRAINDICATED_INTERVENTION = "contraindicated_intervention"
    PREMATURE_INTERVENTION = "premature_intervention"
    BOUNDARY_VIOLATION = "boundary_violation"
    CULTURAL_INSENSITIVITY = "cultural_insensitivity"
    EVIDENCE_CONTRADICTION = "evidence_contradiction"
    DIAGNOSTIC_ERROR = "diagnostic_error"
    CRISIS_MISMANAGEMENT = "crisis_mismanagement"
    QUALITY_INSUFFICIENT = "quality_insufficient"


class ContentType(Enum):
    """Types of content being filtered"""
    CONVERSATION = "conversation"
    THERAPIST_RESPONSE = "therapist_response"
    CLIENT_SCENARIO = "client_scenario"
    INTERVENTION = "intervention"
    DIAGNOSTIC_CONTENT = "diagnostic_content"
    EDUCATIONAL_CONTENT = "educational_content"


class FilterSeverity(Enum):
    """Severity levels for filter violations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class FilterViolation:
    """Individual filter violation"""
    violation_id: str
    violation_type: FilterReason
    severity: FilterSeverity
    description: str
    location: str
    evidence: str
    suggested_modification: str
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate filter violation"""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")


@dataclass
class FilterResult:
    """Result of therapeutic appropriateness filtering"""
    content_id: str
    content_type: ContentType
    decision: FilterDecision
    confidence_score: float
    violations: List[FilterViolation]
    modifications_suggested: List[str]
    approval_reasons: List[str]
    safety_score: float
    ethical_score: float
    clinical_score: float
    overall_appropriateness_score: float
    filter_timestamp: datetime
    reviewer_notes: str = ""
    
    def __post_init__(self):
        """Validate filter result"""
        scores = [self.confidence_score, self.safety_score, self.ethical_score, 
                 self.clinical_score, self.overall_appropriateness_score]
        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError("All scores must be between 0.0 and 1.0")


class TherapeuticAppropriatenessFilter:
    """
    Filters therapeutic content for clinical appropriateness
    
    This system evaluates and filters therapeutic conversations, responses,
    and interventions to ensure they meet clinical standards for:
    - Safety and risk management
    - Ethical compliance
    - Clinical appropriateness
    - Evidence-based practice
    - Cultural sensitivity
    - Professional boundaries
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize therapeutic appropriateness filter"""
        self.config = self._load_configuration(config_path)
        self.safety_filters = self._initialize_safety_filters()
        self.ethical_filters = self._initialize_ethical_filters()
        self.clinical_filters = self._initialize_clinical_filters()
        self.intervention_filters = self._initialize_intervention_filters()
        self.content_patterns = self._initialize_content_patterns()
        self.filter_history: List[FilterResult] = []
        
        logger.info("Therapeutic Appropriateness Filter initialized")
    
    def _load_configuration(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load filter configuration"""
        default_config = {
            'filter_weights': {
                'safety_score': 0.4,
                'ethical_score': 0.3,
                'clinical_score': 0.3
            },
            'acceptance_thresholds': {
                'overall_minimum': 0.7,
                'safety_minimum': 0.8,
                'ethical_minimum': 0.8,
                'clinical_minimum': 0.6
            },
            'severity_thresholds': {
                'critical_reject': 0.9,
                'high_review': 0.7,
                'medium_modify': 0.5
            },
            'auto_reject_violations': [
                'safety_concern', 'ethical_violation', 'crisis_mismanagement'
            ],
            'require_review_violations': [
                'contraindicated_intervention', 'boundary_violation'
            ],
            'enable_modifications': True,
            'strict_mode': False
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_safety_filters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize safety-related filters"""
        filters = {
            'crisis_indicators': {
                'patterns': [
                    r'\b(suicide|kill\s+myself|end\s+my\s+life|want\s+to\s+die)\b',
                    r'\b(self\s*harm|cut\s+myself|hurt\s+myself)\b',
                    r'\b(hopeless|no\s+point|give\s+up|no\s+future)\b',
                    r'\b(overdose|pills|poison|jump)\b'
                ],
                'severity': FilterSeverity.CRITICAL,
                'action': 'require_safety_response'
            },
            'violence_indicators': {
                'patterns': [
                    r'\b(kill|murder|hurt\s+someone|violence)\b',
                    r'\b(weapon|gun|knife|attack)\b',
                    r'\b(rage|fury|explosive\s+anger)\b'
                ],
                'severity': FilterSeverity.HIGH,
                'action': 'assess_violence_risk'
            },
            'substance_abuse': {
                'patterns': [
                    r'\b(overdose|drugs|alcohol|drinking\s+problem)\b',
                    r'\b(addiction|substance|abuse|dependency)\b',
                    r'\b(high|drunk|intoxicated|withdrawal)\b'
                ],
                'severity': FilterSeverity.MEDIUM,
                'action': 'assess_substance_risk'
            },
            'child_safety': {
                'patterns': [
                    r'\b(child\s+abuse|neglect|inappropriate\s+touch)\b',
                    r'\b(minor|underage|child\s+safety)\b'
                ],
                'severity': FilterSeverity.CRITICAL,
                'action': 'mandatory_reporting'
            }
        }
        
        return filters
    
    def _initialize_ethical_filters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize ethical compliance filters"""
        filters = {
            'boundary_violations': {
                'patterns': [
                    r'\b(personal\s+relationship|date|romantic)\b',
                    r'\b(my\s+personal|let\s+me\s+tell\s+you\s+about\s+my)\b',
                    r'\b(meet\s+outside|social\s+media|friend\s+request)\b',
                    r'\b(gift|money|favor|special\s+arrangement)\b'
                ],
                'severity': FilterSeverity.HIGH,
                'action': 'boundary_education'
            },
            'confidentiality_breaches': {
                'patterns': [
                    r'\b(other\s+client|told\s+someone\s+about)\b',
                    r'\b(shared\s+information|discussed\s+with)\b',
                    r'\b(my\s+supervisor\s+said|team\s+meeting)\b'
                ],
                'severity': FilterSeverity.CRITICAL,
                'action': 'confidentiality_review'
            },
            'competence_overreach': {
                'patterns': [
                    r'\b(medical\s+advice|prescribe|medication\s+recommendation)\b',
                    r'\b(legal\s+advice|court|lawsuit)\b',
                    r'\b(financial\s+advice|investment|money\s+management)\b'
                ],
                'severity': FilterSeverity.HIGH,
                'action': 'scope_clarification'
            },
            'discrimination': {
                'patterns': [
                    r'\b(stereotype|prejudice|discriminat)\b',
                    r'\b(all\s+[race|gender|religion].*are)\b',
                    r'\b(typical\s+[group]|those\s+people)\b'
                ],
                'severity': FilterSeverity.HIGH,
                'action': 'cultural_sensitivity_training'
            }
        }
        
        return filters
    
    def _initialize_clinical_filters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize clinical appropriateness filters"""
        filters = {
            'diagnostic_errors': {
                'patterns': [
                    r'\b(you\s+have|diagnosed\s+with|definitely)\b.*\b(disorder|illness)\b',
                    r'\b(clearly|obviously|certainly).*\b(bipolar|schizophrenia|personality)\b'
                ],
                'severity': FilterSeverity.HIGH,
                'action': 'diagnostic_review'
            },
            'premature_interventions': {
                'patterns': [
                    r'\b(deep\s+trauma|childhood\s+abuse).*\b(first\s+session|just\s+met)\b',
                    r'\b(interpretation|unconscious).*\b(early|beginning)\b'
                ],
                'severity': FilterSeverity.MEDIUM,
                'action': 'timing_review'
            },
            'contraindicated_techniques': {
                'patterns': [
                    r'\b(exposure\s+therapy).*\b(psychosis|mania|severe\s+depression)\b',
                    r'\b(cognitive\s+restructuring).*\b(cognitive\s+impairment)\b'
                ],
                'severity': FilterSeverity.HIGH,
                'action': 'technique_review'
            },
            'unsupported_claims': {
                'patterns': [
                    r'\b(always\s+works|guaranteed|cure|never\s+fails)\b',
                    r'\b(miracle|magic|instant\s+results)\b'
                ],
                'severity': FilterSeverity.MEDIUM,
                'action': 'evidence_review'
            }
        }
        
        return filters
    
    def _initialize_intervention_filters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize intervention-specific filters"""
        filters = {
            'cbt_appropriateness': {
                'appropriate_conditions': [
                    'depression', 'anxiety', 'panic', 'ocd', 'ptsd'
                ],
                'contraindications': [
                    'active_psychosis', 'severe_cognitive_impairment', 'acute_mania'
                ],
                'timing_requirements': [
                    'alliance_established', 'crisis_stabilized'
                ]
            },
            'dbt_appropriateness': {
                'appropriate_conditions': [
                    'borderline_personality', 'emotion_dysregulation', 'self_harm'
                ],
                'contraindications': [
                    'active_substance_dependence', 'severe_cognitive_impairment'
                ],
                'timing_requirements': [
                    'commitment_obtained', 'distress_tolerance_assessed'
                ]
            },
            'psychodynamic_appropriateness': {
                'appropriate_conditions': [
                    'personality_disorders', 'relationship_issues', 'insight_goals'
                ],
                'contraindications': [
                    'acute_crisis', 'severe_symptom_focus', 'limited_insight'
                ],
                'timing_requirements': [
                    'strong_alliance', 'psychological_mindedness', 'stability'
                ]
            }
        }
        
        return filters
    
    def _initialize_content_patterns(self) -> Dict[str, List[str]]:
        """Initialize content pattern recognition"""
        patterns = {
            'inappropriate_language': [
                r'\b(fuck|shit|damn|hell)\b',
                r'\b(stupid|idiot|crazy|insane)\b',
                r'\b(retard|psycho|nuts|mental)\b'
            ],
            'unprofessional_tone': [
                r'\b(whatever|duh|obviously|come\s+on)\b',
                r'\b(get\s+over\s+it|just\s+stop|snap\s+out)\b',
                r'\b(that\'s\s+ridiculous|don\'t\s+be\s+silly)\b'
            ],
            'medical_overreach': [
                r'\b(take\s+this\s+medication|stop\s+your\s+meds)\b',
                r'\b(you\s+need\s+surgery|medical\s+procedure)\b',
                r'\b(physical\s+symptoms|medical\s+condition)\b'
            ],
            'legal_overreach': [
                r'\b(sue|lawsuit|legal\s+action)\b',
                r'\b(divorce|custody|court\s+case)\b',
                r'\b(rights|legal\s+advice|attorney)\b'
            ]
        }
        
        return patterns
    
    async def filter_content(
        self,
        content_id: str,
        content: Union[str, List[Any]],
        content_type: ContentType,
        clinical_context: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilterResult:
        """Filter content for therapeutic appropriateness"""
        try:
            logger.info(f"Filtering content {content_id} of type {content_type.value}")
            
            # Initialize filtering components
            violations: List[FilterViolation] = []
            modifications_suggested: List[str] = []
            approval_reasons: List[str] = []
            
            # Convert content to text for analysis
            content_text = self._extract_text_content(content, content_type)
            
            # Apply safety filters
            safety_violations = await self._apply_safety_filters(content_text, clinical_context)
            violations.extend(safety_violations)
            
            # Apply ethical filters
            ethical_violations = await self._apply_ethical_filters(content_text, clinical_context)
            violations.extend(ethical_violations)
            
            # Apply clinical filters
            clinical_violations = await self._apply_clinical_filters(content_text, clinical_context, content_type)
            violations.extend(clinical_violations)
            
            # Calculate component scores
            safety_score = self._calculate_safety_score(safety_violations)
            ethical_score = self._calculate_ethical_score(ethical_violations)
            clinical_score = self._calculate_clinical_score(clinical_violations)
            
            # Calculate overall appropriateness score
            overall_score = self._calculate_overall_score(safety_score, ethical_score, clinical_score)
            
            # Make filtering decision
            decision = self._make_filter_decision(violations, overall_score, safety_score, ethical_score, clinical_score)
            
            # Generate modifications if needed
            if decision in [FilterDecision.MODIFY, FilterDecision.REVIEW]:
                modifications_suggested = self._generate_modifications(violations, content_text)
            
            # Generate approval reasons for accepted content
            if decision == FilterDecision.ACCEPT:
                approval_reasons = self._generate_approval_reasons(safety_score, ethical_score, clinical_score)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(violations, overall_score)
            
            # Create filter result
            result = FilterResult(
                content_id=content_id,
                content_type=content_type,
                decision=decision,
                confidence_score=confidence_score,
                violations=violations,
                modifications_suggested=modifications_suggested,
                approval_reasons=approval_reasons,
                safety_score=safety_score,
                ethical_score=ethical_score,
                clinical_score=clinical_score,
                overall_appropriateness_score=overall_score,
                filter_timestamp=datetime.now()
            )
            
            # Store in history
            self.filter_history.append(result)
            
            logger.info(f"Content {content_id} filtered: {decision.value} (score: {overall_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error filtering content {content_id}: {e}")
            return self._create_error_result(content_id, content_type, str(e))
    
    def _extract_text_content(self, content: Union[str, List[Any]], content_type: ContentType) -> str:
        """Extract text content for analysis"""
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            # Extract text from conversation turns
            text_parts = []
            for item in content:
                if hasattr(item, 'content'):
                    text_parts.append(getattr(item, 'content', ''))
                elif isinstance(item, dict):
                    text_parts.append(item.get('content', ''))
                elif isinstance(item, str):
                    text_parts.append(item)
            
            return ' '.join(text_parts)
        
        return str(content)
    
    async def _apply_safety_filters(self, content_text: str, clinical_context: Any) -> List[FilterViolation]:
        """Apply safety-related filters"""
        violations = []
        
        for filter_name, filter_config in self.safety_filters.items():
            patterns = filter_config.get('patterns', [])
            severity = filter_config.get('severity', FilterSeverity.MEDIUM)
            
            for pattern in patterns:
                matches = re.finditer(pattern, content_text, re.IGNORECASE)
                for match in matches:
                    violation = FilterViolation(
                        violation_id=f"safety_{filter_name}_{len(violations)}",
                        violation_type=FilterReason.SAFETY_CONCERN,
                        severity=severity,
                        description=f"Safety concern detected: {filter_name}",
                        location=f"position_{match.start()}-{match.end()}",
                        evidence=match.group(),
                        suggested_modification=self._get_safety_modification(filter_name, match.group()),
                        confidence_score=0.8
                    )
                    violations.append(violation)
        
        return violations
    
    async def _apply_ethical_filters(self, content_text: str, clinical_context: Any) -> List[FilterViolation]:
        """Apply ethical compliance filters"""
        violations = []
        
        for filter_name, filter_config in self.ethical_filters.items():
            patterns = filter_config.get('patterns', [])
            severity = filter_config.get('severity', FilterSeverity.MEDIUM)
            
            for pattern in patterns:
                matches = re.finditer(pattern, content_text, re.IGNORECASE)
                for match in matches:
                    violation = FilterViolation(
                        violation_id=f"ethical_{filter_name}_{len(violations)}",
                        violation_type=FilterReason.ETHICAL_VIOLATION,
                        severity=severity,
                        description=f"Ethical concern detected: {filter_name}",
                        location=f"position_{match.start()}-{match.end()}",
                        evidence=match.group(),
                        suggested_modification=self._get_ethical_modification(filter_name, match.group()),
                        confidence_score=0.7
                    )
                    violations.append(violation)
        
        return violations
    
    async def _apply_clinical_filters(self, content_text: str, clinical_context: Any, content_type: ContentType) -> List[FilterViolation]:
        """Apply clinical appropriateness filters"""
        violations = []
        
        for filter_name, filter_config in self.clinical_filters.items():
            patterns = filter_config.get('patterns', [])
            severity = filter_config.get('severity', FilterSeverity.MEDIUM)
            
            for pattern in patterns:
                matches = re.finditer(pattern, content_text, re.IGNORECASE)
                for match in matches:
                    violation = FilterViolation(
                        violation_id=f"clinical_{filter_name}_{len(violations)}",
                        violation_type=FilterReason.INAPPROPRIATE_CONTENT,
                        severity=severity,
                        description=f"Clinical concern detected: {filter_name}",
                        location=f"position_{match.start()}-{match.end()}",
                        evidence=match.group(),
                        suggested_modification=self._get_clinical_modification(filter_name, match.group()),
                        confidence_score=0.6
                    )
                    violations.append(violation)
        
        # Additional content pattern checks
        for pattern_name, patterns in self.content_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content_text, re.IGNORECASE)
                for match in matches:
                    violation = FilterViolation(
                        violation_id=f"content_{pattern_name}_{len(violations)}",
                        violation_type=FilterReason.INAPPROPRIATE_CONTENT,
                        severity=FilterSeverity.MEDIUM,
                        description=f"Inappropriate content: {pattern_name}",
                        location=f"position_{match.start()}-{match.end()}",
                        evidence=match.group(),
                        suggested_modification=self._get_content_modification(pattern_name, match.group()),
                        confidence_score=0.5
                    )
                    violations.append(violation)
        
        return violations
    
    def _calculate_safety_score(self, safety_violations: List[FilterViolation]) -> float:
        """Calculate safety score based on violations"""
        if not safety_violations:
            return 1.0
        
        # Weight violations by severity
        total_deduction = 0.0
        for violation in safety_violations:
            if violation.severity == FilterSeverity.CRITICAL:
                total_deduction += 0.5
            elif violation.severity == FilterSeverity.HIGH:
                total_deduction += 0.3
            elif violation.severity == FilterSeverity.MEDIUM:
                total_deduction += 0.2
            else:
                total_deduction += 0.1
        
        return max(0.0, 1.0 - total_deduction)
    
    def _calculate_ethical_score(self, ethical_violations: List[FilterViolation]) -> float:
        """Calculate ethical score based on violations"""
        if not ethical_violations:
            return 1.0
        
        total_deduction = 0.0
        for violation in ethical_violations:
            if violation.severity == FilterSeverity.CRITICAL:
                total_deduction += 0.4
            elif violation.severity == FilterSeverity.HIGH:
                total_deduction += 0.25
            elif violation.severity == FilterSeverity.MEDIUM:
                total_deduction += 0.15
            else:
                total_deduction += 0.1
        
        return max(0.0, 1.0 - total_deduction)
    
    def _calculate_clinical_score(self, clinical_violations: List[FilterViolation]) -> float:
        """Calculate clinical score based on violations"""
        if not clinical_violations:
            return 1.0
        
        total_deduction = 0.0
        for violation in clinical_violations:
            if violation.severity == FilterSeverity.CRITICAL:
                total_deduction += 0.3
            elif violation.severity == FilterSeverity.HIGH:
                total_deduction += 0.2
            elif violation.severity == FilterSeverity.MEDIUM:
                total_deduction += 0.15
            else:
                total_deduction += 0.1
        
        return max(0.0, 1.0 - total_deduction)
    
    def _calculate_overall_score(self, safety_score: float, ethical_score: float, clinical_score: float) -> float:
        """Calculate overall appropriateness score"""
        weights = self.config['filter_weights']
        
        weighted_score = (
            safety_score * weights['safety_score'] +
            ethical_score * weights['ethical_score'] +
            clinical_score * weights['clinical_score']
        )
        
        return weighted_score
    
    def _make_filter_decision(
        self,
        violations: List[FilterViolation],
        overall_score: float,
        safety_score: float,
        ethical_score: float,
        clinical_score: float
    ) -> FilterDecision:
        """Make filtering decision based on scores and violations"""
        thresholds = self.config['acceptance_thresholds']
        auto_reject = self.config['auto_reject_violations']
        require_review = self.config['require_review_violations']
        
        # Check for auto-reject violations
        for violation in violations:
            if violation.violation_type.value in auto_reject:
                return FilterDecision.REJECT
        
        # Check minimum thresholds
        if (safety_score < thresholds['safety_minimum'] or
            ethical_score < thresholds['ethical_minimum'] or
            overall_score < thresholds['overall_minimum']):
            return FilterDecision.REJECT
        
        # Check for review requirements
        for violation in violations:
            if violation.violation_type.value in require_review:
                return FilterDecision.REVIEW
        
        # Check for modification needs
        if clinical_score < thresholds['clinical_minimum']:
            return FilterDecision.MODIFY
        
        # Check severity thresholds
        critical_violations = [v for v in violations if v.severity == FilterSeverity.CRITICAL]
        high_violations = [v for v in violations if v.severity == FilterSeverity.HIGH]
        
        if critical_violations:
            return FilterDecision.REJECT
        elif high_violations and len(high_violations) > 2:
            return FilterDecision.REVIEW
        elif len(violations) > 5:
            return FilterDecision.MODIFY
        
        return FilterDecision.ACCEPT
    
    def _generate_modifications(self, violations: List[FilterViolation], content_text: str) -> List[str]:
        """Generate suggested modifications"""
        modifications = []
        
        # Group violations by type
        violation_groups = {}
        for violation in violations:
            violation_type = violation.violation_type.value
            if violation_type not in violation_groups:
                violation_groups[violation_type] = []
            violation_groups[violation_type].append(violation)
        
        # Generate modifications for each type
        for violation_type, group_violations in violation_groups.items():
            if violation_type == 'safety_concern':
                modifications.append("Add appropriate safety assessment and planning")
                modifications.append("Include crisis intervention protocols")
            elif violation_type == 'ethical_violation':
                modifications.append("Review and correct ethical compliance issues")
                modifications.append("Ensure professional boundaries are maintained")
            elif violation_type == 'inappropriate_content':
                modifications.append("Replace inappropriate language with professional alternatives")
                modifications.append("Ensure clinical accuracy and appropriateness")
        
        # Add specific modifications from violations
        for violation in violations[:5]:  # Top 5 violations
            if violation.suggested_modification not in modifications:
                modifications.append(violation.suggested_modification)
        
        return modifications[:10]  # Limit to 10 modifications
    
    def _generate_approval_reasons(self, safety_score: float, ethical_score: float, clinical_score: float) -> List[str]:
        """Generate reasons for content approval"""
        reasons = []
        
        if safety_score >= 0.9:
            reasons.append("Excellent safety considerations")
        elif safety_score >= 0.8:
            reasons.append("Appropriate safety measures")
        
        if ethical_score >= 0.9:
            reasons.append("Strong ethical compliance")
        elif ethical_score >= 0.8:
            reasons.append("Appropriate ethical standards")
        
        if clinical_score >= 0.9:
            reasons.append("High clinical appropriateness")
        elif clinical_score >= 0.8:
            reasons.append("Clinically appropriate content")
        
        if not reasons:
            reasons.append("Meets minimum appropriateness standards")
        
        return reasons
    
    def _calculate_confidence_score(self, violations: List[FilterViolation], overall_score: float) -> float:
        """Calculate confidence in filtering decision"""
        if not violations:
            return 0.9
        
        # Average confidence of violations
        avg_violation_confidence = sum(v.confidence_score for v in violations) / len(violations)
        
        # Adjust based on overall score
        score_confidence = overall_score * 0.5 + 0.5
        
        # Combine confidences
        combined_confidence = (avg_violation_confidence * 0.6) + (score_confidence * 0.4)
        
        return min(1.0, max(0.1, combined_confidence))
    
    def _get_safety_modification(self, filter_name: str, evidence: str) -> str:
        """Get safety-specific modification suggestion"""
        modifications = {
            'crisis_indicators': "Include immediate safety assessment and crisis intervention",
            'violence_indicators': "Add violence risk assessment and safety planning",
            'substance_abuse': "Include substance abuse assessment and referral options",
            'child_safety': "Follow mandatory reporting protocols and safety procedures"
        }
        
        return modifications.get(filter_name, "Address safety concerns appropriately")
    
    def _get_ethical_modification(self, filter_name: str, evidence: str) -> str:
        """Get ethical-specific modification suggestion"""
        modifications = {
            'boundary_violations': "Maintain appropriate professional boundaries",
            'confidentiality_breaches': "Ensure confidentiality is protected",
            'competence_overreach': "Stay within scope of practice",
            'discrimination': "Use culturally sensitive and inclusive language"
        }
        
        return modifications.get(filter_name, "Address ethical concerns")
    
    def _get_clinical_modification(self, filter_name: str, evidence: str) -> str:
        """Get clinical-specific modification suggestion"""
        modifications = {
            'diagnostic_errors': "Use tentative language and avoid definitive diagnoses",
            'premature_interventions': "Build rapport before deep exploration",
            'contraindicated_techniques': "Select appropriate interventions for presentation",
            'unsupported_claims': "Use evidence-based language and realistic expectations"
        }
        
        return modifications.get(filter_name, "Improve clinical appropriateness")
    
    def _get_content_modification(self, pattern_name: str, evidence: str) -> str:
        """Get content-specific modification suggestion"""
        modifications = {
            'inappropriate_language': "Use professional and respectful language",
            'unprofessional_tone': "Maintain therapeutic and supportive tone",
            'medical_overreach': "Refer to appropriate medical professionals",
            'legal_overreach': "Refer to appropriate legal professionals"
        }
        
        return modifications.get(pattern_name, "Improve content appropriateness")
    
    def _create_error_result(self, content_id: str, content_type: ContentType, error_message: str) -> FilterResult:
        """Create error result when filtering fails"""
        return FilterResult(
            content_id=content_id,
            content_type=content_type,
            decision=FilterDecision.REJECT,
            confidence_score=0.1,
            violations=[],
            modifications_suggested=[f"Review filtering error: {error_message}"],
            approval_reasons=[],
            safety_score=0.0,
            ethical_score=0.0,
            clinical_score=0.0,
            overall_appropriateness_score=0.0,
            filter_timestamp=datetime.now(),
            reviewer_notes=f"Filtering error: {error_message}"
        )
    
    async def batch_filter_content(
        self,
        content_items: List[Dict[str, Any]]
    ) -> List[FilterResult]:
        """Filter multiple content items in batch"""
        results = []
        
        for item in content_items:
            try:
                result = await self.filter_content(
                    content_id=item.get('content_id', f'batch_{len(results)}'),
                    content=item.get('content', ''),
                    content_type=ContentType(item.get('content_type', 'conversation')),
                    clinical_context=item.get('clinical_context'),
                    metadata=item.get('metadata')
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch filtering item {len(results)}: {e}")
                error_result = self._create_error_result(
                    f'batch_{len(results)}',
                    ContentType.CONVERSATION,
                    str(e)
                )
                results.append(error_result)
        
        return results
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        if not self.filter_history:
            return {
                'total_filtered': 0,
                'decision_distribution': {},
                'average_scores': {},
                'common_violations': {},
                'content_type_distribution': {}
            }
        
        # Decision distribution
        decisions = [result.decision.value for result in self.filter_history]
        decision_counts = {decision: decisions.count(decision) for decision in set(decisions)}
        
        # Average scores
        safety_scores = [result.safety_score for result in self.filter_history]
        ethical_scores = [result.ethical_score for result in self.filter_history]
        clinical_scores = [result.clinical_score for result in self.filter_history]
        overall_scores = [result.overall_appropriateness_score for result in self.filter_history]
        
        # Common violations
        all_violations = []
        for result in self.filter_history:
            all_violations.extend(result.violations)
        
        violation_types = {}
        for violation in all_violations:
            violation_type = violation.violation_type.value
            violation_types[violation_type] = violation_types.get(violation_type, 0) + 1
        
        # Content type distribution
        content_types = [result.content_type.value for result in self.filter_history]
        content_type_counts = {ct: content_types.count(ct) for ct in set(content_types)}
        
        return {
            'total_filtered': len(self.filter_history),
            'decision_distribution': decision_counts,
            'average_scores': {
                'safety': sum(safety_scores) / len(safety_scores),
                'ethical': sum(ethical_scores) / len(ethical_scores),
                'clinical': sum(clinical_scores) / len(clinical_scores),
                'overall': sum(overall_scores) / len(overall_scores)
            },
            'common_violations': dict(sorted(violation_types.items(), key=lambda x: x[1], reverse=True)[:10]),
            'content_type_distribution': content_type_counts
        }
    
    def export_filter_data(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export filter data"""
        data = {
            'configuration': self.config,
            'filter_history': [
                {
                    'content_id': result.content_id,
                    'content_type': result.content_type.value,
                    'decision': result.decision.value,
                    'confidence_score': result.confidence_score,
                    'safety_score': result.safety_score,
                    'ethical_score': result.ethical_score,
                    'clinical_score': result.clinical_score,
                    'overall_appropriateness_score': result.overall_appropriateness_score,
                    'violations_count': len(result.violations),
                    'modifications_count': len(result.modifications_suggested),
                    'approval_reasons': result.approval_reasons,
                    'filter_timestamp': result.filter_timestamp.isoformat(),
                    'violations': [
                        {
                            'violation_type': v.violation_type.value,
                            'severity': v.severity.value,
                            'description': v.description,
                            'confidence_score': v.confidence_score
                        }
                        for v in result.violations
                    ]
                }
                for result in self.filter_history
            ],
            'statistics': self.get_filter_statistics()
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2)
        else:
            return data


# Example usage and testing
if __name__ == "__main__":
    async def test_therapeutic_appropriateness_filter():
        """Test the therapeutic appropriateness filter"""
        filter_system = TherapeuticAppropriatenessFilter()
        
        # Test appropriate content
        appropriate_content = "How are you feeling today? I'd like to understand your experience better."
        
        result = await filter_system.filter_content(
            content_id="test_001",
            content=appropriate_content,
            content_type=ContentType.THERAPIST_RESPONSE
        )
        
        print(f"Appropriate content result: {result.decision.value} (score: {result.overall_appropriateness_score:.3f})")
        
        # Test inappropriate content
        inappropriate_content = "You're clearly bipolar and need to just get over it. Take these pills I recommend."
        
        result = await filter_system.filter_content(
            content_id="test_002",
            content=inappropriate_content,
            content_type=ContentType.THERAPIST_RESPONSE
        )
        
        print(f"Inappropriate content result: {result.decision.value} (score: {result.overall_appropriateness_score:.3f})")
        print(f"Violations: {len(result.violations)}")
        print(f"Modifications: {result.modifications_suggested}")
        
        # Get statistics
        stats = filter_system.get_filter_statistics()
        print(f"Filter statistics: {stats}")
    
    # Run test
    asyncio.run(test_therapeutic_appropriateness_filter())
