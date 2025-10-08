"""
Enhanced safety and content filtering system for Pixelated Empathy AI project.
Ensures all inference outputs pass rigorous safety checks before being returned.
"""

import re
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


logger = logging.getLogger(__name__)


class SafetyCategory(Enum):
    """Categories of safety concerns"""
    CRISIS = "crisis"
    HARM = "harm"
    TOXICITY = "toxicity"
    PRIVACY = "privacy"
    INAPPROPRIATE = "inappropriate"
    BIASED = "biased"
    LEGAL = "legal"
    MEDICAL = "medical"


class SafetyLevel(Enum):
    """Levels of safety filtering"""
    LENIENT = "lenient"      # Minimal filtering
    MODERATE = "moderate"    # Standard filtering
    STRICT = "strict"        # Aggressive filtering
    PARANOID = "paranoid"    # Maximum filtering


@dataclass
class SafetyCheckResult:
    """Result of a safety check"""
    is_safe: bool
    overall_score: float  # 0.0 to 1.0, where 1.0 is completely safe
    category_scores: Dict[SafetyCategory, float]
    flagged_categories: List[SafetyCategory]
    confidence: float  # Confidence in the safety assessment
    explanation: str
    filtered_content: Optional[str] = None
    redacted_sections: Optional[List[Dict[str, Any]]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CrisisDetectionResult:
    """Result of crisis detection"""
    is_crisis: bool
    crisis_type: Optional[str]
    confidence: float
    urgency_level: str  # low, medium, high, immediate
    recommended_action: str
    resources: Optional[List[str]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class EnhancedSafetyFilter:
    """Enhanced safety filtering system with multiple layers of protection"""
    
    def __init__(self, safety_level: SafetyLevel = SafetyLevel.MODERATE):
        self.safety_level = safety_level
        self.logger = logging.getLogger(__name__)
        
        # Initialize safety models
        self.toxicity_classifier = None
        self.bias_classifier = None
        self.medical_advice_classifier = None
        
        # Initialize crisis detection patterns
        self.crisis_patterns = self._initialize_crisis_patterns()
        self.harm_patterns = self._initialize_harm_patterns()
        self.privacy_patterns = self._initialize_privacy_patterns()
        self.inappropriate_patterns = self._initialize_inappropriate_patterns()
        
        self._init_safety_models()
    
    def _init_safety_models(self):
        """Initialize safety classification models"""
        try:
            # Toxicity classifier
            self.toxicity_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                tokenizer="unitary/toxic-bert"
            )
            self.logger.info("Toxicity classifier loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load toxicity classifier: {e}")
            self.toxicity_classifier = None
        
        try:
            # Bias classifier (simplified - you'd use a proper bias detection model)
            self.bias_classifier = pipeline(
                "text-classification",
                model="d4data/bias-detection-model" if torch.cuda.is_available() else None,
                tokenizer="d4data/bias-detection-model" if torch.cuda.is_available() else None
            )
            self.logger.info("Bias classifier loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load bias classifier: {e}")
            self.bias_classifier = None
    
    def _initialize_crisis_patterns(self) -> Dict[str, List[str]]:
        """Initialize crisis detection patterns"""
        return {
            'suicide': [
                r'\bkill myself\b', r'\bsuicide\b', r'\bharm myself\b', r'\bend it all\b',
                r'\bnot want to live\b', r'\boverdose\b', r'\bcut\b', r'\bhurt myself\b',
                r'\bself-harm\b', r'\bself injury\b', r'\bhanging\b', r'\bjump off\b'
            ],
            'harm_to_others': [
                r'\bkill.*[a-z ]*other\b', r'\bhurt.*people\b', r'\bviolence\b', r'\battack\b',
                r'\bharm.*others\b', r'\bmurder\b', r'\bphysically.*harm\b', r'\bhurt.*someone\b'
            ],
            'substance_abuse': [
                r'\boverdose\b', r'\bdrugs\b', r'\balcohol\b', r'\bsubstance\b',
                r'\baddiction\b', r '\babuse\b', r '\brehab\b', r '\bdetox\b'
            ],
            'eating_disorders': [
                r '\banorexia\b', r '\bbulimia\b', r '\beating disorder\b', r '\bpurge\b',
                r '\bvomit\b', r '\bstomach pump\b', r '\bthin\b', r '\bskinny\b'
            ]
        }
    
    def _initialize_harm_patterns(self) -> Dict[str, List[str]]:
        """Initialize harm detection patterns"""
        return {
            'physical_harm': [
                r '\bhit\b', r '\bpunch\b', r '\bkick\b', r '\bstrike\b',
                r '\bbeat\b', r '\binjure\b', r '\bhurt\b', r '\bdamage\b'
            ],
            'emotional_harm': [
                r '\bhurt feelings\b', r '\bemotional abuse\b', r '\bmanipulat\b',
                r '\bgaslight\b', r '\bcontrol\b', r '\bdominat\b', r '\bintimidat\b'
            ],
            'sexual_harm': [
                r '\bsexual\b', r '\bharass\b', r '\bgrope\b', r '\btouch\b.*\binappropriate\b',
                r '\binappropriate contact\b', r '\bnon-consensual\b'
            ]
        }
    
    def _initialize_privacy_patterns(self) -> Dict[str, str]:
        """Initialize privacy detection patterns"""
        return {
            'ssn': r '\b\d{3}-\d{2}-\d{4}\b',
            'email': r '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r '\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'address': r '\b\d{1,4} [A-Za-z ]+ [A-Za-z. ]+\b',
            'credit_card': r '\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b',
            'medical_record': r '\b[A-Z]{2,3}\d{2,}\b'  # Simplified medical record pattern
        }
    
    def _initialize_inappropriate_patterns(self) -> List[str]:
        """Initialize inappropriate content patterns"""
        return [
            r '\bexplicit\b', r '\bporn\b', r '\badult content\b', r '\bnudity\b',
            r '\bsexual content\b', r '\berotic\b', r '\bxxx\b', r '\badult\b',
            r '\bviolence\b.*\bdetail\b', r '\bgore\b', r '\bgraphic\b.*\bviolence\b'
        ]
    
    def check_output_safety(self, 
                           content: str,
                           user_context: Optional[Dict[str, Any]] = None,
                           request_metadata: Optional[Dict[str, Any]] = None) -> SafetyCheckResult:
        """Check safety of model output content"""
        start_time = time.time()
        
        if not content or not isinstance(content, str):
            return SafetyCheckResult(
                is_safe=True,
                overall_score=1.0,
                category_scores={},
                flagged_categories=[],
                confidence=1.0,
                explanation="Empty or invalid content"
            )
        
        # Perform comprehensive safety checks
        category_scores = {}
        flagged_categories = []
        redacted_sections = []
        
        # Check for crisis content
        crisis_result = self._check_crisis_content(content)
        category_scores[SafetyCategory.CRISIS] = crisis_result.confidence if crisis_result.is_crisis else 0.0
        if crisis_result.is_crisis:
            flagged_categories.append(SafetyCategory.CRISIS)
            if crisis_result.urgency_level in ['high', 'immediate']:
                # For high urgency crisis, we must flag regardless of safety level
                pass
        
        # Check for harmful content
        harm_score = self._check_harm_content(content)
        category_scores[SafetyCategory.HARM] = harm_score
        if harm_score > self._get_threshold_for_category(SafetyCategory.HARM):
            flagged_categories.append(SafetyCategory.HARM)
        
        # Check for toxic content
        toxicity_score = self._check_toxicity_content(content)
        category_scores[SafetyCategory.TOXICITY] = toxicity_score
        if toxicity_score > self._get_threshold_for_category(SafetyCategory.TOXICITY):
            flagged_categories.append(SafetyCategory.TOXICITY)
        
        # Check for privacy violations
        privacy_violations = self._check_privacy_content(content)
        privacy_score = len(privacy_violations) * 0.3  # Weight privacy violations heavily
        category_scores[SafetyCategory.PRIVACY] = min(1.0, privacy_score)
        if privacy_violations:
            flagged_categories.append(SafetyCategory.PRIVACY)
            redacted_sections.extend(privacy_violations)
        
        # Check for inappropriate content
        inappropriate_score = self._check_inappropriate_content(content)
        category_scores[SafetyCategory.INAPPROPRIATE] = inappropriate_score
        if inappropriate_score > self._get_threshold_for_category(SafetyCategory.INAPPROPRIATE):
            flagged_categories.append(SafetyCategory.INAPPROPRIATE)
        
        # Check for biased content
        bias_score = self._check_bias_content(content)
        category_scores[SafetyCategory.BIASED] = bias_score
        if bias_score > self._get_threshold_for_category(SafetyCategory.BIASED):
            flagged_categories.append(SafetyCategory.BIASED)
        
        # Check for legal/medical advice
        legal_medical_score = self._check_legal_medical_advisory(content)
        category_scores[SafetyCategory.LEGAL] = legal_medical_score
        category_scores[SafetyCategory.MEDICAL] = legal_medical_score
        if legal_medical_score > self._get_threshold_for_category(SafetyCategory.LEGAL):
            flagged_categories.append(SafetyCategory.LEGAL)
            flagged_categories.append(SafetyCategory.MEDICAL)
        
        # Calculate overall safety score
        max_category_score = max(category_scores.values()) if category_scores else 0.0
        overall_score = 1.0 - max_category_score
        
        # Determine if content is safe
        is_safe = overall_score >= self._get_safety_threshold()
        
        # Apply filtering/redaction if content is not safe
        filtered_content = content
        if not is_safe and redacted_sections:
            filtered_content = self._redact_content(content, redacted_sections)
        
        # Generate explanation
        explanation_parts = []
        if flagged_categories:
            explanation_parts.append(f"Flagged categories: {[cat.value for cat in flagged_categories]}")
            max_score_cat = max(category_scores.items(), key=lambda x: x[1])
            explanation_parts.append(f"Highest risk: {max_score_cat[0].value} ({max_score_cat[1]:.2f})")
        else:
            explanation_parts.append("No safety concerns detected")
        
        explanation = "; ".join(explanation_parts)
        
        processing_time = (time.time() - start_time) * 1000
        
        result = SafetyCheckResult(
            is_safe=is_safe,
            overall_score=overall_score,
            category_scores=category_scores,
            flagged_categories=flagged_categories,
            confidence=self._calculate_assessment_confidence(category_scores),
            explanation=explanation,
            filtered_content=filtered_content if not is_safe else None,
            redacted_sections=redacted_sections if redacted_sections else None,
            timestamp=datetime.utcnow().isoformat()
        )
        
        self.logger.debug(f"Safety check completed in {processing_time:.2f}ms - Safe: {is_safe}")
        
        return result
    
    def _check_crisis_content(self, content: str) -> CrisisDetectionResult:
        """Check for crisis-related content"""
        content_lower = content.lower()
        
        # Check each crisis pattern category
        for crisis_type, patterns in self.crisis_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    # Determine urgency based on context and keywords
                    urgency = self._determine_crisis_urgency(content_lower, crisis_type)
                    recommended_action = self._get_crisis_action(crisis_type, urgency)
                    resources = self._get_crisis_resources(crisis_type)
                    
                    return CrisisDetectionResult(
                        is_crisis=True,
                        crisis_type=crisis_type,
                        confidence=min(1.0, len(re.findall('|'.join(patterns), content_lower)) * 0.4),
                        urgency_level=urgency,
                        recommended_action=recommended_action,
                        resources=resources
                    )
        
        return CrisisDetectionResult(
            is_crisis=False,
            crisis_type=None,
            confidence=0.0,
            urgency_level="low",
            recommended_action="continue monitoring",
            resources=[]
        )
    
    def _determine_crisis_urgency(self, content: str, crisis_type: str) -> str:
        """Determine the urgency level of a crisis"""
        urgency_indicators = {
            'immediate': [r'\bnow\b', r'\bimmediately\b', r'\btonight\b', r'\btoday\b'],
            'high': [r'\bsoon\b', r'\blater\b', r'\btomorrow\b', r '\bthis week\b'],
            'medium': [r'\bweek\b', r '\bmonth\b', r '\bsoon\b', r '\bplanning\b'],
            'low': [r '\bthinking\b', r '\bwondering\b', r '\bconsidering\b', r '\bmaybe\b']
        }
        
        immediate_matches = sum(1 for pattern in urgency_indicators['immediate'] if re.search(pattern, content, re.IGNORECASE))
        if immediate_matches > 0:
            return 'immediate'
        
        high_matches = sum(1 for pattern in urgency_indicators['high'] if re.search(pattern, content, re.IGNORECASE))
        if high_matches > 0:
            return 'high'
        
        medium_matches = sum(1 for pattern in urgency_indicators['medium'] if re.search(pattern, content, re.IGNORECASE))
        if medium_matches > 0:
            return 'medium'
        
        return 'low'
    
    def _get_crisis_action(self, crisis_type: str, urgency: str) -> str:
        """Get recommended action for a crisis"""
        actions = {
            'suicide': {
                'immediate': 'Contact emergency services immediately',
                'high': 'Escalate to crisis intervention team',
                'medium': 'Schedule urgent follow-up',
                'low': 'Monitor closely and provide resources'
            },
            'harm_to_others': {
                'immediate': 'Contact law enforcement and emergency services',
                'high': 'Notify authorities and safety team',
                'medium': 'Increase supervision and monitoring',
                'low': 'Document and develop prevention plan'
            }
        }
        
        crisis_actions = actions.get(crisis_type, {})
        return crisis_actions.get(urgency, 'Continue monitoring and provide support')
    
    def _get_crisis_resources(self, crisis_type: str) -> List[str]:
        """Get crisis resources for a specific crisis type"""
        resources = {
            'suicide': [
                'National Suicide Prevention Lifeline: 988',
                'Crisis Text Line: Text HOME to 741741',
                'Emergency Services: 911'
            ],
            'harm_to_others': [
                'Law Enforcement: 911',
                'Local Crisis Center Hotline',
                'Mental Health Emergency Services'
            ]
        }
        
        return resources.get(crisis_type, ['Emergency Services: 911'])
    
    def _check_harm_content(self, content: str) -> float:
        """Check for harmful content"""
        content_lower = content.lower()
        harm_score = 0.0
        
        # Check each harm pattern category
        for harm_type, patterns in self.harm_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, content_lower, re.IGNORECASE))
            harm_score += matches * 0.2  # Weight each match
        
        return min(1.0, harm_score)
    
    def _check_toxicity_content(self, content: str) -> float:
        """Check for toxic content using classifier or keyword matching"""
        if self.toxicity_classifier:
            try:
                result = self.toxicity_classifier(content[:512])  # Limit length for model
                if result and isinstance(result, list) and len(result) > 0:
                    toxicity_result = result[0]
                    if toxicity_result['label'] == 'TOXIC':
                        return min(1.0, toxicity_result['score'])
            except Exception as e:
                self.logger.warning(f"Toxicity classifier error: {e}")
        
        # Fallback to keyword matching
        toxic_matches = sum(1 for pattern in self.harm_patterns['physical_harm'] 
                           if re.search(pattern, content.lower()))
        return min(1.0, toxic_matches * 0.1)
    
    def _check_privacy_content(self, content: str) -> List[Dict[str, Any]]:
        """Check for privacy violations and return redaction information"""
        violations = []
        
        for privacy_type, pattern in self.privacy_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                violations.append({
                    "type": privacy_type,
                    "matched_text": match.group(),
                    "start_position": match.start(),
                    "end_position": match.end(),
                    "replacement": self._get_privacy_replacement(privacy_type)
                })
        
        return violations
    
    def _get_privacy_replacement(self, privacy_type: str) -> str:
        """Get replacement text for privacy violations"""
        replacements = {
            'ssn': '[REDACTED_SSN]',
            'email': '[REDACTED_EMAIL]',
            'phone': '[REDACTED_PHONE]',
            'address': '[REDACTED_ADDRESS]',
            'credit_card': '[REDACTED_CC]',
            'medical_record': '[REDACTED_MR]'
        }
        return replacements.get(privacy_type, '[REDACTED]')
    
    def _check_inappropriate_content(self, content: str) -> float:
        """Check for inappropriate content"""
        inappropriate_matches = sum(1 for pattern in self.inappropriate_patterns 
                                  if re.search(pattern, content.lower()))
        return min(1.0, inappropriate_matches * 0.3)
    
    def _check_bias_content(self, content: str) -> float:
        """Check for biased content"""
        if self.bias_classifier:
            try:
                result = self.bias_classifier(content[:256])  # Shorter text for bias detection
                if result and isinstance(result, list) and len(result) > 0:
                    bias_result = result[0]
                    if bias_result['label'] in ['BIASED', 'STEREOTYPE']:
                        return min(1.0, bias_result['score'])
            except Exception as e:
                self.logger.warning(f"Bias classifier error: {e}")
        
        # Fallback to simple keyword matching for biased content
        bias_keywords = [
            r'\bracist\b', r '\bsexist\b', r '\bageist\b', r '\bhomophobic\b',
            r '\btransphobic\b', r '\bxenophobic\b', r '\bislamophobic\b',
            r '\bantisemitic\b', r '\bcolorist\b'
        ]
        
        bias_matches = sum(1 for pattern in bias_keywords if re.search(pattern, content.lower()))
        return min(1.0, bias_matches * 0.2)
    
    def _check_legal_medical_advisory(self, content: str) -> float:
        """Check for legal or medical advisory content that should not be given by AI"""
        advisory_patterns = [
            r '\byou should\b.*\blawyer\b', r '\blegal advice\b', r '\bmedical diagnosis\b',
            r '\bprescribe\b', r '\bdiagnose\b', r '\btreatment plan\b',
            r '\bmedication\b.*\bdosage\b', r '\bsurgery\b', r '\bprocedure\b.*\bmedical\b'
        ]
        
        advisory_matches = sum(1 for pattern in advisory_patterns if re.search(pattern, content.lower()))
        return min(1.0, advisory_matches * 0.4)
    
    def _redact_content(self, content: str, redactions: List[Dict[str, Any]]) -> str:
        """Redact sensitive content from text"""
        # Sort redactions by position (reverse order) to maintain indices
        sorted_redactions = sorted(redactions, key=lambda x: x['start_position'], reverse=True)
        
        redacted_content = content
        for redaction in sorted_redactions:
            start = redaction['start_position']
            end = redaction['end_position']
            replacement = redaction['replacement']
            redacted_content = redacted_content[:start] + replacement + redacted_content[end:]
        
        return redacted_content
    
    def _get_threshold_for_category(self, category: SafetyCategory) -> float:
        """Get safety threshold for a specific category based on safety level"""
        thresholds = {
            SafetyLevel.LENIENT: {
                SafetyCategory.CRISIS: 0.9,      # Still strict on crisis
                SafetyCategory.HARM: 0.7,
                SafetyCategory.TOXICITY: 0.6,
                SafetyCategory.PRIVACY: 0.1,     # Strict on privacy
                SafetyCategory.INAPPROPRIATE: 0.5,
                SafetyCategory.BIASED: 0.6,
                SafetyCategory.LEGAL: 0.8,       # Strict on legal/medical
                SafetyCategory.MEDICAL: 0.8
            },
            SafetyLevel.MODERATE: {
                SafetyCategory.CRISIS: 0.7,
                SafetyCategory.HARM: 0.5,
                SafetyCategory.TOXICITY: 0.4,
                SafetyCategory.PRIVACY: 0.1,
                SafetyCategory.INAPPROPRIATE: 0.3,
                SafetyCategory.BIASED: 0.4,
                SafetyCategory.LEGAL: 0.6,
                SafetyCategory.MEDICAL: 0.6
            },
            SafetyLevel.STRICT: {
                SafetyCategory.CRISIS: 0.5,
                SafetyCategory.HARM: 0.3,
                SafetyCategory.TOXICITY: 0.2,
                SafetyCategory.PRIVACY: 0.1,
                SafetyCategory.INAPPROPRIATE: 0.2,
                SafetyCategory.BIASED: 0.2,
                SafetyCategory.LEGAL: 0.4,
                SafetyCategory.MEDICAL: 0.4
            },
            SafetyLevel.PARANOID: {
                SafetyCategory.CRISIS: 0.3,
                SafetyCategory.HARM: 0.1,
                SafetyCategory.TOXICITY: 0.1,
                SafetyCategory.PRIVACY: 0.1,
                SafetyCategory.INAPPROPRIATE: 0.1,
                SafetyCategory.BIASED: 0.1,
                SafetyCategory.LEGAL: 0.2,
                SafetyCategory.MEDICAL: 0.2
            }
        }
        
        return thresholds[self.safety_level].get(category, 0.5)
    
    def _get_safety_threshold(self) -> float:
        """Get overall safety threshold based on safety level"""
        thresholds = {
            SafetyLevel.LENIENT: 0.7,
            SafetyLevel.MODERATE: 0.8,
            SafetyLevel.STRICT: 0.9,
            SafetyLevel.PARANOID: 0.95
        }
        return thresholds[self.safety_level]
    
    def _calculate_assessment_confidence(self, category_scores: Dict[SafetyCategory, float]) -> float:
        """Calculate confidence in the safety assessment"""
        if not category_scores:
            return 1.0
        
        # Use the maximum score as inverse confidence (higher risk = lower confidence)
        max_score = max(category_scores.values())
        return 1.0 - (max_score * 0.5)  # Scale down the impact
    
    def filter_response(self, 
                        response: str,
                        user_context: Optional[Dict[str, Any]] = None,
                        request_metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, SafetyCheckResult]:
        """Filter a response to ensure safety, returning (is_safe, filtered_response, safety_result)"""
        safety_result = self.check_output_safety(response, user_context, request_metadata)
        
        if safety_result.is_safe:
            return True, response, safety_result
        elif safety_result.filtered_content:
            return False, safety_result.filtered_content, safety_result
        else:
            # If filtering failed, return a safe default response
            safe_response = "[Response filtered for safety - content requires human review]"
            return False, safe_response, safety_result
    
    def batch_filter_responses(self, 
                              responses: List[str],
                              user_context: Optional[Dict[str, Any]] = None,
                              request_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[bool, str, SafetyCheckResult]]:
        """Filter multiple responses in batch"""
        results = []
        for response in responses:
            result = self.filter_response(response, user_context, request_metadata)
            results.append(result)
        return results


class CrisisInterventionSystem:
    """System for handling crisis situations detected by safety filters"""
    
    def __init__(self, safety_filter: EnhancedSafetyFilter):
        self.safety_filter = safety_filter
        self.logger = logging.getLogger(__name__)
        self.crisis_handlers = self._initialize_crisis_handlers()
    
    def _initialize_crisis_handlers(self) -> Dict[str, callable]:
        """Initialize crisis handlers for different crisis types"""
        return {
            'suicide': self._handle_suicide_crisis,
            'harm_to_others': self._handle_harm_to_others_crisis,
            'substance_abuse': self._handle_substance_abuse_crisis,
            'eating_disorders': self._handle_eating_disorder_crisis
        }
    
    def handle_crisis_detection(self, 
                              crisis_result: CrisisDetectionResult,
                              user_context: Optional[Dict[str, Any]] = None,
                              content: Optional[str] = None) -> Dict[str, Any]:
        """Handle a detected crisis situation"""
        if not crisis_result.is_crisis:
            return {"status": "no_crisis", "action_taken": "none"}
        
        crisis_type = crisis_result.crisis_type
        handler = self.crisis_handlers.get(crisis_type)
        
        if handler:
            try:
                result = handler(crisis_result, user_context, content)
                self.logger.info(f"Handled {crisis_type} crisis: {result}")
                return result
            except Exception as e:
                self.logger.error(f"Failed to handle {crisis_type} crisis: {e}")
                # Fallback to basic crisis response
                return self._basic_crisis_response(crisis_result, user_context)
        else:
            # Fallback to basic crisis response
            return self._basic_crisis_response(crisis_result, user_context)
    
    def _handle_suicide_crisis(self,
                              crisis_result: CrisisDetectionResult,
                              user_context: Optional[Dict[str, Any]] = None,
                              content: Optional[str] = None) -> Dict[str, Any]:
        """Handle suicide crisis detection"""
        # Log the crisis event
        crisis_log = {
            "crisis_type": "suicide",
            "urgency": crisis_result.urgency_level,
            "timestamp": datetime.utcnow().isoformat(),
            "user_context": user_context,
            "content_preview": content[:100] + "..." if content and len(content) > 100 else content,
            "confidence": crisis_result.confidence
        }
        
        self.logger.critical(f"SUICIDE CRISIS DETECTED: {crisis_log}")
        
        # Generate crisis response
        response_content = {
            "message": "I'm concerned about what you're going through. Your safety is important.",
            "resources": crisis_result.resources or [
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741"
            ],
            "recommended_action": crisis_result.recommended_action,
            "urgency_level": crisis_result.urgency_level
        }
        
        # Trigger external alerting system (in production, this would notify crisis teams)
        self._trigger_external_alert(crisis_log)
        
        return {
            "status": "crisis_handled",
            "crisis_type": "suicide",
            "response_content": response_content,
            "logged": True,
            "external_alert_triggered": True,
            "crisis_log": crisis_log
        }
    
    def _handle_harm_to_others_crisis(self,
                                    crisis_result: CrisisDetectionResult,
                                    user_context: Optional[Dict[str, Any]] = None,
                                    content: Optional[str] = None) -> Dict[str, Any]:
        """Handle harm to others crisis detection"""
        # Log the crisis event
        crisis_log = {
            "crisis_type": "harm_to_others",
            "urgency": crisis_result.urgency_level,
            "timestamp": datetime.utcnow().isoformat(),
            "user_context": user_context,
            "content_preview": content[:100] + "..." if content and len(content) > 100 else content,
            "confidence": crisis_result.confidence
        }
        
        self.logger.critical(f"POTENTIAL HARM TO OTHERS DETECTED: {crisis_log}")
        
        # Generate crisis response
        response_content = {
            "message": "I'm concerned about safety. It's important to seek help immediately.",
            "resources": crisis_result.resources or [
                "Emergency Services: 911",
                "Local Crisis Center Hotline"
            ],
            "recommended_action": crisis_result.recommended_action,
            "urgency_level": crisis_result.urgency_level
        }
        
        # Trigger external alerting system
        self._trigger_external_alert(crisis_log)
        
        return {
            "status": "crisis_handled",
            "crisis_type": "harm_to_others",
            "response_content": response_content,
            "logged": True,
            "external_alert_triggered": True,
            "crisis_log": crisis_log
        }
    
    def _handle_substance_abuse_crisis(self,
                                      crisis_result: CrisisDetectionResult,
                                      user_context: Optional[Dict[str, Any]] = None,
                                      content: Optional[str] = None) -> Dict[str, Any]:
        """Handle substance abuse crisis detection"""
        crisis_log = {
            "crisis_type": "substance_abuse",
            "urgency": crisis_result.urgency_level,
            "timestamp": datetime.utcnow().isoformat(),
            "user_context": user_context,
            "content_preview": content[:100] + "..." if content and len(content) > 100 else content,
            "confidence": crisis_result.confidence
        }
        
        self.logger.warning(f"SUBSTANCE ABUSE INDICATORS DETECTED: {crisis_log}")
        
        response_content = {
            "message": "Substance use concerns can be challenging. Support is available.",
            "resources": [
                "SAMHSA National Helpline: 1-800-662-4357",
                "Alcoholics Anonymous: aa.org",
                "Narcotics Anonymous: na.org"
            ],
            "recommended_action": "Provide substance abuse resources",
            "urgency_level": crisis_result.urgency_level
        }
        
        return {
            "status": "resources_provided",
            "crisis_type": "substance_abuse",
            "response_content": response_content,
            "logged": True,
            "crisis_log": crisis_log
        }
    
    def _handle_eating_disorder_crisis(self,
                                      crisis_result: CrisisDetectionResult,
                                      user_context: Optional[Dict[str, Any]] = None,
                                      content: Optional[str] = None) -> Dict[str, Any]:
        """Handle eating disorder crisis detection"""
        crisis_log = {
            "crisis_type": "eating_disorders",
            "urgency": crisis_result.urgency_level,
            "timestamp": datetime.utcnow().isoformat(),
            "user_context": user_context,
            "content_preview": content[:100] + "..." if content and len(content) > 100 else content,
            "confidence": crisis_result.confidence
        }
        
        self.logger.warning(f"EATING DISORDER INDICATORS DETECTED: {crisis_log}")
        
        response_content = {
            "message": "Concerns about eating patterns can be addressed with professional support.",
            "resources": [
                "National Eating Disorders Association: nationaleatingdisorders.org",
                "ANAD Eating Disorder Helpline: 630-577-1330"
            ],
            "recommended_action": "Provide eating disorder resources",
            "urgency_level": crisis_result.urgency_level
        }
        
        return {
            "status": "resources_provided",
            "crisis_type": "eating_disorders",
            "response_content": response_content,
            "logged": True,
            "crisis_log": crisis_log
        }
    
    def _basic_crisis_response(self,
                               crisis_result: CrisisDetectionResult,
                               user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Basic crisis response for unhandled crisis types"""
        basic_response = {
            "message": "I'm concerned about what you're experiencing. Professional help is available.",
            "resources": [
                "Emergency Services: 911",
                "Crisis Text Line: Text HOME to 741741"
            ],
            "recommended_action": "Seek immediate professional help",
            "urgency_level": "high"
        }
        
        return {
            "status": "basic_response",
            "crisis_type": crisis_result.crisis_type or "unknown",
            "response_content": basic_response,
            "logged": True
        }
    
    def _trigger_external_alert(self, crisis_log: Dict[str, Any]):
        """Trigger external alerting system for urgent crises"""
        # In a real system, this would:
        # 1. Send alerts to crisis intervention teams
        # 2. Notify supervisors
        # 3. Log to security incident management system
        # 4. Potentially contact emergency services for immediate threats
        self.logger.info(f"External alert triggered for crisis: {crisis_log.get('crisis_type')}")


# Global safety filter instance
enhanced_safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)
crisis_intervention_system = CrisisInterventionSystem(enhanced_safety_filter)


# Decorator for automatic safety filtering on API endpoints
def safety_filtered_endpoint(func):
    """Decorator to automatically apply safety filtering to API endpoint responses"""
    def wrapper(*args, **kwargs):
        # Get the original response
        result = func(*args, **kwargs)
        
        # Extract response content (adjust based on your API response structure)
        if isinstance(result, dict) and 'content' in result:
            content = result['content']
            user_context = kwargs.get('user_context') or getattr(args[0] if args else None, 'user_context', None)
            request_metadata = kwargs.get('request_metadata') or getattr(args[0] if args else None, 'request_metadata', None)
            
            # Apply safety filtering
            is_safe, filtered_content, safety_result = enhanced_safety_filter.filter_response(
                content, user_context, request_metadata
            )
            
            # Update the response
            result['content'] = filtered_content
            result['safety_filtered'] = not is_safe
            result['safety_score'] = safety_result.overall_score
            result['safety_categories'] = [cat.value for cat in safety_result.flagged_categories]
            
            # Handle crisis situations
            if safety_result.category_scores.get(SafetyCategory.CRISIS, 0) > 0.5:
                # Extract crisis detection result from safety result
                crisis_result = CrisisDetectionResult(
                    is_crisis=True,
                    crisis_type="detected_from_safety_filter",
                    confidence=safety_result.category_scores[SafetyCategory.CRISIS],
                    urgency_level="high",
                    recommended_action="seek_help"
                )
                crisis_response = crisis_intervention_system.handle_crisis_detection(
                    crisis_result, user_context, content
                )
                result['crisis_intervention'] = crisis_response
            
            return result
        else:
            return result
    
    return wrapper


# Integration functions for the inference API
def integrate_safety_filtering_with_api():
    """Integrate safety filtering with the inference API"""
    # This function would be called during API initialization
    # to set up safety filtering for all endpoints
    
    logger.info("Safety filtering integrated with inference API")
    
    # Example of how to wrap API endpoints
    def safety_wrapper(endpoint_func):
        def wrapped(*args, **kwargs):
            # Apply safety filtering to the endpoint
            response = endpoint_func(*args, **kwargs)
            
            # If response contains content that should be filtered
            if hasattr(response, 'content') or (isinstance(response, dict) and 'content' in response):
                content = response.content if hasattr(response, 'content') else response['content']
                user_context = kwargs.get('user_context', {})
                request_metadata = kwargs.get('request_metadata', {})
                
                is_safe, filtered_content, safety_result = enhanced_safety_filter.filter_response(
                    content, user_context, request_metadata
                )
                
                # Update response with safety information
                if hasattr(response, '__dict__'):
                    response.content = filtered_content
                    response.safety_filtered = not is_safe
                    response.safety_score = safety_result.overall_score
                elif isinstance(response, dict):
                    response['content'] = filtered_content
                    response['safety_filtered'] = not is_safe
                    response['safety_score'] = safety_result.overall_score
                    response['safety_categories'] = [cat.value for cat in safety_result.flagged_categories]
            
            return response
        return wrapped
    return safety_wrapper


# Example usage and testing
def test_enhanced_safety_filter():
    """Test the enhanced safety filtering system"""
    logger.info("Testing Enhanced Safety Filtering System...")
    
    # Test safety filter with various content types
    test_cases = [
        # Safe content
        ("I'm feeling a bit down today, but I'm managing.", "safe_content"),
        
        # Crisis content - suicide
        ("I've been thinking about ending it all. Life isn't worth living anymore.", "suicide_crisis"),
        
        # Crisis content - harm to others
        ("I'm so angry I could hurt someone. People need to pay for what they've done.", "harm_to_others"),
        
        # Harmful content
        ("I hate that person. I wish I could punch them in the face.", "physical_harm"),
        
        # Toxic content
        ("This stupid idiot is completely worthless and should just disappear.", "toxic_content"),
        
        # Privacy violation
        ("My social security number is 123-45-6789. Please don't share this.", "privacy_violation"),
        
        # Inappropriate content
        ("I found some explicit adult content online that was really graphic.", "inappropriate_content"),
        
        # Biased content
        ("All people from that country are lazy and criminals.", "biased_content"),
        
        # Legal/medical advice
        ("You should definitely sue your employer for that. I recommend this specific lawyer.", "legal_advice"),
        
        # Mixed content with multiple issues
        ("I'm thinking about overdosing on my medication tonight. My phone number is 555-123-4567.", "mixed_crisis_privacy")
    ]
    
    # Test each case
    for content, test_type in test_cases:
        print(f"\n--- Testing {test_type} ---")
        print(f"Input: {content[:50]}{'...' if len(content) > 50 else ''}")
        
        # Check output safety
        safety_result = enhanced_safety_filter.check_output_safety(content)
        
        print(f"Safe: {safety_result.is_safe}")
        print(f"Overall Score: {safety_result.overall_score:.3f}")
        print(f"Confidence: {safety_result.confidence:.3f}")
        print(f"Explanation: {safety_result.explanation}")
        
        if safety_result.flagged_categories:
            print(f"Flagged Categories: {[cat.value for cat in safety_result.flagged_categories]}")
            for category, score in safety_result.category_scores.items():
                if score > 0.1:
                    print(f"  {category.value}: {score:.3f}")
        
        if safety_result.filtered_content:
            print(f"Filtered Content: {safety_result.filtered_content[:50]}{'...' if len(safety_result.filtered_content) > 50 else ''}")
        
        # Test filtering
        is_safe, filtered_content, filter_result = enhanced_safety_filter.filter_response(content)
        print(f"Filter Result - Safe: {is_safe}")
        if not is_safe:
            print(f"Final Filtered Content: {filtered_content[:50]}{'...' if len(filtered_content) > 50 else ''}")
        
        # Test crisis detection specifically
        if "crisis" in test_type or any(word in content.lower() for word in ["suicide", "kill", "hurt", "harm"]):
            # Extract crisis-related content for testing
            crisis_test_content = "I'm thinking about suicide" if "suicide" in test_type else content
            crisis_result = enhanced_safety_filter._check_crisis_content(crisis_test_content)
            if crisis_result.is_crisis:
                print(f"Crisis Detected: {crisis_result.crisis_type}")
                print(f"Urgency: {crisis_result.urgency_level}")
                print(f"Confidence: {crisis_result.confidence:.3f}")
                print(f"Recommended Action: {crisis_result.recommended_action}")
                
                # Test crisis intervention
                crisis_response = crisis_intervention_system.handle_crisis_detection(
                    crisis_result, 
                    user_context={"user_id": "test_user_123"},
                    content=crisis_test_content
                )
                print(f"Crisis Response Status: {crisis_response['status']}")
    
    # Test batch filtering
    print("\n--- Testing Batch Filtering ---")
    batch_content = [
        "I'm having a good day today.",
        "I don't want to live anymore.",
        "The weather is nice.",
        "I'm going to hurt someone."
    ]
    
    batch_results = enhanced_safety_filter.batch_filter_responses(batch_content)
    
    for i, (is_safe, filtered_content, safety_result) in enumerate(batch_results):
        print(f"Batch Item {i+1}: Safe={is_safe}, Score={safety_result.overall_score:.3f}")
    
    # Test different safety levels
    print("\n--- Testing Different Safety Levels ---")
    test_content = "I'm feeling really depressed and thinking about ending it all."
    
    for safety_level in [SafetyLevel.LENIENT, SafetyLevel.MODERATE, SafetyLevel.STRICT, SafetyLevel.PARANOID]:
        filter_instance = EnhancedSafetyFilter(safety_level)
        result = filter_instance.check_output_safety(test_content)
        print(f"{safety_level.value}: Safe={result.is_safe}, Score={result.overall_score:.3f}")
    
    print("\nEnhanced safety filtering system tests completed!")


if __name__ == "__main__":
    test_enhanced_safety_filter()