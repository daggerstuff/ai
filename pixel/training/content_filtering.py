
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union


class ContentFilter:
    """
    Main ContentFilter class that provides is_safe method for voice training pipeline.
    Compatible with the voice training system that expects 'ContentFilter' object with 'is_safe' attribute.
    """
    
    def __init__(self):
        """Initialize the ContentFilter with the comprehensive filtering system."""
        self.filtering_system = None  # Will be initialized lazily
    
    def _get_filtering_system(self):
        """Lazy initialization of the filtering system to avoid circular imports."""
        if self.filtering_system is None:
            self.filtering_system = ContentFiltering()
        return self.filtering_system
    
    def is_safe(self, content: str) -> bool:
        """
        Check if content is safe for training.
        
        Args:
            content: Text content to check
            
        Returns:
            bool: True if content is safe, False otherwise
        """
        try:
            # Use the comprehensive filtering system
            filtering_system = self._get_filtering_system()
            result = filtering_system.filter_content(content)
            
            # Content is safe if no critical or error level issues found
            critical_issues = [
                issue for issue in result.validation_results 
                if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]
            ]
            
            # Also check if content has harmful PII or safety violations
            pii_critical = any(
                pii.pii_type in [PiiType.SSN, PiiType.CREDIT_CARD] 
                for pii in result.pii_detections
            )
            
            return len(critical_issues) == 0 and not pii_critical
            
        except Exception as e:
            # Default to unsafe if filtering fails
            print(f"ContentFilter error: {e}")
            return False
    
    # Alias method for compatibility
    @property
    def is_safe_method(self):
        """Property to access is_safe as an attribute (for compatibility)."""
        return self.is_safe

class PiiType(Enum):
    """Enumeration of PII types that can be detected."""
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    ADDRESS = "ADDRESS"
    NAME = "NAME"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"

class ValidationSeverity(Enum):
    """Severity levels for content validation."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class SafetyGateType(Enum):
    """Types of safety gates."""
    CONTENT_LENGTH = "CONTENT_LENGTH"
    PROFANITY = "PROFANITY"
    HATE_SPEECH = "HATE_SPEECH"
    THERAPEUTIC_CONTEXT = "THERAPEUTIC_CONTEXT"
    CRISIS_ESCALATION = "CRISIS_ESCALATION"
    MEDICAL_ADVICE = "MEDICAL_ADVICE"
    PERSONAL_DISCLOSURE = "PERSONAL_DISCLOSURE"

@dataclass
class PiiDetectionResult:
    """Data structure for holding PII detection results."""
    pii_type: PiiType
    value: str
    start: int
    end: int
    confidence: float = 1.0
    redacted_value: str = ""

@dataclass
class ContentValidationResult:
    """Data structure for holding content validation results."""
    validation_rule: str
    is_valid: bool
    severity: ValidationSeverity
    details: str
    suggestions: List[str] = None

@dataclass
class SafetyGateResult:
    """Data structure for holding safety gate results."""
    gate_type: SafetyGateType
    gate_name: str
    passed: bool
    severity: ValidationSeverity
    details: str
    recommendations: List[str] = None
    confidence: float = 1.0

class ContentFiltering:
    """
    Enhanced content filter for therapeutic AI systems.
    
    Provides comprehensive PII detection, content validation, and safety gates
    specifically designed for mental health and therapeutic contexts.
    """

    def __init__(self, enable_crisis_integration: bool = True):
        """
        Initializes the ContentFilter.
        
        Args:
            enable_crisis_integration: Whether to integrate with crisis detection
        """
        self.enable_crisis_integration = enable_crisis_integration
        
        # Enhanced PII patterns with higher accuracy
        self.pii_patterns = {
            PiiType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            PiiType.PHONE: re.compile(
                r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
            ),
            PiiType.SSN: re.compile(
                r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'
            ),
            PiiType.CREDIT_CARD: re.compile(
                r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'
            ),
            PiiType.DATE_OF_BIRTH: re.compile(
                r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b'
            ),
            PiiType.IP_ADDRESS: re.compile(
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
            ),
            PiiType.URL: re.compile(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            ),
            PiiType.ADDRESS: re.compile(
                r'\b\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',
                re.IGNORECASE
            ),
            PiiType.NAME: re.compile(
                r'\b(?:my name is|i am|i\'m|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                re.IGNORECASE
            )
        }
        
        # Therapeutic content validation rules
        self.validation_rules = {
            'MIN_LENGTH': {'min_chars': 5, 'severity': ValidationSeverity.ERROR},
            'MAX_LENGTH': {'max_chars': 10000, 'severity': ValidationSeverity.WARNING},
            'THERAPEUTIC_CONTEXT': {'severity': ValidationSeverity.INFO},
            'PROFESSIONALISM': {'severity': ValidationSeverity.WARNING},
            'COHERENCE': {'severity': ValidationSeverity.WARNING}
        }
        
        # Safety gate patterns for mental health contexts
        self.safety_patterns = {
            SafetyGateType.PROFANITY: {
                'patterns': [
                    r'\b(?:fucking|fuck|shit|damn|bitch|asshole|bastard)\b'
                ],
                'severity': ValidationSeverity.WARNING
            },
            SafetyGateType.HATE_SPEECH: {
                'patterns': [
                    r'\b(?:hate|kill|murder|die|suicide|harm)\s+(?:yourself|myself|themselves)\b'
                ],
                'severity': ValidationSeverity.CRITICAL
            },
            SafetyGateType.MEDICAL_ADVICE: {
                'patterns': [
                    r'\b(?:take|stop|start|increase|decrease)\s+(?:medication|pills|drugs|antidepressants)\b',
                    r'\b(?:diagnose|prescription|dosage|mg|ml)\b'
                ],
                'severity': ValidationSeverity.ERROR
            },
            SafetyGateType.PERSONAL_DISCLOSURE: {
                'patterns': [
                    r'\b(?:i am a|i work as|i\'m a)\s+(?:licensed\s+)?(?:therapist|psychologist|psychiatrist|counselor|doctor)\b'
                ],
                'severity': ValidationSeverity.WARNING
            }
        }
        
        # Crisis keywords that should never be filtered
        self.crisis_keywords = {
            'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
            'self-harm', 'cutting', 'hurting myself', 'overdose', 'pills',
            'psychosis', 'hearing voices', 'seeing things', 'paranoid',
            'panic attack', 'can\'t breathe', 'heart racing', 'dying'
        }

    def detect_pii(self, text: str) -> List[PiiDetectionResult]:
        """
        Detects PII in the given text using enhanced patterns.

        Args:
            text: The text to scan for PII.

        Returns:
            A list of PII detection results with confidence scores.
        """
        pii_results = []
        
        for pii_type, pattern in self.pii_patterns.items():
            for match in pattern.finditer(text):
                # Calculate confidence based on pattern specificity
                confidence = self._calculate_pii_confidence(pii_type, match.group(0))
                
                # Generate redacted value
                redacted_value = self._generate_redacted_value(pii_type, match.group(0))
                
                pii_results.append(
                    PiiDetectionResult(
                        pii_type=pii_type,
                        value=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        redacted_value=redacted_value
                    )
                )
        
        return pii_results

    def remove_pii(self, text: str, preserve_crisis_keywords: bool = True) -> str:
        """
        Removes detected PII from the text while preserving crisis-related content.

        Args:
            text: The text to remove PII from.
            preserve_crisis_keywords: Whether to preserve crisis-related keywords.

        Returns:
            The text with PII removed.
        """
        pii_results = self.detect_pii(text)
        
        # Filter out PII that might overlap with crisis keywords
        if preserve_crisis_keywords:
            pii_results = self._filter_crisis_overlaps(text, pii_results)
        
        # Sort by start position in reverse order for safe replacement
        for result in sorted(pii_results, key=lambda r: r.start, reverse=True):
            replacement = f"[{result.pii_type.value}]"
            if result.redacted_value:
                replacement = result.redacted_value
            text = text[:result.start] + replacement + text[result.end:]
        
        return text

    def validate_content(self, text: str) -> List[ContentValidationResult]:
        """
        Validates content using therapeutic AI-specific rules.

        Args:
            text: The text to validate.

        Returns:
            A list of content validation results with suggestions.
        """
        validation_results = []
        
        # Length validation
        if len(text) < self.validation_rules['MIN_LENGTH']['min_chars']:
            validation_results.append(
                ContentValidationResult(
                    validation_rule="MIN_LENGTH",
                    is_valid=False,
                    severity=self.validation_rules['MIN_LENGTH']['severity'],
                    details=f"Text too short ({len(text)} chars). Minimum: {self.validation_rules['MIN_LENGTH']['min_chars']}",
                    suggestions=["Add more context", "Provide specific details", "Expand your thoughts"]
                )
            )
        
        if len(text) > self.validation_rules['MAX_LENGTH']['max_chars']:
            validation_results.append(
                ContentValidationResult(
                    validation_rule="MAX_LENGTH",
                    is_valid=False,
                    severity=self.validation_rules['MAX_LENGTH']['severity'],
                    details=f"Text too long ({len(text)} chars). Maximum: {self.validation_rules['MAX_LENGTH']['max_chars']}",
                    suggestions=["Break into smaller parts", "Focus on key points", "Remove redundant information"]
                )
            )
        
        # Therapeutic context validation
        therapeutic_indicators = self._check_therapeutic_context(text)
        if not therapeutic_indicators['has_context']:
            validation_results.append(
                ContentValidationResult(
                    validation_rule="THERAPEUTIC_CONTEXT",
                    is_valid=False,
                    severity=ValidationSeverity.INFO,
                    details="Content may lack therapeutic context",
                    suggestions=therapeutic_indicators['suggestions']
                )
            )
        
        # Coherence validation
        coherence_score = self._assess_coherence(text)
        if coherence_score < 0.6:
            validation_results.append(
                ContentValidationResult(
                    validation_rule="COHERENCE",
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    details=f"Low coherence score: {coherence_score:.2f}",
                    suggestions=["Check for clear structure", "Ensure logical flow", "Clarify relationships between ideas"]
                )
            )
        
        return validation_results

    def enforce_safety_gates(self, text: str) -> List[SafetyGateResult]:
        """
        Enforces comprehensive safety gates for therapeutic AI content.

        Args:
            text: The text to enforce safety gates on.

        Returns:
            A list of safety gate results with recommendations.
        """
        safety_gate_results = []
        text.lower()
        
        # Check each safety gate type
        for gate_type, gate_config in self.safety_patterns.items():
            gate_violations = []
            
            for pattern_str in gate_config['patterns']:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                matches = pattern.findall(text)
                if matches:
                    gate_violations.extend(matches)
            
            if gate_violations:
                recommendations = self._get_safety_recommendations(gate_type, gate_violations)
                confidence = min(1.0, len(gate_violations) * 0.3)  # Scale confidence by violations
                
                safety_gate_results.append(
                    SafetyGateResult(
                        gate_type=gate_type,
                        gate_name=gate_type.value,
                        passed=False,
                        severity=gate_config['severity'],
                        details=f"Detected {len(gate_violations)} {gate_type.value.lower()} violation(s): {', '.join(gate_violations[:3])}",
                        recommendations=recommendations,
                        confidence=confidence
                    )
                )
        
        # Content length safety gate
        if len(text) == 0:
            safety_gate_results.append(
                SafetyGateResult(
                    gate_type=SafetyGateType.CONTENT_LENGTH,
                    gate_name="EMPTY_CONTENT",
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    details="Empty content detected",
                    recommendations=["Provide meaningful content", "Add context or details"],
                    confidence=1.0
                )
            )
        
        return safety_gate_results

    def _calculate_pii_confidence(self, pii_type: PiiType, value: str) -> float:
        """Calculate confidence score for PII detection."""
        if pii_type == PiiType.EMAIL:
            return 0.95 if '@' in value and '.' in value else 0.7
        elif pii_type == PiiType.PHONE:
            digits = re.sub(r'[^\d]', '', value)
            return 0.9 if len(digits) == 10 or len(digits) == 11 else 0.6
        elif pii_type == PiiType.SSN:
            return 0.8 if len(re.sub(r'[^\d]', '', value)) == 9 else 0.5
        elif pii_type == PiiType.CREDIT_CARD:
            return 0.9 if len(re.sub(r'[^\d]', '', value)) == 16 else 0.6
        else:
            return 0.7

    def _generate_redacted_value(self, pii_type: PiiType, value: str) -> str:
        """Generate appropriate redacted replacement for PII."""
        if pii_type == PiiType.EMAIL:
            return "[EMAIL_REDACTED]"
        elif pii_type == PiiType.PHONE:
            return "[PHONE_REDACTED]"
        elif pii_type == PiiType.NAME:
            return "[NAME_REDACTED]"
        elif pii_type == PiiType.ADDRESS:
            return "[ADDRESS_REDACTED]"
        else:
            return f"[{pii_type.value}_REDACTED]"

    def _filter_crisis_overlaps(self, text: str, pii_results: List[PiiDetectionResult]) -> List[PiiDetectionResult]:
        """Filter out PII detections that overlap with crisis keywords."""
        filtered_results = []
        text_lower = text.lower()
        
        for result in pii_results:
            overlaps_crisis = False
            text[result.start:result.end].lower()
            
            for crisis_keyword in self.crisis_keywords:
                if crisis_keyword in text_lower:
                    # Check if PII detection overlaps with crisis keyword
                    crisis_start = text_lower.find(crisis_keyword)
                    crisis_end = crisis_start + len(crisis_keyword)
                    
                    if (result.start < crisis_end and result.end > crisis_start):
                        overlaps_crisis = True
                        break
            
            if not overlaps_crisis:
                filtered_results.append(result)
        
        return filtered_results

    def _check_therapeutic_context(self, text: str) -> Dict[str, Union[bool, List[str]]]:
        """Check if text has appropriate therapeutic context."""
        therapeutic_keywords = [
            'feel', 'feeling', 'emotion', 'therapy', 'counseling', 'support',
            'help', 'struggle', 'difficulty', 'cope', 'coping', 'stress',
            'anxiety', 'depression', 'mental health', 'wellbeing', 'healing'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in therapeutic_keywords if kw in text_lower]
        
        has_context = len(found_keywords) > 0
        suggestions = [
            "Include emotional context",
            "Mention feelings or experiences",
            "Add therapeutic language",
            "Focus on mental health aspects"
        ] if not has_context else []
        
        return {'has_context': has_context, 'suggestions': suggestions}

    def _assess_coherence(self, text: str) -> float:
        """Assess text coherence using basic heuristics."""
        if not text.strip():
            return 0.0
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return 0.0
        
        # Basic coherence scoring
        score = 0.3  # Lower base score
        
        # Sentence count factor - prefer 2-5 sentences
        if 2 <= len(sentences) <= 5:
            score += 0.2
        elif len(sentences) == 1:
            score += 0.1  # Single sentence can be coherent
        elif len(sentences) > 10:
            score -= 0.2
        
        # Average sentence length - prefer 5-20 words
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if 5 <= avg_length <= 20:
            score += 0.3
        elif avg_length < 3:  # Very short fragments
            score -= 0.2
        elif avg_length > 30:  # Very long run-ons
            score -= 0.1
        
        # Capitalization and punctuation
        if len(sentences) > 0:
            properly_capitalized = sum(1 for s in sentences if len(s) > 0 and s[0].isupper()) / len(sentences)
            score += properly_capitalized * 0.1
        
        # Check for incomplete fragments (words without proper sentence structure)
        words = text.split()
        if len(words) > 3:
            # Look for connecting words that suggest coherence
            connecting_words = ['and', 'but', 'because', 'so', 'however', 'therefore', 'also', 'then']
            has_connections = any(word.lower() in connecting_words for word in words)
            if has_connections:
                score += 0.1
            
            # Penalize if it's mostly just isolated words
            if len(sentences) == 1 and len(words) > 5 and '.' not in text and '!' not in text and '?' not in text:
                score -= 0.3  # Likely just word fragments
        
        return min(1.0, max(0.0, score))

    def _get_safety_recommendations(self, gate_type: SafetyGateType, violations: List[str]) -> List[str]:
        """Get recommendations based on safety gate violations."""
        recommendations_map = {
            SafetyGateType.PROFANITY: [
                "Use professional language",
                "Replace with appropriate alternatives",
                "Focus on constructive expression"
            ],
            SafetyGateType.HATE_SPEECH: [
                "Remove harmful language",
                "Contact crisis support immediately",
                "Seek professional help"
            ],
            SafetyGateType.MEDICAL_ADVICE: [
                "Avoid giving medical advice",
                "Suggest consulting healthcare providers",
                "Focus on emotional support"
            ],
            SafetyGateType.PERSONAL_DISCLOSURE: [
                "Maintain appropriate boundaries",
                "Focus on client needs",
                "Avoid personal information sharing"
            ]
        }
        
        return recommendations_map.get(gate_type, ["Review content for appropriateness"])

