"""
Safety content filter for Pixelated Empathy AI project.
Implements safety checking for inputs and outputs in model inference.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


logger = logging.getLogger(__name__)


class SafetyCategory(Enum):
    """Categories of safety content to filter"""
    CRISIS = "crisis"
    HARM = "harm"
    TOXICITY = "toxicity"
    PRIVACY = "privacy"
    INAPPROPRIATE = "inappropriate"
    BIASED = "biased"


@dataclass
class SafetyCheckResult:
    """Result of a safety check"""
    is_safe: bool
    category: SafetyCategory
    confidence: float
    explanation: str
    filtered_content: Optional[str] = None


@dataclass
class SafetyScore:
    """Safety scoring for content"""
    overall_score: float  # 0.0 to 1.0, where 1.0 is completely safe
    category_scores: Dict[SafetyCategory, float]
    flagged_categories: List[SafetyCategory]
    details: Optional[Dict[str, Any]] = None


class SafetyFilter:
    """Main safety filtering class"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.crisis_keywords = [
            'kill myself', 'suicide', 'harm myself', 'end it all', 'not want to live',
            'overdose', 'cut', 'hurt myself', 'self-harm', 'self injury',
            'hurt others', 'kill others', 'murder', 'violence', 'attack'
        ]
        self.toxicity_keywords = [
            'hate', 'angry', 'furious', 'kill', 'destroy', 'attack', 'harm', 'violence',
            'stupid', 'idiot', 'worthless', 'useless', 'pathetic', 'disgusting'
        ]
        self.privacy_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{10,15}\b',  # Phone number
            r'\b\d{1,4} [A-Za-z ]+ [A-Za-z. ]+\b'  # Basic address pattern
        ]
        
        # Initialize safety models if available
        self.toxicity_classifier = None
        self._init_safety_models()
    
    def _init_safety_models(self):
        """Initialize safety checking models"""
        try:
            # Initialize toxicity classifier
            self.toxicity_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                tokenizer="unitary/toxic-bert"
            )
            self.logger.info("Toxicity classifier loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load toxicity classifier: {e}")
            self.toxicity_classifier = None
    
    def check_input_safety(self, input_text: str) -> SafetyScore:
        """Check safety of input text"""
        return self._check_text_safety(input_text)
    
    def check_output_safety(self, output_text: str) -> SafetyScore:
        """Check safety of model output"""
        return self._check_text_safety(output_text)
    
    def _check_text_safety(self, text: str) -> SafetyScore:
        """Internal method to check text safety"""
        if not text or not isinstance(text, str):
            return SafetyScore(
                overall_score=1.0,
                category_scores={},
                flagged_categories=[]
            )
        
        text_lower = text.lower()
        category_scores = {}
        flagged_categories = []
        
        # Check for crisis content
        crisis_score = self._check_crisis_content(text_lower)
        category_scores[SafetyCategory.CRISIS] = crisis_score
        if crisis_score > 0.7:
            flagged_categories.append(SafetyCategory.CRISIS)
        
        # Check for harmful content
        harm_score = self._check_harm_content(text_lower)
        category_scores[SafetyCategory.HARM] = harm_score
        if harm_score > 0.7:
            flagged_categories.append(SafetyCategory.HARM)
        
        # Check for toxicity
        toxicity_score = self._check_toxicity_content(text_lower)
        category_scores[SafetyCategory.TOXICITY] = toxicity_score
        if toxicity_score > 0.5:
            flagged_categories.append(SafetyCategory.TOXICITY)
        
        # Check for privacy violations
        privacy_score = self._check_privacy_content(text)
        category_scores[SafetyCategory.PRIVACY] = privacy_score
        if privacy_score > 0.5:
            flagged_categories.append(SafetyCategory.PRIVACY)
        
        # Calculate overall score (lower of all category scores)
        if not category_scores:
            overall_score = 1.0
        else:
            overall_score = 1.0 - max(category_scores.values())
        
        return SafetyScore(
            overall_score=max(0.0, min(1.0, overall_score)),
            category_scores=category_scores,
            flagged_categories=flagged_categories
        )
    
    def _check_crisis_content(self, text: str) -> float:
        """Check for crisis-related content"""
        crisis_matches = [kw for kw in self.crisis_keywords if kw in text]
        if crisis_matches:
            return min(1.0, len(crisis_matches) * 0.3)  # Weight crisis content heavily
        return 0.0
    
    def _check_harm_content(self, text: str) -> float:
        """Check for harmful content"""
        harm_keywords = [kw for kw in self.crisis_keywords if 'harm' in kw or 'kill' in kw]
        harm_matches = [kw for kw in harm_keywords if kw in text]
        if harm_matches:
            return min(1.0, len(harm_matches) * 0.4)
        return 0.0
    
    def _check_toxicity_content(self, text: str) -> float:
        """Check for toxic content using classifier or keywords"""
        if self.toxicity_classifier:
            try:
                result = self.toxicity_classifier(text[:512])  # Limit length for model
                if result and isinstance(result, list) and len(result) > 0:
                    toxicity_result = result[0]
                    if toxicity_result['label'] == 'TOXIC' and 'score' in toxicity_result:
                        return min(1.0, toxicity_result['score'])
            except Exception as e:
                self.logger.warning(f"Toxicity classifier error: {e}")
        
        # Fallback to keyword matching
        toxic_matches = [kw for kw in self.toxicity_keywords if kw in text]
        return min(1.0, len(toxic_matches) * 0.1) if toxic_matches else 0.0
    
    def _check_privacy_content(self, text: str) -> float:
        """Check for privacy-violating content"""
        for pattern in self.privacy_patterns:
            if re.search(pattern, text):
                return 1.0  # High probability of privacy violation
        return 0.0
    
    def filter_response(self, response: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """Filter a response to remove unsafe content"""
        if isinstance(response, str):
            return self._filter_text_response(response)
        elif isinstance(response, dict):
            return self._filter_dict_response(response)
        else:
            return response
    
    def _filter_text_response(self, text: str) -> str:
        """Filter text response"""
        safety_score = self._check_text_safety(text)
        
        if safety_score.overall_score < 0.5:  # Too unsafe
            return "[Response filtered for safety]"
        
        # For partially unsafe content, we could implement redaction instead of complete filtering
        # For now, return the original text if it passes the threshold
        return text
    
    def _filter_dict_response(self, response_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Filter dictionary response (like API responses)"""
        filtered_dict = response_dict.copy()
        
        # Filter text fields in the response
        for key, value in response_dict.items():
            if isinstance(value, str):
                safety_score = self._check_text_safety(value)
                if safety_score.overall_score < 0.5:
                    filtered_dict[key] = "[Content filtered for safety]"
                elif safety_score.flagged_categories:
                    # Consider redacting specific parts based on category
                    filtered_dict[key] = self._apply_redactions(value, safety_score.flagged_categories)
            elif isinstance(value, dict):
                filtered_dict[key] = self._filter_dict_response(value)
            elif isinstance(value, list):
                filtered_list = []
                for item in value:
                    if isinstance(item, str):
                        safety_score = self._check_text_safety(item)
                        if safety_score.overall_score < 0.5:
                            filtered_list.append("[Content filtered for safety]")
                        else:
                            filtered_list.append(self._apply_redactions(item, safety_score.flagged_categories))
                    else:
                        filtered_list.append(item)
                filtered_dict[key] = filtered_list
        
        return filtered_dict
    
    def _apply_redactions(self, text: str, flagged_categories: List[SafetyCategory]) -> str:
        """Apply redactions to sensitive content based on category"""
        result = text
        
        # Apply different redaction strategies based on category
        for category in flagged_categories:
            if category == SafetyCategory.CRISIS:
                # For crisis content, we might want to provide helpful resources instead
                if 'crisis' in [cat.value for cat in flagged_categories]:
                    # In a real implementation, we'd replace with appropriate resources
                    pass
            elif category == SafetyCategory.PRIVACY:
                # Redact privacy-related patterns
                for pattern in self.privacy_patterns:
                    result = re.sub(pattern, "[REDACTED]", result)
        
        return result
    
    def get_safety_explanation(self, text: str) -> str:
        """Get a human-readable explanation of safety issues"""
        safety_score = self._check_text_safety(text)
        
        explanations = []
        
        for category, score in safety_score.category_scores.items():
            if score > 0.3:  # Only report significant issues
                explanations.append(f"{category.value}: {score:.2f}")
        
        if explanations:
            return f"Safety concerns detected: {', '.join(explanations)}"
        else:
            return "Content appears safe"
    
    def moderate_conversation(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Moderate an entire conversation"""
        moderated_messages = []
        
        for message in messages:
            moderated_message = message.copy()
            if 'content' in message:
                safety_score = self._check_text_safety(message['content'])
                if safety_score.overall_score < 0.3:  # Very unsafe
                    moderated_message['content'] = "[Message filtered for safety]"
                elif safety_score.flagged_categories:
                    # Apply selective filtering
                    moderated_message['content'] = self._apply_redactions(
                        message['content'], 
                        safety_score.flagged_categories
                    )
                # Add safety metadata
                moderated_message['safety_score'] = safety_score.overall_score
                moderated_message['flagged_categories'] = [cat.value for cat in safety_score.flagged_categories]
            
            moderated_messages.append(moderated_message)
        
        return moderated_messages


class CrisisDetector:
    """Specialized detector for crisis situations"""
    
    def __init__(self):
        self.crisis_patterns = {
            'suicide': [
                r'suicid', r'kill.*myself', r'end.*life', r'not.*want.*live',
                r'hang myself', r'no reason to live', r'rather be dead'
            ],
            'harm_to_others': [
                r'kill.*[a-z ]*other', r'hurt.*people', r'violence', r'attack',
                r'harm.*others', r'murder', r'physically.*harm'
            ],
            'self_harm': [
                r'cut.*myself', r'hurt.*myself', r'self.*harm', r'self.*injury',
                r'harm.*[a-z ]*self', r'overdose', r'self.*destruct'
            ]
        }
        self.crisis_keywords = [
            'crisis', 'emergency', 'help', 'immediate', 'urgent', 'danger',
            'unsafe', 'harm', 'kill', 'die', 'death', 'end it', 'can\'t go on'
        ]
    
    def detect_crisis(self, text: str) -> Dict[str, Union[bool, float, List[str]]]:
        """Detect crisis-related content in text"""
        text_lower = text.lower()
        detected_types = []
        confidence = 0.0
        
        # Check pattern-based detection
        for crisis_type, patterns in self.crisis_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    detected_types.append(crisis_type)
                    confidence = max(confidence, 0.8)  # High confidence for pattern matches
        
        # Check keyword-based detection
        keyword_matches = [kw for kw in self.crisis_keywords if kw in text_lower]
        if keyword_matches:
            confidence = max(confidence, 0.6)  # Medium confidence for keywords
        
        return {
            'is_crisis': len(detected_types) > 0,
            'crisis_types': detected_types,
            'confidence': confidence,
            'keywords_found': keyword_matches
        }


# Global instance for use in other modules
safety_filter = SafetyFilter()
crisis_detector = CrisisDetector()


def apply_safety_filter(text: str, filter_type: str = "both") -> tuple[bool, str, float]:
    """
    Apply safety filter to text
    Returns: (is_safe, filtered_text, confidence_score)
    """
    if filter_type in ["input", "both"]:
        safety_score = safety_filter.check_input_safety(text)
    else:
        safety_score = safety_filter.check_output_safety(text)
    
    is_safe = safety_score.overall_score >= 0.5
    filtered_text = safety_filter._filter_text_response(text) if not is_safe else text
    confidence_score = safety_score.overall_score
    
    return is_safe, filtered_text, confidence_score


def check_crisis_content(text: str) -> Dict[str, Union[bool, float, List[str]]]:
    """Check specifically for crisis content"""
    return crisis_detector.detect_crisis(text)


# Example usage
def test_safety_filter():
    """Test the safety filter functionality"""
    logger.info("Testing Safety Filter...")
    
    test_cases = [
        "I'm feeling really down today and having thoughts about ending it all",
        "This is a normal conversation about therapy",
        "I hate everyone and want to hurt people",
        "My phone number is 123-456-7890 and my email is test@example.com",
        "I love helping people feel better"
    ]
    
    for i, text in enumerate(test_cases):
        print(f"\nTest case {i+1}: {text}")
        
        # Check safety
        safety_score = safety_filter.check_input_safety(text)
        print(f"Safety score: {safety_score.overall_score:.2f}")
        print(f"Flagged categories: {[cat.value for cat in safety_score.flagged_categories]}")
        
        # Check for crisis specifically
        crisis_result = check_crisis_content(text)
        print(f"Crisis detected: {crisis_result['is_crisis']}, Types: {crisis_result['crisis_types']}")
        
        # Filter the response
        filtered = safety_filter.filter_response(text)
        print(f"Filtered: {filtered}")
    
    print("\nSafety filter tests completed!")


if __name__ == "__main__":
    test_safety_filter()