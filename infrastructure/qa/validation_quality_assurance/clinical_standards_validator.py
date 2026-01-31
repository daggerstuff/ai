#!/usr/bin/env python3
"""
Clinical Standards Validator - Task 5.7.2.1
Implements conversation validation against clinical standards including DSM-5, therapeutic guidelines, and professional ethics.
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClinicalValidationResult:
    """Results from clinical standards validation"""
    conversation_id: str
    dsm5_compliance: float
    therapeutic_boundaries: float
    ethical_guidelines: float
    crisis_intervention: float
    evidence_based_practice: float
    cultural_competency: float
    safety_protocols: float
    overall_clinical_score: float
    violations: List[str]
    recommendations: List[str]
    validation_timestamp: str

class ClinicalStandardsValidator:
    """Enterprise-grade clinical standards validation system"""
    
    def __init__(self):
        """Initialize clinical validation system"""
        self.nlp = None
        self._load_nlp_model()
        self._load_clinical_patterns()
        self.validation_stats = {
            'total_validated': 0,
            'violations_found': 0,
            'average_clinical_score': 0.0,
            'common_violations': {}
        }
        
    def _load_nlp_model(self):
        """Load spaCy NLP model for clinical analysis"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy model for clinical analysis")
        except OSError:
            logger.warning("⚠️ spaCy model not found, using basic text analysis")
            self.nlp = None
    
    def _load_clinical_patterns(self):
        """Load clinical validation patterns and guidelines"""
        
        # DSM-5 Compliance Patterns
        self.dsm5_patterns = {
            'diagnostic_criteria': [
                r'\b(meets criteria|diagnostic criteria|DSM-5|ICD-10)\b',
                r'\b(symptoms include|presenting symptoms|clinical presentation)\b',
                r'\b(duration|onset|severity|frequency)\b'
            ],
            'inappropriate_diagnosis': [
                r'\b(you have|you are|diagnosed with)\b(?!.*professional)',
                r'\b(definitely|certainly|obviously) (depressed|anxious|bipolar)\b',
                r'\bself-diagnos(e|is|ing)\b'
            ],
            'professional_language': [
                r'\b(consider|might|could|appears|suggests)\b',
                r'\b(professional help|qualified therapist|mental health professional)\b'
            ]
        }
        
        # Therapeutic Boundaries
        self.boundary_patterns = {
            'appropriate_boundaries': [
                r'\b(professional relationship|therapeutic relationship)\b',
                r'\b(within our session|during therapy|in treatment)\b',
                r'\b(confidentiality|privacy|professional boundaries)\b'
            ],
            'boundary_violations': [
                r'\b(personal relationship|be friends|outside of therapy)\b',
                r'\b(my personal|my own experience|happened to me)\b',
                r'\b(give you my number|meet outside|personal contact)\b'
            ],
            'dual_relationships': [
                r'\b(also my|both your|multiple roles)\b',
                r'\b(business relationship|romantic|dating)\b'
            ]
        }
        
        # Ethical Guidelines
        self.ethical_patterns = {
            'informed_consent': [
                r'\b(informed consent|understand the process|risks and benefits)\b',
                r'\b(confidentiality limits|mandatory reporting|duty to warn)\b'
            ],
            'competence': [
                r'\b(within my expertise|qualified to|specialized training)\b',
                r'\b(refer you to|outside my scope|need specialist)\b'
            ],
            'non_maleficence': [
                r'\b(do no harm|avoid harm|protect from harm)\b',
                r'\b(safe environment|safety first|your wellbeing)\b'
            ]
        }
        
        # Crisis Intervention
        self.crisis_patterns = {
            'suicide_assessment': [
                r'\b(suicidal thoughts|harm yourself|end your life)\b',
                r'\b(safety plan|crisis hotline|emergency contact)\b',
                r'\b(immediate danger|imminent risk|safety assessment)\b'
            ],
            'crisis_response': [
                r'\b(emergency services|call 911|crisis team)\b',
                r'\b(immediate help|urgent care|crisis intervention)\b'
            ],
            'inappropriate_crisis': [
                r'\b(just think positive|it will pass|everyone feels)\b',
                r'\b(not that serious|overreacting|dramatic)\b'
            ]
        }
        
        # Evidence-Based Practice
        self.evidence_patterns = {
            'therapeutic_techniques': [
                r'\b(CBT|cognitive behavioral|mindfulness|DBT)\b',
                r'\b(evidence-based|research shows|studies indicate)\b',
                r'\b(therapeutic technique|intervention|treatment approach)\b'
            ],
            'non_evidence_based': [
                r'\b(just believe|positive thinking will cure|simply choose)\b',
                r'\b(alternative medicine|crystals|energy healing)\b'
            ]
        }
        
        # Cultural Competency
        self.cultural_patterns = {
            'cultural_awareness': [
                r'\b(cultural background|cultural factors|diversity)\b',
                r'\b(respect your culture|cultural sensitivity|inclusive)\b'
            ],
            'cultural_insensitivity': [
                r'\b(all people|everyone should|universal truth)\b',
                r'\b(your culture is|that\'s weird|strange custom)\b'
            ]
        }
        
        # Safety Protocols
        self.safety_patterns = {
            'safety_measures': [
                r'\b(safety plan|coping strategies|support system)\b',
                r'\b(emergency contact|crisis resources|safety net)\b'
            ],
            'safety_violations': [
                r'\b(keep this secret|don\'t tell anyone|between us)\b',
                r'\b(handle this alone|no need for help|keep quiet)\b'
            ]
        }
        
        logger.info("✅ Loaded clinical validation patterns")
    
    def validate_conversation(self, conversation: Dict[str, Any]) -> ClinicalValidationResult:
        """Validate a conversation against clinical standards"""
        
        conversation_id = conversation.get('id', 'unknown')
        full_text = self._extract_conversation_text(conversation)
        
        # Perform clinical validations
        dsm5_score = self._validate_dsm5_compliance(full_text)
        boundaries_score = self._validate_therapeutic_boundaries(full_text)
        ethics_score = self._validate_ethical_guidelines(full_text)
        crisis_score = self._validate_crisis_intervention(full_text)
        evidence_score = self._validate_evidence_based_practice(full_text)
        cultural_score = self._validate_cultural_competency(full_text)
        safety_score = self._validate_safety_protocols(full_text)
        
        # Calculate overall clinical score
        scores = [dsm5_score, boundaries_score, ethics_score, crisis_score, 
                 evidence_score, cultural_score, safety_score]
        overall_score = np.mean(scores)
        
        # Identify violations and recommendations
        violations = self._identify_violations(full_text)
        recommendations = self._generate_recommendations(scores, violations)
        
        # Update statistics
        self._update_validation_stats(overall_score, violations)
        
        return ClinicalValidationResult(
            conversation_id=conversation_id,
            dsm5_compliance=dsm5_score,
            therapeutic_boundaries=boundaries_score,
            ethical_guidelines=ethics_score,
            crisis_intervention=crisis_score,
            evidence_based_practice=evidence_score,
            cultural_competency=cultural_score,
            safety_protocols=safety_score,
            overall_clinical_score=overall_score,
            violations=violations,
            recommendations=recommendations,
            validation_timestamp=datetime.now().isoformat()
        )
    
    def _extract_conversation_text(self, conversation: Dict[str, Any]) -> str:
        """Extract full text from conversation for analysis"""
        text_parts = []
        
        if 'messages' in conversation:
            for message in conversation['messages']:
                if isinstance(message, dict) and 'content' in message:
                    text_parts.append(message['content'])
                elif isinstance(message, str):
                    text_parts.append(message)
        elif 'conversation' in conversation:
            text_parts.append(str(conversation['conversation']))
        elif 'text' in conversation:
            text_parts.append(str(conversation['text']))
        
        return ' '.join(text_parts)
    
    def _validate_dsm5_compliance(self, text: str) -> float:
        """Validate DSM-5 compliance and diagnostic appropriateness"""
        score = 0.5  # Base score
        
        # Check for appropriate diagnostic language
        for pattern in self.dsm5_patterns['diagnostic_criteria']:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        for pattern in self.dsm5_patterns['professional_language']:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        # Penalize inappropriate diagnosis
        for pattern in self.dsm5_patterns['inappropriate_diagnosis']:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _validate_therapeutic_boundaries(self, text: str) -> float:
        """Validate therapeutic boundaries and professional relationships"""
        score = 0.5  # Base score
        
        # Check for appropriate boundaries
        for pattern in self.boundary_patterns['appropriate_boundaries']:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.15
        
        # Penalize boundary violations
        for pattern in self.boundary_patterns['boundary_violations']:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.3
        
        for pattern in self.boundary_patterns['dual_relationships']:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _validate_ethical_guidelines(self, text: str) -> float:
        """Validate ethical guidelines compliance"""
        score = 0.5  # Base score
        
        # Check for ethical practices
        for pattern in self.ethical_patterns['informed_consent']:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        for pattern in self.ethical_patterns['competence']:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        for pattern in self.ethical_patterns['non_maleficence']:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _validate_crisis_intervention(self, text: str) -> float:
        """Validate crisis intervention and safety protocols"""
        score = 0.7  # Higher base score for non-crisis conversations
        
        # Check if crisis content is present
        crisis_present = any(re.search(pattern, text, re.IGNORECASE) 
                           for pattern in self.crisis_patterns['suicide_assessment'])
        
        if crisis_present:
            score = 0.3  # Lower base for crisis situations
            
            # Check for appropriate crisis response
            for pattern in self.crisis_patterns['crisis_response']:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 0.2
            
            # Penalize inappropriate crisis handling
            for pattern in self.crisis_patterns['inappropriate_crisis']:
                if re.search(pattern, text, re.IGNORECASE):
                    score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _validate_evidence_based_practice(self, text: str) -> float:
        """Validate evidence-based practice usage"""
        score = 0.5  # Base score
        
        # Check for evidence-based techniques
        for pattern in self.evidence_patterns['therapeutic_techniques']:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.15
        
        # Penalize non-evidence-based approaches
        for pattern in self.evidence_patterns['non_evidence_based']:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _validate_cultural_competency(self, text: str) -> float:
        """Validate cultural competency and sensitivity"""
        score = 0.6  # Base score
        
        # Check for cultural awareness
        for pattern in self.cultural_patterns['cultural_awareness']:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        # Penalize cultural insensitivity
        for pattern in self.cultural_patterns['cultural_insensitivity']:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _validate_safety_protocols(self, text: str) -> float:
        """Validate safety protocols and measures"""
        score = 0.6  # Base score
        
        # Check for safety measures
        for pattern in self.safety_patterns['safety_measures']:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        # Penalize safety violations
        for pattern in self.safety_patterns['safety_violations']:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _identify_violations(self, text: str) -> List[str]:
        """Identify specific clinical violations"""
        violations = []
        
        # Check for diagnostic violations
        for pattern in self.dsm5_patterns['inappropriate_diagnosis']:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append("Inappropriate diagnostic language detected")
        
        # Check for boundary violations
        for pattern in self.boundary_patterns['boundary_violations']:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append("Therapeutic boundary violation detected")
        
        # Check for crisis handling violations
        for pattern in self.crisis_patterns['inappropriate_crisis']:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append("Inappropriate crisis intervention detected")
        
        # Check for cultural insensitivity
        for pattern in self.cultural_patterns['cultural_insensitivity']:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append("Cultural insensitivity detected")
        
        # Check for safety violations
        for pattern in self.safety_patterns['safety_violations']:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append("Safety protocol violation detected")
        
        return violations
    
    def _generate_recommendations(self, scores: List[float], violations: List[str]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        score_names = ['DSM-5 Compliance', 'Therapeutic Boundaries', 'Ethical Guidelines',
                      'Crisis Intervention', 'Evidence-Based Practice', 'Cultural Competency',
                      'Safety Protocols']
        
        # Recommend improvements for low scores
        for i, score in enumerate(scores):
            if score < 0.6:
                recommendations.append(f"Improve {score_names[i]} (current: {score:.2f})")
        
        # Add specific recommendations based on violations
        if violations:
            recommendations.append("Address identified clinical violations")
            recommendations.append("Review professional guidelines and standards")
        
        # General recommendations
        if np.mean(scores) < 0.7:
            recommendations.append("Consider additional clinical training")
            recommendations.append("Implement regular supervision and consultation")
        
        return recommendations
    
    def _update_validation_stats(self, score: float, violations: List[str]):
        """Update validation statistics"""
        self.validation_stats['total_validated'] += 1
        
        if violations:
            self.validation_stats['violations_found'] += 1
            for violation in violations:
                if violation in self.validation_stats['common_violations']:
                    self.validation_stats['common_violations'][violation] += 1
                else:
                    self.validation_stats['common_violations'][violation] = 1
        
        # Update running average
        total = self.validation_stats['total_validated']
        current_avg = self.validation_stats['average_clinical_score']
        self.validation_stats['average_clinical_score'] = ((current_avg * (total - 1)) + score) / total
    
    def validate_batch(self, conversations: List[Dict[str, Any]]) -> List[ClinicalValidationResult]:
        """Validate a batch of conversations"""
        results = []
        
        logger.info(f"🔍 Starting clinical validation of {len(conversations)} conversations...")
        
        for i, conversation in enumerate(conversations):
            try:
                result = self.validate_conversation(conversation)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"✅ Validated {i + 1}/{len(conversations)} conversations")
                    
            except Exception as e:
                logger.error(f"❌ Error validating conversation {i}: {e}")
                continue
        
        logger.info(f"🎯 Clinical validation complete: {len(results)} conversations validated")
        return results
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics and insights"""
        return {
            'validation_stats': self.validation_stats,
            'clinical_patterns_loaded': {
                'dsm5_patterns': len(self.dsm5_patterns),
                'boundary_patterns': len(self.boundary_patterns),
                'ethical_patterns': len(self.ethical_patterns),
                'crisis_patterns': len(self.crisis_patterns),
                'evidence_patterns': len(self.evidence_patterns),
                'cultural_patterns': len(self.cultural_patterns),
                'safety_patterns': len(self.safety_patterns)
            },
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def export_validation_results(self, results: List[ClinicalValidationResult], 
                                output_path: str) -> bool:
        """Export validation results to JSON file"""
        try:
            output_data = {
                'validation_results': [
                    {
                        'conversation_id': result.conversation_id,
                        'clinical_scores': {
                            'dsm5_compliance': result.dsm5_compliance,
                            'therapeutic_boundaries': result.therapeutic_boundaries,
                            'ethical_guidelines': result.ethical_guidelines,
                            'crisis_intervention': result.crisis_intervention,
                            'evidence_based_practice': result.evidence_based_practice,
                            'cultural_competency': result.cultural_competency,
                            'safety_protocols': result.safety_protocols,
                            'overall_clinical_score': result.overall_clinical_score
                        },
                        'violations': result.violations,
                        'recommendations': result.recommendations,
                        'validation_timestamp': result.validation_timestamp
                    }
                    for result in results
                ],
                'summary_statistics': self.get_validation_statistics(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Exported clinical validation results to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting validation results: {e}")
            return False

def main():
    """Test clinical standards validation system"""
    validator = ClinicalStandardsValidator()
    
    # Test conversation
    test_conversation = {
        'id': 'test_001',
        'messages': [
            {'role': 'user', 'content': 'I feel really depressed and have been having suicidal thoughts.'},
            {'role': 'assistant', 'content': 'I understand you\'re going through a difficult time. These feelings are concerning and I want to help ensure your safety. Have you thought about harming yourself or ending your life? It\'s important we develop a safety plan and connect you with crisis resources like the National Suicide Prevention Lifeline at 988.'}
        ]
    }
    
    # Validate conversation
    result = validator.validate_conversation(test_conversation)
    
    print(f"\n🔍 Clinical Validation Results:")
    print(f"Overall Clinical Score: {result.overall_clinical_score:.3f}")
    print(f"DSM-5 Compliance: {result.dsm5_compliance:.3f}")
    print(f"Therapeutic Boundaries: {result.therapeutic_boundaries:.3f}")
    print(f"Ethical Guidelines: {result.ethical_guidelines:.3f}")
    print(f"Crisis Intervention: {result.crisis_intervention:.3f}")
    print(f"Evidence-Based Practice: {result.evidence_based_practice:.3f}")
    print(f"Cultural Competency: {result.cultural_competency:.3f}")
    print(f"Safety Protocols: {result.safety_protocols:.3f}")
    
    if result.violations:
        print(f"\n⚠️ Violations Found: {len(result.violations)}")
        for violation in result.violations:
            print(f"  - {violation}")
    
    if result.recommendations:
        print(f"\n💡 Recommendations: {len(result.recommendations)}")
        for rec in result.recommendations:
            print(f"  - {rec}")
    
    # Get statistics
    stats = validator.get_validation_statistics()
    print(f"\n📊 Validation Statistics:")
    print(f"Total Validated: {stats['validation_stats']['total_validated']}")
    print(f"Average Clinical Score: {stats['validation_stats']['average_clinical_score']:.3f}")

if __name__ == "__main__":
    main()
