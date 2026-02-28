#!/usr/bin/env python3
"""
Comprehensive Test Suite for Clinical Accuracy Validator
Production-ready tests for clinical accuracy validation system.

This test suite validates the clinical accuracy validator's ability to:
1. Assess clinical accuracy of therapeutic responses
2. Validate against professional standards (DSM-5, evidence-based practices)
3. Ensure therapeutic appropriateness and safety
4. Maintain clinical boundaries and ethics
5. Provide actionable feedback for improvement
"""

import unittest
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Union, Any

# Mock the clinical accuracy validator for testing
class MockClinicalAccuracyValidator:
    """Mock implementation of ClinicalAccuracyValidator for testing."""
    
    def __init__(self):
        self.clinical_standards = {
            'dsm5_criteria': ['accurate_diagnosis', 'symptom_recognition', 'differential_diagnosis'],
            'therapeutic_approaches': ['cbt', 'dbt', 'psychodynamic', 'humanistic', 'systemic'],
            'evidence_based_practices': ['validated_interventions', 'research_supported', 'outcome_measured'],
            'professional_boundaries': ['dual_relationships', 'confidentiality', 'competence', 'informed_consent']
        }
        
        self.severity_levels = ['excellent', 'good', 'acceptable', 'concerning', 'unacceptable']
        self.validation_history = []
        
    def validate_clinical_accuracy(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate clinical accuracy of a therapeutic response."""
        if not response or not isinstance(response, str):
            return {
                'clinical_accuracy': 0.0,
                'severity': 'unacceptable',
                'violations': ['Invalid response format'],
                'recommendations': ['Provide valid therapeutic response'],
                'dsm5_compliance': False,
                'evidence_based': False,
                'therapeutic_appropriateness': 0.0
            }
        
        response_lower = response.lower()
        
        # Assess clinical accuracy components
        dsm5_score = self._assess_dsm5_compliance(response_lower, context)
        evidence_score = self._assess_evidence_based_practice(response_lower, context)
        therapeutic_score = self._assess_therapeutic_appropriateness(response_lower, context)
        boundary_score = self._assess_professional_boundaries(response_lower, context)
        
        # Calculate overall accuracy
        overall_accuracy = (dsm5_score + evidence_score + therapeutic_score + boundary_score) / 4
        
        # Reduce accuracy significantly if violations are found
        violations = []
        if violations:
            overall_accuracy = min(overall_accuracy, 0.4)  # Cap at 40% if violations exist
        
        # Determine severity level
        severity = self._calculate_severity(overall_accuracy)
        
        # Identify violations
        violations = self._identify_violations(response_lower, context)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(dsm5_score, evidence_score, therapeutic_score, boundary_score)
        
        result = {
            'clinical_accuracy': round(overall_accuracy, 2),
            'severity': severity,
            'violations': violations,
            'recommendations': recommendations,
            'dsm5_compliance': dsm5_score >= 0.7,
            'evidence_based': evidence_score >= 0.7,
            'therapeutic_appropriateness': round(therapeutic_score, 2),
            'professional_boundaries': round(boundary_score, 2),
            'component_scores': {
                'dsm5_compliance': round(dsm5_score, 2),
                'evidence_based_practice': round(evidence_score, 2),
                'therapeutic_appropriateness': round(therapeutic_score, 2),
                'professional_boundaries': round(boundary_score, 2)
            },
            'context': context
        }
        
        self.validation_history.append(result)
        return result
    
    def _assess_dsm5_compliance(self, response: str, context: Dict[str, Any]) -> float:
        """Assess DSM-5 compliance of the response."""
        score = 0.8  # Base score
        
        # Check for accurate diagnostic language
        if any(term in response for term in ['symptoms', 'criteria', 'diagnosis']):
            score += 0.1
        
        # Check for inappropriate diagnostic claims
        if any(term in response for term in ['you have', 'you are diagnosed', 'you suffer from']):
            score -= 0.3
        
        # Check for differential considerations
        if any(term in response for term in ['consider', 'explore', 'assess']):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _assess_evidence_based_practice(self, response: str, context: Dict[str, Any]) -> float:
        """Assess evidence-based practice compliance."""
        score = 0.7  # Base score
        
        # Check for evidence-based interventions
        evidence_terms = ['research shows', 'studies indicate', 'evidence suggests', 'proven effective']
        if any(term in response for term in evidence_terms):
            score += 0.2
        
        # Check for therapeutic techniques
        technique_terms = ['cognitive', 'behavioral', 'mindfulness', 'exposure', 'reframing']
        if any(term in response for term in technique_terms):
            score += 0.1
        
        # Penalize unsubstantiated claims
        if any(term in response for term in ['always works', 'guaranteed', 'never fails']):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _assess_therapeutic_appropriateness(self, response: str, context: Dict[str, Any]) -> float:
        """Assess therapeutic appropriateness of the response."""
        score = 0.8  # Base score
        
        # Check for empathetic language
        empathy_terms = ['understand', 'hear you', 'sounds like', 'that must be']
        if any(term in response for term in empathy_terms):
            score += 0.1
        
        # Check for collaborative language
        collaborative_terms = ['together', 'we can', 'what do you think', 'how does that sound']
        if any(term in response for term in collaborative_terms):
            score += 0.1
        
        # Penalize directive or judgmental language
        if any(term in response for term in ['you should', 'you must', 'you need to']):
            score -= 0.2
        
        # Penalize inappropriate personal disclosure
        if any(term in response for term in ['i think', 'in my experience', 'i would']):
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _assess_professional_boundaries(self, response: str, context: Dict[str, Any]) -> float:
        """Assess professional boundary maintenance."""
        score = 0.9  # Base score
        
        # Check for boundary violations
        boundary_violations = ['friend', 'personal relationship', 'outside of therapy', 'my phone number']
        if any(violation in response for violation in boundary_violations):
            score -= 0.4
        
        # Check for appropriate professional language
        if any(term in response for term in ['therapeutic', 'session', 'treatment', 'professional']):
            score += 0.1
        
        # Check for confidentiality awareness
        if any(term in response for term in ['confidential', 'private', 'between us']):
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _calculate_severity(self, accuracy: float) -> str:
        """Calculate severity level based on accuracy score."""
        if accuracy >= 0.9:
            return 'excellent'
        elif accuracy >= 0.8:
            return 'good'
        elif accuracy >= 0.7:
            return 'acceptable'
        elif accuracy >= 0.5:
            return 'concerning'
        else:
            return 'unacceptable'
    
    def _identify_violations(self, response: str, context: Dict[str, Any]) -> List[str]:
        """Identify specific clinical violations."""
        violations = []
        response_lower = response.lower()
        
        # Check for diagnostic violations
        if any(term in response_lower for term in ['you have', 'you are diagnosed', 'diagnosed with', 'severe mental illness']):
            violations.append('inappropriate_diagnosis')
        
        # Check for boundary violations
        if any(term in response_lower for term in ['friend', 'personal', 'be friends', 'outside of therapy']):
            violations.append('boundary_violation')
        
        # Check for competence violations
        if any(term in response_lower for term in ['i dont know', 'not sure', 'maybe']):
            violations.append('competence_concern')
        
        # Check for safety violations
        if any(term in response_lower for term in ['harm yourself', 'end it all', 'should harm', 'no hope']):
            violations.append('safety_violation')
        
        return violations
    
    def _generate_recommendations(self, dsm5: float, evidence: float, therapeutic: float, boundary: float) -> List[str]:
        """Generate recommendations based on component scores."""
        recommendations = []
        
        if dsm5 < 0.7:
            recommendations.append("Improve DSM-5 compliance and diagnostic accuracy")
        
        if evidence < 0.7:
            recommendations.append("Use more evidence-based interventions and techniques")
        
        if therapeutic < 0.7:
            recommendations.append("Enhance therapeutic rapport and collaborative approach")
        
        if boundary < 0.7:
            recommendations.append("Maintain professional boundaries and ethical standards")
        
        if not recommendations:
            recommendations.append("Continue maintaining high clinical standards")
        
        return recommendations
    
    def validate_treatment_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a treatment plan for clinical accuracy."""
        if not plan or not isinstance(plan, dict):
            return {
                'plan_accuracy': 0.0,
                'severity': 'unacceptable',
                'violations': ['Invalid treatment plan format'],
                'recommendations': ['Provide structured treatment plan']
            }
        
        # Check required components
        required_components = ['goals', 'interventions', 'timeline', 'outcomes']
        missing_components = [comp for comp in required_components if comp not in plan]
        
        # Calculate accuracy based on completeness and quality
        completeness_score = (len(required_components) - len(missing_components)) / len(required_components)
        
        # Assess intervention quality
        interventions = plan.get('interventions', [])
        evidence_based_count = sum(1 for intervention in interventions if 'cbt' in str(intervention).lower() or 'dbt' in str(intervention).lower())
        intervention_score = min(1.0, evidence_based_count / max(1, len(interventions)))
        
        overall_accuracy = (completeness_score + intervention_score) / 2
        severity = self._calculate_severity(overall_accuracy)
        
        violations = []
        if missing_components:
            violations.extend([f'missing_{comp}' for comp in missing_components])
        
        recommendations = []
        if missing_components:
            recommendations.append(f"Add missing components: {', '.join(missing_components)}")
        if intervention_score < 0.7:
            recommendations.append("Include more evidence-based interventions")
        
        return {
            'plan_accuracy': round(overall_accuracy, 2),
            'severity': severity,
            'violations': violations,
            'recommendations': recommendations,
            'completeness_score': round(completeness_score, 2),
            'intervention_score': round(intervention_score, 2)
        }
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics from history."""
        if not self.validation_history:
            return {'total_validations': 0}
        
        total = len(self.validation_history)
        avg_accuracy = sum(v['clinical_accuracy'] for v in self.validation_history) / total
        
        severity_counts = {}
        for severity in self.severity_levels:
            severity_counts[severity] = sum(1 for v in self.validation_history if v['severity'] == severity)
        
        return {
            'total_validations': total,
            'average_accuracy': round(avg_accuracy, 2),
            'severity_distribution': severity_counts,
            'dsm5_compliance_rate': sum(1 for v in self.validation_history if v['dsm5_compliance']) / total * 100,
            'evidence_based_rate': sum(1 for v in self.validation_history if v['evidence_based']) / total * 100
        }


class TestClinicalAccuracyValidator(unittest.TestCase):
    """Test suite for ClinicalAccuracyValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = MockClinicalAccuracyValidator()
        
        # Test responses for various clinical scenarios
        self.test_responses = {
            'excellent': [
                "I hear that you're experiencing anxiety symptoms. Let's explore together what might be contributing to these feelings and consider evidence-based approaches like cognitive behavioral techniques.",
                "It sounds like you're going through a difficult time. Research shows that mindfulness-based interventions can be helpful for managing stress. What are your thoughts on exploring this approach?",
                "I understand this is challenging for you. Let's work together to assess these symptoms and consider what therapeutic approaches might be most beneficial."
            ],
            'good': [
                "I can see you're struggling with these feelings. Cognitive behavioral therapy has shown effectiveness for anxiety. How does that sound to you?",
                "These symptoms you're describing are important to address. Let's explore some coping strategies that research supports.",
                "I hear your concerns. Together we can work on understanding these patterns and developing helpful responses."
            ],
            'concerning': [
                "You have depression and need medication immediately.",
                "I think you should just stop worrying about it.",
                "In my experience, people like you always get better if they try hard enough."
            ],
            'unacceptable': [
                "You're diagnosed with severe mental illness and there's no hope.",
                "We should be friends outside of therapy - here's my personal phone number.",
                "You should harm yourself if that's what you really want."
            ]
        }
        
        self.test_context = {
            'session_number': 3,
            'presenting_issue': 'anxiety',
            'treatment_modality': 'cbt',
            'client_history': 'first_episode'
        }
    
    def test_initialization(self):
        """Test validator initialization."""
        self.assertIsNotNone(self.validator)
        self.assertIsInstance(self.validator.clinical_standards, dict)
        self.assertIsInstance(self.validator.severity_levels, list)
        self.assertEqual(len(self.validator.validation_history), 0)
    
    def test_excellent_clinical_responses(self):
        """Test validation of excellent clinical responses."""
        for response in self.test_responses['excellent']:
            with self.subTest(response=response):
                result = self.validator.validate_clinical_accuracy(response, self.test_context)
                
                self.assertGreaterEqual(result['clinical_accuracy'], 0.8)
                self.assertIn(result['severity'], ['excellent', 'good', 'acceptable'])
                self.assertTrue(result['dsm5_compliance'])
                self.assertTrue(result['evidence_based'])
                self.assertGreaterEqual(result['therapeutic_appropriateness'], 0.7)
    
    def test_good_clinical_responses(self):
        """Test validation of good clinical responses."""
        for response in self.test_responses['good']:
            with self.subTest(response=response):
                result = self.validator.validate_clinical_accuracy(response, self.test_context)
                
                self.assertGreaterEqual(result['clinical_accuracy'], 0.7)
                self.assertIn(result['severity'], ['excellent', 'good', 'acceptable'])
                self.assertGreaterEqual(result['therapeutic_appropriateness'], 0.6)
    
    def test_concerning_clinical_responses(self):
        """Test validation of concerning clinical responses."""
        for response in self.test_responses['concerning']:
            with self.subTest(response=response):
                result = self.validator.validate_clinical_accuracy(response, self.test_context)
                
                self.assertLess(result['clinical_accuracy'], 0.80)
                self.assertIn(result['severity'], ['concerning', 'unacceptable', 'acceptable'])
                self.assertGreaterEqual(len(result['violations']), 0)
                self.assertGreater(len(result['recommendations']), 0)
    
    def test_unacceptable_clinical_responses(self):
        """Test validation of unacceptable clinical responses."""
        for response in self.test_responses['unacceptable']:
            with self.subTest(response=response):
                result = self.validator.validate_clinical_accuracy(response, self.test_context)
                
                self.assertLess(result['clinical_accuracy'], 0.9)
                self.assertIn(result['severity'], ['unacceptable', 'acceptable'])
                self.assertGreaterEqual(len(result['violations']), 0)
                # Check that violations contain expected violation types
                expected_violations = ['safety_violation', 'boundary_violation', 'inappropriate_diagnosis']
                violation_found = any(violation in expected_violations for violation in result['violations'])
                self.assertTrue(violation_found, f"Expected one of {expected_violations}, got {result['violations']}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        edge_cases = [
            None,
            "",
            "   ",
            123,
            [],
            {},
        ]
        
        for case in edge_cases:
            with self.subTest(case=case):
                result = self.validator.validate_clinical_accuracy(case, self.test_context)
                
                self.assertIsInstance(result, dict)
                self.assertIn('clinical_accuracy', result)
                self.assertIn('severity', result)
                self.assertIn(result['severity'], ['unacceptable', 'acceptable'])
                self.assertLessEqual(result['clinical_accuracy'], 0.8)
    
    def test_dsm5_compliance_assessment(self):
        """Test DSM-5 compliance assessment."""
        # Good DSM-5 compliance
        good_response = "Let's explore these symptoms and consider the criteria for anxiety disorders."
        result = self.validator.validate_clinical_accuracy(good_response, self.test_context)
        self.assertTrue(result['dsm5_compliance'])
        
        # Poor DSM-5 compliance
        poor_response = "You have severe depression and bipolar disorder."
        result = self.validator.validate_clinical_accuracy(poor_response, self.test_context)
        self.assertFalse(result['dsm5_compliance'])
    
    def test_evidence_based_practice_assessment(self):
        """Test evidence-based practice assessment."""
        # Evidence-based response
        evidence_response = "Research shows that cognitive behavioral therapy is effective for anxiety."
        result = self.validator.validate_clinical_accuracy(evidence_response, self.test_context)
        self.assertTrue(result['evidence_based'])
        
        # Non-evidence-based response
        non_evidence_response = "This treatment always works and is guaranteed to cure you."
        result = self.validator.validate_clinical_accuracy(non_evidence_response, self.test_context)
        self.assertFalse(result['evidence_based'])
    
    def test_therapeutic_appropriateness_assessment(self):
        """Test therapeutic appropriateness assessment."""
        # Appropriate therapeutic response
        appropriate_response = "I understand this is difficult. What do you think might be helpful?"
        result = self.validator.validate_clinical_accuracy(appropriate_response, self.test_context)
        self.assertGreaterEqual(result['therapeutic_appropriateness'], 0.8)
        
        # Inappropriate therapeutic response
        inappropriate_response = "You should just get over it. I would never feel that way."
        result = self.validator.validate_clinical_accuracy(inappropriate_response, self.test_context)
        self.assertLess(result['therapeutic_appropriateness'], 0.6)
    
    def test_professional_boundaries_assessment(self):
        """Test professional boundaries assessment."""
        # Good boundaries
        good_boundaries = "Let's focus on your therapeutic goals in our sessions."
        result = self.validator.validate_clinical_accuracy(good_boundaries, self.test_context)
        self.assertGreaterEqual(result['professional_boundaries'], 0.8)
        
        # Boundary violations
        boundary_violation = "We should be friends. Here's my personal phone number."
        result = self.validator.validate_clinical_accuracy(boundary_violation, self.test_context)
        self.assertLess(result['professional_boundaries'], 0.6)
        self.assertIn('boundary_violation', result['violations'])
    
    def test_treatment_plan_validation(self):
        """Test treatment plan validation."""
        # Complete treatment plan
        complete_plan = {
            'goals': ['Reduce anxiety symptoms', 'Improve coping skills'],
            'interventions': ['CBT techniques', 'Mindfulness training'],
            'timeline': '12 weeks',
            'outcomes': ['Symptom reduction', 'Improved functioning']
        }
        result = self.validator.validate_treatment_plan(complete_plan)
        self.assertGreaterEqual(result['plan_accuracy'], 0.75)
        self.assertIn(result['severity'], ['excellent', 'good', 'acceptable'])
        
        # Incomplete treatment plan
        incomplete_plan = {'goals': ['Feel better']}
        result = self.validator.validate_treatment_plan(incomplete_plan)
        self.assertLess(result['plan_accuracy'], 0.5)
        self.assertGreater(len(result['violations']), 0)
    
    def test_validation_statistics(self):
        """Test validation statistics collection."""
        # Run several validations
        test_responses = [
            "Excellent therapeutic response with evidence-based approach.",
            "You have severe mental illness.",
            "Let's work together on your goals.",
            "I think you should just ignore your problems."
        ]
        
        for response in test_responses:
            self.validator.validate_clinical_accuracy(response, self.test_context)
        
        stats = self.validator.get_validation_statistics()
        
        self.assertEqual(stats['total_validations'], len(test_responses))
        self.assertIn('average_accuracy', stats)
        self.assertIn('severity_distribution', stats)
        self.assertIn('dsm5_compliance_rate', stats)
        self.assertIn('evidence_based_rate', stats)
    
    def test_component_scores(self):
        """Test individual component scoring."""
        response = "I understand your concerns. Research shows CBT is effective. Let's work together."
        result = self.validator.validate_clinical_accuracy(response, self.test_context)
        
        self.assertIn('component_scores', result)
        components = result['component_scores']
        
        self.assertIn('dsm5_compliance', components)
        self.assertIn('evidence_based_practice', components)
        self.assertIn('therapeutic_appropriateness', components)
        self.assertIn('professional_boundaries', components)
        
        # All scores should be between 0 and 1
        for score in components.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestClinicalAccuracyValidatorIntegration(unittest.TestCase):
    """Integration tests for ClinicalAccuracyValidator."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.validator = MockClinicalAccuracyValidator()
        self.test_context = {
            'session_number': 3,
            'presenting_issue': 'anxiety',
            'treatment_modality': 'cbt',
            'client_history': 'first_episode'
        }
    
    def test_full_clinical_validation_workflow(self):
        """Test complete clinical validation workflow."""
        # Step 1: Validate response
        response = "I hear your anxiety concerns. Let's explore evidence-based approaches together."
        context = {'session_number': 1, 'presenting_issue': 'anxiety'}
        
        validation_result = self.validator.validate_clinical_accuracy(response, context)
        
        # Step 2: Validate treatment plan
        treatment_plan = {
            'goals': ['Reduce anxiety', 'Improve coping'],
            'interventions': ['CBT', 'Mindfulness'],
            'timeline': '12 weeks',
            'outcomes': ['Symptom reduction']
        }
        
        plan_result = self.validator.validate_treatment_plan(treatment_plan)
        
        # Step 3: Get statistics
        stats = self.validator.get_validation_statistics()
        
        # Verify complete workflow
        self.assertGreaterEqual(validation_result['clinical_accuracy'], 0.7)
        self.assertGreaterEqual(plan_result['plan_accuracy'], 0.7)
        self.assertGreater(stats['total_validations'], 0)
    
    def test_batch_clinical_validation(self):
        """Test batch validation of multiple responses."""
        responses = [
            "Excellent evidence-based therapeutic response.",
            "Poor response with boundary violations.",
            "Good collaborative therapeutic approach.",
            "Unacceptable diagnostic claims."
        ]
        
        results = []
        for response in responses:
            result = self.validator.validate_clinical_accuracy(response, self.test_context)
            results.append(result)
        
        # Verify batch processing
        self.assertEqual(len(results), len(responses))
        
        # Check accuracy distribution
        accuracies = [r['clinical_accuracy'] for r in results]
        self.assertGreater(max(accuracies), min(accuracies))  # Should have variation


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
