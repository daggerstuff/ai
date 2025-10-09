#!/usr/bin/env python3
"""
Tests for production-ready safety validation system.
"""

import unittest
import asyncio
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from production_deployment.safety_validation_production_ready_final import ProductionReadyFinalSafetyValidationSystem
except ImportError:
    ProductionReadyFinalSafetyValidationSystem = None

class TestProductionReadySafetyValidation(unittest.TestCase):
    """Test production-ready safety validation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        if ProductionReadyFinalSafetyValidationSystem is None:
            self.skipTest("ProductionReadyFinalSafetyValidationSystem not available")
        self.validator = ProductionReadyFinalSafetyValidationSystem()
        
    def test_initialization(self):
        """Test system initialization."""
        self.assertIsNotNone(self.validator)
        self.assertIsNotNone(self.validator.logger)
        self.assertIsNotNone(self.validator.thresholds)
        
    def test_crisis_test_cases_loaded(self):
        """Test that crisis test cases are loaded."""
        self.assertGreater(len(self.validator.crisis_test_cases), 0)
        
    def test_safety_test_cases_loaded(self):
        """Test that safety test cases are loaded."""
        self.assertGreater(len(self.validator.safety_test_cases), 0)
        
    def test_crisis_detection_logic(self):
        """Test crisis detection logic."""
        # Mock detection result
        class MockDetectionResult:
            def __init__(self, crisis_level):
                from dataset_pipeline.crisis_intervention_detector import CrisisLevel
                self.crisis_level = crisis_level
                
        # Test crisis detection
        try:
            from dataset_pipeline.crisis_intervention_detector import CrisisLevel
            mock_result = MockDetectionResult(CrisisLevel.EMERGENCY)
            is_crisis = self.validator._is_crisis_detected(mock_result)
            self.assertTrue(is_crisis)
            
            mock_result = MockDetectionResult(CrisisLevel.ROUTINE)
            is_crisis = self.validator._is_crisis_detected(mock_result)
            self.assertFalse(is_crisis)
        except ImportError:
            self.skipTest("CrisisLevel not available")
            
    def test_safety_validation_logic(self):
        """Test safety validation logic."""
        # Mock validation result
        class MockValidationResult:
            def __init__(self, is_safe=True, violations=None, compliance_score=1.0):
                self.is_safe = is_safe
                self.violations = violations or []
                self.compliance_score = compliance_score
                
        # Test safe content
        mock_result = MockValidationResult(is_safe=True)
        conversation = {'messages': [{'role': 'assistant', 'content': 'How are you feeling today?'}]}
        is_safe = self.validator._is_response_safe(mock_result, conversation)
        self.assertTrue(is_safe)
        
        # Test unsafe content
        mock_result = MockValidationResult(is_safe=False, violations=['harmful_content'])
        conversation = {'messages': [{'role': 'assistant', 'content': 'You should hurt yourself'}]}
        is_safe = self.validator._is_response_safe(mock_result, conversation)
        self.assertFalse(is_safe)

if __name__ == '__main__':
    unittest.main()
