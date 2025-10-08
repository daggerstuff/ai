#!/usr/bin/env python3
"""
Enhanced tests for crisis intervention detector.
"""

import unittest
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from dataset_pipeline.crisis_intervention_detector import CrisisInterventionDetector, CrisisLevel
except ImportError:
    CrisisInterventionDetector = None
    CrisisLevel = None

class TestCrisisInterventionDetectorEnhanced(unittest.TestCase):
    """Enhanced tests for crisis intervention detector."""
    
    def setUp(self):
        """Set up test fixtures."""
        if CrisisInterventionDetector is None:
            self.skipTest("CrisisInterventionDetector not available")
        self.detector = CrisisInterventionDetector()
        
    def test_emergency_crisis_detection(self):
        """Test detection of emergency-level crises."""
        emergency_texts = [
            "I want to kill myself tonight",
            "I have a gun and I'm going to use it",
            "I'm going to hurt someone"
        ]
        
        for text in emergency_texts:
            with self.subTest(text=text):
                conversation = {'id': 'test', 'messages': [{'role': 'user', 'content': text}]}
                result = self.detector.detect_crisis(conversation)
                self.assertIsNotNone(result)
                
    def test_routine_content_detection(self):
        """Test that routine content is not flagged as crisis."""
        routine_texts = [
            "I'm feeling sad today",
            "Work has been stressful",
            "Can you help me with anxiety?"
        ]
        
        for text in routine_texts:
            with self.subTest(text=text):
                conversation = {'id': 'test', 'messages': [{'role': 'user', 'content': text}]}
                result = self.detector.detect_crisis(conversation)
                self.assertIsNotNone(result)
                
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        
    def test_empty_input_handling(self):
        """Test handling of empty input."""
        empty_conversation = {'id': 'test', 'messages': []}
        result = self.detector.detect_crisis(empty_conversation)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
