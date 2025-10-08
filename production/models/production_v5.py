#!/usr/bin/env python3
"""Enhanced V5 Production Wrapper"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from crisis_detector_v5 import EnhancedCrisisDetectorV5
from datetime import datetime
import logging

class ProductionV5:
    def __init__(self):
        self.detector = EnhancedCrisisDetectorV5()
        
    def detect_crisis(self, text, user_id=None):
        """Production crisis detection"""
        try:
            result = self.detector.detect_crisis(text)
            result['production_timestamp'] = datetime.now().isoformat()
            result['user_id'] = user_id
            return result
        except Exception as e:
            return {
                'is_crisis': False,
                'confidence': 0.0,
                'error': str(e),
                'production_timestamp': datetime.now().isoformat()
            }

# Global production instance
production_v5 = ProductionV5()

def detect_crisis(text, user_id=None):
    """Main production interface"""
    return production_v5.detect_crisis(text, user_id)

if __name__ == "__main__":
    # Test production system
    test_result = detect_crisis("I want to kill myself")
    print(f"Production test - Crisis: {test_result['is_crisis']}")
    print("✅ Enhanced V5 Production System Ready")
